# QAT（量化感知训练）执行计划

## 目标

通过量化感知训练，将 SenseVoiceSmall Encoder 的 INT8 推理精度从 **cos=0.9804** 提升到 **cos≥0.99**，同时保持 INT8 的延迟优势（~170ms）。

## 模型架构分析

### 配置
| 参数 | 值 |
|------|-----|
| 模型 | SenseVoiceSmall (FunASR v1.3.1) |
| Encoder | SenseVoiceEncoderSmall |
| output_size | 512 |
| attention_heads | 4 |
| linear_units | 2048 |
| num_blocks | 50 (1 encoders0 + 49 encoders) |
| tp_blocks | 20 |
| total 层数 | 70 EncoderLayerSANM |
| 量化敏感层 | LayerNorm (exNorm) × 72 (70 block × 2 + after_norm + tp_norm) |

### 模型结构

```
SenseVoiceSmall
  ├── embed (language/text_norm query)
  └── encoder (SenseVoiceEncoderSmall)
        ├── embed (SinusoidalPositionEncoder)
        ├── encoders0 (1× EncoderLayerSANM)
        │     └── [norm1 → self_attn (SANM) → norm2 → ffn]
        ├── encoders (49× EncoderLayerSANM)    ← 量化瓶颈
        │     └── [norm1 → self_attn (SANM) → norm2 → ffn]
        ├── after_norm (LayerNorm)
        ├── tp_encoders (20× EncoderLayerSANM)
        │     └── [norm1 → self_attn (SANM) → norm2 → ffn]
        └── tp_norm (LayerNorm)
  └── ctc (CTC)
        └── ctc_lo (Linear)
```

### 量化敏感层定位

已有实验证明：
- **norm1 层**：量化敏感度最高，输出直接送入 self_attn MatMul
- **norm2 层**：custom_hybrid 保留 FP16 无改善，说明非主要瓶颈
- **self_attn 的 MatMul**：对输入精度敏感
- **ctc_lo (Linear)**：输出层，精度影响 cosine similarity

---

## 阶段一：环境准备（1天）

### 1.1 创建 QAT 环境

```bash
# 基于现有 FunASR 环境
conda create -n sensevoice_qat python=3.10
conda activate sensevoice_qat

# 安装 PyTorch（与 RKNN 兼容的版本）
pip install torch==2.0.1 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu

# 安装 FunASR（与当前版本一致）
pip install funasr==1.3.1

# 安装 NVIDIA QAT 工具
pip install nvidia-pyindex
pip install pytorch-quantization==2.1.2
```

### 1.2 验证模型加载

```python
from funasr.models.sense_voice.model import SenseVoiceSmall
model, params = SenseVoiceSmall.from_pretrained(
    model="iic/SenseVoiceSmall", device="cpu"
)
print(sum(p.numel() for p in model.parameters()))  # 验证参数量
```

### 1.3 校准数据准备

使用现有的 `calib_data_large`（143 个 .npy 特征文件），为 QAT 微调提供输入分布。

---

## 阶段二：插入 FakeQuantize 节点（1~2天）

### 2.1 量化位宽策略

| 层类型 | 量化精度 | 说明 |
|-------|---------|------|
| norm1 (LayerNorm) | **FP16** 或 8-bit with per-channel | QAT 中保留高精度或 per-channel量化 |
| norm2 (LayerNorm) | INT8 | 普通 8-bit |
| self_attn MatMul | INT8 | 标准量化 |
| ffn Linear | INT8 | 标准量化 |
| ctc_lo (Linear) | INT8 | 标准量化 |

### 2.2 代码实现

创建 `qat_sensevoice.py`，核心逻辑：

```python
import torch
import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules import \
    TensorQuantizer
from funasr.models.sense_voice.model import SenseVoiceSmall

# 自定义 LayerNorm，支持 per-channel 量化
class QuantLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self._input_quantizer = TensorQuantizer(
            quant_nn.QuantConv2d.default_quant_desc_input
        )
    
    def forward(self, x):
        x = self._input_quantizer(x)
        return super().forward(x)

def replace_module(model, condition_fn, new_module_fn):
    """递归替换满足条件的模块"""
    for name, child in model.named_children():
        if condition_fn(name, child):
            setattr(model, name, new_module_fn(child))
        else:
            replace_module(child, condition_fn, new_module_fn)
    return model

def prepare_qat_model(model):
    """插入 FakeQuantize 节点到所有 Linear 和 Conv1d"""
    # 1. 替换 nn.Linear → QuantLinear
    model = replace_module(
        model,
        lambda n, m: isinstance(m, torch.nn.Linear),
        lambda m: quant_nn.QuantLinear.from_module(m)
    )
    # 2. 替换 nn.Conv1d → QuantConv1d
    model = replace_module(
        model,
        lambda n, m: isinstance(m, torch.nn.Conv1d),
        lambda m: quant_nn.QuantConv1d.from_module(m)
    )
    # 3. 可选择：norm1 层替换为 QuantLayerNorm (per-channel)
    #    norm2 层保持原样（INT8 即可）
    # 4. 启用量化
    model = model.cpu()
    model.eval()
    # 打开所有量化器的 calibrate 模式
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.enable_calib()
            module.disable_quant()  # 先收集统计量，不量化
    return model

# 加载原始模型
model, params = SenseVoiceSmall.from_pretrained(
    model="iic/SenseVoiceSmall", device="cpu"
)

# 应用 QAT 修改
model = prepare_qat_model(model)

# 重定向 forward（参考 export_onnx.py 的修改方式）
model.forward = modified_forward  # 使用原 export 中的 wrapper
model.encoder.forward = encoder_forward
```

### 2.3 关键修改点

| 文件 | 修改内容 |
|------|---------|
| `qat_sensevoice.py` | 主脚本，加载模型、插入 QAT 节点、训练循环 |
| `model.py` (funasr) | 无需修改源码，通过 monkey-patch 替换层 |
| `export_qat_onnx.py` | 导出 QAT 后的 ONNX |

---

## 阶段三：Calibration + QAT 微调（2~3天）

### 3.1 校准统计量收集

```python
def calibrate(model, calib_loader, num_steps=100):
    """用校准数据收集每层的激活值统计量"""
    model.eval()
    for i, (x, x_length, language, text_norm) in enumerate(calib_loader):
        if i >= num_steps:
            break
        with torch.no_grad():
            _ = model(x, x_length, language, text_norm)
    # 关闭校准模式，开启量化
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.disable_calib()
            module.enable_quant()
```

### 3.2 微调策略

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 | 2e-5（比原始训练低 10x） |
| 批量大小 | 8~16（取决于内存） |
| 微调步数 | **500 ~ 1000** 步 |
| 损失函数 | CTC Loss |
| 冻结策略 | 前 40 层 frozen，后 30 层 + norm 层可训练 |
| 混合精度 | 不需要（QAT 本身模拟量化） |

```python
def qat_finetune(model, train_loader, val_loader, num_steps=500):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    
    for step, batch in enumerate(train_loader):
        x, x_length, text, text_length, language, text_norm = batch
        optimizer.zero_grad()
        
        # 前向（QAT 自动模拟量化）
        logits = model(x, x_length, language, text_norm)
        
        # CTC Loss
        input_lengths = torch.full(
            (x.size(0),), logits.size(1), dtype=torch.int32
        )
        loss = ctc_loss(logits.permute(1, 0, 2), text, input_lengths, text_length)
        
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            cos_sim = evaluate_cosine(model, val_loader)
            print(f"Step {step}: loss={loss:.4f}, cos={cos_sim:.6f}")
        
        if step >= num_steps:
            break
```

### 3.3 精度验证

每次验证时：
1. 用 QAT 模型推理（FP32 但模拟 INT8）
2. 导出 ONNX → 转 RKNN → 板端推理
3. 计算 cosine similarity vs FP16

关键验证点：

| 验证点 | 预期 cos | 动作 |
|-------|---------|------|
| QAT 前 (纯 FP32) | 1.0 | 基线确认 |
| 插入 FakeQuantize 后 (Calibrate 模式) | ~0.98 | 确认量化误差可观测 |
| QAT 微调 100 步 | ~0.985 | 趋势向上 |
| QAT 微调 500 步 | ~0.990 | 接近目标 |
| QAT 微调 1000 步 | 0.990+ | 达标 |

---

## 阶段四：导出与部署（1天）

### 4.1 导出 QAT-ONNX

```python
def export_qat_onnx(model, output_path="sensevoice_qat.onnx"):
    """导出带 FakeQuantize 节点的 ONNX"""
    # 1. fold 量化参数（将 scale/zero_point 融合到权重）
    model.eval()
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    # 2. 导出 ONNX
    x = torch.randn(1, 124, 560)
    x_length = torch.tensor([100], dtype=torch.int32)
    language = torch.tensor([3], dtype=torch.int32)
    text_norm = torch.tensor([15], dtype=torch.int32)
    
    torch.onnx.export(
        model,
        (x, x_length, language, text_norm),
        output_path,
        opset_version=13,
        input_names=["x", "x_length", "language", "text_norm"],
        output_names=["logits"],
    )
```

### 4.2 转 RKNN + 测试

```bash
# 用现有 convert.py 转 RKNN
python3 convert.py sensevoice_qat.onnx rk3588 i8 sensevoice_qat.rknn

# 推送板端测试
adb push sensevoice_qat.rknn /data/
adb shell "cd /data && python3 test_ser_qat.py"
```

### 4.3 关键风险点

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| ONNX 导出时 FakeQuantize 被 fold 掉 | 中等 | 导出后检查 ONNX 图中是否包含 QuantizeLinear/DequantizeLinear 节点 |
| RKNN 不识别 QAT 导出的节点 | 高 | 先用简化 ONNX（onnxsim）再转；使用 RKNN toolkit 的 `do_quantization=False` 跳过二次量化 |
| QAT 精度不达标（<0.99） | 低 | 增加微调步数、提高 learning rate、调大校准集 |
| 板端延迟超过 170ms | 低 | 确认所有算子运行在 NPU 而非 CPU |

---

## 阶段五：回退方案（1天）

如果 QAT 全流程遇到不可克服的障碍，回退方案：

| 优先级 | 方案 | 预期精度 | 工作量 |
|-------|------|---------|-------|
| 1 | auto_hybrid + 更优校准数据（尝试不同分布的数据） | 0.990 | 1天 |
| 2 | 手工拆分模型：Encoders 0~49 INT8 + tp_encoders FP16 | 0.985 | 2天 |
| 3 | 接受 INT8 精度 cos=0.980，通过业务后处理弥补 | 0.980 | 0 |

---

## 时间线汇总

| 阶段 | 内容 | 预计耗时 | 交付物 |
|------|------|---------|-------|
| 一 | 环境搭建 + 模型加载验证 | 1天 | 可运行的 QAT 环境 |
| 二 | 插入 FakeQuantize + 导出 ONNX | 1~2天 | `qat_sensevoice.py` |
| 三 | Calibration + 微调 + 验证 | 2~3天 | QAT 模型 checkpoint |
| 四 | 导出 → 转 RKNN → 板端测试 | 1天 | `sensevoice_qat.rknn` |
| 五 | 回退方案（如有需要） | 1天 | 备用方案模型 |

**最少耗时：5 天（全流程顺利）**
**最大耗时：8 天（含问题排查）**

---

## 附录：所需文件清单

### 已有资源
| 文件 | 路径 | 说明 |
|------|------|------|
| 原始 checkpoint | `/home/ubuntu/.cache/modelscope/hub/models/iic/SenseVoiceSmall/model.pt` | 1.8B 参数权重 |
| 模型配置 | `/home/ubuntu/.cache/modelscope/hub/models/iic/SenseVoiceSmall/config.yaml` | 超参数配置 |
| 校准数据（大） | `/home/ubuntu/桌面/ai/calib_data_large/` | 143 个 .npy 文件 |
| 校准数据（真实） | `/home/ubuntu/桌面/ai/calib_data_real/` | 13 个真实情感特征 |
| ONNX 导出脚本 | `/home/ubuntu/桌面/ai/lubancat_ai_manual_code/example/sense-voice/tools/export_onnx.py` | 参考实现 |

### 需新建文件
| 文件 | 说明 |
|------|------|
| `qat_sensevoice.py` | 主脚本：加载模型、插入 QAT 节点、训练循环 |
| `export_qat_onnx.py` | 导出 QAT 后的 ONNX 模型 |
| `train_ctc_dataset.py` | 训练数据加载器（将音频转 CTC 训练格式） |
| `test_ser_qat_board.py` | 板端测试脚本 |

### 训练数据格式要求

CTC 训练需要 `(audio_features, text_tokens)` 对：
- audio_features: (1, 100, 560) float32
- text_tokens: (1, seq_len) int32

当前已有 EPD 特征文件，但缺少对应的文本标注。需要从原始的音频数据生成标注，或使用合成数据。

---

## 决策检查点

在执行过程中遇到以下情况需要暂停并评估：

1. **第 2 天结束**：如果 ONNX 导出后 FakeQuantize 节点丢失，评估是否需要切换方案
2. **第 4 天结束**：如果 QAT 微调 500 步后 cos < 0.985，检查训练数据质量或调整学习率
3. **第 5 天结束**：如果 RKNN 转换后精度 < 0.990，启动回退方案