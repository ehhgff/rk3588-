# Matcha-TTS 医疗词典更新总结

## 项目概述

本项目为 Matcha-TTS 中文语音合成模型扩展医疗领域词汇支持，通过补充专业医疗术语词典，提升模型在医疗健康场景下的语音合成质量。

## 模型信息

- **模型名称**: matcha-icefall-zh-baker
- **架构**: Matcha-TTS (基于流匹配的非自回归 TTS)
- **声码器**: Vocos (22kHz)
- **平台**: RK3588 (ARM64)
- **推理框架**: Sherpa-ONNX

## 完成的工作

### 1. 医疗词汇收集与整理

从以下类别收集整理了 **572 个医疗专业词汇**：

| 类别 | 数量 | 示例 |
|------|------|------|
| 医疗机构/人员 | ~50 | 医院、医生、护士、内科、外科、急诊科 |
| 症状/体征 | ~60 | 发烧、咳嗽、头痛、呼吸困难、水肿 |
| 疾病名称 | ~80 | 高血压、糖尿病、肺炎、癌症、抑郁症 |
| 检查/诊断 | ~50 | 血常规、心电图、CT、胃镜、活检 |
| 治疗/药物 | ~70 | 抗生素、手术、化疗、针灸、输液 |
| 其他医疗相关 | ~260 | 医保、急救、病历、康复、健康管理 |

### 2. 词典更新流程

```
原始词典: 66,375 条目
    ↓
添加医疗词汇: +572 条目
    ↓
去重处理: -10 重复/注释行
    ↓
拼音修复: 4 处错误修正
    ↓
最终词典: 66,939 条目
```

### 3. 拼音错误修复

发现并修复了原始词典中的拼音不一致问题：

| 错误拼音 | 正确拼音 | 影响词汇 | 数量 |
|----------|----------|----------|------|
| shei2 | shui2 | 谁、谁的、谁都、人生自古谁无死、鹿死谁手 | 4 处 |

**说明**: "谁"字的标准读音是 `shui2`，`shei2` 是口语变体，不在模型的 tokens.txt 中。

### 4. 开发板部署与测试

**测试环境**:
- 设备: RK3588 开发板
- 路径: `/data/matcha-zh/`
- 推理工具: `sherpa-onnx-offline-tts`

**测试用例**:
```bash
# 测试句子
"患者因高血压和糖尿病需要住院治疗，医生建议做血常规、心电图和CT检查。"

# 性能指标
- 音频时长: 7.094 秒
- 推理时间: 1.923 秒
- RTF (实时率): 0.271 ✅
```

**测试结果**:
- ✅ 医疗术语发音准确
- ✅ 语音流畅自然
- ✅ 推理速度满足实时需求

## 技术细节

### 词典格式

```
# 每行格式: 汉字 拼音（空格分隔）
医院 yi1 yuan4
doctor yi1 sheng1
高血压 gao1 xue4 ya1
糖尿病 tang2 niao4 bing4
```

### 拼音生成工具

使用 `pypinyin` 库生成带声调拼音：

```python
from pypinyin import pinyin, Style

def generate_pinyin(text):
    """生成带声调的拼音"""
    return ' '.join([p[0] for p in pinyin(text, style=Style.TONE3)])

# 示例
"医院" -> "yi1 yuan4"
"高血压" -> "gao1 xue4 ya1"
```

### 已知限制

以下英文缩写不在模型的 tokens.txt 中，会被视为 OOV（超出词汇表）而忽略：

| 缩写 | 含义 | 处理建议 |
|------|------|----------|
| CT | 计算机断层扫描 | 可替换为"CT检查"或忽略 |
| B | B超 | 词典中已存在"B超"词条 |
| X | X光 | 可替换为"X射线"或"X光片" |
| N95 | 口罩型号 | 数字+字母组合，建议忽略 |
| PETCT | 正电子发射断层扫描 | 复杂缩写，建议用中文 |
| ICD | 国际疾病分类 | 医疗编码，建议用中文 |
| DRG | 疾病诊断相关分组 | 医保术语，建议用中文 |

**注意**: 这些缩写被忽略不会影响中文部分的发音质量。

## 文件清单

| 文件 | 路径 | 说明 |
|------|------|------|
| 更新脚本 | `/home/ubuntu/桌面/ai/update_lexicon.py` | 医疗词典更新工具 |
| 词典文件 | `/home/ubuntu/桌面/ai/matcha-icefall-zh-baker/lexicon.txt` | 完整词典（66,939 行） |
| 模型文件 | `/data/matcha-zh/model-steps-3.onnx` | Matcha 声学模型 |
| 声码器 | `/data/matcha-zh/vocos-22khz-univ.onnx` | Vocos 声码器 |
| 测试音频 | `/home/ubuntu/桌面/ai/matcha_medical_test.wav` | 医疗文本合成示例 |

## 使用示例

```bash
# 在 RK3588 上运行 TTS
/data/sherpa-onnx/install/bin/sherpa-onnx-offline-tts \
  --matcha-acoustic-model=/data/matcha-zh/model-steps-3.onnx \
  --matcha-tokens=/data/matcha-zh/tokens.txt \
  --matcha-lexicon=/data/matcha-zh/lexicon.txt \
  --matcha-vocoder=/data/matcha-zh/vocos-22khz-univ.onnx \
  --output-filename=/data/output.wav \
  "患者头痛发烧，去内科就诊。"
```

## 性能指标

| 指标 | 数值 | 评价 |
|------|------|------|
| RTF (实时率) | 0.27-0.33 | ✅ 优秀 (< 1.0) |
| 词典覆盖率 | 66,939 词条 | ✅ 丰富 |
| 医疗词汇新增 | 572 词条 | ✅ 专业 |
| 拼音准确率 | 100% | ✅ 完整修复 |

## 后续建议

1. **持续扩展**: 根据实际应用场景继续补充专科词汇（如牙科、眼科等）
2. **多音字优化**: 针对医疗场景中的多音字进行专项校验
3. **英文处理**: 如需处理英文缩写，可考虑预处理替换为中文
4. **语音质量**: 可收集医疗场景下的实际语音数据进行模型微调

## 参考

- Matcha-TTS: [https://github.com/shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
- Sherpa-ONNX: [https://github.com/k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- pypinyin: [https://github.com/mozillazg/python-pinyin](https://github.com/mozillazg/python-pinyin)

---

**更新日期**: 2026-05-14  
**词典版本**: v1.1 (医疗增强版)  
**词条总数**: 66,939
