#!/usr/bin/env python3
"""
SER 100帧: custom_hybrid 混合量化
精确指定哪些子图保留 FP16 精度
"""
import os, sys, time, json
import numpy as np
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
HYBRID_RKNN = "models/sensevoice_encoder_ctc_100f_hybrid_custom.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 60)
log("SER 100f: custom_hybrid 混合量化")
log("=" * 60)
log(f"ONNX: {ONNX_PATH}")
log(f"校准集: {DATASET}")

# =========================================================
# Step 1: hybrid_quantization_step1
# 使用 custom_hybrid 指定 FP16 子图
# 格式: [[input_name, output_name], ...]
# =========================================================
log("\nStep 1: 生成混合量化配置...")

rknn = RKNN(verbose=True)
rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
rknn.load_onnx(model=ONNX_PATH)

# 只用 ctc_lo (最终分类层) 做 FP16
# ctc_lo 输入来自 tp_norm 输出
# 格式: [输入tensor名, 输出tensor名]
custom_hybrid = [
    ["/encoder/tp_norm/Mul_output_0", "ctc_logits"],  # tp_norm→ctc_lo→LogSoftmax
]

log(f"custom_hybrid: {json.dumps(custom_hybrid)}")
log("(保留 ctc_lo + LogSoftmax 为 FP16, 其余 INT8)")

# 不使用 proposal, 只使用 custom_hybrid
ret = rknn.hybrid_quantization_step1(
    dataset=DATASET,
    proposal=False,
    custom_hybrid=custom_hybrid,
)

if ret != 0:
    log(f"❌ hybrid_quantization_step1 失败! ret={ret}")
    log("尝试其他 custom_hybrid 配置...")
    rknn.release()
    sys.exit(1)

# 检查生成的配置文件
cfg_files = [f for f in os.listdir('.') if f.endswith('.quantization.cfg')]
if not cfg_files:
    # 查看当前目录下所有可能的文件
    log("当前目录下文件:")
    for f in sorted(os.listdir('.')):
        sz = os.path.getsize(f) if os.path.isfile(f) else 0
        log(f"  {f} ({sz} bytes)")
    sys.exit(1)

quant_cfg = cfg_files[0]
log(f"✅ 量化配置: {quant_cfg}")

with open(quant_cfg, 'r') as f:
    content = f.read()
log(f"配置内容 ({len(content)} chars):")
for line in content.split('\n')[:30]:
    log(f"  {line}")

model_file = os.path.splitext(ONNX_PATH)[0] + ".model"
data_file = os.path.splitext(ONNX_PATH)[0] + ".data"

if not os.path.exists(model_file):
    log(f"⚠️ model file 不存在: {model_file}")
    # 查找自动生成的 .model 文件
    model_candidates = [f for f in os.listdir('.') if f.endswith('.model')]
    if model_candidates:
        model_file = model_candidates[0]
        log(f"使用: {model_file}")
    else:
        log("❌ 未找到 .model 文件")
        sys.exit(1)

if not os.path.exists(data_file):
    data_candidates = [f for f in os.listdir('.') if f.endswith('.data')]
    if data_candidates:
        data_file = data_candidates[0]
        log(f"使用: {data_file}")
    else:
        log("⚠️ .data 文件不存在, 尝试 fallback...")
        rknn.release()
        sys.exit(1)

# =========================================================
# Step 2: hybrid_quantization_step2
# =========================================================
log("\nStep 2: 生成混合量化模型...")
rknn.release()

# 重新创建 RKNN 实例
rknn = RKNN(verbose=True)
ret = rknn.hybrid_quantization_step2(
    model_input=model_file,
    data_input=data_file,
    model_quantization_cfg=quant_cfg,
)
if ret != 0:
    log(f"❌ hybrid_quantization_step2 失败! ret={ret}")
    sys.exit(1)

log("导出模型...")
ret = rknn.export_rknn(HYBRID_RKNN)
if ret == 0:
    size_mb = os.path.getsize(HYBRID_RKNN) / (1024*1024)
    log(f"✅ 混合量化模型导出成功!")
    log(f"   路径: {HYBRID_RKNN}")
    log(f"   大小: {size_mb:.1f} MB")
else:
    log(f"❌ 导出失败! ret={ret}")
    sys.exit(1)

rknn.release()
log("\n混合量化完成!")