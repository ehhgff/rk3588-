#!/usr/bin/env python3
"""
使用 custom_hybrid 构建 SER 模型：
- 所有 48 个 encoder 块的 norm2 层 → FP16
- 其余层 → INT8

从 inspect_norm_tensors.py 获取的配置
"""
import os, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
OUTPUT = "models/sensevoice_encoder_ctc_100f_norm2_fp16.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# custom_hybrid 配置：48 个 norm2 子图保持 FP16
# ============================================================
custom_hybrid = [
    ['/encoder/encoders.0/Add_output_0-rs', '/encoder/encoders.0/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.1/Add_output_0-rs', '/encoder/encoders.1/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.2/Add_output_0-rs', '/encoder/encoders.2/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.3/Add_output_0-rs', '/encoder/encoders.3/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.4/Add_output_0-rs', '/encoder/encoders.4/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.5/Add_output_0-rs', '/encoder/encoders.5/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.6/Add_output_0-rs', '/encoder/encoders.6/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.7/Add_output_0-rs', '/encoder/encoders.7/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.8/Add_output_0-rs', '/encoder/encoders.8/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.9/Add_output_0-rs', '/encoder/encoders.9/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.10/Add_output_0-rs', '/encoder/encoders.10/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.11/Add_output_0-rs', '/encoder/encoders.11/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.12/Add_output_0-rs', '/encoder/encoders.12/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.13/Add_output_0-rs', '/encoder/encoders.13/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.14/Add_output_0-rs', '/encoder/encoders.14/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.15/Add_output_0-rs', '/encoder/encoders.15/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.16/Add_output_0-rs', '/encoder/encoders.16/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.17/Add_output_0-rs', '/encoder/encoders.17/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.18/Add_output_0-rs', '/encoder/encoders.18/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.19/Add_output_0-rs', '/encoder/encoders.19/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.20/Add_output_0-rs', '/encoder/encoders.20/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.21/Add_output_0-rs', '/encoder/encoders.21/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.22/Add_output_0-rs', '/encoder/encoders.22/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.23/Add_output_0-rs', '/encoder/encoders.23/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.24/Add_output_0-rs', '/encoder/encoders.24/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.25/Add_output_0-rs', '/encoder/encoders.25/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.26/Add_output_0-rs', '/encoder/encoders.26/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.27/Add_output_0-rs', '/encoder/encoders.27/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.28/Add_output_0-rs', '/encoder/encoders.28/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.29/Add_output_0-rs', '/encoder/encoders.29/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.30/Add_output_0-rs', '/encoder/encoders.30/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.31/Add_output_0-rs', '/encoder/encoders.31/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.32/Add_output_0-rs', '/encoder/encoders.32/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.33/Add_output_0-rs', '/encoder/encoders.33/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.34/Add_output_0-rs', '/encoder/encoders.34/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.35/Add_output_0-rs', '/encoder/encoders.35/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.36/Add_output_0-rs', '/encoder/encoders.36/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.37/Add_output_0-rs', '/encoder/encoders.37/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.38/Add_output_0-rs', '/encoder/encoders.38/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.39/Add_output_0-rs', '/encoder/encoders.39/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.40/Add_output_0-rs', '/encoder/encoders.40/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.41/Add_output_0-rs', '/encoder/encoders.41/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.42/Add_output_0-rs', '/encoder/encoders.42/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.43/Add_output_0-rs', '/encoder/encoders.43/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.44/Add_output_0-rs', '/encoder/encoders.44/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.45/Add_output_0-rs', '/encoder/encoders.45/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.46/Add_output_0-rs', '/encoder/encoders.46/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.47/Add_output_0-rs', '/encoder/encoders.47/norm2/Add_1_output_0_tp-rs'],
    ['/encoder/encoders.48/Add_output_0-rs', '/encoder/encoders.48/norm2/Add_1_output_0_tp-rs'],
]

# 检查是否已经有输出
if os.path.exists(OUTPUT):
    log(f"✅ 已存在: {OUTPUT} ({os.path.getsize(OUTPUT)/1024/1024:.1f}MB)")
    reply = input("重新构建？(y/N): ")
    if reply.lower() != 'y':
        log("跳过")
        sys.exit(0)

log("=" * 60)
log(f"构建 custom_hybrid 模型: {len(custom_hybrid)} 个 norm2 子图 FP16")
log("=" * 60)

rknn = RKNN(verbose=False)

log("配置 RKNN...")
rknn.config(
    target_platform="rk3588",
    optimization_level=3,
    quantized_dtype="w8a8",
)

log(f"加载 ONNX: {ONNX_PATH}")
rknn.load_onnx(model=ONNX_PATH)

log("运行 hybrid_quantization_step1 (with custom_hybrid)...")
ret = rknn.hybrid_quantization_step1(
    dataset=DATASET,
    proposal=False,
    custom_hybrid=custom_hybrid,
)

if ret != 0:
    log(f"❌ step1 失败: ret={ret}")
    rknn.release()
    sys.exit(1)

log("✅ step1 成功")

# 列出生成的文件
for f in sorted(os.listdir('.')):
    if '100f' in f and (f.endswith('.model') or f.endswith('.data') or f.endswith('.quantization.cfg')):
        size = os.path.getsize(f)
        if size > 1024*1024:
            log(f"  生成: {f} ({size/1024/1024:.1f}MB)")
        else:
            log(f"  生成: {f} ({size/1024:.1f}KB)")

rknn.release()

# ============================================================
# Step 2
# ============================================================
model_path = "sensevoice_encoder_ctc_100f.model"
data_path = "sensevoice_encoder_ctc_100f.data"
cfg_path = "sensevoice_encoder_ctc_100f.quantization.cfg"

if not all(os.path.exists(f) for f in [model_path, data_path, cfg_path]):
    log(f"❌ 缺少文件: 检查 {model_path}, {data_path}, {cfg_path}")
    sys.exit(1)

log("\n" + "=" * 60)
log("运行 hybrid_quantization_step2...")
log("=" * 60)

rknn = RKNN(verbose=False)
ret = rknn.hybrid_quantization_step2(model_path, data_path, cfg_path)

if ret != 0:
    log(f"❌ step2 失败: ret={ret}")
    rknn.release()
    sys.exit(1)

log("✅ step2 成功")

log("导出 RKNN 模型...")
rknn.export_rknn(OUTPUT)
rknn.release()

size = os.path.getsize(OUTPUT)
log(f"✅ 模型已保存: {OUTPUT} ({size/1024/1024:.1f}MB)")