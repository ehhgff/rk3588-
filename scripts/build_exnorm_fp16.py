#!/usr/bin/env python3
"""
hybrid_quantization 两步法:
1. step1 生成量化配置
2. 修改配置: exNorm 层保留 FP16, 其余 INT8
3. step2 导出模型
"""
import os, sys, time, re, json
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
OUTPUT = "models/sensevoice_encoder_ctc_100f_exnorm_fp16.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# Step 1: hybrid_quantization_step1 (NO custom_hybrid, just analyze)
# ============================================================
log("=" * 60)
log("Step 1: 生成量化配置")
log("=" * 60)

if os.path.exists(OUTPUT):
    log(f"✅ 已存在: {OUTPUT}")
    sys.exit(0)

rknn = RKNN(verbose=True)
rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
rknn.load_onnx(model=ONNX_PATH)

ret = rknn.hybrid_quantization_step1(
    dataset=DATASET,
    proposal=False,  # 不使用自动建议
    # 不使用 custom_hybrid
)
if ret != 0:
    log(f"❌ step1 失败: ret={ret}")
    rknn.release()
    sys.exit(1)

# 找到生成的文件
model_file = None
data_file = None
cfg_file = None
for f in os.listdir('.'):
    if f.endswith('.model'):
        model_file = f
    elif f.endswith('.data'):
        data_file = f
    elif f.endswith('.quantization.cfg'):
        cfg_file = f

log(f"model: {model_file} ({os.path.getsize(f)>>20 if model_file else 0}MB)")
log(f"data:  {data_file} ({os.path.getsize(f)>>20 if data_file else 0}MB)")
log(f"cfg:   {cfg_file}")

if not all([model_file, data_file, cfg_file]):
    log("❌ 缺少必要文件")
    sys.exit(1)

# ============================================================
# Step 2: 读取并修改配置文件
# ============================================================
log("\n" + "=" * 60)
log("Step 2: 修改配置 - exNorm 层用 FP16")
log("=" * 60)

with open(cfg_file, 'r') as f:
    cfg_content = f.read()

# 解析配置文件
lines = cfg_content.split('\n')
log(f"配置文件共 {len(lines)} 行")

# 统计原始配置
exnorm_count = 0
for line in lines:
    if 'exNorm' in line or 'exnorm' in line.lower():
        exnorm_count += 1
log(f"exNorm 节点数: {exnorm_count}")

# 找到量化配置行并修改 exNorm 的 quantized_dtype
# 配置文件格式类似于 JSON
# 需要找到 exNorm 层对应的量化配置

# 打印前100行和后100行
log("\n" + "=" * 40)
log("前80行配置:")
for i, line in enumerate(lines[:80]):
    log(f"  [{i:3d}] {line}")
log("\n后80行配置:")
for i, line in enumerate(lines[-80:]):
    log(f"  [{len(lines)-80+i:3d}] {line}")

log("\n包含 'exNorm' 的行:")
for i, line in enumerate(lines):
    if 'exNorm' in line:
        log(f"  [{i:4d}] {line.strip()[:120]}")

rknn.release()
log("\nStep 1+2 完成. 需要手动修改 cfg 后运行 step2.")