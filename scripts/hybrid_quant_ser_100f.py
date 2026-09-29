#!/usr/bin/env python3
"""
SER 100帧 混合量化 (Hybrid Quantization)
使用真实音频特征作为校准集

工作流:
  1. hybrid_quantization_step1(proposal=True)
     → 逐层分析量化误差, 生成 .cfg 配置文件
  2. hybrid_quantization_step2(model_quantization_cfg=cfg)
     → 敏感层 FP16 + 其余 INT8, 导出混合量化模型
"""
import os, sys, time, json
import numpy as np
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
HYBRID_RKNN = "models/sensevoice_encoder_ctc_100f_hybrid.rknn"
HYBRID_CFG = None  # step1 生成后赋值

def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

# =========================================================
# Step 1: hybrid_quantization_step1 (proposal mode)
# =========================================================
log("=" * 60)
log("混合量化 Step 1: 逐层分析量化误差, 生成配置建议")
log("=" * 60)

log(f"ONNX: {ONNX_PATH}")
log(f"校准集: {DATASET}")
log(f"校准集大小: {len(open(DATASET).readlines()) if os.path.exists(DATASET) else 0} 个样本")

rknn = RKNN(verbose=True)
ret = rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
assert ret == 0, "config failed"

ret = rknn.load_onnx(model=ONNX_PATH)
assert ret == 0, "load_onnx failed"

log("\n运行 hybrid_quantization_step1 (proposal=True)...")
log("(正在逐层分析, 预计 5-15 分钟)")
t0 = time.time()
ret = rknn.hybrid_quantization_step1(
    dataset=DATASET,
    proposal=True,
    proposal_dataset_size=len(open(DATASET).readlines()),
)
t1 = time.time()
log(f"step1 耗时: {t1-t0:.0f}秒")

if ret != 0:
    log(f"❌ hybrid_quantization_step1 失败! ret={ret}")
    sys.exit(1)

# 查找生成的配置文件
cfg_candidates = [f for f in os.listdir('.') if f.endswith('.cfg')]
if not cfg_candidates:
    # 可能在其他路径, 试试看
    cfg_candidates = [f for f in os.listdir('.') if 'quant' in f.lower() and (f.endswith('.cfg') or f.endswith('.config'))]
if not cfg_candidates:
    log("❌ 未找到生成的配置文件!")
    log("  检查以下路径:")
    for root, dirs, files in os.walk('.'):
        for fn in files:
            if fn.endswith('.cfg'):
                log(f"  {os.path.join(root, fn)}")
    # 尝试使用模板
    log("  尝试直接创建 hybrid_quantization.cfg 模板...")
    sys.exit(1)

HYBRID_CFG = cfg_candidates[0]
log(f"\n✅ 生成的配置文件: {HYBRID_CFG}")

with open(HYBRID_CFG, 'r') as f:
    content = f.read()
log(f"配置文件大小: {len(content)} 字符")
log("\n配置文件内容 (前3000字符):")
log("-" * 40)
for line in content.split('\n')[:80]:
    log(line)
log("-" * 40)

rknn.release()
log("Step 1 完成!")

# =========================================================
# Step 2: hybrid_quantization_step2
# =========================================================
log("\n" + "=" * 60)
log("混合量化 Step 2: 应用配置, 导出混合量化模型")
log("=" * 60)

log(f"使用配置: {HYBRID_CFG}")
log(f"目标模型: {HYBRID_RKNN}")

ret = rknn.hybrid_quantization_step2(
    model_input=ONNX_PATH,
    data_input=DATASET,
    model_quantization_cfg=HYBRID_CFG,
)
if ret != 0:
    log(f"❌ hybrid_quantization_step2 失败! ret={ret}")

    # 尝试用合并的 npy 文件
    log("  尝试用合并 npy 文件...")
    all_data = []
    with open(DATASET, 'r') as f:
        for line in f:
            path = line.strip()
            data = np.load(path)
            all_data.append(data)

    data_input_path = "hybrid_calib_data.npy"
    all_data_np = np.concatenate(all_data, axis=0)
    np.save(data_input_path, all_data_np)
    log(f"  合并校准数据: {data_input_path}, shape={all_data_np.shape}")

    ret = rknn.hybrid_quantization_step2(
        model_input=ONNX_PATH,
        data_input=data_input_path,
        model_quantization_cfg=HYBRID_CFG,
    )
    if ret != 0:
        log(f"❌ hybrid_quantization_step2 再次失败! ret={ret}")
        sys.exit(1)

log("\n导出混合量化模型...")
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
log("\n" + "=" * 60)
log("混合量化完成!")
log(f"模型: {HYBRID_RKNN}")
log(f"校准集: {DATASET}")
log(f"配置: {HYBRID_CFG}")
log("=" * 60)