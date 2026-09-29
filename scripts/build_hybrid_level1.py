#!/usr/bin/env python3
"""
尝试 quantized_hybrid_level 进行混合量化
这是最简洁的方案: 直接量化时保留部分层为 FP16
"""
import os, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
OUTPUT = "models/sensevoice_encoder_ctc_100f_mixed_l1.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

if os.path.exists(OUTPUT):
    log(f"⚠️ 已存在: {OUTPUT}")
    exit(0)

log("=" * 50)
log("混合量化: quantized_hybrid_level=1")
log("=" * 50)

t0 = time.time()
rknn = RKNN(verbose=True)
rknn.config(
    target_platform="rk3588",
    optimization_level=3,
    quantized_dtype="w8a8",
    quantized_hybrid_level=1,
)
ret = rknn.load_onnx(model=ONNX_PATH)
assert ret == 0, "load_onnx failed"

log("正在构建 (混合量化 level=1)...")
ret = rknn.build(do_quantization=True, dataset=DATASET)
if ret != 0:
    log(f"❌ build 失败: {ret}")
    rknn.release()
    sys.exit(1)

ret = rknn.export_rknn(OUTPUT)
if ret == 0:
    sz = os.path.getsize(OUTPUT) / (1024*1024)
    log(f"\n✅ {OUTPUT} ({sz:.0f}MB) 耗时={time.time()-t0:.0f}s")
else:
    log(f"❌ 导出失败: {ret}")

rknn.release()