#!/usr/bin/env python3
"""
auto_hybrid_cos_thresh=0.99: 保留更多FP16层以提升精度
"""
import os, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
OUTPUT = "models/sensevoice_encoder_ctc_100f_auto_th99.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

if os.path.exists(OUTPUT):
    sz = os.path.getsize(OUTPUT)/(1024*1024)
    log(f"⚠️ 已存在: {OUTPUT} ({sz:.0f}MB)")
    exit(0)

log("=" * 50)
log("auto_hybrid_cos_thresh=0.99")
log("=" * 50)

t0 = time.time()
rknn = RKNN(verbose=True)
rknn.config(
    target_platform="rk3588",
    optimization_level=3,
    quantized_dtype="w8a8",
    auto_hybrid_cos_thresh=0.99,
)
rknn.load_onnx(model=ONNX_PATH)

log("构建中 (auto_hybrid)...")
ret = rknn.build(do_quantization=True, dataset=DATASET, auto_hybrid=True)
if ret != 0:
    log(f"❌ build 失败: {ret}")
    rknn.release()
    sys.exit(1)

ret = rknn.export_rknn(OUTPUT)
if ret == 0:
    sz = os.path.getsize(OUTPUT) / (1024*1024)
    log(f"\n✅ {sz:.0f}MB ({time.time()-t0:.0f}s)")
else:
    log(f"❌ 导出失败: {ret}")

rknn.release()