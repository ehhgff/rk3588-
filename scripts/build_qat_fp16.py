#!/usr/bin/env python3
"""
Build QAT FP16 reference (no quantization, same structure as QAT model)
Compares QAT INT8 vs QAT FP16 to get valid cosine similarity
"""

import os, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/qat_model_fp16.onnx"
RKNN_FP16 = "models/qat_ser_fp16.rknn"

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def build():
    log("=" * 60)
    log("Build QAT FP16 reference (no quantization)")
    log("=" * 60)
    
    if not os.path.exists(ONNX_PATH):
        log(f"ERROR: {ONNX_PATH} not found!")
        return False

    size = os.path.getsize(ONNX_PATH) / (1024*1024)
    log(f"ONNX: {ONNX_PATH} ({size:.1f} MB)")

    t0 = time.time()
    rknn = RKNN(verbose=True)
    
    log("Configuring...")
    ret = rknn.config(target_platform="rk3588", optimization_level=3)
    if ret != 0:
        log("Config failed!")
        return False

    log("Loading ONNX...")
    ret = rknn.load_onnx(model=ONNX_PATH)
    if ret != 0:
        log("Load ONNX failed!")
        return False

    log("Building (no quantization)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        log("Build failed!")
        return False

    log(f"Exporting: {RKNN_FP16}")
    ret = rknn.export_rknn(RKNN_FP16)
    if ret != 0:
        log("Export failed!")
        return False

    rknn_size = os.path.getsize(RKNN_FP16) / (1024*1024)
    log(f"RKNN FP16: {RKNN_FP16} ({rknn_size:.1f} MB)")
    log(f"Time: {time.time()-t0:.0f}s")
    rknn.release()
    return True

if __name__ == "__main__":
    build()