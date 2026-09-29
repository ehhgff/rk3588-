#!/usr/bin/env python3
"""
用修复后的 ONNX 重建 Head FP16
"""
import os, sys, time
from rknn.api import RKNN

HEAD_ONNX = "models/sensevoice_encoder_ctc_100f_head_fixed.onnx"
HEAD_RKNN = "models/sensevoice_encoder_ctc_100f_head_fp16.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("构建 Head FP16 (修复形状)")
if os.path.exists(HEAD_RKNN):
    sz = os.path.getsize(HEAD_RKNN) / (1024*1024)
    log(f"⚠️ 已存在: {sz:.0f}MB")
    exit(0)

t0 = time.time()
rknn = RKNN(verbose=True)
rknn.config(target_platform="rk3588", optimization_level=0)
ret = rknn.load_onnx(model=HEAD_ONNX)
if ret != 0:
    log(f"❌ load_onnx 失败: {ret}")
    sys.exit(1)

log("正在构建...")
ret = rknn.build(do_quantization=False)
if ret != 0:
    log(f"❌ optimization_level=0 失败: {ret}")
    log("尝试 optimization_level=0 + w8a8...")
    rknn.release()
    rknn = RKNN(verbose=True)
    rknn.config(target_platform="rk3588", optimization_level=0, quantized_dtype="w8a8")
    rknn.load_onnx(model=HEAD_ONNX)
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        log(f"❌ 仍然失败: {ret}")
        sys.exit(1)

ret = rknn.export_rknn(HEAD_RKNN)
if ret == 0:
    sz = os.path.getsize(HEAD_RKNN) / (1024*1024)
    log(f"✅ {sz:.0f}MB ({time.time()-t0:.0f}s)")
else:
    log(f"❌ 导出失败: {ret}")

rknn.release()
log("完成!")