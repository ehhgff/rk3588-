#!/usr/bin/env python3
"""
量化两个子模型:
1. encoder → INT8
2. head → FP16 (不量化)
"""
import os, sys, time
from rknn.api import RKNN

ENC_ONNX = "models/sensevoice_encoder_ctc_100f_encoder.onnx"
HEAD_ONNX = "models/sensevoice_encoder_ctc_100f_head.onnx"
DATASET = "calib_data/data_real/dataset.txt"
ENC_RKNN = "models/sensevoice_encoder_ctc_100f_enc_int8.rknn"
HEAD_RKNN = "models/sensevoice_encoder_ctc_100f_head_fp16.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def quantize(onnx_path, rknn_path, do_quant, label):
    log(f"\n{'='*50}")
    log(f"量化: {label}")
    log(f"  ONNX: {onnx_path}")
    log(f"  RKNN: {rknn_path}")
    log(f"  量化: {'INT8' if do_quant else 'FP16 (不量化)'}")
    log('='*50)
    
    t0 = time.time()
    rknn = RKNN(verbose=True)
    rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
    rknn.load_onnx(model=onnx_path)
    
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET if do_quant else None)
    if ret != 0:
        log(f"  ❌ build failed: {ret}")
        rknn.release()
        return False
    
    ret = rknn.export_rknn(rknn_path)
    if ret == 0:
        sz = os.path.getsize(rknn_path) / (1024*1024)
        log(f"  ✅ {sz:.0f}MB ({time.time()-t0:.0f}s)")
        rknn.release()
        return True
    
    log(f"  ❌ export failed: {ret}")
    rknn.release()
    return False

# 1. Encoder INT8
ok = quantize(ENC_ONNX, ENC_RKNN, do_quant=True, label="Encoder INT8")
if not ok:
    sys.exit(1)

# 2. Head FP16
ok = quantize(HEAD_ONNX, HEAD_RKNN, do_quant=False, label="Head FP16")
if not ok:
    sys.exit(1)

# 汇总
log("\n\n=== 结果 ===")
for f in [ENC_RKNN, HEAD_RKNN]:
    if os.path.exists(f):
        sz = os.path.getsize(f) / (1024*1024)
        log(f"  {f}: {sz:.0f} MB")
    else:
        log(f"  {f}: ❌ 不存在")

# 验证 ONNX 输入输出
import onnx
enc_onnx = onnx.load(ENC_ONNX)
head_onnx = onnx.load(HEAD_ONNX)
log(f"\n=== 验证连接性 ===")
log(f"  Encoder输出: {[o.name for o in enc_onnx.graph.output]}")
log(f"  Head输入: {[i.name for i in head_onnx.graph.input]}")
log(f"  {'✅ 匹配!' if [o.name for o in enc_onnx.graph.output] == [i.name for i in head_onnx.graph.input] else '❌ 不匹配!'}")