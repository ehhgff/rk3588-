#!/usr/bin/env python3
"""
快速测试 auto_hybrid_cos_thresh 和 quantized_hybrid_level
"""
import os, sys, time, json
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def try_build(label, config_kwargs, use_auto_hybrid=False):
    out_path = f"sensevoice_encoder_ctc_100f_{label}.rknn"
    if os.path.exists(out_path):
        sz = os.path.getsize(out_path) / (1024*1024)
        log(f"⚠️ {label}: 已存在 ({sz:.0f}MB)")
        return sz
    
    log(f"\n>>> {label}: {json.dumps(config_kwargs)}")
    t0 = time.time()
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588", **config_kwargs)
    rknn.load_onnx(model=ONNX_PATH)
    
    if use_auto_hybrid:
        ret = rknn.build(do_quantization=True, dataset=DATASET, auto_hybrid=True)
    else:
        ret = rknn.build(do_quantization=True, dataset=DATASET)
    
    if ret != 0:
        log(f"  ❌ build failed")
        rknn.release()
        return None
    
    ret = rknn.export_rknn(out_path)
    t1 = time.time()
    
    if ret == 0:
        sz = os.path.getsize(out_path) / (1024*1024)
        log(f"  ✅ {sz:.0f}MB ({t1-t0:.0f}s)")
        rknn.release()
        return sz
    
    log(f"  ❌ export failed")
    rknn.release()
    return None

# --- 实验 1: quantized_hybrid_level ---
log("\n" + "=" * 50)
log("实验 1: quantized_hybrid_level")
log("=" * 50)

for level in [1, 2, 3]:
    try_build(f"hybrid_l{level}", {
        "quantized_dtype": "w8a8",
        "optimization_level": 3,
        "quantized_hybrid_level": level,
    })

# --- 实验 2: auto_hybrid_cos_thresh ---
log("\n" + "=" * 50)
log("实验 2: auto_hybrid_cos_thresh")
log("=" * 50)

for thresh in [0.95, 0.99, 0.999]:
    try_build(f"auto_cos{thresh}", {
        "quantized_dtype": "w8a8",
        "optimization_level": 3,
        "auto_hybrid_cos_thresh": thresh,
    }, use_auto_hybrid=True)

# --- 汇总 ---
log("\n\n=== 模型大小对比 ===")
refs = {
    "int8_240MB": 240,
    "fp16_460MB": 460,
    "hybrid_auto_417MB": 417,
}
for name, ref in refs.items():
    log(f"  {name}")

for f in sorted(os.listdir('.')):
    if f.endswith('.rknn') and ('hybrid' in f or 'auto' in f or 'int8' in f):
        sz = os.path.getsize(f) / (1024*1024)
        log(f"  {f:50s}: {sz:.0f} MB")