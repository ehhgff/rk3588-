#!/usr/bin/env python3
"""build with quantized_algorithm='kl_divergence' using nohup"""
import os, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"

for algo in ["kl_divergence", "gdq"]:
    out = f"models/sensevoice_encoder_ctc_100f_int8_{algo}.rknn"
    if os.path.exists(out):
        print(f"✅ {out}: {os.path.getsize(out)/(1024*1024):.0f}MB (exists)")
        continue
    
    print(f"\n{'='*50}")
    print(f"Building {algo}...")
    print(f"{'='*50}")
    sys.stdout.flush()
    
    t0 = time.time()
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588", optimization_level=3, 
                quantized_dtype="w8a8", quantized_algorithm=algo)
    assert rknn.load_onnx(ONNX_PATH) == 0
    
    ret = rknn.build(do_quantization=True, dataset=DATASET)
    if ret == 0:
        rknn.export_rknn(out)
        sz = os.path.getsize(out)/(1024*1024)
        print(f"✅ {algo}: {sz:.0f}MB ({time.time()-t0:.0f}s)")
    else:
        print(f"❌ {algo}: build failed ({ret})")
    rknn.release()
    sys.stdout.flush()