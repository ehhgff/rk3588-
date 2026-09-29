#!/usr/bin/env python3
"""
使用 auto_hybrid 以较低阈值保留最敏感层为 FP16。
threshold=0.95 表示只有当层量化后的 cosine similarity < 0.95 时才保留 FP16，
这样可以只保留 exNorm 等最敏感层。
"""
import os, time, sys
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"

# 尝试多个阈值，构建不同大小的模型
thresholds = [0.95, 0.97]

for th in thresholds:
    OUTPUT = f"models/sensevoice_encoder_ctc_100f_hybrid_th{th:.2f}.rknn"
    
    # 去掉小数点
    OUTPUT = OUTPUT.replace('.', '_point_')
    
    print(f"[{time.strftime('%H:%M:%S')}] ===== auto_hybrid threshold={th} =====", flush=True)
    
    if os.path.exists(OUTPUT):
        size = os.path.getsize(OUTPUT) / 1024 / 1024
        print(f"  已存在: {OUTPUT} ({size:.1f}MB)", flush=True)
        continue
    
    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform="rk3588",
        optimization_level=3,
        quantized_dtype="w8a8",
        auto_hybrid=True,
        quantized_hybrid_threshold=th,
    )
    
    print(f"  load_onnx...", flush=True)
    rknn.load_onnx(model=ONNX_PATH)
    
    print(f"  build...", flush=True)
    t0 = time.time()
    ret = rknn.build(do_quantization=True, dataset=DATASET)
    elapsed = time.time() - t0
    print(f"  build 耗时: {elapsed:.0f}s", flush=True)
    
    if ret != 0:
        print(f"  ❌ build 失败: {ret}", flush=True)
        rknn.release()
        continue
    
    rknn.export_rknn(OUTPUT)
    rknn.release()
    
    size = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"  ✅ {OUTPUT} ({size:.1f}MB)", flush=True)

print(f"\n[{time.strftime('%H:%M:%S')}] 完成!", flush=True)