#!/usr/bin/env python3
"""
使用 auto_hybrid 以不同阈值保留敏感层为 FP16。
auto_hybrid_cos_thresh=0.95 → 仅当量化后 cosine similarity < 0.95 才保留FP16
这样只保留 exNorm 等最敏感层，模型大小接近纯INT8。
"""
import os, time, sys
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"

# 尝试不同阈值 - 较低阈值 = 更少FP16层
thresholds = [0.95, 0.93, 0.90]

for th in thresholds:
    # 格式化阈值用于文件名
    th_str = f"{th:.2f}".replace('.', 'p')
    OUTPUT = f"models/sensevoice_encoder_ctc_100f_hybrid_th{th_str}.rknn"
    
    print(f"[{time.strftime('%H:%M:%S')}] ===== auto_hybrid threshold={th} =====", flush=True)
    
    if os.path.exists(OUTPUT):
        size = os.path.getsize(OUTPUT) / 1024 / 1024
        print(f"  已存在: {OUTPUT} ({size:.1f}MB)", flush=True)
        continue
    
    rknn = RKNN(verbose=True)
    rknn.config(
        target_platform="rk3588",
        optimization_level=3,
        quantized_dtype="w8a8",
        auto_hybrid_cos_thresh=th,
    )
    
    print(f"  load_onnx...", flush=True)
    ret = rknn.load_onnx(model=ONNX_PATH)
    assert ret == 0, f"load_onnx failed: {ret}"
    
    print(f"  build (auto_hybrid=True)...", flush=True)
    t0 = time.time()
    ret = rknn.build(do_quantization=True, dataset=DATASET, auto_hybrid=True)
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