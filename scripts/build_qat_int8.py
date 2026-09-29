#!/usr/bin/env python3
"""
Build QAT-trained ONNX → RKNN (INT8) model
Uses existing .npy calibration data from calib_data/data_real/
"""

import os
import sys
import time
import numpy as np
from rknn.api import RKNN

ONNX_PATH = "models/qat_model_fp16.onnx"
RKNN_PATH = "models/qat_ser_int8.rknn"
CALIB_DIR = "calib_data/data_real"
DATASET_PATH = os.path.join(CALIB_DIR, "dataset_qat.txt")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def create_dataset_txt():
    if not os.path.exists(CALIB_DIR):
        log(f"ERROR: {CALIB_DIR} not found!")
        return False
    
    npy_files = sorted([f for f in os.listdir(CALIB_DIR) if f.endswith('.npy')])
    if not npy_files:
        log(f"ERROR: No .npy files in {CALIB_DIR}!")
        return False
    
    with open(DATASET_PATH, 'w') as f:
        for fn in npy_files:
            f.write(os.path.abspath(os.path.join(CALIB_DIR, fn)) + '\n')
    
    log(f"Created {DATASET_PATH} with {len(npy_files)} samples")
    return True

def build_rknn():
    log("=" * 60)
    log("QAT Model: ONNX → RKNN INT8")
    log("=" * 60)
    
    if not os.path.exists(ONNX_PATH):
        log(f"ERROR: {ONNX_PATH} not found!")
        return False
    
    onnx_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    log(f"ONNX: {ONNX_PATH} ({onnx_size:.2f} MB)")
    
    if not create_dataset_txt():
        return False
    
    t0 = time.time()
    
    log("Initializing RKNN...")
    rknn = RKNN(verbose=True)
    
    log("Configuring (rk3588, w8a8)...")
    ret = rknn.config(
        target_platform="rk3588",
        optimization_level=3,
        quantized_dtype="w8a8",
    )
    if ret != 0:
        log("Config failed!")
        return False
    
    log(f"Loading ONNX: {ONNX_PATH}")
    ret = rknn.load_onnx(model=ONNX_PATH)
    if ret != 0:
        log("Load ONNX failed!")
        return False
    
    log("Building RKNN (do_quantization=True)...")
    ret = rknn.build(do_quantization=True, dataset=DATASET_PATH)
    if ret != 0:
        log(f"Build failed! ret={ret}")
        return False
    
    log(f"Exporting RKNN: {RKNN_PATH}")
    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0:
        log("Export failed!")
        return False
    
    rknn_size = os.path.getsize(RKNN_PATH) / (1024 * 1024)
    log(f"RKNN model: {RKNN_PATH} ({rknn_size:.2f} MB)")
    log(f"Total time: {time.time()-t0:.0f}s")
    
    rknn.release()
    return True

if __name__ == "__main__":
    success = build_rknn()
    sys.exit(0 if success else 1)