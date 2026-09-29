#!/usr/bin/env python3
"""
SER 100帧: w8a16 混合量化
权重 INT8 (省大小) + 激活 FP16 (保精度)

比 w8a8 (INT8) 更准, 比 w16a16 (FP16) 更小
"""
import os, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
OUTPUT_RKNN = "sensevoice_encoder_ctc_100f_w8a16.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 60)
log("SER 100f: w8a16 混合量化")
log("=" * 60)
log(f"ONNX: {ONNX_PATH}")
log(f"校准集: {DATASET}")

t_start = time.time()

rknn = RKNN(verbose=True)

# 关键: quantized_dtype='w8a16' - 权重INT8, 激活FP16
ret = rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a16")
assert ret == 0, "config failed"

ret = rknn.load_onnx(model=ONNX_PATH)
assert ret == 0, "load_onnx failed"

log("\n开始 w8a16 量化...")
ret = rknn.build(do_quantization=True, dataset=DATASET)
if ret != 0:
    log(f"❌ w8a16 量化失败! ret={ret}")
    sys.exit(1)

log("\n导出模型...")
ret = rknn.export_rknn(OUTPUT_RKNN)
if ret == 0:
    size_mb = os.path.getsize(OUTPUT_RKNN) / (1024*1024)
    log(f"✅ 模型导出成功!")
    log(f"   路径: {OUTPUT_RKNN}")
    log(f"   大小: {size_mb:.1f} MB")
else:
    log(f"❌ 导出失败! ret={ret}")
    sys.exit(1)

t_elapsed = time.time() - t_start
log(f"总耗时: {t_elapsed:.0f}秒 ({t_elapsed/60:.1f}分)")
rknn.release()
log("完成!")