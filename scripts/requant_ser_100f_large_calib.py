#!/usr/bin/env python3
"""
SER 100帧: 使用大校准集 (143个) 重新量化 INT8
"""
import os, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_large/dataset.txt"
OUTPUT_RKNN = "models/sensevoice_encoder_ctc_100f_int8_large.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 60)
log("SER 100f: INT8 量化 (大校准集)")
log("=" * 60)

with open(DATASET) as f:
    n = len(f.readlines())
log(f"校准集: {DATASET} ({n} 个样本)")

t_start = time.time()

rknn = RKNN(verbose=True)
ret = rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
assert ret == 0, "config failed"

ret = rknn.load_onnx(model=ONNX_PATH)
assert ret == 0, "load_onnx failed"

log("\n开始量化...")
ret = rknn.build(do_quantization=True, dataset=DATASET)
if ret != 0:
    log(f"❌ 量化失败! ret={ret}")
    sys.exit(1)

log("\n导出模型...")
ret = rknn.export_rknn(OUTPUT_RKNN)
if ret == 0:
    size_mb = os.path.getsize(OUTPUT_RKNN) / (1024*1024)
    log(f"✅ 导出成功! {OUTPUT_RKNN} ({size_mb:.1f} MB)")
else:
    log(f"❌ 导出失败! ret={ret}")
    sys.exit(1)

t_elapsed = time.time() - t_start
log(f"总耗时: {t_elapsed:.0f}秒 ({t_elapsed/60:.1f}分)")
rknn.release()
log("完成!")