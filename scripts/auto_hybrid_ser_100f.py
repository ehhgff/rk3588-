#!/usr/bin/env python3
"""
SER 100帧: 自动混合量化 (auto_hybrid=True)
直接用 build 的 auto_hybrid 功能, 自动识别敏感层保留 FP16
"""
import os, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
HYBRID_RKNN = "models/sensevoice_encoder_ctc_100f_hybrid_auto.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 60)
log("SER 100f: 自动混合量化 (auto_hybrid=True)")
log("=" * 60)
log(f"ONNX: {ONNX_PATH}")
log(f"校准集: {DATASET}")
with open(DATASET) as f:
    log(f"样本数: {len(f.readlines())}")

t_start = time.time()

rknn = RKNN(verbose=True)
ret = rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
assert ret == 0, "config failed"

ret = rknn.load_onnx(model=ONNX_PATH)
assert ret == 0, "load_onnx failed"

log("\n开始量化 (auto_hybrid=True)...")
ret = rknn.build(do_quantization=True, dataset=DATASET, auto_hybrid=True)
if ret != 0:
    log(f"❌ auto_hybrid 量化失败! ret={ret}")
    log("尝试 save_hybrid_cfg 方式...")
    sys.exit(1)

log("\n导出混合量化模型...")
ret = rknn.export_rknn(HYBRID_RKNN)
if ret == 0:
    size_mb = os.path.getsize(HYBRID_RKNN) / (1024*1024)
    log(f"✅ 混合量化模型导出成功!")
    log(f"   路径: {HYBRID_RKNN}")
    log(f"   大小: {size_mb:.1f} MB")
else:
    log(f"❌ 导出失败! ret={ret}")
    sys.exit(1)

t_elapsed = time.time() - t_start
log(f"总耗时: {t_elapsed:.0f}秒 ({t_elapsed/60:.1f}分)")
rknn.release()
log("完成!")