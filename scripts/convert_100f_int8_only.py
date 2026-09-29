#!/usr/bin/env python3
"""仅执行 INT8 量化（FP16 已生成，校准数据已就绪）"""
import os, sys
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
RKNN_PATH = "models/sensevoice_encoder_ctc_100f_int8.rknn"
DATASET = "calib_data/data_100f/dataset.txt"

rknn = RKNN(verbose=True)
print("配置 INT8 量化参数...")
rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")

print(f"加载 ONNX: {ONNX_PATH}")
ret = rknn.load_onnx(model=ONNX_PATH)
assert ret == 0, "加载 ONNX 失败"

print(f"构建 INT8 模型 (dataset={DATASET})...")
ret = rknn.build(do_quantization=True, dataset=DATASET)
assert ret == 0, "构建失败"

print(f"导出: {RKNN_PATH}")
ret = rknn.export_rknn(RKNN_PATH)
assert ret == 0, "导出失败"

size = os.path.getsize(RKNN_PATH) / (1024*1024)
print(f"INT8 模型大小: {size:.1f} MB")
rknn.release()
print("完成!")