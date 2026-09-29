#!/usr/bin/env python3
"""
SenseVoice 100帧模型 INT8 量化导出脚本
输入: models/sensevoice_encoder_ctc_100f.onnx → (1, 100, 560)
输出: models/sensevoice_encoder_ctc_100f_int8.rknn
"""
import os, sys
import numpy as np
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
RKNN_PATH = "models/sensevoice_encoder_ctc_100f_int8.rknn"
CALIB_DIR = "calib_data/data_100f"
CALIB_SIZE = 100
INPUT_SHAPE = (1, 100, 560)

os.makedirs(CALIB_DIR, exist_ok=True)

print("=" * 60)
print("SenseVoice 100帧 INT8 量化导出")
print("=" * 60)

# 1. 生成校准数据
print(f"\n生成 {CALIB_SIZE} 组校准数据, shape={INPUT_SHAPE} ...")
dataset_txt = os.path.join(CALIB_DIR, "dataset.txt")
with open(dataset_txt, 'w') as f:
    for i in range(CALIB_SIZE):
        data_path = os.path.join(CALIB_DIR, f"calib_{i}.npy")
        data = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        np.save(data_path, data)
        f.write(f"{data_path}\n")
print(f"校准数据已保存到 {CALIB_DIR}/")

# 2. FP16 转换（参考）
print("\n" + "-" * 60)
print("步骤1: 导出 FP16 参考模型")
print("-" * 60)
rknn_fp16 = RKNN(verbose=True)
rknn_fp16.config(target_platform="rk3588", optimization_level=3)
ret = rknn_fp16.load_onnx(model=ONNX_PATH)
if ret != 0: print("FP16 加载 ONNX 失败!"); sys.exit(1)
ret = rknn_fp16.build(do_quantization=False)
if ret != 0: print("FP16 构建失败!"); sys.exit(1)
ret = rknn_fp16.export_rknn("models/sensevoice_encoder_ctc_100f_fp16.rknn")
if ret != 0: print("FP16 导出失败!"); sys.exit(1)
size_fp16 = os.path.getsize("models/sensevoice_encoder_ctc_100f_fp16.rknn") / (1024*1024)
print(f"FP16 模型大小: {size_fp16:.1f} MB")
rknn_fp16.release()

# 3. INT8 量化转换
print("\n" + "-" * 60)
print("步骤2: 导出 INT8 量化模型")
print("-" * 60)
rknn_int8 = RKNN(verbose=True)
rknn_int8.config(
    target_platform="rk3588",
    optimization_level=3,
    quantized_dtype="w8a8",
)
ret = rknn_int8.load_onnx(model=ONNX_PATH)
if ret != 0: print("INT8 加载 ONNX 失败!"); sys.exit(1)
ret = rknn_int8.build(do_quantization=True, dataset=dataset_txt)
if ret != 0: print("INT8 构建失败!"); sys.exit(1)
ret = rknn_int8.export_rknn(RKNN_PATH)
if ret != 0: print("INT8 导出失败!"); sys.exit(1)
size_int8 = os.path.getsize(RKNN_PATH) / (1024*1024)
print(f"INT8 模型大小: {size_int8:.1f} MB")
rknn_int8.release()

print("\n" + "=" * 60)
print(f"转换完成!")
print(f"  FP16: models/sensevoice_encoder_ctc_100f_fp16.rknn ({size_fp16:.1f} MB)")
print(f"  INT8: {RKNN_PATH} ({size_int8:.1f} MB)")
print(f"  压缩比: {size_int8/size_fp16*100:.0f}%")
print("=" * 60)