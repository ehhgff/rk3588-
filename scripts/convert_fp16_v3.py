#!/usr/bin/env python3
"""将 Matcha-TTS ONNX 模型转换为 FP16 格式 - 使用 onnxsim 预处理"""

import onnx
from onnxconverter_common import float16
import onnxsim
import argparse
import os


def convert_to_fp16(input_path, output_path):
    """将 ONNX 模型转换为 FP16"""
    print(f"Loading model from: {input_path}")
    model = onnx.load(input_path)
    
    # 先简化模型
    print("Simplifying model...")
    model_simp, check = onnxsim.simplify(
        model,
        dynamic_input_shape=False,
        skip_fuse_bn=False,
    )
    if not check:
        print("Simplification failed, using original model")
        model_simp = model
    else:
        print("Model simplified successfully")
    
    print("Converting to FP16...")
    # 转换权重为 FP16，但保持计算图为 FP32
    # 这样可以减小模型大小，同时保持兼容性
    
    # 方法：手动转换所有 initializer 为 FP16
    from onnx import TensorProto
    import numpy as np
    
    # 创建新模型
    model_fp16 = onnx.ModelProto()
    model_fp16.CopyFrom(model_simp)
    
    # 转换所有权重为 FP16
    fp32_count = 0
    fp16_count = 0
    for init in model_fp16.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            # 转换为 numpy 数组
            arr = onnx.numpy_helper.to_array(init)
            # 转换为 FP16
            arr_fp16 = arr.astype(np.float16)
            # 更新 initializer
            new_init = onnx.numpy_helper.from_array(arr_fp16, init.name)
            init.CopyFrom(new_init)
            fp16_count += 1
        elif init.data_type == TensorProto.FLOAT16:
            fp16_count += 1
    
    print(f"Converted {fp16_count} tensors to FP16")
    
    # 保持输入输出为 FP32
    # 不需要修改输入输出类型
    
    print(f"Saving FP16 model to: {output_path}")
    onnx.save(model_fp16, output_path)
    
    # 比较文件大小
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    fp16_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\nConversion complete!")
    print(f"Original size: {original_size:.2f} MB")
    print(f"FP16 size: {fp16_size:.2f} MB")
    print(f"Compression ratio: {fp16_size/original_size:.2%}")
    
    return model_fp16


def main():
    parser = argparse.ArgumentParser(description="Convert Matcha-TTS ONNX model to FP16")
    parser.add_argument("--input", "-i", required=True, help="Input ONNX model path")
    parser.add_argument("--output", "-o", required=True, help="Output FP16 model path")
    args = parser.parse_args()
    
    convert_to_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
