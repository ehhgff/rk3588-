#!/usr/bin/env python3
"""将 Matcha-TTS ONNX 模型转换为 FP16 格式 - 仅转换权重"""

import onnx
from onnx import TensorProto
import numpy as np
import argparse
import os


def convert_to_fp16(input_path, output_path):
    """将 ONNX 模型权重转换为 FP16，保持计算图为 FP32"""
    print(f"Loading model from: {input_path}")
    model = onnx.load(input_path)
    
    print("Converting weights to FP16...")
    
    # 创建新模型
    model_fp16 = onnx.ModelProto()
    model_fp16.CopyFrom(model)
    
    # 转换所有 initializer 为 FP16
    fp32_count = 0
    fp16_count = 0
    for init in model_fp16.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            # 转换为 numpy 数组
            arr = onnx.numpy_helper.to_array(init)
            # 转换为 FP16
            arr_fp16 = arr.astype(np.float16)
            # 创建新的 initializer
            new_init = onnx.numpy_helper.from_array(arr_fp16, init.name)
            init.CopyFrom(new_init)
            fp16_count += 1
        elif init.data_type == TensorProto.FLOAT16:
            fp16_count += 1
    
    print(f"Converted {fp16_count} tensors to FP16")
    
    # 保持输入输出为 FP32（不修改）
    # 这样可以确保兼容性
    
    print(f"Saving FP16 model to: {output_path}")
    onnx.save(model_fp16, output_path)
    
    # 比较文件大小
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    fp16_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\nConversion complete!")
    print(f"Original size: {original_size:.2f} MB")
    print(f"FP16 size: {fp16_size:.2f} MB")
    print(f"Compression ratio: {fp16_size/original_size:.2%}")
    
    # 验证模型
    print("\nValidating model...")
    onnx.checker.check_model(model_fp16)
    print("Model validation passed!")
    
    return model_fp16


def main():
    parser = argparse.ArgumentParser(description="Convert Matcha-TTS ONNX model to FP16")
    parser.add_argument("--input", "-i", required=True, help="Input ONNX model path")
    parser.add_argument("--output", "-o", required=True, help="Output FP16 model path")
    args = parser.parse_args()
    
    convert_to_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
