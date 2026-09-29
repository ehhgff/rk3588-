#!/usr/bin/env python3
"""将 Matcha-TTS ONNX 模型转换为 FP16 格式 - 使用 onnxruntime 工具"""

import onnx
import argparse
import os


def convert_to_fp16(input_path, output_path):
    """使用 onnxruntime 工具转换模型为 FP16"""
    print(f"Loading model from: {input_path}")
    
    try:
        # 使用 onnxruntime 的转换工具
        from onnxruntime.transformers.float16 import convert_float_to_float16
        
        print("Converting to FP16 using onnxruntime...")
        convert_float_to_float16(
            input_path,
            output_path,
            keep_io_types=True,
        )
        
    except Exception as e:
        print(f"onnxruntime conversion failed: {e}")
        print("Falling back to onnxconverter-common...")
        
        from onnxconverter_common import float16
        
        model = onnx.load(input_path)
        
        # 转换
        model_fp16 = float16.convert_float_to_float16(
            model,
            min_positive_val=1e-7,
            max_finite_val=1e4,
            keep_io_types=True,
            disable_shape_infer=False,
        )
        
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
    model_fp16 = onnx.load(output_path)
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
