#!/usr/bin/env python3
"""将 Matcha-TTS ONNX 模型转换为 FP16 格式 - 使用 onnxruntime 工具"""

import onnx
import argparse
import os


def convert_to_fp16_ort(input_path, output_path):
    """使用 ONNX Runtime 的工具转换模型为 FP16"""
    print(f"Loading model from: {input_path}")
    
    # 使用 onnxruntime 的转换工具
    try:
        from onnxruntime.tools import optimizer
        from onnxruntime.tools.convert_onnx_models_to_ort import convert_onnx_model
        
        # 先优化模型
        print("Optimizing model...")
        optimized_model = optimizer.optimize_model(
            input_path,
            model_type='bert',  # 使用通用优化
            use_gpu=False,
            num_heads=0,
            hidden_size=0,
        )
        
        # 保存优化后的模型
        temp_path = input_path + '.opt.onnx'
        optimized_model.save_model_to_file(temp_path)
        
        # 转换为 FP16
        print("Converting to FP16...")
        from onnxruntime.transformers import float16 as ort_fp16
        ort_fp16.convert_float_to_float16(
            temp_path,
            output_path,
            keep_io_types=True,
        )
        
        # 删除临时文件
        os.remove(temp_path)
        
    except ImportError:
        print("onnxruntime-tools not available, using fallback method...")
        # 回退方法：使用 onnxconverter-common 但添加更多参数
        from onnxconverter_common import float16
        
        model = onnx.load(input_path)
        
        # 获取输入输出名称
        input_names = [inp.name for inp in model.graph.input]
        output_names = [out.name for out in model.graph.output]
        
        print(f"Inputs: {input_names}")
        print(f"Outputs: {output_names}")
        
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


def main():
    parser = argparse.ArgumentParser(description="Convert Matcha-TTS ONNX model to FP16")
    parser.add_argument("--input", "-i", required=True, help="Input ONNX model path")
    parser.add_argument("--output", "-o", required=True, help="Output FP16 model path")
    args = parser.parse_args()
    
    convert_to_fp16_ort(args.input, args.output)


if __name__ == "__main__":
    main()
