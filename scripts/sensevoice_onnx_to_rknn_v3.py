#!/usr/bin/env python3
"""
SenseVoice ONNX 转 RKNN 脚本 v3
使用固定输入尺寸
"""

import os
import sys
import numpy as np
from rknn.api import RKNN


def convert_onnx_to_rknn(
    onnx_path,
    rknn_path,
    target_platform="rk3588",
    do_quantization=False,
    dataset_path=None,
    fixed_seq_len=500,
):
    """
    转换 ONNX 模型为 RKNN
    """
    print(f"初始化 RKNN...")
    rknn = RKNN(verbose=True)
    
    # 配置模型参数
    print(f"\n配置 RKNN 参数...")
    print(f"  Target platform: {target_platform}")
    print(f"  Quantization: {do_quantization}")
    print(f"  Fixed seq_len: {fixed_seq_len}")
    
    # 设置配置参数
    config_kwargs = {
        "mean_values": [[0.0]],
        "std_values": [[1.0]],
        "target_platform": target_platform,
        "optimization_level": 3,
    }
    
    if do_quantization:
        config_kwargs["quantized_dtype"] = "w8a8"
    
    ret = rknn.config(**config_kwargs)
    if ret != 0:
        print("配置失败!")
        return False
    
    # 加载 ONNX 模型，使用固定输入尺寸
    print(f"\n加载 ONNX 模型: {onnx_path}")
    ret = rknn.load_onnx(
        model=onnx_path,
        input_size_list=[[1, fixed_seq_len, 560]],
    )
    if ret != 0:
        print("加载 ONNX 模型失败!")
        return False
    
    # 构建 RKNN 模型
    print(f"\n构建 RKNN 模型...")
    ret = rknn.build(do_quantization=do_quantization, dataset=dataset_path)
    if ret != 0:
        print("构建 RKNN 模型失败!")
        return False
    
    # 导出 RKNN 模型
    print(f"\n导出 RKNN 模型: {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("导出 RKNN 模型失败!")
        return False
    
    # 获取模型大小
    file_size = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"RKNN 模型大小: {file_size:.2f} MB")
    
    # 释放 RKNN 对象
    rknn.release()
    
    print(f"\n转换完成!")
    return True


def main():
    # 路径配置
    onnx_path = "models/sensevoice_encoder_ctc.onnx"
    rknn_path_fp16 = "models/sensevoice_encoder_ctc_fp16.rknn"
    rknn_path_int8 = "models/sensevoice_encoder_ctc_int8.rknn"
    
    # 检查 ONNX 文件
    if not os.path.exists(onnx_path):
        print(f"错误: ONNX 文件不存在: {onnx_path}")
        return
    
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX 模型大小: {onnx_size:.2f} MB")
    
    # 固定序列长度
    fixed_seq_len = 500  # 约5秒音频 (500帧 * 10ms = 5s)
    
    # 1. 转换为 FP16 RKNN
    print("\n" + "=" * 60)
    print("1. 转换为 FP16 RKNN")
    print("=" * 60)
    
    success = convert_onnx_to_rknn(
        onnx_path=onnx_path,
        rknn_path=rknn_path_fp16,
        target_platform="rk3588",
        do_quantization=False,
        fixed_seq_len=fixed_seq_len,
    )
    
    if not success:
        print("FP16 转换失败!")
        return
    
    # 2. 转换为 INT8 RKNN
    print("\n" + "=" * 60)
    print("2. 转换为 INT8 RKNN")
    print("=" * 60)
    
    # 创建量化数据集
    dataset_path = "other/quantization_dataset.txt"
    if not os.path.exists(dataset_path):
        print(f"创建量化数据集: {dataset_path}")
        with open(dataset_path, 'w') as f:
            for i in range(100):
                data = np.random.randn(1, fixed_seq_len, 560).astype(np.float32)
                data_path = f"quant_data_{i}.npy"
                np.save(data_path, data)
                f.write(f"{data_path}\n")
    
    success = convert_onnx_to_rknn(
        onnx_path=onnx_path,
        rknn_path=rknn_path_int8,
        target_platform="rk3588",
        do_quantization=True,
        dataset_path=dataset_path,
        fixed_seq_len=fixed_seq_len,
    )
    
    if success:
        print("INT8 转换成功!")
    else:
        print("INT8 转换失败，但 FP16 模型已生成。")


if __name__ == "__main__":
    main()