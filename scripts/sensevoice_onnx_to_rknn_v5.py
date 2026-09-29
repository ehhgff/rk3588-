#!/usr/bin/env python3
"""
SenseVoice ONNX 转 RKNN 脚本 v5
修正 mean_values 维度
"""

import os
import numpy as np
from rknn.api import RKNN


def convert_onnx_to_rknn(
    onnx_path,
    rknn_path,
    target_platform="rk3588",
    do_quantization=False,
    dataset_path=None,
):
    print(f"初始化 RKNN...")
    rknn = RKNN(verbose=True)
    
    print(f"\n配置 RKNN 参数...")
    print(f"  Target platform: {target_platform}")
    print(f"  Quantization: {do_quantization}")
    
    # 输入维度为 560，mean_values 和 std_values 需要 560 个值
    input_dim = 560
    
    config_kwargs = {
        "mean_values": [[0.0] * input_dim],
        "std_values": [[1.0] * input_dim],
        "target_platform": target_platform,
        "optimization_level": 3,
    }
    
    if do_quantization:
        config_kwargs["quantized_dtype"] = "w8a8"
    
    ret = rknn.config(**config_kwargs)
    if ret != 0:
        print("配置失败!")
        return False
    
    print(f"\n加载 ONNX 模型: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("加载 ONNX 模型失败!")
        return False
    
    print(f"\n构建 RKNN 模型...")
    ret = rknn.build(do_quantization=do_quantization, dataset=dataset_path)
    if ret != 0:
        print("构建 RKNN 模型失败!")
        return False
    
    print(f"\n导出 RKNN 模型: {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("导出 RKNN 模型失败!")
        return False
    
    file_size = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"RKNN 模型大小: {file_size:.2f} MB")
    
    rknn.release()
    print(f"\n转换完成!")
    return True


def main():
    onnx_path = "models/models/sensevoice_encoder_ctc_fixed.onnx"
    rknn_path_fp16 = "models/sensevoice_encoder_ctc_fp16.rknn"
    rknn_path_int8 = "models/sensevoice_encoder_ctc_int8.rknn"
    
    if not os.path.exists(onnx_path):
        print(f"错误: ONNX 文件不存在: {onnx_path}")
        return
    
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX 模型大小: {onnx_size:.2f} MB")
    
    # 1. FP16
    print("\n" + "=" * 60)
    print("1. 转换为 FP16 RKNN")
    print("=" * 60)
    
    success = convert_onnx_to_rknn(
        onnx_path=onnx_path,
        rknn_path=rknn_path_fp16,
        target_platform="rk3588",
        do_quantization=False,
    )
    
    if not success:
        print("FP16 转换失败!")
        return
    
    # 2. INT8
    print("\n" + "=" * 60)
    print("2. 转换为 INT8 RKNN")
    print("=" * 60)
    
    dataset_path = "other/quantization_dataset.txt"
    if not os.path.exists(dataset_path):
        print(f"创建量化数据集: {dataset_path}")
        with open(dataset_path, 'w') as f:
            for i in range(100):
                data = np.random.randn(1, 500, 560).astype(np.float32)
                data_path = f"quant_data_{i}.npy"
                np.save(data_path, data)
                f.write(f"{data_path}\n")
    
    success = convert_onnx_to_rknn(
        onnx_path=onnx_path,
        rknn_path=rknn_path_int8,
        target_platform="rk3588",
        do_quantization=True,
        dataset_path=dataset_path,
    )
    
    if success:
        print("INT8 转换成功!")
    else:
        print("INT8 转换失败，但 FP16 模型已生成。")


if __name__ == "__main__":
    main()