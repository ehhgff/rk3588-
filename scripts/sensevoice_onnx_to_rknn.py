#!/usr/bin/env python3
"""
SenseVoice ONNX 转 RKNN 脚本
将 SenseVoice ONNX 模型转换为 RK3588 可用的 RKNN 格式
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 导入 RKNN API
try:
    from rknn.api import RKNN
except ImportError:
    print("错误: 未找到 RKNN-Toolkit2")
    print("请先安装 RKNN-Toolkit2:")
    print("  cd /home/ubuntu/桌面/ai/lubancat_ai_manual_code/dev_env/rknn_toolkit2")
    print("  pip install packages/rknn_toolkit2-*.whl")
    sys.exit(1)


def convert_onnx_to_rknn(
    onnx_path,
    rknn_path,
    target_platform="rk3588",
    do_quantization=True,
    dataset_path=None,
    quantized_dtype="asymmetric_quantized-u8",
    mean_values=None,
    std_values=None,
):
    """
    将 ONNX 模型转换为 RKNN 模型
    
    Args:
        onnx_path: 输入 ONNX 模型路径
        rknn_path: 输出 RKNN 模型路径
        target_platform: 目标平台 (rk3588/rk3568/etc)
        do_quantization: 是否进行量化
        dataset_path: 量化校准数据集路径 (.txt 文件，每行一个 npy 文件路径)
        quantized_dtype: 量化类型
        mean_values: 预处理均值
        std_values: 预处理标准差
    """
    
    # 创建 RKNN 对象
    print(f"初始化 RKNN...")
    rknn = RKNN(verbose=True)
    
    # 配置模型参数
    print(f"配置 RKNN 参数...")
    print(f"  Target platform: {target_platform}")
    print(f"  Quantization: {do_quantization}")
    print(f"  Quantized dtype: {quantized_dtype}")
    
    # 设置预处理参数
    # SenseVoice 使用 80 维梅尔频谱，通常不需要额外的归一化
    if mean_values is None:
        mean_values = [[0.0] * 80]  # 80 维特征
    if std_values is None:
        std_values = [[1.0] * 80]   # 不归一化
    
    rknn.config(
        mean_values=mean_values,
        std_values=std_values,
        target_platform=target_platform,
        quantized_dtype=quantized_dtype,
        optimization_level=3,  # 最高优化级别
    )
    
    # 加载 ONNX 模型
    print(f"\n加载 ONNX 模型: {onnx_path}")
    ret = rknn.load_onnx(
        model=onnx_path,
        # 如果模型有多个输入，需要指定输入名称和形状
        # input_size_list=[[1, 3000, 80]],  # [batch, seq_len, feature_dim]
    )
    if ret != 0:
        print("加载 ONNX 模型失败!")
        return False
    print("ONNX 模型加载成功")
    
    # 构建 RKNN 模型
    print(f"\n构建 RKNN 模型...")
    if do_quantization:
        if dataset_path is None or not os.path.exists(dataset_path):
            print(f"警告: 未找到校准数据集 {dataset_path}")
            print("将生成随机数据进行量化...")
            dataset_path = generate_dummy_dataset()
        
        print(f"使用校准数据集: {dataset_path}")
        ret = rknn.build(
            do_quantization=True,
            dataset=dataset_path,
            rknn_batch_size=1,
        )
    else:
        ret = rknn.build(
            do_quantization=False,
            rknn_batch_size=1,
        )
    
    if ret != 0:
        print("构建 RKNN 模型失败!")
        return False
    print("RKNN 模型构建成功")
    
    # 导出 RKNN 模型
    print(f"\n导出 RKNN 模型: {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("导出 RKNN 模型失败!")
        return False
    print("RKNN 模型导出成功")
    
    # 释放 RKNN 对象
    rknn.release()
    
    # 打印模型信息
    print_model_info(rknn_path)
    
    return True


def generate_dummy_dataset(num_samples=10, output_dir="./calibration_data"):
    """
    生成随机校准数据
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_txt = os.path.join(output_dir, "dataset.txt")
    
    with open(dataset_txt, "w") as f:
        for i in range(num_samples):
            # 生成随机梅尔频谱数据
            # 形状: [1, seq_len, 80]
            seq_len = np.random.randint(1000, 3000)
            data = np.random.randn(1, seq_len, 80).astype(np.float32)
            
            # 保存为 npy 文件
            npy_path = os.path.join(output_dir, f"sample_{i:03d}.npy")
            np.save(npy_path, data)
            f.write(npy_path + "\n")
    
    print(f"生成校准数据集: {dataset_txt}")
    return dataset_txt


def print_model_info(rknn_path):
    """
    打印 RKNN 模型信息
    """
    print("\n" + "="*60)
    print("RKNN 模型信息")
    print("="*60)
    
    file_size = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"模型文件: {rknn_path}")
    print(f"文件大小: {file_size:.2f} MB")
    
    # 重新加载模型获取详细信息
    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    if ret == 0:
        print("模型加载成功")
    rknn.release()


def prepare_real_dataset(audio_files, output_dir="./calibration_data"):
    """
    从真实音频文件准备校准数据集
    
    Args:
        audio_files: 音频文件列表 (.wav)
        output_dir: 输出目录
    """
    import librosa
    
    os.makedirs(output_dir, exist_ok=True)
    dataset_txt = os.path.join(output_dir, "dataset.txt")
    
    with open(dataset_txt, "w") as f:
        for i, audio_path in enumerate(audio_files):
            print(f"处理音频 {i+1}/{len(audio_files)}: {audio_path}")
            
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 提取梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=80,
                n_fft=400,
                hop_length=160,
            )
            
            # 转换为对数刻度
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 转置为 [seq_len, 80]
            log_mel_spec = log_mel_spec.T
            
            # 添加 batch 维度 [1, seq_len, 80]
            data = np.expand_dims(log_mel_spec, axis=0).astype(np.float32)
            
            # 保存
            npy_path = os.path.join(output_dir, f"sample_{i:03d}.npy")
            np.save(npy_path, data)
            f.write(npy_path + "\n")
    
    print(f"\n校准数据集已生成: {dataset_txt}")
    return dataset_txt


def main():
    parser = argparse.ArgumentParser(
        description="Convert SenseVoice ONNX to RKNN for RK3588"
    )
    parser.add_argument(
        "--onnx", "-i",
        required=True,
        help="Input ONNX model path"
    )
    parser.add_argument(
        "--rknn", "-o",
        default="sensevoice_small.rknn",
        help="Output RKNN model path"
    )
    parser.add_argument(
        "--target",
        default="rk3588",
        choices=["rk3588", "rk3568", "rk3566", "rv1106"],
        help="Target platform (default: rk3588)"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable quantization (use FP16)"
    )
    parser.add_argument(
        "--dataset",
        help="Path to quantization dataset txt file"
    )
    parser.add_argument(
        "--quantized-dtype",
        default="asymmetric_quantized-u8",
        choices=["asymmetric_quantized-u8", "dynamic_fixed_point-i8", "dynamic_fixed_point-i16"],
        help="Quantization data type"
    )
    parser.add_argument(
        "--generate-dummy-data",
        action="store_true",
        help="Generate dummy calibration data"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.onnx):
        print(f"错误: 找不到 ONNX 模型文件: {args.onnx}")
        sys.exit(1)
    
    # 生成虚拟校准数据
    if args.generate_dummy_data and args.dataset is None:
        args.dataset = generate_dummy_dataset()
    
    # 执行转换
    success = convert_onnx_to_rknn(
        onnx_path=args.onnx,
        rknn_path=args.rknn,
        target_platform=args.target,
        do_quantization=not args.no_quantization,
        dataset_path=args.dataset,
        quantized_dtype=args.quantized_dtype,
    )
    
    if success:
        print("\n" + "="*60)
        print("转换成功!")
        print("="*60)
        print(f"RKNN 模型已保存至: {args.rknn}")
        print("\n下一步:")
        print(f"  1. 将模型推送到开发板: adb push {args.rknn} /data/")
        print("  2. 使用 RKNN C++ API 或 Python API 进行推理")
    else:
        print("\n转换失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
