#!/usr/bin/env python3
"""
转换 Qwen3-1.7B 模型为 RKLLM 格式，max_context=2048
优化版本：减少内存使用
"""
import os
import sys
import gc
from rkllm.api import RKLLM

# 模型路径
model_path = "/home/ubuntu/桌面/ai/Qwen3-1.7B"

# 输出路径
output_path = "/home/ubuntu/桌面/ai/Qwen3-1.7B_W8A8_RK3588_2048.rkllm"

def convert_model():
    print("=" * 60)
    print("Qwen3-1.7B RKLLM 模型转换 (max_context=2048)")
    print("=" * 60)
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        sys.exit(1)
    
    print(f"模型路径: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"目标平台: RK3588")
    print(f"量化类型: W8A8")
    print(f"最大上下文: 2048")
    print("-" * 60)
    
    # 创建 RKLLM 实例
    rkllm = RKLLM()
    
    # 加载模型
    print("\n[1/4] 加载 HuggingFace 模型...")
    ret = rkllm.load_huggingface(model=model_path)
    if ret != 0:
        print(f"加载模型失败: {ret}")
        sys.exit(1)
    print("✓ 模型加载成功")
    
    # 强制垃圾回收
    gc.collect()
    
    # 构建模型 - 使用 optimization_level=1 获得更好性能
    print("\n[2/4] 构建 RKLLM 模型 (W8A8量化, max_context=2048)...")
    print("注意: 使用 optimization_level=1 获得更好性能")
    ret = rkllm.build(
        do_quantization=True,
        optimization_level=1,  # 使用标准优化级别
        quantized_dtype='w8a8',
        target_platform='rk3588',
        max_context=2048
    )
    if ret != 0:
        print(f"构建模型失败: {ret}")
        sys.exit(1)
    print("✓ 模型构建成功")
    
    # 强制垃圾回收
    gc.collect()
    
    # 导出模型
    print("\n[3/4] 导出 RKLLM 模型...")
    ret = rkllm.export_rkllm(output_path)
    if ret != 0:
        print(f"导出模型失败: {ret}")
        sys.exit(1)
    print(f"✓ 模型导出成功: {output_path}")
    
    # 验证模型
    print("\n[4/4] 验证模型...")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ 模型文件大小: {size_mb:.1f} MB")
        print("\n" + "=" * 60)
        print("转换完成!")
        print("=" * 60)
        print(f"\n新模型: {output_path}")
        print("特点:")
        print("  - max_context: 2048 (原模型为4096)")
        print("  - 加载时间: 预计减少 30-50%")
        print("  - 内存占用: 预计减少 30-50%")
    else:
        print("错误: 模型导出失败")
        sys.exit(1)

if __name__ == "__main__":
    convert_model()
