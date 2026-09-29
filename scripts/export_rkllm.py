#!/usr/bin/env python3
"""将 DeepSeek-R1-Distill-Qwen-1.5B 医疗微调模型转换为 RKLLM 格式"""

from rkllm.api import RKLLM

model_path = "/home/ubuntu/桌面/ai/DeepSeek-R1-CMtMedQA-Merged"
output_path = "/home/ubuntu/桌面/ai/DeepSeek-R1-CMtMedQA-w8a8.rkllm"

print("=" * 50)
print("DeepSeek-R1-Distill-Qwen-1.5B 中文微调模型 RKNN 转换")
print("=" * 50)

llm = RKLLM()

print(f"\n[1/4] 加载模型: {model_path}")
ret = llm.load_huggingface(model=model_path)
if ret != 0:
    print(f"加载模型失败! 错误码: {ret}")
    exit(ret)
print("模型加载成功!")

print(f"\n[2/4] 构建模型 (w8a8 量化, optimization_level=0)")
ret = llm.build(
    do_quantization=True,
    optimization_level=0,
    quantized_dtype="w8a8",
    target_platform="rk3588"
)
if ret != 0:
    print(f"构建模型失败! 错误码: {ret}")
    exit(ret)
print("模型构建成功!")

print(f"\n[3/4] 导出 RKLLM 模型: {output_path}")
ret = llm.export_rkllm(output_path)
if ret != 0:
    print(f"导出模型失败! 错误码: {ret}")
    exit(ret)
print("模型导出成功!")

print("\n" + "=" * 50)
print("转换完成!")
print(f"输出文件: {output_path}")
print("=" * 50)
