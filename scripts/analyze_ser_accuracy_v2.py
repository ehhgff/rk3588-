#!/usr/bin/env python3
"""
SER 100f 精度分析: 从ONNX构建两模型 → PC端推理 → 对比输出
"""
import os, sys, json
import numpy as np
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
INPUT_SHAPE = (1, 100, 560)
NUM_TESTS = 10
DATASET = "calib_data/data_100f/dataset.txt"

results = {}

# ========== 1. 构建 FP16 参考模型 ==========
print("=" * 60)
print("1. 构建 FP16 参考模型")
print("=" * 60)
rknn_f = RKNN(verbose=False)
rknn_f.config(target_platform="rk3588", optimization_level=3)
rknn_f.load_onnx(model=ONNX_PATH)
rknn_f.build(do_quantization=False)
rknn_f.init_runtime()
print("FP16 模型构建 + runtime 初始化完成")

# ========== 2. 构建 INT8 量化模型 ==========
print("\n" + "=" * 60)
print("2. 构建 INT8 量化模型")
print("=" * 60)
rknn_i = RKNN(verbose=False)
rknn_i.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
rknn_i.load_onnx(model=ONNX_PATH)
rknn_i.build(do_quantization=True, dataset=DATASET)
rknn_i.init_runtime()
print("INT8 模型构建 + runtime 初始化完成")

# ========== 3. 精度对比 ==========
print("\n" + "=" * 60)
print("3. 精度对比 (PC Simulator)")
print("=" * 60)

cos_sims, max_diffs, mean_diffs = [], [], []

for t in range(NUM_TESTS):
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    out_f16 = rknn_f.inference(inputs=[dummy])[0].flatten()
    out_i8 = rknn_i.inference(inputs=[dummy])[0].flatten()
    
    cos = float(np.dot(out_i8, out_f16) / (np.linalg.norm(out_i8) * np.linalg.norm(out_f16)))
    max_diff = float(np.max(np.abs(out_i8 - out_f16)))
    mean_diff = float(np.mean(np.abs(out_i8 - out_f16)))
    
    cos_sims.append(cos)
    max_diffs.append(max_diff)
    mean_diffs.append(mean_diff)
    print(f"  测试 {t+1:2d}: cos={cos:.6f}  max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}")

results["accuracy"] = {
    "cosine_similarity": {
        "mean": float(f"{np.mean(cos_sims):.4f}"),
        "min": float(f"{np.min(cos_sims):.4f}"),
        "std": float(f"{np.std(cos_sims):.6f}"),
    },
    "max_abs_diff_mean": float(f"{np.mean(max_diffs):.4f}"),
    "mean_abs_diff_mean": float(f"{np.mean(mean_diffs):.4f}"),
    "num_tests": NUM_TESTS,
    "threshold": 0.99,
    "passed": bool(np.mean(cos_sims) >= 0.99),
}

print(f"\n{'='*60}")
print(f"平均余弦相似度: {results['accuracy']['cosine_similarity']['mean']:.4f}")
print(f"最低余弦相似度: {results['accuracy']['cosine_similarity']['min']:.4f}")
print(f"标准差:         {results['accuracy']['cosine_similarity']['std']:.6f}")
print(f"阈值:           0.99")
print(f"结论:           {'✅ 精度达标' if results['accuracy']['passed'] else '❌ 精度损失显著，需混合量化'}")

# 导出模型
ret_f = rknn_f.export_rknn(FP16_OUT := "models/sensevoice_encoder_ctc_100f_fp16_v2.rknn")
ret_i = rknn_i.export_rknn(INT8_OUT := "models/sensevoice_encoder_ctc_100f_int8_v2.rknn")
if ret_f == 0 and ret_i == 0:
    fp16_size = os.path.getsize(FP16_OUT) / (1024*1024)
    int8_size = os.path.getsize(INT8_OUT) / (1024*1024)
    print(f"\n模型已重新导出:")
    print(f"  FP16: {FP16_OUT} ({fp16_size:.1f} MB)")
    print(f"  INT8: {INT8_OUT} ({int8_size:.1f} MB) ↓{int8_size/fp16_size*100:.0f}%")

with open("ser_100f_accuracy.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: ser_100f_accuracy.json")

rknn_f.release()
rknn_i.release()