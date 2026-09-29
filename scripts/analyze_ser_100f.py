#!/usr/bin/env python3
"""
SER 100f 模型深度分析:
  1. eval_perf - PC端性能预估
  2. accuracy_analysis - INT8 vs FP16 余弦相似度
"""
import os, sys, json
import numpy as np
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
FP16_RKNN = "models/sensevoice_encoder_ctc_100f_fp16.rknn"
INT8_RKNN = "models/sensevoice_encoder_ctc_100f_int8.rknn"
INPUT_SHAPE = (1, 100, 560)
DATASET = "calib_data/data_100f/dataset.txt"

results = {}

# =========================================================
# 1. eval_perf (需重新 build)
# =========================================================
print("=" * 60)
print("1. eval_perf - 性能预估")
print("=" * 60)

rknn = RKNN(verbose=False)
rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
rknn.load_onnx(model=ONNX_PATH)
rknn.build(do_quantization=True, dataset=DATASET)

print("\n运行 eval_perf ...")
perf = rknn.eval_perf(is_print=True, fix_freq=True)
print(f"\neval_perf 返回类型: {type(perf)}")
print(f"eval_perf 返回内容: {str(perf)[:500]}")
results["eval_perf_raw"] = str(perf)

rknn.release()

# =========================================================
# 2. accuracy_analysis
# =========================================================
print("\n" + "=" * 60)
print("2. accuracy_analysis - 精度分析")
print("=" * 60)

print("加载 INT8 模型...")
rknn_i = RKNN(verbose=False)
rknn_i.load_rknn(INT8_RKNN)
rknn_i.init_runtime()

print("加载 FP16 模型...")
rknn_f = RKNN(verbose=False)
rknn_f.load_rknn(FP16_RKNN)
rknn_f.init_runtime()

# 多组输入测试
num_tests = 10
cos_sims = []
max_diffs = []
mean_diffs = []

for t in range(num_tests):
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    out_i8 = rknn_i.inference(inputs=[dummy])[0].flatten()
    out_f16 = rknn_f.inference(inputs=[dummy])[0].flatten()
    
    cos = np.dot(out_i8, out_f16) / (np.linalg.norm(out_i8) * np.linalg.norm(out_f16))
    max_diff = np.max(np.abs(out_i8 - out_f16))
    mean_diff = np.mean(np.abs(out_i8 - out_f16))
    
    cos_sims.append(cos)
    max_diffs.append(max_diff)
    mean_diffs.append(mean_diff)
    print(f"  测试 {t+1}: cos={cos:.6f}, max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")

results["accuracy"] = {
    "cosine_similarity_mean": float(np.mean(cos_sims)),
    "cosine_similarity_min": float(np.min(cos_sims)),
    "cosine_similarity_std": float(np.std(cos_sims)),
    "max_abs_diff_mean": float(np.mean(max_diffs)),
    "mean_abs_diff_mean": float(np.mean(mean_diffs)),
    "num_tests": num_tests,
    "threshold": 0.99,
    "passed": bool(np.mean(cos_sims) >= 0.99),
}

print(f"\n  平均余弦相似度: {results['accuracy']['cosine_similarity_mean']:.4f}")
print(f"  最低余弦相似度: {results['accuracy']['cosine_similarity_min']:.4f}")
print(f"  判定阈值: 0.99")
print(f"  {'✅ 通过' if results['accuracy']['passed'] else '❌ 未通过'}")

rknn_i.release()
rknn_f.release()

# =========================================================
# 输出
# =========================================================
print("\n" + "=" * 60)
print("分析结果汇总")
print("=" * 60)
print(json.dumps(results, indent=2, ensure_ascii=False))

with open("ser_100f_analysis_result.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\n结果已保存到 ser_100f_analysis_result.json")