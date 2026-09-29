#!/usr/bin/env python3
"""
SER 100f 精度分析: INT8 vs FP16 余弦相似度对比
"""
import os, sys, json
import numpy as np
from rknn.api import RKNN

INT8_RKNN = "models/sensevoice_encoder_ctc_100f_int8.rknn"
FP16_RKNN = "models/sensevoice_encoder_ctc_100f_fp16.rknn"
INPUT_SHAPE = (1, 100, 560)

print("=" * 60)
print("SER 100帧 INT8 vs FP16 精度分析")
print("=" * 60)

print("加载 INT8 模型...")
rknn_i = RKNN()
rknn_i.load_rknn(INT8_RKNN)
rknn_i.init_runtime()

print("加载 FP16 模型...")
rknn_f = RKNN()
rknn_f.load_rknn(FP16_RKNN)
rknn_f.init_runtime()

# 10组随机输入测试
num_tests = 10
cos_sims, max_diffs, mean_diffs = [], [], []

for t in range(num_tests):
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    out_i8 = rknn_i.inference(inputs=[dummy])[0].flatten()
    out_f16 = rknn_f.inference(inputs=[dummy])[0].flatten()
    
    cos = float(np.dot(out_i8, out_f16) / (np.linalg.norm(out_i8) * np.linalg.norm(out_f16)))
    max_diff = float(np.max(np.abs(out_i8 - out_f16)))
    mean_diff = float(np.mean(np.abs(out_i8 - out_f16)))
    
    cos_sims.append(cos)
    max_diffs.append(max_diff)
    mean_diffs.append(mean_diff)
    print(f"  测试 {t+1:2d}: cos={cos:.6f}  max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}")

results = {
    "cosine_similarity": {
        "mean": float(np.mean(cos_sims)),
        "min": float(np.min(cos_sims)),
        "std": float(np.std(cos_sims)),
    },
    "max_abs_diff_mean": float(np.mean(max_diffs)),
    "mean_abs_diff_mean": float(np.mean(mean_diffs)),
    "num_tests": num_tests,
    "threshold": 0.99,
    "passed": bool(np.mean(cos_sims) >= 0.99),
}

print(f"\n{'='*60}")
print(f"平均余弦相似度: {results['cosine_similarity']['mean']:.4f}")
print(f"最低余弦相似度: {results['cosine_similarity']['min']:.4f}")
print(f"标准差:         {results['cosine_similarity']['std']:.6f}")
print(f"最大绝对差异:   {results['max_abs_diff_mean']:.4f}")
print(f"判定阈值:       0.99")
print(f"结论:           {'✅ 精度达标 (cos >= 0.99)' if results['passed'] else '❌ 精度损失显著 (cos < 0.99)'}")

with open("ser_100f_accuracy.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n结果已保存: ser_100f_accuracy.json")

rknn_i.release()
rknn_f.release()