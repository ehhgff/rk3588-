#!/usr/bin/env python3
"""
Local accuracy test: QAT INT8 vs FP16 reference
Uses random inputs and computes cosine similarity
"""

import os, sys, time
import numpy as np
from rknn.api import RKNN

QAT_RKNN = "models/qat_ser_int8.rknn"
FP16_RKNN = "models/sensevoice_encoder_ctc_100f_fp16.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def test():
    log("=" * 60)
    log("QAT INT8 vs FP16 精度对比")
    log("=" * 60)

    for path in [QAT_RKNN, FP16_RKNN]:
        if not os.path.exists(path):
            log(f"ERROR: {path} not found!")
            return

    size_qat = os.path.getsize(QAT_RKNN) / (1024*1024)
    size_fp16 = os.path.getsize(FP16_RKNN) / (1024*1024)
    log(f"QAT INT8:  {QAT_RKNN} ({size_qat:.1f} MB)")
    log(f"FP16 ref:  {FP16_RKNN} ({size_fp16:.1f} MB)")

    log("\nLoading QAT INT8 model...")
    rknn_qat = RKNN(verbose=False)
    rknn_qat.load_rknn(QAT_RKNN)
    rknn_qat.init_runtime()
    log("  QAT INT8 loaded")

    log("Loading FP16 reference model...")
    rknn_fp16 = RKNN(verbose=False)
    rknn_fp16.load_rknn(FP16_RKNN)
    rknn_fp16.init_runtime()
    log("  FP16 loaded")

    INPUT_SHAPE = (1, 100, 560)
    DTYPE = np.float32
    NUM_TESTS = 50

    cos_sims = []
    max_diffs = []

    log(f"\nRunning {NUM_TESTS} random tests...")
    for t in range(NUM_TESTS):
        inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)

        out_qat = rknn_qat.inference(inputs=[inp])[0].flatten()
        out_fp16 = rknn_fp16.inference(inputs=[inp])[0].flatten()

        a = out_qat.astype(np.float64)
        b = out_fp16.astype(np.float64)

        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(dot / norm) if norm > 1e-10 else 1.0
        max_diff = float(np.max(np.abs(a - b)))

        cos_sims.append(cos)
        max_diffs.append(max_diff)

        if (t+1) % 10 == 0:
            avg_cos = np.mean(cos_sims[-10:])
            log(f"  #{t+1:2d}: avg_cos={avg_cos:.6f}")

    rknn_qat.release()
    rknn_fp16.release()

    cos_arr = np.array(cos_sims)
    log(f"\n{'='*60}")
    log(f"结果汇总 ({NUM_TESTS}次测试)")
    log(f"{'='*60}")
    log(f"  平均余弦相似度: {cos_arr.mean():.6f}")
    log(f"  最小余弦相似度:  {cos_arr.min():.6f}")
    log(f"  最大余弦相似度:  {cos_arr.max():.6f}")
    log(f"  平均最大差异:    {np.mean(max_diffs):.6f}")
    log(f"  >= 0.99 比例:    {(cos_arr >= 0.99).mean()*100:.1f}%")

    with open("qat_accuracy_result.txt", 'w') as f:
        f.write(f"QAT INT8 Accuracy Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: {QAT_RKNN}\n")
        f.write(f"Reference: {FP16_RKNN}\n")
        f.write(f"Tests: {NUM_TESTS}\n")
        f.write(f"Avg Cosine Similarity: {cos_arr.mean():.6f}\n")
        f.write(f"Min Cosine Similarity: {cos_arr.min():.6f}\n")
        f.write(f"Max Cosine Similarity: {cos_arr.max():.6f}\n")
        f.write(f"Avg Max Diff: {np.mean(max_diffs):.6f}\n")
        f.write(f"Ratio >= 0.99: {(cos_arr >= 0.99).mean()*100:.1f}%\n")

    log(f"\n结果已保存到 qat_accuracy_result.txt")

if __name__ == "__main__":
    test()