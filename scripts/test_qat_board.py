#!/usr/bin/env python3
"""
板端测试: QAT INT8 vs FP16 精度 + 延迟
"""
import os, sys, time
import numpy as np
from rknnlite.api import RKNNLite

RKNN_QAT_INT8 = "/data/voice_assistant/models/qat_ser_int8.rknn"
RKNN_QAT_FP16 = "/data/voice_assistant/models/qat_ser_fp16.rknn"
INPUT_SHAPE = (1, 100, 560)
FLOAT32 = np.float32
NUM_TESTS = 50

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_rknn(path, name=""):
    log(f"加载 {name}: {os.path.basename(path)}")
    rknn = RKNNLite()
    ret = rknn.load_rknn(path)
    if ret != 0:
        log(f"  ❌ load_rknn 失败: {path}")
        sys.exit(1)
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        log(f"  ❌ init_runtime 失败")
        sys.exit(1)
    size = os.path.getsize(path) / (1024*1024)
    log(f"  ✅ 加载成功 ({size:.0f} MB)")
    return rknn

def test_accuracy():
    log("=" * 60)
    log("QAT INT8 vs FP16 精度对比 (板端)")
    log("=" * 60)

    rknn_qat = load_rknn(RKNN_QAT_INT8, "QAT INT8")
    rknn_fp16 = load_rknn(RKNN_QAT_FP16, "QAT FP16")

    cos_sims = []
    max_diffs = []

    log(f"\n运行 {NUM_TESTS} 次随机测试...")
    for t in range(NUM_TESTS):
        inp = np.random.randn(*INPUT_SHAPE).astype(FLOAT32)

        out_qat = rknn_qat.inference(inputs=[inp])[0]
        out_fp16 = rknn_fp16.inference(inputs=[inp])[0]

        # Both models have same structure (prepend 3 tokens, 103 output frames)
        min_len = min(out_qat.shape[1], out_fp16.shape[1])
        a = out_qat[0, -min_len:, :].astype(np.float64).flatten()
        b = out_fp16[0, -min_len:, :].astype(np.float64).flatten()

        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(dot / norm) if norm > 1e-10 else 1.0
        max_diff = float(np.max(np.abs(a - b)))

        cos_sims.append(cos)
        max_diffs.append(max_diff)

        if (t + 1) % 10 == 0:
            log(f"  #{t+1:3d}: cos={np.mean(cos_sims[-10:]):.6f}")

    rknn_qat.release()
    rknn_fp16.release()

    cos_arr = np.array(cos_sims)
    log(f"\n{'='*60}")
    log(f"精度结果")
    log(f"{'='*60}")
    log(f"  平均余弦相似度: {cos_arr.mean():.6f}")
    log(f"  最小余弦相似度:  {cos_arr.min():.6f}")
    log(f"  >= 0.99 比例:    {(cos_arr >= 0.99).mean()*100:.1f}%")
    log(f"  平均最大差异:    {np.mean(max_diffs):.6f}")

def test_latency():
    log(f"\n{'='*60}")
    log(f"QAT INT8 延迟测试 ({NUM_TESTS}次)")
    log(f"{'='*60}")

    rknn_qat = load_rknn(RKNN_QAT_INT8, "QAT INT8")

    inp = np.random.randn(*INPUT_SHAPE).astype(FLOAT32)

    # Warmup
    for _ in range(5):
        rknn_qat.inference(inputs=[inp])

    latencies = []
    for t in range(NUM_TESTS):
        t0 = time.perf_counter()
        rknn_qat.inference(inputs=[inp])
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

    rknn_qat.release()

    lat_arr = np.array(latencies)
    log(f"\n  平均延迟: {lat_arr.mean():.1f} ms")
    log(f"  最小延迟:  {lat_arr.min():.1f} ms")
    log(f"  最大延迟:  {lat_arr.max():.1f} ms")
    log(f"  P95延迟:   {np.percentile(lat_arr, 95):.1f} ms")
    return lat_arr.mean()

if __name__ == "__main__":
    test_accuracy()
    avg_lat = test_latency()
    log(f"\n{'='*60}")
    log(f"最终结果")
    log(f"{'='*60}")
    log(f"  QAT INT8 平均延迟: {avg_lat:.1f} ms")