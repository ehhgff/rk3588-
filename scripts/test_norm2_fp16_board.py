#!/usr/bin/env python3
"""
板端测试: norm2_fp16 (custom_hybrid, 48个norm2层FP16) vs FP16
"""
import os, sys, time, json
import numpy as np
from rknnlite.api import RKNNLite

RKNN_HYBRID = "/data/models/sensevoice_encoder_ctc_100f_norm2_fp16.rknn"
RKNN_FP16 = "/data/models/sensevoice_encoder_ctc_100f_fp16.rknn"
INPUT_SHAPE = (1, 100, 560)
DTYPE = np.float32
WARMUP = 5
LOOPS = 50
N_ACC = 50

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_model(path, label):
    log(f"加载 {label}: {path}")
    if not os.path.exists(path):
        log(f"  ❌ 文件不存在: {path}")
        return None
    rknn = RKNNLite()
    ret = rknn.load_rknn(path)
    assert ret == 0, f"{label} load failed"
    ret = rknn.init_runtime()
    assert ret == 0, f"{label} init_runtime failed"
    sz = os.path.getsize(path) / (1024*1024)
    log(f"  ✅ {label} ({sz:.0f}MB)")
    return rknn

def benchmark_latency(rknn, label):
    log(f"\n--- {label} 延迟测试 ---")
    inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
    for _ in range(WARMUP):
        rknn.inference(inputs=[inp])
    lats = []
    for _ in range(LOOPS):
        t0 = time.perf_counter()
        rknn.inference(inputs=[inp])
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    avg = np.mean(lats); med = np.median(lats)
    p90 = lats[int(0.9*len(lats))]; p10 = lats[int(0.1*len(lats))]
    log(f"  Avg={avg:.1f}ms  Med={med:.1f}ms  P10={p10:.1f}ms  P90={p90:.1f}ms")
    return avg, med, p90

def accuracy_test(rknn_hyb, rknn_fp16):
    log(f"\n--- 精度对比 (N={N_ACC}) ---")
    cos_sims, max_diffs = [], []
    for t in range(N_ACC):
        inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
        out_hyb = rknn_hyb.inference(inputs=[inp])[0].flatten()
        out_fp16 = rknn_fp16.inference(inputs=[inp])[0].flatten()
        a, b = out_hyb.astype(np.float64), out_fp16.astype(np.float64)
        dot = np.dot(a, b); norm = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(dot / norm) if norm > 1e-10 else 1.0
        md = float(np.max(np.abs(a - b)))
        cos_sims.append(cos); max_diffs.append(md)
        if t < 3 or t == N_ACC-1:
            log(f"  [{t+1}/{N_ACC}] cos={cos:.6f}  max_diff={md:.4f}")
    avg_cos = np.mean(cos_sims); min_cos = np.min(cos_sims)
    avg_md = np.mean(max_diffs)
    log(f"\n  📊 Cosine: avg={avg_cos:.6f}  min={min_cos:.6f}")
    log(f"  📊 MaxDiff: avg={avg_md:.6f}")
    status = "✅ PASS" if avg_cos >= 0.99 else "❌ FAIL"
    log(f"  {status} (threshold: cos>=0.99)")
    return avg_cos, avg_md, cos_sims

# ======== MAIN ========
log("=" * 60)
log("板端测试: norm2_fp16 (custom_hybrid)")
log("=" * 60)

rknn_hyb = load_model(RKNN_HYBRID, "norm2_fp16")
rknn_fp16 = load_model(RKNN_FP16, "FP16")

if rknn_hyb is None or rknn_fp16 is None:
    sys.exit(1)

lat_hyb, med_hyb, p90_hyb = benchmark_latency(rknn_hyb, "norm2_fp16")
lat_fp16, med_fp16, p90_fp16 = benchmark_latency(rknn_fp16, "FP16")

avg_cos, avg_md, cos_sims = accuracy_test(rknn_hyb, rknn_fp16)

rknn_hyb.release(); rknn_fp16.release()

sz = os.path.getsize(RKNN_HYBRID) / (1024*1024)
log("\n" + "=" * 60)
log("结果汇总")
log("=" * 60)
log(f"  模型: norm2_fp16 (48个norm2层FP16)")
log(f"  模型大小: {sz:.0f} MB")
log(f"  norm2_fp16延迟: avg={lat_hyb:.1f}ms, med={med_hyb:.1f}ms, p90={p90_hyb:.1f}ms")
log(f"  FP16延迟: avg={lat_fp16:.1f}ms, med={med_fp16:.1f}ms, p90={p90_fp16:.1f}ms")
log(f"  Cosine Similarity: {avg_cos:.6f} (min: {min(cos_sims):.6f})")

result = {
    "model": "norm2_fp16 (custom_hybrid)",
    "description": "48个encoder块的norm2层保持FP16，其余INT8",
    "size_mb": round(sz, 1),
    "lat_avg_ms": round(lat_hyb, 1),
    "lat_med_ms": round(med_hyb, 1),
    "lat_p90_ms": round(p90_hyb, 1),
    "cos_avg": round(avg_cos, 6),
    "cos_min": round(min(cos_sims), 6),
    "max_diff_avg": round(avg_md, 6),
    "fp16_lat_avg_ms": round(lat_fp16, 1),
    "fp16_lat_med_ms": round(med_fp16, 1),
    "fp16_lat_p90_ms": round(p90_fp16, 1),
}
log(f"\nJSON: {json.dumps(result, indent=2)}")