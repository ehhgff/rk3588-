#!/usr/bin/env python3
"""
板端测试: 链式推理 (Encoder INT8 + Head FP16) vs FP16
测试: 延迟 + 精度
"""
import os, sys, time, json
import numpy as np
from rknnlite.api import RKNNLite

ENC_RKNN = "models/sensevoice_encoder_ctc_100f_enc_int8.rknn"
HEAD_RKNN = "models/sensevoice_encoder_ctc_100f_head_fp16.rknn"
FP16_RKNN = "models/sensevoice_encoder_ctc_100f_fp16.rknn"
INPUT_SHAPE = (1, 100, 560)
DTYPE = np.float32
WARMUP = 3
LOOPS = 50
N_ACC = 50

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_model(path, label):
    log(f"加载 {label}: {path}")
    rknn = RKNNLite()
    ret = rknn.load_rknn(path)
    assert ret == 0, f"{label} load failed"
    ret = rknn.init_runtime()
    assert ret == 0, f"{label} init_runtime failed"
    sz = os.path.getsize(path) / (1024*1024)
    log(f"  ✅ {label} ({sz:.0f}MB)")
    return rknn

def chain_inference(rknn_enc, rknn_head, inp):
    """链式推理: encoder INT8 → head FP16"""
    enc_out = rknn_enc.inference(inputs=[inp])[0]
    # 检查输出类型
    if enc_out.dtype != np.float32:
        enc_out = enc_out.astype(np.float32)
    out = rknn_head.inference(inputs=[enc_out])[0]
    return out

def benchmark_latency(rknn_enc, rknn_head, label):
    log(f"\n--- {label} 延迟 ---")
    inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
    for _ in range(WARMUP):
        chain_inference(rknn_enc, rknn_head, inp)
    lats = []
    for _ in range(LOOPS):
        t0 = time.perf_counter()
        chain_inference(rknn_enc, rknn_head, inp)
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    avg = np.mean(lats); med = np.median(lats)
    p90 = lats[int(0.9*len(lats))]; p10 = lats[int(0.1*len(lats))]
    log(f"  Avg={avg:.1f}ms  Med={med:.1f}ms  P10={p10:.1f}ms  P90={p90:.1f}ms")
    return avg, med, p90

def benchmark_single(rknn, label):
    log(f"\n--- {label} 延迟 ---")
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

def accuracy_test(rknn_enc, rknn_head, rknn_fp16):
    log(f"\n--- 精度对比 (N={N_ACC}) ---")
    cos_sims, max_diffs = [], []
    for t in range(N_ACC):
        inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
        out_chain = chain_inference(rknn_enc, rknn_head, inp).flatten()
        out_fp16 = rknn_fp16.inference(inputs=[inp])[0].flatten()
        a, b = out_chain.astype(np.float64), out_fp16.astype(np.float64)
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
log("板端测试: Encoder INT8(227MB) + Head FP16(25MB)")
log("=" * 60)

rknn_enc = load_model(ENC_RKNN, "Encoder INT8")
rknn_head = load_model(HEAD_RKNN, "Head FP16")
rknn_fp16 = load_model(FP16_RKNN, "FP16 full")

lat_chain, med_chain, p90_chain = benchmark_latency(rknn_enc, rknn_head, "链式推理(INT8+FP16)")
lat_fp16, med_fp16, p90_fp16 = benchmark_single(rknn_fp16, "FP16 full")

avg_cos, avg_md, cos_sims = accuracy_test(rknn_enc, rknn_head, rknn_fp16)

rknn_enc.release(); rknn_head.release(); rknn_fp16.release()

enc_sz = os.path.getsize(ENC_RKNN) / (1024*1024)
head_sz = os.path.getsize(HEAD_RKNN) / (1024*1024)
log("\n" + "=" * 60)
log("结果汇总")
log("=" * 60)
log(f"  Encoder INT8: {enc_sz:.0f} MB")
log(f"  Head FP16: {head_sz:.0f} MB")
log(f"  合计: {enc_sz+head_sz:.0f} MB (FP16: 460MB)")
log(f"  链式延迟: avg={lat_chain:.1f}ms, med={med_chain:.1f}ms")
log(f"  FP16延迟: avg={lat_fp16:.1f}ms")
log(f"  Cosine Similarity: {avg_cos:.6f}")

result = {
    "model": "chain_enc_int8_head_fp16",
    "total_size_mb": round(enc_sz+head_sz, 1),
    "enc_size_mb": round(enc_sz, 1),
    "head_size_mb": round(head_sz, 1),
    "lat_avg_ms": round(lat_chain, 1),
    "lat_med_ms": round(med_chain, 1),
    "lat_p90_ms": round(p90_chain, 1),
    "cos_avg": round(avg_cos, 6),
    "cos_min": round(min(cos_sims), 6),
    "max_diff_avg": round(avg_md, 6),
    "fp16_lat_ms": round(lat_fp16, 1),
}
log(f"\nJSON: {json.dumps(result, indent=2)}")