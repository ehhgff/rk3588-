#!/usr/bin/env python3
"""ASR (Zipformer) RKNN 模型性能深度分析 — v2 修正版
在板端运行，分析:
  - Encoder / Decoder / Joiner 三模型分别推理延迟
  - 端到端 ASR pipeline 耗时分解
  - 各阶段 CPU vs NPU 开销比例
"""
import os, sys, time, json
import numpy as np

ENCODER_PATH = "/data/zipformer/model/encoder-epoch-99-avg-1.rknn"
DECODER_PATH = "/data/zipformer/model/decoder-epoch-99-avg-1.rknn"
JOINER_PATH = "/data/zipformer/model/joiner-epoch-99-avg-1.rknn"
WARMUP = 5
ROUNDS = 50

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def timed_inference_rknn(rknn, inputs, label="", rounds=ROUNDS):
    for _ in range(WARMUP):
        rknn.inference(inputs=inputs)
    latencies = []
    for i in range(rounds):
        t0 = time.perf_counter()
        out = rknn.inference(inputs=inputs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    arr = np.array(latencies)
    result = {
        "label": label,
        "avg_ms": round(float(np.mean(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "cv_pct": round(float(np.std(arr) / np.mean(arr) * 100), 2),
    }
    return result, out

def get_npu_load():
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            return f.read().strip()
    except:
        return "N/A"

def analyze_models():
    from rknnlite.api import RKNNLite

    results = {}

    # ─── Encoder ───
    log("\n2.1 Encoder 模型分析")
    rknn = RKNNLite()
    rknn.load_rknn(ENCODER_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    npu_load_0 = get_npu_load()
    log(f"  NPU负载(推理前): {npu_load_0}")

    # 尝试多种输入形状 (encoder 是 2D 输入: frames x feat_dim)
    test_shapes = [
        ((100, 512), "2D (100,512)"),
        ((200, 512), "2D (200,512)"),
        ((500, 512), "2D (500,512)"),
        ((1000, 512), "2D (1000,512)"),
    ]
    enc_results = []
    for shape, label in test_shapes:
        try:
            inp = np.random.randn(*shape).astype(np.float32)
            r, out = timed_inference_rknn(rknn, [inp], label=label, rounds=20)
            r["shape"] = str(shape)
            r["output_shape"] = [str(o.shape) for o in out] if isinstance(out, list) else str(out.shape)
            enc_results.append(r)
            log(f"  {label}: avg={r['avg_ms']:.1f}ms  out={r['output_shape']}")
        except Exception as e:
            log(f"  {label}: FAIL - {str(e)[:60]}")

    time.sleep(0.1)
    npu_load_1 = get_npu_load()
    log(f"  NPU负载(推理后): {npu_load_1}")
    rknn.release()
    results["encoder"] = enc_results

    # ─── Decoder ───
    log("\n2.2 Decoder 模型分析")
    rknn2 = RKNNLite()
    rknn2.load_rknn(DECODER_PATH)
    rknn2.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    dec_results = []
    test_decoder_shapes = [
        ((1, 512, 100), "3D (1,512,100)"),
        ((1, 256, 50), "3D (1,256,50)"),
        ((512, 100), "2D (512,100)"),
    ]
    for shape, label in test_decoder_shapes:
        try:
            inp = np.random.randn(*shape).astype(np.float32)
            r, out = timed_inference_rknn(rknn2, [inp], label=label, rounds=20)
            r["shape"] = str(shape)
            r["output_shape"] = [str(o.shape) for o in out] if isinstance(out, list) else str(out.shape)
            dec_results.append(r)
            log(f"  {label}: avg={r['avg_ms']:.1f}ms  out={r['output_shape']}")
        except Exception as e:
            log(f"  {label}: FAIL - {str(e)[:60]}")
    rknn2.release()
    results["decoder"] = dec_results

    # ─── Joiner ───
    log("\n2.3 Joiner 模型分析 (双输入)")
    rknn3 = RKNNLite()
    rknn3.load_rknn(JOINER_PATH)
    rknn3.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    join_results = []
    for dim in [256, 512]:
        for label_prefix, (e_inp, d_inp) in [
                (f"1D({dim})", (np.random.randn(dim).astype(np.float32), np.random.randn(dim).astype(np.float32))),
                (f"2D(1,{dim})", (np.random.randn(1, dim).astype(np.float32), np.random.randn(1, dim).astype(np.float32)))]:
            try:
                r, out = timed_inference_rknn(rknn3, [e_inp, d_inp], label=label_prefix, rounds=20)
                r["dim"] = dim
                r["output_shape"] = [str(o.shape) for o in out] if isinstance(out, list) else str(out.shape)
                join_results.append(r)
                log(f"  {label_prefix}: avg={r['avg_ms']:.1f}ms  out={r['output_shape']}")
            except Exception as e:
                log(f"  {label_prefix}: FAIL - {str(e)[:60]}")
    rknn3.release()
    results["joiner"] = join_results

    # ─── Pipeline 估算 ───
    log("\n3. ASR Pipeline 延迟估算")
    if enc_results and dec_results and join_results:
        valid_enc = [r for r in enc_results if r.get("avg_ms", 0) > 1]
        valid_dec = [r for r in dec_results if r.get("avg_ms", 0) > 1]
        valid_join = [r for r in join_results if r.get("avg_ms", 0) > 1]
        if valid_enc and valid_dec and valid_join:
            enc_best = min(valid_enc, key=lambda x: x["avg_ms"])
            dec_best = min(valid_dec, key=lambda x: x["avg_ms"])
            join_best = min(valid_join, key=lambda x: x["avg_ms"])
            log(f"  Encoder: {enc_best['avg_ms']:.1f}ms  ({enc_best.get('shape','')})")
            log(f"  Decoder: {dec_best['avg_ms']:.1f}ms  ({dec_best.get('shape','')})")
            log(f"  Joiner:  {join_best['avg_ms']:.1f}ms  ({join_best.get('shape','')})")
            join_total = join_best["avg_ms"] * 30
            total = enc_best["avg_ms"] + dec_best["avg_ms"] + join_total
            log(f"  Pipeline (~30 tokens): {enc_best['avg_ms']:.0f}+{dec_best['avg_ms']:.0f}+{join_total:.0f}={total:.0f}ms")
            results["pipeline"] = {
                "encoder_ms": enc_best["avg_ms"],
                "decoder_ms": dec_best["avg_ms"],
                "joiner_per_token_ms": join_best["avg_ms"],
                "joiner_30tokens_ms": round(join_total, 1),
                "total_ms": round(total, 1),
            }

    return results

def main():
    log("=" * 60)
    log("ASR (Zipformer) RKNN 模型深度性能分析 v2")
    log("=" * 60)

    results = {
        "model": "Zipformer ASR",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 模型文件信息
    for name, path in [("Encoder", ENCODER_PATH), ("Decoder", DECODER_PATH), ("Joiner", JOINER_PATH)]:
        sz = os.path.getsize(path)
        log(f"  {name}: {round(sz/1e6,1)}MB")
        results[f"{name.lower()}_size_mb"] = round(sz / 1e6, 1)

    # 分析
    results["analysis"] = analyze_models()

    # 输出
    out_path = "/tmp/analyze_asr_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n结果已保存: {out_path}")

    return results

if __name__ == "__main__":
    main()