#!/usr/bin/env python3
"""ASR (Zipformer) RKNN 模型性能深度分析
在板端运行，分析:
  - Encoder / Decoder / Joiner 三模型分别推理延迟
  - 端到端 ASR pipeline 耗时分解
  - 不同音频长度对延迟的影响
  - NPU 负载与频率
  - 各阶段 CPU vs NPU 开销比例
"""
import os, sys, time, json, struct, subprocess
import numpy as np

ENCODER_PATH = "/data/zipformer/model/encoder-epoch-99-avg-1.rknn"
DECODER_PATH = "/data/zipformer/model/decoder-epoch-99-avg-1.rknn"
JOINER_PATH = "/data/zipformer/model/joiner-epoch-99-avg-1.rknn"
VOCAB_PATH = "/data/zipformer/model/vocab.txt"
WARMUP = 5
ROUNDS = 50

# Zipformer encoder 典型输入: 帧数可变, 每帧 512-dim float32
# 测试不同音频长度 (假设 30ms/帧)
FRAME_MS = 30
TEST_DURATIONS = [1, 2, 3, 5, 10]  # 秒
ENCODER_FEAT_DIM = 512

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_npu_load():
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            return f.read().strip()
    except:
        return "N/A"

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
    return {
        "label": label,
        "avg_ms": round(float(np.mean(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "cv_pct": round(float(np.std(arr) / np.mean(arr) * 100), 2),
    }

def analyze_encoder():
    """Encoder 模型分析: 不同音频长度"""
    log("\n2.1 Encoder 模型 — 不同音频长度")
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    rknn.load_rknn(ENCODER_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

    results = []
    for dur in TEST_DURATIONS:
        frames = int(dur * 1000 / FRAME_MS)
        inp = np.random.randn(1, frames, ENCODER_FEAT_DIM).astype(np.float32)
        r = timed_inference_rknn(rknn, [inp], label=f"encoder_{dur}s_{frames}fr", rounds=20)
        r["duration_s"] = dur
        r["frames"] = frames
        r["rtf"] = round(r["avg_ms"] / (dur * 1000), 4)
        results.append(r)
        log(f"  {dur}s ({frames}帧): avg={r['avg_ms']:6.1f}ms  rtf={r['rtf']:.4f}  cv={r['cv_pct']:.1f}%")
    rknn.release()
    return results

def analyze_decoder():
    """Decoder 模型分析"""
    log("\n2.2 Decoder 模型")
    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    rknn.load_rknn(DECODER_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

    # Decoder 输入: encoder_out (1, C, T)  典型 C=512, T=帧数
    dummy_enc = np.random.randn(1, 512, 100).astype(np.float32)
    r = timed_inference_rknn(rknn, [dummy_enc], label="decoder_100fr", rounds=30)
    log(f"  decoder: avg={r['avg_ms']:6.1f}ms  min={r['min_ms']:5.1f}ms  cv={r['cv_pct']:.1f}%")
    rknn.release()
    return r

def analyze_joiner():
    """Joiner 模型分析"""
    log("\n2.3 Joiner 模型")
    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    rknn.load_rknn(JOINER_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

    # Joiner 输入: encoder_out (1, C) + decoder_out (1, C)  简单拼接
    dummy_enc = np.random.randn(1, 512).astype(np.float32)
    dummy_dec = np.random.randn(1, 512).astype(np.float32)
    r = timed_inference_rknn(rknn, [dummy_enc, dummy_dec], label="joiner", rounds=30)
    log(f"  joiner: avg={r['avg_ms']:6.1f}ms  min={r['min_ms']:5.1f}ms  cv={r['cv_pct']:.1f}%")
    rknn.release()
    return r

def estimate_pipeline(enc_results, dec_result, join_result, avg_tokens=30):
    """估算端到端 ASR 流水线延迟分解"""
    log("\n3. ASR Pipeline 延迟分解估算")
    log(f"  假设: 平均输出 {avg_tokens} 个 token (每帧需 joiner 一次)")

    pipeline = []
    for er in enc_results:
        enc_ms = er["avg_ms"]
        dec_ms = dec_result["avg_ms"]
        # Joiner 每帧一次
        join_total_ms = join_result["avg_ms"] * avg_tokens
        total = enc_ms + dec_ms + join_total_ms
        pipeline.append({
            "audio_len_s": er["duration_s"],
            "encoder_ms": enc_ms,
            "decoder_ms": dec_ms,
            "joiner_total_ms": round(join_total_ms, 1),
            "joiner_per_token_ms": join_result["avg_ms"],
            "total_estimate_ms": round(total, 1),
            "real_time_factor": round(total / (er["duration_s"] * 1000), 3),
        })
        log(f"  {er['duration_s']}s音频: encoder={enc_ms:.0f}ms + decoder={dec_ms:.0f}ms + joiner({avg_tokens}次)={join_total_ms:.0f}ms = {total:.0f}ms (RTF={pipeline[-1]['real_time_factor']:.3f})")
    return pipeline

def model_file_info():
    """模型文件信息"""
    log("\n1. 模型文件信息")
    info = []
    for name, path in [("Encoder", ENCODER_PATH), ("Decoder", DECODER_PATH), ("Joiner", JOINER_PATH)]:
        sz = os.path.getsize(path)
        info.append({"name": name, "path": path, "size_mb": round(sz / 1e6, 1)})
        log(f"  {name}: {round(sz/1e6,1)}MB")
    return info

def main():
    log("=" * 60)
    log("ASR (Zipformer) RKNN 模型深度性能分析")
    log("=" * 60)

    results = {
        "model": "Zipformer ASR",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }

    # 1. 文件信息
    results["models"]["files"] = model_file_info()

    # 2. 环境
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            results["environment"] = {"npu_load_idle": f.read().strip()}
    except:
        pass

    # 3. 各模型分析
    enc_results = analyze_encoder()
    results["models"]["encoder"] = enc_results

    dec_result = analyze_decoder()
    results["models"]["decoder"] = dec_result

    join_result = analyze_joiner()
    results["models"]["joiner"] = join_result

    # 4. Pipeline 估算
    pipeline = estimate_pipeline(enc_results, dec_result, join_result)
    results["pipeline_estimate"] = pipeline

    # 5. 输出 JSON
    out_path = "/tmp/analyze_asr_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n结果已保存: {out_path}")

    # 摘要
    best_rtf = min(pipeline, key=lambda x: x["real_time_factor"])
    log(f"\n最佳 RTF: {best_rtf['real_time_factor']:.3f} ({best_rtf['audio_len_s']}s 音频)")
    log(f"主要瓶颈: encoder (占 {enc_results[2]['avg_ms']/(enc_results[2]['avg_ms']+dec_result['avg_ms']+join_result['avg_ms']*30)*100:.0f}%)")

    return results

if __name__ == "__main__":
    main()