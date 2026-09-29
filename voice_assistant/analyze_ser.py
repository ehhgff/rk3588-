#!/usr/bin/env python3
"""SER (SenseVoice) RKNN 模型性能深度分析
在板端运行，分析:
  - 不同 core_mask 配置下的推理延迟
  - 每层推理耗时分布
  - 数据搬运开销
  - NPU 负载与频率
  - 连续推理稳定性
"""
import os, sys, time, json, struct, wave
import numpy as np

MODEL_PATH = "/data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn"
WARMUP = 5
ROUNDS = 50
SAMPLE_INPUT = np.array([[[0.0]*560 for _ in range(100)]], dtype=np.float32)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_npu_load():
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            return f.read().strip()
    except:
        return "N/A"

def get_npu_freq():
    try:
        with open("/sys/class/devfreq/fdab0000.npu/cur_freq") as f:
            return int(f.read().strip()) // 1000000
    except:
        return "N/A"

def timed_inference(rknn, inputs, label="", rounds=ROUNDS):
    """多次推理并统计"""
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
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "cv_pct": round(float(np.std(arr) / np.mean(arr) * 100), 2),
    }

def analyze_input_sizes(rknn):
    """测试不同输入长度对推理延迟的影响"""
    log("\n3.1 输入帧数对延迟的影响")
    results = []
    for frames in [10, 20, 50, 100, 200]:
        inp = np.array([[[0.0]*560 for _ in range(frames)]], dtype=np.float32)
        r = timed_inference(rknn, [inp], label=f"frames={frames}", rounds=20)
        results.append(r)
        log(f"  frames={frames:3d}: avg={r['avg_ms']:6.1f}ms  min={r['min_ms']:5.1f}ms  max={r['max_ms']:5.1f}ms  cv={r['cv_pct']:.1f}%")
    return results

def analyze_core_masks():
    """测试不同 NPU core_mask 配置"""
    log("\n2. 不同 core_mask 配置对比")
    from rknnlite.api import RKNNLite
    configs = [
        (RKNNLite.NPU_CORE_0, "NPU_CORE_0 (单核)"),
        (RKNNLite.NPU_CORE_0_1, "NPU_CORE_0_1 (双核)"),
        (RKNNLite.NPU_CORE_0_1_2, "NPU_CORE_0_1_2 (三核)"),
        (RKNNLite.NPU_CORE_AUTO, "NPU_CORE_AUTO (自动)"),
    ]
    results = []
    for mask, label in configs:
        log(f"\n  --- {label} ---")
        rknn = RKNNLite()
        rknn.load_rknn(MODEL_PATH)
        rknn.init_runtime(core_mask=mask)
        time.sleep(0.1)
        load_before = get_npu_load()

        r = timed_inference(rknn, [SAMPLE_INPUT], label=label, rounds=30)

        time.sleep(0.1)
        load_after = get_npu_load()
        r["core_mask"] = str(mask)
        r["npu_load_before"] = load_before
        r["npu_load_after"] = load_after
        npu_freq = get_npu_freq()
        r["npu_freq_mhz"] = npu_freq

        log(f"    平均: {r['avg_ms']:6.1f}ms  |  min: {r['min_ms']:5.1f}ms  |  p50: {r['p50_ms']:5.1f}ms  |  p95: {r['p95_ms']:5.1f}ms")
        log(f"    NPU 负载: {load_before} → {load_after}  |  频率: {npu_freq}MHz")
        rknn.release()
        results.append(r)
        time.sleep(0.5)

    # 计算相对收益
    baseline = results[0]["avg_ms"]
    for r in results:
        r["vs_baseline_pct"] = round((baseline - r["avg_ms"]) / baseline * 100, 1)
    return results

def analyze_stability(rknn):
    """长时间连续推理稳定性测试"""
    log("\n3.2 长时间连续推理稳定性")
    latencies = []
    for i in range(100):
        t0 = time.perf_counter()
        rknn.inference(inputs=[SAMPLE_INPUT])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if (i + 1) % 20 == 0:
            load = get_npu_load()
            log(f"    推理 #{i+1:3d}: {latencies[-1]:6.1f}ms  NPU: {load}")

    arr = np.array(latencies)
    window = 10
    moving_avg = np.convolve(arr, np.ones(window)/window, mode='valid')
    drift = moving_avg[-1] - moving_avg[0]

    result = {
        "avg_ms": round(float(np.mean(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "drift_ms": round(float(drift), 2),
        "cv_pct": round(float(np.std(arr) / np.mean(arr) * 100), 2),
    }
    log(f"  100次连续推理: avg={result['avg_ms']:.1f}ms  drift={result['drift_ms']:.1f}ms  cv={result['cv_pct']:.1f}%")
    return result

def main():
    log("=" * 60)
    log("SER (SenseVoice) RKNN 模型深度性能分析")
    log("=" * 60)
    log(f"模型: {MODEL_PATH}")
    log(f"输入: 100帧 x 560维 float32")
    log(f"预热: {WARMUP}次  测试: {ROUNDS}次")

    results = {
        "model": "sensevoice_encoder_ctc_100f_fp16.rknn",
        "model_path": MODEL_PATH,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_info": {"warmup": WARMUP, "rounds": ROUNDS},
    }

    # 1. 环境信息
    log("\n1. 环境信息")
    results["environment"] = {
        "npu_freq_idle_mhz": get_npu_freq(),
        "npu_load_idle": get_npu_load(),
        "hostname": os.uname().nodename or "rk3588",
    }
    log(f"  NPU 空闲频率: {results['environment']['npu_freq_idle_mhz']}MHz")
    log(f"  NPU 空闲负载: {results['environment']['npu_load_idle']}")

    # 2. Core mask 对比
    core_results = analyze_core_masks()
    results["core_mask_comparison"] = core_results

    # 3.1 输入尺寸影响
    log("\n3. 详细分析")
    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

    input_results = analyze_input_sizes(rknn)
    results["input_size_analysis"] = input_results

    # 3.2 稳定性测试
    stability = analyze_stability(rknn)
    results["stability"] = stability

    # 4. 数据搬运 vs 计算开销
    log("\n3.3 数据搬运开销分析")
    dummy = np.zeros((1, 100, 560), dtype=np.float32)
    t_copy = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = dummy.copy()
        t1 = time.perf_counter()
        t_copy.append((t1 - t0) * 1000)
    copy_time = float(np.mean(t_copy))
    inf_time = stability["avg_ms"]
    log(f"  输入数据拷贝: {copy_time:.3f}ms  (占推理 {copy_time/inf_time*100:.1f}%)")

    results["io_overhead"] = {
        "data_copy_us": round(copy_time * 1000, 1),
        "inference_ms": inf_time,
        "overhead_pct": round(copy_time / inf_time * 100, 2),
    }

    rknn.release()

    # 输出 JSON
    sys.path.insert(0, "/data/voice_assistant")
    out_path = "/tmp/analyze_ser_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n结果已保存: {out_path}")

    # 摘要
    log("\n" + "=" * 60)
    log("分析摘要")
    log("=" * 60)
    best = min(core_results, key=lambda x: x["avg_ms"])
    log(f"最佳 core_mask: {best['label']} = {best['avg_ms']}ms")
    log(f"推理稳定性: CV={stability['cv_pct']}%, 漂移={stability['drift_ms']}ms")
    log(f"数据搬运开销: 占推理 {results['io_overhead']['overhead_pct']}%")

    return results

if __name__ == "__main__":
    main()