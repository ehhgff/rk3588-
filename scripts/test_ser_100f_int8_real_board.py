#!/usr/bin/env python3
"""
板端测试: SER 100帧 INT8 (真实校准集)
models/sensevoice_encoder_ctc_100f_int8_real.rknn
"""
import os, sys, time, json, struct, numpy as np
sys.path.insert(0, "/usr/lib/python3.10/site-packages")
from rknnlite.api import RKNNLite

MODEL_INT8 = "/data/sensevoice/models/sensevoice_encoder_ctc_100f_int8_real.rknn"
MODEL_FP16 = "/data/sensevoice/models/sensevoice_encoder_ctc_100f_fp16.rknn"
OUT_DIR = "/data/ser_int8_test"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "ser_100f_int8_real_test.log")
JSON_OUT = os.path.join(OUT_DIR, "ser_100f_int8_real_results.json")

INPUT_SHAPE = (1, 100, 560)
DTYPE = np.float32
WARMUP = 5
ROUNDS = 50

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def section(title):
    sep = "=" * 60
    log("")
    log(sep)
    log(title)
    log(sep)

def read_sys(param):
    try:
        with open(param, "r") as f:
            return f.read().strip()
    except:
        return "N/A"

# =========================================================
section("1. 环境检测")
# =========================================================
log(f"Model: {MODEL_INT8}")
log(f"模型大小: {os.path.getsize(MODEL_INT8) / 1024 / 1024:.1f} MB")
log(f"FP16模型: {MODEL_FP16} (存在={os.path.exists(MODEL_FP16)})")

# NPU频率
freq = read_sys("/sys/class/devfreq/fdab0000.npu/cur_freq")
log(f"NPU频率: {freq}")
avail = read_sys("/sys/class/devfreq/fdab0000.npu/available_frequencies")
log(f"可用频率: {avail}")
governor = read_sys("/sys/class/devfreq/fdab0000.npu/governor")
log(f"Governor: {governor}")

# 温度
temp = read_sys("/sys/class/thermal/thermal_zone0/temp")
log(f"SoC温度: {temp}")

# 内存
mem_total = read_sys("/proc/meminfo")
for l in mem_total.split("\n"):
    if "MemTotal" in l or "MemAvailable" in l or "MemFree" in l:
        log(f"  内存: {l.strip()}")

# =========================================================
section("2. 加载 INT8 模型")
# =========================================================
log("2.1 创建 RKNNLite 实例...")
rknn = RKNNLite(verbose=False)

log("2.2 load_rknn ...")
t0 = time.perf_counter()
rknn.load_rknn(MODEL_INT8)
t_load = (time.perf_counter() - t0) * 1000
log(f"   load_rknn 耗时: {t_load:.1f} ms")

log("2.3 init_runtime (NPU_CORE_0_1_2)...")
t0 = time.perf_counter()
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
t_init = (time.perf_counter() - t0) * 1000
log(f"   init_runtime 耗时: {t_init:.1f} ms")

# 内存检查
mem_after = read_sys("/proc/meminfo")
for l in mem_after.split("\n"):
    if "MemAvailable" in l:
        log(f"   加载后内存: {l.strip()}")

# =========================================================
section("3. 热身 (Warmup)")
# =========================================================
inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
for i in range(WARMUP):
    t0 = time.perf_counter()
    outputs = rknn.inference(inputs=[inp])
    t_inf = (time.perf_counter() - t0) * 1000
    log(f"   Warmup #{i+1}: {t_inf:.3f} ms")

# =========================================================
section("4. 性能测试 (50轮)")
# =========================================================
latencies = []
results_shape = None
for r in range(ROUNDS):
    t0 = time.perf_counter()
    outputs = rknn.inference(inputs=[inp])
    t_inf = (time.perf_counter() - t0) * 1000
    latencies.append(t_inf)
    if results_shape is None:
        results_shape = [o.shape for o in outputs]
    if (r + 1) % 10 == 0:
        log(f"   Round {r+1:3d}/{ROUNDS}: {t_inf:.3f} ms")

lat = np.array(latencies)
summary = {
    "count": ROUNDS,
    "avg_ms": float(np.mean(lat)),
    "median_ms": float(np.median(lat)),
    "min_ms": float(np.min(lat)),
    "max_ms": float(np.max(lat)),
    "std_ms": float(np.std(lat)),
    "cv_pct": float(np.std(lat) / np.mean(lat) * 100),
    "p50_ms": float(np.percentile(lat, 50)),
    "p90_ms": float(np.percentile(lat, 90)),
    "p95_ms": float(np.percentile(lat, 95)),
    "p99_ms": float(np.percentile(lat, 99)),
    "range_ms": float(np.max(lat) - np.min(lat)),
}

log(f"\n--- 性能汇总 ---")
log(f"  平均: {summary['avg_ms']:.1f} ms")
log(f"  中位: {summary['median_ms']:.1f} ms")
log(f"  最小: {summary['min_ms']:.1f} ms")
log(f"  最大: {summary['max_ms']:.1f} ms")
log(f"  P90:  {summary['p90_ms']:.1f} ms")
log(f"  标准差: {summary['std_ms']:.2f} ms")
log(f"  CV:   {summary['cv_pct']:.2f}%")

# =========================================================
section("5. 环境状态")
# =========================================================
temp_end = read_sys("/sys/class/thermal/thermal_zone0/temp")
log(f"SoC温度: {temp_end}")
npu_load = read_sys("/sys/class/devfreq/fdab0000.npu/load")
log(f"NPU负载: {npu_load}")
mem_end = read_sys("/proc/meminfo")
for l in mem_end.split("\n"):
    if "MemAvailable" in l:
        log(f"内存: {l.strip()}")

# =========================================================
section("6. 精度对比 (INT8 vs FP16)")
# =========================================================
log("6.1 加载 FP16 参考模型 ...")
rknn_fp16 = RKNNLite(verbose=False)
rknn_fp16.load_rknn(MODEL_FP16)
rknn_fp16.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
log("   FP16 模型加载完成")

log("6.2 对比推理 (20组随机输入) ...")
cos_sims = []
max_diffs = []
for t in range(20):
    inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
    out_i8 = rknn.inference(inputs=[inp])[0].flatten()
    out_fp16 = rknn_fp16.inference(inputs=[inp])[0].flatten()
    a = out_i8.astype(np.float64)
    b = out_fp16.astype(np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(dot / norm) if norm > 1e-10 else 1.0
    max_diff = float(np.max(np.abs(a - b)))
    cos_sims.append(cos)
    max_diffs.append(max_diff)
    log(f"   #{t+1:2d}: cos={cos:.6f}  max_diff={max_diff:.4f}")

rknn_fp16.release()

cos_arr = np.array(cos_sims)
md_arr = np.array(max_diffs)
log(f"\n--- 精度汇总 ---")
log(f"  平均余弦相似度: {cos_arr.mean():.6f}")
log(f"  最小余弦相似度:  {cos_arr.min():.6f}")
log(f"  余弦相似度标准差: {cos_arr.std():.6f}")
log(f"  平均最大差异:    {md_arr.mean():.4f}")

pass_flag = bool(cos_arr.mean() >= 0.99)
log(f"\n  判定: {'✅ 精度达标! cos >= 0.99' if pass_flag else '❌ 精度不足: cos < 0.99'}")

# =========================================================
section("7. 释放资源")
# =========================================================
log("7.1 释放 INT8 模型...")
rknn.release()
log("   OK")

# =========================================================
section("8. 保存结果")
# =========================================================
results = {
    "model": "models/sensevoice_encoder_ctc_100f_int8_real.rknn",
    "model_size_mb": round(os.path.getsize(MODEL_INT8) / (1024*1024), 1),
    "input_shape": str(INPUT_SHAPE),
    "warmup": WARMUP,
    "rounds": ROUNDS,
    "core_mask": "NPU_CORE_0_1_2",
    "latency": summary,
    "first_inference_ms": round(latencies[0], 3),
    "accuracy": {
        "avg_cosine_similarity": round(float(cos_arr.mean()), 6),
        "min_cosine_similarity": round(float(cos_arr.min()), 6),
        "cos_std": round(float(cos_arr.std()), 6),
        "avg_max_diff": round(float(md_arr.mean()), 4),
    },
    "accuracy_pass": pass_flag,
    "calibration": "real_audio_features",
    "environment_end": {
        "npu_freq_mhz": int(freq) if freq.isdigit() else 0,
        "npu_load": npu_load,
    },
}

with open(JSON_OUT, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log(f"结果已保存: {JSON_OUT}")

log("\n" + "=" * 60)
log("测试完成!")
log(f"日志: {LOG_FILE}")
log(f"JSON: {JSON_OUT}")
log("=" * 60)