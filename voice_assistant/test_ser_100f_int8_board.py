#!/usr/bin/env python3
"""SER 100帧 INT8 模型 — 板端性能测试 + 详细日志

用法:
  1. ADB 连接板端后:
     adb push sensevoice_encoder_ctc_100f_int8.rknn /data/sensevoice/
     adb push test_ser_100f_int8_board.py /data/voice_assistant/
     adb shell "cd /data/voice_assistant && python3 test_ser_100f_int8_board.py"

  2. 日志实时输出到终端, 同时保存到文件: /data/ser_int8_test/ser_100f_int8_test.log
"""
import os, sys, time, json, gc
import numpy as np

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "/data/sensevoice/sensevoice_encoder_ctc_100f_int8.rknn"
MODEL_FP16_PATH = "/data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn"  # 可选精度对比
INPUT_SHAPE = (1, 100, 560)
DTYPE = np.float32
WARMUP = 5
ROUNDS = 50
OUTPUT_DIR = "/data/ser_int8_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "ser_100f_int8_test.log")

# ============================================================
# 日志工具
# ============================================================
def log(msg, end="\n"):
    t = time.strftime("%H:%M:%S.%f")[:12]
    line = f"[{t}] {msg}"
    print(line, end=end, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + end)

def section(title):
    log("")
    log("=" * 65)
    log(f"  {title}")
    log("=" * 65)

def divider():
    log("-" * 65)

# ============================================================
# 系统监控工具
# ============================================================
SYSFS_NPU_LOAD = "/sys/kernel/debug/rknpu/load"
SYSFS_NPU_FREQ = "/sys/class/devfreq/fdab0000.npu/cur_freq"
SYSFS_NPU_AVAIL_FREQ = "/sys/class/devfreq/fdab0000.npu/available_frequencies"

def read_sysfs(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception as e:
        return f"<{e}>"

def get_npu_load():
    return read_sysfs(SYSFS_NPU_LOAD)

def get_npu_freq():
    try:
        return int(read_sysfs(SYSFS_NPU_FREQ)) // 1000000
    except:
        return -1

def get_mem_info():
    """返回 (总MB, 可用MB, 已用MB)"""
    try:
        with open("/proc/meminfo") as f:
            data = f.read()
        total = int([l for l in data.split("\n") if "MemTotal" in l][0].split()[1]) // 1024
        avail = int([l for l in data.split("\n") if "MemAvailable" in l][0].split()[1]) // 1024
        return total, avail, total - avail
    except:
        return -1, -1, -1

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip()) / 1000
    except:
        return -1

def get_npu_temp():
    try:
        zone = "/sys/class/thermal/thermal_zone1/temp"
        if os.path.exists(zone):
            return float(read_sysfs(zone)) / 1000
        return -1
    except:
        return -1

def log_environment(prefix=""):
    """打印当前系统状态"""
    mem_total, mem_avail, mem_used = get_mem_info()
    npu_freq = get_npu_freq()
    npu_load = get_npu_load()
    cpu_temp = get_cpu_temp()
    npu_temp_val = get_npu_temp()
    parts = [f"NPU:{npu_freq}MHz load={npu_load}"]
    if mem_total > 0:
        parts.append(f"RAM:{mem_used}/{mem_total}MB (可用{mem_avail}MB)")
    if cpu_temp > 0:
        parts.append(f"CPU_TEMP:{cpu_temp:.1f}°C")
    if npu_temp_val > 0:
        parts.append(f"NPU_TEMP:{npu_temp_val:.1f}°C")
    log(f"  {prefix}{' | '.join(parts)}")

# ============================================================
# 模型推理测试工具
# ============================================================
def verify_input(ndarray):
    """验证输入数据的合法性"""
    shape = ndarray.shape
    dtype = ndarray.dtype
    rng = (float(ndarray.min()), float(ndarray.max()))
    stats = {
        "shape": str(shape),
        "dtype": str(dtype),
        "range": f"[{rng[0]:.4f}, {rng[1]:.4f}]",
        "mean": float(ndarray.mean()),
        "std": float(ndarray.std()),
        "has_nan": bool(np.any(np.isnan(ndarray))),
        "has_inf": bool(np.any(np.isinf(ndarray))),
    }
    if stats["has_nan"]:
        log("  ⚠️  WARNING: 输入包含 NaN !")
    if stats["has_inf"]:
        log("  ⚠️  WARNING: 输入包含 Inf !")
    if ndarray.nbytes > 100 * 1024 * 1024:
        log(f"  ⚠️  WARNING: 输入数据过大 ({ndarray.nbytes/1024/1024:.0f}MB)")
    return stats

def verify_output(ndarray, label="output"):
    """验证输出数据的合法性, 返回统计"""
    shape = ndarray.shape
    flat = ndarray.flatten()
    rng = (float(flat.min()), float(flat.max()))
    stats = {
        "label": label,
        "shape": str(shape),
        "dtype": str(ndarray.dtype),
        "nbytes_mb": round(ndarray.nbytes / 1024 / 1024, 2),
        "range": f"[{rng[0]:.4f}, {rng[1]:.4f}]",
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "has_nan": bool(np.any(np.isnan(flat))),
        "has_inf": bool(np.any(np.isinf(flat))),
        "n_zero": int(np.count_nonzero(flat == 0)),
        "n_unique": len(np.unique(flat[:1000])),
    }
    # 计算 top-k 活跃度
    abs_flat = np.abs(flat)
    top10_mean = float(np.sort(abs_flat)[-min(10, len(flat)):].mean())
    stats["top10_abs_mean"] = round(top10_mean, 4)
    saturation = float(np.mean(abs_flat > 0.99 * abs_flat.max()))
    stats["saturation_ratio"] = round(saturation, 6)
    if stats["saturation_ratio"] > 0.5:
        log(f"  ⚠️  WARNING: {label} 输出饱和 (top-10均值={top10_mean:.2f})")
    if stats["has_nan"]:
        log(f"  ❌ ERROR: {label} 输出包含 NaN !")
    if stats["has_inf"]:
        log(f"  ❌ ERROR: {label} 输出包含 Inf !")
    if stats["n_zero"] / len(flat) > 0.99:
        log(f"  ⚠️  WARNING: {label} 输出几乎全零 ({stats['n_zero']}/{len(flat)})")
    return stats

def timed_inference(rknn, inputs, label="inference", detail=False):
    """单次推理 + 阶段耗时分解"""
    # 阶段1: 输入准备 (模拟)
    t_prep_start = time.perf_counter()
    inp_list = inputs if isinstance(inputs, list) else [inputs]
    t_prep = (time.perf_counter() - t_prep_start) * 1000 * 0  # 几乎为0, 仅占位

    # 阶段2: NPU 推理
    t_inf_start = time.perf_counter()
    outputs = rknn.inference(inputs=inp_list)
    t_inf = (time.perf_counter() - t_inf_start) * 1000

    # 阶段3: 输出数据处理
    t_post_start = time.perf_counter()
    out_shapes = [o.shape for o in outputs]
    out_bytes = sum(o.nbytes for o in outputs)
    t_post = (time.perf_counter() - t_post_start) * 1000 * 0  # 几乎为0

    total = t_prep + t_inf + t_post

    result = {
        "total_ms": round(total, 3),
        "inference_ms": round(t_inf, 3),
        "output_shapes": [str(s) for s in out_shapes],
        "output_bytes_mb": round(out_bytes / 1024 / 1024, 2),
    }

    if detail:
        result["output_stats"] = [verify_output(o, f"out[{i}]") for i, o in enumerate(outputs)]

    return result, outputs

# ============================================================
# 主测试流程
# ============================================================
def main():
    # 清除旧日志
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log("")
    log("╔" + "═" * 63 + "╗")
    log("║  SER 100帧 INT8 模型 — 板端性能测试                      ║")
    log("╚" + "═" * 63 + "╝")
    log(f"模型: {MODEL_PATH}")
    log(f"输入: {INPUT_SHAPE}")
    log(f"预热: {WARMUP} 次  |  测试: {ROUNDS} 次")
    log(f"日志: {LOG_FILE}")
    log("")

    # ============================================================
    # 阶段0: 环境检查
    # ============================================================
    section("0. 环境检查")

    log("0.1 NPU 信息:")
    log(f"  可用频率: {read_sysfs(SYSFS_NPU_AVAIL_FREQ)}")
    log_environment("空闲: ")

    log("0.2 模型文件检查:")
    for path, label in [(MODEL_PATH, "INT8"), (MODEL_FP16_PATH, "FP16 (精度对比用)")]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            log(f"  ✅ {label}: {path} ({size_mb:.0f}MB)")
        else:
            log(f"  ⚠️  {label}: {path} (不存在, 跳过相关测试)")

    log("0.3 输入数据准备:")
    log(f"  生成随机输入 {INPUT_SHAPE} ...")
    t0 = time.perf_counter()
    dummy_input = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
    t_prep = (time.perf_counter() - t0) * 1000
    inp_stats = verify_input(dummy_input)
    log(f"  输入生成耗时: {t_prep:.2f}ms")
    for k, v in inp_stats.items():
        log(f"    {k}: {v}")

    # ============================================================
    # 阶段1: 模型加载
    # ============================================================
    section("1. 模型加载")

    log("1.1 加载 RKNN 模型 ...")
    log_environment("加载前: ")

    t0 = time.perf_counter()
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(MODEL_PATH)
    t_load = (time.perf_counter() - t0) * 1000
    log(f"  load_rknn 返回: {ret}  |  耗时: {t_load:.1f}ms")
    if ret != 0:
        log(f"  ❌ ERROR: load_rknn 失败, 返回码={ret}")
        sys.exit(1)

    log("")
    log("1.2 初始化 Runtime (NPU 三核) ...")
    log_environment("初始化前: ")

    t0 = time.perf_counter()
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    t_init = (time.perf_counter() - t0) * 1000
    mem_total, mem_avail, mem_used = get_mem_info()
    log(f"  init_runtime 返回: {ret}  |  耗时: {t_init:.1f}ms")
    if ret != 0:
        log(f"  ❌ ERROR: init_runtime 失败, 返回码={ret}")
        sys.exit(1)

    log_environment("加载后: ")
    if mem_total > 0:
        log(f"  模型加载内存开销: ~{mem_used - (mem_total - mem_avail - (mem_total - mem_avail))}MB (待精确计算)")

    # ============================================================
    # 阶段2: 预热推理 + 首次推理分析
    # ============================================================
    section("2. 预热推理 (分析首次/冷启动延迟)")

    log(f"2.1 首次推理 (cold start, 含 JIT/cache 初始化) ...")
    log_environment("推理前: ")
    t0 = time.perf_counter()
    cold_result, cold_outputs = timed_inference(rknn, dummy_input, "cold", detail=True)
    t_cold = (time.perf_counter() - t0) * 1000
    log(f"  首次推理: {cold_result['total_ms']:.3f}ms  (调用方耗时: {t_cold:.3f}ms)")
    log(f"  输出形状: {cold_result['output_shapes']}")
    log(f"  输出大小: {cold_result['output_bytes_mb']}MB")
    for os_ in cold_result.get("output_stats", []):
        log(f"  输出检验: shape={os_['shape']} range={os_['range']} "
            f"mean={os_['mean']:.4f} std={os_['std']:.4f} "
            f"nan={os_['has_nan']} inf={os_['has_inf']} "
            f"zero={os_['n_zero']}/{os_['n_unique']}uniq")
    log_environment("推理后: ")

    log("")
    log(f"2.2 预热 {WARMUP} 次 (稳定 NPU 频率/缓存) ...")
    warmup_times = []
    for i in range(WARMUP):
        r, _ = timed_inference(rknn, dummy_input)
        warmup_times.append(r["total_ms"])
        if i == 0 or i == WARMUP - 1:
            log(f"  warmup #{i+1:2d}: {r['total_ms']:.3f}ms")
    log(f"  预热完成: {WARMUP}次, 末次 {warmup_times[-1]:.3f}ms")
    if warmup_times[-1] < warmup_times[0] * 0.9:
        log(f"  ℹ️  首次推理比末次慢 {((warmup_times[0]/warmup_times[-1])-1)*100:.0f}%, 冷启动影响较大")

    # ============================================================
    # 阶段3: 正式性能测试
    # ============================================================
    section("3. 正式性能测试")

    log(f"3.1 运行 {ROUNDS} 次推理 (逐次计时) ...")
    divider()

    all_times = []
    all_inf_times = []
    all_outputs = []
    rolling_window = 10
    min_time = float("inf")
    max_time = 0.0
    outliers = []
    last_report = 0

    # 记录每个 inference 的详细阶段
    phase_log = {"prep_ms": [], "inference_ms": [], "post_ms": []}

    for i in range(ROUNDS):
        # 每轮都重新生成相同分布的输入 (模拟真实场景)
        if i % 20 == 0:
            test_input = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
        else:
            test_input = dummy_input

        t_start = time.perf_counter()
        result, outputs = timed_inference(rknn, test_input, f"run #{i+1}")
        t_total = (time.perf_counter() - t_start) * 1000

        all_times.append(result["total_ms"])
        all_inf_times.append(result["inference_ms"])
        all_outputs.append(outputs[0].copy() if outputs else None)

        min_time = min(min_time, result["total_ms"])
        max_time = max(max_time, result["total_ms"])

        # 检测异常值 (超过中位数 30%)
        if len(all_times) >= 3:
            median_sofar = np.median(all_times)
            if result["total_ms"] > median_sofar * 1.3:
                outliers.append({
                    "index": i + 1,
                    "time_ms": result["total_ms"],
                    "median_sofar": round(median_sofar, 2),
                    "ratio": round(result["total_ms"] / median_sofar, 2),
                })

        # 每轮都打印详细日志
        marker = ""
        if result["total_ms"] > np.median(warmup_times) * 1.3:
            marker = "  ← ⚠️ 异常偏高"
        out_info = f"out={result['output_shapes'][0]} {result['output_bytes_mb']:.1f}MB"
        log(f"  [#{i+1:3d}/{ROUNDS}] "
            f"推理={result['inference_ms']:.3f}ms "
            f"总={result['total_ms']:.3f}ms "
            f"{out_info}"
            f"{marker}")

        # 每 10 轮输出一次滚动统计
        if (i + 1) % rolling_window == 0:
            window = all_times[-rolling_window:]
            log(f"  --- 最近{rolling_window}轮统计: "
                f"avg={np.mean(window):.3f}ms  "
                f"min={min(window):.3f}ms  "
                f"max={max(window):.3f}ms  "
                f"std={np.std(window):.3f}ms  "
                f"cv={np.std(window)/np.mean(window)*100:.2f}%  "
                f"累计avg={np.mean(all_times):.3f}ms ---")

    divider()

    # ============================================================
    # 阶段4: 结果统计分析
    # ============================================================
    section("4. 结果统计分析")

    times_arr = np.array(all_times)
    inf_arr = np.array(all_inf_times)
    times_sorted = np.sort(times_arr)
    n = len(times_sorted)

    stats = {
        "count": n,
        "avg_ms": round(float(np.mean(times_arr)), 3),
        "median_ms": round(float(np.median(times_arr)), 3),
        "min_ms": round(float(np.min(times_arr)), 3),
        "max_ms": round(float(np.max(times_arr)), 3),
        "std_ms": round(float(np.std(times_arr)), 3),
        "cv_pct": round(float(np.std(times_arr) / np.mean(times_arr) * 100), 3),
        "p50_ms": round(float(np.median(times_arr)), 3),
        "p90_ms": round(float(times_sorted[int(n * 0.90)]), 3),
        "p95_ms": round(float(times_sorted[int(n * 0.95)]), 3),
        "p99_ms": round(float(times_sorted[int(n * 0.99)]), 3),
        "range_ms": round(float(max_time - min_time), 3),
    }

    log("4.1 延迟统计:")
    log(f"  测试次数: {stats['count']}")
    log(f"  平均延迟: {stats['avg_ms']:.3f}ms")
    log(f"  中位延迟: {stats['median_ms']:.3f}ms")
    log(f"  P90 延迟: {stats['p90_ms']:.3f}ms")
    log(f"  P95 延迟: {stats['p95_ms']:.3f}ms")
    log(f"  P99 延迟: {stats['p99_ms']:.3f}ms")
    log(f"  最小延迟: {stats['min_ms']:.3f}ms")
    log(f"  最大延迟: {stats['max_ms']:.3f}ms")
    log(f"  标准差:   {stats['std_ms']:.3f}ms")
    log(f"  变异系数: {stats['cv_pct']:.2f}%")
    log(f"  抖动范围: {stats['range_ms']:.3f}ms")

    log("")
    log("4.2 直方图分布 (10 bins):")
    bins = 10
    bin_edges = np.linspace(times_arr.min(), times_arr.max(), bins + 1)
    for b in range(bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        count = int(np.sum((times_arr >= lo) & (times_arr < hi)) if b < bins - 1
                    else np.sum((times_arr >= lo) & (times_arr <= hi)))
        bar = "█" * count + "░" * (max(0, count))
        pct = count / n * 100
        log(f"  [{lo:7.2f} ~ {hi:7.2f}] {bar} {count:3d} ({pct:4.1f}%)")

    log("")
    log("4.3 异常值分析:")
    if outliers:
        log(f"  发现 {len(outliers)} 个异常值 (>中位数1.3x):")
        for o in outliers:
            log(f"    #{o['index']:3d}: {o['time_ms']:.3f}ms (中位数 {o['median_sofar']:.3f}ms, {o['ratio']:.2f}x)")
    else:
        log(f"  未发现异常值 (所有值在 ±30% 中位数范围内)")

    log("")
    log("4.4 推理耗时 vs 总耗时:")
    inf_ratio = float(np.mean(inf_arr) / np.mean(times_arr) * 100)
    log(f"  推理调用平均: {np.mean(inf_arr):.3f}ms")
    log(f"  总耗时平均:   {np.mean(times_arr):.3f}ms")
    log(f"  推理占比:     {inf_ratio:.1f}%")
    log(f"  额外开销:     {np.mean(times_arr) - np.mean(inf_arr):.3f}ms "
        f"({100-inf_ratio:.1f}%)")

    # ============================================================
    # 阶段5: 输出稳定性分析
    # ============================================================
    section("5. 输出稳定性分析")

    # 检查每次推理的输出是否一致 (用最后10次)
    log("5.1 输出一致性检查 (最后10次推理):")
    if len(all_outputs) >= 10:
        ref_out = all_outputs[-1]
        diffs = []
        for i in range(10):
            idx = len(all_outputs) - 10 + i
            o = all_outputs[idx]
            if o is not None and ref_out is not None:
                diff = float(np.max(np.abs(o - ref_out)))
                diffs.append(diff)
        max_diff = max(diffs) if diffs else -1
        log(f"  最大逐输出差异: {max_diff:.6f}")
        if max_diff < 1e-4:
            log(f"  ✅ 输出一致性好 (差异 < 1e-4)")
        elif max_diff < 0.01:
            log(f"  ⚠️  输出略有浮动 (差异 ~{max_diff:.6f})")
        else:
            log(f"  ❌ 输出差异大 (差异 {max_diff:.6f}), 可能推理不稳定")

    log("")
    log("5.2 输出分布检查 (末次推理):")
    final_out = all_outputs[-1]
    if final_out is not None:
        flat = final_out.flatten()
        log(f"  形状: {final_out.shape}")
        log(f"  dtype: {final_out.dtype}")
        log(f"  范围: [{flat.min():.4f}, {flat.max():.4f}]")
        log(f"  均值: {flat.mean():.4f}")
        log(f"  标准差: {flat.std():.4f}")
        log(f"  NaN: {np.any(np.isnan(flat))}  Inf: {np.any(np.isinf(flat))}")
        # 分布分位数
        for q in [1, 5, 25, 50, 75, 95, 99]:
            v = np.percentile(flat, q)
            log(f"  P{q:2d}: {v:.4f}")
        # 零值占比
        zeros = np.count_nonzero(flat == 0)
        log(f"  零值: {zeros}/{len(flat)} ({zeros/len(flat)*100:.4f}%)")

    # ============================================================
    # 阶段6: 环境稳定性监控
    # ============================================================
    section("6. 环境稳定性监控")

    log("6.1 推理前后环境对比:")
    log_environment("测试结束: ")
    mem_total_end, mem_avail_end, mem_used_end = get_mem_info()
    log(f"  内存变化: 测试前 {mem_used}MB → 测试后 {mem_used_end}MB"
        f" ({mem_used_end - mem_used:+d}MB)")

    log("")
    log("6.2 模型资源释放 (移至精度对比后):")
    # release 移到阶段7之后, 避免 INT8 模型被提前释放

    # ============================================================
    # 阶段7: (可选) 精度对比 — 与 FP16 参考模型比较
    # ============================================================
    has_fp16 = os.path.exists(MODEL_FP16_PATH)
    if has_fp16:
        section("7. 精度对比 (INT8 vs FP16)")

        log("7.1 加载 FP16 参考模型 ...")
        rknn_fp16 = RKNNLite(verbose=False)
        rknn_fp16.load_rknn(MODEL_FP16_PATH)
        rknn_fp16.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

        log("7.2 对比推理 (10组随机输入) ...")
        cos_sims = []
        max_diffs = []
        mean_diffs = []
        for t in range(10):
            inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
            out_i8 = rknn.inference(inputs=[inp])[0].flatten()
            out_fp16 = rknn_fp16.inference(inputs=[inp])[0].flatten()

            # 余弦相似度
            a = out_i8.astype(np.float64)
            b = out_fp16.astype(np.float64)
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            cos = float(dot / norm) if norm > 1e-10 else 1.0
            max_d = float(np.max(np.abs(a - b)))
            mean_d = float(np.mean(np.abs(a - b)))

            cos_sims.append(cos)
            max_diffs.append(max_d)
            mean_diffs.append(mean_d)
            log(f"  #{t+1:2d}: cos={cos:.6f}  max_diff={max_d:.4f}  mean_diff={mean_d:.4f}")

        log("")
        log("7.3 精度汇总:")
        cos_arr = np.array(cos_sims)
        log(f"  平均余弦相似度: {cos_arr.mean():.6f}")
        log(f"  最低余弦相似度: {cos_arr.min():.6f}")
        log(f"  余弦相似度标准差: {cos_arr.std():.6f}")
        log(f"  平均最大绝对差异: {np.mean(max_diffs):.4f}")
        log(f"  平均绝对差异: {np.mean(mean_diffs):.4f}")

        threshold = 0.99
        if cos_arr.mean() >= threshold:
            log(f"  ✅ 精度合格: cos={cos_arr.mean():.6f} >= {threshold}")
        else:
            log(f"  ❌ 精度不足: cos={cos_arr.mean():.6f} < {threshold}")
            log(f"  建议: 使用真实音频校准集重新量化, 或实施混合量化")

        rknn_fp16.release()

    # 释放 INT8 模型资源
    log("")
    log("6.2 释放 INT8 模型资源:")
    t0 = time.perf_counter()
    rknn.release()
    t_rel = (time.perf_counter() - t0) * 1000
    log(f"  release 耗时: {t_rel:.2f}ms")
    time.sleep(0.3)
    log_environment("释放后: ")

    # ============================================================
    # 阶段8: 汇总
    # ============================================================
    section("8. 测试汇总")

    log(f"模型: {MODEL_PATH}")
    log(f"输入: {INPUT_SHAPE}")
    log(f"推理配置: NPU 三核 | 预热 {WARMUP}次 | 测试 {ROUNDS}次")
    log("")

    summary = {
        "model": os.path.basename(MODEL_PATH),
        "model_size_mb": round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 1) if os.path.exists(MODEL_PATH) else -1,
        "input_shape": str(INPUT_SHAPE),
        "warmup": WARMUP,
        "rounds": ROUNDS,
        "core_mask": "NPU_CORE_0_1_2",
    }
    summary["latency"] = stats
    summary["first_inference_ms"] = cold_result["total_ms"]
    summary["inference_ratio_pct"] = round(inf_ratio, 1)
    if has_fp16:
        summary["accuracy"] = {
            "avg_cosine_similarity": round(float(cos_arr.mean()), 6),
            "min_cosine_similarity": round(float(cos_arr.min()), 6),
            "cos_std": round(float(cos_arr.std()), 6),
            "avg_max_diff": round(float(np.mean(max_diffs)), 4),
        }
        summary["accuracy_pass"] = bool(cos_arr.mean() >= 0.99)
    else:
        summary["accuracy"] = None
    summary["environment_end"] = {
        "npu_freq_mhz": get_npu_freq(),
        "npu_load": get_npu_load(),
    }

    json_path = os.path.join(OUTPUT_DIR, "ser_100f_int8_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"JSON 结果: {json_path}")

    log("")
    log("=" * 65)
    log("  测试完成!")
    log(f"  延迟: avg={stats['avg_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  p90={stats['p90_ms']:.1f}ms")
    if has_fp16:
        log(f"  精度: cos={cos_arr.mean():.4f} {'✅' if cos_arr.mean() >= 0.99 else '❌'}")
    log(f"  详细日志: {LOG_FILE}")
    log("=" * 65)

    # 对比预期: 若延迟远高于预期~220ms, 打印诊断
    expected_ms = 220
    if stats["avg_ms"] > expected_ms * 1.5:
        log("")
        log("⚠️  ============ 延迟诊断 ============")
        log(f"⚠️  实际延迟 {stats['avg_ms']:.0f}ms 远超预期 {expected_ms}ms")
        log(f"⚠️  可能原因排查:")
        log(f"⚠️    1. NPU 频率是否过低? (当前={get_npu_freq()}MHz)")
        log(f"⚠️    2. NPU 负载是否过高? (负载={get_npu_load()})")
        log(f"⚠️    3. 模型是否完整加载? 是否存在内存 swap? (内存={get_mem_info()})")
        log(f"⚠️    4. 温度是否过高导致降频? (CPU={get_cpu_temp():.1f}°C NPU={get_npu_temp():.1f}°C)")
        log(f"⚠️    5. 输入形状是否正确? (shape={INPUT_SHAPE})")
        log(f"⚠️    6. 模型是否被人为降级到单核? (实际 core_mask=NPU_CORE_0_1_2)")
        log(f"⚠️    7. 首次推理冷启动是否被计入统计? (首次={cold_result['total_ms']:.0f}ms)")
        log(f"⚠️    8. 对比 FP16 基线: 预期 FP16 ~272ms, INT8 应快 ~20%")
        log(f"⚠️  =================================")

if __name__ == "__main__":
    from rknnlite.api import RKNNLite
    main()