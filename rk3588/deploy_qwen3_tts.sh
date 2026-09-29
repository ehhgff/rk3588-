#!/bin/bash
#====================================================================
# Qwen3-TTS 模型部署 + 常驻/RTF 测试脚本
#
# 用法:
#   ./deploy_qwen3_tts.sh              # 推送模型并运行测试
#   ./deploy_qwen3_tts.sh --test-only  # 只运行测试（模型已推送）
#   ./deploy_qwen3_tts.sh --push-only  # 只推送模型
#   ./deploy_qwen3_tts.sh --help       # 显示帮助
#====================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_SRC="/home/ubuntu/桌面/ai/qwen3-tts-rk3588"
BOARD_DIR="/userdata/qwen3-tts"
ADB="adb"

do_push=1
do_test=1

for arg in "$@"; do
    case "$arg" in
        --test-only) do_push=0; do_test=1 ;;
        --push-only) do_push=1; do_test=0 ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --test-only   只运行测试（模型已推送）"
            echo "  --push-only   只推送模型"
            echo "  --help        显示此帮助"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo " Qwen3-TTS 模型部署 + 常驻/RTF 测试"
echo "=========================================="

# --- 检查 ADB ---
echo ""
echo "[检查] ADB 连接..."
if ! $ADB devices 2>/dev/null | grep -q "device$"; then
    echo "  ❌ ADB 未连接或没有设备"
    exit 1
fi
echo "  ✅ ADB 已连接"

# --- 检查模型源目录 ---
if [ ! -d "$MODEL_SRC" ]; then
    echo "  ❌ 模型源目录不存在: $MODEL_SRC"
    exit 1
fi

# ============ 推送模型 ============
if [ "$do_push" -eq 1 ]; then
    echo ""
    echo "=========================================="
    echo " [1/3] 推送模型文件到开发板"
    echo "=========================================="

    # 检查板子磁盘空间
    avail=$(adb shell "df -m /userdata | tail -1 | awk '{print \$4}'")
    echo "  板子可用空间: ${avail}MB"
    echo "  模型总大小约: 3.5GB"
    echo ""

    # 创建目录结构
    echo "  创建目录结构..."
    $ADB shell "mkdir -p ${BOARD_DIR}/predictor/code_predictor"
    $ADB shell "mkdir -p ${BOARD_DIR}/predictor/embeddings"
    $ADB shell "mkdir -p ${BOARD_DIR}/talker/embeddings"
    $ADB shell "mkdir -p ${BOARD_DIR}/vocoder"

    # 推送文件
    echo ""
    echo "  --- vocoder ---"
    echo "  推送 vocoder.onnx (436MB)..."
    $ADB push "${MODEL_SRC}/vocoder/vocoder.onnx" "${BOARD_DIR}/vocoder/vocoder.onnx" 2>&1 | tail -1

    echo ""
    echo "  --- predictor ---"
    echo "  推送 code_predictor_core.onnx..."
    $ADB push "${MODEL_SRC}/predictor/code_predictor/code_predictor_core.onnx" "${BOARD_DIR}/predictor/code_predictor/code_predictor_core.onnx" 2>&1 | tail -1
    echo "  推送 code_predictor_core.onnx.data (315MB)..."
    $ADB push "${MODEL_SRC}/predictor/code_predictor/code_predictor_core.onnx.data" "${BOARD_DIR}/predictor/code_predictor/code_predictor_core.onnx.data" 2>&1 | tail -1
    echo "  推送 config.json..."
    $ADB push "${MODEL_SRC}/predictor/code_predictor/config.json" "${BOARD_DIR}/predictor/code_predictor/config.json" 2>&1 | tail -1
    echo "  推送 predictor/embeddings/..."
    $ADB push "${MODEL_SRC}/predictor/embeddings/" "${BOARD_DIR}/predictor/embeddings/" 2>&1 | tail -1

    echo ""
    echo "  --- talker ---"
    echo "  推送 tokenizer.json (11MB)..."
    $ADB push "${MODEL_SRC}/talker/tokenizer.json" "${BOARD_DIR}/talker/tokenizer.json" 2>&1 | tail -1
    echo "  推送 talker/embeddings/ (1.4GB, 耗时较长)..."
    $ADB push "${MODEL_SRC}/talker/embeddings/" "${BOARD_DIR}/talker/embeddings/" 2>&1 | tail -1

    # 可选：推送 GGUF 模型（如果空间允许）
    echo ""
    echo "  --- GGUF 模型 (可选) ---"
    talker_gguf_size=$(du -m "${MODEL_SRC}/talker/talker-q8_0.gguf" | cut -f1)
    echo "  talker-q8_0.gguf (${talker_gguf_size}MB) [如果空间允许]"
    $ADB push "${MODEL_SRC}/talker/talker-q8_0.gguf" "${BOARD_DIR}/talker/talker-q8_0.gguf" 2>&1 | tail -1

    echo ""
    echo "  ✅ 模型文件推送完成"
fi

# ============ 推送测试脚本 ============
echo ""
echo "=========================================="
echo " [2/3] 推送测试脚本"
echo "=========================================="

# 生成测试脚本
TEST_SCRIPT="${SCRIPT_DIR}/qwen3_tts_test.py"
cat > "$TEST_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS 常驻 + RTF 测试脚本
在 RK3588 开发板上执行

测试内容:
  1. 模型加载时间 (预测器 + 声码器)
  2. 模型常驻 - 长时间保持加载状态
  3. RTF (Real-Time Factor) - 推理时间 / 音频时长
  4. 内存占用
"""

import os
import sys
import time
import json
import numpy as np
import threading
import signal

# ============ 配置 ============
MODEL_DIR = "/userdata/qwen3-tts"
PREDICTOR_MODEL = os.path.join(MODEL_DIR, "predictor/code_predictor/code_predictor_core.onnx")
PREDICTOR_DATA = os.path.join(MODEL_DIR, "predictor/code_predictor/code_predictor_core.onnx.data")
PREDICTOR_CONFIG = os.path.join(MODEL_DIR, "predictor/code_predictor/config.json")
PREDICTOR_EMBED = os.path.join(MODEL_DIR, "predictor/embeddings")
VOCODER_MODEL = os.path.join(MODEL_DIR, "vocoder/vocoder.onnx")
TALKER_EMBED = os.path.join(MODEL_DIR, "talker/embeddings")
TALKER_TOKENIZER = os.path.join(MODEL_DIR, "talker/tokenizer.json")
TALKER_GGUF = os.path.join(MODEL_DIR, "talker/talker-q8_0.gguf")

LOG_FILE = "/data/qwen3_tts_test_result.json"
SAMPLE_RATE = 24000

# 测试配置
INFERENCE_WARMUP = 3      # 预热次数
INFERENCE_ROUNDS = 10     # 正式测试次数
RESIDENCY_TIME = 60       # 常驻测试时间（秒）
SEQ_LENGTHS = [32, 64, 128, 256]  # 测试不同序列长度

# 全局变量
predictor_sess = None
vocoder_sess = None
predictor_dims = {}
vocoder_dims = {}
keep_running = True


def log(msg):
    """带时间戳的日志"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_memory_mb():
    """获取当前进程内存占用 (MB)"""
    try:
        import subprocess
        pid = os.getpid()
        result = subprocess.run(
            ["cat", f"/proc/{pid}/status"],
            capture_output=True, text=True, timeout=2
        )
        for line in result.stdout.split("\n"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except:
        pass
    return 0


def get_system_memory():
    """获取系统内存信息"""
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                if parts[0].rstrip(":") in ["MemTotal", "MemFree", "MemAvailable"]:
                    mem[parts[0].rstrip(":")] = int(parts[1])
            return mem
    except:
        return {}


def load_models():
    """加载 ONNX 模型"""
    global predictor_sess, vocoder_sess, predictor_dims, vocoder_dims
    results = {}
    mem_before = get_memory_mb()

    log("=" * 60)
    log("Qwen3-TTS 测试 - 模型加载")
    log("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        log("❌ onnxruntime 未安装")
        return None

    # 检查文件
    models_to_check = {
        "Predictor ONNX": PREDICTOR_MODEL,
        "Vocoder ONNX": VOCODER_MODEL,
    }
    all_ok = True
    for name, path in models_to_check.items():
        if not os.path.exists(path):
            log(f"❌ {name} 不存在: {path}")
            all_ok = False
    if not all_ok:
        return None

    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.enable_cpu_mem_arena = True
    sess_options.enable_mem_reuse = True

    # 加载 Predictor
    log(f"\n📦 加载 Predictor: {os.path.basename(PREDICTOR_MODEL)}")
    try:
        t0 = time.perf_counter()
        predictor_sess = ort.InferenceSession(
            PREDICTOR_MODEL, sess_options, providers=providers
        )
        t_load = time.perf_counter() - t0
        predictor_dims = {
            "inputs": [
                {"name": i.name, "shape": i.shape, "type": i.type}
                for i in predictor_sess.get_inputs()
            ],
            "outputs": [
                {"name": o.name, "shape": o.shape, "type": o.type}
                for o in predictor_sess.get_outputs()
            ]
        }
        log(f"  ✅ 加载成功: {t_load:.2f}s")
        log(f"     输入: {predictor_sess.get_inputs()[0].name} {predictor_sess.get_inputs()[0].shape}")
        log(f"     输出: {predictor_sess.get_outputs()[0].name} {predictor_sess.get_outputs()[0].shape}")
        results["predictor"] = {"load_time_s": round(t_load, 3)}
    except Exception as e:
        log(f"  ❌ 加载失败: {e}")
        results["predictor"] = {"error": str(e)}

    # 加载 Vocoder
    log(f"\n📦 加载 Vocoder: {os.path.basename(VOCODER_MODEL)}")
    try:
        t0 = time.perf_counter()
        vocoder_sess = ort.InferenceSession(
            VOCODER_MODEL, sess_options, providers=providers
        )
        t_load = time.perf_counter() - t0
        vocoder_dims = {
            "inputs": [
                {"name": i.name, "shape": i.shape, "type": i.type}
                for i in vocoder_sess.get_inputs()
            ],
            "outputs": [
                {"name": o.name, "shape": o.shape, "type": o.type}
                for o in vocoder_sess.get_outputs()
            ]
        }
        input_shape = vocoder_sess.get_inputs()[0].shape
        output_shape = vocoder_sess.get_outputs()[0].shape
        log(f"  ✅ 加载成功: {t_load:.2f}s")
        log(f"     输入: {vocoder_sess.get_inputs()[0].name} {input_shape}")
        log(f"     输出: {vocoder_sess.get_outputs()[0].name} {output_shape}")

        # 计算 Vocoder RTF 基准
        if output_shape[1] and input_shape[1] and input_shape[2]:
            audio_len_samples = output_shape[1]
            audio_duration = audio_len_samples / SAMPLE_RATE
            log(f"     每帧音频: {audio_len_samples} samples ({audio_duration:.2f}s @ {SAMPLE_RATE}Hz)")
            results["vocoder"] = {
                "load_time_s": round(t_load, 3),
                "audio_samples_per_inference": audio_len_samples,
                "audio_duration_per_inference_s": round(audio_duration, 3)
            }
        else:
            results["vocoder"] = {"load_time_s": round(t_load, 3)}
    except Exception as e:
        log(f"  ❌ 加载失败: {e}")
        results["vocoder"] = {"error": str(e)}

    mem_after = get_memory_mb()
    log(f"\n💾 内存占用: {mem_before:.0f}MB → {mem_after:.0f}MB (+{mem_after - mem_before:.0f}MB)")

    results["memory_mb"] = {
        "before": round(mem_before, 1),
        "after": round(mem_after, 1),
        "model_only": round(mem_after - mem_before, 1)
    }
    results["system_memory"] = get_system_memory()

    return results


def test_predictor_inference(seq_len):
    """测试 Predictor 推理速度"""
    global predictor_sess
    if predictor_sess is None:
        return None

    batch = 1
    hidden_size = 1024

    # 创建随机输入
    hidden_states = np.random.randn(batch, seq_len, hidden_size).astype(np.float32)
    position_ids = np.tile(np.arange(seq_len, dtype=np.int64), (batch, 1))

    input_names = [i.name for i in predictor_sess.get_inputs()]
    feed = {
        input_names[0]: hidden_states,
        input_names[1]: position_ids
    }

    # 预热
    for _ in range(INFERENCE_WARMUP):
        predictor_sess.run(None, feed)

    # 正式测试
    times = []
    for _ in range(INFERENCE_ROUNDS):
        t0 = time.perf_counter()
        outputs = predictor_sess.run(None, feed)
        t = time.perf_counter() - t0
        times.append(t)

    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times) * 1000
    max_ms = np.max(times) * 1000
    std_ms = np.std(times) * 1000

    log(f"  SeqLen={seq_len:3d}: {avg_ms:.1f}ms ± {std_ms:.1f}ms (min={min_ms:.1f}ms, max={max_ms:.1f}ms)")

    return {
        "seq_len": seq_len,
        "avg_ms": round(avg_ms, 1),
        "min_ms": round(min_ms, 1),
        "max_ms": round(max_ms, 1),
        "std_ms": round(std_ms, 1),
        "runs": INFERENCE_ROUNDS
    }


def test_vocoder_inference():
    """测试 Vocoder 推理速度"""
    global vocoder_sess
    if vocoder_sess is None:
        return None

    try:
        # 获取输入输出形状
        input_shape = vocoder_sess.get_inputs()[0].shape
        output_shape = vocoder_sess.get_outputs()[0].shape

        # 创建随机 audio_codes
        # shape: [1, num_codebooks, tokens_per_frame]
        num_codebooks = input_shape[1] if input_shape[1] else 64
        tokens_per_frame = input_shape[2] if input_shape[2] else 16

        audio_codes = np.random.randint(0, 1024, size=(1, num_codebooks, tokens_per_frame), dtype=np.int64)

        feed = {vocoder_sess.get_inputs()[0].name: audio_codes}

        # 预热
        for _ in range(INFERENCE_WARMUP):
            vocoder_sess.run(None, feed)

        # 正式测试
        times = []
        for _ in range(INFERENCE_ROUNDS):
            t0 = time.perf_counter()
            outputs = vocoder_sess.run(None, feed)
            t = time.perf_counter() - t0
            times.append(t)

        avg_ms = np.mean(times) * 1000
        min_ms = np.min(times) * 1000
        max_ms = np.max(times) * 1000
        std_ms = np.std(times) * 1000

        # 计算 RTF
        audio_samples = output_shape[1]  # 122880 for 5.12s @ 24kHz
        audio_duration = audio_samples / SAMPLE_RATE  # seconds
        avg_inference_s = avg_ms / 1000
        rtf = avg_inference_s / audio_duration

        log(f"  Vocoder 推理: {avg_ms:.1f}ms ± {std_ms:.1f}ms")
        log(f"  输出音频: {audio_samples} samples ({audio_duration:.2f}s)")
        log(f"  RTF: {avg_inference_s:.3f}s / {audio_duration:.2f}s = {rtf:.4f}")
        log(f"    → 实时比: 1:{1/rtf:.1f}x 实时" if rtf < 1 else f"    → 慢于实时: {rtf:.2f}x")

        return {
            "avg_ms": round(avg_ms, 1),
            "min_ms": round(min_ms, 1),
            "max_ms": round(max_ms, 1),
            "std_ms": round(std_ms, 1),
            "audio_duration_s": round(audio_duration, 3),
            "rtf": round(rtf, 4),
            "rtf_description": f"1:{1/rtf:.1f}x 实时" if rtf < 1 else f"{rtf:.2f}x 慢于实时",
            "runs": INFERENCE_ROUNDS
        }
    except Exception as e:
        log(f"  ❌ Vocoder 推理失败: {e}")
        return {"error": str(e)}


def test_residency():
    """测试模型常驻能力 - 保持加载状态并监控"""
    global keep_running
    log("\n" + "=" * 60)
    log("常驻测试 - 模型保持加载状态")
    log("=" * 60)
    log(f"保持加载 {RESIDENCY_TIME} 秒，监控内存变化")

    memory_samples = []
    start = time.time()

    # 定期执行推理并记录内存
    for i in range(RESIDENCY_TIME):
        if not keep_running:
            break
        time.sleep(1)

        # 每 5 秒记录一次内存
        if i % 5 == 0:
            mem = get_memory_mb()
            memory_samples.append({"t": i, "memory_mb": round(mem, 1)})

            # 每 10 秒做一次推理确认
            if i % 10 == 0 and i > 0:
                try:
                    if vocoder_sess is not None:
                        audio_codes = np.random.randint(0, 1024, size=(1, 64, 16), dtype=np.int64)
                        t0 = time.perf_counter()
                        vocoder_sess.run(None, {vocoder_sess.get_inputs()[0].name: audio_codes})
                        inf_time = (time.perf_counter() - t0) * 1000
                        log(f"  常驻检查 [{i}s]: 推理正常 ({inf_time:.0f}ms), 内存={mem:.0f}MB")
                except Exception as e:
                    log(f"  ❌ 常驻检查 [{i}s]: 推理失败: {e}")
                    return {"status": "failed", "error": str(e), "runtime_s": i}

    elapsed = time.time() - start
    mem_start = memory_samples[0]["memory_mb"] if memory_samples else 0
    mem_end = memory_samples[-1]["memory_mb"] if memory_samples else 0
    mem_max = max(m["memory_mb"] for m in memory_samples) if memory_samples else 0

    log(f"\n✅ 常驻测试完成: {elapsed:.0f}s")
    log(f"  内存变化: {mem_start:.0f}MB → {mem_end:.0f}MB (峰值: {mem_max:.0f}MB)")
    log(f"  内存泄漏: {'无 (稳定)' if abs(mem_end - mem_start) < 10 else f'疑似泄漏 {mem_end - mem_start:.0f}MB'}")

    return {
        "status": "passed",
        "runtime_s": elapsed,
        "memory_samples": memory_samples,
        "memory_start_mb": round(mem_start, 1),
        "memory_end_mb": round(mem_end, 1),
        "memory_peak_mb": round(mem_max, 1),
        "memory_leak_mb": round(mem_end - mem_start, 1)
    }


def signal_handler(signum, frame):
    global keep_running
    log(f"\n收到信号 {signum}，正在停止...")
    keep_running = False


def main():
    global keep_running
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log("=" * 60)
    log("Qwen3-TTS 常驻 & RTF 测试")
    log(f"设备: {os.uname().nodename}")
    log(f"Python: {sys.version}")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    results = {
        "test_info": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": os.uname().nodename,
            "python_version": sys.version,
            "model_dir": MODEL_DIR,
        }
    }

    # ============ 1. 加载模型 ============
    log("\n" + "=" * 60)
    log("阶段 1: 模型加载")
    log("=" * 60)

    load_results = load_models()
    if load_results is None:
        log("\n❌ 模型加载失败，退出测试")
        results["status"] = "failed"
        results["error"] = "model loading failed"
        save_results(results)
        sys.exit(1)

    results["model_loading"] = load_results

    # ============ 2. Predictor 推理性能 ============
    log("\n" + "=" * 60)
    log("阶段 2: Predictor 推理性能 (不同序列长度)")
    log("=" * 60)

    if predictor_sess is not None:
        predictor_results = {}
        for seq_len in SEQ_LENGTHS:
            r = test_predictor_inference(seq_len)
            if r:
                predictor_results[f"seq_{seq_len}"] = r
        results["predictor_benchmark"] = predictor_results
    else:
        log("  ⏭️  Predictor 未加载，跳过")

    # ============ 3. Vocoder 推理性能 + RTF ============
    log("\n" + "=" * 60)
    log("阶段 3: Vocoder 推理性能 & RTF")
    log("=" * 60)

    if vocoder_sess is not None:
        vocoder_result = test_vocoder_inference()
        if vocoder_result:
            results["vocoder_benchmark"] = vocoder_result
    else:
        log("  ⏭️  Vocoder 未加载，跳过")

    # ============ 4. 常驻测试 ============
    log("\n" + "=" * 60)
    log("阶段 4: 模型常驻测试")
    log("=" * 60)

    residency_result = test_residency()
    results["residency_test"] = residency_result

    # ============ 5. 总结 ============
    log("\n" + "=" * 60)
    log("测试总结")
    log("=" * 60)

    if predictor_sess is not None and vocoder_sess is not None:
        log("  ✅ Predictor + Vocoder 均成功加载")

        if "vocoder_benchmark" in results:
            vb = results["vocoder_benchmark"]
            if "rtf" in vb:
                rtf = vb["rtf"]
                if rtf < 0.5:
                    log(f"  🟢 Vocoder RTF 优秀: {rtf:.4f} ({vb['rtf_description']})")
                elif rtf < 1.0:
                    log(f"  🟡 Vocoder RTF 可接受: {rtf:.4f} ({vb['rtf_description']})")
                else:
                    log(f"  🔴 Vocoder RTF 需要优化: {rtf:.4f} ({vb['rtf_description']})")

    if "residency_test" in results:
        rt = results["residency_test"]
        if rt.get("status") == "passed":
            log(f"  🟢 常驻测试通过: {rt['runtime_s']}s 无异常")
            if rt.get("memory_leak_mb", 0) < 10:
                log(f"  🟢 内存稳定: {rt['memory_start_mb']}MB → {rt['memory_end_mb']}MB")
            else:
                log(f"  🟡 内存轻微增长: {rt['memory_leak_mb']}MB")
        else:
            log(f"  🔴 常驻测试失败: {rt.get('error', '未知错误')}")

    # ============ 保存结果 ============
    save_results(results)

    log(f"\n📝 测试结果已保存到: {LOG_FILE}")
    log("✅ 测试完成")
    return 0


def save_results(results):
    """保存测试结果到 JSON 文件"""
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log(f"  保存结果失败: {e}")

    # 也打印保存路径到 stdout 方便 adb 获取
    print(f"\n---RESULT_JSON---\n{json.dumps(results, indent=2, ensure_ascii=False)}\n---RESULT_JSON_END---")


if __name__ == "__main__":
    sys.exit(main())
PYEOF

echo "  ✅ 测试脚本已创建: ${TEST_SCRIPT}"

echo ""
echo "  推送测试脚本到板子..."
$ADB push "${TEST_SCRIPT}" "${BOARD_DIR}/qwen3_tts_test.py" 2>&1 | tail -1
$ADB shell "chmod +x ${BOARD_DIR}/qwen3_tts_test.py"
echo "  ✅ 脚本已推送"

# ============ 运行测试 ============
if [ "$do_test" -eq 1 ]; then
    echo ""
    echo "=========================================="
    echo " [3/3] 运行测试"
    echo "=========================================="
    echo ""
    echo "模型目录: ${BOARD_DIR}"
    echo "测试脚本: ${BOARD_DIR}/qwen3_tts_test.py"
    echo "结果文件: /data/qwen3_tts_test_result.json"
    echo ""
    echo "启动测试..."
    echo "=========================================="
    echo ""

    $ADB shell "python3 ${BOARD_DIR}/qwen3_tts_test.py" 2>&1

    echo ""
    echo "=========================================="
    echo " 测试完成"
    echo "=========================================="
    echo ""
    echo "获取测试结果:"
    echo "  adb shell cat /data/qwen3_tts_test_result.json"
else
    echo ""
    echo "=========================================="
    echo " 推送完成。运行测试:"
    echo "  adb shell python3 ${BOARD_DIR}/qwen3_tts_test.py"
    echo "=========================================="
fi