#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vocoder.rknn RTF 测试 + 音频播放
在 RK3588 开发板上使用 NPU 推理

测试内容:
  1. RKNN 模型加载时间
  2. 多轮推理 RTF (Real-Time Factor)
  3. 生成音频并播放
"""

import os
import sys
import time
import json
import struct
import wave
import numpy as np

# ============ 配置 ============
VOCODER_RKNN = "/userdata/qwen3-tts/vocoder/vocoder.rknn"
SAMPLE_RATE = 24000
WARMUP = 5
ROUNDS = 10
OUTPUT_WAV = "/data/vocoder_rknn_test.wav"
LOG_FILE = "/data/vocoder_rknn_test_result.json"

# RKNNLite 核心绑定 (NPU)
CORE_MASK = 1  # 0=auto, 1=NPU0, 2=NPU1, 3=NPU0+NPU1


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_memory_mb():
    try:
        pid = os.getpid()
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except:
        pass
    return 0


def arr_to_wav(audio_float32: np.ndarray, filepath: str, sample_rate: int = SAMPLE_RATE):
    """浮点音频 [-1,1] → 16-bit PCM WAV 文件"""
    audio_int16 = np.clip(audio_float32, -1.0, 1.0) * 32767
    audio_int16 = audio_int16.astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    duration = len(audio_int16) / sample_rate
    return duration


def main():
    log("=" * 60)
    log("vocoder.rknn RTF 测试 + 音频播放")
    log(f"设备: {os.uname().nodename}")
    log("=" * 60)

    results = {
        "test_info": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": os.uname().nodename,
            "model": VOCODER_RKNN,
        }
    }

    # ---------- 1. 加载 RKNN 模型 ----------
    if not os.path.exists(VOCODER_RKNN):
        log(f"❌ 模型不存在: {VOCODER_RKNN}")
        sys.exit(1)

    log(f"\n📦 加载 vocoder.rknn...")
    mem_before = get_memory_mb()

    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        log("❌ rknnlite 未安装 (需要 rknn-toolkit-lite2)")
        sys.exit(1)

    rknn = RKNNLite()
    t0 = time.perf_counter()
    ret = rknn.load_rknn(VOCODER_RKNN)
    if ret != 0:
        log(f"❌ load_rknn 失败: ret={ret}")
        sys.exit(1)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    ret = rknn.init_runtime(core_mask=CORE_MASK)
    if ret != 0:
        log(f"❌ init_runtime 失败: ret={ret}")
        sys.exit(1)
    t_init = time.perf_counter() - t1
    mem_after = get_memory_mb()

    log(f"  ✅ 加载成功: {t_load:.2f}s + init {t_init:.2f}s")
    log(f"  💾 内存: {mem_before:.0f}MB → {mem_after:.0f}MB (+{mem_after-mem_before:.0f}MB)")

    results["model_loading"] = {
        "load_time_s": round(t_load, 3),
        "init_time_s": round(t_init, 3),
        "memory_mb": {
            "before": round(mem_before, 1),
            "after": round(mem_after, 1),
            "model_only": round(mem_after - mem_before, 1)
        }
    }

    # ---------- 2. 获取模型 I/O 信息 ----------
    sdk_info = rknn.get_sdk_version()
    log(f"\n  SDK: {sdk_info}")

    # vocoder.rknn 固定输入输出 (已知 from ONNX 模型)
    num_codebooks = 64
    tokens_per_frame = 16
    audio_len = 122880
    audio_duration = audio_len / SAMPLE_RATE

    log(f"  输入: audio_codes [1, {num_codebooks}, {tokens_per_frame}] int64")
    log(f"  输出: audio_values [1, {audio_len}] float32")
    log(f"  每帧音频: {audio_len} samples ({audio_duration:.2f}s @ {SAMPLE_RATE}Hz)")

    results["model_info"] = {
        "input_shape": [1, num_codebooks, tokens_per_frame],
        "output_shape": [1, audio_len],
        "audio_duration_s": round(audio_duration, 3)
    }

    # ---------- 3. 推理性能测试 ----------
    log("\n" + "=" * 60)
    log("推理性能测试")
    log("=" * 60)

    # 构造输入: audio_codes [1, 64, 16] int64
    test_input = np.random.randint(0, 1024, size=(1, num_codebooks, tokens_per_frame), dtype=np.int64)
    inputs = [test_input]

    # 预热
    log(f"  预热 {WARMUP} 次...")
    for i in range(WARMUP):
        outputs = rknn.inference(inputs=inputs)

    # 正式测试
    log(f"  正式测试 {ROUNDS} 次...")
    times = []
    for i in range(ROUNDS):
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=inputs)
        t = time.perf_counter() - t0
        times.append(t)

        if (i + 1) % 5 == 0:
            log(f"    [{i+1}/{ROUNDS}] {t*1000:.1f}ms")

    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times) * 1000
    max_ms = np.max(times) * 1000
    std_ms = np.std(times) * 1000

    avg_inference_s = avg_ms / 1000
    rtf = avg_inference_s / audio_duration

    log(f"\n  📊 推理性能:")
    log(f"  平均: {avg_ms:.1f}ms")
    log(f"  最小: {min_ms:.1f}ms")
    log(f"  最大: {max_ms:.1f}ms")
    log(f"  标准差: {std_ms:.1f}ms")
    log(f"  输出音频: {audio_duration:.2f}s")
    log(f"  RTF: {avg_inference_s:.4f}s / {audio_duration:.2f}s = {rtf:.4f}")

    if rtf < 0.3:
        log(f"  🟢 实时比: 1:{1/rtf:.0f}x 实时 (优秀)")
    elif rtf < 0.5:
        log(f"  🟢 实时比: 1:{1/rtf:.0f}x 实时 (良好)")
    elif rtf < 1.0:
        log(f"  🟡 实时比: 1:{1/rtf:.0f}x 实时 (可接受)")
    else:
        log(f"  🔴 实时比: {rtf:.2f}x 慢于实时 (需优化)")

    # 与 ONNX CPU 对比 (之前测试 RTF=0.8887)
    log(f"\n  对比: ONNX CPU RTF = 0.8887 (之前测试)")
    log(f"        RKNN NPU RTF = {rtf:.4f}")
    speedup = 0.8887 / rtf if rtf > 0 else float('inf')
    log(f"        NPU 加速比: {speedup:.1f}x")

    results["benchmark"] = {
        "avg_ms": round(avg_ms, 1),
        "min_ms": round(min_ms, 1),
        "max_ms": round(max_ms, 1),
        "std_ms": round(std_ms, 1),
        "audio_duration_s": round(audio_duration, 3),
        "rtf": round(rtf, 4),
        "speedup_vs_onnx_cpu": round(speedup, 1),
        "runs": ROUNDS,
        "warmup": WARMUP
    }

    # ---------- 4. 生成真实音频并播放 ----------
    log("\n" + "=" * 60)
    log("生成音频并播放")
    log("=" * 60)

    # 使用一些有规律的 codec tokens（而不是纯随机）来产生有意义的音频
    # 实际上，随机 codec tokens 会产生噪声，但可以验证管道的完整性
    log(f"  构造音频输入 (1, {num_codebooks}, {tokens_per_frame})...")

    # 为更好的听觉效果，使用带一定结构的数据
    # 低频区域集中一些值，高频区域用零
    structured_input = np.zeros((1, num_codebooks, tokens_per_frame), dtype=np.int64)
    for cb in range(min(8, num_codebooks)):
        # 前8个 codebook 填充一些规律性的值
        structured_input[0, cb, :] = np.random.randint(0, 256, size=tokens_per_frame)
    # 中间codebook 填充中等值
    for cb in range(8, min(32, num_codebooks)):
        structured_input[0, cb, :] = np.random.randint(0, 512, size=tokens_per_frame)

    log(f"  运行 RKNN 推理...")
    t0 = time.perf_counter()
    outputs = rknn.inference(inputs=[structured_input])
    gen_time = time.perf_counter() - t0

    audio_data = outputs[0]
    log(f"  输出形状: {audio_data.shape}")
    log(f"  输出范围: [{audio_data.min():.4f}, {audio_data.max():.4f}]")

    # 保存为 WAV
    duration = arr_to_wav(audio_data.flatten(), OUTPUT_WAV)
    log(f"  ✅ 音频已保存: {OUTPUT_WAV} ({duration:.2f}s)")
    log(f"  生成耗时: {gen_time*1000:.1f}ms")

    results["audio_generation"] = {
        "wav_file": OUTPUT_WAV,
        "duration_s": round(duration, 3),
        "gen_time_ms": round(gen_time*1000, 1)
    }

    # 播放音频
    log(f"\n  🔊 播放音频 (aplay)...")
    os.system(f"aplay -D plughw:0,0 {OUTPUT_WAV} 2>/dev/null || aplay {OUTPUT_WAV} 2>/dev/null")
    log(f"  ✅ 播放完成")

    # ---------- 5. 常驻检查 ----------
    log("\n" + "=" * 60)
    log("常驻确认")
    log("=" * 60)

    # 再次推理确认模型保持加载
    t0 = time.perf_counter()
    outputs2 = rknn.inference(inputs=inputs)
    t2 = time.perf_counter() - t0
    log(f"  常驻推理正常: {t2*1000:.1f}ms")
    mem_now = get_memory_mb()
    log(f"  当前内存: {mem_now:.0f}MB")

    results["residency_confirm"] = {
        "status": "passed" if outputs2 is not None else "failed",
        "inference_ms": round(t2*1000, 1),
        "memory_mb": round(mem_now, 1)
    }

    # ---------- 释放资源 ----------
    rknn.release()

    # ---------- 保存结果 ----------
    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\n📝 结果已保存: {LOG_FILE}")

    # 打印 JSON 摘要到 stdout
    print(f"\n---RESULT_JSON---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"---RESULT_JSON_END---")

    log(f"\n✅ 测试完成")


if __name__ == "__main__":
    main()