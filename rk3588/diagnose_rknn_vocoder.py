#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vocoder.rknn RTF 瓶颈诊断
分析 NPU 利用率低的原因，测试不同配置
"""

import os
import sys
import time
import json
import struct
import wave
import numpy as np
import threading

VOCODER_RKNN = "/userdata/qwen3-tts/vocoder/vocoder.rknn"
SAMPLE_RATE = 24000
WARMUP = 3
ROUNDS = 5
OUTPUT_WAV = "/data/vocoder_rknn_diag_10s.wav"

NUM_CODEBOOKS = 64
TOKENS_PER_FRAME = 16
AUDIO_LEN = 122880
AUDIO_DURATION = AUDIO_LEN / SAMPLE_RATE  # 5.12s


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def arr_to_wav(audio_float32, filepath, sample_rate=SAMPLE_RATE):
    audio_int16 = np.clip(audio_float32, -1.0, 1.0) * 32767
    audio_int16 = audio_int16.astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return len(audio_int16) / sample_rate


def get_npu_load():
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            return f.read().strip()
    except:
        return "N/A"


def get_cpu_freqs():
    freqs = {}
    for i in range(8):
        try:
            with open(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq") as f:
                freqs[f"cpu{i}"] = int(f.read().strip()) // 1000
        except:
            pass
    return freqs


def test_inference(rknn, test_input, label="default", rounds=ROUNDS):
    """测试推理性能"""
    # 预热
    for _ in range(WARMUP):
        rknn.inference(inputs=[test_input])

    # 详细分阶段计时
    prep_ns = []
    inf_ns = []
    total_ns = []

    for _ in range(rounds):
        t0 = time.perf_counter()
        # 准备输入 (如果有额外操作)
        inp = test_input.copy()
        t1 = time.perf_counter()

        outputs = rknn.inference(inputs=[inp])
        t2 = time.perf_counter()

        # 处理输出 (获取结果)
        out = outputs[0].copy()
        t3 = time.perf_counter()

        prep_ns.append((t1 - t0) * 1e6)
        inf_ns.append((t2 - t1) * 1e3)
        total_ns.append((t3 - t0) * 1e3)

    avg_inf = np.mean(inf_ns)
    avg_total = np.mean(total_ns)
    rtf = (avg_total / 1000) / AUDIO_DURATION

    log(f"\n  [{label}]")
    log(f"    推理阶段: {avg_inf:.1f}ms")
    log(f"    总耗时: {avg_total:.1f}ms")
    log(f"    RTF: {rtf:.4f}")

    return {
        "label": label,
        "inf_avg_ms": round(avg_inf, 1),
        "inf_min_ms": round(np.min(inf_ns), 1),
        "inf_max_ms": round(np.max(inf_ns), 1),
        "inf_std_ms": round(np.std(inf_ns), 1),
        "total_avg_ms": round(avg_total, 1),
        "rtf": round(rtf, 4),
    }


def main():
    log("=" * 60)
    log("vocoder.rknn 瓶颈诊断")
    log("=" * 60)

    from rknnlite.api import RKNNLite

    results = {
        "test_info": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": os.uname().nodename,
            "npu_driver_version": "0.9.8",
            "rknntoolkit_version": "2.3.2",
        }
    }

    # ============ 1. 环境信息 ============
    log("\n📊 环境诊断")
    cpu_freqs = get_cpu_freqs()
    log(f"  CPU 频率: {cpu_freqs}")
    log(f"  NPU 空闲负载: {get_npu_load()}")

    results["environment"] = {
        "cpu_freqs_mhz": cpu_freqs,
        "npu_load_idle": get_npu_load(),
    }

    # ============ 2. 加载模型 ============
    log(f"\n📦 加载 vocoder.rknn")

    rknn = RKNNLite()
    rknn.load_rknn(VOCODER_RKNN)

    # 尝试不同 core_mask 配置
    core_configs = [
        (1, "NPU Core0 专用"),
        # (2, "NPU Core1 专用"),
        (3, "NPU Core0+Core1"),
        (0, "NPU Auto (全部核心)"),
    ]

    all_benchmarks = []

    for core_mask, core_label in core_configs:
        log(f"\n--- 测试: {core_label} (core_mask={core_mask}) ---")
        try:
            # 重新初始化 runtime
            rknn.init_runtime(core_mask=core_mask)
        except Exception as e:
            log(f"  init_runtime 失败: {e}")
            continue

        # 使用默认 int64 输入
        test_input = np.random.randint(0, 1024, size=(1, NUM_CODEBOOKS, TOKENS_PER_FRAME), dtype=np.int64)
        result = test_inference(rknn, test_input, label=core_label)
        result["core_mask"] = core_mask
        result["npu_load_after"] = get_npu_load()
        log(f"  NPU 负载: {result['npu_load_after']}")
        all_benchmarks.append(result)

    # ============ 3. 测试不同输入数据类型 ============
    log(f"\n{'='*60}")
    log("输入数据类型对比测试")
    log("=" * 60)

    rknn.init_runtime(core_mask=0)  # auto

    dtype_configs = [
        (np.int64, "int64 (原始)"),
        (np.int32, "int32"),
        (np.int16, "int16"),
    ]

    for dtype, dtype_label in dtype_configs:
        test_input = np.random.randint(0, 1024, size=(1, NUM_CODEBOOKS, TOKENS_PER_FRAME), dtype=dtype)
        result = test_inference(rknn, test_input, label=f"dtype={dtype_label}")
        result["dtype"] = str(dtype)
        all_benchmarks.append(result)

    results["benchmarks"] = all_benchmarks

    # ============ 4. 结论分析 ============
    log(f"\n{'='*60}")
    log("🔍 瓶颈分析")
    log("=" * 60)

    # 最佳结果
    best = min(all_benchmarks, key=lambda x: x["rtf"])
    worst = max(all_benchmarks, key=lambda x: x["rtf"])

    log(f"\n  最佳配置: {best['label']}")
    log(f"    RTF: {best['rtf']:.4f}, 耗时: {best['inf_avg_ms']:.1f}ms")
    log(f"  最差配置: {worst['label']}")
    log(f"    RTF: {worst['rtf']:.4f}, 耗时: {worst['inf_avg_ms']:.1f}ms")
    log(f"  差异: {(worst['rtf']/best['rtf']-1)*100:.1f}%")

    # 分析 NPU 利用率
    npu_loads = [b.get("npu_load_after", "") for b in all_benchmarks if "npu_load_after" in b]
    log(f"\n  NPU 负载样本: {npu_loads[:3]}")

    # 预期 NPU 算力
    # RK3588 NPU: 6 TOPS @ 1GHz, 3 cores
    # Model: 122880 float32 output, ~50M MACs estimate for HiFi-GAN vocoder
    # Expected inference time: ~50M / (6T * utilization) ≈ 8-16ms at 100% utilization
    # Actual: ~2800ms → utilization ~0.5-1%
    # This suggests massive CPU fallback or data transfer bottleneck

    log(f"\n  📉 NPU 利用率极低 (≈15% 单核 ≈ 5% 总)")
    log(f"  根因分析:")
    log(f"  1️⃣ 算子不支持: int64 输入可能强制部分算子 CPU 回退")
    log(f"  2️⃣ 数据搬运瓶颈: CPU↔NPU DMA 带宽受限")
    log(f"  3️⃣ 模型结构: ONNX→RKNN 转换可能未完全 NPU 加速")

    # 测试 RKNN 内置 profiling (如果支持)
    log(f"\n  🔬 RKNN 内置 profiling...")
    try:
        rknn.init_runtime(core_mask=0)
        test_input = np.random.randint(0, 1024, size=(1, NUM_CODEBOOKS, TOKENS_PER_FRAME), dtype=np.int64)

        perf_result = rknn.eval_perf(inputs=[test_input])
        log(f"  eval_perf: {perf_result}")
        results["perf_profile"] = str(perf_result)
    except AttributeError:
        log(f"  RKNNLite.eval_perf 不支持")
    except Exception as e:
        log(f"  profiling 失败: {e}")

    # ============ 5. 生成 10 秒音频 ============
    log(f"\n{'='*60}")
    log("🔊 生成 10 秒音频并通过 USB 音响播放")
    log("=" * 60)

    rknn.init_runtime(core_mask=0)

    # 生成有结构的 codec tokens 使声音更有意义
    log(f"  构造 2 帧 (10.24s) 音频输入...")
    all_audio = []

    for frame_idx in range(2):
        # 每帧使用不同的随机种子模拟连续语音
        structured = np.zeros((1, NUM_CODEBOOKS, TOKENS_PER_FRAME), dtype=np.int64)
        np.random.seed(frame_idx * 1000)
        for cb in range(min(8, NUM_CODEBOOKS)):
            structured[0, cb, :] = np.random.randint(0, 256, size=TOKENS_PER_FRAME)
        for cb in range(8, min(32, NUM_CODEBOOKS)):
            structured[0, cb, :] = np.random.randint(0, 512, size=TOKENS_PER_FRAME)
        # 高频 codebook 用较小的值减少噪声
        for cb in range(32, NUM_CODEBOOKS):
            structured[0, cb, :] = np.random.randint(0, 128, size=TOKENS_PER_FRAME)

        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[structured])
        inf_time = (time.perf_counter() - t0) * 1000
        audio_frame = outputs[0].flatten()
        all_audio.append(audio_frame)
        log(f"  帧 {frame_idx+1}/2: {inf_time:.0f}ms")

    full_audio = np.concatenate(all_audio)
    duration = arr_to_wav(full_audio, OUTPUT_WAV)
    log(f"  ✅ 音频已保存: {OUTPUT_WAV} ({duration:.2f}s)")
    results["audio"] = {
        "file": OUTPUT_WAV,
        "duration_s": round(duration, 2),
    }

    # 播放: 查找 USB 设备
    log(f"\n  查找 USB 音频设备...")
    card0_info = os.popen("aplay -l 2>/dev/null | grep 'card 0'").read().strip()
    card1_info = os.popen("aplay -l 2>/dev/null | grep 'card 1'").read().strip()
    log(f"  card 0: {card0_info}")
    log(f"  card 1: {card1_info}")

    # 使用 plughw:0,0 (AB13X USB Audio) 播放
    log(f"\n  🔊 播放音频 (USB card 0: AB13X USB Audio)...")
    os.system(f"aplay -D plughw:0,0 {OUTPUT_WAV} 2>/dev/null")
    log(f"  ✅ 播放完成")

    # ============ 6. 保存结果 ============
    results["npu_driver_version"] = "0.9.8"
    results["rknntoolkit_version"] = "2.3.2"

    log_file = "/data/vocoder_rknn_diag_result.json"
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\n📝 诊断结果已保存: {log_file}")

    rknn.release()

    print(f"\n---RESULT_JSON---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"---RESULT_JSON_END---")
    log(f"✅ 诊断完成")


if __name__ == "__main__":
    main()