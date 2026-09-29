#!/usr/bin/env python3
"""
性能测试脚本 v2 - 验证本次优化效果
修复协议: LLM 用 prompt, SER 用二进制协议
"""
import socket
import json
import time
import os
import sys
import subprocess
import struct
import numpy as np

SER_SOCK = "/tmp/sensevoice_server.sock"
LLM_SOCK = "/tmp/qwen3_llm_streaming.sock"
TTS_PIPE = "/tmp/tts_pipe"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

def wait_for_socket(sock_path, max_wait=60):
    for i in range(max_wait):
        if os.path.exists(sock_path):
            return True
        time.sleep(1)
    return False

# ─────────────────────────────────────────
# 测试1: SER 模型加载 + NPU 初始化时间
# ─────────────────────────────────────────
def test_ser_model_load():
    log("\n═══ 测试1: SER 模型加载 + NPU 三核初始化 ═══")
    os.system("pkill -9 -f sensevoice_server 2>/dev/null")
    time.sleep(1.0)

    # 使用 time 命令精确测量
    import subprocess as _sp
    ts = time.time()
    proc = _sp.Popen(
        ["python3", "-c", """
from rknnlite.api import RKNNLite
import time
t0 = time.time()
rknn = RKNNLite()
ret = rknn.load_rknn('/data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn')
print(f'LOAD_OK:{ret}', flush=True)
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
t = (time.time() - t0) * 1000
print(f'INIT_OK:{ret}:{t:.0f}ms', flush=True)
# 快速推理一次验证
dummy = [[0.0]*560 for _ in range(100)]
import numpy as np
feats = np.array([dummy], dtype=np.float32)
t1 = time.time()
out = rknn.inference(inputs=[feats])
t2 = (time.time() - t1) * 1000
print(f'INFER_OK:{t2:.1f}ms', flush=True)
"""],
        stdout=_sp.PIPE, stderr=_sp.PIPE
    )
    proc.wait(timeout=60)
    stdout, stderr = proc.communicate()
    total = (time.time() - ts) * 1000
    log(f"  SER 总耗时: {total:.0f}ms")
    for line in stdout.decode().split('\n'):
        if line.strip():
            log(f"  {line}")

    # NPU 负载检查
    time.sleep(0.2)
    try:
        load = subprocess.check_output("cat /sys/kernel/debug/rknpu/load 2>/dev/null", shell=True, timeout=2).decode().strip()
        log(f"  推理后 NPU load: {load}")
    except:
        pass

    return {"total_ms": round(total, 0)}

# ─────────────────────────────────────────
# 测试2: TTS 合成延迟（10ms 轮询 vs 50ms 理论对比）
# ─────────────────────────────────────────
def test_tts_latency():
    log("\n═══ 测试2: TTS 合成延迟（10ms 轮询） ═══")
    if not os.path.exists(TTS_PIPE):
        log("  TTS pipe 不存在，跳过")
        return {"error": "TTS pipe 不存在"}

    import glob

    # 先预热一下
    existing = set(glob.glob("/tmp/tts_out_*.wav"))
    with open(TTS_PIPE, "w") as pipe:
        pipe.write("您好\n")
        pipe.flush()
    time.sleep(1)
    for f in set(glob.glob("/tmp/tts_out_*.wav")) - existing:
        try: os.remove(f)
        except: pass

    results = []
    short_texts = ["好的", "您好", "是的", "请说", "明白"]

    for txt in short_texts:
        existing = set(glob.glob("/tmp/tts_out_*.wav"))
        ts = time.time()

        with open(TTS_PIPE, "w") as pipe:
            pipe.write(txt + "\n")
            pipe.flush()

        pipe_us = (time.time() - ts) * 1_000_000

        # 10ms 轮询检测
        poll_start = time.time()
        found = None
        polls = 0
        for _ in range(500):
            time.sleep(0.01)
            polls += 1
            for f in glob.glob("/tmp/tts_out_*.wav"):
                if f not in existing:
                    found = f
                    break
            if found:
                break

        poll_ms = (time.time() - poll_start) * 1000
        total_ms = (time.time() - ts) * 1000

        if found:
            try: os.remove(found)
            except: pass

        results.append({
            "text": txt,
            "total_ms": round(total_ms, 1),
            "pipe_us": round(pipe_us, 0),
            "poll_ms": round(poll_ms, 1),
            "polls": polls,
        })
        log(f"  '{txt}': 总计 {total_ms:.1f}ms (pipe {pipe_us:.0f}us, 轮询 {poll_ms:.1f}ms/{polls}次)")
        time.sleep(0.3)

    avg = sum(r["total_ms"] for r in results) / len(results)
    avg_poll = sum(r["poll_ms"] for r in results) / len(results)
    log(f"  ─── 平均合成延迟: {avg:.1f}ms, 平均轮询检测: {avg_poll:.1f}ms")

    return {"avg_ms": round(avg, 1), "avg_poll_ms": round(avg_poll, 1), "samples": results}


# ─────────────────────────────────────────
# 测试3: NPU 三核负载（串行推理测试）
# ─────────────────────────────────────────
def test_npu_parallel():
    log("\n═══ 测试3: NPU 三核并行推理负载 ═══")

    # 直接运行 NPU 三核推理
    test_code = """
import time
import numpy as np
from rknnlite.api import RKNNLite

rknn = RKNNLite()
ret = rknn.load_rknn('/data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn')
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

# 批量推理给 NPU 加压
dummy = np.array([[[0.0]*560 for _ in range(100)]], dtype=np.float32)

# 连续推理 10 次
for i in range(10):
    t0 = time.time()
    out = rknn.inference(inputs=[dummy])
    t = (time.time() - t0) * 1000
    print(f'INFER_{i}: {t:.1f}ms', flush=True)
    time.sleep(0.05)

print('DONE', flush=True)
"""
    proc = subprocess.Popen(
        ["python3", "-c", test_code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    proc.wait(timeout=60)
    stdout, stderr = proc.communicate()

    latencies = []
    for line in stdout.decode().split('\n'):
        line = line.strip()
        if line.startswith('INFER_'):
            parts = line.split(':')
            ms = float(parts[1].replace('ms',''))
            latencies.append(ms)
            log(f"  推理 #{len(latencies)}: {ms:.1f}ms")
        elif line:
            log(f"  {line}")

    if latencies:
        log(f"  ─── 平均推理延迟: {sum(latencies)/len(latencies):.1f}ms")

    # NPU 负载
    time.sleep(0.3)
    try:
        load = subprocess.check_output("cat /sys/kernel/debug/rknpu/load 2>/dev/null", shell=True, timeout=2).decode().strip()
        log(f"  NPU 推理负载: {load}")
    except:
        pass

    return {"latencies_ms": latencies}


# ─────────────────────────────────────────
# 测试4: LLM 端到端延迟（正确协议）
# ─────────────────────────────────────────
def test_llm_latency():
    log("\n═══ 测试4: LLM 端到端延迟 ═══")
    if not os.path.exists(LLM_SOCK):
        log(f"  LLM socket {LLM_SOCK} 不存在，尝试其他路径...")
        for p in ["/tmp/qwen3_llm.sock", "/tmp/voice_assistant.sock"]:
            if os.path.exists(p):
                LLM_SOCK_GLOBAL = p
                log(f"  使用 {p}")
                break
        else:
            log("  跳过 LLM 测试")
            return {"error": "LLM socket 不存在"}

    questions = [
        "感冒的症状是什么",
        "发烧了怎么办", 
        "高血压需要注意什么",
    ]

    latencies = []
    for q in questions:
        payload = json.dumps({
            "prompt": q,
            "max_tokens": 60,
            "temperature": 0.7,
            "stream": True
        }).encode()

        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(30)
            s.connect(LLM_SOCK)
            s.sendall(payload)

            # 读取流式响应
            response = ""
            first_token_time = None
            buf = b""
            ts_start = time.time()

            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
                # 处理可能的换行分隔 JSON
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.decode())
                        if first_token_time is None and "sentence" in data:
                            first_token_time = time.time()
                        if "done" in data and data["done"]:
                            response = data.get("response", "")
                            break
                    except:
                        continue
                if "done" in globals() and data.get("done"):
                    break

            s.close()
            total_ms = (time.time() - ts_start) * 1000

            if first_token_time:
                first_token_ms = (first_token_time - ts_start) * 1000
            else:
                first_token_ms = total_ms

            log(f"  '{q}': 首token {first_token_ms:.0f}ms, 总计 {total_ms:.0f}ms (resp: {len(response)} chars)")
            latencies.append({
                "question": q,
                "ttft_ms": round(first_token_ms, 0),
                "total_ms": round(total_ms, 0),
                "resp_len": len(response)
            })

        except Exception as e:
            log(f"  '{q}' 错误: {e}")

        time.sleep(0.5)

    if latencies:
        avg = sum(l["total_ms"] for l in latencies) / len(latencies)
        log(f"  ─── LLM 平均延迟: {avg:.0f}ms")
        return {"avg_ms": round(avg, 0), "samples": latencies}
    return {"error": "无有效数据"}


# ─────────────────────────────────────────
# 测试5: 系统资源监控
# ─────────────────────────────────────────
def test_system_resources():
    log("\n═══ 测试5: 系统资源状态 ═══")
    import subprocess as _sp

    resources = {}

    # CPU 频率
    try:
        freqs = []
        for i in range(8):
            f = _sp.check_output(f"cat /sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq 2>/dev/null", shell=True, timeout=2).decode().strip()
            freqs.append(int(f) // 1000)
        resources["cpu_freqs_khz"] = freqs
        log(f"  CPU 频率 (MHz): {[f//1000 for f in freqs]}")
    except:
        pass

    # 内存
    try:
        mem = _sp.check_output("free -m | grep Mem", shell=True, timeout=2).decode().strip().split()
        resources["mem_total_mb"] = int(mem[1])
        resources["mem_used_mb"] = int(mem[2])
        resources["mem_avail_mb"] = int(mem[6])
        log(f"  内存: 已用 {mem[2]}MB / 总计 {mem[1]}MB (可用 {mem[6]}MB)")
    except:
        pass

    # Swap
    try:
        swap = _sp.check_output("free -m | grep Swap", shell=True, timeout=2).decode().strip().split()
        resources["swap_total_mb"] = int(swap[1])
        resources["swap_used_mb"] = int(swap[2])
        log(f"  Swap: {swap[2]}MB / {swap[1]}MB")
    except:
        pass

    # NPU
    try:
        freq = _sp.check_output("cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null", shell=True, timeout=2).decode().strip()
        resources["npu_freq_hz"] = int(freq)
        log(f"  NPU 频率: {int(freq)//1000000}MHz")
    except:
        pass

    # 温度
    try:
        temps = {}
        for zone in ["soc-thermal", "bigcore0-thermal", "bigcore1-thermal", "npu-thermal"]:
            zpath = f"/sys/class/thermal/thermal_zone*/temp"
            out = _sp.check_output(f"for f in {zpath}; do name=$(cat $(dirname $f)/type 2>/dev/null); if [ \"$name\" = \"{zone}\" ]; then cat $f; fi; done", shell=True, timeout=2).decode().strip()
            if out:
                temps[zone] = round(int(out) / 1000, 1)
        resources["temps_c"] = temps
        log(f"  温度: {temps}")
    except:
        pass

    # 负载
    try:
        load = open("/proc/loadavg").read().strip().split()[:3]
        resources["load_avg"] = [float(x) for x in load]
        log(f"  系统负载: {load[0]} / {load[1]} / {load[2]}")
    except:
        pass

    return resources


# ─────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────
def main():
    log("=" * 56)
    log("性能测试 v2 - 优化效果验证")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 56)

    results = {}

    # 测试0: 系统基线
    results["system"] = test_system_resources()

    # 测试1: SER 模型加载时间
    results["ser"] = test_ser_model_load()

    # 测试2: TTS 合成延迟
    results["tts"] = test_tts_latency()

    # 测试3: NPU 三核并行
    results["npu_parallel"] = test_npu_parallel()

    # 测试5: LLM 延迟
    results["llm"] = test_llm_latency()

    # 输出汇总
    log("\n" + "=" * 56)
    log("测试结果汇总")
    log("=" * 56)

    if "ser" in results:
        log(f"  SER 模型加载+初始化: {results['ser'].get('total_ms', 'N/A')}ms")

    if "tts" in results and "avg_ms" in results["tts"]:
        log(f"  TTS 平均合成延迟: {results['tts']['avg_ms']}ms (10ms 轮询)")
        log(f"  TTS 平均轮询检测: {results['tts']['avg_poll_ms']}ms")

    if "npu_parallel" in results and "latencies_ms" in results["npu_parallel"]:
        lats = results["npu_parallel"]["latencies_ms"]
        log(f"  NPU 三核推理延迟: avg={sum(lats)/len(lats):.1f}ms, min={min(lats):.1f}ms, max={max(lats):.1f}ms")

    if "llm" in results and "avg_ms" in results["llm"]:
        log(f"  LLM 平均端到端延迟: {results['llm']['avg_ms']}ms")

    log("\n测试完成")
    return results


if __name__ == "__main__":
    results = main()
    import json as _json
    with open("/tmp/perf_v2_results.json", "w") as f:
        _json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"结果已保存到 /tmp/perf_v2_results.json")