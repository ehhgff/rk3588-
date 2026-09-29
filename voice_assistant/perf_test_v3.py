#!/usr/bin/env python3
"""完整性能测试 v3 - 全部协议已修正"""
import socket
import json
import time
import os
import sys
import subprocess
import struct
import numpy as np
import glob

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


# ─── 测试1: SER 模型加载 + NPU 三核初始化 ───
def test_ser_model_load():
    log("\n═══ 测试1: SER 模型加载 + NPU 三核初始化 ═══")
    os.system("pkill -9 -f sensevoice_server 2>/dev/null")
    time.sleep(1.0)

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
dummy = [[0.0]*560 for _ in range(100)]
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

    time.sleep(0.2)
    try:
        load = subprocess.check_output("cat /sys/kernel/debug/rknpu/load 2>/dev/null", shell=True, timeout=2).decode().strip()
        log(f"  推理后 NPU load: {load}")
    except:
        pass
    return {"total_ms": round(total, 0)}


# ─── 测试2: TTS 合成延迟（10ms 轮询） ───
def test_tts_latency():
    log("\n═══ 测试2: TTS 合成延迟（10ms 轮询） ═══")
    if not os.path.exists(TTS_PIPE):
        log("  TTS pipe 不存在，跳过")
        return {"error": "TTS pipe 不存在"}

    existing = set(glob.glob("/tmp/tts_out_*.wav"))
    log("  正在预热 TTS...")
    with open(TTS_PIPE, "w") as pipe:
        pipe.write("您好\n")
        pipe.flush()
    time.sleep(1.5)
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
            "text": txt, "total_ms": round(total_ms, 1),
            "pipe_us": round(pipe_us, 0), "poll_ms": round(poll_ms, 1), "polls": polls
        })
        log(f"  '{txt}': {total_ms:.1f}ms (pipe {pipe_us:.0f}us, 轮询{poll_ms:.1f}ms/{polls}次)")
        time.sleep(0.3)

    avg = sum(r["total_ms"] for r in results) / len(results)
    avg_poll = sum(r["poll_ms"] for r in results) / len(results)
    log(f"  ─── 平均合成延迟: {avg:.1f}ms, 平均轮询: {avg_poll:.1f}ms")
    return {"avg_ms": round(avg, 1), "avg_poll_ms": round(avg_poll, 1), "samples": results}


# ─── 测试3: NPU 三核推理负载 ───
def test_npu_parallel():
    log("\n═══ 测试3: NPU 三核并行推理负载 ═══")
    test_code = """
import time
import numpy as np
from rknnlite.api import RKNNLite
rknn = RKNNLite()
ret = rknn.load_rknn('/data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn')
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
dummy = np.array([[[0.0]*560 for _ in range(100)]], dtype=np.float32)
for i in range(10):
    t0 = time.time()
    out = rknn.inference(inputs=[dummy])
    t = (time.time() - t0) * 1000
    print(f'INFER_{i}: {t:.1f}ms', flush=True)
    time.sleep(0.05)
print('DONE', flush=True)
"""
    proc = subprocess.Popen(["python3", "-c", test_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

    if latencies:
        avg = sum(latencies) / len(latencies)
        log(f"  ─── 平均推理延迟: {avg:.1f}ms, min={min(latencies):.1f}ms, max={max(latencies):.1f}ms")

    time.sleep(0.3)
    try:
        load = subprocess.check_output("cat /sys/kernel/debug/rknpu/load 2>/dev/null", shell=True, timeout=2).decode().strip()
        log(f"  NPU 负载: {load}")
    except:
        pass
    return {"latencies_ms": latencies, "avg_ms": round(sum(latencies)/len(latencies), 1) if latencies else 0}


# ─── 测试4: LLM 端到端延迟（正确协议：末尾加 \\n） ───
def test_llm_latency():
    log("\n═══ 测试4: LLM 端到端延迟（修正协议） ═══")
    if not os.path.exists(LLM_SOCK):
        log(f"  LLM socket 不存在，跳过")
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
        }).encode() + b'\n'

        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(30)
            t0 = time.time()
            s.connect(LLM_SOCK)
            s.sendall(payload)

            done = False
            buf = b""
            response = ""
            first_token_time = None

            while not done:
                chunk = s.recv(8192)
                if not chunk:
                    break
                if first_token_time is None:
                    first_token_time = time.time()
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.decode())
                        if "done" in data and data.get("done"):
                            response = data.get("response", "")
                            done = True
                    except:
                        continue

            s.close()
            total_ms = (time.time() - t0) * 1000
            ttft_ms = (first_token_time - t0) * 1000 if first_token_time else total_ms

            log(f"  '{q[:15]}': TTFT={ttft_ms:.0f}ms Total={total_ms:.0f}ms Resp={len(response)}chars")
            latencies.append({
                "question": q,
                "ttft_ms": round(ttft_ms, 0),
                "total_ms": round(total_ms, 0),
                "resp_len": len(response)
            })

        except Exception as e:
            log(f"  '{q[:15]}' 错误: {e}")

        time.sleep(1)

    if latencies:
        avg_ttft = sum(l["ttft_ms"] for l in latencies) / len(latencies)
        avg_total = sum(l["total_ms"] for l in latencies) / len(latencies)
        log(f"  ─── 平均 TTFT: {avg_ttft:.0f}ms, 平均总延迟: {avg_total:.0f}ms")
        return {
            "avg_ttft_ms": round(avg_ttft, 0),
            "avg_total_ms": round(avg_total, 0),
            "samples": latencies
        }
    return {"error": "无有效数据"}


# ─── 测试5: 系统资源状态 ───
def test_system():
    log("\n═══ 测试5: 系统资源状态 ═══")
    r = {}
    # CPU
    try:
        freqs = []
        for i in range(8):
            f = subprocess.check_output(f"cat /sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq 2>/dev/null", shell=True, timeout=2).decode().strip()
            freqs.append(int(f)//1000)
        r["cpu_freqs_mhz"] = [f//1000 for f in freqs]
        log(f"  CPU 频率 (MHz): {[f//1000 for f in freqs]}")
    except: pass
    # NPU freq
    try:
        freq = subprocess.check_output("cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null", shell=True, timeout=2).decode().strip()
        r["npu_freq_mhz"] = int(freq)//1000000
        log(f"  NPU 频率: {int(freq)//1000000}MHz")
    except: pass
    # Memory
    try:
        mem = subprocess.check_output("free -m | grep Mem", shell=True, timeout=2).decode().strip().split()
        r["mem_total_mb"] = int(mem[1])
        r["mem_used_mb"] = int(mem[2])
        r["mem_avail_mb"] = int(mem[6])
        log(f"  内存: 已用 {mem[2]}MB / {mem[1]}MB (可用 {mem[6]}MB)")
    except: pass
    # Temp
    try:
        for zone in ["soc-thermal", "npu-thermal"]:
            zpath = f"/sys/class/thermal/thermal_zone*/temp"
            out = subprocess.check_output(f"for f in {zpath}; do name=$(cat $(dirname $f)/type 2>/dev/null); if [ \"$name\" = \"{zone}\" ]; then cat $f; fi; done", shell=True, timeout=2).decode().strip()
            if out:
                r[f"{zone}_c"] = round(int(out)/1000, 1)
        log(f"  温度: SOC={r.get('soc-thermal_c','?')}°C, NPU={r.get('npu-thermal_c','?')}°C")
    except: pass
    return r


def main():
    log("=" * 56)
    log("性能测试 v3 - 修正协议")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 56)

    results = {}
    results["system"] = test_system()
    results["ser"] = test_ser_model_load()
    results["tts"] = test_tts_latency()
    results["npu_parallel"] = test_npu_parallel()
    results["llm"] = test_llm_latency()

    log("\n" + "=" * 56)
    log("最终结果汇总")
    log("=" * 56)
    log(f"  系统: {results['system']}")
    log(f"  SER加载: {results['ser'].get('total_ms','N/A')}ms")
    if 'tts' in results and 'avg_ms' in results['tts']:
        tts = results['tts']
        log(f"  TTS合成: avg={tts['avg_ms']}ms (轮询10ms, 检测={tts['avg_poll_ms']}ms)")
    if 'npu_parallel' in results and 'avg_ms' in results['npu_parallel']:
        npu = results['npu_parallel']
        log(f"  NPU推理: avg={npu['avg_ms']}ms, min={min(npu['latencies_ms']):.1f}ms")
    if 'llm' in results and 'avg_total_ms' in results['llm']:
        llm = results['llm']
        log(f"  LLM延迟: TTFT={llm['avg_ttft_ms']}ms, Total={llm['avg_total_ms']}ms")

    import json as _json
    with open("/tmp/perf_v3_results.json", "w") as f:
        _json.dump(results, f, indent=2, ensure_ascii=False)
    log("\n结果已保存到 /tmp/perf_v3_results.json")
    return results

if __name__ == "__main__":
    main()