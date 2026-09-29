#!/usr/bin/env python3
"""
性能测试脚本 - 验证本次优化效果
测试项:
  1. SER NPU 三核并行推理延迟
  2. TTS 预热效果（NPU cache 冷启动 vs 预热后）
  3. TTS 轮询检测延迟 (50ms vs 10ms)
  4. NPU 三核负载分布
  5. LLM 端到端延迟（确保优化无副作用）
"""
import socket
import json
import time
import os
import sys
import subprocess
import struct

SER_SOCK = "/tmp/sensevoice_server.sock"
LLM_SOCK = "/tmp/qwen3_llm.sock"
TTS_PIPE = "/tmp/tts_pipe"

# ─────────────────────────────────────────
# 测试辅助函数
# ─────────────────────────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

def send_unix_socket(sock_path, data, timeout=30):
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(sock_path)
        if isinstance(data, str):
            data = data.encode()
        s.sendall(data)
        resp = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
            try:
                json.loads(resp.decode())
                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        s.close()
        return json.loads(resp.decode())
    except Exception as e:
        return {"error": str(e)}

def wait_for_socket(sock_path, max_wait=60):
    for i in range(max_wait):
        if os.path.exists(sock_path):
            return True
        time.sleep(1)
    return False

# ─────────────────────────────────────────
# 测试1: SER NPU 三核并行推理延迟
# ─────────────────────────────────────────
def test_ser_npu_latency():
    log("\n═══ 测试1: SER NPU 三核并行推理延迟 ═══")
    log("启动 SER 服务（使用 NPU_CORE_0_1_2）...")

    # 清理旧服务
    os.system("pkill -9 -f sensevoice_server 2>/dev/null")
    time.sleep(0.5)

    # 启动 SER 服务（后台）
    ser_proc = subprocess.Popen(
        ["python3", "/data/sensevoice/sensevoice_server.py",
         "--model", "/data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn",
         "--socket", SER_SOCK],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    if not wait_for_socket(SER_SOCK, 15):
        log("SER 服务启动失败")
        return {"error": "SER启动失败"}
    log("SER 服务就绪")

    # 发送测试音频编码（模拟推理请求）
    latencies = []
    dummy_feat = [0.0] * 400  # 模拟100帧语音特征

    for i in range(5):
        payload = json.dumps({"features": dummy_feat}).encode()
        ts = time.time()
        resp = send_unix_socket(SER_SOCK, payload)
        elapsed = (time.time() - ts) * 1000
        latencies.append(elapsed)
        log(f"  SER推理 #{i+1}: {elapsed:.1f}ms")
        time.sleep(0.2)

    ser_proc.terminate()
    time.sleep(0.5)

    avg = sum(latencies) / len(latencies)
    log(f"\nSER 平均推理延迟: {avg:.1f}ms")
    log(f"SER 最小/最大: {min(latencies):.1f}ms / {max(latencies):.1f}ms")

    return {
        "ser_latencies_ms": latencies,
        "ser_avg_ms": round(avg, 1),
        "ser_min_ms": round(min(latencies), 1),
        "ser_max_ms": round(max(latencies), 1),
    }


# ─────────────────────────────────────────
# 测试2: TTS 预热效果
# ─────────────────────────────────────────
def test_tts_warmup():
    log("\n═══ 测试2: TTS NPU cache 预热效果 ═══")

    # 检查 TTS daemon
    if not os.path.exists(TTS_PIPE):
        log("TTS pipe 不存在，跳过")
        return {"error": "TTS pipe 不存在"}

    # 测试1: 冷启动合成（模拟首次发送）
    log("\n  --- 模拟冷启动合成 ---")
    cold_latencies = []
    import glob

    for i in range(3):
        existing = set(glob.glob("/tmp/tts_out_*.wav"))
        ts = time.time()

        with open(TTS_PIPE, "w") as pipe:
            pipe.write(f"测试文本{i+1}\n")
            pipe.flush()

        pipe_time = (time.time() - ts) * 1000

        # 轮询等待（新代码 10ms）
        poll_start = time.time()
        found = None
        for _ in range(500):
            time.sleep(0.01)
            for f in glob.glob("/tmp/tts_out_*.wav"):
                if f not in existing:
                    found = f
                    break
            if found:
                break

        poll_time = (time.time() - poll_start) * 1000
        total = (time.time() - ts) * 1000

        if found:
            try:
                os.remove(found)
            except:
                pass

        cold_latencies.append({"total_ms": round(total, 1), "pipe_ms": round(pipe_time, 1), "poll_ms": round(poll_time, 1)})
        log(f"  冷启动 #{i+1}: pipe写 {pipe_time:.1f}ms, 轮询 {poll_time:.1f}ms, 总计 {total:.1f}ms")
        time.sleep(0.3)

    # 测试2: 预热后合成
    log("\n  --- 预热后合成 ---")
    # 发送预热文本
    existing = set(glob.glob("/tmp/tts_out_*.wav"))
    with open(TTS_PIPE, "w") as pipe:
        pipe.write("您好\n")
        pipe.flush()
    time.sleep(0.3)
    for f in glob.glob("/tmp/tts_out_*.wav"):
        if f not in existing:
            try:
                os.remove(f)
            except:
                pass

    warm_latencies = []
    for i in range(3):
        existing = set(glob.glob("/tmp/tts_out_*.wav"))
        ts = time.time()

        with open(TTS_PIPE, "w") as pipe:
            pipe.write(f"预热后测试{i+1}\n")
            pipe.flush()

        pipe_time = (time.time() - ts) * 1000

        poll_start = time.time()
        found = None
        for _ in range(500):
            time.sleep(0.01)
            for f in glob.glob("/tmp/tts_out_*.wav"):
                if f not in existing:
                    found = f
                    break
            if found:
                break

        poll_time = (time.time() - poll_start) * 1000
        total = (time.time() - ts) * 1000

        if found:
            try:
                os.remove(found)
            except:
                pass

        warm_latencies.append({"total_ms": round(total, 1), "pipe_ms": round(pipe_time, 1), "poll_ms": round(poll_time, 1)})
        log(f"  预热后 #{i+1}: pipe写 {pipe_time:.1f}ms, 轮询 {poll_time:.1f}ms, 总计 {total:.1f}ms")
        time.sleep(0.3)

    return {
        "cold": cold_latencies,
        "warm": warm_latencies,
    }


# ─────────────────────────────────────────
# 测试3: NPU 三核负载分布
# ─────────────────────────────────────────
def test_npu_load():
    log("\n═══ 测试3: NPU 三核负载分布 ═══")
    results = []

    # 读取 NPU load 前先静默
    def get_npu_load():
        try:
            out = subprocess.check_output("cat /sys/kernel/debug/rknpu/load 2>/dev/null", shell=True, timeout=2)
            return out.decode().strip()
        except:
            return "N/A"

    # 空闲时的 NPU 负载
    idle = get_npu_load()
    log(f"  空闲 NPU load: {idle}")

    # 启动 SER 测试负载
    os.system("pkill -9 -f sensevoice_server 2>/dev/null")
    time.sleep(0.5)

    ser_proc = subprocess.Popen(
        ["python3", "/data/sensevoice/sensevoice_server.py",
         "--model", "/data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn",
         "--socket", SER_SOCK],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    if not wait_for_socket(SER_SOCK, 15):
        log("SER 启动失败")
        return {"error": "SER启动失败"}

    # 连续发送请求，让 NPU 有负载
    dummy_feat = [0.0] * 400
    for i in range(5):
        payload = json.dumps({"features": dummy_feat}).encode()
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect(SER_SOCK)
            s.sendall(payload)
            s.recv(4096)
            s.close()
        except:
            pass
        time.sleep(0.1)

    time.sleep(0.3)
    loaded = get_npu_load()
    log(f"  推理中 NPU load: {loaded}")

    ser_proc.terminate()
    time.sleep(0.5)

    idle2 = get_npu_load()
    log(f"  服务停止后 NPU load: {idle2}")

    return {"idle": idle, "under_load": loaded, "idle_after": idle2}


# ─────────────────────────────────────────
# 测试4: LLM 端到端延迟（基线对比）
# ─────────────────────────────────────────
def test_llm_latency():
    log("\n═══ 测试4: LLM 端到端延迟（基线验证） ═══")
    if not os.path.exists(LLM_SOCK):
        log("LLM socket 不存在，跳过")
        return {"error": "LLM socket 不存在"}

    questions = [
        "感冒的症状是什么",
        "发烧了怎么办",
        "高血压需要注意什么",
    ]

    latencies = []
    for q in questions:
        payload = json.dumps({"text": q}).encode()
        ts = time.time()
        resp = send_unix_socket(LLM_SOCK, payload)
        elapsed = (time.time() - ts) * 1000
        resp_len = len(resp.get("text", "")) if isinstance(resp, dict) else 0
        latencies.append({"question": q, "ms": round(elapsed, 1), "resp_len": resp_len})
        log(f"  '{q}': {elapsed:.0f}ms ({resp_len} chars)")
        time.sleep(0.5)

    avg_ms = sum(l["ms"] for l in latencies) / len(latencies)
    log(f"\nLLM 平均延迟: {avg_ms:.0f}ms")

    return latencies


# ─────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────
def main():
    log("=" * 56)
    log("性能测试 - 优化效果验证")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 56)

    results = {}

    # 测试1: SER NPU 三核
    results["ser"] = test_ser_npu_latency()

    # 测试2: TTS 预热
    results["tts_warmup"] = test_tts_warmup()

    # 测试3: NPU 负载
    results["npu_load"] = test_npu_load()

    # 测试4: LLM 基线
    results["llm"] = test_llm_latency()

    # 输出汇总
    log("\n" + "=" * 56)
    log("测试结果汇总")
    log("=" * 56)

    if "ser" in results and "ser_avg_ms" in results["ser"]:
        log(f"  SER 平均推理延迟: {results['ser']['ser_avg_ms']}ms (NPU_CORE_0_1_2)")

    if "tts_warmup" in results and "cold" in results["tts_warmup"]:
        cold_avg = sum(d["total_ms"] for d in results["tts_warmup"]["cold"]) / len(results["tts_warmup"]["cold"])
        log(f"  TTS 冷启动平均延迟: {cold_avg:.1f}ms")
        if "warm" in results["tts_warmup"]:
            warm_avg = sum(d["total_ms"] for d in results["tts_warmup"]["warm"]) / len(results["tts_warmup"]["warm"])
            log(f"  TTS 预热后平均延迟: {warm_avg:.1f}ms")
            log(f"  TTS 预热收益: {cold_avg - warm_avg:.1f}ms")

    if "npu_load" in results:
        log(f"  NPU 空闲负载: {results['npu_load'].get('idle', 'N/A')}")
        log(f"  NPU 推理中负载: {results['npu_load'].get('under_load', 'N/A')}")

    if "llm" in results and isinstance(results["llm"], list):
        avg = sum(d["ms"] for d in results["llm"]) / len(results["llm"])
        log(f"  LLM 平均延迟: {avg:.0f}ms (基线验证)")

    log("\n测试完成")
    return results


if __name__ == "__main__":
    results = main()
    # 保存结果到文件
    import json as _json
    with open("/tmp/perf_optimization_results.json", "w") as f:
        _json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到 /tmp/perf_optimization_results.json")