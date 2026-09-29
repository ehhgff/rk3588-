#!/usr/bin/env python3
"""LLM 延迟测试 v3 - 精确的时间分布测量"""
import socket
import json
import time

LLM_SOCK = "/tmp/qwen3_llm_streaming.sock"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

payload = json.dumps({
    "prompt": "感冒的症状是什么",
    "max_tokens": 60,
    "temperature": 0.7,
    "stream": True
}).encode()

log("LLM 精确时间分布测试")
log("=" * 56)

total_try = 3
for attempt in range(total_try):
    log(f"\n--- 尝试 {attempt+1}/{total_try} ---")
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30)

        t0 = time.time()
        s.connect(LLM_SOCK)
        t1 = time.time()
        log(f"  connect: {(t1-t0)*1000:.1f}ms")

        s.sendall(payload)
        t2 = time.time()
        log(f"  send:    {(t2-t1)*1000:.1f}ms")

        # 测量第一次 recv 返回的时间
        first_data_time = None
        done = False
        buf = b""
        t_recv_start = time.time()

        while not done:
            if first_data_time is None:
                chunk = s.recv(8192)
                first_data_time = time.time()
                log(f"  first_recv: {(first_data_time-t_recv_start)*1000:.1f}ms (大小={len(chunk)}B)")
            else:
                chunk = s.recv(8192)

            if not chunk:
                break

            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                if not line.strip():
                    continue
                try:
                    data = json.loads(line.decode())
                    if "sentence" in data:
                        log(f"  sentence: '{data['sentence'][:30]}'")
                    if "done" in data:
                        log(f"  done: True, response_len={len(data.get('response',''))}")
                        done = True
                except:
                    continue

        s.close()
        total_ms = (time.time() - t0) * 1000
        log(f"  总耗时: {total_ms:.0f}ms")
        log(f"  服务器推理(~估计): {total_ms - (first_data_time - t_recv_start)*1000:.0f}ms")
        time.sleep(2)

    except Exception as e:
        log(f"  错误: {e}")
        time.sleep(2)