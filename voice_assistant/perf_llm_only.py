#!/usr/bin/env python3
"""LLM 延迟测试 - 新会话，无上下文累积"""
import socket
import json
import time

LLM_SOCK = "/tmp/qwen3_llm_streaming.sock"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

questions = [
    "感冒的症状是什么",
    "发烧了怎么办",
    "高血压需要注意什么",
]

log("LLM 延迟测试（新会话）")
log("=" * 50)

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

        response = ""
        first_token_time = None
        buf = b""
        ts_start = time.time()

        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
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

        s.close()
        total_ms = (time.time() - ts_start) * 1000

        if first_token_time:
            first_token_ms = (first_token_time - ts_start) * 1000
        else:
            first_token_ms = total_ms

        log(f"  ✅ '{q}'")
        log(f"     TTFT: {first_token_ms:.0f}ms | 总计: {total_ms:.0f}ms | 响应: {len(response)} chars")

    except Exception as e:
        log(f"  ❌ '{q}' 错误: {e}")

    time.sleep(1)