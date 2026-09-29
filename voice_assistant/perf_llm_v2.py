#!/usr/bin/env python3
"""LLM 延迟测试 v2 - 修复 socket 读取逻辑"""
import socket
import json
import time
import select

LLM_SOCK = "/tmp/qwen3_llm_streaming.sock"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

questions = [
    "感冒的症状是什么",
    "发烧了怎么办",
    "高血压需要注意什么",
]

log("LLM 延迟测试 v2（修正协议）")
log("=" * 56)

for q in questions:
    # 测试流式模式
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
        
        ts_start = time.time()
        s.sendall(payload)
        t_sent = time.time()
        
        response = ""
        first_token_time = None
        done = False
        buf = b""
        
        while not done:
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
                    if "sentence" in data and first_token_time is None:
                        first_token_time = time.time()
                    if "done" in data and data.get("done"):
                        response = data.get("response", "")
                        done = True
                except:
                    continue
        
        s.close()
        total_ms = (time.time() - ts_start) * 1000
        ttft_ms = (first_token_time - ts_start) * 1000 if first_token_time else total_ms
        
        log(f"  ✅ '{q[:20]}...'")
        log(f"     TTFT={ttft_ms:.0f}ms  Total={total_ms:.0f}ms  Resp={len(response)}chars")
        
    except Exception as e:
        log(f"  ❌ '{q[:20]}' 错误: {e}")
    
    time.sleep(1)

log("\n--- 非流式模式对比测试 ---")

for q in questions[:1]:
    payload = json.dumps({
        "prompt": q,
        "max_tokens": 60,
        "temperature": 0.7,
        "stream": False
    }).encode()

    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30)
        s.connect(LLM_SOCK)
        
        ts_start = time.time()
        s.sendall(payload)
        
        # 读取所有数据
        buf = b""
        while True:
            chunk = s.recv(8192)
            if not chunk:
                break
            buf += chunk
        
        s.close()
        total_ms = (time.time() - ts_start) * 1000
        
        try:
            resp = json.loads(buf.decode())
            log(f"  非流式 '{q[:20]}': {total_ms:.0f}ms, resp={len(resp.get('response',''))}chars")
        except:
            log(f"  非流式 '{q[:20]}': {total_ms:.0f}ms, raw={buf[:100]}")
        
    except Exception as e:
        log(f"  ❌ '{q[:20]}' 错误: {e}")