import socket, json, time

BENCH_SOCK = "/tmp/qwen3_llm.sock"

def test_llm(prompt, max_tokens=30, stream=False):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(60)
    sock.connect(BENCH_SOCK)
    req = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7, "stream": stream}) + "\n"
    start = time.time()
    sock.sendall(req.encode())
    
    if stream:
        ttft = None
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            text = data.decode("utf-8", errors="replace").strip()
            if "\n" in text:
                lines = text.split("\n")
                for line in lines:
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "sentence" in obj and ttft is None:
                            ttft = (time.time() - start) * 1000
                        if obj.get("done"):
                            total = (time.time() - start) * 1000
                            sock.close()
                            return ttft or total, total, obj.get("response", "")
                    except:
                        pass
        total = (time.time() - start) * 1000
        sock.close()
        return ttft or total, total, ""
    else:
        data = sock.recv(65536)
        total = (time.time() - start) * 1000
        sock.close()
        try:
            obj = json.loads(data.decode())
            return total, total, obj.get("response", "")
        except:
            return total, total, data.decode()

# ==== A/B 测试 ====
import os
affinity_mask = os.popen("taskset -p %d" % os.getpid()).read().strip()
print("=" * 65)
print("  LLM A/B 测试 - 当前 taskset: %s" % affinity_mask)
print("=" * 65)

PROMPTS = [
    ("简单介绍一下感冒的症状", 20),
    ("高血压患者需要注意什么", 20),
]

# 先测试 stream=True (当前 LLM 服务模式)
for label, stream_mode in [("stream=False (非流式)", False), ("stream=True (流式)", True)]:
    print("\n--- %s ---" % label)
    all_ttft, all_total = [], []
    for prompt, max_tok in PROMPTS:
        for r in range(2):
            time.sleep(0.5)
            try:
                ttft, total, text = test_llm(prompt, max_tok, stream=stream_mode)
                all_ttft.append(ttft)
                all_total.append(total)
                print("  [%d/2] TTFT=%6.0fms total=%6.0fms | %s..." % (r+1, ttft, total, text[:25] if text else "(empty)"))
            except Exception as e:
                print("  [%d/2] ERROR: %s" % (r+1, e))
    
    if all_ttft:
        avg_ttft = sum(all_ttft)/len(all_ttft)
        avg_total = sum(all_total)/len(all_total)
        print("  >>> 平均: TTFT=%.0fms, Total=%.0fms" % (avg_ttft, avg_total))

# 保存结果
import json as _json
result = {
    "affinity": affinity_mask,
    "ttft": all_ttft,
    "total": all_total,
}
with open("/tmp/llm_bench_result.json", "w") as f:
    _json.dump(result, f)
print("\n结果已保存")