import socket, json, time, os

PROMPTS = [
    "简单介绍一下感冒的症状",
    "高血压患者需要注意什么",
    "发烧了应该怎么办",
    "糖尿病的早期症状有哪些",
]

BENCH_SOCK = "/tmp/qwen3_llm.sock"

def test_llm(prompt, max_tokens=30):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(60)
    sock.connect(BENCH_SOCK)
    req = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7, "stream": True}) + "\n"
    start = time.time()
    sock.sendall(req.encode())
    ttft = None
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        lines = data.decode("utf-8", errors="replace").strip().split("\n")
        for line in lines:
            try:
                obj = json.loads(line)
                if "sentence" in obj and ttft is None:
                    ttft = (time.time() - start) * 1000
                if obj.get("done"):
                    break
            except:
                pass
        else:
            continue
        break
    total = (time.time() - start) * 1000
    sock.close()
    return ttft or total, total

print("=" * 60)
print("  A: LLM on BIG cores (CPU6-7) - taskset=c0")
print("=" * 60)
all_ttft, all_total = [], []
for p in PROMPTS:
    for r in range(3):
        time.sleep(1)
        try:
            ttft, total = test_llm(p)
            all_ttft.append(ttft)
            all_total.append(total)
            print(f"  [{r+1}/3] TTFT={ttft:6.0f}ms  total={total:6.0f}ms  | {p[:15]}...")
        except Exception as e:
            print(f"  [{r+1}/3] ERROR: {e}")

if all_ttft:
    avg_ttft = sum(all_ttft)/len(all_ttft)
    avg_total = sum(all_total)/len(all_total)
    print(f"\n>>> 大核结果: TTFT avg={avg_ttft:.0f}ms, Total avg={avg_total:.0f}ms")

# 保存结果
save = {"label": "BIG_CORE_67", "ttft": all_ttft, "total": all_total}
with open("/tmp/llm_bench_big.json", "w") as f:
    json.dump(save, f)
print("\n结果已保存到 /tmp/llm_bench_big.json")