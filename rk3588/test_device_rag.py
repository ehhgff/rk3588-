#!/usr/bin/env python3
"""设备端 RAG 测试脚本"""
import socket, json, sys

SOCK = "/tmp/patient_rag.sock"

def test(query, section=None):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(3.0)
    sock.connect(SOCK)
    req = {"query": query, "threshold": 0.30}
    if section:
        req["section"] = section
    sock.sendall((json.dumps(req) + "\n").encode())
    resp = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        resp += chunk
        if b"\n" in chunk:
            break
    sock.close()
    return json.loads(resp)


passed = 0
failed = 0

def check(category, query, section, expected_substr):
    global passed, failed
    r = test(query, section)
    ans = r.get("answer", "")
    conf = r.get("confidence", 0)
    ok = expected_substr in ans if ans else False
    sym = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    ot = " [离题]" if r.get("is_off_topic") else ""
    print(f"  {sym:4s} | {category:<12s} | {query:<20s} | conf={conf:.2f}{ot} | {ans[:40]}")
    if not ok:
        print(f"      期望包含: {expected_substr}")


print("=" * 80)
print(" Patient RAG 测试报告")
print("=" * 80)

# ═══ 直接问 ═══
print("\n── 直接问 ──────────────────────────────────────────────")
check("直接问", "你叫什么名字？", "发热", "李华")
check("直接问", "发烧吗？", "发热", "发烧")
check("直接问", "您现在哪里不舒服？", "贫血", "头晕")
check("直接问", "血压高多久了？", None, "5年")

# ═══ 确认正确 ═══
print("\n── 确认正确 ────────────────────────────────────────────")
check("确认正确", "您是35岁吗？", "发热", "是的")
check("确认正确", "月经量多吗？", "贫血", "比较多")
check("确认正确", "有痰吗？", None, "痰液")

# 用 "咳嗽时有没有咳痰？" 的预期也试一下
check("确认正确", "咳嗽时有没有咳痰？", None, "痰液")

# ═══ 确认错误 ═══
print("\n── 确认错误 ────────────────────────────────────────────")
check("确认错误", "你的名字是张三吗？", "发热", "不是")
check("确认错误", "您是医生吗？", "发热", "不是")
check("确认错误", "痰是黄的吗？", None, "不是")

print("\n" + "=" * 80)
print(f" 总计: {passed + failed} 题 | 通过: {passed} | 失败: {failed}")
print("=" * 80)
sys.exit(0 if failed == 0 else 1)