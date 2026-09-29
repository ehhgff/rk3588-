#!/usr/bin/env python3
"""Patient RAG 综合测试脚本

两种模式：
  1. 本地模式 (默认) — 直接测试 bigram Jaccard 匹配算法，不需要连接设备
  2. 远程模式 --remote — 连接设备上的 patient_rag.socket 测试

用法：
  python3 test_patient_rag.py              # 本地测试（推荐）
  python3 test_patient_rag.py --remote     # 连接设备测试
"""
import json
import sys
import os
from collections import Counter

# ─── Bigram Jaccard 相似度 ───
def char_bigram(s):
    return set(s[i:i+2] for i in range(len(s) - 1))

def jaccard_sim(a, b):
    if not a or not b:
        return 0.0
    ba, bb = char_bigram(a), char_bigram(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def load_patient_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "patient_data.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def query_local(data, query, threshold=0.30):
    """本地模拟 RAG 查询"""
    best_sim = 0.0
    best_entry = None
    close_matches = []

    for entry in data:
        sim = jaccard_sim(query, entry["question"])
        if sim > best_sim:
            best_sim = sim
            best_entry = entry
        if sim >= 0.15:
            close_matches.append({
                "question": entry["question"],
                "answer": entry["answer"],
                "topic": entry.get("topic", ""),
                "score": round(sim, 4)
            })

    close_matches.sort(key=lambda x: x["score"], reverse=True)
    close_matches = close_matches[:5]

    is_off_topic = False
    answer = ""
    topic = ""

    if best_entry and best_sim >= threshold:
        answer = best_entry["answer"]
        topic = best_entry.get("topic", "")
        if topic == "off_topic":
            is_off_topic = True
    elif best_entry and best_sim >= 0.20:
        answer = best_entry["answer"]
        topic = best_entry.get("topic", "")

    return {
        "status": "ok",
        "confidence": round(best_sim, 4),
        "answer": answer,
        "topic": topic,
        "is_off_topic": is_off_topic,
        "close_matches": close_matches,
        "matched_question": best_entry["question"] if best_entry else "",
    }


# ─── 测试用例 ───
# (query, expected_keyword, category, mode)
# mode: "exact"=答案包含关键词, "hit"=置信度≥0.30即可, "off_topic"=离题检测

TEST_CASES = [
    # ═══ 直接问 ═══
    # 发热
    ("你叫什么名字？",           "李华",          "直接问-发热", "exact"),
    ("发烧吗？",               "有发烧",        "直接问-发热", "exact"),
    ("您今年多大年纪？",         "35",           "直接问-发热", "exact"),
    ("您从事什么职业？",         "教师",          "直接问-发热", "exact"),
    ("您是哪个民族的？",         "汉族",          "直接问-发热", "exact"),
    ("您结婚了吗？",            "已经结婚",       "直接问-发热", "exact"),
    ("不舒服的症状持续有多久？",   "3天",          "直接问-发热", "exact"),
    ("发烧的时候体温最高到多少度？","39",          "直接问-发热", "exact"),
    ("吃过什么药了吗？",         "布洛芬",        "直接问-发热", "exact"),
    ("有过敏史吗？",            "没有",          "直接问-发热", "exact"),

    # 贫血
    ("您现在哪里不舒服？",       "头晕",          "直接问-贫血", "exact"),
    ("月经量多吗？",            "比较多",         "直接问-贫血", "exact"),
    ("最近食欲怎么样？",         "不好",          "直接问-贫血", "exact"),
    ("平时饮食喜欢吃什么？",      "油腻",          "直接问-贫血", "exact"),
    ("有没有做过检查？",         "血常规",        "直接问-贫血", "exact"),
    ("睡眠怎么样？",            "不太好",         "直接问-贫血", "exact"),
    ("有没有头晕眼前发黑的情况？","眼前发黑",       "直接问-贫血", "exact"),
    ("面色怎么样？",            "苍白",          "直接问-贫血", "exact"),

    # 慢性支气管炎
    ("您咳嗽有多久了？",         "3个多月",       "直接问-慢支", "exact"),
    ("咳嗽时有没有咳痰？",       "痰液量多",      "直接问-慢支", "exact"),
    ("咳的痰液是什么颜色的？",   "白色",          "直接问-慢支", "exact"),
    ("有喘息的情况吗？",         "有",            "直接问-慢支", "exact"),
    ("咳嗽一般发生在什么时间？",  "早晚",          "直接问-慢支", "exact"),
    ("有没有做过肺功能检查？",   "肺功能",         "直接问-慢支", "exact"),

    # 原发性高血压
    ("您的高血压病史有多久了？",  "5年",           "直接问-高血压", "exact"),
    ("有没有按时服用降压药物呢？","忘记吃",         "直接问-高血压", "exact"),
    ("服用的是哪些降压药物？",   "硝苯地平",       "直接问-高血压", "exact"),
    ("平时血压控制在多少？",     "140",           "直接问-高血压", "exact"),
    ("有没有做过动态血压监测？",  "没有",          "直接问-高血压", "exact"),

    # ═══ 正确确认问 ═══
    ("你是李华吗？",            "是的",          "确认正确-发热", "exact"),
    ("您是35岁吗？",            "是的",          "确认正确-发热", "hit"),
    ("你是教师吗？",            "是的",          "确认正确-发热", "exact"),
    ("咳痰吗？",               "是的",          "确认正确-慢支", "hit"),
    ("有痰吗？",               "有痰",          "确认正确-慢支", "hit"),
    ("症状是一下子出现的还是慢慢加重的？","满满加重","确认正确-贫血", "exact"),
    ("加重吗？",               "是的",          "确认正确-贫血", "hit"),

    # ═══ 错误确认问 ═══
    ("你的名字是张三吗？",       "不是",          "确认错误-发热", "exact"),
    ("您是医生吗？",            "不是",          "确认错误-发热", "exact"),
    ("您是45岁吗？",            "不是",          "确认错误-发热", "exact"),
    ("痰是黄的吗？",            "不是",          "确认错误-慢支", "exact"),

    # ═══ 离题检测 ═══
    ("唱首歌吧",               "离题",           "离题检测", "off_topic"),
    ("今天天气怎么样",          "离题",           "离题检测", "off_topic"),
    ("你会跳舞吗",             "离题",           "离题检测", "off_topic"),
    ("讲个笑话",               "离题",           "离题检测", "off_topic"),
]


def run_local_tests(data):
    """本地运行所有测试"""
    passed = 0
    failed = 0
    results = []

    print(f"{'='*70}")
    print(f"  Patient RAG 本地综合测试 — {len(data)} 条数据")
    print(f"{'='*70}")

    for query, expected, category, mode in TEST_CASES:
        res = query_local(data, query)
        answer = res["answer"]
        conf = res["confidence"]
        topic = res["topic"]
        off_topic = res["is_off_topic"]
        matched_q = res["matched_question"]

        if mode == "off_topic":
            is_match = off_topic
        elif mode == "hit":
            is_match = conf >= 0.30 and len(answer) > 0
        else:  # exact
            is_match = conf >= 0.30 and expected in answer

        tag = "✅" if is_match else "❌"
        if is_match:
            passed += 1
        else:
            failed += 1

        print(f"\n  {tag} [{category}]")
        print(f"     问: {query}")
        print(f"     匹配: 「{matched_q[:30]}…」 conf={conf:.2f}")
        print(f"     答: {answer[:50]}")
        if not is_match:
            print(f"     期望: {expected}")

        results.append({"query": query, "ok": is_match, "conf": conf,
                        "answer": answer, "off_topic": off_topic, "mode": mode})

    return passed, failed, results


# ─── 覆盖率统计 ───
def coverage_check(data, results):
    """检查测试覆盖了哪些 topic"""
    topics_in_data = set(e.get("topic", "") for e in data)
    topics_covered = set()

    for r in results:
        if r["ok"]:
            # 找到匹配的 entry
            for e in data:
                if e["question"] == r.get("query"):
                    topics_covered.add(e.get("topic", ""))
                    break

    print(f"\n{'='*70}")
    print(f"  覆盖率统计")
    print(f"{'='*70}")
    print(f"  数据中 Topic: {', '.join(sorted(topics_in_data))}")
    print(f"  测试覆盖 Topic: {', '.join(sorted(topics_covered))}")

    # 统计每个 category 通过率
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"pass": 0, "total": 0})
    for query, expected, category, mode in TEST_CASES:
        for r in results:
            if r["query"] == query:
                cat_stats[category]["total"] += 1
                if r["ok"]:
                    cat_stats[category]["pass"] += 1
                break

    print(f"\n  各类别通过率:")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        pct = s["pass"] / s["total"] * 100 if s["total"] > 0 else 0
        tag = "✅" if pct == 100 else "⚠️"
        print(f"    {tag} {cat}: {s['pass']}/{s['total']} ({pct:.0f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patient RAG 测试")
    parser.add_argument("--remote", action="store_true",
                       help="连接设备 RAG socket 测试")
    args = parser.parse_args()

    data = load_patient_data()

    if args.remote:
        # 远程测试：连接设备 socket
        import socket
        PATIENT_RAG_SOCK = "/tmp/patient_rag.sock"
        print(f"远程模式 — 连接 {PATIENT_RAG_SOCK}")

        def query_remote(query, threshold=0.30):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(PATIENT_RAG_SOCK)
            req = json.dumps({"query": query, "threshold": threshold})
            sock.sendall((req + "\n").encode("utf-8"))
            resp = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                resp += chunk
                if b"\n" in chunk:
                    break
            sock.close()
            return json.loads(resp.decode("utf-8"))

        passed = 0
        failed = 0
        for query, expected, category, mode in TEST_CASES:
            res = query_remote(query)
            answer = res.get("answer", "")
            conf = res.get("confidence", 0)
            off_topic = res.get("is_off_topic", False)

            if mode == "off_topic":
                is_match = off_topic
            elif mode == "hit":
                is_match = conf >= 0.30 and len(answer) > 0
            else:
                is_match = conf >= 0.30 and expected in answer

            tag = "✅" if is_match else "❌"
            if is_match:
                passed += 1
            else:
                failed += 1
            print(f"  {tag} [{category}] {query[:30]:<30s} conf={conf:.2f} → {answer[:30]}")

        print(f"\n总计: {passed}/{passed+failed} ✅")

    else:
        # 本地模式
        passed, failed, results = run_local_tests(data)
        coverage_check(data, results)

        print(f"\n{'='*70}")
        print(f"  总计: {passed+failed} 题")
        print(f"  ✅ 通过: {passed}")
        print(f"  ❌ 失败: {failed}")
        print(f"{'='*70}")
        sys.exit(0 if failed == 0 else 1)