#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patient RAG Server — 患者模拟器快速检索服务
=============================================
为问诊训练提供 <50ms 的 RAG 检索，支持：
  - 四种问法（直接问、确认问-正确、确认问-错误）自动匹配
  - 四科室（发热/贫血/慢性支气管炎/原发性高血压）
  - 恶意/离题问题矫正机制
  - UNIX Socket + JSON 协议

架构：字符二元组 (bigram) Jaccard 相似度 + 同义词扩展
  无需 NPU，纯 CPU 匹配 <5ms（RK3588 A76）

请求格式 (JSON):
  {"query": "你叫什么名字？", "section": "发热", "threshold": 0.25}

响应格式 (JSON):
  {"status": "ok", "answer": "...", "confidence": 0.95,
   "matched_id": "fever_name", "section": "发热", "is_off_topic": false}

Socket: /tmp/patient_rag.sock
"""

import os
import sys
import json
import time
import socket
import threading
import signal

# ── 配置 ──────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "patient_data.json")
SOCKET_PATH = "/tmp/patient_rag.sock"
DEFAULT_THRESHOLD = 0.30        # 低于此值视为离题
DEFAULT_TOP_K = 3

import random

# ── 医学同义词（提升匹配率） ──────────────────────
SYNONYMS = {
    "发热": "发烧",
    "咳嗽": "咳",
    "咳痰": "痰咳嗽",
    "着凉": "受凉淋雨",
    "头痛": "头疼头昏",
    "乏力": "没力气疲劳累",
    "恶心": "想吐",
    "腹痛": "肚子疼肚子痛",
    "腹泻": "拉肚子",
    "心悸": "心慌心跳",
    "胸闷": "胸口闷",
    "气短": "喘不上气呼吸困难气喘",
    "失眠": "睡不着",
    "多梦": "做梦",
    "盗汗": "夜间出汗晚上出汗",
    "寒战": "发抖打冷战",
    "抽搐": "抽筋",
    "皮疹": "疹子红点",
    "瘙痒": "痒",
    "头晕": "头昏眩晕",
    "厌食": "没胃口吃不下不想吃",
    "消瘦": "瘦了体重下降",
    "水肿": "肿浮肿",
    "关节痛": "关节疼",
    "35": "三十五",
}

# ── 离题矫正语池（每科室多种风格，随机选用） ──────
CORRECTION_POOLS = {
    "发热": [
        # 疑惑型
        "医生，您怎么问这个？我主要来看发烧咳嗽的。",
        # 急切型
        "您先别问这些了，我烧到39度了快帮我看看吧！",
        # 提醒型
        "医生，我是来看发烧的，您是不是走神了？",
        # 抱怨型
        "您怎么老问不相干的，我发烧难受着呢。",
        # 拉回型
        "这个跟看病有关系吗？我发烧咳嗽3天了，您先帮我看看这个吧。",
        # 自嘲型
        "哎哟医生，您再问下去我病都自己好了，先看看发烧行不？",
        # 装傻型
        "我哪知道这个，我现在脑子都烧糊涂了，您快帮我看看吧。",
    ],
    "贫血": [
        "医生，还是先看看我的贫血问题吧，头晕得厉害。",
        "您问这个干嘛，我贫血头晕好久了。",
        "您别岔开话题呀，我是来看贫血的。",
        "我头晕眼花站都站不稳了，您先帮我看看贫血吧。",
        "医生，您是不是搞错病历了？我主要问题是贫血。",
    ],
    "慢性支气管炎": [
        "医生，还是先看看我的咳嗽吧，3个多月了不见好。",
        "您别管别的了，我这咳嗽咳得觉都睡不好。",
        "医生，我主要是来看咳嗽的，您先帮我听听肺？",
        "您问的这些跟我咳嗽有关系吗？帮我看看气管吧。",
    ],
    "原发性高血压": [
        "医生，先看看我的血压吧，最近老是头晕。",
        "您怎么说起这个来了，我来看高血压的。",
        "我血压高您还没给我看完呢，别扯远了。",
        "医生，我吃了药脸发红，您先帮我看看血压的问题吧。",
    ],
}

# ── 离题关键词 → 针对性矫正 ──────────────────────
OFF_TOPIC_INTENTS = [
    # (关键词, 概率权重, 响应)
    (["吃了吗", "吃了没", "吃饭", "吃什么"], 0.6,
     "您怎么总关心我吃没吃，我发烧您管不管？"),
    (["天气", "下雨", "冷吗", "热吗"], 0.5,
     "医生您怎么聊起天气来了，我来看病的！"),
    (["唱歌", "唱首", "表演", "才艺"], 0.7,
     "我又不是来表演的，我是来看病的！"),
    (["好看", "漂亮", "帅"], 0.5,
     "您别夸我了，快帮我看看病吧。"),
    (["电话", "微信", "手机"], 0.5,
     "医生，先看病吧，其他的一会儿再说。"),
    (["多少钱", "收费", "贵"], 0.4,
     "您别跟我谈钱，先看病行不行？"),
]


class PatientRAG:
    """患者数据检索引擎 — 字符 bigram Jaccard 相似度"""

    def __init__(self, data_path=DATA_FILE):
        self.data_path = data_path
        self.sections = {}          # section_name -> section_data
        self.flat_index = []        # 展平的检索条目列表
        self._load_data()

    # ── 加载数据 ──

    def _load_data(self):
        if not os.path.exists(self.data_path):
            print(f"[PatientRAG] 数据文件不存在: {self.data_path}")
            sys.exit(1)

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.sections = raw.get("sections", {})

        # 构建展平索引
        idx = 0
        for section_name, section_data in self.sections.items():
            for qa in section_data.get("qa_pairs", []):
                qa_id = qa.get("id", f"q{idx}")
                topic = qa.get("topic", "")
                for vi, var in enumerate(qa.get("variations", [])):
                    text = self._normalize(var["text"])
                    bigrams = self._char_bigrams(text)
                    self.flat_index.append({
                        "index": idx,
                        "section": section_name,
                        "qa_id": qa_id,
                        "topic": topic,
                        "variation_text": var["text"],
                        "answer": var["answer"],
                        "normalized": text,
                        "bigrams": bigrams,
                    })
                    idx += 1

        total_qas = sum(len(sd.get("qa_pairs", []))
                        for sd in self.sections.values())
        print(f"[PatientRAG] 已加载 {len(self.sections)} 科室, "
              f"{total_qas} 条问答, {len(self.flat_index)} 条问法变体")

    # ── 文本处理 ──

    @staticmethod
    def _normalize(text):
        """归一化：去标点、去空格、小写"""
        import re
        text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
        return text.strip()

    @staticmethod
    def _char_bigrams(text):
        """提取字符二元组"""
        return set(text[i:i+2] for i in range(len(text) - 1))

    def _expand_synonyms(self, text):
        """同义词扩展：在原文后追加同义词供匹配"""
        expanded = text
        for word, syn_str in SYNONYMS.items():
            if word in text:
                expanded += syn_str
        return expanded

    # ── 相似度计算 ──

    def _jaccard(self, bigrams_a, bigrams_b):
        """Jaccard 相似度"""
        if not bigrams_a or not bigrams_b:
            return 0.0
        intersection = bigrams_a & bigrams_b
        union = bigrams_a | bigrams_b
        return len(intersection) / len(union)

    # ── 检索 ──

    def search(self, query, section=None, threshold=DEFAULT_THRESHOLD,
               top_k=DEFAULT_TOP_K):
        """
        检索最匹配的问答条目。

        Returns:
            dict: {
                "answer": "...",
                "confidence": 0.xx,
                "matched_id": "...",
                "matched_variation": "...",
                "section": "...",
                "is_off_topic": bool,
                "close_matches": [...]  # top_k 候选
            }
        """
        query_norm = self._normalize(query)
        query_expanded = self._expand_synonyms(query_norm)
        query_bigrams = self._char_bigrams(query_expanded)

        # 计算所有条目的相似度
        scored = []
        for entry in self.flat_index:
            if section and entry["section"] != section:
                continue
            sim = self._jaccard(query_bigrams, entry["bigrams"])
            if sim > 0:
                scored.append((sim, entry))

        # 按相似度降序排列
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return self._off_topic_response(query, section)

        best_sim, best_entry = scored[0]

        # 低于阈值 → 离题
        if best_sim < threshold:
            return self._off_topic_response(query, section, best_entry)

        # 构造响应
        result = {
            "status": "ok",
            "answer": best_entry["answer"],
            "confidence": round(best_sim, 4),
            "matched_id": best_entry["qa_id"],
            "matched_variation": best_entry["variation_text"],
            "section": best_entry["section"],
            "topic": best_entry["topic"],
            "is_off_topic": False,
            "close_matches": [
                {
                    "answer": e["answer"],
                    "confidence": round(s, 4),
                    "qa_id": e["qa_id"],
                    "topic": e["topic"],
                    "variation": e["variation_text"],
                }
                for s, e in scored[:top_k]
            ],
        }
        return result

    def _off_topic_response(self, query, section=None, best_entry=None):
        """生成离题/未命中的矫正响应 — 多样化的有趣引导"""
        # 1) 检查是否有针对性关键词命中（更有趣）
        query_norm = self._normalize(query)
        for keywords, prob, response in OFF_TOPIC_INTENTS:
            if any(kw in query_norm for kw in keywords):
                if random.random() < prob:
                    return self._build_off_topic_result(response, query,
                                                        section, best_entry)

        # 2) 从科室语池随机选
        pool = CORRECTION_POOLS.get(section, CORRECTION_POOLS.get("发热"))
        off_topic_msg = random.choice(pool)

        return self._build_off_topic_result(off_topic_msg, query,
                                            section, best_entry)

    def _build_off_topic_result(self, message, query, section, best_entry):
        """构造离题响应的 JSON"""
        result = {
            "status": "ok",
            "answer": message,
            "confidence": 0.0,
            "matched_id": None,
            "matched_variation": None,
            "section": section or "unknown",
            "topic": None,
            "is_off_topic": True,
            "close_matches": [],
        }
        if best_entry:
            result["close_matches"] = [{
                "answer": best_entry["answer"],
                "confidence": round(
                    self._jaccard(
                        self._char_bigrams(self._expand_synonyms(
                            self._normalize(query))),
                        best_entry["bigrams"]
                    ), 4),
                "qa_id": best_entry["qa_id"],
                "topic": best_entry["topic"],
                "variation": best_entry["variation_text"],
            }]
        return result


# ════════════════════════════════════════════════════
#   Socket 服务
# ════════════════════════════════════════════════════

class PatientRAGServer:
    """UNIX Socket JSON 服务"""

    def __init__(self, socket_path=SOCKET_PATH):
        self.socket_path = socket_path
        self.rag = None
        self.server_socket = None
        self.running = False

    def start(self):
        """启动服务器"""
        print("[PatientRAG] 初始化检索引擎...")
        self.rag = PatientRAG(DATA_FILE)

        # 清理旧 socket
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)

        self.running = True
        print(f"[PatientRAG] 服务已启动: {self.socket_path}")

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        while self.running:
            try:
                client, addr = self.server_socket.accept()
                # 每个客户端连接交给线程处理
                t = threading.Thread(target=self._handle_client,
                                     args=(client,))
                t.daemon = True
                t.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[PatientRAG] 接受连接异常: {e}")

        self._cleanup()

    def stop(self):
        """停止服务器"""
        self.running = False

    def _signal_handler(self, signum, frame):
        print(f"\n[PatientRAG] 收到信号 {signum}，准备退出...")
        self.stop()

    def _cleanup(self):
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        print("[PatientRAG] 已退出")

    def _handle_client(self, client):
        """处理单个客户端请求"""
        try:
            client.settimeout(10.0)
            data = b""
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk
                if len(chunk) < 4096:
                    break

            if not data:
                client.close()
                return

            # 解析请求
            try:
                req = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as e:
                self._send_json(client, {
                    "status": "error",
                    "message": f"JSON 解析失败: {e}"
                })
                client.close()
                return

            query = req.get("query", "").strip()
            if not query:
                self._send_json(client, {
                    "status": "error",
                    "message": "query 不能为空"
                })
                client.close()
                return

            section = req.get("section", None)
            threshold = req.get("threshold", DEFAULT_THRESHOLD)

            # 检索
            t0 = time.time()
            result = self.rag.search(query, section, threshold)
            elapsed_ms = (time.time() - t0) * 1000
            result["elapsed_ms"] = round(elapsed_ms, 2)

            self._send_json(client, result)

        except Exception as e:
            try:
                self._send_json(client, {
                    "status": "error",
                    "message": str(e)
                })
            except Exception:
                pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    @staticmethod
    def _send_json(sock, data):
        """发送 JSON 响应"""
        resp = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
        sock.sendall(resp)


# ════════════════════════════════════════════════════
#   命令行测试 / 客户端
# ════════════════════════════════════════════════════

def query_server(query, section=None, socket_path=SOCKET_PATH,
                 threshold=DEFAULT_THRESHOLD):
    """查询正在运行的服务（同步）"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    sock.connect(socket_path)

    req = json.dumps({
        "query": query,
        "section": section,
        "threshold": threshold,
    }, ensure_ascii=False) + "\n"
    sock.sendall(req.encode("utf-8"))

    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in chunk:
            break

    sock.close()
    return json.loads(data.decode("utf-8"))


def run_standalone_test():
    """独立测试模式（无需服务器）"""
    print("=" * 50)
    print("Patient RAG — 独立测试模式")
    print("=" * 50)
    rag = PatientRAG(DATA_FILE)

    test_queries = [
        # 发热 - 姓名 三种问法
        ("发热", "你叫什么名字"),
        ("发热", "你是李华吗"),
        ("发热", "你的名字是张三吗"),
        # 发热 - 主要症状
        ("发热", "您现在主要哪里不舒服"),
        ("发热", "发烧吗"),
        ("发热", "有没有咳嗽"),
        # 发热 - 体温
        ("发热", "体温最高到多少度"),
        ("发热", "烧到39度了吗"),
        # 贫血
        ("贫血", "您现在哪里不舒服"),
        ("贫血", "月经量多不多"),
        ("贫血", "有生过小孩吗"),
        # 慢支
        ("慢性支气管炎", "您咳嗽有多久了"),
        ("慢性支气管炎", "有没有喘"),
        # 高血压
        ("原发性高血压", "高血压多久了"),
        ("原发性高血压", "按时吃药了吗"),
        # 离题问题
        ("发热", "今天天气怎么样"),
        ("发热", "你吃了吗"),
        ("发热", "唱首歌"),
        ("发热", "你是猪吗"),
    ]

    for section, query in test_queries:
        t0 = time.time()
        result = rag.search(query, section)
        elapsed = (time.time() - t0) * 1000
        symbol = "🔄" if result["is_off_topic"] else "✅"
        print(f"\n{symbol} [{section}] \"{query}\" ({elapsed:.1f}ms)")
        print(f"   → {result['answer']} (conf={result['confidence']:.2f})")
        if result["is_off_topic"] and result.get("close_matches"):
            cm = result["close_matches"][0]
            print(f"   (最佳候选: {cm['topic']} conf={cm['confidence']:.2f})")


def main():
    """入口：启动服务或测试"""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_standalone_test()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--query":
        # 查询运行中的服务：python3 patient_rag_server.py --query "你叫什么名字" 发热
        query = sys.argv[2] if len(sys.argv) > 2 else "你好"
        section = sys.argv[3] if len(sys.argv) > 3 else None
        result = query_server(query, section)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 启动服务
    server = PatientRAGServer()
    server.start()


if __name__ == "__main__":
    main()
