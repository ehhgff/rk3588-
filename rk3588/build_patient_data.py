#!/usr/bin/env python3
"""
将 as.txt（医生-病人问诊表）转换为 patient_data.json（RAG 扩展版）

为每个问题生成三种问法：
  1. 直接问：原始医生问法 → 病人原始回答
  2. 正确确认："你是X吗？" → "是的，我是X"
  3. 错误确认："你是Y吗？"（Y≠X）→ "不是，我是X"

保留离题矫正机制。
"""
import json, re, sys

INPUT = r"/home/ubuntu/桌面/ai/as.txt"
OUTPUT = r"/home/ubuntu/桌面/ai/voice_assistant/rk3588/patient_data.json"

# ─── 读取 as.txt ───
def parse_as_txt(path):
    """解析 as.txt，返回 [(topic, question, answer), ...]
    
    格式（Tab 分隔）：
        \t医生\t病人          ← 表头，跳过
        发热\t您叫什么名字？\t李华   ← 首行有 topic
        \t您今年多大年纪？\t35 岁   ← 后续行 topic 为空，沿用上一个
    """
    records = []
    current_topic = None
    
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if i == 0:
            continue  # 表头
        
        parts = line.strip().split("\t")
        
        if len(parts) >= 3:
            col0, col1, col2 = parts[0].strip(), parts[1].strip(), parts[2].strip()
        elif len(parts) == 2:
            col0, col1, col2 = "", parts[0].strip(), parts[1].strip()
        else:
            continue
        
        # 更新当前 topic
        if col0 and col0 not in ["医生", "病人"]:
            current_topic = col0
        
        if col1 and col2 and current_topic:
            records.append((current_topic, col1, col2))
    
    return records


# ─── 提取关键信息用于生成变体 ───
def extract_entity(q, a):
    """从问答中提取关键实体，用于生成变体问法"""
    q = q.strip()
    a = a.strip()
    
    # 名字类
    name_match = re.search(r'叫什么名字', q)
    if name_match:
        # 从回答提取名字
        name = a.replace("叫", "").replace("我叫", "").replace("李华", "李华").strip()
        if "李华" in a:
            return {"type": "name", "correct": "李华", "wrong": "张三"}
    
    # 职业类
    job_match = re.search(r'(从事|职业)', q)
    if job_match:
        job = a.replace("是", "").replace("我是", "").replace("教师", "教师").strip()
        if "教师" in a:
            return {"type": "job", "correct": "教师", "wrong": "医生"}
        if "煤矿" in a or "工人" in a:
            return {"type": "job", "correct": "煤矿工人", "wrong": "教师"}
    
    # 年龄类
    age_match = re.search(r'(年纪|年龄|多大)', q)
    if age_match:
        age = re.search(r'(\d+)', a)
        if age:
            return {"type": "age", "correct": age.group(1), "wrong": str(int(age.group(1)) + 10)}
        if "35" in a:
            return {"type": "age", "correct": "35", "wrong": "45"}
    
    # 婚姻类
    if re.search(r'结婚', q):
        if "已经结婚" in a or "结婚了" in a or "已婚" in a:
            return {"type": "marry_yes", "yes": "已经结婚了", "no": "还没有结婚"}
        return {"type": "marry_no", "yes": "已经结婚了", "no": "还没有结婚"}
    
    # 民族类
    if re.search(r'民族', q):
        ethnic = a.replace("是", "").replace("我是", "").replace("汉族", "汉族").strip()
        if "汉族" in a:
            return {"type": "ethnic", "correct": "汉族", "wrong": "回族"}
    
    # 是否/有没有类
    yesno_match = re.search(r'(有没有|是不是|会不会|能不能|要不要|是否|有没|能\S+吗|会\S+吗|是\S+吗)', q)
    if yesno_match:
        is_yes = any(w in a for w in ["有", "是", "会", "能", "经常", "是的"])
        return {"type": "yesno", "yes": a, "no": _negate_answer(q, a), "is_yes": is_yes}
    
    # 什么类（开放式）
    if re.search(r'什么|怎么|哪些|多少|多久', q):
        return {"type": "what", "answer": a}
    
    # 数字类
    num = re.search(r'(\d+)', a)
    if num:
        return {"type": "number", "correct": num.group(1), "wrong": str(int(num.group(1)) * 2), "answer": a}
    
    return {"type": "generic", "answer": a}


def _negate_answer(q, a):
    """生成否定回答"""
    # 处理常见的肯定回答模式
    if any(w in a for w in ["有", "是", "会", "能"]):
        # "有" -> "没有"（但只替换第一个肯定词）
        for pattern, replacement in [("有时候", "没有"), ("有的", "没有的"),
                                      ("没有", "没有"),  # 保护"没有"不变
                                      ("有", "没有"), ("是", "不是"),
                                      ("会", "不会"), ("能", "不能")]:
            if pattern in a and replacement != pattern:
                new_a = a.replace(pattern, replacement, 1)
                if new_a != a and '不不' not in new_a and '没没有' not in new_a:
                    a = new_a
                    break
        return a
    return a


def make_positive_confirmation(q, entity):
    """生成正确确认问法"""
    if entity["type"] == "name":
        return f"你是{entity['correct']}吗？"
    elif entity["type"] == "age":
        return f"您是{entity['correct']}岁吗？"
    elif entity["type"] == "job":
        return f"你是{entity['correct']}吗？"
    elif entity["type"] == "ethnic":
        return f"你是{entity['correct']}吗？"
    elif entity["type"] == "yesno":
        # 从原问题提取动词和关键词
        match = re.search(r'(有[没有]*|是[不是]*|会[不会]*|能[不能]*)(\S+?)[吗？]', q)
        if match:
            prefix = match.group(1).rstrip("没有不是不会不能")
            keyword = match.group(2)
            return f"{prefix}{keyword}吗？"
        return f"是{entity['yes'][:4]}吗？"
    elif entity["type"] == "marry_yes":
        return "是已经结婚吗？"
    elif entity["type"] == "marry_no":
        return "是已经结婚吗？"
    elif entity["type"] == "number":
        return f"是{entity['correct']}吗？"
    return None


def make_negative_confirmation(q, entity):
    """生成错误确认问法"""
    if entity["type"] == "name":
        return f"你的名字是{entity['wrong']}吗？"
    elif entity["type"] == "age":
        return f"您是{entity['wrong']}岁吗？"
    elif entity["type"] == "job":
        return f"你是{entity['wrong']}吗？"
    elif entity["type"] == "ethnic":
        return f"你是{entity['wrong']}吗？"
    elif entity["type"] == "yesno":
        if entity["is_yes"]:
            # 把"有没有X"变成"没有X吗"，把"是X"变成"不是X吗"
            # 优先匹配完整模式
            neg_q = re.sub(r'有没有', '没有', q)
            if neg_q != q:
                return neg_q.rstrip('？') + '吗？'
            neg_q = re.sub(r'(是|会|能)', '不是', q, count=1)
            if neg_q != q:
                return neg_q
            # 通用否定：添加"没有"
            q_clean = q.rstrip('？?')
            if not q_clean.endswith('没有'):
                return q_clean + '没有吗？'
            return q_clean + '吗？'
        else:
            # 把"没有X"变成"有X吗"（原问为否定，错误确认应变为肯定）
            # 先处理"有没有"情况，避免变成"有有"
            pos_q = re.sub(r'有没有', '有', q)
            if pos_q != q:
                return pos_q.rstrip('？?') + '吗？'
            pos_q = re.sub(r'没有', '有', q, count=1)
            if pos_q != q:
                return pos_q.rstrip('？?') + '吗？'
            pos_q = re.sub(r'(不是|不会|不能)', '', q, count=1)
            if pos_q != q:
                return pos_q.rstrip('？?') + '吗？'
            return q
    elif entity["type"] == "marry_yes":
        return "您还没有结婚吗？"
    elif entity["type"] == "marry_no":
        return "您还没有结婚吗？"
    elif entity["type"] == "number":
        return f"是{entity['wrong']}吗？"
    return None


def make_positive_answer(q, a, entity):
    """生成正确确认的回答"""
    if entity["type"] == "name":
        return f"是的，我叫{entity['correct']}"
    elif entity["type"] == "age":
        return f"是的，我今年{entity['correct']}岁"
    elif entity["type"] == "job":
        return f"是的，我是{entity['correct']}"
    elif entity["type"] == "ethnic":
        return f"是的，我是{entity['correct']}"
    elif entity["type"] == "yesno":
        return f"是的，{a}"
    elif entity["type"] in ("marry_yes", "marry_no"):
        return f"是的，{entity['yes']}"
    return a


def make_negative_answer(q, a, entity):
    """生成错误确认的回答"""
    if entity["type"] == "name":
        return f"不是，我叫{entity['correct']}"
    elif entity["type"] == "age":
        return f"不是，我今年{entity['correct']}岁"
    elif entity["type"] == "job":
        return f"不是，我是{entity['correct']}"
    elif entity["type"] == "ethnic":
        return f"不是，我是{entity['correct']}"
    elif entity["type"] == "yesno":
        if entity["is_yes"]:
            return f"不是，{entity['no']}"
        else:
            return f"是的，{entity['yes']}"
    elif entity["type"] in ("marry_yes", "marry_no"):
        return f"不是，{entity['no']}"
    elif entity["type"] == "number":
        return f"不是，是{entity['correct']}"
    return a


def generate_variants(records):
    """为每条记录生成三种问法"""
    entries = []
    
    for topic, q, a in records:
        # 跳过复合问句（含内部问号），难以生成正确的变体
        if '？' in q[:-1] or '?' in q[:-1]:
            entries.append({
                "question": q,
                "answer": a,
                "topic": topic,
            })
            continue
        
        entity = extract_entity(q, a)
        if not entity:
            continue
        
        # 1. 直接问
        entries.append({
            "question": q,
            "answer": a,
            "topic": topic,
        })
        
        # 2. 正确确认
        pos_q = make_positive_confirmation(q, entity)
        if pos_q and pos_q != q:
            pos_a = make_positive_answer(q, a, entity)
            entries.append({
                "question": pos_q,
                "answer": pos_a,
                "topic": topic,
            })
        
        # 3. 错误确认
        neg_q = make_negative_confirmation(q, entity)
        if neg_q and neg_q != q:
            neg_a = make_negative_answer(q, a, entity)
            entries.append({
                "question": neg_q,
                "answer": neg_a,
                "topic": topic,
            })
    
    return entries


def main():
    records = parse_as_txt(INPUT)
    print(f"解析到 {len(records)} 条问答记录")
    
    # 按 topic 统计
    from collections import Counter
    topics = Counter(t for t, _, _ in records)
    for t, c in topics.most_common():
        print(f"  {t}: {c} 条")
    
    entries = generate_variants(records)
    print(f"生成 {len(entries)} 条扩展条目")
    
    # 手动补充别名（因文本差异大、Jaccard 相似度低的问法）
    MANUAL_ENTRIES = [
        # 发热 — "您现在主要哪里不舒服？" 的别名
        {"question": "发烧吗？", "answer": "有发烧和咳嗽的症状", "topic": "发热"},
        {"question": "有发烧吗？", "answer": "有发烧和咳嗽的症状", "topic": "发热"},
        # 高血压 — "您的高血压病史有多久了？" 的别名
        {"question": "血压高多久了？", "answer": "大概有5年了", "topic": "原发性高血压"},
        {"question": "高血压多久了？", "answer": "大概有5年了", "topic": "原发性高血压"},
        # 慢支 — "咳的痰液是什么颜色的？" 的错误确认问法
        {"question": "痰是黄的吗？", "answer": "不是，是白色的黏痰", "topic": "慢性支气管炎"},
        {"question": "痰是黄色的吗？", "answer": "不是，是白色的黏痰", "topic": "慢性支气管炎"},
    ]
    entries.extend(MANUAL_ENTRIES)
    print(f"手动补充 {len(MANUAL_ENTRIES)} 条别名")
    
    # 添加离题问题
    offtopic_questions = [
        {"question": "唱首歌吧", "answer": "患者说了与看病无关的内容", "topic": "off_topic"},
        {"question": "今天天气怎么样", "answer": "患者说了与看病无关的内容", "topic": "off_topic"},
        {"question": "你会跳舞吗", "answer": "患者说了与看病无关的内容", "topic": "off_topic"},
        {"question": "讲个笑话", "answer": "患者说了与看病无关的内容", "topic": "off_topic"},
    ]
    entries.extend(offtopic_questions)
    
    # 构建 patient_rag_server.py 所需的嵌套格式
    # { "sections": { "科室名": { "qa_pairs": [ { "id": "q0", "topic": "...", "variations": [...] } ] } } }
    sections = {}
    qa_index = {}
    
    for e in entries:
        topic = e.get("topic", "其他")
        q = e["question"]
        a = e["answer"]
        
        # 用 topic+问题原文 作为分组 key（同一原始问题归类到同一个 qa_pair）
        # 确认问法通过原始问题(去掉确认后缀)关联, 这里简单用共享 topic 聚合
        # 为了保证同一原始问题的多个变体在同一个 qa_pair 里, 按 (topic, 原始问题) 分组
        
        if topic not in sections:
            sections[topic] = {"qa_pairs": []}
            qa_index[topic] = {}
        
        # 找原始问题: 如果是确认问法, 尝试反向还原
        # 简单策略: 按 (topic, question) 精确聚合
        orig_key = q  # 每个不同的问题文本作为一个独立 qa_pair
        
        if orig_key not in qa_index[topic]:
            qa_id = f"q{len(sections[topic]['qa_pairs'])}"
            qa_index[topic][orig_key] = {
                "id": qa_id,
                "topic": topic,
                "variations": []
            }
            sections[topic]["qa_pairs"].append(qa_index[topic][orig_key])
        
        qa_index[topic][orig_key]["variations"].append({
            "text": q,
            "answer": a
        })
    
    output_data = {"sections": sections}
    
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n写入 {OUTPUT}")
    total_qa = sum(len(sd["qa_pairs"]) for sd in sections.values())
    total_var = sum(len(qa["variations"]) for sd in sections.values() for qa in sd["qa_pairs"])
    print(f"总计 {total_qa} 条问答, {total_var} 条问法变体, {len(sections)} 个科室")
    print("Done.")


if __name__ == "__main__":
    main()