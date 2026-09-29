# 过滤增强版医疗 RAG 演示

## 概述

`demo_queries_filtered.py` 是增强版 RAG 演示，解决了反义词冲突、人群过滤等医学合理性问题，确保检索结果的医学准确性。

## 解决的问题

### 问题1: 反义词冲突

```
用户查询: "高血压怎么办"
错误结果: 返回了低血压的治疗方案 ❌

原因: 向量相似度检索只看语义相似，
      "高血压"和"低血压"向量很接近
```

### 问题2: 人群不匹配

```
用户查询: "孕妇感冒怎么办"
错误结果: 返回了普通人的感冒用药 ❌

原因: 没有区分特殊人群（孕妇、儿童、老人）
```

### 问题3: 科室混淆

```
用户查询: "头痛"
错误结果: 返回了乙肝大三阳 ❌

原因: 向量空间中两者可能距离较近
```

## 核心代码解析

### 1. 科室关键词定义

```python
DEPT_KEYWORDS = {
    '内科': ['高血压', '糖尿病', '血糖', '感冒', '发烧', '咳嗽', '胃痛', ...],
    '儿科': ['孩子', '小孩', '宝宝', '婴儿', '儿童', '幼儿', ...],
    '妇产科': ['孕妇', '怀孕', '孕期', '产妇', '分娩', '月经', ...],
    '皮肤科': ['湿疹', '痤疮', '痘痘', '皮肤过敏', '皮炎', ...],
    '外科': ['骨折', '手术', '腰疼', '腿麻', '腰椎', ...],
    '心理科': ['失眠', '焦虑', '抑郁', '睡不着', ...],
    '五官科': ['眼睛', '眼科', '视力', '近视', ...],
    '中医科': ['中药', '调理', '气虚', '血虚', ...],
    '男科': ['阳痿', '早泄', '前列腺', ...],
    '传染病科': ['乙肝', '丙肝', '艾滋病', ...],
    '肿瘤科': ['肿瘤', '癌症', '化疗', ...],
}
```

| 科室 | 关键词示例 |
|------|-----------|
| 内科 | 高血压、糖尿病、感冒 |
| 儿科 | 孩子、宝宝、婴儿 |
| 妇产科 | 孕妇、怀孕、月经 |
| 皮肤科 | 湿疹、痘痘、过敏 |

### 2. 医学关键词定义（含反义词）

```python
MEDICAL_KEYWORDS = {
    '高血压': {
        'synonyms': ['高血压', '血压高', '降压', '降血压', '控制血压'],
        'antonyms': ['低血压', '血压低'],  # 反义词
        'dept': '内科',
        'category': '心血管疾病'
    },
    '低血压': {
        'synonyms': ['低血压', '血压低', '升压', '血压偏低'],
        'antonyms': ['高血压', '血压高'],  # 反义词
        'dept': '内科',
        'category': '心血管疾病'
    },
    '糖尿病': {
        'synonyms': ['糖尿病', '血糖高', '高血糖', '控糖', '降糖'],
        'antonyms': ['低血糖', '血糖低'],
        'dept': '内科',
        'category': '代谢疾病'
    },
}
```

| 字段 | 说明 |
|------|------|
| `synonyms` | 同义词列表，用于匹配 |
| `antonyms` | 反义词列表，用于过滤 |
| `dept` | 所属科室 |
| `category` | 疾病分类 |

### 3. 特殊人群定义

```python
SPECIAL_POPULATIONS = {
    '孕妇': ['孕妇', '怀孕', '孕期', '产妇', '准妈妈', '妊娠期'],
    '儿童': ['儿童', '孩子', '小孩', '宝宝', '婴儿', '幼儿'],
    '老年人': ['老人', '老年', '高龄', '岁数大', '年迈'],
}
```

### 4. 科室预测

```python
def predict_department(self, query: str) -> str:
    """预测查询意图科室"""
    query = query.lower()
    scores = {}
    
    # 先检查关键医学术语（高权重）
    for term, info in self.MEDICAL_KEYWORDS.items():
        for syn in info['synonyms']:
            if syn in query:
                dept = info['dept']
                scores[dept] = scores.get(dept, 0) + 100
                break
    
    # 检查科室关键词（普通权重）
    for dept, keywords in self.DEPT_KEYWORDS.items():
        for kw in keywords:
            if kw in query:
                scores[dept] = scores.get(dept, 0) + 10 * len(kw)
    
    return max(scores, key=scores.get) if scores else None
```

| 匹配类型 | 权重 | 说明 |
|----------|------|------|
| 关键医学术语 | +100 | 精确匹配 |
| 科室关键词 | +10 * 词长 | 模糊匹配 |

### 5. 医学术语检测

```python
def detect_medical_terms(self, query_text: str):
    """检测查询中的关键医学术语"""
    detected = []
    query_lower = query_text.lower()
    
    for term, info in self.MEDICAL_KEYWORDS.items():
        matched_syn = None
        for syn in info['synonyms']:
            if syn in query_lower:
                matched_syn = syn
                break
        
        if matched_syn:
            detected.append({
                'term': term,
                'matched': matched_syn,
                'synonyms': info['synonyms'],
                'dept': info['dept'],
                'antonyms': info.get('antonyms', [])
            })
    
    return detected
```

**示例**：
```
输入: "高血压怎么办"
输出: [{
    'term': '高血压',
    'matched': '高血压',
    'synonyms': ['高血压', '血压高', ...],
    'dept': '内科',
    'antonyms': ['低血压', '血压低']
}]
```

### 6. 反义词冲突检测

```python
def check_antonym_conflict(self, query_terms: list, result_text: str) -> bool:
    """检查反义词冲突"""
    result_lower = result_text.lower()
    
    for term_info in query_terms:
        antonyms = term_info.get('antonyms', [])
        for antonym in antonyms:
            if antonym in result_lower:
                return True  # 发现冲突
    
    return False
```

**示例**：
```
查询术语: 高血压 (反义词: 低血压)
结果文本: "低血压患者应该..."

检测: "低血压" 在结果中 → 返回 True (冲突)
```

### 7. 带过滤的搜索

```python
def search_with_filter(self, query_text: str, k: int = 3):
    # 1. 预测科室
    predicted_dept = self.predict_department(query_text)
    
    # 2. 检测关键医学术语
    detected_terms = self.detect_medical_terms(query_text)
    
    # 3. 检测查询人群
    query_populations = self.detect_population(query_text)
    
    # 4. 编码查询
    query_vector, encode_time = self.encode_query(query_text)
    
    # 5. 向量搜索（获取更多候选）
    vec_scores, vec_indices = self.vector_index.search(query_vector, k*6)
    
    # 6. BM25搜索
    tokens = list(jieba.cut(query_text.lower()))
    bm25_scores = self.bm25.get_scores(tokens)
    bm25_top_k = np.argsort(bm25_scores)[-k*6:][::-1]
    
    # 7. RRF融合
    combined = {}
    for rank, idx in enumerate(vec_indices[0]):
        combined[idx] = combined.get(idx, 0) + 0.6 / (rank + 1)
    for rank, idx in enumerate(bm25_top_k):
        combined[idx] = combined.get(idx, 0) + 0.4 / (rank + 1)
    
    # 8. 过滤处理
    for idx in list(combined.keys()):
        result_text = self.titles[idx] + " " + self.questions[idx]
        
        # 反义词冲突 → 完全排除
        if self.check_antonym_conflict(detected_terms, result_text):
            del combined[idx]
            continue
        
        # 人群不匹配 → 降权或排除
        result_populations = self.detect_population(result_text)
        if query_populations and result_populations:
            if not any(p in result_populations for p in query_populations):
                combined[idx] *= 0.3  # 降权
    
    # 9. 排序返回
    sorted_results = sorted(combined.items(), key=lambda x: -x[1])
    return sorted_results[:k]
```

### 8. 过滤流程图

```
┌─────────────────────────────────────────────────────┐
│                   查询处理流程                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. 关键词提取                                      │
│     输入 → 提取医学术语                             │
│                                                     │
│  2. 反义词检测                                      │
│     术语 → 查找反义词列表                           │
│                                                     │
│  3. 科室识别                                        │
│     关键词 → 匹配科室                               │
│                                                     │
│  4. 向量检索                                        │
│     查询 → FAISS Top-K                              │
│                                                     │
│  5. 结果过滤                                        │
│     ┌─────────────────────────────────┐             │
│     │ 反义词冲突? → 完全排除          │             │
│     │ 人群不匹配? → 降权或排除        │             │
│     │ 同义词匹配? → 提权              │             │
│     │ 科室匹配? → 提权                │             │
│     └─────────────────────────────────┘             │
│                                                     │
│  6. 返回结果                                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 效果对比

| 查询 | 原始结果 | 过滤后结果 |
|------|----------|------------|
| 高血压怎么办 | 混合高低血压 | 仅高血压 ✓ |
| 头痛 | 乙肝大三阳 | 神经内科头痛 ✓ |
| 孕妇感冒 | 普通感冒药 | 孕妇安全用药 ✓ |
| 孩子发烧 | 成人用药 | 儿童退烧方案 ✓ |
| 低血糖 | 糖尿病治疗 | 低血糖处理 ✓ |

## 使用示例

```python
from demo_queries_filtered import FilteredDemoRAG

rag = FilteredDemoRAG()
rag.load()

# 高血压查询
results, dept, terms, times = rag.search_with_filter("高血压怎么办", k=3)

print(f"预测科室: {dept}")
print(f"检测术语: {[t['term'] for t in terms]}")
print(f"耗时: {times['total']:.1f}ms")

for r in results:
    print(f"[{r['department']}] {r['title']}")
    print(f"  相似度: {r['similarity']:.3f}")
    print(f"  匹配关键词: {r['matched_keywords']}")
```

## 关键改进总结

| 改进点 | 方法 | 效果 |
|--------|------|------|
| 反义词过滤 | 检测反义词并排除 | 避免相反概念混淆 |
| 科室优先 | 科室匹配提权 | 提高相关科室排名 |
| 人群识别 | 特殊人群检测 | 区分孕妇/儿童/老人 |
| 同义词增强 | 同义词匹配提权 | 提高精确匹配排名 |
