# 交互式医疗 RAG 系统

## 概述

`interactive_rag.py` 提供命令行交互界面，用户可以直接输入问题进行医疗查询，支持向量检索和 BM25 关键词检索的混合检索。

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│              InteractiveRAG 类                       │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  对话数据   │  │ FAISS索引   │  │  BM25索引   │ │
│  │  68023条    │  │  向量检索   │  │  关键词检索 │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│         │                │                │        │
│         └────────────────┼────────────────┘        │
│                          ▼                          │
│                   ┌─────────────┐                   │
│                   │  RRF融合    │                   │
│                   │  结果排序   │                   │
│                   └─────────────┘                   │
└─────────────────────────────────────────────────────┘
```

## 核心代码解析

### 1. 类初始化

```python
class InteractiveRAG:
    def __init__(self):
        self.dialogues = []      # 对话数据列表
        self.titles = []         # 标题列表
        self.questions = []      # 问题列表
        self.answers = []        # 答案列表
        self.departments = []    # 科室列表
        self.vector_index = None # FAISS 向量索引
        self.bm25 = None         # BM25 索引
        self.tokenized_corpus = []  # 分词后的语料
        self.loaded = False      # 加载状态标志
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `dialogues` | list | 原始对话数据 |
| `vector_index` | faiss.Index | FAISS 向量索引 |
| `bm25` | BM25Okapi | BM25 关键词索引 |
| `loaded` | bool | 防止重复加载 |

### 2. 资源加载

```python
def load(self):
    if self.loaded:
        return True
    
    # [1/3] 加载医疗数据
    data_path = "/userdata/medical_rag/sbert_768_final.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    self.dialogues = data['dialogues']
    for d in self.dialogues:
        self.titles.append(d.get('title', ''))
        self.questions.append(d.get('question', ''))
        self.answers.append(d.get('answer', ''))
        self.departments.append(d.get('department', '未知'))
    
    # [2/3] 加载FAISS向量索引
    import faiss
    index_path = "/userdata/medical_rag/sbert_768_final_vector.faiss"
    self.vector_index = faiss.read_index(index_path)
    
    # [3/3] 加载BM25索引
    import pickle
    bm25_path = "/userdata/medical_rag/sbert_768_final_bm25.pkl"
    with open(bm25_path, 'rb') as f:
        data = pickle.load(f)
    self.bm25 = data['bm25']
    self.tokenized_corpus = data['tokenized_corpus']
    
    self.loaded = True
    return True
```

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1 | sbert_768_final.json | 68023 条对话数据 |
| 2 | sbert_768_final_vector.faiss | FAISS 向量索引 |
| 3 | sbert_768_final_bm25.pkl | BM25 关键词索引 |

### 3. 查询编码（演示版）

```python
def encode_query(self, query_text):
    """编码查询 (使用随机向量模拟SBERT)"""
    # 实际应用中应该使用SBERT模型编码
    # 这里使用随机向量作为演示
    np.random.seed(hash(query_text) % 2**32)
    return np.random.random((1, 768)).astype('float32')
```

| 代码 | 说明 |
|------|------|
| `hash(query_text) % 2**32` | 将文本哈希转为随机种子 |
| `np.random.random((1, 768))` | 生成 768 维向量 |

**注意**：演示版使用随机向量，实际应替换为 SBERT 编码。

### 4. 混合检索核心

```python
def search(self, query_text, k=3):
    # 1. 编码查询
    query_vector = self.encode_query(query_text)
    
    # 2. 向量搜索
    vec_scores, vec_indices = self.vector_index.search(query_vector, k*2)
    
    # 3. BM25搜索
    import jieba
    tokens = list(jieba.cut(query_text.lower()))
    tokens = [t for t in tokens if t.strip()]
    bm25_scores = self.bm25.get_scores(tokens)
    bm25_top_k = np.argsort(bm25_scores)[-k*2:][::-1]
    
    # 4. RRF融合
    combined = {}
    for rank, idx in enumerate(vec_indices[0]):
        if idx >= 0:
            combined[idx] = combined.get(idx, 0) + 0.6 / (rank + 1)
    
    for rank, idx in enumerate(bm25_top_k):
        combined[idx] = combined.get(idx, 0) + 0.4 / (rank + 1)
    
    # 5. 排序取Top-K
    sorted_results = sorted(combined.items(), key=lambda x: -x[1])
    final_indices = [idx for idx, _ in sorted_results[:k]]
    
    return results, total_time
```

### 5. RRF 融合算法详解

```
RRF (Reciprocal Rank Fusion) 公式:

RRF(d) = Σ 1/(k + rank(d))

本实现:
- 向量检索权重: 0.6
- BM25检索权重: 0.4
- rank从1开始
```

| 检索方式 | 权重 | 说明 |
|----------|------|------|
| 向量检索 | 0.6 | 语义相似度 |
| BM25 检索 | 0.4 | 关键词匹配 |

**融合示例**：
```
查询: "头痛怎么办"

向量检索结果: [idx_1, idx_5, idx_10]
  idx_1: 0.6/1 = 0.6
  idx_5: 0.6/2 = 0.3
  idx_10: 0.6/3 = 0.2

BM25检索结果: [idx_5, idx_20, idx_1]
  idx_5: 0.4/1 = 0.4
  idx_20: 0.4/2 = 0.2
  idx_1: 0.4/3 = 0.13

最终得分:
  idx_1: 0.6 + 0.13 = 0.73
  idx_5: 0.3 + 0.4 = 0.7
  idx_10: 0.2
  idx_20: 0.2

排序: [idx_1, idx_5, idx_10]
```

### 6. 交互模式

```python
def interactive_mode(self):
    while True:
        query = input("请输入问题: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if query.lower() == 'stats':
            self.show_stats()
            continue
        
        results, elapsed = self.search(query, k=3)
        
        for i, r in enumerate(results, 1):
            print(f"[{i}] [{r['department']}] {r['title']}")
            print(f"    问题: {r['question'][:60]}...")
            print(f"    回答: {r['answer'][:80]}...")
```

| 命令 | 功能 |
|------|------|
| 直接输入问题 | 执行检索 |
| `stats` | 显示系统统计 |
| `quit/exit/q` | 退出程序 |

### 7. 统计信息

```python
def show_stats(self):
    print(f"数据规模: {len(self.dialogues)} 条对话")
    print(f"向量维度: 768")
    print(f"向量索引: {self.vector_index.ntotal} 向量")
    
    # 科室分布
    dept_counts = {}
    for dept in self.departments:
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
    
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {dept}: {count}条")
```

## 使用示例

```bash
python3 interactive_rag.py

# 输出
======================================================================
加载医疗RAG系统...
======================================================================

[1/3] 加载医疗数据...
  ✓ 加载完成: 68023 条对话

[2/3] 加载FAISS向量索引...
  ✓ 索引加载完成: 68023 向量

[3/3] 加载BM25索引...
  ✓ BM25加载完成

======================================================================
系统加载完成! 可以开始查询
======================================================================

请输入问题: 头痛怎么办

搜索: 头痛怎么办
----------------------------------------------------------------------
检索时间: 15.23ms
找到结果: 3 条

[1] [神经内科] 偏头痛
    问题: 偏头痛怎么治疗？...
    回答: 偏头痛的治疗包括药物治疗和非药物治疗...

请输入问题: stats

系统统计
======================================================================
数据规模: 68023 条对话
向量维度: 768
向量索引: 68023 向量

科室分布 (前10):
  内科: 15000条
  外科: 8000条
  ...
```

## 数据文件说明

| 文件 | 大小 | 内容 |
|------|------|------|
| sbert_768_final.json | ~50MB | 对话数据 |
| sbert_768_final_vector.faiss | ~200MB | FAISS 索引 |
| sbert_768_final_bm25.pkl | ~30MB | BM25 索引 |

## 注意事项

1. **内存需求**：约 500MB 加载数据
2. **首次加载**：需要 2-3 秒
3. **演示版限制**：使用随机向量，实际需 SBERT 模型
4. **路径硬编码**：数据路径写死在代码中
