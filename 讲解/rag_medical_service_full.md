# rag_medical_service_full.py 代码详解

## 文件信息

- **路径**: `/home/ubuntu/桌面/ai/voice_assistant/rag_medical_service_full.py`
- **作用**: 医疗RAG服务核心类（完整数据版本）
- **功能**: 68,023条医疗知识检索、反义词冲突检测、提示词生成

---


## 逐行详解

### 1. 导入模块

```python
import sys
import os
import json
import numpy as np
import pickle
from pathlib import Path
```

| 模块 | 作用 |
|------|------|
| `sys` | 系统相关功能 |
| `os` | 文件操作、环境变量 |
| `json` | JSON 数据解析 |
| `numpy` | 数值计算 |
| `pickle` | Python 对象序列化 |
| `pathlib` | 路径操作 |

### 2. 离线模式设置

```python
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

| 环境变量 | 作用 |
|----------|------|
| `HF_HUB_OFFLINE` | Hugging Face 离线模式 |
| `TRANSFORMERS_OFFLINE` | Transformers 离线模式 |

**作用**: 避免网络请求，使用本地缓存模型。

### 3. 反义词对配置

```python
ANTONYM_PAIRS = [
    ('高血压', '低血压'),
    ('高血糖', '低血糖'),
    ('甲亢', '甲减'),
    ('失眠', '嗜睡'),
    ('便秘', '腹泻'),
    ('肥胖', '消瘦'),
]
```

| 反义词对 | 说明 |
|----------|------|
| `高血压/低血压` | 血压异常类型 |
| `高血糖/低血糖` | 血糖异常类型 |
| `甲亢/甲减` | 甲状腺功能异常 |
| `失眠/嗜睡` | 睡眠障碍类型 |
| `便秘/腹泻` | 消化系统异常 |
| `肥胖/消瘦` | 体重异常类型 |

### 4. 类初始化

```python
def __init__(self, data_dir="/userdata/medical_rag_full"):
    self.data_dir = data_dir
    self.dialogues = []
    self.titles = []
    self.questions = []
    self.answers = []
    self.departments = []
    self.vector_index = None
    self.bm25 = None
    self.encoder = None
    self.encoder_loaded = False
    self.loaded = False
    
    # 预加载所有资源
    self._preload()
```

| 属性 | 作用 |
|------|------|
| `dialogues` | 完整对话数据 |
| `titles` | 标题列表 |
| `questions` | 问题列表 |
| `answers` | 答案列表 |
| `departments` | 科室列表 |
| `vector_index` | FAISS 向量索引 |
| `bm25` | BM25 关键词索引 |
| `encoder` | SBERT 编码器 |
| `encoder_loaded` | 编码器加载状态 |
| `loaded` | 整体加载状态 |

### 5. 预加载方法

```python
def _preload(self):
    """预加载所有资源"""
    print("[RAG服务-完整版] 预加载资源中...", file=sys.stderr)
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `print("...", file=sys.stderr)` | 输出到标准错误流 | 避免干扰正常输出，便于日志收集 |

**设计思路**: 采用一次性预加载策略，牺牲启动时间换取查询性能。

```python
    # 加载数据
    data_path = f"{self.data_dir}/sbert_768_full.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `f"{self.data_dir}/sbert_768_full.json"` | 构建数据文件路径 | 使用 f-string 格式化，确保路径正确 |
| `encoding='utf-8'` | 指定 UTF-8 编码 | 确保中文字符正确读取 |
| `json.load(f)` | 解析 JSON 数据 | 将文件内容转为 Python 字典 |

```python
    self.dialogues = data['dialogues']
    for d in self.dialogues:
        self.titles.append(d.get('title', ''))
        self.questions.append(d.get('question', ''))
        self.answers.append(d.get('answer', ''))
        self.departments.append(d.get('department', '未知'))
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `data['dialogues']` | 提取对话列表 | JSON 结构中的 dialogues 字段 |
| `d.get('title', '')` | 安全获取字段值 | 如果字段不存在，返回默认值 |
| `append()` | 添加到列表 | 构建四个独立的列表便于索引 |

**数据结构设计**:
- `dialogues`: 完整对话对象列表
- `titles`: 标题列表（用于显示）
- `questions`: 问题列表（用于检索）
- `answers`: 答案列表（用于生成回复）
- `departments`: 科室列表（用于分类）

```python
    # 加载FAISS索引
    import faiss
    self.vector_index = faiss.read_index(f"{self.data_dir}/sbert_768_full_vector.faiss")
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `import faiss` | 导入 FAISS 库 | Facebook 开发的向量检索库 |
| `faiss.read_index()` | 读取索引文件 | 从磁盘加载预训练的向量索引 |
| `.faiss` 文件 | FAISS 索引格式 | 包含向量数据和索引结构 |

**FAISS 索引特点**:
- 支持快速近似最近邻搜索
- 优化内存使用和查询速度
- 支持 GPU 加速（本项目中未使用）

```python
    # 加载BM25索引
    with open(f"{self.data_dir}/sbert_768_full_bm25.pkl", 'rb') as f:
        bm25_data = pickle.load(f)
    self.bm25 = bm25_data['bm25']
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `'rb'` 模式 | 二进制读取模式 | 适用于 pickle 序列化文件 |
| `pickle.load(f)` | 反序列化对象 | 恢复 BM25 索引对象状态 |
| `bm25_data['bm25']` | 提取 BM25 对象 | pickle 文件中存储的字典结构 |

**BM25 索引作用**:
- 提供关键词匹配能力
- 作为向量检索的补充
- 提高检索的召回率

### 6. 编码器加载

```python
def _load_encoder(self):
    """加载SBERT编码器"""
    try:
        from sentence_transformers import SentenceTransformer
        
        model_path = '/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots/183bb99aa7af74355fb58d16edf8c13ae7c5433e'
        self.encoder = SentenceTransformer(model_path)
        self.encoder_loaded = True
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `try:` | 异常处理开始 | 防止模型加载失败导致整个服务崩溃 |
| `from sentence_transformers import SentenceTransformer` | 导入句子编码器 | 使用 Hugging Face 的 Sentence-BERT 实现 |
| `model_path` | 本地模型路径 | 离线模式下的模型缓存位置 |
| `SentenceTransformer(model_path)` | 创建编码器实例 | 加载预训练的 SBERT 模型 |
| `self.encoder_loaded = True` | 设置加载标志 | 用于后续的状态检查 |

**SBERT 模型特点**:
- **模型名称**: text2vec-base-chinese
- **开发者**: shibing624
- **向量维度**: 768 维
- **语言**: 中文优化
- **用途**: 将文本转换为语义向量

**模型路径结构解析**:
```
/root/.cache/huggingface/hub/
├── models--shibing624--text2vec-base-chinese/
    └── snapshots/
        └── 183bb99aa7af74355fb58d16edf8c13ae7c5433e/  # 模型文件
```

**离线模式优势**:
1. **启动速度**: 无需网络下载，直接加载本地缓存
2. **稳定性**: 不依赖外部网络连接
3. **一致性**: 确保模型版本固定

### 7. 反义词冲突检测

```python
def _check_antonym_conflict(self, query, result_text):
    """检查反义词冲突"""
    for pos_word, neg_word in self.ANTONYM_PAIRS:
        if pos_word in query and neg_word in result_text:
            return True
        if neg_word in query and pos_word in result_text:
            return True
    return False
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `for pos_word, neg_word in self.ANTONYM_PAIRS:` | 遍历反义词对 | 检查所有预定义的反义词组合 |
| `pos_word in query and neg_word in result_text` | 正向冲突检测 | 查询包含正向词，结果包含负向词 |
| `neg_word in query and pos_word in result_text` | 反向冲突检测 | 查询包含负向词，结果包含正向词 |
| `return True` | 发现冲突 | 立即返回，不再检查其他对 |

**冲突检测逻辑示例**:
```
查询: "高血压吃什么药"
结果: "低血压患者饮食注意事项"
检测: "高血压" in 查询 and "低血压" in 结果 → 冲突！
```

**设计目的**: 避免语义混淆，确保检索结果与查询意图一致。

### 8. 查询编码

```python
def encode_query(self, query_text):
    """编码查询文本"""
    if self.encoder is None:
        return None
    try:
        embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        return embedding.astype('float32')
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `if self.encoder is None:` | 编码器检查 | 确保编码器已成功加载 |
| `try:` | 异常处理 | 防止编码过程出错 |
| `[query_text]` | 列表包装 | SBERT 要求输入为列表格式 |
| `convert_to_numpy=True` | 转为 NumPy 数组 | 便于 FAISS 处理 |
| `astype('float32')` | 单精度转换 | 减少内存占用，提高计算效率 |

**编码过程详解**:
1. **输入**: "高血压吃什么药" (字符串)
2. **SBERT处理**: 分词 → 词向量 → 池化 → 归一化
3. **输出**: [0.123, 0.456, ..., 0.789] (768维向量)

**精度选择原因**:
- `float32`: 精度足够，内存占用小
- `float64`: 精度更高，但内存占用翻倍

### 9. 搜索方法

```python
def search(self, query_text, k=2):
    """搜索医疗知识"""
    if not self.loaded:
        return None
    
    # 编码查询
    query_vector = self.encode_query(query_text)
    if query_vector is None:
        return None
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `if not self.loaded:` | 服务状态检查 | 确保所有资源已加载完成 |
| `query_vector = self.encode_query(query_text)` | 文本编码 | 将自然语言转为向量表示 |
| `if query_vector is None:` | 编码结果检查 | 处理编码失败的情况 |

**参数设计**:
- `query_text`: 用户查询的自然语言文本
- `k=2`: 默认返回2条最相关结果

```python
    # 向量检索
    vec_scores, vec_indices = self.vector_index.search(query_vector, k*3)
```

| 代码 | 详细解释 | 技术细节 |
|------|----------|----------|
| `self.vector_index.search()` | FAISS 搜索 | 执行近似最近邻搜索 |
| `query_vector` | 查询向量 | 768维的语义表示 |
| `k*3` | 检索数量 | 检索3倍结果用于后续过滤 |
| `vec_scores` | 相似度分数 | 余弦相似度值数组 |
| `vec_indices` | 索引位置 | 对应数据集中条目的索引 |

**检索策略**:
- **检索数量**: k*3 = 6条（当k=2时）
- **过滤原因**: 后续进行反义词冲突检测和相似度调整
- **性能优化**: 一次检索多个结果，减少FAISS调用次数

```python
    # 构建结果
    results = []
    for rank, idx in enumerate(vec_indices[0]):
        if 0 <= idx < len(self.dialogues):
            result_text = self.titles[idx] + " " + self.questions[idx]
            
            # 检查反义词冲突
            if self._check_antonym_conflict(query_text, result_text):
                continue
```

**作用**: 过滤反义词冲突的结果。

```python
            # 计算相似度
            cosine_sim = float(vec_scores[0][rank])
            cosine_sim = max(0.0, min(1.0, cosine_sim))
            
            results.append({
                'department': self.departments[idx],
                'title': self.titles[idx],
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'similarity': cosine_sim
            })
```

| 字段 | 作用 |
|------|------|
| `department` | 科室信息 |
| `title` | 对话标题 |
| `question` | 用户问题 |
| `answer` | 医生回答 |
| `similarity` | 余弦相似度 |

### 10. 提示词生成

```python
def format_prompt(self, query, results):
    """格式化提示词"""
    if not results:
        return f"用户问：{query}\n请基于医疗知识回答。"
    
    context = ""
    for i, r in enumerate(results, 1):
        context += f"\n参考{i} [{r['department']}]：\n"
        context += f"问题：{r['question']}\n"
        context += f"答案：{r['answer'][:200]}...\n"
    
    prompt = f"基于以下医疗知识回答用户问题：\n{context}\n\n用户问：{query}\n请给出专业、准确的医疗建议。"
    return prompt
```

**格式示例**:
```
基于以下医疗知识回答用户问题：

参考1 [内科]：
问题：高血压患者吃什么药比较好
答案：建议服用降压药，如氨氯地平...

用户问：高血压吃什么药
请给出专业、准确的医疗建议。
```

### 11. 兼容接口

```python
def generate_prompt(self, query, k=2):
    """生成提示词 (兼容简化版服务接口)"""
    # 先执行搜索
    results = self.search(query, k=k)
    # 然后格式化提示词
    return self.format_prompt(query, results)
```

**作用**: 提供与简化版服务相同的接口。

---

## 数据流程

```
用户查询 → 编码向量 → FAISS检索 → 过滤冲突 → 格式化提示词 → LLM生成
```

---

## 性能特点

| 特性 | 说明 |
|------|------|
| 数据量 | 68,023 条医疗对话 |
| 检索方式 | FAISS 向量检索 |
| 冲突检测 | 反义词对过滤 |
| 编码器 | SBERT 中文模型 |
| 加载方式 | 预加载（启动慢，查询快） |
| 兼容性 | 与简化版服务接口一致 |

---

## 总结

`rag_medical_service_full.py` 是医疗RAG系统的**核心检索引擎**，具有以下特点：

1. **完整数据支持**: 68,023条医疗知识
2. **智能过滤**: 反义词冲突检测，避免错误结果
3. **高性能检索**: FAISS向量索引，毫秒级响应
4. **离线模式**: 无需网络连接，使用本地缓存
5. **兼容接口**: 与简化版服务无缝对接

这个类为整个医疗语音助手提供了强大的知识检索能力！