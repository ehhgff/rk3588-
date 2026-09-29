# 语音助手 RAG (检索增强生成) 部署总结

## 一、RAG 简介

**RAG (Retrieval-Augmented Generation)** 是一种将检索系统与生成模型结合的技术，让 AI 能够基于私有知识库回答问题。

### 1.1 为什么需要 RAG？

| 问题 | 解决方案 |
|------|---------|
| LLM 知识有截止日期 | RAG 提供最新知识 |
| LLM 可能产生幻觉 | RAG 基于事实回答 |
| 无法访问私有数据 | RAG 使用本地知识库 |
| 回答不够精准 | RAG 提供相关上下文 |

### 1.2 RAG 工作流程

```
用户提问 → 向量化 → 相似度检索 → 获取相关文档 → 拼接上下文 → LLM生成 → 回答
                ↓
          本地知识库 (文本文件)
```

---

## 二、系统架构

### 2.1 RAG 增强版语音助手架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG 增强版语音助手                         │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Zipformer  │ →  │  RAG检索    │ →  │   InternVL3-1B      │  │
│  │  (语音识别)  │    │  (知识检索)  │    │   (RAG增强生成)      │  │
│  └─────────────┘    └──────┬──────┘    └──────────┬──────────┘  │
│                            ↓                       ↓             │
│                     ┌─────────────┐         ┌─────────────┐     │
│                     │ 本地知识库   │         │   MeloTTS   │     │
│                     │ (TF-IDF索引)│         │  (语音合成)  │     │
│                     └─────────────┘         └─────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

| 组件 | 技术 | 功能 |
|------|------|------|
| **嵌入模型** | TF-IDF | 将文本转换为向量 |
| **向量存储** | Python Dict | 轻量级存储，无外部依赖 |
| **相似度计算** | 余弦相似度 | 计算查询与文档的相似度 |
| **检索策略** | Top-K | 返回最相关的 K 个文档 |

---

## 三、部署步骤

### 3.1 前置条件

确保基础版语音助手已部署：
- ✅ Zipformer: `/data/zipformer/`
- ✅ InternVL3-1B: `/data/internvl3/`
- ✅ MeloTTS: `/data/melotts_deploy/`

### 3.2 部署 RAG 模块

```bash
# 上传 RAG 检索脚本
adb push ~/桌面/ai/voice_assistant/rag_retrieve.py /data/voice_assistant/

# 上传 RAG 增强版语音助手脚本
adb push ~/桌面/ai/voice_assistant/voice_assistant_rag.sh /data/voice_assistant/

# 添加执行权限
adb shell "chmod +x /data/voice_assistant/voice_assistant_rag.sh"
```

### 3.3 开发板文件结构

```
/data/voice_assistant/
├── voice_assistant_test.sh      # 基础版语音助手
├── voice_assistant_rag.sh       # RAG 增强版语音助手
├── rag_retrieve.py              # RAG 检索模块
└── rag_index.pkl                # 知识库索引 (自动生成)
```

---

## 四、使用方法

### 4.1 运行 RAG 增强版语音助手

```bash
# 使用默认测试音频
cd /data/voice_assistant
./voice_assistant_rag.sh

# 使用指定音频文件
./voice_assistant_rag.sh /path/to/your/audio.wav
```

### 4.2 测试 RAG 检索功能

```bash
# 单独测试 RAG 检索
python3 /data/voice_assistant/rag_retrieve.py "怎么控制灯光"
```

**输出示例：**
```
基于以下信息回答问题（如果信息不足，请根据你的知识回答）：

[1] 智能灯光控制：说'打开客厅灯'可以打开客厅的灯光。说'关闭所有灯'可以关闭家里的所有灯光。支持亮度调节和色温调节。
[2] 常见问题：语音识别不准确时请确保环境安静、说话清晰。支持灯光、空调、窗帘等智能家居设备控制。

用户问题：怎么控制灯光
请回答：
```

### 4.3 完整流程演示

**运行命令：**
```bash
adb shell "/data/voice_assistant/voice_assistant_rag.sh"
```

**输出示例：**
```
========================================
    RAG 增强版语音助手系统
========================================

[INFO] RAG 模块已加载
[INFO] 语音助手启动成功！

[STEP] 步骤1/5: 准备音频文件
[INFO] 使用音频文件: /data/zipformer/model/test.wav

[STEP] 步骤2/5: 语音识别 (Zipformer)
Zipformer output: 怎么打开客厅灯
[RESULT] 识别结果: 怎么打开客厅灯

[STEP] 步骤3/5: RAG 知识检索
[RAG] 查询: 怎么打开客厅灯
[RAG] 检索完成
[INFO] 检索到的相关信息:
[1] 智能灯光控制：说'打开客厅灯'可以打开客厅的灯光...
[2] 常见问题：语音识别不准确时请确保环境安静...

[STEP] 步骤4/5: 对话理解 (InternVL3-1B + RAG)
[INFO] 输入文本: 怎么打开客厅灯
[INFO] 使用 RAG 增强的上下文
[INFO] 生成回复中...
[RESULT] 助手回复: 您可以说"打开客厅灯"来控制灯光...

[STEP] 步骤5/5: 语音合成 (MeloTTS)
[INFO] 合成文本: 您可以说"打开客厅灯"来控制灯光...
[RESULT] 语音合成完成

========================================
[INFO] 语音助手流程完成
[INFO] 总耗时: 12 秒
========================================
```

---

## 五、知识库管理

### 5.1 默认知识库内容

```python
# 智能家居
"智能灯光控制：说'打开客厅灯'可以打开客厅的灯光..."
"空调控制：说'打开空调'启动空调..."
"窗帘控制：说'打开窗帘'打开所有窗帘..."

# 设备信息
"本设备是 LubanCat 4 开发板，基于 RK3588 芯片..."
"语音助手功能：支持中英文语音识别..."

# 使用帮助
"如何使用语音助手：对设备说唤醒词唤醒..."
"常见问题：语音识别不准确时请确保环境安静..."
```

### 5.2 添加自定义知识

编辑 `rag_retrieve.py` 文件中的 `_create_default_kb` 方法：

```python
def _create_default_kb(self):
    default_kb = [
        # 添加你的知识
        "你的知识内容1...",
        "你的知识内容2...",
        
        # 原有知识...
    ]
    self.knowledge_base = default_kb
    self._build_index()
```

然后删除旧的索引文件，重新运行：
```bash
rm /data/voice_assistant/rag_index.pkl
python3 /data/voice_assistant/rag_retrieve.py "测试"
```

### 5.3 知识库格式

**推荐格式：**
```
主题：详细描述
- 要点1
- 要点2
- 示例：具体例子
```

**示例：**
```python
"天气查询：说'今天天气怎么样'可以查询当前天气。支持查询未来3天天气预报。",
"音乐播放：说'播放周杰伦的歌'可以播放音乐。支持暂停、下一首、音量调节。",
```

---

## 六、技术实现

### 6.1 TF-IDF 算法

**词频 (TF)：**
```
TF(t,d) = 词 t 在文档 d 中出现的次数
```

**逆文档频率 (IDF)：**
```
IDF(t) = log(N / (DF(t) + 1)) + 1
N = 总文档数
DF(t) = 包含词 t 的文档数
```

**TF-IDF：**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

### 6.2 余弦相似度

```
sim(A,B) = (A·B) / (||A|| × ||B||)

其中：
A·B = 向量 A 和 B 的点积
||A|| = 向量 A 的 L2 范数
```

### 6.3 检索流程

```python
def search(query, top_k=2):
    # 1. 查询分词
    query_tokens = tokenize(query)
    
    # 2. 计算查询向量
    query_vec = compute_tfidf(query_tokens)
    
    # 3. 计算与所有文档的相似度
    similarities = []
    for doc_vec in doc_vectors:
        sim = cosine_similarity(query_vec, doc_vec)
        similarities.append(sim)
    
    # 4. 返回 Top-K
    return top_k_documents
```

---

## 七、性能优化

### 7.1 当前性能

| 步骤 | 耗时 | 说明 |
|------|------|------|
| RAG 检索 | < 100ms | 纯 Python，轻量级 |
| 索引构建 | ~1s | 首次运行时 |
| 内存占用 | ~10MB | 小型知识库 |

### 7.2 优化建议

**1. 知识库优化**
- 控制知识库大小（< 1000 条）
- 文档长度适中（100-500 字）
- 定期清理无效知识

**2. 检索优化**
- 使用更复杂的分词（如 jieba）
- 添加同义词扩展
- 实现重排序机制

**3. 索引优化**
- 使用 FAISS 加速（需要安装）
- 预加载索引到内存
- 定期更新索引

---

## 八、应用场景

### 8.1 智能家居控制

```
用户：怎么打开客厅灯？
RAG检索：智能灯光控制文档
助手：您可以说"打开客厅灯"来控制灯光。也支持调节亮度，比如"把灯调暗一点"。
```

### 8.2 设备使用帮助

```
用户：这个设备是什么？
RAG检索：设备信息文档
助手：这是 LubanCat 4 开发板，基于 RK3588 芯片，支持语音助手功能。
```

### 8.3 产品知识问答

```
用户：你们的产品有什么功能？
RAG检索：产品功能文档
助手：我们的语音助手支持：1. 中英文语音识别 2. 智能家居控制 3. 本地知识问答...
```

---

## 九、故障排查

### 9.1 常见问题

**问题 1：RAG 检索不到相关内容**
```
原因：查询与知识库不匹配
解决：
1. 检查知识库是否包含相关内容
2. 调整相似度阈值（默认 0.05）
3. 优化知识库表达方式
```

**问题 2：检索速度慢**
```
原因：知识库过大
解决：
1. 减少知识库大小
2. 使用 FAISS 加速
3. 预加载索引
```

**问题 3：回答不够准确**
```
原因：上下文不够精确
解决：
1. 减少 Top-K 数量
2. 添加重排序机制
3. 优化知识库质量
```

### 9.2 调试方法

```bash
# 1. 测试检索功能
python3 /data/voice_assistant/rag_retrieve.py "你的查询"

# 2. 查看知识库内容
python3 -c "
import pickle
with open('/data/voice_assistant/rag_index.pkl', 'rb') as f:
    data = pickle.load(f)
    for doc in data['knowledge_base']:
        print(doc[:100])
        print('---')
"

# 3. 手动测试相似度
python3 -c "
from rag_retrieve import SimpleRAGRetriever
retriever = SimpleRAGRetriever()
retriever.load_index()
results = retriever.search('测试查询', top_k=3)
for doc, score in results:
    print(f'Score: {score:.3f}')
    print(f'Doc: {doc[:50]}...')
    print()
"
```

---

## 十、总结

### 10.1 已完成工作

✅ **RAG 增强版语音助手**
- 轻量级 TF-IDF 检索实现
- 纯 Python，无外部依赖
- 支持自定义知识库
- 集成到语音助手流程

✅ **知识库管理**
- 默认知识库（智能家居、设备信息、使用帮助）
- 支持动态添加知识
- 自动构建和保存索引

✅ **测试验证**
- RAG 检索功能正常
- 集成到语音助手流程
- 总耗时增加 < 100ms

### 10.2 系统能力对比

| 功能 | 基础版 | RAG版 |
|------|--------|-------|
| 语音识别 | ✅ | ✅ |
| 对话理解 | ✅ | ✅+RAG增强 |
| 语音合成 | ✅ | ✅ |
| 私有知识 | ❌ | ✅ |
| 精准回答 | 一般 | 更好 |

### 10.3 使用建议

- ✅ **产品说明书问答**：上传产品文档，实现智能客服
- ✅ **智能家居控制**：提供设备控制指令说明
- ✅ **企业内部知识**：上传内部文档，构建企业助手
- ✅ **教育辅助**：上传教材内容，实现智能答疑

---

## 十一、参考文档

- [语音助手部署总结](./语音助手部署总结.md)
- [MeloTTS 部署总结](./melotts部署总结.md)
- [InternVL3-1B 部署总结](./internvl3-1b部署总结.md)
- [Zipformer 部署总结](./zipformer部署总结.md)

---

**创建时间：** 2026-03-26  
**版本：** v1.0 - RAG 增强版  
**状态：** 已验证，可正常使用
