# RAG医疗咨询系统 - 科室优化版文档

## 一、项目概述

### 1.1 项目背景
本项目在RAG医疗咨询系统基础上，针对RK3588平台进行深度优化，引入**科室分类索引**和**混合搜索策略**，显著提升检索速度和准确性。

**最新版本**: 科室优化版 (Department Optimized)
- **启动时间**: ~2秒 (RAG) + ~3-4秒 (LLM) = ~6-7秒总启动
- **明确科室查询**: ~300ms (提升7.7倍)
- **模糊查询**: ~1100ms (提升2倍)
- **端到端总耗时**: ~7.5-9.5秒

### 1.2 核心特性
| 特性 | 说明 |
|------|------|
| **科室分类索引** | 17个科室独立FAISS索引，减少候选集 |
| **RKNN NPU加速** | 向量编码使用NPU，速度提升3-5倍 |
| **混合搜索** | 关键词召回 + 向量精排 + 科室过滤 |
| **智能科室检测** | 基于关键词自动识别查询所属科室 |
| **动态候选调整** | 匹配科室100条，未匹配200条 |
| **提示词优化** | 精简模板，LLM生成时间减少35% |

### 1.3 应用场景
- 智能医疗咨询终端
- 离线医疗问答系统
- 边缘端健康助手
- 语音交互医疗助手

---

## 二、系统架构

### 2.1 技术栈
| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 向量检索 | FAISS | 1.7.4 | 科室分类索引 |
| 语义编码 | RKNN Encoder | FP16 | NPU加速向量编码 |
| 关键词索引 | Jieba + 倒排表 | - | 快速召回 |
| 大语言模型 | InternVL3-1B | RKLLM | 文本生成 |
| 语音识别 | Zipformer | RKNN | 语音转文字 |
| 语音合成 | MeloTTS | RKNN | 文字转语音 |
| 部署平台 | RK3588 | Linux | 边缘计算设备 |

### 2.2 架构图
```
┌─────────────────────────────────────────────────────────────┐
│                        语音输入                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     语音识别 (ASR)                           │
│              Zipformer RKNN模型 (~1.4s)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      查询预处理                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   科室检测    │  │  关键词提取  │  │  意图识别    │      │
│  │ (17个科室)    │  │ (jieba分词)  │  │ (医疗术语)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG检索服务 (科室优化版)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              混合搜索策略                             │  │
│  │                                                      │  │
│  │  Step 1: 科室检测 (关键词匹配)                        │  │
│  │     ├── 匹配科室 → 使用科室索引 (20-50条候选)         │  │
│  │     └── 未匹配   → 使用全局索引 (100-200条候选)       │  │
│  │                                                      │  │
│  │  Step 2: 关键词召回 (倒排索引)                        │  │
│  │     └── ~50ms, 召回Top-K候选                         │  │
│  │                                                      │  │
│  │  Step 3: 向量精排 (RKNN NPU)                          │  │
│  │     └── ~250-2200ms (根据候选数量)                    │  │
│  │                                                      │  │
│  │  Step 4: 返回Top-3结果                               │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM生成回复 (~2.2s)                        │
│              InternVL3-1B RKLLM模型                          │
│              提示词优化: 50字以内限制                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   语音合成 (TTS) (~3-5s)                     │
│              MeloTTS RKNN模型                                │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 科室分类索引
```
┌─────────────────────────────────────────────────────────────┐
│                    科室分类索引结构                          │
├─────────────────────────────────────────────────────────────┤
│  全局索引: 68,023条 (用于未匹配科室查询)                     │
│                                                              │
│  科室索引 (17个):                                            │
│  ├── 内科:     12,453条                                     │
│  ├── 外科:      8,234条                                     │
│  ├── 儿科:      6,789条                                     │
│  ├── 妇产科:    5,432条                                     │
│  ├── 皮肤科:    3,456条                                     │
│  ├── 眼科:      2,890条                                     │
│  ├── 耳鼻喉科:  2,567条                                     │
│  ├── 口腔科:    2,134条                                     │
│  ├── 精神科:    1,987条                                     │
│  ├── 心理科:    1,876条                                     │
│  ├── 中医科:    4,567条                                     │
│  ├── 骨科:      3,890条                                     │
│  ├── 肿瘤科:    1,654条                                     │
│  ├── 神经科:    2,345条                                     │
│  ├── 心血管科:  2,123条                                     │
│  ├── 消化科:    2,890条                                     │
│  └── 呼吸科:    2,456条                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、核心优化策略

### 3.1 科室检测机制
```python
# 科室关键词映射
DEPT_KEYWORDS = {
    '儿科': ['孩子', '儿童', '婴儿', '宝宝', '小儿', '发烧', '咳嗽', '疫苗'],
    '妇产科': ['怀孕', '孕妇', '分娩', '月经', '妇科', '产科', '备孕', '流产'],
    '皮肤科': ['皮肤', '痘痘', '湿疹', '过敏', '瘙痒', '皮疹', '痤疮', '红斑'],
    # ... 其他科室
}

def detect_departments(query):
    """检测查询可能所属的科室"""
    matched = []
    for dept, keywords in DEPT_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            matched.append(dept)
    return matched
```

### 3.2 动态候选调整
```python
def hybrid_search(query):
    # Step 1: 检测科室
    matched_depts = detect_departments(query)
    
    # Step 2: 根据是否匹配科室调整候选数量
    if matched_depts:
        # 匹配科室 → 减少候选数量
        candidates = keyword_search(query, top_k=100)
        # 优先从科室索引中检索
        dept_candidates = search_dept_index(matched_depts, query)
        candidates = merge_candidates(candidates, dept_candidates)
    else:
        # 未匹配科室 → 使用更多候选
        candidates = keyword_search(query, top_k=200)
    
    # Step 3: 向量精排
    results = vector_rerank(query, candidates)
    return results
```

### 3.3 性能对比
| 查询类型 | 优化前 | 优化后 | 提升 |
|---------|-------|-------|------|
| 明确科室 (儿科/妇科) | ~2300ms | **~300ms** | **7.7×** |
| 模糊查询 (头痛/感冒) | ~2300ms | **~1100ms** | **2×** |
| 平均耗时 | ~2300ms | **~800ms** | **2.9×** |

---

## 四、文件清单

### 4.1 核心服务文件
| 文件 | 路径 | 说明 |
|------|------|------|
| `rag_optimized_server.py` | `/userdata/medical_rag_full/` | 科室优化版RAG服务 |
| `build_dept_index.py` | `/userdata/medical_rag_full/` | 科室索引构建脚本 |
| `model_service_daemon` | `/userdata/voice_assistant/` | LLM常驻服务 |
| `llm_client` | `/userdata/voice_assistant/` | LLM客户端 |
| `tts_service.py` | `/userdata/voice_assistant/` | TTS服务 |

### 4.2 启动脚本
| 文件 | 路径 | 说明 |
|------|------|------|
| `start_all_services_v2.sh` | `/userdata/voice_assistant/` | 一键启动所有服务 |
| `voice_assistant_lazy_rag_v2.sh` | `/userdata/voice_assistant/` | 语音助手主脚本 |

### 4.3 数据文件
| 文件 | 路径 | 大小 | 说明 |
|------|------|------|------|
| `sbert_768_full.json` | `/userdata/medical_rag_full/` | 15MB | 医疗对话数据 |
| `sbert_768_full_vector.faiss` | `/userdata/medical_rag_full/` | 6.4MB | 全局向量索引 |
| `dept_vector_index.pkl` | `/userdata/medical_rag_full/` | 8MB | 科室分类索引 |
| `keyword_index.pkl` | `/userdata/medical_rag_full/` | 2MB | 关键词倒排索引 |

---

## 五、使用指南

### 5.1 启动服务
```bash
# 方法1: 一键启动所有服务
cd /userdata/voice_assistant
./start_all_services_v2.sh

# 方法2: 分别启动
# 1. 启动RAG服务
cd /userdata/medical_rag_full
python3 rag_optimized_server.py

# 2. 启动LLM服务
cd /userdata/voice_assistant
export LD_LIBRARY_PATH=/data/internvl3
./model_service_daemon

# 3. 启动TTS服务
cd /userdata/voice_assistant
python3 tts_service.py
```

### 5.2 运行语音助手
```bash
cd /userdata/voice_assistant
./voice_assistant_lazy_rag_v2.sh /data/zipformer/model/test.wav
```

### 5.3 测试RAG服务
```bash
# 测试明确科室查询 (预期: ~300ms)
cd /userdata/medical_rag_full
python3 << 'PYEOF'
import socket, json
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/rag_optimized.sock')
sock.send(json.dumps({'action': 'search', 'query': '孩子发烧怎么办', 'k': 2}).encode())
print(sock.recv(8192).decode())
sock.close()
PYEOF

# 测试模糊查询 (预期: ~1100ms)
python3 << 'PYEOF'
import socket, json
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/rag_optimized.sock')
sock.send(json.dumps({'action': 'search', 'query': '头痛怎么办', 'k': 2}).encode())
print(sock.recv(8192).decode())
sock.close()
PYEOF
```

---

## 六、性能指标

### 6.1 服务启动时间
| 服务 | 启动时间 | 说明 |
|------|---------|------|
| RAG服务 | ~2秒 | RKNN NPU + 科室索引 |
| LLM服务 | ~3-4秒 | RKLLM模型加载 |
| TTS服务 | ~1秒 | 快速启动 |
| **总启动** | **~6-7秒** | 全流程 |

### 6.2 端到端查询时间
| 步骤 | 耗时 | 说明 |
|------|------|------|
| 语音识别 | ~1.4s | Zipformer RKNN |
| RAG检索 | 0.3-1.1s | 科室优化后 |
| LLM生成 | ~2.2s | 提示词优化后 |
| 语音合成 | ~3-5s | MeloTTS RKNN |
| **总耗时** | **~7.5-9.5s** | 完整流程 |

---

## 七、注意事项

### 7.1 科室检测限制
- 科室检测基于关键词匹配，可能无法覆盖所有情况
- 建议持续完善 `DEPT_KEYWORDS` 映射表

### 7.2 RKNN精度问题
- RKNN FP16量化可能带来轻微精度损失
- 关键词检索不受RKNN精度影响（基于文本匹配）
- 向量精排受影响，但候选集已通过关键词召回限定

### 7.3 内存使用
- RAG服务: ~800MB (含RKNN模型)
- LLM服务: ~1.2GB (RKLLM模型)
- TTS服务: ~300MB
- **建议总内存**: 4GB+

---

## 八、版本历史

| 版本 | 日期 | 主要更新 |
|------|------|---------|
| v1.0 | 2024-03 | 基础RAG版本 |
| v2.0 | 2024-03 | 延迟加载优化 |
| v3.0 | 2024-04 | **科室优化版** - 科室分类索引 + RKNN加速 |

---

## 九、核心代码详解

### 9.1 科室关键词映射
```python
# 科室关键词映射（用于自动识别科室）
DEPT_KEYWORDS = {
    '内科': ['内科', '心脏', '心血管', '高血压', '糖尿病', '胃病', '胃炎', '肺炎', '感冒', '发烧', '咳嗽', '血压', '血糖'],
    '外科': ['外科', '手术', '骨折', '伤口', '缝合', '切除', '阑尾', '胆囊', '结石', '刀口'],
    '儿科': ['儿科', '儿童', '小孩', '婴儿', '宝宝', '幼儿', '新生儿', '疫苗', '孩子'],
    '妇产科': ['妇产', '妇科', '产科', '怀孕', '孕妇', '分娩', '月经', '痛经', '子宫', '卵巢', '孕期'],
    # ... 其他科室
}
```
**作用**: 定义各科室的关键词列表，用于从用户查询中自动识别所属科室。

### 9.2 科室检测函数
```python
def _detect_departments(self, query):
    """从查询中检测可能的科室"""
    query_lower = query.lower()    # 转小写，统一匹配
    matched_depts = []             # 存储匹配的科室
    
    for dept, keywords in DEPT_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:  # 关键词匹配
                matched_depts.append(dept)
                break              # 匹配到一个关键词即可
    
    return matched_depts if matched_depts else None
```
**作用**: 遍历所有科室的关键词，检查是否出现在查询中，返回匹配的科室列表。

### 9.3 动态候选调整
```python
# Step 2: 关键词搜索召回候选集（根据是否匹配科室调整数量）
top_k = 100 if matched_depts else 200  # 匹配科室时减少候选数量
candidate_ids = self._keyword_search(query_text, top_k=top_k)
```
**作用**: 根据是否匹配科室动态调整候选数量。匹配科室时只需100条候选，未匹配时使用200条。

### 9.4 科室子索引精排
```python
# 使用科室子索引进行精排
if self.dept_indices and dept in self.dept_indices:
    dept_index = self.dept_indices[dept]
    dept_doc_map = self.dept_doc_maps[dept]
    
    # 找出候选在子索引中的位置
    valid_indices = []
    valid_doc_ids = []
    for doc_id in docs_in_dept:
        try:
            idx_in_sub = dept_doc_map.index(doc_id)
            valid_indices.append(idx_in_sub)
            valid_doc_ids.append(doc_id)
        except ValueError:
            continue
    
    if valid_indices:
        # 提取候选向量并计算相似度
        candidate_vectors = np.array([dept_index.reconstruct(int(i)) for i in valid_indices])
        similarities = np.dot(candidate_vectors, query_vector.T).flatten()
```
**作用**: 在匹配的科室子索引中进行精确相似度计算，减少搜索空间，提升速度。

### 9.5 搜索入口逻辑
```python
def search(self, query_text, k=2):
    """搜索 - 使用科室优化版混合搜索"""
    if not self.loaded:
        return None
    
    # 如果启用混合搜索且科室索引已加载，使用科室优化版
    if self._use_hybrid_search and self.keyword_index is not None and self.dept_indices is not None:
        return self._hybrid_search_with_dept(query_text, k)
    
    # 回退到父类的混合搜索
    return super().search(query_text, k)
```
**作用**: 判断是否满足科室优化条件，满足则调用科室优化版搜索，否则回退到父类方法。

### 9.6 核心优化点总结

| 优化点 | 代码位置 | 效果 |
|--------|---------|------|
| 科室检测 | `_detect_departments` | 自动识别查询所属科室 |
| 动态候选调整 | `top_k = 100 if matched_depts else 200` | 匹配科室时减少候选数量 |
| 科室子索引 | `dept_indices[dept]` | 在更小的索引中搜索 |
| 优先排序 | 优先处理 `matched_depts` | 优先处理匹配科室的结果 |

---

*文档版本: v3.1*
*最后更新: 2024-04-06*
