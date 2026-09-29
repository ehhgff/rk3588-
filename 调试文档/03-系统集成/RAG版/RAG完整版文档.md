# RAG医疗咨询系统完整版文档

## 一、项目概述

### 1.1 项目背景
本项目实现了一个面向RK3588平台的医疗咨询检索增强生成(RAG)系统，支持中文医疗问答，采用混合检索策略结合向量检索和关键词匹配，针对边缘设备进行了深度优化。

**最新版本**: 延迟加载版 (Lazy Loading)
- 启动时间: 0.5秒 (vs 13秒预加载版)
- 首次查询: ~13秒 (含SBERT模型加载)
- 后续查询: ~200ms

### 1.2 核心特性
| 特性 | 说明 |
|------|------|
| **延迟加载** | SBERT模型首次查询时加载，加速启动 |
| **混合检索** | FAISS向量检索 + 关键词匹配 |
| **D2嵌入策略** | 使用title+question组合生成语义嵌入 |
| **意图识别** | 基于医学术语的科室预测 |
| **过滤机制** | 反义词冲突检测 + 人群匹配过滤 |
| **语音集成** | 完整ASR+RAG+LLM+TTS流程 |

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
| 向量检索 | FAISS | 1.7.4 | 高效向量相似度搜索 |
| 语义编码 | Sentence-BERT | text2vec-base-chinese | 中文语义嵌入 |
| 大语言模型 | InternVL3-1B | RKNN | 文本生成 |
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
│              Zipformer RKNN模型                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      查询预处理                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   意图识别    │  │  术语检测    │  │  人群检测    │      │
│  │ (科室预测)    │  │ (关键词提取) │  │ (特殊人群)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG检索服务                               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              延迟加载版RAG服务                        │  │
│  │                                                      │  │
│  │  ┌─────────────┐    ┌─────────────┐                 │  │
│  │  │  轻量数据   │    │  FAISS索引  │                 │  │
│  │  │  (68K条)   │    │  (6.4MB)   │                 │  │
│  │  └─────────────┘    └─────────────┘                 │  │
│  │         │                    │                       │  │
│  │         └────────┬───────────┘                       │  │
│  │                  ▼                                   │  │
│  │         ┌─────────────┐                              │  │
│  │         │ SBERT编码器 │ ← 首次查询时加载 (391MB)     │  │
│  │         │  (延迟加载) │                              │  │
│  │         └─────────────┘                              │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM生成回复                                │
│              InternVL3-1B RKNN模型                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   语音合成 (TTS)                             │
│              MeloTTS RKNN模型                                │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 核心模块

#### 2.3.1 延迟加载RAG服务
```python
class SimpleRAGService:
    """简化版RAG服务 - 延迟加载"""
    
    def __init__(self):
        self.dialogues = None
        self.faiss_index = None
        self.encoder = None
        self.encoder_loaded = False
        
        # 立即加载轻量级数据
        self._load_data()      # 68,023条对话
        self._load_faiss()     # 6.4MB压缩索引
        # SBERT编码器延迟加载
    
    def _load_encoder(self):
        """延迟加载SBERT编码器"""
        if self.encoder_loaded:
            return
        
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_path, device='cpu')
        self.encoder_loaded = True
```

#### 2.3.2 语音助手集成
```bash
# 完整流程
voice_assistant_lazy_rag_v2.sh
    ├── 启动LLM服务 (model_service_daemon)
    ├── 启动RAG服务 (rag_medical_server_simple.py)
    ├── 启动TTS服务 (tts_service.py)
    ├── 语音识别 (zipformer)
    ├── RAG检索 (socket通信)
    ├── LLM生成 (socket通信)
    └── 语音合成 (socket通信)
```

---

## 三、数据流程

### 3.1 数据预处理流程
```
原始CMtMedQA数据 (280,000条)
       │
       ▼
┌──────────────┐
│   数据清洗    │  → 去除重复、无效条目
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   分层采样    │  → 确保科室平衡
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   D2嵌入     │  → title+question组合
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   索引构建    │  → FAISS IVF-PQ压缩
└──────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  最终数据: 68,023条                  │
│  - sbert_768_full.json (15MB)       │
│  - sbert_768_full_vector.faiss (6.4MB) │
└─────────────────────────────────────┘
```

### 3.2 查询处理流程
```
用户语音输入
   │
   ▼
┌──────────────┐
│   语音识别    │  → Zipformer (1.4s)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   查询编码    │  → SBERT编码 (首次8s, 后续200ms)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   向量检索    │  → FAISS (1-2ms)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   结果过滤    │  → 反义词/人群过滤
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   LLM生成    │  → InternVL3-1B (3s)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   语音合成    │  → MeloTTS (5s)
└──────────────┘
```

---

## 四、核心文件清单

### 4.1 源代码文件
| 文件 | 路径 | 说明 |
|------|------|------|
| rag_medical_server_simple.py | /userdata/medical_rag_full/ | 延迟加载版RAG服务 |
| voice_assistant_lazy_rag_v2.sh | /userdata/voice_assistant/ | 语音助手主脚本 |
| start_all_services_v2.sh | /userdata/voice_assistant/ | 一键启动脚本 |
| tts_service.py | /userdata/voice_assistant/ | TTS常驻服务 |
| compress_faiss_index.py | voice_assistant/ | FAISS压缩工具 |

### 4.2 数据文件
| 文件 | 路径 | 大小 | 说明 |
|------|------|------|------|
| sbert_768_full.json | /userdata/medical_rag_full/ | ~15MB | 68,023条对话数据 |
| sbert_768_full_vector.faiss | /userdata/medical_rag_full/ | ~6.4MB | IVF-PQ压缩索引 |

### 4.3 模型文件
| 路径 | 大小 | 说明 |
|------|------|------|
| /root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese | ~391MB | SBERT中文模型 |
| /data/internvl3/internvl3-1b_w8a8_rk3588.rkllm | ~1.1GB | LLM模型 |
| /userdata/melotts_deploy/model/*.rknn | ~100MB | TTS模型 |

---

## 五、性能指标

### 5.1 完整流程性能
| 步骤 | 首次耗时 | 后续耗时 | 说明 |
|------|----------|----------|------|
| 服务启动 | 0.5s | - | 延迟加载版 |
| 语音识别 | 1.4s | 1.4s | Zipformer |
| RAG检索 | 13s | 0.2s | 含SBERT加载 |
| LLM生成 | 3s | 3s | InternVL3-1B |
| 语音合成 | 5s | 5s | MeloTTS |
| **总耗时** | **23s** | **10s** | 端到端 |

### 5.2 资源占用
| 指标 | 数值 | 说明 |
|------|------|------|
| 内存占用 | ~1.5GB | 含所有服务 |
| Swap使用 | ~3MB | 2GB Swap配置 |
| 磁盘空间 | ~2GB | 模型+数据 |

### 5.3 准确率
| 模型 | 维度 | 准确率 |
|------|------|--------|
| SBERT | 768 | 87.5% |

---

## 六、部署配置

### 6.1 环境要求
```
RK3588开发板
- 内存: 4GB (建议)
- 存储: 8GB+
- 系统: Linux
- Swap: 2GB (必需)
```

### 6.2 部署步骤
```bash
# 1. 设置Swap
adb shell 'dd if=/dev/zero of=/swapfile bs=1M count=2048'
adb shell 'mkswap /swapfile && swapon /swapfile'

# 2. 推送数据
adb push sbert_768_full.json /userdata/medical_rag_full/
adb push sbert_768_full_vector.faiss /userdata/medical_rag_full/

# 3. 推送模型
adb push text2vec-base-chinese /root/.cache/huggingface/hub/

# 4. 启动服务
adb shell '/userdata/voice_assistant/start_all_services_v2.sh'

# 5. 运行语音助手
adb shell '/userdata/voice_assistant/voice_assistant_lazy_rag_v2.sh'
```

### 6.3 服务管理
```bash
# 查看服务状态
ps | grep -E "python|model_service"
ls -la /tmp/*.sock

# 查看日志
tail -f /tmp/rag_lazy_server.log
tail -f /tmp/llm_service.log
tail -f /tmp/tts_service.log

# 停止服务
pkill -f "rag_medical\|model_service\|tts_service"
```

---

## 七、Socket通信协议

### 7.1 RAG服务协议
```python
# 请求格式
{
    "action": "search",
    "query": "高血压吃什么药",
    "k": 2
}

# 响应格式
{
    "status": "ok",
    "data": {
        "results": [
            {
                "department": "内科",
                "title": "心血管内科",
                "answer": "..."
            }
        ],
        "time_ms": 200.5,
        "encoder_loaded": True
    }
}
```

### 7.2 LLM服务协议
```bash
# Ping测试
./llm_client ping

# 生成文本
./llm_client generate "提示词"
```

### 7.3 TTS服务协议
```python
# 请求格式
{
    "command": "synthesize",
    "text": "你好",
    "output": "/tmp/output.wav"
}

# 响应格式
{
    "status": "ok",
    "message": "合成完成",
    "output": "/tmp/output.wav"
}
```

---

## 八、优化建议

### 8.1 性能优化
1. **预加载SBERT** - 如需更快首次响应，可预加载编码器
2. **量化压缩** - 使用INT8量化减少内存占用
3. **批量查询** - 支持批量编码提高效率

### 8.2 功能扩展
1. **多轮对话** - 添加对话历史管理
2. **意图分类** - 更精细的科室分类
3. **知识更新** - 支持动态添加医疗知识

---

## 九、参考文档

- [RAG完整版问题解决汇总](../05-问题排查/RAG完整版问题解决汇总.md)
- [RK3588测试指南](../../voice_assistant/RK3588_TEST_GUIDE.md)
- [MeloTTS部署文档](../02-模型部署/MeloTTS/melotts部署总结.md)
