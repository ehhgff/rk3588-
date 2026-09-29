# core - 核心服务文件

本目录包含语音助手系统的核心服务程序，这些服务通过Unix Socket提供常驻服务。

## 文件说明

### streaming_vad.py
**主程序入口**
- 功能: 完整的语音助手流程，集成VAD、ASR、RAG、LLM、TTS
- 使用: `python3 streaming_vad.py`
- 依赖: 需要先启动RAG、LLM、TTS服务
- 流程: 语音输入 → VAD检测 → Zipformer识别 → RAG检索 → LLM生成 → TTS合成 → 语音输出

### melotts_service_rknn.py
**TTS语音合成服务 (RKNN优化版)**
- 功能: 提供基于MeloTTS的语音合成服务，使用RK3588 NPU加速
- Socket: `/tmp/melotts_service.sock`
- 使用: `python3 melotts_service_rknn.py`
- 参数: 支持语速调整 (0.5-2.0，默认0.6)
- 输出: 44.1kHz Float32音频

### qwen3_llm_service.py
**LLM大语言模型服务**
- 功能: 提供基于Qwen3-0.6B的文本生成服务
- Socket: `/tmp/qwen3_llm.sock`
- 使用: `python3 qwen3_llm_service.py`
- 模型: Qwen3-0.6B (W8A8量化，约800MB)
- 启动时间: ~3-4秒

### rag_optimized_server.py
**RAG检索服务 (科室优化版)**
- 功能: 提供医疗知识检索服务，支持科室分类索引和混合搜索
- Socket: `/tmp/rag_optimized.sock`
- 使用: `python3 rag_optimized_server.py`
- 特性: 
  - 17个科室分类索引
  - 关键词召回 + 向量精排
  - RKNN NPU加速
- 性能: 明确科室查询~300ms，模糊查询~1100ms

## 启动顺序

1. 启动RAG服务: `python3 rag_optimized_server.py`
2. 启动LLM服务: `python3 qwen3_llm_service.py`
3. 启动TTS服务: `python3 melotts_service_rknn.py`
4. 运行主程序: `python3 streaming_vad.py`

或使用一键启动脚本: `../scripts/start_all_services_v2.sh`

## 依赖关系

```
streaming_vad.py (主程序)
    ├── rag_optimized_server.py (RAG检索)
    ├── qwen3_llm_service.py (LLM生成)
    └── melotts_service_rknn.py (TTS合成)
```
