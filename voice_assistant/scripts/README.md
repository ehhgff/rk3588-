# scripts - 启动脚本

本目录包含各种启动脚本，用于简化服务启动和测试流程。

## 主要脚本

### start_all_services_v2.sh
**一键启动所有服务（推荐）**
- 功能: 自动设置Swap、清理旧服务、启动RAG/LLM/TTS服务
- 使用: `./start_all_services_v2.sh`
- 流程:
  1. 设置Swap (2GB)
  2. 清理旧服务进程
  3. 启动RAG服务 (~2秒)
  4. 启动LLM服务 (~3-4秒)
  5. 启动TTS服务 (~1-2秒)
- 总启动时间: ~6-8秒

### start_services.sh
**启动服务（简化版）**
- 功能: 启动核心服务，不包含Swap设置
- 使用: `./start_services.sh`

### start_qwen3.sh
**仅启动Qwen3 LLM服务**
- 功能: 单独启动LLM服务
- 使用: `./start_qwen3.sh`

### voice_assistant_lazy_rag_v2.sh
**语音助手完整流程脚本**
- 功能: 执行完整的语音助手流程（录音→识别→RAG→LLM→TTS）
- 使用: `./voice_assistant_lazy_rag_v2.sh [音频文件]`
- 注意: 需要先启动各项服务


## 使用建议

1. **首次启动**: 使用 `start_all_services_v2.sh` 一键启动所有服务
2. **单独调试**: 使用对应的单个服务启动脚本
