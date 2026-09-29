# Voice Assistant - 语音助手系统

基于RK3588的医疗语音助手系统，支持语音输入、智能问答和语音输出。

## 系统架构

```
语音输入 → VAD检测 → 语音识别 → RAG检索 → LLM生成 → TTS合成 → 语音输出
```

## 目录结构

```
voice_assistant/
├── README.md                    # 项目主README
├── requirements.txt             # Python依赖
│
├── core/                        # 核心服务文件
│   ├── README.md
│   ├── streaming_vad.py         # 主程序
│   ├── melotts_service_rknn.py  # TTS服务
│   ├── qwen3_llm_service.py     # LLM服务
│   └── rag_optimized_server.py  # RAG服务
│
├── client/                      # 客户端文件
│   ├── README.md
│   └── qwen3_llm_client.py      # LLM客户端
│
├── scripts/                     # 启动脚本
│   ├── README.md
│   ├── start_all_services_v2.sh # 一键启动
│   ├── start_services.sh        # 简化版启动
│   ├── start_qwen3.sh           # 仅启动LLM
│   ├── voice_assistant_lazy_rag_v2.sh  # 完整流程
│   └── voice_assistant_streaming_v2.sh # 流式处理
│
├── tools/                       # 工具和索引构建
│   ├── README.md
│   ├── build_dept_index.py      # 构建科室索引
│   ├── build_text2vec_index.py  # 构建向量索引
│   ├── build_text2vec_index_vm.py # VM版本
│   ├── compress_faiss_index.py  # 压缩索引
│   └── check_rag_service.py     # 检查RAG服务
│
├── config/                      # 配置文件
│   ├── README.md
│   ├── config.yaml              # 默认配置
│   ├── config_optimized.yaml    # 优化配置
│   ├── config_fast.yaml         # 快速配置
│   └── config_cot.yaml          # CoT配置
│
├── docs/                        # 文档
│   ├── README.md
│   ├── 流式语音助手部署检查清单.md
│   ├── 流式语音助手架构设计.md
│   └── 医疗RAG语音助手测试指南.md
│
├── build/                       # 构建输出
│   ├── README.md
│   ├── llm_client               # LLM客户端可执行文件
│   └── model_service_daemon     # 模型服务守护进程
│
├── logs/                        # 日志文件
│
├── rk3588_deploy/               # RK3588部署文件
│   ├── README.md
│   ├── demo_queries_filtered.py
│   ├── interactive_rag.py
│   ├── medical_terminology_extended.py
│   └── rk3588_server.py
│
└── 讲解/                        # 讲解资料
    └── README.md
```

## 快速开始

### 1. 启动服务

```bash
cd scripts
./start_all_services_v2.sh
```

### 2. 运行语音助手

```bash
cd core
python3 streaming_vad.py
```

### 3. 对着麦克风说话

- 连续说5个字触发录音
- 说完等待0.7秒自动停止
- 系统自动回复

## 核心功能

- **语音活动检测 (VAD)**: Silero VAD，自动检测语音起止
- **语音识别 (ASR)**: Zipformer RKNN，流式实时识别
- **知识检索 (RAG)**: 科室优化版，17个科室分类索引
- **文本生成 (LLM)**: Qwen3-0.6B，医疗问答
- **语音合成 (TTS)**: MeloTTS RKNN，NPU加速

## 性能指标

| 指标 | 数值 |
|------|------|
| 服务启动时间 | ~6-8秒 |
| 端到端响应时间 | ~5-7秒 |
| 内存占用 | ~2.1GB |
| RAG检索速度 | 0.3-1.1秒 |

## 详细文档

- [核心服务说明](core/README.md)
- [客户端使用](client/README.md)
- [启动脚本说明](scripts/README.md)
- [工具使用](tools/README.md)
- [配置说明](config/README.md)
- [文档资料](docs/README.md)
- [构建输出](build/README.md)
- [RK3588部署](rk3588_deploy/README.md)
- [讲解资料](讲解/README.md)

## 调试文档

更详细的调试文档位于:
- `/home/ubuntu/桌面/ai/调试文档/03-系统集成/流式处理版/`

## 许可证

MIT License
