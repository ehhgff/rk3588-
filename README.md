# RK3588 端侧医疗语音助手

> 在 **RK3588（8GB）** 开发板上构建的**完全本地化、低延迟、隐私保护**的中文医疗问诊语音助手系统。
> 推理全程离线，NPU 加速，端到端响应约 **4 秒**。

---

## 一、项目简介

系统模拟"患者—医生"问诊流程：患者用语音提问 → 本地识别 → 检索患者档案 → 大模型生成回答 → 语音播报。
所有数据不出设备，兼顾**隐私、延迟、成本**。

**核心能力**

- 语音识别（ASR）：中文实时识别，支持流式 VAD 断句
- 患者档案 RAG：基于字符 bigram Jaccard 相似度的轻量检索，覆盖"直接问 / 正确确认 / 错误确认"多种问法
- 大模型对话（LLM）：DRAG 蒸馏后的 Qwen2-0.5B，医学问答无幻觉
- 语音合成（TTS）：常驻进程 + FIFO 管道，低延迟流式播报
- 离题矫正：识别无关问题并引导患者回到问诊

---

## 二、系统架构

```
 麦克风
   │  音频流
   ▼
┌─────────┐   ┌───────────┐   ┌────────────┐   ┌──────────┐
│  VAD    │──▶│  ASR      │──▶│  Patient   │──▶│   LLM    │
│ 断句    │   │ SenseVoice│   │   RAG      │   │ Qwen2-0.5B│
└─────────┘   └───────────┘   └────────────┘   └──────────┘
                                    │                │
                                    ▼                ▼
                              患者档案检索      生成回答文本
                                                     │
                                                     ▼
                                              ┌──────────┐
                                              │   TTS    │──▶ 扬声器
                                              │ Matcha   │
                                              └──────────┘
```

各模块通过 **UNIX Domain Socket / FIFO 管道** 常驻通信，避免反复加载模型的启动开销。

---

## 三、技术栈

| 模块 | 方案 | 加速 | 说明 |
|------|------|------|------|
| **ASR** | SenseVoice（sherpa-onnx） | RKNN / NPU | 中文语音识别，流式 VAD 断句 |
| **LLM** | Qwen2-0.5B-Instruct（DRAG 蒸馏） | RKLLM / NPU | w8a8 量化，40+ tokens/s |
| **RAG** | 患者档案检索（bigram Jaccard） | CPU | 轻量、无向量模型，ms 级响应 |
| **TTS** | Matcha TTS + Vocos | CPU 多线程 | 常驻进程，RTF ≈ 0.087（4 线程） |

**关键约束**：TTS 与 LLM 不争抢 NPU —— TTS 走 CPU，3 个 NPU 核心全部留给 LLM，保证推理速度。

---

## 四、分支说明

本仓库按**系统演进版本**组织，`main` 为最终版，其余为各阶段版本分支。

| 分支 | 说明 | 文件数 |
|------|------|--------|
| **main** ⭐ | **最终版（4s 延迟版）**：`rk3588/` 部署代码 + `scripts/` 模型转换量化脚本 + `model_config/` 模型配置 + 全部文档 | 184 |
| `流式处理版` | 流式语音助手：ASR 流式处理、`core/` 服务、`deploy/` 部署脚本、`讲解/` 源码解读、`other/` 实验代码（SenseVoice C++ 等） | 139 |
| `RAG科室优化版` | RAG 科室优化：`rk3588_deploy/` 服务、`tools/` 索引构建、`core/rag_optimized_server.py` | 28 |
| `模型常驻版` | 模型常驻内存服务：`build/`（`model_service_daemon` / `llm_client`） | 9 |
| `基础版` | 初版语音助手：仅文档（源码已归档清理） | 6 |

```bash
# 查看所有分支
git ls-remote --heads https://github.com/ehhgff/rk3588-

# 切换版本
git checkout 流式处理版
```

---

## 五、目录结构（main 分支）

```
rk3588-/
├── rk3588/                     # 最终版板端部署代码
│   ├── streaming_vad.py        # 主程序：VAD + ASR + RAG + LLM + TTS 编排
│   ├── patient_rag_server.py   # 患者档案 RAG 服务（bigram Jaccard）
│   ├── patient_data.json       # 患者问答数据（多问法变体）
│   ├── build_patient_data.py   # 由 as.txt 生成 patient_data.json
│   ├── qwen3_llm_streaming_service.py  # LLM 流式服务（RKLLM）
│   ├── sherpa_tts_daemon.c     # Matcha TTS C 常驻守护进程
│   ├── sherpa_tts_service.py   # TTS 服务管理
│   ├── run.sh / deploy_all.sh   # 启动与部署脚本
│   └── test_*.py               # 测试脚本
├── scripts/                    # 模型导出与量化脚本（80 个）
│   ├── sensevoice_onnx_to_rknn_v6.py   # SenseVoice → RKNN（最终版）
│   ├── qat_sensevoice.py               # QAT 量化
│   └── export_rkllm.py                 # LLM → RKLLM
├── model_config/               # 最终版模型配置（checkpoint-9375）
│   ├── config.json / generation_config.json
│   └── tokenizer.json / tokenizer_config.json
└── 调试文档/                    # 完整调试与部署文档
```

---

## 六、快速开始（板端）

```bash
# 1. 启动全部常驻服务（ASR / RAG / LLM / TTS）
adb shell "cd /userdata/voice_assistant && ./run.sh"

# 2. 运行语音助手主程序
adb shell "cd /userdata/voice_assistant && python3 streaming_vad.py"

# 3. 单独测试 RAG
adb shell "python3 test_patient_rag.py"
```

> 具体路径与依赖以 [`调试文档/`](./调试文档) 内对应版本文档为准。

---

## 七、模型文件说明

模型权重体积过大（合计约 **3GB**），**未纳入 Git**，需另行获取：

| 文件 | 大小 | 用途 |
|------|------|------|
| `qwen2-0.5b-medical-w8a8.rkllm` | 762 MB | **最终版 LLM**（DRAG 蒸馏 + w8a8） |
| `Qwen3-0.6B_W8A8_RK3588_2048.rkllm` | 889 MB | 早期 LLM |
| `sensevoice_encoder_ctc_100f.model` / `.data` | 893 / 37 MB | ASR 编码器 |
| `matcha_ljspeech.ckpt`、`matcha-icefall-zh-baker.tar.bz2` | 209 / 72 MB | Matcha TTS |
| `decoder/encoder-ZH_MIX_EN.onnx` / `.rknn` | ~320 MB | MeloTTS |
| `text2vec-*.onnx.data` / `*.faiss` / `*.rknn` | ~1 GB | RAG 向量索引 |

> GitHub 单文件上限 100MB，上述文件建议通过 **GitHub Releases** 或 **Git LFS** 分发。

---

## 八、文档索引

`调试文档/` 目录结构：

| 目录 | 内容 |
|------|------|
| `01-环境准备` | 开发环境搭建、依赖、网络配置 |
| `02-模型部署` | InternVL3-1B / MeloTTS / Zipformer 独立部署 |
| `03-系统集成` | **4s延迟版** / 流式处理版 / RAG版 / RAG科室优化版 / 模型常驻版 / 基础版 / DRAG.md |
| `05-问题排查` | 各类问题与解决方案汇总 |
| `06-使用指南` | 完整部署清单与使用说明 |
| `07-面试准备` | 性能报告与源码详解 |

---

## 九、性能指标

| 指标 | 数值 |
|------|------|
| 端到端响应 | ~4 s |
| LLM 推理速度 | 40+ tokens/s |
| 首 Token 延迟 | < 200 ms |
| TTS RTF | 0.087（4 线程） |
| 隐私 | 全程离线，无数据上传 |

---

## 十、许可与声明

本项目用于技术研究与学习交流。模型与数据集版权归各自原作者所有。