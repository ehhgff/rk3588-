# RK3588 AI 模型部署调试文档总览

> **项目概述**：在 LubanCat 4 (RK3588) 开发板上部署多个 AI 模型，构建完整的语音助手系统
> 
> **创建时间**：2026-03-26
> 
> **文档版本**：v2.0

---

## 📂 文档结构

```
调试文档/
├── 01-环境准备/          # 环境配置相关
├── 02-模型部署/          # 各模型独立部署
│   ├── MeloTTS/
│   ├── InternVL3-1B/
│   └── Zipformer/
├── 03-系统集成/          # 系统集成方案
│   ├── 基础版/
│   ├── RAG版/
│   └── 模型常驻版/
├── 04-性能优化/          # 性能测试与优化
├── 05-问题排查/          # 常见问题与解决方案
├── 06-使用指南/          # 部署清单和使用说明
└── README.md            # 本文档
```

---

## 📚 文档目录

### 01-环境准备
| 文档 | 说明 | 状态 |
|------|------|------|
| [环境配置.md](./01-环境准备/环境配置.md) | 开发环境搭建、依赖安装、网络配置 | ✅ 完成 |

### 02-模型部署

#### 2.1 语音合成 (TTS)
| 文档 | 说明 | 状态 |
|------|------|------|
| [melotts部署总结.md](./02-模型部署/MeloTTS/melotts部署总结.md) | MeloTTS 完整部署流程 | ✅ 完成 |
| [melotts调试文档.md](./02-模型部署/MeloTTS/melotts调试文档.md) | MeloTTS 调试过程记录 | ✅ 完成 |

#### 2.2 多模态对话 (VLM)
| 文档 | 说明 | 状态 |
|------|------|------|
| [internvl3-1b部署总结.md](./02-模型部署/InternVL3-1B/internvl3-1b部署总结.md) | InternVL3-1B 部署与优化 | ✅ 完成 |
| [internvl3-1b.md](./02-模型部署/InternVL3-1B/internvl3-1b.md) | InternVL3-1B 快速参考 | ✅ 完成 |

#### 2.3 语音识别 (ASR)
| 文档 | 说明 | 状态 |
|------|------|------|
| [zipformer部署总结.md](./02-模型部署/Zipformer/zipformer部署总结.md) | Zipformer 语音识别部署 | ✅ 完成 |

### 03-系统集成

#### 3.1 基础版
| 文档 | 说明 | 状态 |
|------|------|------|
| [语音助手系统方案.md](./03-系统集成/基础版/语音助手系统方案.md) | 语音助手架构设计方案 | ✅ 完成 |
| [语音助手部署总结.md](./03-系统集成/基础版/语音助手部署总结.md) | 基础版语音助手部署 | ✅ 完成 |

#### 3.2 RAG 版
| 文档 | 说明 | 状态 |
|------|------|------|
| [语音助手RAG部署总结.md](./03-系统集成/RAG版/语音助手RAG部署总结.md) | RAG 增强版语音助手 | ✅ 完成 |

#### 3.3 模型常驻版 ⭐ 推荐
| 文档 | 说明 | 状态 |
|------|------|------|
| [模型常驻服务部署文档.md](./03-系统集成/模型常驻版/模型常驻服务部署文档.md) | 模型常驻内存服务部署 | ✅ 完成 |

### 04-性能优化
| 文档 | 说明 | 状态 |
|------|------|------|
| [性能测试报告.md](./04-性能优化/性能测试报告.md) | 详细性能测试与优化建议 | ✅ 完成 |

### 05-问题排查
| 文档 | 说明 | 状态 |
|------|------|------|
| [问题与解决方案汇总.md](./05-问题排查/问题与解决方案汇总.md) | 所有问题及解决方案汇总 | ✅ 完成 |

### 06-使用指南
| 文档 | 说明 | 状态 |
|------|------|------|
| [完整部署清单.md](./06-使用指南/完整部署清单.md) | 部署清单和快速检查 | ✅ 完成 |
| [模型常驻服务使用说明.md](./06-使用指南/模型常驻服务使用说明.md) | 模型常驻服务使用指南 | ✅ 完成 |

---

## 🎯 项目成果

### 已部署模型

| 模型 | 类型 | 大小 | 功能 | 路径 |
|------|------|------|------|------|
| **MeloTTS** | TTS | ~160MB | 文本转语音 | `/data/melotts_deploy/` |
| **InternVL3-1B** | VLM | ~1.4GB | 多模态对话 | `/data/internvl3/` |
| **Zipformer** | ASR | ~128MB | 语音识别 | `/data/zipformer/` |

### 系统集成方案对比

| 系统 | 功能 | 路径 | 耗时 | 推荐度 |
|------|------|------|------|--------|
| **基础版** | ASR → LLM → TTS | `voice_assistant_test.sh` | ~36s | ⭐⭐ |
| **RAG版** | ASR → RAG → LLM → TTS | `voice_assistant_rag.sh` | ~36s | ⭐⭐⭐ |
| **优化版** | 优化后的完整流程 | `voice_assistant_optimized.sh` | ~24s | ⭐⭐⭐⭐ |
| **超优化版** | 进一步减少 token | `voice_assistant_super.sh` | ~18s | ⭐⭐⭐⭐⭐ |
| **模型常驻版** ⭐ | 模型常驻内存 | `voice_assistant_daemon.sh` | **~8.76s** | ⭐⭐⭐⭐⭐ |

---

## 📊 性能指标

### 模型性能

| 模型 | RTF | 延迟 | 评价 |
|------|-----|------|------|
| Zipformer (ASR) | 0.17 | ~1s | ⭐⭐⭐⭐⭐ 优秀 |
| InternVL3-1B (LLM) | - | ~2-20s | ⭐⭐⭐ 一般 |
| MeloTTS (TTS) | 0.38 | ~3s | ⭐⭐⭐⭐ 良好 |

### 系统性能演进

| 版本 | 总耗时 | 对话理解 | 优化效果 |
|------|--------|---------|---------|
| 原版 | ~36s | ~30s | 基准 |
| 超优化版 | ~18s | ~13s | -50% |
| **模型常驻版** | **~8.76s** | **~2.38s** | **-76%** |

---

## 🚀 快速开始

### 推荐：模型常驻版（最快）

```bash
# 1. 在 adb shell 中启动服务（保持运行）
adb shell
> cd /data/voice_assistant
> LD_LIBRARY_PATH=/data/internvl3 nohup ./model_service_daemon > /tmp/llm_service.log 2>&1 &
> sleep 30  # 等待模型加载

# 2. 在另一个终端运行语音助手
adb shell "/data/voice_assistant/voice_assistant_daemon.sh"
```

### 单次使用：超优化版

```bash
# 无需启动服务，直接运行
adb shell "/data/voice_assistant/voice_assistant_super.sh"
```

---

## 📁 关键文件位置

### 源代码
```
~/桌面/ai/
├── voice_assistant/
│   ├── model_service_daemon.cpp    # 模型常驻服务源码
│   ├── llm_client.cpp              # 客户端源码
│   ├── voice_assistant_daemon.sh   # 模型常驻版语音助手
│   ├── voice_assistant_super.sh    # 超优化版语音助手
│   └── build/                      # 编译输出
└── 调试文档/                        # 本文档所在目录
```

### 开发板部署
```
/data/
├── voice_assistant/                # 语音助手程序
│   ├── model_service_daemon        # 模型常驻服务
│   ├── llm_client                  # 客户端
│   └── voice_assistant_daemon.sh   # 语音助手脚本
├── internvl3/                      # InternVL3-1B 模型
├── melotts_deploy/                 # MeloTTS 模型
└── zipformer/                      # Zipformer 模型
```

---

## 🛠️ 常用命令

### 服务管理
```bash
# 启动服务
LD_LIBRARY_PATH=/data/internvl3 nohup ./model_service_daemon > /tmp/llm_service.log 2>&1 &

# 查看服务状态
ps | grep model_service
cat /proc/$(pgrep model_service_daemon)/status | grep VmRSS

# 停止服务
pkill -f model_service_daemon
```

### 测试命令
```bash
# 测试服务连接
./llm_client ping

# 查看模型状态
./llm_client status

# 测试生成
./llm_client generate '你好'
```

---

## 🔮 未来优化方向

1. **专用硬件加速**：使用 RK3588 的 NPU 专用算子
2. **模型量化**：进一步压缩模型体积
3. **流式推理**：边生成边输出，减少等待时间
4. **模型预热**：服务启动时预热，减少首次响应
5. **多客户端支持**：支持同时处理多个请求

---

## 📝 更新日志

### v2.0 (2026-03-26)
- ✅ 重构文档结构，分阶段组织
- ✅ 新增模型常驻内存服务
- ✅ 性能优化：总耗时 36s → 8.76s
- ✅ 新增详细部署文档

### v1.0 (2026-03-26)
- ✅ 完成三个模型部署
- ✅ 完成基础版语音助手
- ✅ 完成 RAG 增强版
- ✅ 完成性能优化

---

**维护者**: AI Assistant  
**最后更新**: 2026-03-26
