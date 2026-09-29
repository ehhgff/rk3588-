# RK3588 医疗RAG语音助手测试指南

## 系统概述

基于RK3588的医疗RAG语音助手系统（延迟加载版），集成：
- **语音识别**：Zipformer (RKNN NPU加速)
- **医疗RAG检索**：SBERT-768 + FAISS (68,023条医疗知识)
- **对话理解**：InternVL3-1B (RKLLM NPU加速)
- **语音合成**：MeloTTS (RKNN NPU加速)

**最新版本**: 延迟加载版 (Lazy Loading)
- 启动时间: 0.5秒
- 首次查询: ~13秒（含SBERT模型加载）
- 后续查询: ~200ms

---

## 文件说明

### 核心文件（P0 - 必须）

| 文件 | 说明 | 优先级 |
|------|------|--------|
| `start_all_services_v2.sh` | **一键启动所有服务（推荐）** | P0 |
| `voice_assistant_lazy_rag_v2.sh` | **语音助手主脚本（推荐）** | P0 |
| `rag_medical_server_simple.py` | RAG延迟加载服务 | P0 |
| `tts_service.py` | TTS常驻服务 | P0 |

### 服务类文件（P1 - 重要）

| 文件 | 说明 | 优先级 |
|------|------|--------|
| `rag_medical_service_full.py` | RAG服务核心类（被simple版调用） | P1 |
| `compress_faiss_index.py` | FAISS索引压缩工具（数据生成） | P1 |

### 测试与工具文件（P2 - 可选）

| 文件 | 说明 | 优先级 |
|------|------|--------|
| `test_rag_lazy_client.py` | RAG客户端测试工具 | P2 |
| `optimize_sbert_loading.py` | SBERT加载优化实验 | P2 |

### 构建目录（build/）

| 文件 | 说明 |
|------|------|
| `build/llm_client` | LLM客户端（C++二进制） |
| `build/model_service_daemon` | LLM常驻服务（C++二进制） |

### 部署目录（rk3588_deploy/）

| 文件 | 说明 |
|------|------|
| `rk3588_deploy/sbert_768_final.json` | 68,023条医疗对话数据 |
| `rk3588_deploy/sbert_768_final_vector.faiss` | FAISS向量索引（6.4MB） |
| `rk3588_deploy/sbert_768_final_bm25.pkl` | BM25索引（备用） |
| `rk3588_deploy/full_data/` | 完整数据集备份目录 |
| `rk3588_deploy/medical_terminology_extended.py` | 医学术语库扩展 |
| `rk3588_deploy/interactive_rag.py` | 交互式RAG测试工具 |
| `rk3588_deploy/demo_queries_filtered.py` | 演示查询过滤工具 |
| `rk3588_deploy/rk3588_server.py` | RK3588服务端（旧版） |

---

## 快速测试

### 1. 确保设备已连接

```bash
# 检查设备连接
adb devices

# 如使用网络连接
adb connect 192.168.1.100:5555
```

### 2. 一键启动所有服务

```bash
# 在RK3588上执行
adb shell '/userdata/voice_assistant/start_all_services_v2.sh'
```

输出示例：
```
========================================
    启动医疗RAG语音助手服务
========================================
[INFO] 启用Swap...
[INFO] 内存状态:
Mem:           3.8Gi       960Mi        80Mi
Swap:          2.0Gi          0B       2.0Gi
[INFO] 清理旧服务...
[INFO] 启动RAG服务（延迟加载版）...
[INFO] ✓ RAG服务已就绪
[INFO] 启动LLM服务...
[INFO] 等待LLM模型加载（约30秒）...
[INFO] ✓ LLM服务已就绪
[INFO] 启动TTS服务...
[INFO] ✓ TTS服务已就绪
========================================
    所有服务已启动
========================================
```

### 3. 运行语音助手

```bash
# 使用默认测试音频
adb shell '/userdata/voice_assistant/voice_assistant_lazy_rag_v2.sh'

# 或使用指定音频
adb shell '/userdata/voice_assistant/voice_assistant_lazy_rag_v2.sh /data/zipformer/model/test.wav'
```

---

## 版本对比

| 版本 | 特点 | 启动时间 | 首次查询 | 适用场景 |
|------|------|----------|----------|----------|
| **延迟加载版** | SBERT延迟加载 | ~0.5秒 | ~13秒 | **日常使用（推荐）** |
| 预加载版 | SBERT预加载 | ~13秒 | ~0.2秒 | 高频查询场景 |

---

## 性能指标

### 完整流程（首次查询）

| 步骤 | 耗时 | 说明 |
|------|------|------|
| 服务启动 | ~0.5s | 延迟加载版 |
| 语音识别 | ~1.4s | Zipformer推理 |
| RAG检索 | ~13s | 含SBERT模型加载(391MB) |
| LLM生成 | ~3s | InternVL3-1B推理 |
| 语音合成 | ~5s | MeloTTS推理 |
| **总耗时** | **~23秒** | 完整流程 |

### 后续查询（SBERT已加载）

| 步骤 | 耗时 | 说明 |
|------|------|------|
| 语音识别 | ~1.4s | Zipformer推理 |
| RAG检索 | ~0.2s | FAISS向量检索 |
| LLM生成 | ~3s | InternVL3-1B推理 |
| 语音合成 | ~5s | MeloTTS推理 |
| **总耗时** | **~10秒** | 完整流程 |

---

## 服务管理

### 查看服务状态

```bash
# 查看所有服务进程
adb shell 'ps | grep -E "python|model_service"'

# 查看socket文件
adb shell 'ls -la /tmp/*.sock'

# 检查内存使用
adb shell 'free -h'
```

### 查看日志

```bash
# RAG服务日志
adb shell 'tail -f /tmp/rag_lazy_server.log'

# LLM服务日志
adb shell 'tail -f /tmp/llm_service.log'

# TTS服务日志
adb shell 'tail -f /tmp/tts_service.log'
```

### 停止所有服务

```bash
adb shell 'pkill -f "rag_medical\|model_service\|tts_service"'
```

### 单独测试服务

```bash
# 测试LLM服务
adb shell '/userdata/voice_assistant/llm_client ping'
adb shell '/userdata/voice_assistant/llm_client generate "你好"'

# 测试RAG服务
adb shell 'python3 << EOF
import socket, json
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/rag_medical_lazy.sock")
sock.send(json.dumps({"action": "stats"}).encode())
print(sock.recv(1024).decode())
EOF'

# 测试TTS服务
adb shell 'python3 << EOF
import socket, json
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/tts_service.sock")
request = {"command": "synthesize", "text": "你好", "output": "/tmp/test.wav"}
sock.send(json.dumps(request).encode())
print(sock.recv(1024).decode())
EOF'
```

---

## 常见问题

### 1. RAG服务启动超时

**现象**：提示"RAG服务启动超时"

**解决**：
```bash
# 检查RAG服务日志
adb shell 'cat /tmp/rag_lazy_server.log'

# 手动启动RAG服务
adb shell 'cd /userdata/medical_rag_full && python3 rag_medical_server_simple.py &'

# 等待3秒后检查
adb shell 'ls -la /tmp/rag_medical_lazy.sock'
```

### 2. LLM服务启动超时

**现象**：提示"LLM服务启动超时"

**解决**：
```bash
# 手动启动LLM服务
adb shell 'cd /userdata/voice_assistant && export LD_LIBRARY_PATH=/data/internvl3 && ./model_service_daemon &'

# 等待30秒后测试
sleep 30
adb shell '/userdata/voice_assistant/llm_client ping'
```

### 3. TTS服务启动超时

**现象**：提示"TTS服务启动超时"

**解决**：
```bash
# 手动启动TTS服务
adb shell 'cd /userdata/voice_assistant && python3 tts_service.py &'

# 检查socket
adb shell 'ls -la /tmp/tts_service.sock'
```

### 4. 语音识别失败

**现象**：提示"语音识别失败"

**解决**：
```bash
# 检查音频文件是否存在
adb shell 'ls -la /data/zipformer/model/test.wav'

# 检查Zipformer模型
adb shell 'ls -la /data/zipformer/model/*.rknn'
```

### 5. 语音合成失败

**现象**：提示"语音合成失败"

**解决**：
```bash
# 检查MeloTTS模型
adb shell 'ls -la /userdata/melotts_deploy/model/*.rknn'

# 检查输出目录
adb shell 'ls -la /tmp/voice_assistant/'
```

### 6. 内存不足（OOM）

**现象**：服务被Kill，提示"Killed"

**解决**：
```bash
# 添加Swap（2GB）
adb shell 'dd if=/dev/zero of=/swapfile bs=1M count=2048'
adb shell 'mkswap /swapfile && swapon /swapfile'

# 验证Swap
adb shell 'free -h'
```

---

## 测试用例

### 医疗查询测试

```bash
# 测试1: 高血压
adb shell 'python3 << EOF
import socket, json
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/rag_medical_lazy.sock")
sock.send(json.dumps({"action": "search", "query": "高血压吃什么药", "k": 2}).encode())
response = json.loads(sock.recv(8192).decode())
if response["status"] == "ok":
    r = response["data"]["results"][0]
    print(f"科室: {r[chr(39)+chr(39)]department[chr(39)+chr(39)]}")
    print(f"标题: {r[chr(39)+chr(39)]title[chr(39)+chr(39)]}")
EOF'

# 预期输出:
# 科室: 内科
# 标题: 心血管内科
```

### 完整流程测试

```bash
# 运行完整测试
adb shell '/userdata/voice_assistant/voice_assistant_lazy_rag_v2.sh'

# 预期输出:
# [INFO] ✓ LLM服务正常
# [INFO] ✓ RAG服务已就绪
# [INFO] ✓ TTS服务已就绪
# [RESULT] 识别结果: ...
# [INFO] RAG检索耗时: ...ms
# [RESULT] 回复: ...
# [INFO] 语音合成完成: ...
```

---

## 数据文件

| 文件 | 路径 | 大小 | 说明 |
|------|------|------|------|
| sbert_768_full.json | /userdata/medical_rag_full/ | ~15MB | 68,023条对话数据 |
| sbert_768_full_vector.faiss | /userdata/medical_rag_full/ | ~6.4MB | FAISS压缩索引 |

---

## 模型文件

| 路径 | 大小 | 说明 |
|------|------|------|
| /root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese | ~391MB | SBERT中文模型 |
| /data/internvl3/internvl3-1b_w8a8_rk3588.rkllm | ~1.1GB | LLM模型 |
| /userdata/melotts_deploy/model/*.rknn | ~100MB | TTS模型 |
| /data/zipformer/model/*.rknn | ~50MB | ASR模型 |

---

## 参考文档

- [RAG完整版文档](../调试文档/03-系统集成/RAG版/RAG完整版文档.md)
- [问题解决汇总](../调试文档/05-问题排查/RAG完整版问题解决汇总.md)

---

## 更新记录

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-03-30 | v2.0 | 更新为延迟加载版，68,023条数据 |
| 2026-03-28 | v1.0 | 初始版本，4,649条数据 |
