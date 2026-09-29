# 语音助手启动配置说明

本文档说明语音助手系统的启动流程、依赖服务配置及部署方式。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                   语音助手主程序                              │
│              streaming_vad.py (主循环)                       │
│  ┌──────────┬──────────┬──────────┬──────────────────┐      │
│  │ ASR      │ SER      │ RAG      │ LLM              │ TTS  │
│  │ Paraformer│ SenseVoice│ 医疗RAG  │ Qwen3-0.6B      │Matcha│
│  │ sherpa-  │ RKNN NPU │ 相似度   │ RKLLM NPU       │ONNX  │
│  │ onnx-alsa│ socket   │ 匹配     │ socket           │socket│
│  └──────────┴──────────┴──────────┴──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 服务通信方式

所有依赖服务均通过 **Unix Socket** 与本机通信：

| 服务 | Socket 路径 | 协议 |
|------|-------------|------|
| Patient RAG | `/tmp/patient_rag.sock` | JSON over Unix Socket |
| Qwen3 LLM | `/tmp/qwen3_llm.sock` | JSON over Unix Socket |
| Matcha-TTS | `/tmp/matcha_tts.sock` | JSON over Unix Socket |
| SenseVoice SER | `/tmp/sensevoice_server.sock` | JSON over Unix Socket |

---

## 启动脚本

### 1. 主启动脚本

**文件**：`/userdata/voice_assistant/run.sh`

**用途**：开发板上的主要启动入口，启动所有依赖服务后运行主程序。

**用法**：

```bash
./run.sh                    # 正常启动
./run.sh --verbose          # 显示详细环境诊断信息
./run.sh --help             # 显示帮助
./run.sh -- <python_args>   # 传递参数给 streaming_vad.py
```

**启动流程**：

```
run.sh
  ├─ 1. 设置 LD_LIBRARY_PATH → sherpa-onnx bundled libstdc++
  ├─ 2. 启动 Patient RAG 服务 (patient_rag_server.py)
  ├─ 3. 启动 Qwen3 LLM 服务 (qwen3_llm_streaming_service.py)
  ├─ 4. 启动 Matcha-TTS 服务 (matcha_tts_service.py)
  ├─ 5. 启动 SenseVoice SER 服务 (sensevoice_server.py)
  ├─ 6. 初始化音频设备 (amixer set Master 40%)
  ├─ 7. 显示诊断信息（--verbose 模式）
  └─ 8. exec streaming_vad.py（主程序）
```

### 2. 服务管理脚本

**文件**：`/userdata/voice_assistant/scripts/start_services.sh`

**用途**：独立管理所有依赖服务的启停。

**用法**：

```bash
./start_services.sh              # 启动所有服务（默认）
./start_services.sh start        # 启动所有服务
./start_services.sh stop         # 停止所有服务
./start_services.sh status       # 查看服务状态
./start_services.sh restart      # 重启所有服务
```

### 3. systemd 启动脚本

**文件**：`/userdata/voice_assistant/deploy/voice-assistant-start.sh`

**用途**：由 systemd 在系统启动时调用，以守护进程方式运行。

**特点**：
- 并行启动所有服务（后台等待就绪）
- 记录详细日志（`logger -t voice-assistant`）
- 监控服务进程健康状态
- 自动初始化音频设备

---

## 服务配置详情

### Patient RAG 服务

| 项目 | 配置 |
|------|------|
| 脚本路径 | `/userdata/voice_assistant/patient_rag_server.py` |
| 工作目录 | `/userdata/medical_rag_full` |
| Socket | `/tmp/patient_rag.sock` |
| 启动超时 | 120 秒 |
| 日志文件 | `/tmp/patient_rag_server.log` |

### Qwen3 LLM 服务

| 项目 | 配置 |
|------|------|
| 脚本路径 | `/userdata/voice_assistant/qwen3_llm_streaming_service.py` |
| 工作目录 | `/userdata/voice_assistant` |
| Socket | `/tmp/qwen3_llm.sock` |
| 启动超时 | 120 秒 |
| 环境变量 | `LD_LIBRARY_PATH=/data/qwen3:$LD_LIBRARY_PATH` |
| 日志文件 | `/tmp/qwen3.log` |

### Matcha-TTS 服务

| 项目 | 配置 |
|------|------|
| 脚本路径 | `/userdata/voice_assistant/matcha_tts_service.py` |
| 工作目录 | `/userdata/voice_assistant` |
| Socket | `/tmp/matcha_tts.sock` |
| 启动超时 | 120 秒 |
| 模型文件 | `/data/matcha_tts/model-steps-3.onnx` |
| 声码器 | `/data/matcha_tts/matcha_vocoder.onnx` |
| 词典 | `/data/matcha_tts/lexicon.txt` |
| Token 映射 | `/data/matcha_tts/tokens.txt` |
| 日志文件 | `/tmp/matcha_tts_service.log` |

### SenseVoice SER 服务

| 项目 | 配置 |
|------|------|
| 脚本路径 | `/data/sensevoice/sensevoice_server.py` |
| 工作目录 | `/data/sensevoice` |
| Socket | `/tmp/sensevoice_server.sock` |
| 启动超时 | 120 秒 |
| 模型参数 | `--model /data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn --socket /tmp/sensevoice_server.sock` |
| 日志文件 | `/tmp/sensevoice_server.log` |

---

## 启动顺序与依赖

```
服务启动顺序（无依赖并行启动）：
  ┌─ RAG ──── 无依赖，独立启动
  ├─ LLM ──── 无依赖，独立启动
  ├─ TTS ──── 无依赖，独立启动
  └─ SER ──── 无依赖，独立启动

主程序启动条件：
  └─ 所有服务启动完成（超时 120 秒，失败仅警告不阻塞）
```

**服务启动失败处理**：
- 任一服务启动失败，主程序仍会启动
- 启动失败的服务功能在运行时不可用
- 日志中会记录失败信息

---

## 音频配置

| 项目 | 配置 |
|------|------|
| 音量 | 40%（`amixer set Master 40%`） |
| 输入设备 | USB 麦克风（`plughw:1,0`） |
| 输出设备 | USB 音箱（默认 ALSA 输出） |
| 采样率 | 16000 Hz（ASR）/ 22050 Hz（TTS） |

---

## 部署说明

### 首次部署

```bash
# 1. 将项目文件复制到开发板
adb push voice_assistant/ /userdata/

# 2. 部署 SenseVoice SER 模型
adb push sensevoice/ /data/

# 3. 部署 Matcha-TTS 模型
adb push matcha_tts/ /data/

# 4. 部署 Paraformer 模型
adb push paraformer/ /data/

# 5. 设置执行权限
adb shell "chmod +x /userdata/voice_assistant/run.sh"
adb shell "chmod +x /userdata/voice_assistant/scripts/start_services.sh"
```

### 手动启动

```bash
# 开发板上执行
cd /userdata/voice_assistant
./run.sh
```

### 开机自启（systemd）

```bash
# 安装 systemd 服务
cp /userdata/voice_assistant/deploy/voice-assistant-start.sh /usr/local/bin/
chmod +x /usr/local/bin/voice-assistant-start.sh

# 创建 systemd service 文件
cat > /etc/systemd/system/voice-assistant.service << 'EOF'
[Unit]
Description=Voice Assistant Service
After=network.target

[Service]
Type=forking
ExecStart=/usr/local/bin/voice-assistant-start.sh
ExecStop=/usr/local/bin/voice-assistant-stop.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动
systemctl enable voice-assistant
systemctl start voice-assistant
```

---

## 日志文件

| 日志文件 | 对应服务 |
|----------|----------|
| `/tmp/patient_rag_server.log` | Patient RAG |
| `/tmp/qwen3.log` | Qwen3 LLM |
| `/tmp/matcha_tts_service.log` | Matcha-TTS |
| `/tmp/sensevoice_server.log` | SenseVoice SER |
| `/tmp/streaming_vad.log` | 主程序 |

查看实时日志：

```bash
tail -f /tmp/matcha_tts_service.log
```

---

## 故障排查

### 服务无法启动

```bash
# 1. 检查服务状态
./start_services.sh status

# 2. 查看对应日志
cat /tmp/sensevoice_server.log

# 3. 手动启动测试
python3 /data/sensevoice/sensevoice_server.py \
  --model /data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn \
  --socket /tmp/sensevoice_server.sock
```

### Socket 连接失败

```bash
# 检查 socket 文件是否存在
ls -la /tmp/*.sock

# 测试 socket 连通性
python3 -c "
import socket
s = socket.socket(socket.AF_UNIX)
s.settimeout(2)
s.connect('/tmp/matcha_tts.sock')
print('OK')
"
```

### 音频问题

```bash
# 检查音频设备
aplay -l
arecord -l

# 测试录音
arecord -d 3 -f S16_LE -r 16000 test.wav

# 测试播放
aplay test.wav

# 调整音量
amixer set Master 40%
```

### 进程管理

```bash
# 查看所有语音助手进程
ps aux | grep -E "streaming_vad|sensevoice|matcha_tts|qwen3_llm|patient_rag"

# 强制清理所有服务
pkill -9 -f "sensevoice_server|matcha_tts_service|qwen3_llm|patient_rag"
rm -f /tmp/*.sock
```