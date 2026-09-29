# tts_service.py 代码详解

## 文件信息

- **路径**: `/home/ubuntu/桌面/ai/voice_assistant/tts_service.py`
- **作用**: MeloTTS 常驻服务
- **功能**: 通过 Unix Socket 提供语音合成服务

---

## 逐行详解

### 1. 导入模块

```python
import socket
import os
import sys
import json
import subprocess
import signal
import threading
```

| 模块 | 作用 |
|------|------|
| `socket` | Unix Socket 通信 |
| `os` | 文件操作、进程管理 |
| `sys` | 系统相关功能 |
| `json` | JSON 数据解析 |
| `subprocess` | 调用外部命令 |
| `signal` | 信号处理 |
| `threading` | 多线程处理 |

### 2. 服务配置

```python
SOCKET_PATH = "/tmp/tts_service.sock"
PID_FILE = "/tmp/tts_service.pid"
MELOTTS_DIR = "/userdata/melotts_deploy"
```

| 配置项 | 说明 |
|--------|------|
| `SOCKET_PATH` | Unix Socket 文件路径 |
| `PID_FILE` | PID 文件路径 |
| `MELOTTS_DIR` | MeloTTS 部署目录 |

### 3. 类初始化

```python
class TTSService:
    def __init__(self):
        self.running = False
        self.sock = None
```

| 属性 | 作用 |
|------|------|
| `running` | 服务运行状态标志 |
| `sock` | Socket 对象 |

### 4. 启动服务

```python
def start(self):
    # 清理旧的socket文件
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
```

| 代码 | 作用 |
|------|------|
| `os.path.exists()` | 检查文件是否存在 |
| `os.remove()` | 删除文件 |

```python
    # 创建socket
    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.sock.bind(SOCKET_PATH)
    self.sock.listen(5)
```

| 代码 | 作用 |
|------|------|
| `socket.AF_UNIX` | Unix 域套接字 |
| `socket.SOCK_STREAM` | TCP 流式套接字 |
| `bind()` | 绑定到文件路径 |
| `listen(5)` | 最大等待连接数5 |

```python
    # 设置权限
    os.chmod(SOCKET_PATH, 0o777)
```

**作用**: 允许所有用户访问 socket 文件。

```python
    # 接受连接
    while self.running:
        try:
            self.sock.settimeout(1.0)
            conn, addr = self.sock.accept()
            threading.Thread(target=self.handle_client, args=(conn,)).start()
```

| 代码 | 作用 |
|------|------|
| `settimeout(1.0)` | 设置1秒超时 |
| `accept()` | 接受客户端连接 |
| `threading.Thread()` | 创建新线程处理请求 |

### 5. 处理客户端请求

```python
def handle_client(self, conn):
    data = conn.recv(4096).decode()
    request = json.loads(data)
```

| 代码 | 作用 |
|------|------|
| `recv(4096)` | 接收最多4096字节数据 |
| `decode()` | 字节转字符串 |
| `json.loads()` | 解析 JSON 数据 |

```python
    if command == 'ping':
        response = {'status': 'ok', 'message': 'TTS服务运行正常'}
```

**作用**: 健康检查命令。

```python
    elif command == 'synthesize':
        text = request.get('text', '')
        output_file = request.get('output', '/tmp/tts_output.wav')
```

| 代码 | 作用 |
|------|------|
| `request.get('text', '')` | 获取text字段，默认空字符串 |
| `request.get('output', ...)` | 获取output字段，默认路径 |

### 6. 语音合成

```python
def synthesize(self, text, output_file):
    cmd = [
        f"{MELOTTS_DIR}/melotts_demo",
        "--input_text", text,
        "--encoder_model_path", f"{MELOTTS_DIR}/model/encoder-ZH_MIX_EN.rknn",
        "--decoder_model_path", f"{MELOTTS_DIR}/model/decoder-ZH_MIX_EN.rknn",
        "--output_filename", output_file,
        "--language", "ZH"
    ]
```

**作用**: 构建 melotts_demo 命令行参数。

```python
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f"{MELOTTS_DIR}/lib:{env.get('LD_LIBRARY_PATH', '')}"
```

**作用**: 设置库路径，确保能找到 RKNN 运行时库。

```python
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30
    )
```

| 参数 | 作用 |
|------|------|
| `env=env` | 使用自定义环境变量 |
| `capture_output=True` | 捕获标准输出和错误 |
| `text=True` | 以文本模式返回 |
| `timeout=30` | 30秒超时 |

```python
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return True, f"合成完成: {output_file}"
```

| 代码 | 作用 |
|------|------|
| `os.path.exists()` | 检查文件是否存在 |
| `os.path.getsize()` | 获取文件大小 |

### 7. 停止服务

```python
def stop(self):
    self.running = False
    if self.sock:
        self.sock.close()
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
```

**作用**: 清理资源，关闭 socket，删除文件。

### 8. 信号处理

```python
def signal_handler(sig, frame):
    print("\n[TTS服务] 收到停止信号", file=sys.stderr)
    service.stop()
    sys.exit(0)
    
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

| 信号 | 说明 |
|------|------|
| `SIGTERM` | 终止信号（kill 默认发送） |
| `SIGINT` | 中断信号（Ctrl+C） |

---

## 通信协议

### 请求格式

```json
{
    "command": "synthesize",
    "text": "你好医生",
    "output": "/tmp/output.wav"
}
```

### 响应格式

```json
{
    "status": "ok",
    "message": "合成完成: /tmp/output.wav",
    "output": "/tmp/output.wav"
}
```

---

## 服务架构

```
┌─────────────────┐
│   客户端脚本     │
│ (voice_assistant)│
└────────┬────────┘
         │ JSON over Unix Socket
         ▼
┌─────────────────┐
│  tts_service.py │
│  (常驻服务)      │
└────────┬────────┘
         │ subprocess
         ▼
┌─────────────────┐
│ melotts_demo    │
│ (RKNN推理)      │
└─────────────────┘
```

---

## 启动方式

```bash
# 后台启动
nohup python3 tts_service.py > /tmp/tts_service.log 2>&1 &

# 检查状态
python3 -c "
import socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/tts_service.sock')
sock.send(b'{\"command\":\"ping\"}')
print(sock.recv(1024).decode())
"
```

---

## 总结

| 特性 | 说明 |
|------|------|
| 通信方式 | Unix Socket |
| 协议格式 | JSON |
| 并发处理 | 多线程 |
| 超时机制 | 30秒 |
| 容错处理 | 文件存在性+大小检查 |
| 信号处理 | 支持优雅退出 |
