# RK3588 医疗 RAG HTTP 服务

## 概述

`rk3588_server.py` 是运行在 RK3588 开发板上的 HTTP API 服务，提供医疗 RAG 查询接口，支持通过 ADB 反向代理从开发机访问。

## 架构设计

```
┌─────────────────────────────────────────────────────┐
│                   RK3588 开发板                       │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │           rk3588_server.py                   │   │
│  │         (HTTP Server :8081)                  │   │
│  └─────────────────┬───────────────────────────┘   │
│                    │                                │
│  ┌─────────────────▼───────────────────────────┐   │
│  │        HybridMedicalRAGv2                    │   │
│  │    (RKNN NPU 加速 + FAISS 检索)              │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
         │
         │ ADB 反向代理: adb reverse tcp:8081 tcp:8081
         ▼
┌─────────────────────────────────────────────────────┐
│              开发机 (虚拟机)                         │
│           localhost:8081                            │
└─────────────────────────────────────────────────────┘
```

## 核心代码解析

### 1. 导入和全局变量

```python
import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_medical_rag_v2 import HybridMedicalRAGv2

rag = None  # 全局RAG实例
```

| 代码 | 说明 |
|------|------|
| `HTTPServer` | Python 内置 HTTP 服务器 |
| `BaseHTTPRequestHandler` | 请求处理器基类 |
| `sys.path.insert(0, ...)` | 添加当前目录到模块搜索路径 |
| `rag = None` | 全局变量存储 RAG 实例，避免每次请求重新加载 |

### 2. RAGHandler 类 - 请求处理器

```python
class RAGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {args[0]}")
```

| 方法 | 说明 |
|------|------|
| `log_message` | 重写日志方法，添加时间戳格式化 |

### 3. GET 请求处理

```python
def do_GET(self):
    if self.path == '/':
        # 首页 - 返回 HTML 表单
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        html = '''<html>...</html>'''
        self.wfile.write(html.encode('utf-8'))
        
    elif self.path == '/health':
        # 健康检查 - 返回 JSON 状态
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            'status': 'ok',
            'device': 'RK3588',
            'acceleration': 'RKNN NPU',
            'data_count': len(rag.dialogues) if rag else 0,
            'embedding_dim': rag.embedding_dim if rag else 0
        }
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode())
```

| 路径 | 功能 | 返回格式 |
|------|------|----------|
| `/` | 首页 | HTML 表单 |
| `/health` | 健康检查 | JSON 状态信息 |

### 4. POST 请求处理

```python
def do_POST(self):
    if self.path == '/query':
        # 读取请求体
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        # 解析查询 - 支持 JSON 和表单两种格式
        try:
            data = json.loads(post_data)
            query = data.get('query', '')
        except:
            # 表单格式: query=xxx
            params = {}
            for param in post_data.split('&'):
                if '=' in param:
                    k, v = param.split('=', 1)
                    params[k] = v
            query = params.get('query', '')
        
        # 执行查询
        start = time.time()
        results, predicted = rag.search(query, top_k=3, final_k=3, use_intent=True)
        elapsed = (time.time() - start) * 1000
        
        # 构建响应
        response = {
            'query': query,
            'predicted_department': predicted,
            'latency_ms': round(elapsed, 2),
            'results': [...]
        }
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode())
```

| 步骤 | 代码 | 说明 |
|------|------|------|
| 1 | `self.rfile.read()` | 读取请求体数据 |
| 2 | `json.loads()` | 尝试 JSON 解析 |
| 3 | `post_data.split('&')` | 回退到表单解析 |
| 4 | `rag.search()` | 执行 RAG 检索 |
| 5 | `json.dumps()` | 返回 JSON 响应 |

### 5. 服务启动

```python
def main():
    global rag
    
    # 加载 RAG 索引
    rag = HybridMedicalRAGv2(
        embedding_dim=768,
        index_path="/userdata/medical_rag/sbert_768_final"
    )
    rag.load()
    
    # 启动 HTTP 服务
    port = 8081
    server = HTTPServer(('127.0.0.1', port), RAGHandler)
    server.serve_forever()
```

| 参数 | 值 | 说明 |
|------|------|------|
| `embedding_dim` | 768 | 向量维度 |
| `index_path` | /userdata/medical_rag/... | 索引文件路径 |
| `port` | 8081 | 监听端口 |
| `127.0.0.1` | 本地地址 | 仅本机访问 |

## API 接口详情

### POST /query

**请求示例**：
```bash
curl -X POST http://localhost:8081/query \
  -H "Content-Type: application/json" \
  -d '{"query": "高血压怎么办"}'
```

**响应示例**：
```json
{
  "query": "高血压怎么办",
  "predicted_department": "内科",
  "latency_ms": 250.5,
  "results": [
    {
      "department": "内科",
      "title": "高血压",
      "similarity": 0.85,
      "source": "vector"
    }
  ]
}
```

### GET /health

**响应示例**：
```json
{
  "status": "ok",
  "device": "RK3588",
  "acceleration": "RKNN NPU",
  "data_count": 68023,
  "embedding_dim": 768
}
```

## ADB 反向代理

```bash
# 在开发机上执行
adb reverse tcp:8081 tcp:8081

# 然后可以访问
curl http://localhost:8081/health
```

## 运行方式

```bash
# 在 RK3588 上运行
python3 rk3588_server.py

# 输出
======================================================================
RK3588医疗RAG HTTP服务
======================================================================

加载RAG索引...
✓ 加载完成
  数据量: 68023
  维度: 768

✓ HTTP服务已启动
  地址: http://127.0.0.1:8081
```
