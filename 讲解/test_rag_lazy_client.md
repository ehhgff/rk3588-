# test_rag_lazy_client.py 代码详解

## 文件信息

- **路径**: `/home/ubuntu/桌面/ai/voice_assistant/test_rag_lazy_client.py`
- **作用**: RAG延迟加载服务测试客户端
- **功能**: 测试服务状态、首次查询、二次查询性能

---

## 完整代码

```python
#!/usr/bin/env python3
"""测试RAG延迟加载服务客户端"""

import socket
import json

def client_stats():
    """获取服务状态"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/tmp/rag_medical_lazy.sock")
    
    request = {'action': 'stats'}
    sock.send(json.dumps(request).encode('utf-8'))
    
    response = sock.recv(4096).decode('utf-8')
    sock.close()
    
    return json.loads(response)

def client_search(query, k=5):
    """搜索"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/tmp/rag_medical_lazy.sock")
    
    request = {'action': 'search', 'query': query, 'k': k}
    sock.send(json.dumps(request).encode('utf-8'))
    
    response = sock.recv(8192).decode('utf-8')
    sock.close()
    
    return json.loads(response)

if __name__ == '__main__':
    print("=== 测试RAG延迟加载服务 ===")
    
    # 获取状态
    print("\n[1/3] 获取服务状态...")
    result = client_stats()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 首次查询
    print("\n[2/3] 首次查询（加载SBERT）...")
    result = client_search('高血压吃什么药', k=3)
    if result['status'] == 'ok':
        data = result['data']
        print(f"耗时: {data['time_ms']:.1f}ms")
        print(f"编码器加载: {data['encoder_load_time']:.1f}s")
        if data['results']:
            print(f"结果: [{data['results'][0]['department']}] {data['results'][0]['title'][:30]}...")
    
    # 第二次查询
    print("\n[3/3] 第二次查询（已加载）...")
    result = client_search('感冒发烧', k=3)
    if result['status'] == 'ok':
        data = result['data']
        print(f"耗时: {data['time_ms']:.1f}ms")
        if data['results']:
            print(f"结果: [{data['results'][0]['department']}] {data['results'][0]['title'][:30]}...")
    
    print("\n=== 测试完成 ===")
```

---

## 逐行详解

### 1. 导入模块

```python
import socket
import json
```

| 模块 | 作用 |
|------|------|
| `socket` | Unix Socket 通信 |
| `json` | JSON 数据解析 |

### 2. 获取服务状态函数

```python
def client_stats():
    """获取服务状态"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/tmp/rag_medical_lazy.sock")
```

| 代码 | 作用 |
|------|------|
| `socket.AF_UNIX` | Unix 域套接字 |
| `socket.SOCK_STREAM` | TCP 流式套接字 |
| `connect("/tmp/rag_medical_lazy.sock")` | 连接到 RAG 服务 |

```python
    request = {'action': 'stats'}
    sock.send(json.dumps(request).encode('utf-8'))
```

| 代码 | 作用 |
|------|------|
| `{'action': 'stats'}` | 构建状态查询请求 |
| `json.dumps()` | 将字典转为 JSON 字符串 |
| `.encode('utf-8')` | 编码为字节串 |
| `sock.send()` | 发送请求数据 |

```python
    response = sock.recv(4096).decode('utf-8')
    sock.close()
    
    return json.loads(response)
```

| 代码 | 作用 |
|------|------|
| `sock.recv(4096)` | 接收最多4096字节响应 |
| `.decode('utf-8')` | 字节串转字符串 |
| `sock.close()` | 关闭连接 |
| `json.loads()` | 解析 JSON 响应 |

### 3. 搜索函数

```python
def client_search(query, k=5):
    """搜索"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/tmp/rag_medical_lazy.sock")
```

| 参数 | 作用 |
|------|------|
| `query` | 搜索查询文本 |
| `k=5` | 返回结果数量，默认5条 |

```python
    request = {'action': 'search', 'query': query, 'k': k}
    sock.send(json.dumps(request).encode('utf-8'))
```

| 请求字段 | 作用 |
|----------|------|
| `action: 'search'` | 指定搜索操作 |
| `query` | 搜索内容 |
| `k` | 返回结果数量 |

```python
    response = sock.recv(8192).decode('utf-8')
    sock.close()
    
    return json.loads(response)
```

**注意**: 搜索响应缓冲区更大（8192字节），因为包含更多数据。

### 4. 主测试流程

```python
if __name__ == '__main__':
    print("=== 测试RAG延迟加载服务 ===")
```

**作用**: 确保代码只在直接运行时执行。

```python
    # 获取状态
    print("\n[1/3] 获取服务状态...")
    result = client_stats()
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

| 参数 | 作用 |
|------|------|
| `indent=2` | 缩进2个空格 |
| `ensure_ascii=False` | 允许中文字符显示 |

```python
    # 首次查询
    print("\n[2/3] 首次查询（加载SBERT）...")
    result = client_search('高血压吃什么药', k=3)
```

**测试目的**: 验证首次查询时的 SBERT 模型加载性能。

```python
    if result['status'] == 'ok':
        data = result['data']
        print(f"耗时: {data['time_ms']:.1f}ms")
        print(f"编码器加载: {data['encoder_load_time']:.1f}s")
```

| 字段 | 作用 |
|------|------|
| `time_ms` | 总查询耗时（毫秒） |
| `encoder_load_time` | SBERT 编码器加载时间（秒） |

```python
        if data['results']:
            print(f"结果: [{data['results'][0]['department']}] {data['results'][0]['title'][:30]}...")
```

| 字段 | 作用 |
|------|------|
| `department` | 科室信息 |
| `title[:30]` | 标题前30个字符 |

```python
    # 第二次查询
    print("\n[3/3] 第二次查询（已加载）...")
    result = client_search('感冒发烧', k=3)
```

**测试目的**: 验证模型已加载后的查询性能。

---

## 通信协议

### 请求格式

```json
{
    "action": "search",
    "query": "高血压吃什么药",
    "k": 3
}
```

### 响应格式

```json
{
    "status": "ok",
    "data": {
        "time_ms": 1560.2,
        "encoder_load_time": 1.3,
        "encoder_loaded": true,
        "results": [
            {
                "department": "内科",
                "title": "高血压患者吃什么药比较好",
                "answer": "建议服用降压药..."
            }
        ]
    }
}
```

---

## 测试场景分析

### 场景1：服务状态检查
```python
client_stats()
```
**目的**: 验证服务是否正常运行

### 场景2：首次查询（冷启动）
```python
client_search('高血压吃什么药', k=3)
```
**目的**: 测试 SBERT 模型加载时间
**预期**: 总耗时 = 模型加载时间 + 查询时间

### 场景3：二次查询（热启动）
```python
client_search('感冒发烧', k=3)
```
**目的**: 测试模型已加载后的性能
**预期**: 总耗时 ≈ 查询时间（无模型加载）

---

## 延迟加载效果对比

| 查询类型 | 模型加载 | 总耗时 | 性能提升 |
|----------|----------|--------|----------|
| 首次查询 | ✅ 需要加载 | 1.5-2秒 | - |
| 后续查询 | ❌ 已加载 | 0.1-0.3秒 | **5-10倍** |

---

## 运行示例

```bash
python3 test_rag_lazy_client.py
```

**输出示例**:
```
=== 测试RAG延迟加载服务 ===

[1/3] 获取服务状态...
{
  "status": "ok",
  "data": {
    "encoder_loaded": false,
    "startup_time": 0.5
  }
}

[2/3] 首次查询（加载SBERT）...
耗时: 1560.2ms
编码器加载: 1.3s
结果: [内科] 高血压患者吃什么药比较好...

[3/3] 第二次查询（已加载）...
耗时: 120.5ms
结果: [内科] 感冒发烧应该吃什么药...

=== 测试完成 ===
```

---

## 总结

| 特性 | 说明 |
|------|------|
| 测试目的 | 验证延迟加载性能优势 |
| 通信方式 | Unix Socket + JSON |
| 关键指标 | 模型加载时间、查询耗时 |
| 测试场景 | 冷启动 vs 热启动对比 |
| 性能提升 | 5-10倍速度提升 |

这个测试客户端完美展示了 RAG 延迟加载服务的核心优势：**快速启动 + 按需加载**！