# client - 客户端文件

本目录包含用于连接服务的客户端程序。

## 文件说明

### qwen3_llm_client.py
**LLM服务客户端**
- 功能: 通过Unix Socket连接LLM服务，发送请求并接收回复
- Socket: `/tmp/qwen3_llm.sock`
- 使用: `python3 qwen3_llm_client.py "你的问题"`
- 示例:
  ```bash
  python3 qwen3_llm_client.py "你好，请介绍一下自己"
  ```
- 返回: LLM生成的文本回复

## 使用场景

1. **测试LLM服务**: 快速验证LLM服务是否正常运行
2. **命令行调用**: 不需要启动完整语音助手，直接调用LLM
3. **调试**: 排查LLM相关问题

## 工作原理

```python
# 创建Unix Socket连接
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/qwen3_llm.sock')

# 发送请求
request = {'prompt': '你的问题', 'max_tokens': 150}
sock.send(json.dumps(request).encode())

# 接收回复
response = sock.recv(4096).decode()
result = json.loads(response)
print(result['text'])
```
