# 医疗RAG语音助手问题解决汇总

## 问题分类

- [内存问题](#内存问题)
- [服务启动问题](#服务启动问题)
- [通信问题](#通信问题)
- [模型加载问题](#模型加载问题)
- [性能问题](#性能问题)
- [其他问题](#其他问题)

---

## 内存问题

### 1. RAG服务被系统Kill (OOM)

**现象**:
```
[1]-  Killed                  python3 rag_medical_server_simple.py
```

**原因分析**:
- RK3588只有3.8GB内存
- SBERT模型加载需要~400MB内存
- 与LLM服务(~1.2GB)同时运行时内存不足

**解决方案**:
```bash
# 创建2GB Swap文件
dd if=/dev/zero of=/swapfile bs=1M count=2048
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# 验证
free -h
```

**预防措施**:
- 始终确保Swap已启用
- 监控内存使用：`watch -n 1 free -h`

---

## 服务启动问题

### 2. TTS服务启动超时

**现象**:
```
[WARN] TTS服务启动超时，将使用备用方案
```

**原因分析**:
- 脚本中检查的socket路径错误
- 实际路径: `/tmp/tts_service.sock`
- 脚本检查: `/tmp/tts.sock`

**解决方案**:
修改脚本中的socket路径：
```bash
# 错误
if [ -S "/tmp/tts.sock" ]; then
    sock.connect('/tmp/tts.sock')

# 正确
if [ -S "/tmp/tts_service.sock" ]; then
    sock.connect('/tmp/tts_service.sock')
```

### 3. LLM服务启动失败

**现象**:
```
[ERROR] LLM服务启动超时
```

**原因分析**:
- 模型加载需要约30秒
- 旧服务进程未清理
- Socket文件残留

**解决方案**:
```bash
# 清理旧服务
pkill -9 -f model_service_daemon
rm -f /tmp/voice_assistant.sock

# 重新启动
export LD_LIBRARY_PATH=/data/internvl3
./model_service_daemon &

# 等待30秒后测试
./llm_client ping
```

### 4. RAG服务启动但无法连接

**现象**:
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**原因分析**:
- Socket文件存在但服务进程已退出
- 多线程实现导致连接问题

**解决方案**:
- 使用简化版单线程服务：`rag_medical_server_simple.py`
- 检查服务进程：`ps | grep python`
- 清理残留socket：`rm -f /tmp/rag_medical*.sock`

---

## 通信问题

### 5. TTS服务通信协议错误

**现象**:
```
[ERROR] 语音合成失败
```

**原因分析**:
- TTS服务使用JSON协议
- 脚本直接发送纯文本
- 服务期望格式：`{"command": "synthesize", "text": "...", "output": "..."}`

**解决方案**:
```python
# 错误
sock.send(b'要合成的文本')

# 正确
request = {
    'command': 'synthesize',
    'text': '要合成的文本',
    'output': '/tmp/output.wav'
}
sock.send(json.dumps(request).encode())
```

### 6. Socket权限问题

**现象**:
```
Permission denied: '/tmp/rag_medical_lazy.sock'
```

**解决方案**:
```bash
# 设置权限
chmod 777 /tmp/*.sock

# 或在服务代码中设置
os.chmod(SOCKET_PATH, 0o777)
```

---

## 模型加载问题

### 7. SBERT模型下载失败

**现象**:
```
Cannot send a request, as the client has been closed
No sentence-transformers model found
```

**原因分析**:
- RK3588无网络连接
- 尝试从HuggingFace下载模型

**解决方案**:
```python
# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 使用本地模型路径
model_path = "/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots/183bb99aa7af74355fb58d16edf8c13ae7c5433e"
encoder = SentenceTransformer(model_path, device='cpu')
```

**部署步骤**:
```bash
# 在开发机下载模型
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('shibing624/text2vec-base-chinese')"

# 推送到RK3588
adb push /root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese /root/.cache/huggingface/hub/
```

### 8. FAISS索引加载失败

**现象**:
```
RuntimeError: Error in faiss::FileIOReader
```

**原因分析**:
- 索引文件损坏
- 路径错误
- 版本不兼容

**解决方案**:
```bash
# 验证文件完整性
ls -lh /userdata/medical_rag_full/*.faiss
md5sum /userdata/medical_rag_full/*.faiss

# 重新生成索引
python3 compress_faiss_index.py
```

---

## 性能问题

### 9. 首次RAG查询耗时过长

**现象**:
```
[INFO] RAG检索耗时: 13055ms
```

**原因分析**:
- SBERT模型首次加载需要~8秒
- 延迟加载策略导致首次查询慢

**解决方案**:

**方案A：接受延迟（推荐）**
- 延迟加载减少服务启动时间
- 仅首次查询慢，后续查询<200ms

**方案B：预加载模型**
```python
# 在rag_medical_server_simple.py中
class SimpleRAGService:
    def __init__(self):
        # ...
        self._load_encoder()  # 取消延迟加载
```

**方案C：预热查询**
```bash
# 服务启动后执行一次虚拟查询
python3 -c "
import socket, json
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/rag_medical_lazy.sock')
sock.send(json.dumps({'action': 'search', 'query': '高血压', 'k': 1}).encode())
sock.recv(8192)
sock.close()
"
```

### 10. 语音合成耗时过长

**现象**:
```
[INFO] 耗时: 5249ms
```

**原因分析**:
- MeloTTS模型推理较慢
- CPU资源竞争

**优化方案**:
- 使用更短的回复文本
- 优化MeloTTS参数
- 考虑使用更轻量的TTS模型

---

## 其他问题

### 11. 语音识别结果不准确

**现象**:
```
[RESULT] 识别结果: 对我做了介绍那么我想说的是...
```

**原因分析**:
- 输入音频质量问题
- Zipformer模型限制

**解决方案**:
- 使用更清晰的音频输入
- 调整音频预处理参数
- 考虑使用更大的语音识别模型

### 12. RAG检索结果不准确

**现象**:
- "高血压"返回妇产科结果
- 查询与结果不匹配

**原因分析**:
- 向量相似度被关键词主导
- 未区分近义词（高vs低）

**解决方案**:
- 添加医学术语检测
- 实现反义词冲突检测
- 使用关键词加权

```python
# 示例：反义词检测
antonym_pairs = [
    ('高血压', '低血压'),
    ('高血糖', '低血糖'),
    ('高热', '低热'),
]

for high, low in antonym_pairs:
    if high in query and low in result:
        # 降低相似度分数
        score *= 0.5
```

### 13. 脚本语法错误

**现象**:
```
line 310: syntax error near unexpected token `then'
```

**原因分析**:
- Here Document缩进问题
- 特殊字符未转义

**解决方案**:
```bash
# 错误 - 缩进导致问题
if python3 << EOF
code
EOF
then
    ...

# 正确 - 使用不同分隔符
python3 << PYEOF
code
PYEOF
if [ $? -eq 0 ]; then
    ...
```

### 14. 模块导入错误

**现象**:
```
ModuleNotFoundError: No module named 'faiss'
ModuleNotFoundError: No module named 'sentence_transformers'
```

**解决方案**:
```bash
# 在线安装
pip3 install faiss-cpu sentence-transformers numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 离线安装
pip3 install /userdata/packages/faiss_cpu-*.whl --no-index
pip3 install /userdata/packages/sentence_transformers-*.whl --no-index
```

---

## 调试技巧

### 查看服务日志

```bash
# RAG服务日志
tail -f /tmp/rag_lazy_server.log

# LLM服务日志
tail -f /tmp/llm_service.log

# TTS服务日志
tail -f /tmp/tts_service.log

# 语音助手日志
tail -f /tmp/voice_assistant.log
```

### 测试单个组件

```bash
# 测试RAG
curl -X POST unix:///tmp/rag_medical_lazy.sock \
  -d '{"action": "search", "query": "高血压", "k": 2}'

# 测试LLM
echo "测试" | ./llm_client generate

# 测试TTS
python3 -c "
import socket, json
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/tts_service.sock')
sock.send(json.dumps({'command': 'synthesize', 'text': '测试', 'output': '/tmp/test.wav'}).encode())
print(sock.recv(4096).decode())
"
```

### 监控资源使用

```bash
# 内存监控
watch -n 1 'free -h'

# CPU监控
top

# 进程监控
ps aux | grep -E "python|model_service"

# 磁盘使用
df -h
```

---

## 最佳实践

1. **始终配置Swap** - 防止OOM导致服务被Kill
2. **使用离线模型** - 避免RK3588网络依赖
3. **延迟加载策略** - 平衡启动时间和首次查询速度
4. **服务健康检查** - 不仅检查socket，还要测试实际连接
5. **日志记录** - 保留详细日志便于排查问题
6. **定期重启** - 长时间运行后建议重启释放内存

---

## 联系支持

如遇到未列出的问题，请收集以下信息：
1. 完整的错误日志
2. `free -h` 输出
3. `ps | grep -E "python|model_service"` 输出
4. `ls -la /tmp/*.sock` 输出
