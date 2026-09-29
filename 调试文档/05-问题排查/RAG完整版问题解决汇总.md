# RAG完整版问题解决汇总

## 一、部署问题

### 1.1 RAG服务启动失败

**现象**: 
```
ModuleNotFoundError: No module named 'faiss'
```

**原因**: FAISS库未安装

**解决**:
```bash
# 使用pip安装
pip install faiss-cpu

# 或从源码编译安装
```

### 1.2 LLM服务启动失败

**现象**:
```
No module named 'rkllm_api'
```

**原因**: 缺少RKLLM Python API

**解决**: 使用C++版本替代
```bash
# 使用model_service_daemon (C++版本)
./model_service_daemon
```

### 1.3 TTS服务合成失败

**现象**: 输出文件大小为0字节

**原因**: tts_service_daemon二进制文件参数格式不兼容

**解决**: 创建Python版本TTS服务
```python
# tts_service.py - 调用melotts_demo的正确参数
./melotts_demo \
    --input_text "文本" \
    --encoder_model_path model/encoder-ZH_MIX_EN.rknn \
    --decoder_model_path model/decoder-ZH_MIX_EN.rknn \
    --output_filename output.wav \
    --language ZH
```

## 二、数据问题

### 2.1 检索结果不准确

**现象**: "高血压"返回妇产科结果

**原因**: 向量相似度被"血压"关键词主导，未区分"高"vs"低"

**解决**: 添加反义词冲突检测
```python
ANTONYM_PAIRS = [
    ('高血压', '低血压'),
    ('高血糖', '低血糖'),
    ('甲亢', '甲减'),
]

def detect_antonym_conflict(query, results):
    for pos, neg in ANTONYM_PAIRS:
        if pos in query:
            results = [r for r in results if neg not in r.get('question', '')]
    return results
```

### 2.2 科室分布不均

**现象**: 所有结果都是"男科"

**原因**: 数据加载或分层采样逻辑错误

**解决**: 实现分层采样
```python
def stratified_sample(df, n_samples, dept_col='department'):
    departments = df[dept_col].unique()
    samples_per_dept = n_samples // len(departments)
    
    sampled = []
    for dept in departments:
        dept_df = df[df[dept_col] == dept]
        n = min(samples_per_dept, len(dept_df))
        sampled.append(dept_df.sample(n))
    
    return pd.concat(sampled).sample(frac=1)
```

## 三、性能问题

### 3.1 RAG检索时间过长

**现象**: 检索耗时30秒以上

**原因**: 每次查询都重新加载模型

**解决**: 实现模型常驻服务
```python
class MedicalRAGService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._preload()
        return cls._instance
```

### 3.2 内存不足(OOM)

**现象**: 服务被系统杀死

**原因**: RK3588内存有限(4GB)，同时加载多个模型

**解决**: 配置Swap空间
```bash
# 创建4GB Swap
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# 添加到fstab
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### 3.3 LLM生成超时

**现象**: LLM响应时间过长(>10秒)

**原因**: 模型未使用NPU加速或batch size过大

**解决**: 
- 使用W8A8量化模型
- 减小max_token长度
- 使用模型常驻服务避免重复加载

## 四、网络问题

### 4.1 模型下载失败

**现象**: 
```
Cannot send a request, as the client has been closed
```

**原因**: RK3588无法连接HuggingFace

**解决**: 离线部署模型
```bash
# 开发环境下载
huggingface-cli download shibing624/text2vec-base-chinese

# 推送到RK3588
adb push ~/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese /userdata/models/
```

### 4.2 ADB连接不稳定

**现象**: ADB频繁断开

**解决**:
```bash
# 使用adb over TCP
adb tcpip 5555
adb connect <rk3588_ip>:5555

# 或使用USB连接时检查线缆
```

## 五、兼容性问题

### 5.1 Python版本不兼容

**现象**: 
```
SyntaxError: invalid character '✓' (U+2713)
```

**原因**: Python 3.8不支持某些Unicode字符

**解决**: 替换特殊字符
```python
# 替换前
print("✓ 成功")

# 替换后
print("[OK] 成功")
```

### 5.2 Socket连接失败

**现象**: 
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**原因**: 服务未启动或socket文件权限问题

**解决**:
```bash
# 检查socket文件
ls -la /tmp/*.sock

# 设置权限
chmod 777 /tmp/rag_medical_full.sock

# 重启服务
pkill -f rag_medical_server
python3 rag_medical_server_full_v2.py start
```

## 六、模型转换问题

### 6.1 RKNN转换失败

**现象**: 
```
TypeError: quantize_dynamic() got an unexpected keyword argument 'optimize_model'
```

**原因**: Optimum版本不兼容

**解决**:
```python
# 移除不支持的参数
quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
    # 删除 optimize_model=True
)
```

### 6.2 ONNX导出失败

**现象**: 动态轴导致导出失败

**解决**:
```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 1: 'sequence'},
        'output': {0: 'batch_size'}
    }
)
```

## 七、语音相关问题

### 7.1 语音识别失败

**现象**: 识别结果为空或不准确

**原因**: 音频格式不匹配

**解决**:
```bash
# 转换为正确格式
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f wav output.wav
```

### 7.2 TTS合成音质差

**现象**: 语音不自然或断续

**原因**: MeloTTS模型参数设置不当

**解决**:
```bash
# 调整speed参数
./melotts_demo \
    --input_text "文本" \
    --speed 0.9 \
    --output_filename output.wav
```

## 八、系统集成问题

### 8.1 服务启动顺序

**问题**: 服务间依赖导致启动失败

**解决**: 按顺序启动
```bash
#!/bin/bash
# 1. 启动LLM服务
./model_service_daemon &
sleep 30

# 2. 启动RAG服务
python3 rag_medical_server_full_v2.py start &
sleep 10

# 3. 启动TTS服务
python3 tts_service.py &
```

### 8.2 日志查看

**各服务日志位置**:
```
/tmp/rag_full_server.log    # RAG服务
/tmp/llm_service.log        # LLM服务
/tmp/tts_service.log        # TTS服务
```

**实时监控**:
```bash
tail -f /tmp/rag_full_server.log
```

## 九、调试技巧

### 9.1 单独测试组件

**测试RAG**:
```python
import socket, json
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/rag_medical_full.sock')
sock.sendall(json.dumps({'query': '测试', 'k': 2}).encode())
print(json.loads(sock.recv(8192).decode()))
```

**测试LLM**:
```bash
./llm_client ping
./llm_client generate "你好"
```

**测试TTS**:
```bash
./tts_client.py synthesize "你好" /tmp/test.wav
```

### 9.2 性能分析

**查看内存使用**:
```bash
free -h
cat /proc/$(pgrep model_service_daemon)/status | grep VmRSS
```

**查看CPU使用**:
```bash
top -p $(pgrep model_service_daemon)
```

**查看NPU使用**:
```bash
cat /sys/kernel/debug/rknpu/load
```

## 十、最佳实践

1. **服务管理**: 使用start_all_services.sh统一管理
2. **日志记录**: 所有服务重定向日志到/tmp
3. **错误处理**: 客户端添加超时和重试机制
4. **资源监控**: 定期监控内存和NPU温度
5. **数据备份**: 定期备份医疗知识库索引

## 参考文档

- [RAG完整版文档](../03-系统集成/RAG版/RAG完整版文档.md)
- [RK3588测试指南](../../voice_assistant/RK3588_TEST_GUIDE.md)
- [MeloTTS部署文档](../02-模型部署/MeloTTS/melotts部署总结.md)
