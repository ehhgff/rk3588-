# voice_assistant_lazy_rag_v2.sh 代码详解

## 文件信息

- **路径**: `/home/ubuntu/桌面/ai/voice_assistant/voice_assistant_lazy_rag_v2.sh`
- **作用**: 医疗RAG版语音助手主脚本
- **功能**: 语音识别 → RAG检索 → LLM生成 → 语音合成
---

## 逐行详解

### 1. 脚本头部

```bash
#!/bin/bash
# 医疗RAG版语音助手 - 使用延迟加载版RAG服务 (修复版)
# 快速启动 + InternVL3-1B模型

set -e
```

**解释**:
- `#!/bin/bash`: 使用Bash解释器
- `set -e`: 遇到错误立即退出

---

### 2. 配置路径

```bash
ZIPFORMER_DIR="/data/zipformer"
INTERNVL_DIR="/data/internvl3"
VOICE_ASSISTANT_DIR="/userdata/voice_assistant"
MEDICAL_RAG_DIR="/userdata/medical_rag_full"
WORK_DIR="/tmp/voice_assistant"
```

**解释**: 定义各组件的安装路径，方便后续引用和修改。

---

### 3. 模型路径

```bash
ENCODER_MODEL="$ZIPFORMER_DIR/model/encoder-epoch-99-avg-1.rknn"
DECODER_MODEL="$ZIPFORMER_DIR/model/decoder-epoch-99-avg-1.rknn"
JOINER_MODEL="$ZIPFORMER_DIR/model/joiner-epoch-99-avg-1.rknn"
```

**解释**: Zipformer语音识别模型包含三个组件：编码器、解码器、连接器。

---

### 4. 工作文件

```bash
INPUT_AUDIO="${1:-$ZIPFORMER_DIR/model/test.wav}"
OUTPUT_AUDIO="$WORK_DIR/output.wav"
```

**解释**:
- `${1:-默认值}`: 如果第一个参数$1未定义，使用默认值
- 输入音频默认为test.wav，也可通过命令行参数指定

---

### 5. 创建目录

```bash
mkdir -p $WORK_DIR
```

**解释**: `-p`表示递归创建，如果目录已存在不报错。

---

### 6. 颜色输出定义

```bash
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
```

**解释**: ANSI颜色转义码，用于彩色日志输出。

---

### 7. 日志函数

```bash
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}
```

**解释**: 封装带颜色的日志输出函数。

---

### 8. 启动LLM服务

```bash
start_llm_service() {
    log_info "检查LLM服务..."
```

**功能**: 检查并启动LLM常驻服务。

```bash
    if [ -S "/tmp/voice_assistant.sock" ]; then
        if ./llm_client ping > /dev/null 2>&1; then
            log_info "✓ LLM服务正常"
            return 0
        fi
    fi
```

**解释**: 双重检查：socket存在 + ping命令成功。

```bash
    local start_time=$(date +%s%N)
```

**解释**: `local`定义局部变量，`date +%s%N`获取纳秒级时间戳。

```bash
    pkill -9 -f model_service_daemon 2>/dev/null || true
    rm -f /tmp/voice_assistant.sock
    sleep 1
```

**解释**: 清理旧服务，确保干净启动。

```bash
    export LD_LIBRARY_PATH=/data/internvl3
    nohup ./model_service_daemon > /tmp/llm_service.log 2>&1 &
```

**解释**: 设置库路径，后台启动LLM服务。

```bash
    for i in {1..60}; do
        if [ -S "/tmp/voice_assistant.sock" ]; then
            sleep 2
            if ./llm_client ping > /dev/null 2>&1; then
                local end_time=$(date +%s%N)
                local duration=$(( (end_time - start_time) / 1000000 ))
                log_info "✓ LLM服务已就绪，启动耗时: ${duration}ms"
                return 0
            fi
        fi
        sleep 1
    done
```

**解释**: 轮询等待最多60秒，计算启动耗时（纳秒转毫秒）。

---

### 9. 启动RAG服务

```bash
start_rag_service_lazy() {
    log_info "检查RAG服务..."
```

**功能**: 检查并启动RAG服务。

```bash
    if python3 -c "
import socket
import json
import sys
try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect('/tmp/rag_medical_lazy.sock')
    sock.send(json.dumps({'action': 'ping'}).encode())
    response = sock.recv(1024).decode()
    sock.close()
    data = json.loads(response)
    if data.get('status') == 'ok':
        sys.exit(0)
except:
    pass
sys.exit(1)
" 2>/dev/null; then
```

**解释**: 内嵌Python代码测试RAG服务，发送ping命令验证。

---

### 10. RAG检索函数

```bash
rag_retrieve_lazy() {
    local query="$1"
```

**功能**: 调用RAG服务进行医疗知识检索。

```bash
    python3 << EOF
import socket
import json
import sys

try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect('/tmp/rag_medical_lazy.sock')
    
    request = {'action': 'search', 'query': '''$query''', 'k': 2}
    sock.send(json.dumps(request).encode('utf-8'))
    
    response = sock.recv(8192).decode('utf-8')
    data = json.loads(response)
```

**解释**: 通过Unix Socket发送查询请求，接收JSON响应。

```bash
    if data.get('status') == 'ok':
        results = data['data']['results']
        time_ms = data['data']['time_ms']
        encoder_loaded = data['data']['encoder_loaded']
```

**解释**: 解析响应数据，获取检索结果、耗时、编码器状态。

```bash
        prompt = f"""你是专业的医疗助手...

用户问题：$query

参考医疗知识：
"""
        for i, r in enumerate(results[:2], 1):
            prompt += f"\n参考{i}:\n"
            prompt += f"科室：{r['department']}\n"
            prompt += f"问题：{r['title']}\n"
            prompt += f"答案：{r['answer'][:300]}\n"
```

**解释**: 构建LLM提示词，整合检索到的医疗知识。

---

### 11. TTS合成函数

```bash
tts_synthesize() {
    local text="$1"
    local output="$2"
```

 $1, $2, $3... 函数的第1、2、3个参数 
 $@ 所有参数 
 $# 参数个数 
 local 声明局部变量 
 "$1" 带引号，防止空格问题


**功能**: 调用TTS服务合成语音。

```bash
    if [ $? -eq 0 ] && [ -f "$output" ] && [ -s "$output" ]; then
        return 0
    fi
```


代码                        含义
 [ $? -eq 0 ]         上一个命令的退出码是否为0（成功） 
 &&                   逻辑与，所有条件都满足才为真 
 [ -f "$output" ]     文件是否存在 
 [ -s "$output" ]     文件是否存在且非空（大小>0）



**解释**: 三重检查：命令成功 + 文件存在 + 文件非空。

```bash
    log_warn "TTS服务调用失败，使用备用方案"
    return 1
```

**解释**: TTS失败不退出，允许流程继续。

---

### 12. 主流程

```bash
main() {
    echo "========================================"
    echo "    医疗RAG版语音助手系统"
    echo "========================================"
    echo ""
```

**功能**: 主函数，编排整个流程。

```bash
    start_llm_service || exit 1
    start_rag_service_lazy || exit 1
    start_tts_service
```

**解释**: 启动三个服务，LLM和RAG失败则退出，TTS失败继续。

```bash
    total_start=$(date +%s%N)
```

**解释**: 记录总开始时间。

#### 步骤1: 语音识别

```bash
    log_step "步骤1/4: 语音识别"
    step1_start=$(date +%s%N)
    
    cd $ZIPFORMER_DIR
    recognized_text=$(./rknn_zipformer_demo \
        $ENCODER_MODEL $DECODER_MODEL $JOINER_MODEL $INPUT_AUDIO 2>/dev/null | \
        grep "Zipformer output:" | sed 's/Zipformer output: //')
```

| 符号 | 作用 |
|------|------|
| `$(...)` | 命令替换，获取命令输出 |
| `\` | 行续符，命令跨多行 |
| `\|` | 管道，传递数据 |
| `2>/dev/null` | 重定向错误输出到黑洞 |
| `grep "Zipformer output:"` | 过滤文本 |
| `sed 's/旧/新/'` | 替换文本 |

**解释**: 调用Zipformer模型进行语音识别，提取输出文本。

```bash
    step1_end=$(date +%s%N)
    step1_time=$(( (step1_end - step1_start) / 1000000 ))
```

**解释**: 计算步骤耗时（纳秒转毫秒）。

#### 步骤2: RAG检索

```bash
    log_step "步骤2/4: 医疗RAG检索 (68,023条知识库)"
    step2_start=$(date +%s%N)
    
    prompt=$(rag_retrieve_lazy "$recognized_text" 2>&1)
```

| 符号 | 作用 |
|------|------|
| `$(...)` | 命令替换，获取函数输出 |
| `2>&1` | 将标准错误重定向到标准输出 |
| `"$recognized_text"` | 传递语音识别结果作为查询 |

**解释**: 调用RAG检索函数，获取增强提示词。

#### 步骤3: LLM生成

```bash
    log_step "步骤3/4: LLM生成回复 (InternVL3-1B)"
    step3_start=$(date +%s%N)
    
    reply_text=$(./llm_client generate "$prompt" 2>/dev/null | grep "回复:" | sed 's/回复: //')
```

**解释**: 调用LLM服务生成回复。

```bash
    if [ -z "$reply_text" ]; then
        log_warn "LLM生成失败，使用RAG检索结果直接回复"
        reply_text=$(echo "$prompt" | grep -A 3 "参考1" | grep "答案：" | head -1 | sed 's/答案：//' | cut -c1-150)
    fi
```

| 符号 | 作用 |
|------|------|
| `[ -z "$var" ]` | 检查变量是否为空 |
| `grep -A 3` | 匹配行及后3行 |
| `head -1` | 取第一行 |
| `sed 's/旧/新/'` | 替换文本 |
| `cut -c1-150` | 截取前150个字符 |

**解释**: 容错处理：LLM失败时直接使用RAG结果。

#### 步骤4: 语音合成

```bash
    log_step "步骤4/4: 语音合成"
    step4_start=$(date +%s%N)
    
    if tts_synthesize "$reply_text" $OUTPUT_AUDIO; then
        log_info "语音合成完成: $OUTPUT_AUDIO"
    else
        log_error "语音合成失败"
    fi
```

**解释**: 调用TTS合成语音，失败不影响流程。

#### 总耗时统计

```bash
    total_end=$(date +%s%N)
    total_time=$(( (total_end - total_start) / 1000000 ))
    
    echo "========================================"
    echo "    总耗时: ${total_time}ms"
    echo "========================================"
```

**解释**: 计算并显示总耗时。

---

## 执行流程图

```
开始
  │
  ▼
启动服务 ──→ LLM服务 → RAG服务 → TTS服务
  │
  ▼
步骤1/4 ──→ 语音识别 (Zipformer)
  │
  ▼
步骤2/4 ──→ RAG检索 (68,023条知识)
  │
  ▼
步骤3/4 ──→ LLM生成 (InternVL3-1B)
  │
  ▼
步骤4/4 ──→ 语音合成 (MeloTTS)
  │
  ▼
显示总耗时
  │
  ▼
完成
```

---

## 关键设计要点

### 1. 服务容错
- LLM/RAG失败：脚本退出
- TTS失败：继续执行，不影响主流程

### 2. 性能计时
- 每个步骤单独计时
- 纳秒级精度，毫秒显示

### 3. 错误恢复
- LLM失败：使用RAG结果直接回复
- 多重检查确保服务可用

### 4. 延迟加载
- RAG服务快速启动（0.5秒）
- SBERT模型首次查询时加载

---

## 故障排查

### 问题1: 语音识别失败
```bash
# 检查模型文件
ls -la /data/zipformer/model/*.rknn

# 手动测试
./rknn_zipformer_demo encoder.rknn decoder.rknn joiner.rknn test.wav
```

### 问题2: RAG检索失败
```bash
# 检查RAG服务
cat /tmp/rag_lazy_server.log

# 手动测试
python3 -c "import socket; sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); sock.connect('/tmp/rag_medical_lazy.sock')"
```

### 问题3: LLM生成失败
```bash
# 检查LLM服务
./llm_client ping

# 查看日志
cat /tmp/llm_service.log
```
