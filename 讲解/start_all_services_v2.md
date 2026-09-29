# start_all_services_v2.sh 代码详解

## 文件信息

- **路径**: `/home/ubuntu/桌面/ai/voice_assistant/start_all_services_v2.sh`
- **作用**: 一键启动医疗RAG语音助手的所有服务
- **功能**: Swap设置、服务启动、状态检查

---

## 逐行详解

### 1. 脚本头部

```bash
#!/bin/bash
```
**解释**: 指定使用Bash解释器执行此脚本。

```bash
# 一键启动所有服务（优化版）
# 包含Swap设置、服务启动和状态检查
```
**解释**: 注释说明脚本功能。

```bash
set -e
```
**解释**: **关键设置**。遇到错误立即退出。如果任何命令返回非零状态，脚本立即停止。
**为什么**: 防止错误继续传播，确保问题被及时发现。

---

### 2. 欢迎信息

```bash
echo "========================================"
echo "    启动医疗RAG语音助手服务"
echo "========================================"
echo ""
```
**解释**: 打印视觉分隔线和标题，让用户知道脚本正在运行。

---

### 3. 日志函数

```bash
log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}
```
**逐部分解释**:
- `log_info()` - 定义名为log_info的函数
- `echo -e` - 启用转义字符解释
- `\033[0;32m` - ANSI颜色码，绿色
- `[INFO]` - 日志级别标签
- `\033[0m` - 重置颜色
- `$1` - 函数的第一个参数

```bash
log_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $1"
}
```
**解释**: 黄色警告日志。`\033[1;33m`表示黄色高亮。

```bash
log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}
```
**解释**: 红色错误日志。`\033[0;31m`表示红色。

**为什么**: 彩色输出让日志更易读，快速区分信息级别。

---

### 4. Swap设置

#### 4.1 创建Swap文件

```bash
if [ ! -f /swapfile ]; then
```
**解释**: 检查swap文件是否已存在。`-f`测试文件是否存在，`!`表示否定。

```bash
    log_info "创建Swap文件（2GB）..."
```
**解释**: 输出信息提示用户。

```bash
    dd if=/dev/zero of=/swapfile bs=1M count=2048 2>/dev/null
```
**逐部分解释**:
- `dd` - 数据复制命令
- `if=/dev/zero` - 输入文件，/dev/zero产生无限个0字节
- `of=/swapfile` - 输出文件，即swap文件路径
- `bs=1M` - 块大小1MB
- `count=2048` - 复制2048块 = 2GB
- `2>/dev/null` - 将错误输出丢弃

**为什么**: 创建一个2GB的空文件作为swap空间。

```bash
    chmod 600 /swapfile
```
**解释**: 设置文件权限为仅root可读写（rw-------）。
**为什么**: 安全考虑，防止其他用户访问swap内容。

```bash
    mkswap /swapfile 2>/dev/null
```
**解释**: 将文件格式化为swap格式。

#### 4.2 启用Swap

```bash
if ! swapon -s | grep -q /swapfile; then
```
**逐部分解释**:
- `swapon -s` - 显示当前启用的swap设备
- `grep -q /swapfile` - 静默搜索swapfile，`-q`只返回状态不输出
- `!` - 如果不存在则执行

**为什么**: 检查swap是否已启用，避免重复启用错误。

```bash
    log_info "启用Swap..."
    swapon /swapfile 2>/dev/null || log_warn "Swap启用失败"
```
**解释**: 启用swap文件，如果失败则输出警告但不退出。
**为什么**: `|| log_warn`防止`set -e`导致脚本退出。

#### 4.3 显示内存状态

```bash
log_info "内存状态:"
free -h | grep -E "Mem|Swap"
echo ""
```
**逐部分解释**:
- `free -h` - 显示内存使用，`-h`人类可读格式
- `grep -E "Mem|Swap"` - 只显示Mem和Swap行

**为什么**: 让用户确认swap已成功启用。

---

### 5. 清理旧服务

```bash
log_info "清理旧服务..."
```
**解释**: 提示用户正在清理。

```bash
pkill -9 -f "rag_medical\|model_service\|tts_service" 2>/dev/null || true
```
**逐部分解释**:
- `pkill` - 根据名称发送信号给进程
- `-9` - SIGKILL信号，强制终止
- `-f` - 匹配完整命令行
- `"rag_medical\|model_service\|tts_service"` - 正则表达式，匹配任一关键词
- `2>/dev/null` - 丢弃错误输出
- `|| true` - 如果失败返回true，防止`set -e`退出

**为什么**: 杀死残留的旧服务进程，释放资源。

```bash
sleep 2
```
**解释**: 等待2秒，让进程完全终止。
**为什么**: 确保进程退出，socket文件被释放。

```bash
rm -f /tmp/*.sock
```
**解释**: 删除所有socket文件，`-f`强制不提示。
**为什么**: 防止"Address already in use"错误。

---

### 6. 启动RAG服务

```bash
log_info "启动RAG服务（延迟加载版）..."
```
**解释**: 提示正在启动RAG服务。

```bash
cd /userdata/medical_rag_full
```
**解释**: 切换到RAG数据目录。
**为什么**: RAG服务需要在此目录找到数据文件。

```bash
nohup python3 rag_medical_server_simple.py > /tmp/rag_server.log 2>&1 &
```
**逐部分解释**:
- `nohup` - no hang up，忽略SIGHUP信号
- `python3 rag_medical_server_simple.py` - 启动RAG服务
- `> /tmp/rag_server.log` - 标准输出重定向到日志
- `2>&1` - 标准错误重定向到标准输出
- `&` - 后台运行

**为什么**: 
- `nohup`确保SSH断开后服务继续运行
- 日志重定向方便排查问题
- 后台运行不阻塞脚本

#### 等待RAG服务就绪

```bash
for i in {1..30}; do
```
**解释**: 循环30次，最多等待9秒。

```bash
    if [ -S "/tmp/rag_medical_lazy.sock" ]; then
```
**解释**: 检查socket文件是否存在且是socket类型。`-S`测试是否为socket。

```bash
        log_info "✓ RAG服务已就绪"
        break
```
**解释**: 如果socket存在，输出成功信息并跳出循环。

```bash
    fi
    sleep 0.3
done
```
**解释**: 每次等待0.3秒。

**为什么**: 轮询检查服务是否就绪，比固定等待更精确。

---

### 7. 启动LLM服务

```bash
log_info "启动LLM服务..."
cd /userdata/voice_assistant
```
**解释**: 切换到voice_assistant目录。

```bash
export LD_LIBRARY_PATH=/data/internvl3
```
**解释**: 设置动态库搜索路径。
**为什么**: 让程序能找到RKLLM等依赖库。

```bash
nohup ./model_service_daemon > /tmp/llm_service.log 2>&1 &
```
**解释**: 后台启动LLM服务（C++二进制）。

#### 等待LLM服务

```bash
log_info "等待LLM模型加载（约30秒）..."
```
**解释**: 提示用户LLM加载需要较长时间。

```bash
for i in {1..60}; do
```
**解释**: 最多等待60秒。

```bash
    if [ -S "/tmp/voice_assistant.sock" ]; then
        sleep 2
        if ./llm_client ping > /dev/null 2>&1; then
            log_info "✓ LLM服务已就绪"
            break
        fi
    fi
```
**解释**: 先检查socket，再用ping命令确认服务真正可用。

```bash
    sleep 1
    if [ $((i % 10)) -eq 0 ]; then
        echo "  已等待 ${i} 秒..."
    fi
done
```
**解释**: 每秒检查一次，每10秒输出进度。

**为什么**: LLM模型加载慢，需要双重检查确保就绪。

---

### 8. 启动TTS服务

```bash
log_info "启动TTS服务..."
cd /userdata/voice_assistant
nohup python3 tts_service.py > /tmp/tts_service.log 2>&1 &
```
**解释**: 后台启动TTS服务。

#### 等待TTS服务

```bash
for i in {1..20}; do
    if [ -S "/tmp/tts_service.sock" ]; then
        log_info "✓ TTS服务已就绪"
        break
    fi
    sleep 0.5
done
```
**解释**: 最多等待10秒（20×0.5秒）。

**为什么**: TTS启动较快，等待时间较短。

---

### 9. 完成信息

```bash
echo ""
echo "========================================"
echo "    所有服务已启动"
echo "========================================"
echo ""
```
**解释**: 打印完成分隔线。

#### 显示服务状态

```bash
log_info "服务状态:"
echo "  RAG: /tmp/rag_medical_lazy.sock"
echo "  LLM: /tmp/voice_assistant.sock"
echo "  TTS: /tmp/tts_service.sock"
echo ""
```
**解释**: 显示三个服务的socket路径。
**为什么**: 方便用户手动测试服务。

#### 显示内存使用

```bash
log_info "内存使用:"
free -h | grep -E "Mem|Swap"
echo ""
```
**解释**: 显示当前内存和swap使用情况。

#### 显示运行进程

```bash
log_info "运行进程:"
ps | grep -E "python|model_service" | grep -v grep
echo ""
```
**解释**: 显示正在运行的服务进程。
**为什么**: 确认服务确实在运行。

#### 提示下一步

```bash
echo "========================================"
echo "    现在可以运行语音助手:"
echo "    ./voice_assistant_lazy_rag_v2.sh"
echo "========================================"
```
**解释**: 提示用户如何运行语音助手主脚本。

---

## 关键设计要点

### 1. 错误处理
- `set -e` 确保错误立即退出
- `|| true` 防止非致命错误中断脚本

### 2. 资源管理
- 创建Swap防止OOM
- 清理旧服务释放资源

### 3. 服务编排
- 按依赖顺序启动
- 轮询检查确保就绪

### 4. 用户体验
- 彩色日志输出
- 进度提示
- 状态汇总

---

## 执行流程图

```
开始
  │
  ▼
设置Swap ──→ 创建/启用Swap文件
  │
  ▼
清理旧服务 ──→ 杀死旧进程 → 删除socket
  │
  ▼
启动RAG ──→ 后台运行 → 检查socket (最多9秒)
  │
  ▼
启动LLM ──→ 设置环境变量 → 后台运行 → ping检查 (最多60秒)
  │
  ▼
启动TTS ──→ 后台运行 → 检查socket (最多10秒)
  │
  ▼
显示状态 ──→ socket路径 → 内存使用 → 运行进程
  │
  ▼
完成
```

---

## 故障排查

### 问题1: RAG服务启动超时
```bash
# 查看日志
cat /tmp/rag_server.log

# 手动启动查看错误
python3 /userdata/medical_rag_full/rag_medical_server_simple.py
```

### 问题2: LLM服务无法连接
```bash
# 检查模型文件
ls -la /data/internvl3/*.rkllm

# 检查依赖库
ldd /userdata/voice_assistant/model_service_daemon

# 查看日志
cat /tmp/llm_service.log
```

### 问题3: 内存不足
```bash
# 查看内存
free -h

# 查看OOM日志
dmesg | grep -i kill

# 增加Swap
fallocate -l 4G /swapfile2
mkswap /swapfile2
swapon /swapfile2
```
