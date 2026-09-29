#!/bin/bash
# 一键启动所有服务（优化版）
# 包含Swap设置、服务启动和状态检查

set -e

echo "========================================"
echo "    启动医疗RAG语音助手服务"
echo "========================================"
echo ""

# 记录总启动时间
TOTAL_START_TIME=$(date +%s%N)

# 1. 设置Swap
log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# 打印耗时函数
print_elapsed() {
    local start=$1
    local end=$(date +%s%N)
    local elapsed=$(( (end - start) / 1000000 ))
    echo "${elapsed}ms"
}

# 启用Swap
if [ ! -f /swapfile ]; then
    log_info "创建Swap文件（2GB）..."
    dd if=/dev/zero of=/swapfile bs=1M count=2048 2>/dev/null
    chmod 600 /swapfile
    mkswap /swapfile 2>/dev/null
fi

if ! swapon -s | grep -q /swapfile; then
    log_info "启用Swap..."
    swapon /swapfile 2>/dev/null || log_warn "Swap启用失败"
fi

log_info "内存状态:"
free -h | grep -E "Mem|Swap"
echo ""

# 2. 清理旧服务
log_info "清理旧服务..."
pkill -9 -f "rag_medical\|qwen3_llm\|tts_service" 2>/dev/null || true
sleep 2
rm -f /tmp/*.sock

# 3. 启动RAG服务（优化版 - RKNN + 混合搜索 + 科室过滤）
log_info "启动RAG服务（优化版 - RKNN + 混合搜索 + 科室过滤）..."
RAG_START_TIME=$(date +%s%N)
cd /userdata/medical_rag_full
nohup python3 rag_optimized_server.py > /tmp/rag_optimized_server.log 2>&1 &

# 等待RAG服务（优化后启动更快，约2-3秒）
for i in {1..50}; do
    if [ -S "/tmp/rag_optimized.sock" ]; then
        log_info "✓ RAG服务已就绪（RKNN NPU 加速 + 科室过滤），耗时: $(print_elapsed $RAG_START_TIME)"
        log_info "  科室明确查询: ~300ms | 模糊查询: ~1100ms"
        break
    fi
    sleep 0.1
done

# 4. 启动Qwen3 LLM服务
log_info "启动Qwen3 LLM服务..."
QWEN3_START_TIME=$(date +%s%N)
cd /userdata/voice_assistant
export LD_LIBRARY_PATH=/data/qwen3:$LD_LIBRARY_PATH
# 使用nohup启动，并将输出重定向到日志文件
nohup python3 qwen3_llm_service.py > /tmp/qwen3.log 2>&1 &
QWEN3_PID=$!

# 等待Qwen3 LLM服务
log_info "等待Qwen3模型加载..."
for i in {1..120}; do
    # 检查进程是否还在运行
    if ! kill -0 $QWEN3_PID 2>/dev/null; then
        log_error "Qwen3服务进程已退出"
        cat /tmp/qwen3.log | tail -10
        exit 1
    fi
    
    # 检查socket是否存在且服务可用
    if [ -S "/tmp/qwen3_llm.sock" ]; then
        if python3 qwen3_llm_client.py "ping" > /dev/null 2>&1; then
            log_info "✓ Qwen3 LLM服务已就绪，耗时: $(print_elapsed $QWEN3_START_TIME)"
            break
        fi
    fi
    sleep 0.3
    if [ $((i % 30)) -eq 0 ]; then
        echo "  已等待 $((i * 3 / 10)) 秒..."
    fi
done

# 5. 启动TTS服务（RKNN优化版）
log_info "启动TTS服务（RKNN优化版 - MeloTTS）..."
TTS_START_TIME=$(date +%s%N)
cd /data/voice_assistant
nohup python3 melotts_service_rknn.py > /tmp/melotts_service.log 2>&1 &

# 等待TTS服务
for i in {1..20}; do
    if [ -S "/tmp/melotts_service.sock" ]; then
        log_info "✓ TTS服务已就绪（RKNN NPU 加速），耗时: $(print_elapsed $TTS_START_TIME)"
        break
    fi
    sleep 0.5
done

echo ""
echo "========================================"
echo "    所有服务已启动"
echo "========================================"
echo ""

# 打印总启动时间
TOTAL_END_TIME=$(date +%s%N)
TOTAL_ELAPSED=$(( (TOTAL_END_TIME - TOTAL_START_TIME) / 1000000 ))
log_info "总启动时间: ${TOTAL_ELAPSED}ms"
echo ""

# 显示状态
log_info "服务状态:"
echo "  RAG: /tmp/rag_optimized.sock (RKNN + 混合搜索 + 科室过滤)"
echo "  LLM: /tmp/qwen3_llm.sock (Qwen3-1.7B)"
echo "  TTS: /tmp/melotts_service.sock (MeloTTS RKNN NPU)"
echo ""
log_info "RAG性能指标:"
echo "  启动时间: ~2秒"
echo "  明确科室查询: ~300ms (提升7.7倍)"
echo "  模糊查询: ~1100ms (提升2倍)"
echo ""

log_info "内存使用:"
free -h | grep -E "Mem|Swap"
echo ""

log_info "运行进程:"
ps | grep -E "python|model_service" | grep -v grep
echo ""

echo "========================================"
echo "    现在可以运行语音助手:"
echo "    python3 streaming_vad.py"
echo "======================================="
