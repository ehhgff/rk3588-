#!/bin/bash
# 一键启动所有服务（V3优化版）
# 优化点：就绪通知机制 + 快速清理 + 预检查

set -e

echo "========================================"
echo "    启动医疗RAG语音助手服务(V3优化)"
echo "========================================"
echo ""

# 记录总启动时间
TOTAL_START_TIME=$(date +%s%N)

# 就绪标记目录
READY_DIR="/tmp/service_ready"
mkdir -p $READY_DIR
rm -f $READY_DIR/*

# 日志函数
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

# 1. 快速Swap检查（假设已预创建）
SWAP_START=$(date +%s%N)
if ! swapon -s | grep -q /swapfile; then
    log_warn "Swap未启用，尝试启用..."
    swapon /swapfile 2>/dev/null || log_warn "Swap启用失败，继续启动"
fi
log_info "Swap检查耗时: $(print_elapsed $SWAP_START)"

log_info "内存状态:"
free -h | grep -E "Mem|Swap"
echo ""

# 2. 快速清理旧服务（非阻塞）
CLEAN_START=$(date +%s%N)
log_info "快速清理旧服务..."
pkill -9 -f "rag_medical\|qwen3_llm\|tts_service\|melotts" 2>/dev/null &
PKILL_PID=$!

# 同时删除socket文件
rm -f /tmp/*.sock 2>/dev/null
rm -f /tmp/service_ready/* 2>/dev/null

# 等待pkill完成（最多0.5秒）
wait $PKILL_PID 2>/dev/null || true
sleep 0.3  # 缩短等待时间

log_info "清理耗时: $(print_elapsed $CLEAN_START)"
echo ""

# 3. 串行启动服务（保持稳定性）
log_info "启动服务（带就绪通知）..."

# 3.1 启动RAG服务
RAG_START=$(date +%s%N)
log_info "启动RAG服务..."
cd /userdata/medical_rag_full
nohup python3 rag_optimized_server.py > /tmp/rag_optimized_server.log 2>&1 &
RAG_PID=$!

# 后台等待RAG就绪
(
    for i in {1..100}; do
        if [ -S "/tmp/rag_optimized.sock" ]; then
            touch $READY_DIR/rag
            break
        fi
        sleep 0.05
    done
) &

# 3.2 启动Qwen3 LLM服务
QWEN3_START=$(date +%s%N)
log_info "启动Qwen3 LLM服务..."
cd /userdata/voice_assistant
export LD_LIBRARY_PATH=/data/qwen3:$LD_LIBRARY_PATH
nohup python3 qwen3_llm_service.py > /tmp/qwen3.log 2>&1 &
QWEN3_PID=$!

# 后台等待LLM就绪
(
    for i in {1..200}; do
        if [ -S "/tmp/qwen3_llm.sock" ]; then
            if python3 qwen3_llm_client.py "ping" > /dev/null 2>&1; then
                touch $READY_DIR/llm
                break
            fi
        fi
        sleep 0.1
    done
) &

# 3.3 启动TTS服务
TTS_START=$(date +%s%N)
log_info "启动TTS服务..."
cd /data/voice_assistant
nohup python3 melotts_service_rknn.py > /tmp/melotts_service.log 2>&1 &
TTS_PID=$!

# 后台等待TTS就绪
(
    for i in {1..50}; do
        if [ -S "/tmp/melotts_service.sock" ]; then
            touch $READY_DIR/tts
            break
        fi
        sleep 0.1
    done
) &

# 4. 等待所有服务就绪（带超时）
log_info "等待服务就绪..."
WAIT_START=$(date +%s%N)

rag_ready=false
llm_ready=false
tts_ready=false
timeout=150  # 15秒超时

for i in $(seq 1 $timeout); do
    # 检查RAG
    if [ "$rag_ready" = false ] && [ -f $READY_DIR/rag ]; then
        log_info "✓ RAG服务已就绪，耗时: $(print_elapsed $RAG_START)"
        rag_ready=true
    fi
    
    # 检查LLM
    if [ "$llm_ready" = false ] && [ -f $READY_DIR/llm ]; then
        log_info "✓ Qwen3 LLM服务已就绪，耗时: $(print_elapsed $QWEN3_START)"
        llm_ready=true
    fi
    
    # 检查TTS
    if [ "$tts_ready" = false ] && [ -f $READY_DIR/tts ]; then
        log_info "✓ TTS服务已就绪，耗时: $(print_elapsed $TTS_START)"
        tts_ready=true
    fi
    
    # 全部就绪
    if [ "$rag_ready" = true ] && [ "$llm_ready" = true ] && [ "$tts_ready" = true ]; then
        break
    fi
    
    # 检查进程是否存活
    if ! kill -0 $RAG_PID 2>/dev/null; then
        log_error "RAG服务进程已退出"
        cat /tmp/rag_optimized_server.log | tail -5
        exit 1
    fi
    if ! kill -0 $QWEN3_PID 2>/dev/null; then
        log_error "Qwen3服务进程已退出"
        cat /tmp/qwen3.log | tail -5
        exit 1
    fi
    if ! kill -0 $TTS_PID 2>/dev/null; then
        log_error "TTS服务进程已退出"
        cat /tmp/melotts_service.log | tail -5
        exit 1
    fi
    
    sleep 0.1
done

# 检查是否超时
if [ "$rag_ready" = false ] || [ "$llm_ready" = false ] || [ "$tts_ready" = false ]; then
    log_warn "部分服务未在预期时间内就绪，继续执行..."
fi

log_info "等待就绪总耗时: $(print_elapsed $WAIT_START)"

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
[ "$rag_ready" = true ] && echo "  ✓ RAG: /tmp/rag_optimized.sock" || echo "  ✗ RAG: 未就绪"
[ "$llm_ready" = true ] && echo "  ✓ LLM: /tmp/qwen3_llm.sock" || echo "  ✗ LLM: 未就绪"
[ "$tts_ready" = true ] && echo "  ✓ TTS: /tmp/melotts_service.sock" || echo "  ✗ TTS: 未就绪"
echo ""

log_info "RAG性能指标:"
echo "  明确科室查询: ~300ms"
echo "  模糊查询: ~1100ms"
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
