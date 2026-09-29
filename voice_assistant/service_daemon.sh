#!/bin/bash
# 服务守护进程 - 保持服务常驻内存，实现秒级启动
# 使用方法: ./service_daemon.sh start|stop|status

SERVICE_DIR="/userdata/voice_assistant"
LOG_DIR="/tmp"
PID_FILE="/tmp/voice_assistant_daemon.pid"

# 颜色输出
log_info() { echo -e "\033[0;32m[INFO]\033[0m $1"; }
log_warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }
log_success() { echo -e "\033[0;32m[✓]\033[0m $1"; }

# 检查服务状态
check_services() {
    local rag_ok=0
    local llm_ok=0
    local tts_ok=0
    
    [ -S "/tmp/rag_optimized.sock" ] && rag_ok=1
    [ -S "/tmp/qwen3_llm.sock" ] && llm_ok=1
    [ -S "/tmp/melotts_service.sock" ] && tts_ok=1
    
    echo "RAG:$rag_ok LLM:$llm_ok TTS:$tts_ok"
}

# 启动服务
start_services() {
    log_info "启动服务守护进程..."
    
    # 清理旧服务
    pkill -9 -f "rag_optimized\|qwen3_llm\|melotts" 2>/dev/null || true
    rm -f /tmp/*.sock 2>/dev/null || true
    sleep 0.5
    
    # 确保swap
    if [ -f /swapfile ] && ! cat /proc/swaps | grep -q /swapfile; then
        swapon /swapfile 2>/dev/null || true
    fi
    
    # 并行启动
    (
        cd /userdata/medical_rag_full
        exec python3 rag_optimized_server.py > $LOG_DIR/rag_optimized_server.log 2>&1
    ) &
    
    (
        cd $SERVICE_DIR
        export LD_LIBRARY_PATH=/data/qwen3:$LD_LIBRARY_PATH
        exec python3 qwen3_llm_service.py > $LOG_DIR/qwen3.log 2>&1
    ) &
    
    (
        cd /data/voice_assistant
        exec python3 melotts_service_rknn.py > $LOG_DIR/melotts_service.log 2>&1
    ) &
    
    # 保存PID
    sleep 1
    pgrep -f "rag_optimized_server\|qwen3_llm_service\|melotts_service_rknn" > $PID_FILE 2>/dev/null || true
    
    log_info "服务启动中，等待就绪..."
    
    # 等待就绪（最多30秒）
    for i in {1..60}; do
        local status=$(check_services)
        if [[ "$status" == "RAG:1 LLM:1 TTS:1" ]]; then
            log_success "所有服务已就绪！"
            return 0
        fi
        sleep 0.5
        if [ $((i % 10)) -eq 0 ]; then
            echo "  等待中... ($((i/2))s) - $status"
        fi
    done
    
    log_warn "服务启动可能未完成，当前状态: $(check_services)"
    return 1
}

# 停止服务
stop_services() {
    log_info "停止服务..."
    pkill -9 -f "rag_optimized\|qwen3_llm\|melotts" 2>/dev/null || true
    rm -f /tmp/*.sock $PID_FILE 2>/dev/null || true
    log_success "服务已停止"
}

# 查看状态
show_status() {
    local status=$(check_services)
    echo "========================================"
    echo "    服务状态"
    echo "========================================"
    echo ""
    
    if [[ "$status" == "RAG:1 LLM:1 TTS:1" ]]; then
        log_success "所有服务运行中"
    else
        log_warn "部分服务未运行"
    fi
    
    echo "  RAG服务: $([ -S /tmp/rag_optimized.sock ] && echo '✓ 运行中' || echo '✗ 未运行')"
    echo "  LLM服务: $([ -S /tmp/qwen3_llm.sock ] && echo '✓ 运行中' || echo '✗ 未运行')"
    echo "  TTS服务: $([ -S /tmp/melotts_service.sock ] && echo '✓ 运行中' || echo '✗ 未运行')"
    echo ""
    
    log_info "内存使用:"
    free -h | grep -E "Mem|Swap" | head -1
    echo ""
    
    log_info "运行进程:"
    ps | grep -E "rag_optimized|qwen3_llm|melotts" | grep -v grep || echo "  无相关进程"
}

# 快速重启（用于开发测试）
fast_restart() {
    log_info "快速重启服务..."
    stop_services
    sleep 1
    start_services
}

# 主命令处理
case "${1:-status}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        fast_restart
        ;;
    status)
        show_status
        ;;
    *)
        echo "使用方法: $0 {start|stop|restart|status}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动所有服务"
        echo "  stop    - 停止所有服务"
        echo "  restart - 快速重启服务"
        echo "  status  - 查看服务状态"
        exit 1
        ;;
esac
