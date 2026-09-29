#!/bin/bash
# Voice Assistant 服务管理脚本（BusyBox 兼容版）
# 替代 systemd 的功能

SERVICE_NAME="voice-assistant"
PID_FILE="/var/run/voice-assistant.pid"
LOCK_FILE="/var/lock/voice-assistant.lock"
LOG_FILE="/tmp/voice-assistant-service.log"

# 服务配置
RAG_DIR="/userdata/medical_rag_full"
VA_DIR="/userdata/voice_assistant"
TTS_DIR="/data/voice_assistant"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# 检查服务是否运行
check_running() {
    if [ -f $PID_FILE ]; then
        # 读取PID并检查进程是否存在
        PIDS=$(cat $PID_FILE 2>/dev/null)
        for PID in $PIDS; do
            if kill -0 $PID 2>/dev/null; then
                return 0  # 有进程在运行
            fi
        done
    fi
    return 1  # 没有运行
}

# 启动服务
do_start() {
    if check_running; then
        log "服务已经在运行"
        return 0
    fi
    
    log "=== 启动 Voice Assistant 服务 ==="
    
    # 创建锁文件
    touch $LOCK_FILE
    
    # 预启动检查
    log "执行预启动检查..."
    
    # 检查并启用swap
    if [ -f /swapfile ]; then
        swapon /swapfile 2>/dev/null || true
    fi
    
    # 设置swappiness
    echo 10 > /proc/sys/vm/swappiness 2>/dev/null || true
    
    # 清理旧socket
    rm -f /tmp/*.sock 2>/dev/null
    rm -f /tmp/service_ready/* 2>/dev/null
    
    # 启动RAG服务
    log "启动RAG服务..."
    cd $RAG_DIR
    nohup python3 rag_optimized_server.py > /tmp/rag_optimized_server.log 2>&1 &
    RAG_PID=$!
    
    # 启动LLM服务
    log "启动LLM服务..."
    cd $VA_DIR
    export LD_LIBRARY_PATH=/data/qwen3:$LD_LIBRARY_PATH
    nohup python3 qwen3_llm_service.py > /tmp/qwen3.log 2>&1 &
    LLM_PID=$!
    
    # 不启动TTS服务（按需启动）
    log "TTS服务将在需要时由客户端启动"
    
    # 保存PID（只有RAG和LLM）
    echo "$RAG_PID $LLM_PID" > $PID_FILE
    
    log "核心服务已启动，PID: $RAG_PID $LLM_PID"
    
    # 等待服务就绪
    log "等待服务就绪..."
    READY_DIR="/tmp/service_ready"
    mkdir -p $READY_DIR
    
    for i in {1..150}; do
        # 检查各服务socket（只检查RAG和LLM）
        [ -S "/tmp/rag_optimized.sock" ] && touch $READY_DIR/rag 2>/dev/null
        [ -S "/tmp/qwen3_llm.sock" ] && python3 $VA_DIR/qwen3_llm_client.py "ping" > /dev/null 2>&1 && touch $READY_DIR/llm 2>/dev/null
        
        # 核心服务就绪（RAG + LLM）
        if [ -f $READY_DIR/rag ] && [ -f $READY_DIR/llm ]; then
            log "核心服务已就绪（RAG + LLM）"
            break
        fi
        
        # 检查进程存活
        if ! kill -0 $RAG_PID 2>/dev/null || ! kill -0 $LLM_PID 2>/dev/null; then
            log "错误: 有服务进程异常退出"
            do_stop
            return 1
        fi
        
        sleep 0.1
    done
    
    log "服务启动完成"
    return 0
}

# 停止服务
do_stop() {
    log "=== 停止 Voice Assistant 服务 ==="
    
    if [ -f $PID_FILE ]; then
        PIDS=$(cat $PID_FILE)
        log "停止进程: $PIDS"
        
        # 先尝试优雅停止
        for PID in $PIDS; do
            kill $PID 2>/dev/null || true
        done
        
        # 等待进程退出
        for i in {1..30}; do
            ALL_DEAD=true
            for PID in $PIDS; do
                if kill -0 $PID 2>/dev/null; then
                    ALL_DEAD=false
                    break
                fi
            done
            if [ "$ALL_DEAD" = true ]; then
                break
            fi
            sleep 0.1
        done
        
        # 强制终止残留进程
        for PID in $PIDS; do
            if kill -0 $PID 2>/dev/null; then
                log "强制终止进程 $PID"
                kill -9 $PID 2>/dev/null || true
            fi
        done
        
        rm -f $PID_FILE
    fi
    
    # 清理其他相关进程
    pkill -9 -f "rag_optimized_server" 2>/dev/null || true
    pkill -9 -f "qwen3_llm_service" 2>/dev/null || true
    pkill -9 -f "melotts_service_rknn" 2>/dev/null || true
    
    # 清理socket
    rm -f /tmp/rag_optimized.sock
    rm -f /tmp/qwen3_llm.sock
    rm -f /tmp/melotts_service.sock
    rm -f /tmp/service_ready/*
    rm -f $LOCK_FILE
    
    log "服务已停止"
    return 0
}

# 重启服务
do_restart() {
    do_stop
    sleep 1
    do_start
}

# 查看状态
do_status() {
    echo "=== Voice Assistant 服务状态 ==="
    echo ""
    
    if check_running; then
        echo "服务状态: 运行中"
        echo "PID文件: $PID_FILE"
        echo "进程ID: $(cat $PID_FILE 2>/dev/null)"
    else
        echo "服务状态: 未运行"
    fi
    
    echo ""
    echo "Socket文件:"
    for sock in /tmp/rag_optimized.sock /tmp/qwen3_llm.sock /tmp/melotts_service.sock; do
        if [ -S "$sock" ]; then
            echo "  ✓ $sock"
        else
            echo "  ✗ $sock (不存在)"
        fi
    done
    
    echo ""
    echo "进程状态:"
    ps | grep -E "rag_optimized|qwen3_llm|melotts_service" | grep -v grep || echo "  无运行进程"
    
    echo ""
    echo "内存使用:"
    free -h | grep -E "Mem|Swap"
}

# 监控模式（自动重启）
do_monitor() {
    log "进入监控模式..."
    while true; do
        if ! check_running; then
            log "检测到服务未运行，自动重启..."
            do_start
        fi
        sleep 5
    done
}

# 主程序
case "${1:-start}" in
    start)
        do_start
        exit $?
        ;;
    stop)
        do_stop
        exit $?
        ;;
    restart)
        do_restart
        exit $?
        ;;
    status)
        do_status
        exit 0
        ;;
    monitor)
        do_monitor
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|monitor}"
        echo ""
        echo "Commands:"
        echo "  start    - 启动服务"
        echo "  stop     - 停止服务"
        echo "  restart  - 重启服务"
        echo "  status   - 查看状态"
        echo "  monitor  - 监控模式（自动重启）"
        exit 1
        ;;
esac
