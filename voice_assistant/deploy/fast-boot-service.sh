#!/bin/bash
# 语音助手快速启动服务 - 开机自动启动 + 常驻内存方案
# 实现重启后秒级启动

SERVICE_NAME="voice-assistant-fast"
PID_FILE="/var/run/voice-assistant.pid"
LOCK_FILE="/var/lock/voice-assistant.lock"
LOG_FILE="/tmp/voice-assistant-fast.log"

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
        PIDS=$(cat $PID_FILE 2>/dev/null)
        for PID in $PIDS; do
            if kill -0 $PID 2>/dev/null; then
                return 0
            fi
        done
    fi
    return 1
}

# 快速启动服务（带预加载）
do_fast_start() {
    log "=== 快速启动 Voice Assistant 服务 ==="
    
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
    
    # 快速预加载关键模型（只加载最大的几个）
    log "快速预加载关键模型..."
    (
        # 后台预加载，不阻塞服务启动
        cat /data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm > /dev/null 2>&1 &
        cat /userdata/medical_rag_full/text2vec-small-chinese_fp16.rknn > /dev/null 2>&1 &
    )
    
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
    
    # 保存PID
    echo "$RAG_PID $LLM_PID" > $PID_FILE
    
    log "核心服务已启动，PID: $RAG_PID $LLM_PID"
    
    # 等待服务就绪（优化等待时间）
    log "等待服务就绪..."
    READY_DIR="/tmp/service_ready"
    mkdir -p $READY_DIR
    
    for i in {1..200}; do
        [ -S "/tmp/rag_optimized.sock" ] && touch $READY_DIR/rag 2>/dev/null
        [ -S "/tmp/qwen3_llm.sock" ] && python3 $VA_DIR/qwen3_llm_client.py "ping" > /dev/null 2>&1 && touch $READY_DIR/llm 2>/dev/null
        
        if [ -f $READY_DIR/rag ] && [ -f $READY_DIR/llm ]; then
            log "核心服务已就绪（RAG + LLM）"
            break
        fi
        
        if ! kill -0 $RAG_PID 2>/dev/null || ! kill -0 $LLM_PID 2>/dev/null; then
            log "错误: 有服务进程异常退出"
            do_stop
            return 1
        fi
        
        sleep 0.05
done
    
    log "服务启动完成"
    return 0
}

# 暂停服务（保持内存常驻）
do_pause() {
    log "=== 暂停 Voice Assistant 服务 ==="
    
    if [ -f $PID_FILE ]; then
        PIDS=$(cat $PID_FILE)
        # 发送暂停信号（SIGUSR1）
        for PID in $PIDS; do
            kill -USR1 $PID 2>/dev/null || true
        done
        log "服务已暂停（内存常驻）"
    fi
}

# 恢复服务
do_resume() {
    log "=== 恢复 Voice Assistant 服务 ==="
    
    if [ -f $PID_FILE ]; then
        PIDS=$(cat $PID_FILE)
        # 发送恢复信号（SIGUSR2）
        for PID in $PIDS; do
            kill -USR2 $PID 2>/dev/null || true
        done
        log "服务已恢复"
    else
        do_fast_start
    fi
}

# 停止服务
do_stop() {
    log "=== 停止 Voice Assistant 服务 ==="
    
    if [ -f $PID_FILE ]; then
        PIDS=$(cat $PID_FILE)
        log "停止进程: $PIDS"
        
        for PID in $PIDS; do
            kill $PID 2>/dev/null || true
        done
        
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
        
        for PID in $PIDS; do
            if kill -0 $PID 2>/dev/null; then
                log "强制终止进程 $PID"
                kill -9 $PID 2>/dev/null || true
            fi
        done
        
        rm -f $PID_FILE
    fi
    
    pkill -9 -f "rag_optimized_server" 2>/dev/null || true
    pkill -9 -f "qwen3_llm_service" 2>/dev/null || true
    pkill -9 -f "melotts_service_rknn" 2>/dev/null || true
    
    rm -f /tmp/rag_optimized.sock
    rm -f /tmp/qwen3_llm.sock
    rm -f /tmp/melotts_service.sock
    rm -f /tmp/service_ready/*
    rm -f $LOCK_FILE
    
    log "服务已停止"
    return 0
}

# 查看状态
do_status() {
    echo "=== Voice Assistant 快速启动服务状态 ==="
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
            echo "  ✗ $sock"
        fi
    done
    
    echo ""
    echo "内存使用:"
    free -h | grep -E 'Mem|Swap'
    
    return 0
}

# 主函数
case "$1" in
    start)
        do_fast_start
        ;;
    pause)
        do_pause
        ;;
    resume)
        do_resume
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_stop
        sleep 1
        do_fast_start
        ;;
    status)
        do_status
        ;;
    *)
        echo "用法: $0 {start|pause|resume|stop|restart|status}"
        echo ""
        echo "命令说明:"
        echo "  start   - 快速启动服务（带预加载）"
        echo "  pause   - 暂停服务（保持内存常驻）"
        echo "  resume  - 恢复服务"
        echo "  stop    - 停止服务"
        echo "  restart - 重启服务"
        echo "  status  - 查看状态"
        exit 1
        ;;
esac

exit 0
