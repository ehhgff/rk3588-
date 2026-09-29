#!/bin/bash
# 语音助手快速启动服务 - 开机自动启动 + 常驻内存方案
# 启动 RAG + sherpa-onnx Matcha TTS + SenseVoice SER + LLM + streaming_vad.py

SERVICE_NAME="voice-assistant-fast"
PID_FILE="/var/run/voice-assistant.pid"
LOCK_FILE="/var/lock/voice-assistant.lock"
LOG_FILE="/tmp/voice-assistant-fast.log"
RUN_SCRIPT="/data/voice_assistant/run.sh"

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

    # 防重复启动
    if [ -f $LOCK_FILE ]; then
        if check_running; then
            log "服务已在运行中，跳过启动"
            return 0
        fi
        log "检测到上次启动未正常退出，清理残留..."
        rm -f $LOCK_FILE
    fi
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

    # ==========================================
    # 调用 run.sh 启动所有服务
    #   - Patient RAG
    #   - Qwen3 LLM
    #   - sherpa-onnx Matcha TTS
    #   - SenseVoice SER
    #   - streaming_vad.py (Zipformer RKNN ASR)
    # ==========================================
    log "启动所有服务 (run.sh)..."
    cd /data/voice_assistant
    nohup ./run.sh > /tmp/voice_assistant_run.log 2>&1 &
    RUN_PID=$!
    echo "$RUN_PID" > $PID_FILE
    log "run.sh 已启动，PID: $RUN_PID"

    log "服务启动完成，后台日志: /tmp/voice_assistant_run.log"
    return 0
}

# 停止服务
do_stop() {
    log "=== 停止 Voice Assistant 服务 ==="

    # 停止 run.sh（它会连带停止所有子服务）
    pkill -9 -f "run.sh" 2>/dev/null || true
    sleep 1

    # 停止所有相关进程
    pkill -9 -f "streaming_vad" 2>/dev/null || true
    pkill -9 -f "qwen3_llm_streaming" 2>/dev/null || true
    pkill -9 -f "sherpa_tts_daemon" 2>/dev/null || true
    pkill -9 -f "sherpa_tts_service" 2>/dev/null || true
    pkill -9 -f "rag_optimized_server" 2>/dev/null || true
    pkill -9 -f "patient_rag_server" 2>/dev/null || true
    pkill -9 -f "sensevoice_server" 2>/dev/null || true
    pkill -9 -f "sherpa-onnx-alsa" 2>/dev/null || true

    # 清理 socket 文件
    rm -f /tmp/tts_pipe
    rm -f /tmp/qwen3_llm.sock
    rm -f /tmp/qwen3_llm_streaming.sock
    rm -f /tmp/qwen2_llm_streaming.sock
    rm -f /tmp/rag_optimized.sock
    rm -f /tmp/patient_rag.sock
    rm -f /tmp/sensevoice_server.sock
    rm -f /tmp/service_ready/*

    rm -f $PID_FILE
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
    echo "Socket 文件状态:"
    if [ -f "/tmp/tts_pipe" ]; then
            echo "  ✓ /tmp/tts_pipe"
        else
            echo "  ✗ /tmp/tts_pipe"
        fi
        for sock in /tmp/qwen3_llm.sock /tmp/rag_optimized.sock /tmp/patient_rag.sock /tmp/sensevoice_server.sock; do
        if [ -S "$sock" ]; then
            echo "  ✓ $sock"
        else
            echo "  ✗ $sock"
        fi
    done

    echo ""
    echo "服务进程:"
    for proc in sherpa_tts_daemon qwen3_llm_streaming rag_optimized_server patient_rag_server sensevoice_server streaming_vad; do
        PIDS=$(pgrep -f "$proc" 2>/dev/null | tr '\n' ' ')
        if [ -n "$PIDS" ]; then
            echo "  ✓ $proc (PID: $PIDS)"
        else
            echo "  ✗ $proc"
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
        echo "用法: $0 {start|stop|restart|status}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动所有服务（RAG + LLM + Matcha TTS + SER + Zipformer ASR）"
        echo "  stop    - 停止所有服务"
        echo "  restart - 重启所有服务"
        echo "  status  - 查看服务状态"
        exit 1
        ;;
esac

exit 0