#!/bin/bash
# Qwen3-1.7B LLM 服务启动脚本

export LD_LIBRARY_PATH=/data/qwen3:$LD_LIBRARY_PATH
export RKLLM_LIB_PATH=/data/qwen3/librkllmrt.so

SERVICE_DIR="/userdata/voice_assistant"
SERVICE_NAME="qwen3_llm_service.py"
LOG_FILE="/tmp/qwen3_llm_service.log"
SOCKET_FILE="/tmp/qwen3_llm.sock"

start_service() {
    echo "启动 Qwen3-1.7B LLM 服务..."

    # 杀死旧进程
    pkill -f "$SERVICE_NAME" 2>/dev/null
    sleep 1

    # 启动服务
    cd "$SERVICE_DIR"
    nohup python3 "$SERVICE_NAME" > "$LOG_FILE" 2>&1 &

    # 等待服务就绪
    echo "等待服务启动..."
    for i in {1..60}; do
        if [ -S "$SOCKET_FILE" ]; then
            # 测试连接
            if python3 -c "import socket, json; s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(1); s.connect('$SOCKET_FILE'); s.send(json.dumps({'prompt':'test','max_tokens':1}).encode()); s.recv(1024); s.close()" 2>/dev/null; then
                echo "✓ Qwen3-1.7B LLM 服务已就绪"
                return 0
            fi
        fi
        sleep 0.5
    done

    echo "✗ 服务启动失败，请检查日志: $LOG_FILE"
    return 1
}

stop_service() {
    echo "停止 Qwen3-1.7B LLM 服务..."
    pkill -f "$SERVICE_NAME" 2>/dev/null
    rm -f "$SOCKET_FILE"
    echo "✓ 服务已停止"
}

status() {
    if pgrep -f "$SERVICE_NAME" > /dev/null; then
        echo "✓ Qwen3-1.7B LLM 服务正在运行"
        echo "  Socket: $SOCKET_FILE"
        echo "  日志: $LOG_FILE"
    else
        echo "✗ Qwen3-1.7B LLM 服务未运行"
    fi
}

case "$1" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        sleep 2
        start_service
        ;;
    status)
        status
        ;;
    log)
        tail -50 "$LOG_FILE"
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|log}"
        exit 1
        ;;
esac
