#!/bin/bash
# sherpa_tts_service.sh - TTS 常驻服务管理脚本
# 用法:
#   ./sherpa_tts_service.sh start    # 启动常驻服务
#   ./sherpa_tts_service.sh stop     # 停止常驻服务
#   ./sherpa_tts_service.sh restart  # 重启常驻服务
#   ./sherpa_tts_service.sh status   # 查看服务状态
#   ./sherpa_tts_service.sh speak <text>  # 直接合成并播放
#   ./sherpa_tts_service.sh speakf <file> # 从文件读取文本并播放

MODEL_DIR="/data/sherpa-onnx/matcha-zh-baker"
DAEMON="/data/sherpa_tts_daemon"
PIPE="/tmp/tts_pipe"
PIDFILE="/tmp/tts_daemon.pid"
OUTDIR="/tmp"

# USB 音响设备号
AUDIO_DEV="plughw:1,0"

export LD_LIBRARY_PATH="/userdata/sherpa-onnx/install/lib:/usr/lib"

start() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "服务已在运行中 (PID $(cat $PIDFILE))"
        return 0
    fi

    echo "启动 TTS 常驻服务..."
    echo "  模型目录: $MODEL_DIR"
    echo "  管道文件: $PIPE"

    # 先测试模型文件是否存在
    if [ ! -f "$MODEL_DIR/model-steps-3.onnx" ]; then
        echo "错误: 找不到 $MODEL_DIR/model-steps-3.onnx"
        exit 1
    fi

    nohup $DAEMON $MODEL_DIR $PIPE > /tmp/tts_daemon.log 2>&1 &
    echo $! > "$PIDFILE"

    sleep 2
    if kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "服务启动成功 (PID $(cat $PIDFILE))"
    else
        echo "服务启动失败，日志:"
        tail -20 /tmp/tts_daemon.log
        rm -f "$PIDFILE"
        return 1
    fi
}

stop() {
    if [ ! -f "$PIDFILE" ]; then
        echo "服务未运行 (无 PID 文件)"
        pkill -f "$DAEMON" 2>/dev/null
        rm -f "$PIPE"
        return 0
    fi

    pid=$(cat "$PIDFILE")
    echo "停止 TTS 常驻服务 (PID $pid)..."
    kill $pid 2>/dev/null
    sleep 1
    rm -f "$PIDFILE" "$PIPE"
    echo "已停止"
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "TTS 常驻服务: 运行中"
        echo "  PID: $(cat $PIDFILE)"
        echo "  管道: $PIPE"
        echo "  输出: $OUTDIR/tts_out_*.wav"
        echo "  日志: /tmp/tts_daemon.log"
    else
        echo "TTS 常驻服务: 未运行"
    fi
}

speak() {
    local text="$1"
    if [ -z "$text" ]; then
        echo "请提供要合成的文本"
        echo "用法: $0 speak <text>"
        return 1
    fi

    # 如果服务没在运行，自动启动
    if [ ! -f "$PIDFILE" ] || ! kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        start
    fi

    echo "合成: $text"
    echo "$text" > "$PIPE"

    # 等待新文件出现
    sleep 1
    local latest
    latest=$(ls -t $OUTDIR/tts_out_*.wav 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        sleep 2  # 等文件写完成
        echo "播放: $latest"
        aplay -D "$AUDIO_DEV" "$latest" 2>/dev/null || \
        aplay "$latest" 2>/dev/null || \
        echo "播放失败，请检查音响设备"
    fi
}

speakf() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "文件不存在: $file"
        return 1
    fi
    local text
    text=$(tr -d '\n' < "$file")
    speak "$text"
}

case "${1:-}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    status)
        status
        ;;
    speak)
        speak "${@:2}"
        ;;
    speakf)
        speakf "$2"
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|speak <text>|speakf <file>}"
        echo ""
        echo "示例:"
        echo "  $0 start"
        echo "  $0 speak '你好世界，这是一段测试语音'"
        echo "  $0 speakf /tmp/text.txt"
        echo "  $0 stop"
        exit 1
        ;;
esac