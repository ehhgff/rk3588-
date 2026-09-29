#!/bin/bash
#====================================================================
# 语音助手依赖服务启动脚本
# 启动所有后台服务：RAG、LLM、Paroli TTS、SenseVoice SER
#
# 用法:
#   ./start_services.sh              # 启动所有服务
#   ./start_services.sh --status     # 查看服务状态
#   ./start_services.sh --stop       # 停止所有服务
#====================================================================

VOICE_ASSISTANT_DIR="/userdata/voice_assistant"
SENSEVOICE_DIR="/data/sensevoice"
SERVICE_TIMEOUT=120

# Socket 路径
RAG_SOCK="/tmp/patient_rag.sock"
LLM_SOCK="/tmp/qwen3_llm.sock"
TTS_SOCK="/tmp/paroli_tts.sock"
SER_SOCK="/tmp/sensevoice_server.sock"

# 服务脚本
RAG_SCRIPT="${VOICE_ASSISTANT_DIR}/patient_rag_server.py"
LLM_SCRIPT="${VOICE_ASSISTANT_DIR}/qwen3_llm_streaming_service.py"
TTS_SCRIPT="${VOICE_ASSISTANT_DIR}/paroli_tts_service.py"
SER_SCRIPT="${SENSEVOICE_DIR}/sensevoice_server.py"
SER_ARGS="--model ${SENSEVOICE_DIR}/sensevoice_encoder_ctc_100f_fp16.rknn --socket ${SER_SOCK}"

# --- 辅助函数 ---

check_socket() {
    [ -S "$1" ] || return 1
    python3 -c "import socket; s=socket.socket(socket.AF_UNIX); s.settimeout(1.5); s.connect('$1'); s.close()" 2>/dev/null
}

kill_stale() {
    local pattern="$1"
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "  ⚠️  发现残留 $pattern 进程 (PID: $(echo $pids | tr '\n' ' '))，正在清理..."
        kill $pids 2>/dev/null
        sleep 1
        pids=$(pgrep -f "$pattern" 2>/dev/null)
        [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    fi
}

wait_socket() {
    local sock="$1"
    local name="$2"
    local waited=0
    while [ $waited -lt $SERVICE_TIMEOUT ]; do
        if check_socket "$sock"; then
            echo "  ✅ ${name} 服务已就绪 (${waited}s)"
            return 0
        fi
        waited=$((waited + 1))
        if [ $((waited % 5)) -eq 0 ]; then
            echo "  ⏳ ${name} 加载中 (${waited}s) ..."
        fi
        sleep 1
    done
    echo "  ❌ ${name} 服务启动超时 (${SERVICE_TIMEOUT}s)"
    return 1
}

start_service() {
    local script="$1"
    local sock="$2"
    local name="$3"

    if check_socket "$sock"; then
        echo "  ✅ ${name} 服务已在运行"
        return 0
    fi

    echo "  🔄 启动 ${name} 服务..."
    rm -f "$sock"
    local script_path="${script%% *}"
    nohup python3 $script > "/tmp/$(basename $script_path .py).log" 2>&1 &
    local pid=$!
    echo "  PID: ${pid}"

    wait_socket "$sock" "$name"
    return $?
}

stop_service() {
    local pattern="$1"
    local sock="$2"
    local name="$3"
    echo "  🛑 停止 ${name} 服务..."
    kill_stale "$pattern"
    rm -f "$sock"
}

show_status() {
    echo "=== 服务状态 ==="
    for entry in "RAG:${RAG_SOCK}" "LLM:${LLM_SOCK}" "Paroli TTS:${TTS_SOCK}" "SenseVoice SER:${SER_SOCK}"; do
        local name="${entry%%:*}"
        local sock="${entry##*:}"
        if check_socket "$sock"; then
            echo "  ✅ ${name}: 运行中 (${sock})"
        else
            echo "  ❌ ${name}: 未运行 (${sock})"
        fi
    done
}

# --- 主逻辑 ---

case "${1:-start}" in
    start)
        echo ""
        echo "=========================================="
        echo " 启动语音助手依赖服务"
        echo "=========================================="

        # 1. RAG 服务
        echo "[1/4] Patient RAG 服务..."
        kill_stale "patient_rag"
        start_service "$RAG_SCRIPT" "$RAG_SOCK" "RAG"

        # 2. LLM 服务
        echo "[2/4] Qwen3 LLM 服务..."
        kill_stale "qwen3_llm"
        start_service "$LLM_SCRIPT" "$LLM_SOCK" "LLM"

        # 3. Paroli TTS 服务
        echo "[3/4] Paroli TTS 服务（C++ 常驻进程，模型常驻内存）..."
        kill_stale "paroli_tts_service"
        kill_stale "paroli-socket-server"
        start_service "$TTS_SCRIPT" "$TTS_SOCK" "Paroli TTS"

        # 4. SenseVoice SER 服务
        echo "[4/4] SenseVoice SER 服务..."
        kill_stale "sensevoice_server"
        start_service "$SER_SCRIPT $SER_ARGS" "$SER_SOCK" "SenseVoice SER"

        echo ""
        echo "=========================================="
        echo " 所有服务启动完成"
        echo "=========================================="
        show_status
        echo ""
        ;;

    stop)
        echo ""
        echo "=== 停止所有服务 ==="
        stop_service "sensevoice_server" "$SER_SOCK" "SenseVoice SER"
        stop_service "paroli_tts_service" "$TTS_SOCK" "Paroli TTS"
        stop_service "qwen3_llm" "$LLM_SOCK" "LLM"
        stop_service "patient_rag" "$RAG_SOCK" "RAG"
        echo "  ✅ 所有服务已停止"
        echo ""
        ;;

    status)
        show_status
        ;;

    restart)
        $0 stop
        sleep 1
        $0 start
        ;;

    *)
        echo "用法: $0 {start|stop|status|restart}"
        echo ""
        echo "   start    启动所有服务（默认）"
        echo "   stop     停止所有服务"
        echo "   status   查看服务状态"
        echo "   restart  重启所有服务"
        exit 1
        ;;
esac