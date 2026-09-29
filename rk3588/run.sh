#!/bin/sh
#====================================================================
# 语音助手启动脚本
# 自动设置 LD_LIBRARY_PATH 指向 sherpa-onnx bundled libstdc++
# 自动检测并启动依赖服务（RAG、LLM、SenseVoice SER）
#
# 用法:
#   ./run.sh                    # 正常启动
#   ./run.sh --verbose          # 显示详细环境信息
#   ./run.sh --help             # 显示帮助
#   ./run.sh <python_args>      # 传递参数给 streaming_vad.py
#====================================================================

SCRIPT_DIR="/userdata/voice_assistant"
SHERPA_ONNX_LIB="/userdata/sherpa-onnx/install/lib"
MAIN_SCRIPT="${SCRIPT_DIR}/streaming_vad.py"

# === 依赖服务配置 ===
RAG_SCRIPT="${SCRIPT_DIR}/patient_rag_server.py"
RAG_SOCK="/tmp/patient_rag.sock"

LLM_SCRIPT="${SCRIPT_DIR}/qwen3_llm_streaming_service.py"
LLM_SOCK="/tmp/qwen3_llm.sock"

# sherpa-onnx Matcha TTS 常驻服务配置
TTS_SCRIPT="${SCRIPT_DIR}/sherpa_tts_service.py"
TTS_PIPE="/tmp/tts_pipe"
TTS_PID_FILE="/tmp/sherpa_tts.pid"

SER_SCRIPT="/data/sensevoice/sensevoice_server.py"
SER_SOCK="/tmp/sensevoice_server.sock"
SER_ARGS="--model /data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn --socket ${SER_SOCK}"

# 服务启动超时（秒）
SERVICE_TIMEOUT=120

# --- 解析参数 ---
VERBOSE=0
PY_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --verbose|-v)
            VERBOSE=1
            ;;
        --help|-h)
            echo "用法: $0 [选项] [-- python_args]"
            echo ""
            echo "选项:"
            echo "  --verbose, -v    显示详细的环境诊断信息"
            echo "  --help, -h       显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  LD_LIBRARY_PATH  当前: ${LD_LIBRARY_PATH:-<未设置>}"
            echo ""
            echo "说明:"
            echo "  自动设置 LD_LIBRARY_PATH=${SHERPA_ONNX_LIB}"
            echo "  确保 sherpa-onnx 的 bundled libstdc++ 被正确加载"
            echo "  自动初始化音频设备（音量 40%）"
            echo "  自动启动依赖服务：RAG / LLM / sherpa-onnx Matcha TTS / SenseVoice SER"
            exit 0
            ;;
        --)
            shift
            PY_ARGS="$@"
            break
            ;;
        *)
            PY_ARGS="$@"
            break
            ;;
    esac
    shift
done

# --- 设置 LD_LIBRARY_PATH ---
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="${SHERPA_ONNX_LIB}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${SHERPA_ONNX_LIB}"
fi

# --- 辅助函数 ---

# 检查 socket 是否可用（能连上）
check_socket() {
    [ -S "$1" ] || return 1
    python3 -c "import socket; s=socket.socket(socket.AF_UNIX); s.settimeout(1.5); s.connect('$1'); s.close()" 2>/dev/null
}

# 验证 socket 是否存在（不检查连通性）
verify_socket() {
    [ -S "$1" ]
}

# 杀死匹配名称的残留进程
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
            echo ""
            echo "  ✅ ${name} 服务已就绪 (${waited}s)"
            return 0
        fi
        waited=$((waited + 1))
        if [ $((waited % 5)) -eq 0 ]; then
            echo "  ⏳ ${name} 加载中 (${waited}s) ..."
        fi
        sleep 1
    done
    echo ""
    echo "  ❌ ${name} 服务启动超时 (${SERVICE_TIMEOUT}s)"
    return 1
}

# 启动一个后台服务
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
    # 提取脚本路径（去掉参数）用于日志文件名
    local script_path="${script%% *}"
    nohup python3 $script > "/tmp/$(basename $script_path .py).log" 2>&1 &
    local pid=$!
    echo "  PID: ${pid}"

    wait_socket "$sock" "$name"
    return $?
}

# --- 启动所有依赖服务 ---
echo ""
echo "=========================================="
echo " 检查并启动依赖服务..."
echo "=========================================="

# 1. Patient RAG 服务
kill_stale "patient_rag"
start_service "$RAG_SCRIPT" "$RAG_SOCK" "RAG"

# 2. LLM 服务
kill_stale "qwen3_llm"
start_service "$LLM_SCRIPT" "$LLM_SOCK" "LLM"
LLM_OK=$?

# 3. sherpa-onnx Matcha TTS 常驻服务
if [ -f "$TTS_PID_FILE" ] && kill -0 $(cat "$TTS_PID_FILE") 2>/dev/null; then
    echo "  ✅ sherpa-onnx Matcha TTS 常驻服务已就绪 (PID $(cat $TTS_PID_FILE))"
    TTS_OK=0
else
    echo "  🔄 启动 sherpa-onnx Matcha TTS 常驻服务..."
    kill_stale "sherpa_tts_service"
    kill_stale "sherpa_tts_daemon"
    rm -f "$TTS_PIPE"
    nohup python3 "$TTS_SCRIPT" > /tmp/sherpa_tts_service.log 2>&1 &
    TTS_PID=$!
    echo "  PID: ${TTS_PID}"
    # 等待 pipe 就绪（C daemon 创建 pipe 后即就绪）
    waited=0
    while [ $waited -lt $SERVICE_TIMEOUT ]; do
        if [ -p "$TTS_PIPE" ] && [ -f "$TTS_PID_FILE" ] && kill -0 $(cat "$TTS_PID_FILE") 2>/dev/null; then
            echo "  ✅ sherpa-onnx Matcha TTS 常驻服务已就绪 (${waited}s)"
            TTS_OK=0
            break
        fi
        waited=$((waited + 1))
        if [ $((waited % 5)) -eq 0 ]; then
            echo "  ⏳ TTS 加载中 (${waited}s) ..."
        fi
        sleep 1
    done
    if [ $TTS_OK -ne 0 ]; then
        echo "  ❌ TTS 服务启动超时"
        TTS_OK=1
    fi
fi

# 4. SenseVoice SER 服务
kill_stale "sensevoice_server"
start_service "$SER_SCRIPT $SER_ARGS" "$SER_SOCK" "SenseVoice SER"
SER_OK=$?

echo "=========================================="
echo ""

# 如果有服务启动失败，仅警告不阻塞
if [ $LLM_OK -ne 0 ] || [ $TTS_OK -ne 0 ] || [ $SER_OK -ne 0 ]; then
    echo "  ⚠️  部分服务启动失败："
    [ $LLM_OK -ne 0 ] && echo "     - LLM: 失败"
    [ $TTS_OK -ne 0 ] && echo "     - TTS: 失败"
    [ $SER_OK -ne 0 ] && echo "     - SER: 失败"
    echo ""
fi

# --- 音频设备初始化 ---
echo "🔊 初始化音频设备..."
amixer set Master 40% 2>/dev/null && echo "  音量设置为 40%"
echo ""

# --- 诊断信息 ---
if [ "$VERBOSE" -eq 1 ]; then
    echo "=========================================="
    echo " 语音助手 - 环境诊断"
    echo "=========================================="
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo ""

    echo "--- libstdc++ 版本检查 ---"
    if [ -f "${SHERPA_ONNX_LIB}/libstdc++.so.6" ]; then
        REAL=$(readlink -f "${SHERPA_ONNX_LIB}/libstdc++.so.6")
        echo "  bundled: ${REAL}"
        strings "${REAL}" 2>/dev/null | grep GLIBCXX | tail -5 | sed 's/^/    /'
    else
        echo "  bundled: NOT FOUND"
    fi
    SYS_LIBSTD="/lib/libstdc++.so.6"
    if [ -f "$SYS_LIBSTD" ]; then
        SYS_REAL=$(readlink -f "$SYS_LIBSTD")
        echo "  system:  ${SYS_REAL}"
        strings "${SYS_REAL}" 2>/dev/null | grep GLIBCXX | tail -5 | sed 's/^/    /'
    fi
    echo ""

    echo "--- sherpa-onnx 库检查 ---"
    for lib in libsherpa-onnx-c-api.so libstdc++.so.6; do
        if [ -f "${SHERPA_ONNX_LIB}/${lib}" ]; then
            echo "  ✅ ${lib}"
        else
            echo "  ❌ ${lib} (NOT FOUND)"
        fi
    done
    echo ""

    echo "--- 主脚本检查 ---"
    if [ -f "$MAIN_SCRIPT" ]; then
        echo "  ✅ ${MAIN_SCRIPT}"
    else
        echo "  ❌ ${MAIN_SCRIPT} (NOT FOUND)"
    fi
    echo "=========================================="
    echo ""
fi

# --- 释放被占用的音频设备 ---
echo "🔄 释放音频设备（清理 pulseaudio）..."
pkill -9 -f pulseaudio 2>/dev/null
sleep 0.3

# --- 启动主程序 ---
echo "🚀 启动语音助手主程序..."
echo ""
exec python3 ${MAIN_SCRIPT} ${PY_ARGS}