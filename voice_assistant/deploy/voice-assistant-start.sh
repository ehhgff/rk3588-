#!/bin/bash
# Voice Assistant 服务启动脚本（systemd 入口）
# 由 systemd 在 ExecStart 阶段调用
# 启动所有依赖服务后，保持守护状态

PID_FILE="/var/run/voice-assistant.pid"
READY_DIR="/tmp/service_ready"
mkdir -p $READY_DIR
rm -f $READY_DIR/*

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    logger -t voice-assistant "$1"
}

log "=== 启动 Voice Assistant 服务 ==="

START_TIME=$(date +%s%N)

# Socket 路径
RAG_SOCK="/tmp/patient_rag.sock"
LLM_SOCK="/tmp/qwen3_llm.sock"
TTS_SOCK="/tmp/paroli_tts.sock"
SER_SOCK="/tmp/sensevoice_server.sock"

# 服务脚本
VOICE_ASSISTANT_DIR="/userdata/voice_assistant"
SENSEVOICE_DIR="/data/sensevoice"

RAG_SCRIPT="${VOICE_ASSISTANT_DIR}/patient_rag_server.py"
LLM_SCRIPT="${VOICE_ASSISTANT_DIR}/qwen3_llm_streaming_service.py"
TTS_SCRIPT="${VOICE_ASSISTANT_DIR}/paroli_tts_service.py"
SER_SCRIPT="${SENSEVOICE_DIR}/sensevoice_server.py"
SER_ARGS="--model ${SENSEVOICE_DIR}/sensevoice_encoder_ctc_100f_fp16.rknn --socket ${SER_SOCK}"

# --- 1. RAG 服务 ---
log "启动 RAG 服务..."
cd /userdata/medical_rag_full
nohup python3 rag_optimized_server.py > /tmp/rag_optimized_server.log 2>&1 &
RAG_PID=$!
log "RAG 服务 PID: $RAG_PID"

(
    for i in {1..100}; do
        if [ -S "$RAG_SOCK" ]; then
            touch $READY_DIR/rag
            log "RAG 服务就绪"
            break
        fi
        sleep 0.05
    done
) &

# --- 2. LLM 服务 ---
log "启动 LLM 服务..."
cd $VOICE_ASSISTANT_DIR
export LD_LIBRARY_PATH=/data/qwen3:$LD_LIBRARY_PATH
nohup python3 qwen3_llm_streaming_service.py > /tmp/qwen3.log 2>&1 &
LLM_PID=$!
log "LLM 服务 PID: $LLM_PID"

(
    for i in {1..200}; do
        if [ -S "$LLM_SOCK" ]; then
            if python3 qwen3_llm_client.py "ping" > /dev/null 2>&1; then
                touch $READY_DIR/llm
                log "LLM 服务就绪"
                break
            fi
        fi
        sleep 0.1
    done
) &

# --- 3. Paroli TTS 服务 ---
log "启动 Paroli TTS 服务..."
cd $VOICE_ASSISTANT_DIR
nohup python3 paroli_tts_service.py > /tmp/paroli_tts_service.log 2>&1 &
TTS_PID=$!
log "Paroli TTS 服务 PID: $TTS_PID"

(
    for i in {1..60}; do
        if [ -S "$TTS_SOCK" ]; then
            touch $READY_DIR/tts
            log "Paroli TTS 服务就绪"
            break
        fi
        sleep 0.1
    done
) &

# --- 4. SenseVoice SER 服务 ---
log "启动 SenseVoice SER 服务..."
cd $SENSEVOICE_DIR
nohup python3 sensevoice_server.py $SER_ARGS > /tmp/sensevoice_server.log 2>&1 &
SER_PID=$!
log "SenseVoice SER 服务 PID: $SER_PID"

(
    for i in {1..60}; do
        if [ -S "$SER_SOCK" ]; then
            touch $READY_DIR/ser
            log "SenseVoice SER 服务就绪"
            break
        fi
        sleep 0.1
    done
) &

# --- 保存 PID ---
echo "$RAG_PID $LLM_PID $TTS_PID $SER_PID" > $PID_FILE

# --- 等待所有服务就绪（最多 20 秒）---
log "等待所有服务就绪..."
for i in {1..200}; do
    if [ -f $READY_DIR/rag ] && [ -f $READY_DIR/llm ] && [ -f $READY_DIR/tts ] && [ -f $READY_DIR/ser ]; then
        break
    fi

    # 检查进程存活
    for pid_entry in "$RAG_PID:RAG" "$LLM_PID:LLM" "$TTS_PID:TTS" "$SER_PID:SER"; do
        local pid="${pid_entry%%:*}"
        local name="${pid_entry##*:}"
        if ! kill -0 $pid 2>/dev/null; then
            log "错误: ${name} 服务异常退出"
            exit 1
        fi
    done

    sleep 0.1
done

# --- 音频初始化 ---
log "初始化音频设备..."
amixer set Master 40% 2>/dev/null && log "音量设置为 40%"

# --- 计算启动时间 ---
END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
log "所有服务已就绪，总启动时间: ${ELAPSED_MS}ms"

# --- 守护状态（systemd Type=forking 需要）---
log "服务启动完成，进入守护状态"
while true; do
    if ! kill -0 $RAG_PID 2>/dev/null || \
       ! kill -0 $LLM_PID 2>/dev/null || \
       ! kill -0 $TTS_PID 2>/dev/null || \
       ! kill -0 $SER_PID 2>/dev/null; then
        log "错误: 有服务进程退出"
        exit 1
    fi
    sleep 5
done