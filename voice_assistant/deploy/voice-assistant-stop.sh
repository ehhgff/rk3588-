#!/bin/bash
# Voice Assistant 服务停止脚本
# 由systemd在ExecStop阶段调用

PID_FILE="/var/run/voice-assistant.pid"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    logger -t voice-assistant "$1"
}

log "=== 停止 Voice Assistant 服务 ==="

# 1. 读取PID
if [ -f $PID_FILE ]; then
    PIDS=$(cat $PID_FILE)
    log "找到服务PID: $PIDS"
    
    # 优雅停止
    for PID in $PIDS; do
        if kill -0 $PID 2>/dev/null; then
            log "停止进程 $PID..."
            kill -TERM $PID 2>/dev/null || true
        fi
    done
    
    # 等待进程退出（最多3秒）
    for i in {1..30}; do
        all_dead=true
        for PID in $PIDS; do
            if kill -0 $PID 2>/dev/null; then
                all_dead=false
                break
            fi
        done
        if [ "$all_dead" = true ]; then
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

# 2. 清理相关进程
log "清理Python服务进程..."
pkill -9 -f "rag_optimized_server" 2>/dev/null || true
pkill -9 -f "qwen3_llm_service" 2>/dev/null || true
pkill -9 -f "melotts_service_rknn" 2>/dev/null || true

# 3. 清理socket文件
log "清理socket文件..."
rm -f /tmp/rag_optimized.sock
rm -f /tmp/qwen3_llm.sock
rm -f /tmp/melotts_service.sock
rm -f /tmp/service_ready/*

log "服务已停止"
exit 0
