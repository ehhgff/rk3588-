#!/bin/bash
# Voice Assistant 服务预启动脚本
# 由systemd在ExecStartPre阶段调用

LOG_FILE="/tmp/voice-assistant-pre.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "=== Voice Assistant 预启动 ==="

# 1. 确保Swap已启用
if ! swapon -s | grep -q /swapfile; then
    log "启用Swap..."
    if [ -f /swapfile ]; then
        swapon /swapfile 2>/dev/null || log "警告: Swap启用失败"
    else
        log "警告: Swap文件不存在"
    fi
fi

# 2. 设置swappiness
if [ -f /proc/sys/vm/swappiness ]; then
    current=$(cat /proc/sys/vm/swappiness)
    if [ "$current" -gt "10" ]; then
        echo 10 > /proc/sys/vm/swappiness
        log "设置vm.swappiness=10"
    fi
fi

# 3. 清理旧的socket文件
log "清理旧socket文件..."
rm -f /tmp/*.sock 2>/dev/null
rm -f /tmp/service_ready/* 2>/dev/null

# 4. 检查必要目录
for dir in /userdata/voice_assistant /userdata/medical_rag_full /data/voice_assistant /data/qwen3; do
    if [ ! -d "$dir" ]; then
        log "错误: 目录不存在 $dir"
        exit 1
    fi
done

# 5. 检查关键文件
CRITICAL_FILES=(
    "/data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm"
    "/userdata/voice_assistant/qwen3_llm_service.py"
    "/userdata/medical_rag_full/rag_optimized_server.py"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        log "错误: 关键文件不存在 $file"
        exit 1
    fi
done

# 6. 内存检查
mem_available=$(free -m | awk '/^Mem:/{print $7}')
if [ "$mem_available" -lt "512" ]; then
    log "警告: 可用内存不足 ${mem_available}MB，建议至少512MB"
fi

log "预启动检查完成"
exit 0
