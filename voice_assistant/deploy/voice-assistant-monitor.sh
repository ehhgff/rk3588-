#!/bin/bash
# Voice Assistant 服务监控守护进程
# 功能：监控服务状态，崩溃时自动重启

PID_FILE="/var/run/voice-assistant-monitor.pid"
LOG_FILE="/tmp/voice-assistant-monitor.log"
CHECK_INTERVAL=5  # 检查间隔（秒）
MAX_RESTARTS=3    # 最大连续重启次数
RESTART_WINDOW=60 # 重启窗口（秒）

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# 记录重启历史
RESTART_HISTORY="/tmp/va_restart_history"

# 检查是否需要重启
check_restart_limit() {
    local now=$(date +%s)
    local count=0
    
    # 清理过期的重启记录
    if [ -f $RESTART_HISTORY ]; then
        while read -r timestamp; do
            if [ $((now - timestamp)) -lt $RESTART_WINDOW ]; then
                count=$((count + 1))
            fi
        done < $RESTART_HISTORY
    fi
    
    # 记录本次重启
    echo $now >> $RESTART_HISTORY
    
    # 检查是否超过限制
    if [ $count -ge $MAX_RESTARTS ]; then
        log "错误: 连续重启次数超过限制($MAX_RESTARTS次/$RESTART_WINDOW秒)"
        log "停止自动重启，请检查服务配置"
        return 1
    fi
    
    return 0
}

# 检查服务健康状态
check_health() {
    local healthy=true
    
    # 检查socket文件
    if [ ! -S "/tmp/rag_optimized.sock" ]; then
        log "健康检查失败: RAG socket不存在"
        healthy=false
    fi
    
    if [ ! -S "/tmp/qwen3_llm.sock" ]; then
        log "健康检查失败: LLM socket不存在"
        healthy=false
    fi
    
    if [ ! -S "/tmp/melotts_service.sock" ]; then
        log "健康检查失败: TTS socket不存在"
        healthy=false
    fi
    
    # 检查进程
    local VA_DIR="/userdata/voice_assistant"
    if ! ps | grep -q "rag_optimized_server"; then
        log "健康检查失败: RAG进程不存在"
        healthy=false
    fi
    
    if ! ps | grep -q "qwen3_llm_service"; then
        log "健康检查失败: LLM进程不存在"
        healthy=false
    fi
    
    if ! ps | grep -q "melotts_service_rknn"; then
        log "健康检查失败: TTS进程不存在"
        healthy=false
    fi
    
    if [ "$healthy" = true ]; then
        return 0
    else
        return 1
    fi
}

# 启动监控
start_monitor() {
    log "=== 启动 Voice Assistant 监控守护进程 ==="
    
    # 检查是否已在运行
    if [ -f $PID_FILE ]; then
        local old_pid=$(cat $PID_FILE)
        if kill -0 $old_pid 2>/dev/null; then
            log "监控进程已在运行 (PID: $old_pid)"
            return 0
        fi
    fi
    
    # 保存PID
    echo $$ > $PID_FILE
    
    log "监控进程已启动 (PID: $$)"
    log "检查间隔: ${CHECK_INTERVAL}秒"
    log "最大重启次数: ${MAX_RESTARTS}次/${RESTART_WINDOW}秒"
    
    # 监控循环
    while true; do
        if ! check_health; then
            log "检测到服务异常"
            
            if check_restart_limit; then
                log "正在重启服务..."
                /usr/local/bin/voice-assistant-service.sh restart
                
                # 等待服务启动
                sleep 10
                
                if check_health; then
                    log "服务重启成功"
                else
                    log "服务重启失败"
                fi
            else
                log "重启次数超限，停止监控"
                break
            fi
        fi
        
        sleep $CHECK_INTERVAL
    done
    
    rm -f $PID_FILE
    log "监控进程已退出"
}

# 停止监控
stop_monitor() {
    if [ -f $PID_FILE ]; then
        local pid=$(cat $PID_FILE)
        if kill -0 $pid 2>/dev/null; then
            log "停止监控进程 (PID: $pid)"
            kill $pid
            rm -f $PID_FILE
        else
            log "监控进程未运行"
            rm -f $PID_FILE
        fi
    else
        log "监控进程未运行"
    fi
}

# 查看监控状态
status_monitor() {
    if [ -f $PID_FILE ]; then
        local pid=$(cat $PID_FILE)
        if kill -0 $pid 2>/dev/null; then
            echo "监控进程: 运行中 (PID: $pid)"
            echo "日志文件: $LOG_FILE"
            echo ""
            echo "最近日志:"
            tail -10 $LOG_FILE
        else
            echo "监控进程: 未运行"
            rm -f $PID_FILE
        fi
    else
        echo "监控进程: 未运行"
    fi
}

# 主程序
case "${1:-start}" in
    start)
        # 后台启动监控
        start_monitor &
        echo "监控进程已在后台启动"
        ;;
    stop)
        stop_monitor
        ;;
    status)
        status_monitor
        ;;
    restart)
        stop_monitor
        sleep 1
        start_monitor &
        echo "监控进程已重启"
        ;;
    run)
        # 前台运行模式（用于调试）
        start_monitor
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart|run}"
        echo ""
        echo "Commands:"
        echo "  start    - 后台启动监控"
        echo "  stop     - 停止监控"
        echo "  status   - 查看监控状态"
        echo "  restart  - 重启监控"
        echo "  run      - 前台运行（调试用）"
        exit 1
        ;;
esac
