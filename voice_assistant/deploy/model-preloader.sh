#!/bin/bash
# 模型预加载守护进程
# 功能：系统启动时将模型文件预加载到内存缓存，加速服务启动

LOG_FILE="/tmp/model-preloader.log"
PID_FILE="/var/run/model-preloader.pid"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# 检查是否以root运行
if [ "$EUID" -ne 0 ]; then
    echo "请以root权限运行"
    exit 1
fi

# 模型文件列表
MODEL_FILES=(
    "/data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm"
    "/data/qwen3/librkllmrt.so"
    "/data/zipformer/model/encoder-epoch-99-avg-1.rknn"
    "/data/zipformer/model/decoder-epoch-99-avg-1.rknn"
    "/data/zipformer/model/joiner-epoch-99-avg-1.rknn"
    "/data/melotts_deploy/model/encoder-ZH_MIX_EN.rknn"
    "/data/melotts_deploy/model/decoder-ZH_MIX_EN.rknn"
)

# 预加载函数
preload_models() {
    log "=== 开始预加载模型文件 ==="
    
    for model in "${MODEL_FILES[@]}"; do
        if [ -f "$model" ]; then
            log "预加载: $model"
            
            # 方法1: 使用cat读取到/dev/null（简单有效）
            cat "$model" > /dev/null 2>&1
            
            # 方法2: 如果vmtouch可用，使用vmtouch锁定到内存
            if command -v vmtouch >/dev/null 2>&1; then
                vmtouch -vt "$model" >> $LOG_FILE 2>&1
            fi
            
            # 方法3: 使用dd读取（兼容性最好）
            # dd if="$model" of=/dev/null bs=1M 2>/dev/null
        else
            log "警告: 模型文件不存在: $model"
        fi
    done
    
    log "=== 模型预加载完成 ==="
}

# 守护进程模式
daemon_mode() {
    log "启动模型预加载守护进程"
    
    # 保存PID
    echo $$ > $PID_FILE
    
    # 首次预加载
    preload_models
    
    # 守护循环：定期检查并重新加载（防止被swap出去）
    while true; do
        sleep 300  # 每5分钟检查一次
        
        # 检查内存压力
        mem_available=$(free -m | awk '/^Mem:/{print $7}')
        if [ "$mem_available" -gt "1024" ]; then
            # 内存充足，重新预加载
            log "内存充足(${mem_available}MB)，重新预加载模型..."
            preload_models
        else
            log "内存紧张(${mem_available}MB)，跳过预加载"
        fi
    done
}

# 单次预加载模式
single_mode() {
    log "执行单次模型预加载"
    preload_models
}

# 停止守护进程
stop_daemon() {
    if [ -f $PID_FILE ]; then
        PID=$(cat $PID_FILE)
        if kill -0 $PID 2>/dev/null; then
            log "停止守护进程 $PID"
            kill $PID
            rm -f $PID_FILE
        else
            log "守护进程未运行"
            rm -f $PID_FILE
        fi
    else
        log "PID文件不存在"
    fi
}

# 状态检查
status() {
    if [ -f $PID_FILE ]; then
        PID=$(cat $PID_FILE)
        if kill -0 $PID 2>/dev/null; then
            log "守护进程运行中 (PID: $PID)"
            return 0
        else
            log "守护进程未运行"
            return 1
        fi
    else
        log "守护进程未运行"
        return 1
    fi
}

# 主程序
case "${1:-single}" in
    start|daemon)
        if status >/dev/null 2>&1; then
            log "守护进程已在运行"
            exit 0
        fi
        daemon_mode
        ;;
    single|once)
        single_mode
        ;;
    stop)
        stop_daemon
        ;;
    status)
        status
        ;;
    restart)
        stop_daemon
        sleep 1
        daemon_mode
        ;;
    *)
        echo "用法: $0 {start|single|stop|status|restart}"
        echo "  start   - 启动守护进程模式"
        echo "  single  - 单次预加载（默认）"
        echo "  stop    - 停止守护进程"
        echo "  status  - 查看状态"
        echo "  restart - 重启守护进程"
        exit 1
        ;;
esac
