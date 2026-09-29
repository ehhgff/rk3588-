#!/bin/bash
# 设置语音助手快速启动 - 开机自动启动 + 常驻内存

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1"
}

# 1. 安装快速启动服务脚本
log_info "安装快速启动服务脚本..."
cp "$SCRIPT_DIR/fast-boot-service.sh" /usr/local/bin/
chmod +x /usr/local/bin/fast-boot-service.sh

# 2. 创建快捷命令
cat > /usr/local/bin/va-fast-start << 'EOF'
#!/bin/bash
/usr/local/bin/fast-boot-service.sh start
EOF
chmod +x /usr/local/bin/va-fast-start

cat > /usr/local/bin/va-fast-stop << 'EOF'
#!/bin/bash
/usr/local/bin/fast-boot-service.sh stop
EOF
chmod +x /usr/local/bin/va-fast-stop

cat > /usr/local/bin/va-pause << 'EOF'
#!/bin/bash
/usr/local/bin/fast-boot-service.sh pause
EOF
chmod +x /usr/local/bin/va-pause

cat > /usr/local/bin/va-resume << 'EOF'
#!/bin/bash
/usr/local/bin/fast-boot-service.sh resume
EOF
chmod +x /usr/local/bin/va-resume

# 3. 配置开机自动启动
log_info "配置开机自动启动..."

# 创建启动脚本
STARTUP_SCRIPT="/etc/init.d/voice-assistant-fast"
cat > $STARTUP_SCRIPT << 'EOF'
#!/bin/bash
# Voice Assistant Fast Boot Service

case "$1" in
    start)
        echo "Starting Voice Assistant Fast Boot Service..."
        # 等待系统完全启动
        sleep 5
        /usr/local/bin/fast-boot-service.sh start
        ;;
    stop)
        echo "Stopping Voice Assistant Fast Boot Service..."
        /usr/local/bin/fast-boot-service.sh stop
        ;;
    restart)
        $0 stop
        sleep 1
        $0 start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac
EOF
chmod +x $STARTUP_SCRIPT

# 添加到rc.local（如果不存在）
if [ -f /etc/rc.local ]; then
    if ! grep -q "voice-assistant-fast" /etc/rc.local; then
        # 在exit 0之前添加启动命令
        sed -i '/^exit 0/i\/etc/init.d/voice-assistant-fast start' /etc/rc.local
        log_info "已添加到 /etc/rc.local"
    fi
else
    # 创建rc.local
    cat > /etc/rc.local << 'EOF'
#!/bin/bash
# rc.local for Voice Assistant

# 启动语音助手快速服务
/etc/init.d/voice-assistant-fast start

exit 0
EOF
    chmod +x /etc/rc.local
    log_info "已创建 /etc/rc.local"
fi

# 4. 创建内存保护脚本（防止服务被OOM killer终止）
log_info "创建内存保护脚本..."
cat > /usr/local/bin/va-memory-protect.sh << 'EOF'
#!/bin/bash
# 保护语音助手服务不被OOM killer终止

PID_FILE="/var/run/voice-assistant.pid"

check_and_protect() {
    if [ -f $PID_FILE ]; then
        PIDS=$(cat $PID_FILE 2>/dev/null)
        for PID in $PIDS; do
            if kill -0 $PID 2>/dev/null; then
                # 设置OOM分数为-1000（永不终止）
                echo -1000 > /proc/$PID/oom_score_adj 2>/dev/null || true
            fi
        done
    fi
}

# 每分钟检查一次
while true; do
    check_and_protect
    sleep 60
done
EOF
chmod +x /usr/local/bin/va-memory-protect.sh

# 5. 添加到crontab（开机启动内存保护）
if ! grep -q "va-memory-protect" /etc/crontab 2>/dev/null; then
    echo "@reboot root /usr/local/bin/va-memory-protect.sh > /dev/null 2>&1 &" >> /etc/crontab
    log_info "已添加内存保护到crontab"
fi

# 6. 测试启动
log_info "测试快速启动..."
/usr/local/bin/fast-boot-service.sh status

echo ""
echo "========================================"
echo "    快速启动方案安装完成"
echo "========================================"
echo ""
echo "使用方法:"
echo "  va-fast-start  - 快速启动服务"
echo "  va-fast-stop   - 停止服务"
echo "  va-pause       - 暂停服务（内存常驻）"
echo "  va-resume      - 恢复服务"
echo ""
echo "开机自动启动:"
echo "  已配置到 /etc/rc.local"
echo "  开机后5秒自动启动服务"
echo ""
echo "内存保护:"
echo "  已启用OOM保护，防止服务被系统终止"
echo ""
echo "预期效果:"
echo "  重启后服务自动启动，约5-6秒完成"
echo "  用户无需等待，开机即可使用"
echo "========================================"
