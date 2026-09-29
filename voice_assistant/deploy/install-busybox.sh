#!/bin/bash
# Voice Assistant BusyBox 环境安装脚本
# 替代 systemd 的方案

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VA_DIR="/userdata/voice_assistant"

echo "========================================"
echo "    Voice Assistant BusyBox 安装"
echo "========================================"
echo ""

log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# 检查root权限
if [ "$EUID" -ne 0 ]; then
    log_error "请以root权限运行此脚本"
    exit 1
fi

# 创建必要的目录
log_info "创建必要的目录..."
mkdir -p /usr/local/bin
mkdir -p /var/run
mkdir -p /var/lock
mkdir -p /etc/init.d

# 1. 安装服务管理脚本
log_info "安装服务管理脚本..."
cp "$SCRIPT_DIR/voice-assistant-service.sh" /usr/local/bin/
chmod +x /usr/local/bin/voice-assistant-service.sh

# 2. 安装监控脚本
log_info "安装监控守护脚本..."
cp "$SCRIPT_DIR/voice-assistant-monitor.sh" /usr/local/bin/
chmod +x /usr/local/bin/voice-assistant-monitor.sh

# 3. 安装Swap设置脚本
log_info "安装Swap设置脚本..."
cp "$SCRIPT_DIR/swap-setup.sh" /usr/local/bin/
chmod +x /usr/local/bin/swap-setup.sh

# 4. 创建快捷命令
log_info "创建快捷命令..."

# va-start - 启动服务
cat > /usr/local/bin/va-start << 'EOF'
#!/bin/bash
/usr/local/bin/voice-assistant-service.sh start
EOF
chmod +x /usr/local/bin/va-start

# va-stop - 停止服务
cat > /usr/local/bin/va-stop << 'EOF'
#!/bin/bash
/usr/local/bin/voice-assistant-service.sh stop
EOF
chmod +x /usr/local/bin/va-stop

# va-restart - 重启服务
cat > /usr/local/bin/va-restart << 'EOF'
#!/bin/bash
/usr/local/bin/voice-assistant-service.sh restart
EOF
chmod +x /usr/local/bin/va-restart

# va-status - 查看状态
cat > /usr/local/bin/va-status << 'EOF'
#!/bin/bash
/usr/local/bin/voice-assistant-service.sh status
EOF
chmod +x /usr/local/bin/va-status

# va-monitor - 启动监控
cat > /usr/local/bin/va-monitor << 'EOF'
#!/bin/bash
case "${1:-start}" in
    start)
        /usr/local/bin/voice-assistant-monitor.sh start
        ;;
    stop)
        /usr/local/bin/voice-assistant-monitor.sh stop
        ;;
    status)
        /usr/local/bin/voice-assistant-monitor.sh status
        ;;
    restart)
        /usr/local/bin/voice-assistant-monitor.sh restart
        ;;
    *)
        echo "Usage: va-monitor {start|stop|status|restart}"
        exit 1
        ;;
esac
EOF
chmod +x /usr/local/bin/va-monitor

# 6. 配置开机自启（使用 /etc/init.d 或 /etc/rc.local）
log_info "配置开机自启..."

# 创建 init.d 脚本
cat > /etc/init.d/voice-assistant << 'EOF'
#!/bin/bash
# Voice Assistant service for BusyBox

case "$1" in
    start)
        echo "Starting Voice Assistant..."
        /usr/local/bin/voice-assistant-service.sh start
        # 可选：同时启动监控
        # /usr/local/bin/voice-assistant-monitor.sh start
        ;;
    stop)
        echo "Stopping Voice Assistant..."
        /usr/local/bin/voice-assistant-service.sh stop
        /usr/local/bin/voice-assistant-monitor.sh stop 2>/dev/null || true
        ;;
    restart)
        $0 stop
        sleep 1
        $0 start
        ;;
    status)
        /usr/local/bin/voice-assistant-service.sh status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
EOF
chmod +x /etc/init.d/voice-assistant

# 添加到 rc.local（如果存在）
if [ -f /etc/rc.local ]; then
    if ! grep -q "voice-assistant-service.sh" /etc/rc.local; then
        # 在 exit 0 之前添加
        if grep -q "^exit 0" /etc/rc.local; then
            sed -i '/^exit 0/i\/usr/local/bin/swap-setup.sh > /tmp/swap-setup.log 2>&1 &\n/usr/local/bin/voice-assistant-service.sh start > /tmp/va-autostart.log 2>&1 &' /etc/rc.local
        else
            echo "/usr/local/bin/swap-setup.sh > /tmp/swap-setup.log 2>&1 &" >> /etc/rc.local
            echo "/usr/local/bin/voice-assistant-service.sh start > /tmp/va-autostart.log 2>&1 &" >> /etc/rc.local
        fi
        log_info "已添加到 /etc/rc.local"
    fi
else
    # 创建 rc.local
    cat > /etc/rc.local << 'EOF'
#!/bin/bash
# 系统启动脚本

# 设置Swap
/usr/local/bin/swap-setup.sh > /tmp/swap-setup.log 2>&1 &

# 启动Voice Assistant
/usr/local/bin/voice-assistant-service.sh start > /tmp/va-autostart.log 2>&1 &

exit 0
EOF
    chmod +x /etc/rc.local
    log_info "已创建 /etc/rc.local"
fi

# 7. 立即执行Swap设置
log_info "执行Swap设置..."
/usr/local/bin/swap-setup.sh || log_warn "Swap设置可能失败，继续安装"

echo ""
echo "========================================"
echo "    安装完成！"
echo "========================================"
echo ""
echo "服务管理命令:"
echo "  va-start          - 启动服务"
echo "  va-stop           - 停止服务"
echo "  va-restart        - 重启服务"
echo "  va-status         - 查看状态"
echo ""
echo "监控管理命令:"
echo "  va-monitor start  - 启动监控守护（自动重启）"
echo "  va-monitor stop   - 停止监控守护"
echo "  va-monitor status - 查看监控状态"
echo ""
echo "完整命令:"
echo "  voice-assistant-service.sh {start|stop|restart|status}"
echo "  voice-assistant-monitor.sh {start|stop|status|restart}"
echo ""
echo "开机自启:"
echo "  已配置到 /etc/rc.local"
echo "  系统启动时将自动启动服务"
echo ""
echo "其他工具:"
echo "  swap-setup.sh     - Swap设置"
echo ""
