#!/bin/bash
# 设置语音助手开机自动启动 - 在系统最后启动

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

log_info() {
    echo "[INFO] $1"
}

# 1. 安装自动启动脚本
log_info "安装自动启动脚本..."
cp "$SCRIPT_DIR/S99voice-assistant-autostart" /etc/init.d/
chmod +x /etc/init.d/S99voice-assistant-autostart

# 2. 清理 rc.local 中的重复和冲突
log_info "优化 rc.local..."
cat > /etc/rc.local << 'EOF'
#!/bin/bash
# 系统启动脚本
# 语音助手服务由 /etc/init.d/S99voice-assistant-autostart 管理

exit 0
EOF
chmod +x /etc/rc.local

# 3. 确保快速启动服务已安装
if [ ! -x /usr/local/bin/fast-boot-service.sh ]; then
    log_info "安装快速启动服务..."
    cp "$SCRIPT_DIR/fast-boot-service.sh" /usr/local/bin/
    chmod +x /usr/local/bin/fast-boot-service.sh
fi

# 4. 创建快捷命令（如果不存在）
if [ ! -x /usr/local/bin/va-fast-start ]; then
    log_info "创建快捷命令..."
    
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
fi

# 5. 测试配置
echo ""
echo "=== 自动启动配置 ==="
echo "启动脚本: /etc/init.d/S99voice-assistant-autostart"
echo "启动顺序: 系统最后（S99）"
echo "启动延迟: 3秒（等待系统就绪）"
echo ""
echo "=== 启动流程 ==="
echo "1. 系统启动所有 Sxx 服务"
echo "2. 执行 S99voice-assistant-autostart"
echo "3. 等待3秒确保系统就绪"
echo "4. 启动语音助手服务"
echo "5. 服务启动完成（约4-5秒）"
echo ""
echo "=== 预期效果 ==="
echo "- 开机后约8-10秒语音助手就绪"
echo "- 用户无需手动操作"
echo "- 服务自动在后台启动"
echo ""
echo "如需立即测试，请重启系统: reboot"
