#!/bin/bash
# 设置语音助手开机自动启动 - 在系统启动最后阶段
# 这个脚本需要在开发板上以 root 权限运行

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1"
}

# 检查是否以 root 运行
if [ "$EUID" -ne 0 ] && [ "$(id -u)" -ne 0 ]; then
    log_error "请使用 root 权限运行此脚本"
    exit 1
fi

# 1. 安装启动脚本到 /etc/init.d/
log_info "安装启动脚本到 /etc/init.d/..."
if [ -f "$SCRIPT_DIR/S99voice-assistant" ]; then
    cp "$SCRIPT_DIR/S99voice-assistant" /etc/init.d/
    chmod +x /etc/init.d/S99voice-assistant
    log_info "启动脚本已安装: /etc/init.d/S99voice-assistant"
else
    log_error "未找到启动脚本: $SCRIPT_DIR/S99voice-assistant"
    exit 1
fi

# 2. 确保快速启动服务已安装
if [ ! -x /usr/local/bin/fast-boot-service.sh ]; then
    log_info "安装快速启动服务..."
    if [ -f "$SCRIPT_DIR/fast-boot-service.sh" ]; then
        cp "$SCRIPT_DIR/fast-boot-service.sh" /usr/local/bin/
        chmod +x /usr/local/bin/fast-boot-service.sh
        log_info "快速启动服务已安装"
    else
        log_error "未找到快速启动服务脚本"
    fi
else
    log_info "快速启动服务已存在"
fi

# 3. 确保标准服务脚本已安装
if [ ! -x /usr/local/bin/voice-assistant-service.sh ]; then
    log_info "安装标准服务脚本..."
    if [ -f "$SCRIPT_DIR/voice-assistant-service.sh" ]; then
        cp "$SCRIPT_DIR/voice-assistant-service.sh" /usr/local/bin/
        chmod +x /usr/local/bin/voice-assistant-service.sh
        log_info "标准服务脚本已安装"
    else
        log_error "未找到标准服务脚本"
    fi
else
    log_info "标准服务脚本已存在"
fi

# 4. 清理 rc.local 中的重复和冲突（如果有的话）
log_info "检查并优化 rc.local..."
if [ -f /etc/rc.local ]; then
    # 备份原文件
    cp /etc/rc.local /etc/rc.local.backup.$(date +%Y%m%d%H%M%S)
    
    # 创建新的 rc.local，移除旧的语音助手启动命令
    grep -v "voice-assistant" /etc/rc.local > /tmp/rc.local.tmp
    grep -v "swap-setup.sh" /tmp/rc.local.tmp > /etc/rc.local
    rm -f /tmp/rc.local.tmp
    
    # 确保 rc.local 有可执行权限
    chmod +x /etc/rc.local
    log_info "rc.local 已优化"
fi

# 5. 删除旧的语音助手脚本（如果有）
log_info "清理旧的启动脚本..."
if [ -f /etc/init.d/voice-assistant ]; then
    rm -f /etc/init.d/voice-assistant
    log_info "已删除旧的 voice-assistant 脚本"
fi

if [ -f /etc/init.d/voice-assistant-fast ]; then
    rm -f /etc/init.d/voice-assistant-fast
    log_info "已删除旧的 voice-assistant-fast 脚本"
fi

# 6. 验证安装
log_info "验证安装..."
if [ -x /etc/init.d/S99voice-assistant ]; then
    log_info "✓ 启动脚本已正确安装"
else
    log_error "✗ 启动脚本安装失败"
    exit 1
fi

# 7. 显示启动顺序
log_info "当前启动顺序中的 S99 服务:"
ls -1 /etc/init.d/S99* 2>/dev/null | while read line; do
    echo "  - $(basename $line)"
done

echo ""
log_info "========================================"
log_info "配置完成！"
log_info "========================================"
echo ""
echo "语音助手将在系统启动的最后阶段自动启动。"
echo ""
echo "常用命令:"
echo "  /etc/init.d/S99voice-assistant status  - 查看状态"
echo "  /etc/init.d/S99voice-assistant stop    - 停止服务"
echo "  /etc/init.d/S99voice-assistant start   - 启动服务"
echo "  /etc/init.d/S99voice-assistant restart - 重启服务"
echo ""
echo "日志文件: /tmp/voice-assistant-boot.log"
echo ""
echo "下次重启后将自动生效。"
