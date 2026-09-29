#!/bin/bash
# Voice Assistant 优化部署脚本
# 一键安装所有优化组件

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VA_DIR="/userdata/voice_assistant"

echo "========================================"
echo "    Voice Assistant 优化部署"
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
mkdir -p /etc/systemd/system

# 1. 安装Swap预创建脚本
log_info "安装Swap预创建脚本..."
cp "$SCRIPT_DIR/swap-setup.sh" /usr/local/bin/
chmod +x /usr/local/bin/swap-setup.sh

# 添加到rc.local实现开机自动配置
if [ -f /etc/rc.local ]; then
    if ! grep -q "swap-setup.sh" /etc/rc.local; then
        # 在exit 0之前添加
        sed -i '/^exit 0/i\/usr/local/bin/swap-setup.sh > /tmp/swap-setup.log 2>&1 &' /etc/rc.local
        log_info "已添加到/etc/rc.local"
    fi
else
    # 创建rc.local
    cat > /etc/rc.local << 'EOF'
#!/bin/bash
/usr/local/bin/swap-setup.sh > /tmp/swap-setup.log 2>&1 &
exit 0
EOF
    chmod +x /etc/rc.local
fi

# 立即执行swap设置
log_info "执行Swap设置..."
/usr/local/bin/swap-setup.sh

# 2. 安装systemd服务
log_info "安装systemd服务..."
cp "$SCRIPT_DIR/voice-assistant.service" /etc/systemd/system/
cp "$SCRIPT_DIR/voice-assistant-pre.sh" /usr/local/bin/
cp "$SCRIPT_DIR/voice-assistant-start.sh" /usr/local/bin/
cp "$SCRIPT_DIR/voice-assistant-stop.sh" /usr/local/bin/

chmod +x /usr/local/bin/voice-assistant-*.sh

# 重新加载systemd
systemctl daemon-reload 2>/dev/null || log_warn "systemctl不可用，跳过服务注册"

# 3. 复制优化版启动脚本到voice_assistant目录
log_info "安装优化版启动脚本..."
cp "$SCRIPT_DIR/../scripts/start_all_services_v3_optimized.sh" "$VA_DIR/" 2>/dev/null || \
    log_warn "无法复制到$VA_DIR，请手动复制"

# 5. 创建启动快捷方式
cat > /usr/local/bin/va-start << 'EOF'
#!/bin/bash
# Voice Assistant 快速启动

VA_DIR="/userdata/voice_assistant"

if [ -f "$VA_DIR/start_all_services_v3_optimized.sh" ]; then
    cd "$VA_DIR"
    bash start_all_services_v3_optimized.sh
else
    echo "错误: 启动脚本不存在"
    exit 1
fi
EOF
chmod +x /usr/local/bin/va-start

cat > /usr/local/bin/va-stop << 'EOF'
#!/bin/bash
# Voice Assistant 快速停止

pkill -9 -f "rag_optimized_server\|qwen3_llm_service\|melotts_service" 2>/dev/null || true
rm -f /tmp/*.sock
echo "Voice Assistant 服务已停止"
EOF
chmod +x /usr/local/bin/va-stop

cat > /usr/local/bin/va-status << 'EOF'
#!/bin/bash
# Voice Assistant 状态检查

echo "=== Voice Assistant 服务状态 ==="
echo ""

echo "Socket文件:"
for sock in /tmp/rag_optimized.sock /tmp/qwen3_llm.sock /tmp/melotts_service.sock; do
    if [ -S "$sock" ]; then
        echo "  ✓ $sock"
    else
        echo "  ✗ $sock (不存在)"
    fi
done

echo ""
echo "进程状态:"
ps | grep -E "rag_optimized|qwen3_llm|melotts_service" | grep -v grep || echo "  无运行进程"

echo ""
echo "内存使用:"
free -h | grep -E "Mem|Swap"
EOF
chmod +x /usr/local/bin/va-status

echo ""
echo "========================================"
echo "    部署完成！"
echo "========================================"
echo ""
echo "可用命令:"
echo "  va-start    - 快速启动服务(V3优化版)"
echo "  va-stop     - 停止所有服务"
echo "  va-status   - 查看服务状态"
echo ""
echo "systemd服务:"
echo "  systemctl start voice-assistant   - 启动服务"
echo "  systemctl stop voice-assistant    - 停止服务"
echo "  systemctl enable voice-assistant  - 开机自启"
echo ""
echo "优化版启动脚本位置:"
echo "  $VA_DIR/start_all_services_v3_optimized.sh"
echo ""
