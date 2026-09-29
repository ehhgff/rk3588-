#!/bin/bash
# Swap预创建脚本
# 建议添加到 /etc/rc.local 系统启动时执行

SWAP_FILE="/swapfile"
SWAP_SIZE_MB=2048

log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $1"
}

# 检查是否已存在swap文件
if [ -f "$SWAP_FILE" ]; then
    log_info "Swap文件已存在: $SWAP_FILE"
else
    log_info "创建Swap文件 (${SWAP_SIZE_MB}MB)..."
    
    # 使用fallocate快速创建（比dd快）
    if command -v fallocate >/dev/null 2>&1; then
        fallocate -l ${SWAP_SIZE_MB}M $SWAP_FILE 2>/dev/null || \
            dd if=/dev/zero of=$SWAP_FILE bs=1M count=$SWAP_SIZE_MB 2>/dev/null
    else
        dd if=/dev/zero of=$SWAP_FILE bs=1M count=$SWAP_SIZE_MB 2>/dev/null
    fi
    
    chmod 600 $SWAP_FILE
    mkswap $SWAP_FILE 2>/dev/null
    log_info "Swap文件创建完成"
fi

# 检查swap是否已启用
if swapon -s | grep -q "$SWAP_FILE"; then
    log_info "Swap已启用"
else
    log_info "启用Swap..."
    swapon $SWAP_FILE 2>/dev/null || log_warn "Swap启用失败"
fi

# 添加到fstab确保开机自动挂载
if ! grep -q "$SWAP_FILE" /etc/fstab 2>/dev/null; then
    log_info "添加Swap到fstab..."
    echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
fi

# 设置swappiness优化内存使用
current_swappiness=$(cat /proc/sys/vm/swappiness 2>/dev/null || echo "60")
if [ "$current_swappiness" -gt "10" ]; then
    log_info "设置vm.swappiness=10 (当前: $current_swappiness)"
    echo 10 > /proc/sys/vm/swappiness
    
    # 持久化配置
    if ! grep -q "vm.swappiness" /etc/sysctl.conf 2>/dev/null; then
        echo "vm.swappiness=10" >> /etc/sysctl.conf
    fi
fi

log_info "Swap设置完成"
free -h | grep -E "Mem|Swap"
