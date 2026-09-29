#!/bin/bash
# 批量修改启动脚本，禁用开机自动启动

SCRIPTS_DIR="/home/ubuntu/桌面/ai/voice_assistant/deploy/board_init_scripts"

# 需要修改的脚本列表
SCRIPTS=(
    "S35iptables"
    "S36wifibt-init.sh"
    "S40bluetoothd"
    "S40network"
    "S40rkaiq_3A"
    "S45connman"
    "S49ntp"
    "S49weston"
    "S50nginx"
    "S50sshd"
    "S50systemui"
    "S60nfs"
    "S70vsftpd"
    "S80dnsmasq"
    "S91smb"
    "S99chromium-wayland.sh"
)

echo "开始修改启动脚本..."
echo ""

for script in "${SCRIPTS[@]}"; do
    filepath="$SCRIPTS_DIR/$script"
    if [ ! -f "$filepath" ]; then
        echo "跳过: $script (文件不存在)"
        continue
    fi

    echo "处理: $script"

    # 创建临时文件
    temp_file="$filepath.tmp"

    # 读取第一行（shebang）
    head -1 "$filepath" > "$temp_file"

    # 添加注释和 AUTO_START 变量
    cat >> "$temp_file" << 'EOF'
# 注意: 此脚本已被修改，默认不在启动时执行
# 如需启动，请运行: /etc/init.d/SCRIPT_NAME start
# 或运行: start-optional-services.sh

# 开机自动启动开关 (设置为 true 启用，false 禁用)
AUTO_START=false

EOF

    # 将 SCRIPT_NAME 替换为实际脚本名
    sed -i "s/SCRIPT_NAME/$script/g" "$temp_file"

    # 追加原文件的其余内容（跳过第一行）
    tail -n +2 "$filepath" >> "$temp_file"

    # 替换原文件
    mv "$temp_file" "$filepath"
    chmod +x "$filepath"

done

echo ""
echo "修改完成！"
echo ""
echo "以下脚本已禁用开机自动启动:"
for script in "${SCRIPTS[@]}"; do
    echo "  - $script"
done
