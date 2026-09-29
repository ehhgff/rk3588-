#!/bin/bash
# 启动可选服务脚本
# 用于手动启动被禁用的开机服务

SCRIPT_DIR="/etc/init.d"

# 定义服务启动顺序（按照依赖关系）
OPTIONAL_SERVICES=(
    # 网络和无线
    "S35iptables"           # 防火墙
    "S36wifibt-init.sh"     # WiFi/BT 初始化
    "S40network"            # 网络配置
    "S40bluetoothd"         # 蓝牙服务
    "S45connman"            # 网络管理器
    "S49ntp"                # 时间同步
    "S80dnsmasq"            # DNS/DHCP

    # 显示和图形界面
    "S40rkaiq_3A"           # 图像质量（ISP）
    "S49weston"             # Wayland 显示服务器
    "S50systemui"           # Qt 系统界面
    "S99chromium-wayland.sh" # Chromium 环境

    # 网络服务
    "S50nginx"              # Web 服务器
    "S50sshd"               # SSH 服务器
    "S60nfs"                # NFS 文件共享
    "S70vsftpd"             # FTP 服务器
    "S91smb"                # Samba 文件共享
)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
启动可选服务脚本

用法: $0 [选项] [服务名]

选项:
    start [服务名]      启动指定服务或所有可选服务
    stop [服务名]       停止指定服务或所有可选服务
    restart [服务名]    重启指定服务或所有可选服务
    status              查看所有可选服务的状态
    list                列出所有可选服务
    help                显示此帮助信息

示例:
    $0 start                    # 启动所有可选服务
    $0 start S50nginx           # 只启动 nginx
    $0 stop S50sshd             # 停止 sshd
    $0 status                   # 查看所有服务状态
    $0 list                     # 列出所有可选服务

EOF
}

# 列出所有可选服务
list_services() {
    echo "可选服务列表:"
    echo ""
    printf "%-30s %-20s %s\n" "服务名" "类别" "说明"
    echo "────────────────────────────────────────────────────────────────"

    declare -A SERVICE_DESC=(
        ["S35iptables"]="防火墙规则"
        ["S36wifibt-init.sh"]="WiFi/BT 初始化"
        ["S40network"]="网络配置"
        ["S40bluetoothd"]="蓝牙服务"
        ["S40rkaiq_3A"]="图像质量 (ISP)"
        ["S45connman"]="网络管理器"
        ["S49ntp"]="时间同步 (NTP)"
        ["S49weston"]="Wayland 显示服务器"
        ["S50nginx"]="Web 服务器"
        ["S50sshd"]="SSH 服务器"
        ["S50systemui"]="Qt 系统界面"
        ["S60nfs"]="NFS 文件共享"
        ["S70vsftpd"]="FTP 服务器"
        ["S80dnsmasq"]="DNS/DHCP 服务"
        ["S91smb"]="Samba 文件共享"
        ["S99chromium-wayland.sh"]="Chromium 环境"
    )

    declare -A SERVICE_CATEGORY=(
        ["S35iptables"]="网络"
        ["S36wifibt-init.sh"]="网络"
        ["S40network"]="网络"
        ["S40bluetoothd"]="网络"
        ["S40rkaiq_3A"]="显示"
        ["S45connman"]="网络"
        ["S49ntp"]="网络"
        ["S49weston"]="显示"
        ["S50nginx"]="网络服务"
        ["S50sshd"]="网络服务"
        ["S50systemui"]="显示"
        ["S60nfs"]="文件共享"
        ["S70vsftpd"]="文件共享"
        ["S80dnsmasq"]="网络"
        ["S91smb"]="文件共享"
        ["S99chromium-wayland.sh"]="显示"
    )

    for service in "${OPTIONAL_SERVICES[@]}"; do
        category="${SERVICE_CATEGORY[$service]:-其他}"
        desc="${SERVICE_DESC[$service]:-未知服务}"
        printf "%-30s %-20s %s\n" "$service" "$category" "$desc"
    done
}

# 检查服务是否存在
check_service() {
    local service=$1
    if [ ! -f "$SCRIPT_DIR/$service" ]; then
        log_error "服务不存在: $service"
        return 1
    fi
    return 0
}

# 启动单个服务
start_service() {
    local service=$1
    if ! check_service "$service"; then
        return 1
    fi

    log_info "启动 $service ..."
    if $SCRIPT_DIR/$service start; then
        log_info "$service 启动成功"
        return 0
    else
        log_error "$service 启动失败"
        return 1
    fi
}

# 停止单个服务
stop_service() {
    local service=$1
    if ! check_service "$service"; then
        return 1
    fi

    log_info "停止 $service ..."
    if $SCRIPT_DIR/$service stop; then
        log_info "$service 停止成功"
        return 0
    else
        log_error "$service 停止失败"
        return 1
    fi
}

# 启动所有服务
start_all() {
    log_info "开始启动所有可选服务..."
    echo ""

    local success=0
    local failed=0

    for service in "${OPTIONAL_SERVICES[@]}"; do
        if start_service "$service"; then
            ((success++))
        else
            ((failed++))
        fi
        echo ""
    done

    log_info "启动完成: 成功 $success 个, 失败 $failed 个"
}

# 停止所有服务（逆序）
stop_all() {
    log_info "开始停止所有可选服务..."
    echo ""

    local success=0
    local failed=0

    # 逆序停止
    for ((i=${#OPTIONAL_SERVICES[@]}-1; i>=0; i--)); do
        service="${OPTIONAL_SERVICES[$i]}"
        if stop_service "$service"; then
            ((success++))
        else
            ((failed++))
        fi
        echo ""
    done

    log_info "停止完成: 成功 $success 个, 失败 $failed 个"
}

# 查看服务状态
show_status() {
    echo "可选服务状态:"
    echo ""
    printf "%-30s %-15s %s\n" "服务名" "状态" "说明"
    echo "────────────────────────────────────────────────────────────────"

    for service in "${OPTIONAL_SERVICES[@]}"; do
        if [ -f "$SCRIPT_DIR/$service" ]; then
            # 尝试获取服务状态
            status="未知"
            if $SCRIPT_DIR/$service status >/dev/null 2>&1; then
                status="运行中"
            else
                # 检查进程是否存在
                case "$service" in
                    S35iptables)
                        iptables -L >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S40bluetoothd)
                        pgrep -x bluetoothd >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S40network)
                        ip addr show | grep -q "inet " && status="运行中" || status="已停止"
                        ;;
                    S45connman)
                        pgrep -x connmand >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S49ntp)
                        pgrep -x ntpd >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S49weston)
                        pgrep -x weston >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S50nginx)
                        pgrep -x nginx >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S50sshd)
                        pgrep -x sshd >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S50systemui)
                        pgrep -x systemui >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S60nfs)
                        pgrep -x rpc.nfsd >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S70vsftpd)
                        pgrep -x vsftpd >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S80dnsmasq)
                        pgrep -x dnsmasq >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    S91smb)
                        pgrep -x smbd >/dev/null 2>&1 && status="运行中" || status="已停止"
                        ;;
                    *)
                        status="未知"
                        ;;
                esac
            fi

            if [ "$status" = "运行中" ]; then
                printf "%-30s ${GREEN}%-15s${NC} %s\n" "$service" "$status" ""
            else
                printf "%-30s ${YELLOW}%-15s${NC} %s\n" "$service" "$status" ""
            fi
        else
            printf "%-30s ${RED}%-15s${NC} %s\n" "$service" "不存在" ""
        fi
    done
}

# 主逻辑
case "$1" in
    start)
        if [ -n "$2" ]; then
            start_service "$2"
        else
            start_all
        fi
        ;;
    stop)
        if [ -n "$2" ]; then
            stop_service "$2"
        else
            stop_all
        fi
        ;;
    restart)
        if [ -n "$2" ]; then
            stop_service "$2"
            sleep 1
            start_service "$2"
        else
            stop_all
            sleep 2
            start_all
        fi
        ;;
    status)
        show_status
        ;;
    list)
        list_services
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
