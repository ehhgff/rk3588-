#!/bin/bash
#
# test_rktop_on_board.sh - 在 RK3588 板端通过 SSH 运行 rktop 并采集详细系统指标
#
# 用法:
#   ./test_rktop_on_board.sh                          # 使用默认 IP
#   ./test_rktop_on_board.sh 192.168.1.100            # 指定板端 IP
#   BOARD_IP=192.168.1.100 ./test_rktop_on_board.sh   # 通过环境变量指定
#
# 前置条件:
#   - 板端已安装 /userdata/rktop（静态链接 ARM64 二进制）
#   - 板端已启用 SSH 服务（sshd）
#   - 本机可无密码 SSH 到板端 root 用户（或输入密码）
#

set -euo pipefail

# ============================================================
# 配置
# ============================================================
BOARD_IP="${1:-${BOARD_IP:-192.168.138.136}}"
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
RTOP_CMD="${RTOP_CMD:-/userdata/rktop}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
FORCE_ADB="${FORCE_ADB:-0}"
LOG_FILE="/tmp/rktop_remote_test_$(date +%Y%m%d_%H%M%S).log"

SCRIPT_START=$(date +%s)
STEP_NUM=0

# ============================================================
# 日志函数
# ============================================================
log() {
    local level="$1"
    local msg="$2"
    local elapsed
    elapsed=$(( $(date +%s) - SCRIPT_START ))
    printf "[%5ds] [%-5s] %s\n" "$elapsed" "$level" "$msg" | tee -a "$LOG_FILE"
}

log_step() {
    STEP_NUM=$((STEP_NUM + 1))
    log "======" "============================================================"
    log "STEP " "步骤 $STEP_NUM: $1"
    log "======" "============================================================"
}

log_ok()    { log "OK   " "$1"; }
log_warn()  { log "WARN " "$1"; }
log_err()   { log "ERROR" "$1"; }
log_data()  { log "DATA " "$1"; }

# ============================================================
# 远程执行函数
# ============================================================
REMOTE_MODE=""

detect_remote() {
    if [ "$FORCE_ADB" = "1" ]; then
        if command -v adb &>/dev/null && adb devices 2>/dev/null | grep -qE "device$"; then
            REMOTE_MODE="adb"
            log_ok "ADB 模式 (FORCE_ADB=1)"
            return 0
        fi
        log_err "FORCE_ADB=1 但 ADB 不可用"
        exit 1
    fi
    if ssh -p "$SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
        "${SSH_USER}@${BOARD_IP}" "echo OK" 2>/dev/null | grep -q "OK"; then
        REMOTE_MODE="ssh"
        log_ok "SSH 连接成功: ${SSH_USER}@${BOARD_IP}:${SSH_PORT}"
        return 0
    fi
    if command -v adb &>/dev/null && adb devices 2>/dev/null | grep -qE "device$"; then
        REMOTE_MODE="adb"
        log_warn "SSH 不可用，切换到 ADB 模式"
        log_ok "ADB 设备已连接"
        return 0
    fi
    log_err "SSH 和 ADB 均无法连接到板端"
    exit 1
}

run_remote() {
    if [ "$REMOTE_MODE" = "ssh" ]; then
        ssh -p "$SSH_PORT" "${SSH_USER}@${BOARD_IP}" "$@"
    else
        local cmd="$*"
        printf '%s\n' "$cmd" | adb shell "su -c 'sh -s'"
    fi
}

run_remote_script() {
    local label="$1"
    local script="$2"
    log_data "========== $label =========="
    if [ "$REMOTE_MODE" = "ssh" ]; then
        ssh -p "$SSH_PORT" "${SSH_USER}@${BOARD_IP}" "$script" 2>&1 | tee -a "$LOG_FILE"
    else
        printf '%s\n' "$script" | adb shell "su -c 'sh -s'" 2>&1 | tee -a "$LOG_FILE"
    fi
}

# ============================================================
# 采集函数 - 系统各项指标
# ============================================================

collect_system_info() {
    local label="$1"

    # --- 基础信息 ---
    run_remote_script "基础信息 [$label]" '
        echo "主机名  : $(hostname)"
        echo "内核    : $(uname -a)"
        echo "启动时间: $(uptime -s 2>/dev/null || echo N/A)"
        echo "运行时间: $(awk "{printf \"%.1f 天\", \$1/86400}" /proc/uptime 2>/dev/null)"
        echo "设备模型: $(cat /proc/device-tree/model 2>/dev/null | tr -d "\0" || echo N/A)"
    '

    # --- CPU ---
    run_remote_script "CPU [$label]" '
        echo "核心数  : $(nproc)"
        echo "负载均值: $(cat /proc/loadavg)"
        echo ""
        echo "--- 每核状态 ---"
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
            core=$(echo $cpu | sed "s/.*cpu\\([0-9]\\+\\).*/\\1/")
            freq=$(cat $cpu 2>/dev/null)
            gov=$(cat $(dirname $cpu)/scaling_governor 2>/dev/null)
            min=$(cat $(dirname $cpu)/scaling_min_freq 2>/dev/null)
            max=$(cat $(dirname $cpu)/scaling_max_freq 2>/dev/null)
            printf "  CPU%s: %d MHz | %s | %d-%d MHz\n" "$core" "$((freq/1000))" "$gov" "$((min/1000))" "$((max/1000))"
        done
        echo ""
        echo "--- CPU 使用率 (仅 aggregate) ---"
        awk "/^cpu /{total=\$2+\$3+\$4+\$5+\$6+\$7+\$8+\$9+\$10; printf \"  使用率: %.1f%% (user:%.1f%% sys:%.1f%% iowait:%.1f%% idle:%.1f%%)\\n\", (1-\$5/total)*100, \$2/total*100, \$4/total*100, \$6/total*100, \$5/total*100}" /proc/stat
    '

    # --- 内存 ---
    run_remote_script "内存 [$label]" '
        echo "--- free -m ---"
        free -m
        echo ""
        echo "--- /proc/meminfo 关键项 ---"
        grep -E "^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree|Dirty|AnonPages|Shmem)" /proc/meminfo
    '

    # --- 温度 ---
    run_remote_script "温度 [$label]" '
        max_temp=0
        for zone in /sys/class/thermal/thermal_zone*; do
            name=$(cat $zone/type 2>/dev/null)
            temp=$(cat $zone/temp 2>/dev/null)
            if [ -n "$temp" ] && [ "$temp" -gt 0 ] 2>/dev/null; then
                c=$((temp/1000))
                printf "  %s: %d°C\n" "$name" "$c"
                [ "$c" -gt "$max_temp" ] && max_temp=$c
            fi
        done
        echo "  最高温度: ${max_temp}°C"
    '

    # --- NPU ---
    run_remote_script "NPU [$label]" '
        if [ -d /sys/class/devfreq/fdab0000.npu ]; then
            echo "NPU频率:"
            for f in cur_freq available_frequencies; do
                val=$(cat /sys/class/devfreq/fdab0000.npu/$f 2>/dev/null)
                echo "  $f: $val"
            done
        else
            echo "  NPU sysfs: N/A"
        fi
        echo "NPU驱动版本:"
        cat /sys/kernel/debug/rknpu/driver_version 2>/dev/null || cat /proc/rknpu/driver_version 2>/dev/null || echo "  N/A"
        echo "NPU负载:"
        cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo "  N/A"
        echo "NPU内存分配:"
        cat /sys/kernel/debug/rknpu/alloc 2>/dev/null | head -8 || echo "  N/A"
    '

    # --- GPU ---
    run_remote_script "GPU [$label]" '
        found=0
        for dev in /sys/class/devfreq/fb000000.gpu /sys/class/devfreq/fb000000.gpu-panthor; do
            if [ -f "$dev/cur_freq" ]; then
                cur=$(cat $dev/cur_freq 2>/dev/null)
                gov=$(cat $dev/governor 2>/dev/null)
                printf "  GPU频率: %d MHz | Governor: %s\n" "$((cur/1000000))" "$gov"
                found=1
            fi
        done
        [ "$found" -eq 0 ] && echo "  GPU: N/A"
        echo "GPU利用率:"
        cat /sys/class/devfreq/fb000000.gpu/gpu_load 2>/dev/null || echo "  N/A"
    '

    # --- 磁盘 ---
    run_remote_script "磁盘 [$label]" '
        echo "--- 分区使用 ---"
        df -h | grep -vE "tmpfs|devtmpfs|overlay" | awk "{printf \"  %-20s %s/%s (%s)\n\", \$1, \$3, \$2, \$5}"
        echo ""
        echo "--- 磁盘 I/O 统计 ---"
        cat /proc/diskstats | grep -E "mmcblk|nvme|sda" | head -6 | awk "{printf \"  %s: 读次数=%s 写次数=%s\n\", \$3, \$4, \$8}"
    '

    # --- 进程 ---
    run_remote_script "进程 [$label]" '
        echo "总进程数: $(ps aux 2>/dev/null | wc -l)"
        echo "CPU Top 8:"
        ps aux 2>/dev/null | sort -k3 -rn | head -8 | awk "{printf \"  PID=%-6s CPU=%-5s MEM=%-5s %s\n\", \$2, \$3\"%\", \$4\"%\", \$11}"
        echo ""
        echo "内存 Top 8:"
        ps aux 2>/dev/null | sort -k4 -rn | head -8 | awk "{printf \"  PID=%-6s CPU=%-5s MEM=%-5s %s\n\", \$2, \$3\"%\", \$4\"%\", \$11}"
    '

    # --- DDR ---
    run_remote_script "DDR [$label]" '
        if [ -f /sys/class/devfreq/dmc/cur_freq ]; then
            cur=$(cat /sys/class/devfreq/dmc/cur_freq 2>/dev/null)
            echo "DDR频率: $((cur/1000000)) MHz"
        else
            echo "DDR: N/A"
        fi
    '
}

# ============================================================
# 主流程
# ============================================================

# 初始化日志文件
echo "" > "$LOG_FILE"
{
    echo "============================================================"
    echo " RK3588 rktop 远程测试日志"
    echo " 生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo " 板端 IP   : $BOARD_IP"
    echo " SSH 用户   : $SSH_USER"
    echo " rktop路径 : $RTOP_CMD"
    echo " 超时时间  : ${TIMEOUT_SEC}s"
    echo " 日志文件  : $LOG_FILE"
    echo "============================================================"
    echo ""
} | tee -a "$LOG_FILE"

# Step 1: 检测远程连接
log_step "检测远程连接方式"
detect_remote

# Step 2: 检查 rktop 二进制
log_step "检查 rktop 二进制"
if ! run_remote "ls -lh $RTOP_CMD" 2>&1 | tee -a "$LOG_FILE"; then
    log_err "板端未找到 $RTOP_CMD"
    log_err "请先推送: adb push <本地路径> /userdata/"
    exit 1
fi
log_ok "rktop 存在"

# Step 3: 检查板端内核接口
log_step "检查 Rockchip 内核接口"
run_remote_script "内核接口" '
    echo "NPU设备节点:"
    ls -la /dev/dri/ 2>/dev/null | grep -i npu || echo "  /dev/dri/npu: N/A"
    ls -d /sys/class/devfreq/fdab0000.npu 2>/dev/null || echo "  /sys/class/devfreq/fdab0000.npu: N/A"
    echo "GPU设备节点:"
    ls -d /sys/class/devfreq/fb000000.gpu* 2>/dev/null || echo "  /sys/class/devfreq/fb000000.gpu: N/A"
    ls -d /dev/rga /dev/mpp_service 2>/dev/null || echo "  无 RGA/MPP 设备节点"
'

# Step 4: 运行前系统快照
log_step "采集运行前系统快照"
collect_system_info "BEFORE"

# Step 5: 运行 rktop
log_step "启动 rktop TUI 监控"

if [ "$REMOTE_MODE" = "adb" ]; then
    log_warn "ADB 无法直接运行 TUI 程序，请手动执行:"
    log_warn "  adb shell"
    log_warn "  su -c '$RTOP_CMD'"
    log_warn "或: ssh -t root@$BOARD_IP '$RTOP_CMD'"
else
    log_ok "通过 SSH 启动 rktop (伪终端 -t)"
    log_ok "按 Ctrl+C 提前退出，上限 ${TIMEOUT_SEC}s"
    echo ""

    log "------" "-------- rktop 启动 --------"
    RTOP_START=$(date +%s)

    ssh -p "$SSH_PORT" -t -o ConnectTimeout=10 \
        "${SSH_USER}@${BOARD_IP}" \
        "RTOP_CMD=$RTOP_CMD && \
         echo '[rktop] 启动时间: \$(date)'; \
         echo '[rktop] 负载: \$(cat /proc/loadavg)'; \
         echo '[rktop] 内存: \$(free -m | grep Mem | awk \"{print \\\$3\\\"M/\\\"\\\$2\\\"M\\\"}\")'; \
         echo '[rktop]'; \
         exec \$RTOP_CMD" 2>&1 | tee -a "$LOG_FILE"

    RTOP_DURATION=$(( $(date +%s) - RTOP_START ))
    log "------" "-------- rktop 退出 (运行 ${RTOP_DURATION}s) --------"

    # 退出后瞬时状态
    run_remote_script "rktop 退出后瞬时状态" '
        echo "时间: $(date)"
        echo "负载: $(cat /proc/loadavg)"
        echo "内存: $(free -m | grep Mem | awk "{print \$3\"M/\"\$2\"M\"}")"
        max_temp=0
        for z in /sys/class/thermal/thermal_zone*; do
            t=$(cat $z/temp 2>/dev/null)
            n=$(cat $z/type 2>/dev/null)
            [ -n "$t" ] && [ "$t" -gt 0 ] 2>/dev/null && c=$((t/1000)) && printf "  温度: %s %d°C\n" "$n" "$c" && [ "$c" -gt "$max_temp" ] && max_temp=$c
        done
        echo "  最高温度: ${max_temp}°C"
    '
fi

# Step 6: 运行后系统快照
log_step "采集运行后系统快照"
collect_system_info "AFTER"

# Step 7: 对比摘要
log_step "生成性能对比摘要"
TOTAL_DURATION=$(( $(date +%s) - SCRIPT_START ))

{
    echo ""
    echo "============================================================"
    echo "  测试摘要"
    echo "============================================================"
    echo "  板端 IP        : $BOARD_IP"
    echo "  远程模式       : $REMOTE_MODE"
    echo "  测试总耗时     : ${TOTAL_DURATION}s"
    if [ -n "${RTOP_DURATION:-}" ]; then
        echo "  rktop 运行时长 : ${RTOP_DURATION}s"
    fi
    echo "  日志文件       : $LOG_FILE"
    echo "============================================================"
    echo ""
    echo "  [CPU 负载]"
    echo -n "    运行前: "
    run_remote "cat /proc/loadavg" 2>/dev/null | tee -a "$LOG_FILE"
    echo -n "    运行后: "
    run_remote "cat /proc/loadavg" 2>/dev/null | tee -a "$LOG_FILE"

    echo ""
    echo "  [内存]"
    echo -n "    运行前: "
    run_remote "free -m | grep Mem | awk '{print \$3\"M / \"\$2\"M\"}'" 2>/dev/null | tee -a "$LOG_FILE"
    echo -n "    运行后: "
    run_remote "free -m | grep Mem | awk '{print \$3\"M / \"\$2\"M\"}'" 2>/dev/null | tee -a "$LOG_FILE"

    echo ""
    echo "  [温度峰值]"
    echo -n "    运行前: "
    run_remote "for z in /sys/class/thermal/thermal_zone*; do cat \$z/temp 2>/dev/null; done | sort -rn | head -1 | awk '{printf \"%d°C\n\", \$1/1000}'" 2>/dev/null | tee -a "$LOG_FILE"
    echo -n "    运行后: "
    run_remote "for z in /sys/class/thermal/thermal_zone*; do cat \$z/temp 2>/dev/null; done | sort -rn | head -1 | awk '{printf \"%d°C\n\", \$1/1000}'" 2>/dev/null | tee -a "$LOG_FILE"

    echo ""
    echo "  [NPU 频率]"
    echo -n "    运行前: "
    run_remote "cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null | awk '{printf \"%d MHz\n\", \$1/1000000}'" 2>/dev/null | tee -a "$LOG_FILE"
    echo -n "    运行后: "
    run_remote "cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null | awk '{printf \"%d MHz\n\", \$1/1000000}'" 2>/dev/null | tee -a "$LOG_FILE"

    echo ""
    echo "============================================================"
    echo "  测试结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
} | tee -a "$LOG_FILE"

echo ""
log_ok "日志已保存: $LOG_FILE"
echo "  查看: less $LOG_FILE"
echo "  概况: grep -E '\[STEP\]|\[OK\]|\[ERROR\]|\[WARN\]' $LOG_FILE"
echo ""