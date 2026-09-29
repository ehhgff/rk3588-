#!/bin/bash
#====================================================================
# 一键部署脚本 - 将所有优化推送到板子并应用
#
# 用法:
#   ./deploy_all.sh                    # 部署所有更改
#   ./deploy_all.sh --check            # 只检查状态，不做更改
#   ./deploy_all.sh --skip-vad         # 跳过 VAD 调优
#   ./deploy_all.sh --skip-asr         # 跳过 ASR 初始化修复
#   ./deploy_all.sh --help             # 显示帮助
#====================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOARD_DIR="/userdata/voice_assistant"
ADB="adb"

# --- 解析参数 ---
CHECK_ONLY=0
SKIP_VAD=0
SKIP_ASR=0

for arg in "$@"; do
    case "$arg" in
        --check)    CHECK_ONLY=1 ;;
        --skip-vad) SKIP_VAD=1 ;;
        --skip-asr) SKIP_ASR=1 ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --check       只检查状态，不做更改"
            echo "  --skip-vad    跳过 VAD 调优"
            echo "  --skip-asr    跳过 ASR 初始化修复"
            echo "  --help        显示此帮助"
            exit 0
            ;;
    esac
done

# --- 检查 ADB ---
echo "=========================================="
echo " 一键部署 - 语音助手优化"
echo "=========================================="
echo ""

echo "[1/4] 检查 ADB 连接..."
if ! $ADB devices 2>/dev/null | grep -q "device$"; then
    echo "  ❌ ADB 未连接或没有设备"
    echo "  请确保板子已连接且 ADB 可用"
    exit 1
fi
echo "  ✅ ADB 已连接"
echo ""

# --- Step 1: Push run.sh ---
echo "[2/4] 推送启动脚本 run.sh..."
if [ "$CHECK_ONLY" -eq 0 ]; then
    # Check if run.sh exists locally
    if [ -f "${SCRIPT_DIR}/run.sh" ]; then
        $ADB push "${SCRIPT_DIR}/run.sh" "${BOARD_DIR}/run.sh" 2>/dev/null
        $ADB shell "chmod +x ${BOARD_DIR}/run.sh"
        echo "  ✅ run.sh 已推送"
    else
        echo "  ⚠️  run.sh 不存在 (跳过)"
    fi
else
    $ADB shell "test -f ${BOARD_DIR}/run.sh && echo '  ✅ run.sh 存在' || echo '  ❌ run.sh 不存在'"
fi
echo ""

# --- Step 2: Fix ASR init ---
echo "[3/4] 修复 ASR 初始化..."
if [ "$SKIP_ASR" -eq 0 ]; then
    if [ "$CHECK_ONLY" -eq 0 ]; then
        python3 "${SCRIPT_DIR}/fix_asr_init.py"
    else
        python3 "${SCRIPT_DIR}/fix_asr_init.py" --check
    fi
else
    echo "  ⏭️  已跳过"
fi
echo ""

# --- Step 3: VAD Tuning ---
echo "[4/4] VAD 参数调优..."
if [ "$SKIP_VAD" -eq 0 ]; then
    if [ "$CHECK_ONLY" -eq 0 ]; then
        python3 "${SCRIPT_DIR}/fix_vad_tuning.py"
    else
        echo "  (使用 --check 运行 fix_vad_tuning.py 进行检查)"
    fi
else
    echo "  ⏭️  已跳过"
fi
echo ""

# --- Summary ---
echo "=========================================="
echo " 部署完成"
echo "=========================================="
echo ""
echo "启动管道:"
echo "  adb shell /userdata/voice_assistant/run.sh"
echo ""
echo "查看日志:"
echo "  adb shell /userdata/voice_assistant/run.sh --verbose"
echo "=========================================="
