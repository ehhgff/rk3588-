#!/bin/bash
"""  # (shebang for editor)
echo "这是板端运行脚本，请在 shell 中执行"
exit 0
"""
# ============================================================
# 性能分析主脚本 — 在 RK3588 板端执行
# 依次运行: SER → ASR → TTS → LLM
# ============================================================
SCRIPT_DIR="/data/voice_assistant"
RESULTS_DIR="/data/perf_analysis"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$RESULTS_DIR/run.log"; }

log "========================================"
log "RK3588 模型性能深度分析"
log "========================================"
log ""

# 1. 环境信息
log "--- 环境信息 ---"
echo "NPU频率:" > "$RESULTS_DIR/environment.txt"
cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null >> "$RESULTS_DIR/environment.txt"
echo "CPU信息:" >> "$RESULTS_DIR/environment.txt"
cat /proc/cpuinfo | grep "model name" | head -1 >> "$RESULTS_DIR/environment.txt"
echo "内存:" >> "$RESULTS_DIR/environment.txt"
free -h >> "$RESULTS_DIR/environment.txt"
echo "NPU负载(空闲):" >> "$RESULTS_DIR/environment.txt"
cat /sys/kernel/debug/rknpu/load 2>/dev/null >> "$RESULTS_DIR/environment.txt"
echo "NPU可用核心:" >> "$RESULTS_DIR/environment.txt"
cat /sys/class/devfreq/fdab0000.npu/available_frequencies 2>/dev/null >> "$RESULTS_DIR/environment.txt"
echo "rknn_server版本:" >> "$RESULTS_DIR/environment.txt"
rknn_server --version 2>/dev/null || echo "N/A" >> "$RESULTS_DIR/environment.txt"
cat "$RESULTS_DIR/environment.txt"

# 2. SER 分析
log ""
log "--- [1/4] SER (SenseVoice) 分析 ---"
if python3 "$SCRIPT_DIR/analyze_ser.py" 2>&1 | tee "$RESULTS_DIR/ser_analysis.log"; then
    cp /tmp/analyze_ser_results.json "$RESULTS_DIR/" 2>/dev/null
    log "SER 分析完成"
else
    log "SER 分析失败"
fi

# 3. ASR 分析
log ""
log "--- [2/4] ASR (Zipformer) 分析 ---"
if python3 "$SCRIPT_DIR/analyze_asr.py" 2>&1 | tee "$RESULTS_DIR/asr_analysis.log"; then
    cp /tmp/analyze_asr_results.json "$RESULTS_DIR/" 2>/dev/null
    log "ASR 分析完成"
else
    log "ASR 分析失败"
fi

# 4. TTS 分析
log ""
log "--- [3/4] TTS (Matcha-TTS) 分析 ---"
if python3 "$SCRIPT_DIR/analyze_tts.py" 2>&1 | tee "$RESULTS_DIR/tts_analysis.log"; then
    cp /tmp/analyze_tts_results.json "$RESULTS_DIR/" 2>/dev/null
    log "TTS 分析完成"
else
    log "TTS 分析失败"
fi

# 5. LLM 分析
log ""
log "--- [4/4] LLM (Qwen3-0.6B) 分析 ---"
if python3 "$SCRIPT_DIR/analyze_llm.py" 2>&1 | tee "$RESULTS_DIR/llm_analysis.log"; then
    cp /tmp/analyze_llm_results.json "$RESULTS_DIR/" 2>/dev/null
    log "LLM 分析完成"
else
    log "LLM 分析失败"
fi

log ""
log "========================================"
log "所有分析完成"
log "结果目录: $RESULTS_DIR"
log "========================================"
ls -lh "$RESULTS_DIR/"