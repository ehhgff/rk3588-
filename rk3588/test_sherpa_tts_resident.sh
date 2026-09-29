#!/bin/bash
# 使用 sherpa-onnx 测试 TTS 模型常驻 + RTF
# 使用已有的 VITS AISHELL3 模型
# 生成目标文本并通过 USB 播放

SHERPA_BIN="/userdata/sherpa-onnx/install/bin"
SHERPA_LIB="/userdata/sherpa-onnx/install/lib"
MODEL_DIR="/userdata/sherpa-onnx/models"
OUT_DIR="/data"
PLAY_DEVICE="plughw:1,0"
export LD_LIBRARY_PATH=$SHERPA_LIB:$LD_LIBRARY_PATH

TEXT="来哥哥再给你唱首歌儿哎呦把伴奏给我放起来放就放嘛还要多人家钩子"

echo "============================================"
echo " sherpa-onnx TTS 常驻 + RTF 测试"
echo "============================================"
echo ""
echo "模型: vits-aishell3.int8.onnx"
echo "文本: $TEXT"
echo ""

# ----------------- 第1步：环境检查 -----------------
echo "--- [1/4] 环境检查 ---"
echo "  $(ls -lh $MODEL_DIR/vits-aishell3.int8.onnx | awk '{print $5}') VITS model"
echo "  $(wc -l < $MODEL_DIR/tokens.txt) tokens, $(wc -l < $MODEL_DIR/lexicon.txt) lexicon entries"
mem=$(free -m | grep Mem | awk '{print $3}')
echo "  当前内存: ${mem}MB used"
echo ""

# ----------------- 第2步：CPU 多轮常驻测试 -----------------
echo "--- [2/4] CPU TTS 多轮常驻测试 (5轮) ---"
TOTAL_CPU_MS=0
for i in $(seq 1 5); do
    t0=$(date +%s%N)
    $SHERPA_BIN/sherpa-onnx-offline-tts \
        --vits-model=$MODEL_DIR/vits-aishell3.int8.onnx \
        --vits-tokens=$MODEL_DIR/tokens.txt \
        --vits-lexicon=$MODEL_DIR/lexicon.txt \
        --output-filename=$OUT_DIR/tts_round_${i}.wav \
        --provider=cpu \
        "$TEXT" 2>&1 | grep -E 'Elapsed|duration|RTF|Saved'
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    TOTAL_CPU_MS=$((TOTAL_CPU_MS + ms))
    echo "  第${i}轮总耗时: ${ms}ms"
done
AVG_CPU=$((TOTAL_CPU_MS / 5))
echo "  CPU 平均: ${AVG_CPU}ms/轮"
echo ""

# ----------------- 第3步：内存变化监控 -----------------
echo "--- [3/4] 常驻内存分析 ---"
mem_after=$(free -m | grep Mem | awk '{print $3}')
echo "  测试前内存: ${mem}MB used"
echo "  测试后内存: ${mem_after}MB used"
echo "  增量: $((mem_after - mem))MB"
echo ""

# ----------------- 第4步：直接生成并播放 -----------------
echo "--- [4/4] 直接生成并 USB 播放 ---"
# 先生成 WAV
t0=$(date +%s%N)
$SHERPA_BIN/sherpa-onnx-offline-tts \
    --vits-model=$MODEL_DIR/vits-aishell3.int8.onnx \
    --vits-tokens=$MODEL_DIR/tokens.txt \
    --vits-lexicon=$MODEL_DIR/lexicon.txt \
    --output-filename=$OUT_DIR/tts_final.wav \
    --provider=cpu \
    "$TEXT" 2>&1 | grep -E 'Elapsed|duration|RTF|Saved'
t1=$(date +%s%N)
gen_ms=$(( (t1 - t0) / 1000000 ))
echo "  生成耗时: ${gen_ms}ms"

# 重采样到 48kHz 立体声
python3 -c "
import wave, numpy as np
with wave.open('$OUT_DIR/tts_final.wav', 'rb') as wf:
    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    sr = wf.getframerate()
n = int(len(data) * 48000 / sr)
x_old = np.linspace(0, 1, len(data))
x_new = np.linspace(0, 1, n)
resampled = np.interp(x_new, x_old, data).astype(np.int16)
stereo = np.column_stack((resampled, resampled))
with wave.open('$OUT_DIR/tts_final_48k.wav', 'wb') as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(48000)
    wf.writeframes(stereo.tobytes())
dur = len(resampled) / 48000
print(f'  重采样完成: {dur:.1f}s')
" 2>&1

# 播放
echo "  播放..."
aplay -D $PLAY_DEVICE $OUT_DIR/tts_final_48k.wav 2>&1 && echo "  ✅ 播放完成"
echo ""

echo "============================================"
echo " ✅ 测试完成"
echo "============================================"