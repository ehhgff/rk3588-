#!/bin/bash
# Matcha TTS: RTF + 多线程 + 进程常驻 测试

SHERPA_BIN="/userdata/sherpa-onnx/install/bin"
SHERPA_LIB="/userdata/sherpa-onnx/install/lib"
MODEL_DIR="/userdata/sherpa-onnx/matcha-zh-baker"
OUT_DIR="/data"
export LD_LIBRARY_PATH=$SHERPA_LIB:$LD_LIBRARY_PATH
TEXT="来哥哥再给你唱首歌儿哎呦把伴奏给我放起来放就放嘛还要多人家钩子"

echo "============================================"
echo " Matcha TTS + Vocos 多线程 RTF 测试"
echo "============================================"
echo "模型: Matcha-icefall-zh-baker + Vocos-22khz"
echo ""

# ------ 1. 环境检查 ------
echo "--- [1/5] 环境检查 ---"
echo "  acoustic: $(ls -lh $MODEL_DIR/model-steps-3.onnx | awk '{print $5}')"
echo "  vocoder:  $(ls -lh $MODEL_DIR/vocos-22khz-univ.onnx | awk '{print $5}')"
echo "  tokens:   $(wc -l < $MODEL_DIR/tokens.txt)"
echo "  lexicon:  $(wc -l < $MODEL_DIR/lexicon.txt)"
free -m | head -2
echo ""

# ------ 2. 多线程 RTF 对比 (1~4 threads) ------
echo "--- [2/5] 多线程 RTF 对比 ---"
for t in 1 2 3 4; do
    t0=$(date +%s%N)
    $SHERPA_BIN/sherpa-onnx-offline-tts \
        --num-threads=$t \
        --matcha-acoustic-model=$MODEL_DIR/model-steps-3.onnx \
        --matcha-vocoder=$MODEL_DIR/vocos-22khz-univ.onnx \
        --matcha-lexicon=$MODEL_DIR/lexicon.txt \
        --matcha-tokens=$MODEL_DIR/tokens.txt \
        --output-filename=$OUT_DIR/matcha_t${t}.wav \
        "$TEXT" 2>&1 | grep -E 'Elapsed|duration|RTF|Saved'
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    echo "    --num-threads=$t: ${ms}ms"
done
echo ""

# ------ 3. 进程常驻测试 (单进程连续推理) ------
echo "--- [3/5] 常驻测试: 最快配置下连续推理3轮 ---"
BEST_THREADS=4
for i in 1 2 3; do
    t0=$(date +%s%N)
    $SHERPA_BIN/sherpa-onnx-offline-tts \
        --num-threads=$BEST_THREADS \
        --matcha-acoustic-model=$MODEL_DIR/model-steps-3.onnx \
        --matcha-vocoder=$MODEL_DIR/vocos-22khz-univ.onnx \
        --matcha-lexicon=$MODEL_DIR/lexicon.txt \
        --matcha-tokens=$MODEL_DIR/tokens.txt \
        --output-filename=$OUT_DIR/matcha_resident_${i}.wav \
        "$TEXT" 2>&1 | grep -E 'Elapsed|duration|RTF|Saved'
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    echo "    第${i}轮: ${ms}ms"
done
echo ""

# ------ 4. 内存分析 ------
echo "--- [4/5] 内存占用 ---"
free -m | head -2
echo ""

# ------ 5. 生成最终音频并 USB 播放 ------
echo "--- [5/5] 生成 + USB 播放 ---"
PLAY_DEVICE="plughw:1,0"

$SHERPA_BIN/sherpa-onnx-offline-tts \
    --num-threads=$BEST_THREADS \
    --matcha-acoustic-model=$MODEL_DIR/model-steps-3.onnx \
    --matcha-vocoder=$MODEL_DIR/vocos-22khz-univ.onnx \
    --matcha-lexicon=$MODEL_DIR/lexicon.txt \
    --matcha-tokens=$MODEL_DIR/tokens.txt \
    --output-filename=$OUT_DIR/matcha_final.wav \
    "$TEXT" 2>&1 | grep -E 'Elapsed|duration|RTF|Saved'

python3 -c "
import wave, numpy as np
with wave.open('$OUT_DIR/matcha_final.wav', 'rb') as wf:
    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    sr = wf.getframerate()
    dur = wf.getnframes() / sr
    print(f'  原始: {dur:.2f}s @ {sr}Hz')
n = int(len(data) * 48000 / sr)
x_old = np.linspace(0, 1, len(data))
x_new = np.linspace(0, 1, n)
resampled = np.interp(x_new, x_old, data).astype(np.int16)
stereo = np.column_stack((resampled, resampled))
with wave.open('$OUT_DIR/matcha_final_48k.wav', 'wb') as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(48000)
    wf.writeframes(stereo.tobytes())
" 2>&1

aplay -D $PLAY_DEVICE $OUT_DIR/matcha_final_48k.wav 2>&1 && echo "  ✅ 播放完成"

echo ""
echo "============================================"
echo " ✅ 测试完成"
echo "============================================"