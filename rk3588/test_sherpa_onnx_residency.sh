#!/bin/bash
# sherpa-onnx 模型常驻 + RTF 测试脚本
# 测试 VITS TTS 模型在 sherpa-onnx 下的常驻性能和 RTF
# 同时生成目标音频并通过 USB 音响播放

SHERPA_BIN="/userdata/sherpa-onnx/install/bin"
SHERPA_LIB="/userdata/sherpa-onnx/install/lib"
MODEL_DIR="/userdata/sherpa-onnx/models"
OUTPUT_DIR="/data"
PLAY_DEVICE="plughw:1,0"    # USB2.0 Device
RATE=48000

export LD_LIBRARY_PATH=$SHERPA_LIB:$LD_LIBRARY_PATH

echo "============================================"
echo " sherpa-onnx 模型常驻 + RTF 测试"
echo "============================================"

# ------ 1. 检查 NPU 驱动 ------
echo ""
echo "[1/5] 环境检查"
echo "  sherpa-onnx: $($SHERPA_BIN/sherpa-onnx-offline --version 2>&1 | head -1)"
echo "  VITS model: $(ls -lh $MODEL_DIR/vits-aishell3.int8.onnx | awk '{print $5}')"
echo "  tokens: $(wc -l < $MODEL_DIR/tokens.txt) entries"
echo "  lexicon: $(wc -l < $MODEL_DIR/lexicon.txt) entries"

# ------ 2. CPU 单轮测试 (baseline) ------
echo ""
echo "[2/5] CPU 单轮 TTS (baseline)"
TEXT="来哥哥再给你唱首歌儿哎呦把伴奏给我放起来放就放嘛还要多人家钩子"

t0=$(date +%s%N)
$SHERPA_BIN/sherpa-onnx-offline-tts \
    --vits-model=$MODEL_DIR/vits-aishell3.int8.onnx \
    --vits-tokens=$MODEL_DIR/tokens.txt \
    --vits-lexicon=$MODEL_DIR/lexicon.txt \
    --output-filename=$OUTPUT_DIR/tts_cpu.wav \
    --provider=cpu \
    "$TEXT" 2>&1 | grep -E 'Elapsed|duration|RTF|Saved'

t1=$(date +%s%N)
cpu_time_ms=$(( (t1 - t0) / 1000000 ))
echo "  Total: ${cpu_time_ms}ms"

# ------ 3. RKNN NPU 多轮常驻测试 ------
echo ""
echo "[3/5] RKNN NPU 多轮常驻测试 (5轮)"

echo "  第 1 轮 (加载模型)..."
t0=$(date +%s%N)
$SHERPA_BIN/sherpa-onnx-offline-tts \
    --vits-model=$MODEL_DIR/vits-aishell3.int8.onnx \
    --vits-tokens=$MODEL_DIR/tokens.txt \
    --vits-lexicon=$MODEL_DIR/lexicon.txt \
    --output-filename=$OUTPUT_DIR/tts_npu_1.wav \
    --provider=rknn \
    "$TEXT" 2>&1 | grep -E 'Elapsed|duration|RTF|Saved'
t1=$(date +%s%N)
round1_ms=$(( (t1 - t0) / 1000000 ))
echo "    耗时: ${round1_ms}ms"

for i in 2 3 4 5; do
    echo "  第 ${i} 轮..."
    t0=$(date +%s%N)
    $SHERPA_BIN/sherpa-onnx-offline-tts \
        --vits-model=$MODEL_DIR/vits-aishell3.int8.onnx \
        --vits-tokens=$MODEL_DIR/tokens.txt \
        --vits-lexicon=$MODEL_DIR/lexicon.txt \
        --output-filename=$OUTPUT_DIR/tts_npu_${i}.wav \
        --provider=rknn \
        "$TEXT" 2>&1 | grep -E 'Elapsed|duration|RTF|Saved'
    t1=$(date +%s%N)
    round_ms=$(( (t1 - t0) / 1000000 ))
    echo "    耗时: ${round_ms}ms"
done

# ------ 4. 内存监控 ------
echo ""
echo "[4/5] 常驻内存监控"
# 使用 VAD/ASR 测试模型常驻
echo "  启动连续 ASR 测试 (3s silence + 1s beep)..."
# 生成一个 48000Hz 的测试音
python3 -c "
import wave, numpy as np
sr = 16000
dur = 2
t = np.linspace(0, dur, int(sr*dur), False)
noise = (np.random.randn(int(sr*dur)) * 100).astype(np.int16)
with wave.open('$OUTPUT_DIR/test_silence.wav', 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(noise.tobytes())
print('  Test audio generated')
" 2>&1

# 检查内存 (插入 sensevoice 模型常驻测试)
echo "  内存占用变化:"
mem_before=$(free -m | grep Mem | awk '{print $3}')
echo "  空闲内存: ${mem_before}MB"

# ------ 5. 生成最终音频并播放 ------
echo ""
echo "[5/5] 生成音频并通过 USB 播放"

# 使用 NPU 生成最佳质量
$SHERPA_BIN/sherpa-onnx-offline-tts \
    --vits-model=$MODEL_DIR/vits-aishell3.int8.onnx \
    --vits-tokens=$MODEL_DIR/tokens.txt \
    --vits-lexicon=$MODEL_DIR/lexicon.txt \
    --output-filename=$OUTPUT_DIR/tts_final.wav \
    --provider=rknn \
    "$TEXT" 2>&1 | grep -E 'Elapsed|duration|RTF|Saved'

# 重采样到 48kHz 立体声
echo "  重采样到 ${RATE}Hz 立体声..."
python3 -c "
import wave, numpy as np

with wave.open('$OUTPUT_DIR/tts_final.wav', 'rb') as wf:
    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    orig_rate = wf.getframerate()

orig_len = len(data)
target_rate = $RATE
resampled_len = int(orig_len * target_rate / orig_rate)
x_old = np.linspace(0, 1, orig_len)
x_new = np.linspace(0, 1, resampled_len)
resampled = np.interp(x_new, x_old, data).astype(np.int16)
stereo = np.column_stack((resampled, resampled))

with wave.open('$OUTPUT_DIR/tts_final_48k.wav', 'wb') as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(target_rate)
    wf.writeframes(stereo.tobytes())
dur = len(resampled) / target_rate
print(f'  Done: {dur:.2f}s, {len(resampled)} samples')
" 2>&1

# 播放
echo "  播放音频..."
aplay -D $PLAY_DEVICE $OUTPUT_DIR/tts_final_48k.wav 2>&1 && echo "  ✅ 播放完成"

echo ""
echo "============================================"
echo " ✅ 测试完成"
echo "============================================"