#!/bin/bash
# sherpa-onnx Matcha TTS Pipe 延迟测试
# 测试从发送文本到 pipe 到生成 WAV 文件的端到端延迟

echo "===== sherpa-onnx Matcha TTS Pipe 延迟测试 ====="
echo ""

# 清理旧文件
rm -f /tmp/tts_out_*.wav

declare -a TEXTS
TEXTS[0]="你好。"
TEXTS[1]="你好，请问有什么可以帮助你的吗？"
TEXTS[2]="今天天气真不错，我们一起去公园散步吧。"
TEXTS[3]="夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转。"

for i in "${!TEXTS[@]}"; do
    text="${TEXTS[$i]}"
    chars=$(echo -n "$text" | wc -c)

    echo "--- 测试 $((i+1)): ${chars}字 ---"
    echo "文本: $text"

    start=$(date +%s%N)

    echo "$text" > /tmp/tts_pipe

    out_file=""
    existing=$(ls /tmp/tts_out_*.wav 2>/dev/null | sort)
    for n in $(seq 1 200); do
        new_files=$(ls /tmp/tts_out_*.wav 2>/dev/null | sort)
        diff=$(comm -13 <(echo "$existing") <(echo "$new_files") 2>/dev/null)
        if [ -n "$diff" ]; then
            out_file=$(echo "$diff" | head -1)
            break
        fi
        sleep 0.01
    done

    end=$(date +%s%N)
    elapsed_ms=$(( (end - start) / 1000000 ))

    if [ -n "$out_file" ]; then
        file_size=$(stat -c%s "$out_file" 2>/dev/null || echo "?")
        echo "   文件: $out_file"
        echo "   大小: ${file_size} bytes"
        echo "   ⏱ 延迟: ${elapsed_ms}ms"
    else
        echo "   ❌ 超时未生成文件"
    fi
    echo ""
done

echo "===== 测试完成 ====="