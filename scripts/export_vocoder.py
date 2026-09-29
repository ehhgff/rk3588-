#!/usr/bin/env python3
"""导出 HiFiGAN Vocoder 为 ONNX 格式"""

import torch
from matcha.cli import load_hifigan

# 加载 vocoder
print("Loading HiFiGAN vocoder...")
vocoder = load_hifigan("generator_v1", "cpu")
vocoder.eval()

# 创建 dummy input (mel spectrogram)
dummy_input = torch.randn(1, 80, 100)  # [batch, n_mels, time]

# 导出 ONNX
output_path = "models/matcha_vocoder.onnx"
print(f"Exporting vocoder to {output_path}...")

torch.onnx.export(
    vocoder,
    dummy_input,
    output_path,
    input_names=["mel"],
    output_names=["audio"],
    dynamic_axes={
        "mel": {0: "batch_size", 2: "time"},
        "audio": {0: "batch_size", 2: "audio_length"}
    },
    opset_version=15,
    export_params=True,
    do_constant_folding=True
)

print(f"Vocoder exported to {output_path}")
