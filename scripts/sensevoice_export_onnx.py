#!/usr/bin/env python3
"""
SenseVoice 模型 ONNX 导出脚本
用于将 SenseVoice-Small 模型导出为 ONNX 格式，便于后续转换为 RKNN
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from pathlib import Path

# 添加 funasr 路径
import sys
sys.path.insert(0, '/home/ubuntu/桌面/ai/FunASR')

try:
    from funasr import AutoModel
except ImportError:
    print("请先安装 FunASR: pip install -U funasr")
    raise


class SenseVoiceEncoder(nn.Module):
    """
    SenseVoice 编码器包装类
    提取音频特征并进行情感识别
    """
    def __init__(self, model_dir="iic/SenseVoiceSmall"):
        super().__init__()
        # 加载预训练模型
        self.model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            disable_pbar=True,
        )
        self.model.eval()
        
    def forward(self, audio_input):
        """
        前向传播
        Args:
            audio_input: [batch, seq_len, feature_dim] 梅尔频谱特征
        Returns:
            情感识别结果
        """
        # 通过模型编码器
        encoder_out = self.model.model.encoder(audio_input)
        
        # 情感分类头
        # SenseVoice 使用 CTC 输出，包含情感标签
        ctc_logits = self.model.model.ctc(encoder_out)
        
        return ctc_logits


def export_sensevoice_to_onnx(
    model_dir="iic/SenseVoiceSmall",
    output_path="sensevoice_small.onnx",
    opset_version=12,
    max_seq_len=3000,
    feature_dim=80,
):
    """
    将 SenseVoice 模型导出为 ONNX 格式
    
    Args:
        model_dir: 模型目录或 HuggingFace 模型名
        output_path: 输出 ONNX 文件路径
        opset_version: ONNX opset 版本 (RKNN 推荐 12)
        max_seq_len: 最大序列长度
        feature_dim: 特征维度 (默认 80 维梅尔频谱)
    """
    print(f"Loading SenseVoice model from: {model_dir}")
    
    # 创建模型实例
    model = SenseVoiceEncoder(model_dir)
    model.eval()
    
    # 创建示例输入
    batch_size = 1
    dummy_input = torch.randn(batch_size, max_seq_len, feature_dim)
    
    print(f"Exporting to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Opset version: {opset_version}")
    
    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_input'],
        output_names=['ctc_logits'],
        dynamic_axes={
            'audio_input': {0: 'batch_size', 1: 'seq_len'},
            'ctc_logits': {0: 'batch_size', 1: 'seq_len'}
        }
    )
    
    print(f"ONNX model exported to: {output_path}")
    
    # 验证模型
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # 打印模型信息
    print("\nModel info:")
    print(f"  Input: {onnx_model.graph.input[0]}")
    print(f"  Output: {onnx_model.graph.output[0]}")
    
    return output_path


def export_with_frontend(
    model_dir="iic/SenseVoiceSmall",
    output_path="sensevoice_with_frontend.onnx",
    opset_version=12,
):
    """
    导出包含前处理（梅尔频谱提取）的完整模型
    输入为原始音频波形
    """
    print(f"Loading SenseVoice model with frontend from: {model_dir}")
    
    # 加载模型
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        disable_pbar=True,
    )
    model.eval()
    
    # 创建示例音频输入 (16kHz, 10秒)
    sample_rate = 16000
    duration = 10
    dummy_audio = torch.randn(1, sample_rate * duration)
    
    print(f"Exporting full model to ONNX...")
    print(f"  Audio input shape: {dummy_audio.shape}")
    
    # 导出
    torch.onnx.export(
        model.model,
        dummy_audio,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_waveform'],
        output_names=['ctc_output'],
        dynamic_axes={
            'audio_waveform': {0: 'batch_size', 1: 'audio_len'},
            'ctc_output': {0: 'batch_size', 1: 'seq_len'}
        }
    )
    
    print(f"Full ONNX model exported to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export SenseVoice to ONNX")
    parser.add_argument("--model-dir", default="iic/SenseVoiceSmall",
                        help="Model directory or HuggingFace model name")
    parser.add_argument("--output", "-o", default="sensevoice_small.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version (default: 12 for RKNN)")
    parser.add_argument("--with-frontend", action="store_true",
                        help="Include audio frontend (mel-spectrogram extraction)")
    parser.add_argument("--max-seq-len", type=int, default=3000,
                        help="Maximum sequence length")
    parser.add_argument("--feature-dim", type=int, default=80,
                        help="Feature dimension (default: 80 for mel-spectrogram)")
    
    args = parser.parse_args()
    
    if args.with_frontend:
        export_with_frontend(
            model_dir=args.model_dir,
            output_path=args.output,
            opset_version=args.opset,
        )
    else:
        export_sensevoice_to_onnx(
            model_dir=args.model_dir,
            output_path=args.output,
            opset_version=args.opset,
            max_seq_len=args.max_seq_len,
            feature_dim=args.feature_dim,
        )


if __name__ == "__main__":
    main()
