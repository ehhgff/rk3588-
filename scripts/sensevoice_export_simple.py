#!/usr/bin/env python3
"""
SenseVoice 简化版 ONNX 导出脚本
由于 SenseVoice 模型结构复杂，此脚本创建一个简化版本用于 RKNN 转换演示
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse


class SimpleEmotionRecognizer(nn.Module):
    """
    简化的情感识别模型
    模拟 SenseVoice 的编码器 + CTC 结构
    """
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=4, num_emotions=7):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 编码器层 (使用 Transformer 编码器)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CTC 分类头
        self.ctc_head = nn.Linear(hidden_dim, num_emotions + 1)  # +1 for blank
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] 梅尔频谱
        Returns:
            logits: [batch, seq_len, num_classes] CTC logits
        """
        # 输入投影
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # 编码
        x = self.encoder(x)  # [batch, seq_len, hidden_dim]
        
        # CTC 输出
        logits = self.ctc_head(x)  # [batch, seq_len, num_classes]
        
        return logits


def export_sensevoice_simple(
    output_path="sensevoice_simple.onnx",
    input_dim=80,
    hidden_dim=256,
    num_layers=4,
    num_emotions=7,
    opset_version=12,
):
    """
    导出简化版 SenseVoice 模型
    
    Args:
        output_path: 输出 ONNX 路径
        input_dim: 输入特征维度 (默认 80 维梅尔频谱)
        hidden_dim: 隐藏层维度
        num_layers: 编码器层数
        num_emotions: 情感类别数
        opset_version: ONNX opset 版本
    """
    print("创建简化版 SenseVoice 模型...")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  编码器层数: {num_layers}")
    print(f"  情感类别: {num_emotions}")
    
    # 创建模型
    model = SimpleEmotionRecognizer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_emotions=num_emotions
    )
    model.eval()
    
    # 创建示例输入
    batch_size = 1
    seq_len = 1000
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"\n导出 ONNX...")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  Opset 版本: {opset_version}")
    
    # 导出 ONNX (使用 dynamo=False 避免 PyTorch 2.x 的新导出器)
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
        },
        dynamo=False,  # 使用旧版导出器
    )
    
    print(f"\nONNX 模型已导出到: {output_path}")
    
    # 验证模型
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型验证通过!")
    
    # 打印模型信息
    print(f"\n模型信息:")
    print(f"  输入: {onnx_model.graph.input[0].name}")
    print(f"  输出: {onnx_model.graph.output[0].name}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 获取文件大小
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {file_size:.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export simplified SenseVoice to ONNX")
    parser.add_argument("--output", "-o", default="sensevoice_simple.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--input-dim", type=int, default=80,
                        help="Input feature dimension (default: 80 for mel-spectrogram)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension (default: 256)")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of encoder layers (default: 4)")
    parser.add_argument("--num-emotions", type=int, default=7,
                        help="Number of emotion classes (default: 7)")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version (default: 12 for RKNN)")
    
    args = parser.parse_args()
    
    export_sensevoice_simple(
        output_path=args.output,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_emotions=args.num_emotions,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
