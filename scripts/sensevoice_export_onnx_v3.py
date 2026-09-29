#!/usr/bin/env python3
"""
SenseVoice ONNX 导出脚本 v3
导出 encoder + CTC (log_softmax) 部分
输入: 560-dim 特征（经过前端 LFR 处理后）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from funasr import AutoModel


class SenseVoiceEncoderCTC(nn.Module):
    """
    SenseVoice 编码器 + CTC 封装（推理模式）
    输入: 560-dim 特征（80-dim mel * 7 LFR 帧）
    输出: CTC log_softmax logits
    """
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.ctc_lo = model.model.ctc.ctc_lo  # Linear(512, 25055)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 560] LFR 特征
        Returns:
            logits: [batch, seq_len, vocab_size] CTC log_softmax
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        ilens = torch.tensor([seq_len] * batch_size, dtype=torch.long)
        
        # 编码器
        encoder_out, encoder_out_lens = self.encoder(x, ilens)
        
        # CTC 线性层 + log_softmax
        ctc_logits = F.log_softmax(self.ctc_lo(encoder_out), dim=2)
        
        return ctc_logits


def export_sensevoice_onnx(
    output_path="models/sensevoice_encoder_ctc.onnx",
    opset_version=14,
    seq_len=500,
):
    """
    导出 SenseVoice encoder + CTC 到 ONNX
    """
    print("加载 SenseVoice 模型...")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        disable_pbar=True,
    )
    model.model.eval()
    
    # 创建封装模型
    wrapped_model = SenseVoiceEncoderCTC(model)
    wrapped_model.eval()
    
    # 输入维度: 560 (80 mel * 7 LFR 帧)
    input_dim = 560
    
    # 创建示例输入
    batch_size = 1
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"\n导出 ONNX...")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  Opset 版本: {opset_version}")
    
    # 导出 ONNX (使用 dynamo=False 强制使用旧版导出器)
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['lfr_features'],
        output_names=['ctc_logits'],
        dynamic_axes={
            'lfr_features': {0: 'batch_size', 1: 'seq_len'},
            'ctc_logits': {0: 'batch_size', 1: 'seq_len'}
        },
        dynamo=False,
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
    
    # 获取文件大小
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {file_size:.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export SenseVoice encoder+CTC to ONNX")
    parser.add_argument("--output", "-o", default="models/sensevoice_encoder_ctc.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version (default: 14)")
    parser.add_argument("--seq-len", type=int, default=500,
                        help="Sequence length for dummy input (default: 500)")
    
    args = parser.parse_args()
    
    export_sensevoice_onnx(
        output_path=args.output,
        opset_version=args.opset,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()