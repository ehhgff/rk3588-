#!/usr/bin/env python3
"""
SenseVoice ONNX 导出脚本 v4
使用固定输入尺寸（兼容 RKNN）
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
    SenseVoice 编码器 + CTC 封装（推理模式，固定输入尺寸）
    """
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.ctc_lo = model.model.ctc.ctc_lo
        
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        ilens = torch.tensor([seq_len] * batch_size, dtype=torch.long)
        encoder_out, encoder_out_lens = self.encoder(x, ilens)
        ctc_logits = F.log_softmax(self.ctc_lo(encoder_out), dim=2)
        return ctc_logits


def export_sensevoice_onnx_fixed(
    output_path="models/models/sensevoice_encoder_ctc_fixed.onnx",
    opset_version=14,
    batch_size=1,
    seq_len=500,
):
    """
    导出 SenseVoice encoder + CTC 到 ONNX（固定输入尺寸）
    """
    print("加载 SenseVoice 模型...")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        disable_pbar=True,
    )
    model.model.eval()
    
    wrapped_model = SenseVoiceEncoderCTC(model)
    wrapped_model.eval()
    
    input_dim = 560
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"\n导出 ONNX...")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  Opset 版本: {opset_version}")
    
    # 导出 ONNX - 不使用 dynamic_axes
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['lfr_features'],
        output_names=['ctc_logits'],
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
    for inp in onnx_model.graph.input:
        print(f"  输入: {inp.name}, shape: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    for out in onnx_model.graph.output:
        print(f"  输出: {out.name}, shape: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {file_size:.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export SenseVoice to ONNX (fixed shape)")
    parser.add_argument("--output", "-o", default="models/models/sensevoice_encoder_ctc_fixed.onnx")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=500,
                        help="Fixed sequence length (default: 500 = ~5s audio)")
    
    args = parser.parse_args()
    
    export_sensevoice_onnx_fixed(
        output_path=args.output,
        opset_version=args.opset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()