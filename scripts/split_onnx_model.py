#!/usr/bin/env python3
"""
拆分 ONNX 模型为两个子模型:
1. encoder: lfr_features → tp_encoders.19 输出 (INT8量化)
2. head: tp_encoders.19 输出 → ctc_logits (FP16)
"""
import onnx
from onnx.utils import Extractor

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
ENC_OUTPUT = "models/sensevoice_encoder_ctc_100f_encoder.onnx"
HEAD_OUTPUT = "models/sensevoice_encoder_ctc_100f_head.onnx"

print("=" * 60)
print("拆分 ONNX 模型")
print("=" * 60)

model = onnx.load(ONNX_PATH)
print(f"原始模型: {len(model.graph.node)} 节点")

# ---- 1. 提取 Encoder ----
# 输入: lfr_features, 输出: tp_encoders.19/Add_1_output_0
print("\n1. 提取 Encoder (lfr_features → tp_encoders.19)...")
try:
    extractor = Extractor(model)
    enc_model = extractor.extract_model(
        input_names=["lfr_features"],
        output_names=["/encoder/tp_encoders.19/Add_1_output_0"],
    )
    onnx.save(enc_model, ENC_OUTPUT)
    print(f"   ✅ {ENC_OUTPUT}")
    print(f"      inputs: {[i.name for i in enc_model.graph.input]}")
    print(f"      outputs: {[o.name for o in enc_model.graph.output]}")
    print(f"      节点数: {len(enc_model.graph.node)}")
except Exception as e:
    print(f"   ❌ Extractor 失败: {e}")
    print("   尝试手动提取...")

# ---- 2. 提取 Head (tp_norm + ctc_lo + LogSoftmax) ----
print("\n2. 提取 Head (tp_encoders.19 → ctc_logits)...")
try:
    head_model = extractor.extract_model(
        input_names=["/encoder/tp_encoders.19/Add_1_output_0"],
        output_names=["ctc_logits"],
    )
    onnx.save(head_model, HEAD_OUTPUT)
    print(f"   ✅ {HEAD_OUTPUT}")
    print(f"      inputs: {[i.name for i in head_model.graph.input]}")
    print(f"      outputs: {[o.name for o in head_model.graph.output]}")
    print(f"      节点数: {len(head_model.graph.node)}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# ---- 检查 ----
import os
for f in [ENC_OUTPUT, HEAD_OUTPUT]:
    if os.path.exists(f):
        sz = os.path.getsize(f) / (1024*1024)
        m = onnx.load(f)
        print(f"\n{f}: {sz:.1f}MB, inputs={[i.name for i in m.graph.input]}, outputs={[o.name for o in m.graph.output]}")
    else:
        print(f"\n⚠️ {f} 不存在")