#!/usr/bin/env python3
"""
查看 ONNX 模型的输入输出节点名
用于找到 custom_hybrid 需要的正确 tensor 名
"""
import onnx

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"

model = onnx.load(ONNX_PATH)
graph = model.graph

print("=" * 60)
print("输入节点:")
for inp in graph.input:
    print(f"  {inp.name}")

print("\n输出节点:")
for out in graph.output:
    print(f"  {out.name}")

print("\n所有节点 (前50个):")
for i, node in enumerate(graph.node[:50]):
    inputs = ", ".join(node.input[:3])
    outputs = ", ".join(node.output[:3])
    print(f"  [{i:3d}] {node.op_type:15s}  {inputs[:60]:60s} → {outputs[:60]}")

print("\n所有节点 (后50个):")
n = len(graph.node)
for i, node in enumerate(graph.node[n-50:n]):
    idx = n - 50 + i
    inputs = ", ".join(node.input[:3])
    outputs = ", ".join(node.output[:3])
    print(f"  [{idx:3d}] {node.op_type:15s}  {inputs[:60]:60s} → {outputs[:60]}")

print("\nCTC相关节点:")
for i, node in enumerate(graph.node):
    if 'ctc' in node.name.lower() or 'log_softmax' in node.op_type.lower():
        inputs = ", ".join(node.input[:5])
        outputs = ", ".join(node.output[:3])
        print(f"  [{i:3d}] {node.op_type:15s}  name={node.name}  inputs({inputs})  →  outputs({outputs})")

print("\ntp_norm/norm相关节点:")
for i, node in enumerate(graph.node):
    if 'tp_norm' in node.name or 'tp' in node.name:
        inputs = ", ".join(node.input[:3])
        outputs = ", ".join(node.output[:3])
        print(f"  [{i:3d}] {node.op_type:15s}  {inputs[:50]} → {outputs[:50]}")