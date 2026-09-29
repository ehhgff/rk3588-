#!/usr/bin/env python3
"""
修复 Head ONNX 模型: 设置正确的输入形状
"""
import onnx
import numpy as np
from onnx import helper, TensorProto

HEAD_ONNX = "models/sensevoice_encoder_ctc_100f_head.onnx"
FIXED_ONNX = "models/sensevoice_encoder_ctc_100f_head_fixed.onnx"

model = onnx.load(HEAD_ONNX)
graph = model.graph

print("修复前:")
for inp in graph.input:
    print(f"  Input: {inp.name}, shape={[d.dim_value for d in inp.type.tensor_type.shape.dim]}")

# 创建新的输入 tensor proto，带正确的形状
new_input = helper.make_tensor_value_info(
    "/encoder/tp_encoders.19/Add_1_output_0",
    TensorProto.FLOAT,
    [1, 100, 512]  # encoder hidden size is 512
)

# 替换输入
graph.input.remove(graph.input[0])
graph.input.insert(0, new_input)

# 修复输出形状
for out in graph.output:
    print(f"  Output: {out.name}, shape={[d.dim_value for d in out.type.tensor_type.shape.dim]}")

print("\n修复后:")
for inp in graph.input:
    print(f"  Input: {inp.name}, shape={[d.dim_value for d in inp.type.tensor_type.shape.dim]}")

onnx.save(model, FIXED_ONNX)
print(f"\n已保存: {FIXED_ONNX}")

# 验证
m2 = onnx.load(FIXED_ONNX)
print(f"验证 - 输入: {[i.name for i in m2.graph.input]}")
for i in m2.graph.input:
    print(f"  shape: {[d.dim_value for d in i.type.tensor_type.shape.dim]}")

# 试试 onnx.checker
try:
    onnx.checker.check_model(m2)
    print("onnx.checker: ✅ PASS")
except Exception as e:
    print(f"onnx.checker: ⚠️ {e}")