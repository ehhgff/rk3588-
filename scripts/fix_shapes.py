#!/usr/bin/env python3
"""
修复 Encoder 和 Head ONNX 模型的输入/输出形状
用 shape_inference 自动推断
"""
import onnx
from onnx import helper, TensorProto
from onnx.shape_inference import infer_shapes

ENC_ONNX = "models/sensevoice_encoder_ctc_100f_encoder.onnx"
HEAD_ONNX = "models/sensevoice_encoder_ctc_100f_head.onnx"
ENC_FIXED = "models/sensevoice_encoder_ctc_100f_encoder_fixed.onnx"
HEAD_FIXED = "models/sensevoice_encoder_ctc_100f_head_fixed.onnx"

def fix_model(input_path, output_path, input_shapes=None, output_shapes=None):
    print(f"\n修复: {input_path}")
    model = onnx.load(input_path)
    
    # 先尝试 shape_inference
    try:
        model = infer_shapes(model)
        print("  shape_inference: ✅")
    except Exception as e:
        print(f"  shape_inference: ❌ {e}")
    
    # 手动设置形状（如果 shape_inference 没搞定）
    if input_shapes:
        for i, inp in enumerate(model.graph.input):
            if i < len(input_shapes):
                shape = input_shapes[i]
                new_tensor = helper.make_tensor_value_info(inp.name, TensorProto.FLOAT, shape)
                model.graph.input.remove(inp)
                model.graph.input.insert(i, new_tensor)
                print(f"  设置输入 {inp.name} → {shape}")
    
    if output_shapes:
        for i, out in enumerate(model.graph.output):
            if i < len(output_shapes):
                shape = output_shapes[i]
                # 获取当前输出 tensor 的元素类型
                elem_type = out.type.tensor_type.elem_type
                new_tensor = helper.make_tensor_value_info(out.name, elem_type, shape)
                model.graph.output.remove(out)
                model.graph.output.insert(i, new_tensor)
                print(f"  设置输出 {out.name} → {shape}")
    
    # 验证
    print(f"  输入:")
    for inp in model.graph.input:
        print(f"    {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print(f"  输出:")
    for out in model.graph.output:
        print(f"    {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
    
    onnx.save(model, output_path)
    print(f"  已保存: {output_path}")

# Encoder: input=(1,100,560), output=(1,100,512)
fix_model(ENC_ONNX, ENC_FIXED, 
    input_shapes=[(1, 100, 560)],
    output_shapes=[(1, 100, 512)])

# Head: input=(1,100,512), output=(1,100,25055)
fix_model(HEAD_ONNX, HEAD_FIXED,
    input_shapes=[(1, 100, 512)],
    output_shapes=[(1, 100, 25055)])