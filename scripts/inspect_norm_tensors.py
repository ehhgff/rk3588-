#!/usr/bin/env python3
"""
精确找到每个 exNorm 子图的数据输入/输出 tensor，仅使用精确匹配。
"""
import onnx, os, re

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
CFG_PATH = "sensevoice_encoder_ctc_100f.quantization.cfg"

model = onnx.load(ONNX_PATH)
graph = model.graph

# 加载 cfg tensor 名称
cfg_tensors = set()
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(':') and not line.startswith('#') and not line.startswith('-'):
                name = line.rstrip(':').strip()
                name = name.split()[-1] if ' ' in name else name
                cfg_tensors.add(name)

def precise_match(onnx_name):
    """精确匹配 - 只返回确定性的匹配"""
    if onnx_name in cfg_tensors:
        return onnx_name
    # 常见后缀
    for suffix in ['-rs', '_tp-rs', '_int8']:
        cand = onnx_name + suffix
        if cand in cfg_tensors:
            return cand
    # 精确路径匹配：保留完整路径，只尝试后缀
    if '/' in onnx_name:
        parts = onnx_name.rsplit('/', 1)
        for suffix in ['-rs', '_tp-rs', '_int8']:
            cand = parts[0] + '/' + parts[1] + suffix
            if cand in cfg_tensors:
                return cand
    return None

# 找到所有 norm 子图路径
norm_paths = set()
pat = re.compile(r'/encoder/encoders\d*\.?\d+/(norm[12])/')
for node in graph.node:
    m = pat.search(node.name)
    if m:
        path = node.name[:m.end()].rstrip('/')
        norm_paths.add(path)
norm_paths = sorted(norm_paths)

print(f"找到 {len(norm_paths)} 个 exNorm 子图\n")

custom_hybrid = []
errors = 0

for np in norm_paths:
    norm_nodes = [n for n in graph.node if n.name.startswith(np + '/')]
    if not norm_nodes:
        continue
    
    # 找到子图的输入/输出 tensor
    internal_outputs = set()
    internal_inputs = set()
    for n in norm_nodes:
        for o in n.output:
            internal_outputs.add(o)
        for i in n.input:
            internal_inputs.add(i)
    
    # 数据输入：外部产生、内部消费，且不是 weight/bias
    data_inputs = [t for t in (internal_inputs - internal_outputs) if not t.startswith('onnx::')]
    
    # 数据输出：内部产生，且被外部 node 消费
    data_outputs = []
    for t in internal_outputs:
        for n in graph.node:
            if not n.name.startswith(np + '/'):
                if t in n.input:
                    data_outputs.append(t)
                    break
    # fallback: 取最后一个 node 的输出
    if not data_outputs:
        data_outputs = [norm_nodes[-1].output[0]]
    
    if not data_inputs or not data_outputs:
        errors += 1
        continue
    
    # 尝试所有可能的 data_input，找到第一个匹配 cfg 的
    in_cfg = None
    in_match = None
    for t in data_inputs:
        m = precise_match(t)
        if m:
            in_cfg = m
            in_match = t
            break
    
    # 尝试所有可能的 data_output，找到第一个匹配 cfg 的
    out_cfg = None
    out_match = None
    for t in data_outputs:
        m = precise_match(t)
        if m:
            out_cfg = m
            out_match = t
            break
    
    if in_cfg and out_cfg:
        custom_hybrid.append([in_cfg, out_cfg])
    else:
        errors += 1
        if errors <= 10:
            print(f"  ⚠️ {np}")
            if not in_cfg:
                print(f"      inputs: {data_inputs}")
                print(f"      →  NO MATCH in cfg")
            if not out_cfg:
                print(f"      outputs: {data_outputs}")
                print(f"      →  NO MATCH in cfg")

# 处理 tp_norm
for node in graph.node:
    if 'tp_norm' in node.name and node.op_type == 'LayerNormalization':
        data_input = None
        for inp in node.input:
            if not inp.startswith('onnx::'):
                data_input = inp
                break
        if data_input:
            data_output = node.output[0]
            in_cfg = precise_match(data_input)
            out_cfg = precise_match(data_output)
            if in_cfg and out_cfg:
                custom_hybrid.append([in_cfg, out_cfg])
            else:
                print(f"  ⚠️ tp_norm: no match")
        break

# 去重
seen = set()
unique = []
for ch in custom_hybrid:
    key = (ch[0], ch[1])
    if key not in seen:
        seen.add(key)
        unique.append(ch)

# 验证 - 检查是否有跨块子图
print(f"\n{'='*60}")
print(f"共 {len(unique)} 个 custom_hybrid 子图 (跳过 {errors} 个)")
print(f"{'='*60}")

# 检查跨块
cross_block = 0
for ch in unique:
    in_block = re.search(r'encoders[\d.]+', ch[0])
    out_block = re.search(r'encoders[\d.]+', ch[1])
    if in_block and out_block and in_block.group() != out_block.group():
        cross_block += 1
        if cross_block <= 3:
            print(f"  ⚠️ 跨块: {ch[0]} → {ch[1]}")

# 直接输出（没有跨块问题则使用）
if cross_block == 0:
    print("\n✅ 无跨块子图，可以安全使用")
    
# 打印前 10 个和最后 5 个确认
print("\n--- 前 10 个 ---")
for ch in unique[:10]:
    print(f"  {ch[0]}")
    print(f"  → {ch[1]}")
    print()

print("--- custom_hybrid 配置 ---")
print("custom_hybrid = [")
for ch in unique:
    print(f"    [{ch[0]!r}, {ch[1]!r}],")
print("]")