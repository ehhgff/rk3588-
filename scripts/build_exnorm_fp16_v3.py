#!/usr/bin/env python3
"""
使用 hybrid_quantization_step1 + custom_hybrid 方法，
指定 exNorm 层子图为 FP16。

从 .model 文件中解析 exNorm 节点的输入/输出 tensor 名称，
然后用 custom_hybrid 参数指定每个 exNorm 子图。
"""
import os, re, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
OUTPUT = "models/sensevoice_encoder_ctc_100f_exnorm_fp16.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# 1. 从 .model 文件中解析 exNorm 节点的输入/输出 tensor
# ============================================================
# .model 文件必须存在（由之前的 step1 生成）
model_file = None
for f in os.listdir('.'):
    if f.endswith('.model') and '100f' in f:
        model_file = f
        break

if not model_file:
    log("❌ 找不到 .model 文件")
    sys.exit(1)

log(f"解析 {model_file}...")

with open(model_file, 'rb') as f:
    raw = f.read()

# 找到所有 "exNorm" 的位置
exnorm_positions = []
idx = 0
while True:
    idx = raw.find(b'exNorm', idx)
    if idx == -1:
        break
    exnorm_positions.append(idx)
    idx += 1

log(f"找到 {len(exnorm_positions)} 个 exNorm 节点")

# 对每个 exNorm，提取其输入 tensor 名称（在 node 定义之前的名称）
# .model 文件格式: ...input_tensor_name\n...node_name"exNorm
# 输入 tensor 是 exNorm node 之前最后一个非 node 行
# node 行以 "op_type" 结尾
# 非 node 行是简单的 tensor 名称

def is_node_line(text):
    """判断一行是否为 node 定义（以 "op_type 结尾）"""
    return '"' in text.rstrip()[-3:] or '#' in text

def extract_tensor_name(line):
    """从二进制行提取 tensor 名称"""
    line = line.rstrip(b'\x00').rstrip()
    # 去掉可能的前缀字符（长度字节）
    while line and (line[0:1] in [b'!', b'"', b'#', b'$', b'%', b'&', b"'", b'(', b')', b'*', b'+', b',', b'-', b'.', b'/'] or line[0:1].isdigit()):
        # 这些可能是二进制格式的长度前缀或其他标记
        if line[0:1] == b'\x00' or line[0:1] == b'\x01':
            break
        line = line[1:]
    return line.decode('ascii', errors='ignore').strip()

# 对每个 exNorm 位置，往前搜索 tensor 名称
custom_hybrid = []

for pos in exnorm_positions:
    # 搜索区域: 从 exNorm 位置往前 500 字节
    start = max(0, pos - 500)
    chunk = raw[start:pos]
    
    # 分割成行（以 \n 分割）
    lines = chunk.split(b'\n')
    
    # 从后往前找：找到第一个非 node 行（这就是输入 tensor）
    input_tensor = None
    output_tensor = None
    
    # 先提取输出 tensor（可能在 exNorm 同一行中）
    # 格式可能是 "output_tensornode_name"exNorm
    exnorm_line_start = max(0, pos - 200)
    exnorm_chunk = raw[exnorm_line_start:pos+6]
    
    # 在 exNorm 之前提取 output tensor name
    # 格式: output_name"exNorm
    out_match = re.search(rb'([\w/\.\-_]+)"exNorm', exnorm_chunk)
    if out_match:
        output_tensor = out_match.group(1).decode('ascii', errors='ignore')
    
    # 从行中找输入 tensor
    for line in reversed(lines):
        line_str = line.decode('ascii', errors='ignore').strip()
        if not line_str or line_str.startswith('x00'):
            continue
        # 跳过 node 定义行（包含 "op_type）
        if '"' in line_str and ('"' + line_str.split('"')[-1]).strip():
            # 可能是 node 行
            if any(op in line_str for op in ['"exNorm', '"Conv', '"Add', '"Mul', '"Reshape', '"Transpose', '"Split', '"Relu', '"Softmax', '"Concat', '"Sigmoid', '"exSDPAttention', '"ReduceMean', '"Clip', '"Sub', '"Pow', '"Div', '"Sqrt', '"Cast', '"Pad']):
                continue
        
        # 这是一个 tensor 名称行
        # 提取名称（去掉前缀字符）
        tname = line_str
        while tname and (not tname[0].isalnum() and tname[0] not in ['/', '_']):
            tname = tname[1:]
        
        if tname and ('onnx' in tname or 'encoder' in tname or tname.startswith('/')):
            if 'onnx::Mul' not in tname and 'onnx::Add' not in tname:
                # 这不是 weight/bias，而是数据 tensor
                input_tensor = tname
                break
    
    if input_tensor and output_tensor:
        # 检查是否在 cfg 中存在
        custom_hybrid.append([input_tensor, output_tensor])
        log(f"  子图: [{input_tensor[:50]}...] → [{output_tensor[:50]}...]")
    else:
        log(f"  ⚠️ 无法提取输入/输出 at pos {pos}")

log(f"\n共提取 {len(custom_hybrid)} 个 exNorm 子图")

# ============================================================
# 2. 运行 step1 + step2
# ============================================================
log("=" * 60)
log("运行 hybrid_quantization_step1 (with custom_hybrid)")
log("=" * 60)

if os.path.exists(OUTPUT):
    log(f"✅ 已存在: {OUTPUT}")
    sys.exit(0)

rknn = RKNN(verbose=False)
rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")

rknn.load_onnx(model=ONNX_PATH)

ret = rknn.hybrid_quantization_step1(
    dataset=DATASET,
    proposal=False,
    custom_hybrid=custom_hybrid,
)

if ret != 0:
    log(f"❌ step1 失败: ret={ret}")
    # 常见的错误: tensor name 不存在
    # 尝试不使用 custom_hybrid，只做一般 step1
    log("尝试不使用 custom_hybrid 重新运行 step1...")
    rknn.release()
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
    rknn.load_onnx(model=ONNX_PATH)
    ret = rknn.hybrid_quantization_step1(dataset=DATASET, proposal=False)
    if ret != 0:
        log(f"step1 也失败: ret={ret}")
        sys.exit(1)
    log("step1 成功（without custom_hybrid）")
    
    # 尝试用 save_hybrid_quantization_config 保存配置
    try:
        cfg = rknn.save_hybrid_quantization_config()
        log(f"save_hybrid_quantization_config: {cfg}")
    except:
        pass

# 输出生成的文件
for f in sorted(os.listdir('.')):
    if '100f' in f and (f.endswith('.model') or f.endswith('.data') or f.endswith('.cfg')):
        size = os.path.getsize(f)
        log(f"  生成: {f} ({size/1024/1024:.1f}MB)" if size > 1024*1024 else f"  生成: {f} ({size/1024:.1f}KB)")

rknn.release()

# 如果模型没有直接生成，检查是否需要运行 step2
model_path = None
data_path = None
for f in sorted(os.listdir('.')):
    if f.endswith('.model') and '100f' in f:
        model_path = f
    elif f.endswith('.data') and '100f' in f:
        data_path = f

if not os.path.exists(OUTPUT) and model_path and data_path:
    log("\n" + "=" * 60)
    log("运行 hybrid_quantization_step2")
    log("=" * 60)
    rknn = RKNN(verbose=False)
    
    # 找 quantization.cfg
    cfg_path = None
    for f in os.listdir('.'):
        if f.endswith('.quantization.cfg') and '100f' in f:
            cfg_path = f
            break
    
    if not cfg_path:
        log("找不到 quantization.cfg")
        sys.exit(1)
    
    ret = rknn.hybrid_quantization_step2(model_path, data_path, cfg_path)
    if ret != 0:
        log(f"step2 失败: ret={ret}")
        sys.exit(1)
    
    rknn.export_rknn(OUTPUT)
    rknn.release()
    log(f"✅ 模型已保存: {OUTPUT}")

if os.path.exists(OUTPUT):
    size = os.path.getsize(OUTPUT)
    log(f"✅ 最终模型: {OUTPUT} ({size/1024/1024:.1f}MB)")