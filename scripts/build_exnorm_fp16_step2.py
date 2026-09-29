#!/usr/bin/env python3
"""
从 step1 生成的 .quantization.cfg 中找到 exNorm 相关 tensor，
将它们的 dtype 改为 float16，然后运行 step2 构建混合量化模型。
"""
import os, re, sys, time
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
OUTPUT = "models/sensevoice_encoder_ctc_100f_exnorm_fp16.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# 1. 找到 step1 生成的文件
# ============================================================
model_file = None
data_file = None
cfg_file = None
for f in os.listdir('.'):
    if f.endswith('.model') and '100f' in f:
        model_file = f
    elif f.endswith('.data') and '100f' in f:
        data_file = f
    elif f.endswith('.quantization.cfg') and '100f' in f:
        cfg_file = f

log(f"model: {model_file}")
log(f"data:  {data_file}")
log(f"cfg:   {cfg_file}")

if not all([model_file, data_file, cfg_file]):
    log("缺少必要文件，运行 step1...")
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
    rknn.load_onnx(model=ONNX_PATH)
    ret = rknn.hybrid_quantization_step1(dataset=DATASET, proposal=False)
    if ret != 0:
        log(f"step1 失败: ret={ret}")
        sys.exit(1)
    # 重新查找
    for f in os.listdir('.'):
        if f.endswith('.model') and '100f' in f:
            model_file = f
        elif f.endswith('.data') and '100f' in f:
            data_file = f
        elif f.endswith('.quantization.cfg') and '100f' in f:
            cfg_file = f
    log(f"step1 完成: {model_file}, {data_file}, {cfg_file}")
    rknn.release()

# ============================================================
# 2. 解析 .model 文件找到 exNorm 相关的 tensor 名称
# ============================================================
log("解析 .model 文件中的 exNorm 节点...")

with open(model_file, 'rb') as f:
    raw = f.read()

# .model 是混合二进制/文本格式，搜索 exNorm 节点
# 格式: "tensor_name/tensor_name"op_type"
# exNorm 节点格式: [...input_tensors...]output_names/output_name"exNorm"

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

# 对所有 ONNX 节点（可能包含 fp32 weights），找出所有 unique 的 onnx::Mul 和 onnx::Add 名称
# 这些是 exNorm 的 weight 和 bias
exnorm_tensors = set()

# 对于每个 exNorm，往回找 weight/bias tensor 名称
for pos in exnorm_positions:
    # 从 exNorm 位置往前 500 字节查找
    start = max(0, pos - 500)
    chunk = raw[start:pos]
    
    # 找到 chunk 中所有 tensor 名称（以换行或特殊字符分隔）
    # 提取 onnx::Mul_XXXX 和 onnx::Add_XXXX 名称
    for m in re.finditer(rb'([\w/\.]+_output_\d[\w\-]*|onnx::\w+_\d+)', chunk):
        name = m.group(1).decode('ascii', errors='ignore')
        exnorm_tensors.add(name)

log(f"提取到 {len(exnorm_tensors)} 个候选 tensor")

# 从 model 中找出所有 exNorm 节点的 output tensor names
exnorm_outputs = set()
for pos in exnorm_positions:
    # 从 exNorm 位置往回找 output name（在 "exNorm" 之前的字符串）
    start_region = max(0, pos - 200)
    chunk = raw[start_region:pos]
    # 找可能的 output tensor name: 以 / 或 onnx:: 开头，紧接着 exNorm
    # 在二进制格式中，output name 和 node name 可能重复
    for m in re.finditer(rb'([\w/\.]+_output_\d[\w\-]*|[/\w]+[\w\.]*\w)', chunk):
        name = m.group(1).decode('ascii', errors='ignore')
        if 'onnx' in name or 'encoder' in name or name.startswith('/'):
            exnorm_tensors.add(name)

# ============================================================
# 3. 找到所有 norm 相关的 tensor
# ============================================================
norm_patterns = [
    r'encoders\d+\.\d+/norm[12]/',
    r'encoders\.\d+/norm[12]/',
    r'tp_norm/',
]
exnorm_tensors_norm = set()
for t in exnorm_tensors:
    for pat in norm_patterns:
        if re.search(pat, t):
            exnorm_tensors_norm.add(t)

log(f"norm 相关 tensor: {len(exnorm_tensors_norm)}")
for t in sorted(exnorm_tensors_norm)[:10]:
    log(f"  {t}")

# 也需要找出 onnx::Mul_XXXX/onnx::Add_XXXX 中与 exNorm 关联的
# 从 model 中更仔细地提取
exnorm_weight_tensors = set()
with open(model_file, 'rb') as f:
    content = f.read()

# 对每个 exNorm 位置，提取其前面最近的 onnx::Mul 和 onnx::Add
for pos in exnorm_positions:
    start = max(0, pos - 300)
    chunk = content[start:pos].decode('ascii', errors='ignore')
    # 提取 chunk 中所有 onnx::Mul_ 和 onnx::Add_ 模式的名称
    for m in re.finditer(r'(onnx::(?:Mul|Add)_\d+)', chunk):
        exnorm_weight_tensors.add(m.group(1))

log(f"exNorm weight/bias tensors: {len(exnorm_weight_tensors)}")
for t in sorted(list(exnorm_weight_tensors))[:5]:
    log(f"  {t}")

# ============================================================
# 4. 修改 .quantization.cfg
# ============================================================
log("修改量化配置...")

with open(cfg_file, 'r') as f:
    cfg = f.read()

# 找到所有 norm 相关 tensor 在 cfg 中的完整条目并修改 dtype
# 使用正则替换
modified_count = 0

# 修正 tensor 名称: .model 文件可能包含 "./" 前缀
def normalize_tensor_name(name):
    name = name.strip()
    if name.startswith('./'):
        name = name[1:]
    return name

# 修改 norm output tensors
for t_raw in exnorm_tensors_norm:
    t = normalize_tensor_name(t_raw)
    # 在 cfg 中查找该 tensor 名称后紧跟的 dtype: int8
    pattern = re.compile(
        rf'({re.escape(t)}:\s*\n(?:.*\n)*?)(\s+)(dtype:\s*)int8',
        re.MULTILINE
    )
    new_cfg, count = pattern.subn(r'\1\2\3float16', cfg)
    if count > 0:
        modified_count += count
        cfg = new_cfg
        log(f"  ✅ 修改: {t}")
    else:
        log(f"  ❌ 未找到: {t}")

# 修改 weight/bias tensors
for t_raw in exnorm_weight_tensors:
    t = normalize_tensor_name(t_raw)
    pattern = re.compile(
        rf'({re.escape(t)}:\s*\n(?:.*\n)*?)(\s+)(dtype:\s*)int8',
        re.MULTILINE
    )
    new_cfg, count = pattern.subn(r'\1\2\3float16', cfg)
    if count > 0:
        modified_count += count
        cfg = new_cfg
        log(f"  ✅ 修改: {t}")
    else:
        log(f"  ❌ 未找到: {t}")

# 还要修改所有 ReduceMean_2ln 开头的 tensor（可能是 exNorm 的 output）
# 这些已经在 exnorm_tensors_norm 中了

log(f"修改了 {modified_count} 个 tensor 的 dtype")

# 保存修改后的 cfg
cfg_modified = cfg_file + '.modified'
with open(cfg_modified, 'w') as f:
    f.write(cfg)

# ============================================================
# 5. 运行 step2
# ============================================================
log("运行 hybrid_quantization_step2...")

rknn = RKNN(verbose=False)
ret = rknn.hybrid_quantization_step2(model_file, data_file, cfg_modified)
if ret != 0:
    log(f"step2 失败: ret={ret}")
    sys.exit(1)

rknn.export_rknn(OUTPUT)
rknn.release()
log(f"✅ 模型已保存: {OUTPUT}")