#!/usr/bin/env python3
"""
修改 .quantization.cfg，仅将 exNorm 的 weight/bias tensor 设为 float16，
跳过 output/activation tensor（RKNN 不允许修改那些）。
"""
import os, re, sys, time
from rknn.api import RKNN

DATASET = "calib_data/data_real/dataset.txt"
ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
OUTPUT = "models/sensevoice_encoder_ctc_100f_exnorm_fp16.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# 1. 找到文件
# ============================================================
model_file = data_file = cfg_file = None
for f in os.listdir('.'):
    if f.endswith('.model') and '100f' in f: model_file = f
    elif f.endswith('.data') and '100f' in f: data_file = f
    elif f.endswith('.quantization.cfg') and '100f' in f: cfg_file = f

if not all([model_file, data_file, cfg_file]):
    log("运行 step1...")
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
    rknn.load_onnx(model=ONNX_PATH)
    ret = rknn.hybrid_quantization_step1(dataset=DATASET, proposal=False)
    assert ret == 0, f"step1 failed: {ret}"
    for f in os.listdir('.'):
        if f.endswith('.model') and '100f' in f: model_file = f
        elif f.endswith('.data') and '100f' in f: data_file = f
        elif f.endswith('.quantization.cfg') and '100f' in f: cfg_file = f
    rknn.release()

log(f"model: {model_file}")
log(f"data:  {data_file}")
log(f"cfg:   {cfg_file}")

# ============================================================
# 2. 解析 .model 找到 exNorm 的 weight/bias tensor 名称
# ============================================================
with open(model_file, 'rb') as f:
    raw = f.read()

# 找到所有 exNorm 位置
exnorm_positions = []
idx = 0
while True:
    idx = raw.find(b'exNorm', idx)
    if idx == -1: break
    exnorm_positions.append(idx)
    idx += 1

log(f"找到 {len(exnorm_positions)} 个 exNorm 节点")

# 对每个 exNorm，提取其前面的 onnx::Mul_ 和 onnx::Add_ 名称
weight_bias_tensors = set()
for pos in exnorm_positions:
    # 前 500 字节读取
    start = max(0, pos - 500)
    chunk = raw[start:pos].decode('ascii', errors='ignore')
    for m in re.finditer(r'(onnx::(?:Mul|Add)_\d+)', chunk):
        weight_bias_tensors.add(m.group(1))

log(f"提取到 {len(weight_bias_tensors)} 个 weight/bias tensor")

# ============================================================
# 3. 修改 cfg
# ============================================================
with open(cfg_file, 'r') as f:
    cfg_text = f.read()

lines = cfg_text.split('\n')
modified = 0
new_lines = []
in_entry = False
current_tensor = None

for i, line in enumerate(lines):
    stripped = line.rstrip()
    
    # 检查是否是新的 tensor entry 开始
    entry_match = re.match(r'^    ([/\w\.:_\-]+):\s*$', stripped)
    if entry_match:
        current_tensor = entry_match.group(1)
        new_lines.append(line)
        continue
    
    # 检查是否在 weight/bias tensor 条目中
    if current_tensor in weight_bias_tensors:
        # 查找 dtype 行并修改
        if re.match(r'^\s+dtype:\s+int8$', stripped):
            new_lines.append(line.replace('int8', 'float16'))
            modified += 1
            log(f"  ✅ {current_tensor}: int8 → float16")
            in_entry = False
            current_tensor = None
            continue
    
    new_lines.append(line)

log(f"\n修改了 {modified} 个 tensor 的 dtype")

if modified == 0:
    log("⚠️ 没有修改任何 tensor! 检查 tensor 名称是否匹配")
    # 打印前 10 个 weight/bias tensor
    for t in sorted(list(weight_bias_tensors))[:10]:
        log(f"  weight/bias: {t}")
    sys.exit(1)

# 保存修改后的 cfg
cfg_modified = cfg_file + '.modified'
with open(cfg_modified, 'w') as f:
    f.write('\n'.join(new_lines))

# ============================================================
# 4. step2
# ============================================================
log("运行 hybrid_quantization_step2...")
rknn = RKNN(verbose=False)
ret = rknn.hybrid_quantization_step2(model_file, data_file, cfg_modified)
if ret != 0:
    log(f"❌ step2 失败: ret={ret}")
    sys.exit(1)

rknn.export_rknn(OUTPUT)
rknn.release()
size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
log(f"✅ 模型已保存: {OUTPUT} ({size_mb:.1f}MB)")