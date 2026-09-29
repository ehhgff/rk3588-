#!/usr/bin/env python3
"""
尝试手动创建混合量化配置 and 使用 custom_hybrid
用 /encoder/tp_norm/Add_1_output_0 (tp_norm 最终输出)
"""
import os, sys, json, time
import numpy as np
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
HYBRID_RKNN = "models/sensevoice_encoder_ctc_100f_hybrid_custom.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 60)
log("SER 100f: custom_hybrid v2")
log("=" * 60)

rknn = RKNN(verbose=False)
rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
rknn.load_onnx(model=ONNX_PATH)

# 只保留 ctc_lo + LogSoftmax 为 FP16
# tp_norm add_1_output → ctc_lo MatMul → ctc_lo Add → LogSoftmax → ctc_logits
custom_hybrid = [
    ["/encoder/tp_norm/Add_1_output_0", "ctc_logits"],
]

log(f"custom_hybrid: {json.dumps(custom_hybrid)}")

ret = rknn.hybrid_quantization_step1(
    dataset=DATASET,
    proposal=False,
    custom_hybrid=custom_hybrid,
)

if ret != 0:
    log(f"❌ 仍然失败! ret={ret}")
    log("尝试只保留 LogSoftmax 层为 FP16...")
    rknn.release()
    
    rknn2 = RKNN(verbose=False)
    rknn2.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
    rknn2.load_onnx(model=ONNX_PATH)
    
    custom_hybrid2 = [
        ["/ctc_lo/Add_output_0", "ctc_logits"],
    ]
    log(f"custom_hybrid: {json.dumps(custom_hybrid2)}")
    ret2 = rknn2.hybrid_quantization_step1(
        dataset=DATASET,
        proposal=False,
        custom_hybrid=custom_hybrid2,
    )
    if ret2 != 0:
        log(f"❌ 仍然失败! ret={ret2}")
        rknn2.release()
        sys.exit(1)
    
    # 找到了 cfg
    model_file = [f for f in os.listdir('.') if f.endswith('.model')][0]
    data_file = [f for f in os.listdir('.') if f.endswith('.data')][0]
    cfg_file = [f for f in os.listdir('.') if f.endswith('.quantization.cfg')][0]
    
    log(f"cfg: {cfg_file}")
    with open(cfg_file) as f:
        log(f"内容: {f.read()[:2000]}")
    
    rknn2.hybrid_quantization_step2(
        model_input=model_file,
        data_input=data_file,
        model_quantization_cfg=cfg_file,
    )
    rknn2.export_rknn(HYBRID_RKNN)
    if os.path.exists(HYBRID_RKNN):
        sz = os.path.getsize(HYBRID_RKNN) / (1024*1024)
        log(f"✅ 模型: {HYBRID_RKNN} ({sz:.1f}MB)")
    rknn2.release()
    sys.exit(0)

# 成功了
model_file = [f for f in os.listdir('.') if f.endswith('.model')][0]
data_file = [f for f in os.listdir('.') if f.endswith('.data')][0]
cfg_file = [f for f in os.listdir('.') if f.endswith('.quantization.cfg')][0]

log(f"✅ cfg_file: {cfg_file}")
with open(cfg_file) as f:
    log(f"内容: {f.read()[:2000]}")

log("\nStep2: 导出混合模型...")
rknn.hybrid_quantization_step2(
    model_input=model_file,
    data_input=data_file,
    model_quantization_cfg=cfg_file,
)
rknn.export_rknn(HYBRID_RKNN)
if os.path.exists(HYBRID_RKNN):
    sz = os.path.getsize(HYBRID_RKNN) / (1024*1024)
    log(f"✅ 模型: {HYBRID_RKNN} ({sz:.1f}MB)")

rknn.release()