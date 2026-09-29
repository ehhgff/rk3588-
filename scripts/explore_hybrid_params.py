#!/usr/bin/env python3
"""
SER 100帧: 探索混合量化参数
测试 quantized_hybrid_level / auto_hybrid_cos_thresh 等参数
"""
import os, sys, time, json
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
RESULTS = {}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def try_build(label, config_kwargs):
    log(f"\n{'='*50}")
    log(f"尝试: {label}")
    log(f"配置: {json.dumps(config_kwargs)}")
    log('='*50)
    
    out_path = f"models/sensevoice_encoder_ctc_100f_{label}.rknn"
    if os.path.exists(out_path):
        sz = os.path.getsize(out_path) / (1024*1024)
        log(f"⚠️ 已存在, 跳过 ({sz:.0f}MB)")
        RESULTS[label] = {"size_mb": round(sz, 1), "skipped": True}
        return
    
    try:
        rknn = RKNN(verbose=False)
        rknn.config(target_platform="rk3588", **config_kwargs)
        rknn.load_onnx(model=ONNX_PATH)
        
        t0 = time.time()
        if config_kwargs.get('auto_hybrid', False):
            ret = rknn.build(do_quantization=True, dataset=DATASET, auto_hybrid=True)
        else:
            ret = rknn.build(do_quantization=True, dataset=DATASET)
        t1 = time.time()
        
        if ret != 0:
            log(f"❌ 构建失败! ret={ret}")
            rknn.release()
            return
        
        ret = rknn.export_rknn(out_path)
        if ret == 0:
            sz = os.path.getsize(out_path) / (1024*1024)
            log(f"✅ 成功! ({sz:.0f}MB, {t1-t0:.0f}s)")
            RESULTS[label] = {"size_mb": round(sz, 1), "time_s": round(t1-t0, 1)}
        else:
            log(f"❌ 导出失败! ret={ret}")
        
        rknn.release()
    except Exception as e:
        log(f"❌ 异常: {e}")

# ====== 实验 ======

# 1. baseline INT8
try_build("int8_baseline", {
    "quantized_dtype": "w8a8",
    "optimization_level": 3,
})

# 2. 不同 optimization_level
for ol in [0, 1, 2]:
    try_build(f"int8_ol{ol}", {
        "quantized_dtype": "w8a8",
        "optimization_level": ol,
    })

# 3. quantized_hybrid_level 不同值
for level in [1, 2, 3, 4, 5]:
    try_build(f"hybrid_l{level}", {
        "quantized_dtype": "w8a8",
        "optimization_level": 3,
        "quantized_hybrid_level": level,
    })

# 4. auto_hybrid 不同阈值
for thresh in [0.9, 0.95, 0.98, 0.99, 0.995]:
    try_build(f"auto_cos{thresh}", {
        "quantized_dtype": "w8a8",
        "optimization_level": 3,
        "auto_hybrid_cos_thresh": thresh,
    })

# 5. quantized_algorithm
for algo in ['normal', 'mmse', 'kl']:
    try_build(f"alg_{algo}", {
        "quantized_dtype": "w8a8",
        "optimization_level": 3,
        "quantized_algorithm": algo,
    })

# 6. quantized_method
for method in ['channel', 'layer']:
    try_build(f"method_{method}", {
        "quantized_dtype": "w8a8",
        "optimization_level": 3,
        "quantized_method": method,
    })

# 汇总
log("\n\n" + "=" * 60)
log("实验结果汇总")
log("=" * 60)
for label, result in sorted(RESULTS.items()):
    sz = result.get("size_mb", "?")
    skipped = result.get("skipped", False)
    flag = " (已存在)" if skipped else ""
    log(f"  {label:20s}: {sz:>5} MB{flag}")