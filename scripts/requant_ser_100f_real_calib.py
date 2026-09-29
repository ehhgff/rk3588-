#!/usr/bin/env python3
"""
SER 100帧: 用真实音频特征作为校准集, 重新 INT8 量化
先用这个试试 -> 如果精度够就不需要混合量化了
"""
import os, sys, time, json
import numpy as np
from rknn.api import RKNN

ONNX_PATH = "models/sensevoice_encoder_ctc_100f.onnx"
DATASET = "calib_data/data_real/dataset.txt"
INT8_RKNN = "models/sensevoice_encoder_ctc_100f_int8_real.rknn"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 60)
log("SER 100f: 真实校准集 INT8 量化")
log("=" * 60)
log(f"ONNX: {ONNX_PATH}")
log(f"校准集: {DATASET}")
with open(DATASET) as f:
    n_samples = len(f.readlines())
log(f"样本数: {n_samples}")

t_start = time.time()

rknn = RKNN(verbose=True)
ret = rknn.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
assert ret == 0, "config failed"

ret = rknn.load_onnx(model=ONNX_PATH)
assert ret == 0, "load_onnx failed"

log("\n开始量化 (使用真实校准集)...")
ret = rknn.build(do_quantization=True, dataset=DATASET)
if ret != 0:
    log(f"❌ 量化失败! ret={ret}")
    sys.exit(1)

log("\n导出 RKNN 模型...")
ret = rknn.export_rknn(INT8_RKNN)
if ret == 0:
    size_mb = os.path.getsize(INT8_RKNN) / (1024*1024)
    log(f"✅ 模型导出成功!")
    log(f"   路径: {INT8_RKNN}")
    log(f"   大小: {size_mb:.1f} MB")
else:
    log(f"❌ 导出失败! ret={ret}")
    sys.exit(1)

t_elapsed = time.time() - t_start
log(f"总耗时: {t_elapsed:.0f}秒 ({t_elapsed/60:.1f}分)")

# 精度对比
log("\n" + "=" * 60)
log("精度对比 (INT8 真实校准 vs FP16 参考)")
log("=" * 60)

# 加载 FP16 参考模型
FP16_RKNN = "models/sensevoice_encoder_ctc_100f_fp16.rknn"
if not os.path.exists(FP16_RKNN):
    log(f"⚠️ FP16 参考模型不存在: {FP16_RKNN}")
    log("  跳过精度对比, 板端测试时再测")
    rknn.release()
    sys.exit(0)

rknn_fp16 = RKNN(verbose=False)
rknn_fp16.load_rknn(FP16_RKNN)
rknn_fp16.init_runtime()

INPUT_SHAPE = (1, 100, 560)
DTYPE = np.float32

cos_sims = []
max_diffs = []
for t in range(20):
    inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
    out_i8 = rknn.inference(inputs=[inp])[0].flatten()
    out_fp16 = rknn_fp16.inference(inputs=[inp])[0].flatten()
    
    a = out_i8.astype(np.float64)
    b = out_fp16.astype(np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(dot / norm) if norm > 1e-10 else 1.0
    max_diff = float(np.max(np.abs(a - b)))
    
    cos_sims.append(cos)
    max_diffs.append(max_diff)
    log(f"  #{t+1:2d}: cos={cos:.6f}  max_diff={max_diff:.4f}")

rknn_fp16.release()
rknn.release()

cos_arr = np.array(cos_sims)
log(f"\n平均余弦相似度: {cos_arr.mean():.6f}")
log(f"最小余弦相似度:  {cos_arr.min():.6f}")
log(f"平均最大差异:    {np.mean(max_diffs):.4f}")

pass_threshold = cos_arr.mean() >= 0.99
log(f"\n{'✅ 精度达标!' if pass_threshold else '❌ 精度不足:'} cos={cos_arr.mean():.6f} {'>= 0.99' if pass_threshold else '< 0.99'}")

if not pass_threshold:
    log("\n💡 建议: 实施混合量化")
    log("   手动标记敏感层为 FP16, 其余层保持 INT8")
    log("   目标: 混合量化模型 ~260MB, 延迟 ~170ms, 精度 cos>0.99")

print(f"\n总耗时: {t_elapsed:.0f}秒")