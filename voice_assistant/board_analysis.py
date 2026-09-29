#!/usr/bin/env python3
"""板端内存友好型性能分析脚本 v3"""
import os, json, time
import numpy as np

WORK_DIR = "/data/rknn_analysis"
os.makedirs(WORK_DIR, exist_ok=True)
LOG_FILE = os.path.join(WORK_DIR, "board_analysis.log")

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def section(title):
    log(""); log("="*60); log(f"  {title}"); log("="*60)

from rknnlite.api import RKNNLite

results = {}

# ============================================================
# 1. SER INT8 性能 (先测 INT8 再测 FP16, 避免内存叠加)
# ============================================================
section("1. SER INT8 性能分析")
SER_INT8 = "/userdata/sensevoice/sensevoice_encoder_ctc_int8.rknn"
SER_SHAPE = (1, 700, 400)
dummy = np.random.randn(*SER_SHAPE).astype(np.float32)

rknn = RKNNLite(verbose=False)
rknn.load_rknn(SER_INT8)
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

for _ in range(3):
    rknn.inference(inputs=[dummy])
times = []
for _ in range(30):
    t0 = time.perf_counter()
    rknn.inference(inputs=[dummy])
    t1 = time.perf_counter()
    times.append((t1-t0)*1000)
times.sort()
n = len(times)
int8_3c = {"avg_ms": round(np.mean(times),3), "med_ms": round(np.median(times),3),
           "p90_ms": round(times[int(n*0.9)],3), "min_ms": round(min(times),3)}
log(f"INT8 三核: avg={int8_3c['avg_ms']:.3f} med={int8_3c['med_ms']:.3f} p90={int8_3c['p90_ms']:.3f}")
results["SER_INT8_3core"] = int8_3c

# 保存 INT8 输出用于精度对比 (仅取前 100 个元素)
int8_sample = rknn.inference(inputs=[dummy])[0].flatten()[:100].copy()
rknn.release()

# ============================================================
# 2. SER FP16 性能
# ============================================================
section("2. SER FP16 性能分析")
SER_FP16 = "/userdata/sensevoice/sensevoice_encoder_ctc_fp16.rknn"

rknn = RKNNLite(verbose=False)
rknn.load_rknn(SER_FP16)
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

for _ in range(3):
    rknn.inference(inputs=[dummy])
times = []
for _ in range(30):
    t0 = time.perf_counter()
    rknn.inference(inputs=[dummy])
    t1 = time.perf_counter()
    times.append((t1-t0)*1000)
times.sort()
n = len(times)
fp16_3c = {"avg_ms": round(np.mean(times),3), "med_ms": round(np.median(times),3),
           "p90_ms": round(times[int(n*0.9)],3), "min_ms": round(min(times),3)}
log(f"FP16 三核: avg={fp16_3c['avg_ms']:.3f} med={fp16_3c['med_ms']:.3f} p90={fp16_3c['p90_ms']:.3f}")
results["SER_FP16_3core"] = fp16_3c

fp16_sample = rknn.inference(inputs=[dummy])[0].flatten()[:100].copy()
rknn.release()

log(f"\n加速比 INT8/FP16: {fp16_3c['avg_ms']/int8_3c['avg_ms']:.2f}x")

# ============================================================
# 3. 量化误差分析 (用采样片段)
# ============================================================
section("3. 量化误差分析 (采样 100 个元素)")
af = int8_sample.astype(np.float64)
bf = fp16_sample.astype(np.float64)
dot = np.dot(af, bf)
norm = np.linalg.norm(af) * np.linalg.norm(bf)
cos = float(dot/norm) if norm > 0 else 1.0
ad = np.abs(af-bf)
log(f"余弦相似度: {cos:.6f}")
log(f"最大差异: {ad.max():.6f}")
log(f"平均差异: {ad.mean():.6f}")
log(f"INT8 范围: [{int8_sample.min():.4f}, {int8_sample.max():.4f}]")
log(f"FP16 范围: [{fp16_sample.min():.4f}, {fp16_sample.max():.4f}]")
results["quant_error"] = {"cos_similarity": round(cos,6), "max_diff": round(ad.max(),6)}

# ============================================================
# 4. ASR Decoder
# ============================================================
section("4. ASR Decoder")
dec_path = "/userdata/zipformer/model/decoder-epoch-99-avg-1.rknn"
if os.path.exists(dec_path):
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn(dec_path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    inp = np.random.randn(1,256).astype(np.float32)
    for _ in range(5):
        rknn.inference(inputs=[inp])
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        rknn.inference(inputs=[inp])
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    times.sort()
    n = len(times)
    s = {"avg_ms": round(np.mean(times),3), "p90_ms": round(times[int(n*0.9)],3)}
    log(f"Decoder: avg={s['avg_ms']:.3f} p90={s['p90_ms']:.3f}")
    results["ASR_Decoder"] = s
    rknn.release()

# ============================================================
# 5. ASR Joiner
# ============================================================
section("5. ASR Joiner")
join_path = "/userdata/zipformer/model/joiner-epoch-99-avg-1.rknn"
if os.path.exists(join_path):
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn(join_path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    inp = [np.random.randn(1,512).astype(np.float32),
           np.random.randn(1,512).astype(np.float32)]
    for _ in range(5):
        rknn.inference(inputs=inp)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        rknn.inference(inputs=inp)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    times.sort()
    n = len(times)
    s = {"avg_ms": round(np.mean(times),3), "p90_ms": round(times[int(n*0.9)],3)}
    log(f"Joiner: avg={s['avg_ms']:.3f} p90={s['p90_ms']:.3f}")
    results["ASR_Joiner"] = s
    rknn.release()

# ============================================================
# 6. 混合量化建议
# ============================================================
section("6. 混合量化建议")
if cos < 0.99:
    log(f"余弦相似度 {cos:.4f} < 0.99, 建议混合量化")
    log("流程: 1) 收集校准集 → 2) hybrid_quantization_step1 分析 →")
    log("      3) 敏感层保留 FP16 → 4) 重新量化")
else:
    log(f"余弦相似度 {cos:.4f} >= 0.99, 量化精度良好")

# 保存
with open(os.path.join(WORK_DIR,"results.json"),"w") as f:
    json.dump(results, f, indent=2)
log(f"\n结果保存: {os.path.join(WORK_DIR,'results.json')}")
log("完成!")