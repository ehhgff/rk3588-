#!/usr/bin/env python3
"""
板端测试: PTQ INT8 best 精度 + 延迟
"""
import os, sys, time
import numpy as np
from rknnlite.api import RKNNLite

MODEL = "/data/voice_assistant/ser_int8_best.rknn"
FP16 = "/data/sensevoice/models/sensevoice_encoder_ctc_100f_fp16.rknn"
INPUT_SHAPE = (1, 100, 560)
DTYPE = np.float32
N = 50

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load(path, name):
    log(f"加载 {name}: {os.path.basename(path)}")
    rknn = RKNNLite()
    rknn.load_rknn(path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    s = os.path.getsize(path)/(1024*1024)
    log(f"  ✅ ({s:.0f} MB)")
    return rknn

log("="*60)
log("PTQ INT8 Best: 精度 + 延迟测试")
log("="*60)

rknn = load(MODEL, "PTQ INT8")
rknn_fp16 = load(FP16, "FP16 ref")

cos_sims = []
log(f"\n精度测试 {N} 次...")
for t in range(N):
    inp = np.random.randn(*INPUT_SHAPE).astype(DTYPE)
    out = rknn.inference(inputs=[inp])[0].flatten()
    ref = rknn_fp16.inference(inputs=[inp])[0].flatten()
    a = out.astype(np.float64)
    b = ref.astype(np.float64)
    cos = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>1e-10 else 1.0
    cos_sims.append(cos)
    if (t+1)%10==0:
        log(f"  #{t+1:3d}: cos={np.mean(cos_sims[-10:]):.6f}")
rknn_fp16.release()
c = np.array(cos_sims)
log(f"\n精度结果:")
log(f"  平均余弦相似度: {c.mean():.6f}")
log(f"  最小余弦相似度:  {c.min():.6f}")
log(f"  >= 0.99:        {(c>=0.99).mean()*100:.1f}%")

log(f"\n延迟测试 {N}x warmup...")
for _ in range(10):
    rknn.inference(inputs=[np.random.randn(*INPUT_SHAPE).astype(DTYPE)])
lats = []
log(f"延迟测试 {N} 次...")
for t in range(N):
    t0 = time.perf_counter()
    rknn.inference(inputs=[np.random.randn(*INPUT_SHAPE).astype(DTYPE)])
    lats.append((time.perf_counter()-t0)*1000)
rknn.release()
la = np.array(lats)
log(f"\n延迟结果:")
log(f"  平均: {la.mean():.1f} ms")
log(f"  最小:  {la.min():.1f} ms")
log(f"  最大:  {la.max():.1f} ms")
log(f"  P95:   {np.percentile(la,95):.1f} ms")

# Save result
with open("/data/voice_assistant/ptq_best_result.txt", 'w') as f:
    f.write(f"PTQ INT8 Best Results\n")
    f.write(f"cos_sim_mean={c.mean():.6f}\n")
    f.write(f"cos_sim_min={c.min():.6f}\n")
    f.write(f"latency_mean={la.mean():.1f}\n")
    f.write(f"latency_p95={np.percentile(la,95):.1f}\n")
log(f"\n结果已保存")