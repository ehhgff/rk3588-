#!/usr/bin/env python3
"""TTS ONNX 板端分析脚本 - 处理 token 输入范围"""
import os, json, time
import numpy as np
import onnxruntime as ort

WORK_DIR = "/data/rknn_analysis"
LOG_FILE = os.path.join(WORK_DIR, "tts_analysis.log")

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

MODELS = {
    "matcha_acoustic": "/userdata/matcha-tts/matcha_acoustic.onnx",
    "matcha_vocoder": "/userdata/matcha-tts/matcha_vocoder.onnx",
}
results = {}

for name, path in MODELS.items():
    if not os.path.exists(path):
        log(f"⚠️ {path} 不存在, 跳过")
        continue
    log(f"\n=== {name} ===")
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    in_info = sess.get_inputs()
    out_info = sess.get_outputs()
    log(f"输入: {[(i.name, i.shape) for i in in_info]}")
    log(f"输出: {[(o.name, o.shape) for o in out_info]}")

    # 生成有效输入
    feeds = {}
    for i in in_info:
        shape = tuple(s if isinstance(s, int) and s > 0 else 1 for s in i.shape)
        if i.type == 'tensor(int64)':
            # Token 输入: 范围 0-177 (vocab size 178)
            feeds[i.name] = np.random.randint(0, 177, size=shape, dtype=np.int64)
            log(f"  {i.name}: int64 tensor, shape={shape}, values=[0,177]")
        elif i.type == 'tensor(float)':
            feeds[i.name] = np.random.randn(*shape).astype(np.float32)
            log(f"  {i.name}: float32 tensor, shape={shape}")
        else:
            feeds[i.name] = np.random.randn(*shape).astype(np.float32)
            log(f"  {i.name}: {i.type}, shape={shape}")

    # 验证推理
    try:
        out = sess.run(None, feeds)
        log(f"推理成功: {[o.shape for o in out]}")
    except Exception as e:
        log(f"推理失败: {e}")
        continue

    # 预热 + 测时
    for _ in range(5):
        sess.run(None, feeds)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    times.sort()
    n = len(times)
    stats = {"avg_ms": round(np.mean(times),3), "med_ms": round(np.median(times),3),
             "p90_ms": round(times[int(n*0.9)],3), "p99_ms": round(times[int(n*0.99)],3),
             "min_ms": round(min(times),3)}
    log(f"  avg={stats['avg_ms']:.3f} med={stats['med_ms']:.3f} "
        f"p90={stats['p90_ms']:.3f} p99={stats['p99_ms']:.3f}")
    results[name] = stats

    # 尝试不同 batch/seq_len
    for scale_name, scale_shape in [("seq8", (8,)), ("seq16", (16,)), ("seq32", (32,))]:
        feeds2 = {}
        try:
            for i in in_info:
                shape = list(feeds[i.name].shape)
                if len(shape) == 1:
                    shape[0] = int(scale_shape[0])
                feeds2[i.name] = feeds[i.name][:shape[0]].copy() if feeds[i.name].shape == shape else np.random.randint(0, 177, size=shape, dtype=np.int64) if i.type == 'tensor(int64)' else np.random.randn(*shape).astype(np.float32)
            # Regenerate with correct shape
            for i in in_info:
                shape = tuple(s if isinstance(s, int) and s > 0 else 1 for s in i.shape)
                if len(shape) == 1:
                    shape = (int(scale_shape[0]),)
                if i.type == 'tensor(int64)':
                    feeds2[i.name] = np.random.randint(0, 177, size=shape, dtype=np.int64)
                else:
                    feeds2[i.name] = np.random.randn(*shape).astype(np.float32)
            for _ in range(3):
                sess.run(None, feeds2)
            t2 = time.perf_counter()
            for _ in range(20):
                sess.run(None, feeds2)
            t3 = time.perf_counter()
            avg2 = (t3-t2)/20*1000
            log(f"  {scale_name}: avg={avg2:.3f}ms")
            results[f"{name}_{scale_name}"] = {"avg_ms": round(avg2,3)}
        except Exception as e:
            log(f"  {scale_name}: 失败 {str(e)[:60]}")

with open(os.path.join(WORK_DIR,"tts_results.json"),"w") as f:
    json.dump(results, f, indent=2)
log(f"\n结果保存: {os.path.join(WORK_DIR,'tts_results.json')}")