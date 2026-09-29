#!/usr/bin/env python3
"""
RKNN 模型深度性能分析 (RKNN Toolkit 2.3.2)
- eval_perf(): 逐层耗时/内存评估 (ADB 连接板端)
- accuracy_analysis(): 量化误差分析, 定位问题层
- eval_memory(): 内存占用分析
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, "/home/ubuntu/anaconda3/envs/python3.12-tk2-2.3/lib/python3.10/site-packages")
from rknn.api import RKNN

# ============================================================
# 配置
# ============================================================
SER_INT8  = "/tmp/sensevoice_int8.rknn"     # 246MB (INT8 量化)
SER_FP16  = "/tmp/sensevoice_fp16.rknn"     # 472MB (FP16, 作参考)
OUTPUT_DIR = "/tmp/rknn_analysis"
ASR_ENCODER = ""  # 如果需要分析 ASR, 从这里拉取

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "analysis.log")

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(log_file, "a") as f:
        f.write(line + "\n")

def section(title):
    log("")
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)

def parse_perf(perf_dict):
    """解析 eval_perf 返回的 dict, 提取总耗时和每层数据"""
    total_us = 0
    layers = []
    if isinstance(perf_dict, dict):
        # RKNN Toolkit 2.3.2 返回格式
        total_us = perf_dict.get("total_time_us", 0) or perf_dict.get("total", 0)
        # 尝试不同可能的 key
        for key in ["perf_data", "layers", "layer_metrics", "ops"]:
            if key in perf_dict and perf_dict[key]:
                layers = perf_dict[key]
                break
    return total_us, layers

# ============================================================
# 0. 确认 ADB 连接
# ============================================================
section("0. ADB 连接确认")
import subprocess
ret_adb = subprocess.run(["adb", "devices"], capture_output=True, text=True)
log(f"adb devices:\n{ret_adb.stdout.strip()}")
if "device" not in ret_adb.stdout or "offline" in ret_adb.stdout:
    log("⚠️  ADB 设备状态异常, 请检查连接")
else:
    log("✅ ADB 连接正常")

# ============================================================
# 1. eval_perf() - INT8 模型 - 单核 vs 三核
# ============================================================
section("1. eval_perf() - INT8 逐层性能分析")

dummy_shape = (1, 300, 512)  # SenseVoice 典型输入: batch=1, T=300, feat=512
dummy_input = np.random.randn(*dummy_shape).astype(np.float32)

log(f"使用虚拟输入: shape={dummy_input.shape}, dtype={dummy_input.dtype}")
log(f"输入大小: {dummy_input.nbytes/1024:.1f} KB")

# ------ 1a. INT8 单核 ------
log("\n--- [1a] INT8 单核 (NPU_CORE_0) ---")
rknn = RKNN(verbose=False)
ret = rknn.load_rknn(path=SER_INT8)
assert ret == 0, f"load_rknn({SER_INT8}) failed: {ret}"
log("✅ load_rknn 成功")

ret = rknn.init_runtime(target='rk3588', perf_debug=True, eval_mem=False)
assert ret == 0, f"init_runtime failed: {ret}"

log("预热推理...")
for _ in range(5):
    rknn.inference(inputs=[dummy_input])

log("运行 eval_perf + eval_memory...")
try:
    perf_single = rknn.eval_perf(is_print=True, fix_freq=True)
    mem_single = rknn.eval_memory()
except Exception as e:
    log(f"eval_perf 在模拟器模式下不支持: {e}")
    perf_single = {"total_time_us": 0, "error": str(e)}
    mem_single = {}
rknn.release()

log("INT8 单核 eval_perf 完成")

# ------ 1b. INT8 三核 ------
log("\n--- [1b] INT8 三核 (NPU_CORE_0_1_2) ---")
rknn = RKNN(verbose=False)
rknn.load_rknn(SER_INT8)
ret = rknn.init_runtime(target='rk3588', perf_debug=True, eval_mem=True,
                         core_mask=RKNN.NPU_CORE_0_1_2)
assert ret == 0, f"init_runtime failed: {ret}"

for _ in range(5):
    rknn.inference(inputs=[dummy_input])

perf_triple = rknn.eval_perf(is_print=True, fix_freq=True)
mem_triple = rknn.eval_memory()
rknn.release()
log("INT8 三核 eval_perf 完成")

# ------ 1c. FP16 单核 (对比) ------
log("\n--- [1c] FP16 单核 (NPU_CORE_0) ---")
rknn = RKNN(verbose=False)
rknn.load_rknn(SER_FP16)
ret = rknn.init_runtime(target='rk3588', perf_debug=True, eval_mem=True)
assert ret == 0, f"init_runtime failed: {ret}"

for _ in range(3):
    rknn.inference(inputs=[dummy_input])

perf_fp16 = rknn.eval_perf(is_print=True, fix_freq=True)
mem_fp16 = rknn.eval_memory()
rknn.release()
log("FP16 单核 eval_perf 完成")

# 保存所有 raw 结果
for name, perf, mem in [("int8_single", perf_single, mem_single),
                          ("int8_triple", perf_triple, mem_triple),
                          ("fp16_single", perf_fp16, mem_fp16)]:
    with open(os.path.join(OUTPUT_DIR, f"perf_{name}.json"), "w") as f:
        json.dump({"perf": perf, "memory": mem}, f, indent=2, default=str)

# ------ 性能对比汇总 ------
log("\n" + "-" * 40)
log("性能对比汇总:")
log("-" * 40)

t_s, _ = parse_perf(perf_single)
t_t, _ = parse_perf(perf_triple)
t_f, _ = parse_perf(perf_fp16)

log(f"INT8 单核: {t_s/1000:.2f} ms")
log(f"INT8 三核: {t_t/1000:.2f} ms")
log(f"FP16 单核: {t_f/1000:.2f} ms")

if t_s and t_t:
    log(f"三核加速比: {t_s/t_t:.2f}x")
if t_s and t_f:
    log(f"INT8 vs FP16 加速比: {t_f/t_s:.2f}x")

# 内存对比
for label, mem in [("INT8 单核", mem_single), ("INT8 三核", mem_triple), ("FP16 单核", mem_fp16)]:
    if isinstance(mem, dict):
        log(f"{label} 内存: total={mem.get('total_weight_mem','?')}, inputs={mem.get('total_inputs_mem','?')}, outputs={mem.get('total_outputs_mem','?')}")

# ============================================================
# 2. accuracy_analysis() - 量化误差分析
# ============================================================
section("2. accuracy_analysis() - 量化误差分析")

npy_path = os.path.join(OUTPUT_DIR, "test_input.npy")
np.save(npy_path, dummy_input)
log(f"测试输入保存: {npy_path}")

# 对 INT8 模型做精度分析 (对比 FP32 仿真结果)
log("\n运行 accuracy_analysis (INT8 vs FP32 target=rk3588)...")
snapshot_dir = os.path.join(OUTPUT_DIR, "snapshot_int8")
os.makedirs(snapshot_dir, exist_ok=True)

rknn = RKNN(verbose=False)
rknn.load_rknn(SER_INT8)
ret = rknn.accuracy_analysis(
    inputs=[npy_path],
    output_dir=snapshot_dir,
    target='rk3588'
)
log(f"accuracy_analysis 返回值: {ret}")
rknn.release()

# 解析结果
cosine_file = os.path.join(snapshot_dir, "cosine_similarity.json")
alt_cosine = os.path.join(snapshot_dir, "output", "cosine_similarity.json")

for cf in [cosine_file, alt_cosine]:
    if os.path.exists(cf):
        with open(cf) as f:
            cos_data = json.load(f)
        log(f"\n解析 {cf} ...")
        log(f"共 {len(cos_data)} 层数据")

        # 找问题层 (余弦相似度 < 0.99)
        problem_layers = []
        all_sims = []
        for layer in cos_data:
            name = layer.get("name", layer.get("layer_name", "unknown"))
            cos_sim = layer.get("cosine_similarity", layer.get("similarity", 1.0))
            if isinstance(cos_sim, str):
                try:
                    cos_sim = float(cos_sim)
                except:
                    cos_sim = 1.0
            all_sims.append(cos_sim)
            if cos_sim < 0.99:
                problem_layers.append((name, cos_sim))

        if problem_layers:
            log(f"\n⚠️  发现 {len(problem_layers)} 个问题层 (cos_sim < 0.99):")
            problem_layers.sort(key=lambda x: x[1])
            for name, cos_sim in problem_layers[:30]:
                log(f"  {name}: cos_sim={cos_sim:.6f}")
        else:
            log(f"\n✅ 所有层余弦相似度 >= 0.99!")

        # 统计
        if all_sims:
            log(f"\n相似度统计:")
            log(f"  最小: {min(all_sims):.6f}")
            log(f"  平均: {sum(all_sims)/len(all_sims):.6f}")
            below_99 = sum(1 for s in all_sims if s < 0.99)
            log(f"  < 0.99: {below_99}/{len(all_sims)} ({below_99/len(all_sims)*100:.1f}%)")
            below_95 = sum(1 for s in all_sims if s < 0.95)
            log(f"  < 0.95: {below_95}/{len(all_sims)} ({below_95/len(all_sims)*100:.1f}%)")
        break
else:
    log(f"⚠️  未找到 cosine_similarity.json")
    log(f"  搜索 {snapshot_dir}:")
    for root, dirs, files in os.walk(snapshot_dir):
        for f in files[:20]:
            log(f"    {os.path.join(root, f)}")

# ============================================================
# 3. hybrid_quantization 方案建议
# ============================================================
section("3. 混合量化 (Hybrid Quantization) 方案")

if os.path.exists(cosine_file) or os.path.exists(alt_cosine):
    cf = cosine_file if os.path.exists(cosine_file) else alt_cosine
    with open(cf) as f:
        cos_data = json.load(f)

    problem_layers = []
    for layer in cos_data:
        name = layer.get("name", layer.get("layer_name", "unknown"))
        cos_sim = layer.get("cosine_similarity", layer.get("similarity", 1.0))
        if isinstance(cos_sim, str):
            try:
                cos_sim = float(cos_sim)
            except:
                cos_sim = 1.0
        if cos_sim < 0.99:
            problem_layers.append(name)

    if problem_layers:
        log(f"\n建议对这 {len(problem_layers)} 个层执行混合量化:")
        log(f"hybrid_quantization_step1 + step2 工作流:")
        log(f"  1. 准备校准数据集 (100-500 张代表性输入)")
        log(f"  2. 指定 'custom_quantize_layers' 为以下层列表:")
        for name in problem_layers[:20]:
            log(f"     - {name}")
        if len(problem_layers) > 20:
            log(f"     ... 共 {len(problem_layers)} 层")
        log(f"  3. 调用 hybrid_quantization_step1() 进行混合量化")
        log(f"  4. 调用 hybrid_quantization_step2() 导出混合量化模型")
        log(f"  预期效果: 问题层保留 FP16 精度, 其余层保持 INT8, 速度损失 < 10%")
    else:
        log("无需混合量化 - 量化误差已在可接受范围内")
else:
    log("跳过混合量化方案 (需先完成 accuracy_analysis)")

# ============================================================
# 4. 摘要
# ============================================================
section("4. 分析摘要")
log(f"模型: SER SenseVoice INT8 ({SER_INT8})")
log(f"输入: {dummy_shape}")
log(f"")
log(f"性能:")
log(f"  INT8 单核: {t_s/1000:.2f} ms" if t_s else "  INT8 单核: N/A")
log(f"  INT8 三核: {t_t/1000:.2f} ms" if t_t else "  INT8 三核: N/A")
log(f"  FP16 单核: {t_f/1000:.2f} ms" if t_f else "  FP16 单核: N/A")

acc_file = cosine_file if os.path.exists(cosine_file) else (alt_cosine if os.path.exists(alt_cosine) else None)
if acc_file:
    log(f"精度分析: {acc_file}")
log(f"所有结果: {OUTPUT_DIR}")
log(f"日志: {log_file}")
log("")