#!/usr/bin/env python3
"""LLM (Qwen3-0.6B) RKLLM 模型性能深度分析
在板端运行，直接调用 RKLLM API，分析:
  - 模型加载时间 (rkllm_init)
  - 系统提示词预填充时间 (prefill)
  - 首 token 延迟 (TTFT)
  - 生成速度 (tokens/s)
  - 不同 prompt 长度的延迟
  - 不同 max_tokens 的扩展性
  - CPU 绑核对生成速度的影响
"""
import os, sys, time, json, ctypes, queue, threading
import numpy as np

# 确保能找到 rkllm 库
sys.path.insert(0, "/userdata/voice_assistant/core")
sys.path.insert(0, "/userdata/voice_assistant")

MODEL_PATH = "/data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm"
SYS_PROMPT = "你是一个医疗助手，请用一句话简短回答患者问题。"
WARMUP = 1
ROUNDS = 3

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_rkllm(model_path, max_context_len=2048, max_new_tokens=80,
               cpu_mask=0x0F, n_batch=1):
    """加载 RKLLM 模型并返回句柄"""
    from rkllm_python_api_streaming import (
        rkllm_lib, RKLLMParam, RKLLMInput,
        RKLLMInferParam, RKLLM_Handle_t
    )
    import queue as _queue

    t0 = time.perf_counter()

    # 回调
    token_queue = _queue.Queue()
    _finished = [False]
    _error = [None]

    LLMResultCallback = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    )

    def callback_impl(result_ptr, userdata, state):
        try:
            if state == 0:
                class R(ctypes.Structure):
                    _fields_ = [("text", ctypes.c_char_p)]
                r = ctypes.cast(result_ptr, ctypes.POINTER(R))
                text = r.contents.text.decode('utf-8')
                token_queue.put(text)
            elif state == 2:
                _finished[0] = True
                token_queue.put(None)
            elif state == 3:
                _error[0] = "RKLLM 运行错误"
                _finished[0] = True
                token_queue.put(None)
        except Exception as e:
            pass
        return 0

    callback = LLMResultCallback(callback_impl)

    # 创建参数
    rkllm_lib.rkllm_createDefaultParam.restype = RKLLMParam
    param = rkllm_lib.rkllm_createDefaultParam()
    param.model_path = bytes(model_path, 'utf-8')
    param.max_context_len = max_context_len
    param.max_new_tokens = max_new_tokens
    param.skip_special_token = True
    param.is_async = True
    param.extend_param.base_domain_id = 0
    param.extend_param.embed_flash = 1
    param.extend_param.enabled_cpus_num = bin(cpu_mask).count('1')
    param.extend_param.enabled_cpus_mask = cpu_mask
    param.extend_param.n_batch = n_batch
    param.extend_param.use_cross_attn = 0

    handle = RKLLM_Handle_t()
    rkllm_lib.rkllm_init.argtypes = [
        ctypes.POINTER(RKLLM_Handle_t),
        ctypes.POINTER(RKLLMParam),
        type(callback),
    ]
    rkllm_lib.rkllm_init.restype = ctypes.c_int

    ret = rkllm_lib.rkllm_init(ctypes.byref(handle), ctypes.byref(param), callback)
    load_ms = (time.perf_counter() - t0) * 1000

    if ret != 0:
        raise RuntimeError(f"rkllm_init 失败: {ret}")

    rkllm_lib.rkllm_run_async.argtypes = [
        RKLLM_Handle_t, ctypes.POINTER(RKLLMInput),
        ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p,
    ]
    rkllm_lib.rkllm_run_async.restype = ctypes.c_int
    rkllm_lib.rkllm_abort.argtypes = [RKLLM_Handle_t]
    rkllm_lib.rkllm_abort.restype = ctypes.c_int
    rkllm_lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
    rkllm_lib.rkllm_destroy.restype = ctypes.c_int

    # 推理参数
    infer_param = RKLLMInferParam()
    ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
    infer_param.mode = 0
    infer_param.lora_params = None
    infer_param.prompt_cache_params = None
    infer_param.keep_history = 1

    return {
        "handle": handle,
        "infer_param": infer_param,
        "callback": callback,
        "token_queue": token_queue,
        "finished": _finished,
        "error": _error,
        "rkllm_lib": rkllm_lib,
        "load_ms": round(load_ms, 0),
    }

def do_prefill(ctx, system_prompt):
    """预填充系统提示词"""
    from rkllm_python_api_streaming import RKLLMInput
    q = ctx["token_queue"]
    f = ctx["finished"]
    f[0] = False

    # 清空队列
    while not q.empty():
        try: q.get_nowait()
        except: break

    input_data = RKLLMInput()
    input_data.role = b"system"
    input_data.enable_thinking = False
    input_data.input_type = 0

    text_bytes = system_prompt.encode('utf-8')
    input_data.prompt_input = ctypes.c_char_p(text_bytes)
    input_data.prompt_length = len(text_bytes)

    t0 = time.perf_counter()
    ret = ctx["rkllm_lib"].rkllm_run_async(
        ctx["handle"], ctypes.byref(input_data),
        ctypes.byref(ctx["infer_param"]), None
    )
    if ret != 0:
        raise RuntimeError(f"prefill rkllm_run_async 失败: {ret}")

    # 立即 abort (只做 prefill)
    ctx["rkllm_lib"].rkllm_abort(ctx["handle"])
    f[0] = True
    ms = (time.perf_counter() - t0) * 1000
    return ms

def do_generate(ctx, prompt, max_tokens=None):
    """生成文本，返回 (ttft_ms, total_ms, tokens, text)"""
    from rkllm_python_api_streaming import RKLLMInput
    q = ctx["token_queue"]
    f = ctx["finished"]
    f[0] = False

    # 清空队列
    while not q.empty():
        try: q.get_nowait()
        except: break

    input_data = RKLLMInput()
    input_data.role = b"user"
    input_data.enable_thinking = False
    input_data.input_type = 0

    text_bytes = prompt.encode('utf-8')
    input_data.prompt_input = ctypes.c_char_p(text_bytes)
    input_data.prompt_length = len(text_bytes)

    if max_tokens:
        pass  # 已在 param 中设置

    t0 = time.perf_counter()
    ret = ctx["rkllm_lib"].rkllm_run_async(
        ctx["handle"], ctypes.byref(input_data),
        ctypes.byref(ctx["infer_param"]), None
    )
    if ret != 0:
        raise RuntimeError(f"generate rkllm_run_async 失败: {ret}")

    # 收集 tokens
    tokens = []
    first_token_time = None
    while True:
        token = q.get()
        if token is None:
            break
        if first_token_time is None:
            first_token_time = time.perf_counter()
        tokens.append(token)

    total_ms = (time.perf_counter() - t0) * 1000
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else total_ms
    text = "".join(tokens)

    return {
        "ttft_ms": round(ttft_ms, 0),
        "total_ms": round(total_ms, 0),
        "num_tokens": len(tokens),
        "tokens_per_sec": round(len(tokens) / (total_ms / 1000), 1) if total_ms > 0 else 0,
        "text_len_chars": len(text),
        "text": text[:50] + "..." if len(text) > 50 else text,
    }

def test_model_loading():
    """模型加载时间"""
    log("\n2.1 模型加载时间")
    results = []
    for mask, label in [(0x0F, "CPU0-3 (小核)"), (0xF0, "CPU4-7 (大核)")]:
        t0 = time.perf_counter()
        ctx = load_rkllm(MODEL_PATH, cpu_mask=mask)
        elapsed = (time.perf_counter() - t0) * 1000
        ctx["rkllm_lib"].rkllm_destroy(ctx["handle"])
        log(f"  {label}: {elapsed:.0f}ms (rkllm_init={ctx['load_ms']:.0f}ms)")
        results.append({"cpu_mask": hex(mask), "label": label, "total_ms": round(elapsed, 0)})
        time.sleep(0.5)
    return results

def test_prefill():
    """系统提示词预填充时间"""
    log("\n2.2 系统提示词预填充时间")
    ctx = load_rkllm(MODEL_PATH, cpu_mask=0x0F)
    ms = do_prefill(ctx, SYS_PROMPT)
    log(f"  系统提示词预填充: {ms:.0f}ms")
    log(f"  提示词长度: {len(SYS_PROMPT)} chars")
    ctx["rkllm_lib"].rkllm_destroy(ctx["handle"])
    return {"prefill_ms": round(ms, 0), "prompt_len": len(SYS_PROMPT)}

def test_inference():
    """推理性能测试 — 不同 prompt 长度"""
    log("\n2.3 推理性能测试")
    ctx = load_rkllm(MODEL_PATH, cpu_mask=0x0F, max_new_tokens=60)

    # 预热: prefill + generate
    do_prefill(ctx, SYS_PROMPT)
    do_generate(ctx, "你好", max_tokens=10)

    test_prompts = [
        ("短(4字)", "感冒的症状是什么"),
        ("中(8字)", "发烧了应该怎么办"),
        ("长(14字)", "高血压患者日常需要注意什么"),
    ]
    results = []
    for label, prompt in test_prompts:
        log(f"\n  --- {label}: '{prompt}' ---")
        for i in range(ROUNDS):
            do_prefill(ctx, SYS_PROMPT)
            r = do_generate(ctx, prompt, max_tokens=60)
            log(f"    尝试{i+1}: TTFT={r['ttft_ms']:.0f}ms  total={r['total_ms']:.0f}ms  "
                f"{r['num_tokens']}tokens ({r['tokens_per_sec']:.1f} tok/s)  "
                f"响应={r['text'][:30]}")
            results.append({"prompt_label": label, "prompt": prompt, "attempt": i+1, **r})
            time.sleep(0.3)
        # 平均
        group = [r for r in results if r["prompt_label"] == label]
        avg_ttft = np.mean([r["ttft_ms"] for r in group])
        avg_total = np.mean([r["total_ms"] for r in group])
        avg_tps = np.mean([r["tokens_per_sec"] for r in group])
        log(f"    → 平均: TTFT={avg_ttft:.0f}ms  total={avg_total:.0f}ms  {avg_tps:.1f} tok/s")

    ctx["rkllm_lib"].rkllm_destroy(ctx["handle"])
    return results

def test_max_tokens_scaling():
    """max_tokens 扩展性测试"""
    log("\n2.4 max_tokens 扩展性测试")
    ctx = load_rkllm(MODEL_PATH, cpu_mask=0x0F, max_new_tokens=120)
    do_prefill(ctx, SYS_PROMPT)

    prompt = "请详细说明感冒的不同症状和治疗方法"
    results = []
    for mt in [20, 40, 60, 80]:
        # 重新加载不同 max_tokens
        ctx["rkllm_lib"].rkllm_destroy(ctx["handle"])
        ctx = load_rkllm(MODEL_PATH, cpu_mask=0x0F, max_new_tokens=mt)
        do_prefill(ctx, SYS_PROMPT)
        r = do_generate(ctx, prompt, max_tokens=mt)
        log(f"  max_tokens={mt:3d}: TTFT={r['ttft_ms']:.0f}ms  total={r['total_ms']:.0f}ms  "
            f"{r['num_tokens']}tokens ({r['tokens_per_sec']:.1f} tok/s)")
        results.append({**r, "max_tokens": mt})
        time.sleep(0.3)

    ctx["rkllm_lib"].rkllm_destroy(ctx["handle"])
    return results

def test_cpu_affinity():
    """CPU 绑核对生成速度的影响"""
    log("\n2.5 CPU 绑核对生成速度的影响")
    configs = [
        (0x0F, "CPU0-3 (小核, 当前配置)"),
        (0xF0, "CPU4-7 (大核)"),
        (0xFF, "CPU0-7 (全部核心)"),
    ]
    results = []
    prompt = "感冒的症状是什么"
    for mask, label in configs:
        log(f"\n  --- {label} ---")
        ctx = load_rkllm(MODEL_PATH, cpu_mask=mask, max_new_tokens=60)
        do_prefill(ctx, SYS_PROMPT)
        r = do_generate(ctx, prompt, max_tokens=60)
        log(f"    TTFT={r['ttft_ms']:.0f}ms  total={r['total_ms']:.0f}ms  "
            f"{r['num_tokens']}tokens ({r['tokens_per_sec']:.1f} tok/s)")
        results.append({"cpu_mask": hex(mask), "label": label, **r})
        ctx["rkllm_lib"].rkllm_destroy(ctx["handle"])
        time.sleep(0.5)

    # 找最佳
    best = min(results, key=lambda x: x["total_ms"])
    log(f"\n  最佳配置: {best['label']} ({best['total_ms']}ms)")
    return results

def get_npu_load():
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            return f.read().strip()
    except:
        return "N/A"

def main():
    log("=" * 60)
    log("LLM (Qwen3-0.6B) RKLLM 模型深度性能分析")
    log("=" * 60)
    log(f"模型: {MODEL_PATH}")
    log(f"系统提示词: '{SYS_PROMPT}'")

    results = {
        "model": "Qwen3-0.6B W8A8",
        "model_path": MODEL_PATH,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1. 环境
    log("\n1. 环境信息")
    try:
        sz = os.path.getsize(MODEL_PATH)
        results["model_size_mb"] = round(sz / 1e6, 1)
        log(f"  模型大小: {results['model_size_mb']}MB")
    except: pass
    try:
        npu_load = get_npu_load()
        results["npu_load_idle"] = npu_load
        log(f"  NPU 空闲负载: {npu_load}")
    except: pass

    # 2. 测试
    results["model_loading"] = test_model_loading()
    results["prefill"] = test_prefill()
    results["inference"] = test_inference()
    results["max_tokens_scaling"] = test_max_tokens_scaling()
    results["cpu_affinity"] = test_cpu_affinity()

    # 3. 计算关键指标
    log("\n" + "=" * 60)
    log("关键指标汇总")
    log("=" * 60)
    inf_results = results["inference"]
    for label in ["短(4字)", "中(8字)", "长(14字)"]:
        group = [r for r in inf_results if r["prompt_label"] == label]
        if group:
            avg_ttft = np.mean([r["ttft_ms"] for r in group])
            avg_total = np.mean([r["total_ms"] for r in group])
            avg_tps = np.mean([r["tokens_per_sec"] for r in group])
            log(f"  {label}: TTFT={avg_ttft:.0f}ms  total={avg_total:.0f}ms  {avg_tps:.1f} tok/s")

    best_aff = min(results["cpu_affinity"], key=lambda x: x["total_ms"])
    log(f"  最佳 CPU 配置: {best_aff['label']} ({best_aff['total_ms']}ms)")

    # 输出 JSON
    out_path = "/tmp/analyze_llm_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存: {out_path}")

    return results

if __name__ == "__main__":
    main()