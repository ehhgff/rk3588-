import os
import sys
import ctypes
import time

# 设置动态库路径
rkllm_lib = ctypes.CDLL('/data/qwen3/librkllmrt.so')

# 定义结构体
RKLLM_Handle_t = ctypes.c_void_p

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("prompt_input", ctypes.c_char_p),
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
    ]

# 回调函数
callback_count = [0]
tokens_received = []

LLMResultCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

def callback_impl(result, userdata, state):
    callback_count[0] += 1
    if state == 0:  # RKLLM_RUN_NORMAL
        text = result.contents.text.decode('utf-8')
        tokens_received.append(text)
        print("[callback] token: '%s'" % text, flush=True)
    elif state == 2:  # RKLLM_RUN_FINISH
        print("[callback] finish", flush=True)
    return 0

callback = LLMResultCallback(callback_impl)

# 使用 rkllm_createDefaultParam 创建默认参数
rkllm_lib.rkllm_createDefaultParam.argtypes = []
rkllm_lib.rkllm_createDefaultParam.restype = RKLLMParam

rkllm_param = rkllm_lib.rkllm_createDefaultParam()
print("Default param created")
print("model_path: %s" % rkllm_param.model_path)
print("max_context_len: %d" % rkllm_param.max_context_len)
print("max_new_tokens: %d" % rkllm_param.max_new_tokens)

# 修改参数
rkllm_param.model_path = b'/data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm'
rkllm_param.max_context_len = 2048
rkllm_param.max_new_tokens = 20
rkllm_param.skip_special_token = True
rkllm_param.is_async = True  # 启用异步

# 设置 extend_param
rkllm_param.extend_param.base_domain_id = 0
rkllm_param.extend_param.embed_flash = 1
rkllm_param.extend_param.enabled_cpus_num = 4
rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)
rkllm_param.extend_param.n_batch = 1
rkllm_param.extend_param.use_cross_attn = 0

handle = RKLLM_Handle_t()
ret = rkllm_lib.rkllm_init(ctypes.byref(handle), ctypes.byref(rkllm_param), callback)
print("init result: %d" % ret)

# 准备输入
prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
input_data = RKLLMInput()
input_data.role = b'user'
input_data.enable_thinking = False
input_data.input_type = 0  # RKLLM_INPUT_PROMPT
input_data.prompt_input = prompt.encode('utf-8')

infer_param = RKLLMInferParam()
infer_param.mode = 0  # RKLLM_INFER_GENERATE
infer_param.lora_params = None
infer_param.prompt_cache_params = None
infer_param.keep_history = 0

# 异步运行
print("start async generation...")
start = time.time()
ret = rkllm_lib.rkllm_run_async(handle, ctypes.byref(input_data), ctypes.byref(infer_param), None)
print("rkllm_run_async return: %d" % ret)

# 等待生成完成
time.sleep(5)

print("\ncallback count: %d" % callback_count[0])
print("tokens received: %d" % len(tokens_received))
print("content: %s" % ''.join(tokens_received))
print("time: %dms" % int((time.time()-start)*1000))

# 销毁
rkllm_lib.rkllm_destroy(handle)
