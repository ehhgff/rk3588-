from rkllm_python_api_streaming import RKLLMStreaming, rkllm_lib, RKLLMInput, _callback_type, _instance_map
import ctypes
import time

print("加载模型...")
llm = RKLLMStreaming("/data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm", max_new_tokens=20)

print("\n测试直接调用 rkllm_run_async:")
print("实例ID: %d" % llm._instance_id)
print("实例是否在映射中: %s" % (llm._instance_id in _instance_map))

# 准备输入
formatted_prompt = b"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

input_data = RKLLMInput()
input_data.role = b'user'
input_data.enable_thinking = False
input_data.input_type = 0
input_data.prompt_input = formatted_prompt

print("调用 rkllm_run_async...")
start = time.time()
ret = rkllm_lib.rkllm_run_async(
    llm.handle,
    ctypes.byref(input_data),
    ctypes.byref(llm.infer_param),
    ctypes.c_void_p(llm._instance_id)
)
print("rkllm_run_async 返回: %d" % ret)

# 等待回调，检查队列
time.sleep(1)
print("队列大小: %d" % llm._token_queue.qsize())
print("是否完成: %s" % llm._finished)

time.sleep(4)
print("5秒后队列大小: %d" % llm._token_queue.qsize())
print("是否完成: %s" % llm._finished)

end = time.time()
print("耗时: %dms" % int((end-start)*1000))

llm.release()
