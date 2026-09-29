from rkllm_python_api_streaming import RKLLMStreaming
import time

print("加载模型...")
llm = RKLLMStreaming("/data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm", max_new_tokens=20)

print("\n测试流式生成:")
start = time.time()

# 使用更短的超时
tokens = []
for token in llm.generate_streaming("你好"):
    tokens.append(token)
    if len(tokens) > 50:
        break

end = time.time()
print(f"\n\n耗时: {(end-start)*1000:.0f}ms")
print(f"Token数: {len(tokens)}")
print(f"内容: {''.join(tokens)}")

llm.release()
