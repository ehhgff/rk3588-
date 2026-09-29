#!/usr/bin/env python3
"""RK3588 上的 DeepSeek-R1-Medical RKLLM 推理测试"""

import sys
sys.path.insert(0, '/usr/lib')

from rkllm.api import RKLLM

def main():
    model_path = "/userdata/llm_models/DeepSeek-R1-Medical-w8a8.rkllm"
    
    print("加载 RKLLM 模型...")
    llm = RKLLM()
    
    # 加载模型
    ret = llm.load_rkllm(model_path)
    if ret != 0:
        print(f"模型加载失败: {ret}")
        return
    
    print("模型加载成功!")
    
    # 测试提示词
    prompts = [
        "患者头痛发热3天，请给出诊断建议。",
        "咳嗽咳痰2周，痰中带血，可能是什么疾病？",
        "65岁糖尿病患者，空腹血糖9.2，如何用药？"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*50}")
        print(f"测试 {i}: {prompt}")
        print('='*50)
        
        # 生成回复
        response = llm.inference(prompt)
        print(f"回复: {response}")
    
    # 释放模型
    llm.release()
    print("\n测试完成!")

if __name__ == "__main__":
    main()
