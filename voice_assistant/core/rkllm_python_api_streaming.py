#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKLLM Python API 流式封装
支持真正的逐 token 流式输出
使用 rkllm_run_async 实现异步生成
"""

import os
import sys
import ctypes
import time
import threading
import queue

# 设置动态库路径
rkllm_lib = ctypes.CDLL(os.environ.get('RKLLM_LIB_PATH', '/data/qwen3/librkllmrt.so'))

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


class RKLLMStreaming:
    """支持流式输出的 RKLLM 封装"""
    
    def __init__(self, model_path, max_context_len=2048, max_new_tokens=45, platform="rk3588"):
        self.model_path = model_path
        self.max_context_len = max_context_len
        self.max_new_tokens = max_new_tokens
        self.platform = platform
        
        # 流式相关
        self._token_queue = queue.Queue()
        self._finished = False
        self._error = None
        
        # 回调相关（必须作为实例属性保存，防止被垃圾回收）
        self._callback_ref = None  # 保持回调引用
        
        # 初始化 RKLLM
        self._init_rkllm()
    
    def _create_callback(self):
        """创建并返回回调函数"""
        # 回调函数类型
        LLMResultCallback = ctypes.CFUNCTYPE(
            ctypes.c_int, 
            ctypes.POINTER(RKLLMResult), 
            ctypes.c_void_p, 
            ctypes.c_int
        )
        
        def callback_impl(result, userdata, state):
            """回调函数实现"""
            try:
                if state == 0:  # RKLLM_RUN_NORMAL
                    text = result.contents.text.decode('utf-8')
                    self._token_queue.put(text)
                elif state == 2:  # RKLLM_RUN_FINISH
                    self._finished = True
                    self._token_queue.put(None)  # 结束标记
                elif state == 3:  # RKLLM_RUN_ERROR
                    self._error = "RKLLM运行错误"
                    self._finished = True
                    self._token_queue.put(None)
            except Exception as e:
                print("回调异常: %s" % e, file=sys.stderr)
            return 0
        
        # 创建回调函数并保存引用
        callback = LLMResultCallback(callback_impl)
        return callback
    
    def _init_rkllm(self):
        """初始化 RKLLM"""
        # 使用 rkllm_createDefaultParam 创建默认参数
        rkllm_lib.rkllm_createDefaultParam.argtypes = []
        rkllm_lib.rkllm_createDefaultParam.restype = RKLLMParam
        
        rkllm_param = rkllm_lib.rkllm_createDefaultParam()
        
        # 修改参数
        rkllm_param.model_path = bytes(self.model_path, 'utf-8')
        rkllm_param.max_context_len = self.max_context_len
        rkllm_param.max_new_tokens = self.max_new_tokens
        rkllm_param.skip_special_token = True
        rkllm_param.is_async = True  # 启用异步模式
        
        # 设置 extend_param
        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.enabled_cpus_num = 4
        if self.platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0)|(1 << 1)|(1 << 2)|(1 << 3)
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0
        
        self.handle = RKLLM_Handle_t()
        
        # 创建回调函数（必须在 init 之前）
        self._callback_ref = self._create_callback()
        
        # 初始化
        rkllm_lib.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), type(self._callback_ref)]
        rkllm_lib.rkllm_init.restype = ctypes.c_int
        
        ret = rkllm_lib.rkllm_init(
            ctypes.byref(self.handle), 
            ctypes.byref(rkllm_param), 
            self._callback_ref
        )
        if ret != 0:
            raise RuntimeError("rkllm init failed: %d" % ret)
        
        # 异步运行
        rkllm_lib.rkllm_run_async.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        rkllm_lib.rkllm_run_async.restype = ctypes.c_int
        
        # 销毁
        rkllm_lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        rkllm_lib.rkllm_destroy.restype = ctypes.c_int
        
        # 推理参数
        self.infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        self.infer_param.mode = 0  # RKLLM_INFER_GENERATE
        self.infer_param.lora_params = None
        self.infer_param.prompt_cache_params = None
        self.infer_param.keep_history = 0
        
        print("[RKLLMStreaming] 模型加载成功 (异步模式)", file=sys.stderr)
    
    def generate_streaming(self, prompt, max_tokens=None, temperature=0.7):
        """
        流式生成回答
        
        Yields:
            str: 每个生成的token
        """
        # 重置状态
        self._token_queue = queue.Queue()
        self._finished = False
        self._error = None
        
        # 格式化提示词
        formatted_prompt = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % prompt
        
        # 准备输入
        input_data = RKLLMInput()
        input_data.role = b'user'
        input_data.enable_thinking = False
        input_data.input_type = 0  # RKLLM_INPUT_PROMPT
        input_data.prompt_input = formatted_prompt.encode('utf-8')
        
        # 异步运行
        ret = rkllm_lib.rkllm_run_async(
            self.handle,
            ctypes.byref(input_data),
            ctypes.byref(self.infer_param),
            None  # userdata 不需要，因为回调已经绑定了实例
        )
        
        if ret != 0:
            raise RuntimeError("rkllm_run_async failed: %d" % ret)
        
        # 从队列中读取token并yield
        while True:
            try:
                token = self._token_queue.get(timeout=30)
                if token is None:  # 结束标记
                    break
                yield token
            except queue.Empty:
                break
        
        if self._error:
            raise RuntimeError(self._error)
    
    def generate(self, prompt, max_tokens=None, temperature=0.7):
        """
        同步生成回答（兼容旧接口）
        
        Returns:
            str: 完整回复
        """
        tokens = []
        for token in self.generate_streaming(prompt, max_tokens, temperature):
            tokens.append(token)
        return ''.join(tokens)
    
    def release(self):
        """释放资源"""
        if self.handle:
            rkllm_lib.rkllm_destroy(self.handle)
            self.handle = None
    
    def __del__(self):
        self.release()


if __name__ == "__main__":
    # 测试
    model_path = "/data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm"
    
    print("加载模型...", file=sys.stderr)
    llm = RKLLMStreaming(model_path, max_new_tokens=20)
    
    print("\n测试流式生成:", file=sys.stderr)
    for token in llm.generate_streaming("Hello"):
        print(token, end='', flush=True)
    print()
    
    llm.release()
