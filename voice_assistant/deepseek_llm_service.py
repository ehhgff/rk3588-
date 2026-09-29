#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-R1-Medical LLM 服务程序
基于 RKLLMStreaming（异步回调），支持逐句流式输出
Unix Socket 通信，JSON Lines 协议

与 run.sh 兼容的配置:
  - Socket: /tmp/qwen3_llm.sock
  - 客户端协议与 qwen3_llm_client.py 兼容

请求:
  {"prompt": "...", "max_tokens": 80, "temperature": 0.7, "stream": false}

非流式响应:
  {"response": "...", "time_ms": 1234}

流式响应（stream=true，每行一个 JSON）:
  {"sentence": "..."}           # 完整句子到达
  {"done": true, "response": "..."}  # 生成结束
"""

import os
import sys
import socket
import json
import time
import threading
import signal
import queue
import ctypes

# ─── 配置 ───
MODEL_PATH = "/data/llm_models/DeepSeek-R1-Medical-w8a8.rkllm"
SOCKET_PATH = "/tmp/qwen3_llm.sock"
MAX_CONTEXT_LEN = 2048
MAX_NEW_TOKENS = 80
LOG_FILE = "/tmp/deepseek_llm_service.log"

# ─── 全局 ───
llm = None
service_running = False


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


# ═══════════════════════════════════════════════
# DeepSeekRKLLM — 继承流式 API，专为 DeepSeek-R1 优化
# ═══════════════════════════════════════════════

class DeepSeekRKLLM:
    """DeepSeek-R1-Medical 流式推理封装

    与 RKLLMStreaming 的区别：
      - enable_thinking=True（DeepSeek-R1 支持推理过程）
      - 不手动格式化 prompt（使用模型内置模板）
      - 含重复检测（遇重复字符自动终止）
    """

    def __init__(self, model_path, max_context_len=2048, max_new_tokens=80, platform="rk3588"):
        self.model_path = model_path
        self.max_context_len = max_context_len
        self.max_new_tokens = max_new_tokens
        self.platform = platform

        # 流式相关
        self._token_queue = queue.Queue()
        self._finished = False
        self._error = None
        self._callback_ref = None

        self._init_rkllm()

    def _create_callback(self):
        """创建 RKLLM 异步回调"""
        LLMResultCallback = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,  # RKLLMResult*
            ctypes.c_void_p,  # userdata
            ctypes.c_int      # state
        )

        def callback_impl(result_ptr, userdata, state):
            try:
                if state == 0:  # RKLLM_RUN_NORMAL
                    # 从 result 结构体读取 text 字段
                    text_ptr = ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_char_p))
                    text = text_ptr[0].decode("utf-8")
                    self._token_queue.put(text)
                elif state == 2:  # RKLLM_RUN_FINISH
                    self._finished = True
                    self._token_queue.put(None)
                elif state == 3:  # RKLLM_RUN_ERROR
                    self._error = "RKLLM 运行错误"
                    self._finished = True
                    self._token_queue.put(None)
            except Exception as e:
                log(f"回调异常: {e}")
            return 0

        callback = LLMResultCallback(callback_impl)
        return callback

    def _init_rkllm(self):
        """初始化 RKLLM 引擎"""
        from rkllm_python_api_streaming import rkllm_lib, RKLLMParam, RKLLMInput, RKLLMInferParam, RKLLM_Handle_t

        self._rkllm_lib = rkllm_lib

        # 创建默认参数
        rkllm_lib.rkllm_createDefaultParam.argtypes = []
        rkllm_lib.rkllm_createDefaultParam.restype = RKLLMParam
        rkllm_param = rkllm_lib.rkllm_createDefaultParam()

        rkllm_param.model_path = bytes(self.model_path, "utf-8")
        rkllm_param.max_context_len = self.max_context_len
        rkllm_param.max_new_tokens = self.max_new_tokens
        rkllm_param.skip_special_token = True
        rkllm_param.is_async = True

        # CPU 亲和性设置
        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.enabled_cpus_num = 4
        if self.platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0

        self.handle = RKLLM_Handle_t()

        # 创建回调
        self._callback_ref = self._create_callback()

        # 初始化
        rkllm_lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam),
            type(self._callback_ref),
        ]
        rkllm_lib.rkllm_init.restype = ctypes.c_int

        ret = rkllm_lib.rkllm_init(
            ctypes.byref(self.handle),
            ctypes.byref(rkllm_param),
            self._callback_ref,
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_init 失败: {ret}")

        # 设置 rkllm_run_async 参数类型
        rkllm_lib.rkllm_run_async.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        rkllm_lib.rkllm_run_async.restype = ctypes.c_int

        # rkllm_abort 用于中断生成
        rkllm_lib.rkllm_abort.argtypes = [RKLLM_Handle_t]
        rkllm_lib.rkllm_abort.restype = ctypes.c_int

        # rkllm_destroy
        rkllm_lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        rkllm_lib.rkllm_destroy.restype = ctypes.c_int

        # 推理参数
        self.infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        self.infer_param.mode = 0       # RKLLM_INFER_GENERATE
        self.infer_param.lora_params = None
        self.infer_param.prompt_cache_params = None
        self.infer_param.keep_history = 0

        log("DeepSeek-R1-Medical 模型加载成功（异步模式）")
        print("[DeepSeek-LLM] ✓ DeepSeek-R1-Medical 模型加载完成", file=sys.stderr)

    def generate_streaming(self, prompt, max_tokens=None, temperature=0.7):
        """流式生成，支持 DeepSeek 推理过程

        Yields:
            str: 每个生成的 token
        """
        self._token_queue = queue.Queue()
        self._finished = False
        self._error = None

        from rkllm_python_api_streaming import RKLLMInput, RKLLMInferParam

        # DeepSeek-R1：使用模型内置模板，不手动格式化
        # enable_thinking=True 启用推理过程（thinking tags）
        input_data = RKLLMInput()
        input_data.role = b"user"
        input_data.enable_thinking = True
        input_data.input_type = 0  # RKLLM_INPUT_PROMPT
        input_data.prompt_input = prompt.encode("utf-8")

        ret = self._rkllm_lib.rkllm_run_async(
            self.handle,
            ctypes.byref(input_data),
            ctypes.byref(self.infer_param),
            None,
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_run_async 失败: {ret}")

        # 从队列读取 token
        repeat_count = 0
        last_char = None

        while True:
            try:
                token = self._token_queue.get(timeout=30)
                if token is None:
                    break
                yield token

                # 重复检测
                if len(token) == 1 and token == last_char:
                    repeat_count += 1
                else:
                    repeat_count = 0
                last_char = token if len(token) == 1 else None

                if repeat_count >= 6:
                    self._rkllm_lib.rkllm_abort(self.handle)
                    # 清空队列
                    while True:
                        try:
                            t = self._token_queue.get(timeout=0.5)
                            if t is None:
                                break
                        except queue.Empty:
                            break
                    break

            except queue.Empty:
                break

        if self._error:
            raise RuntimeError(self._error)

    def generate(self, prompt, max_tokens=None, temperature=0.7):
        """同步生成完整回复"""
        tokens = []
        for token in self.generate_streaming(prompt, max_tokens, temperature):
            tokens.append(token)
        return "".join(tokens)

    def release(self):
        if self.handle:
            try:
                self._rkllm_lib.rkllm_destroy(self.handle)
            except Exception:
                pass
            self.handle = None

    def __del__(self):
        self.release()


# ═══════════════════════════════════════════════
# Socket 服务端
# ═══════════════════════════════════════════════

def sentence_split(text):
    """按句子分割文本，支持中英文标点"""
    import re
    parts = re.split(r"(。|！|？|\.|\!|\?|；|;|\n)", text)
    sentences = []
    buf = ""
    for part in parts:
        buf += part
        if re.match(r"^[。！？\.\!\?；;\n]$", part):
            s = buf.strip()
            if s:
                sentences.append(s)
            buf = ""
    rest = buf.strip()
    if rest:
        sentences.append(rest)
    return sentences


def handle_client(conn):
    """处理单个客户端连接"""
    try:
        # ── 接收请求 ──
        data = b""
        conn.settimeout(10)
        while True:
            try:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break
            except socket.timeout:
                break

        if not data:
            return

        request = json.loads(data.decode("utf-8"))
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", MAX_NEW_TOKENS)
        temperature = request.get("temperature", 0.7)
        stream_mode = request.get("stream", False)

        # ping 健康检查
        if prompt == "ping":
            result = {"response": "pong", "time_ms": 0}
            conn.sendall(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            conn.close()
            return

        log(f"收到请求: {prompt[:60]}... (stream={stream_mode})")
        print(f"[DeepSeek-LLM] 收到请求: {prompt[:60]}... (stream={stream_mode})", file=sys.stderr)

        start_time = time.time()

        # ── 流式生成 ──
        full_response = ""
        sentence_buffer = ""

        for token in llm.generate_streaming(prompt, max_tokens=max_tokens, temperature=temperature):
            full_response += token
            sentence_buffer += token

            # 遇到句子结束标点时切分
            if token and token[-1] in "。！？.!?\n":
                sentence = sentence_buffer.strip()
                if sentence:
                    if stream_mode:
                        msg = json.dumps({"sentence": sentence}, ensure_ascii=False) + "\n"
                        conn.sendall(msg.encode("utf-8"))
                    sentence_buffer = ""

        # 剩余未结束的句子
        rest = sentence_buffer.strip()
        if rest:
            if stream_mode:
                msg = json.dumps({"sentence": rest}, ensure_ascii=False) + "\n"
                conn.sendall(msg.encode("utf-8"))

        elapsed_ms = (time.time() - start_time) * 1000
        full_response = full_response.strip()

        log(f"生成完成: {len(full_response)} chars, {elapsed_ms:.0f}ms")
        print(f"[DeepSeek-LLM] 生成完成: {len(full_response)} chars, {elapsed_ms:.0f}ms", file=sys.stderr)

        # ── 发送完成事件 ──
        if stream_mode:
            done_msg = json.dumps({"done": True, "response": full_response}, ensure_ascii=False) + "\n"
            conn.sendall(done_msg.encode("utf-8"))
        else:
            result = {"response": full_response, "time_ms": elapsed_ms}
            conn.sendall(json.dumps(result, ensure_ascii=False).encode("utf-8"))

    except json.JSONDecodeError as e:
        log(f"JSON解析错误: {e}")
        try:
            err = json.dumps({"error": "无效的JSON请求"}) + "\n"
            conn.sendall(err.encode("utf-8"))
        except Exception:
            pass
    except Exception as e:
        log(f"处理错误: {e}")
        print(f"[DeepSeek-LLM] 处理错误: {e}", file=sys.stderr)
        try:
            err = json.dumps({"error": str(e), "done": True}) + "\n"
            conn.sendall(err.encode("utf-8"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def signal_handler(signum, frame):
    global service_running
    log(f"收到信号 {signum}")
    service_running = False


def main():
    global service_running, llm

    # 清空日志
    open(LOG_FILE, "w").close()
    log("DeepSeek-R1-Medical LLM 服务启动")

    # 信号处理
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # ── 设置环境变量和导入路径 ──
    os.environ["RKLLM_LIB_PATH"] = "/data/qwen3/librkllmrt.so"
    sys.path.insert(0, "/userdata/voice_assistant")
    sys.path.insert(0, "/userdata/voice_assistant/core")

    # ── 加载模型 ──
    print("[DeepSeek-LLM] 正在加载 DeepSeek-R1-Medical 模型 (约 2.0GB)...", file=sys.stderr)
    log("正在加载模型...")

    try:
        llm = DeepSeekRKLLM(
            MODEL_PATH,
            max_context_len=MAX_CONTEXT_LEN,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    except Exception as e:
        log(f"模型加载失败: {e}")
        print(f"[DeepSeek-LLM] ❌ 模型加载失败: {e}", file=sys.stderr)
        sys.exit(1)

    # ── 启动 Socket 服务 ──
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(SOCKET_PATH)
    server.listen(10)
    os.chmod(SOCKET_PATH, 0o666)

    service_running = True
    log(f"服务已启动: {SOCKET_PATH}")
    print(f"[DeepSeek-LLM] 服务已启动: {SOCKET_PATH}", file=sys.stderr)
    print(f"[DeepSeek-LLM] 日志: {LOG_FILE}", file=sys.stderr)
    print(f"[DeepSeek-LLM] 模型: DeepSeek-R1-Medical-w8a8 (enable_thinking=True)", file=sys.stderr)

    try:
        while service_running:
            try:
                server.settimeout(1)
                conn, addr = server.accept()
                thread = threading.Thread(target=handle_client, args=(conn,))
                thread.daemon = True
                thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                log(f"接受连接错误: {e}")
    except Exception as e:
        log(f"服务错误: {e}")
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        if llm:
            try:
                llm.release()
            except Exception:
                pass
        log("服务已停止")
        print("[DeepSeek-LLM] 服务已停止", file=sys.stderr)


if __name__ == "__main__":
    main()
