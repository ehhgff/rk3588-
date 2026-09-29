#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-R1-CMtMedQA LLM 流式服务程序 — keep_history=1 预填充版
- 启动时预填充系统提示词到 NPU KV cache（内置模板，role=b"system"）
- 每次请求用内置模板传用户输入（role=b"user"），省去手动格式化
- enable_thinking=False：关闭思考链，直接输出回答，避免思考泄漏到结果中
- Unix Socket 通信，JSON Lines 协议

请求:
  {"prompt": "...", "max_tokens": 80, "temperature": 0.7, "stream": true}

流式响应（每行一个 JSON）:
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
import ctypes
import queue
import re

# ─── 配置 ───
MODEL_PATH = "/data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm"
SOCKET_PATH = "/tmp/qwen3_llm_streaming.sock"
MAX_CONTEXT_LEN = 2048
MAX_NEW_TOKENS = 80
LOG_FILE = "/tmp/deepseek_llm_service.log"

# 系统提示词（预填充到 NPU，通过内置模板 role=b"system"）
SYSTEM_PROMPT = "你是一个医疗助手，请用一句话简短回答患者问题。"

# ─── 全局 ───
llm = None
service_running = False


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


# ═══════════════════════════════════════════════
# PrefilledRKLLM — keep_history=1 预填充 RKLLM 封装
# ═══════════════════════════════════════════════

# ── 思考链过滤 ──
_THINK_TAGS = re.compile(r'结构.*?结构|躯.*?躯|<think>.*?</think>', re.DOTALL)

def strip_thinking(text):
    """移除 DeepSeek-R1 思考链内容（支持 结构…结构、躯…躯、<think>…</think>）"""
    return _THINK_TAGS.sub('', text).strip()


class PrefilledRKLLM:
    """
    支持 keep_history=1 + 系统提示词预填充的 RKLLM 封装（DeepSeek-R1 版）
    - 启动时预填充 SYSTEM_PROMPT（通过内置模板 role=b"system"）
    - 每次请求传原始用户输入（role=b"user", enable_thinking=True）
    - 自动过滤思考链内容（结构…结构）
    - 回答结束后自动 reset 上下文，供下一轮使用
    """

    def __init__(self, model_path, max_context_len=2048, max_new_tokens=80, platform="rk3588"):
        self.model_path = model_path
        self.max_context_len = max_context_len
        self.max_new_tokens = max_new_tokens
        self.platform = platform

        # 流式
        self._token_queue = queue.Queue()
        self._finished = False
        self._error = None
        self._callback_ref = None

        self._init_rkllm()

    # ── ctypes 回调 ──
    def _create_callback(self):
        LLMResultCallback = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        )

        def callback_impl(result_ptr, userdata, state):
            try:
                if state == 0:  # RKLLM_RUN_NORMAL
                    # 从 RKLLMResult 读取 text
                    class RKLLMResult(ctypes.Structure):
                        _fields_ = [("text", ctypes.c_char_p)]
                    r = ctypes.cast(result_ptr, ctypes.POINTER(RKLLMResult))
                    text = r.contents.text.decode('utf-8')
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

        # 默认参数
        rkllm_lib.rkllm_createDefaultParam.argtypes = []
        rkllm_lib.rkllm_createDefaultParam.restype = RKLLMParam
        rkllm_param = rkllm_lib.rkllm_createDefaultParam()

        rkllm_param.model_path = bytes(self.model_path, 'utf-8')
        rkllm_param.max_context_len = self.max_context_len
        rkllm_param.max_new_tokens = self.max_new_tokens
        rkllm_param.skip_special_token = True
        rkllm_param.is_async = True

        # CPU 亲和性 (大核 4-7)
        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.enabled_cpus_num = 4
        if self.platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0

        self.handle = RKLLM_Handle_t()
        self._callback_ref = self._create_callback()

        rkllm_lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam),
            type(self._callback_ref),
        ]
        rkllm_lib.rkllm_init.restype = ctypes.c_int

        ret = rkllm_lib.rkllm_init(
            ctypes.byref(self.handle), ctypes.byref(rkllm_param), self._callback_ref
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_init 失败: {ret}")

        # 异步运行
        rkllm_lib.rkllm_run_async.argtypes = [
            RKLLM_Handle_t, ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p,
        ]
        rkllm_lib.rkllm_run_async.restype = ctypes.c_int

        # 中止
        rkllm_lib.rkllm_abort.argtypes = [RKLLM_Handle_t]
        rkllm_lib.rkllm_abort.restype = ctypes.c_int

        # 销毁
        rkllm_lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        rkllm_lib.rkllm_destroy.restype = ctypes.c_int

        # 推理参数 — keep_history=1 启用上下文持久化
        self.infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        self.infer_param.mode = 0       # RKLLM_INFER_GENERATE
        self.infer_param.lora_params = None
        self.infer_param.prompt_cache_params = None
        self.infer_param.keep_history = 1

        print(f"[PrefilledLLM] ✓ 模型加载完成 (keep_history=1)", file=sys.stderr)

    # ── 预填充 ──
    def prefill_system(self, system_prompt):
        """
        预填充系统提示词到 NPU KV cache。
        调用 rkllm_run_async 后立即 abort，只做 prefill 不生成 token。
        """
        self._token_queue = queue.Queue()
        self._finished = False
        self._error = None

        from rkllm_python_api_streaming import RKLLMInput

        # 预填充只需要存系统提示词到 KV cache，不需要生成思考链
        input_data = RKLLMInput()
        input_data.role = b"system"
        input_data.enable_thinking = False
        input_data.input_type = 0
        input_data.prompt_input = system_prompt.encode('utf-8')

        ret = self._rkllm_lib.rkllm_run_async(
            self.handle, ctypes.byref(input_data), ctypes.byref(self.infer_param), None
        )
        if ret != 0:
            log(f"prefill_system rkllm_run_async 失败: {ret}")
            return False

        # 给 prefill 一点时间完成（同步部分），然后 abort 生成
        time.sleep(0.05)
        self._rkllm_lib.rkllm_abort(self.handle)

        # 排空可能已生成的 token
        while True:
            try:
                t = self._token_queue.get(timeout=0.5)
                if t is None:
                    break
            except queue.Empty:
                break

        print(f"[PrefilledLLM] ✓ 系统提示词预填充完成: \"{system_prompt}\"", file=sys.stderr)
        log(f"系统提示词预填充完成: {system_prompt}")
        return True

    # ── 流式生成 ──
    def generate_streaming(self, prompt, max_tokens=None, temperature=0.7):
        """
        流式生成（DeepSeek-R1 内置模板版）。
        - keep_history=1：系统提示词已在 KV cache 中
        - 传原始 prompt，内置模板处理 role=b"user" 格式
        - enable_thinking=True → 输出可能含 结构…结构 思考链，自动过滤
        """
        self._token_queue = queue.Queue()
        self._finished = False
        self._error = None

        from rkllm_python_api_streaming import RKLLMInput

        # 直接回答，不生成思考链（避免思考泄漏到输出中）
        input_data = RKLLMInput()
        input_data.role = b"user"
        input_data.enable_thinking = False
        input_data.input_type = 0
        input_data.prompt_input = prompt.encode('utf-8')

        ret = self._rkllm_lib.rkllm_run_async(
            self.handle, ctypes.byref(input_data), ctypes.byref(self.infer_param), None
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_run_async 失败: {ret}")

        # 读取 token 队列 + 过滤思考链
        # 状态机：UTTER = 0, IN_THINK = 1
        think_state = 0  # 0=UTTER, 1=IN_THINK
        think_starters = ('结构', '躯', '<think>')
        think_enders = ('结构', '躯', '</think>')

        while True:
            try:
                token = self._token_queue.get(timeout=30)
                if token is None:
                    break
            except queue.Empty:
                break

            if not token:
                continue

            if think_state == 0:
                # ── 不在思考块中 ──
                # 找 think 开始标记
                earliest_pos = len(token)
                earliest_marker = None
                for marker in think_starters:
                    pos = token.find(marker)
                    if pos != -1 and pos < earliest_pos:
                        earliest_pos = pos
                        earliest_marker = marker

                if earliest_marker is not None:
                    # 遇到思考开始
                    before = token[:earliest_pos]
                    if before:
                        yield before
                    think_state = 1
                    after = token[earliest_pos + len(earliest_marker):]
                    if after:
                        # 在同一 chunk 内查找思考结束
                        end_pos = -1
                        for end_marker in think_enders:
                            ep = after.find(end_marker)
                            if ep != -1 and (end_pos == -1 or ep < end_pos):
                                end_pos = ep
                                earliest_end_marker = end_marker
                        if end_pos != -1:
                            # 思考在一段内结束
                            think_state = 0
                            remainder = after[end_pos + len(earliest_end_marker):]
                            if remainder:
                                yield remainder
                else:
                    yield token
            else:
                # ── 在思考块中 ──
                # 找思考结束标记
                end_pos = -1
                for marker in think_enders:
                    pos = token.find(marker)
                    if pos != -1 and (end_pos == -1 or pos < end_pos):
                        end_pos = pos
                        earliest_end_marker = marker
                if end_pos != -1:
                    think_state = 0
                    after = token[end_pos + len(earliest_end_marker):]
                    if after:
                        yield after

        if self._error:
            raise RuntimeError(self._error)

    # ── 重置上下文 ──
    def reset_context(self):
        """
        重置模型上下文（销毁 → 重新初始化 → 预填充）。
        耗时约 2-3s，调用方应在 ASR 阶段提前调用以隐藏延迟。
        """
        if not self.handle:
            return False

        log("正在重置 LLM 上下文...")
        print(f"[PrefilledLLM] 重置上下文...", file=sys.stderr)

        # 先 abort 当前生成
        try:
            self._rkllm_lib.rkllm_abort(self.handle)
        except:
            pass
        time.sleep(0.05)

        # 销毁旧 handle
        try:
            self._rkllm_lib.rkllm_destroy(self.handle)
        except:
            pass
        self.handle = None

        # 重新初始化
        from rkllm_python_api_streaming import RKLLMParam, RKLLMInput, RKLLMInferParam, RKLLM_Handle_t

        rkllm_lib = self._rkllm_lib

        rkllm_lib.rkllm_createDefaultParam.argtypes = []
        rkllm_lib.rkllm_createDefaultParam.restype = RKLLMParam
        rkllm_param = rkllm_lib.rkllm_createDefaultParam()

        rkllm_param.model_path = bytes(self.model_path, 'utf-8')
        rkllm_param.max_context_len = self.max_context_len
        rkllm_param.max_new_tokens = self.max_new_tokens
        rkllm_param.skip_special_token = True
        rkllm_param.is_async = True
        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.enabled_cpus_num = 4
        rkllm_param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0

        new_handle = RKLLM_Handle_t()
        rkllm_lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam),
            type(self._callback_ref),
        ]
        rkllm_lib.rkllm_init.restype = ctypes.c_int

        ret = rkllm_lib.rkllm_init(
            ctypes.byref(new_handle), ctypes.byref(rkllm_param), self._callback_ref
        )
        if ret != 0:
            log(f"重置上下文失败: rkllm_init={ret}")
            print(f"[PrefilledLLM] ❌ 重置上下文失败", file=sys.stderr)
            return False

        self.handle = new_handle
        self.infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        self.infer_param.mode = 0
        self.infer_param.lora_params = None
        self.infer_param.prompt_cache_params = None
        self.infer_param.keep_history = 1

        # 重新预填充系统提示词
        self.prefill_system(SYSTEM_PROMPT)

        print(f"[PrefilledLLM] ✓ 上下文重置完成", file=sys.stderr)
        log("LLM 上下文重置完成")
        return True

    def release(self):
        if self.handle:
            try:
                self._rkllm_lib.rkllm_destroy(self.handle)
            except:
                pass
            self.handle = None

    def __del__(self):
        self.release()


# ═══════════════════════════════════════════════
# Socket 服务
# ═══════════════════════════════════════════════

_need_reset = False        # 需要重置上下文的标志
_resetting = False         # 正在重置中
_reset_lock = threading.Lock()


def handle_client(conn):
    """处理单个客户端连接"""
    global _need_reset, _resetting

    # 如果在重置中，等待重置完成（最多等 10s）
    for _ in range(100):
        if not _resetting:
            break
        time.sleep(0.1)
    else:
        try: conn.close()
        except: pass
        return

    try:
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

        request = json.loads(data.decode('utf-8'))
        prompt = request.get('prompt', '')
        max_tokens = request.get('max_tokens', MAX_NEW_TOKENS)
        temperature = request.get('temperature', 0.7)
        stream_mode = request.get('stream', False)

        if prompt == 'ping':
            conn.sendall(json.dumps({'response': 'pong', 'time_ms': 0}).encode('utf-8'))
            conn.close()
            return

        # __RESET__：客户端主动重置对话上下文（15分钟无语音时调用）
        if prompt == '__RESET__':
            log("收到 RESET 命令，重置上下文...")
            print(f"[Qwen3-LLM] 收到 RESET 命令", file=sys.stderr)

            def _do_reset():
                global _resetting
                with _reset_lock:
                    _resetting = True
                try:
                    llm.reset_context()
                finally:
                    with _reset_lock:
                        _resetting = False
            threading.Thread(target=_do_reset, daemon=True).start()

            conn.sendall(json.dumps({'response': 'reset_ok'}).encode('utf-8'))
            conn.close()
            return

        log(f"收到请求: {prompt[:50]}... (stream={stream_mode})")
        print(f"[Qwen3-LLM] 收到请求: {prompt[:50]}...", file=sys.stderr)

        start_time = time.time()

        # ── 流式生成 ──
        full_response = ""
        sentence_buffer = ""

        for token in llm.generate_streaming(prompt, max_tokens=max_tokens, temperature=temperature):
            full_response += token
            sentence_buffer += token

            if token and token[-1] in '。！？.!?\n':
                sentence = sentence_buffer.strip()
                if sentence:
                    if stream_mode:
                        msg = json.dumps({'sentence': sentence}, ensure_ascii=False) + '\n'
                        conn.sendall(msg.encode('utf-8'))
                    sentence_buffer = ""

        rest = sentence_buffer.strip()
        if rest:
            if stream_mode:
                msg = json.dumps({'sentence': rest}, ensure_ascii=False) + '\n'
                conn.sendall(msg.encode('utf-8'))

        elapsed_ms = (time.time() - start_time) * 1000
        full_response = full_response.strip()

        log(f"生成完成: {len(full_response)} chars, {elapsed_ms:.0f}ms")
        print(f"[Qwen3-LLM] 生成完成: {len(full_response)} chars, {elapsed_ms:.0f}ms", file=sys.stderr)

        if stream_mode:
            done_msg = json.dumps({'done': True, 'response': full_response}, ensure_ascii=False) + '\n'
            conn.sendall(done_msg.encode('utf-8'))
        else:
            result = {'response': full_response, 'time_ms': elapsed_ms}
            conn.sendall(json.dumps(result, ensure_ascii=False).encode('utf-8'))

        # 多轮对话：不自动重置，客户端在 15 分钟超时后发 __RESET__ 命令
        # （reset_worker 线程只响应 __RESET__ 触发的重置）

    except json.JSONDecodeError as e:
        log(f"JSON解析错误: {e}")
        try:
            conn.sendall((json.dumps({'error': '无效的JSON请求'}) + '\n').encode('utf-8'))
        except:
            pass
    except Exception as e:
        log(f"处理错误: {e}")
        print(f"[Qwen3-LLM] 处理错误: {e}", file=sys.stderr)
        try:
            err = json.dumps({'error': str(e), 'done': True}) + '\n'
            conn.sendall(err.encode('utf-8'))
        except:
            pass
    finally:
        try:
            conn.close()
        except:
            pass


def reset_worker():
    """后台线程：检测 _need_reset 标志，在下一轮请求前重置上下文"""
    global _need_reset, _resetting
    while service_running:
        if _need_reset:
            with _reset_lock:
                _need_reset = False
                _resetting = True
            log("后台重置上下文...")
            print(f"[Qwen3-LLM] 后台重置上下文...", file=sys.stderr)
            try:
                llm.reset_context()
            except Exception as e:
                log(f"重置上下文异常: {e}")
            finally:
                with _reset_lock:
                    _resetting = False
        time.sleep(0.5)


def signal_handler(signum, frame):
    global service_running
    log(f"收到信号 {signum}")
    print(f"[Qwen3-LLM] 收到信号 {signum}", file=sys.stderr)
    service_running = False


def main():
    global service_running, llm

    open(LOG_FILE, 'w').close()
    log("DeepSeek-R1-CMtMedQA LLM 服务启动 (keep_history=1 预填充版)")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # 加载模型
    os.environ['RKLLM_LIB_PATH'] = '/data/qwen3/librkllmrt.so'
    sys.path.insert(0, '/userdata/voice_assistant')
    sys.path.insert(0, '/userdata/voice_assistant/core')

    print("[Qwen3-LLM] 正在加载 Qwen3-0.6B (890MB)...", file=sys.stderr)
    log("正在加载模型...")

    try:
        llm = PrefilledRKLLM(
            MODEL_PATH,
            max_context_len=MAX_CONTEXT_LEN,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    except Exception as e:
        log(f"模型加载失败: {e}")
        print(f"[Qwen3-LLM] ❌ 模型加载失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 预填充系统提示词
    print(f"[Qwen3-LLM] 预填充系统提示词...", file=sys.stderr)
    llm.prefill_system(SYSTEM_PROMPT)

    # 启动后台重置线程
    threading.Thread(target=reset_worker, daemon=True).start()

    # Socket 服务
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(SOCKET_PATH)
    server.listen(10)
    os.chmod(SOCKET_PATH, 0o666)

    old_sock = "/tmp/qwen3_llm.sock"
    if os.path.exists(old_sock):
        os.remove(old_sock)
    try:
        os.symlink(SOCKET_PATH, old_sock)
    except:
        pass

    service_running = True
    log(f"服务已启动: {SOCKET_PATH}")
    print(f"[Qwen3-LLM] 服务已启动: {SOCKET_PATH}", file=sys.stderr)
    print(f"[Qwen3-LLM] 系统提示词: \"{SYSTEM_PROMPT}\"", file=sys.stderr)

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
        for f in [SOCKET_PATH, old_sock]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        if llm:
            try:
                llm.release()
            except:
                pass
        log("服务已停止")
        print("[Qwen3-LLM] 服务已停止", file=sys.stderr)


if __name__ == '__main__':
    main()
