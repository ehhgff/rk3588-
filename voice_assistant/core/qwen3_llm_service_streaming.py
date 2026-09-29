#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-0.6B LLM 流式服务程序
基于 RKLLM Runtime，支持 token-by-token 流式输出
使用 Unix Socket 通信
"""

import os
import sys
import socket
import json
import time
import threading
import signal
import select
from ctypes import CFUNCTYPE, c_char_p, c_int, c_void_p

# 配置
MODEL_PATH = "/data/qwen3_2048/Qwen3-0.6B_W8A8_RK3588_2048.rkllm"
SOCKET_PATH = "/tmp/qwen3_llm_streaming.sock"
MAX_CONTEXT_LEN = 2048
MAX_NEW_TOKENS = 45  # 流式模式可以生成更多token
LOG_FILE = "/tmp/qwen3_llm_streaming_service.log"

# 全局变量
llm_instance = None
service_running = False
lock = threading.Lock()

# 流式输出相关
streaming_clients = {}  # client_conn -> buffer_list
streaming_lock = threading.Lock()


def log(msg):
    """日志输出到文件"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def load_model():
    """加载 Qwen3 模型"""
    global llm_instance

    os.environ['RKLLM_LIB_PATH'] = '/data/qwen3/librkllmrt.so'
    sys.path.insert(0, '/userdata/voice_assistant')

    try:
        # 尝试使用流式API
        from rkllm_python_api_streaming import RKLLMStreaming
        
        log("正在加载 Qwen3 模型 (流式模式)...")
        print("[Qwen3-LLM流式服务] 正在加载 Qwen3 模型 (流式模式)...", file=sys.stderr)

        llm_instance = RKLLMStreaming(MODEL_PATH, MAX_CONTEXT_LEN, MAX_NEW_TOKENS)

        log("模型加载成功 (流式模式)")
        print("[Qwen3-LLM流式服务] ✓ Qwen3 模型加载完成 (流式模式)", file=sys.stderr)
        return True

    except Exception as e:
        log(f"流式模型加载失败: {e}，尝试回退到标准模式")
        print(f"[Qwen3-LLM流式服务] 流式模型加载失败: {e}，尝试回退到标准模式", file=sys.stderr)
        
        try:
            from rkllm_python_api import RKLLM
            llm_instance = RKLLM(MODEL_PATH, MAX_CONTEXT_LEN)
            log("标准模型加载成功")
            print("[Qwen3-LLM流式服务] ✓ Qwen3 标准模型加载完成", file=sys.stderr)
            return True
        except Exception as e2:
            log(f"标准模型加载也失败: {e2}")
            print(f"[Qwen3-LLM流式服务] 模型加载失败: {e2}", file=sys.stderr)
            return False


def generate_streaming(prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.7, 
                       token_callback=None, sentence_callback=None):
    """
    流式生成回答 - 使用真正的异步流式输出
    
    Args:
        prompt: 输入提示词
        max_tokens: 最大生成token数
        temperature: 温度参数
        token_callback: 每个token生成时的回调函数(token_text)
        sentence_callback: 每个完整句子生成时的回调函数(sentence_text)
    
    Returns:
        完整回复文本
    """
    global llm_instance

    if not llm_instance:
        return "错误: 模型未加载"

    try:
        # 检查是否支持流式生成
        if hasattr(llm_instance, 'generate_streaming'):
            # 使用真正的流式生成
            result_buffer = []
            sentence_buffer = []
            current_sentence = ""
            
            for token in llm_instance.generate_streaming(prompt, max_tokens, temperature):
                if token:
                    result_buffer.append(token)
                    
                    # 调用token回调
                    if token_callback:
                        token_callback(token)
                    
                    # 构建句子并检测句子结束
                    current_sentence += token
                    
                    # 检查是否有完整句子
                    for char in ['。', '！', '？', '.', '!', '?', '；', ';', '\n']:
                        if char in current_sentence:
                            parts = current_sentence.split(char)
                            for i in range(len(parts) - 1):
                                sentence = parts[i] + char
                                if sentence.strip():
                                    sentence_buffer.append(sentence.strip())
                                    # 立即调用句子回调（在服务端就触发）
                                    print(f"[服务端检测] 发现完整句子: '{sentence.strip()}'", file=sys.stderr)
                                    if sentence_callback:
                                        sentence_callback(sentence.strip())
                            current_sentence = parts[-1]
                            break
            
            # 处理最后剩余的文本
            if current_sentence.strip():
                sentence_buffer.append(current_sentence.strip())
                if sentence_callback:
                    sentence_callback(current_sentence.strip())
            
            return ''.join(result_buffer)
        else:
            # 回退到标准生成
            with lock:
                response = llm_instance.generate(prompt, max_tokens, temperature)
            
            # 模拟流式输出
            if token_callback:
                for char in response:
                    token_callback(char)
            
            if sentence_callback:
                sentences = response.split('。')
                for sentence in sentences:
                    if sentence.strip():
                        sentence_callback(sentence.strip() + '。')
            
            return response.strip()
            
    except Exception as e:
        log(f"流式生成错误: {e}")
        return f"生成错误: {e}"


def generate(prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.7):
    """标准同步生成"""
    global llm_instance

    if not llm_instance:
        return "错误: 模型未加载"

    try:
        with lock:
            response = llm_instance.generate(prompt, max_tokens, temperature)
        return response.strip()
    except Exception as e:
        log(f"生成错误: {e}")
        return f"生成错误: {e}"


def send_with_timeout(conn, data, timeout=180):
    """带超时的发送数据"""
    try:
        conn.setblocking(False)
        total_sent = 0
        data_bytes = data.encode('utf-8')
        
        while total_sent < len(data_bytes):
            ready = select.select([], [conn], [], timeout)
            if not ready[1]:
                raise socket.timeout("发送超时")
            try:
                sent = conn.send(data_bytes[total_sent:])
                total_sent += sent
            except BlockingIOError:
                time.sleep(0.01)
                continue
        return True
    except Exception as e:
        log(f"发送错误: {e}")
        return False
    finally:
        try:
            conn.setblocking(True)
        except:
            pass


def handle_streaming_client(conn):
    """处理流式客户端连接"""
    client_id = id(conn)
    
    try:
        # 接收数据
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

        # 解析请求
        request = json.loads(data.decode('utf-8'))
        prompt = request.get('prompt', '')
        max_tokens = request.get('max_tokens', MAX_NEW_TOKENS)
        temperature = request.get('temperature', 0.7)
        stream_mode = request.get('stream', False)

        # 特殊处理 ping 请求
        if prompt == 'ping':
            result = {'response': 'pong', 'time_ms': 0}
            conn.sendall(json.dumps(result).encode('utf-8'))
            return

        log(f"收到流式请求: {prompt[:50]}...")
        print(f"[Qwen3-LLM流式服务] 收到请求: {prompt[:50]}...", file=sys.stderr)

        if stream_mode:
            # 流式模式：逐个token发送，并在检测到完整句子时立即发送sentence事件
            start_time = time.time()
            
            def token_callback(token_text):
                """每个token的回调"""
                try:
                    chunk = {
                        'token': token_text,
                        'done': False
                    }
                    conn.sendall(json.dumps(chunk, ensure_ascii=False).encode('utf-8') + b'\n')
                except:
                    pass
            
            def sentence_callback(sentence_text):
                """每个完整句子的回调 - 立即发送sentence事件"""
                try:
                    chunk = {
                        'token': '',
                        'sentence': sentence_text,
                        'done': False
                    }
                    conn.sendall(json.dumps(chunk, ensure_ascii=False).encode('utf-8') + b'\n')
                    print(f"[流式服务] 发送句子: {sentence_text}", file=sys.stderr)
                except:
                    pass
            
            # 生成回复（流式）
            response = generate_streaming(
                prompt, 
                max_tokens=max_tokens, 
                temperature=temperature,
                token_callback=token_callback,
                sentence_callback=sentence_callback
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 发送完成标记
            final_chunk = {
                'token': '',
                'response': response,
                'done': True,
                'time_ms': elapsed_ms
            }
            conn.sendall(json.dumps(final_chunk, ensure_ascii=False).encode('utf-8') + b'\n')
            
            log(f"流式生成完成，耗时: {elapsed_ms:.0f}ms")
            
        else:
            # 非流式模式：一次性返回
            start_time = time.time()
            response = generate(prompt, max_tokens, temperature)
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = {
                'response': response,
                'time_ms': elapsed_ms
            }
            
            if not send_with_timeout(conn, json.dumps(result, ensure_ascii=False), timeout=180):
                log("发送响应失败")

    except socket.timeout:
        log("客户端接收超时")
        try:
            error = {'error': '请求接收超时'}
            conn.sendall(json.dumps(error).encode('utf-8'))
        except:
            pass
    except json.JSONDecodeError as e:
        log(f"JSON解析错误: {e}")
        try:
            error = {'error': '无效的JSON请求'}
            conn.sendall(json.dumps(error).encode('utf-8'))
        except:
            pass
    except Exception as e:
        log(f"处理错误: {e}")
        print(f"[Qwen3-LLM流式服务] 处理错误: {e}", file=sys.stderr)
        try:
            error = {'error': str(e)}
            conn.sendall(json.dumps(error).encode('utf-8'))
        except:
            pass
    finally:
        try:
            conn.close()
        except:
            pass


def handle_client(conn):
    """处理标准客户端连接（兼容旧版）"""
    try:
        # 接收数据
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

        # 解析请求
        request = json.loads(data.decode('utf-8'))
        prompt = request.get('prompt', '')
        max_tokens = request.get('max_tokens', MAX_NEW_TOKENS)
        temperature = request.get('temperature', 0.7)

        # 特殊处理 ping 请求
        if prompt == 'ping':
            result = {'response': 'pong', 'time_ms': 0}
            conn.sendall(json.dumps(result).encode('utf-8'))
            return

        log(f"收到请求: {prompt[:50]}...")
        print(f"[Qwen3-LLM流式服务] 收到标准请求: {prompt[:50]}...", file=sys.stderr)

        # 生成回答
        start_time = time.time()
        response = generate(prompt, max_tokens, temperature)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        log(f"生成完成，耗时: {elapsed_ms:.0f}ms")
        print(f"[Qwen3-LLM流式服务] 生成完成，耗时: {elapsed_ms:.0f}ms", file=sys.stderr)

        # 发送响应
        result = {
            'response': response,
            'time_ms': elapsed_ms
        }
        
        if not send_with_timeout(conn, json.dumps(result, ensure_ascii=False), timeout=180):
            log("发送响应失败")

    except socket.timeout:
        log("客户端接收超时")
        try:
            error = {'error': '请求接收超时'}
            conn.sendall(json.dumps(error).encode('utf-8'))
        except:
            pass
    except json.JSONDecodeError as e:
        log(f"JSON解析错误: {e}")
        try:
            error = {'error': '无效的JSON请求'}
            conn.sendall(json.dumps(error).encode('utf-8'))
        except:
            pass
    except Exception as e:
        log(f"处理错误: {e}")
        print(f"[Qwen3-LLM流式服务] 处理错误: {e}", file=sys.stderr)
        try:
            error = {'error': str(e)}
            conn.sendall(json.dumps(error).encode('utf-8'))
        except:
            pass
    finally:
        try:
            conn.close()
        except:
            pass


def signal_handler(signum, frame):
    """信号处理"""
    global service_running
    log("收到停止信号")
    service_running = False


def main():
    """主函数"""
    global service_running

    # 清空日志文件
    open(LOG_FILE, 'w').close()
    log("流式服务启动")

    # 注册信号处理
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # 加载模型
    if not load_model():
        sys.exit(1)

    # 删除旧的 socket
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    # 创建 socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(SOCKET_PATH)
    server.listen(10)
    os.chmod(SOCKET_PATH, 0o666)

    service_running = True

    log(f"流式服务已启动: {SOCKET_PATH}")
    print(f"[Qwen3-LLM流式服务] 服务已启动: {SOCKET_PATH}", file=sys.stderr)
    print(f"[Qwen3-LLM流式服务] 日志文件: {LOG_FILE}", file=sys.stderr)

    try:
        while service_running:
            try:
                server.settimeout(1)
                conn, addr = server.accept()
                thread = threading.Thread(target=handle_streaming_client, args=(conn,))
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
        if llm_instance:
            llm_instance.release()
        log("服务已停止")
        print("[Qwen3-LLM流式服务] 服务已停止", file=sys.stderr)


if __name__ == '__main__':
    main()
