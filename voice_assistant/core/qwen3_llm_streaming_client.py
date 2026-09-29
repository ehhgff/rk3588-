#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-0.6B LLM 流式客户端
支持接收 token-by-token 流式输出
"""

import socket
import json
import time

DEFAULT_TIMEOUT = 60
LLM_SERVICE_SOCK = "/tmp/qwen3_llm_streaming.sock"


def query_llm_streaming(prompt, max_tokens=45, temperature=0.7, timeout=DEFAULT_TIMEOUT):
    """
    流式查询 LLM 服务
    
    Args:
        prompt: 输入提示词
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        timeout: 超时时间（秒）
    
    Returns:
        生成器，逐个产生 (token, is_done, full_response)
    """
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(LLM_SERVICE_SOCK)

        request = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': True
        }
        sock.send(json.dumps(request).encode() + b'\n')

        # 接收流式响应
        buffer = b""
        full_response = ""
        
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                
                buffer += chunk
                
                # 处理完整行
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line.decode('utf-8'))
                        
                        if 'error' in data:
                            yield None, True, f"错误: {data['error']}"
                            return
                        
                        token = data.get('token', '')
                        is_done = data.get('done', False)
                        
                        if token:
                            full_response += token
                            yield token, False, full_response
                        
                        if is_done:
                            final_response = data.get('response', full_response)
                            yield None, True, final_response
                            return
                            
                    except json.JSONDecodeError:
                        continue
                        
            except socket.timeout:
                yield None, True, f"错误: 接收超时"
                return
            except Exception as e:
                yield None, True, f"错误: {e}"
                return

    except Exception as e:
        yield None, True, f"错误: {e}"
    finally:
        if sock:
            try:
                sock.close()
            except:
                pass


def query_llm(prompt, max_tokens=45, temperature=0.7, timeout=DEFAULT_TIMEOUT):
    """
    标准同步查询 LLM 服务（兼容旧版）
    
    Args:
        prompt: 输入提示词
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        timeout: 超时时间（秒）
    
    Returns:
        回复文本字符串，或 None（失败时）
    """
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(LLM_SERVICE_SOCK)

        request = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': False
        }
        sock.send(json.dumps(request).encode() + b'\n')

        # 接收响应
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in chunk:
                break

        if not data:
            return None

        response = json.loads(data.decode('utf-8'))
        
        if 'error' in response:
            print(f"LLM 服务错误: {response['error']}")
            return None
        
        return response.get('response', '')

    except Exception as e:
        print(f"LLM 查询失败: {e}")
        return None
    finally:
        if sock:
            try:
                sock.close()
            except:
                pass


if __name__ == '__main__':
    # 测试流式输出
    prompt = "感冒了怎么办？"
    
    print("测试流式输出:")
    print("=" * 50)
    
    for token, is_done, full_response in query_llm_streaming(prompt, max_tokens=45):
        if is_done:
            print(f"\n完整回复: {full_response}")
        else:
            print(token, end='', flush=True)
    
    print("\n" + "=" * 50)
