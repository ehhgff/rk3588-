#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-1.7B LLM 客户端
通过 Unix Socket 与服务通信
"""

import socket
import json
import sys

SOCKET_PATH = "/tmp/qwen3_llm.sock"
DEFAULT_TIMEOUT = 180  # 3分钟超时，因为LLM生成较慢

def query_llm(prompt, max_tokens=20, temperature=0.7, timeout=DEFAULT_TIMEOUT):
    """
    查询 LLM 服务

    参数:
        prompt: 输入问题
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        timeout: 超时时间（秒）

    返回:
        dict: {'response': str, 'time_ms': float} 或 {'error': str}
    """
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(SOCKET_PATH)

        # 发送请求
        request = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        sock.sendall(json.dumps(request, ensure_ascii=False).encode('utf-8'))
        sock.shutdown(socket.SHUT_WR)

        # 接收响应
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk

        return json.loads(response.decode('utf-8'))

    except socket.timeout:
        return {'error': '请求超时'}
    except ConnectionRefusedError:
        return {'error': '连接被拒绝，服务可能未启动'}
    except Exception as e:
        return {'error': str(e)}
    finally:
        if sock:
            sock.close()


def interactive_test():
    """交互式测试"""
    print("=" * 50)
    print("Qwen3-1.7B LLM 客户端 (交互模式)")
    print("=" * 50)
    print("输入 'exit' 或 'quit' 退出")
    print()

    while True:
        try:
            prompt = input("用户: ").strip()
            if prompt.lower() in ['exit', 'quit', '退出']:
                break
            if not prompt:
                continue

            print("助手: ", end='', flush=True)

            result = query_llm(prompt)

            if 'error' in result:
                print(f"错误: {result['error']}")
            else:
                print(result['response'])
                print(f"[耗时: {result.get('time_ms', 0):.0f}ms]")

            print()

        except KeyboardInterrupt:
            print("\n退出...")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def batch_test():
    """批量测试"""
    print("=" * 50)
    print("Qwen3-1.7B LLM 客户端 (批量测试模式)")
    print("=" * 50)
    print()

    test_queries = [
        "你好，请介绍一下你自己",
        "什么是人工智能？",
        "1+1等于几？",
        "用一句话解释量子计算",
        "推荐5本科值得读的科幻小说"
    ]

    total_time = 0
    success_count = 0

    for i, query in enumerate(test_queries):
        print(f"【查询 {i+1}/{len(test_queries)}】: {query}")
        print("助手: ", end='', flush=True)

        result = query_llm(query)

        if 'error' in result:
            print(f"错误: {result['error']}")
            print()
        else:
            print(result['response'])
            elapsed = result.get('time_ms', 0)
            print(f"[耗时: {elapsed:.0f}ms]")
            total_time += elapsed
            success_count += 1
            print()

    print("=" * 50)
    print(f"测试完成: {success_count}/{len(test_queries)} 成功")
    if success_count > 0:
        print(f"平均耗时: {total_time/success_count:.0f}ms")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch' or sys.argv[1] == '-b':
            batch_test()
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("用法:")
            print("  python3 qwen3_llm_client.py          交互模式")
            print("  python3 qwen3_llm_client.py --batch  批量测试模式")
            print("  python3 qwen3_llm_client.py --help   显示帮助")
        else:
            # 单次查询
            prompt = ' '.join(sys.argv[1:])
            print("用户:", prompt)
            print("助手: ", end='', flush=True)

            result = query_llm(prompt)

            if 'error' in result:
                print(f"错误: {result['error']}")
            else:
                print(result['response'])
                print(f"[耗时: {result.get('time_ms', 0):.0f}ms]")
    else:
        interactive_test()


if __name__ == '__main__':
    main()
