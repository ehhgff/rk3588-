import socket
import json
import time

# 连接流式服务
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(30)
sock.connect('/tmp/qwen3_llm_streaming.sock')

# 发送请求
request = {
    'prompt': '感冒了怎么办？',
    'max_tokens': 45,
    'temperature': 0.7,
    'stream': True
}

print("发送请求...")
start = time.time()
sock.send(json.dumps(request).encode() + b'\n')

# 接收流式响应
print("\n=== 流式响应 ===")
buffer = b""
full_response = ""
token_count = 0
sentence_count = 0

while True:
    try:
        chunk = sock.recv(4096)
        if not chunk:
            break
        
        buffer += chunk
        
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            if not line:
                continue
            
            try:
                data = json.loads(line.decode('utf-8'))
                
                token = data.get('token', '')
                sentence = data.get('sentence', '')
                is_done = data.get('done', False)
                
                if token:
                    token_count += 1
                    full_response += token
                    print("[Token %d] %s" % (token_count, token), end='', flush=True)
                
                if sentence:
                    sentence_count += 1
                    elapsed = (time.time() - start) * 1000
                    print("\n[句子 %d] (%dms): %s" % (sentence_count, int(elapsed), sentence))
                
                if is_done:
                    final_response = data.get('response', full_response)
                    elapsed = (time.time() - start) * 1000
                    print("\n\n=== 完成 ===")
                    print("总 Token 数: %d" % token_count)
                    print("总句子数: %d" % sentence_count)
                    print("总耗时: %dms" % int(elapsed))
                    print("完整回复: %s" % final_response)
                    break
                    
            except json.JSONDecodeError:
                continue
                
        if is_done:
            break
            
    except socket.timeout:
        print("\n接收超时")
        break
    except Exception as e:
        print("\n异常: %s" % e)
        break

sock.close()
