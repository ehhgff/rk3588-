#!/usr/bin/env python3
"""Fix llm_generate_streaming and _llm_worker exception handling.

Changes:
  1. Move response_parts = [] before try block so it's defined on all paths
  2. Add try/except to _llm_worker so pipeline doesn't hang on LLM failure
"""
import subprocess
import sys

FILE = '/userdata/voice_assistant/streaming_vad.py'


def adb_shell(cmd):
    return subprocess.run(['adb', 'shell', cmd], capture_output=True, text=True)


def adb_pull(remote, local):
    return subprocess.run(['adb', 'pull', remote, local], capture_output=True)


def adb_push(local, remote):
    return subprocess.run(['adb', 'push', local, remote], capture_output=True)


# Pull
result = adb_pull(FILE, '/tmp/fix_llm_exc_tmp.py')
if result.returncode != 0:
    print("❌ adb pull failed")
    sys.exit(1)

with open('/tmp/fix_llm_exc_tmp.py', 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# --- Fix 1: Move response_parts before try block ---
old1 = '''    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect("/tmp/qwen3_llm_streaming.sock")

        request = {
            'prompt': prompt,
            'max_tokens': 60,
            'temperature': 0.7,
            'stream': True
        }
        sock.sendall(json.dumps(request, ensure_ascii=False).encode('utf-8') + b'\\n')

        response_parts = []'''

new1 = '''    response_parts = []

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect("/tmp/qwen3_llm_streaming.sock")

        request = {
            'prompt': prompt,
            'max_tokens': 60,
            'temperature': 0.7,
            'stream': True
        }
        sock.sendall(json.dumps(request, ensure_ascii=False).encode('utf-8') + b'\\n')'''

if old1 in content:
    content = content.replace(old1, new1, 1)
    print("✅ Fix 1: response_parts moved before try block")
else:
    print("⚠️  Fix 1: pattern not found, trying alt...")
    # Try without \n escaping
    old1b = '''        sock.sendall(json.dumps(request, ensure_ascii=False).encode('utf-8') + b'\\n')

        response_parts = []'''
    new1b = '''        sock.sendall(json.dumps(request, ensure_ascii=False).encode('utf-8') + b'\\n')'''
    if old1b in content:
        content = content.replace(old1b, new1b, 1)
        # Now insert response_parts before try
        old_insert = '''    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)'''
        new_insert = '''    response_parts = []

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)'''
        if old_insert in content:
            content = content.replace(old_insert, new_insert, 1)
            print("✅ Fix 1: response_parts moved (alt pattern)")
        else:
            print("❌ Fix 1: cannot find try block")
    else:
        print("❌ Fix 1: pattern not found at all")


# --- Fix 2: Add exception handling to _llm_worker ---
old2 = '''        def _llm_worker():
            """后台线程：流式 LLM 生成"""
            reply = llm_generate_streaming(prompt, on_sentence=on_sentence)
            reply_holder.append(reply)
            tts_queue.put(None)  # LLM 完成哨兵'''

new2 = '''        def _llm_worker():
            """后台线程：流式 LLM 生成"""
            try:
                reply = llm_generate_streaming(prompt, on_sentence=on_sentence)
                reply_holder.append(reply)
            except Exception as e:
                log_warn(f"_llm_worker 异常: {e}")
            finally:
                tts_queue.put(None)  # 确保即使异常也通知 TTS 线程退出'''

if old2 in content:
    content = content.replace(old2, new2, 1)
    print("✅ Fix 2: _llm_worker exception handling added")
else:
    print("⚠️  Fix 2: pattern not found")
    idx = content.find('_llm_worker')
    if idx >= 0:
        print(f"  Found at char {idx}, context: {repr(content[idx:idx+200])}")


if content != original:
    with open('/tmp/fix_llm_exc_tmp.py', 'w', encoding='utf-8') as f:
        f.write(content)

    result = adb_push('/tmp/fix_llm_exc_tmp.py', FILE)
    if result.returncode == 0:
        print(f"\n✅ Saved! Deployed to {FILE}")
    else:
        print(f"\n❌ adb push failed")
else:
    print(f"\n⚠️  No changes made")

# Verify
print(f"\n{'='*50}")
print("Verification:")
result = adb_shell("grep -n 'response_parts\\|_llm_worker' /userdata/voice_assistant/streaming_vad.py | head -10")
for line in result.stdout.strip().split('\n'):
    print(f"  {line}")
