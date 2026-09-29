#!/usr/bin/env python3
"""Fix thread race condition in record_on_voice_detection():
  - Join reader/stream threads before calling finalize() and destroy()
  - Prevents std::system_error: Invalid argument caused by concurrent
    access to sherpa-onnx ASR engine

Root cause:
  stream_thread calls _asr_engine.process_chunk() in a while loop.
  After recording ends, the main thread calls finalize() + destroy()
  on the same engine. If the stream thread is still running (not joined),
  it may call process_chunk() concurrently, causing C++ ABI errors.
"""
import subprocess
import sys

FILES = [
    '/userdata/voice_assistant/streaming_vad.py',
    '/userdata/voice_assistant/core/streaming_vad.py',
]


def adb_shell(cmd):
    return subprocess.run(['adb', 'shell', cmd], capture_output=True, text=True)


def adb_pull(remote, local):
    return subprocess.run(['adb', 'pull', remote, local], capture_output=True)


def adb_push(local, remote):
    return subprocess.run(['adb', 'push', local, remote], capture_output=True)


for FILE in FILES:
    print(f"\n{'='*60}")
    print(f"File: {FILE}")
    print('='*60)

    # Pull
    result = adb_pull(FILE, '/tmp/fix_thread_tmp.py')
    if result.returncode != 0:
        print(f"  ❌ adb pull failed")
        continue

    with open('/tmp/fix_thread_tmp.py', 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # --- Fix: Add thread joins before finalize/destroy ---
    old = """    # 返回录音帧和流式识别结果
    # 结束流式识别，获取最终结果
    if _asr_engine is not None:"""

    new = """    # 返回录音帧和流式识别结果
    # 等待所有线程安全退出，避免竞争条件
    reader_thread.join(timeout=2)
    stream_thread.join(timeout=2)

    # 结束流式识别，获取最终结果
    if _asr_engine is not None:"""

    if old in content:
        content = content.replace(old, new, 1)
        print(f"  ✅ Added thread joins before finalize")
    else:
        print(f"  ⚠️  Target pattern not found, trying alternative...")
        # Try to find the code block without the comment
        old2 = """    # 结束流式识别，获取最终结果
    if _asr_engine is not None:"""
        new2 = """    # 等待所有线程安全退出，避免竞争条件
    reader_thread.join(timeout=2)
    stream_thread.join(timeout=2)

    # 结束流式识别，获取最终结果
    if _asr_engine is not None:"""
        if old2 in content:
            content = content.replace(old2, new2, 1)
            print(f"  ✅ Added thread joins before finalize (alt pattern)")
        else:
            print(f"  ❌ Could not find target code")
            # Debug
            idx = content.find('finalize')
            if idx >= 0:
                print(f"  Found 'finalize' at char {idx}, context:")
                print(f"  {repr(content[idx-100:idx+100])}")
            continue

    if content != original:
        with open('/tmp/fix_thread_tmp.py', 'w', encoding='utf-8') as f:
            f.write(content)

        result = adb_push('/tmp/fix_thread_tmp.py', FILE)
        if result.returncode == 0:
            print(f"  ✅ Saved to board")
        else:
            print(f"  ❌ adb push failed")
    else:
        print(f"  ⚠️  No changes made")

# --- Verification ---
print(f"\n{'='*60}")
print(f"Verification")
print('='*60)

for FILE in FILES:
    name = FILE.rsplit('/', 1)[-1]
    result = adb_shell(
        f"grep -n 'reader_thread.join\\|stream_thread.join\\|finalize\\|destroy' {FILE} | head -10"
    )
    print(f"\n  {name}:")
    for line in result.stdout.strip().split('\n'):
        print(f"    {line}")

print(f"\nDone! Run: adb shell /userdata/voice_assistant/run.sh")
