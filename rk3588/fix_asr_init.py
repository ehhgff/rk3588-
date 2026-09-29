#!/usr/bin/env python3
"""Fix ASR initialization:
  1. Remove ctypes preload from streaming_vad.py files
  2. Push run.sh wrapper script (sets LD_LIBRARY_PATH at process startup)
  3. Verify the fix

Background:
  - sherpa-onnx bundled libstdc++.so.6 has GLIBCXX up to 3.4.33
  - System /lib/libstdc++.so.6 only has GLIBCXX up to 3.4.28
  - ctypes RTLD_GLOBAL preload causes C++ ABI conflicts → segfault
  - LD_LIBRARY_PATH must be set BEFORE Python starts (dynamic linker reads it at startup)
  - os.environ['LD_LIBRARY_PATH'] at runtime does NOT affect dlopen!
  - sherpa_onnx_stream.py's _get_lib() already loads by absolute path → no change needed

Usage:
  python3 fix_asr_init.py           # Fix files + push run.sh
  python3 fix_asr_init.py --check   # Only check current state
"""
import subprocess
import sys
import os

FILES = [
    '/userdata/voice_assistant/streaming_vad.py',
    '/userdata/voice_assistant/core/streaming_vad.py',
]
RUN_SH_SRC = os.path.join(os.path.dirname(__file__), 'run.sh')
RUN_SH_DST = '/userdata/voice_assistant/run.sh'

DO_FIX = '--check' not in sys.argv


def adb_run(cmd, **kwargs):
    """Run adb command and return CompletedProcess."""
    full_cmd = ['adb'] + cmd
    return subprocess.run(full_cmd, capture_output=True, **kwargs)


def adb_push(local, remote):
    """Push local file to board via adb."""
    result = adb_run(['push', local, remote])
    return result.returncode == 0


def adb_pull(remote, local):
    """Pull file from board via adb."""
    result = adb_run(['pull', remote, local])
    return result.returncode == 0


def adb_shell(command):
    """Run shell command on board."""
    result = adb_run(['shell', command], text=True)
    return result.stdout.strip()


# ─── Step 1: Remove ctypes preload from streaming_vad.py files ───
def remove_ctypes_preload():
    """Remove ctypes CDLL + RTLD_GLOBAL preload block from both files."""
    old = '''import subprocess
import ctypes

# 预加载 sherpa-onnx 自带的 libstdc++（含 GLIBCXX_3.4.32）
_libstdcxx_path = '/userdata/sherpa-onnx/install/lib/libstdc++.so.6'
if os.path.exists(_libstdcxx_path):
    ctypes.CDLL(_libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
from sherpa_onnx_stream import SherpaOnnxStreamingASR'''

    new = '''import subprocess
from sherpa_onnx_stream import SherpaOnnxStreamingASR'''

    for FILE in FILES:
        print(f"\n{'='*60}")
        print(f"File: {FILE}")
        print('='*60)

        if not DO_FIX:
            # Check only
            result = adb_shell(f'grep -c "ctypes.CDLL" {FILE} 2>/dev/null || echo 0')
            if result and result != '0':
                print(f"  ⚠️  ctypes preload STILL PRESENT")
            else:
                print(f"  ✅ ctypes preload already removed")
            continue

        if not adb_pull(FILE, '/tmp/fix_asr_tmp.py'):
            print(f"  ❌ adb pull failed")
            continue

        with open('/tmp/fix_asr_tmp.py', 'r') as f:
            content = f.read()

        if old in content:
            content = content.replace(old, new, 1)
            with open('/tmp/fix_asr_tmp.py', 'w') as f:
                f.write(content)
            if adb_push('/tmp/fix_asr_tmp.py', FILE):
                print(f"  ✅ Removed ctypes preload")
            else:
                print(f"  ❌ adb push failed")
        elif 'ctypes.CDLL' in content:
            print(f"  ⚠️  ctypes preload found but pattern mismatch")
            print(f"  Running grep for diagnosis...")
            for line in adb_shell(f'grep -n "ctypes\\|RTLD_GLOBAL\\|libstdcxx" {FILE}').split('\n'):
                print(f"    {line}")
        else:
            print(f"  ✅ Already clean (no ctypes preload)")


# ─── Step 2: Push run.sh wrapper script ───
def push_run_sh():
    """Push the run.sh wrapper script to the board."""
    if not DO_FIX:
        return

    print(f"\n{'='*60}")
    print(f"Push: {RUN_SH_DST}")
    print('='*60)

    if not os.path.exists(RUN_SH_SRC):
        print(f"  ❌ run.sh not found at {RUN_SH_SRC}")
        return

    if adb_push(RUN_SH_SRC, RUN_SH_DST):
        print(f"  ✅ Pushed run.sh")
        adb_shell(f'chmod +x {RUN_SH_DST}')
        print(f"  ✅ Set executable")
    else:
        print(f"  ❌ adb push failed")


# ─── Step 3: Verification ───
def verify():
    """Verify all fixes are applied."""
    print(f"\n{'='*60}")
    print(f"Verification")
    print('='*60)

    # Check ctypes preload removed
    for FILE in FILES:
        result = adb_shell(f'grep -n "ctypes.CDLL\\|RTLD_GLOBAL\\|libstdcxx" {FILE} | head -5')
        if result:
            print(f"  ❌ {FILE}: ctypes preload STILL PRESENT")
            for line in result.split('\n'):
                print(f"       {line}")
        else:
            name = FILE.rsplit('/', 1)[-1]
            print(f"  ✅ {name}: ctypes preload removed OK")

    # Check run.sh
    result = adb_shell(f'test -f {RUN_SH_DST} && echo "exists" || echo "missing"')
    if result == 'exists':
        print(f"  ✅ run.sh: exists on board")
    else:
        print(f"  ❌ run.sh: NOT FOUND on board")

    # Check sherpa_onnx_stream.py _get_lib() — should load by absolute path
    result = adb_shell(
        r"grep -n 'os.path.exists\|LoadLibrary' /userdata/voice_assistant/sherpa_onnx_stream.py | head -5"
    )
    print(f"  sherpa_onnx_stream.py _get_lib():")
    for line in result.split('\n'):
        print(f"    {line}")


if __name__ == '__main__':
    if not DO_FIX:
        print("Check mode: verifying current state only\n")

    remove_ctypes_preload()
    push_run_sh()
    verify()

    if DO_FIX:
        print(f"\n{'='*60}")
        print(f"All done! Run the pipeline with:")
        print(f"  adb shell /userdata/voice_assistant/run.sh")
        print(f"{'='*60}")
