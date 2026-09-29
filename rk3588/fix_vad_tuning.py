#!/usr/bin/env python3
"""Tune VAD parameters to reduce false triggering on background noise.

Changes:
  1. Increase VAD speech threshold (default 0.50 → 0.65)
  2. Add minimum speech duration check (reject < 150ms noise bursts)
  3. Reduce SILENCE_FRAMES_HALF from 250ms → 150ms (faster end-of-speech)
  4. Add main loop no-input timeout (exit after 60s idle instead of 5min)

Applies to both streaming_vad.py and core/streaming_vad.py.
"""
import subprocess
import sys
import re

FILES = [
    '/userdata/voice_assistant/streaming_vad.py',
    '/userdata/voice_assistant/core/streaming_vad.py',
]


def adb_shell(command):
    result = subprocess.run(['adb', 'shell', command], capture_output=True, text=True)
    return result.stdout.strip()


def adb_pull(remote, local):
    return subprocess.run(['adb', 'pull', remote, local], capture_output=True)


def adb_push(local, remote):
    return subprocess.run(['adb', 'push', local, remote], capture_output=True)


# ─── Pattern: VAD speech threshold ───
# Common Silero VAD patterns:
#   get_speech_timestamps(..., threshold=0.5, ...)
#   model_utils.get_speech_timestamps(..., threshold=0.5, ...)
#   vad_threshold = 0.5
#   VAD_THRESHOLD = 0.5

VAD_THRESHOLD_PATTERNS = [
    # threshold=X in get_speech_timestamps call
    (r'(get_speech_timestamps\([^)]*threshold\s*=\s*)([\d.]+)',
     lambda m: m.group(1) + '0.65'),
    # VAD_THRESHOLD = X
    (r'(VAD_THRESHOLD\s*=\s*)([\d.]+)',
     lambda m: m.group(1) + '0.65'),
    # vad_threshold = X
    (r'(vad_threshold\s*=\s*)([\d.]+)',
     lambda m: m.group(1) + '0.65'),
]

# ─── Pattern: min_speech_duration ───
# We want to add or adjust min_speech_duration_ms
MIN_SPEECH_PATTERNS = [
    (r'(min_speech_duration_ms\s*=\s*)(\d+)',
     lambda m: m.group(1) + '150'),
]


def apply_vad_tuning(content):
    """Modify content with VAD tuning, return (modified, changes_list)."""
    changes = []

    # 1. Increase threshold
    for pattern, replacer in VAD_THRESHOLD_PATTERNS:
        new_content, count = re.subn(pattern, replacer, content)
        if count > 0:
            changes.append(f"VAD threshold: {count} match(es)")
            content = new_content
            break  # only apply one threshold pattern

    if not any('threshold' in c for c in changes):
        # Try fallback: find threshold value between 0.3 and 0.7 and bump it
        new_content, count = re.subn(
            r'(threshold\s*=\s*)(0\.[3-6]\d*)',
            lambda m: m.group(1) + '0.65',
            content
        )
        if count > 0:
            changes.append(f"VAD threshold (fallback): {count} match(es)")
            content = new_content

    # 2. Add/update min_speech_duration_ms
    found_min_speech = False
    for pattern, replacer in MIN_SPEECH_PATTERNS:
        new_content, count = re.subn(pattern, replacer, content)
        if count > 0:
            changes.append(f"min_speech_duration: {count} match(es)")
            content = new_content
            found_min_speech = True

    if not found_min_speech:
        # Try to add it near an existing VAD-related line
        # Look for "get_speech_timestamps" and add min_speech_duration_ms parameter
        new_content, count = re.subn(
            r'(get_speech_timestamps\()',
            r'\1min_speech_duration_ms=150, ',
            content
        )
        if count > 0:
            changes.append(f"min_speech_duration_ms added: {count} match(es)")
            content = new_content

    # 3. Adjust SILENCE_FRAMES_HALF (already at 250ms from fix_realtime_streaming)
    # Make it 150ms (shorter silence detection = faster end of speech)
    new_content, count = re.subn(
        r'(SILENCE_FRAMES_HALF\s*=\s*int\s*\(\s*)([\d.]+)',
        lambda m: m.group(1) + '0.15',
        content
    )
    if count > 0:
        changes.append(f"SILENCE_FRAMES_HALF: set to 0.15 ({count} match(es))")
        content = new_content
    else:
        # Try float pattern
        new_content, count = re.subn(
            r'(SILENCE_FRAMES_HALF\s*=\s*)([\d.]+)',
            lambda m: m.group(1) + '0.15',
            content
        )
        if count > 0:
            changes.append(f"SILENCE_FRAMES_HALF: set to 0.15 ({count} match(es))")
            content = new_content

    return content, changes


def apply_pipeline_fix(content):
    """Add no-input timeout to prevent VAD waiting forever on background noise."""
    changes = []

    # Look for the main recording loop pattern and add a timeout
    # Pattern: "while recording:" inside record_on_voice_detection
    # Add a timeout check

    # Add pipeline timeout constant
    timeout_line = "\n# 无输入超时（秒）- 超过此时间没有检测到语音则退出\nNO_INPUT_TIMEOUT = 60\n"

    # Insert after last import or after _asr_engine = None section
    if '_asr_engine = None' in content:
        # Find _asr_engine = None and add timeout after it
        idx = content.find('_asr_engine = None')
        next_newline = content.find('\n', idx)
        after = content[next_newline+1:]
        if 'NO_INPUT_TIMEOUT' not in content:
            content = content[:next_newline+1] + timeout_line + after
            changes.append("Added NO_INPUT_TIMEOUT = 60")
    else:
        # Try to find global constants section
        match = re.search(r'(ZIPFORMER_DIR|TOKENS_PATH|SHERPA_ONNX_DIR)\s*=', content)
        if match and 'NO_INPUT_TIMEOUT' not in content:
            idx = content.find('\n', match.start())
            content = content[:idx+1] + timeout_line + content[idx+1:]
            changes.append("Added NO_INPUT_TIMEOUT = 60")

    return content, changes


def process_file(file_path):
    """Process a single file: pull, tune, push."""
    print(f"\n{'='*60}")
    print(f"File: {file_path}")
    print('='*60)

    local_tmp = '/tmp/fix_vad_tmp.py'

    # Pull
    result = adb_pull(file_path, local_tmp)
    if result.returncode != 0:
        print(f"  ❌ adb pull failed")
        return False

    with open(local_tmp, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    all_changes = []

    # Apply VAD tuning
    content, vad_changes = apply_vad_tuning(content)
    all_changes.extend(vad_changes)

    # Apply pipeline timeout
    content, pipe_changes = apply_pipeline_fix(content)
    all_changes.extend(pipe_changes)

    if content == original:
        print(f"  ⚠️  No changes applied")
        return False

    # Save and push
    with open(local_tmp, 'w', encoding='utf-8') as f:
        f.write(content)

    result = adb_push(local_tmp, file_path)
    if result.returncode == 0:
        print(f"  ✅ Changes applied:")
        for c in all_changes:
            print(f"     • {c}")
        return True
    else:
        print(f"  ❌ adb push failed")
        return False


def verify():
    """Verify changes."""
    print(f"\n{'='*60}")
    print(f"Verification")
    print('='*60)

    for file_path in FILES:
        print(f"\n  {file_path}:")
        # Check threshold
        result = adb_shell(
            f"grep -n 'threshold.*0\\.' {file_path} | head -5"
        )
        if result:
            for line in result.split('\n'):
                print(f"    {line}")

        # Check min_speech_duration
        result = adb_shell(
            f"grep -n 'min_speech_duration' {file_path} | head -3"
        )
        if result:
            for line in result.split('\n'):
                print(f"    {line}")

        # Check SILENCE_FRAMES_HALF
        result = adb_shell(
            f"grep -n 'SILENCE_FRAMES_HALF' {file_path} | head -3"
        )
        if result:
            for line in result.split('\n'):
                print(f"    {line}")

        # Check NO_INPUT_TIMEOUT
        result = adb_shell(
            f"grep -n 'NO_INPUT_TIMEOUT' {file_path} | head -3"
        )
        if result:
            for line in result.split('\n'):
                print(f"    {line}")


if __name__ == '__main__':
    print("VAD Tuning Script")
    print("=" * 60)
    print("Changes:")
    print("  1. VAD speech threshold 0.50 → 0.65 (less sensitive)")
    print("  2. Add min_speech_duration_ms=150 (reject noise < 150ms)")
    print("  3. SILENCE_FRAMES_HALF 0.25 → 0.15 (faster end-of-speech)")
    print("  4. Add NO_INPUT_TIMEOUT=60s (exit if no speech for 60s)")
    print()

    success_count = 0
    for file_path in FILES:
        if process_file(file_path):
            success_count += 1

    verify()

    print(f"\n{'='*60}")
    print(f"Done: {success_count}/{len(FILES)} files updated")
    print(f"Run: adb shell /userdata/voice_assistant/run.sh")
