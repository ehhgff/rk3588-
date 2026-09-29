import socket
import json
import struct
import numpy as np
import sys
import os
import time

sys.path.insert(0, '/tmp')
import sensevoice_feat_extract as sfe

SER_SOCK = '/tmp/sensevoice_server.sock'
SER_TIMEOUT = 10.0

EMOTION_LABELS = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'angry',
    4: 'fearful',
    5: 'disgusted',
    6: 'surprised',
    7: 'other'
}

EMOTION_CN = {
    'neutral': '中性',
    'happy': '开心',
    'sad': '悲伤',
    'angry': '生气',
    'fearful': '恐惧',
    'disgusted': '厌恶',
    'surprised': '惊讶',
    'other': '其他'
}


def ser_from_wav(wav_path, max_frames=100):
    feats = sfe.extract_mel_lfr(wav_path, max_frames)
    return ser_from_feats(feats)


def ser_from_audio(audio, max_frames=100):
    feats = sfe.extract_mel_lfr_from_audio(audio, max_frames)
    return ser_from_feats(feats)


def ser_from_feats(feats):
    if feats.ndim == 2:
        feats = feats[np.newaxis, :, :]

    data = feats.astype(np.float32).tobytes()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(SER_TIMEOUT)
    try:
        sock.connect(SER_SOCK)
        sock.sendall(struct.pack('!Q', len(data)) + data)

        header = sock.recv(8)
        if len(header) < 8:
            sock.close()
            return {'error': '服务端响应不完整'}

        resp_len = struct.unpack('!Q', header)[0]
        resp = b''
        remaining = resp_len
        while remaining > 0:
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                break
            resp += chunk
            remaining -= len(chunk)

        sock.close()
        return json.loads(resp.decode('utf-8'))
    except socket.timeout:
        return {'error': f'SER service timeout ({SER_TIMEOUT}s)'}
    except Exception as e:
        return {'error': str(e)}


def format_emotion(result):
    if 'error' in result:
        return f'[SER错误] {result["error"]}'
    emotion_id = result.get('emotion_id', -1)
    emotion_en = result.get('emotion', 'UNKNOWN').lower()
    emotion_cn = EMOTION_CN.get(emotion_en, emotion_en)
    prob = result.get('probability', 0)
    text = result.get('text', '')
    parts = []
    parts.append(f'情感: {emotion_cn} ({emotion_en})')
    parts.append(f'置信度: {prob:.2%}')
    if text:
        parts.append(f'识别文本: {text}')
    return ' | '.join(parts)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python3 sensevoice_ser_client.py <wav_path> [max_frames]')
        print('示例: python3 sensevoice_ser_client.py /data/test.wav 100')
        sys.exit(1)
    wav_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    t0 = time.time()
    result = ser_from_wav(wav_path, max_frames)
    elapsed = time.time() - t0
    print(f'耗时: {elapsed*1000:.0f}ms')
    print(format_emotion(result))