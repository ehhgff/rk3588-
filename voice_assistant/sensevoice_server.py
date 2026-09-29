#!/usr/bin/env python3
"""
SenseVoice RKNN 常驻服务 (v2 - 支持 ASR+SER)
在 RK3588 上后台运行，避免重复加载模型
"""
import sys
import json
import struct
import socket
import time
import os
import signal
import numpy as np
from rknnlite.api import RKNNLite

MODEL_PATH = "/data/sensevoice/sensevoice_encoder_ctc_100f_fp16.rknn"
SOCKET_PATH = "/tmp/sensevoice_server.sock"
TOKEN_PATH = "/data/sensevoice/tokens.json"
MAX_FRAMES = 100
BLANK_ID = 0
SPECIAL_TOKEN_START = 24992

EMO_MAP = {
    25001: "HAPPY", 25002: "SAD", 25003: "ANGRY",
    25004: "NEUTRAL", 25005: "FEARFUL", 25006: "DISGUSTED",
    25007: "SURPRISED", 25008: "OTHER", 25009: "EMO_UNKNOWN",
}

EMO_ID_MAP = {
    "NEUTRAL": 0, "HAPPY": 1, "SAD": 2, "ANGRY": 3,
    "FEARFUL": 4, "DISGUSTED": 5, "SURPRISED": 6,
    "OTHER": 7, "EMO_UNKNOWN": 7, "UNKNOWN": 7,
}


class SenseVoiceServer:
    def __init__(self, model_path, socket_path, token_path=TOKEN_PATH):
        self.model_path = model_path
        self.socket_path = socket_path
        self.token_path = token_path
        self.rknn = None
        self.server = None
        self.running = False
        self.tokens = None

    def load_tokenizer(self):
        if os.path.exists(self.token_path):
            with open(self.token_path, "r", encoding="utf-8") as f:
                self.tokens = json.load(f)
            print(f"[服务] 加载词表: {len(self.tokens)} tokens")
        else:
            print(f"[服务] 警告: 未找到词表文件 {self.token_path}")

    def decode_text(self, token_ids):
        if self.tokens is None:
            return ""
        text_parts = []
        for tid in token_ids:
            if tid >= SPECIAL_TOKEN_START:
                continue
            if tid < len(self.tokens):
                token = self.tokens[tid]
                if token == "<unk>":
                    continue
                text_parts.append(token)
        text = "".join(text_parts)
        text = text.replace("\u2581", " ")
        return text.strip()

    def load_model(self):
        print(f"[服务] 加载模型: {self.model_path}")
        t0 = time.time()
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError("模型加载失败")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError("运行时初始化失败")
        elapsed = time.time() - t0
        print(f"[服务] 模型加载完成 ({elapsed*1000:.0f}ms)")
        self.load_tokenizer()

    def start(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.load_model()

        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.socket_path)
        self.server.listen(5)
        os.chmod(self.socket_path, 0o777)

        self.running = True
        print(f"[服务] 监听: {self.socket_path}")

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        while self.running:
            try:
                conn, _ = self.server.accept()
                self.handle_client(conn)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[服务] 连接错误: {e}")

        self.cleanup()

    def handle_client(self, conn):
        try:
            header = conn.recv(8)
            if len(header) < 8:
                return

            if header[0:1] == b'{':
                self._handle_json(conn, header)
            else:
                self._handle_binary(conn, header)
        except Exception as e:
            print(f"[服务] 处理请求错误: {e}")
            try:
                err = json.dumps({"error": str(e)}).encode('utf-8')
                conn.sendall(struct.pack('!Q', len(err)) + err)
            except:
                pass
        finally:
            conn.close()

    def _handle_json(self, conn, first_chunk):
        buf = first_chunk
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b'\n' in chunk:
                break
        try:
            req = json.loads(buf.decode('utf-8'))
        except:
            return

        mode = req.get('mode', 'asr')
        feats_list = req.get('features')
        if feats_list is None:
            return

        feats = np.array(feats_list, dtype=np.float32)
        if feats.ndim == 2:
            feats = feats[np.newaxis, :, :]

        result = self._run_inference(feats)

        if mode == 'ser':
            ser_result = {
                'emotion_id': result['emotion_id'],
                'emotion': result['emotion'],
                'probability': result['probability'],
                'text': result['text'],
                'infer_time_ms': result['infer_time_ms'],
            }
            resp = json.dumps(ser_result) + '\n'
        else:
            resp = json.dumps(result) + '\n'

        conn.sendall(resp.encode('utf-8'))

    def _handle_binary(self, conn, header):
        data_len = struct.unpack('!Q', header)[0]
        data = b''
        remaining = data_len
        while remaining > 0:
            chunk = conn.recv(min(remaining, 65536))
            if not chunk:
                break
            data += chunk
            remaining -= len(chunk)

        if len(data) != data_len:
            return

        feats = np.frombuffer(data, dtype=np.float32).reshape(1, -1, 560)
        result = self._run_inference(feats)

        resp = json.dumps(result).encode('utf-8')
        conn.sendall(struct.pack('!Q', len(resp)) + resp)

    def _run_inference(self, feats):
        actual_frames = feats.shape[1]
        pad_feats = np.zeros((1, MAX_FRAMES, 560), dtype=np.float32)
        n = min(actual_frames, MAX_FRAMES)
        pad_feats[:, :n, :] = feats[:, :n, :]

        t0 = time.time()
        outputs = self.rknn.inference(inputs=[pad_feats])
        infer_time = time.time() - t0

        ctc_logits = outputs[0]
        yseq = ctc_logits.argmax(axis=-1)
        yseq = self.unique_consecutive(yseq)
        mask = yseq != BLANK_ID
        token_ids = yseq[mask].tolist()

        emotion = "UNKNOWN"
        emotion_id = 7
        for tid in token_ids:
            if tid in EMO_MAP:
                emotion = EMO_MAP[tid]
                emotion_id = EMO_ID_MAP.get(emotion, 7)
                break

        text = self.decode_text(token_ids)

        return {
            "emotion": emotion,
            "emotion_id": emotion_id,
            "probability": 1.0 if emotion != "UNKNOWN" else 0.0,
            "text": text,
            "token_ids": token_ids[:30],
            "infer_time_ms": round(infer_time * 1000, 1),
            "actual_frames": actual_frames,
        }

    @staticmethod
    def unique_consecutive(arr):
        if len(arr) == 0:
            return arr
        result = [arr[0]]
        for i in range(1, len(arr)):
            if arr[i] != arr[i-1]:
                result.append(arr[i])
        return np.array(result, dtype=arr.dtype)

    def stop(self, signum=None, frame=None):
        print(f"\n[服务] 正在关闭...")
        self.running = False

    def cleanup(self):
        if self.rknn:
            self.rknn.release()
        if self.server:
            self.server.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        print("[服务] 已关闭")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SenseVoice RKNN 常驻服务")
    parser.add_argument("--model", default=MODEL_PATH, help="RKNN 模型路径")
    parser.add_argument("--socket", default=SOCKET_PATH, help="Unix socket 路径")
    args = parser.parse_args()

    server = SenseVoiceServer(args.model, args.socket)
    server.start()


if __name__ == "__main__":
    main()