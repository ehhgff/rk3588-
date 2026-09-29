#!/usr/bin/env python3
"""Matcha-TTS 常驻服务（ONNX Runtime Python 版）
- 声学模型 + 声码器一次性加载到内存
- 每次合成仅需推理时间（~0.7秒）
- 通过 Unix Socket 提供服务
- 多音字消歧（poly lexicon 最长词匹配）
- RMS 归一化 + 头部静音切除
"""
import os
import sys
import json
import time
import socket
import signal
import re
import threading
import struct
import io
import wave as wave_module

import numpy as np
import onnxruntime as ort

SOCKET_PATH = "/tmp/matcha_tts.sock"
PID_FILE = "/tmp/matcha_tts.pid"

MATCHA_DIR = "/data/matcha_tts"
ACOUSTIC_MODEL = f"{MATCHA_DIR}/model-steps-3.onnx"
VOCODER_MODEL = f"{MATCHA_DIR}/matcha_vocoder.onnx"
TOKENS_FILE = f"{MATCHA_DIR}/tokens.txt"
LEXICON_FILE = f"{MATCHA_DIR}/lexicon.txt"
# Poly lexicon: 跟随主 lexicon 的 symlink 指向目录
POLY_LEXICON_FILE = os.path.join(os.path.dirname(os.path.realpath(LEXICON_FILE)), "lexicon_poly.txt")

# RMS 目标响度 (归一化目标)
TARGET_RMS = 0.12
# 头部静音检测阈值
TRIM_THRESHOLD = 0.005
# 淡入时长 (秒)
FADE_IN_SEC = 0.005

# ── 中文标点 → ASCII 映射 ──
CN_PUNCT_MAP = str.maketrans({
    "，": ",", "。": ".", "？": "?", "！": "!",
    "：": ":", "；": ";", "、": ",",
    "（": "(", "）": ")", "【": "[", "】": "]",
    "—": "-", "…": "...",
})

# 多音词最大长度（字符数）
MAX_POLY_WORD_LEN = 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tokens(filename):
    """加载 tokens.txt → {token: id}"""
    ans = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 1:
                ans[" "] = int(fields[0])
            else:
                ans[fields[0]] = int(fields[1])
    return ans


def load_lexicon(filename, token2id):
    """加载 lexicon.txt → {char/word: [id1, id2, ...]}"""
    ans = {}
    unk_id = token2id.get("_", 1)
    with open(filename, encoding="utf-8") as f:
        for line in f:
            # 跳过空行和损坏行（含非音素内容）
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            word = fields[0]
            tokens = fields[1:]
            # 检查所有音素是否都是合法 token（跳过损坏行）
            valid = True
            ids = []
            for t in tokens:
                if t in token2id:
                    ids.append(token2id[t])
                else:
                    valid = False
                    break
            if valid and len(ids) == len(word):
                ans[word] = ids
    return ans


class MatchaTTSInference:
    """Matcha-TTS 推理引擎（模型常驻内存）"""

    def __init__(self):
        self.am_session = None
        self.vocoder_session = None
        self.token2id = None
        self.id2token = None
        self.lexicon = None        # 主词典（含单字 + 多字词）
        self.poly_lexicon = None   # 多音字词典（优先匹配）
        self.sample_rate = 22050
        self._loaded = False
        self._istft_n_fft = 1024
        self._istft_hop_length = 256
        self._istft_win_length = 1024

    def load(self):
        """加载所有模型和词典（启动时调用一次）"""
        log("加载 tokens...")
        self.token2id = load_tokens(TOKENS_FILE)
        self.id2token = {i: t for t, i in self.token2id.items()}
        log(f"  {len(self.token2id)} tokens")

        log("加载 lexicon...")
        self.lexicon = load_lexicon(LEXICON_FILE, self.token2id)
        log(f"  {len(self.lexicon)} 词条")

        # 加载多音字词典（如果存在）
        self.poly_lexicon = {}
        if os.path.exists(POLY_LEXICON_FILE):
            self.poly_lexicon = load_lexicon(POLY_LEXICON_FILE, self.token2id)
            log(f"  多音字词典: {len(self.poly_lexicon)} 词条")
        else:
            log(f"  多音字词典不存在: {POLY_LEXICON_FILE}")

        log("加载声学模型...")
        t0 = time.time()
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 2
        self.am_session = ort.InferenceSession(
            ACOUSTIC_MODEL,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.am_session.get_modelmeta().custom_metadata_map
        self.sample_rate = int(meta.get("sample_rate", 22050))
        log(f"  完成 ({(time.time()-t0)*1000:.0f}ms, {self.sample_rate}Hz)")

        log("加载声码器...")
        t0 = time.time()
        self.vocoder_session = ort.InferenceSession(
            VOCODER_MODEL,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        vmeta = self.vocoder_session.get_modelmeta().custom_metadata_map
        self._istft_n_fft = int(vmeta.get("n_fft", self._istft_n_fft))
        self._istft_hop_length = int(vmeta.get("hop_length", self._istft_hop_length))
        self._istft_win_length = int(vmeta.get("win_length", self._istft_win_length))
        log(f"  完成 ({(time.time()-t0)*1000:.0f}ms)")
        log(f"  ISTFT: n_fft={self._istft_n_fft}, hop={self._istft_hop_length}, win={self._istft_win_length}")

        self._loaded = True
        log("Matcha-TTS 模型就绪")

    def text_to_ids(self, text):
        """文本 → token ID 序列（含多音字消歧）

        流程:
          1. 中文标点 → ASCII 标点
          2. 正则分词: 中文段 / 英文数字 / 单个标点
          3. 中文段: 最长词匹配 poly → 逐字回退 lexicon
          4. 英文数字: 逐字母查 token2id
        """
        # 1. 中文标点映射为 ASCII
        text = text.translate(CN_PUNCT_MAP)

        ids = []
        pattern = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+|.")

        for match in pattern.finditer(text):
            segment = match.group()

            # 完整匹配（标点、特殊 token）
            if segment in self.token2id:
                ids.append(self.token2id[segment])
                continue

            # 中文段：poly 最长词匹配 → 逐字回退
            if re.match(r"[\u4e00-\u9fff]+", segment):
                i = 0
                while i < len(segment):
                    matched = False
                    # poly lexicon 最长匹配
                    if self.poly_lexicon:
                        max_wlen = min(MAX_POLY_WORD_LEN, len(segment) - i)
                        for wlen in range(max_wlen, 1, -1):
                            word = segment[i:i + wlen]
                            if word in self.poly_lexicon:
                                ids.extend(self.poly_lexicon[word])
                                i += wlen
                                matched = True
                                break
                    if not matched:
                        # 逐字：先查主 lexicon（可能有多字词条，但只取当前字）
                        ch = segment[i]
                        if ch in self.lexicon:
                            ids.extend(self.lexicon[ch])
                        else:
                            log(f"  忽略未知字符: U+{ord(ch):04X} '{ch}'")
                        i += 1
                continue

            # 英文/数字：逐字母
            if re.match(r"[a-zA-Z0-9]+", segment):
                for ch in segment.lower():
                    if ch in self.token2id:
                        ids.append(self.token2id[ch])
                continue

            # 其他（漏网标点等）
            if segment in self.token2id:
                ids.append(self.token2id[segment])

        return ids

    def _istft(self, Z):
        """Kaldi-native-fbank 风格的 ISTFT

        与 kaldi-native-fbank 的 IStft::Compute 实现一致：
        - 使用 periodic Hann 窗口
        - center=True，裁剪 n_fft//2 首尾填充
        - 输出长度 = n_frames * hop_length
        - 参数从 Vocos 模型 metadata 读取

        Args:
            Z: 复数频谱, shape (n_fft//2+1, n_frames)
        """
        n_fft = self._istft_n_fft
        hop_length = self._istft_hop_length
        win_length = self._istft_win_length
        n_frames = Z.shape[1]

        a = 2 * np.pi / win_length
        w = 0.5 - 0.5 * np.cos(a * np.arange(win_length))

        num_samples = n_fft + (n_frames - 1) * hop_length
        samples = np.zeros(num_samples, dtype=np.float64)
        denom = np.zeros(num_samples, dtype=np.float64)

        for i in range(n_frames):
            frame = np.fft.irfft(Z[:, i], n=n_fft)
            frame = frame[:win_length] * w
            start = i * hop_length
            end = start + win_length
            samples[start:end] += frame
            denom[start:end] += w ** 2

        denom = np.where(denom > 1e-10, denom, 1.0)
        samples = samples / denom
        samples = samples[n_fft // 2: -(n_fft // 2)]
        return samples

    def synthesize(self, text, output_path, speed=1.0, noise_scale=0.667):
        """合成语音，写入 WAV 文件

        Args:
            text: 输入文本
            output_path: 输出 WAV 文件路径
            speed: 语速控制 (1.0=正常, 0.5=慢速, 2.0=快速)
            noise_scale: 噪声尺度 (0.667 默认, 越小越稳定)
        """
        if not self._loaded:
            self.load()

        # 1. 文本 → token IDs
        t0 = time.time()
        ids = self.text_to_ids(text)
        g2p_time = (time.time() - t0) * 1000
        if not ids:
            log("文本转 ID 结果为空")
            return False
        log(f"  G2P: {len(ids)} tokens in {g2p_time:.1f}ms")

        # 2. 声学模型推理 → mel 频谱
        tokens = np.array([ids], dtype=np.int64)

        x_lengths = np.array([tokens.shape[1]], dtype=np.int64)
        length_scale = np.array([max(0.5, 1.0 / max(speed, 0.1))], dtype=np.float32)
        noise_scale_arr = np.array([noise_scale], dtype=np.float32)

        t0 = time.time()
        mel = self.am_session.run(
            [self.am_session.get_outputs()[0].name],
            {
                self.am_session.get_inputs()[0].name: tokens,
                self.am_session.get_inputs()[1].name: x_lengths,
                self.am_session.get_inputs()[2].name: noise_scale_arr,
                self.am_session.get_inputs()[3].name: length_scale,
            },
        )[0]
        am_time = (time.time() - t0) * 1000
        log(f"  声学模型: {mel.shape[2]} 帧, {am_time:.1f}ms")

        # 3. 声码器推理 → 直接输出音频波形 (Vocos 单输出版)
        t0 = time.time()
        audio = self.vocoder_session.run(
            [self.vocoder_session.get_outputs()[0].name],
            {self.vocoder_session.get_inputs()[0].name: mel},
        )[0]
        # audio shape: (1, 1, num_samples)
        audio = audio[0, 0, :]  # → (num_samples,)
        voc_time = (time.time() - t0) * 1000
        log(f"  声码器: {len(audio)} samples, {voc_time:.1f}ms, max={audio.max():.4f}")

        # 5. 切除头部无声段
        start_idx = 0
        trim_len = min(len(audio), int(self.sample_rate * 0.5))
        for i in range(trim_len):
            if abs(audio[i]) > TRIM_THRESHOLD:
                start_idx = max(0, i - int(self.sample_rate * 0.01))
                break
        if start_idx > 0:
            audio = audio[start_idx:]

        # 6. 短淡入 (5ms)
        fade_len = int(self.sample_rate * FADE_IN_SEC)
        if fade_len > 0 and fade_len < len(audio):
            fade_in = np.linspace(0, 1, fade_len)
            audio[:fade_len] *= fade_in

        # 7. RMS 归一化到目标响度
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 1e-6:
            gain = TARGET_RMS / current_rms
            audio = audio * gain
            # 防止削波（软限幅）
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio * (0.95 / peak)
            log(f"  RMS: {current_rms:.4f} → {TARGET_RMS:.2f} (gain={gain:.2f}x, peak={peak:.4f})")

        # 8. 写入 WAV
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        with wave_module.open(output_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(audio_int16.tobytes())

        return True


class MatchaTTSService:
    def __init__(self):
        self.running = False
        self.sock = None
        self.tts = MatchaTTSInference()
        self.request_count = 0
        self.total_time = 0.0

    def start(self):
        log("启动 Matcha-TTS 服务...")

        for f in [ACOUSTIC_MODEL, VOCODER_MODEL, TOKENS_FILE, LEXICON_FILE]:
            if not os.path.exists(f):
                log(f"错误: 文件不存在 {f}")
                sys.exit(1)

        log("模型文件就绪")

        # 加载模型（常驻内存）
        self.tts.load()

        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o777)
        self.sock.listen(5)

        self.running = True
        log(f"服务已启动，监听 {SOCKET_PATH}")

        while self.running:
            try:
                self.sock.settimeout(1.0)
                conn, _ = self.sock.accept()
                threading.Thread(target=self.handle, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    log(f"Accept 异常: {e}")

    def handle(self, conn):
        try:
            data = conn.recv(8192).decode()
            request = json.loads(data)
            cmd = request.get("command", "")

            if cmd == "ping":
                resp = {"status": "ok", "model_loaded": self.tts._loaded}
            elif cmd == "synthesize":
                text = request.get("text", "")
                output = request.get("output", "/tmp/matcha_out.wav")
                speed = request.get("speed", 1.0)
                noise_scale = request.get("noise_scale", 0.667)
                if text:
                    out_dir = os.path.dirname(output)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                    t0 = time.time()
                    if self.tts.synthesize(text, output, speed, noise_scale):
                        elapsed = (time.time() - t0) * 1000
                        self.request_count += 1
                        self.total_time += elapsed
                        avg = self.total_time / self.request_count
                        resp = {
                            "status": "ok",
                            "output": output,
                            "sample_rate": self.tts.sample_rate,
                            "time_ms": round(elapsed, 1),
                            "avg_time_ms": round(avg, 1),
                        }
                    else:
                        resp = {"status": "error", "message": "合成失败"}
                else:
                    resp = {"status": "error", "message": "文本为空"}
            elif cmd == "stats":
                avg = self.total_time / self.request_count if self.request_count > 0 else 0
                resp = {
                    "status": "ok",
                    "request_count": self.request_count,
                    "total_time_ms": round(self.total_time, 1),
                    "avg_time_ms": round(avg, 1),
                }
            else:
                resp = {"status": "error", "message": f"未知命令: {cmd}"}
        except Exception as e:
            log(f"处理请求异常: {e}")
            resp = {"status": "error", "message": str(e)}

        try:
            conn.sendall(json.dumps(resp, ensure_ascii=False).encode())
        except:
            pass
        conn.close()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        for f in [SOCKET_PATH, PID_FILE]:
            if os.path.exists(f):
                os.remove(f)
        log("服务已停止")


def main():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    service = MatchaTTSService()

    def signal_handler(signum, frame):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    service.start()


if __name__ == "__main__":
    main()
