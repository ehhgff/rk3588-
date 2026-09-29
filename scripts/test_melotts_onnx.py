#!/usr/bin/env python3
"""
MeloTTS ONNX 模型测试 - 评估生成速度
纯 ONNX Runtime + numpy 实现，无需 PyTorch
"""

import sys
import os
import time
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/home/ubuntu/桌面/ai/MeloTTS/mmontol-MeloTTS")

import soundfile as sf

# 从 MeloTTS-Chinese config.json 加载正确符号表（112 个符号，匹配 ONNX 模型）
import json
with open("/home/ubuntu/桌面/ai/MeloTTS/MeloTTS/MeloTTS-Chinese/config.json") as f:
    MODEL_CONFIG = json.load(f)
ONNX_SYMBOLS = MODEL_CONFIG["symbols"]  # 共 112 个符号

ONNX_DIR = "/home/ubuntu/桌面/ai/MeloTTS/mmontol-MeloTTS"
LANGUAGE = "ZH_MIX_EN"
SPEAKER_ID = 0
SAMPLE_RATE = 44100

TEST_TEXTS = [
    "您好，我是智能语音助手，请问有什么可以帮您的吗？",
    "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。",
    "根据最新的临床指南，高血压患者的血压控制目标应当低于130/80毫米汞柱。",
    "糖尿病是一种慢性代谢性疾病，主要表现为血糖水平升高，需要长期管理和治疗。",
]


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    return np.arange(max_length, dtype=length.dtype)[None, :] < length[:, None]


class TextProcessor:
    def __init__(self):
        from melo.text import cleaned_text_to_sequence
        from melo.text.cleaner import clean_text
        self.clean_text = clean_text
        self.cleaned_text_to_sequence = cleaned_text_to_sequence
        self._symbol_to_id = {s: i for i, s in enumerate(ONNX_SYMBOLS)}

    def process(self, text):
        norm_text, phone, tone, word2ph = self.clean_text(text, LANGUAGE)
        phone, tone, language = self.cleaned_text_to_sequence(
            phone, tone, LANGUAGE, self._symbol_to_id)
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language = intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
        phone = np.array(phone, dtype=np.int64)
        tone = np.array(tone, dtype=np.int64)
        language = np.array(language, dtype=np.int64)
        return phone, tone, language, len(phone)


ONNX_X_LENGTH = 256

def prepare_encoder_inputs(phone, tone, language, x_length):
    x = phone.reshape(1, -1)
    x_lengths = np.array([min(x_length, ONNX_X_LENGTH)], dtype=np.int64)
    
    pad_len = ONNX_X_LENGTH - x.shape[1]
    if pad_len > 0:
        x = np.pad(x, ((0, 0), (0, pad_len)), constant_values=0)
        tone_input = np.pad(tone.reshape(1, -1), ((0, 0), (0, pad_len)), constant_values=0)
        lang_ids = np.pad(language.reshape(1, -1), ((0, 0), (0, pad_len)), constant_values=0)
    else:
        tone_input = tone.reshape(1, -1)
        lang_ids = language.reshape(1, -1)
    
    sid = np.array([SPEAKER_ID], dtype=np.int64)
    ja_bert = np.zeros((1, 768, ONNX_X_LENGTH), dtype=np.float32)
    noise_scale_w = np.array([0.8], dtype=np.float32)
    sdp_ratio = np.array([0.2], dtype=np.float32)
    return {
        'x': x, 'x_lengths': x_lengths, 'sid': sid,
        'tone': tone_input, 'lang_ids': lang_ids,
        'ja_bert': ja_bert, 'noise_scale_w': noise_scale_w,
        'sdp_ratio': sdp_ratio,
    }


def generate_path(duration, mask):
    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, -1)
    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
    path = path.reshape(b, t_x, t_y)
    pad = np.zeros((b, 1, t_y), dtype=path.dtype)
    path_shifted = np.concatenate([pad, path[:, :-1]], axis=1)
    path = path - path_shifted
    path = path[:, np.newaxis, :, :].transpose(0, 1, 3, 2) * mask
    return path


def encoder_postprocess(encoder_out):
    logw, x_mask, g, m_p, logs_p = encoder_out
    length_scale = 1.0
    w = np.exp(logw) * x_mask * length_scale
    w_ceil = np.ceil(w)
    # 固定尺寸：匹配 ONNX 导出配置（x_length=256 → y_length=512）
    y_length = int(2 * ONNX_X_LENGTH)
    y_lengths = np.array([y_length], dtype=np.float32)
    y_mask = sequence_mask(y_lengths)[:, None, :].astype(x_mask.dtype)
    x_mask_exp = np.expand_dims(x_mask, 2)
    y_mask_exp = np.expand_dims(y_mask, -1)
    attn_mask = x_mask_exp * y_mask_exp
    attn = generate_path(w_ceil, attn_mask)
    attn = attn.squeeze(1)
    return attn, y_mask, g, m_p, logs_p


def run_test():
    print("=" * 70)
    print("  MeloTTS ONNX 推理速度测试")
    print("=" * 70)

    print("\n[1] 加载 ONNX 模型...")
    t0 = time.time()
    ort_opts = ort.SessionOptions()
    ort_opts.intra_op_num_threads = 4
    ort_opts.inter_op_num_threads = 2

    enc_path = f"{ONNX_DIR}/encoder-ZH_MIX_EN.onnx"
    dec_path = f"{ONNX_DIR}/decoder-ZH_MIX_EN.onnx"
    enc_size = os.path.getsize(enc_path)
    dec_size = os.path.getsize(dec_path)
    print(f"   Encoder: {enc_path} ({enc_size/1024/1024:.1f}MB)")
    print(f"   Decoder: {dec_path} ({dec_size/1024/1024:.1f}MB)")

    enc_session = ort.InferenceSession(enc_path, sess_options=ort_opts, providers=['CPUExecutionProvider'])
    dec_session = ort.InferenceSession(dec_path, sess_options=ort_opts, providers=['CPUExecutionProvider'])
    print(f"   加载耗时: {time.time()-t0:.2f}s")

    enc_in_names = [inp.name for inp in enc_session.get_inputs()]
    dec_in_names = [inp.name for inp in dec_session.get_inputs()]
    print(f"   Encoder inputs: {enc_in_names}")
    print(f"   Decoder inputs: {dec_in_names}")

    print("\n[2] 初始化文本处理器...")
    t0 = time.time()
    tp = TextProcessor()
    print(f"   加载耗时: {time.time()-t0:.2f}s")

    print("\n[3] 推理测试\n")
    total_audio_dur = 0
    total_infer_time = 0

    for idx, text in enumerate(TEST_TEXTS):
        print(f"  {'=' * 68}")
        print(f"  测试 {idx+1}/{len(TEST_TEXTS)}: \"{text}\"")
        print(f"  {'-' * 68}")

        t0 = time.time()
        phone, tone, language, x_len = tp.process(text)
        text_ms = (time.time() - t0) * 1000
        print(f"   文本处理: {text_ms:.1f}ms | phones={x_len}")

        enc_in = prepare_encoder_inputs(phone, tone, language, x_len)
        enc_feed = {name: enc_in[name] for name in enc_in_names}
        t0 = time.time()
        enc_out = enc_session.run(None, enc_feed)
        enc_ms = (time.time() - t0) * 1000
        logw, x_mask, g, m_p, logs_p = enc_out
        print(f"   Encoder:  {enc_ms:7.1f}ms | "
              f"m_p={m_p.shape} logs_p={logs_p.shape}")
        print(f"   logw:     min={logw.min():.4f} max={logw.max():.4f} "
              f"mean={logw.mean():.4f} sum_exp={np.exp(logw).sum():.1f}")
        print(f"   x_mask:   sum={x_mask.sum().astype(int)} "
              f"m_p={m_p.mean():.4f}±{m_p.std():.4f} "
              f"logs_p={logs_p.mean():.4f}±{logs_p.std():.4f}")

        t0 = time.time()
        attn, y_mask, g, m_p, logs_p = encoder_postprocess(enc_out)
        post_ms = (time.time() - t0) * 1000
        print(f"   后处理:   {post_ms:7.1f}ms | attn={attn.shape} y={y_mask.shape}")

        dec_feed = {
            dec_in_names[0]: attn.astype(np.float32),
            dec_in_names[1]: y_mask.astype(np.float32),
            dec_in_names[2]: g.astype(np.float32),
            dec_in_names[3]: m_p.astype(np.float32),
            dec_in_names[4]: logs_p.astype(np.float32),
            dec_in_names[5]: np.array([0.6], dtype=np.float32),
        }
        t0 = time.time()
        [audio] = dec_session.run(None, dec_feed)
        dec_ms = (time.time() - t0) * 1000

        audio_dur = audio.shape[-1] / SAMPLE_RATE
        infer_total = (enc_ms + dec_ms) / 1000
        rtf = infer_total / audio_dur
        audio_float = audio[0, 0, :]

        print(f"   Decoder:  {dec_ms:7.1f}ms | "
              f"音频={audio_dur:.2f}s | RTF={rtf:.3f}")
        print(f"   波形:     max={audio_float.max():.8f} "
              f"min={audio_float.min():.8f} "
              f"mean={audio_float.mean():.8f} "
              f"nonzero={np.count_nonzero(np.abs(audio_float)>1e-8)}/{audio_float.size}")

        total_audio_dur += audio_dur
        total_infer_time += infer_total

        out_path = f"/tmp/mts_test_{idx}.wav"
        norm_audio = audio_float.astype(np.float32)
        mx = np.abs(norm_audio).max()
        if mx > 0:
            norm_audio = norm_audio / mx * 0.95
        sf.write(out_path, norm_audio, SAMPLE_RATE)
        print(f"   保存:     {out_path} ({os.path.getsize(out_path)} bytes)\n")

    print(f"  {'=' * 68}")
    print("  汇总")
    print(f"  {'=' * 68}")
    print(f"   测试文本数:   {len(TEST_TEXTS)}")
    print(f"   总音频时长:   {total_audio_dur:.2f}s")
    print(f"   总推理时间:   {total_infer_time:.2f}s")
    print(f"   平均 RTF:     {total_infer_time/total_audio_dur:.3f}")
    print(f"   平均速度:     {total_audio_dur/total_infer_time:.2f}x realtime")
    print(f"  {'=' * 68}")

    print("\n 播放测试:")
    for idx in range(len(TEST_TEXTS)):
        print(f"   aplay /tmp/mts_test_{idx}.wav")


if __name__ == "__main__":
    run_test()