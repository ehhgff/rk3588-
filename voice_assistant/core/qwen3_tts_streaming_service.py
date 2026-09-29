#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS 流式生成服务 (CPU 推理)
不依赖 NPU，纯 CPU 流式推理

流式架构:
    文本 → Tokenizer → Talker (逐 token 生成) → Speech Decoder (分块解码) → 音频流

特点:
    - 首包延迟低 (~1-2秒)
    - 边生成边播放
    - 支持打断和取消
    - 纯 CPU 运行，无需 NPU

文件位置: /userdata/voice_assistant/qwen3_tts_streaming_service.py
"""

import os
import sys
import json
import time
import socket
import struct
import threading
import queue
import wave
import io
from typing import Generator, Optional, List
import numpy as np

# 设置代理
os.environ['http_proxy'] = 'http://192.168.56.150:7897'
os.environ['https_proxy'] = 'http://192.168.56.150:7897'

# ============ 配置 ============
MODEL_DIR = "/home/ubuntu/桌面/ai/llm_models/Qwen3-TTS-ONNX-Real"
TALKER_ENCODER = os.path.join(MODEL_DIR, "talker_encoder.onnx")
TALKER_DECODER = os.path.join(MODEL_DIR, "talker_decoder.onnx")
SPEECH_DECODER = os.path.join(MODEL_DIR, "speech_decoder.onnx")
MODEL_PATH = "/home/ubuntu/桌面/ai/llm_models/Qwen3-TTS-12Hz-0.6B-CustomVoice"

SERVICE_SOCK = "/tmp/qwen3_tts_streaming.sock"
LOG_FILE = "/tmp/qwen3_tts_streaming.log"

# 音频配置
SAMPLE_RATE = 24000
CHUNK_DURATION = 0.3  # 每块音频时长 (秒)
CHUNK_TOKENS = 4      # 每块解码的 token 数 (12Hz * 0.3s ≈ 4)

# 全局会话
talker_encoder_sess = None
talker_decoder_sess = None
speech_decoder_sess = None
tokenizer = None


def log(msg):
    """记录日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except:
        pass


def init_service():
    """初始化服务"""
    global talker_encoder_sess, talker_decoder_sess, speech_decoder_sess, tokenizer
    
    log("=" * 60)
    log("Qwen3-TTS 流式服务初始化")
    log("=" * 60)
    
    try:
        import onnxruntime as ort
        
        # CPU 推理配置
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # RK3588 有 4 个 A76 核心
        
        # 加载 Talker Encoder
        if os.path.exists(TALKER_ENCODER):
            talker_encoder_sess = ort.InferenceSession(
                TALKER_ENCODER, sess_options, providers=providers
            )
            log(f"✓ Talker Encoder 加载成功")
        else:
            log(f"⚠️ Talker Encoder 不存在: {TALKER_ENCODER}")
        
        # 加载 Talker Decoder
        if os.path.exists(TALKER_DECODER):
            talker_decoder_sess = ort.InferenceSession(
                TALKER_DECODER, sess_options, providers=providers
            )
            log(f"✓ Talker Decoder 加载成功")
        else:
            log(f"⚠️ Talker Decoder 不存在: {TALKER_DECODER}")
        
        # 加载 Speech Decoder
        if os.path.exists(SPEECH_DECODER):
            speech_decoder_sess = ort.InferenceSession(
                SPEECH_DECODER, sess_options, providers=providers
            )
            log(f"✓ Speech Decoder 加载成功")
        else:
            log(f"⚠️ Speech Decoder 不存在: {SPEECH_DECODER}")
        
        # 加载 Tokenizer
        sys.path.insert(0, os.path.dirname(__file__))
        from qwen3_tts_tokenizer import Qwen3TTSTokenizer
        tokenizer = Qwen3TTSTokenizer(MODEL_PATH)
        log(f"✓ Tokenizer 加载成功")
        
        return True
        
    except Exception as e:
        log(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


class StreamingTTSEngine:
    """
    流式 TTS 引擎
    
    工作流程:
    1. 接收文本，构建 prompt
    2. 自回归生成 token (流式)
    3. 累积到一定数量后解码为音频块
    4. 输出音频块
    """
    
    def __init__(self):
        self.cancelled = False
        self.generated_tokens = []
        self.audio_buffer = []
        
    def cancel(self):
        """取消生成"""
        self.cancelled = True
        log("生成已取消")
    
    def generate_tokens_streaming(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 500,
        temperature: float = 0.9
    ) -> Generator[int, None, None]:
        """
        流式生成 token
        
        每生成一个 token 就 yield 出来
        
        参数:
            prompt_ids: 输入 prompt token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            
        Yields:
            生成的 token ID
        """
        if talker_encoder_sess is None or talker_decoder_sess is None:
            log("Talker 模型未加载")
            return
        
        generated = list(prompt_ids)
        
        # TTS 结束 token
        tts_eod = 151673
        codec_eos = 2150
        
        log(f"开始流式生成，prompt 长度: {len(prompt_ids)}")
        start_time = time.time()
        
        for i in range(max_new_tokens):
            if self.cancelled:
                log("生成被中断")
                break
            
            # 准备输入 (只取最后 256 个 token)
            seq = generated[-256:]
            input_tensor = np.expand_dims(np.array(seq, dtype=np.int64), axis=0)
            
            # Encoder: token -> hidden states
            enc_outputs = talker_encoder_sess.run(None, {'input_ids': input_tensor})
            hidden_states = enc_outputs[0]  # [1, seq_len, hidden_size]
            
            # Decoder: hidden states -> logits (只取最后一个位置)
            last_hidden = hidden_states[:, -1:, :]  # [1, 1, hidden_size]
            dec_outputs = talker_decoder_sess.run(None, {'hidden_states': last_hidden})
            logits = dec_outputs[0]  # [1, 1, vocab_size]
            
            # 采样
            next_token_logits = logits[0, 0, :] / temperature
            
            # 应用 softmax
            probs = np.exp(next_token_logits - np.max(next_token_logits))
            probs = probs / np.sum(probs)
            
            # 采样
            next_token = int(np.random.choice(len(probs), p=probs))
            
            generated.append(next_token)
            
            # 检查结束条件
            if next_token in [tts_eod, codec_eos]:
                log(f"检测到结束 token: {next_token}")
                break
            
            # 每 20 个 token 打印进度
            if i > 0 and i % 20 == 0:
                elapsed = time.time() - start_time
                speed = i / elapsed
                log(f"  生成进度: {i} tokens, 速度: {speed:.1f} tok/s")
            
            yield next_token
        
        total_time = time.time() - start_time
        log(f"生成完成: {i+1} tokens, 耗时: {total_time:.1f}s, 平均: {(i+1)/total_time:.1f} tok/s")
    
    def decode_audio_chunk(self, codec_tokens: List[int]) -> Optional[np.ndarray]:
        """
        解码音频块
        
        将 codec tokens 解码为音频波形
        
        参数:
            codec_tokens: codec token 列表
            
        返回:
            音频波形 (numpy array) 或 None
        """
        if speech_decoder_sess is None or len(codec_tokens) < 16:
            return None
        
        try:
            # 重塑为 [batch, num_codebooks, seq_len]
            num_codebooks = 16
            seq_len = len(codec_tokens) // num_codebooks
            
            if seq_len == 0:
                return None
            
            codec_tokens = codec_tokens[:seq_len * num_codebooks]
            codes = np.array(codec_tokens, dtype=np.int64).reshape(1, num_codebooks, seq_len)
            
            # Speech Decoder 推理
            outputs = speech_decoder_sess.run(None, {'codes': codes})
            audio = outputs[0]  # [batch, audio_samples]
            
            return audio[0]  # 返回一维数组
            
        except Exception as e:
            log(f"音频解码失败: {e}")
            return None
    
    def synthesize_streaming(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "chinese",
        instruct: str = ""
    ) -> Generator[bytes, None, None]:
        """
        流式合成音频
        
        参数:
            text: 要合成的文本
            speaker: 说话人
            language: 语言
            instruct: 风格指令
            
        Yields:
            WAV 格式的音频块 (bytes)
        """
        log(f"=" * 50)
        log(f"流式合成请求: {text[:50]}...")
        log(f"说话人: {speaker}, 语言: {language}")
        
        # 1. 构建 prompt
        prompt_ids = tokenizer.build_tts_prompt(text, speaker, language, instruct)
        log(f"Prompt 长度: {len(prompt_ids)} tokens")
        
        # 2. 流式生成 token 并累积解码
        codec_buffer = []
        chunk_count = 0
        
        for token in self.generate_tokens_streaming(prompt_ids):
            if self.cancelled:
                break
            
            # 过滤出 codec token (0-1023)
            if 0 <= token < 1024:
                codec_buffer.append(token)
            
            # 累积到足够数量后解码
            if len(codec_buffer) >= CHUNK_TOKENS * 16:  # 16 codebooks
                audio_chunk = self.decode_audio_chunk(codec_buffer)
                
                if audio_chunk is not None:
                    # 转换为 WAV bytes
                    wav_bytes = self.audio_to_wav_bytes(audio_chunk)
                    yield wav_bytes
                    chunk_count += 1
                    log(f"  输出音频块 #{chunk_count}, 时长: {len(audio_chunk)/SAMPLE_RATE:.2f}s")
                
                # 保留重叠部分 (避免边界不连续)
                overlap = 16  # 1 frame overlap
                codec_buffer = codec_buffer[-overlap:] if len(codec_buffer) > overlap else []
        
        # 解码剩余 token
        if len(codec_buffer) >= 16 and not self.cancelled:
            audio_chunk = self.decode_audio_chunk(codec_buffer)
            if audio_chunk is not None:
                wav_bytes = self.audio_to_wav_bytes(audio_chunk)
                yield wav_bytes
                chunk_count += 1
                log(f"  输出最终音频块 #{chunk_count}")
        
        log(f"流式合成完成，共 {chunk_count} 个音频块")
    
    def audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """
        将音频数组转换为 WAV bytes
        
        参数:
            audio: 音频波形 (float32, [-1, 1])
            
        返回:
            WAV 格式的 bytes
        """
        # 确保音频在 [-1, 1] 范围内
        audio = np.clip(audio, -1.0, 1.0)
        
        # 转换为 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 使用 wave 模块生成 WAV
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()


def handle_streaming_client(conn, addr):
    """处理流式客户端请求"""
    try:
        # 接收请求
        data = b''
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b'\n' in chunk:
                break
        
        request = json.loads(data.decode().strip())
        command = request.get('command')
        
        if command == 'synthesize_streaming':
            text = request.get('text', '')
            speaker = request.get('speaker', 'Vivian')
            language = request.get('language', 'chinese')
            instruct = request.get('instruct', '')
            
            # 创建引擎
            engine = StreamingTTSEngine()
            
            # 发送开始标记
            conn.send(json.dumps({'status': 'started'}).encode() + b'\n')
            
            # 流式发送音频块
            chunk_index = 0
            for wav_bytes in engine.synthesize_streaming(text, speaker, language, instruct):
                if engine.cancelled:
                    break
                
                # 发送块头信息
                header = json.dumps({
                    'status': 'chunk',
                    'index': chunk_index,
                    'size': len(wav_bytes)
                })
                conn.send(header.encode() + b'\n')
                
                # 发送音频数据
                conn.send(wav_bytes)
                chunk_index += 1
            
            # 发送结束标记
            conn.send(json.dumps({'status': 'finished', 'chunks': chunk_index}).encode() + b'\n')
            
        elif command == 'synthesize':
            # 非流式合成 (兼容旧接口)
            text = request.get('text', '')
            output = request.get('output', '/tmp/qwen3_tts_streaming_output.wav')
            speaker = request.get('speaker', 'Vivian')
            language = request.get('language', 'chinese')
            
            engine = StreamingTTSEngine()
            
            # 收集所有音频块
            all_audio = []
            for wav_bytes in engine.synthesize_streaming(text, speaker, language):
                # 解析 WAV bytes 获取音频数据
                audio = wav_bytes_to_audio(wav_bytes)
                if audio is not None:
                    all_audio.append(audio)
            
            if all_audio:
                # 合并音频
                full_audio = np.concatenate(all_audio)
                
                # 保存 WAV
                save_wav(full_audio, output)
                
                response = {
                    'status': 'ok',
                    'output': output,
                    'sample_rate': SAMPLE_RATE,
                    'duration': len(full_audio) / SAMPLE_RATE
                }
            else:
                response = {'status': 'error', 'message': '合成失败'}
            
            conn.send(json.dumps(response).encode())
            
        elif command == 'ping':
            conn.send(json.dumps({'status': 'ok', 'message': 'pong'}).encode())
            
        elif command == 'status':
            conn.send(json.dumps({
                'status': 'ok',
                'encoder_loaded': talker_encoder_sess is not None,
                'decoder_loaded': talker_decoder_sess is not None,
                'speech_loaded': speech_decoder_sess is not None,
                'mode': 'streaming_cpu'
            }).encode())
            
        else:
            conn.send(json.dumps({'status': 'error', 'message': f'未知命令: {command}'}).encode())
            
    except Exception as e:
        log(f"处理请求异常: {e}")
        try:
            conn.send(json.dumps({'status': 'error', 'message': str(e)}).encode())
        except:
            pass
    finally:
        conn.close()


def wav_bytes_to_audio(wav_bytes: bytes) -> Optional[np.ndarray]:
    """从 WAV bytes 提取音频数据"""
    try:
        buffer = io.BytesIO(wav_bytes)
        with wave.open(buffer, 'rb') as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            
            raw_data = wf.readframes(nframes)
            
            if sampwidth == 2:
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32767.0
            else:
                audio = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32) / 128.0
            
            if nchannels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            
            return audio
    except Exception as e:
        log(f"解析 WAV 失败: {e}")
        return None


def save_wav(audio: np.ndarray, output_path: str):
    """保存音频为 WAV 文件"""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def start_service():
    """启动流式 TTS 服务"""
    log("=" * 60)
    log("Qwen3-TTS 流式服务启动 (CPU 推理)")
    log("=" * 60)
    
    if not init_service():
        log("❌ 服务启动失败")
        return
    
    # 清理旧 socket
    if os.path.exists(SERVICE_SOCK):
        os.remove(SERVICE_SOCK)
    
    # 创建 socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SERVICE_SOCK)
    server.listen(5)
    os.chmod(SERVICE_SOCK, 0o777)
    
    log(f"✓ 服务已启动: {SERVICE_SOCK}")
    log("等待请求...")
    
    try:
        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_streaming_client, args=(conn, addr))
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        log("\n服务停止")
    finally:
        server.close()
        if os.path.exists(SERVICE_SOCK):
            os.remove(SERVICE_SOCK)


if __name__ == '__main__':
    start_service()
