#!/usr/bin/env python3
"""
SenseVoice RKNN 推理脚本 (RK3588 开发板端)
使用 RKNN-Lite 在 RK3588 上进行语音情感识别
"""

import os
import sys
import time
import numpy as np
import argparse
from pathlib import Path

# 导入 RKNN Lite API
try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("错误: 未找到 RKNN-Lite")
    print("请在开发板上安装 RKNN-Lite:")
    print("  pip install rknn_toolkit_lite2-*.whl")
    sys.exit(1)

# 尝试导入音频处理库
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("警告: 未找到 librosa，音频预处理功能将受限")


class SenseVoiceRKNN:
    """
    SenseVoice RKNN 推理类
    """
    
    # 情感标签映射
    EMOTION_LABELS = {
        25001: "HAPPY",      # 开心
        25002: "SAD",        # 悲伤
        25003: "ANGRY",      # 愤怒
        25004: "NEUTRAL",    # 中性
        25005: "FEARFUL",    # 恐惧
        25006: "SURPRISED",  # 惊讶
        25007: "DISGUSTED",  # 厌恶
    }
    
    # 情感表情符号
    EMOTION_EMOJI = {
        "HAPPY": "😊",
        "SAD": "😔",
        "ANGRY": "😡",
        "NEUTRAL": "😐",
        "FEARFUL": "😰",
        "SURPRISED": "😮",
        "DISGUSTED": "🤢",
        "UNKNOWN": "❓",
    }
    
    def __init__(self, model_path, target_platform="rk3588", core_mask=RKNNLite.NPU_CORE_AUTO):
        """
        初始化 SenseVoice RKNN 推理器
        
        Args:
            model_path: RKNN 模型路径
            target_platform: 目标平台
            core_mask: NPU 核心掩码
                - RKNNLite.NPU_CORE_AUTO: 自动选择
                - RKNNLite.NPU_CORE_0: 使用 NPU 核心 0
                - RKNNLite.NPU_CORE_1: 使用 NPU 核心 1
                - RKNNLite.NPU_CORE_2: 使用 NPU 核心 2
                - RKNNLite.NPU_CORE_ALL: 使用所有核心
        """
        self.model_path = model_path
        self.target_platform = target_platform
        self.core_mask = core_mask
        self.rknn = None
        
    def init_runtime(self):
        """
        初始化 RKNN 运行时环境
        """
        print(f"初始化 RKNN 运行时...")
        print(f"  模型: {self.model_path}")
        print(f"  平台: {self.target_platform}")
        print(f"  NPU 核心: {self.core_mask}")
        
        # 创建 RKNN Lite 对象
        self.rknn = RKNNLite(verbose=False)
        
        # 加载 RKNN 模型
        print("加载 RKNN 模型...")
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            print("加载 RKNN 模型失败!")
            return False
        
        # 初始化运行时环境
        print("初始化 NPU 运行时...")
        ret = self.rknn.init_runtime(
            core_mask=self.core_mask,
        )
        if ret != 0:
            print("初始化运行时失败!")
            return False
        
        print("RKNN 运行时初始化成功!")
        return True
    
    def extract_mel_spectrogram(self, audio_path, sr=16000, n_mels=80, n_fft=400, hop_length=160):
        """
        从音频文件提取梅尔频谱特征
        
        Args:
            audio_path: 音频文件路径
            sr: 采样率
            n_mels: 梅尔频带数
            n_fft: FFT 窗口大小
            hop_length: 帧移
            
        Returns:
            mel_spec: 梅尔频谱 [1, seq_len, n_mels]
        """
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa 未安装，无法提取音频特征")
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=sr)
        
        # 提取梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        
        # 转换为对数刻度 (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 转置为 [seq_len, n_mels]
        log_mel_spec = log_mel_spec.T
        
        # 添加 batch 维度 [1, seq_len, n_mels]
        log_mel_spec = np.expand_dims(log_mel_spec, axis=0).astype(np.float32)
        
        return log_mel_spec
    
    def preprocess_audio(self, audio_input):
        """
        预处理音频输入
        
        Args:
            audio_input: 音频文件路径或 numpy 数组
            
        Returns:
            input_data: 预处理后的输入数据
        """
        if isinstance(audio_input, str):
            # 从文件加载
            input_data = self.extract_mel_spectrogram(audio_input)
        elif isinstance(audio_input, np.ndarray):
            # 直接使用 numpy 数组
            if audio_input.ndim == 2:
                # [seq_len, feature_dim] -> [1, seq_len, feature_dim]
                input_data = np.expand_dims(audio_input, axis=0).astype(np.float32)
            elif audio_input.ndim == 3:
                input_data = audio_input.astype(np.float32)
            else:
                raise ValueError(f"不支持的输入维度: {audio_input.ndim}")
        else:
            raise ValueError(f"不支持的输入类型: {type(audio_input)}")
        
        return input_data
    
    def inference(self, audio_input):
        """
        执行推理
        
        Args:
            audio_input: 音频文件路径或 numpy 数组
            
        Returns:
            result: 推理结果字典
        """
        # 预处理
        input_data = self.preprocess_audio(audio_input)
        
        # 执行推理
        start_time = time.time()
        outputs = self.rknn.inference(inputs=[input_data])
        inference_time = time.time() - start_time
        
        # 解析输出
        ctc_logits = outputs[0]  # [batch, seq_len, vocab_size]
        
        # 解码情感标签
        emotion_result = self.decode_emotion(ctc_logits)
        
        result = {
            "emotion": emotion_result["emotion"],
            "emotion_emoji": emotion_result["emoji"],
            "confidence": emotion_result["confidence"],
            "inference_time": inference_time,
            "rtf": inference_time / (input_data.shape[1] * 0.01),  # 假设每帧 10ms
        }
        
        return result
    
    def decode_emotion(self, ctc_logits):
        """
        从 CTC 输出解码情感标签
        
        Args:
            ctc_logits: CTC 输出 logits
            
        Returns:
            emotion_result: 情感识别结果
        """
        # 获取每个时间步的最大概率标签
        pred_indices = np.argmax(ctc_logits, axis=-1)[0]  # [seq_len]
        
        # 统计情感标签出现频率
        emotion_counts = {}
        for idx in pred_indices:
            idx = int(idx)
            if idx in self.EMOTION_LABELS:
                emotion = self.EMOTION_LABELS[idx]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if not emotion_counts:
            return {
                "emotion": "UNKNOWN",
                "emoji": self.EMOTION_EMOJI["UNKNOWN"],
                "confidence": 0.0,
            }
        
        # 选择出现频率最高的情感
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        total_count = sum(emotion_counts.values())
        confidence = emotion_counts[dominant_emotion] / total_count
        
        return {
            "emotion": dominant_emotion,
            "emoji": self.EMOTION_EMOJI.get(dominant_emotion, self.EMOTION_EMOJI["UNKNOWN"]),
            "confidence": confidence,
        }
    
    def release(self):
        """
        释放资源
        """
        if self.rknn is not None:
            self.rknn.release()
            print("RKNN 资源已释放")


def benchmark(model_path, audio_path, num_runs=100):
    """
    基准测试
    """
    print("="*60)
    print("SenseVoice RKNN 基准测试")
    print("="*60)
    
    # 初始化
    sv = SenseVoiceRKNN(model_path)
    if not sv.init_runtime():
        return
    
    # 预热
    print("\n预热...")
    input_data = sv.preprocess_audio(audio_path)
    for _ in range(10):
        sv.rknn.inference(inputs=[input_data])
    
    # 正式测试
    print(f"\n运行 {num_runs} 次推理...")
    times = []
    
    for i in range(num_runs):
        start = time.time()
        outputs = sv.rknn.inference(inputs=[input_data])
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{num_runs}")
    
    # 统计结果
    times = np.array(times)
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # 计算 RTF
    audio_duration = input_data.shape[1] * 0.01  # 假设每帧 10ms
    rtf = avg_time / audio_duration
    
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"音频时长: {audio_duration:.2f} s")
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"最小推理时间: {min_time*1000:.2f} ms")
    print(f"最大推理时间: {max_time*1000:.2f} ms")
    print(f"RTF (实时因子): {rtf:.3f}")
    print(f"FPS (每秒处理): {1/avg_time:.1f}")
    
    sv.release()


def main():
    parser = argparse.ArgumentParser(
        description="SenseVoice RKNN Inference on RK3588"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="RKNN model path"
    )
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="Input audio file path"
    )
    parser.add_argument(
        "--core",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2, 3],
        help="NPU core to use (-1=auto, 0=core0, 1=core1, 2=core2, 3=all)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of runs for benchmark"
    )
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.model):
        print(f"错误: 找不到模型文件: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.audio):
        print(f"错误: 找不到音频文件: {args.audio}")
        sys.exit(1)
    
    # 设置 NPU 核心
    core_mask_map = {
        -1: RKNNLite.NPU_CORE_AUTO,
        0: RKNNLite.NPU_CORE_0,
        1: RKNNLite.NPU_CORE_1,
        2: RKNNLite.NPU_CORE_2,
        3: RKNNLite.NPU_CORE_ALL,
    }
    core_mask = core_mask_map.get(args.core, RKNNLite.NPU_CORE_AUTO)
    
    if args.benchmark:
        # 基准测试模式
        benchmark(args.model, args.audio, args.num_runs)
    else:
        # 单次推理模式
        print("="*60)
        print("SenseVoice RKNN 推理")
        print("="*60)
        
        # 初始化
        sv = SenseVoiceRKNN(args.model, core_mask=core_mask)
        if not sv.init_runtime():
            sys.exit(1)
        
        # 执行推理
        print(f"\n处理音频: {args.audio}")
        result = sv.inference(args.audio)
        
        # 显示结果
        print("\n" + "="*60)
        print("推理结果")
        print("="*60)
        print(f"情感: {result['emotion']} {result['emotion_emoji']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"推理时间: {result['inference_time']*1000:.2f} ms")
        print(f"RTF: {result['rtf']:.3f}")
        
        # 释放资源
        sv.release()


if __name__ == "__main__":
    main()
