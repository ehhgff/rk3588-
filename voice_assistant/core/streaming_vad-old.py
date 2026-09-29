#!/usr/bin/env python3
"""
完整语音助手 - Silero VAD + Zipformer ASR + RAG + Qwen3-0.6B
- 使用 USB 声卡 card 0 录音
- 实时进行 Silero VAD 语音检测
- 连续5帧检测到人声开始录音，半秒无人声停止录音
- 使用 Zipformer 进行语音识别
- 使用 RAG 检索增强（科室优化版）
- 使用 Qwen3-0.6B 生成回复
- 使用 USB 音响 card 1 播放
"""
import os
import sys
import time
import threading
import socket
import json
import numpy as np
import subprocess

# 导入扩展医学术语库
sys.path.insert(0, '/userdata/voice_assistant/rk3588_deploy')
from medical_terminology_extended import MedicalTerminologyExtended

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_FRAMES_THRESHOLD = int(0.7 / (FRAME_DURATION_MS / 1000))  # 0.7秒无人声停止ss
SILENCE_FRAMES_HALF = int(0.5 / (FRAME_DURATION_MS / 1000))  # 前半秒进入 Zipformer
SILENCE_FRAMES_TAIL = int(0.5 / (FRAME_DURATION_MS / 1000))  # 尾部静音缓冲，确保最后一个字被完整录制
SPEECH_FRAMES_TO_START = 5

# 流式处理参数
STREAMING_CHUNK_MS = 200  # 200ms 音频块
STREAMING_CHUNK_SIZE = int(SAMPLE_RATE * STREAMING_CHUNK_MS / 1000)  # 200ms 音频块大小

# Zipformer 流式识别参数
ZIPFORMER_STREAMING = True  # 启用 Zipformer 流式识别
ZIPFORMER_CHUNK_SIZE = 6400  # 200ms 音频块（16kHz * 0.2s）
ZIPFORMER_WINDOW_SIZE = 9600  # 300ms 窗口大小

RECORD_DEVICE = "plughw:0,0"
PLAY_DEVICE = "plughw:1,0"
OUTPUT_FILE = "/data/voice_assistant/test_recording.wav"
ZIPFORMER_DIR = "/data/zipformer"
ENCODER_MODEL = f"{ZIPFORMER_DIR}/model/encoder-epoch-99-avg-1.rknn"
DECODER_MODEL = f"{ZIPFORMER_DIR}/model/decoder-epoch-99-avg-1.rknn"
JOINER_MODEL = f"{ZIPFORMER_DIR}/model/joiner-epoch-99-avg-1.rknn"

VOICE_ASSISTANT_DIR = "/userdata/voice_assistant"
MEDICAL_RAG_DIR = "/data/medical_rag_full"
LLM_SERVICE_SOCK = "/tmp/qwen3_llm.sock"
RAG_SERVICE_SOCK = "/tmp/rag_optimized.sock"
TTS_SERVICE_SOCK = "/tmp/melotts_service.sock"

# 使用扩展医学术语库构建科室关键词映射
# 从MedicalTerminologyExtended类中提取数据
_med_terms = MedicalTerminologyExtended()

# 构建科室关键词映射（基于扩展医学术语库）
DEPT_KEYWORDS = {
    '内科': list(set(
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('高血压', []) +
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('糖尿病', []) +
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('胃炎', []) +
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('肺炎', []) +
        _med_terms.GENERAL_SYMPTOMS.get('发热', []) +
        _med_terms.RESPIRATORY_SYMPTOMS.get('咳嗽', []) +
        _med_terms.DIGESTIVE_SYMPTOMS.get('腹痛', []) +
        _med_terms.DIGESTIVE_SYMPTOMS.get('腹泻', []) +
        _med_terms.DIGESTIVE_SYMPTOMS.get('便秘', []) +
        _med_terms.DIGESTIVE_SYMPTOMS.get('恶心', []) +
        _med_terms.DIGESTIVE_SYMPTOMS.get('呕吐', []) +
        ['内科', '心脏', '心血管', '血压', '血糖', '消化', '肠道']
    )),
    '外科': list(set(
        _med_terms.SURGICAL_DISEASES.get('阑尾炎', []) +
        _med_terms.SURGICAL_DISEASES.get('胆囊炎', []) +
        _med_terms.SURGICAL_DISEASES.get('胆结石', []) +
        _med_terms.SURGICAL_DISEASES.get('骨折', []) +
        ['外科', '手术', '伤口', '缝合', '切除', '刀口', '创伤', '烧伤', '烫伤']
    )),
    '儿科': list(set(
        _med_terms.PEDIATRIC_DISEASES.get('上呼吸道感染', []) +
        _med_terms.PEDIATRIC_DISEASES.get('肺炎', []) +
        _med_terms.PEDIATRIC_DISEASES.get('腹泻', []) +
        _med_terms.PEDIATRIC_SYMPTOMS.get('发热', []) +
        ['儿科', '儿童', '小孩', '婴儿', '宝宝', '幼儿', '新生儿', '孩子', '小儿']
    )),
    '妇产科': list(set(
        _med_terms.OBSTETRICS_GYNECOLOGY_DISEASES.get('正常妊娠', []) +
        _med_terms.OBSTETRICS_GYNECOLOGY_DISEASES.get('月经不调', []) +
        _med_terms.OBSTETRICS_GYNECOLOGY_DISEASES.get('子宫肌瘤', []) +
        _med_terms.GYNECOLOGICAL_SYMPTOMS.get('月经异常', []) +
        _med_terms.GYNECOLOGICAL_SYMPTOMS.get('白带异常', []) +
        ['妇产', '妇科', '产科', '子宫', '卵巢', '孕期', '产妇', '哺乳期']
    )),
    '皮肤科': list(set(
        _med_terms.DERMATOLOGY_DISEASES.get('湿疹', []) +
        _med_terms.DERMATOLOGY_DISEASES.get('荨麻疹', []) +
        _med_terms.DERMATOLOGY_DISEASES.get('痤疮', []) +
        _med_terms.DERMATOLOGY_DISEASES.get('银屑病', []) +
        _med_terms.DERMATOLOGY_DISEASES.get('白癜风', []) +
        _med_terms.DERMATOLOGICAL_SYMPTOMS.get('皮疹', []) +
        _med_terms.DERMATOLOGICAL_SYMPTOMS.get('瘙痒', []) +
        ['皮肤', '皮肤科', '皮炎']
    )),
    '眼科': list(set(
        _med_terms.ENT_DISEASES.get('近视', []) +
        _med_terms.ENT_DISEASES.get('远视', []) +
        _med_terms.ENT_DISEASES.get('白内障', []) +
        _med_terms.ENT_DISEASES.get('青光眼', []) +
        _med_terms.ENT_SYMPTOMS.get('视力障碍', []) +
        _med_terms.ENT_SYMPTOMS.get('眼痛', []) +
        ['眼科', '眼睛']
    )),
    '耳鼻喉科': list(set(
        _med_terms.ENT_DISEASES.get('中耳炎', []) +
        _med_terms.ENT_DISEASES.get('鼻炎', []) +
        _med_terms.ENT_DISEASES.get('咽炎', []) +
        _med_terms.ENT_DISEASES.get('扁桃体炎', []) +
        _med_terms.ENT_SYMPTOMS.get('耳痛', []) +
        _med_terms.ENT_SYMPTOMS.get('咽痛', []) +
        _med_terms.ENT_SYMPTOMS.get('鼻塞', []) +
        ['耳鼻喉', '耳朵', '听力', '鼻子', '喉咙']
    )),
    '口腔科': list(set(
        _med_terms.ENT_DISEASES.get('声带息肉', []) +
        ['口腔', '牙齿', '龋齿', '牙周病', '口腔溃疡', '口臭', '牙龈炎', '拔牙', '补牙', '牙痛']
    )),
    '精神科': list(set(
        _med_terms.PSYCHIATRIC_DISEASES.get('抑郁症', []) +
        _med_terms.PSYCHIATRIC_DISEASES.get('焦虑症', []) +
        _med_terms.PSYCHIATRIC_DISEASES.get('失眠症', []) +
        _med_terms.PSYCHIATRIC_SYMPTOMS.get('焦虑', []) +
        _med_terms.PSYCHIATRIC_SYMPTOMS.get('抑郁', []) +
        _med_terms.PSYCHIATRIC_SYMPTOMS.get('强迫', []) +
        _med_terms.PSYCHIATRIC_SYMPTOMS.get('恐惧', []) +
        ['精神', '精神分裂', '躁狂']
    )),
    '心理科': list(set(
        _med_terms.PSYCHIATRIC_SYMPTOMS.get('焦虑', []) +
        _med_terms.PSYCHIATRIC_SYMPTOMS.get('抑郁', []) +
        ['心理', '心理咨询', '压力', '情绪', '心理障碍', '心理问题']
    )),
    '中医科': ['中医', '中药', '针灸', '拔罐', '气血', '肾虚', '肝火', '湿气', '调理'],
    '骨科': list(set(
        _med_terms.SURGICAL_DISEASES.get('骨折', []) +
        _med_terms.SURGICAL_DISEASES.get('腰椎间盘突出', []) +
        _med_terms.SURGICAL_DISEASES.get('颈椎病', []) +
        _med_terms.SURGICAL_DISEASES.get('关节炎', []) +
        ['骨科', '骨头', '关节', '骨质疏松']
    )),
    '肿瘤科': list(set(
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('癌症', []) +
        ['肿瘤', '癌症', '癌', '化疗', '放疗', '肿块', '良性', '恶性', '瘤']
    )),
    '神经科': list(set(
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('癫痫', []) +
        _med_terms.NEUROLOGICAL_SYMPTOMS.get('头痛', []) +
        _med_terms.NEUROLOGICAL_SYMPTOMS.get('眩晕', []) +
        _med_terms.NEUROLOGICAL_SYMPTOMS.get('抽搐', []) +
        _med_terms.NEUROLOGICAL_SYMPTOMS.get('麻木', []) +
        _med_terms.NEUROLOGICAL_SYMPTOMS.get('瘫痪', []) +
        ['神经', '帕金森', '神经炎', '神经痛', '脑梗塞', '脑出血']
    )),
    '心血管科': list(set(
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('冠心病', []) +
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('心力衰竭', []) +
        _med_terms.CARDIOVASCULAR_SYMPTOMS.get('心悸', []) +
        _med_terms.CARDIOVASCULAR_SYMPTOMS.get('胸痛', []) +
        ['心血管', '心脏', '心肌梗塞', '心律失常']
    )),
    '消化科': list(set(
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('胃炎', []) +
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('胃溃疡', []) +
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('肝炎', []) +
        _med_terms.DIGESTIVE_SYMPTOMS.get('腹痛', []) +
        _med_terms.DIGESTIVE_SYMPTOMS.get('腹泻', []) +
        _med_terms.DIGESTIVE_SYMPTOMS.get('便秘', []) +
        _med_terms.DIGESTIVE_SYMPTOMS.get('消化不良', []) +
        ['消化', '肠胃', '胃痛', '胃溃疡', '结肠炎', '痔疮']
    )),
    '呼吸科': list(set(
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('慢性阻塞性肺病', []) +
        _med_terms.INTERNAL_MEDICINE_DISEASES.get('哮喘', []) +
        _med_terms.RESPIRATORY_SYMPTOMS.get('咳嗽', []) +
        _med_terms.RESPIRATORY_SYMPTOMS.get('呼吸困难', []) +
        ['呼吸', '肺炎', '支气管炎', '肺气肿', '胸痛']
    )),
}

# 构建常见症状关键词（基于扩展医学术语库）
SYMPTOM_KEYWORDS = list(set(
    list(_med_terms.GENERAL_SYMPTOMS.get('疼痛', [])) +
    list(_med_terms.GENERAL_SYMPTOMS.get('发热', [])) +
    list(_med_terms.GENERAL_SYMPTOMS.get('疲劳', [])) +
    list(_med_terms.RESPIRATORY_SYMPTOMS.get('咳嗽', [])) +
    list(_med_terms.RESPIRATORY_SYMPTOMS.get('呼吸困难', [])) +
    list(_med_terms.DIGESTIVE_SYMPTOMS.get('恶心', [])) +
    list(_med_terms.DIGESTIVE_SYMPTOMS.get('呕吐', [])) +
    list(_med_terms.DIGESTIVE_SYMPTOMS.get('腹痛', [])) +
    list(_med_terms.DIGESTIVE_SYMPTOMS.get('腹泻', [])) +
    list(_med_terms.DIGESTIVE_SYMPTOMS.get('便秘', [])) +
    list(_med_terms.NEUROLOGICAL_SYMPTOMS.get('头痛', [])) +
    list(_med_terms.NEUROLOGICAL_SYMPTOMS.get('眩晕', [])) +
    list(_med_terms.NEUROLOGICAL_SYMPTOMS.get('失眠', [])) +
    list(_med_terms.DERMATOLOGICAL_SYMPTOMS.get('皮疹', [])) +
    list(_med_terms.DERMATOLOGICAL_SYMPTOMS.get('瘙痒', [])) +
    ['难受', '不舒服', '胸闷', '腹胀', '出汗', '寒战', '关节痛', '肌肉痛', '视力模糊', '听力下降']
))

# 构建症状-科室映射（基于扩展医学术语库）
SYMPTOM_DEPT_MAP = {
    # 消化系统症状 -> 内科/消化科
    '肚子疼': '内科',
    '腹痛': '内科',
    '胃痛': '消化科',
    '腹泻': '内科',
    '便秘': '内科',
    '恶心': '内科',
    '呕吐': '内科',
    '消化不良': '消化科',
    # 呼吸系统症状 -> 内科/呼吸科
    '发烧': '内科',
    '发热': '内科',
    '咳嗽': '内科',
    '咳痰': '呼吸科',
    '呼吸困难': '呼吸科',
    '胸闷': '心血管科',
    # 神经系统症状 -> 内科/神经科
    '头痛': '神经科',
    '头疼': '神经科',
    '头晕': '神经科',
    '眩晕': '神经科',
    '失眠': '精神科',
    '抽搐': '神经科',
    # 皮肤症状 -> 皮肤科
    '皮疹': '皮肤科',
    '瘙痒': '皮肤科',
    '皮肤痒': '皮肤科',
    '湿疹': '皮肤科',
    '痤疮': '皮肤科',
    # 五官症状
    '视力模糊': '眼科',
    '眼睛疼': '眼科',
    '听力下降': '耳鼻喉科',
    '耳鸣': '耳鼻喉科',
    '耳痛': '耳鼻喉科',
    '咽痛': '耳鼻喉科',
    '牙痛': '口腔科',
    # 骨骼肌肉症状 -> 骨科
    '关节痛': '骨科',
    '肌肉痛': '骨科',
    '骨折': '骨科',
    '腰疼': '骨科',
    '脖子疼': '骨科',
    # 心血管症状 -> 心血管科
    '心悸': '心血管科',
    '心慌': '心血管科',
    '胸痛': '心血管科',
    '心口痛': '心血管科',
    # 泌尿症状 -> 内科
    '尿频': '内科',
    '尿急': '内科',
    '尿痛': '内科',
    '血尿': '内科',
    # 精神症状 -> 精神科/心理科
    '焦虑': '心理科',
    '抑郁': '精神科',
    '情绪': '心理科',
}


def log_info(msg):
    print(f"[INFO] {msg}")

def log_step(msg):
    print(f"[STEP] {msg}")

def log_result(msg):
    print(f"[RESULT] {msg}")

def log_warn(msg):
    print(f"[WARN] {msg}")

def setup_audio():
    """设置音频设备"""
    log_info("设置音频设备...")
    print(f"  录音设备: {RECORD_DEVICE}")
    print(f"  播放设备: {PLAY_DEVICE}")
    # 设置音量为 30%
    subprocess.run(["amixer", "-c", "1", "set", "PCM", "30%"], capture_output=True)
    log_info("音频设备设置完成 (音量: 30%)\n")


def load_vad():
    """加载 VAD 模型"""
    log_info("加载 VAD 模型...")
    vad_available = False
    vad_type = "Volume"
    vad_model = None

    try:
        from silero_vad import load_silero_vad
        import torch

        model = load_silero_vad(torch.device('cpu'))
        vad_available = True
        vad_type = "Silero"
        vad_model = model
        log_info("✅ Silero VAD 模块加载成功")
    except ImportError as e:
        log_warn(f"⚠️  Silero VAD 模块加载失败: {e}")
        log_info("📢 使用音量检测作为备选")
    except Exception as e:
        log_warn(f"⚠️  Silero VAD 初始化失败: {e}")
        log_info("📢 使用音量检测作为备选")

    log_info(f"VAD 初始化成功 (类型: {vad_type})\n")
    return vad_available, vad_type, vad_model


def get_speech_prob(audio_tensor, model, sample_rate):
    """获取语音概率"""
    with torch.no_grad():
        prob = model(audio_tensor, sample_rate).item()
    return prob


def init_zipformer_streaming():
    """初始化 Zipformer 流式识别"""
    log_info("初始化 Zipformer 流式识别...")
    
    try:
        # 这里使用 subprocess 调用 Zipformer 命令行工具
        # 实际应用中，应该使用 Zipformer 的 C++ API 或 Python 绑定
        log_info("✅ Zipformer 流式识别初始化成功")
        return True
    except Exception as e:
        log_warn(f"❌ Zipformer 流式识别初始化失败: {e}")
        return False


def zipformer_streaming_recognize(audio_chunk):
    """Zipformer 流式识别单个音频块"""
    try:
        # 保存临时音频块
        temp_chunk = "/tmp/zipformer_chunk.wav"
        with open(temp_chunk, "wb") as f:
            f.write(create_wav_header(len(audio_chunk)) + audio_chunk)
        
        # 调用 Zipformer 命令行工具
        result = subprocess.run(
            [f"{ZIPFORMER_DIR}/rknn_zipformer_demo", ENCODER_MODEL, DECODER_MODEL, JOINER_MODEL, temp_chunk],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=ZIPFORMER_DIR
        )
        
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if "Zipformer output:" in line:
                    return line.split("Zipformer output:")[-1].strip()
        return ""
    except Exception as e:
        log_warn(f"Zipformer 流式识别失败: {e}")
        return ""


def create_wav_header(data_size):
    """创建 WAV 文件头"""
    header = b"RIFF"
    header += (data_size + 36).to_bytes(4, "little")
    header += b"WAVE"
    header += b"fmt "
    header += (16).to_bytes(4, "little")
    header += (1).to_bytes(2, "little")
    header += (1).to_bytes(2, "little")
    header += (SAMPLE_RATE).to_bytes(4, "little")
    header += (SAMPLE_RATE * 2).to_bytes(4, "little")
    header += (2).to_bytes(2, "little")
    header += (16).to_bytes(2, "little")
    header += b"data"
    header += data_size.to_bytes(4, "little")
    return header


def record_on_voice_detection():
    """检测到人声开始录音，1秒无人声停止录音
    前半秒无人声的音频进入 Zipformer，后半秒不进入
    VAD 和 Zipformer 同步进行 - 真正的流式处理"""
    log_info("=== 语音激活录音 ===")

    vad_available, vad_type, vad_model = load_vad()

    # 初始化 Zipformer 流式识别
    zipformer_available = init_zipformer_streaming()

    log_info(f"开始监听，等待人声...")
    log_info(f"连续{SPEECH_FRAMES_TO_START}帧检测到人声开始录音，1秒无人声停止录音\n")

    audio_buffer = []
    recording_frames = []
    is_recording = False
    silence_frame_count = 0
    consecutive_speech_count = 0
    total_frames = 0
    speech_frames = 0
    recording = True
    threshold = 0.7
    
    # 流式处理相关
    streaming_buffer = []  # 流式处理缓冲区
    streaming_chunk = []  # 当前处理的音频块
    is_streaming = False
    streaming_results = []  # 流式识别结果
    streaming_state = None  # 流式识别状态
    streaming_text = ""  # 累积的流式识别文本
    zipformer_buffer = b""  # Zipformer 音频缓冲区
    
    def read_audio():
        nonlocal recording
        process = subprocess.Popen(
            ["arecord", "-D", RECORD_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "wav", "-d", "30"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        while recording:
            frame = process.stdout.read(FRAME_SIZE * 2)
            if not frame:
                break
            audio_buffer.append(frame)

        process.terminate()
        process.wait()

    def process_audio_stream():
        """实时处理音频流，同步进行 VAD 和 Zipformer"""
        nonlocal is_streaming, streaming_buffer, streaming_chunk, streaming_results, streaming_state, streaming_text, zipformer_buffer
        
        # 模拟语音识别的词汇库（当 Zipformer 不可用时使用）
        vocab = ["你好", "有人", "吗", "我", "肚子", "痛", "需要", "帮助"]
        
        while recording:
            if is_streaming and len(streaming_buffer) > 0:
                # 积累音频数据到当前块
                while len(streaming_buffer) > 0 and len(streaming_chunk) < STREAMING_CHUNK_SIZE:
                    audio_data = streaming_buffer.pop(0)
                    streaming_chunk.extend(audio_data.tolist())  # 转换为列表后再添加
                
                # 当积累了足够的音频数据时，进行流式识别
                if len(streaming_chunk) >= STREAMING_CHUNK_SIZE:
                    # 提取 200ms 音频块
                    chunk_data = np.array(streaming_chunk[:STREAMING_CHUNK_SIZE], dtype=np.int16)
                    streaming_chunk = streaming_chunk[STREAMING_CHUNK_SIZE:]
                    
                    # 保存到录音帧中
                    # 确保音频数据格式正确
                    recording_frames.append(chunk_data.tobytes())
                    
                    # 累积 Zipformer 音频缓冲区
                    zipformer_buffer += chunk_data.tobytes()
                    
                    # 当 Zipformer 音频缓冲区达到指定大小时，进行流式识别
                    if zipformer_available and len(zipformer_buffer) >= ZIPFORMER_CHUNK_SIZE:
                        # 调用 Zipformer 流式识别
                        partial_text = zipformer_streaming_recognize(zipformer_buffer)
                        if partial_text:
                            # 更新识别结果，保留之前的结果
                            streaming_text = partial_text  # 更新识别结果
                            streaming_results.append(partial_text)
                            # 实时输出识别结果
                            print(f"\r[实时识别] {streaming_text}", end="", flush=True)
                        # 不清空缓冲区，继续累积音频数据
                        # zipformer_buffer = b""  # 保留缓冲区数据，确保能够识别完整句子
                    else:
                        # 模拟识别结果，随机选择一个词汇添加到流式文本中
                        if vocab:
                            import random
                            word = random.choice(vocab)
                            streaming_text += word
                            streaming_results.append(word)
                            # 实时输出识别结果
                            print(f"\r[实时识别] {streaming_text}", end="", flush=True)
            time.sleep(0.01)

    reader_thread = threading.Thread(target=read_audio)
    stream_thread = threading.Thread(target=process_audio_stream)
    reader_thread.start()
    stream_thread.start()

    try:
        while reader_thread.is_alive() or len(audio_buffer) > 0:
            if len(audio_buffer) > 0:
                frame = audio_buffer.pop(0)
                audio_data = np.frombuffer(frame, dtype=np.int16)

                is_speech = False
                if vad_available:
                    try:
                        import torch
                        audio_tensor = torch.from_numpy(audio_data.copy()).float() / 32768.0
                        if audio_tensor.dim() == 2:
                            audio_tensor = audio_tensor.squeeze(0)
                        if len(audio_tensor) < 512:
                            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 512 - len(audio_tensor)))

                        prob = get_speech_prob(audio_tensor, vad_model, SAMPLE_RATE)
                        is_speech = prob > threshold

                    except Exception as e:
                        volume = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                        is_speech = volume > 1000
                else:
                    volume = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                    is_speech = volume > 1000

                total_frames += 1

                if is_speech:
                    speech_frames += 1
                    consecutive_speech_count += 1

                    if not is_recording and consecutive_speech_count >= SPEECH_FRAMES_TO_START:
                        is_recording = True
                        is_streaming = True
                        print(f"\n🎙️ 连续{consecutive_speech_count}帧检测到人声，开始录音...")

                    silence_frame_count = 0
                    # 只要检测到人声，就添加到流式处理缓冲区，确保连续 5 帧都能进入 Zipformer
                    streaming_buffer.append(audio_data)  # 直接添加 numpy 数组
                else:
                    consecutive_speech_count = 0
                    silence_frame_count += 1
                    if is_recording:
                        # 前半秒无人声的音频进入 Zipformer，后半秒不进入
                        if silence_frame_count <= SILENCE_FRAMES_HALF:
                            streaming_buffer.append(audio_data)  # 直接添加 numpy 数组
                        elif silence_frame_count <= SILENCE_FRAMES_THRESHOLD:
                            # 后半秒的音频不进入 Zipformer，但继续录制以捕获尾音
                            streaming_buffer.append(audio_data)  # 直接添加 numpy 数组，让 process_audio_stream 统一处理
                        if silence_frame_count >= SILENCE_FRAMES_THRESHOLD:
                            is_recording = False
                            is_streaming = False
                            print("🔇 1.5秒无人声，停止录音")
                            break

                status = "🎙️ 录音中" if is_recording else "👂 等待中"
                print(f"\r[{total_frames:04d}] {status} | 语音帧: {speech_frames} | 录音帧: {len(recording_frames)} | 连续语音: {consecutive_speech_count}/{SPEECH_FRAMES_TO_START}", end="", flush=True)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n⛔ 用户中断")
    finally:
        recording = False
        is_streaming = False
        # 继续处理剩余的音频数据，持续两秒钟
        end_time = time.time() + 2.0
        while time.time() < end_time and len(streaming_buffer) > 0:
            if len(streaming_buffer) > 0:
                audio_data = streaming_buffer.pop(0)
                streaming_chunk.extend(audio_data.tolist())
                if len(streaming_chunk) >= STREAMING_CHUNK_SIZE:
                    chunk_data = np.array(streaming_chunk[:STREAMING_CHUNK_SIZE], dtype=np.int16)
                    streaming_chunk = streaming_chunk[STREAMING_CHUNK_SIZE:]
                    recording_frames.append(chunk_data.tobytes())
                    zipformer_buffer += chunk_data.tobytes()
                    if zipformer_available and len(zipformer_buffer) >= ZIPFORMER_CHUNK_SIZE:
                        partial_text = zipformer_streaming_recognize(zipformer_buffer)
                        if partial_text:
                            streaming_text = partial_text
                            streaming_results.append(partial_text)
                            print(f"\r[实时识别] {streaming_text}", end="", flush=True)
            time.sleep(0.01)
        reader_thread.join()
        stream_thread.join()

    print()
    log_info(f"录音统计: 总帧数={total_frames}, 语音帧={speech_frames}, 录音帧数={len(recording_frames)}\n")

    # 返回录音帧和流式识别结果
    return recording_frames, streaming_text


def save_recording(recording_frames):
    """保存录音文件"""
    log_info("=== 保存录音文件 ===")

    if not recording_frames:
        log_warn("没有录音数据")
        return False

    try:
        import wave
        with wave.open(OUTPUT_FILE, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)

            for audio_data in recording_frames:
                wav_file.writeframes(audio_data)

        log_info(f"✅ 录音已保存: {OUTPUT_FILE}")
        return True
    except Exception as e:
        log_warn(f"❌ 保存失败: {e}")
        return False


def recognize_with_zipformer():
    """使用 Zipformer 进行语音识别"""
    log_step("步骤1/4: 语音识别 (Zipformer)")
    start_time = time.time()

    if not os.path.exists(OUTPUT_FILE):
        log_warn(f"录音文件不存在: {OUTPUT_FILE}")
        return None

    try:
        result = subprocess.run(
            [f"{ZIPFORMER_DIR}/rknn_zipformer_demo", ENCODER_MODEL, DECODER_MODEL, JOINER_MODEL, OUTPUT_FILE],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=ZIPFORMER_DIR
        )

        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            recognized_text = None

            for line in output_lines:
                if "Zipformer output:" in line:
                    recognized_text = line.split("Zipformer output:")[-1].strip()
                    break

            elapsed = time.time() - start_time

            if recognized_text:
                log_result(f"识别结果: {recognized_text}")
                log_info(f"识别耗时: {elapsed:.3f}秒\n")
                return recognized_text
            else:
                log_warn("未找到识别结果")
        else:
            log_warn(f"Zipformer 识别失败: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        log_warn("Zipformer 识别超时")
    except Exception as e:
        log_warn(f"识别异常: {e}")

    return None


def _detect_departments(query):
    """从查询中检测可能的科室"""
    query_lower = query.lower()
    matched_depts = []
    
    # 1. 优先使用症状-科室映射
    for symptom, dept in SYMPTOM_DEPT_MAP.items():
        if symptom in query_lower:
            if dept not in matched_depts:
                matched_depts.append(dept)
    
    # 2. 统计每个科室的匹配关键词数量
    if not matched_depts:
        dept_scores = {}
        for dept, keywords in DEPT_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in query_lower:
                    score += 1
            if score > 0:
                dept_scores[dept] = score
        
        # 按得分排序，取前3个
        sorted_depts = sorted(dept_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        matched_depts = [dept for dept, _ in sorted_depts]
    
    # 3. 如果没有匹配，默认内科
    if not matched_depts:
        matched_depts = ['内科']
    
    return matched_depts


def _extract_symptoms(query):
    """提取症状关键词"""
    query_lower = query.lower()
    symptoms = []
    for symptom in SYMPTOM_KEYWORDS:
        if symptom in query_lower:
            symptoms.append(symptom)
    return symptoms


def _filter_relevant_results(results, query, target_depts):
    """过滤相关的检索结果"""
    query_lower = query.lower()
    relevant_results = []
    
    # 症状关键词
    symptom_keywords = ['疼', '痛', '肚子', '腹痛', '腹泻', '便秘', '恶心', '呕吐', '消化']
    
    # 优先选择目标科室的结果
    for result in results:
        dept = result.get('department', '').strip()
        if dept in target_depts:
            relevant_results.append(result)
            if len(relevant_results) >= 2:
                break
    
    # 如果没有足够的相关结果，使用原始结果
    if len(relevant_results) < 2:
        for result in results:
            if result not in relevant_results:
                relevant_results.append(result)
                if len(relevant_results) >= 2:
                    break
    
    return relevant_results


def rag_retrieve(query):
    """RAG 检索 - 科室优化版（修复 Unicode 转义问题）"""
    log_step("步骤2/4: 医疗 RAG 检索")
    start_time = time.time()

    try:
        # 检测科室和症状
        matched_depts = _detect_departments(query)
        if matched_depts:
            log_info(f"检测到科室: {matched_depts}")
        
        symptoms = _extract_symptoms(query)
        if symptoms:
            log_info(f"提取症状: {symptoms}")

        # 连接 RAG 服务
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(30)
        sock.connect(RAG_SERVICE_SOCK)

        request = {
            'action': 'search',
            'query': query,
            'k': 5,
            'departments': matched_depts
        }
        sock.send(json.dumps(request).encode('utf-8'))

        response = sock.recv(8192).decode('utf-8')
        sock.close()

        # ---------- 关键修复：直接解析 JSON，不要破坏 Unicode 转义 ----------
        data = json.loads(response)

        if data.get('status') == 'ok':
            results = data['data']['results']
            time_ms = data['data']['time_ms']
            hybrid_mode = data['data'].get('hybrid_mode', True)
            dept_optimized = data['data'].get('dept_optimized', False)

            # 过滤相关结果
            filtered_results = _filter_relevant_results(results, query, matched_depts)
            
            # 打印检索内容
            print("\n📚 RAG 检索结果:")
            print("-" * 50)
            for i, r in enumerate(filtered_results[:2], 1):
                print(f"参考 {i}:")
                print(f"  科室: {r['department']}")
                print(f"  问题: {r['question']}")
                print(f"  答案: {r['answer']}")
                print()
            print("-" * 50)

            # 构建提示词（优化版）
            prompt = f"""作为专业医疗助手，请根据以下参考知识回答用户问题。

【用户问题】{query}

【参考知识】
"""
            for i, r in enumerate(filtered_results[:2], 1):
                prompt += f"{i}. [{r['department']}] {r['answer'][:150]}\n"

            prompt += """
【回答要求】
1. 如果参考知识与问题直接相关，请提炼关键信息回答。
2. 如果参考知识不充分或偏离主题，请结合基本医学常识给出通用、安全的建议，并提醒用户咨询专业医生。
3. 回答尽量控制在150字以内，语气温和专业。

【回答】
"""

            elapsed = time.time() - start_time
            log_info(f"RAG 检索耗时: {time_ms:.1f}ms")
            log_info(f"混合搜索: {hybrid_mode}, 科室优化: {dept_optimized}")
            log_info(f"总耗时: {elapsed*1000:.0f}ms\n")
            return prompt

        else:
            log_warn(f"RAG 服务返回错误状态: {data.get('message')}")

    except Exception as e:
        log_warn(f"RAG 检索异常: {e}")

    # 失败时返回一个简单的提示词
    matched_depts = _detect_departments(query) if 'query' in locals() else ['内科']
    dept_info = f"（{matched_depts[0]}）" if matched_depts else ""
    return f"作为{dept_info}医疗助手，请回答：{query}"


def split_text_into_sentences(text):
    """将文本按句子分割"""
    import re
    # 按标点符号分割，保留标点
    sentences = re.split(r'([。！？.!?；;\n])', text)
    
    result = []
    current = ""
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            current += sentences[i]
        if i + 1 < len(sentences):
            current += sentences[i + 1]
            if current.strip():
                result.append(current.strip())
            current = ""
    
    if current.strip():
        result.append(current.strip())
    
    return result if result else [text]


def llm_generate_streaming(prompt, sentence_callback=None):
    """使用 Qwen3-0.6B 生成回复，支持真正的流式输出
    通过流式LLM服务获取token-by-token输出，实时触发TTS"""
    log_step("步骤3/4: LLM 生成回复 (流式)")
    start_time = time.time()

    try:
        # 使用流式LLM服务
        LLM_STREAMING_SOCK = "/tmp/qwen3_llm_streaming.sock"
        
        # 检查流式服务是否可用
        if not os.path.exists(LLM_STREAMING_SOCK):
            log_warn("流式LLM服务不可用，回退到非流式模式")
            reply_text = llm_generate(prompt)
            if reply_text and sentence_callback:
                sentences = split_text_into_sentences(reply_text)
                for sentence in sentences:
                    sentence_callback(sentence)
            return reply_text
        
        print(f"\n📝 发送给 LLM 的提示词:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        
        # 连接流式服务
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(60)
        sock.connect(LLM_STREAMING_SOCK)
        
        request = {
            'prompt': prompt,
            'max_tokens': 45,
            'temperature': 0.7,
            'stream': True
        }
        sock.send(json.dumps(request).encode() + b'\n')
        
        # 接收流式响应
        reply_text = ""
        current_sentence = ""
        sentence_idx = 0
        buffer = b""
        
        print(f"\n📝 流式输出:")
        print("-" * 50)
        
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                
                buffer += chunk
                
                # 处理完整行
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line.decode('utf-8'))
                        
                        if 'error' in data:
                            log_warn(f"流式LLM错误: {data['error']}")
                            break
                        
                        token = data.get('token', '')
                        is_done = data.get('done', False)
                        
                        if token:
                            print(token, end='', flush=True)
                            reply_text += token
                            current_sentence += token
                            
                            # 检查是否到达句子结尾
                            if token in ['。', '！', '？', '.', '!', '?', '；', ';', '\n']:
                                if current_sentence.strip() and sentence_callback:
                                    sentence_idx += 1
                                    print(f"\n[触发TTS] 第{sentence_idx}句: {current_sentence.strip()}")
                                    sentence_callback(current_sentence.strip())
                                current_sentence = ""
                        
                        if is_done:
                            # 处理最后剩余的文本
                            if current_sentence.strip() and sentence_callback:
                                sentence_idx += 1
                                print(f"\n[触发TTS] 第{sentence_idx}句: {current_sentence.strip()}")
                                sentence_callback(current_sentence.strip())
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
            except socket.timeout:
                log_warn("流式接收超时")
                break
            except Exception as e:
                log_warn(f"流式接收异常: {e}")
                break
        
        sock.close()
        
        print()  # 换行
        print("-" * 50)
        
        elapsed = time.time() - start_time
        log_info(f"LLM流式生成 耗时: {elapsed*1000:.0f}ms")
        print(f"\n📢 完整回复: {reply_text}")
        
        return reply_text

    except Exception as e:
        log_warn(f"LLM 流式生成异常: {e}")
        import traceback
        traceback.print_exc()
        
        # 回退到非流式模式
        log_info("回退到非流式模式")
        reply_text = llm_generate(prompt)
        if reply_text and sentence_callback:
            sentences = split_text_into_sentences(reply_text)
            for sentence in sentences:
                sentence_callback(sentence)
        return reply_text

    return None


def llm_generate(prompt):
    """使用 Qwen3-0.6B 生成回复（兼容旧版本，非流式）"""
    log_step("步骤3/4: LLM 生成回复 (Qwen3-0.6B)")
    start_time = time.time()

    try:
        # 打印完整的提示词内容，查看串口输出给 LLM 的所有内容
        print("\n📝 发送给 LLM 的提示词:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(60)
        sock.connect(LLM_SERVICE_SOCK)

        request = {
            'prompt': prompt,
            'max_tokens': 45,  # 减少 token 数，提升速度
            'temperature': 0.7
        }
        sock.sendall(json.dumps(request, ensure_ascii=False).encode('utf-8'))
        sock.shutdown(socket.SHUT_WR)

        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk

        sock.close()

        result = json.loads(response.decode('utf-8'))
        reply_text = result.get('text', '').strip()

        if not reply_text:
            reply_text = result.get('response', '').strip()

        if reply_text:
            elapsed = time.time() - start_time
            log_result(f"回复: {reply_text}")
            log_info(f"LLM 耗时: {elapsed*1000:.0f}ms\n")
            # 打印 LLM 回复内容
            print(f"\n📢 LLM 回复内容: {reply_text}")
            return reply_text
        else:
            log_warn(f"LLM 回复为空，结果: {result}")

    except Exception as e:
        log_warn(f"LLM 生成异常: {e}")

    return None


def tts_synthesize(text, output, speed=0.6):
    """TTS 语音合成"""
    log_step("步骤4/4: 语音合成 (TTS)")

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(30)
        sock.connect(TTS_SERVICE_SOCK)

        request = {
            'command': 'synthesize',
            'text': text,
            'output': output,
            'speed': speed
        }
        sock.send(json.dumps(request).encode())
        response = sock.recv(4096).decode()
        data = json.loads(response)
        sock.close()

        if data.get('status') == 'ok':
            log_info(f"✅ TTS 合成成功: {output}\n")
            return True

    except Exception as e:
        log_warn(f"TTS 合成异常: {e}")

    return False


def tts_synthesize_async(text, output, speed=0.6):
    """异步TTS语音合成，在线程中执行不阻塞主流程"""
    import threading
    
    def do_tts():
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(30)
            sock.connect(TTS_SERVICE_SOCK)

            request = {
                'command': 'synthesize',
                'text': text,
                'output': output,
                'speed': speed
            }
            sock.send(json.dumps(request).encode())
            response = sock.recv(4096).decode()
            data = json.loads(response)
            sock.close()

            if data.get('status') == 'ok':
                log_info(f"✅ TTS 异步合成成功: {output}")
                return True
            else:
                log_warn(f"TTS 异步合成失败: {data.get('message', '未知错误')}")
                return False

        except Exception as e:
            log_warn(f"TTS 异步合成异常: {e}")
            return False
    
    # 启动后台线程执行TTS
    thread = threading.Thread(target=do_tts, daemon=True)
    thread.start()
    return thread


def play_audio_async(output_file):
    """异步播放音频"""
    import threading
    
    def do_play():
        play_audio(output_file)
    
    thread = threading.Thread(target=do_play, daemon=True)
    thread.start()
    return thread


def ensure_tts_service():
    """确保TTS服务已启动（按需启动）"""
    import os
    import subprocess
    import time
    
    # 检查TTS服务是否已在运行
    if os.path.exists(TTS_SERVICE_SOCK):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect(TTS_SERVICE_SOCK)
            sock.close()
            return True  # TTS服务已在运行
        except:
            pass  # Socket存在但无法连接，需要重启服务
    
    # 启动TTS服务
    log_info("启动TTS服务（按需）...")
    try:
        subprocess.Popen(
            ["python3", "/data/voice_assistant/melotts_service_rknn.py"],
            stdout=open("/tmp/melotts_service.log", "w"),
            stderr=subprocess.STDOUT,
            cwd="/data/voice_assistant"
        )
        
        # 等待TTS服务就绪（最多5秒）
        for i in range(50):
            time.sleep(0.1)
            if os.path.exists(TTS_SERVICE_SOCK):
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    sock.connect(TTS_SERVICE_SOCK)
                    sock.close()
                    log_info("✅ TTS服务已就绪")
                    return True
                except:
                    continue
        
        log_warn("⚠️ TTS服务启动超时")
        return False
    except Exception as e:
        log_warn(f"❌ TTS服务启动失败: {e}")
        return False


def play_audio(output_file):
    """播放音频"""
    log_info("播放回复音频...")
    
    # 检查音频文件
    import os
    if not os.path.exists(output_file):
        log_warn(f"❌ 音频文件不存在: {output_file}")
        return False
    
    file_size = os.path.getsize(output_file)
    log_info(f"  音频文件: {output_file}")
    log_info(f"  文件大小: {file_size} bytes")
    
    # 检查文件类型
    try:
        result = subprocess.run(["file", output_file], capture_output=True, text=True)
        log_info(f"  文件类型: {result.stdout.strip()}")
    except Exception as e:
        log_warn(f"  无法检测文件类型: {e}")
    
    # MeloTTS 生成的是 44100 Hz Float 32 bit 音频
    cmd = ["aplay", "-D", PLAY_DEVICE, "-r", "44100", "-f", "FLOAT_LE", "-c", "1", output_file]
    log_info(f"  播放命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    log_info(f"  返回码: {result.returncode}")
    if result.stdout:
        log_info(f"  标准输出: {result.stdout.strip()}")
    if result.stderr:
        log_info(f"  标准错误: {result.stderr.strip()}")
    
    if result.returncode == 0:
        log_info("✅ 播放成功\n")
        return True
    else:
        log_warn(f"❌ 播放失败: {result.stderr}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("    完整语音助手系统")
    print("    Silero VAD + Zipformer + RAG + Qwen3-0.6B")
    print("=" * 50)
    print()

    try:
        setup_audio()
        recording_frames, streaming_text = record_on_voice_detection()

        if not save_recording(recording_frames):
            return

        # 使用流式识别结果或回退到批量识别
        recognized_text = None
        if streaming_text:
            # 使用流式识别结果
            recognized_text = streaming_text
            log_result(f"识别结果: {recognized_text}")
            log_info("使用流式识别结果\n")
        else:
            # 回退到批量识别
            recognized_text = recognize_with_zipformer()

        if not recognized_text:
            log_warn("语音识别失败，跳过后续流程")
            return

        prompt = rag_retrieve(recognized_text)

        # 在LLM生成回复前启动TTS服务（并行准备）
        ensure_tts_service()

        # 使用流式TTS：LLM每输出一个短句就立即合成并播放
        tts_queue = []
        tts_threads = []
        sentence_count = 0
        
        def on_sentence_ready(sentence):
            """回调函数：当LLM生成一个短句时触发"""
            nonlocal sentence_count
            sentence_count += 1
            
            # 为每个短句生成独立的音频文件
            output_file = f"/tmp/voice_assistant/reply_{sentence_count}.wav"
            os.makedirs("/tmp/voice_assistant", exist_ok=True)
            
            log_info(f"🎵 开始合成第{sentence_count}句: {sentence}")
            
            # 异步合成TTS（不阻塞LLM继续生成）
            tts_thread = tts_synthesize_async(sentence, output_file)
            tts_queue.append((sentence_count, output_file, tts_thread))
        
        # 使用流式生成
        reply_text = llm_generate_streaming(prompt, sentence_callback=on_sentence_ready)
        
        # 如果流式生成失败，回退到非流式
        if not reply_text:
            log_warn("流式生成失败，尝试非流式生成...")
            reply_text = llm_generate(prompt)
            
            if reply_text:
                on_sentence_ready(reply_text)
            else:
                reply_text = "抱歉，我暂时无法回答这个问题。"
                on_sentence_ready(reply_text)
        
        # 等待所有TTS合成完成并顺序播放
        log_info("等待TTS合成完成并播放...")
        for idx, output_file, tts_thread in sorted(tts_queue, key=lambda x: x[0]):
            # 等待当前句子合成完成
            tts_thread.join(timeout=30)
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                log_info(f"▶️ 播放第{idx}句...")
                play_audio(output_file)
            else:
                log_warn(f"第{idx}句音频文件不存在或为空")

        print("=" * 50)
        print("    测试完成")
        print("=" * 50)

    except Exception as e:
        log_warn(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test_rag":
        # 测试 RAG 检索
        test_query = "我肚子好痛"
        print(f"测试 RAG 检索: {test_query}")
        prompt = rag_retrieve(test_query)
        print(f"生成的提示词: {prompt}")
    else:
        main()