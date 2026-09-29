#!/usr/bin/env python3
"""
完整语音助手 - Silero VAD + Zipformer ASR + SenseVoice SER + RAG + Qwen3-0.6B + Matcha-TTS
- 使用 USB 声卡 card 0 录音
- 实时进行 Silero VAD 语音检测
- 连续5帧检测到人声开始录音，半秒无人声停止录音
- 使用 Zipformer 进行语音识别
- 使用 SenseVoice 进行情感识别（SER）
- 使用 RAG 检索增强（科室优化版）
- 使用 Qwen3-0.6B 生成回复
- 使用 Matcha-TTS 进行语音合成
- 使用 USB 音响 card 1 播放
"""
import os
import sys
import time
import threading
import socket
import json
import subprocess

sys.path.insert(0, "/data/sensevoice")
import queue
import re
import pty
import termios
import tty

# ── 全局音频播放锁 ──
_PLAY_LOCK = threading.Lock()

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_FRAMES_THRESHOLD = int(1.0 / (FRAME_DURATION_MS / 1000))  # 1.0秒无人声停止
SILENCE_FRAMES_HALF = int(0.2 / (FRAME_DURATION_MS / 1000))  # 前半秒进入 Zipformer
SILENCE_FRAMES_TAIL = int(0.8 / (FRAME_DURATION_MS / 1000))  # 尾部静音缓冲0.8s，确保最后一个字被完整录制
SPEECH_FRAMES_TO_START = 5

# 流式处理参数
STREAMING_CHUNK_MS = 30  # 30ms 逐帧识别（每帧直接送入 ASR）

# Zipformer 流式识别参数
ZIPFORMER_STREAMING = True  # 启用 Zipformer 流式识别
ZIPFORMER_CHUNK_SIZE = 6400  # 200ms 音频块（16kHz * 0.2s）
ZIPFORMER_WINDOW_SIZE = 9600  # 300ms 窗口大小

def _scan_audio_devices(list_cmd):
    """扫描 arecord -l / aplay -l，返回 [(card, device, name), ...]"""
    devices = []
    try:
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=5)
        # line format: "card X: ShortName [DeviceName], device Y: USB Audio [USB Audio]"
        for line in result.stdout.split('\n'):
            m = re.match(r'card\s+(\d+):\s+.*?\[(.+?)\].*?device\s+(\d+):', line)
            if m:
                devices.append((m.group(1), m.group(3), m.group(2)))
    except Exception as e:
        print(f"[音频] 设备扫描失败: {e}", file=sys.stderr, flush=True)
    return devices

def _find_device(devices, name_patterns, default):
    """按设备名（子串匹配）查找设备，返回 plughw:X,Y
       name_patterns 可以是字符串或列表，匹配任意一个即返回"""
    if isinstance(name_patterns, str):
        name_patterns = [name_patterns]
    for card, device, name in devices:
        for pattern in name_patterns:
            if pattern in name:
                dev = f"plughw:{card},{device}"
                print(f"[音频] 匹配到 {name}: {dev}", file=sys.stderr, flush=True)
                return dev
    print(f"[音频] 未匹配到 {name_patterns}，使用默认 {default}", file=sys.stderr, flush=True)
    return default

# 按设备名匹配（支持多个名字，兼容不同 USB 麦克风/音箱）
RECORD_DEVICE_NAMES = ["AB13X", "UACDemo"]  # USB 麦克风
PLAY_DEVICE_NAMES = ["USB2.0 Device"]             # USB 音箱

_rec_devices = _scan_audio_devices(['arecord', '-l'])
RECORD_DEVICE = _find_device(_rec_devices, RECORD_DEVICE_NAMES, 'plughw:0,0')

_play_devices = _scan_audio_devices(['aplay', '-l'])
PLAY_DEVICE = _find_device(_play_devices, PLAY_DEVICE_NAMES, 'plughw:1,0')
OUTPUT_FILE = "/data/voice_assistant/test_recording.wav"
ZIPFORMER_DIR = "/data/zipformer"
ENCODER_MODEL = f"{ZIPFORMER_DIR}/model/encoder-epoch-99-avg-1.rknn"
DECODER_MODEL = f"{ZIPFORMER_DIR}/model/decoder-epoch-99-avg-1.rknn"
JOINER_MODEL = f"{ZIPFORMER_DIR}/model/joiner-epoch-99-avg-1.rknn"
SHERPA_ONNX_DIR = "/userdata/sherpa-onnx/install"
SHERPA_ONNX_ALSA = "/data/sherpa-onnx/install/bin/sherpa-onnx-alsa"
SHERPA_ONNX_LIB = "/data/sherpa-onnx/install/lib"
TOKENS_PATH = f"{ZIPFORMER_DIR}/model/vocab.txt"

# 无输入超时（秒）- 超过此时间重置 LLM 上下文（继续等待下一轮）
NO_INPUT_TIMEOUT = 900  # 15分钟

# 各阶段计时器（从说话结束到播放首句）
TIMING = {}
VOICE_ASSISTANT_DIR = "/userdata/voice_assistant"
MEDICAL_RAG_DIR = "/data/medical_rag_full"
LLM_SERVICE_SOCK = "/tmp/qwen3_llm.sock"
RAG_SERVICE_SOCK = "/tmp/rag_optimized.sock"
# TTS 服务 Socket（可切换 MeloTTS 或 Matcha-TTS）
TTS_SERVICE_SOCK = "/tmp/matcha_tts.sock"   # Matcha-TTS
PATIENT_RAG_SOCK = "/tmp/patient_rag.sock"          # Patient RAG（患者模拟器）
SER_SERVICE_SOCK = "/tmp/sensevoice_server.sock"    # SenseVoice SER（情感识别）

# 科室关键词映射（根据文档优化版）
DEPT_KEYWORDS = {
    '内科': ['内科', '心脏', '心血管', '高血压', '糖尿病', '胃病', '胃炎', '肺炎', '感冒', '发烧', '咳嗽', '血压', '血糖', '肚子疼', '腹痛', '腹泻', '便秘', '恶心', '呕吐', '消化', '肠道'],
    '外科': ['外科', '手术', '骨折', '伤口', '缝合', '切除', '阑尾', '胆囊', '结石', '刀口', '创伤', '烧伤', '烫伤', '跌打', '损伤'],
    '儿科': ['儿科', '儿童', '小孩', '婴儿', '宝宝', '幼儿', '新生儿', '疫苗', '孩子', '小儿', '发烧', '咳嗽', '疫苗'],
    '妇产科': ['妇产', '妇科', '产科', '怀孕', '孕妇', '分娩', '月经', '痛经', '子宫', '卵巢', '孕期', '产妇', '哺乳期', '备孕', '流产'],
    '皮肤科': ['皮肤', '皮肤科', '湿疹', '痤疮', '痘痘', '皮炎', '银屑病', '白癜风', '皮疹', '皮肤过敏', '荨麻疹', '红斑'],
    '眼科': ['眼科', '眼睛', '视力', '近视', '远视', '散光', '白内障', '青光眼', '结膜炎', '角膜炎'],
    '耳鼻喉科': ['耳鼻喉', '耳朵', '听力', '耳鸣', '中耳炎', '鼻子', '鼻炎', '鼻窦炎', '喉咙', '咽炎', '扁桃体'],
    '口腔科': ['口腔', '牙齿', '龋齿', '牙周病', '口腔溃疡', '口臭', '牙龈炎', '拔牙', '补牙'],
    '精神科': ['精神', '抑郁', '焦虑', '失眠', '精神分裂', '躁狂', '强迫症', '恐惧症'],
    '心理科': ['心理', '心理咨询', '压力', '情绪', '心理障碍', '心理问题'],
    '中医科': ['中医', '中药', '针灸', '拔罐', '气血', '肾虚', '肝火', '湿气', '调理', '中药', '中医', '针灸'],
    '骨科': ['骨科', '骨头', '关节', '骨折', '骨质疏松', '关节炎', '腰椎', '颈椎', '肩周炎'],
    '肿瘤科': ['肿瘤', '癌症', '癌', '化疗', '放疗', '肿块', '良性', '恶性', '瘤', '癌症', '肿瘤', '化疗'],
    '神经科': ['神经', '头痛', '头晕', '癫痫', '帕金森', '神经炎', '神经痛', '脑梗塞', '脑出血'],
    '心血管科': ['心血管', '心脏', '冠心病', '心肌梗塞', '心力衰竭', '心律失常', '心脏杂音'],
    '消化科': ['消化', '肠胃', '胃痛', '腹痛', '腹泻', '便秘', '胃炎', '胃溃疡', '结肠炎', '痔疮'],
    '呼吸科': ['呼吸', '肺炎', '哮喘', '支气管炎', '肺气肿', '咳嗽', '呼吸困难', '胸痛'],
}

# 常见症状关键词
SYMPTOM_KEYWORDS = [
    '疼', '痛', '难受', '不舒服', '发烧', '发热', '咳嗽', '咳痰', '呼吸困难', '胸闷',
    '头晕', '头痛', '恶心', '呕吐', '腹泻', '便秘', '腹痛', '腹胀', '皮疹', '瘙痒',
    '失眠', '疲劳', '乏力', '出汗', '寒战', '关节痛', '肌肉痛', '视力模糊', '听力下降'
]

# 症状-科室映射
SYMPTOM_DEPT_MAP = {
    '肚子疼': '内科',
    '腹痛': '内科',
    '腹泻': '内科',
    '便秘': '内科',
    '恶心': '内科',
    '呕吐': '内科',
    '发烧': '内科',
    '咳嗽': '内科',
    '头痛': '内科',
    '头晕': '内科',
    '胸闷': '内科',
    '呼吸困难': '内科',
    '关节痛': '骨科',
    '肌肉痛': '骨科',
    '皮疹': '皮肤科',
    '瘙痒': '皮肤科',
    '视力模糊': '眼科',
    '听力下降': '耳鼻喉科',
    '耳鸣': '耳鼻喉科',
    '牙痛': '口腔科',
}


def log_info(msg):
    print(f"[INFO] {msg}", flush=True)

def log_step(msg):
    print(f"[STEP] {msg}", flush=True)

def log_result(msg):
    print(f"[RESULT] {msg}", flush=True)

def log_warn(msg):
    print(f"[WARN] {msg}", flush=True)

def log_timing(label, description=""):
    """记录时间戳到 TIMING 字典"""
    TIMING[label] = time.time()

def _warmup_asr():
    """预热 ASR 模型：启动 sherpa-onnx-alsa 加载完成后立即退出，避免首次说话丢失"""
    log_info("预热 ASR 模型...")
    os.system('pkill -9 -f sherpa-onnx-alsa 2>/dev/null')
    time.sleep(0.3)
    import subprocess as _sp
    try:
        _warmup_cmd = [
            SHERPA_ONNX_ALSA,
            f'--tokens={TOKENS_PATH}',
            f'--encoder={ENCODER_MODEL}',
            f'--decoder={DECODER_MODEL}',
            f'--joiner={JOINER_MODEL}',
            '--provider=rknn', '--num-threads=-4',
            '--decoding-method=greedy_search',
            RECORD_DEVICE
        ]
        _proc = _sp.Popen(
            _warmup_cmd,
            stdout=_sp.DEVNULL, stderr=_sp.PIPE,
            env={**os.environ,
                 'LD_LIBRARY_PATH': f"{SHERPA_ONNX_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                 'RKNN_CORE_MASK': '6'}
        )
        # 等待 "Started!" 表示模型已加载，录音已就绪
        for raw_line in iter(_proc.stderr.readline, b''):
            line = raw_line.decode('utf-8', errors='replace').strip()
            if 'Started!' in line or 'started' in line.lower():
                log_info("ASR 模型加载完成")
                break
        _proc.terminate()
        try:
            _proc.wait(timeout=3)
        except:
            _proc.kill()
    except Exception as e:
        log_warn(f"ASR 预热失败: {e}")
    log_info("ASR 预热结束")

def setup_audio():
    """设置音频设备"""
    log_info("设置音频设备...")
    print(f"  录音设备: {RECORD_DEVICE}")
    print(f"  播放设备: {PLAY_DEVICE}")
    # 设置音量为 40%
    subprocess.run(["amixer", "-c", "1", "set", "PCM", "40%"], capture_output=True)
    log_info("音频设备设置完成 (音量: 40%)\n")


def record_on_voice_detection(ser_ctx=None):
    """启动 sherpa-onnx-alsa 子进程，PTY 强制行缓冲
       ser_ctx: SER 上下文，用于提前推理优化"""
    log_info("=== 启动 sherpa-onnx-alsa 实时语音识别 ===")

    cmd = [
        SHERPA_ONNX_ALSA,
        f'--tokens={TOKENS_PATH}',
        f'--encoder={ENCODER_MODEL}',
        f'--decoder={DECODER_MODEL}',
        f'--joiner={JOINER_MODEL}',
        '--provider=rknn',
        '--num-threads=-4',
        '--decoding-method=greedy_search',
            '--rule2-min-trailing-silence=0.8',
            RECORD_DEVICE
    ]

    log_info("语音识别已启动，请说话...")

    os.system('pkill -9 -f sherpa-onnx-alsa 2>/dev/null')
    time.sleep(0.3)

    # ── PTY: C 程序 stderr → 行缓冲 ──
    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd)          # 禁止终端转义处理

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=slave_fd,
            env={**os.environ,
                 'LD_LIBRARY_PATH': f"{SHERPA_ONNX_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                 'RKNN_CORE_MASK': '6'}
        )
    finally:
        os.close(slave_fd)

    _started = [False]
    _seg_text = [""]
    _all_texts = [""]
    _have_final = [False]
    _done = [False]
    _reader_lines = []
    _last_data_time = [0.0]
    _ser_triggered = [False]

    # ── stderr reader ──
    #  sherpa-onnx 用多个 fprintf 分段发送，文本可能跨 chunk 无分隔符
    #  \rN:text\x1b[2K\r  = 中间更新 (\r 结尾)
    #  \rN:text\n          = 最终结果 (\n 结尾)
    def _extract_text(line, is_final):
        """从一行识别结果中提取文本"""
        clean = re.sub(r'\x1b\[[\d;]*[A-Za-z]', '', line)
        colon_pos = clean.find(':')
        if colon_pos > 0 and clean[:colon_pos].isdigit():
            new_text = clean[colon_pos + 1:]
            _seg_text[0] = new_text
            if len(new_text) > len(_all_texts[0]):
                _all_texts[0] = new_text
            if is_final:
                _have_final[0] = True

    def reader():
        buf = ""
        try:
            while not _done[0]:
                raw = os.read(master_fd, 4096)
                if not raw:
                    break
                _last_data_time[0] = time.time()
                buf += raw.decode('utf-8', errors='replace')
                # 按 \n/\r 分割处理完整行
                while True:
                    nl = buf.find('\n')
                    cr = buf.find('\r')
                    if nl < 0 and cr < 0:
                        break
                    if nl >= 0 and (cr < 0 or nl < cr):
                        line, buf = buf[:nl], buf[nl + 1:]
                        is_final = True
                    else:
                        line, buf = buf[:cr], buf[cr + 1:]
                        is_final = False
                    line = line.strip()
                    if not line:
                        continue
                    _reader_lines.append(line)
                    if 'Started!' in line:
                        _started[0] = True
                        continue
                    _extract_text(line, is_final)
                # 从剩余 buf 提取最新文本（可能缺尾部分隔符）
                if buf.strip():
                    m = re.search(r'(\d+):([^\r\n]*)', buf)
                    if m:
                        raw_text = m.group(2).strip()
                        raw_text = re.sub(r'\x1b\[[\d;]*[A-Za-z]', '', raw_text)
                        if raw_text:
                            _seg_text[0] = raw_text
                            if len(raw_text) > len(_all_texts[0]):
                                _all_texts[0] = raw_text
        except OSError:
            pass  # PTY 断开时正常
        # 缓冲区剩余数据（子进程退出后）
        if buf.strip():
            _extract_text(buf.strip(), False)

    thr = threading.Thread(target=reader, daemon=True)
    thr.start()

    # ── 等待 "Started!" ──
    _t0 = time.time()
    while not _started[0]:
        if proc.poll() is not None:
            log_warn("sherpa-onnx 进程崩溃")
            _done[0] = True
            try:
                os.close(master_fd)
            except OSError:
                pass
            return [], ""
        if time.time() - _t0 > 60:
            log_warn("ASR 启动超时")
            _done[0] = True
            try:
                os.close(master_fd)
            except OSError:
                pass
            return [], ""
        time.sleep(0.1)

    # ── 等待 ASR 结果 ──
    #  退出: 1) \n 到达（endpoint 触发）
    #        2) ASR 连续 800ms 没发数据 && 有 >=2 字文本
    #        3) NO_INPUT_TIMEOUT（安全网）
    _t1 = time.time()
    while time.time() - _t1 < NO_INPUT_TIMEOUT:
        if _have_final[0]:
            break
        
        # SER: 300ms 静音触发提前推理（与 ASR 尾部静音并行）
        # 条件：有文本内容（大于4字）+ 300ms 静音
        if ser_ctx and not _ser_triggered[0] and _last_data_time[0] > 0 and len(_all_texts[0]) > 4:
            if time.time() - _last_data_time[0] >= 0.3:
                _ser_triggered[0] = True
                _do_ser_early(ser_ctx)
        
        if len(_all_texts[0]) >= 2 and _last_data_time[0] > 0:
            if time.time() - _last_data_time[0] >= 0.8:
                break
        time.sleep(0.05)

    seg_elapsed = time.time() - _t1

    text = _all_texts[0] if _all_texts[0] else (_seg_text[0] if _seg_text[0] else "")
    # 清理 ANSI 转义序列（sherpa-onnx 会输出 \x1b[2K 等控制码）
    text = re.sub(r'\x1b\[[\d;]*[A-Za-z]', '', text)
    text = text.replace('\r', '').replace('\n', ' ').strip()
    if text:
        print(f"\n>>> [识别] 「{text}」 ({seg_elapsed:.1f}s)\n", flush=True)
    else:
        log_warn("未检测到语音或超时")

    _done[0] = True
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except:
        proc.kill()
        proc.wait()
    try:
        os.close(master_fd)
    except OSError:
        pass

    return [], text



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
            prompt = f"""作为专业医疗助手，请根据以下参考知识回答用户问题。请只使用中文，不要输出任何英文内容。

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


def llm_generate(prompt):
    """使用非流式 LLM 接口（连接 /tmp/qwen3_llm.sock），返回完整回复"""
    log_step("步骤3/4: LLM 生成 (Qwen3-0.6B)")
    start_time = time.time()

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect("/tmp/qwen3_llm.sock")

        request = {
            'prompt': prompt,
            'max_tokens': 60,
            'temperature': 0.7
        }
        sock.sendall(json.dumps(request, ensure_ascii=False).encode('utf-8') + b'\n')

        # 读取响应（旧服务不带换行符结尾，使用 timeout + recv）
        data = b""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
            except socket.timeout:
                break

        sock.close()

        result = json.loads(data.decode('utf-8'))
        response = result.get('response', '').strip()

        elapsed = time.time() - start_time
        if response:
            log_result(f"回复: {response}")
        else:
            log_warn("LLM 返回空响应")
        log_info(f"LLM 耗时: {elapsed*1000:.0f}ms\n")
        return response

    except Exception as e:
        log_warn(f"LLM 生成异常: {e}")
        return None


def llm_generate_streaming(prompt):
    """流式 LLM 生成（连接 /tmp/qwen3_llm.sock），逐句 yield"""
    log_step("步骤3/4: LLM 流式生成")
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect("/tmp/qwen3_llm.sock")

        request = {
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": True
        }
        sock.sendall(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")

        first_token = True
        buf = b""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    obj = json.loads(line.decode("utf-8"))
                    if "sentence" in obj:
                        if first_token:
                            TIMING['llm_first_token'] = time.time()
                            first_token = False
                        for char in obj["sentence"]:
                            yield char
                    elif "done" in obj:
                        full = obj.get("response", "")
                        log_result(f"回复: {full[:100]}")
                        return
                    elif "error" in obj:
                        log_warn(f"LLM 错误: {obj}")
                        return
            except socket.timeout:
                break
        sock.close()
    except Exception as e:
        log_warn(f"LLM 流式异常: {e}")





def _trim_wav_tail(path, threshold=300, keep_ms=30):
    """去掉 WAV 文件尾部低于阈值的静音噪声"""
    try:
        import struct
        with open(path, 'rb') as f:
            data = f.read()
        # 只处理 16-bit PCM
        if data[34] != 16:  # bits_per_sample
            return
        nchannels = struct.unpack('<H', data[22:24])[0]
        sample_rate_val = struct.unpack('<I', data[24:28])[0]
        data_start = 44 if data[12:16] == b'fmt ' else struct.unpack('<I', data[40:44])[0] + 44
        samples = struct.unpack_from(f'<{len(data[data_start:])//2}h', data, data_start)

        # 从末尾向前找最后一个高于阈值的样本
        last_ok = len(samples) - 1
        while last_ok > 0 and abs(samples[last_ok]) < threshold:
            last_ok -= 1

        # 保留 keep_ms 的尾音缓冲，避免骤然截断
        keep_samples = int(sample_rate_val * keep_ms / 1000) * nchannels
        trim_end = min(last_ok + keep_samples + 1, len(samples))
        trimmed = samples[:trim_end]

        # 重写 WAV
        import io
        import wave
        with io.BytesIO() as buf:
            with wave.open(buf, 'wb') as w:
                w.setnchannels(nchannels)
                w.setsampwidth(2)
                w.setframerate(sample_rate_val)
                w.writeframes(struct.pack(f'<{len(trimmed)}h', *trimmed))
            with open(path, 'wb') as f:
                f.write(buf.getvalue())
    except Exception:
        pass  # 修剪失败不影响主流程


def tts_synthesize(text, output, speed=1.0, noise_scale=0.667):
    """TTS 语音合成（Matcha-TTS 常驻服务，模型在内存中）"""
    log_step("步骤4/4: 语音合成 (TTS)")

    _out_dir = os.path.dirname(output)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)

    if not text.strip():
        log_warn("文本为空，跳过 TTS")
        return False

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(30)
        sock.connect(TTS_SERVICE_SOCK)

        request = {
            'command': 'synthesize',
            'text': text,
            'output': output,
            'speed': speed,
            'noise_scale': noise_scale,
        }

        sock.send(json.dumps(request, ensure_ascii=False).encode())
        response = sock.recv(4096).decode()
        data = json.loads(response)
        sock.close()

        if data.get('status') == 'ok':
            log_info(f"✅ TTS 合成成功: {output}")
            return True

    except Exception as e:
        log_warn(f"TTS 合成异常: {e}")

    return False



def play_audio(output_file):
    """播放音频（自动检测采样率）- 直接播放 WAV 文件，带全局播放锁"""
    log_info("播放回复音频...")
    import os
    if not os.path.exists(output_file):
        log_warn("音频文件不存在: {}".format(output_file))
        return False

    file_size = os.path.getsize(output_file)
    log_info("  音频文件: {}".format(output_file))
    log_info("  文件大小: {} bytes".format(file_size))

    with _PLAY_LOCK:
        try:
            if 'play_start' not in TIMING:
                TIMING['play_start'] = time.time()
            aplay_cmd = ["aplay", "-D", PLAY_DEVICE, "-q", output_file]
            subprocess.run(aplay_cmd, timeout=120, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            log_info("播放成功\n")
            return True
        except subprocess.TimeoutExpired:
            log_warn("播放超时")
        except Exception as e:
            log_warn("播放异常: {}".format(e))
    return False

def tts_warmup():
    """确认 TTS 服务在线（melotts_persistent 已在启动时加载模型，仅 ping 检测）"""
    log_info("TTS 服务检测（ping）...")
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(TTS_SERVICE_SOCK)
        request = {'command': 'ping'}
        sock.send(json.dumps(request).encode())

        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if b"}" in response:
                break
        sock.close()

        data = json.loads(response.decode())
        if data.get('status') == 'ok':
            log_info("TTS 服务在线")
        else:
            log_warn("TTS 服务响应异常: {}".format(data))
    except Exception as e:
        log_warn("TTS 服务检测失败: {}（不影响运行，TTS 可能未就绪）".format(e))


def tts_warmup_async():
    """发送 ping 检查 TTS 服务状态（仅 Dummy，实际已废弃但保留兼容）"""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(TTS_SERVICE_SOCK)
        request = {'command': 'ping'}
        sock.send(json.dumps(request).encode())
        sock.close()
    except Exception:
        pass


def _do_ser_early(ctx):
    """SER 提前推理：停止 arecord 并触发情感识别"""
    try:
        if ctx.get('arecord_proc'):
            ctx['arecord_proc'].terminate()
            try:
                ctx['arecord_proc'].wait(timeout=2)
            except:
                ctx['arecord_proc'].kill()
                ctx['arecord_proc'].wait()
        
        wav_path = ctx.get('wav_path', OUTPUT_FILE)
        if os.path.exists(wav_path):
            emotion, emotion_id, confidence = ser_recognize_emotion(wav_path)
            ctx['result'] = {
                'emotion': emotion,
                'emotion_id': emotion_id,
                'confidence': confidence
            }
            log_info(f"SER 提前推理完成: {emotion}")
    except Exception as e:
        log_warn(f"SER 提前推理异常: {e}")


def _get_ser_emotion_text(ctx):
    """从 SER 上下文提取情感文本"""
    result = ctx.get('result')
    if result:
        return format_emotion_prompt(result.get('emotion', 'neutral'))
    return "患者情绪：中性"


def llm_reset():
    """
    发送 reset 命令到 LLM 服务，清除对话上下文重新预填充系统提示词。
    用于长时间无人说话时重置对话（15分钟无语音）。
    """
    log_info("正在重置 LLM 对话上下文...")
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(LLM_SERVICE_SOCK)
        request = {'prompt': '__RESET__'}
        sock.sendall(json.dumps(request).encode() + b'\n')
        sock.close()
        log_info("LLM 上下文已重置")
    except Exception as e:
        log_warn(f"LLM 重置失败: {e}")


def _ser_service_available():
    """检查 SER 服务是否可用"""
    try:
        if not os.path.exists(SER_SERVICE_SOCK):
            return False
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect(SER_SERVICE_SOCK)
        sock.close()
        return True
    except:
        return False


def ser_recognize_emotion(audio_path):
    """调用 SenseVoice SER 服务进行情感识别"""
    if not _ser_service_available():
        log_warn("SER 服务不可用，跳过情感识别")
        return 'neutral', 0, 0.0
    
    try:
        import sensevoice_ser_client as ser_client
        result = ser_client.ser_from_wav(audio_path, max_frames=100)
        
        if 'error' in result:
            log_warn(f"SER 服务返回错误: {result['error']}")
            return 'neutral', 0, 0.0
        
        emotion_en = result.get('emotion', 'UNKNOWN').lower()
        emotion_id = result.get('emotion_id', 7)
        confidence = result.get('probability', 0.0)
        
        emotion_map = {
            'neutral': 'neutral',
            'happy': 'happy', 
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fearful',
            'disgusted': 'disgusted',
            'surprised': 'surprised',
            'other': 'other',
            'unknown': 'neutral'
        }
        emotion = emotion_map.get(emotion_en, 'neutral')
        
        log_info(f"SER 情感识别结果: {emotion} (ID: {emotion_id}, 置信度: {confidence:.2f})")
        return emotion, emotion_id, confidence
    except Exception as e:
        log_warn(f"SER 情感识别异常: {e}")
        return 'neutral', 0, 0.0


def format_emotion_prompt(emotion):
    """将情感标签格式化为 LLM prompt 注入文本"""
    emotion_map = {
        'neutral': '患者情绪：中性',
        'happy': '患者情绪：开心',
        'sad': '患者情绪：悲伤',
        'angry': '患者情绪：生气',
        'fearful': '患者情绪：恐惧',
        'disgusted': '患者情绪：厌恶',
        'surprised': '患者情绪：惊讶',
    }
    return emotion_map.get(emotion, '患者情绪：中性')


def main():
    """主函数 - sherpa-onnx-alsa (全NPU核) + RAG + Qwen3-0.6B + TTS"""
    print("=" * 50)
    print("    完整语音助手系统 — 多轮对话")
    print("    sherpa-onnx-alsa + RAG + Qwen3-0.6B")
    print("    15分钟无语音自动重置对话上下文")
    print("=" * 50)
    print()

    try:
        setup_audio()

        # TTS 模型预热（启动时提前加载，不占用流水线时间）
        tts_warmup()

        # ASR 模型预热（加载 NPU 模型，避免首次说话丢失）
        _warmup_asr()

        turn_count = 0
        while True:
            turn_count += 1
            print(f"\n{'='*50}")
            print(f"    第 {turn_count} 轮对话")
            print(f"{'='*50}\n")

            TIMING.clear()

            # ── SER 上下文：并行录音 + 提前推理 ──
            _ser_ctx = {
                'wav_path': OUTPUT_FILE,
                'arecord_proc': None,
                'result': None
            }

            # ── ASR（带 SER 提前推理）──
            recording_frames, streaming_text = record_on_voice_detection(_ser_ctx)

            recognized_text = streaming_text
            if not recognized_text:
                log_info("15分钟无语音，重置上下文，等待下一轮...")
                llm_reset()
                continue

            log_timing('speech_end')
            TIMING['asr_done'] = TIMING['speech_end']

            # ── SER 情感识别（优先使用提前推理结果）──
            ser_emotion_text = _get_ser_emotion_text(_ser_ctx)
            if _ser_ctx.get('result'):
                ser_emotion = _ser_ctx['result']['emotion']
                TIMING['ser_done'] = TIMING['speech_end']  # 提前推理已在 ASR 期间完成
                log_info(f"SER 提前推理结果: {ser_emotion}")
            else:
                # 回退到串行推理
                ser_emotion = 'neutral'
                try:
                    TIMING['ser_start'] = time.time()
                    ser_emotion, ser_emotion_id, ser_confidence = ser_recognize_emotion(OUTPUT_FILE)
                    TIMING['ser_done'] = time.time()
                    log_info(f"SER 串行推理耗时: {(TIMING['ser_done'] - TIMING['ser_start']) * 1000:.0f}ms")
                except Exception as e:
                    log_warn(f"SER 调用失败: {e}")
                ser_emotion_text = format_emotion_prompt(ser_emotion)

            # ── RAG 检索 ──
            prompt = f"用户：{recognized_text}\n（请只用中文回答，不要包含任何英文）\n助手："
            TIMING['rag_start'] = time.time()

            # 注入情感信息到 prompt
            prompt = ser_emotion_text + chr(10) + prompt

            # ── Patient RAG 检索（始终查询，结果注入 prompt 作为上下文）──
            rag_context = None
            patient_rag_answer = None
            try:
                rag_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                rag_sock.settimeout(3.0)
                rag_sock.connect(PATIENT_RAG_SOCK)
                rag_req = json.dumps({"query": recognized_text, "threshold": 0.30})
                rag_sock.sendall((rag_req + "\n").encode("utf-8"))
                rag_resp = b""
                while True:
                    chunk = rag_sock.recv(4096)
                    if not chunk:
                        break
                    rag_resp += chunk
                    if b"\n" in chunk:
                        break
                rag_sock.close()
                rag_result = json.loads(rag_resp.decode("utf-8"))
                is_off_topic = rag_result.get("is_off_topic", False)
                if rag_result.get("status") == "ok":
                    topic = rag_result.get("topic", "")
                    answer = rag_result.get("answer", "")
                    # 始终注入 RAG 上下文到 prompt（topic + answer）
                    if topic and answer:
                        rag_context = f"患者档案参考——{topic}：{answer}"
                    elif answer and rag_result.get("close_matches"):
                        # 离题情况：用候选匹配作为参考
                        cm = rag_result["close_matches"][0]
                        rag_context = f"患者档案参考——{cm.get('topic', '相关')}：{cm.get('answer', answer)}"
                    # 高置信度不离题 → 快速通道（跳过 LLM）
                    if (not is_off_topic
                            and rag_result.get("confidence", 0) >= 0.30):
                        patient_rag_answer = rag_result["answer"]
                        log_info(f"Patient RAG 命中 (conf={rag_result['confidence']}): {patient_rag_answer}")
            except Exception as e:
                log_warn(f"Patient RAG 查询异常: {e}")

            TIMING['rag_done'] = time.time()

            # 把 RAG 上下文注入 prompt（如果有）
            if rag_context:
                prompt = f"{rag_context}\n\n{prompt}"

            if patient_rag_answer:
                # ── 直通路径：RAG → TTS → 播放（无延迟）──
                log_step("步骤3/4: Patient RAG 直接回答（跳过 LLM）")
                TIMING['llm_done'] = time.time()  # LLM 被跳过，立即标记完成
                rag_output = '/tmp/voice_assistant/patient_rag_answer.wav'
                os.makedirs('/tmp/voice_assistant', exist_ok=True)
                if tts_synthesize(patient_rag_answer, rag_output):
                    TIMING['tts_first_chunk_done'] = time.time()
                    play_audio(rag_output)
                log_result(f"RAG 回答: {patient_rag_answer}")
                # 打印耗时
                def _fmt_elapsed(label):
                    if label in TIMING and 'speech_end' in TIMING:
                        return f"{(TIMING[label] - TIMING['speech_end']) * 1000:.0f}ms"
                    return "N/A"
                print(f"\n⏱️  说话结束 → SER 完成: {_fmt_elapsed('ser_done')}")
                print(f"⏱️  说话结束 → RAG 完成: {_fmt_elapsed('rag_done')}")
                print(f"⏱️  说话结束 → LLM 完成: {_fmt_elapsed('llm_done')}")
                print(f"⏱️  说话结束 → TTS 就绪: {_fmt_elapsed('tts_first_chunk_done')}")
                print(f"⏱️  说话结束 → 播放首句: {_fmt_elapsed('play_start')}")
                print()
                continue  # 跳过 LLM 路径

            # ── LLM + TTS ──
            import queue
            import threading as _th

            tts_queue = queue.Queue()
            reply_holder = []


            def _strip_markdown(text):
                """去除 LLM 输出中的 markdown 符号"""
                import re
                text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
                text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
                text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
                text = re.sub(r'^\d+[\.、\)]\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
                text = re.sub(r'`{1,3}[^`]+`{1,3}', '', text)
                text = text.replace('**', '').replace('__', '')
                text = re.sub(r'\n{2,}', '\n', text)
                return text.strip()


            def _llm_worker():
                """后台线程：LLM 逐字输出，按标点分句送 TTS"""
                try:
                    # TTS 已在 VAD 前同步预热完成，无需再次预热
                    first = True
                    full = ""
                    buf = ""
                    for char in llm_generate_streaming(prompt):
                        full += char
                        buf += char
                        # 首句加速：逗号、顿号、分号也切
                        if first and char in '，、；' and len(buf) >= 10:
                            cleaned = _strip_markdown(buf)
                            TIMING['llm_first_sentence'] = time.time()
                            log_info(f"LLM 首段(快): {cleaned}")
                            first = False
                            if cleaned:
                                tts_queue.put(cleaned)
                            buf = ""
                        elif char in '。！？' and len(buf) >= (15 if first else 8):
                            cleaned = _strip_markdown(buf)
                            if first:
                                TIMING['llm_first_sentence'] = time.time()
                                log_info(f"LLM 首段: {cleaned}")
                                first = False
                            if cleaned:
                                tts_queue.put(cleaned)
                            buf = ""
                        elif len(buf) >= 45:
                            cleaned = _strip_markdown(buf)
                            if first:
                                TIMING['llm_first_sentence'] = time.time()
                                log_info(f"LLM 首段: {cleaned}")
                                first = False
                            if cleaned:
                                tts_queue.put(cleaned)
                            buf = ""
                    if buf.strip():
                        cleaned = _strip_markdown(buf)
                        if cleaned:
                            tts_queue.put(cleaned)
                    reply_holder.append(full)
                    TIMING['llm_done'] = time.time()
                except Exception as e:
                    log_warn(f"_llm_worker 异常: {e}")
                finally:
                    tts_queue.put(None)

            def _tts_worker():
                """后台线程：每句调 tts_synthesize 合成完整 wav 后送播放"""
                import os as _os
                _os.makedirs('/tmp/voice_assistant', exist_ok=True)
                sentence_idx = 0
                any_sent = False
                try:
                    while True:
                        sentence = tts_queue.get()
                        if sentence is None:
                            break
                        output_file = f'/tmp/voice_assistant/sentence_{sentence_idx:03d}.wav'
                        sentence_idx += 1
                        if tts_synthesize(sentence, output_file):
                            if not any_sent:
                                any_sent = True
                                TIMING['tts_first_chunk_done'] = time.time()
                            play_queue.put(output_file)
                        else:
                            log_warn(f"TTS 合成失败: {sentence[:20]}...")
                except Exception as e:
                    log_warn(f"_tts_worker 异常: {e}")
                finally:
                    play_queue.put(None)

            play_queue = queue.Queue()
            _th.Thread(target=_llm_worker, daemon=True).start()
            _th.Thread(target=_tts_worker, daemon=True).start()

            # ── Play ──
            while True:
                chunk_file = play_queue.get()
                if chunk_file is None:
                    break
                play_audio(chunk_file)

            reply_text = reply_holder[0] if reply_holder else None
            if not reply_text:
                log_warn("LLM 未生成回复，使用备用回复")
                reply_text = "抱歉，我暂时无法回答这个问题。"
                tts_output = "/tmp/voice_assistant/reply.wav"
                if tts_synthesize(reply_text, tts_output):
                    play_audio(tts_output)

            # ── Timing ──
            print("=" * 50)
            print(f"    第 {turn_count} 轮耗时统计")
            print("=" * 50)
            timing_keys = [
                ('speech_end',          '说话结束'),
                ('ser_done',            'SER 情感识别完成'),
                ('rag_done',            'RAG 检索完成'),
                ('llm_done',            'LLM 生成完成'),
                ('tts_first_chunk_done','TTS 合成完成'),
                ('play_start',          '开始播放'),
            ]
            base = TIMING.get('speech_end', 0)
            for i, (key, label) in enumerate(timing_keys):
                if key in TIMING:
                    delta = (TIMING[key] - base) * 1000
                    if i == len(timing_keys) - 1:
                        print(f"🎯 从说话结束到出声: {delta:>8.0f}ms")
                    else:
                        print(f"⏱️  {label}: {delta:>8.0f}ms")
                else:
                    print(f"  {label}:   --- (未记录)")

            if 'speech_end' in TIMING:
                total = (time.time() - TIMING['speech_end']) * 1000
                print(f"⏱️  总耗时(说话结束→播放完毕): {total:>8.0f}ms ({total/1000:.1f}s)")
            print()

            print(f"  第 {turn_count} 轮完成，等待下一轮语音...（按 Ctrl+C 退出）")

    except KeyboardInterrupt:
        print("\n\n用户中断，退出语音助手")
    except Exception as e:
        log_warn(f"主循环异常: {e}")
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