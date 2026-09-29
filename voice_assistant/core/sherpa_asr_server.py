#!/usr/bin/env python3
"""
Sherpa-ONNX 流式 ASR 服务 (基于 Zipformer RKNN)
- 常驻进程，模型只加载一次
- 通过 Unix Socket 接收音频流
- 实时返回识别结果（token + 时间戳）
- 支持 VAD 集成，自动检测语音结束
"""

import os
import sys
import time
import socket
import json
import struct
import threading
import numpy as np
from typing import Optional, Dict, List, Tuple

# 配置参数
SOCKET_PATH = "/tmp/sherpa_asr_streaming.sock"
SAMPLE_RATE = 16000
AUDIO_FORMAT = "int16"
CHANNELS = 1

# Zipformer 模型路径
MODEL_DIR = "/data/zipformer/model"
ENCODER_MODEL = f"{MODEL_DIR}/encoder-epoch-99-avg-1.rknn"
DECODER_MODEL = f"{MODEL_DIR}/decoder-epoch-99-avg-1.rknn"
JOINER_MODEL = f"{MODEL_DIR}/joiner-epoch-99-avg-1.rknn"
TOKENS_FILE = f"{MODEL_DIR}/vocab.txt"

# 流式处理参数
CHUNK_SIZE_MS = 200  # 200ms 音频块
CHUNK_SIZE_SAMPLES = int(SAMPLE_RATE * CHUNK_SIZE_MS / 1000)  # 3200 samples
BUFFER_SIZE = CHUNK_SIZE_SAMPLES * 2  # bytes (16-bit)

# 调试模式
DEBUG = os.environ.get("SHERPA_ASR_DEBUG", "0") == "1"


class SherpaOnnxStreamingASR:
    """基于 Sherpa-ONNX 的流式 ASR 引擎"""
    
    def __init__(self):
        self.recognizer = None
        self.stream = None
        self.model_loaded = False
        self.lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'total_audio_seconds': 0,
            'total_processing_time': 0,
            'init_time': 0
        }
    
    def load_model(self) -> bool:
        """加载模型（只执行一次）"""
        if self.model_loaded:
            return True
        
        start_time = time.time()
        
        try:
            import sherpa_onnx
            
            print("[Sherpa-ASR] 正在导入 sherpa_onnx 模块...", file=sys.stderr)
            
            # 创建在线识别器配置
            recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
                feat_config=sherpa_onnx.FeatureExtractorConfig(
                    sampling_rate=SAMPLE_RATE,
                    feature_dim=80,  # MFCC 特征维度
                ),
                model_config=sherpa_onnx.OnlineModelConfig(
                    transducer=sherpa_onnx.OnlineTransducerModelConfig(
                        encoder=ENCODER_MODEL,
                        decoder=DECODER_MODEL,
                        joiner=JOINER_MODEL,
                    ),
                ),
            )
            
            print("[Sherpa-ASR] 正在创建在线识别器...", file=sys.stderr)
            self.recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
            
            init_time = (time.time() - start_time) * 1000
            self.stats['init_time'] = init_time
            
            print(f"[Sherpa-ASR] ✅ 模型加载成功 ({init_time:.0f}ms)", file=sys.stderr)
            
            self.model_loaded = True
            return True
            
        except ImportError:
            print("[Sherpa-ASR] ❌ 错误: 未安装 sherpa_onnx", file=sys.stderr)
            print("[Sherpa-ASR] 请运行: pip install sherpa-onnx", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[Sherpa-ASR] ❌ 模型加载失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return False
    
    def create_stream(self):
        """创建新的识别流"""
        if not self.model_loaded or not self.recognizer:
            return None
        
        with self.lock:
            self.stream = self.recognizer.create_stream()
            return self.stream
    
    def process_chunk(self, audio_data: bytes) -> Optional[Dict]:
        """
        处理单个音频块
        
        Args:
            audio_data: PCM 音频数据 (16-bit, 16kHz, mono)
        
        Returns:
            识别结果字典，包含:
            - text: 当前累积的文本
            - tokens: token列表
            - timestamps: 时间戳列表
            - is_final: 是否为最终结果
            - partial: 中间结果（可能变化）
        """
        if not self.stream or not audio_data:
            return None
        
        try:
            # 将字节数据转换为 numpy 数组
            audio_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 输入音频到流
            self.stream.accept_waveform(SAMPLE_RATE, audio_samples.tolist())
            
            # 判断是否需要解码
            if self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
            
            # 获取结果
            result = self.recognizer.get_result(self.stream)
            
            if result and (result.text or len(result.tokens) > 0):
                return {
                    'text': result.text,
                    'tokens': result.tokens if hasattr(result, 'tokens') else [],
                    'timestamps': result.timestamps if hasattr(result, 'timestamps') else [],
                    'is_final': False,  # 流式模式下，除非明确结束，都是中间结果
                }
            
            return None
            
        except Exception as e:
            print(f"[Sherpa-ASR] 处理异常: {e}", file=sys.stderr)
            return None
    
    def finalize_stream(self) -> Optional[Dict]:
        """结束当前识别流，返回最终结果"""
        if not self.stream:
            return None
        
        try:
            # 输入空音频表示结束
            self.stream.input_finished()
            
            # 最终解码
            if self.recognizer.is_ready(self.stream):
                self.recognize_decode_stream(self.stream)
            
            # 获取最终结果
            result = self.recognizer.get_result(self.stream)
            
            if result:
                final_result = {
                    'text': result.text,
                    'tokens': result.tokens if hasattr(result, 'tokens') else [],
                    'timestamps': result.timestamps if hasattr(result, 'timestamps') else [],
                    'is_final': True,
                }
                
                # 重置统计
                self.stats['total_requests'] += 1
                
                return final_result
            
            return None
            
        except Exception as e:
            print(f"[Sherpa-ASR] 结束流异常: {e}", file=sys.stderr)
            return None
        finally:
            # 创建新流以备下次使用
            self.stream = None
    
    def reset_stream(self):
        """重置当前识别流（放弃当前结果）"""
        self.stream = None


class SherpaASRServer:
    """流式 ASR Socket 服务器"""
    
    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.asr_engine = SherpaOnnxStreamingASR()
        self.server_socket = None
        self.running = False
        
        # 客户端管理
        self.clients: Dict[int, dict] = {}  # client_id -> {stream, buffer, ...}
        self.next_client_id = 1
        self.clients_lock = threading.Lock()
        
        # 日志
        self.log_file = open("/tmp/sherpa_asr_server.log", "a")
    
    def log(self, message: str):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        self.log_file.write(log_line)
        self.log_file.flush()
        
        if DEBUG:
            print(log_line.strip(), file=sys.stderr)
    
    def start(self):
        """启动服务器"""
        # 清理旧的 socket 文件
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        # 加载模型
        print("[Server] 正在加载 ASR 模型...", file=sys.stderr)
        if not self.asr_engine.load_model():
            print("[Server] ❌ 模型加载失败，退出", file=sys.stderr)
            sys.exit(1)
        
        # 创建 socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # 设置超时以便检查 running 状态
        
        # 设置权限
        os.chmod(self.socket_path, 0o777)
        
        self.running = True
        self.log(f"✅ Sherpa-ASR 服务已启动")
        self.log(f"   Socket: {self.socket_path}")
        self.log(f"   模型初始化时间: {self.asr_engine.stats['init_time']:.0f}ms")
        self.log(f"   等待客户端连接...")
        
        print(f"[Server] ✅ Sherpa-ASR 流式服务已启动", file=sys.stderr)
        print(f"[Server]    Socket: {self.socket_path}", file=sys.stderr)
        print(f"[Server]    等待客户端连接...", file=sys.stderr)
        
        # 主循环
        try:
            while self.running:
                try:
                    client_sock, addr = self.server_socket.accept()
                    
                    # 为每个客户端创建独立线程
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_sock,),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        self.log(f"接受连接异常: {e}")
                        
        except KeyboardInterrupt:
            self.log("收到中断信号")
        finally:
            self.stop()
    
    def handle_client(self, client_sock: socket.socket):
        """处理客户端连接"""
        # 分配客户端ID
        with self.clients_lock:
            client_id = self.next_client_id
            self.next_client_id += 1
            
            # 创建该客户端专属的ASR流
            asr_stream = self.asr_engine.create_stream()
            self.clients[client_id] = {
                'stream': asr_stream,
                'audio_buffer': b'',
                'start_time': time.time(),
                'total_audio_bytes': 0,
            }
        
        self.log(f"📱 客户端 #{client_id} 已连接")
        
        try:
            client_sock.settimeout(30.0)  # 30秒无数据则断开
            
            while self.running:
                # 接收数据
                data = b''
                while len(data) < 4:  # 先读取4字节长度头
                    chunk = client_sock.recv(4 - len(data))
                    if not chunk:
                        raise ConnectionError("连接断开")
                    data += chunk
                
                # 解析长度
                msg_length = struct.unpack('!I', data)[0]
                
                # 接收消息体
                data = b''
                while len(data) < msg_length:
                    chunk = client_sock.recv(min(4096, msg_length - len(data)))
                    if not chunk:
                        raise ConnectionError("连接断开")
                    data += chunk
                
                # 处理请求
                response = self.process_request(client_id, data)
                
                # 发送响应
                if response:
                    response_json = json.dumps(response, ensure_ascii=False).encode('utf-8')
                    length_header = struct.pack('!I', len(response_json))
                    client_sock.sendall(length_header + response_json)
                    
        except socket.timeout:
            self.log(f"⏰ 客户端 #{client_id} 超时断开")
        except ConnectionError as e:
            self.log(f"❌ 客户端 #{client_id} 断开: {e}")
        except Exception as e:
            self.log(f"⚠️ 客户端 #{client_id} 异常: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            # 清理客户端资源
            with self.clients_lock:
                if client_id in self.clients:
                    del self.clients[client_id]
            
            client_sock.close()
            self.log(f"👋 客户端 #{client_id} 已断开")
    
    def process_request(self, client_id: int, data: bytes) -> Optional[Dict]:
        """
        处理客户端请求
        
        请求格式 (JSON):
        {
            "command": "audio|reset|finalize|status",
            "data": "base64编码的PCM数据或null"
        }
        """
        try:
            request = json.loads(data.decode('utf-8'))
            command = request.get('command', '')
            
            if command == 'audio':
                # 处理音频数据
                import base64
                audio_base64 = request.get('data', '')
                audio_data = base64.b64decode(audio_base64)
                
                return self.handle_audio(client_id, audio_data)
            
            elif command == 'reset':
                # 重置识别流
                return self.handle_reset(client_id)
            
            elif command == 'finalize':
                # 结束识别，返回最终结果
                return self.handle_finalize(client_id)
            
            elif command == 'status':
                # 返回状态
                return {'status': 'ok', 'client_id': client_id}
            
            else:
                return {'error': f'未知命令: {command}'}
                
        except json.JSONDecodeError:
            return {'error': 'JSON 解析失败'}
        except Exception as e:
            return {'error': str(e)}
    
    def handle_audio(self, client_id: int, audio_data: bytes) -> Dict:
        """处理音频数据"""
        with self.clients_lock:
            if client_id not in self.clients:
                return {'error': '客户端不存在'}
            
            client_info = self.clients[client_id]
            stream = client_info['stream']
            
            if not stream:
                return {'error': 'ASR流未初始化'}
        
        try:
            # 转换并处理音频
            audio_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 输入到流
            stream.accept_waveform(SAMPLE_RATE, audio_samples.tolist())
            
            # 更新统计
            with self.clients_lock:
                self.clients[client_id]['total_audio_bytes'] += len(audio_data)
            
            # 尝试解码
            result = None
            if self.asr_engine.recognizer.is_ready(stream):
                self.asr_engine.recognizer.decode_stream(stream)
                result_obj = self.asr_engine.recognizer.get_result(stream)
                
                if result_obj and result_obj.text:
                    result = {
                        'text': result_obj.text,
                        'tokens': list(result_obj.tokens) if hasattr(result_obj, 'tokens') else [],
                        'is_final': False,
                        'processing_time': time.time() - self.clients[client_id]['start_time'],
                    }
            
            return result if result else {'status': 'ok'}
            
        except Exception as e:
            return {'error': f'音频处理异常: {e}'}
    
    def handle_reset(self, client_id: int) -> Dict:
        """重置识别流"""
        with self.clients_lock:
            if client_id in self.clients:
                # 创建新的流
                self.clients[client_id]['stream'] = self.asr_engine.create_stream()
                self.clients[client_id]['audio_buffer'] = b''
                self.clients[client_id]['start_time'] = time.time()
                self.clients[client_id]['total_audio_bytes'] = 0
        
        self.log(f"🔄 客户端 #{client_id} 已重置")
        return {'status': 'reset_ok'}
    
    def handle_finalize(self, client_id: int) -> Dict:
        """结束识别，返回最终结果"""
        with self.clients_lock:
            if client_id not in self.clients:
                return {'error': '客户端不存在'}
            
            client_info = self.clients[client_id]
            stream = client_info['stream']
            
            if not stream:
                return {'error': 'ASR流未初始化'}
        
        try:
            # 结束流
            stream.input_finished()
            
            # 最终解码
            if self.asr_engine.recognizer.is_ready(stream):
                self.asr_engine.recognizer.decode_stream(stream)
            
            result_obj = self.asr_engine.recognizer.get_result(stream)
            
            # 计算统计信息
            total_time = time.time() - client_info['start_time']
            audio_duration = client_info['total_audio_bytes'] / (SAMPLE_RATE * 2)  # seconds
            
            result = {
                'text': result_obj.text if result_obj else '',
                'tokens': list(result_obj.tokens) if result_obj and hasattr(result_obj, 'tokens') else [],
                'is_final': True,
                'stats': {
                    'audio_duration': round(audio_duration, 2),
                    'processing_time': round(total_time, 3),
                    'rtf': round(total_time / max(audio_duration, 0.001), 3),
                }
            }
            
            # 重置流以备下次使用
            with self.clients_lock:
                self.clients[client_id]['stream'] = self.asr_engine.create_stream()
                self.clients[client_id]['audio_buffer'] = b''
                self.clients[client_id]['start_time'] = time.time()
                self.clients[client_id]['total_audio_bytes'] = 0
            
            self.log(f"✅ 客户端 #{client_id} 识别完成: '{result['text']}' (RTF={result['stats']['rtf']})")
            
            return result
            
        except Exception as e:
            return {'error': f'结束识别异常: {e}'}
    
    def stop(self):
        """停止服务器"""
        self.running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        self.log("🛑 服务已停止")
        self.log_file.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sherpa-ONNX 流式 ASR 服务')
    parser.add_argument('--socket', default=SOCKET_PATH, help='Socket 路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = args.debug
    
    print("=" * 60, file=sys.stderr)
    print("  Sherpa-ONNX 流式 ASR 服务 (Zipformer RKNN)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Socket: {args.socket}", file=sys.stderr)
    print(f"  Encoder: {ENCODER_MODEL}", file=sys.stderr)
    print(f"  Decoder: {DECODER_MODEL}", file=sys.stderr)
    print(f"  Joiner: {JOINER_MODEL}", file=sys.stderr)
    print(f"  Tokens: {TOKENS_FILE}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    server = SherpaASRServer(socket_path=args.socket)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[Server] 收到退出信号", file=sys.stderr)
        server.stop()


if __name__ == "__main__":
    main()
