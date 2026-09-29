#!/usr/bin/env python3
"""
Sherpa-ONNX 流式 ASR 客户端库
- 连接 sherpa_asr_server 服务
- 发送音频数据并接收实时识别结果
- 提供简洁的 Python API
"""

import os
import sys
import socket
import json
import struct
import base64
import time
from typing import Optional, Dict, List, Callable, Generator

# 默认配置
DEFAULT_SOCKET_PATH = "/tmp/sherpa_asr_streaming.sock"
CONNECT_TIMEOUT = 5.0  # 连接超时（秒）
REQUEST_TIMEOUT = 10.0  # 请求超时（秒）


class SherpaASRClient:
    """Sherpa-ONNX 流式 ASR 客户端"""
    
    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path
        self.sock: Optional[socket.socket] = None
        self.connected = False
        
        # 回调函数
        self.on_partial_result: Optional[Callable[[str], None]] = None
        self.on_final_result: Optional[Callable[[Dict], None]] = None
    
    def connect(self) -> bool:
        """连接到 ASR 服务"""
        try:
            if os.path.exists(self.socket_path):
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(CONNECT_TIMEOUT)
                self.sock.connect(self.socket_path)
                self.connected = True
                
                print(f"[Sherpa-Client] ✅ 已连接到 ASR 服务: {self.socket_path}")
                return True
            else:
                print(f"[Sherpa-Client] ❌ Socket 文件不存在: {self.socket_path}", file=sys.stderr)
                return False
                
        except Exception as e:
            print(f"[Sherpa-Client] ❌ 连接失败: {e}", file=sys.stderr)
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
            self.connected = False
            print("[Sherpa-Client] 已断开连接")
    
    def _send_request(self, command: str, data=None) -> Optional[Dict]:
        """
        发送请求到服务器
        
        Args:
            command: 命令类型 (audio|reset|finalize|status)
            data: 附加数据
        
        Returns:
            服务器响应字典
        """
        if not self.connected or not self.sock:
            return {'error': '未连接'}
        
        try:
            # 构建请求
            request = {
                'command': command,
                'data': base64.b64encode(data).decode('ascii') if data else None
            }
            
            request_json = json.dumps(request, ensure_ascii=False).encode('utf-8')
            
            # 发送长度头 + 数据
            length_header = struct.pack('!I', len(request_json))
            self.sock.sendall(length_header + request_json)
            
            # 接收响应长度
            length_data = b''
            while len(length_data) < 4:
                chunk = self.sock.recv(4 - len(length_data))
                if not chunk:
                    raise ConnectionError("连接断开")
                length_data += chunk
            
            response_length = struct.unpack('!I', length_data)[0]
            
            # 接收响应数据
            response_data = b''
            while len(response_data) < response_length:
                chunk = self.sock.recv(min(4096, response_length - len(response_data)))
                if not chunk:
                    raise ConnectionError("连接断开")
                response_data += chunk
            
            # 解析响应
            response = json.loads(response_data.decode('utf-8'))
            
            if 'error' in response:
                print(f"[Sherpa-Client] ⚠️ 服务器错误: {response['error']}", file=sys.stderr)
            
            return response
            
        except socket.timeout:
            print(f"[Sherpa-Client] ⏰ 请求超时", file=sys.stderr)
            return {'error': '请求超时'}
        except ConnectionError as e:
            print(f"[Sherpa-Client] ❌ 连接断开: {e}", file=sys.stderr)
            self.connected = False
            return {'error': str(e)}
        except Exception as e:
            print(f"[Sherpa-Client] ❌ 请求异常: {e}", file=sys.stderr)
            return {'error': str(e)}
    
    def send_audio(self, audio_data: bytes) -> Optional[str]:
        """
        发送音频块
        
        Args:
            audio_data: PCM 音频数据 (16-bit, 16kHz, mono)
        
        Returns:
            当前识别的文本（如果有更新），否则返回 None
        """
        result = self._send_request('audio', audio_data)
        
        if result and 'text' in result and result['text']:
            text = result['text']
            
            # 调用回调
            if self.on_partial_result:
                self.on_partial_result(text)
            
            return text
        
        return None
    
    def reset(self) -> bool:
        """重置识别状态"""
        result = self._send_request('reset')
        return result.get('status') == 'reset_ok'
    
    def finalize(self) -> Optional[Dict]:
        """
        结束当前识别会话，返回最终结果
        
        Returns:
            最终识别结果字典：
            {
                'text': '完整文本',
                'tokens': ['token1', 'token2', ...],
                'is_final': True,
                'stats': {
                    'audio_duration': 3.5,
                    'processing_time': 0.68,
                    'rtf': 0.194
                }
            }
        """
        result = self._send_request('finalize')
        
        if result and result.get('is_final'):
            # 调用回调
            if self.on_final_result:
                self.on_final_result(result)
        
        return result
    
    def check_status(self) -> bool:
        """检查服务状态"""
        result = self._send_request('status')
        return result.get('status') == 'ok'


class StreamingASRSession:
    """
    流式 ASR 会话管理器
    - 自动管理连接生命周期
    - 提供生成器接口，方便集成到录音循环
    - 支持实时返回中间结果和最终结果
    """
    
    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.client = SherpaASRClient(socket_path)
        self.session_active = False
        self.current_text = ""
        self.start_time = None
        
        # 统计信息
        self.stats = {
            'chunks_sent': 0,
            'total_bytes': 0,
            'first_text_time': None,
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.end_session()
        return False
    
    def start_session(self) -> bool:
        """开始新的识别会话"""
        if not self.client.connect():
            return False
        
        # 重置服务端状态
        self.client.reset()
        
        self.session_active = True
        self.current_text = ""
        self.start_time = time.time()
        self.stats = {
            'chunks_sent': 0,
            'total_bytes': 0,
            'first_text_time': None,
        }
        
        print("[ASR Session] 🎙️ 新会话已开始", file=sys.stderr)
        return True
    
    def end_session(self) -> Optional[Dict]:
        """结束当前会话，返回最终结果"""
        if not self.session_active:
            return None
        
        final_result = None
        
        try:
            # 获取最终结果
            final_result = self.client.finalize()
            
            if final_result and final_result.get('is_final'):
                stats = final_result.get('stats', {})
                
                print(f"\n[ASR Session] ✅ 识别完成:", file=sys.stderr)
                print(f"   文本: '{final_result.get('text', '')}'", file=sys.stderr)
                print(f"   音频时长: {stats.get('audio_duration', 0):.2f}s", file=sys.stderr)
                print(f"   处理耗时: {stats.get('processing_time', 0):.3f}s", file=sys.stderr)
                print(f"   RTF: {stats.get('rtf', 0):.3f}", file=sys.stderr)
                
                if self.stats['first_text_time']:
                    first_text_delay = (self.stats['first_text_time'] - self.start_time) * 1000
                    print(f"   首字延迟: {first_text_delay:.0f}ms", file=sys.stderr)
                
        finally:
            self.client.disconnect()
            self.session_active = False
        
        return final_result
    
    def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """
        处理单个音频块（在录音循环中调用）
        
        Args:
            audio_data: PCM 音频数据 (16-bit, 16kHz, mono)
        
        Returns:
            当前识别文本（如果更新了）
        """
        if not self.session_active:
            return None
        
        # 更新统计
        self.stats['chunks_sent'] += 1
        self.stats['total_bytes'] += len(audio_data)
        
        # 发送到服务端
        current_text = self.client.send_audio(audio_data)
        
        # 记录首次获得文本的时间
        if current_text and not self.stats['first_text_time']:
            self.stats['first_text_time'] = time.time()
            first_text_delay = (self.stats['first_text_time'] - self.start_time) * 1000
            print(f"\n[ASR Session] 🔊 首次识别输出 ({first_text_delay:.0f}ms): '{current_text}'", 
                  file=sys.stderr, end='', flush=True)
        
        self.current_text = current_text or self.current_text
        
        return current_text
    
    @property
    def is_active(self) -> bool:
        return self.session_active


def recognize_file(file_path: str, socket_path: str = DEFAULT_SOCKET_PATH) -> Dict:
    """
    识别音频文件（便捷函数）
    
    Args:
        file_path: WAV 文件路径
        socket_path: ASR 服务 Socket 路径
    
    Returns:
        识别结果字典
    """
    import wave
    
    session = StreamingASRSession(socket_path)
    
    try:
        if not session.start_session():
            return {'error': '无法连接到 ASR 服务'}
        
        # 读取 WAV 文件
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getsampwidth()
            n_frames = wf.getnframes()
            
            print(f"[recognize_file] 音频文件信息:", file=sys.stderr)
            print(f"   采样率: {sample_rate}Hz", file=sys.stderr)
            print(f"   声道数: {n_channels}", file=sys.stderr)
            print(f"   时长: {n_frames / sample_rate:.2f}s", file=sys.stderr)
            
            # 分块读取并发送
            chunk_size = int(sample_rate * 0.2)  # 200ms chunks
            
            while True:
                frames = wf.readframes(chunk_size)
                if not frames:
                    break
                
                # 如果是立体声，转换为单声道
                if n_channels == 2:
                    import numpy as np
                    audio = np.frombuffer(frames, dtype=np.int16)
                    audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    frames = audio.tobytes()
                
                # 处理音频块
                session.process_audio_chunk(frames)
            
            # 结束会话
            return session.end_session()
            
    except Exception as e:
        print(f"[recognize_file] ❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {'error': str(e)}
    finally:
        if session.is_active:
            session.end_session()


def test_connection(socket_path: str = DEFAULT_SOCKET_PATH) -> bool:
    """测试 ASR 服务连接"""
    client = SherpaASRClient(socket_path)
    
    if client.connect():
        status = client.check_status()
        client.disconnect()
        return status
    return False


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sherpa-ONNX ASR 客户端测试工具')
    parser.add_argument('--socket', default=DEFAULT_SOCKET_PATH, help='Socket 路径')
    parser.add_argument('--test-file', help='测试音频文件路径')
    parser.add_argument('--test-connect', action='store_true', help='仅测试连接')
    args = parser.parse_args()
    
    if args.test_connect:
        print(f"\n测试连接: {args.socket}")
        if test_connection(args.socket):
            print("✅ 服务可用\n")
            sys.exit(0)
        else:
            print("❌ 服务不可用\n")
            sys.exit(1)
    
    elif args.test_file:
        print(f"\n识别文件: {args.test_file}")
        result = recognize_file(args.test_file, args.socket)
        
        if result and result.get('is_final'):
            print(f"\n✅ 识别成功!")
            print(f"文本: {result['text']}")
        else:
            print(f"\n❌ 识别失败: {result}")
    
    else:
        print("\n请指定 --test-file 或 --test-connect 参数")
        parser.print_help()
