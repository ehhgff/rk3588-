#!/usr/bin/env python3
"""
MeloTTS RKNN 常驻服务
基于 RK3588 的 TTS 语音合成服务
参考 Zipformer RKNN 部署方式
"""

import socket
import os
import sys
import json
import subprocess
import signal
import threading
import time
from pathlib import Path

# 服务配置
SOCKET_PATH = "/tmp/matcha_tts.sock"
PID_FILE = "/tmp/melotts_service.pid"
MELOTTS_DIR = "/data/melotts_demo"

# RKNN 模型配置
ENCODER_MODEL = f"{MELOTTS_DIR}/model/encoder-ZH_MIX_EN.rknn"
DECODER_MODEL = f"{MELOTTS_DIR}/model/decoder-ZH_MIX_EN.rknn"

class MeloTTSService:
    """MeloTTS RKNN 服务类"""
    
    def __init__(self):
        self.running = False
        self.sock = None
        self.request_count = 0
        self.total_time = 0
        
    def start(self):
        """启动服务"""
        # 清理旧的socket文件
        if os.path.exists(SOCKET_PATH):
            try:
                os.remove(SOCKET_PATH)
            except:
                pass
            
        # 创建socket
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(SOCKET_PATH)
        self.sock.listen(5)
        
        # 设置权限
        os.chmod(SOCKET_PATH, 0o777)
        
        # 保存PID
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        self.running = True
        print(f"[MeloTTS服务] 已启动，监听 {SOCKET_PATH}", file=sys.stderr)
        print(f"[MeloTTS服务] 模型路径: {ENCODER_MODEL}", file=sys.stderr)
        
        # 接受连接
        while self.running:
            try:
                self.sock.settimeout(1.0)
                conn, addr = self.sock.accept()
                threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[MeloTTS服务] 接受连接错误: {e}", file=sys.stderr)
                
    def handle_client(self, conn):
        """处理客户端请求"""
        try:
            # 接收数据
            data = conn.recv(4096).decode()
            request = json.loads(data)
            
            command = request.get('command')
            
            if command == 'ping':
                response = {
                    'status': 'ok', 
                    'message': 'MeloTTS RKNN 服务运行正常',
                    'model': 'encoder-ZH_MIX_EN.rknn + decoder-ZH_MIX_EN.rknn'
                }
                
            elif command == 'synthesize':
                text = request.get('text', '')
                output_file = request.get('output', '/tmp/melotts_output.wav')
                speed = request.get('speed', 1.0)
                
                start_time = time.time()
                # 调用melotts_demo进行合成
                success, message = self.synthesize(text, output_file, speed)
                elapsed = (time.time() - start_time) * 1000
                
                self.request_count += 1
                self.total_time += elapsed
                avg_time = self.total_time / self.request_count
                
                response = {
                    'status': 'ok' if success else 'error',
                    'message': message,
                    'output': output_file if success else None,
                    'time_ms': elapsed,
                    'avg_time_ms': avg_time,
                    'request_count': self.request_count
                }
                
            elif command == 'stats':
                response = {
                    'status': 'ok',
                    'request_count': self.request_count,
                    'total_time_ms': self.total_time,
                    'avg_time_ms': self.total_time / self.request_count if self.request_count > 0 else 0
                }
            else:
                response = {'status': 'error', 'message': f'未知命令: {command}'}
                
        except Exception as e:
            response = {'status': 'error', 'message': str(e)}
            
        # 发送响应
        conn.sendall(json.dumps(response).encode())
        conn.close()
        
    def synthesize(self, text, output_file, speed=1.0):
        """调用melotts_demo合成语音 (RKNN加速)"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 构建命令 - 使用RKNN模型
            # 注意：必须在 MELOTTS_DIR 目录下运行，否则词典加载会失败
            cmd = [
                "./melotts_demo",
                "--input_text", text,
                "--encoder_model_path", ENCODER_MODEL,
                "--decoder_model_path", DECODER_MODEL,
                "--output_filename", output_file,
                "--language", "ZH",
                "--speed", str(speed),
                "--disable_bert", "true"
            ]
            
            print(f"[MeloTTS合成] 命令: {' '.join(cmd)}", file=sys.stderr)
            
            # 设置环境变量
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = f"{MELOTTS_DIR}/lib:{env.get('LD_LIBRARY_PATH', '')}"
            
            # 执行命令 - 在 MELOTTS_DIR 目录下运行
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=MELOTTS_DIR
            )
            
            print(f"[MeloTTS合成] 返回码: {result.returncode}", file=sys.stderr)
            if result.stdout:
                print(f"[MeloTTS合成] stdout: {result.stdout[:500]}", file=sys.stderr)
            if result.stderr:
                print(f"[MeloTTS合成] stderr: {result.stderr[:500]}", file=sys.stderr)
            
            # 检查输出文件
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                # 获取音频信息
                file_size = os.path.getsize(output_file)
                # 检查文件类型
                try:
                    file_result = subprocess.run(["file", output_file], capture_output=True, text=True)
                    print(f"[MeloTTS合成] 文件类型: {file_result.stdout.strip()}", file=sys.stderr)
                except Exception as e:
                    print(f"[MeloTTS合成] 无法检测文件类型: {e}", file=sys.stderr)
                return True, f"RKNN合成完成: {output_file} ({file_size} bytes)"
            else:
                error_msg = result.stderr if result.stderr else "未知错误"
                return False, f"RKNN合成失败: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return False, "RKNN合成超时"
        except Exception as e:
            return False, f"RKNN合成错误: {str(e)}"
            
    def stop(self):
        """停止服务"""
        self.running = False
        if self.sock:
            self.sock.close()
        for f in [SOCKET_PATH, PID_FILE]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        print("[MeloTTS服务] 已停止", file=sys.stderr)

def main():
    service = MeloTTSService()
    
    # 信号处理
    def signal_handler(sig, frame):
        print("\n[MeloTTS服务] 收到停止信号", file=sys.stderr)
        service.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 启动服务
    service.start()

if __name__ == '__main__':
    main()