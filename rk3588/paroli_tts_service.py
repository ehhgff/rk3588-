#!/usr/bin/env python3
"""Paroli TTS 常驻服务（C++ 后端，模型常驻内存）
- 通过 Unix Socket 提供 TTS 服务
- 使用 paroli-socket-server（C++ 守护进程）进行语音合成
- C++ 进程模型常驻内存，无每次请求的模型加载开销
"""
import os
import sys
import json
import time
import socket
import signal
import subprocess
import threading

SOCKET_PATH = "/tmp/paroli_tts.sock"
PID_FILE = "/tmp/paroli_tts.pid"
SERVER_LOG = "/tmp/paroli_socket_server.log"

PAROLI_DIR = "/data/paroli"
PAROLI_SERVER = f"{PAROLI_DIR}/bin/paroli-socket-server"
ENCODER = f"{PAROLI_DIR}/model/encoder_streaming.onnx"
DECODER = f"{PAROLI_DIR}/model/decoder_streaming.rknn"
CONFIG = f"{PAROLI_DIR}/model/zh_CN-huayan-medium.onnx.json"
LIB_PATH = f"{PAROLI_DIR}/lib:{PAROLI_DIR}/lib/aarch64-linux-gnu"

REQUEST_TIMEOUT = 60  # seconds


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ParoliSocketServerManager:
    def __init__(self):
        self.process = None
        self.running = False

    def start(self):
        log("启动 Paroli TTS 服务（C++ 常驻进程）...")

        # Verify files
        for f in [PAROLI_SERVER, ENCODER, DECODER, CONFIG]:
            if not os.path.exists(f):
                log(f"错误: 文件不存在 {f}")
                sys.exit(1)

        log("所有文件就绪")

        # Clean up stale socket
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
            log(f"清理旧 socket")

        # Launch C++ server
        cmd = [
            PAROLI_SERVER,
            "--encoder", ENCODER,
            "--decoder", DECODER,
            "-c", CONFIG,
            "--socket", SOCKET_PATH,
            "--core-mask", "1",
        ]

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = LIB_PATH

        log(f"启动: {' '.join(cmd)}")
        log(f"日志: {SERVER_LOG}")

        with open(SERVER_LOG, "w") as logfile:
            self.process = subprocess.Popen(
                cmd,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=PAROLI_DIR,
            )

        # Wait for socket to appear
        log("等待服务就绪...")
        for i in range(100):
            if os.path.exists(SOCKET_PATH):
                log(f"服务已启动，监听 {SOCKET_PATH}")
                self.running = True
                # Write PID file
                with open(PID_FILE, "w") as f:
                    f.write(str(self.process.pid))
                return True
            time.sleep(0.1)

        # Timed out
        log("错误: 服务启动超时")
        self.stop()
        sys.exit(1)

    def stop(self):
        log("停止 Paroli TTS 服务...")
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

        for f in [SOCKET_PATH, PID_FILE]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        self.running = False
        log("服务已停止")

    def is_alive(self):
        if not self.process:
            return False
        ret = self.process.poll()
        if ret is not None:
            log(f"C++ 进程已退出，返回码: {ret}")
            self.running = False
            return False
        return True


def main():
    manager = ParoliSocketServerManager()

    def signal_handler(signum, frame):
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    manager.start()

    # Monitor the C++ process and forward requests if needed
    # (Currently paroli-socket-server handles requests directly;
    #  we just keep the Python process alive for health monitoring)
    try:
        while manager.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()


if __name__ == "__main__":
    main()