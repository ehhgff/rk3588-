#!/usr/bin/env python3
"""sherpa-onnx Matcha TTS 常驻服务管理
管理 C daemon（sherpa_tts_daemon），通过 FIFO pipe 通信
"""
import os
import sys
import time
import signal
import subprocess

PIPE_PATH = "/tmp/tts_pipe"
PID_FILE = "/tmp/sherpa_tts.pid"
DAEMON_LOG = "/tmp/sherpa_tts_daemon.log"
OUTPUT_DIR = "/tmp"

DAEMON = "/data/sherpa_tts_daemon"
MODEL_DIR = "/data/sherpa-onnx/matcha-zh-baker"
LIB_PATH = "/userdata/sherpa-onnx/install/lib:/usr/lib"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SherpaTTSManager:
    def __init__(self):
        self.process = None
        self.running = False

    def start(self):
        log("启动 sherpa-onnx Matcha TTS 常驻服务...")

        for f in [DAEMON, f"{MODEL_DIR}/model-steps-3.onnx",
                  f"{MODEL_DIR}/vocos-22khz-univ.onnx",
                  f"{MODEL_DIR}/lexicon.txt", f"{MODEL_DIR}/tokens.txt"]:
            if not os.path.exists(f):
                log(f"错误: 文件不存在 {f}")
                sys.exit(1)

        log("所有文件就绪")

        if os.path.exists(PIPE_PATH):
            os.remove(PIPE_PATH)

        cmd = [DAEMON, MODEL_DIR, PIPE_PATH]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = LIB_PATH

        log(f"启动: {' '.join(cmd)}")
        log(f"日志: {DAEMON_LOG}")

        with open(DAEMON_LOG, "w") as logfile:
            self.process = subprocess.Popen(
                cmd, stdout=logfile, stderr=subprocess.STDOUT, env=env
            )

        log("等待服务就绪...")
        for i in range(200):
            ret = self.process.poll()
            if ret is not None:
                log(f"错误: 进程已退出，返回码 {ret}")
                with open(DAEMON_LOG) as f:
                    for line in f.readlines()[-10:]:
                        log(f"  {line.rstrip()}")
                sys.exit(1)
            if os.path.exists(PIPE_PATH):
                log(f"服务已启动，监听 {PIPE_PATH}")
                self.running = True
                with open(PID_FILE, "w") as f:
                    f.write(str(self.process.pid))
                return True
            time.sleep(0.1)

        log("错误: 服务启动超时")
        self.stop()
        sys.exit(1)

    def stop(self):
        log("停止 sherpa-onnx Matcha TTS 服务...")
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        for f in [PIPE_PATH, PID_FILE]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        self.running = False
        log("服务已停止")

    def is_alive(self):
        if not self.process:
            return False
        ret = self.process.poll()
        if ret is not None:
            log(f"C daemon 已退出，返回码: {ret}")
            self.running = False
            return False
        return True


def main():
    manager = SherpaTTSManager()

    def signal_handler(signum, frame):
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    manager.start()

    try:
        while manager.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()


if __name__ == "__main__":
    main()