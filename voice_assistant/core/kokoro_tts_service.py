#!/usr/bin/env python3
import os, sys, json, time, socket, signal, threading, wave, io
import numpy as np

SOCKET_PATH = "/tmp/kokoro_tts.sock"
PID_FILE = "/tmp/kokoro_tts.pid"
MODEL_PATH = "/home/ubuntu/桌面/ai/kokoro/kokoro-v1.1-zh.onnx"
VOICES_PATH = "/home/ubuntu/桌面/ai/kokoro/voices-v1.1-zh.bin"

def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

class KokoroService:
    def __init__(self):
        self.running = False
        self.sock = None

    def start(self):
        from kokoro_onnx import Kokoro
        log("Loading model...")
        t0 = time.time()
        self.kokoro = Kokoro(model_path=MODEL_PATH, voices_path=VOICES_PATH)
        self.g2p = None
        try:
            from misaki import zh
            self.g2p = zh.ZHG2P()
        except:
            pass
        log("Ready in %.1fs" % (time.time() - t0))
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(SOCKET_PATH)
        self.sock.listen(5)
        os.chmod(SOCKET_PATH, 0o777)
        self.running = True
        log("Listening")
        while self.running:
            try:
                self.sock.settimeout(1.0)
                c, _ = self.sock.accept()
                threading.Thread(target=self.handle, args=(c,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    log("Accept: " + str(e))

    def synth(self, text, voice, speed, lang):
        if lang == "zh" and self.g2p:
            ph = self.g2p(text)[0]
            is_ph = True
        else:
            ph, is_ph = text, False
        audio, sr = self.kokoro.create(text=ph, voice=voice, speed=speed, lang="en-us", is_phonemes=is_ph)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        return buf.getvalue(), sr

    def handle(self, conn):
        try:
            data = json.loads(conn.recv(65536).decode())
            cmd = data.get("command", "")
            if cmd == "ping":
                resp = {"status": "ok"}
            elif cmd == "synthesize":
                t0 = time.time()
                text = data.get("text", "")
                out = data.get("output", "/tmp/kokoro_out.wav")
                voice = data.get("voice", "zf_001")
                speed = data.get("speed", 1.0)
                lang = "zh" if "zh" in data.get("language", "chinese").lower() else "en"
                wav, sr = self.synth(text, voice, speed, lang)
                d = os.path.dirname(out)
                if d and not os.path.exists(d):
                    os.makedirs(d)
                with open(out, "wb") as f:
                    f.write(wav)
                dur = len(wav) / 2 / sr
                log("Done: %.1fs audio in %.2fs" % (dur, time.time() - t0))
                resp = {"status": "ok", "output": out, "sample_rate": sr}
            else:
                resp = {"status": "error", "message": "unknown cmd"}
        except Exception as e:
            resp = {"status": "error", "message": str(e)}
        conn.sendall(json.dumps(resp).encode())
        conn.close()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        for f in [SOCKET_PATH, PID_FILE]:
            if os.path.exists(f):
                os.remove(f)

def main():
    svc = KokoroService()
    def handler(s, f):
        svc.stop()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    svc.start()

if __name__ == "__main__":
    main()
