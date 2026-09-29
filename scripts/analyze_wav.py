#!/usr/bin/env python3
"""Analyze and compare WAV files"""
import wave, numpy as np

def analyze(path, label):
    wf = wave.open(path, "rb")
    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    wf.close()
    rms = np.sqrt(np.mean(data**2))
    zc = np.sum(np.abs(np.diff(np.signbit(data.astype(float)))))
    silence = np.sum(np.abs(data) < 100)
    return {
        "len": len(data),
        "max": int(data.max()),
        "min": int(data.min()),
        "std": float(data.std()),
        "rms": float(rms),
        "rms_db": 20 * np.log10(rms + 1e-10),
        "zc": int(zc),
        "zc_pct": 100.0 * zc / len(data),
        "silence_pct": 100.0 * silence / len(data),
        "e1": float(np.sqrt(np.mean(data[:len(data)//2]**2))),
        "e2": float(np.sqrt(np.mean(data[len(data)//2:]**2))),
    }

ref = analyze("/data/matcha_tts/test_matcha.wav", "Ref")
new = analyze("/tmp/test_matcha_fix.wav", "New")

print("=" * 60)
print(f"{'Metric':<25} {'Ref':>15} {'New':>15}")
print("=" * 60)
print(f"{'Length (samples)':<25} {ref['len']:>15} {new['len']:>15}")
print(f"{'Duration (sec)':<25} {ref['len']/22050:>15.2f} {new['len']/22050:>15.2f}")
print(f"{'Max':<25} {ref['max']:>15} {new['max']:>15}")
print(f"{'Min':<25} {ref['min']:>15} {new['min']:>15}")
print(f"{'Std':<25} {ref['std']:>15.1f} {new['std']:>15.1f}")
print(f"{'RMS (dBFS)':<25} {ref['rms_db']:>15.1f} {new['rms_db']:>15.1f}")
print(f"{'Zero crossings (%)':<25} {ref['zc_pct']:>15.1f} {new['zc_pct']:>15.1f}")
print(f"{'Silence (%)':<25} {ref['silence_pct']:>15.1f} {new['silence_pct']:>15.1f}")
print(f"{'Energy 1st half':<25} {ref['e1']:>15.1f} {new['e1']:>15.1f}")
print(f"{'Energy 2nd half':<25} {ref['e2']:>15.1f} {new['e2']:>15.1f}")
print("=" * 60)

print("\n=== 差异诊断 ===")
std_ratio = new['std'] / ref['std'] if ref['std'] > 0 else 1
print(f"Std ratio (new/ref): {std_ratio:.2f}")
print(f"Silence: new={new['silence_pct']:.1f}% vs ref={ref['silence_pct']:.1f}%")
print(f"Duration: new={new['len']/22050:.2f}s vs ref={ref['len']/22050:.2f}s")

db_diff = new['rms_db'] - ref['rms_db']
if db_diff < -6:
    print(">> 新WAV音量显著偏低（{}dB）".format(db_diff))
elif abs(db_diff) < 3:
    print(">> 新WAV音量正常，与参考WAV相当 (diff={:.1f}dB)".format(db_diff))
else:
    print(">> 新WAV音量在可接受范围 (diff={:.1f}dB)".format(db_diff))

if new['silence_pct'] > 80:
    print(">> 新WAV几乎全静音！")
elif new['silence_pct'] > ref['silence_pct'] * 2:
    print(">> 新WAV静音比例偏高，可能有卡顿")
else:
    print(">> 新WAV静音比例正常")