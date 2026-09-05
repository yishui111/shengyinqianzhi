# -*- coding: utf-8 -*-
"""诊断：逐个验证 preprocess_api 里各处理项的真实效果（不启动服务、不跑大模型）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pre_service"))
import preprocess_api as P  # noqa: E402

SR = 44100
rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out")
os.makedirs(OUT, exist_ok=True)


def speech_like(dur, f0_mean=180.0, amp=0.15):
    """谐波叠加大师：有清晰基频的拟语音信号。"""
    t = np.arange(int(SR * dur)) / SR
    f0 = f0_mean * (1 + 0.03 * np.sin(2 * np.pi * 4 * t))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(ph * h) / h for h in (1, 2, 3, 4, 5, 6))
    # 加一点幅度起伏模拟音节
    sig *= 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3 * t))
    return (sig * amp).astype("float32")


def applause_like(dur, amp=0.25):
    """拟掌声：宽带噪声 + 随机密集瞬态脉冲（无基频）。"""
    n = int(SR * dur)
    noise = rng.standard_normal(n).astype("float32")
    clicks = (rng.random(n) > 0.998).astype("float32")
    env = np.convolve(clicks, np.ones(80) / 80, mode="same") * 30
    env = np.clip(env, 0, 1.5) + 0.15
    # 简易带通：FFT 去掉 <300Hz 与 >8kHz
    spec = np.fft.rfft(noise * env)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[(freqs < 300) | (freqs > 8000)] = 0
    return (np.fft.irfft(spec, n) * amp).astype("float32")


print("=" * 70)
print("【测试 1】_voiced_ratio 对不同信号的判断（应为：语音高、掌声低、静音 0）")
for name, seg in [("语音", speech_like(1.0)), ("掌声", applause_like(1.0)),
                  ("静音", np.zeros(SR, dtype="float32"))]:
    print("   %-4s -> voiced_ratio = %.3f" % (name, P._voiced_ratio(seg, SR)))

print("=" * 70)
print("【测试 2】remove_unvoiced_regions（去掌声）：语音3s + 掌声4s + 语音3s")
y = np.concatenate([speech_like(3.0), applause_like(4.0), speech_like(3.0)])
out = P.remove_unvoiced_regions(y.copy(), SR)
print("   输入 %.2fs -> 输出 %.2fs（预期 ~6.3s：掌声被剪掉）" % (len(y) / SR, len(out) / SR))

print("=" * 70)
print("【测试 3】voice_segments（去静音/切片）过切 bug：连续语音，每 0.3s 嵌入 60ms 微停顿")
pieces = []
for k in range(20):  # 共 20*0.3 = 6 秒"连续说话"，中间只有 60ms 的发音间隙
    pieces.append(speech_like(0.3))
    pieces.append(np.zeros(int(SR * 0.06), dtype="float32"))
y2 = np.concatenate(pieces)
segs = P.voice_segments(y2, SR)
print("   6 秒连续说话切出 %d 段（docstring 说 0.45s 静音才分段，合理应为 1 段）" % len(segs))
durs = ["%.2f" % ((b - a) / SR) for a, b in segs[:12]]
print("   前 12 段时长(秒): %s" % ", ".join(durs))
# 真正的：语音 3s + 静音 2s + 语音 3s 应该切成 2 段
y3 = np.concatenate([speech_like(3.0), np.zeros(int(SR * 2), dtype="float32"), speech_like(3.0)])
segs3 = P.voice_segments(y3, SR)
print("   [3s语音+2s静音+3s语音] 切出 %d 段（应为 2）" % len(segs3))

print("=" * 70)
print("【测试 4】真实测试文件：tests/denoise_test/denoise_src.wav")
import soundfile as sf  # noqa: E402
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "denoise_test", "denoise_src.wav")
if os.path.isfile(src):
    yr, srr = sf.read(src)
    if yr.ndim > 1:
        yr = yr.mean(axis=1)
    yr = np.asarray(yr, dtype="float32")
    print("   输入 %.2fs @ %dHz" % (len(yr) / srr, srr))
    vr = P._voiced_ratio(yr, srr)
    print("   整体 voiced_ratio = %.3f" % vr)
    out4 = P.remove_unvoiced_regions(yr.copy(), srr)
    print("   去掌声后 %.2fs" % (len(out4) / srr))
    sf.write(os.path.join(OUT, "denoise_src_after.wav"), out4, srr)
else:
    print("   文件不存在，跳过")

print("=" * 70)
print("【测试 5】说话人过滤：tests/test_twospeaker.wav（加载 ECAPA，CPU 需十几秒）")
spk = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_twospeaker.wav")
if os.path.isfile(spk):
    ys, srs = sf.read(spk)
    if ys.ndim > 1:
        ys = ys.mean(axis=1)
    ys = np.asarray(ys, dtype="float32")
    print("   输入 %.2fs @ %dHz" % (len(ys) / srs, srs))
    try:
        out5 = P.keep_main_speaker(ys.copy(), srs)
        print("   说话人过滤后 %.2fs" % (len(out5) / srs))
        sf.write(os.path.join(OUT, "twospeaker_after.wav"), out5, srs)
    except Exception as exc:
        print("   说话人过滤异常：%r" % exc)
else:
    print("   文件不存在，跳过")

print("完成。输出样例在", OUT)
