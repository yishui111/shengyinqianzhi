# -*- coding: utf-8 -*-
"""端到端冒烟：走完整 process_one 流水线（GPU）。
用例：
  1. 干净录音 + 只勾「切片」      -> 修复点：必须出切片
  2. 带 BGM 素材 + 人声分离       -> 输出应只有人声
  3. 混合素材 + 人声分离+去掌声   -> 掌声段应被剪掉
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pre_service"))
import preprocess_api as P  # noqa: E402

TESTS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(TESTS, "diag_out")
os.makedirs(OUT, exist_ok=True)


def run(case, src, do_sep=False, do_reverb=False, do_speaker=False, do_denoise=False,
        do_boost=False, boost_gain=2.0, do_clean=False, do_norm=False):
    d = os.path.join(OUT, "e2e_%s" % case)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    print("-" * 60)
    print("用例 %s：%s 选项=%s" % (case, os.path.basename(src),
          dict(sep=do_sep, reverb=do_reverb, speaker=do_speaker, denoise=do_denoise,
               clean=do_clean)))
    outs = P.process_one(src, d, do_sep, do_reverb, do_speaker, do_denoise,
                         do_boost, boost_gain, do_clean, do_norm, 44100)
    total = sum(P.duration_sec(o) for o in outs)
    print("  -> 输出 %d 个文件，共 %.1f 秒：" % (len(outs), total))
    for o in outs[:8]:
        print("     %s（%.1fs）" % (os.path.basename(o), P.duration_sec(o)))
    if len(outs) > 8:
        print("     ... 等 %d 个" % len(outs))
    return outs


# 1. 干净录音只切片（修复前：0 个切片）
s1 = run("slice", os.path.join(TESTS, "test_speech.wav"), do_clean=True)

# 2. 带 BGM 只人声分离
s2 = run("bgm", os.path.join(TESTS, "test_bgm.wav"), do_sep=True)

# 3. 混合素材：人声分离 + 去掌声
s3 = run("mix", os.path.join(TESTS, "test_mixed_bgm.wav"), do_sep=True, do_denoise=True)

print("=" * 60)
print("端到端冒烟完成")
