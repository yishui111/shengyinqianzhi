# -*- coding: utf-8 -*-
"""
下载语音转文字（ASR）模型到项目内 models/asr（一次性，之后离线可用）：
  1. SenseVoiceSmall  —— 阿里达摩院多语言/方言语音识别主模型（约 1GB）
  2. speech_fsmn_vad  —— 语音活动检测（VAD），长音频分段用（约 50MB）

用法（在本项目根目录下执行）：
  runtime/py312/python.exe tests/download_asr_model.py   # 作者自带便携运行时
  python tests/download_asr_model.py                      # 全新环境（需 pip install modelscope）

下载目录自动取 <项目根>/models/asr，与主服务 pre_service/preprocess_api.py 的默认路径一致，
放到哪台机器都能用；下载完成后记录版本/时间到 models/asr/download_info.txt 供复查。
"""
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(PROJECT_ROOT, "models", "asr")
os.makedirs(BASE, exist_ok=True)
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(BASE, "_cache"))

from modelscope import snapshot_download  # noqa: E402


def main():
    t0 = time.time()
    p1 = snapshot_download("iic/SenseVoiceSmall", local_dir=os.path.join(BASE, "SenseVoiceSmall"))
    print("SenseVoiceSmall ->", p1, "耗时 %.1fs" % (time.time() - t0))
    p2 = snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                           local_dir=os.path.join(BASE, "fsmn_vad"))
    print("fsmn_vad ->", p2, "耗时 %.1fs" % (time.time() - t0))
    with open(os.path.join(BASE, "download_info.txt"), "w", encoding="utf-8") as f:
        f.write("SenseVoiceSmall: %s\nfsmn_vad: %s\ntime: %s\n" % (
            p1, p2, time.strftime("%Y-%m-%d %H:%M:%S")))
    print("ALL DONE")
    print("模型目录：", BASE)


if __name__ == "__main__":
    main()
