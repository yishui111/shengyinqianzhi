# -*- coding: utf-8 -*-
"""ASR 冒烟测试：加载 SenseVoice 模型，识别一段真实人声，打印原始输出与清洗结果。

用法（先运行 tests/download_asr_model.py 把模型下到 models/asr）：
  python tests/test_asr.py <音频或视频文件路径>

模型与缓存目录都按 <项目根> 自动推导，可在任意位置部署后直接使用。
"""
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASR_DIR = os.path.join(PROJECT_ROOT, "models", "asr")

if len(sys.argv) < 2:
    sys.exit("用法：python tests\\test_asr.py <音频或视频文件路径>"
             "（若未下载模型，先运行 tests\\download_asr_model.py）")
TEST_WAV = sys.argv[1]

os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "models", "hf_cache"))

t0 = time.time()
from funasr import AutoModel  # noqa: E402
import torch  # noqa: E402

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device, file=sys.stderr)

model = AutoModel(
    model=os.path.join(ASR_DIR, "SenseVoiceSmall"),
    trust_remote_code=True,
    remote_code=os.path.join(ASR_DIR, "SenseVoiceSmall", "model.py"),
    vad_model=os.path.join(ASR_DIR, "fsmn_vad"),
    vad_kwargs={"max_single_segment_time": 30000},
    device=device,
    disable_update=True,
)
print("模型加载耗时 %.1fs" % (time.time() - t0), file=sys.stderr)

res = model.generate(
    input=TEST_WAV,
    cache={},
    language="auto",
    use_itn=True,
    batch_size_s=60,
)

print("===== 原始输出 =====")
for r in res:
    print(r)
print("===== 清洗后 =====")
for r in res:
    text = r.get("text", "")
    text = re.sub(r"<\|[^|]+\|>", "", text)
    text = re.sub(r"^\d+", "", text).strip()
    print("seg:", text)
