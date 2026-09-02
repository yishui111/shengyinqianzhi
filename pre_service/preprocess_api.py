# -*- coding: utf-8 -*-
"""
声音素材前置处理项目（E:\\shengyinqianzhi）
=============================================================
把各种音频/视频素材处理成适合训练模型的干净人声：
  1. ffmpeg 从视频/音频提取 44.1k 单声道 wav
  2. pymss 人声分离（去 BGM）bs_roformer_voc_hyperacev2
  3. pymss 去混响（dereverb_mel_band_roformer）
  4. 可选：去静音/切片（1.5~20 秒）+ 质量过滤 + 响度归一化(-16 LUFS)

复用换声项目运行时与模型（不重新下载）：
  E:\\tihuanshengyin\\runtime\\py312  /  rvc_service\\pymss_models  /  runtime\\ffmpeg
"""

import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

# 项目根目录随脚本位置推导（盘符可变，整块硬盘换盘符也能直接用）；
# 换声项目运行时默认取同盘根目录兄弟项目（自动跟随盘符），可用 HUANSHENG_ROOT 环境变量覆盖
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUANSHENG_ROOT = os.environ.get("HUANSHENG_ROOT") or os.path.join(
    os.path.dirname(PROJECT_ROOT), "tihuanshengyin")


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


# 自包含优先：项目内自带运行时/ffmpeg/模型，换电脑直接拷整个文件夹即可；找不到再回退换声项目
PY312 = _first_existing(
    os.path.join(PROJECT_ROOT, "runtime", "py312", "python.exe"),
    os.path.join(HUANSHENG_ROOT, "runtime", "py312", "python.exe"))
FFMPEG = _first_existing(
    os.path.join(PROJECT_ROOT, "ffmpeg", "bin", "ffmpeg.exe"),
    os.path.join(HUANSHENG_ROOT, "runtime", "ffmpeg", "bin", "ffmpeg.exe"))
FFPROBE = _first_existing(
    os.path.join(PROJECT_ROOT, "ffmpeg", "bin", "ffprobe.exe"),
    os.path.join(HUANSHENG_ROOT, "runtime", "ffmpeg", "bin", "ffprobe.exe"))
PYMSS_MODEL_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "models", "pymss"),
    os.path.join(HUANSHENG_ROOT, "rvc_service", "pymss_models"))
SPK_SAVEDIR = _first_existing(
    os.path.join(PROJECT_ROOT, "models", "speaker"),
    os.path.join(HUANSHENG_ROOT, "runtime", "cache", "hf_speaker_model"))
HF_CACHE_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "models", "hf_cache"),
    os.path.join(HUANSHENG_ROOT, "runtime", "cache", "huggingface"))
VOCAL_MODEL = "bs_roformer_voc_hyperacev2"
DEREVERB_MODEL = "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt"
PORT = int(os.environ.get("PRE_PORT", "8070"))
INPUT_ROOT = os.path.join(PROJECT_ROOT, "ziliao", "input")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "ziliao", "output")
MERGED_ROOT = os.path.join(OUTPUT_ROOT, "merged")  # 「合成一个音频」输出目录
ASR_DIR = os.path.join(PROJECT_ROOT, "models", "asr")  # 语音转文字模型（SenseVoice + VAD）
ASR_OUT_ROOT = os.path.join(PROJECT_ROOT, "ziliao", "asr")  # 语音转文字结果（txt）
MAX_SEC_PER_CHUNK = int(os.environ.get("SEPARATE_MAX_SEC", "90"))
MIN_DUR = 1.5
MAX_DUR = 20.0
MIN_RMS = 0.015
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".webm", ".m4v")

os.makedirs(INPUT_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("preprocess")

_sep = None
_dereverb = None
_spk_model = None
_asr_model = None
_model_lock = threading.Lock()
_task_lock = threading.Lock()
_asr_lock = threading.Lock()  # ASR 模型不并发（SenseVoice 单实例）
_state = {"running": False, "task": "", "step": "", "ok": False, "error": "", "log": [], "results": []}


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    with _task_lock:
        _state["log"].append(line)
        _state["log"] = _state["log"][-600:]
    logger.info(msg)


def set_state(**kw):
    with _task_lock:
        _state.update(kw)


def _device():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 6 * 1024 ** 3:
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _get_separator():
    global _sep
    if _sep is None:
        from pymss import MSSeparator
        os.makedirs(PYMSS_MODEL_DIR, exist_ok=True)
        log("初始化人声分离模型（%s，设备 %s）" % (VOCAL_MODEL, _device()))
        _sep = MSSeparator.from_model_name(
            VOCAL_MODEL, model_dir=PYMSS_MODEL_DIR, download=False,
            source="hf-mirror", device=_device(), output_format="wav",
            inference_params={"normalize": True},
        )
    return _sep


def _get_dereverb():
    global _dereverb
    if _dereverb is None:
        from pymss import MSSeparator
        os.makedirs(PYMSS_MODEL_DIR, exist_ok=True)
        log("初始化去混响模型（%s）" % DEREVERB_MODEL)
        _dereverb = MSSeparator.from_model_name(
            DEREVERB_MODEL, model_dir=PYMSS_MODEL_DIR, download=False,
            source="hf-mirror", device=_device(), output_format="wav",
            inference_params={"normalize": True},
        )
    return _dereverb


def _get_speaker_model():
    """ECAPA 声纹模型（speechbrain），用于多人说话时只保留主要说话人。"""
    global _spk_model
    if _spk_model is None:
        import os as _os
        _os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy
        log("初始化说话人识别模型（ECAPA）")
        _spk_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=SPK_SAVEDIR,
            local_strategy=LocalStrategy.COPY,
        )
    return _spk_model


def duration_sec(path):
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path], stderr=subprocess.DEVNULL)
        return float(out.decode().strip())
    except Exception:  # noqa: BLE001
        return 0.0


def extract_audio(src, dst_wav):
    """任意音频/视频 -> 44.1k 单声道 wav"""
    subprocess.run([FFMPEG, "-y", "-i", src, "-ar", "44100", "-ac", "1", dst_wav],
                   check=True, capture_output=True)


def voice_segments(y, sr, min_dur=MIN_DUR, max_dur=MAX_DUR, silence_gap=0.45):
    """基于能量切出连续语音段：静音超 0.45s 分段，长段按 20s 拆开。"""
    win = int(sr * 0.03)
    hop = int(sr * 0.01)
    n = len(y)
    if n <= win:
        return [(0, n)] if n >= min_dur * sr else []
    rms = np.array([float(np.sqrt((y[i:i + win] ** 2).mean()))
                    for i in range(0, max(n - win, 1), hop)])
    voiced = rms > MIN_RMS
    segs = []
    start = None
    for i, v in enumerate(voiced):
        t = i * 0.01
        if v and start is None:
            start = i
        elif not v and start is not None:
            if t - start * 0.01 > silence_gap:
                segs.append((start * hop, i * hop))
                start = None
    if start is not None:
        segs.append((start * hop, n))
    out = []
    for a, b in segs:
        if b - a < min_dur * sr:
            continue
        while b - a > max_dur * sr:
            out.append((a, a + int(max_dur * sr)))
            a += int(max_dur * sr)
        if b - a >= min_dur * sr:
            out.append((a, b))
    return out


def loudness_normalize(src_wav, dst_wav):
    subprocess.run(
        [FFMPEG, "-y", "-i", src_wav, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", dst_wav],
        check=True, capture_output=True)


def _voiced_ratio(seg, sr):
    """有声帧占比：掌声/欢呼/杂音没有清晰基频，占比很低。"""
    try:
        import parselmouth
        snd = parselmouth.Sound(seg, int(sr))
        pitch = snd.to_pitch_ac(time_step=0.02, voicing_threshold=0.35,
                                pitch_floor=60, pitch_ceiling=500)
        f0 = pitch.selected_array["frequency"]
        return float((f0 > 0).mean()) if len(f0) else 0.0
    except Exception:  # noqa: BLE001
        return 1.0


def _smooth_join(pieces, sr, fade=0.08):
    """交叉淡化拼接多段音频，避免段间静音导致的卡顿。"""
    if not pieces:
        return np.zeros(0, dtype="float32")
    if len(pieces) == 1:
        return pieces[0]
    n = int(sr * fade)
    out = pieces[0].astype("float32")
    for a in pieces[1:]:
        a = a.astype("float32")
        if len(out) < n or len(a) < n:
            out = np.concatenate([out, a])
            continue
        ramp = np.linspace(0.0, 1.0, n, dtype="float32")
        out = np.concatenate([out[:-n], out[-n:] * (1.0 - ramp) + a[:n] * ramp, a[n:]])
    return out


def remove_unvoiced_regions(y, sr, min_gap=0.7, keep_pad=0.15):
    """剪掉无基频的长片段（掌声/欢呼/长停顿）：逐帧 f0 检测，>min_gap 的全剪掉，保留两侧 pad。"""
    import parselmouth
    snd = parselmouth.Sound(y, int(sr))
    pitch = snd.to_pitch_ac(time_step=0.05, voicing_threshold=0.35,
                            pitch_floor=60, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    times = pitch.ts()
    unvoiced = f0 <= 0
    cuts = []
    start = None
    for i, uv in enumerate(unvoiced):
        if uv and start is None:
            start = i
        elif not uv and start is not None:
            if times[i] - times[start] > min_gap:
                cuts.append((times[start], times[i]))
            start = None
    if start is not None and times[-1] - times[start] > min_gap:
        cuts.append((times[start], times[-1]))
    if not cuts:
        return y
    log("去非人声：剪掉 %d 段（掌声/欢呼/长停顿）" % len(cuts))
    pieces = []
    prev = 0
    pad = keep_pad * sr
    for a, b in cuts:
        aa = max(0, int(a * sr) - int(pad))
        bb = min(len(y), int(b * sr) + int(pad))
        if aa > prev:
            pieces.append(y[prev:aa])
        prev = bb
    if prev < len(y):
        pieces.append(y[prev:])
    return _smooth_join(pieces, sr)


def keep_main_speaker(y, sr, min_seg=1.0, sim_thr=0.45, voiced_thr=0.25):
    """只保留主要说话人，并去掉掌声/欢呼等非人声：
    切段→基频过滤（去非语音）→声纹聚类（去其他说话人）→交叉淡化拼接。"""
    import torch
    import librosa

    segs = voice_segments(y, sr, min_dur=min_seg, max_dur=30.0, silence_gap=0.6)
    if len(segs) <= 1:
        return y
    model = _get_speaker_model()
    cands = []
    for a, b in segs:
        seg = y[a:b]
        if len(seg) < int(0.6 * sr):
            continue
        vr = _voiced_ratio(seg, sr)
        if vr < voiced_thr:
            log("去非人声：去掉 [%.1f-%.1f]（掌声/欢呼，有声占比 %.2f）" % (a / sr, b / sr, vr))
            continue
        w16 = librosa.resample(seg, orig_sr=sr, target_sr=16000)
        t = torch.from_numpy(np.asarray(w16, dtype="float32")).unsqueeze(0)
        e = model.encode_batch(t).squeeze(0).squeeze(0)
        cands.append((a, b, seg, e))
    if not cands:
        log("说话人过滤：没有留下任何语音段")
        return y
    if len(cands) == 1:
        return cands[0][2]
    E = torch.stack([c[3] for c in cands])
    E = E / E.norm(dim=1, keepdim=True)
    sim = (E @ E.T).cpu().numpy()
    center = int(sim.mean(axis=1).argmax())
    keep = [i for i in range(len(cands)) if sim[center, i] >= sim_thr]
    if len(keep) < len(cands):
        log("说话人识别：共 %d 段，保留主要说话人 %d 段，去掉其他说话人 %d 段" % (
            len(cands), len(keep), len(cands) - len(keep)))
    else:
        log("说话人识别：检测到单一说话人，全部保留")
    keep = sorted(keep, key=lambda i: cands[i][0])
    return _smooth_join([cands[i][2] for i in keep], sr)


def process_one(src, task_dir, do_sep, do_reverb, do_speaker, do_denoise, do_boost, boost_gain, do_clean, do_norm, sample_rate):
    """处理单个文件，返回输出 wav 列表。"""
    base = os.path.splitext(os.path.basename(src))[0]
    workdir = os.path.join(task_dir, "work")
    os.makedirs(workdir, exist_ok=True)
    raw_wav = os.path.join(workdir, "input.wav")
    log("提取音频：%s（%.1f 秒）" % (base, duration_sec(src)))
    extract_audio(src, raw_wav)
    y, sr = sf.read(raw_wav)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype="float32")

    if not (do_sep or do_reverb or do_speaker or do_denoise or do_boost or do_clean or do_norm):
        log("未勾选任何处理项：仅提取音频（视频转音频，原样输出，不做处理）")

    if do_sep:
        log("人声分离：%s" % base)
        sep = _get_separator()
        chunk_len = MAX_SEC_PER_CHUNK * sr
        chunks = []
        for s in range(0, len(y), chunk_len):
            seg = y[s:s + chunk_len]
            stems = sep.separate(seg, pbar=False, stems=["vocals"])
            v = np.asarray(stems["vocals"], dtype="float32")
            if v.ndim > 1:
                v = v.mean(axis=1)
            chunks.append(v)
        y = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if _device() == "cuda":
            import torch
            torch.cuda.empty_cache()

    if do_reverb:
        log("去混响：%s" % base)
        dereverb = _get_dereverb()
        chunk_len = MAX_SEC_PER_CHUNK * sr
        chunks = []
        for s in range(0, len(y), chunk_len):
            stems = dereverb.separate(y[s:s + chunk_len], pbar=False)
            clean = stems.get("noreverb")
            if clean is None:
                clean = stems.get("vocals")
            if clean is None:
                clean = list(stems.values())[0]
            c = np.asarray(clean, dtype="float32")
            if c.ndim > 1:
                c = c.mean(axis=1)
            chunks.append(c)
        y = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if _device() == "cuda":
            import torch
            torch.cuda.empty_cache()

    if do_speaker:
        log("说话人过滤（只保留主要说话人）：%s" % base)
        try:
            y = keep_main_speaker(y, sr)
        except Exception as exc:  # noqa: BLE001
            log("说话人过滤不可用（%s），保持原样" % exc)

    if do_denoise:
        # 去掌声/欢呼/长停顿/无基频音效：剪掉连续无基频的长片段
        log("去掌声/音效：%s" % base)
        try:
            y = remove_unvoiced_regions(y, sr)
        except Exception as exc:  # noqa: BLE001
            log("去掌声/音效不可用（%s），保持原样" % exc)

    # 输出采样率重采样
    if sr != sample_rate:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    if do_boost:
        # 提高音量：整体增益放大（防削波 clip），人声偏小/远距离录音适用
        gain = max(0.1, min(float(boost_gain), 20.0))
        log("提高音量：%.1f 倍" % gain)
        y = np.clip(y * gain, -1.0, 1.0)

    if do_clean:
        segs = voice_segments(y, sr)
        log("%s 切片：%d 段" % (base, len(segs)))
        outs = []
        for i, (a, b) in enumerate(segs):
            seg = y[a:b]
            p = os.path.join(task_dir, "%s_p%02d.wav" % (base, i))
            sf.write(p, seg, sr)
            if do_norm:
                tmp = p + ".tmp.wav"
                loudness_normalize(p, tmp)
                shutil.move(tmp, p)
            outs.append(p)
        return outs

    p = os.path.join(task_dir, "%s_clean.wav" % base)
    sf.write(p, y, sr)
    if do_norm:
        tmp = p + ".tmp.wav"
        loudness_normalize(p, tmp)
        shutil.move(tmp, p)
    return [p]


def run_task(files, opts):
    # running 状态已在请求入口原子置位；这里只负责执行
    task_id = time.strftime("%Y%m%d_%H%M%S")
    task_dir = os.path.join(OUTPUT_ROOT, task_id)
    os.makedirs(task_dir, exist_ok=True)
    set_state(task=task_id, task_dir=task_dir, step="开始处理 %d 个文件" % len(files))
    results = []
    try:
        for i, src in enumerate(files, 1):
            set_state(step="[%d/%d] 处理 %s" % (i, len(files), os.path.basename(src)))
            outs = process_one(
                src, task_dir,
                do_sep=opts.get("separate", False),
                do_reverb=opts.get("dereverb", False),
                do_speaker=opts.get("speaker", False),
                do_denoise=opts.get("denoise", False),
                do_boost=opts.get("boost", False),
                boost_gain=float(opts.get("boost_gain", 2.0)),
                do_clean=opts.get("clean", False),
                do_norm=opts.get("normalize", False),
                sample_rate=int(opts.get("sample_rate", 44100)),
            )
            for o in outs:
                results.append({"file": os.path.basename(o), "size": os.path.getsize(o),
                                "duration": round(duration_sec(o), 1)})
        set_state(running=False, ok=True, step="全部完成", results=results)
        log("完成：共输出 %d 个文件 -> %s" % (len(results), task_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("处理失败")
        set_state(running=False, ok=False, error="处理失败：%s" % exc)
        log("✘ 失败：%s" % exc)
    finally:
        set_state(running=False)


def start_task(files, opts):
    """原子地开始一个任务：已运行则拒绝，避免并发竞态互相覆盖。"""
    with _task_lock:
        if _state["running"]:
            return False
        _state.update(running=True, ok=False, error="", log=[], results=[], task="", step="启动")
    threading.Thread(target=run_task, args=(files, opts), daemon=True).start()
    return True


# ---------- 合成一个音频（把生成好的结果全部拼成一个 wav） ----------

def collect_result_wavs(task_id=None):
    """收集待合并的 wav：
    - task_id 指定单个任务目录（如 "20260819_235149"）
    - task_id 为 None / "all" 时收集所有任务目录的顶层 wav
    只取任务目录顶层的最终结果（*_clean.wav / *_pNN.wav），跳过 work 子目录与 merged 目录。"""
    wavs = []
    if task_id and task_id != "all":
        d = os.path.join(OUTPUT_ROOT, task_id)
        if not os.path.isdir(d):
            raise ValueError("任务不存在：%s" % task_id)
        wavs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".wav")]
    else:
        for d in sorted(os.listdir(OUTPUT_ROOT)):
            dd = os.path.join(OUTPUT_ROOT, d)
            if not os.path.isdir(dd) or d == "merged":
                continue
            for f in sorted(os.listdir(dd)):
                if f.lower().endswith(".wav"):
                    wavs.append(os.path.join(dd, f))
    return wavs


def merge_wavs(wavs, out_path, sample_rate=44100):
    """把多个 wav 按顺序拼成一个 wav：
    先各自统一转成 44.1k 单声道 pcm_s16le（兼容不同参数），再 concat 无损拼接。"""
    tmpdir = tempfile.mkdtemp(prefix="merge_")
    try:
        pieces = []
        for i, w in enumerate(wavs):
            p = os.path.join(tmpdir, "p%04d.wav" % i)
            subprocess.run(
                [FFMPEG, "-y", "-i", w, "-ar", str(sample_rate), "-ac", "1",
                 "-c:a", "pcm_s16le", p],
                check=True, capture_output=True)
            pieces.append(p)
        listf = os.path.join(tmpdir, "list.txt")
        with open(listf, "w", encoding="utf-8") as fp:
            for p in pieces:
                fp.write("file '%s'\n" % p.replace("\\", "/").replace("'", "'\\''"))
        subprocess.run(
            [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", listf,
             "-c", "copy", out_path],
            check=True, capture_output=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path


# ---------- 语音转文字（ASR，funasr + SenseVoice） ----------

def _get_asr():
    """懒加载 SenseVoice 语音识别模型（本地，不联网）。"""
    global _asr_model
    if _asr_model is None:
        import torch
        from funasr import AutoModel
        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log("初始化语音识别模型（SenseVoice，设备 %s）" % device)
        _asr_model = AutoModel(
            model=os.path.join(ASR_DIR, "SenseVoiceSmall"),
            trust_remote_code=True,
            remote_code=os.path.join(ASR_DIR, "SenseVoiceSmall", "model.py"),
            vad_model=os.path.join(ASR_DIR, "fsmn_vad"),
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
        )
    return _asr_model


def _clean_sensevoice(text):
    """去掉 SenseVoice 输出的 <|zh|><|NEUTRAL|><|Speech|><|withitn|> 等标签。"""
    import re
    text = re.sub(r"<\|[^|]+\|>", "", text or "")
    text = re.sub(r"^\d+\s*", "", text)
    return text.strip()


def transcribe_file(src):
    """识别单个音频/视频文件，返回 {text, segments, duration}。"""
    import tempfile
    t0 = time.time()
    model = _get_asr()
    tmpdir = tempfile.mkdtemp(prefix="asr_")
    try:
        wav16 = os.path.join(tmpdir, "input16k.wav")
        # SenseVoice 要求 16k 单声道 wav
        subprocess.run([FFMPEG, "-y", "-i", src, "-ar", "16000", "-ac", "1", wav16],
                       check=True, capture_output=True)
        res = model.generate(input=wav16, cache={}, language="auto",
                             use_itn=True, batch_size_s=60)
        segs = []
        for r in res:
            t = _clean_sensevoice(r.get("text", ""))
            if t:
                segs.append(t)
        text = "\n".join(segs)
        log("语音转文字：%s（%d 段，耗时 %.1fs）" % (
            os.path.basename(src), len(segs), time.time() - t0))
        return {"text": text, "segments": segs, "duration": round(duration_sec(src), 1)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


app = FastAPI(title="声音素材前置处理")

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>声音素材前置处理</title>
<style>
body{font-family:"Microsoft YaHei",sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#222}
h1{border-bottom:2px solid #2e7d32;padding-bottom:8px}
.card{background:#eff7ef;border:1px solid #ddd;border-radius:10px;padding:18px;margin:16px 0}
label{display:block;margin:10px 0 4px;font-weight:600}
input[type=file]{width:100%}
select,input[type=number],input[type=text]{padding:6px;width:260px}
button{background:#2e7d32;color:#fff;border:none;padding:10px 26px;border-radius:6px;font-size:16px;cursor:pointer;margin-top:14px}
button:disabled{background:#aaa}
pre{background:#f0f0f0;padding:12px;border-radius:6px;overflow-x:auto;min-height:120px;max-height:320px;overflow-y:auto}
.err{color:#c0392b;margin-top:8px}
.ok{color:#27ae60;margin-top:8px}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:6px 10px;text-align:left;font-size:14px}
</style>
</head>
<body>
<h1>🎙 声音素材前置处理（训练用干净人声）</h1>
<div class="card">
  <label>① 选择音频/视频文件（可多选，支持 wav/mp3/m4a/flac + mp4/mkv/mov 等）</label>
  <input type="file" id="files" multiple>
  <label>② 或填服务器上的文件夹路径（直接批量读取该目录所有音视频）</label>
  <input type="text" id="dirpath" placeholder="例如 盘符:\\素材\\视频合集（如 F:\\素材\\视频合集）">
  <label>处理选项（<b>什么都不勾 = 只把视频/音频原样转成 44.1k 单声道 wav，不做任何处理</b>，适合方言/干净录音）</label>
  <label style="font-weight:400"><input type="checkbox" id="separate"> 人声分离（去背景音乐/伴奏）</label>
  <label style="font-weight:400"><input type="checkbox" id="dereverb"> 去混响</label>
  <label style="font-weight:400"><input type="checkbox" id="speaker"> 只保留主要说话人（去掉视频里其他人的声音）</label>
  <label style="font-weight:400"><input type="checkbox" id="denoise"> 去掌声/欢呼/音效（剪掉无基频的长片段）</label>
  <label style="font-weight:400"><input type="checkbox" id="boost"> 提高人声音量（人声偏小/录音太远时用）</label>
  <label style="font-weight:400">增益倍数 <input type="number" id="boostGain" value="2" min="1" max="20" step="0.5" style="width:80px">（2 = 音量翻倍）</label>
  <label style="font-weight:400"><input type="checkbox" id="clean"> 额外生成训练切片（1.5~20 秒一段，可选）</label>
  <label style="font-weight:400"><input type="checkbox" id="normalize"> 响度归一化(-16 LUFS)</label>
  <label>输出采样率（训练推荐 44100）</label>
  <select id="sr">
    <option value="44100">44100 Hz（推荐）</option>
    <option value="32000">32000 Hz</option>
    <option value="16000">16000 Hz</option>
  </select>
  <br>
  <button id="btn">开始处理</button>
  <div id="msg"></div>
</div>
<div class="card">
  <b>处理日志</b> <button onclick="poll()" style="margin:0 8px">刷新</button>
  <pre id="log">还没有任务</pre>
</div>
<div class="card">
  <b>处理结果</b>
  <div id="result"></div>
</div>
<div class="card">
  <b>🎬 合成一个音频</b>
  <label>合并范围（把选中的结果 wav 按顺序拼成一个音频）</label>
  <select id="mergeTask"></select>
  <br>
  <button id="btnMerge" onclick="doMerge()">合成并下载</button>
  <div id="msgMerge"></div>
</div>
<div class="card">
  <b>📝 语音转文字</b>
  <label>① 选择音频/视频文件（可多选，方言/普通话都支持）</label>
  <input type="file" id="asrFiles" multiple>
  <label>② 或填服务器上的文件路径</label>
  <input type="text" id="asrPath" placeholder="例如 盘符:\\素材\\xxx.wav（如 F:\\素材\\xxx.wav）">
  <br>
  <button id="btnAsr" onclick="doAsr()">识别文字</button>
  <div id="msgAsr"></div>
  <div id="asrResult"></div>
</div>
<script>
async function poll(){
  const r = await fetch('/api/status'); const j = await r.json();
  document.getElementById('log').textContent = (j.log||[]).join('\\n') || '还没有任务';
  document.getElementById('msg').textContent = j.running ? ('处理中：' + (j.step||'')) : (j.ok ? '✔ 处理完成' : (j.error ? '✘ ' + j.error : ''));
  document.getElementById('btn').disabled = !!j.running;
  const rs = document.getElementById('result');
  if (j.results && j.results.length){
    let h = '<p>输出目录：<code>'+ (j.task_dir||'') +'</code></p>';
    h += '<table><tr><th>文件</th><th>时长</th><th>试听</th><th>下载</th></tr>';
    j.results.forEach(x=>{
      const u = '/api/result/'+j.task+'/file/'+encodeURIComponent(x.file);
      h += '<tr><td>'+x.file+'</td><td>'+x.duration+' s</td>'
         + '<td><audio controls preload="none" src="'+u+'" style="width:240px"></audio></td>'
         + '<td><a href="'+u+'" download>⬇ wav</a></td></tr>';
    });
    h += '</table><br><a href="/api/result/'+j.task+'/zip">⬇ 下载全部结果 zip</a>';
    rs.innerHTML = h;
  }
}
document.getElementById('btn').onclick = async ()=>{
  const files = Array.from(document.getElementById('files').files||[]);
  const dir = document.getElementById('dirpath').value.trim();
  const opts = {
    separate: document.getElementById('separate').checked,
    dereverb: document.getElementById('dereverb').checked,
    speaker: document.getElementById('speaker').checked,
    denoise: document.getElementById('denoise').checked,
    boost: document.getElementById('boost').checked,
    boost_gain: parseFloat(document.getElementById('boostGain').value) || 2,
    clean: document.getElementById('clean').checked,
    normalize: document.getElementById('normalize').checked,
    sample_rate: document.getElementById('sr').value,
  };
  let r;
  if (files.length){
    const fd = new FormData();
    fd.append('opts', JSON.stringify(opts));
    files.forEach(f=>fd.append('files', f));
    r = await fetch('/api/process', {method:'POST', body: fd});
  } else if (dir){
    r = await fetch('/api/process_dir', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({dir, ...opts})});
  } else {
    document.getElementById('msg').textContent = '请先选择文件或填文件夹路径';
    return;
  }
  const j = await r.json(); document.getElementById('msg').textContent = j.detail || j.message || '';
  setTimeout(poll, 800);
};
async function loadTasks(){
  try{
    const r = await fetch('/api/tasks'); const j = await r.json();
    const sel = document.getElementById('mergeTask');
    const tasks = j.tasks||[];
    sel.innerHTML = tasks.map(t=>'<option value="'+t.id+'">'+t.id+'（'+t.count+' 个文件）</option>').join('')
      + '<option value="all">全部输出（所有任务的结果合并，慎用）</option>';
    if (sel.options.length > 1) sel.selectedIndex = 0; // 默认选最近一次任务（一个队列）
  }catch(e){}
}
async function doMerge(){
  const btn = document.getElementById('btnMerge'); btn.disabled = true;
  const msg = document.getElementById('msgMerge');
  msg.textContent = '正在合成…（文件多时需等待）';
  try{
    const r = await fetch('/api/merge', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({task: document.getElementById('mergeTask').value})});
    if (!r.ok){ let e = {}; try{ e = await r.json(); }catch(_){} throw new Error(e.detail || ('合并失败（HTTP ' + r.status + '）')); }
    const blob = await r.blob();
    const cd = r.headers.get('Content-Disposition') || '';
    const m = cd.match(/filename="?([^";]+)"?/);
    const fn = m ? m[1] : ('merged_' + Date.now() + '.wav');
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = fn;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    msg.textContent = '✔ 已合成并开始下载：' + fn;
  }catch(e){ msg.textContent = '✘ ' + e.message; }
  btn.disabled = false;
}
async function doAsr(){
  const btn = document.getElementById('btnAsr'); btn.disabled = true;
  const msg = document.getElementById('msgAsr');
  msg.textContent = '识别中…（首次会加载模型约 10 秒，长音频需等待）';
  const files = Array.from(document.getElementById('asrFiles').files||[]);
  const path = document.getElementById('asrPath').value.trim();
  try{
    let r;
    if (files.length){
      const fd = new FormData();
      files.forEach(f=>fd.append('files', f));
      r = await fetch('/api/asr', {method:'POST', body: fd});
    } else if (path){
      r = await fetch('/api/asr_path', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({path})});
    } else {
      msg.textContent = '请先选择文件或填服务器路径'; btn.disabled = false; return;
    }
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || ('识别失败（HTTP ' + r.status + '）'));
    let h = '';
    (j.results||[]).forEach(x=>{
      h += '<div style="border:1px solid #ccc;border-radius:6px;padding:10px;margin:10px 0">';
      h += '<b>'+x.filename+'</b>（'+x.duration+' 秒） ';
      h += '<button onclick="copyText(this)" data-t="'+encodeURIComponent(x.text||'')+'" style="margin:0 6px;padding:4px 12px;font-size:13px">复制全文</button> ';
      h += '<a href="'+x.txt_url+'" download style="font-size:13px">⬇ 下载 txt 文档</a>';
      h += '<pre style="white-space:pre-wrap;min-height:40px">'+(x.text||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</pre>';
      h += '</div>';
    });
    document.getElementById('asrResult').innerHTML = h;
    msg.textContent = '✔ 识别完成';
  }catch(e){ msg.textContent = '✘ ' + e.message; }
  btn.disabled = false;
}
function copyText(btn){
  const t = decodeURIComponent(btn.getAttribute('data-t'));
  navigator.clipboard.writeText(t).then(()=>{ btn.textContent='已复制 ✓'; setTimeout(()=>btn.textContent='复制全文',1500); });
}
setInterval(poll, 2500); poll(); loadTasks();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "preprocess", "port": PORT}


@app.get("/api/status")
def status():
    with _task_lock:
        return dict(_state)


def _collect_folder(dirpath):
    if not os.path.isdir(dirpath):
        raise ValueError("文件夹不存在：%s" % dirpath)
    files = []
    for root, _, names in os.walk(dirpath):
        for n in sorted(names):
            if n.lower().endswith(AUDIO_EXTS + VIDEO_EXTS):
                files.append(os.path.join(root, n))
    if not files:
        raise ValueError("该文件夹里没有音频/视频文件")
    return files


@app.post("/api/process")
async def process_upload(files: list[UploadFile] = File(...), opts: str = Form("{}")):
    o = json.loads(opts or "{}")
    task_id = time.strftime("%Y%m%d_%H%M%S")
    up_dir = os.path.join(INPUT_ROOT, task_id)
    os.makedirs(up_dir, exist_ok=True)
    paths = []
    for f in files:
        p = os.path.join(up_dir, os.path.basename(f.filename or "file.wav"))
        with open(p, "wb") as fp:
            shutil.copyfileobj(f.file, fp)
        paths.append(p)
    if not start_task(paths, o):
        return JSONResponse({"detail": "已有任务在处理中，请稍候再试"}, status_code=400)
    return {"message": "任务已开始（%d 个文件）" % len(paths)}


@app.post("/api/process_dir")
def process_dir(req: dict):
    try:
        files = _collect_folder(req.get("dir", ""))
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    if not start_task(files, req):
        return JSONResponse({"detail": "已有任务在处理中，请稍候再试"}, status_code=400)
    return {"message": "任务已开始（%d 个文件）" % len(files)}


@app.get("/api/result/{task_id}/zip")
def result_zip(task_id: str):
    d = os.path.join(OUTPUT_ROOT, task_id)
    if not os.path.isdir(d):
        raise ValueError("任务不存在")
    wavs = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav")]
    zip_path = os.path.join(d, "%s_results.zip" % task_id)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for w in wavs:
            z.write(w, os.path.basename(w))
    return FileResponse(zip_path, media_type="application/zip",
                        filename="%s_results.zip" % task_id)


@app.get("/api/result/{task_id}/file/{filename}")
def result_file(task_id: str, filename: str):
    """直接下载/试听单个结果 wav（不再需要 zip 解压）。"""
    d = os.path.join(OUTPUT_ROOT, task_id)
    p = os.path.join(d, os.path.basename(filename))
    if not os.path.isfile(p):
        raise ValueError("文件不存在")
    return FileResponse(p, media_type="audio/wav", filename=os.path.basename(p))


@app.get("/api/tasks")
def tasks_api():
    """列出有输出 wav 的任务目录（最近 10 个），供「合成一个音频」选择。"""
    out = []
    for d in sorted(os.listdir(OUTPUT_ROOT), reverse=True):
        dd = os.path.join(OUTPUT_ROOT, d)
        if not os.path.isdir(dd) or d == "merged":
            continue
        n = len([f for f in os.listdir(dd) if f.lower().endswith(".wav")])
        if n:
            out.append({"id": d, "count": n})
        if len(out) >= 10:
            break
    return {"tasks": out}


@app.post("/api/merge")
def merge_api(req: dict = None):
    """把生成好的结果 wav 按顺序合成一个音频并直接下载。
    body: {"task": "20260819_235149"} 合并指定任务；{"task": "all"} 或省略合并全部输出。"""
    req = req or {}
    task = req.get("task") or "all"
    try:
        wavs = collect_result_wavs(task)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    if not wavs:
        return JSONResponse({"detail": "没有可合并的输出文件"}, status_code=400)
    os.makedirs(MERGED_ROOT, exist_ok=True)
    name = "merged_%s_%s.wav" % (task if task != "all" else "all",
                                 time.strftime("%Y%m%d_%H%M%S"))
    out_path = os.path.join(MERGED_ROOT, name)
    log("合并 %d 个文件 -> %s" % (len(wavs), out_path))
    try:
        merge_wavs(wavs, out_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("合并失败")
        log("✘ 合并失败：%s" % exc)
        return JSONResponse({"detail": "合并失败：%s" % exc}, status_code=500)
    log("合并完成：%s（%d 个文件，共 %.1f 秒）" % (name, len(wavs), duration_sec(out_path)))
    return FileResponse(out_path, media_type="audio/wav",
                        filename=os.path.basename(out_path))


# ---------- 语音转文字接口 ----------

def _asr_save_txt(ts_dir, base, text):
    """把识别文字写成 txt 文档（UTF-8），返回相对下载路径。"""
    os.makedirs(ts_dir, exist_ok=True)
    txt_path = os.path.join(ts_dir, "%s.txt" % base)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    rel = os.path.basename(ts_dir)
    return "/api/asr_file/%s/%s.txt" % (rel, base)


@app.get("/api/asr_file/{ts}/{filename}")
def asr_file(ts: str, filename: str):
    """下载语音转文字生成的 txt 文档。"""
    p = os.path.join(ASR_OUT_ROOT, ts, os.path.basename(filename))
    if not os.path.isfile(p):
        raise ValueError("文件不存在")
    return FileResponse(p, media_type="text/plain; charset=utf-8",
                        filename=os.path.basename(p))


@app.post("/api/asr")
async def asr_upload(files: list[UploadFile] = File(...)):
    """上传音频/视频 → 语音转文字，返回识别文本（含 txt 文档下载链接）。"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    up_dir = os.path.join(INPUT_ROOT, ts)
    os.makedirs(up_dir, exist_ok=True)
    ts_dir = os.path.join(ASR_OUT_ROOT, ts)
    results = []
    with _asr_lock:
        try:
            for f in files:
                src = os.path.join(up_dir, os.path.basename(f.filename or "file.wav"))
                with open(src, "wb") as fp:
                    shutil.copyfileobj(f.file, fp)
                info = transcribe_file(src)
                info["filename"] = os.path.basename(src)
                info["txt_url"] = _asr_save_txt(ts_dir, os.path.splitext(os.path.basename(src))[0],
                                                info["text"])
                results.append(info)
        except Exception as exc:  # noqa: BLE001
            logger.exception("语音转文字失败")
            return JSONResponse({"detail": "语音转文字失败：%s" % exc}, status_code=500)
    return {"task": ts, "results": results}


@app.post("/api/asr_path")
def asr_path(req: dict):
    """服务器上的文件路径 → 语音转文字。"""
    p = req.get("path", "").strip()
    if not os.path.isfile(p):
        return JSONResponse({"detail": "文件不存在：%s" % p}, status_code=400)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ts_dir = os.path.join(ASR_OUT_ROOT, ts)
    try:
        with _asr_lock:
            info = transcribe_file(p)
    except Exception as exc:  # noqa: BLE001
        logger.exception("语音转文字失败")
        return JSONResponse({"detail": "语音转文字失败：%s" % exc}, status_code=500)
    info["filename"] = os.path.basename(p)
    info["txt_url"] = _asr_save_txt(ts_dir, os.path.splitext(os.path.basename(p))[0], info["text"])
    return {"task": ts, "results": [info]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)
