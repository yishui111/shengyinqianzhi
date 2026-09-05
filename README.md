<div align="center">

# 🎙️ 声音素材前置处理

> ⭐ **喜欢这个项目？请先点个 Star ⭐ 支持一下，让更多人看到！**

![GitHub stars](https://img.shields.io/github/stars/yishui111/shengyinqianzhi.svg?style=flat-square&color=orange)
![GitHub forks](https://img.shields.io/github/forks/yishui111/shengyinqianzhi.svg?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/yishui111/shengyinqianzhi.svg?style=flat-square)

**把音频/视频素材一键加工成适合 AI 训练（RVC / GPT-SoVITS 等）的干净人声 —— 网页操作、本地运行、全程不出本机。**

</div>

---

## ✨ 项目简介

声音克隆 / 歌声合成训练之前，最费时间的是**素材清洗**：网上下载的素材带背景音乐、带混响、有多人说话、有掌声欢呼，直接拿去训练会毁掉音色。

本项目是一个**自托管的声音素材前置处理服务**（FastAPI 单文件实现，自带网页界面），把各种音频（wav/mp3/m4a/flac/aac/ogg）和视频（mp4/mkv/mov/avi/flv）处理成适合训练模型的干净人声：

- 从视频中自动提取音轨（ffmpeg）
- 可选**人声分离**去背景音乐/伴奏（pymss `bs_roformer_voc_hyperacev2`）
- 可选**去混响**（`dereverb_mel_band_roformer`）
- 可选**提高人声音量**、**响度归一化**（页面核心选项）
- 进阶选项**只保留主要说话人**、**去掌声/欢呼/音效**、**生成训练切片**：后端仍支持（opts 里 `speaker`/`denoise`/`clean`），页面已按"突出核心"精简不展示（切片因训练中心自会切完整语音而移除）
- 附带**语音转文字**（ASR，SenseVoice，支持中文/方言）与**合成一个音频**（把一批结果 wav 无损拼接）

**设计原则：本地运行、隐私安全** —— 服务只绑本机 `127.0.0.1`（局域网访问不到，素材不出这台电脑），素材只在你自己电脑上处理，不上传任何服务器；**结果不保留**：上传原件、处理结果、合并音频、转写 txt 会在服务启动时和每个新任务开始时自动清空，要留着用的请及时从页面下载；一次只跑一个任务（自动排队），长音频自动按 90 秒分段防止显存溢出，无独立显卡时自动走 CPU。

## 🎯 主要功能

- 🎬 **音视频统一入料**：任意音频/视频 → 44.1k（可选 32k/16k）单声道 wav，纯转码模式什么都不用勾
- 🎤 **人声分离去 BGM**：pymss 分离出纯净人声，长音频自动分段拼接
- 🏠 **去混响**：消除房间回声，让干声更干净
- 🗣️ **只保留主要说话人**：ECAPA 声纹聚类，视频里谁说话最久就留谁（进阶选项，页面不展示，API opts `speaker`）
- 🔇 **去掌声/欢呼/音效**：基频检测剪掉连续 0.7 秒以上的无基频片段（进阶选项，页面不展示，API opts `denoise`；带 BGM 素材须配合人声分离）
- 🔊 **提高人声音量**：1~20 倍增益（防削波），适合人声偏小/远距离录音
- ✂️ **训练切片**：可选把成品切成 1.5~20 秒片段并响度归一化（-16 LUFS）（进阶选项，页面不展示，API opts `clean`；训练中心自会切完整语音）
- 🧩 **合成一个音频**：把单次任务的输出 wav 按文件名顺序无损拼成一个文件
- 📝 **语音转文字**：音频/视频识别成中文文本（方言友好），可复制/下载 txt
- 🖥️ **网页操作**：http://127.0.0.1:8070/，实时日志、结果列表直接**页面试听**与下载

## 🗂️ 目录结构

```
shengyinqianzhi/
├── pre_service/
│   └── preprocess_api.py   # 主服务（FastAPI 单文件，含网页 UI）
├── tests/                  # 自研测试/工具脚本（ASR 模型下载、ASR 冒烟测试）
├── requirements.txt        # 全新环境手动安装依赖参考（见 DEPLOY.md）
├── start.bat               # 一键启动（自动打开浏览器）
├── stop.bat                # 一键停止（按端口精确杀进程）
├── 一键启动素材前置处理.bat  # 原版启动脚本（保留）
├── 关闭素材前置处理.bat      # 原版关闭脚本（保留）
├── 部署方案.md              # 作者原始部署笔记（含自测记录）
├── AGENTS.md               # 项目约定（随项目分发，供 AI/协作者阅读）
├── .gitignore
└── README.md               # 本文件
```

> 💡 本仓库只包含**源代码 / 脚本 / 配置 / 文档**等关键内容。
> 便携 Python 运行时、ffmpeg、模型权重、素材产物等**大文件不随仓库分发**，见下方「大件资源下载」与 [DEPLOY.md](DEPLOY.md)。

## 🚀 快速开始（拉到新电脑即可部署）

### 环境要求

- 操作系统：Windows 10/11（作者环境）；Linux/macOS 亦可自行启动
- 硬件：NVIDIA 显卡建议显存 ≥6GB（人声分离更流畅）；无 GPU 自动走 CPU（较慢但可用）
- 运行时：Python 3.12（**推荐直接使用便携运行时 `runtime\py312`**，见 DEPLOY.md 方案 A）

> 本项目原作者以「整个项目文件夹自包含」的方式分发（运行时 + ffmpeg + 模型全部放进项目内，拷走即用）。
> 拉到新电脑部署的**两条路径**（复制整套运行时 / 全新手动安装）详见 [DEPLOY.md](DEPLOY.md)。

### 1. 克隆

```bash
git clone https://github.com/yishui111/shengyinqianzhi.git
cd shengyinqianzhi
```

### 2. 安装依赖 / 准备大件资源

- **方案 A（推荐，从作者处拿大件）**：复制整套便携运行时 `runtime\py312`（全部依赖已锁定），放到项目根目录即可，无需 `pip install`；
- **方案 B（全新环境）**：安装 Python 3.12 后 `pip install -r requirements.txt`（含 torch 等大依赖），再准备 `ffmpeg` 与 `models`。

详细步骤见 [DEPLOY.md](DEPLOY.md)；大件清单见下方「大件资源下载」表格。

### 3. 配置（可选）

无需配置文件即可运行。可选环境变量：`PRE_PORT`（端口，默认 8070）、`SEPARATE_MAX_SEC`（AI 处理分段秒数，默认 90），完整列表见 [DEPLOY.md](DEPLOY.md)。

### 4. 启动

```bash
# Windows：双击 start.bat（或一键启动素材前置处理.bat），脚本自动打开浏览器
start.bat
```

- `start.bat` **后台启动，不弹黑框窗口**：服务以 pythonw 无窗口运行，关掉任何窗口都不影响；
  运行日志写入 `ziliao\logs\service.log`（自动轮转）+ 网页「处理日志」区；停止用 `stop.bat`。
- 想看前台控制台日志可双击原版「一键启动素材前置处理.bat」（关掉该窗口 = 关闭服务）。

### 5. 验证

- 打开浏览器访问 http://127.0.0.1:8070/ ，看到网页界面即部署成功
- 自检接口：`GET /api/health` 返回 `{"status":"ok",...}`
- 冒烟测试：上传一段带 BGM 的音频 / 一段视频 → 勾选「人声分离」处理 → 输出应只剩干净人声

## 📥 大件资源下载（模型 / 运行时 / ffmpeg）

| 资源 | 用途 | 下载地址 / 获取方式 |
| ---- | ---- | ---- |
| `runtime\py312`（约 9.4GB） | 便携 Python 3.12，含 torch/pymss/speechbrain/funasr 等全部依赖 | 作者机器上整体复制（推荐，版本全部锁定）；或全新安装：Python 3.12 + `pip install -r requirements.txt` |
| `ffmpeg\`（约 0.2GB） | 音视频解码/转码/提取音轨（ffmpeg.exe + ffprobe.exe） | https://www.gyan.dev/ffmpeg/builds/ 下载 release-essentials 版，解压为 `ffmpeg\bin\ffmpeg.exe` / `ffprobe.exe` |
| `models\pymss`（约 1.1GB） | 人声分离 + 去混响模型（`bs_roformer_voc_hyperacev2`、dereverb） | 用 pymss 自带下载器下载到 `models\pymss`（huggingface / hf-mirror / modelscope 源，见 DEPLOY.md） |
| `models\speaker` + `models\hf_cache`（约 0.2GB） | ECAPA 说话人识别（「只保留主要说话人」） | 首次使用自动从 HuggingFace 下载（speechbrain/spkrec-ecapa-voxceleb，约 50MB），需联网 |
| `models\asr`（约 1GB） | 语音转文字（SenseVoiceSmall + FSMN VAD） | 运行 `tests\download_asr_model.py` 从 ModelScope 下载（一次性，之后离线可用） |

> ⚠️ 运行大件全部就绪前，服务可启动但对应功能会报错/跳过，属正常现象。

## 🛠️ 本地开发 & 提交

```bash
git add .
git commit -m "feat: xxx"
git push origin main
```

主服务是**单文件**实现（`pre_service\preprocess_api.py`，含后端 API 与网页 UI 的 HTML），
改完代码直接重启即可，无构建步骤。

## ❓ 常见问题（FAQ）

- **Q：双击 start.bat 提示 python.exe not found？** A：还没准备 `runtime\py312` 便携运行时，请按 [DEPLOY.md](DEPLOY.md) 方案 A 复制整套运行时，或方案 B 装系统 Python 后改用 `python pre_service\preprocess_api.py` 启动。
- **Q：处理很慢？** A：无 GPU 时自动走 CPU（人声分离明显变慢）；建议 NVIDIA 显存 ≥6GB。
- **Q：干净录音处理后人声音高好像变了？** A：人声分离引擎会轻微改变音高（引擎特性）。干净独白素材请**不要勾**「人声分离」，只做纯转码/切片即可。
- **Q：为什么一次只能跑一个任务？** A：任务单队列设计，防止多个任务并发互相干扰模型状态；后台任务会自动排队。
- **Q：长音频会不会爆显存？** A：>90 秒音频自动分段分离再拼接，已内置防爆显存机制。
- **Q：勾了「去掌声/欢呼/音效」没反应？** A：该功能只剪连续 0.7 秒以上的无基频片段（静音/掌声/欢呼）；素材自带背景音乐时音乐有基频、剪不动，请**同时勾「人声分离」**；与人声重叠的掌声也剪不到。
- **Q：训练切片选项怎么没了？** A：2026-09-05 页面精简时移除——训练中心接收完整语音并自行切片，前置阶段切片多余；API 提交 opts 带 `"clean": true` 仍可使用该能力。
- **Q：批量处理里有一个坏文件会整批失败吗？** A：不会，坏文件会被跳过并继续处理剩余文件，结束后页面顶部提示"⚠ 完成，但 N 个文件失败：xxx"。
- **Q：一批里有同名素材（比如不同文件夹都叫 第1集.mp4）会互相覆盖吗？** A：不会，上传和输出都会自动加序号（如 `第1集_2_clean.wav`）。
- **Q：视频里想保留的不是说话最久的人？** A：「只保留主要说话人」按说话时间最长者保留；想留别人请先剪辑素材再处理。

## ⚠️ 注意事项

- 本仓库**不含**任何模型权重、便携运行时、素材文件；素材与测试音频属个人/隐私数据，一律不入库（`models/`、`runtime/`、`ffmpeg/`、`ziliao/` 已在 `.gitignore` 中忽略）；
- 服务默认监听 `0.0.0.0:8070`，仅供本机/局域网使用；请勿暴露到公网；
- 端口可用环境变量 `PRE_PORT` 修改（默认 8070）；
- 本仓库仅供学习交流使用。

## 📄 许可证

MIT License（如项目自带 LICENSE 则以仓库内为准）

## 🙏 支持与致谢

如果这个项目帮到了你，**请点亮右上角的 ⭐ Star**，你的支持是我持续更新的最大动力！
