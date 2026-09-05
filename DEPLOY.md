
## 🚀 换电脑部署（保证可用）

> **方式 A（推荐 · 100% 保证）**：用 U 盘 / 网盘把「原项目整份文件夹」（含全部大件）复制到新电脑 → 双击 `start.bat` 即可。
>
> **方式 B（代码装配）**：`git clone` 本仓库 → 双击 `assemble.bat` 预检大件 → 按提示补齐缺失项（下载地址见下文/README）→ 双击 `start.bat`。

> 说明：引擎、模型、镜像、运行时等大件体积超过 GitHub 单文件 100MB 上限，**不随仓库分发**；本仓库承载全部自研代码与装配指引，"方式 A"是换机部署最稳路径，"方式 B"适合需要重新下载大件的场景。
# 部署方案（DEPLOY.md）— 声音素材前置处理（shengyinqianzhi）

> 在**另一台电脑**上部署出一模一样的服务：把音频/视频素材加工成训练用干净人声（端口 8070）。
> 作者原始部署笔记与每次功能的自测记录见同目录 `部署方案.md`（其中路径为本机当时的布局，仅作背景参考）。

## 0. 部署方式总览：两条路径，任选其一

| 路径 | 适用人群 | 说明 |
| ---- | ---- | ---- |
| **方案 A：复制整套便携运行时** | 从作者机器拿到大件 | 作者以「整个项目文件夹自包含」方式使用：`runtime\py312` + `ffmpeg` + `models` 全在项目内，把整套拷到新电脑即可直接跑，**无需装 Python/联网装依赖**。本仓库不含大件，需从作者处一并复制。 |
| **方案 B：全新手动安装** | 从 GitHub 拉代码的新用户 | 系统装 Python 3.12 → `pip install -r requirements.txt` → 下载 ffmpeg → 下载模型。torch 等依赖较大，按下方步骤执行。 |

无论哪条路径，**代码本体 = 本仓库内容**（拉到项目文件夹，与上面大件目录平级）。

---

## 方案 A：复制整套便携运行时（推荐给有作者大件的场景）

1. 把作者电脑上**完整的原项目文件夹**（含 `runtime\py312`、`ffmpeg`、`models`、`ziliao`）整体复制到新电脑任意位置（整盘移动也兼容，代码按项目根相对路径定位，不写死盘符）；
2. 用本仓库内容覆盖其中的代码/文档部分（`pre_service\`、`tests\`、`README.md`、`DEPLOY.md`、`start.bat`、`stop.bat` 等）；
3. 双击 `start.bat`（或 `一键启动素材前置处理.bat`），浏览器自动打开 http://127.0.0.1:8070/ 即完成；
4. 无独立显卡的新电脑自动走 CPU（人声分离较慢，其余功能可用）。

---

## 方案 B：全新手动安装（从本仓库部署）

### B.1 环境要求

- 操作系统：Windows 10/11 64 位（作者环境；Linux/macOS 可按等价命令自行调整）
- Python：3.10+（作者用 3.12，64 位），安装时勾选 **Add python.exe to PATH**
- 显卡：NVIDIA 建议显存 ≥6GB（人声分离走 GPU 更快）；无 GPU 自动 CPU
- 网络：下载依赖与模型时需要（模型下载完成后离线可用）

### B.2 克隆并准备目录

```bash
git clone https://github.com/yishui111/shengyinqianzhi.git
cd shengyinqianzhi
# 模型/产物目录（代码会自动创建 ziliao，models 需手动准备）
mkdir models
```

### B.3 安装 Python 依赖

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows（Linux/macOS: source .venv/bin/activate）
pip install --upgrade pip

# 1) torch 建议按官方 CUDA 安装指引安装（Windows 默认带 CUDA 12.x）：
#    https://pytorch.org/get-started/locally/
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2) 其余依赖（requirements.txt 内容见仓库，含版本说明注释）：
pip install -r requirements.txt
```

> 若 `pip install -r requirements.txt` 中 torch 版本解析失败，请先按上面前置命令单独装好 torch/torchaudio。

### B.4 准备 ffmpeg

1. 下载 ffmpeg 便携版（release-essentials）：https://www.gyan.dev/ffmpeg/builds/
2. 解压，把其中的 `bin\ffmpeg.exe`、`bin\ffprobe.exe` 放到项目 `ffmpeg\bin\` 下；
   （`start.bat` 与主服务都会自动把 `ffmpeg\bin` 加入搜索路径；若不想放项目内，也可把 ffmpeg 加入系统 PATH。）

### B.5 下载模型（三处，按需）

| 模型目录 | 内容 | 获取方式 |
| ---- | ---- | ---- |
| `models\pymss` | 人声分离 `bs_roformer_voc_hyperacev2` + 去混响 dereverb 模型（约 1.1GB） | 用 pymss 自带下载器，例如：<br>`python -c "from pymss.model_download import download_model; download_model('bs_roformer_voc_hyperacev2', model_dir='models/pymss', source='hf-mirror')"`<br>去混响模型同理（模型名以 pymss 为准，可用 `python -m pymss list` 查看注册表）；也可把作者机器 `models\pymss` 整个复制过来。 |
| `models\speaker` + `models\hf_cache` | ECAPA 说话人识别（约 0.2GB） | **无需手动下载**：首次勾选「只保留主要说话人」时自动从 HuggingFace 拉取（约 50MB），存到项目内，之后离线可用。 |
| `models\asr` | 语音转文字 SenseVoiceSmall + FSMN VAD（约 1GB） | 运行：`python tests\download_asr_model.py`（从 ModelScope 下载，一次性，之后离线可用）。 |

**部署自检（不勾任何处理选项 = 纯转码冒烟）**：

```bash
# 用一段你自己的短音频/视频（或先生成一个 3 秒正弦波）：
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3" -ar 44100 -ac 1 smoke_src.wav
```

启动后在网页上传该文件 → 什么都不勾 → 开始处理 → 结果应得到一个同长 wav，即链路正常。

### B.6 启动 / 停止

```bash
start.bat        # Windows：一键启动 + 自动打开浏览器 http://127.0.0.1:8070/
stop.bat         # 一键停止（按端口 8070 精确杀进程，不会误杀其他项目）
```

- 原版启动脚本同样可用：`一键启动素材前置处理.bat`（前台运行，日志在同一窗口）/ `关闭素材前置处理.bat`
- 手动启动：`python pre_service\preprocess_api.py`
- 健康检查：浏览器或命令行访问 `GET http://127.0.0.1:8070/api/health`

### B.7 使用流程与自检

1. 网页选文件（可多选音频/视频）上传，或填服务器文件夹路径批量读取；
2. 处理选项**默认全部不勾选**：什么都不勾 = 只把视频/音频原样转成 44.1k 单声道 wav（纯转码，适合方言录音/干净素材）。页面核心选项：勾「人声分离」去 BGM、「去混响」去房间回声、「提高人声音量」放大音量、「响度归一化」按需勾选；
   进阶选项「只保留主要说话人」「去掌声/欢呼/音效」「额外生成训练切片」已不在页面展示（2026-09-05 页面精简；切片功能因训练中心自会处理完整语音而移除），后端仍支持：提交 opts 带 `"speaker": true` / `"denoise": true` / `"clean": true` 即可；
3. 点「开始处理」，日志实时显示；完成后结果列表**可直接页面试听、单个 wav 直接下载**；
4. 冒烟验证：一段带 BGM 的音频 + 一段视频 → 勾「人声分离」处理 → 确认输出只剩人声（无伴奏）。

---

## 端口 / 数据目录 / 日志 / 配置说明

| 项 | 说明 |
| ---- | ---- |
| 默认端口 | `8070`（Web UI 与全部 API 同一端口） |
| 数据目录 | `ziliao\input\`（上传的原始素材，按任务时间分目录）、`ziliao\output\<任务时间>\`（结果 wav）、`ziliao\output\merged\`（「合成一个音频」结果）、`ziliao\asr\<时间戳>\`（语音转文字 txt 文档）。**均不持久保留**：服务启动时与每个新任务开始时自动清空，要留的东西请从页面及时下载 |
| 日志 | 无独立日志文件：运行日志实时显示在服务控制台 + 网页「处理日志」区（保留最近 600 行） |
| 配置文件 | 无配置文件；全部通过**环境变量**定制（见下），未设置时用默认值 |
| 模型目录 | `models\pymss`、`models\speaker`、`models\hf_cache`、`models\asr`（见 B.5） |

### 可用环境变量（全部可选）

| 变量 | 默认 | 说明 |
| ---- | ---- | ---- |
| `PRE_PORT` | `8070` | 服务端口（start.bat / stop.bat 与主服务都读取） |
| `PRE_HOST` | `127.0.0.1` | 服务绑定地址（默认仅本机可访问；如需局域网其他设备访问再改为 `0.0.0.0`） |
| `SEPARATE_MAX_SEC` | `90` | 人声分离/去混响自动分段秒数（防爆显存） |
| `HF_HOME` | `models\hf_cache` | HuggingFace 缓存目录 |
| `HUANSHENG_ROOT` | 同盘根目录的 `tihuanshengyin` | 旧版「复用换声项目运行时/模型」的兼容回退路径（自包含模式下不会用到） |

## 本机与目标机器可能不同的项

- **路径**：本项目代码全部用项目根相对路径（`%~dp0` / 脚本所在目录推导），不写死盘符，整个文件夹可放任意位置；`部署方案.md` 中记录的 `E:\shengyinqianzhi`、`E:\tihuanshengyin`、`E:\xunlianzhongxin` 是作者本机当时的布局，**新机器无需保持一致**；
- **端口**：若 8070 被占用，启动前设 `PRE_PORT`（如 `set PRE_PORT=8080`）再运行 start.bat；
- **显卡**：有无 GPU 自动切换，无需配置；
- **依赖安装方式**：方案 A（复制便携运行时）与方案 B（系统 Python + pip）二选一，不要混用同一 `runtime` 目录。

## 常见问题排查（FAQ）

- **Q：双击 start.bat 一闪而过或提示 python.exe not found？** A：确认已准备 `runtime\py312\python.exe`（方案 A）或改用方案 B 手动安装后用 `python pre_service\preprocess_api.py` 启动。
- **Q：`pip install -r requirements.txt` 报 torch 相关错误？** A：按 B.3 先单独安装官方 torch/torchaudio（对应你的 CUDA），再装其余依赖；pymss 要求 torch>=2.7.1。
- **Q：启动后页面能开，但处理报「人声分离不可用/模型不存在」？** A：`models\pymss` 未就绪，见 B.5 第 1 项。
- **Q：「语音转文字」报模型加载失败？** A：先运行 `tests\download_asr_model.py` 下载模型（需联网 + 已装 modelscope）。
- **Q：勾「只保留主要说话人」第一次很慢？** A：首次使用会自动下载 ECAPA 模型（约 50MB，需联网），之后秒级加载。
- **Q：处理很慢 / 显存不足？** A：无 GPU 自动 CPU（慢）；有 GPU 但显存 <6GB 也会自动 CPU；长音频已按 90 秒自动分段，一般不会爆显存。
- **Q：勾「去掌声/欢呼/音效」没反应？** A：该功能只剪连续 0.7 秒以上的无基频片段（静音/掌声/欢呼）；素材自带背景音乐时音乐有基频、剪不动，请**同时勾「人声分离」**；与人声重叠的掌声也剪不到（属预期）。
- **Q：勾「额外生成训练切片」不出切片？** A：2026-09-05 前版本有切分 bug（真实语音被切碎到全部低于 1.5 秒而被过滤），已修复；仍不出说明素材里没有超过 1.5 秒的连续发声段。
- **Q：识别中文带方言口音不准？** A：SenseVoice 对常见方言友好，若结果不理想可换更清晰的源音频或先做去噪。

---

## 维护约定（给后续维护者 / AI）

- 每次功能变更后：同步更新本文件与 `部署方案.md`（活文档，含自测记录）；
- 素材/测试产物一律放项目内 `ziliao\`、`tests\`，不放项目外；
- 大件（runtime/ffmpeg/models）不提交 Git，由方案 A 整套复制或方案 B 按表下载。
