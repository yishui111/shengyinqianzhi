# 项目约定（shengyinqianzhi 声音素材前置处理）

> 本文件随项目分发：项目文件夹复制到其他电脑 / 拉取本仓库后，agent 与协作者都遵守以下约定。
> 部署步骤以 `DEPLOY.md` 为准，作者原始部署笔记见 `部署方案.md`。

## 项目概述

独立的声音素材前置处理服务（端口 8070）：把音频/视频素材处理成训练用的干净人声。
- 主服务：`pre_service\preprocess_api.py`（FastAPI 单文件，含网页 UI，端口 8070）
- 复用/内置：便携 Python 运行时（`runtime\py312`）、ffmpeg（`ffmpeg\bin`）、模型（`models\`）
  —— 大件目录**不随仓库分发**，部署方式见 `DEPLOY.md`（方案 A 整套复制 / 方案 B 手动安装）

## 关键结构

- `pre_service\preprocess_api.py`：主服务（FastAPI，8070；单文件含后端 + 网页 HTML）
- `tests\`：测试/工具脚本（`download_asr_model.py` 下载 ASR 模型、`test_asr.py` ASR 冒烟）
- `ziliao\input\`：上传的原始素材；`ziliao\output\<任务时间>\`：处理结果；`ziliao\asr\`：语音转文字 txt
- `一键启动素材前置处理.bat` / `关闭素材前置处理.bat`：原版启停脚本；`start.bat` / `stop.bat`：标准入口
- `DEPLOY.md`：新机器部署步骤；`部署方案.md`：作者原始笔记（含功能变更自测记录）

## 文件归属

- 素材/测试产物一律放项目内（`ziliao`、`tests`），不放项目目录之外。
- `runtime\`、`ffmpeg\`、`models\` 为大件，不入 Git；修改代码时不得改动其中内容。

## 工作方式（命令优先）

- 启动：双击 `start.bat`（或 `一键启动素材前置处理.bat`），浏览器访问 http://127.0.0.1:8070/
- 停止：双击 `stop.bat`（或 `关闭素材前置处理.bat`，按端口 8070 杀进程）
- 自检：`GET /api/health`；用一段带 BGM 的音频和一段视频冒烟测试，确认输出只有人声
- 部署/实现任务：直接执行到完成、测试通过再一次性汇报，不逐步请示；换整体架构/方案才停下确认

## 本项目关键约定（非显而易见）

- 一次只跑一个任务（并发会干扰模型状态），任务自动排队。
- 人声分离引擎会轻微改变干净录音的音高：干净素材建议只做「去静音切片」，不开「人声分离」。
- 长音频 >90 秒自动分段处理，防止显存溢出。
- 输出采样率默认 44100；RVC / GPT-SoVITS 训练都兼容。
- 「只保留主要说话人」按说话时间最长者保留：想留的不是主角时需先剪辑素材。

## 禁止（Do NOT）

- 不把素材/测试产物放进项目目录之外。
- 不修改 `models\` / `runtime\` 等大件目录里的文件。
- 不把密钥、token、账号密码、个人素材、真人音频写进仓库或提交到 Git。

## 维护

- 活文档：重复踩坑就补规则；每次优化/修复后同步更新 `DEPLOY.md` 与 `部署方案.md`。
---
### 关键点（2026-09-02 上传整理补充）
- pre_service = 自研 FastAPI（端口 8070），tests 为自研（测试需自备音频）
- 大件不入库：runtime\py312(便携Python ~9.4GB)、ffmpeg(~0.2GB)、models\pymss(~1.1GB)/speaker/asr(~1GB)；获取方式在 DEPLOY
- 一键启动/关闭 bat 已规范 ASCII/CRLF；无 runtime 时 start.bat 报错属预期（提示就位）
- requirements.txt 为全新安装参考清单（原工程无）；torch/pymss 安装注意见文件注释
