# SceneRecall 0.4

本地动画资料库：整理视频、字幕与角色证据，用自然语言找回画面和片段。采用 Python / FastAPI 与 React / TypeScript，支持本地模型和可选的外部 AI 服务。

## 基础环境

| 依赖 | 要求与来源 |
| --- | --- |
| Python / uv | Python **3.12–3.13**，推荐 3.12；用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理环境与锁定依赖 |
| Node.js / npm | [Node.js](https://nodejs.org/en/download) **20+**，已验证 Node 22；用于前端构建及 Bangumi 查询依赖 |
| FFmpeg / ffprobe | 安装 [FFmpeg](https://ffmpeg.org/download.html)，确保两者在 PATH 中；Windows 也可放到 `.cache/ffmpeg/bin` |
| GPU（可选） | 主应用不要求 GPU；模型加速需要兼容的 PyTorch / ONNX Runtime 和驱动。Windows CUDA 配置与实测范围见[模型环境说明](docs/WINDOWS_MODELS.md) |

主应用支持 Windows 与 macOS；Linux 尚未完成平台验收。模型环境需另外准备，不能直接共用 Windows 与 WSL 的虚拟环境。

## 安装与启动

以下命令在仓库根目录执行。**基础安装只准备应用及前端，不安装可选模型依赖或下载权重。**

Windows：

```powershell
.\scripts\start_windows.cmd -Setup -DataDir "D:\SceneRecallLibrary"
```

后续启动省略 `-Setup`。解释器、FFmpeg 路径与后台运行配置见 [Windows 环境说明](docs/WINDOWS.md)。

macOS / Linux：

```sh
uv sync --frozen --python 3.12 --extra dev
npm ci --ignore-scripts
npm --prefix frontend ci
npm --prefix frontend run build
uv run scenerecall --data-dir "$HOME/SceneRecallLibrary" --port 8765
```

服务默认仅监听本机，访问地址由启动终端输出。生成式画面理解可接入兼容 API 或已登录的本机 Codex CLI；这与下面的本地模型环境分别配置。

启动后可先进入界面浏览资料库，作品与任务列表独立加载；搜索索引、资料统计和本地检索模型检查在后台准备。搜索页会显示准备状态，就绪后自动启用检索；准备失败可在页面重试，无需重启。统计未完成时显示占位或已有值。队列在索引与首次统计处理结束后开始调度；索引失败时先重试索引。

开始分析、编辑默认配置，以及在任务队列完成角色确认，见 [任务使用说明](docs/TASK_WORKFLOW.md)。

## 模型与独立环境

按所需能力选装，无需全部下载。权重从上游取得并保存在本机；运行时使用本地目录或已有缓存，缺失时不会隐式下载。

| 用途 | 模型 / 下载来源 | 准备要点 |
| --- | --- | --- |
| 动画人物检测 | [DeepGHS anime_person_detection](https://huggingface.co/deepghs/anime_person_detection) | `person_detect_v1.1_m/model.onnx`，与 CCIP 搭配 |
| 角色图像匹配 | [DeepGHS CCIP ONNX](https://huggingface.co/deepghs/ccip_onnx) | 所选预设的 `model_feat.onnx` 与 `model_metrics.onnx`；应用默认 `ccip-caformer-24-randaug-pruned`，现有实测使用 `ccip-caformer-5_fp32` |
| 直接图文检索 | [Google SigLIP 2 base-patch16-256](https://huggingface.co/google/siglip2-base-patch16-256) | 当前暂选编码器；完整 snapshot，含权重、预处理与 tokenizer 文件 |
| 图文检索对照（可选） | [LAION multilingual CLIP ViT-B/32](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k) | 另需本地 XLM-R 配置与 tokenizer；与 SigLIP 2 二选一 |
| 说话人分段（可选） | [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | 需接受上游访问条件，下载完整离线目录及其依赖 |
| 声纹匹配（可选） | [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | 完整 snapshot；按作品、语言与配音版本使用 |
| 语音转写（可选） | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 使用本地 CTranslate2 格式 Whisper 权重；项目未指定必装的模型尺寸 |
| 说话活动检测（实验） | [LR-ASD 代码与基础权重](https://github.com/Junhua-Liao/LR-ASD) | 应用需要经过训练工具导出的模型包；本机动画适配包尚未公开分发，准备方式见 [LR-ASD 文档](docs/LRASD_TRAINING.md) |

依赖分组如下；Windows 的锁定环境与具体安装命令集中在 [WINDOWS_MODELS.md](docs/WINDOWS_MODELS.md)，其他平台的视觉／音频安装见 [CHARACTER_AV.md](docs/CHARACTER_AV.md#安装与运行)。

| 环境 | 关键依赖 | 配置入口 |
| --- | --- | --- |
| 主应用 | 推荐 Python 3.12；可选 `audio` / `asr` extras 安装在该环境 | `pyproject.toml` / `uv.lock` |
| 人物检测 / CCIP | 独立 Python 3.12、imgutils 0.19、NumPy 1.26、ONNX Runtime | [视觉依赖](scripts/requirements-visual-windows.txt)；`SCENERECALL_VISUAL_PYTHON` |
| 图文编码 | 独立 Python 3.12、PyTorch、Transformers / OpenCLIP | [检索依赖](scripts/requirements-retrieval-windows.txt)；资料库 `private/retrieval-encoder.json` |
| LR-ASD | 独立 Python 3.10、NumPy 1.23、PyTorch | [LR-ASD 依赖](scripts/requirements-lrasd-windows.txt)；视听身份设置中的解释器、源码与模型包目录 |

imgutils 的 NumPy 1.x 与 pyannote 4 的 NumPy 2.x 要求冲突，**不要将 `visual-identity` 与 `audio` 装入同一环境**。图文编码配置需固定模型 revision 和本地路径；Windows requirements 中的 CUDA 12.8 版本是已验证配置，其他平台需使用相应构建。

## 开源与许可

SceneRecall 使用 [GNU AGPL v3](LICENSE)。第三方程序、模型权重及外部资料分别遵循原许可，不因本项目的许可证而重新授权。

特别是 CCIP 模型卡仅标注 `OpenRAIL`，具体条款仍需向上游确认；community-1 权重使用 CC BY 4.0；Animated_AD 的可选外部适配未发现明确许可证，不捆绑其实现或权重。项目来源、署名、许可证原文及分发注意事项见 [第三方声明](THIRD_PARTY_NOTICES.md)。

环境与能力细节见 [docs](docs/)，开发进度见 [PLAN.md](PLAN.md)，已完成与尚待完成的验证见 [VALIDATION.md](docs/VALIDATION.md)。
