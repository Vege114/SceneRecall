# SceneRecall 0.4

**角色标注可以只做你要识别的人物。** 在「角色档案 → 开始角色确认」或等待中的任务打开「选择识别角色」，按姓名或别名查找并勾选目标。保存后只需标注所选角色，也可以清空名单后继续分析。同一角色库的剧集共用选择，既有档案和已保存的标注可继续查看。[操作说明](docs/CHARACTER_CONFIRMATION_QUEUE.md)

**搜索结果会自动显示对应画面。** 视觉命中按原片的精确帧时间重建，台词和声音结果按播放位置生成预览。页面出现或滚动到结果时就会加载。普通采样 JPEG 在分析成功、具备可靠重建条件后从档案中释放，原片指纹、帧定位、校验值和检索索引保留。收藏和人工确认的画面另存独立证据。已有资料库的迁移方式和实测见[采样画面存储](docs/SAMPLED_FRAMES.md)。

单集详情可以先打开播放器并定位到搜索命中点，画面证据、字幕时间线与运行参数按所选页签读取。搜索回到原始资料核对时，只恢复候选及其依赖。YumeMita 第 1–4 集的浏览器对照、测量条件与磁盘盘点见 [YumeMita 性能与空间验证](docs/YUMEMITA_PERFORMANCE_20261111.md)。

已有资料库可先用离线工具估算普通人物裁剪的占用，或核验大型人物身份 JSON 能节省的字节。逐文件核验通过后可以在停服期间显式整理。命令、适用范围和副本实测见 [资料库空间整理](docs/ARCHIVE_STORAGE.md)。

人物裁剪 PNG 的保留可独立配置。开发阶段默认保留，设置 `SCENERECALL_STAGE=production` 时默认不保留。识别记录、特征与可信参考图继续保存，缺少普通裁剪时可使用已登记的采样帧和人物框重建。[PNG 保留策略](docs/PNG_RETENTION.md)

检索结果可展开「搜索总耗时与阶段记录」，查看从点击搜索按钮（键盘操作从提交开始）到结果可用的前端计时、后端阶段和实际缓存状态，并下载不含查询原文的记录。测量边界见 [搜索耗时记录](docs/SEARCH_TIMING.md)。

搜索页默认显示匹配分 **严格高于 50%** 的全部合格结果，按分数从高到低排列；阈值可调整，视觉矩阵缓存预算保持 **1 GiB**。有明确人物目标时，词法、语义与概念贡献受命中帧的人物证据约束，窗口里其他帧出现姓名不能给当前帧加满分。分数不是正确概率，小人物漏检和空间关系识别仍有不足。阈值曲线、实测与后续优化方向见 [本轮检索记录](docs/SEARCH_SESSION_20261104.md)。

主搜索框支持按当前作品的角色名录解析别名和部分姓名，例如“华恋”可匹配“爱城华恋”，“华恋与光站在东京塔上”可同时解析两位角色。单字简称需要明确的别名或并列角色语境；有歧义时保留原文。台词、画面及默认混合检索都使用统一的规范名称进行各路召回、排序、定位和复核，使简称与全名具有相同的检索含义。字词检索还使用该角色已登记且无歧义的别名，因此与全名不重字的昵称也能找到原台词。台词支持包含匹配、中文分词及部分词语召回，结果仍显示原始对白；这不等同于任意错别字纠正。解析在本地完成，不增加视觉模型查询次数，也不把人物候选或组合动作自动标成已确认。修复原因、完整结果一致性与真实前端耗时见 [搜索验证记录](docs/SEARCH_QUERY_EQUIVALENCE_20261103.md)。

本地动画资料库：整理视频、字幕与角色证据，用自然语言找回画面和片段。采用 Python / FastAPI 与 React / TypeScript，支持本地模型和可选的外部 AI 服务。

## 近期功能入口（2026-11-03）

人物处理、参考资料和作品知识按以下八组提供。它们保留人工确认与来源校验：候选相似度、角色设定或姓名展开均不能直接证明当前画面中的人物身份与事件。

| 功能 | 用户可见变化 | 使用说明 |
| --- | --- | --- |
| 大角色表的场景分析 | 启用本地人物匹配时，先按当前画面候选选择相关档案，再约束发送给模型的资料体积；名单尾部角色仍可入选 | [场景角色名单](docs/SCENE_LOCAL_ROSTER.md) |
| 离线权重预检 | 人物检测／CCIP 缺少本地权重时，在加载推理运行库前列出所缺文件和缓存位置，不隐式下载 | [视觉模型预检](docs/VISUAL_MODEL_PREFLIGHT.md) |
| 角色确认保存队列 | 跨角色暂存勾选，按顺序保存；逐项显示结果，保留失败项供重试，并提示冲突及未保存选择 | [角色确认队列](docs/CHARACTER_CONFIRMATION_QUEUE.md) |
| Bangumi 参考图分批缓存 | 主角参考优先，分批继续缓存；复用完整本地图、补齐损坏文件，显示剩余数量和失败原因 | [参考图缓存](docs/BANGUMI_REFERENCE_CACHE.md) |
| 人物裁剪 PNG 保留 | 在视听身份配置中选择跟随部署、保留或不保留；缺少普通裁剪时从持久证据按需重建预览 | [PNG 保留策略](docs/PNG_RETENTION.md) |
| 作品知识资料卡与 LightRAG | 导入有出处的文字，核对概念、角色和关系资料卡；可显式建库并选择基线／LightRAG 检索 | [知识库说明](docs/LIGHTRAG.md) |
| 已保存 Bangumi 档案转知识 | 复用本地档案准备来源和待核对角色资料卡；重复整理保留人工决定，不额外联网或调用模型 | [Bangumi 知识资料](docs/BANGUMI_KNOWLEDGE.md) |
| 参考图寻找影片候选 | 用已确认概念参考图与已有影片特征比较，查看原帧并填写依据后，才能确认出场 | [参考图候选发现](docs/REFERENCE_DISCOVERY.md) |

LightRAG 仍是实验选项，默认使用基线；[既有小样本实验](docs/LIGHTRAG_EXPERIMENT_20261028.md)尚未证明留出集净提升。本轮工程回归不替代检索质量验收，[PNG 四集验证](docs/PNG_VALIDATION_20261031.md)也仍未完成。分组测试与完整套件状态见[验证记录](docs/VALIDATION.md#八组功能分拆与累计回归2026-11-03)。

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

已完成单集的详情直接读取保存结果，并核对当前人物关联与人工修订。后台分析的本地读取和任务记账在线程中执行，避免等待资料库锁时冻结界面请求；单集详情超过 30 秒会显示超时与重试入口。详情加载耗时与实际 Chrome 验证见 [验证记录](docs/VALIDATION.md#已完成单集的详情读取2026-10-23)。

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
| 作品知识（可选） | 独立 Python 3.12、LightRAG 1.5.7、FastEmbed 0.9.1 | [知识库接入与试验说明](docs/LIGHTRAG.md)；模型与设置 → 作品知识库 |

imgutils 的 NumPy 1.x 与 pyannote 4 的 NumPy 2.x 要求冲突，**不要将 `visual-identity` 与 `audio` 装入同一环境**。图文编码配置需固定模型 revision 和本地路径；Windows requirements 中的 CUDA 12.8 版本是已验证配置，其他平台需使用相应构建。

作品设定库支持有出处的 Markdown／纯文本资料卡、内部概念及角色关联、基线／LightRAG 对照和确认片段。默认本地 CPU 多语言 embedding，文本抽取可用已有 Codex 或兼容 API 连接；知识检索和参考图候选发现均可复用 core 画面索引。安装、核对、索引重建和评估命令见 [LightRAG 文档](docs/LIGHTRAG.md)。

## 开源与许可

SceneRecall 使用 [GNU AGPL v3](LICENSE)。第三方程序、模型权重及外部资料分别遵循原许可，不因本项目的许可证而重新授权。

特别是 CCIP 模型卡仅标注 `OpenRAIL`，具体条款仍需向上游确认；community-1 权重使用 CC BY 4.0；Animated_AD 的可选外部适配未发现明确许可证，不捆绑其实现或权重。项目来源、署名、许可证原文及分发注意事项见 [第三方声明](THIRD_PARTY_NOTICES.md)。

查询缓存、并发召回、实体出场索引及知识库接口见 [查询优化与实测](docs/QUERY_OPTIMIZATION_20261028.md)。该文记录同一批查询的结果等价验证、速度和内存边界，以及可复现基准命令。

环境与能力细节见 [docs](docs/)，开发进度见 [PLAN.md](PLAN.md)，已完成与尚待完成的验证见 [VALIDATION.md](docs/VALIDATION.md)。
