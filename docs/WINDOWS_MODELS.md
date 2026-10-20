# Windows 原生模型环境

主应用使用 `.venv-windows`（Python 3.12）。视觉与 LR-ASD 各自运行于独立 Windows 环境；不要把 Linux 的 `.venv/bin/python` 设为原生应用的解释器。以下命令在项目根目录的 PowerShell 中执行，依赖安装需要联网，模型推理继续使用已授权的本地权重。

## 直接图文检索独立环境（2026-10-20）

本次目标机器实际报告 **NVIDIA GeForce RTX 4070 Laptop GPU、8188 MiB 显存、驱动 610.62**；torch 设备总内存为 8,585,216,000 字节。这是设备容量，运行时可用显存会受桌面及共享 GPU 工作影响，不能按标称容量假设每次都可独占。正式选择记录 `pytorch-cuda / cuda:0 / float16`；CPU 只在配置中显式选择，不自动降级。

新 `.venv-retrieval-windows` 与 `.venv-visual-windows`（imgutils／CCIP）、`.venv-lrasd-windows` 分开，避免 NumPy／框架依赖冲突：

```powershell
uv venv --python 3.12 .venv-retrieval-windows
uv pip install --python .venv-retrieval-windows\Scripts\python.exe --torch-backend cu128 -r scripts\requirements-retrieval-windows.txt
uv pip check --python .venv-retrieval-windows\Scripts\python.exe
.\.venv-retrieval-windows\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

`requirements-retrieval-windows.txt` 保存本次独立环境的完整依赖 freeze，包括 torch 2.8.0+cu128、torchvision 0.23.0+cu128、Transformers 4.57.1、OpenCLIP 3.2.0、NumPy 2.5.4。安装成功只证明依赖可用，不证明动画质量或吞吐。首次安装／权重下载单独计时；正常 worker／模型加载仍须计入新任务处理墙钟。

已实跑的 SigLIP 2 snapshot 为 `google/siglip2-base-patch16-256@3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`。模型文件位于项目忽略目录 `.cache/retrieval-models`，不是正式库迁移。下面是本机实验配置格式；可保存为隔离实验的 `encoder.json`，不需要改正式库：

```json
{
  "encoder_id": "siglip2-base-p16-256",
  "model_dir": "D:/SceneRecall/.cache/retrieval-models/siglip2-base-patch16-256/3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab",
  "revision": "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab",
  "python": "D:/SceneRecall/.venv-retrieval-windows/Scripts/python.exe",
  "device": "cuda",
  "batch_size": 16,
  "timeout_s": 180
}
```

`scripts/benchmark_retrieval.py` 将该配置复制到自己创建的隔离 runtime；实际应用读取所选资料库的 `private/retrieval-encoder.json`。缺配置、权重或 CUDA 时返回不可用，不隐式下载／安装。`python` 可省略并由 `SCENERECALL_RETRIEVAL_PYTHON` 显式指定。`revision` 必须固定 SHA，不能写 `main`；实现还校验实际权重、预处理、tokenizer 和依赖身份。

第二个且唯一的对照为 `openclip-multilingual-b32`，对应 `laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k`，实跑 snapshot 为 `506d40eb551f4801a1c27fc20a31c7b8f590deda`；配置还需要本地 `text_model_dir`（XLM-R 配置与 tokenizer）。已完成一次同片 90 秒 core smoke；两种模型开发召回打平，暂保留分配显存更低的 SigLIP 2，不宣称最终选型通过。两者官方来源／许可及完整实验命令见 [RETRIEVAL_FIRST.md](RETRIEVAL_FIRST.md)，追加实测和加载方差限制见 [VALIDATION.md](VALIDATION.md#检索优先实验与工程验证2026-10-20)。

常驻 worker 的 `inference_ms / item_count / effective_batch_size` 是最近一次请求的元数据，不是整段累计值；查询后顶层 metadata 还可能对应文本编码。整段采用外层墙钟、任务阶段和索引 `frames / encoded_frames / batches / cache_hits` 计数。PyTorch `peak_allocated_bytes` 也不是全系统显存占用。当前真实记录仅证明一次 90 秒 core 可运行；不代表完整模式或整集实时验收。

## 视觉与逐帧人物检测

```powershell
uv venv --python 3.12 .venv-visual-windows
uv pip install --python .venv-visual-windows\Scripts\python.exe -r scripts\requirements-visual-windows.txt
$env:SCENERECALL_VISUAL_PYTHON = "$PWD\.venv-visual-windows\Scripts\python.exe"
$env:HF_HOME = "D:\SceneRecall-Cache\huggingface"
```

`requirements-visual-windows.txt` 固定本次已验证的 Windows Python 3.12 包版本，包括 dghs-imgutils 0.19.0、NumPy 1.26.4、ONNX Runtime 1.23.2。该环境同时供 CCIP 视觉识别与 LR-ASD 预览人物检测使用。启动脚本会自动发现本项目的 `.venv-visual-windows\Scripts\python.exe`。

本机继续使用 `ccip-caformer-5_fp32`，复用 D 盘 Hugging Face 缓存中的三个文件，未下载或切换模型：

| 仓库与文件 | 原有 snapshot |
| --- | --- |
| deepghs/ccip_onnx · ccip-caformer-5_fp32/model_feat.onnx | eb2acdd29af1703388d3d0c04221add322bc9110 |
| deepghs/ccip_onnx · ccip-caformer-5_fp32/model_metrics.onnx | eb2acdd29af1703388d3d0c04221add322bc9110 |
| deepghs/anime_person_detection · person_detect_v1.1_m/model.onnx | e39c744c22432ad01f91dd254fe2b02c8d878b8c |

此缓存的既有相对符号链接已在本机 Windows 验证可以读取，无需覆盖原 WSL 缓存。Worker 强制离线并使用 UTF-8 JSON 管道；缺少权重时返回明确错误，不自动下载。

## LR-ASD

LR-ASD 保持上游提交 `1b6dcd2d8fc2895683de6508ec6294ec47d388ca` 兼容的 Python 3.10 与 NumPy 1.23.5。为使基础解释器与环境一同保存，可先将 uv 管理的 Windows Python 装入目标目录，再用 `--allow-existing` 创建环境，保留其中的基础解释器：

```powershell
uv python install 3.10.19 --no-bin --no-registry --install-dir .venv-lrasd-windows\runtime
uv venv --allow-existing --python .venv-lrasd-windows\runtime\cpython-3.10.19-windows-x86_64-none\python.exe .venv-lrasd-windows
uv pip install --python .venv-lrasd-windows\Scripts\python.exe --torch-backend cu128 -r scripts\requirements-lrasd-windows.txt
```

这是独立的 Python 3.10 worker，不安装 SceneRecall 主应用包。uv 可能提示它不满足主应用的 Python 3.12 要求，这是这两个环境隔离的原因。本次已装环境的基础解释器位于 `.venv-lrasd-windows\python-base`，运行入口同为 `.venv-lrasd-windows\Scripts\python.exe`。requirements 固定已验证的 44 个包，包括 torch 2.8.0+cu128 与 torchvision 0.23.0+cu128；需要本机可用的 NVIDIA 驱动。

现有资料的音画同步配置迁移为：

| 设置项 | 本机 Windows 路径 |
| --- | --- |
| sync_python_path | D:\SceneRecall\.venv-lrasd-windows\Scripts\python.exe |
| sync_repository_path | D:\SceneRecall-LR-ASD |
| sync_checkpoint_path | D:\SceneRecall\training\lrasd\runs\cmdam-jp2d-20261015-v1\bundle |
| sync_detector_path | D:\SceneRecall\training\lrasd\models\person.onnx |

保留的 bundle checkpoint SHA-256 为 `4ad11260f30fb3693042076acbef31dea1c27d7a5a2c8ab79a7885c60196328a`，人物检测器 SHA-256 为 `6fcdfc6af18b3a26aca62aad0e00faffbeea23413abf868808d2d98d399999c9`。未切换到后续人工微调候选，也未改变模型分数域或人工身份记录。

音频配置原有的 diarization、embedding、ASR 模型路径为空，ASR 关闭；迁移保留该状态，不把它们描述为已装好的声音识别功能。原有 LR-ASD 状态及许可确认也按原设置保留。实际验证结果见 [验证记录](VALIDATION.md)。

## 2026-10-18 原生验证

验证复用资料库既有的 1,199,000–1,203,000 ms 片段，数据复制至项目内 `test-results/native-models` 后执行，没有覆盖正式库中的模型结果或人工身份。

切换后另通过正式 HTTP API 完成同范围的新预览：本地总计 21.406 秒、1 条轨迹、`cache_hit=false`，短片及人物图可读取，0 外部模型请求。该结果作为新预览保存，既有结果和人工身份保留；详细阶段耗时见 [验证记录](VALIDATION.md)。

| 验证 | 结果 |
| --- | --- |
| CCIP 与 anime person worker | 原生 Windows Python 3.12.6 成功加载原 snapshot；单帧检测 2 框，768 维特征，自距离 4.88e-08；本次进程加载 13.435 s、推理 0.574 s |
| LR-ASD 逐帧人物检测 | ONNX Runtime 1.23.2；100 帧、1 条连续轨迹、零插值；所有轨迹和逐帧人物框与既有 WSL 输入完全一致 |
| LR-ASD CUDA 推理 | Windows Python 3.10.19、CUDA 12.8、RTX 4070 Laptop；完整 168 项严格加载；100 帧首次用时 2.385 s，恢复任务后复验 0.836 s，概率范围 0.5379619–0.8768268，均值 0.8211666942 |
| LR-ASD 与 WSL 对比 | 同权重、同音频与 RGB 输入、同 100 个时间点；逐帧最大绝对误差为 0，均值完全相同；输出保留 review_required |
| Windows worker 管道 | 多轮回复、完整行超时、中途退出、中文 UTF-8 JSON，以及正常关闭和超时后的实际 worker PID 回收通过；两个独立环境的 uv pip check 均通过（视觉 65 包、LR-ASD 44 包） |
| Windows worker 退出 | 实际加载 CCIP 后关闭后端，venv 启动器及其基础 Python 子进程均退出，没有遗留模型 worker；正常关闭与部分行超时回收有独立回归测试 |

这些结果证明现有模型可由原生 Windows 环境加载和运行，并且该片段输出与迁移前一致；不扩大模型原有的身份判定、标定或准确率声明。此处耗时为本机单次验证记录。
