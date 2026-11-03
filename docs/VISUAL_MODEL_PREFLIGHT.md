# 视觉识别模型权重的离线预检

视觉身份 worker 在第一次加载模型运行库之前，先检查当前选定模型需要的全部本地权重。缺少文件时直接返回所缺仓库、文件名和缓存目录，避免先初始化运行库后才得到难以定位的加载错误。

## 检查内容

每次新 worker 首次处理请求时，需要在 Hugging Face 本地缓存中找到：

| 仓库 | 必需文件 |
| --- | --- |
| `deepghs/ccip_onnx` | `<所选模型>/model_feat.onnx` |
| `deepghs/ccip_onnx` | `<所选模型>/model_metrics.onnx` |
| `deepghs/anime_person_detection` | `person_detect_v1.1_m/model.onnx` |

文件解析明确使用 `local_files_only=True`。worker 在导入 Hugging Face 或 imgutils 之前设置 `HF_HUB_OFFLINE=1` 和 `HF_HUB_DISABLE_TELEMETRY=1`。预检不会联网下载，也不会自动替换为另一个已经缓存的 CCIP 模型。

若缺少权重，应预先准备界面选定的模型文件，或明确选择已经完整缓存的本地模型，再重试任务。错误信息中的缓存位置来自运行 worker 所使用环境的 `HF_HUB_CACHE`，可能与其他 Python 环境不同。

## 模型身份与失败边界

- 三个文件全部可用后，才执行 ONNX Runtime 配置、导入计算依赖并启动既有模型预热。
- 模型版本标识继续包含选定模型名与缓存快照提交目录；没有快照目录时保留文件修改时间作为既有回退标识。
- 预检确认文件在缓存中且是普通文件，不验证权重数值正确性、模型兼容性或 CUDA 可用性。这些问题仍由后续加载和运行时检查报告。
- 缺文件错误保持现有 JSON 行 worker 协议，不产生成功结果；诊断文字不会污染协议标准输出。
- 既有 CUDA DLL 预加载、TF32 设置和拒绝静默回退 CPU 的规则不变。

## 验证

以 `f907431` 为基线，仅加入本功能的 worker 和测试文件，在独立代码副本中运行：

```powershell
python -m pytest tests/test_visual_worker_runtime.py -q
```

9 项通过。使用假的 Hugging Face 缓存与 ONNX 会话，验证选定模型的三个文件、离线参数、快照身份、逐个文件缺失的错误内容，以及缺文件时不启动计算运行库。测试不下载权重、不调用真实模型或处理真实视频。
