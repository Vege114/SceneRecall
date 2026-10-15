# 独立角色图库、局部身份与基准

本次是在原队列、场景记录和检索投影上增加本地身份阶段。身份未校准时保留 `candidate`；没有参考图、图片质量不合格、外部模型不可用时保留 `unknown` 或明确的阶段降级。测试通过只验证工程约束，不代表动画识别准确率已经验收。

## 参考图与出场证据

`works/<character_scope_id>/reference-images/gallery.json` 保存独立图库。图片来源为 `upload`、`external_cached` 或 `frame_crop`，保留角色 ID、来源 URL / 原片资产与帧 / 裁剪区域、人工确认状态、质量、内容 SHA256、版本和完整修改审计。图片作为本机 PNG 保存；外部 URL 仅作出处，不触发下载。

图库不增加 `appearances`，不构造 `representative`，也不证明角色在某集出现。无出场记录的角色可提前拥有参考图。在从零建档作品中，`POST /api/assets/{asset_id}/characters` 可人工建具名角色；绑定 Bangumi 名单的作品继续限制在允许的角色中。

上传默认 `candidate`。进入可信库须人工确认身份并明确确认图片仅含一个角色，且通过至少 64×64 和边缘方差的基础质量检查。CCIP 提取前再由动漫人物检测器检查单人约束；多人图和检测未确定单人的图片不会进入本次匹配。清晰度门槛是可审计的质量规则，不是人物置信度，也不代表已完成动漫测试集校准。

人工撤销保留图片及历史，立即从可信图库剔除；已有场景的相关自动身份在展示、检索和再次发布时变为待核对。针对具体人物区域的人工确认优先于模型结果。人工撤销区域关联也留审计，不修改原始模型描述。

## 视觉模型与隔离

适配实际的 `imgutils.detect.detect_person`、`ccip_extract_feature` 和 `ccip_difference` API。先检测多人帧、按原图裁剪并检查混入其他人的区域，再提取单角色特征。距离采用 CCIP 自身的学习度量模型，没有替换成余弦距离，也没有计算 `1-distance` 来伪造 `match_confidence`。

所有当前允许角色的可信参考图均参与检索，随后按每个局部区域选择 Top-K，再合并当前批次需要的档案和图片。不会在检索前按更新时间裁掉第 17 个及之后的角色。每个角色用距离最近的两份可信参考图的平均距离（仅一份时用该份）；保存每份引用及候选间差距。正脸、侧脸、换装可分别入库，但是否有效必须用真实素材验证。

`P1/P2` 仅在当前原帧中编号；持久区域 ID 包含资产、原帧、时间与检测模型，不凭相邻或相似外观跨镜头合并。场景实体通过 `identity_region_id` 精确引用区域；事件仍经局部 `actor_ids/target_ids` 关联人物。角色名在读取/检索时解析，不全文替换历史描述。

重型依赖在常驻离线 worker 中运行。应用默认使用自身 Python，可用 `SCENERECALL_VISUAL_PYTHON=/absolute/path/to/visual-venv/bin/python` 指向独立环境。`dghs-imgutils==0.19` 的 NumPy 约束与 pyannote 4 不兼容，应使用独立视觉环境；具体安装命令见主视听文档。worker 仅在启用视觉身份后启动，设置自己的 `HF_HUB_OFFLINE=1`，不会改变其他 provider 的环境或自动下载模型。

需用户预先审阅权重条款并把以下仓库所需文件缓存到该 worker 的 Hugging Face 本地缓存：

- `deepghs/ccip_onnx`：所选模型目录中的 `model_feat.onnx` 与 `model_metrics.onnx`。默认 `ccip-caformer-24-randaug-pruned`，另支持官方列出的 6/5 变体。
- `deepghs/anime_person_detection`：`person_detect_v1.1_m/model.onnx` 与其发布的模型元数据（如 `model_type.json`）。建议预缓存该模型完整目录，避免只复制 ONNX 而遗漏预处理信息。

worker 调用官方预处理，加载检测、特征、度量三个 ONNX 会话并保持常驻；加载计时纳入阶段和总耗时。模型版本包含 imgutils 包版本及本地 Hugging Face snapshot commit。图片特征键包含内容散列、模型与预处理版本；检测键另包含素材指纹、帧及时间。改角色名、确认候选与撤销错误样本不会重新计算不变的图像特征。

官方代码为 [imgutils](https://github.com/deepghs/imgutils)，包声明 MIT；权重和训练数据条款仍须分别核对。本项目只调用 API，没有复制第三方实现。参考 [CCIP 文档](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html)、[人物检测代码](https://github.com/deepghs/imgutils/blob/main/imgutils/detect/person.py)、[CCIP 权重](https://huggingface.co/deepghs/ccip_onnx)、[人物检测权重](https://huggingface.co/deepghs/anime_person_detection)。

## 配置与校准

`GET/PUT /api/assets/{asset_id}/identity/settings`：

```json
{"enabled": false, "model": "ccip-caformer-24-randaug-pruned", "top_k": 5, "calibration": null}
```

默认不自动确认任何 CCIP 预测。拥有适用于当前作品/模型的独立标注测试集后，才可写入校准记录：

```json
{
  "model_version": "使用实际 worker 返回的完整版本字符串",
  "dataset_id": "held-out-set-id",
  "evaluation_id": "reviewed-evaluation-id",
  "aggregation": "nearest-two-mean-v1",
  "max_distance": 0.0,
  "min_margin": 1.0
}
```

这里的数值仅演示 JSON 格式，不能当作推荐阈值。阈值须来自该测试集，且与多参考聚合方式相同；错误模型版本不会启用自动确认。即使通过距离门槛，没有第二候选可计算差距时也不自动确认。

原始检测、特征、各版匹配与区域结果分层保存；人工映射在 `assets/<id>/identity/visual/decisions.json`。图库、审计和视听原始证据参与常规元数据备份，特征缓存可重建。

## 三组冷/暖基准

[`scripts/benchmark_identity.py`](../scripts/benchmark_identity.py) 对相同资产、保存的关键帧计划、字幕、允许角色与参考库比较：

1. `old`：`scene-chunks-v1` 配合改造前冻结的提示词版本，移除本地视听身份阶段。它是当前程序中的旧提示词基线，不是旧提交的二进制程序；原代码按更新时间裁剪角色档案的细节未完整恢复。
2. `new_visual`：`scene-chunks-v1` 加本地视觉身份。
3. `new_audiovisual`：`scene-chunks-v1` 加本地视觉与音频身份。

每组在新进程执行冷启动，再在相同进程执行暖缓存。暖缓存保留局部特征、音频提取/分段与模型会话，但从相同初始档案重建分析和索引，不复制之前的付费模型回执。操作系统文件缓存与远程服务缓存不受控制，报告明确保留此限制。

默认不调用 provider；下列命令仅生成 `not_measured` 报告。路径和 ID 应替换为本机数据，范围须为保存扫描中的完整镜头边界且不超过 120 秒：

```sh
uv run python scripts/benchmark_identity.py \
  --library /path/to/library --asset asset_id --plan plan_id \
  --start-ms 0 --end-ms 30000 --work-dir /tmp/identity-benchmark \
  --output /tmp/identity-benchmark.json
```

完成模型、许可、范围和预算核对并确实授权调用后，添加 `--allow-provider-calls --max-requests 20`。预算是每组每次运行的上限，三组各两次，共六次；最多可能产生 120 次请求，服务内部重试按队列已有规则计费。脚本仅支持显式配置的 Codex CLI 视觉连接，不复制 API 密钥，不启动训练或整集分析。可用 `--visual-batch-size 8` 冻结 1–16 张的分块批次大小，用不同工作目录对照批次策略；不靠单纯提高超时来制造提速。

总耗时包括本地模型加载、音频提取/分段/特征、视觉身份、视觉调用和索引；`visual_model_load_ms` 只单列视觉模型加载，音频模型加载仍计入分段/特征及总时长，没有伪造单独的音频加载数值。原始影片不复制，已有扫描不重做，其耗时明确排除。首次写入可查询索引时记录首次可检索时间，并另外记录首次画面可检索时间。报告统计视觉调用、二分回退、超时、无效输出、角色覆盖率，以及有真值时的错误率和台词说话人准确率。

可传 `--labels /path/to/labels.json`：

```json
{
  "visual": [{"frame_id": "frame_1", "box": [0.1, 0.1, 0.3, 0.8], "character_id": "character_A"}],
  "speech": [{"start_ms": 200, "end_ms": 1800, "character_id": "character_A"}],
  "diarization": [{"start_ms": 200, "end_ms": 1800, "speaker_id": "annotated_speaker_A"}]
}
```

视觉真值通过同帧、区域 IoU≥0.5 的全局一对一匹配评估。语音归属按标注说话时长计，候选/未知为弃权，重叠发言分别计算。DER 使用全局最优说话人置换、0 ms collar，包含重叠；结果同时给漏检、误检和混淆的说话人毫秒数。无真值的准确率为 `null`，不填 0 或从单元测试推算。

本次提交的 [`validation-identity-benchmark-20261015.json`](validation-identity-benchmark-20261015.json) 明确保留六个未实测单元格。尚未安装的权重、真实动画片段与人工标注不能由合成测试替代，也不能据此报告提速或准确率。
