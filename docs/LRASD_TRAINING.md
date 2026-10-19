# LR-ASD 本机动画微调与接入

本流程使用用户授权的本机动画，训练人物全身区域输入的 LR-ASD 学生模型。AVA 与 TalkSet 公开权重接收同时间人物脸部区域，提供机器弱标签。原视频与外挂字幕只读，候选裁图、PCM、标签、训练状态和导出权重写入 `training/lrasd/`；该目录已排除在 Git 跟踪之外。

运行状态以 `label-summary.json`、训练目录的 `progress.json` 和最终 `bundle/metadata.json` 为准。本文描述可重现的操作及验收条件，不报告尚未取得的模型效果。

## 数据与评估范围

少女歌剧 TV 来源按集划分，固定 01–08 集用于训练、09–10 集用于验证、11–12 集用于最终测试。`group` / `evaluation_group` 使用集标识，相邻窗口和同一集不能跨集合。该划分能检查留出集上的机器标签一致性，不能证明跨作品泛化。

已检查的 12 份简体 ASS 中，3183 条 `Default` / `Default2` 对白的 Name 字段全部为空；非空 `shape` 来自 OP 特效。字幕只用于选择对白候选时段和排除已识别的歌曲区间，不能产生人物说话标签。日语音轨通过 `ffprobe` 的流语言标签核实，字幕译文不与日语发音强制对齐。

当前标签没有经过人工逐帧验收。弱标签验证集上的 AP、AUC、F1 或平衡准确率，含义都是与教师共识的吻合程度。教师可能共同出错，学生也可能学习其偏差；这些指标不能作为真实说话人准确率、概率校准或跨剧效果的依据。

## 环境与输入约定

主应用使用 Python 3.12。LR-ASD 使用独立环境，包含 PyTorch、NumPy、OpenCV、SciPy 和 `python_speech_features`；候选检测使用已有的 `.venv-visual` 与本机 `face.onnx`、`person.onnx`。FFmpeg / ffprobe 需要可执行。第三方源码和权重必须预先放在本机，脚本不会自动下载。

以下示例从项目根目录运行，路径变量由运行者填写。`LOCAL_MEDIA_DIRECTORY` 指向已授权素材目录，不在文档中记录私人视频完整路径。

```sh
export LRASD_REPO=/path/to/local/LR-ASD
export LRASD_PYTHON="$LRASD_REPO/.venv/bin/python3.10"
export VISUAL_PYTHON="$PWD/.venv-visual/bin/python"
export APP_PYTHON="$PWD/.venv/bin/python"
export LOCAL_MEDIA_DIRECTORY=/path/to/authorized/media
```

训练及应用推理共享版本化预处理：源视频重采样为 25 fps，整帧经 FFmpeg 双线性缩放到 480×270，按归一化人物全身框取整裁剪，转灰度并双线性缩放到 112×112。训练窗口可选连续 100 帧（4 秒）或适合短镜头的 50 帧（2 秒）。音频取同一原片时间范围的 16 kHz 单声道 PCM16，用原数值尺度计算 13 维 MFCC，每视频帧对应 4 个音频步，分别得到 400 或 200 步。

灰度输入保持 0..255，归一化由原始 `ASD_Model` 内部完成。不能在外部再次除以 255。脸部教师与全身学生使用各自裁图，但必须使用相同窗口和音频；当前预处理标识为 `lrasd-gray112-pcm16-mfcc13-25fps-fullbody-v1`。

## 准备来源与候选

```sh
"$APP_PYTHON" scripts/prepare_lrasd_sources.py \
  --directory "$LOCAL_MEDIA_DIRECTORY" \
  --output training/lrasd/revue

"$VISUAL_PYTHON" scripts/prepare_lrasd_anime.py \
  --sources training/lrasd/revue/sources.json \
  --models training/lrasd/models \
  --output training/lrasd/candidates-2s \
  --frames 50 --max-windows 1000 --max-attempts 2000 \
  --max-tracks-per-window 3 --include-dialogue-gaps
```

来源脚本只匹配已授权的少女歌剧 TV 文件名范围，选择 `.sc.ass`，记录逐集音轨、视频与字幕 SHA-256、样式/Name 统计，以及合并后的对白和音乐区间。歌曲排除来自字幕样式与卡拉 OK 标记的启发式，不能保证覆盖全部歌唱。`clean_dialogue_intervals` 额外排除已识别音乐区间前后各 0.5 秒。

候选准备器在 CPU 上运行本机 ONNX 检测器，检查九个时间点的人物/脸部唯一配对与轨迹连续性，在检测点之间插值人物框，并拒绝明显切镜或歧义轨迹。机器检测、插值框及单人画面都不构成人物身份或说话事实。对白空隙也只是额外候选，不等于已确认静音。

`manifest.jsonl` 每行保留集划分、源时间、人物框、素材哈希和检测配置。学生的 `visual_path` 与教师的 `teacher_visual_path` 分别指向含 `visual` 键的 `uint8[T,112,112]` NPZ；`audio_path` 指向该窗口的 PCM WAV，`audio_start_seconds` 为 0。未标注候选的 `labels` 全为 -1。准备、弱标注和训练的 `--frames` 必须保持一致，不能混用 50 与 100 帧窗口。

## 冻结候选快照

当前 2 秒候选使用三个目录：`candidates-2s`、`candidates-2s-val` 和 `candidates-2s-test`。第一个目录含初期少量验证/测试候选，后续只追加训练候选；后两个目录分别补充固定验证集和测试集。目录名称不能代替行内的 `split` 和集标识，合并时须保留原划分。

弱标注前使用 `freeze_lrasd_candidates.py` 读取各输入的完整字节快照，导出一个固定 manifest；不要让 labeler 直接消费仍在增长的候选文件。`--manifest` 可重复给出，`--output` 及其审计文件必须是新路径：

```sh
"$APP_PYTHON" scripts/freeze_lrasd_candidates.py \
  --manifest training/lrasd/candidates-2s/manifest.jsonl \
  --manifest training/lrasd/candidates-2s-val/manifest.jsonl \
  --manifest training/lrasd/candidates-2s-test/manifest.jsonl \
  --output training/lrasd/frozen/revue-2s-v1.jsonl
```

冻结脚本将资产引用转换为绝对路径，实际计算学生 NPZ、教师 NPZ 和 PCM WAV 的 SHA-256，并与候选记录核对；同时验证统一的 T=50/100、112×112 灰度形状、逐帧框和精确音频长度。一个快照不能混用窗口长度。相同 `sample_id` 只有在时间、标签、人物框、源身份、资产哈希和划分等语义一致时才去重，冲突会报错。源视频路径/哈希和 `group` 都检查跨 split 泄漏；验证/测试集不会按教师分数筛选。

脚本不覆盖已存在的输出或审计文件。后续纳入新候选时使用 `revue-2s-v2.jsonl` 等新版本，并重新生成对应标签目录。默认审计文件为 `revue-2s-v1.audit.json`，保存输入/输出哈希、同义重复的来源、集合计数及来源隔离检查。冻结只固定清单和审计记录，仍引用已生成的资产；后续阶段应保持这些 NPZ/WAV 不变。

## 生成教师机器弱标签

先用独立输出目录进行小批 smoke，确认两套教师能加载并返回有限概率，再对冻结的完整候选 manifest 运行。候选文件、教师权重或阈值变化时使用新的输出目录，避免复用不同配置的预测缓存。

```sh
"$LRASD_PYTHON" scripts/label_lrasd_candidates.py \
  --manifest training/lrasd/frozen/revue-2s-v1.jsonl \
  --output training/lrasd/labels-2s-smoke \
  --source-root "$LRASD_REPO" \
  --frames 50 --max-candidates 12 --batch-size 4 --device cuda

"$LRASD_PYTHON" scripts/label_lrasd_candidates.py \
  --manifest training/lrasd/frozen/revue-2s-v1.jsonl \
  --output training/lrasd/labels-2s \
  --source-root "$LRASD_REPO" \
  --frames 50 --max-candidates 1000 --batch-size 4 --device cuda
```

教师为 `weight/pretrain_AVA.model` 与 `weight/finetuning_TalkSet.model`，均使用原始 LR-ASD 定义、安全张量加载及完整 168 项 `state_dict` 严格检查。各教师的分类概率取 AV 分类头两个类别 logits 的 softmax，不使用上游无标签分支返回的原始 logit。

逐帧规则固定为：两位教师概率均 ≥0.95 才标 1，两位均 ≤0.05 才标 0，其余标 -1。再屏蔽置信类别变化边界附近 ±2 帧及窗口两端，每窗至少保留 40% 的有效帧：100 帧窗口默认 40 帧，50 帧窗口默认 20 帧。`--min-valid-frames` 可提高门槛，不能低于这个比例或超过窗口帧数。批次最多 4 个窗口、总计不超过 400 个视频帧。概率阈值用于保守筛选，尚未经过人工动画集校准。

2 秒窗口同样先做小批 smoke，并执行相同的教师概率阈值、边界遮罩、类别和样本数量检查。缩短窗口不会把机器弱标签变成人工真值，也不会降低逐帧置信标准。4 秒实验则在独立目录准备、冻结和标注 T=100 窗口，所有阶段使用 `--frames 100`。

| 产物 | 用途 |
| --- | --- |
| `candidate-probabilities.jsonl` | 全部已处理候选的两位教师原始逐帧概率、均值、标签和拒绝原因，支持同配置续跑 |
| `labeled.jsonl` | 通过有效帧门槛的窗口，明确记录 `machine_generated=true`、`human_annotated=false` |
| `label-summary.json` | 集合样本/类别数量、教师分歧、来源权重哈希及 `training_ready` |
| `label-progress.json` | 当前处理状态与数量 |

至少需要 20 个训练窗口、5 个验证窗口和 5 个测试窗口，且每个集合都须具有有效正类和负类帧。未满足时保留候选与审计结果，`training_ready=false`，脚本退出码为 2。小批 smoke 通常也会因样本数量不足退出 2；应查看预测和摘要，不应因此放宽标签阈值或绕过完整训练门槛。

## 训练、恢复与留出评估

只有冻结的 `labeled.jsonl` 与 `label-summary.json` 通过门槛后，才运行以下训练命令：

```sh
"$LRASD_PYTHON" scripts/train_lrasd.py \
  --manifest training/lrasd/labels-2s/labeled.jsonl \
  --output training/lrasd/runs/revue-2s-weak-v1 \
  --source-root "$LRASD_REPO" \
  --epochs 20 --patience 4 --batch-size 8 --frames 50 \
  --lr 0.0001 --lr-decay 0.95 --seed 114 --workers 0 \
  --dataset-name "Revue Starlight TV" \
  --dataset-version "2026-10-15-teacher-consensus-v1" \
  --source-domain "Japanese 2D TV anime; machine teacher consensus" \
  --target-domain "Japanese 2D anime; cross-title and human validation pending" \
  --label-source ensemble_teacher_weak --class-balance inverse-frequency \
  --allow-within-show-group-split --split-strategy episode_holdout_same_show \
  --device cuda
```

训练从 AVA 完整权重初始化，FP32 全参数更新，使用 `CE(AV) + 0.5×CE(visual)`、AdamW、梯度裁剪及按有效帧类别频数加权的损失。当前 2 秒训练配置为 T=50、B=8，总计 400 个视频帧；4 秒配置使用 T=100、B≤4。教师标注仍限制 B≤4。最多 20 轮；验证 AP 连续 4 轮未提升时提前停止。不会把不确定的 -1 帧当负样本，也不会仅靠纯正/纯负窗口数量决定类别权重。

训练前保存基线验证结果；每轮只用验证集选择 checkpoint 与阈值，训练结束后再评估固定的测试集并保存基线对照。测试集也是机器弱标签，评估语义仍为教师共识一致性。

`latest.pt` 保存模型、优化器、调度器、随机状态和批次位置；`best.pt` 保存最佳验证 checkpoint。恢复时保留相同 manifest 和训练配置，在原命令后加 `--resume training/lrasd/runs/revue-2s-weak-v1/latest.pt`。修改数据或配置应另开输出目录，不能将 100 帧 checkpoint 直接作为 50 帧运行的恢复状态。`--max-steps` 可用于有界运行并保存恢复状态；它不会导出“已完成”的模型包。运行进度、损失和显存记录分别见 `progress.json`、日志及 `history.jsonl`。

## 模型包验证与 SceneRecall 设置

只有完成训练、验证及最终留出评估后，runner 才会生成 `bundle/model.pt` 和 `bundle/metadata.json`。`latest.pt` / `best.pt` 是恢复状态，不能当作应用模型包。导出记录包含权重 SHA-256、架构、画布与预处理版本、数据划分哈希、弱标签来源、训练轮数、验证指标及未校准/未人工验证限制。

可在主项目根目录核对导出与原始模型严格加载：

```sh
PYTHONPATH=backend "$LRASD_PYTHON" -c '
import os
from pathlib import Path
from scenerecall.av_sync_worker import validate_lrasd_bundle, load_lrasd_model
metadata, checkpoint = validate_lrasd_bundle(Path("training/lrasd/runs/revue-2s-weak-v1/bundle"))
model = load_lrasd_model(Path(os.environ["LRASD_REPO"]), checkpoint, "cpu")
print(metadata["schema_version"], metadata["training"]["status"], len(model.state_dict()))
'
```

应用在资产详情的“视听身份 → 本机模型与音画复核配置”中选择：

| 设置 | 填写内容 |
| --- | --- |
| 音画复核模型 | `LR-ASD`，对应 `sync_model_type=lrasd` |
| 模型代码目录 | 本机 LR-ASD checkout 根目录，包含 `model/Model.py`、`model/Encoder.py`、`model/Classifier.py` 和 `loss.py` |
| LR-ASD Python 路径 | 安装了依赖的独立解释器绝对路径，对应 `sync_python_path` |
| LR-ASD 模型包目录 | 完成导出的 `bundle` 目录，对应 `sync_checkpoint_path`；不填单个 `.pt` 文件 |
| 运行设备 | 与应用实际运行环境一致的 `cpu` 或 `cuda` |

路径须对运行 SceneRecall 的进程可见；WSL 运行时使用 Linux/挂载路径，不能直接使用另一运行环境的解释器。保存配置不会下载权重或自动启用模型。核对导出记录后，按需要勾选歧义与冲突短片段的音画复核并保存。

加载器检查 bundle 版本、完整 168 项权重、SHA-256 和已完成的训练/评估记录。弱标签评估不等于人工验收；模型包中的 `domain_validation`、`calibration` 限制仍保留。LR-ASD 不使用 LWTNet 的同步分数阈值。

短片段复核需要同源连续帧、音频及已有确认视觉身份的人物轨迹，不能用稀疏关键帧拼图代替。结果字段为 `active_speaker_probability` 及帧级概率，始终以候选、待人工复核状态保存。它不自动覆盖人工身份，不自动把预测添加到可信声纹库，也不能单凭听到某个声音断言对应人物可见。

生产使用前仍需独立人工标注的日语动画测试片段，包含画外音、多人同框、重叠、唱歌和切镜等情形，并另用未参与开发的作品验证泛化与校准。相关身份和声音证据规则见[视听身份使用说明](CHARACTER_AV.md)与[声音身份说明](AUDIO_IDENTITY.md)。

## 2026-10-15 本机实验记录

本轮使用辉夜姬电影及少女歌剧 TV 的日语音轨。固定视觉采集预算产生 700 条记录，去重并核验 1865 个资产后冻结为 679 条 T=50 候选；训练、验证、测试分别为 379/150/150 条。双教师共识最终接受 37/29/15 条，三个集合均有正负标签。训练有效标签为 1119 帧，合计约 44.8 秒；这是一轮小规模机器弱标签实验，不能称为充分监督的完整生产模型。

在 RTX 4070 Laptop GPU 上以 FP32、B=8、T=50 运行，最多 20 轮，patience=4；实际完成 5 轮、25 次更新后提前停止，导出验证 AP 最佳的第 1 轮 checkpoint（该 checkpoint 完成 5 次更新）。真实数据训练日志约为 0.40 秒/步；CUDA 分配峰值 2.25 GiB、保留峰值 2.54 GiB、进程峰值 RSS 1.77 GiB。模型在 88°C 时暂停 10 秒并降温后恢复。进程墙钟 71.638 秒，包含初始化和降温，不包含数据准备和教师标注。

| 与机器弱标签的一致性 | AVA 基线 | 导出 checkpoint |
| --- | --- | --- |
| 验证 AP | 0.920447 | 0.920140 |
| 最终测试 AP | 0.822441 | 0.868285 |
| 最终测试 F1（阈值仅由验证集选定） | 0.460905 | 0.637363 |

验证 AP 基本持平、略低，不能声称本轮已取得明确的真实效果提升。测试集标签也来自教师模型，测试结果不代表人工准确率。固定种子及可恢复状态已保存，但 PyTorch 对 CUDA max-pool 反向传播报告了非确定性算子警告，不能保证跨运行逐位相同。

导出位于本机忽略目录 `training/lrasd/runs/jp2d-2s-20261015-v1/bundle/`。真实素材已通过独立的 SceneRecall worker 推理：完整权重加载、SHA、50 帧有限概率和 `candidate` / `review_required` 状态检查通过。相对 AVA 初始化，90 个参数张量均发生变化；这只证明实际更新与推理执行，不证明准确率。未修改用户资产或可信声纹库，未自动设置为默认模型。

专项测试为 32 passed / 1 skipped，前端构建及修改代码的 Ruff 检查通过。此前完整后端测试为 1164 passed / 8 skipped / 2 failed；两个失败为未修改模块中的 Node 子进程启动时间假设和不同 Python 版本的 data URI 解码行为，隔离重跑仍失败。本轮未扩展到这些无关模块。

## CMD-AM 官方注释素材恢复记录

已按固定清单找齐 8 部电影对应的 40 个官方注释 clip。素材恢复分为三个来源，按原始 clip ID 精确匹配，没有下载整套年度视频包：

| 来源 | 数量 | 本地完整性与来源记录 |
| --- | --- | --- |
| [Animated AD 项目](https://www.robots.ox.ac.uk/~vgg/research/animated_ad/)公开 Dropbox 年包中的选定成员 | 30 | HTTP Range 读取 ZIP 目录与选定 MKV 成员；实际文件大小、解压大小和 CRC32 核对通过 |
| 原始 YouTube ID 的公开视频，电影 `Sing` | 5 | 保存原始 ID、公开页面元数据和实际下载文件 SHA-256 |
| [FYQ12138/condensed-movies](https://huggingface.co/datasets/FYQ12138/condensed-movies) 公开第三方 HF 镜像，电影 `The Secret Life of Pets` | 5 | 固定提交 `77e0d3b0b0f072758f6b0df2af4f06bd85e27193`，只取目标 TAR 成员；记录表明 `private=false`、`gated=false`、未使用凭据 |

本机再次核验得到 40 个唯一视频、共 328,453,474 字节，40 份注释的 SHA-256 均匹配选片清单，40 个视频的 SHA-256 均匹配逐片 probe；30 个 Dropbox 文件的 CRC32 和大小均匹配下载审计。HF 镜像元数据未声明 license，来源明确保留为第三方镜像；它不代表作者提供的原始字节，不能仅凭同名 ID 假定时间和画布一致。

补回的 10 个视频都保存了与实际素材 SHA-256 绑定的原生帧检查：首帧、分位锚点、官方关键帧和最后一个标注帧叠加人物框，复核画面内容、人物框位置及原生帧序号。检查记录确认从原生第 0 帧开始的标注前缀；采用固定 `[0, 8, 640, 344]` 黑边裁剪，排除标注前缀之外的片尾。`alignment-overrides.json` 记录精确源哈希、裁剪、前缀长度及报告/联系表证据；这些操作只产生派生资产，不改动原视频。该复核确认素材与官方框的对应关系，没有新增人工逐帧说话标签，也不证明模型准确率。

`prepared-v1` 的完整准备曾在音频尾部共同覆盖检查中停止：派生音频比请求的覆盖范围短 347 个采样点，超过原先 32 点容差。随后使用新目录 `training/lrasd/cmdam/prepared-v2/`，只保留具有共同音视频支持的完整 2 秒窗口，未以大段补零绕过边界检查。该版本现已完成 40/40 个 clip，最终 `label-summary.json` 为 `training_ready=true`，接受 1183 条 T=50 窗口：

| 集合 | 窗口数 |
| --- | --- |
| 训练 | 976 |
| 验证 | 84 |
| 测试 | 123 |

官方训练窗口共有 17,151 个正类、18,905 个负类和 12,744 个遮罩人物帧，有效监督合计 36,056 人物帧，约为先前日语实验 1119 有效人物帧的 32.2 倍。这是人物轨迹上的监督数量；同一视频帧可能包含多条人物轨迹，不能换算成互不重复的原视频时长，也不能据此认定标签质量或准确率提高。

## 2026-10-15 官方训练与日语适配结果

官方来源按电影固定为 6/1/1 划分，seed=114，不因个别素材下载困难而重抽集合：

| 集合 | 电影 | clip 数量 |
| --- | --- | --- |
| 训练 | `Coraline`、`Despicable Me`、`Ice Age`、`Kung Fu Panda`、`Sing`、`The Polar Express` | 28 |
| 验证 | `Spider Man: Into the Spider-Verse` | 7 |
| 最终测试 | `The Secret Life of Pets` | 5 |

官方句级说话人及人物框注释经保守时间对齐投影到帧，空白、边界和歧义处仍遮罩；其来源标识为 `cmdam_official_sentence_alignment`。这些是从官方注释派生的监督，不是人工逐帧嘴部活动真值。此处重用原评估注释训练，不能宣称得到了官方 CMD-AM benchmark 成绩。第一阶段已从完整 AVA 权重初始化并完成训练，输出目录为 `training/lrasd/runs/cmdam-full-20261015-v1/`。该阶段只用官方训练、验证集合选择模型，通过 `--defer-final-test` 生成下一阶段初始化用的中间包；最终测试在阶段二最佳 checkpoint 固定后计算。

第一阶段曾因短时达到 91°C，在第 164 次更新处正常保存并停止，降温后从原 checkpoint 恢复；优化器状态和训练 `config_hash` 保持不变。后续采用 86°C 暂停、降至 82°C 后恢复，每 2 次更新检查一次温度。该操作保留原训练状态，没有重新初始化模型或优化器。

第二阶段已保留既有 `labels-2s-v1` 的全部 37/29/15 条日语训练、验证、测试数据，没有重跑教师或修改标签。少女歌剧继续按集划分，辉夜姬电影样本仍只在原训练集合。`assemble_lrasd_adaptation.py` 从官方训练集合以 seed=114 的 SHA-256 排序固定选取 3×37=111 条不同窗口回放；不足 111 条会拒绝，不隐式重复或降低数量。阶段二训练集共 148 条，验证集保留官方 84 条和日语 29 条，测试集保留官方 123 条和日语 15 条。官方留出电影及其 clip、日语留出集分组和相应源视频哈希均未进入训练回放。下列命令记录本次配置，重跑时必须使用新的输出目录：

```sh
"$LRASD_PYTHON" scripts/assemble_lrasd_adaptation.py \
  --official-manifest training/lrasd/cmdam/prepared-v2/labeled.jsonl \
  --jp-manifest training/lrasd/labels-2s-v1/labeled.jsonl \
  --output training/lrasd/adaptation-v1
```

组装已输出 `labeled.jsonl`、`assembly.json` 和 `recipe.json`，要求全新目录。它核验输入完成摘要、manifest/config 和实际资产 SHA-256、T=50 灰度 NPZ 形状及 PCM 时长，保留每行原始 `labels` / `mask` / `label_source`，记录原行哈希、回放 ID 和来源隔离审计。混合模式标识为 `mixed_official_and_teacher`，验证及测试摘要按两种 `label_source` 分开报告；不会把官方派生标签或教师标签标为人工 ASD 真值。

阶段二配方固定为完整官方第一阶段 `bundle` 初始化、全参数 FP32、T=50、B=8、学习率 1e-5、decay=0.95、最多 10 轮、patience=4、seed=114、有效帧类别频数加权。使用全新优化器，不以第一阶段 `latest.pt` 作为 `--resume`。checkpoint 按两种来源各自验证 AP 的等权平均值 `macro_source_ap` 选择，避免大量官方验证窗口淹没少量日语窗口；概率阈值仍只由验证集有效帧选择。运行命令需明确传入：

```sh
"$LRASD_PYTHON" scripts/train_lrasd.py \
  --manifest training/lrasd/adaptation-v1/labeled.jsonl \
  --output training/lrasd/runs/cmdam-jp2d-20261015-v1 \
  --source-root "$LRASD_REPO" \
  --parent-bundle "$OFFICIAL_PARENT_BUNDLE" \
  --stage-name cmdam-japanese-2d-adaptation-v1 \
  --dataset-name "CMD-AM-Kaguya-Revue-JP2D" \
  --dataset-version "2026-10-15-official-plus-jp-teacher-replay-v1" \
  --source-domain "CMD-AM official-source ASD proxy labels plus Japanese Kaguya/Revue teacher-consensus weak labels; 111 distinct source training replay windows" \
  --target-domain "Japanese 2D anime; held-out Revue episodes from the same show, previously evaluated weak test; human ASD accuracy and cross-title generalization unverified" \
  --label-source mixed_official_and_teacher \
  --frames 50 --batch-size 8 --lr 0.00001 --lr-decay 0.95 \
  --epochs 10 --patience 4 --seed 114 --class-balance inverse-frequency \
  --max-gpu-temperature 86 --resume-gpu-temperature 82 \
  --validation-selection macro_source_ap \
  --allow-within-show-group-split --split-strategy episode_holdout_same_show \
  --previously-evaluated-test-source ensemble_teacher_weak --device cuda
```

其中 `OFFICIAL_PARENT_BUNDLE` 指向完成官方训练与验证的 `training/lrasd/runs/cmdam-full-20261015-v1/bundle/`。最终阶段二模型包位于 `training/lrasd/runs/cmdam-jp2d-20261015-v1/bundle/`；汇总报告为 `training/lrasd/full-training-results.json`。以下结果均来自该报告及对应 bundle 元数据。

### 实际训练时间与资源

| 阶段 | 实际完成 | 导出最佳 checkpoint | 普通更新中位数 | 训练阶段墙钟 |
| --- | --- | --- | --- | --- |
| 官方来源训练 | 6 轮、732 次更新，提前停止 | 第 2 轮、244 次更新 | 0.78146 秒/步 | 22.3517 分钟 |
| 日语适配与官方回放 | 6 轮、114 次更新，提前停止 | 第 2 轮、38 次更新 | 0.60243 秒/步 | 5.5070 分钟 |

两阶段训练墙钟合计 27.8587 分钟，包含初始化、恢复、降温及阶段内评估，不包含素材下载、预处理和独立应用 worker 检查。普通更新统计包含数据加载和计算，排除日志记录的降温区间，不能直接代替端到端耗时。两阶段记录的最大 CUDA 分配量为 2,423,605,248 字节（约 2.26 GiB），最大 CUDA 保留量为 2,759,852,032 字节（约 2.57 GiB），最大进程 RSS 为 1,926,213,632 字节（约 1.79 GiB）；RSS 是训练进程峰值，不是 Windows 或 WSL 全部内存。

### 验证与最终留出结果

官方 84 条验证窗口上的 AP 从 AVA 基线 0.823744 提升至第一阶段 0.882517。阶段二在相同固定留出数据上的分来源验证结果为：

| 验证标签来源 | 官方父包 | 阶段二适配包 |
| --- | --- | --- |
| CMD-AM 句级人物对齐 AP | 0.882517 | 0.883507 |
| 日语教师共识 AP | 0.888155 | 0.889534 |

阶段二按来源 AP 等权平均选择 checkpoint，不能将联合池化 AP 与各来源 AP 混为同一个指标。此前小规模 JP 实验记录的 AVA 日语验证 AP 0.920447 是历史结果，本阶段的父包基线是已经过官方来源训练的模型；不能把 0.888155→0.889534 描述为相对原始 AVA 的日语性能提升。

最终固定 checkpoint 后，在相同两组测试数据上分别比较官方父包和适配包：

| 测试标签来源 | 父包 AP | 适配包 AP | 父包 F1 | 适配包 F1 |
| --- | --- | --- | --- | --- |
| CMD-AM 句级人物对齐，123 条窗口 | 0.787743 | 0.787459 | 0.723124 | 0.720748 |
| 日语教师共识，15 条窗口 | 0.868872 | 0.879359 | 0.649533 | 0.634703 |

父包比较阈值为 0.2381071001，适配包阈值为 0.2866455317，均只在相同的混合验证集上选择，未用测试集调阈值。父包这一阈值是为本次比较重新选择的，和第一阶段仅官方验证集选出的阈值不同。日语弱标签测试 AP 小幅上升，但验证阈值对应的 F1 下降；官方来源测试 AP 与 F1 也略降。因此这次适配不能被描述为运行点 F1 改善、没有遗忘或已经达到生产准确率。

日语测试集已用于前述小规模实验，元数据记录为 `previously_evaluated`；官方来源测试的既往历史记录为 `history_not_declared`，两者均未宣称为未经开发使用的全新测试集。指标分别衡量与官方派生监督或机器教师共识的吻合，不能代替人工日语 ASD 准确率、跨剧泛化或概率校准。模型结果仍只作为待复核候选，不自动写入可信声纹库。

### 固定清单与模型哈希

| 产物 | SHA-256 |
| --- | --- |
| 官方 `prepared-v2/labeled.jsonl` | `3be0232fc0f34e47fc43946fa8122eb631ed8ab6dfbc9008b41bced242b6123a` |
| 日语 `labels-2s-v1/labeled.jsonl` | `74de0a7738a09b094a33cbd2ada06c6eb8da07b90b9f72e98f0ded5a0fdc9eb4` |
| 混合 `adaptation-v1/labeled.jsonl` | `561ed29b6a4406ea34f8d34818b42096739a1c6cf5c1a97d9142ad4d4bd015ee` |
| 官方父包 `model.pt` | `e45f26fdcc4074701efa21695360de34c3cafc548b01f824a6d2885dcc9b7c8d` |
| 阶段二适配包 `model.pt` | `4ad11260f30fb3693042076acbef31dea1c27d7a5a2c8ab79a7885c60196328a` |

### SceneRecall 独立 worker 验证

本机 `training/lrasd/verify_export.py` 已使用真实辉夜姬 2 秒训练窗口，通过 CUDA 独立调用 SceneRecall worker。完整模型与父包哈希链核对通过，50/50 帧均返回有限且在 0..1 内的概率，结果保持 `candidate` / `review_required`。相对官方父包，90/90 个可训练参数张量发生变化，837,156 个被比较元素中有 837,122 个发生变化，统计排除 running buffers。检查记录明确 `writes_user_assets_or_voice_bank=false`。这证明导出模型加载与真实输入推理执行通过，不证明准确率、身份成立或概率已校准。复现命令为：

```sh
"$LRASD_PYTHON" training/lrasd/verify_export.py \
  --bundle training/lrasd/runs/cmdam-jp2d-20261015-v1/bundle \
  --manifest training/lrasd/labels-2s-v1/labeled.jsonl \
  --source-root "$LRASD_REPO" --parent-bundle "$OFFICIAL_PARENT_BUNDLE" \
  --output training/lrasd/worker-smoke-cmdam-jp2d-v1 \
  --python "$LRASD_PYTHON" --device cuda
```

本轮相关专项测试为 54 passed / 1 skipped，修改代码的 Ruff 检查全部通过。运行没有自动切换 SceneRecall 默认模型、改写用户资产或添加可信声纹。

## 应用接入与 Bangumi 联动

正式资料库为 `D:\SceneRecallLibrary`。《超时空辉夜姬》的资产 ID 为 `asset_666d87eb4e274e1fa9b7`，已绑定 Bangumi 条目 `604826`，保存了 17 份角色资料及其图片 URL。新训练模型包为 `D:\SceneRecall\training\lrasd\runs\cmdam-jp2d-20261015-v1\bundle`；WSL 运行应用时对应 `/mnt/d/SceneRecall/training/lrasd/runs/cmdam-jp2d-20261015-v1/bundle`，解释器和源码目录仍需使用同一运行环境可见的路径。当前模型配置仅保存于该辉夜姬资产，新导入的资产不会自动继承。

资产详情中的入口为“视听身份 → LR-ASD短片段试跑”。选择原片中 0.64–8 秒的范围，使用同时间连续画面、人物区域及音频生成发言概率；未知人物和身份候选须核对，不能仅凭分数确定角色。LR-ASD 判断给定人物区域是否在发言，CCIP 使用参考图比较人物视觉身份，两种模型的分数与结论不能互相替代。

本次已通过显式缓存接口将 17 张 Bangumi 图片保存到正式资料库，没有跳过或错误。再次请求返回 `imported=0`、`existing=17`、`errors=[]`，未重复新增图片。所有图片均保持 `candidate`，没有标记为人工确认。当前接口默认每次最多处理 12 张，传入 `max_images=20` 可覆盖本次 17 份角色的下载范围：

```http
POST /api/assets/asset_666d87eb4e274e1fa9b7/references/cache-bangumi

{"max_images":20}
```

实际导入、已存在、跳过及错误数量以接口返回为准。缓存只读取已导入 Bangumi 资料中的官方图片地址，保留来源和哈希，新图保存为待确认候选；打开档案不会自动下载，也不会自动加入可信图库。图片可帮助复用现成人物参照，不能提供原片出场证据、逐帧说话标注或可信声纹，仍需核对角色身份与单人图片。

### 辉夜姬正式短片段试跑

正式预览 v3 的任务为 `3011770fa59e7d7068c077a6`，范围为原片 20:04–20:08，共 100 帧、两条连续人物全身框轨迹。报告保存于 `training/lrasd/app-kaguya/preview-v3-result.json`，小段视频为同目录的 `kaguya-20m04s-20m08s.mp4`。UI 提供经本地会话认证的 H.264/AAC 小段播放器，播放器 0–4 秒对应上述原片范围，无需生成整片代理视频。

| 轨迹与画面核对角色 | LR-ASD 发言分数 | CCIP 第一候选及距离 | 对应角色的候选排序及距离 |
| --- | --- | --- | --- |
| 左侧，辉夜 | 0.339578 | 绫䌷芦花，0.246569 | 辉夜，第 2 位，0.311376 |
| 右侧，酒寄彩叶 | 0.167244 | 酒寄红叶，0.265237 | 酒寄彩叶，第 3 位，0.320222 |

LR-ASD 分数均未校准，不能据此直接确定发言人。CCIP 距离越小排序越靠前，不是身份概率；这次两条轨迹的首选身份都误排，因此不能写成角色识别准确或自动确认成功。结果保留为待核对候选，图库图片也未因此获得人工确认。

本次总耗时 104.004 秒，记录到的子阶段如下，另有任务调度等开销；视觉特征提取共 19 次。

| 子阶段 | 秒 |
| --- | --- |
| 画面解码 | 12.852 |
| 小段视频编码 | 0.431 |
| 检测模型加载 | 0.655 |
| 人物检测 | 4.017 |
| LR-ASD 推理 | 37.017 |
| 视觉参考匹配 | 46.015 |

此前 19:59–20:03 首测同为 100 帧，但只形成一条宽人物框轨迹，框内两张脸重叠；其 0.821167 分数无法归属具体人物，未作为有效的角色或发言归属证明。

## 2026-10-18 人工标注微调与同片留出评估

本轮已经完成固定场景计划的人工标注导出、基线验证、微调和独立最终比较。24 个 2–6 秒片段合计 80 秒，按场景合并为 18 组；训练、验证和测试分别包含 16/4/4 个片段、10/4/4 个场景组。同场景的全部人物和窗口使用同一划分，跨划分的原片时间至少间隔 90 秒。原始标注及训练前备份保留，未将模型预测、字幕或角色图片当作人工说话标签。

准备检查为 24/24 就绪，导出 69 条连续 50 帧、25 fps 的两秒人物轨迹窗口，跳过 3 条全未知轨迹。数据量与指标单位如下：

| 集合 | 人物轨迹窗口 | 说话人物帧 | 未说话人物帧 | 遮罩人物帧 | 有效人物帧 |
| --- | --- | --- | --- | --- | --- |
| 训练 | 35 | 1,018 | 697 | 35 | 1,715 |
| 验证 | 16 | 346 | 454 | 0 | 800 |
| 测试 | 18 | 477 | 423 | 0 | 900 |

一条样本是一名人物在两秒连续画面中的轨迹，指标按未遮罩的人物帧计算。同一原片画面可以对应多个人物，邻帧也高度相关；69 条窗口不等于 138 秒互不重复的原片，900 个测试人物帧不等于 900 个独立场景。人工标签表示本轮已保存、明确复核的逐帧 ASD 标注，仍受标注质量及选片范围限制。

微调从前述官方来源与日语弱标签适配包初始化，并新建优化器；本轮只使用 35 条人工训练窗口，没有官方回放、蒸馏或参数锚定。配置为 FP32、T=50、B=4、学习率 2e-5、seed=114，最多 6 轮、验证 AP 连续 3 轮无提升后停止。实际完成 4 轮、36 次更新，导出第 1 轮、完成 9 次更新的最佳 checkpoint；训练记录墙钟约 18.315 秒，不包含标注、导出和独立最终比较。最大 CUDA 分配量为 1,252,680,192 字节，保留量为 1,440,743,424 字节，训练进程峰值 RSS 为 1,825,251,328 字节。

训练阶段通过 `--defer-final-test` 留出测试集；随后独立比较程序核对冻结数据、父包和候选包，再对同一测试集运行两个模型。以下全部使用独立 `comparison.json` 的统一评估口径，没有混用训练日志中的近似数值：

| 指标 | 既有基线 | 人工微调候选 |
| --- | --- | --- |
| 验证 AP，800 个有效人物帧 | 0.869690 | 0.866273 |
| 最终测试 AP，900 个有效人物帧 | 0.945100 | 0.943927 |
| 最终测试 F1 | 0.798544 | 0.771144 |
| 最终测试精确率 | 0.948127 | 0.948012 |
| 最终测试召回率 | 0.689727 | 0.649895 |
| 仅由各自验证集选定的阈值 | 0.7737769485 | 0.8140922189 |

候选的验证 AP、测试 AP 和测试 F1 均下降，因此本轮结论为**保留既有基线**，没有自动切换应用模型、覆盖可信身份或添加声纹。本轮没有评估官方来源域的遗忘，也没有证明跨作品泛化或概率校准。基线此前接触过本片少量机器弱标签；已检查的旧清单与本轮留出范围无冲突，但不能排除其他既往曝光，不能把它称为从未见过本片的模型。结果只覆盖同一影片的固定场景留出；测试结果已查看，后续若据此调整超参数或选择模型，需另收集最终测试场景。

| 冻结产物 | SHA-256 |
| --- | --- |
| 人工数据 manifest | `9ad1b6394b7fa95bcdfd7d183fafc993c58c5b4ccd2649bf990d56253189aad1` |
| 场景计划，规范化 JSON | `75887c8706c17d9d2c8b008fdc9645889e084fbe63a2245f21deafe4899c2959` |
| 既有基线 `model.pt` | `4ad11260f30fb3693042076acbef31dea1c27d7a5a2c8ab79a7885c60196328a` |
| 人工微调候选 `bundle/model.pt` | `441a978d2a1f770a08cc5e5207befc90d1532582ff71f1c52e45ef8ecdc78e4d` |
| 候选 `bundle/metadata.json` | `17303208a994e1dbfb751726dd82892dbeb66f1806a6610a9834a2824846dcfd` |

场景计划哈希按排序键、紧凑分隔符、UTF-8 中文 JSON 计算，不等同于缩进文件的字节哈希。训练内部的 `best.pt` 与导出的 `bundle/model.pt` 也有不同哈希；上表明确记录可供应用加载的导出包。候选包仍保留训练阶段的 `training_and_validation_only` 和 `deferred_not_evaluated` 状态，独立最终比较不会自动晋升该包的验收状态。

本机 `result-summary.json` 记录相关回归为 26 passed / 8 subtests passed、独立标注导出审计通过且权重实际发生变化；这是既有执行证据。本次文档同步重新计算并核对了 manifest、规范化场景计划、两个模型包及其 metadata 的 SHA-256，没有重新训练或运行模型推理。去除私人绝对路径、访问凭证、素材、权重及标注正文的公开摘要见 [人工 ASD 验证摘要](validation-lrasd-human-20261018.json)。
