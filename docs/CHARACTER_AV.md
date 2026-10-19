# 角色中心的视听理解

本次是在原有 FastAPI/React、场景队列、JSON/JSONL 证据和可重建索引上增量改造。模型功能默认关闭；没有本地依赖或权重时明确显示未启用/不可用。下文的初次工程验收不代表动画人物准确率、DER 或真实提速；后续 LR-ASD 真实训练、短片段预览及人工标注微调记录见 [LR-ASD 实验](LRASD_TRAINING.md)。2026-10-18 人工微调候选的留出指标下降，继续保留既有模型。

## 先核实，再修改

修改前提交：`3728bb241c7f913000699b50e15596a360edfe9f`。

| 疑点 | 实际检查结果 | 本轮处理 |
| --- | --- | --- |
| 首次没有图片参考 | `jobs.character_context` 只读已关联 `representative`；外部档案的 URL 没有进入图像输入 | 新增独立参考图库，支持本机上传、已缓存外部图、原片裁剪；从零建档可先人工建立姓名 |
| 16 个角色截断 | `CharacterStore.context` 没有 16 限制，但 `scene_jobs._bounded_scene_profiles` 按更新时间截取 16，参考图又取最近 8 个 | 完整允许图库检索后按每个区域 Top-K 选择；无匹配器时压缩全部名单文字，不按更新时间排除角色 |
| 身份不明反复深入 | 分块确实使用 320×180 预览，prompt 将身份不明列入 `needs_deep` | 原图检测裁剪 + 清晰区域补充，独立 `needs_identity_review` / `needs_action_deep`，保留分块拼图用途 |
| 已关联却不参与检索 | 旧 `decorate_observations` / `Library.records` 已投影角色名；不是完全未关联 | 轻量分块新增 events，actor/target 引用本地实体；读取时投影参与者姓名；声音独立 `kind=audio` |

## 安装与运行

基础环境仍按 README 安装。以下命令会下载 Python 软件依赖，**不会下载模型权重**。建议 Python 3.12。imgutils 0.19 要求 NumPy 1.x，而 pyannote 4 要求 NumPy 2.x，不能把两个 extras 装在一个环境中。锁文件明确声明互斥，视觉使用独立常驻子进程。

```sh
# 主应用与音频环境
uv sync --python 3.12 --extra dev --extra audio
# 可选：无可靠文字字幕时本地 ASR
uv sync --python 3.12 --extra dev --extra audio --extra asr

# 独立视觉 worker；保留主 .venv
UV_PROJECT_ENVIRONMENT=.venv-visual uv sync --python 3.12 --extra visual-identity

npm ci --ignore-scripts
npm --prefix frontend ci
npm --prefix frontend run build
SCENERECALL_VISUAL_PYTHON="$PWD/.venv-visual/bin/python" \
  .venv/bin/scenerecall --data-dir "$HOME/SceneRecallLibrary" --port 8765
```

Torch/Torchaudio 固定在 2.8，TorchCodec 固定在相应的 0.7 系列；依据 [TorchCodec 官方兼容表](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)。FFmpeg/ffprobe 必须可用。CPU 可运行本地模型，但本轮未测算模型 RAM、GPU 显存或吞吐；CUDA/MPS 支持取决于对应适配器和实际模型。Animated_AD 同步 scorer 仅提供 CPU/CUDA 路径。

先自行核对许可证、取得访问权限并下载权重，然后在资产详情的“视听身份”中配置本机目录：

- [imgutils / CCIP](https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html)：`dghs-imgutils` 软件使用 MIT，实际检测与 CCIP 权重还需核对各模型卡。worker 强制离线，需预缓存 imgutils 使用的检测和 CCIP 模型及度量文件。CCIP 要求单角色输入，不能拿整张多人图片直接提特征。模型默认 `ccip-caformer-24-randaug-pruned`。
- [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)：接受模型访问条款后取得 Hugging Face 访问令牌，自行下载完整模型目录（包括本地 segmentation、embedding、PLDA 依赖）。令牌不保存在 SceneRecall。使用普通 speaker_diarization，保留多人重叠；不使用排他分段输出抹去重叠。
- [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)：下载完整本机 snapshot，至少含 `hyperparams.yaml`、`embedding_model.ckpt`、`mean_var_norm_emb.ckpt`、`classifier.ckpt`、`label_encoder.txt`。模型卡标注 Apache-2.0；VoxCeleb 真人声音训练并不证明动画角色识别效果。
- 可选 Faster Whisper：提供本地模型目录；复用可靠文字字幕优先。译文只按时间交集引用，不做跨语言强制音素对齐，也不把一个字幕段当成一个说话人。
- Animated_AD：见[实际代码核查](ANIMATED_AD_REVIEW.md)与[音频说明](AUDIO_IDENTITY.md)。没有公开可核实的动画 temporal-adapter 权重时保持关闭。不能仅勾选“动画已验证”就声称模型效果已验证。

为避免意外长时间占用，独立音频试跑默认限制 60 秒；按需明确提高本地时长预算。主场景任务冻结该资产当时的身份配置；改变设置后创建新任务，原图像/音频特征缓存仍可复用。旧任务保留旧提示词与成功回执。

## 使用顺序

1. 资产详情 → 角色档案。Bangumi 档案使用当前作品允许名单；从零建档可先“新建角色档案”。打开“参考图库与声音样本”，上传已有图片或有权使用的音频。外部图片只作为身份参考，绝不成为原片出场证据。
2. 人工核对参考图只含一人、身份正确。清晰度/尺寸质量不合格或混入多人不能用于可信匹配。参考状态为 candidate / confirmed / revoked，记录来源和版本。机器未确认预测不会自动加入可信图库。
3. 在“视听身份”保存视觉设置；对已扫描原片帧做区域识别，查看 P1/P2 候选及距离/差距。每个区域绑定原帧时间、bbox 与独立 ID；同为 P1 或相邻画面不意味着同一角色。
4. 选择音轨、语言、配音版本和模型目录，保存后试跑短片段。试听原音频，核对字幕时间关联、重叠状态及候选，再人工确认或撤销。不同文件的 speaker_id 有各自作用域；同一声优不直接等于同一角色。
5. 运行原来的场景分析。人物与声音处理先于视觉剧情解释；分块/深入模型收到当前区域映射和带 speaker_id 的声音记录。音频可证明“听到谁”，不能单独证明人物可见、动作、口型或关系。

角色改名、别名更新和区域/说话人纠错只修改映射与读取投影，保留原始模型 prose、JSON/JSONL 与审计。新场景用独立语义版本避免纯改名使所有剧情失效；旧版已冻结任务仍按原 revision 检查，修改运行中任务的角色档案会拒绝过期提交。引用了被纠正身份的剧情推断及其实际后序依赖会标记待复核，纯动作映射不因此重跑。改变外观语义、合并/删除角色仍可能使派生剧情需要复核；不会自动重跑整集。

## 证据与恢复

图像参考存于 `works/<scope>/reference-images/`；检测、区域和人工决定存于 `assets/<asset>/identity/visual/`。图像特征按内容、模型和预处理版本缓存，匹配缓存另含参考图版本与校准参数。声音原始记录、PCM、分段、模型/时间证据存于 `assets/<asset>/audio/`；可信声音库按作品、语言、配音版本、角色隔离，撤销样本不会删除历史记录。任务内 `identity-prelude.json` 冻结已经完成的两条身份链路；恢复不重复提特征。后续模型请求重新投影当前身份，失败请求的旧输入留存于审计目录，已成功的付费回执继续复用。

重型推理共享 `local_resources.LOCAL_MODEL_GATE`，解码和独立观察可并行，剧情整理继续保持原有顺序。阶段计时包含本地音频处理与模型加载；远程视觉仍使用既有累计预算、成功回执、取消和二分回退。特征缓存可删除重建，JSON/JSONL 与人工决定是可核对的来源。索引重建包含独立声音记录，不把旁白虚构成当前画面人物。

## 验收边界

工程测试覆盖首次无参考、超过 16 角色、多人物区域、候选/未知、换装侧脸候选保守处理、原片时间偏移、无音轨、重叠、画外音、配音隔离、跨语言字幕、样本撤销和恢复复用。合成图片/特征及替身分段模型只能证明约束与流程，不能证明真实侧脸、换装、唱歌或抢话识别质量。

音画同步有真实本地 LWTNet/temporal-adapter 调用桥，使用同时间范围连续 25 fps 帧与音频，拒绝稀疏关键帧拼图。但本轮缺少授权权重与已确认连续人物轨迹，未做模型推理验收；目前通过显式短片段复核 API 提交轨迹，不自动从稀疏关键帧臆造全片轨迹。同步分数仍作为待复核证据，不直接污染声音库。

三方案同素材基准见 `scripts/benchmark_identity.py` 和 [验证记录](VALIDATION.md)。未提供已授权真实动画、标签与本机权重，因此真实冷/热耗时、首次检索耗时、调用/回退差异、角色错误率、台词归属准确率和 DER 均尚未测得。没有对本轮声称 ≥50% 提速；README 中历史实验属于旧流程的独立测量。
