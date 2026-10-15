# 本机音频角色识别

此模块增量接入既有场景队列，也可在资产详情单独处理短片段。默认关闭，不上传媒体、不自动下载权重、不启动训练、不调用付费 ASR。关闭、没有音轨、模型不可用和处理时长超预算均返回明确状态；视觉旧流程可继续。

## 运行准备

音频模型使用独立 Python 环境或安装 optional extras，避免强制所有用户安装 PyTorch：

```sh
uv sync --extra audio --extra asr --extra dev
uv run scenerecall --data-dir "$HOME/SceneRecallLibrary"
```

需要 FFmpeg/ffprobe。`audio` 包含 pyannote.audio 4.x、SpeechBrain、PyTorch/torchaudio；`asr` 是可选 faster-whisper。若当前前端/后端环境安装版本与重型包不兼容，请使用 README 的音频独立环境安装方式。没有依赖也可以正常启动应用和查看已有记录。音频与视觉本地模型通过共享 `LOCAL_MODEL_GATE` 控制竞争；CPU 默认可运行，CUDA 需兼容驱动，MPS 的具体模型支持尚未验证。同步上游 scorer 当前仅支持 CPU/CUDA。模型内存和实际耗时由素材、硬件、权重决定，本次没有测得动画实际吞吐。

手动获取以下模型，再在资产“音频角色识别”的高级设置填写**完整本机目录**：

1. [community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)：先由用户接受访问条件并使用自己的 HF 访问令牌下载完整离线目录；权重 CC-BY-4.0。访问令牌不写入 SceneRecall 配置、API 或日志。保留 config、segmentation、embedding、PLDA 等子文件。运行时检查本机引用，不用在线模型 ID 代替本地路径；禁用 pyannote 使用统计。
2. [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)：Apache-2.0。完整目录包括 `hyperparams.yaml`、`embedding_model.ckpt`、`mean_var_norm_emb.ckpt`、`classifier.ckpt` 和 `label_encoder.txt`。适配器覆盖 YAML 的 `pretrained_path` 为当前目录，不沿用远程下载位置。
3. 可选 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)：本机 CTranslate2 格式 Whisper 权重，设置 `asr_enabled=true` 和 `asr_model_path`。代码和模型各自遵循其许可证。本机模型调用启用 `local_files_only`。有可靠文字字幕时跳过 ASR；翻译字幕只按时间相交关联，不做跨语言强制发音对齐。

配置字段示例（路径仅为用户自行安装后的示例）：

```json
{
  "enabled": true,
  "stream_index": 1,
  "language": "ja",
  "dub_version": "Japanese-original-TV",
  "diarization_model_path": "/absolute/models/community-1",
  "embedding_model_path": "/absolute/models/ecapa",
  "asr_enabled": false,
  "device": "cpu",
  "max_duration_ms": 60000
}
```

音轨使用 FFmpeg **全局 stream index**。默认选 default 音轨，再回退第一条。`language` 是原音语言，`dub_version` 是用户明确的配音版本，不能使用字幕翻译语言替代。默认本地处理预算为 60 秒；整集必须明确提高 `max_duration_ms`。超预算返回 `budget_exceeded`，不会截断后宣称已完整识别。音频不是远程请求，其时长预算与现有视觉请求/费用预算分别管理。

## 处理与证据

流程为选轨 → 以原视频播放原点归一 PTS → FFmpeg 提取 16 kHz mono PCM16 → 自动 VAD/说话人分段聚类 → ECAPA 片段特征 → 当前作品的完整可信声音库 → 字幕时间关联（必要时 ASR）→ 场景 `speaker_dialogue`。

先保留音轨真实起点和静音缺口，再裁剪。不能把第一包音频强制设成片段零点；重采样后的样本数必须和处理时间范围相同。所有时间为原片毫秒，音轨索引、源引用、范围、模型及预处理版本都持久化。单条台词可与多个 speaker 段关联；抢话仍保留多个重叠记录，混音片段不会自动确认或直接进入可信声音库。

`speaker_id` 带 asset、音轨/范围提取身份和分段模型作用域，不能跨文件、音轨或重新分段直接合并。音频确认只表示听到该角色，不证明角色可见、动作、口型、关系或叙事推断。`frame_ids` 是相关时间的帧引用，`evidence_frame_ids=[]` 与 `visual_evidence=false` 保持这种区别。

原始记录包含 `audio_stream_index/start_ms/end_ms/speaker_id/character_id/candidate_character_id/association_status/association_source/audio_ref/subtitle_ids/frame_ids/model_versions/overlap` 等字段。听到的证据与字幕翻译的完整词句不混为一谈，`subtitle_association=time_overlap_only`，多 speaker 覆盖同字幕时 `dialogue_attribution=ambiguous`。没有可信转写则保留空文本和降级原因。

时间统计 `timings` 的字段均以 `_s` 结尾，单位秒，包含提取、分段、转写、特征、匹配和总耗时；模型首次加载发生在对应阶段，计入总耗时。`cache_hits` 为命中次数，不是声称性能提升的比例。

## 样本与人工纠正

角色声音库在 `works/<scope>/voice-samples.jsonl`，音频在同目录 `voice-audio/`。样本必须注明角色、语言、配音版本、来源、质量、版本和确认来源。支持原片音频记录和有权使用的外部 WAV；当前外部上传要求 16 kHz mono PCM16，大小不超过 10 MB。质量过滤排除小于一秒、基本静音和严重削波样本；这不代替听辨背景音乐、唱歌或混入别人。

`proposed` 不参与匹配，只有明确人工确认的 `confirmed` 样本进入可信库。`revoked` 立即退出当前匹配投影，旧样本与审计保留。视觉已确认人物与语音同时出现仅可构成提议，不能自动证明其说话；目前通过原片记录创建未确认样本，并显示相关帧供核对。

多参考样本按每个角色最近的最多三项距离聚合，候选在完整当前作品库中检索，再取 Top-K；不同语言/配音版本绝不交叉使用。不存在“同声优自动扩充角色样本”的逻辑。相似度/距离、候选差距和身份状态分别保存。没有校准文件时，非空声音库最多给出 `candidate`，不会因分数看起来高就自动确认。

可手动提供校准 JSON，设置 `calibration_path`。文件须包含 `id/evaluation_set_id/model_version/scope_id/language/dub_version/candidate_max_distance/confirmed_max_distance/min_margin`，模型和作品范围必须匹配。确认阈值须严于候选阈值；单一候选库缺少真实竞争差距时不自动确认。本次未提供动画标注校准，因此默认不发布自动确认阈值。

人工修改单段身份写入 `audio/identity-overrides.json` 及 `identity-audit.jsonl`，不替换原始描述或文本。当前档案名字在读取/索引时投影。改名不重提声纹；撤销样本只改变映射，昂贵特征仍缓存。

合并角色时通过角色档案重定向投影既有样本和人工台词关联，保留原 ID 及审计。删除角色后其样本退出可信库，历史文件仍可备份恢复。人工身份实际变化只使明确引用该台词且包含身份推断的派生产物失效，名字变动不触发剧情重算。

## 缓存、续跑、取消与接口

素材/音轨/范围/源 PTS/预处理版本决定 PCM 缓存；分段模型和特征模型各自有版本键，声音特征还包含音频内容 SHA256。样本改名不改变缓存，换模型、范围、配音版本和撤销信任不会错误复用身份结论。每个完成特征立即原子写盘，取消后续跑复用已完成分段及特征。JSON/JSONL 是事实来源，搜索可重建。

记录 ID 基于原始声音证据及分段身份；仅调整预算、Top-K 或同步设置不丢失人工确认。校准文件在明确执行分析时核实并保存为运行内快照，后续读取不依赖外部配置文件仍然存在。重跑局部区间时，跨边界的旧段投影为可追溯的剩余片段，保留原记录 ID 引用及人工映射，试听按原 PCM 精确裁剪；未重跑区间不会从搜索中消失。恢复备份时检查音频文件路径、作用域、范围、原始记录及样本内容，拒绝跨目录文件引用。

独立分析可用 `POST /api/assets/{id}/audio/cancel` 请求取消；FFmpeg 提取期间和每阶段/片段检查取消。模型正在计算时等该次本机调用完成再停，不强杀共享 GPU。主场景任务复用原队列 checkpoint，可暂停/恢复/取消。删除资产会在下一检查点停止。音频特征和记录在 asset 目录内，遵循已有资产删除/回收逻辑；作品共享声音样本保留独立来源审计。

HTTP 请求任务被取消时先通知并等待其后台工作线程停止，再释放该资产的分析槽，避免遗留线程继续写入或与下次请求冲突。

接口：

| 方法 | 路径 | 内容 |
| --- | --- | --- |
| GET / PUT | `/api/assets/{id}/audio/settings` | 配置及依赖/权重可用性；可用性不等于效果验证 |
| GET | `/api/assets/{id}/audio/tracks` | 本机音轨及推荐全局索引 |
| POST | `/api/assets/{id}/audio/analyze` | `{start_ms,end_ms}` 本地同步处理，可断点重试 |
| POST | `/api/assets/{id}/audio/cancel` | 请求独立音频任务在检查点停止 |
| GET | `/api/assets/{id}/audio/records` | 当前身份投影、耗时及降级原因 |
| POST | `/api/assets/{id}/audio/records/{record}/confirm` | `{character_id:null或ID,note}` 人工覆盖及撤销 |
| GET | `/api/assets/{id}/audio/records/{record}/clip` | 本机 WAV 试听 |
| GET / POST | `/api/assets/{id}/characters/{character}/voice-samples` | 声音库及新增样本 |
| PATCH / DELETE | `/api/assets/{id}/characters/{character}/voice-samples/{sample}` | 确认、提议或撤销（删除保留审计） |
| GET | `/api/assets/{id}/characters/{character}/voice-samples/{sample}/audio` | 声音样本试听 |
| POST | `/api/assets/{id}/audio/records/{record}/sync-review` | 配置完整时调用连续人物轨迹的同步复核 |

接口使用既有本机会话保护；音频、路径不发送给远程模型。读取记录不会偷偷加载重型模型；加入新声音样本后重新执行短片段任务，只补新样本特征并更新身份，不重复计算已有原片特征。

## 音画同步复核边界

详见 [官方代码核实](ANIMATED_AD_REVIEW.md)。设置本机 `sync_repository_path`、`sync_checkpoint_path`，并明确确认 `sync_license_acknowledged` 与 `sync_animation_verified` 后才可启用。缺少动画完整权重、预处理未经验证或许可未确认时返回未启用原因。不要把这些标志置为 true 来绕过缺失验证。

实际适配器为 `av_sync.py` / `av_sync_worker.py`，子进程 import 本机 Animated_AD 的 LWTNet temporal adapter 和 scorer。严格要求两路 adapter 参数完整，使用 `torch.load(weights_only=True)`、严格权重形状校验、真实连续视频帧和同源音频。输入 `tracks` 每项包含 `id/character_id/association_status=confirmed/boxes:[{at_ms,box:[x,y,w,h]}]`，要求稠密时间轨迹覆盖片段；关键帧列表不符合要求。每次最多八个候选、八秒区间，只有歧义或冲突身份才复核。

同步分数单独保存为待核对证据，不自动改变人物动作事实、人工身份或声音库。上游运行所需额外包（例如 librosa、decord、torchvision）需按其代码环境另行安装；本次不把无明确许可证的上游实现打包到应用。没有权重、没有稠密真实人物轨迹时，端到端同步精度仍**未验证**。

## 本轮验证及未验证项

`uv run pytest tests/test_audio_identity.py tests/test_audio_identity_api.py` 覆盖空声音库、未知身份、多个说话人/重叠、不同配音版本、画外音不成为可见证据、跨语言字幕、字幕跨 speaker、无音轨、ASR 条件调用、外部样本撤销、人工覆盖与改名/合并/删除、取消/续跑缓存、局部重跑边界、备份路径校验和同步缺权重/稀疏帧拒绝。

FFmpeg 测试实际生成 48 kHz、延迟 500 ms 的音轨与视频容器，裁剪从原片 250 ms 开始并重采样至 16 kHz；验证前段应有静音、裁剪后样本数以及音轨索引拒绝。该测试验证时间工程逻辑，不是语音识别效果。其余模型调用使用明确的测试 doubles，不能当作动画识别成功。

本轮未获得真实动画语音标注集、可用 community-1 / ECAPA 权重及动画 LWTNet 权重，未执行真实自动分段、声纹识别、ASR 或同步推理。因此不报告虚构准确率、DER 或加速倍数。唱歌/背景乐检测、音源分离、端到端稠密人物跟踪和自动可靠唇音验证仍未验收；这些情况应保留未知/候选并人工核对。
