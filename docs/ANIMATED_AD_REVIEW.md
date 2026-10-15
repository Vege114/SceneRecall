# Animated_AD 官方代码核实记录

核实日期：2026-10-15。仓库：[zhrgui/Animated_AD](https://github.com/zhrgui/Animated_AD)，本地审阅提交 `e2681a470f79f7157218a10f038441d99dda64b6`。本次仅下载公开源码用于阅读，没有下载数据集或模型、没有训练，也没有复制第三方实现到 SceneRecall。

已实际阅读根 README、`character_recognition/README.md`、`character_recognition/audio_recognition/README.md`，以及下面的实现：

| 官方文件 | 核实到的行为 | SceneRecall 的适配 |
| --- | --- | --- |
| [audio_recognition.py](https://github.com/zhrgui/Animated_AD/blob/e2681a470f79f7157218a10f038441d99dda64b6/character_recognition/audio_recognition/audio_recognition.py) | 输入 `audio_annotation_file` 的预标注时间段；SpeechBrain ECAPA 提特征并匹配多声音样本；有聚类分支和固定 0.25 相似度判据；会补入声优声音库。 | 不依赖评测时间段。community-1 自动 VAD/分段/聚类；作品、语言、配音版本、角色隔离。不沿用固定阈值，也不把声优当角色。 |
| [preprocess/build_voice_examples.py](https://github.com/zhrgui/Animated_AD/blob/e2681a470f79f7157218a10f038441d99dda64b6/character_recognition/audio_recognition/preprocess/build_voice_examples.py) | 从视觉轨迹挑高分样本，样本不足时补入低分轨迹；包含特定角色的例外。 | 视觉轨迹只能提议。原片声音样本经人确认后才可信；不为凑数量降低质量，也不依据“单人画面有人说话”确认说话者。 |
| [visual_enhancement.py](https://github.com/zhrgui/Animated_AD/blob/e2681a470f79f7157218a10f038441d99dda64b6/character_recognition/audio_recognition/visual_enhancement.py) | 加载 `LWTNet_temporal_adapter`，截取候选轨迹，调用同步 scorer。原脚本含数据集目录布局和 FPS 换算。 | 短歧义片段按源 PTS 重采样为连续 25 fps，用同源 16 kHz 音频及稠密人物框调用真实 scorer。不能用关键帧拼图代替同步输入。 |
| [visual_enhancement_postprocessing.py](https://github.com/zhrgui/Animated_AD/blob/e2681a470f79f7157218a10f038441d99dda64b6/character_recognition/audio_recognition/visual_enhancement_postprocessing.py) | 用同步分数、视觉分类分数及外部 `alpha` 组合更新弱声音预测。 | 各分数分开保存；不照搬乘积分数为置信度，不自动覆盖人工判断。无动画校准时只提供待核对同步分数。 |
| [model/checkpoint.py](https://github.com/zhrgui/Animated_AD/blob/e2681a470f79f7157218a10f038441d99dda64b6/character_recognition/audio_recognition/model/checkpoint.py) | 宽松权重加载允许缺失层保留初始参数，适于微调初始化。 | 推理适配器使用安全 `weights_only=True` 和严格 `load_state_dict`；必须包含两路 temporal adapter，拒绝随机初始化层冒充有效模型。 |
| [model/lwtnet.py](https://github.com/zhrgui/Animated_AD/blob/e2681a470f79f7157218a10f038441d99dda64b6/character_recognition/audio_recognition/model/lwtnet.py)、[model/sync_scorer.py](https://github.com/zhrgui/Animated_AD/blob/e2681a470f79f7157218a10f038441d99dda64b6/character_recognition/audio_recognition/model/sync_scorer.py)、[data/audio.py](https://github.com/zhrgui/Animated_AD/blob/e2681a470f79f7157218a10f038441d99dda64b6/character_recognition/audio_recognition/data/audio.py)、[data/datasets.py](https://github.com/zhrgui/Animated_AD/blob/e2681a470f79f7157218a10f038441d99dda64b6/character_recognition/audio_recognition/data/datasets.py) | 15 视频帧/60 mel 步窗口；25 fps、16 kHz mono、RGB float32 的 0..255 输入，音频前端接收 PCM16 数值尺度。 | ECAPA 用 [-1,1]，同步模型单独保留原 PCM16 数值尺度；版本化两套预处理，不能共用错误归一化。 |

官方同步 README 明确指出基础 LWTNet 针对真人训练，需要动画微调；旧版按 0..1 微调的权重与当前 0..255 输入不匹配。读取的 `config/finetune.yaml` 是真人数据配置；README 引用的 `config/finetune_animated.yaml` 在该提交中不存在。仓库忽略 `.pt`/`.pth` 和 checkpoints，未发现可公开下载的动画 temporal-adapter 权重链接。因此本轮同步模块默认**未启用、效果未验证**，不会把真人基础权重输出当完成的动画验证。

该提交根目录及音频子目录未发现 LICENSE 文件。SceneRecall 不捆绑或改写其模型代码；只提供独立进程调用本机外部 checkout 的适配器。启用者须自行确认代码和权重使用权。`sync_license_acknowledged` 记录这一配置，不代表 SceneRecall 授予第三方许可。

community-1 的[模型卡](https://huggingface.co/pyannote/speaker-diarization-community-1)标明 16 kHz 单声道输入、重叠说话人输出及本机离线使用方式，权重为 CC-BY-4.0，并需要接受访问条件。使用普通 `speaker_diarization` 输出，不用会消除重叠的 exclusive 版本。

SpeechBrain ECAPA 的[模型卡](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)标明 Apache-2.0、VoxCeleb 训练及 16 kHz 单声道约束；`encode_batch` 调用者须自行满足采样要求。其真人基准不能作为动画角色准确率。SceneRecall 对原片和声音库统一提取 PCM 后调用该接口，模型常驻、特征缓存，阈值必须另用动画标注集校准。
