# 第三方项目、模型与许可证

核查日期：2026-10-22。SceneRecall 自有代码按根目录 [LICENSE](LICENSE) 的 GNU AGPL v3 发布；第三方代码、模型、数据及服务保留各自条款。本文件不改变这些条款，也不把软件许可证扩展为影视素材、角色图片、声音或训练数据的授权。

本仓库通过包管理器使用运行依赖；前端构建会包含 React、React DOM 和 Lucide 的部分代码。以下模型权重、FFmpeg 二进制及 LR-ASD / Animated_AD 外部 checkout **不随本仓库分发**。核查的 Git 跟踪源码中未包含这些项目的模型架构副本；同步适配器从用户提供的本机 checkout 导入模型。许可全文与来源索引见 [licenses/README.md](licenses/README.md)。

## 角色、声音与视觉模型

“模型卡标注”指发布者公开的模型元数据，不代表 SceneRecall 对训练数据权利作出保证。下载时应同时保留所选 revision 的模型卡、许可证及权重来源。

| 项目与准确来源 | 在 SceneRecall 中的用途 | 已核实的许可与范围 |
| --- | --- | --- |
| [DeepGHS CCIP ONNX](https://huggingface.co/deepghs/ccip_onnx) | 动画角色特征及距离模型；支持 `ccip-caformer-24-randaug-pruned`、`ccip-caformer-6-randaug-pruned_fp32`、`ccip-caformer-5_fp32` | 模型卡只标注 **`openrail`**。本次检查模型卡及根文件树未找到明确变体和完整许可文本；不能据此认定为 CreativeML OpenRAIL-M、OpenRAIL++ 或其他特定版本，也不据此宣称可商用或可再分发。需要此类用途时，应先向发布者确认具体条款。调用库 imgutils 的 MIT 许可不等于 CCIP 权重许可。 |
| [DeepGHS anime_person_detection](https://huggingface.co/deepghs/anime_person_detection) | 应用检测器 `person_detect_v1.1_m`；本机训练采集也使用 `person_detect_v1.1_n` | 模型卡标注 **MIT**；模型权重与其训练数据是不同对象。 |
| [DeepGHS anime_face_detection](https://huggingface.co/deepghs/anime_face_detection) | 本机训练准备使用 `face_detect_v1.4_n` | 模型卡标注 **MIT**。它不是 LR-ASD 演示脚本的 S3FD。 |
| [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | 可选本地说话人分段与聚类 | 权重为 **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**，由 pyannote 团队发布；访问文件需接受上游条件。使用或分享受许可材料时保留作者、模型来源和许可链接，并标明修改。`pyannote.audio` 代码的 MIT 许可与此不同。 |
| [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | 可选本地 ECAPA 声纹特征 | 模型卡标注 **Apache-2.0**；训练来源包含 VoxCeleb 1 / 2。SpeechBrain 库也是 Apache-2.0，均不授予用户影视音轨或声音样本的权利。 |
| [google/siglip2-base-patch16-256](https://huggingface.co/google/siglip2-base-patch16-256) | 可选通用视觉语义检索 | 模型卡标注 **Apache-2.0**。 |
| [laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k) | 可选多语言 OpenCLIP 视觉语义检索 | 模型卡标注 **MIT**。 |
| 用户配置的 faster-whisper 本地模型 | 可选 ASR | [faster-whisper 代码](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE)为 MIT；应用不固定某个权重发布者，所选 checkpoint 的模型卡和许可证须单独核对，不能用库许可证替代。 |

## LR-ASD、相关实现与研究来源

| 项目 | 关系及许可依据 |
| --- | --- |
| [LR-ASD](https://github.com/Junhua-Liao/LR-ASD) | 同步检测和本机训练从外部 checkout 加载 `model.Model.ASD_Model`、`lossAV`、`lossV`。文档验证提交为 `1b6dcd2d8fc2895683de6508ec6294ec47d388ca`；[根许可证](https://github.com/Junhua-Liao/LR-ASD/blob/1b6dcd2d8fc2895683de6508ec6294ec47d388ca/LICENSE)为 MIT，Copyright (c) 2025 Liao Junhua，全文保存在 [LR-ASD-MIT.txt](licenses/LR-ASD-MIT.txt)。官方权重为 [pretrain_AVA.model](https://github.com/Junhua-Liao/LR-ASD/blob/1b6dcd2d8fc2895683de6508ec6294ec47d388ca/weight/pretrain_AVA.model) 和 [finetuning_TalkSet.model](https://github.com/Junhua-Liao/LR-ASD/blob/1b6dcd2d8fc2895683de6508ec6294ec47d388ca/weight/finetuning_TalkSet.model)，未发现单列权重许可；这里记录仓库许可，不对 AVA、TalkSet 或原始视频权利作额外推断。 |
| [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) | LR-ASD 的前序工作和上游致谢来源，SceneRecall 不单独导入其模型。MIT，Copyright (c) 2023 Liao Junhua；保留 [许可全文](licenses/Light-ASD-MIT.txt)。 |
| [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) | LR-ASD / Light-ASD 的上游实现来源，未单独捆绑到 SceneRecall。MIT，Copyright (c) 2021 Tao Ruijie；保留原始 `LICENSE.md` 的 [全文](licenses/TalkNet-ASD-MIT.txt)。 |
| LR-ASD 演示代码中的 S3FD | LR-ASD 的 [faceDetector/README.md](https://github.com/Junhua-Liao/LR-ASD/blob/1b6dcd2d8fc2895683de6508ec6294ec47d388ca/model/faceDetector/README.md) 明确说其检测器改编自 [cs-giung/face-detection-pytorch](https://github.com/cs-giung/face-detection-pytorch)。本次在该来源根目录未找到 LICENSE，`sfd_face.pth` 的独立许可也未核实。SceneRecall 的适配器不运行上游完整演示、不导入 S3FD；不能把 LR-ASD 顶层 MIT 自动当作这部分代码和权重的授权。 |
| [Animated_AD](https://github.com/zhrgui/Animated_AD) | 研究参考及可选外部 LWTNet temporal-adapter 桥接。已审阅提交 `e2681a470f79f7157218a10f038441d99dda64b6` 的根目录及音频子目录未发现 LICENSE，本次复核根目录仍未见明确许可；未复制其模型实现，未捆绑权重。启用者须确认其所用代码和权重的授权；`sync_license_acknowledged` 仅记录这一确认，并不是授权书。详见 [研究记录](docs/ANIMATED_AD_REVIEW.md)。 |

感谢 Zhongrui Gui、Junyu Xie、Tengda Han、Weidi Xie、Andrew Zisserman 的 [Character-Centric Understanding of Animated Movies](https://www.robots.ox.ac.uk/~vgg/research/animated_ad/) 研究。其项目网站页脚的 CC BY-SA 4.0 声明指向网站，不能据此推断 Animated_AD 代码、CMD-AM 注释或电影视频均采用同一许可。本机训练记录还引用了 [FYQ12138/condensed-movies](https://huggingface.co/datasets/FYQ12138/condensed-movies) 第三方镜像；该数据集卡未声明许可证。公开视频可下载、镜像存在、论文可引用，都不等于获得素材再分发授权。本仓库不分发这些影视片段或训练权重；训练来源及修改记录保存在 [LR-ASD 训练文档](docs/LRASD_TRAINING.md)。

## 主要软件依赖

下表的版本是本次保留许可文本的包版本；实际安装以锁文件和对应包内声明为准。包内代码未因本项目改动而重新许可。

| 组件 | 用途 | 许可证原文 |
| --- | --- | --- |
| [@aronnaxlin/bgm-cli 1.1.7](https://github.com/aronnaxlin/bgm-cli) | 必需的本地 Bangumi CLI，作者 aronnaxlin；经根目录 npm 依赖安装并作为外部进程调用 | **AGPL-3.0-only**（发行包 `package.json` 与 README），保留包内 [许可全文](licenses/bgm-cli-AGPL-3.0.txt)；这项代码许可与 Bangumi 服务和图片等内容的条款不同 |
| React / React DOM 18.3.1 | 前端界面 | MIT：[React](licenses/React-MIT.txt)、[React DOM](licenses/ReactDOM-MIT.txt)；Copyright (c) Facebook, Inc. and its affiliates |
| Lucide React 0.468.0 | 界面图标 | [ISC](licenses/Lucide-ISC.txt)，含 Lucide Contributors 与 Feather / Cole Bemis 归属；另保留 [Feather MIT](licenses/Feather-MIT.txt) |
| FastAPI 0.141.1 | HTTP API | [MIT](licenses/FastAPI-MIT.txt) |
| Uvicorn 0.53.0 | HTTP 服务 | [BSD-3-Clause](licenses/Uvicorn-BSD-3-Clause.txt) |
| Pydantic 2.13.5 | 数据校验 | [MIT](licenses/Pydantic-MIT.txt) |
| HTTPX 0.28.1 | HTTP 客户端 | [BSD-3-Clause](licenses/HTTPX-BSD-3-Clause.txt) |
| python-multipart 0.0.32 | 上传表单解析 | [Apache-2.0](licenses/python-multipart-Apache-2.0.txt) |
| keyring 25.7.0 | 系统凭据存储 | [MIT](licenses/keyring-MIT.txt) |
| Pillow 12.3.0 | 图片读取与裁剪 | [MIT-CMU / PIL 与 Pillow 声明](licenses/Pillow-MIT-CMU.txt) |
| PySceneDetect 0.7.1 | 镜头检测 | [BSD-3-Clause](licenses/PySceneDetect-BSD-3-Clause.txt) |
| [PyAV 18.1.0](https://github.com/PyAV-Org/PyAV) | 关键帧扫描的原生 PTS 解码与 libavfilter 图像转换（2026-10-23 新增核查） | [BSD-3-Clause 原文](licenses/PyAV-BSD-3-Clause.txt)；wheel 携带或链接的 FFmpeg 库保留其各自许可，不能用 PyAV 的代码许可替代 |
| LanceDB 0.39.0 | 本地检索索引 | [Apache-2.0](licenses/LanceDB-Apache-2.0.txt) |
| jieba 0.42.1 | 中文分词 | [MIT](licenses/jieba-MIT.txt) |
| pysubs2 1.9.0 | 字幕解析 | [MIT](licenses/pysubs2-MIT.txt) |
| psutil 7.2.2 | 进程与资源信息 | [BSD-3-Clause](licenses/psutil-BSD-3-Clause.txt) |
| opencc-python-reimplemented 0.1.7 | 中文简繁归一化 | [Apache-2.0](licenses/OpenCC-Python-Apache-2.0.txt)，保留包含 OpenCC 字典来源的 [NOTICE](licenses/OpenCC-Python-NOTICE.txt) |
| pykakasi 2.3.0 | 日文读音归一化 | **GPL-3.0-or-later**（包元数据），保留 [GPL v3 全文](licenses/pykakasi-GPL-3.0.txt)；这是 copyleft 依赖，不能作为 MIT 依赖分发 |

可选计算环境另使用以下上游库，均经包管理器安装，未把其实现复制到本项目源码：

| 组件 | 上游代码许可 |
| --- | --- |
| [dghs-imgutils](https://github.com/deepghs/imgutils) | [MIT 原文副本](licenses/imgutils-MIT.txt)，Copyright (c) 2023 DeepGHS |
| PyTorch / torchaudio / TorchCodec | 分别为 [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE)、[BSD-2-Clause](https://github.com/pytorch/audio/blob/main/LICENSE)、[BSD-3-Clause](https://github.com/pytorch/torchcodec/blob/main/LICENSE)；发行包内附带组件另有声明 |
| pyannote.audio | [MIT](https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE)，Copyright (c) 2020 CNRS |
| SpeechBrain | [Apache-2.0](https://github.com/speechbrain/speechbrain/blob/develop/LICENSE) |
| Transformers | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| OpenCLIP | [MIT](https://github.com/mlfoundations/open_clip/blob/main/LICENSE) |
| faster-whisper | [MIT](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE)，Copyright (c) 2023 SYSTRAN |

FFmpeg / ffprobe 由用户安装后作为外部进程运行。其[上游许可说明](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)区分 LGPL-2.1-or-later、启用 GPL 组件及其他构建情况；实际二进制应查供应方声明、`ffmpeg -L` 和构建参数，不能统一视作某一种宽松许可。

## 分发与保留说明

- MIT、ISC、BSD 组件应保留相应版权、许可条件和免责声明；BSD-3-Clause 还包含不得用作者名称背书的条件。Apache-2.0 组件应保留许可证及适用 NOTICE，并注明修改。
- 若分发包含 pykakasi 等 GPL 组件的版本，须满足适用的许可证、对应源代码等义务；bgm-cli 的 AGPL-3.0-only 与本项目的 AGPL 条款同样继续适用，包括各自适用的网络交互要求。不能仅附本文件就视为完成所有分发义务。
- 此文件覆盖主要直接运行依赖和明确使用、研究或引用的模型，不是所有平台 wheel、GPU 运行库及传递依赖的完整清单。打包安装环境或二进制时，应保留实际随包第三方 LICENSE / NOTICE，并按实际版本补充清单。
- Bangumi、在线模型 API 与 Codex 等为外部服务；其服务条款以及返回图片、文本、音频的权利与软件许可证分开处理。角色档案或素材来源链接不改变原权利人的权利。
