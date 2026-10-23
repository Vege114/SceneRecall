# 检索优先实验策略：retrieval-first-v1

2026-10-20 起，任务入口更新于 2026-10-23。本文区分已实现的工程行为与待完成的质量、性能验收。**应用初始默认仍为 `scene-chunks-v1`；P4 验收通过前不改为检索策略。** 用户可以在共享任务配置中保存自己的默认策略。初始工程验证使用隔离资料库；后续按用户明确要求，通过 GUI 为《少女歌剧》第 1、2 集建立全片无 VLM 核心检索索引。已有历史报告保持不变。

## 入口与完成契约

已有原片与可用文字字幕 → 本地镜头和短窗口 → 直接图像特征、字幕索引 → 有序局部观察与身份决策（complete）→ 混合检索 → 原片时间与原帧证据。

通过已登录的 `POST /api/jobs` 创建任务：

```json
{
  "asset_id": "替换为已导入资产ID",
  "task_pipeline": "unified-v1",
  "analysis_strategy": "retrieval-first-v1",
  "retrieval_mode": "core",
  "start_ms": 0,
  "end_ms": null,
  "stages": ["vision"],
  "max_requests": 100,
  "max_cost": null
}
```

| 模式 | 必需工作 | 完成语义 |
| --- | --- | --- |
| `core` | 本地采样、直接视觉索引、已有字幕索引、独立本地人物匹配（启用时）、最终发布 | 所请求核心工作完成；任务可为 `completed`，但 `retrieval.status=core_ready`，`default_completed_at=null` |
| `complete`（REST 字段默认值） | core 全部工作，加全部窗口的局部人物／物体／地点／事件记录、身份决策与最终发布 | 所有默认 capability 均 `completed` 才写 `retrieval.status=completed` 和 `default_completed_at` |

GUI 选择“画面检索 · 可选 VLM”后，“使用 VLM 生成局部描述与事件”的初始默认不勾选，之后使用用户保存的默认值。未勾选提交 `core`，无需 vision 连接，不提交模型绑定／覆盖或费用预算，不调用生成式 VLM。新任务冻结 `retrieval_contract_version=2`，关闭 VLM 不再跳过独立的本地人物检测、CCIP 和身份投影；是否启用本地识别由“视听身份”中的独立设置控制。勾选 VLM 后提交 `complete`，要求选择有 `vision` 能力的模型连接。默认范围为全片，批量每集使用自身完整时长；“先试运行前 60 秒”可以显式缩小范围。两种模式都保留本地图文编码器。

单集／批量提交与“模型与设置 → 默认任务配置”共用同一组件，集中设置策略、VLM、连接／模型、执行方式和预算。本地关键帧、人物检测／CCIP、当前图文编码器和字幕索引显示为锁定选项，沿用当前实现。提交前无需另行扫描关键帧或打开角色选择题；点击“开始任务”即创建任务。新 GUI 提交 `task_pipeline:"unified-v1"`，任务内依次管理关键帧、角色确认、分析处理；有待核对角色或采样仍在运行时进入 `waiting_input`，在任务队列“确认角色并继续”完成确认或明确跳过，再继续本地索引及所选 VLM 阶段。详见 [任务使用说明](TASK_WORKFLOW.md)。省略统一流程标记的旧接口和历史任务继续使用原有生命周期。

旧 contract 1 core 的回放仍不执行人物匹配，不改变旧任务配置和 `identity=not_requested` 的含义。已发布画面索引可在详情页点击“补充人物匹配”，调用 `POST /api/assets/{asset_id}/retrieval/identity`、body `{}` 创建独立本地任务；复用原帧及检测／特征缓存，不重新扫描、调用通用图像编码器或生成式 VLM，也不改旧任务。该入口要求已启用本地识别并存在画面窗口；同集活跃补匹配任务重复提交返回同一任务，其他活跃任务会阻止并发修改。原片暂时离线时仍可使用已登记的原帧。

统一任务通过 `model_bindings.vision` 选择已有 API 或 Codex 连接；未指定时继承全局 vision 分配。Codex 模型和思考等级继承所选连接／模型默认值，也可用 `model_overrides.vision` 显式选择，任务冻结实际结果，不暗中改用其他模型。省略 `task_pipeline` 的旧检索请求保留 `gpt-6.1-sol`、`low` 轻量预设及显式覆盖规则。完整模式已完成一次真实 6 秒功能 smoke，包含预算暂停、超时及重试，**性能未通过，不进入 complete 整集验证**。全片 core 试用不代表 complete 的实时或质量门槛通过。

`retrieval` 持久状态包含 `status`、`capabilities`、`coverage`、`first_searchable_at`、`default_completed_at`。capability 为 `sampling / visual_index / subtitle_index / local_records / identity / final_publish`。coverage 同时报告选定时间范围、总窗口数、视觉已发布数和已生成局部记录数。`first_searchable_at` 表示至少一个完整视觉窗口已发布；字幕另有 `subtitle_searchable_at`。没有字幕也会完成“检查及索引可用字幕”工作，但 `subtitles_available=false`，不会虚构字幕覆盖。

这里“身份决策完成”包括正确保留候选或未知，并不表示每个人都已实名，更不代替实名质量验收。contract 2 的人物工作产生独立不可变附属观察，按当前区域决定、角色名和别名动态投影；不凭候选名字拼接跨帧人物轨迹。候选姓名可召回并明确展示为候选，不能进入已确认角色过滤，也不能被文本判定或视觉复核升级成完整实名匹配。人工姓名纠错须唯一对应作品角色，保存正式区域决定；后续改名、撤销或重绑不会被旧标注复活。启用的本地身份失败时 core／complete 均不能成功，已发布视觉索引保留；独立关闭时显示 `disabled` 及原因，不能写成已识别。contract 2 core 的 `processed_units` 统计本次核心工作已处理窗口，旧 contract 1 计数口径不变。

任务队列的批次摘要按各集实际能力状态显示本地人物匹配已完成、已启用或已关闭；关闭 VLM 不等于人物匹配关闭。旧 contract 1 仍保留其“未启用局部事件与身份”的原始含义，core 就绪也不标成默认完整流程完成。

未归项目、明确绑定同一 Bangumi subject 和 season 的剧集，离线幂等归并到 `cast_anime_*` 角色作用域，共用稳定角色 ID、档案与参考图库。不按片名合并，不同作品／季隔离；已归项目的旧系列作用域保留兼容。启动、档案应用、新增集和组织移动均应用共享规则，旧 scope、角色 ID 映射、参考图 ID／确认撤销状态与审计保留；原始观察不重写。引用 PNG 先写入目标，角色、图库和成员映射通过同一恢复事务发布。图库下载成功仍只产生候选；可信参考和身份校准不会因共享而自动获得。身份配置仍可按集控制，角色资料及参考图跨集共用。

旧 `scene-chunks-v1` 的输入、成功回执、剧情上下文与完成条件不变。新策略不读取前序深分析作为依赖，也不调用自动深入、整集剧情连续推理、上下文刷新或 LR-ASD。额外 OCR 单独运行；本策略仅接收 `stages=["vision"]` 并索引已有文字字幕。

## 采样、局部记录与恢复

- `keyframes.py` 的本地镜头检测和原始 PTS 时间证据复用；新 `retrieval_sampling.py` 通过已解码低分辨率帧的 observer 收集证据。镜头边界、检索窗口和模型请求批次是三种不同对象。
- `retrieval-sampling-v1` 保留每镜头首末帧、约 500 ms 周期帧，以及 4×4 局部变化网格中超过 0.12 的变化峰。周期采样不是总帧率上限；短暂物品及快速切镜可能使实际保留帧数增加。没有以超时或固定全片帧数丢弃后续镜头。
- 核心窗口通常 6 秒，另记录前后 1 秒上下文。窗口保留全部选中源帧；送局部 VLM 的每窗口最多 6 帧，优先保住每个镜头代表帧。若镜头数量超出上限，会拆窗口。长镜头持续采样；VFR 持帧保存真实源时间，不制造新时间戳。
- `retrieval-local-observation-v1` 每次最多 2 窗口、12 张当前帧，另有总量受限的参考图。请求宽度不超过 4，并受任务与连接并发限制。不按每帧发 VLM。
- 严格 schema 和服务端校验共同约束局部实体 ID、候选角色／概念 ID、当前证据帧、事件主客体与时间范围。时序动作必须提供两个不同源时间的有序帧；模型自报置信度不产生实名。设定参考不能作为当前事件帧。
- 作品上下文按当前帧与已确认参考图的直接视觉候选、局部字幕取少量相关资料；每窗口至多五个概念。相似度仅选择待核对档案，不写入已确认绑定。角色候选先在允许的完整图库匹配，再限制注入数量。当前没有每镜头联网检索；无相关资料时仍可描述可见外观。
- 原片指纹、字幕摘要、采样参数、模型／编码器身份、策略和 prompt 版本被冻结。统一任务在角色确认完成、正式分析前冻结当前参考图库身份，使本任务能使用刚确认的参考；后续变更仍受依赖校验约束。配置变化要求新任务。局部成功回执、不可变原始观察和发布指针支持恢复；损坏的原始观察不能被新哈希重新认证。恢复不把成功回放的速度当成首次处理速度。

模型批次支持有限时序证据，但不保证解决方向、否定、多人主客体关系或所有短暂动作；这些是留出集必须检验的能力。

## 编码、搜索与缓存

当前仅有两种编码候选，不接入时序视频编码器：

| 配置 ID | 权重与实现 | 状态 |
| --- | --- | --- |
| `siglip2-base-p16-256` | Google `siglip2-base-patch16-256`，Transformers | 已完成冷进程和暖进程新内容 core smoke；暂保留为首选，尚未通过最终选型验收 |
| `openclip-multilingual-b32` | LAION `CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k`，OpenCLIP + 本地 XLM-R tokenizer | 已完成同一 90 秒开发片段的 core 对照 smoke |

[SigLIP 2 官方模型卡](https://huggingface.co/google/siglip2-base-patch16-256)标注 Apache-2.0；[该 OpenCLIP 权重模型卡](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k)标注 MIT；[OpenCLIP 上游实现](https://github.com/mlfoundations/open_clip)提供对应实现。模型卡及服务器测量不证明本机动画效果或端到端速度。

追加的本机 core 实测均使用 90 秒片段：SigLIP 2 新进程暖机段 45.405 秒（首次视觉可搜索 34.455 秒、模型加载 6.687 秒），同进程另一段新内容 29.026 秒（首次视觉可搜索 20.600 秒），后者在新的隔离库中重新计算图像特征，未使用图像缓存回放。OpenCLIP 冷进程为 53.438 秒（首次视觉可搜索 43.269 秒、加载 12.281 秒）。原首轮 SigLIP 2 的 89.779 秒／加载 50.485 秒报告保留，不改写。

两编码器在相同 6 条开发正例上均有 Top-5 候选落在预标区间，2 条无答案问题仍返回 partial，不算正确拒答。SigLIP 2 与 OpenCLIP 的 PyTorch 峰值分配分别为 856,508,928 和 1,683,879,936 字节；这不是整个 GPU 的峰值占用。当前暂保留 SigLIP 2，依据仅是此小型 smoke 召回打平且分配显存较低。正常加载从 50.485 秒变为 6.687 秒，反映冷启动与 OS／驱动／文件缓存等环境状态的明显波动，具体归因未被控制实验确定；不能据 89.779 对 53.438 宣称 OpenCLIP 快约 40%。最终采用或淘汰仍需受控重复、未调参留出及 complete 预算验证。

`visual_encoder.py / visual_encoder_worker.py` 运行独立常驻 Python worker，只加载固定 revision 的本地文件；禁止推理时隐式下载。CUDA 不可用时明确失败，CPU 仅显式选择。批量编码使用兼容的图像塔与文本塔、L2 归一化；OOM 缩小批次后有界重试，不静默改用 CPU。worker 复用既有 Windows 监督和本地模型资源门控。

缓存身份包含编码器、权重哈希、预处理及 tokenizer 文件、实现版本／代码、依赖、device/dtype，以及源内容、帧 PTS／time_base、像素、区域。图文必须使用同一编码空间；不同签名的向量和原始分数不混用。文本查询缓存独立保存，改名不重算无关图像特征。区域编码采用归一化 `xyxy`；人工图像裁剪／绑定接口沿用现有 `xywh`，两个协议不能混用。

搜索继续使用现有 FTS、可选文本向量和范围过滤，增加直接视觉与作品概念分支，按排名倒数融合（RRF，常数 60）。中文原问题直接进入视觉分支；字符规范化、已确认姓名和别名进入文字索引。日中专名不能仅凭通用图文模型获得实名。可选查询理解仍保留原问题的字面召回；视觉和候选复核使用原问题，保留角色、物体、动作、否定和方向约束。

每窗口取最佳帧作为 `hit_frame_id / seek_ms`，返回有序源帧及原始窗口时间；前端优先跳 `seek_ms`。没有 caption 的窗口显示“画面证据”。界面默认显示最多八张有序代表帧并保留命中帧，可展开全部证据，不删索引帧。同一记录跨分支去重；相邻候选仅在同作品／原片、共享命中原帧或相同稳定参与者事件证据时保守合并，间隔不超过 1 秒、合并跨度不超过 12 秒，且合并键须在全部成员上共同成立。结果保留 `member_ids / source_spans`、全部有序原帧和最高排名代表的命中帧；局部实体／事件不跨观察拼接。合并是展示去重，不生成新的连续动作，也不能用合并区间冒充精确事件边界。

`POST /api/search` 的 `visual_verification` 与 `enhancements` 默认均 `false`。视觉候选没有仅凭相似度成为“完整匹配”的权限；无答案问题仍可能返回 partial 候选，目前没有校准的视觉拒答保证。

超过当前文本塔 token 上限的长查询明确停用该次直接视觉分支，并保留原问题、文字／概念召回及降级原因；不会截断约束。目前尚未实现保留完整条件的分句视觉编码或三表达别名扩展，相关长查询能力仍须后续验证。

动作定位仅在已有当前局部事件通过 `retrieval_events.event_seek()` 的受限中文约束校验时返回 `seek_basis=local_event / event_evidence`：唯一同类动作、角色／物体／接收者全部可消解、无否定或多动作歧义、至少两个有序不同源时间，且起点正好有当前证据帧。原 `hit_frame_id` 仍表示真正的视觉命中，匹配不会因此升级 full。其余查询保持原帧定位，不以窗口起点伪装动作起点；复杂表述仍需局部核验，尚无动作边界质量达标声明。

显式 `visual_verification=true` 使用唯一候选复核方案：最多 Top-3 候选、每候选最多 4 帧、单次 VLM 请求、最长 30 秒。缓存绑定原问题、候选证据、模型和复核策略；命中缓存本次请求数为 0。不可用、失败或超时保留原候选，不隐式启动任务或整集分析；返回 `verification.status/request_count/candidate_count/deadline_ms`。`enhancements=true` 单独启用既有查询理解／文字决策／解释能力，其请求预算不属于这个单次视觉复核上限。

## 角色身份与作品实体

`identity-binding-v2` 只处理显式标记的新检索观察，旧 scene 协议保持不变。`BindingResolver` 批量读时投影稳定角色 ID：人工决定优先；否则可靠当前自动绑定；证据不足保留候选、歧义或未知。当前观察绑定与可信参考入库是两回事。

自动绑定要求有效的已审参考、未撤销的版本／像素哈希、匹配模型与预处理／校准身份、当前合法区域、不同源时间的同镜头双观察一致；独立单帧路径须有单独留出校准。校准声明必须满足 `auto_precision≥0.98`、`eligible_coverage≥0.60`、`cross_character_error_rate≤0.01`。CCIP 距离和差距不是概率，模型 confidence 不参与晋升。没有提供真实动画留出校准，本轮不会造出一份“已通过”校准。

自动观察不更新可信图库或供后续识别使用的外观档案。人工确认、未知、撤销与冲突会重新投影；改名和别名更新搜索／展示／事件参与者，不改写原始描述、观察或特征。对应入口仍在“角色档案／视听身份”。

`work-entities-v1` 单独建立持久 `work_scope_id`，由用户显式创建或把其他集加入同一 ID；不拿每集 `work_id` 或可变 `anime_id` 代替。局部 `Entity.id` 只在当前观察内有效；`character_id` 指向已有角色库；`concept_id` 指向道具、机体、场所或组织符号，物体不伪装成角色。

详情页“作品实体”可编辑正式／原文／译名／别名、区分性外观、形态、功能／归属／设定事实、必要关系、出处和适用阶段；复杂字段提供 JSON 表单。可上传本地参考图或选已注册原帧，支持整景、地标与物体区域，人工确认／撤销及审计。资料包采用受限 ZIP 导入，不跟随链接下载。参考图与资料可跨集复用；Bangumi 继续初始化已有角色，未被当成完整道具地点百科。

主要接口为 `/api/work-entity-scopes`、`/api/assets/{id}/work-entity-scope`、`/work-entities`、`/work-entities/{concept_id}/references`、`/references/from-frame`、`/occurrences`、`/work-entity-bindings`。当前可从画面局部实体直接打开人工绑定表单，并记录证据帧、区域、原因和状态。

原始观察、概念绑定及证据、设定事实、当前事件分开保存。功能查询可返回该概念已确认的出场，但 `event_verified=false`；“正在停止时间”必须有当前事件证据。场所参考保留完整景别／地标，不套用人物 CCIP 规则。意象先保存可见特征，设定解释不能伪装成当前事实。P3 工程已接通，**真实作品最小样例库及对照样本尚未整理，尚无设定增益结论**。

## LR-ASD 与资源策略

新普通画面流程固定 `speech_activity=off`，完整模式也不调用音频身份链。需要“谁正在说话”时显式使用已有短窗口 LR-ASD 入口；它提供人物轨迹上的说话活动证据，再关联角色、字幕及人工核对。画外音、重叠语音和字幕仅提及不证明在画面中说话。

有界扩大的说话覆盖保留为独立增强，不是本版本默认能力。已有模型、标注和训练成果保留；没有新的大规模训练或微调候选部署。优化先围绕新路径的解码、批量图像编码、参考缓存、上下文和索引发布，不能用更高并发挤占共享 GPU 后再声称加速。

## 实验命令与验收门槛

Windows 目标设备与安装见 [WINDOWS_MODELS.md](WINDOWS_MODELS.md)。以下命令在仓库根目录执行；所有真实实验使用隔离资料目录，输入 manifest 只读引用已导入资产及原片位置，不向正式库写配置。

```powershell
# 查看入口；不会运行模型
.\.venv-windows\Scripts\python.exe scripts\benchmark_retrieval.py --help

# 在现有开发清单上新建一次独立冷进程实验；会运行本地图文模型
$experimentStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
.\.venv-windows\Scripts\python.exe scripts\benchmark_retrieval.py `
  --manifest test-results\retrieval-v1\dev.json `
  --encoder-config test-results\retrieval-v1\encoder.json `
  --encoder siglip2-base-p16-256 --run-mode cold-content --allow-local-inference `
  --work-dir "test-results\retrieval-v1\siglip2-$experimentStamp" `
  --output "test-results\retrieval-v1\siglip2-$experimentStamp.json"
```

本机 `dev.json / encoder.json` 与实验产物位于忽略目录。已核对的开发清单另保留在 [examples/retrieval-smoke.windows.json](examples/retrieval-smoke.windows.json)，只含来源映射和查询，不分发媒体或权重，也不代替留出集。新机器先准备清单；最小结构为 `{"clips":[{"id":"clip01","source_library":"D:/ExistingLibrary","asset_id":"实际资产ID","start_ms":0,"end_ms":60000,"queries":["红色物体"]}]}`。编码配置格式见 Windows 模型文档。入口限制每段不超过 90 秒，拒绝源库／实验目录交叠、覆盖已有报告和非空首次处理目标。

`cold-content` 每段重新启动编码器；`warm-process-new-content` 至少两段不同内容，第一段单列暖机；`receipt-replay` 仅接受同实验目录中已经成功的相同范围、编码器、core 任务。回放不是首次内容性能。此脚本禁止远程模型请求，只测 core，不能用其报告验收 complete；后者必须另以真实局部 VLM 短片任务记录。

```powershell
# 合成工程回归，不调用真实模型；workspace 临时目录避免 Windows 沙箱 TEMP 限制
$testTemp = Join-Path $PWD '.cache\retrieval-validation-temp'
New-Item -ItemType Directory -Force -Path $testTemp | Out-Null
$env:TEMP = $testTemp
$env:TMP = $testTemp
.\.venv-windows\Scripts\python.exe -m pytest -q `
  tests/test_retrieval_sampling.py tests/test_retrieval_protocol.py tests/test_retrieval_jobs.py `
  tests/test_retrieval_complete.py tests/test_retrieval_api.py tests/test_retrieval_verification.py `
  tests/test_visual_encoder.py tests/test_visual_retrieval.py tests/test_identity_bindings.py `
  tests/test_work_entities.py --basetemp="$testTemp\pytest"
Push-Location frontend
npm.cmd test -- tests/retrieval_search.test.tsx tests/work_entities.test.tsx
npm.cmd run typecheck
Pop-Location
```

部分既有 asyncio／HTTP／子进程测试需要允许本机回环 socket；沙箱 `socketpair` 阻塞属于环境限制，不能当成模型效果结果。只运行与新行为和兼容风险有关的测试；旧历史性能报告保留。

| 阶段 | 已有工程／实验 | 仍需完成的进入下一阶段条件 |
| --- | --- | --- |
| P0 | 版本化契约、完成边界、隔离实验入口、复用计时和回执 | 冻结至少 30 条有目标＋10 条无答案查询、分离开发／留出；明确单帧可见区间与时序事件边界 |
| P1 | 图像＋字幕检索闭环；两编码器同片开发 core smoke；SigLIP 2 暖进程新内容测量 | 受控重复两编码器对照与未参与调参的短片验证；比较质量、首搜、总墙钟、显存、依赖与许可后才最终选型 |
| P2 | 严格局部事件协议、持久化、实名投影和前端；合成回归；6 秒真实 VLM 功能 smoke | 更换或精简局部生成后重测质量与吞吐；至少 100 个分层人物观察留出，自动精确率≥98%、eligible 覆盖≥60%、跨角色错连≤1%；同时公布全部观察未知率，不能全面拒识过关 |
| P3 | 作品概念、参考图、绑定、功能检索及 GUI | 人工整理最小高频道具／机体／场所库；同一查询集对比基础、图文设定，必要时加纯文字设定；专名有增益且不增加设定冒充事件的错误 |
| P4 | 局部资源门控、缓存、恢复和增量发布机制 | 新路径热点优化；留出 Top-5≥80%；适用事件边界误差≤1秒；真实整集 complete 总墙钟≤原片时长；兼容恢复、预算失败、索引发布、Windows 清理和备份恢复通过后才讨论默认切换 |

30＋10 必须覆盖实名／别名、多人递交、连续动作、短镜头、无台词、专有道具、驾驶舱／特殊地点、背景和意象；负例包括相似实体、跨作品同名、台词提及但未出场、画外音、功能未发动。静态目标以人工可见区间内的命中源帧计命中；有清晰起止的事件才统计两侧边界误差，整段窗口范围不能冒充 1 秒定位。无答案集报告错误完整匹配、候选返回和拒答情况，不能把 partial 默认等同正确拒答。

完整性能应同时记录总墙钟、首次视觉可搜索、默认完成、阶段、请求、GPU／内存占用。一次性安装／下载单列；正常模型加载计入处理；首次作品资料准备与跨集复用分开记录。90 秒 core 接近实时不是 24 分钟 complete 达标。如果局部 VLM、采样密度或首次加载预算不支持目标，应先小规模比较减少请求、替换组件或退出默认能力的策略版本，再决定是否扩大；不得靠后台挪走必需工作、漏窗或到时停止达标。

当前真实结果及其局限见 [VALIDATION.md](VALIDATION.md#检索优先实验与工程验证2026-10-20)。本文件不宣称 P0–P4 已全部验收。
