# 作品知识检索与 LightRAG 试验

知识库默认关闭。作品资料、概念、角色和人工影片绑定保存在 SceneRecall；LightRAG 的图谱和向量是可删除、可重建的派生索引。知识命中与影片命中分别返回，设定不能证明当前片段发生了某个事件。

## 安装与模型配置

首轮固定 `lightrag-hku==1.5.7`，上游提交 `28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c`。Python 3.12 独立环境的传递依赖锁文件位于 `scripts/knowledge-runtime/uv.lock`，不加入主应用环境，不部署上游 WebUI 或 HTTP server。

在仓库根目录执行：

```powershell
.\scripts\setup_knowledge.ps1
```

脚本使用 `uv sync --frozen`，显式准备 tokenizer，并显示知识 Python 路径。在“模型与设置 → 模型连接 → 作品知识库后端 · LightRAG”中填写该解释器路径，启用知识后端，选择文本连接和明确模型，再检查连接。文本可复用既有 Codex 连接；API 文本连接也沿用现有连接管理。Codex 使用订阅额度，金额保持未知；不能把它记成零。

默认 embedding 是本地 CPU 多语言 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，输出 384 维。独立环境使用锁定的 FastEmbed/ONNX Runtime；模型采用 `Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` 的固定 revision `faf4aa4225822f3bc6376869cb1164e8e3feedd0`。安装环境后显式准备模型：

```powershell
.\.cache\knowledge-runtime\Scripts\python.exe scripts/download_knowledge_embedding.py `
  --destination D:\SceneRecall\.cache\knowledge-models\multilingual-minilm-l12-v2
```

在设置中选择“本地多语言 embedding（CPU）”，填写脚本输出的绝对模型目录。worker 校验模型清单和文件 SHA-256，只从本地加载；缺文件或校验失败时提示准备模型，不在查询中下载。超过模型上下文的输入使用固定的分窗、加权平均与归一化策略，策略随运行代码和模型签名一起冻结。本地模型文件与每部作品的派生索引分别计量。

可选“已有 embedding API 连接”需要声明 `embedding` 能力、实际 Base URL、模型 ID 和凭据；Codex CLI 不提供 embedding。连接检查取得真实维度并冻结模型签名。更换本地模型或 API 模型均须重新预检、重建派生知识索引。

API 文本连接提供“知识文本 API 思考模式”：服务默认不发送 `thinking` 参数；只对支持该参数的服务显式选择关闭或启用（配置字段 `text_thinking=null|disabled|enabled`）。例如 DeepSeek 的小样本抽取可以显式选择关闭思考后测量效果与代价。该选择参与索引签名，不影响其他模型连接。Codex 使用独立的思考等级字段，不能接受这个 API 专用参数。

当前真实试验按用户后续选择使用 DeepSeek 非思考文本模型与本地 CPU embedding。配置请求的模型名与供应商响应中的模型名分别记录，响应别名不覆盖冻结的请求配置；实际调用量及样本数以隔离库的建库回执和实验报告为准。

模型回调经过主进程 ProviderManager，密钥不会进入作品文件、worker 协议或派生图。文本回调保留 system、prompt、history 的角色顺序；Codex 的 `{text:string}` 仅是传输外壳。入库实体/关系抽取、描述归并、查询关键词、embedding 的调用用途分别记录。重排关闭，`aquery_data` 不生成最终回答。

## 导入、核对与查询

### 复用 Bangumi 初始档案

选择 Bangumi 档案导入动画时，会在本地复用已获取的作品简介、角色简介、名称/别名和 infobox 字段，保存带条目 URL、获取时间和内容哈希的知识来源快照。角色简介及非名称资料字段会生成关联现有内部角色 ID 的 **candidate** 资料卡。作品简介先作为来源正文保存，可从中选取片段并关联概念；不会凭简介中的名字自动创建正式组织、能力或世界规则。

此步骤不额外调用文本模型、不启动 LightRAG 建库，也不会生成确认出场。Bangumi 的“主角/配角/客串”表示角色与作品的关联，不代表角色之间的关系；声优和制作人员字段也不解释为剧情关系。评分、收藏量和评论不作为设定知识。

没有资料库的作品会按当前具体动画身份与 Bangumi 条目创建独立作用域；同版本的各集复用它，不仅按作品显示名合并。已有明确归属的资料库继续使用当前归属，不能通过此功能跨作品换绑。

旧影片可在“作品图文设定库”点击“从已有 Bangumi 档案准备知识资料”，或在知识面板点击“整理已有 Bangumi 资料”。操作仅读取本地已保存档案，不再次访问网络；缺失的内容仍可手动补充。重复操作不重复生成资料卡、不重置已核对或已撤销的决定。更新档案后变化的来源保存为新快照，旧导入来源失效，新卡片重新待核对；不会自动把新内容视为已经审核。

角色暂时没有唯一内部身份时先保存来源；身份补齐后再次整理可补建候选卡。已有资料卡的身份引用若失效，会提示编辑卡片并重新选择内部对象，不会自动改写已经审核的关联。单纯移动项目不会更换知识作用域；季或分集/电影版本确实变化时仍拒绝混入旧自动资料库。

按下列步骤核对自动生成的卡片后，再显式启动建库。导入任务会显示整理数量和可恢复的错误；资料整理失败不阻止影片导入。

隔离检查复用了本机已保存的《少女☆歌剧 Revue Starlight》（Bangumi #214265）档案：14 个角色整理为 15 份来源、45 张候选资料卡，重复整理新增数为零。没有联网刷新、模型调用、已核对知识或确认出场；这个数量仅描述该份缓存档案，不代表所有条目都有相同覆盖。

1. 在作品设定库创建明确的作品版本作用域；同一 TV 版本的多集可显式加入同一作用域。电影或实质不同版本使用独立作用域。
2. 上传少量 UTF-8 `.md` / `.txt`，填写标题及出处 URL。原始字节、SHA-256 和快照 ID 会保存；资料中的 URL 和 Markdown 图片不会触发抓取。
3. 选取原文片段，关联已有概念或内部角色，填写资料卡、条件、代价、例外和适用阶段。概念不要求存在 Bangumi 条目。新自动抽取的名字只作为候选。
4. 人工核对后将卡片设为 reviewed，再启动建库。未核对、来源已撤销或目标已失效的卡片不会发布到索引。
5. 在知识面板查询资料，展开来源原文；有有效绑定的可见概念可以查找片段。角色出场需要明确姓名查询或显式选择角色。规则、能力等抽象概念可以只有知识结果。
6. 在镜头检索中选择单一作品版本，切换 baseline / LightRAG。全库多作品查询暂退回基线，并显示原因。

现有人工出场界面支持 core 窗口，不需要局部 VLM 实体。确认绑定必须引用当前原帧和真实记录。抽象概念不能直接保存“确认出场”；应通过可见对象、人物、台词或已有事件证据关联。

知识片段的 `setting_concept_appearance` 和 `knowledge_character_appearance` 始终保留 `partial`、`event_verified=false`。字幕提及只提供字幕证据。拥有能力、关系或约定不能证明当前画面正在发动能力、履行关系或完成约定。

## 索引、标识与一致性

每张已核对资料卡生成确定性知识文本，同时供词法与 LightRAG 使用。LightRAG 使用显式 `sr_doc_<card_id>`、唯一 basename，并从实际 `doc_status.chunks_list` / `text_chunks` 读取块 ID 建立持久映射。块到卡片按多对多表达；未消歧的共享块只能显示知识，不能选择影片身份。

`entity_name`、`src_id/tgt_id` 是上游名称，`source_id` 是块引用，`reference_id` 是查询内临时序号。名称、别名、正文中的 ID 字符串都不能替代正式映射。图节点需要来源锚定的人工对齐；同名合并对应多个内部 ID 时标为 ambiguous，不合并正式概念。

在“抽取候选与内部身份核对”中选择图节点或关系、内部对象及已核对的支持资料卡。关系键为有方向的 `source->target`，须选择相应关系资料卡。只有唯一 reviewed 映射、有效支持卡和同代完整图谱块追踪共同成立时，图谱命中才能通过该映射恢复正式目标；即使上游查询返回的 `source_id` 被截断，也使用建库时保存的完整追踪验证出处。未核对、冲突、撤销或支持资料失效的映射不能选择影片身份。

知识结果的 `mapping_basis=document_card` 表示通过文档资料卡取得主目标，并不宣称图节点身份已确认；`reviewed_graph_alignment` 表示使用了上述人工核对的图谱身份关联。两条路径最终都只投影当前影片证据，不确认事件。

worker 生命周期固定为 `(work_scope_id, work_version_id, generation)`；workspace 在实例构造时设置，独立目录隔离。首轮最多一个活动操作，建库或忙碌时查询直接退回词法。模型回调与查询响应分别调度，取消和 EOF 会清理进程及调用账本。

默认文件存储会把 JSON、NetworkX 图和 NanoVectorDB 向量装入内存。关闭或切换作品时退出 worker，释放其进程内存；磁盘持久化不等于低内存。

知识输入、别名或映射变更立即令旧代 stale，显式按作品重建，校验后原子发布。查询不能触发重建。绑定、人物确认和帧证据变化只失效影片投影，不要求重建图谱。查询前后及 HTTP 返回前检查当前版本；已撤销资料或绑定不能借旧缓存或回滚复活。

身份版本检查覆盖角色档案、人工决定、校准设置、图库信任与参考图、帧登记、原片指纹以及当前检索发布指针。它仅读取这些发布文件和目录的身份，不递归扫描特征、裁剪或历史目录，也不读取影片原帧像素。当前区域/匹配的原子发布会失效版本；最终返回仍从权威记录重新验证身份和证据。

API 使用既有本机会话认证：

| 路径 | 用途 |
|---|---|
| `/api/knowledge/settings`、`/api/knowledge/check` | 配置及预检 |
| `/api/assets/{asset}/knowledge/bangumi`、`.../knowledge/bangumi` | 从已保存 Bangumi 档案准备来源和候选卡，不联网或建库 |
| `/api/work-entity-scopes/{scope}/knowledge/sources` | 来源导入、查看和撤销 |
| `.../knowledge/cards` | 来源资料卡及核对状态 |
| `.../knowledge/build`、`.../knowledge/status` | 现有任务队列中的建库、状态和错误 |
| `.../knowledge/query` | `knowledge_hits` 与 `film_hits` |
| `.../knowledge/alignments`、`.../knowledge/graph-candidates` | 图谱候选与来源锚定的核对映射 |

关闭知识后端或切回 baseline 即可回退；资料和确认绑定保留。未配置、未建库、超时、进程失败或版本不符时保留其他召回分支，并返回降级原因。


## 实验准备与运行

`docs/examples/knowledge-evaluation.json` 提供三份官方短节选、十个概念草案、十张候选资料卡和 24 条问题（16 条开发、8 条留出）。摘录保留原 URL 和字节；它们是节选，不冒充完整网页快照。资料卡默认 candidate，须核对原文、翻译和概念区分后使用。

示例不含伪造的影片时间或身份确认。`expected` 中 `null` 表示未标注，`[]` 表示人工确认该层无答案。知识卡、目标、影片三层都标注完成后才能形成完整质量结论；影片可用 `record_ids` 或 `film_spans` 标注。要求当前事件的正例填写 `required_evidence_type:"event"`，此时普通 partial 出场不会计为正确事件片段。三个跨作品/撤销问题初始禁用，必须在独立状态中准备后分别测量。开发和留出划分在调参前固定；不得用留出答案补充别名后又报告留出成绩。

先使用隔离目录复用已发布证据：

```powershell
.\.venv-windows\Scripts\python.exe scripts/prepare_knowledge_trial.py `
  --source-library D:\SceneRecallLibrary `
  --asset-id asset_49d289a62d9740babce8 `
  --output-dir D:\SceneRecall\.cache\knowledge-trial\library
```

输出目录必须不存在。原帧在可用时硬链接，作为只读输入；可变 JSON 和选定 SQLite 记录独立复制，不重算图像特征。不要编辑这些硬链接图像，也不要将运行实验的资料库指向正式库。

在隔离库导入、核对资料，复用真实内部角色 ID，检查原帧并建立绑定。给 manifest 填写真实 `work_scope_id`、卡片和目标 ID、片段 ID 或时间区间。保持所有模式的资料、绑定、过滤和其他检索分支一致。最低可先验证基线入口：

```powershell
.\.venv-windows\Scripts\python.exe scripts/benchmark_knowledge.py `
  --library D:\SceneRecall\.cache\knowledge-trial\library `
  --profiles-dir D:\SceneRecallLibrary\private `
  --manifest D:\SceneRecall\.cache\knowledge-trial\manifest.json `
  --output-dir D:\SceneRecall\.cache\knowledge-trial\baseline-check `
  --mode baseline --cold-runs 0 --repeats 1
```

`--profiles-dir` 复用既有连接和 keyring 命名空间，避免复制明文凭据或改变 keyring 路径身份。只有模型配置使用该路径，影片和知识写入仍在隔离库。

连接检查和建库通过后运行对照：

```powershell
.\.venv-windows\Scripts\python.exe scripts/benchmark_knowledge.py `
  --library D:\SceneRecall\.cache\knowledge-trial\library `
  --profiles-dir D:\SceneRecallLibrary\private `
  --manifest D:\SceneRecall\.cache\knowledge-trial\manifest.json `
  --output-dir D:\SceneRecall\.cache\knowledge-trial\comparison-01 `
  --mode baseline --mode lightrag --mode naive `
  --cold-runs 3 --repeats 5
```

工具使用真实 ASGI `/api/search`，包括最终权威证据校验；不启动后台入库任务。默认关闭 enhancements 和视觉复核，保留配置一致的字幕、语义及视觉分支。`--disable-visual` 仅用于双方相同的知识链路诊断，不能据此宣称完整检索效果。

每次模式/轮次使用新应用进程。cold 轮在每问前关闭知识 worker 并清空可丢弃 catalog/矩阵缓存；它表示 worker/应用缓存条件，不表示清除了 OS、持久关键词或文本向量缓存。warm 轮先预热全部查询，预热响应保存但不计量。轮次不是独立问题数，报告同时给出 distinct_queries。

输出目录拒绝覆盖，包含逐查询 JSONL、原始 HTTP 响应、日志、每轮资源记录和 `summary.json`。每问完成后立即 fsync 到 `<mode>-<regime>-<round>.partial.jsonl`，长任务中也能查看完整进度；中断保留已经完成的行。父进程中断时只清理本次启动的实验子进程及其后代，避免 Windows 遗留 worker：

- 分别记录知识、内部目标、影片的 Top-1/5/10、空答案及来源追溯；未标注层保持未知。
- 记录端到端、知识阶段和 HTTP 阶段耗时；回调用途、真实尝试、token 和费用保留原始账本。不可获得的费用或 token 不填零。
- 每作品分别记录快照、共享 catalog 和派生索引磁盘字节，排除模型、原片、现有影片索引及参考图片。新增占用需用前后差值。
- RSS 以 20 ms 采样，主进程、知识 worker、其他子进程分开；不是 GPU 或分配器峰值保证。
- 检查跨作用域误命中、已知错误身份及错误事件断言；没有人工身份金标时不能宣称身份零错误。

只有金标已完成、LightRAG 确实参与、知识输入在模式间相同且未发生变动时，`ready_for_effect_claims` 才可能为 true。后端降级为词法的结果不会被当作 LightRAG 成绩。来源、绑定撤销和版本冲突另用独立快照状态运行，避免前一个模式改变后一个模式的输入。

首次建库/重建时长和调用账本来自 build manifest 与任务回执；查询工具不会自动发起重建。将新增资料、资料撤销、参考图新增绑定的收益分别报告。若语义/关系的留出改善不能复现，或主要收益由资料覆盖而非 LightRAG 解释，应报告这一结果；不以服务启动或生成回答作为完成证明。

上游依据：[v1.5.7 发布](https://github.com/HKUDS/LightRAG/releases/tag/v1.5.7)、[结构化 SDK 查询](https://github.com/HKUDS/LightRAG/blob/v1.5.7/lightrag/lightrag.py#L3998)、[实际返回转换](https://github.com/HKUDS/LightRAG/blob/v1.5.7/lightrag/utils.py#L6851)。
