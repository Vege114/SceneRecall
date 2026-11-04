# 针对现有代码的搜索优化初步核对（2026-11-04）

前半部分记录对已有耗时、当前代码及官方资料的初步核对；末节补入用户要求的 ChatGPT Pro 项目新会话研究及独立来源核验。**本文不是已批准的最终优化路线。** 本轮未实施性能优化、未运行新搜索、未测量提速；模型建议、成熟机制与项目适配推断分别标注，不能预设缓存就是答案。

## 已知事实与研究范围

昨日12集暖查询：点击到结果63.890秒，服务63.869秒。互不重叠的主阶段为来源重建33.601秒、三次版本核验16.950秒、并行召回12.742秒、其他0.576秒。[原始计时](search-detail-full-warm-timing-20261103.json)、[基准条件](search-detail-benchmark-20261103.json)

- 10条结果涉及5个素材，却重建了4,387条记录。
- 知识projection执行5次，共15.715秒，已经包含在核验和召回中，不能重复相加。每次读取并解析JSON 316次，仅涉及7个不同JSON路径。
- 全请求文本读取累计18.776秒，JSON解析累计11.203秒；这些操作已包含在阶段时间中，且不是纯物理磁盘或CPU时间。
- library锁等待2.202秒，其中约2.136秒位于projection。未记录持锁者，不能归因于后台导入。
- 视觉热召回仅0.128秒，查询向量和矩阵缓存均命中，矩阵预算1GiB。

这是返回10条结果的一次暖样本；阈值模式可能返回更多候选，不能按上述比例直接预测提速。接下来应抽象比较读多写少、本地检索、多源权威证据与身份一致性、读取放大、来源恢复及变更可见性等架构问题。

## 可供后续比较的现成方案

以下是静态适用性判断，不表示确定实施顺序。

| 候选方案 | 当前函数落点与适用性 | 正确性/实施边界 |
| --- | --- | --- |
| 按请求或稳定版本复用读取结果 | `KnowledgeService.asearch`已有projection，`KnowledgeStore.lexical_hits`又生成一次；目录和角色资料也重复读取 | 版本探测不能被固定缓存隐藏；返回对象应只读或隔离修改；请求缓存也需要容量和并发管理 |
| 按候选及其依赖加载权威来源 | `main.current_candidates`已计算requested_ids，随后仍调用`Library.records(owner)`全量重建 | 要展开member_ids/source_ids及场景、字幕、身份依赖，不能直接使用可能滞后的索引JSON替代权威结果 |
| 按完整依赖版本缓存，并合并相同key的并发构建 | 可进一步跨请求复用projection；`WorkEntityRecall._load`已有文件身份前后检查可供参考 | key、失效、失败/取消、外部文件变化和内存均需审计；普通LRU不保证只构建一次 |
| 将版本探测与内容生成分开 | 三次核验主要成本不在SQLite版本读取，而在知识读取与投影 | 三次核验仍需探测新状态，不能直接复用query_token返回值 |
| 变更驱动的增量投影/持久化读模型 | 减少读取时重新组装，按受影响record或scope维护派生数据 | 需依赖索引、版本标记、幂等更新、崩溃恢复和全量重建后备，复杂度较高 |

## 官方依据与具体代码核对

**请求复用与批量加载。** DataLoader官方文档描述按请求memo、批量加载、更新后清缓存的模式；可借鉴设计，无需引入GraphQL。[DataLoader](https://github.com/graphql/dataloader#caching)。Python的ContextVar可保存请求上下文，to_thread会传播它；共享对象仍需线程同步。[ContextVar](https://docs.python.org/3.12/library/contextvars.html)、[to_thread](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.to_thread)

对应入口：[asearch](../backend/scenerecall/knowledge_service.py)、[projection](../backend/scenerecall/knowledge_store.py)、[lexical_hits](../backend/scenerecall/knowledge_store.py)。不宜第一步就在通用read_json上无差别缓存全部文件。

**定向来源读取。** [current_candidates](../backend/scenerecall/main.py)与[Library.records](../backend/scenerecall/library.py)之间缺少按ID恢复权威记录的接口。现有[filtered_records](../backend/scenerecall/search.py)已经每256个ID分批SQL读取并使用事务，records.id已有主键。因此瓶颈不能简单概括为“缺少批量SQL”。[SQLite查询计划](https://sqlite.org/queryplanner.html)、[参数限制](https://sqlite.org/limits.html)

定向加载仍应保留[RetrievalStore.records](../backend/scenerecall/retrieval_store.py)使用的active指针、ID、摘要和来源验证，以及当前身份、注释、场景替代、字幕和合并成员的投影规则。

**缓存机制与失效。** lru_cache结构线程安全不代表同key只计算一次；cachetools的condition可抑制相同key并发重复构建，容量与可变值仍由使用方管理。[Python functools](https://docs.python.org/3.12/library/functools.html#functools.lru_cache)、[cachetools](https://cachetools.readthedocs.io/en/latest/)。Cache-Aside要求写入后失效，并单独处理一致性。[官方说明](https://learn.microsoft.com/en-us/azure/architecture/patterns/cache-aside)

项目可参考[WorkEntityRecall._load](../backend/scenerecall/knowledge_recall.py)。projection依赖涉及作品目录、来源状态/摘要、卡片/alignment、membership、素材归档策略、角色目录/允许名单及实现版本，不能只用scope ID作key。

**增量读模型。** Materialized View模式支持预生成适合读取的投影，CQRS文档提醒读模型更新延迟和维护复杂度。这里只是本地设计的参考，不要求部署Azure或Redis。[Materialized View](https://learn.microsoft.com/en-us/azure/architecture/patterns/materialized-view)、[CQRS](https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs)

## 不可省略的边界与验证建议

1. **正常路径三轮freshness核验不能直接memo。** [query_token](../backend/scenerecall/search.py)、[版本比较](../backend/scenerecall/search.py)和[最终核验](../backend/scenerecall/search.py)必须探测新状态。内容仅能在依赖版本一致时复用；构建中变化仍需重试或失败，不能缓存混合快照。
2. SQLite旧读事务保持旧快照，不能据此判断外部新提交，也不能覆盖JSON文件。[隔离规则](https://sqlite.org/isolation.html)。PRAGMA data_version只能比较同一连接的先后值，不反映本连接自身提交，不宜直接替代现有[query_revision表及触发器](../backend/scenerecall/query_cache.py)。[官方语义](https://sqlite.org/pragma.html#pragma_data_version)
3. 不宜直接全局缓存[Library.get_asset](../backend/scenerecall/library.py)：返回值可修改，且[organization.project_asset](../backend/scenerecall/organization.py)发现未登记素材时可能写目录。若缩短锁范围，也必须先捕获一致快照、发布前重新验版本。

未来正确性测试建议（本轮未执行）：

- 三轮核验间改名、撤销来源/身份，命中缓存也能检测变化。
- 构建中替换/删除文件、改变active指针或损坏摘要，不缓存混合状态；缺失后恢复能重新读取。
- 并发同key、失败和取消，不串请求、不重复构建或永久等待；修改返回值不污染缓存。
- 定向与全量加载逐条对照，覆盖字幕来源、合并成员、场景替代、注释、跨素材拒绝及超过旧上限的阈值候选。
- 缓存关闭、极小容量、淘汰、重启、外部改写，结果和证据有效性与无缓存路径一致。

性能验证应固定查询模式、范围、预算和结果条件，记录冷暖分布、唯一/重复读取、projection次数、锁等待和内存；不能重复累加父子阶段，也不能把单次样本写成稳定提速结论。

## ChatGPT Pro项目新会话研究补充

[项目新会话：检索系统架构优化](https://chatgpt.com/g/g-p-6ae1878c2e64819195f7b7bd05053d31-scenerecall/c/6aead4ff-cc6c-83ec-aa00-2a28ce29290f)，回复原文保存在 `.cache/optimization-research-20261104/pro-response-dom.txt`。提问先抽象为“本地读多写少、多源权威证据、人工纠错必须及时生效”的架构问题，再要求联网研究；没有预设增大缓存。以下为回复摘要与核对后的设计建议，均未实施。

### 问题抽象与官方机制

Pro 将现象归为：**read amplification / over-fetching**（少量命中却恢复大量对象）、**late materialization**（先判断资格，再恢复必要的宽字段和关联）、**dependency tracking / revision-based validation**（低成本判断输入是否变化）、**read model / materialized projection**（维护适合查询的派生表示），以及 **optimistic validation / publication ordering**（计算后验版本，协调发布与纠错）。这是本场景的分类判断，不是这些名词自动保证提速或一致性。

本次独立打开并核对以下四项官方资料：

| 成熟机制与依据 | 对项目的适用判断及边界 |
| --- | --- |
| [SQLite：许多小查询可以高效](https://www.sqlite.org/np1queryprob.html)：同进程调用没有客户端数据库的逐查询网络往返 | 不因 SQL 次数多就认定 N+1 是主因；应测扫描量、对象构造和重复解析。`filtered_records` 已批量读，优先检查 `current_candidates → Library.records` 的恢复范围 |
| [CQRS：同一数据存储中分离读写模型](https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs)：独立读模型不要求拆数据库或引入分布式服务 | 可以研究在同一 SQLite 中维护候选到证据的窄读模型；是否同步维护、哪些 JSON 迁入数据库，是本项目待验证选择 |
| [Salsa 增量算法](https://salsa-rs.github.io/salsa/reference/algorithm.html)：记录输入 revision 和依赖，未变则复用；派生值未变可避免继续重算 | 为 `KnowledgeStore.projection` 等纯投影建立完整依赖版本有成熟依据；不建议直接移植 Rust 框架，也不能忽略读取时的隐式输入 |
| [Bazel Skyframe](https://bazel.build/reference/skyframe)：显式依赖图支持失效与重算，未登记的文件读取会破坏增量正确性 | 版本协议必须覆盖文件、角色规则、集合成员关系及“当前无人工 override”等负依赖；只缓存已读到的记录无法发现新增成员 |

资料证明这些机制存在及其前提，**不证明本机能达到某个延迟**。未采纳回复中未经本次独立核验的库版本、部署兼容性或 SQLite 特定缺陷声明。

### 必须分开的四个正确性契约

下表是建议明确的目标，不表示当前系统已经全部实现。

| 契约 | 需要保证什么 | 仅靠什么不够 |
| --- | --- | --- |
| 结果内部一致 | 身份、来源、绑定、引用和版本来自一致的已提交状态 | 多连接各读一次不是共同快照；SQLite 事务也不覆盖可变 JSON 文件 |
| 发布时有效 | 在明确的最终校验点，相关依赖仍有效 | 在同一旧读事务重读 revision 只会看到旧快照；须重新观察当前状态 |
| UI 纠错后撤销 | 受控修改确认成功后，旧显示已失效，晚到的旧响应不能重新装入 | 清服务端缓存或最终查库不能关闭“校验后、UI 安装前”的竞态；需要结果版本与修改/发布顺序 |
| 新增召回及排序时效 | 修改后新增的匹配项及时进入候选，相关排序也更新 | 从旧候选中过滤失效项无法补出漏召回；需同步索引、索引水位屏障或完整的增量补充路径 |

例如把身份 A 改成 B：权威回源可以排除查询 A 的旧命中，却不能让旧索引未返回的记录自动出现在查询 B 中。不能把“展示有效”宣称成“最新召回完整”。SQLite 快照的边界已由上文[官方隔离文档](https://sqlite.org/isolation.html)核对；UI 协调、索引水位如何落地仍是项目设计工作。

### 推荐的渐进方向与验证条件

1. **先减少同版本重复工作和恢复范围。** 在 `KnowledgeService.asearch → projection/lexical_hits` 复用已捕获版本的纯结果，并合并同 key 在途构建；在 `current_candidates → Library.records` 按候选及依赖恢复权威证据。阈值模式不能为加速偷偷截断候选，依赖展开和资格判断也不能延后到错误的位置。
2. **再设计覆盖所有写入口的变更协议。** 明确每类事实的唯一权威；探索在同事务更新事实、revision、有效性与恢复日志。先采用少数 scope/内容/身份版本域，只有无关失效显著时才细化依赖图。外部文件若改为“导入提交后生效”，这是语义变化，不能静默替换目前的文件检查。
3. **按剩余成本决定读模型与跨请求复用。** 便宜投影可同步维护，昂贵投影同步置无效后重建；缓存 key 必须覆盖输入、策略和投影实现版本。旧 builder 只能写旧版本 key，更新当前指针或发布时再次核验；清缓存本身不能防止旧结果复活。暂不引入大型服务，视觉矩阵预算继续为 1GiB。
4. **用四个契约验收，再考虑改核验流程。** 目前三轮 `query_token` freshness 检查仍不可直接 memo 或删除；只有新的写入、读取、发布及 UI 协议证明覆盖同等保证后，才能评估替换。固定状态对照身份、来源版本、引用与排序，再用 barrier/event 控制撤销、晚到响应、旧构建回填、新增 override/成员及重启交错。

下一轮测量应分别记录实际依赖数/恢复对象数、每个版本键的重复构建、相关与无关修改的失效范围、修改提交到 UI 撤销/新记录可召回的时延，以及新增写入成本。先证明工作量减少且四契约不退化，再报告冷暖延迟分布；已有 63.89 秒单次样本不能用于承诺提速倍数。
