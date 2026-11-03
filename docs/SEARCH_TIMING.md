# 搜索耗时记录

搜索结果下方的「搜索总耗时与阶段记录」展示本次前端与后端的实际计时，并可下载 JSON。计时不改变查询、排序、缓存或模型选择，不会因打开诊断而调用模型。记录仅包含随机请求 ID、查询字符数、类型、筛选数量、返回数量、阶段时长和缓存状态；不保存查询原文、角色名、路径、凭据或图片。记录随当前结果保留在页面内存，用户可主动导出。

## 测量边界

| 记录 | 实际边界 |
| --- | --- |
| 点击到结果 DOM 可用 | 指针点击搜索按钮产生的 `click` 事件时间戳，到 React 提交本次结果 DOM。时间戳统一到浏览器 performance 时钟；不兼容时退回点击处理函数入口，并以 `click_time_source` 标明。包含点击事件分发到表单提交的时间，不包含按下指针到松开生成 click 的时间，也不含缩略图加载。键盘激活和程序提交使用 `trigger=form_submit`，不伪造点击指标。 |
| 点击到表单提交／绘制机会 | `click_to_submit_ms` 补足原提交计时之前的区间；`click_to_paint_opportunity_ms` 到两次 requestAnimationFrame 的机会点。前台页面可减少浏览器节流，但仍不证明图片已加载或像素实际显示。 |
| 提交到结果 DOM 可用 | 浏览器的表单提交处理函数进入，到 React 提交本次结果 DOM。包括组装请求、请求等待、解析和界面提交；不含用户输入过程、事件进入处理函数之前的等待或缩略图加载。当前没有查询输入防抖，`debounce_ms=0`。 |
| 提交到绘制机会 | 提交到两次 `requestAnimationFrame` 后的机会点；不是画面已实际显示或图片加载完毕的保证。后台标签页的回调可能延迟。 |
| 前端准备请求 | 提交处理函数进入到调用 `fetch`。包含组装与序列化请求、同步页面状态更新。 |
| 请求发起到响应头 | `fetch` 调用到其 Promise 返回响应。包含浏览器调度、连接与传输、未单独测得的服务入站等待以及后端处理。 |
| 响应头到 JSON 已解析 | 收取响应体、JSON 解析及 JavaScript 调度。不能将其全部解释为下载速度。 |
| 后端应用总时长 | 搜索计时中间件入口到响应头可交付；包含会话检查、路由、搜索、来源核对、JSON 序列化。操作系统、ASGI 接收前或事件循环进入该中间件之前的等待无法单独区分。 |
| 请求到 JSON 的剩余项 | 前端请求到 JSON 的持续时间减去后端应用持续时间；包含传输、未测入站等待、响应体、解析与调度，不能称为「纯网络」。这里只比较持续时间，不跨前后端时钟相减时间戳。 |
| 浏览器资源计时 | 浏览器能提供本次完整 `PerformanceResourceTiming` 时，单独列出请求前准备、请求到首字节、响应体传输；未提供时明确留空，不填零。首字节时间包含服务端处理。 |

## 后端阶段

`X-Search-Request-ID` 关联请求；`X-Search-Timing` 返回 `search-timing-v1` JSON；`Server-Timing` 返回后端应用总时长。前端 API 读取最终响应头，获得包含 `response.json` 的完整记录。JSON 正文本身继续保持原搜索协议。无需读取进程全局「最后一次查询」来取得记录，因此同时搜索的阶段不会串到另一条请求上。

每个 span 包含 `id`、`parent_id`、`name`、相对请求起点的 `start_ms` / `end_ms`、`elapsed_ms` 和实际状态。所有后端偏移使用同一进程的单调时钟，线程中的记录也使用同一时钟。父阶段包含子阶段，四路召回并行，**不要把子阶段与父阶段相加，也不要把四路召回直接相加当成总时长**。比较区间可识别重叠；顶层区间之外的空白仍是未进一步拆分的应用调度／准备开销。

- `snapshot.*`：取得与验证索引、作品资料和编码器版本；`.work` 是实际执行线程内的区间。
- `query.character_names`：按检索范围解析角色姓名、简称和别名；`query.character_names.revision` 核对名录版本。状态只记录命中数量、是否展开等信息，不含姓名或查询原文。
- `recall.parallel` 与 `recall.lexical/semantic/visual/concept`：完整并行召回墙钟以及各路召回区间。未配置分支仍显示短区间及零候选。
- `lexical.sql_tokenize.work`：SQLite 字面检索、分词及结果读取。
- `semantic.*`：文字语义索引快照、实际查询编码及索引读取排序。没有文字语义模型时不会伪造编码阶段。
- `visual.model_identity` / `encoder.identity.work`：本地图文模型文件身份与运行依赖核对。身份核对可命中本进程内部文件哈希缓存，该区间不等同模型加载。
- `visual.query_cache_lookup`：磁盘查询向量缓存命中状态。命中时不出现 `visual.query_encode`。
- `encoder.prepare_and_wait`：编码器请求线程开始到启动／复用进程之前，包括请求锁与本地模型资源等待、读取设置、核对模型身份；这些部分未单独拆分，不能把整个区间称为纯排队。`visual.query_encode` 到 `encoder.request.work` 之间还包括异步线程调度。
- `encoder.worker_start` / `encoder.worker_exchange`：实际启动或复用进程，以及向编码进程发送请求到收到结果。`worker_model_load_ms` 只在这次启动了新进程时引用 worker 测得的模型加载持续时间，热进程记零；`worker_inference_ms` 来自这次 worker 编码。worker 持续时间只作区间内说明，不把其时钟偏移放入父进程时间线。
- `visual.matrix_cache`：矩阵命中、是否实际构建与流式回退。`fallback_reason` 区分 `cache_disabled`（缓存预算为零）、`retained_budget_exceeded`（保守留存估算超过字节预算）、`allocation_error`（确实捕获 Python `MemoryError`）、`builder_declined`（其他构建器明确返回空，原因未细分）。`memoized_fallback=true` 表示复用该索引版本与范围先前记下的回退结果，本次未重建；它不会把上次预算回退改称内存分配失败。`cache_budget_bytes` 是本次上限；存在时 `attempted_retained_bytes` 是构建器最后已计算的保守估算，`retained_bytes_phase` 指明记录元数据或向量阶段，并非系统实测 RSS 或完整索引的最终内存量。`visual.matrix_build.work` 是确实发生的索引解码与矩阵构建；`visual.matrix_score`、`visual.top_results` 分别计相似度计算、排序与结果读取。现有重试会全部保留在时间线内，`snapshot_retries` 是实际重试次数，父级总时长包括这些尝试。
- `results.*`：各路合并排序、可选模型重排、展示去重与选择、可选解释生成。禁用模型重排时该区间标明 `enabled=false`。
- `canonical.source_records` / `.work`：读取当前原始证据，父区间包含线程调度；`canonical.enrich` 核对与补齐来源、播放定位和展示字段。
- `visual_verification`：只有主动启用候选视觉复核时出现。
- `response.json`：实际 JSONResponse 序列化，包含响应字节数。

编码器计时通过独立适配器实现，原 `visual_encoder.py` 与 worker 的实现指纹保持不变。增加计时不会使已有视觉向量、磁盘查询缓存和任务快照失效；实际权重、预处理或编码实现变化仍按原规则隔离。

## 复测方法

先等待分析、索引和后台准备任务完成，记录代码提交、库大小、模型／硬件和是否仍有任务运行。分别选择台词、画面和角色查询，每种记录第一次与多次重复请求，保存请求 ID 和导出的 JSON。查询家族由测试者标注，程序不通过语言猜测「角色查询」。

首次请求不必然是冷模型：启动准备可能已预热模型，磁盘查询向量缓存也可能来自之前的会话。按 `query_cache_hit`、`matrix_cache_hit`、`matrix_built`、`worker_started` 等实际状态描述每条；不要为了制造冷状态随意删除共享缓存。若已确认没有运行任务而重启服务，应注明重启只清除进程内状态，不清除磁盘缓存；首次进程内矩阵构建和首次模型加载也可能出现在不同请求。真实测试结果另随本轮验证文档记录，不用早期人工观察的宽泛秒数区间推断瓶颈或宣称提速。

最新实测见 [2026-11-03 简称与全名一致性验证](SEARCH_QUERY_EQUIVALENCE_20261103.md)：在同一作品、相同筛选下从前端分别点击搜索，保存实际响应并深比较完整有序结果。台词、画面和混合模式的姓名解析与别名扩展一致；解析本身不增加视觉模型查询次数。计时差异与结果差异应分别判断。
