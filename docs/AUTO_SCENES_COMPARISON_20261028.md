# Auto-scenes-extraction 与 SceneRecall 本机对比

日期：2026-10-28。结论：**在本次相同90秒片段、基础图文检索任务中，Auto-scenes-extraction 的任务耗时约为 SceneRecall 的29%，查询也更快；质量各有取舍，现有开发标注不足以证明谁整体更准确。** 对方的流水线和拒绝低分结果的做法值得借鉴，不能据此认定替换数据库或模型就能复现全部收益。

本文保存当次部署与实验的历史记录。下文 `test-results/` 中的原始数据、日志、图片和实验脚本，以及 `.cache/` 中的外部项目部署，均为本地忽略内容，**未随仓库发布**。代码标记中的路径均相对于仓库根目录，其他环境需自行准备相应素材和外部项目。

## 本地部署与使用

- 源码目录：Auto-scenes-extraction（`.cache/benchmarks/Auto-scenes-extraction/`）。上游：[TuanAMV/Auto-scenes-extraction](https://github.com/TuanAMV/Auto-scenes-extraction)，固定提交 `7ff094a458307365b64dc29137e41caf58e276a7`。
- 语义搜索已启动：[http://127.0.0.1:5006](http://127.0.0.1:5006)。索引流水线已启动：[http://127.0.0.1:5005](http://127.0.0.1:5005)。仅监听本机。
- 重启入口：start_search_local.cmd（`.cache/benchmarks/Auto-scenes-extraction/start_search_local.cmd`）、start_pipeline_local.cmd（`.cache/benchmarks/Auto-scenes-extraction/start_pipeline_local.cmd`）。已有服务运行时不必重复启动。
- 已放入本次片段的 Lance 索引，并保存“本机基础检索对比”“本机基础索引对比”预设。搜索选择 `qihoo360-fg-clip2-base` 的索引，阈值10，输入“建筑外面的红色金属楼梯”即可复现。
- 两个页面、模型列表、真实中文搜索及结果图片预览均通过 HTTP 验证，证据：http-smoke.json（`test-results/auto-scenes-comparison-20261028/http-smoke.json`）。

使用官方 [qihoo360/fg-clip2-base](https://huggingface.co/qihoo360/fg-clip2-base)，固定 revision `430fbc8a912c86fd4de601381b6245a0edab22f0`；权重约1.54 GB，下载约202秒，未计入推理耗时。该分支可直接做中文查询，避免拿英文 CLIP 的语言限制与中文检索直接比较。

实际运行库为 Python 3.12.6、PyTorch 2.8.0+cu128、Transformers 4.57.1。外部项目有单独 `.venv`，通过 `.pth` 复用现有检索环境的 CUDA/PyTorch 等基础库，Flask、pylance、pyarrow 等装在其自身环境；因此移动部署时也须保留 `.venv-retrieval-windows`。两部分依赖版本已分别保存。

克隆版不包含 README 提到的嵌入式 Python、FFmpeg 和模型，原 bat 无法直接用。本次补充了启动器、依赖和本机 FFmpeg。FFmpeg 9 已移除上游仍使用的 `-vsync`，首次原样运行失败；已将6处调用改为等价的 `-fps_mode vfr/passthrough`，补丁见 deployment-compatibility.patch（`test-results/auto-scenes-comparison-20261028/deployment-compatibility.patch`）。没有修改抽帧、切景、编码、排名算法。

**本次部署和实测覆盖索引、中文语义搜索、预览。MiniCPM 自动提纯、Qwen 重排、景别分析未下载/运行，不能将其宣传效果计入本次质量结论。**

## 实验口径

- 硬件：Windows 11，RTX 4070 Laptop GPU，8188 MiB 显存，驱动610.62；两边均 CUDA FP16，串行占用 GPU。
- SceneRecall 提交：`029c3902ad2df1b93f3963eac30997fd0a80f49f`。编码器为现行 SigLIP2 base-patch16-256，revision `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`，batch16。
- 同一《超时空辉夜姬》原片181.306–271.306秒，先制成双方共用的 H.264 CRF0 无损视频；实际媒体时长90.007秒，SceneRecall分析0–90.000秒，对方处理完整片段，末尾差7毫秒。文件 SHA256 `7f395737e45b163218ea4eb37b0080ae58dee5bcd4a04d7df667b7e60ada8fa0`。片段准备25.919秒，双方均不计入建库时间。
- 两边各3次新进程、空索引、无图像特征缓存。操作系统文件缓存未清空，不能称磁盘冷启动；运行顺序是 SceneRecall 三轮后对方三轮，没有随机交替，不能据此作精细的模型加载速度排名。
- 当前实际使用记录已核对为 `retrieval-first-v1/core`。本次复用正式 core 实现，但隔离库未复制人物图库，身份阶段关闭；没有远程模型调用。因此不是带 CCIP、人物确认和生成式理解的完整任务耗时。
- 对方使用随仓库预设的 sample_interval=3、短边256、阈值0.85、localmax_order=1、最短7源帧、黑白帧过滤、batch128、8个I/O线程。sample_interval=3 是每3个源帧，约8fps，**不是每3秒**。batch128也是上游函数/提供预设中的值，索引网页初始值和 README 可能不同。

## 速度实测

主表为**应用顶层模块导入之后的任务墙钟，包含各自模型准备、处理和索引写入**；下载、安装、片段准备不计入。SceneRecall 的 worker 启动/导入仍计在其任务内，这是实际进程架构产生的开销。对方顶层导入提前完成，另列在后表；不能把主表解读为从双击程序开始的启动加速。

| 项目 | SceneRecall | Auto-scenes / FG-CLIP2 |
|---|---:|---:|
| 第1次任务 | 57.360 s | 13.405 s |
| 第2次任务 | 41.293 s | 11.299 s |
| 第3次任务 | 39.442 s | 11.866 s |
| **任务中位数** | **41.293 s** | **11.866 s** |
| 首次视觉可搜，中位数 | 29.037 s，逐步发布 | 完整索引约11.866 s后可搜 |
| 实际编码帧数 | 533 | 625（720采样，过滤95） |
| 持久化检索向量 | 533 × 768维 FP32 | 69 × 768维 FP16 |
| 检索单元 | 18个默认6秒窗口 | 23个场景，每景3个特征 |
| PyTorch峰值分配 | 0.80 GiB | 1.80 GiB |

该任务口径下为 **3.48倍吞吐，耗时减少71.3%**。显存是模型进程的 PyTorch 分配峰值，不是整机显存占用；较快的外部流程使用了更多模型显存。

| 计时边界补充 | SceneRecall | Auto-scenes |
|---|---:|---:|
| 顶层导入前至构建完成 | 本次脚本未独立记录此精确边界 | 18.685 / 17.163 / 16.500 s |
| 外层进程总时长 | 59.681 / 42.553 / 40.486 s，**还含8问和退出** | 未用同一边界记录 |
| 对方顶层模块导入 | — | 5.280 / 5.864 / 4.634 s |
| 记录的模型准备中位数 | 8.094 s，含校验、ML导入及加载 | 1.948 s，ML顶层导入已发生 |

不要把这几种边界混算成“整体冷启动快几倍”。也没有把“总耗时减模型加载”冒充同进程新内容的真实热启动实验。

查询均先暖模型；每轮8问，各重复3次，均含真实索引读取和排名：

| 查询中位数 | SceneRecall原生（含融合） | SceneRecall纯视觉 | Auto-scenes |
|---|---:|---:|---:|
| 未缓存该文本 | 41.17 ms | 35.25 ms | 9.52 ms |
| 重复查询 | 23.66 ms | 17.58 ms | 9.43 ms |

SceneRecall重复查询复用文本向量；对方仍每次编码文本、整表读Lance、上传GPU。对方本片只检索69个向量，SceneRecall检索533个；不是同等数据量的数据库性能测试。两边都没有在本次纯视觉路径中使用 ANN。

## 质量：看到什么，以及不能据此推断什么

采用原有6个开发正例、2个无答案查询，保留全部原始输出和时间诊断：quality-summary.md（`test-results/auto-scenes-comparison-20261028/quality-summary.md`）。主配置中，双方前5候选与旧预标区间相交均为6/6，三轮各自排名、分数及时间均一致。**这个6/6不是准确率，也不能证明全部语义正确。**

实际返回帧对照保存在本地实验目录（左列是旧标注区间中点，仅作参考；不是人工真值图），图片未随仓库发布：

六题首条结果画面对照（`test-results/auto-scenes-comparison-20261028/top1-visual-review.jpg`）

对实际返回帧作目视复核后，有以下具体发现：

1. 红色楼梯、红色售货机、手机音乐播放界面、街边银发角色大屏幕，两边首条都能找到相关画面。
2. 旧“售货机”标注从片段4.694秒开始，但 SceneRecall 的4.046秒帧中售货机已清晰可见；时间点落在旧区间外并不等于误检。
3. “紫色短发女生在室外看手机”，对方首条5.881秒清楚显示女生、手机和街头环境，却落在旧标注10.694–30秒之外。SceneRecall首条20.103秒更偏手机近景。这条说明**只算命中旧时间区间会误判质量**。
4. “两个学生在夜晚街道”一题，旧标注中点及双方首条帧都只清楚见到一名学生，不能用该标注宣称复合条件识别通过，需要重新核对和标注。
5. 两条无答案“绿色恐龙驾驶飞机”“使用能停止时间的秒表”：SceneRecall均返回5个候选，其中原生搜索带partial等证据状态，不能把partial当成已确认正确或已拒答；对方阈值10时均不返回。**本次对方的低分过滤更利于避免无答案查询产生候选**，但只有2题，尚未标定拒答能力。
6. 对方阈值提高到12，会没有“女生看手机”的返回；提高到14，又丢掉“售货机”。阈值10、12、14都能挡住这两条负例，不能仅看拒答而忽略正例损失。不同模型分数不在同一尺度，不能照搬“10”到SigLIP。

外部产品只返回场景范围，没有最佳命中帧。结果JSON中的 `evaluation_best_feature_ms` 是本次评价从原三向量重建的最大特征时间，**不是该产品原有定位功能**；只有两帧的场景，中间平均向量甚至没有对应真实帧。SceneRecall则原生返回 `seek_ms`。IoU也受固定6秒窗口和旧目标区间长短影响，仅作为诊断，没有拿它作为总体质量排名。

另做一次只修改采样密度的消融：sample_interval=12（约2fps），模型/阈值等不变，任务7.856秒，编码155帧，19景/57向量；阈值10时“售货机”0返回，旧标注Top5重叠从6/6变5/6。完整消融（`test-results/auto-scenes-comparison-20261028/quality-interval12.md`）。说明稀疏采样会损失线索；这是单次消融，不是最终采样参数选择。

## 差异具体来自哪里

| 环节 | SceneRecall当前实现 | 对方实现及实际影响 |
|---|---|---|
| 解码/抽帧 | 先逐帧低分辨率扫描画面变化、周期和局部变化选帧，再按源PTS精确抽原尺寸JPEG；选帧后另有抽图开销 | FFmpeg一次连续解码，定距采样时直接缩小到454×256并写JPEG，同时GPU编码；大部分工作重叠 |
| 采样和编码流水 | 选帧阶段中位15.235秒；视觉索引22.046秒，内含模型准备8.094秒。每16图串行送worker再写库，未显式双缓冲 | 抽帧加编码整体中位9.416秒，之后切景等约0.27秒；I/O线程池、双缓冲预取。**FG-CLIP2内部仍逐图推理**，不能归功于“一次128图GPU批推理” |
| 场景形成 | 像素内容切镜和时间采样结合，检索按默认6秒窗口；保留更密的证据帧及真实时间 | 相邻语义embedding低于0.85且为局部最小才切分；只存首中尾。相似语义镜头可能合并，动作只在非三点时刻出现可能漏掉 |
| 图像模型 | SigLIP2整图缩到256×256，使用较慢的processor；小batch16 | FG-CLIP2动态长宽比patch；实测输入原图454×256，模型看到18×32=576个patch（不是简单224方形，也不只是README推算的28×16） |
| 模型准备 | 独立worker、权重哈希复验、依赖身份检查、ML导入和加载计入任务 | Flask模块导入阶段已加载ML软件库，模型在主进程直接构建；更轻的任务启动路径，但校验和隔离契约不同 |
| 特征传输/存储 | FP16推理后FP32归一化，经CPU list/JSON管道，SQLite FP32 BLOB，保留533条 | 以FP16二进制写Lance，只留69条；仅向量payload约少15.4倍，信息量也少了，不能称相同信息的无损压缩 |
| 查询 | SQL过滤后CPU NumPy分块扫描、窗口内max，再与字幕等多路RRF融合 | Lance整表读取、所有向量上传GPU、矩阵乘、每场景三帧max；本片更快，但库大时仍随向量总量增长，不是ANN |
| 拒答 | 视觉相似度为正即可进入候选，partial表明证据不足，尚缺统一无答案标定 | 用 `logit_scale × cosine` 显式截断；减少低分候选，也会漏正例 |

还有一个直接可行动的遗漏：普通关键帧任务已接入的 **PyAV优化没有接到 retrieval-first 这条采样路径**。`retrieval_jobs.py` 调 `scan_keyframes` 没有传decoder，仍用FFmpeg默认值。不能把历史普通关键帧任务的提速当成当前检索入库已经获得的收益。

对方场景边界有采样量化，结束取下一边界前的采样帧，存在帧间空隙；并非帧级连续镜头真值。23景里还有长约15.64秒的场景。对方“手机音乐界面”首条范围约17.392–27.402秒，SceneRecall为24–30秒，说明“找到相关帧”和“剪出干净片段”是两件需要分别验收的事。未制作人工切镜真值、未比较导出视频，不宣称谁切镜/剪辑总体更准。

以上是实测结合代码的归因；由于模型、采样、保留信息和进程契约同时不同，**没有单独量化每一项的因果贡献**。特别不能将总差距都归给 Lance、GPU排名或batch大小。详细源码证据：SceneRecall审计（`test-results/auto-scenes-comparison-20261028/scenerecall-code-audit.md`）、外部审计（`test-results/auto-scenes-comparison-20261028/autoscenes-code-audit.md`）。

README还存在需要注意的偏差：大batch描述不适用于FG逐张分支；其自报索引时间漏模型加载及末尾写盘等待；单句 `text_search` 的Qwen重排开关没有实际Qwen重排调用，实际实现的另一分支是MiniCPM首帧验证。未执行的可选功能不能加到本次实测质量上。

## 对 SceneRecall 的改进优先级

1. 先把已验证的PyAV扫描接到retrieval-first，并减少扫描后再解码抽JPEG的重复工作；这是当前15秒采样阶段最直接的优化入口。保留源PTS和原图证据契约，做输出等价性验证。
2. 将读图/预处理和GPU编码流水化，实测batch32/64；比较processor快慢实现的图像/向量差异，不直接切换。FG分支并不是真批量，没必要照搬它的逐图循环。
3. 避免每批向量转Python list再JSON传输；缓存已验证的模型身份，并控制复验时机。进程隔离和内容完整性检查仍有价值，应测量后优化，不能直接删除。
4. 把“检索代表帧”和“详细证据帧”分开维护：检索先扫更小的代表集合，命中后读取密帧。以短动作、边缘物件作为不漏检约束，避免固定只留三帧造成退化。
5. 重标本次失真的开发查询，补充跨作品留出集和足量无答案题，再给SigLIP分数做拒答/复核策略；不能把对方阈值数字直接移植。FG-CLIP2可作为后续同采样同索引的模型对照，本次没有证明更换它必然提升整体质量。

## 原始记录和复现

结果根：auto-scenes-comparison-20261028（`test-results/auto-scenes-comparison-20261028/`）。包含全部逐题输出、每轮命令、依赖版本、共享视频哈希、补丁、HTTP验证、可视复核图和评价脚本。

- comparison-summary.json（`test-results/auto-scenes-comparison-20261028/comparison-summary.json`）：汇总速度与阶段。
- SceneRecall运行记录（`test-results/auto-scenes-comparison-20261028/scenerecall/run-summary.json`）：3轮任务/查询，正式库保护文件前后哈希一致。
- 外部原生benchmark脚本（`test-results/auto-scenes-comparison-20261028/run_autoscenes_benchmark.py`）：直接调用上游 `process_video`、`scenes_to_lance_rows`、`AsyncLanceWriter` 和 `SemanticSearchEngine`；只加计时、写入校验和评价取证。成功主轮编号r2/r3/r4；r1为FFmpeg旧参数失败，未纳入统计。
- quality-summary.json（`test-results/auto-scenes-comparison-20261028/quality-summary.json`）：时间区间指标和三轮一致性，未冒充语义准确率。

本次没有修改SceneRecall应用代码或正式资料库。结果是单片段、开发查询、小规模索引的本机评估，不外推整集HEVC吞吐、大库查询扩展性、人物识别、生成式理解或MiniCPM最终复核质量。
