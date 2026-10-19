# 人工标注、微调与评估

这轮使用《超时空辉夜姬》24 个分散片段（2–6 秒，总计 80 秒），计划文件为
`training/lrasd/manual-kaguya/annotation-plan.json`。场景预先分为 16 个训练、4 个验证、4 个测试；同一场景的所有人物和两秒窗口跟随场景，不随机拆分帧或轨迹。不同划分的原片时间至少间隔 90 秒。

这 24 个片段保守合并为 18 个场景组，比赛、祭典等相近内容保持在同一训练组。原来 20 分钟附近的练习标注完整保留，但不自动并入这轮固定划分；以后如要纳入训练，需要重新检查它们与验证测试场景的距离。推荐队列已按检测框完整度从简单到复杂排序，人物框仍需你核对。

这能测量同一影片的新场景，不能代表其他动画或跨作品效果。现有模型曾用本片少量机器弱标签适配，因此基线也不能称作从未接触本片。准备脚本读取旧适配清单，检查新验证/测试片段是否靠近已知旧窗口，并把已知曝光记录写入报告。

## 你需要完成的内容

按场景清单逐段看画面、听日语原声，检查人物框，填写说话/未说话区间。拿不准保持“不确定”。角色身份未知也可以标 ASD；角色图和字幕不能自动证明谁在说话。确认每条有效人物轨迹后勾选复核、保存。误检测的轨迹可排除；不要为了完成进度把未知帧批量变成未说话。

只有已经保存、明确复核、未排除且带人工输入来源的轨迹可进入数据集。标签 `-1` 始终遮罩，不参与损失、阈值选择和指标；缺框只能保留未知。检测器的框来源仍记录为检测器提议，人工复核不被改称人工绘制。标注后不会自动训练或替换应用模型。

## 检查准备程度

在 WSL 中，从项目根目录运行；每次输出使用新目录，旧报告和导出不会被覆盖。

```bash
cd /mnt/d/SceneRecall
source /mnt/d/SceneRecall-Setup/runtime-env.sh
.venv-visual/bin/python scripts/prepare_lrasd_human_round.py \
  --annotations-root training/lrasd/manual-kaguya/clips \
  --plan training/lrasd/manual-kaguya/annotation-plan.json \
  --output training/lrasd/human-check-v1
```

`readiness.json` 列出缺失片段、未保存复核、全未知轨迹、媒体/时间来源错误等阻塞项。退出码 `2` 表示还需标注；不会假造标签、写回标注或启动训练。输出的计划副本与其 SHA256 固定这一轮划分。若修改场景或划分，必须另起一轮，不能保留旧比较结论。

## 导出冻结数据

所有计划片段完成后，增加 `--export`：

```bash
.venv-visual/bin/python scripts/prepare_lrasd_human_round.py \
  --annotations-root training/lrasd/manual-kaguya/clips \
  --plan training/lrasd/manual-kaguya/annotation-plan.json \
  --output training/lrasd/kaguya-human-round-v1 \
  --export
```

每段按连续 50 帧切成两秒窗口。图像为 112×112 全身灰度裁剪，音频为严格对齐的单声道 16kHz PCM16。尾部不足两秒丢弃并报告；不插框、不补标签、不把不同镜头拼起来。导出保留原片 SHA256、原片时间、音轨、媒体哈希、标注修订号和人工复核信息。

只有计划全部完成，且 train/val/test 分别至少 20/5/5 个有效窗口、每个划分都包含说话和未说话帧，才会标记 `training_ready=true` 并生成 `commands.json`。未满足时仍不训练。

可选 `--official-replay-manifest <官方 CMD-AM manifest>` 加入官方训练集重放；只选择其 `split=train`，保持原标签和逐来源信息。其监督语义仍是官方句级说话人/人物框投影，不能改称人工逐帧真值。人工验证/测试集保持不变。

直接调用旧导出器且不传 `--plan` 仍兼容全部 train 的行为，这类导出不会被当作完整微调轮次。

## 固定顺序：基线、微调、最终比较

导出生成的 `commands.json` 给出三个可直接运行的命令，运行环境为已有的 `/mnt/d/SceneRecall-LR-ASD/.venv/bin/python`：

1. `evaluate_lrasd_human.py` 冻结现有模型，仅跑人工 **val**，保存基线指标和在 val 上选择的阈值，不读取 test 做推理。
2. `train_lrasd.py` 用同一冻结清单的 **train** 微调；从现有完成的 bundle 初始化、重置优化器。默认学习率 `2e-5`、最多 6 轮、早停 3 轮。只按人工 val AP 选择 checkpoint 和阈值，`--defer-final-test` 保留 test。
3. `evaluate_lrasd_human.py --candidate-bundle ... --final-test --baseline-report ...` 同时比较冻结的基线和最佳新模型。每个模型的阈值只从 val 产生，再用于同一 test；不会在 test 上调阈值。此命令核对先前基线报告的模型/数据哈希，以及候选模型确实用本轮数据训练。

报告包含 AP、ROC AUC、F1、精确率、召回率、平衡准确率及混淆计数；逐来源保留人工作品 ASD 与官方投影标签语义，禁止把混合得分称作纯人工 ASD 准确率。这里的逐帧样本有时间相关性，数百帧不等于数百个独立场景。

首次基线评估只用 val；test 留到这轮模型与超参数固定后再执行最终比较。若看到 test 后继续反复调参，这个 test 已经参与选择，应另收集场景做新的最终测试。程序不会自动宣称泛化、自动批准部署或替换应用权重。

准备流程不会因为文件里有模型预测、角色图相似度或字幕说话时间而生成“人工”标签。真实模型效果和是否值得替换，必须等你的人工标注完成后才能知道。
