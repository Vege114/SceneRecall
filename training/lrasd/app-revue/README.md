# 《少女歌剧》第一集运行与性能报告

本目录保存本次运行的离线报告、采集与巡检工具。仅第一集属于完成范围；第二、三集由用户取消，历史请求单列，不计作失败。

- `performance_report.html`：完整中文耗时报告，可直接在浏览器打开。
- `revue-ep1-timing.html`：交互式可视化片段，供 visualize 容器渲染。
- `performance_data.json`、`performance_annotations.json`：报告证据元数据与性能注释；不含视频、模型权重或原始模型响应。
- `performance_collect.py`：只读采集正式库证据；默认路径是本次运行的本机路径，其他机器使用前须检查参数。
- `performance_report.py`、`performance_report_template.html`：从本目录 JSON 构建离线 HTML。
- `stall_watch.py`：只读巡检正式库和结果文件元数据，仅把巡检基线写入工作区。输入及 inflight 活动不等同于成功结果。

在仓库根目录运行（Windows）：

```powershell
.\.venv-windows\Scripts\python.exe training\lrasd\app-revue\performance_report.py
.\.venv-windows\Scripts\python.exe training\lrasd\app-revue\performance_report_test.py
```

最终结果为 88/88 场景完成。报告分别标注墙钟、并发累计请求计时和未知等待；不能据局部微基准宣称整条链路提速。JSON 中的本机路径是证据来源标识，远程检出后不能据此访问原始正式库。虚拟环境、视频、模型、正式库、临时监控状态和原始调用材料不纳入提交。
