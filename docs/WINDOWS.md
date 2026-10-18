# Windows 与 WSL

SceneRecall 主应用支持原生 Windows。`scripts/start_windows.cmd` 使用 Windows PowerShell 和独立 `.venv-windows`，不调用 WSL；`.venv/bin/python` 属于 Linux，不能共用。2026-10-18 本机已有资料库与模型配置已迁到 Windows，完整验证见 [验证记录](VALIDATION.md)。

## 安装与启动

需要 Windows Python 3.12／3.13、uv、Node.js 20+ 和 Windows FFmpeg。在项目根目录运行：

```powershell
.\scripts\start_windows.cmd -Setup -DataDir "D:\SceneRecallLibrary"
.\scripts\start_windows.cmd -Check -DataDir "D:\SceneRecallLibrary"
```

首次用 `-Setup` 安装锁定的主应用与前端依赖并构建页面；它不安装重型模型。以后省略 `-Setup`，前台运行期间保持终端打开，按 Ctrl+C 正常退出。默认 `python` 过旧时可在准备阶段传 `-Python "C:\path\to\Python312\python.exe"`。模型环境另见 [Windows 模型说明](WINDOWS_MODELS.md)。

FFmpeg 与 ffprobe 可放在 `.cache/ffmpeg/bin`，启动器会自动添加到当前进程及其子进程的 PATH；也支持 `-FfmpegDirectory "D:\tools\ffmpeg"` 或已有 PATH。本机已安装 FFmpeg 9.0.2 essentials，来自 [FFmpeg 官方下载页](https://ffmpeg.org/download.html)列出的 [Gyan Windows builds](https://www.gyan.dev/ffmpeg/builds/)，归档 SHA-256 为 `60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba`，下载来源和校验记录保存在本机 `.cache/ffmpeg/install.json`。未修改系统全局 PATH。

## 后台运行与原有入口

```powershell
.\.venv-windows\Scripts\python.exe scripts\manage_windows_runtime.py start --data-dir "D:\SceneRecallLibrary" --port 8765 --open
.\.venv-windows\Scripts\python.exe scripts\manage_windows_runtime.py status --data-dir "D:\SceneRecallLibrary" --port 8765
.\.venv-windows\Scripts\python.exe scripts\manage_windows_runtime.py stop --data-dir "D:\SceneRecallLibrary" --port 8765
```

后台管理器创建隐藏的独立控制台，核实系统用户、进程创建时间、完整启动参数、Python 环境、后代监听及控制台成员后才发送 CTRL_BREAK，等待 Uvicorn 正常收尾；超时不会强杀。`stop --dry-run` 只检查身份。管理器只停止由它启动的实例；前台启动的服务用所在终端 Ctrl+C 退出。

状态和日志存放于 `.cache/runtime`。日志包含本机访问会话链接，应保留在本机；`--open` 自动打开新链接。服务重启后旧网页需要新会话。不要同时用 Windows 与 WSL 后端写同一资料库。

本机 `D:\SceneRecall-Setup\Start-SceneRecall.ps1`／`Stop-SceneRecall.ps1` 已更新为此原生后台管理器；原入口在 `D:\SceneRecall-Backups\windows-migration-20261018\launchers` 中保留。它们不再调用 WSL。独立标注工具的脚本和 Linux 环境不在本次切换范围内。

## 模型缓存与本机设置

默认读取被 Git 忽略的 `.cache/windows-runtime.json`，或用 `-RuntimeConfig` 显式指定。示例：

```json
{
  "environment": {
    "HF_HOME": "D:\\SceneRecall-Cache\\huggingface",
    "TORCH_HOME": "D:\\SceneRecall-Cache\\torch"
  }
}
```

只接受 `SCENERECALL_*` 及 Hugging Face／Torch 缓存、离线环境变量；只作用于本次进程。启动器自动发现 `.venv-visual-windows\Scripts\python.exe`。主应用、视觉与 LR-ASD 使用三个独立原生环境，均不提交仓库，不覆盖原 Linux 环境。LR-ASD 的解释器及模型目录保存在各资产音频设置中；本机迁移保持原模型和 CUDA 配置。

## 从已有 WSL 资料库迁移

先停两端主服务，确认没有活动任务。脚本默认只读 dry-run，检查正在使用的源文件和模型路径，不能把 Linux venv 的 `bin/python` 直接改成不存在的 `Scripts/python.exe`；先安装独立 Windows 环境，再明确映射：

```powershell
.\.venv-windows\Scripts\python.exe scripts\migrate_windows_library.py --library "D:\SceneRecallLibrary" --python-map "/mnt/d/SceneRecall-LR-ASD/.venv/bin/python=D:\SceneRecall\.venv-lrasd-windows\Scripts\python.exe"
```

核对摘要后加 `--apply --backup-dir "D:\SceneRecall-Backups\fresh-migration"`。备份目录必须全新且在资料库外；工具先完整备份、校验库文件及原片 SHA-256，再原子更新操作配置，并保留迁移 journal。`--verify-backup BACKUP_DIR` 可再次核验。只转换明确的运行路径字段，不改原片、字幕、模型权重、人工结果和历史回执正文。

本机备份为 `D:\SceneRecall-Backups\windows-migration-20261018`，验证了 87 个库文件；只更新源媒体位置和 LR-ASD 设置两个 JSON，队列原为 0 项。迁移后已通过正式 API 检查源文件、历史预览和模型路径，并运行一段本地预览。其他库若含暂停任务、旧 identity-prelude 或旧式冒号命名回执，仍需单独验证断点兼容，不能据此保证所有旧任务无缝跨系统恢复。

回退时先停止 Windows 主服务，再运行 `scripts/migrate_windows_library.py --rollback BACKUP_DIR`，并恢复备份的两个 WSL 启停入口。自动回退若发现迁移后配置又被修改，会拒绝覆盖；先保留这些修改并人工核对。原 Linux 解释器、缓存和模型均保留，供排查或回退使用。

## 系统文件窗口

原生 Windows 由 `pythonw.exe` 调用系统 `IFileOpenDialog`，直接显示资源管理器风格窗口，无需 PowerShell。支持视频／外挂字幕／ZIP 筛选、单选／多选、中文和空格路径、初始目录及取消。对话框独立显示，不把任意前台程序当作模态所有者；返回后仍检查本机文件类型与真实路径。

WSL 保留 Windows PowerShell／WinForms 路由，起始目录和返回路径由 `wslpath` 转换；普通 Linux 使用 Zenity，macOS 使用 AppKit。Windows 盘根目录保留 `C:\`，不截成相对路径 `C:`。库内相对路径统一新写入 `/`，可读取既有安全反斜杠引用；盘符、UNC、父级穿越、越界和 ZIP 成员反斜杠仍受严格检查。

## Codex 订阅连接

SceneRecall 通过本机 `codex exec` 复用 Codex CLI 的 ChatGPT 登录；桌面客户端是否打开不直接决定可用性。模型设置中的状态显示后端与 CLI 环境，状态检查不执行推理。

原生 Windows 优先使用 `SCENERECALL_CODEX_PATH`，然后检查 PATH 和桌面 Codex 安装目录；npm `codex.cmd` 解析到原生 `codex.exe`。本机 CLI 0.163.0 已登录，正式接口报告后端／CLI 均为 `windows`、兼容并返回 7 个模型；合成图片与严格 schema 的真实推理通过，无需 WSL 中转，也不复制认证文件。

保留 WSL 时，自动发现 Windows CLI 并转换临时目录、图片、schema 等路径，固定 PowerShell 包装用 Job Object 管理子进程树；Windows CLI 使用 Windows 用户自己的登录及缓存，Linux `CODEX_HOME` 不转发。需要明确选择 CLI 时可在启动前设置 `SCENERECALL_CODEX_PATH`。官方说明见 [Codex CLI](https://learn.chatgpt.com/docs/cli)与[身份验证](https://learn.chatgpt.com/docs/auth)。真实能力测试会消耗订阅额度。
