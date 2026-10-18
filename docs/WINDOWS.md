# Windows 与 WSL

SceneRecall 主应用支持原生 Windows。仓库内的 `scripts/start_windows.cmd` 使用 Windows PowerShell 和独立 `.venv-windows`，不会调用 WSL；已有 `.venv/bin/python` 则属于 Linux/WSL，两种环境不能共用解释器。

## 原生 Windows 启动

在 Windows 安装 uv、Python 3.12/3.13、Node.js 20+，并让 `ffmpeg.exe`、`ffprobe.exe` 在启动终端的 PATH 中可见。在项目根目录运行：

```powershell
.\scripts\start_windows.cmd -Setup -DataDir "D:\SceneRecallLibrary"
```

`-Setup` 按锁文件准备 Python 和前端依赖并构建页面，不安装重型音频／视觉模型。以后省略该参数；检查环境而不启动服务使用：

```powershell
.\scripts\start_windows.cmd -Check -DataDir "D:\SceneRecallLibrary"
```

若本机默认 `python` 是旧版本，可以在准备时指定 `-Python "C:\path\to\Python312\python.exe"`。`-Port 8877` 修改监听端口。启动终端保持打开，按 `Ctrl+C` 停止；停止后用新打印的完整会话链接重新打开页面。

Python 依赖存放于 `.venv-windows`，uv 缓存存放于 `.cache/uv`，不会覆盖 `.venv`、`.venv-visual` 或 LR-ASD 的已有 Linux 环境。这些环境和缓存均不提交 Git。

## 已有 WSL 用户

若原启动入口内部调用 `wsl.exe ... start-scenerecall.sh`，应用后端实际运行于 WSL；浏览器在 Windows 打开不改变后端系统。可以保留现有入口，并在更新后重启后端使用本次桥接。

WSL 下的视频、字幕、原文件重新定位和 ZIP 恢复统一通过 Windows PowerShell 打开资源管理器样式的系统文件窗口。多选、类型筛选和取消行为与原生 Windows 一致。窗口使用置顶所有者，中文输入／返回采用 UTF-8；起始目录与返回路径经 `wslpath` 转换，后端仍校验真实文件及扩展名。不需要通过 Linux Zenity 选 Windows 文件。普通 Linux 桌面仍使用 Zenity。

Windows 盘根目录必须保留分隔符，例如 `C:\`，不能变成相对目录 `C:`。应用现已保留盘根和 UNC 共享根，避免再次选择视频或字幕时被后端拒绝。

已有 WSL 资料库的源视频和本地模型配置可能保存 `/mnt/d/...`、Linux 解释器或模型目录。切换原生 Windows 不会自动迁移这些路径，也不会转换模型依赖。需要立即继续使用已有资料与重型模型时，沿用 WSL 桥接；新建原生 Windows 资料库时使用 Windows 路径。不要同时用两个后端写同一资料库。

## Codex 订阅连接

SceneRecall 通过本机 `codex exec` 复用 Codex CLI 的 ChatGPT 登录，不连接桌面窗口的聊天进程。桌面客户端是否打开不直接决定连接是否可用。模型与设置中的状态检查会显示后端运行环境与实际 CLI 位置；状态检查不执行模型推理。

原生 Windows 优先使用显式的 `SCENERECALL_CODEX_PATH`，再查 PATH 和桌面 Codex 的常见安装目录。npm 的 `codex.cmd` 会解析到包内原生 `codex.exe`，不会通过 `cmd.exe` 重新解释模型参数和路径。

WSL 自动发现 Windows Codex CLI，并使用 Windows 用户自己的登录和模型缓存；找不到时保留 Linux CLI 回退，`SCENERECALL_CODEX_PATH` 可明确指定需要的 CLI。Windows 桥接将请求临时目录、图片、结构化输出 schema 与指令文件路径转换为 Windows 可访问路径，提示词以数据通过 stdin 传输。固定 PowerShell 启动器通过 Windows Job Object 管理本次 CLI 子进程，超时和取消会回收整棵进程树。Linux 的 `CODEX_HOME` 不传给 Windows CLI；应用不读取、复制或保存认证文件。

若 WSL 禁用了 Windows 程序互操作，需恢复互操作后使用桥接，或在 WSL 内安装并登录 Linux Codex CLI。原生 Windows 可在启动前设置：

```powershell
$env:SCENERECALL_CODEX_PATH = "C:\path\to\codex.exe"
.\scripts\start_windows.cmd -DataDir "D:\SceneRecallLibrary"
```

Codex 版本、ChatGPT 登录、所需执行参数均须通过状态检查；真实能力测试会消耗订阅额度。安装和登录的官方说明见 [Codex CLI](https://learn.chatgpt.com/docs/cli)与[身份验证](https://learn.chatgpt.com/docs/auth)。本次实际验证范围及尚未验收项见[验证记录](VALIDATION.md)。
