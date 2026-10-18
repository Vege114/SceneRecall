"""Open the operating system's file chooser without uploading local media."""
from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EXTENSIONS = {
    "video": {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm", ".mpg", ".mpeg",
              ".ts", ".m2ts", ".mts", ".wmv", ".flv", ".ogv", ".vob"},
    "subtitle": {".srt", ".ass", ".vtt"},
    "backup": {".zip"},
}
TITLES = {"video": "SceneRecall · 选择视频", "subtitle": "SceneRecall · 选择外挂字幕", "backup": "SceneRecall · 选择备份"}
_dialog_lock = threading.Lock()


class FilePickerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["video", "subtitle", "backup"] = "video"
    multiple: bool = Field(default=False, strict=True)
    directory: str | None = Field(default=None, min_length=1, max_length=32768)

    @model_validator(mode="after")
    def single_backup(self):
        if self.kind == "backup" and self.multiple:
            raise ValueError("备份恢复一次只能选择一个 ZIP 文件")
        return self


class RestoreLocalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=32768)


class FilePickerBusy(RuntimeError):
    pass


class FilePickerUnavailable(RuntimeError):
    pass


def _is_wsl() -> bool:
    return sys.platform.startswith("linux") and bool(
        os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP")
        or "microsoft" in platform.release().lower())


def _wsl_path(raw: str, target: str) -> str:
    """Use WSL's mount configuration, rather than assuming drives live in /mnt."""
    command = shutil.which("wslpath")
    if not command:
        raise FilePickerUnavailable("WSL 文件选择需要 wslpath，请检查 WSL 安装")
    result = _run([command, target, raw])
    value = result.stdout.removesuffix("\n")
    if result.returncode or not value or "\x00" in value:
        raise FilePickerUnavailable("所选 Windows 路径无法由 WSL 访问，请选择已挂载的本机文件")
    return value


def _requested_local_path(raw: str) -> Path:
    if "\x00" in raw or "://" in raw:
        raise ValueError("请选择本机文件或目录")
    if _is_wsl() and PureWindowsPath(raw).is_absolute():
        raw = _wsl_path(raw, "-u")
    requested = Path(raw).expanduser()
    if not requested.is_absolute():
        raise ValueError("请选择本机文件或目录的绝对路径")
    return requested


def local_path(raw: str) -> Path:
    return _requested_local_path(raw).resolve(strict=True)


def starting_directory(raw: str) -> Path:
    """A moved video should not prevent opening the chooser to locate it."""
    requested = _requested_local_path(raw)
    while True:
        try:
            directory = requested.resolve(strict=True)
        except FileNotFoundError:
            if requested.parent == requested:
                return Path.home().resolve(strict=True)
            requested = requested.parent
            continue
        except NotADirectoryError as exc:
            raise ValueError("文件选择的起始位置必须是目录") from exc
        if not directory.is_dir():
            raise ValueError("文件选择的起始位置必须是目录")
        return directory


def local_file(raw: str, kind: str) -> Path:
    path = local_path(raw)
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS[kind]:
        raise ValueError({"video": "请选择支持的视频文件", "subtitle": "请选择 SRT、VTT 或 ASS 字幕文件",
                          "backup": "请选择 ZIP 备份文件"}[kind])
    return path


# Only fixed code is passed to the interpreter. Filenames and other input are
# serialized separately, including Unicode, quotes and embedded newlines.
_MAC_SCRIPT = r'''
ObjC.import('AppKit');
function run(argv) {
    const options = JSON.parse(argv[0]);
    // Present directly from this process. Finder Apple Events can wait behind
    // Finder's own work and require an unrelated Automation permission prompt.
    const application = $.NSApplication.sharedApplication;
    application.setActivationPolicy($.NSApplicationActivationPolicyAccessory);
    const panel = $.NSOpenPanel.openPanel;
    panel.title = options.title;
    panel.canChooseFiles = true;
    panel.canChooseDirectories = false;
    panel.allowsMultipleSelection = options.multiple;
    // allowedFileTypes is available on older macOS versions too, and accepts
    // extensions such as MKV and ASS without relying on installed UTI handlers.
    panel.allowedFileTypes = options.extensions;
    panel.allowsOtherFileTypes = false;
    panel.treatsFilePackagesAsDirectories = false;
    if (options.directory) panel.directoryURL = $.NSURL.fileURLWithPath(options.directory);
    application.activateIgnoringOtherApps(true);
    if (Number(panel.runModal) !== Number($.NSModalResponseOK)) {
        return JSON.stringify({paths: [], cancelled: true});
    }
    const urls = panel.URLs;
    const paths = [];
    for (let index = 0; index < Number(urls.count); index++) {
        paths.push(ObjC.unwrap(urls.objectAtIndex(index).path));
    }
    return JSON.stringify({paths: paths, cancelled: false});
}
'''

_WINDOWS_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Windows.Forms
$options = [Console]::In.ReadToEnd() | ConvertFrom-Json
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$owner = New-Object System.Windows.Forms.Form
try {
    # A background web server has no foreground window to own the chooser.
    # An invisible topmost owner keeps the Explorer dialog above the browser.
    $owner.ShowInTaskbar = $false
    $owner.TopMost = $true
    $owner.Opacity = 0
    $owner.Show()
    $owner.Activate() | Out-Null
    $dialog.Title = $options.title
    $dialog.Multiselect = $options.multiple
    $dialog.CheckFileExists = $true
    $dialog.AutoUpgradeEnabled = $true
    $dialog.RestoreDirectory = $true
    $dialog.Filter = 'SceneRecall|' + (($options.extensions | ForEach-Object { '*.' + $_ }) -join ';')
    if ($options.directory) { $dialog.InitialDirectory = $options.directory }
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        @{paths=@($dialog.FileNames);cancelled=$false} | ConvertTo-Json -Compress
    } else {
        @{paths=@();cancelled=$true} | ConvertTo-Json -Compress
    }
} finally {
    $dialog.Dispose()
    $owner.Dispose()
}
'''


def _run(command: list[str], *, input: str | None = None):
    try:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        return subprocess.run(command, input=input, capture_output=True, text=True, encoding="utf-8", check=False,
                              **options)
    except OSError as exc:
        raise FilePickerUnavailable("无法打开系统文件选择窗口，请检查本机桌面环境后重试") from exc


def _json_result(result):
    if result.returncode:
        raise FilePickerUnavailable("无法打开系统文件选择窗口，请检查本机桌面环境后重试")
    try:
        value = json.loads(result.stdout.lstrip("\ufeff"))
        if not isinstance(value, dict) or not isinstance(value.get("cancelled"), bool):
            raise ValueError
        if not isinstance(value.get("paths"), list) or any(not isinstance(path, str) for path in value["paths"]):
            raise ValueError
        if value["cancelled"] and value["paths"]:
            raise ValueError
        return value
    except (ValueError, TypeError) as exc:
        raise FilePickerUnavailable("系统文件选择窗口返回了无法读取的结果，请重试") from exc


def _open_windows_dialog(options: dict, *, wsl: bool = False):
    names = ("powershell.exe", "pwsh.exe") if wsl else ("powershell", "pwsh")
    command = next((found for name in names if (found := shutil.which(name))), None)
    if not command and wsl:
        # WSL may disable imported Windows PATH entries while interop is enabled.
        candidate = _wsl_path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe", "-u")
        if Path(candidate).is_file():
            command = candidate
    if not command:
        raise FilePickerUnavailable("WSL 文件选择需要 Windows PowerShell，请启用 WSL 的 Windows 程序互操作"
                                    if wsl else "系统文件选择需要本机 PowerShell")
    if wsl and options.get("directory"):
        options = {**options, "directory": _wsl_path(options["directory"], "-w")}
    encoded = base64.b64encode(_WINDOWS_SCRIPT.encode("utf-16-le")).decode("ascii")
    return _json_result(_run([command, "-NoProfile", "-STA", "-EncodedCommand", encoded,
                              "-NonInteractive", "-OutputFormat", "Text"],
                            input=json.dumps(options, ensure_ascii=False)))


def _open_dialog(options: dict):
    if sys.platform == "darwin":
        return _json_result(_run(["/usr/bin/osascript", "-l", "JavaScript", "-e", _MAC_SCRIPT,
                                  json.dumps(options, ensure_ascii=False)]))
    if sys.platform == "win32":
        return _open_windows_dialog(options)
    if sys.platform.startswith("linux"):
        if _is_wsl():
            return _open_windows_dialog(options, wsl=True)
        separator = "\x1f"
        if command := shutil.which("zenity"):
            args = [command, "--file-selection", f"--title={options['title']}",
                    "--file-filter=SceneRecall | " + " ".join(f"*.{ext}" for ext in options["extensions"]),
                    f"--separator={separator}"]
            if options["multiple"]:
                args.append("--multiple")
            if options["directory"]:
                args.append(f"--filename={options['directory']}/")
            result = _run(args)
            if result.returncode == 1:
                return {"paths": [], "cancelled": True}
            if result.returncode:
                raise FilePickerUnavailable("无法打开系统文件选择窗口，请检查本机桌面环境后重试")
            return {"paths": result.stdout.rstrip("\n").split(separator), "cancelled": False}
        raise FilePickerUnavailable("系统文件选择需要本机桌面环境和 Zenity，请安装 Zenity 后重试")
    raise FilePickerUnavailable("当前系统暂不支持原生文件选择窗口")


def pick_local_files(payload: FilePickerInput) -> dict:
    directory = None
    if payload.directory:
        directory = starting_directory(payload.directory)
    if not _dialog_lock.acquire(blocking=False):
        raise FilePickerBusy("系统文件选择窗口已打开，请先完成或取消当前选择")
    try:
        result = _open_dialog({"title": TITLES[payload.kind], "multiple": payload.multiple,
                               "directory": str(directory) if directory else None,
                               "extensions": sorted(ext[1:] for ext in EXTENSIONS[payload.kind])})
        if result["cancelled"]:
            return {"paths": [], "cancelled": True}
        if not result["paths"] or (not payload.multiple and len(result["paths"]) != 1):
            raise FilePickerUnavailable("系统文件选择窗口返回了错误的文件数量，请重试")
        paths = list(dict.fromkeys(str(local_file(raw, payload.kind)) for raw in result["paths"]))
        return {"paths": paths, "cancelled": False}
    finally:
        _dialog_lock.release()
