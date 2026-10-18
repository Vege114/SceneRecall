"""Windows CLI discovery and WSL path conversion, without accessing credentials."""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

_cached: tuple[float, dict] | None = None


def powershell() -> str | None:
    found = shutil.which("powershell.exe")
    if found:
        return found
    try:
        candidate = Path(linux_path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"))
        return str(candidate) if candidate.is_file() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def windows_path(path: str | Path) -> str:
    value = str(path)
    if len(value) > 2 and value[1] == ":" or value.startswith("\\\\"):
        return value
    result = subprocess.run(["wslpath", "-w", str(Path(value).resolve())], capture_output=True,
                            timeout=5, check=True)
    return result.stdout.decode("utf-8").strip()


def linux_path(path: str) -> str:
    result = subprocess.run(["wslpath", "-u", path], capture_output=True, timeout=5, check=True)
    return result.stdout.decode("utf-8").strip()


def discover(environment: dict[str, str]) -> dict:
    """Query only executable locations and the Windows profile directory.

    PowerShell executes a constant encoded script; prompts, model selections,
    paths from requests and credentials are never interpolated into that script.
    """
    global _cached
    now = time.monotonic()
    if _cached and now - _cached[0] < 15:
        return _cached[1]
    empty = {"paths": [], "home": None, "profile": None}
    shell = powershell()
    if not shell:
        return empty
    script = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$paths = @((Get-Command codex -All -CommandType Application | Select-Object -ExpandProperty Source))
$local = [Environment]::GetFolderPath('LocalApplicationData')
$paths += (Join-Path $local 'Programs\OpenAI\Codex\bin\codex.exe')
$paths += (Join-Path $local 'OpenAI\Codex\bin\codex.exe')
$paths += @(Get-ChildItem (Join-Path $local 'OpenAI\Codex\bin\*\codex.exe') |
    Sort-Object LastWriteTime -Descending | Select-Object -ExpandProperty FullName)
$paths += @(Get-Process codex | Select-Object -ExpandProperty Path)
$profilePath = [Environment]::GetFolderPath('UserProfile')
@{ paths = @($paths | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
    Select-Object -Unique); home = (Join-Path $profilePath '.codex'); profile = $profilePath } |
    ConvertTo-Json -Compress
"""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    # Never route a Linux CODEX_HOME into a Windows process.
    env = {key: value for key, value in environment.items() if key.upper() not in {"CODEX_HOME", "HOME"}}
    try:
        process = subprocess.run([shell, "-NoLogo", "-NoProfile", "-NonInteractive",
                                  "-EncodedCommand", encoded], env=env, capture_output=True, timeout=8)
        if process.returncode or len(process.stdout) > 64 * 1024:
            return empty
        result = json.loads(process.stdout.decode("utf-8-sig"))
        if (not isinstance(result, dict) or not isinstance(result.get("paths"), list)
                or not all(isinstance(path, str) for path in result["paths"])
                or not isinstance(result.get("home"), str) or not isinstance(result.get("profile"), str)):
            return empty
        _cached = (now, result)
        return result
    except (OSError, ValueError, subprocess.SubprocessError):
        return empty


def command(args: list[str], directory: Path) -> list[str]:
    converted, path_next = [], False
    for value in args:
        if path_next:
            value = windows_path(value)
        elif value.startswith("model_instructions_file="):
            path = json.loads(value.split("=", 1)[1])
            value = "model_instructions_file=" + json.dumps(windows_path(path))
        converted.append(value)
        path_next = value in {"--image", "-i", "--output-schema", "--cd", "-C"}
    if len(args) > 1 and args[1] == "exec" and "--help" not in args and "-h" not in args:
        converted[2:2] = ["--cd", windows_path(directory)]
    return converted


def launch(args: list[str], directory: Path, stdin: bytes, environment: dict[str, str]) -> tuple[list[str], bytes]:
    shell = powershell()
    if not shell:
        raise OSError("Windows PowerShell is unavailable")
    converted = command(args, directory)
    # Windows C-runtime quoting is data in JSON, never a shell command. This
    # preserves spaces, Unicode, quotes and trailing backslashes in arguments.
    overrides = {key.upper(): value for key, value in environment.items() if key.upper() in {
        "CODEX_HOME", "USERPROFILE", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    }}
    payload = {"executable": windows_path(converted[0]),
               "arguments": subprocess.list2cmdline(converted[1:]), "directory": windows_path(directory),
               "input": base64.b64encode(stdin).decode("ascii"), "environment": overrides}
    script = Path(__file__).with_name("codex_windows_bridge.ps1").read_text(encoding="utf-8")
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return ([shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            json.dumps(payload).encode("utf-8"))
