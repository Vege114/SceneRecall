"""Start a hidden native Windows console and stop only its verified server tree.

The state file is a discovery hint, never authority to signal a PID. Stop checks
the current user, creation time, exact launcher, Python arguments and listener
again before requesting CTRL_BREAK. It never calls TerminateProcess or taskkill.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
import uuid
import webbrowser

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PYTHON = PROJECT_ROOT / ".venv-windows/Scripts/python.exe"
LAUNCHER = PROJECT_ROOT / "scripts/start_windows.ps1"
RUNTIME_ROOT = PROJECT_ROOT / ".cache/runtime"


class RuntimeErrorDetail(Exception):
    """A failed or unsafe operation that can be displayed without a traceback."""


def positive_timeout(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return number


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    cli.add_argument("command", choices=("start", "stop", "status"))
    cli.add_argument("--data-dir", type=Path, default=Path(os.environ.get(
        "SCENERECALL_DATA", str(Path.home() / "SceneRecallLibrary"))))
    cli.add_argument("--port", type=int, default=8765)
    cli.add_argument("--runtime-config", type=Path)
    cli.add_argument("--ffmpeg-directory", type=Path)
    cli.add_argument("--timeout", type=positive_timeout, default=30.0)
    cli.add_argument("--open", action="store_true", help="open the local session in a browser after start")
    cli.add_argument("--dry-run", action="store_true", help="verify a stop target without sending a signal")
    return cli


def normalized(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def powershell() -> Path:
    return Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"


def current_user() -> str:
    return psutil.Process().username().casefold()


def state_path(data_dir: Path, port: int) -> Path:
    digest = hashlib.sha256(normalized(data_dir).encode("utf-8")).hexdigest()[:16]
    return RUNTIME_ROOT / f"native-{port}-{digest}.json"


@contextmanager
def runtime_lock(path: Path):
    import msvcrt

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeErrorDetail("Another start/stop/status is in progress; retry shortly.") from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def command(data_dir: Path, port: int, runtime_config: Path | None, ffmpeg_directory: Path | None) -> list[str]:
    argv = [str(powershell()), "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER),
            "-DataDir", str(data_dir), "-Port", str(port)]
    if runtime_config is not None:
        argv += ["-RuntimeConfig", str(runtime_config.resolve())]
    if ffmpeg_directory is not None:
        argv += ["-FfmpegDirectory", str(ffmpeg_directory.resolve())]
    return argv


def load_state(path: Path, data_dir: Path, port: int) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = command(data_dir, port,
                           Path(value["runtime_config"]) if value["runtime_config"] else None,
                           Path(value["ffmpeg_directory"]) if value["ffmpeg_directory"] else None)
        if (value["version"] != 1 or normalized(value["project_root"]) != normalized(PROJECT_ROOT)
                or normalized(value["data_dir"]) != normalized(data_dir) or value["port"] != port
                or value["username"] != current_user() or value["launcher_command"] != expected
                or type(value["launcher_pid"]) is not int or value["launcher_pid"] <= 0
                or not math.isfinite(value["launcher_created"])):
            raise ValueError("identity mismatch")
        if value.get("server_pid") is not None:
            if (type(value["server_pid"]) is not int or value["server_pid"] <= 0
                    or not math.isfinite(value["server_created"]) or not isinstance(value["server_command"], list)):
                raise ValueError("invalid server identity")
        return value
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeErrorDetail(f"Invalid runtime state; no process was signalled: {path}") from exc


def save_state(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # On Windows an exclusive bind, rather than SO_REUSEADDR, detects owners.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            if exc.winerror in (10013, 10048):
                return True
            raise
    return False


def launcher_process(state: dict) -> psutil.Process | None:
    try:
        process = psutil.Process(state["launcher_pid"])
        if abs(process.create_time() - state["launcher_created"]) > 0.01:
            return None  # A recycled PID is stale metadata, not a stop target.
        checks = {"user": process.username().casefold() == current_user(),
                  "executable": normalized(process.exe()) == normalized(powershell()),
                  "arguments": process.cmdline() == state["launcher_command"],
                  "directory": normalized(process.cwd()) == normalized(PROJECT_ROOT)}
        if not all(checks.values()):
            mismatch = ", ".join(name for name, matches in checks.items() if not matches)
            raise RuntimeErrorDetail(f"Launcher identity changed ({mismatch}); no process was signalled.")
        return process
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def python_executables() -> set[str]:
    # Windows venv redirectors may create a child from the base interpreter.
    result = subprocess.run([str(APP_PYTHON), "-c", "import sys; print(sys._base_executable)"],
                            check=True, capture_output=True, text=True, encoding="utf-8", timeout=10)
    return {normalized(APP_PYTHON), normalized(result.stdout.strip())}


def server_process(launcher: psutil.Process, data_dir: Path, port: int,
                   executable_paths: set[str]) -> psutil.Process | None:
    candidates = []
    expected_tail = ["-m", "scenerecall.main", "--data-dir", str(data_dir), "--port", str(port)]
    try:
        children = launcher.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    for child in children:
        try:
            argv = child.cmdline()
            if (len(argv) != 7 or normalized(argv[0]) not in executable_paths
                    or argv[1:] != expected_tail or child.username().casefold() != current_user()
                    or normalized(child.exe()) not in executable_paths
                    or normalized(child.cwd()) != normalized(PROJECT_ROOT)):
                continue
            if any(connection.status == psutil.CONN_LISTEN and connection.laddr.ip == "127.0.0.1"
                   and connection.laddr.port == port for connection in child.net_connections(kind="tcp4")):
                child.create_time()
                candidates.append(child)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    if len(candidates) > 1:
        raise RuntimeErrorDetail("Multiple verified listeners found; no process was signalled.")
    return candidates[0] if candidates else None


def health_ready(port: int) -> bool:
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, ValueError):
        return False


def check_server_identity(state: dict, server: psutil.Process) -> None:
    if state.get("server_pid") is not None:
        if (server.pid != state["server_pid"] or abs(server.create_time() - state["server_created"]) > 0.01
                or server.cmdline() != state["server_command"]):
            raise RuntimeErrorDetail("Server identity changed; no process was signalled.")


def inspect(state: dict | None, data_dir: Path, port: int, executable_paths: set[str]) -> dict:
    launcher = launcher_process(state) if state else None
    server = server_process(launcher, data_dir, port, executable_paths) if launcher else None
    if server:
        try:
            check_server_identity(state, server)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            server = None
    if not server and launcher:
        # A graceful shutdown may remove the launcher between its identity
        # check and child enumeration (or the server's second identity check).
        launcher = launcher_process(state)
    if server:
        status = "running"
    elif port_busy(port):
        status = "unverified_port_owner"
    elif launcher:
        status = "starting_or_stopping"
    else:
        status = "stopped"
    return {"status": status, "data_dir": str(data_dir), "port": port,
            "launcher_pid": launcher.pid if launcher else None,
            "server_pid": server.pid if server else None,
            "log_path": state["log_path"] if state else None}


def request_break(state: dict, data_dir: Path, port: int, executable_paths: set[str]) -> None:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.AttachConsole.argtypes = [wintypes.DWORD]
    kernel.AttachConsole.restype = wintypes.BOOL
    kernel.FreeConsole.restype = wintypes.BOOL
    kernel.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
    kernel.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    kernel.GetConsoleProcessList.restype = wintypes.DWORD
    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    kernel.SetConsoleCtrlHandler.argtypes = [handler_type, wintypes.BOOL]
    kernel.SetConsoleCtrlHandler.restype = wintypes.BOOL
    handler = handler_type(lambda event: True)
    kernel.FreeConsole()  # Detach only this manager, not its calling terminal.
    if not kernel.AttachConsole(state["launcher_pid"]):
        raise RuntimeErrorDetail(f"Cannot attach to the verified console: {ctypes.WinError(ctypes.get_last_error())}")
    try:
        if not kernel.SetConsoleCtrlHandler(handler, True):
            raise ctypes.WinError(ctypes.get_last_error())
        launcher = launcher_process(state)
        server = server_process(launcher, data_dir, port, executable_paths) if launcher else None
        if not launcher or not server or not launcher.is_running() or not server.is_running():
            raise RuntimeErrorDetail("Target identity or listener changed; no process was signalled.")
        check_server_identity(state, server)
        expected_console = {os.getpid(), launcher.pid, *(child.pid for child in launcher.children(recursive=True))}
        members = (wintypes.DWORD * 4096)()
        count = kernel.GetConsoleProcessList(members, len(members))
        if count == 0 or count > len(members) or not set(members[:count]).issubset(expected_console):
            raise RuntimeErrorDetail("The console contains an unrelated process; no process was signalled.")
        if not kernel.GenerateConsoleCtrlEvent(1, launcher.pid):  # CTRL_BREAK_EVENT to this dedicated group.
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.FreeConsole()
        kernel.SetConsoleCtrlHandler(handler, False)
        kernel.AttachConsole(wintypes.DWORD(-1).value)


def start(args: argparse.Namespace, path: Path, state: dict | None, executable_paths: set[str]) -> dict:
    current = inspect(state, args.data_dir, args.port, executable_paths)
    if current["status"] == "running":
        return current
    if current["status"] != "stopped":
        raise RuntimeErrorDetail(f"Cannot start: {current['status']}; no process was stopped.")
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    log_path = RUNTIME_ROOT / f"native-{args.port}-{stamp}.log"
    argv = command(args.data_dir, args.port, args.runtime_config, args.ffmpeg_directory)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(argv, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                                        "SCENERECALL_MANAGED_CONSOLE": "1"},
                                   startupinfo=startup,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP)
    state = {"version": 1, "project_root": str(PROJECT_ROOT), "data_dir": str(args.data_dir), "port": args.port,
             "username": current_user(), "launcher_pid": process.pid,
             "launcher_created": psutil.Process(process.pid).create_time(), "launcher_command": argv,
             "runtime_config": str(args.runtime_config.resolve()) if args.runtime_config else None,
             "ffmpeg_directory": str(args.ffmpeg_directory.resolve()) if args.ffmpeg_directory else None,
             "log_path": str(log_path)}
    save_state(path, state)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        current = inspect(state, args.data_dir, args.port, executable_paths)
        if current["status"] == "running" and health_ready(args.port):
            server = psutil.Process(current["server_pid"])
            state.update(server_pid=server.pid, server_created=server.create_time(), server_command=server.cmdline())
            save_state(path, state)
            return current
        if process.poll() is not None:
            raise RuntimeErrorDetail(f"The launcher exited with code {process.returncode}; inspect {log_path}")
        time.sleep(0.2)
    raise RuntimeErrorDetail(f"Startup did not finish in {args.timeout:g}s; no force stop was attempted. Log: {log_path}")


def stop(args: argparse.Namespace, path: Path, state: dict | None, executable_paths: set[str]) -> dict:
    current = inspect(state, args.data_dir, args.port, executable_paths)
    if current["status"] == "stopped":
        path.unlink(missing_ok=True)
        return current
    if current["status"] != "running" or not state:
        raise RuntimeErrorDetail(f"Cannot verify a running stop target ({current['status']}); no signal was sent.")
    if args.dry_run:
        return {**current, "stop_verified": True, "signal_sent": False}
    request_break(state, args.data_dir, args.port, executable_paths)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        current = inspect(state, args.data_dir, args.port, executable_paths)
        if current["status"] == "stopped":
            path.unlink(missing_ok=True)
            return {**current, "signal_sent": True}
        time.sleep(0.1)
    raise RuntimeErrorDetail(f"Requested graceful shutdown, but still running after {args.timeout:g}s; no force stop was used.")


def open_session(result: dict, port: int) -> None:
    log = Path(result["log_path"]).read_text(encoding="utf-8", errors="replace")
    matches = re.findall(rf"http://127\.0\.0\.1:{port}/\?token=[A-Za-z0-9_-]+", log)
    if not matches:
        raise RuntimeErrorDetail("Server is running, but its local session link is not yet available in the log.")
    webbrowser.open(matches[-1])


def main(argv: list[str] | None = None) -> int:
    # PowerShell 5.1 otherwise decodes redirected Chinese output as a legacy code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    cli = parser()
    args = cli.parse_args(argv)
    if os.name != "nt":
        cli.error("This manager runs on native Windows only.")
    if not 1024 <= args.port <= 65535:
        cli.error("port must be between 1024 and 65535")
    if (args.open and args.command != "start") or (args.dry_run and args.command != "stop"):
        cli.error("--open is only for start; --dry-run is only for stop")
    args.data_dir = args.data_dir.expanduser().resolve()
    try:
        path = state_path(args.data_dir, args.port)
        with runtime_lock(path):
            state = load_state(path, args.data_dir, args.port)
            executable_paths = python_executables()
            if args.command == "start":
                result = start(args, path, state, executable_paths)
            elif args.command == "stop":
                result = stop(args, path, state, executable_paths)
            else:
                result = inspect(state, args.data_dir, args.port, executable_paths)
        if args.open:
            open_session(result, args.port)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (RuntimeErrorDetail, psutil.Error, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
