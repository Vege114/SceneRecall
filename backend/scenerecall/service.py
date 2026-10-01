"""Conservative, stateless discovery of this installation's local CLI server.

Never use a port or a saved PID as authority to signal a process. Discover the
launcher, user, interpreter, arguments and listener together on every invocation.
"""
from __future__ import annotations

import argparse
import errno
import math
import os
from pathlib import Path
import socket
import sys
import sysconfig
import time

import psutil


class ServiceError(Exception):
    """An unsafe or unsuccessful stop, suitable for display in the terminal."""


def positive_timeout(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("等待秒数必须是有限的正数")
    return number


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="SceneRecall 本地影视资料库", allow_abbrev=False)
    cli.add_argument("command", nargs="?", choices=("start", "stop"), default="start",
                     help="启动（默认）或安全停止本安装的服务")
    cli.add_argument("--data-dir", type=Path, default=Path.home() / "SceneRecallLibrary", help="资料目录")
    cli.add_argument("--port", type=int, default=8765, help="本机端口（默认 8765）")
    cli.add_argument("--timeout", type=positive_timeout, help="停止等待秒数（默认 30）")
    cli.add_argument("--dry-run", action="store_true", help="只检查停止目标，不发送信号")
    return cli


def _path(value: str, cwd: Path, home: Path) -> Path:
    # Resolve paths in the server's context, not in the stop command's cwd/HOME.
    if value == "~" or value.startswith("~/"):
        value = str(home) + value[1:]
    return (cwd / value).resolve()


def _launch_options(argv: list[str], cwd: Path, home: Path) -> tuple[Path, int] | None:
    """Recognize only the public start syntax, including pre-stop CLI versions."""
    data_dir, port = home / "SceneRecallLibrary", 8765
    command_seen = False
    args = iter(argv)
    try:
        for arg in args:
            if arg == "start" and not command_seen:
                command_seen = True
            elif arg == "--data-dir" or arg.startswith("--data-dir="):
                data_dir = _path(next(args) if arg == "--data-dir" else arg.split("=", 1)[1], cwd, home)
            elif arg == "--port" or arg.startswith("--port="):
                port = int(next(args) if arg == "--port" else arg.split("=", 1)[1])
            else:
                return None
    except (StopIteration, ValueError):
        return None
    return data_dir.resolve(), port


def _matches(process: psutil.Process, data_dir: Path, port: int) -> bool:
    if process.uids().effective != os.geteuid():
        return False
    argv = process.cmdline()
    # The normal console script has exactly one Python executable before it.
    # Refuse python -c, uvicorn, wrappers, and similarly named scripts elsewhere.
    if len(argv) < 2 or Path(argv[1]).name != "scenerecall":
        return False
    cwd = Path(process.cwd())
    launcher = Path(sysconfig.get_path("scripts")) / "scenerecall"
    if (cwd / argv[1]).resolve() != launcher.resolve():
        return False
    if Path(process.exe()).resolve() != Path(sys.executable).resolve():
        return False
    # The server may have been started with a different HOME under the same UID.
    home = Path(process.environ().get("HOME") or Path.home())
    return _launch_options(argv[2:], cwd, home) == (data_dir, port)


def _listens(process: psutil.Process, port: int) -> bool:
    return any(connection.status == psutil.CONN_LISTEN
               and connection.laddr.ip == "127.0.0.1" and connection.laddr.port == port
               for connection in process.net_connections(kind="tcp4"))


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # Closed HTTP connections can leave TIME_WAIT sockets after shutdown.
        # They must not make an already stopped server look like a live owner.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return True
            raise
    return False


def find_service(data_dir: Path, port: int) -> psutil.Process | None:
    """Read-only discovery; also used for --dry-run and legacy CLI instances."""
    candidates = []
    denied = False
    for process in psutil.process_iter():
        try:
            if _matches(process, data_dir, port):
                # Do not accept psutil's partial identity if process creation
                # time could not be read when the Process object was created.
                process.create_time()
                candidates.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            denied = True
    if len(candidates) > 1:
        raise ServiceError("发现多个匹配的 SceneRecall 进程，无法唯一确认目标；未发送停止信号。")
    if candidates:
        process = candidates[0]
        try:
            if _listens(process, port):
                return process
        except psutil.NoSuchProcess:
            return None
        raise ServiceError("匹配的 SceneRecall 进程尚未监听目标端口，可能正在启动或退出；未发送停止信号，请稍后重试。")
    if _port_busy(port):
        detail = "部分进程信息无权读取；" if denied else ""
        raise ServiceError(f"{detail}端口 {port} 已被占用，但无法确认是本安装及指定资料目录的 SceneRecall；未发送停止信号。")
    return None


def _wait_for_exit(process: psutil.Process, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise psutil.TimeoutExpired(timeout, pid=process.pid)
        try:
            process.wait(timeout=min(0.1, remaining))
            return
        except psutil.TimeoutExpired:
            # A non-child zombie has already exited, even if its parent has not
            # reaped it. Check creation identity too, rather than waiting on a
            # replacement that happens to acquire this PID.
            continue


def stop_service(data_dir: Path, port: int, timeout: float = 30.0, dry_run: bool = False) -> str:
    if os.name != "posix":
        raise ServiceError("stop 目前仅支持 macOS / Linux；请在启动服务的终端按 Ctrl+C 正常退出。")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ServiceError("等待秒数必须是有限的正数。")
    try:
        data_dir = data_dir.expanduser().resolve()
        process = find_service(data_dir, port)
        if process is None:
            return f"SceneRecall 未运行（端口 {port}，资料目录 {data_dir}）。"
        # Keep this Process object: psutil's signal method checks PID + creation
        # time again, preventing a stale PID from targeting its replacement.
        if not process.is_running() or not _matches(process, data_dir, port) or not _listens(process, port):
            raise ServiceError("目标进程身份或监听状态已变化；未发送停止信号，请重试。")
        if dry_run:
            return f"检查通过：可安全停止 SceneRecall（PID {process.pid}，端口 {port}，资料目录 {data_dir}）；未发送信号。"
        process.terminate()  # SIGTERM on POSIX; never escalate to SIGKILL.
        try:
            _wait_for_exit(process, timeout)
        except psutil.TimeoutExpired as exc:
            raise ServiceError(f"已请求 SceneRecall 正常退出，但等待 {timeout:g} 秒后进程仍在运行；未强制终止。请检查启动终端或稍后重试。") from exc
        return f"SceneRecall 已停止（端口 {port}）。资料和配置已保留：{data_dir}"
    except psutil.NoSuchProcess:
        return f"SceneRecall 已退出（端口 {port}）；资料和配置保留。"
    except (psutil.AccessDenied, PermissionError) as exc:
        raise ServiceError("权限不足，无法核实或停止 SceneRecall；请使用启动服务的同一系统用户，并检查终端的进程访问权限。") from exc
    except OSError as exc:
        raise ServiceError(f"无法检查或停止 SceneRecall：{exc.strerror or exc}。") from exc
    except RuntimeError as exc:
        raise ServiceError("无法解析资料目录，请使用可访问的绝对路径。") from exc
