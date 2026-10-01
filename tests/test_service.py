"""Stop only the verified local CLI server; real instances use isolated libraries."""
from contextlib import contextmanager
from functools import partial
import http.client
import os
from pathlib import Path
import socket
import subprocess
import sys
import sysconfig
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from scenerecall import service


TEST_PORT = 19321


@pytest.fixture
def target(tmp_path, monkeypatch):
    data_dir = (tmp_path / "library").resolve()
    process = Mock(spec=psutil.Process)
    process.pid = 12345
    process.uids.return_value = SimpleNamespace(effective=os.geteuid())
    process.cwd.return_value = str(tmp_path)
    process.exe.return_value = sys.executable
    process.environ.return_value = {"HOME": str(tmp_path / "home")}
    process.cmdline.return_value = [sys.executable, str(Path(sysconfig.get_path("scripts")) / "scenerecall"),
                                   "--data-dir", str(data_dir), "--port", str(TEST_PORT)]
    process.is_running.return_value = True
    process.status.return_value = psutil.STATUS_RUNNING
    process.net_connections.return_value = [SimpleNamespace(
        status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.1", port=TEST_PORT))]
    monkeypatch.setattr(service.psutil, "process_iter", lambda: iter([process]))
    monkeypatch.setattr(service, "_port_busy", lambda port: False)
    return process, data_dir


@pytest.mark.parametrize("args,relative_dir,port", [
    ([], "home/SceneRecallLibrary", 8765),
    (["start"], "home/SceneRecallLibrary", 8765),
    (["--data-dir", "relative/library", "--port", "19001"], "relative/library", 19001),
    (["start", "--data-dir=relative/library", "--port=19002"], "relative/library", 19002),
    (["--port=19003", "start", "--data-dir=~/library"], "home/library", 19003),
])
def test_launch_arguments_resolve_in_server_context(tmp_path, args, relative_dir, port):
    assert service._launch_options(args, tmp_path, tmp_path / "home") == (
        (tmp_path / relative_dir).resolve(), port)


@pytest.mark.parametrize("args", [
    ["stop"], ["start", "start"], ["--data-dir"], ["--port"], ["--port=no"],
    ["--reload"], ["--dry-run"], ["--timeout=30"], ["--por", "8765"],
])
def test_ambiguous_or_non_start_arguments_are_not_targets(tmp_path, args):
    assert service._launch_options(args, tmp_path, tmp_path) is None


@pytest.mark.parametrize("mismatch", ["uid", "launcher", "interpreter", "data", "port", "wrapper"])
def test_unrelated_process_identity_is_rejected(target, tmp_path, mismatch):
    process, data_dir = target
    if mismatch == "uid":
        process.uids.return_value.effective += 1
    elif mismatch == "launcher":
        process.cmdline.return_value[1] = str(tmp_path / "another-install" / "scenerecall")
    elif mismatch == "interpreter":
        process.exe.return_value = str(tmp_path / "another-python")
    elif mismatch == "data":
        process.cmdline.return_value[3] = str(tmp_path / "another-library")
    elif mismatch == "port":
        process.cmdline.return_value[5] = str(TEST_PORT + 1)
    else:
        process.cmdline.return_value = [sys.executable, "-c", "import scenerecall.main"]
    assert service.find_service(data_dir, TEST_PORT) is None
    process.terminate.assert_not_called()


def test_relative_launcher_and_equals_arguments_are_identified(target):
    process, data_dir = target
    process.cmdline.return_value[1] = os.path.relpath(process.cmdline.return_value[1], process.cwd.return_value)
    process.cmdline.return_value[2:] = ["start", "--data-dir=library", f"--port={TEST_PORT}"]
    assert service.find_service(data_dir, TEST_PORT) is process
    process.terminate.assert_not_called()


@pytest.mark.parametrize("wrong_listener", [
    SimpleNamespace(status=psutil.CONN_ESTABLISHED, laddr=SimpleNamespace(ip="127.0.0.1", port=TEST_PORT)),
    SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="0.0.0.0", port=TEST_PORT)),
    SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.1", port=TEST_PORT + 1)),
])
def test_process_without_exact_loopback_listener_is_not_signalled(target, wrong_listener):
    process, data_dir = target
    process.net_connections.return_value = [wrong_listener]
    with pytest.raises(service.ServiceError, match="尚未监听"):
        service.stop_service(data_dir, TEST_PORT)
    process.terminate.assert_not_called()


def test_multiple_matching_processes_are_not_signalled(target, monkeypatch):
    process, data_dir = target
    monkeypatch.setattr(service.psutil, "process_iter", lambda: iter([process, process]))
    with pytest.raises(service.ServiceError, match="多个匹配"):
        service.stop_service(data_dir, TEST_PORT)
    process.terminate.assert_not_called()


@pytest.mark.parametrize("unreadable", ["cmdline", "create_time"])
def test_busy_port_with_unreadable_process_reports_safe_refusal(target, monkeypatch, unreadable):
    process, data_dir = target
    getattr(process, unreadable).side_effect = psutil.AccessDenied(process.pid)
    monkeypatch.setattr(service, "_port_busy", lambda port: True)
    with pytest.raises(service.ServiceError, match="部分进程信息无权读取.*未发送停止信号"):
        service.stop_service(data_dir, TEST_PORT)
    process.terminate.assert_not_called()


@pytest.mark.parametrize("failure", [psutil.NoSuchProcess(12345), psutil.ZombieProcess(12345)])
def test_disappearing_process_during_scan_is_ignored(target, failure):
    process, data_dir = target
    process.cmdline.side_effect = failure
    assert "未运行" in service.stop_service(data_dir, TEST_PORT)
    process.terminate.assert_not_called()


def test_not_running_does_not_create_library(tmp_path, monkeypatch):
    monkeypatch.setattr(service.psutil, "process_iter", lambda: iter([]))
    monkeypatch.setattr(service, "_port_busy", lambda port: False)
    absent = tmp_path / "nonexistent"
    assert "未运行" in service.stop_service(absent, TEST_PORT)
    assert not absent.exists()


def test_dry_run_checks_identity_but_never_signals(target):
    process, data_dir = target
    result = service.stop_service(data_dir, TEST_PORT, dry_run=True)
    assert "检查通过" in result and "未发送信号" in result
    assert str(process.pid) in result
    process.terminate.assert_not_called()
    process.wait.assert_not_called()


@pytest.mark.parametrize("change", ["pid_reused", "arguments", "listener"])
def test_identity_or_listener_change_before_signal_is_rejected(target, change):
    process, data_dir = target
    if change == "pid_reused":
        process.is_running.return_value = False
    elif change == "arguments":
        original = list(process.cmdline.return_value)
        process.cmdline.side_effect = [original, [sys.executable, "another-command"]]
    else:
        process.net_connections.side_effect = [process.net_connections.return_value, []]
    with pytest.raises(service.ServiceError, match="身份或监听状态已变化"):
        service.stop_service(data_dir, TEST_PORT)
    process.terminate.assert_not_called()


@pytest.mark.parametrize("stage", ["is_running", "terminate", "wait"])
def test_exit_race_reports_already_exited(target, stage):
    process, data_dir = target
    getattr(process, stage).side_effect = psutil.NoSuchProcess(process.pid)
    assert "已退出" in service.stop_service(data_dir, TEST_PORT)
    process.kill.assert_not_called()


@pytest.mark.parametrize("stage,failure", [
    ("net_connections", psutil.AccessDenied(12345)),
    ("terminate", psutil.AccessDenied(12345)),
    ("terminate", PermissionError("not permitted")),
])
def test_permission_errors_are_actionable_without_force(target, stage, failure):
    process, data_dir = target
    getattr(process, stage).side_effect = failure
    with pytest.raises(service.ServiceError, match="权限不足.*同一系统用户"):
        service.stop_service(data_dir, TEST_PORT)
    process.kill.assert_not_called()


def test_graceful_stop_waits_on_same_process(target):
    process, data_dir = target
    assert "已停止" in service.stop_service(data_dir, TEST_PORT, timeout=2)
    process.terminate.assert_called_once_with()
    assert process.wait.called
    assert all(0 < call.kwargs["timeout"] <= 2 for call in process.wait.call_args_list)
    process.kill.assert_not_called()


def test_timeout_never_escalates_to_kill(target):
    process, data_dir = target

    def timed_wait(timeout):
        time.sleep(timeout)
        raise psutil.TimeoutExpired(timeout, pid=process.pid)

    process.wait.side_effect = timed_wait
    with pytest.raises(service.ServiceError, match="等待 0.001 秒.*未强制终止"):
        service.stop_service(data_dir, TEST_PORT, timeout=0.001)
    process.terminate.assert_called_once_with()
    assert all(0 < call.kwargs["timeout"] <= 0.001 for call in process.wait.call_args_list)
    process.kill.assert_not_called()


@pytest.mark.parametrize("exited_state", ["zombie", "pid_reused"])
def test_exited_process_is_not_awaited_or_signalled_again(target, exited_state):
    process, data_dir = target
    if exited_state == "zombie":
        process.status.return_value = psutil.STATUS_ZOMBIE
    else:
        process.is_running.side_effect = [True, False]
    assert "已停止" in service.stop_service(data_dir, TEST_PORT, timeout=0.001)
    process.terminate.assert_called_once_with()
    process.wait.assert_not_called()
    process.kill.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_timeout_is_rejected_before_discovery(tmp_path, monkeypatch, timeout):
    discover = Mock()
    monkeypatch.setattr(service, "find_service", discover)
    with pytest.raises(service.ServiceError, match="有限的正数"):
        service.stop_service(tmp_path, TEST_PORT, timeout=timeout)
    discover.assert_not_called()


def test_unknown_home_is_reported_before_process_discovery(monkeypatch):
    discover = Mock()
    monkeypatch.setattr(service, "find_service", discover)
    monkeypatch.setattr(Path, "expanduser", Mock(side_effect=RuntimeError("Could not determine home directory")))
    with pytest.raises(service.ServiceError, match="无法解析资料目录.*绝对路径"):
        service.stop_service(Path("~missing-test-user/library"), TEST_PORT)
    discover.assert_not_called()


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "-inf", "not-a-number"])
def test_cli_rejects_invalid_timeout(timeout):
    with pytest.raises(SystemExit) as error:
        service.parser().parse_args(["stop", f"--timeout={timeout}"])
    assert error.value.code == 2


@pytest.mark.parametrize("option", ["--dry-run", "--timeout=30"])
def test_start_rejects_stop_options_before_creating_library(tmp_path, monkeypatch, option):
    from scenerecall import main

    create_app = Mock()
    monkeypatch.setattr(main, "create_app", create_app)
    monkeypatch.setattr(sys, "argv", ["scenerecall", "start", "--data-dir", str(tmp_path), option])
    with pytest.raises(SystemExit) as error:
        main.run()
    assert error.value.code == 2
    create_app.assert_not_called()


def _unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _http_status(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


@contextmanager
def _running_server(command, cwd, port, health_path, log_path):
    # A waiter reaps the child even while a separate stop CLI waits for its exit.
    with log_path.open("w+") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        waiter = threading.Thread(target=process.wait, daemon=True)
        waiter.start()
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(f"isolated test server exited with {process.returncode}; log: {log_path}")
                try:
                    if _http_status(port, health_path) == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(0.05)
            else:
                pytest.fail(f"isolated test server did not become healthy; log: {log_path}")
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
            waiter.join(timeout=10)
            if waiter.is_alive():
                process.kill()  # Cleanup only this test-owned Popen child.
                waiter.join(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="stop supports macOS and Linux")
@pytest.mark.parametrize("start_command", [[], ["start"]], ids=["legacy-start", "explicit-start"])
def test_real_cli_stop_preserves_library_and_refuses_wrong_directory(tmp_path, start_command):
    launcher = Path(sysconfig.get_path("scripts")) / "scenerecall"
    assert launcher.is_file(), "install the project console script before running tests"
    port = _unused_port()
    data_dir = tmp_path / "isolated-library"
    data_dir.mkdir()
    preserved = data_dir / "user-notes.txt"
    preserved.write_text("保留用户资料", encoding="utf-8")
    # Relative paths and equals syntax must also work for real process argv.
    command = [str(launcher), *start_command, "--data-dir=isolated-library", f"--port={port}"]
    with _running_server(command, tmp_path, port, "/api/health", tmp_path / "server.log") as process:
        settings = data_dir / "private" / "settings.json"
        settings.write_text('{"defaults": {"window_ms": 1234}}', encoding="utf-8")
        run_stop = partial(subprocess.run, cwd=tmp_path, capture_output=True, text=True, timeout=15)
        wrong = run_stop([str(launcher), "stop", "--data-dir", str(tmp_path / "wrong"), "--port", str(port)])
        assert wrong.returncode == 1 and "无法确认" in wrong.stderr
        assert process.poll() is None and _http_status(port, "/api/health") == 200
        dry_run = run_stop([str(launcher), "stop", "--data-dir", str(data_dir), "--port", str(port), "--dry-run"])
        assert dry_run.returncode == 0 and "未发送信号" in dry_run.stdout
        assert process.poll() is None and _http_status(port, "/api/health") == 200
        stop = run_stop([str(launcher), "stop", "--data-dir", str(data_dir), "--port", str(port), "--timeout=5"])
        assert stop.returncode == 0, stop.stderr
        assert "已停止" in stop.stdout and process.wait(timeout=5) is not None
        assert "Application shutdown complete." in (tmp_path / "server.log").read_text(encoding="utf-8")
        assert preserved.read_text(encoding="utf-8") == "保留用户资料"
        assert settings.read_text(encoding="utf-8") == '{"defaults": {"window_ms": 1234}}'
        repeat = run_stop([str(launcher), "stop", "--data-dir", str(data_dir), "--port", str(port)])
        assert repeat.returncode == 0 and "未运行" in repeat.stdout


@pytest.mark.skipif(os.name != "posix", reason="stop supports macOS and Linux")
def test_real_unrelated_http_server_is_not_stopped(tmp_path):
    port = _unused_port()
    command = [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(tmp_path)]
    with _running_server(command, tmp_path, port, "/", tmp_path / "http.log") as process:
        with pytest.raises(service.ServiceError, match="无法确认.*未发送停止信号"):
            service.stop_service(tmp_path / "not-a-library", port)
        assert process.poll() is None and _http_status(port, "/") == 200
