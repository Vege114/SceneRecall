"""Stop only the verified local CLI server; real instances use isolated libraries."""
from contextlib import contextmanager
from functools import partial
import http.client
import os
from pathlib import Path
import re
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
from scenerecall import windows_service


# This module verifies the POSIX CLI's UID/SIGTERM lifecycle. Windows uses
# manage_windows_runtime.py, whose process identity and CTRL_BREAK integration
# are exercised by test_windows_runtime_manager.py.
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX service lifecycle")


TEST_PORT = 19321


@pytest.fixture
def target(tmp_path, monkeypatch):
    # Keep the existing POSIX safety cases runnable on Windows too; dedicated
    # cases below exercise the native Windows identity/control path.
    monkeypatch.setattr(service, "WINDOWS", False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    data_dir = (tmp_path / "library").resolve()
    process = Mock(spec=psutil.Process)
    process.pid = 12345
    process.uids = Mock(return_value=SimpleNamespace(effective=os.geteuid()))
    process.cwd.return_value = str(tmp_path)
    process.exe.return_value = psutil.Process().exe()
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


def test_posix_backslash_spelling_retains_its_existing_path_meaning(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "WINDOWS", False)
    assert service._path("~\\library", tmp_path, tmp_path / "home") == (tmp_path / "~\\library").resolve()


def test_posix_port_check_does_not_need_windows_socket_constants(monkeypatch):
    monkeypatch.setattr(service, "WINDOWS", False)
    monkeypatch.delattr(socket, "SO_EXCLUSIVEADDRUSE", raising=False)
    assert not service._port_busy(0)


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


def test_relative_launcher_and_equals_arguments_are_identified(target, tmp_path, monkeypatch):
    process, data_dir = target
    # A relative launcher must share the server cwd's volume. CI can install
    # the real virtual environment on D: while pytest temp directories use C:.
    scripts = tmp_path / "installation"
    monkeypatch.setattr(service.sysconfig, "get_path", lambda name: str(scripts))
    process.cmdline.return_value[1] = os.path.relpath(scripts / "scenerecall", process.cwd.return_value)
    process.cmdline.return_value[2:] = ["start", "--data-dir=library", f"--port={TEST_PORT}"]
    assert service.find_service(data_dir, TEST_PORT) is process
    process.terminate.assert_not_called()


def test_macos_framework_launcher_matches_the_current_os_interpreter(target, tmp_path, monkeypatch):
    process, data_dir = target
    framework = tmp_path / "Python.framework" / "Versions" / "3.12"
    public_python = framework / "bin" / "python3.12"
    real_python = framework / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    monkeypatch.setattr(sys, "executable", str(public_python))
    monkeypatch.setattr(service.psutil, "Process", lambda: SimpleNamespace(exe=lambda: str(real_python)))
    process.exe.return_value = str(real_python)
    assert service.find_service(data_dir, TEST_PORT) is process
    # A different OS executable remains unrelated even with a matching script.
    process.exe.return_value = str(framework / "another-python")
    assert service.find_service(data_dir, TEST_PORT) is None
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


def test_posix_start_keeps_original_uvicorn_entry_point(tmp_path, monkeypatch):
    from scenerecall import main
    import uvicorn

    app = SimpleNamespace(state=SimpleNamespace(session_token="synthetic-test-token"))
    monkeypatch.setattr(main, "create_app", Mock(return_value=app))
    monkeypatch.setattr(main, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(sys, "argv", ["scenerecall", "start", "--data-dir", str(tmp_path), "--port", str(TEST_PORT)])
    serve = Mock()
    monkeypatch.setattr(uvicorn, "run", serve)
    monkeypatch.setattr(windows_service, "shutdown_control", Mock(side_effect=AssertionError("POSIX used Windows control")))
    main.run()
    serve.assert_called_once_with(app, host="127.0.0.1", port=TEST_PORT, access_log=False)


@pytest.fixture
def windows_target(target, monkeypatch):
    process, data_dir = target
    monkeypatch.setattr(service, "WINDOWS", True)
    current = SimpleNamespace(username=lambda: "MACHINE\\tester", exe=lambda: sys.executable)
    monkeypatch.setattr(service.psutil, "Process", lambda: current)
    process.exe.return_value = sys.executable
    process.username.return_value = "machine\\TESTER"
    process.cmdline.return_value[1] += ".exe"
    process.environ.return_value = {"USERPROFILE": process.environ.return_value["HOME"]}
    return process, data_dir


@pytest.mark.parametrize("mismatch", ["user", "launcher", "redirector", "data", "port", "wrapper"])
def test_windows_unrelated_or_redirector_process_is_rejected(windows_target, tmp_path, mismatch):
    process, data_dir = windows_target
    if mismatch == "user":
        process.username.return_value = "MACHINE\\another-user"
    elif mismatch == "launcher":
        process.cmdline.return_value[1] = str(tmp_path / "another-install" / "scenerecall.exe")
    elif mismatch == "redirector":
        process.exe.return_value = str(tmp_path / "venv-redirector.exe")
    elif mismatch == "data":
        process.cmdline.return_value[3] = str(tmp_path / "another-library")
    elif mismatch == "port":
        process.cmdline.return_value[5] = str(TEST_PORT + 1)
    else:
        process.cmdline.return_value = [sys.executable, "-c", "import scenerecall.main"]
    assert service.find_service(data_dir, TEST_PORT) is None
    process.terminate.assert_not_called()


def test_windows_profile_and_case_insensitive_owner_match(windows_target):
    process, _ = windows_target
    process.cmdline.return_value[2:] = ["start"]
    profile = Path(process.environ.return_value["USERPROFILE"])
    assert service._matches(process, profile / "SceneRecallLibrary", 8765)


@pytest.mark.parametrize("dry_run", [False, True])
def test_windows_stop_uses_cooperative_event_never_terminate(windows_target, monkeypatch, dry_run):
    process, data_dir = windows_target
    request = Mock()
    monkeypatch.setattr(windows_service, "request_shutdown", request)
    result = service.stop_service(data_dir, TEST_PORT, dry_run=dry_run)
    assert ("检查通过" if dry_run else "已停止") in result
    request.assert_called_once_with(process, data_dir, TEST_PORT, dry_run=dry_run)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    assert process.wait.called is not dry_run


@pytest.mark.parametrize("dry_run", [False, True])
def test_windows_missing_control_channel_refuses_even_dry_run(windows_target, monkeypatch, dry_run):
    process, data_dir = windows_target
    monkeypatch.setattr(windows_service, "request_shutdown", Mock(side_effect=FileNotFoundError()))
    with pytest.raises(service.ServiceError, match="没有可用的安全停止通道.*Ctrl\\+C"):
        service.stop_service(data_dir, TEST_PORT, dry_run=dry_run)
    process.terminate.assert_not_called()
    process.wait.assert_not_called()


def test_windows_control_access_denied_never_terminates(windows_target, monkeypatch):
    process, data_dir = windows_target
    monkeypatch.setattr(windows_service, "request_shutdown", Mock(side_effect=PermissionError()))
    with pytest.raises(service.ServiceError, match="权限不足.*同一系统用户"):
        service.stop_service(data_dir, TEST_PORT)
    process.terminate.assert_not_called()
    process.wait.assert_not_called()


def test_windows_stop_timeout_never_terminates(windows_target, monkeypatch):
    process, data_dir = windows_target
    monkeypatch.setattr(windows_service, "request_shutdown", Mock())
    monkeypatch.setattr(service, "_wait_for_exit", Mock(side_effect=psutil.TimeoutExpired(0.001)))
    with pytest.raises(service.ServiceError, match="未强制终止"):
        service.stop_service(data_dir, TEST_PORT, timeout=0.001)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_windows_control_name_binds_process_creation_installation_and_target(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=123, create_time=lambda: 1000.25)
    original = windows_service._event_name(process, tmp_path, TEST_PORT)
    assert original.startswith("Global\\SceneRecall.Stop.")
    assert windows_service._event_name(process, tmp_path / "other", TEST_PORT) != original
    assert windows_service._event_name(process, tmp_path, TEST_PORT + 1) != original
    assert windows_service._event_name(SimpleNamespace(pid=124, create_time=process.create_time), tmp_path, TEST_PORT) != original
    assert windows_service._event_name(SimpleNamespace(pid=123, create_time=lambda: 1001.25), tmp_path, TEST_PORT) != original
    monkeypatch.setattr(windows_service.sysconfig, "get_path", lambda key: str(tmp_path / "other-install"))
    assert windows_service._event_name(process, tmp_path, TEST_PORT) != original


@pytest.mark.parametrize("dry_run", [False, True])
def test_windows_event_client_checks_handle_and_closes_it(tmp_path, monkeypatch, dry_run):
    events = Mock()
    monkeypatch.setattr(windows_service, "_Events", lambda: events)
    monkeypatch.setattr(windows_service, "_event_name", lambda *args: "verified-event")
    windows_service.request_shutdown(Mock(), tmp_path, TEST_PORT, dry_run=dry_run)
    events.open.assert_called_once_with("verified-event")
    events.close.assert_called_once_with(events.open.return_value)
    assert events.signal.called is not dry_run


def test_windows_event_server_requests_normal_shutdown_and_closes_handle(tmp_path, monkeypatch):
    events = Mock()
    events.wait.return_value = True
    monkeypatch.setattr(windows_service, "_Events", lambda: events)
    monkeypatch.setattr(windows_service, "_event_name", lambda *args: "verified-event")
    server = SimpleNamespace(should_exit=False)
    with windows_service.shutdown_control(server, tmp_path, TEST_PORT):
        deadline = time.monotonic() + 2
        while not server.should_exit and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.should_exit
    events.create.assert_called_once_with("verified-event")
    events.close.assert_called_once_with(events.create.return_value)


@pytest.mark.skipif(os.name != "nt", reason="native Windows object ACL")
def test_native_windows_event_acl_and_lifetime(tmp_path):
    import ctypes
    from ctypes import wintypes

    events = windows_service._Events()
    events.security.GetSecurityDescriptorOwner.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                                           ctypes.POINTER(wintypes.BOOL)]
    events.security.GetSecurityDescriptorOwner.restype = wintypes.BOOL

    def numeric_sid(trustee):
        # SDDL abbreviates built-in identities, including the Administrator
        # account as LA. Ask Windows to resolve the actual SID rather than
        # comparing aliases or granting any additional trustee permission.
        descriptor = ctypes.c_void_p()
        assert events.security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"O:{trustee}", 1, ctypes.byref(descriptor), None)
        try:
            sid, defaulted = ctypes.c_void_p(), wintypes.BOOL()
            assert events.security.GetSecurityDescriptorOwner(descriptor, ctypes.byref(sid), ctypes.byref(defaulted))
            text = wintypes.LPWSTR()
            assert events.security.ConvertSidToStringSidW(sid, ctypes.byref(text))
            try:
                return text.value
            finally:
                events.kernel.LocalFree(text)
        finally:
            events.kernel.LocalFree(descriptor)

    name = windows_service._event_name(psutil.Process(), tmp_path, TEST_PORT)
    handle = events.create(name)
    try:
        with pytest.raises(FileExistsError):
            events.create(name)
        events.security.GetSecurityInfo.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.DWORD,
                                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        events.security.GetSecurityInfo.restype = wintypes.DWORD
        descriptor = ctypes.c_void_p()
        assert events.security.GetSecurityInfo(handle, 6, 4, None, None, None, None, ctypes.byref(descriptor)) == 0
        try:
            convert = events.security.ConvertSecurityDescriptorToStringSecurityDescriptorW
            convert.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                ctypes.POINTER(wintypes.LPWSTR), ctypes.c_void_p]
            convert.restype = wintypes.BOOL
            text = wintypes.LPWSTR()
            assert convert(descriptor, 1, 4, ctypes.byref(text), None)
            try:
                assert text.value.startswith("D:P")
                trustees = re.findall(r";;;([^)]*)\)", text.value)
                assert len(trustees) == 2
                assert {numeric_sid(trustee) for trustee in trustees} == {"S-1-5-18", events._user_sid()}
            finally:
                events.kernel.LocalFree(text)
        finally:
            events.kernel.LocalFree(descriptor)
        opened = events.open(name)
        events.signal(opened)
        events.close(opened)
        assert events.wait(handle)
    finally:
        events.close(handle)
    with pytest.raises(FileNotFoundError):
        events.open(name)


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


def _server_diagnostics(process, port, log_path):
    """Inspect only the isolated test child; never print its environment."""
    details = [f"test PID={process.pid}, port={port}, sys.executable={sys.executable}"]
    try:
        details.append(f"current OS executable={psutil.Process().exe()}")
        owner = psutil.Process(process.pid)
        for child in (owner, *owner.children(recursive=True)):
            try:
                details.append(str(child.as_dict(attrs=["pid", "exe", "cmdline", "cwd"])))
            except psutil.Error as exc:
                details.append(f"test child identity unavailable: {exc}")
    except psutil.Error as exc:
        details.append(f"test process inspection unavailable: {exc}")
    try:
        output = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        details.append("isolated test log:\n" + output)
    except OSError as exc:
        details.append(f"test log unavailable: {exc}")
    return re.sub(r"([?&]token=)[^&\s]+", r"\1<redacted>", "\n".join(details))


def test_synthetic_server_failure_diagnostics_redact_session_tokens(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("startup http://127.0.0.1:19321/?token=synthetic-secret-token\nHTTP 503", encoding="utf-8")
    diagnostic = _server_diagnostics(SimpleNamespace(pid=os.getpid()), TEST_PORT, log)
    assert "synthetic-secret-token" not in diagnostic
    assert "token=<redacted>" in diagnostic and "HTTP 503" in diagnostic


@contextmanager
def _running_server(command, cwd, port, health_path, log_path):
    # A waiter reaps the child even while a separate stop CLI waits for its exit.
    with log_path.open("w+") as log:
        environment = {**os.environ, "PYTHONUTF8": "1"}
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=environment)
        waiter = threading.Thread(target=process.wait, daemon=True)
        waiter.start()
        try:
            deadline = time.monotonic() + 30
            last_result = "no HTTP response"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(f"isolated test server exited with {process.returncode}\n"
                                + _server_diagnostics(process, port, log_path))
                try:
                    status = _http_status(port, health_path)
                    last_result = f"HTTP {status} for {health_path}"
                    if status == 200:
                        break
                except (OSError, http.client.HTTPException) as exc:
                    last_result = f"{type(exc).__name__}: {exc}"
                time.sleep(0.05)
            else:
                pytest.fail(f"isolated test server did not become healthy: {last_result}\n"
                            + _server_diagnostics(process, port, log_path))
            yield process
        finally:
            if process.poll() is None:
                # Windows console/venv launchers wait for a Python child. If a
                # test fails, clean up all of this test-owned launcher's children
                # as well, so a failed assertion cannot leave a listener behind.
                if os.name == "nt":
                    try:
                        children = psutil.Process(process.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        children = []
                    for child in reversed(children):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                process.terminate()
            waiter.join(timeout=10)
            if waiter.is_alive():
                process.kill()  # Cleanup only this test-owned Popen child.
                waiter.join(timeout=5)


@pytest.mark.parametrize("start_command", [[], ["start"]], ids=["legacy-start", "explicit-start"])
def test_real_cli_stop_preserves_library_and_refuses_wrong_directory(tmp_path, start_command):
    launcher = Path(sysconfig.get_path("scripts")) / ("scenerecall.exe" if os.name == "nt" else "scenerecall")
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
        run_stop = partial(subprocess.run, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=15,
                           env={**os.environ, "PYTHONUTF8": "1"})
        wrong = run_stop([str(launcher), "stop", "--data-dir", str(tmp_path / "wrong"), "--port", str(port)])
        assert wrong.returncode == 1 and "无法确认" in wrong.stderr
        assert process.poll() is None and _http_status(port, "/api/health") == 200
        dry_run = run_stop([str(launcher), "stop", "--data-dir", str(data_dir), "--port", str(port), "--dry-run"])
        assert dry_run.returncode == 0 and "未发送信号" in dry_run.stdout, (
            dry_run.stderr + "\n" + _server_diagnostics(process, port, tmp_path / "server.log"))
        assert process.poll() is None and _http_status(port, "/api/health") == 200
        stop = run_stop([str(launcher), "stop", "--data-dir", str(data_dir), "--port", str(port), "--timeout=5"])
        assert stop.returncode == 0, stop.stderr
        assert "已停止" in stop.stdout and process.wait(timeout=5) is not None
        assert "Application shutdown complete." in (tmp_path / "server.log").read_text(encoding="utf-8")
        assert preserved.read_text(encoding="utf-8") == "保留用户资料"
        assert settings.read_text(encoding="utf-8") == '{"defaults": {"window_ms": 1234}}'
        repeat = run_stop([str(launcher), "stop", "--data-dir", str(data_dir), "--port", str(port)])
        assert repeat.returncode == 0 and "未运行" in repeat.stdout


def test_real_unrelated_http_server_is_not_stopped(tmp_path):
    port = _unused_port()
    (tmp_path / "health.txt").write_text("synthetic test server", encoding="utf-8")
    # HTTPServer.server_bind() performs getfqdn() before listening. A hosted
    # runner's reverse DNS must not determine whether this unrelated local
    # process can be tested. TCPServer needs only the numeric loopback address;
    # retain the real stdlib HTTP handler and fail if DNS is ever reintroduced.
    script = """
from functools import partial
from http.server import SimpleHTTPRequestHandler
import socket
from socketserver import TCPServer
import sys

def forbid_dns(*args):
    raise AssertionError('the isolated loopback fixture must not perform DNS')

socket.getfqdn = socket.gethostbyaddr = forbid_dns
handler = partial(SimpleHTTPRequestHandler, directory=sys.argv[2])
with TCPServer(('127.0.0.1', int(sys.argv[1])), handler) as server:
    print('synthetic HTTP server ready', flush=True)
    server.serve_forever()
"""
    command = [sys.executable, "-u", "-c", script, str(port), str(tmp_path)]
    with _running_server(command, tmp_path, port, "/health.txt", tmp_path / "http.log") as process:
        with pytest.raises(service.ServiceError, match="无法确认.*未发送停止信号"):
            service.stop_service(tmp_path / "not-a-library", port)
        assert process.poll() is None and _http_status(port, "/health.txt") == 200
