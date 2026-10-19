"""Windows background control refuses unrelated PIDs and exits through Uvicorn."""
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest


PROCESS_CLASS = psutil.Process
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/manage_windows_runtime.py"
SPEC = importlib.util.spec_from_file_location("windows_runtime_manager", SCRIPT)
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


@pytest.fixture
def identity(tmp_path, monkeypatch):
    data_dir = (tmp_path / "library").resolve()
    monkeypatch.setattr(manager, "current_user", lambda: "test-user")
    state = {"version": 1, "project_root": str(manager.PROJECT_ROOT), "data_dir": str(data_dir),
             "port": 19391, "username": "test-user", "launcher_pid": 12345, "launcher_created": 100.0,
             "launcher_command": manager.command(data_dir, 19391, None, None),
             "runtime_config": None, "ffmpeg_directory": None, "log_path": str(tmp_path / "server.log")}
    launcher = Mock(spec=PROCESS_CLASS)
    launcher.pid = 12345
    launcher.create_time.return_value = 100.0
    launcher.username.return_value = "TEST-USER"
    launcher.exe.return_value = str(manager.powershell())
    launcher.cmdline.return_value = list(state["launcher_command"])
    launcher.cwd.return_value = str(manager.PROJECT_ROOT)
    monkeypatch.setattr(manager.psutil, "Process", lambda pid: launcher)
    return data_dir, state, launcher


def test_recycled_pid_is_not_a_target(identity):
    _, state, launcher = identity
    launcher.create_time.return_value = 101.0
    assert manager.launcher_process(state) is None
    launcher.cmdline.assert_not_called()
    launcher.terminate.assert_not_called()


@pytest.mark.parametrize("mismatch", ["user", "exe", "arguments", "directory"])
def test_launcher_requires_live_identity(identity, mismatch, tmp_path):
    _, state, launcher = identity
    if mismatch == "user":
        launcher.username.return_value = "another-user"
    elif mismatch == "exe":
        launcher.exe.return_value = str(tmp_path / "another-powershell.exe")
    elif mismatch == "arguments":
        launcher.cmdline.return_value += ["-Command", "unrelated"]
    else:
        launcher.cwd.return_value = str(tmp_path)
    with pytest.raises(manager.RuntimeErrorDetail, match="identity changed"):
        manager.launcher_process(state)
    launcher.terminate.assert_not_called()


@pytest.mark.parametrize("mismatch", ["launcher_command", "username", "project_root", "data_dir", "port"])
def test_state_cannot_authorize_another_target(identity, tmp_path, mismatch):
    data_dir, state, _ = identity
    if mismatch == "launcher_command":
        state[mismatch][6] = str(tmp_path / "another-start.ps1")
    elif mismatch == "port":
        state[mismatch] += 1
    else:
        state[mismatch] = "unrelated"
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(manager.RuntimeErrorDetail, match="Invalid runtime state"):
        manager.load_state(path, data_dir, 19391)


@pytest.mark.parametrize("mismatch", [None, "user", "exe", "module", "data", "port", "listener"])
def test_server_requires_descendant_interpreter_arguments_and_loopback_listener(identity, tmp_path, mismatch):
    data_dir, _, launcher = identity
    child = Mock(spec=PROCESS_CLASS)
    child.pid = 12346
    child.cmdline.return_value = [str(manager.APP_PYTHON), "-m", "scenerecall.main", "--data-dir", str(data_dir),
                                 "--port", "19391"]
    child.username.return_value = "test-user"
    child.exe.return_value = str(manager.APP_PYTHON)
    child.cwd.return_value = str(manager.PROJECT_ROOT)
    child.net_connections.return_value = [SimpleNamespace(
        status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.1", port=19391))]
    launcher.children.return_value = [child]
    if mismatch == "user":
        child.username.return_value = "another-user"
    elif mismatch == "exe":
        child.exe.return_value = str(tmp_path / "another-python.exe")
    elif mismatch == "module":
        child.cmdline.return_value[2] = "another.main"
    elif mismatch == "data":
        child.cmdline.return_value[4] = str(tmp_path / "another-library")
    elif mismatch == "port":
        child.cmdline.return_value[6] = "19392"
    elif mismatch == "listener":
        child.net_connections.return_value[0].laddr.ip = "0.0.0.0"
    actual = manager.server_process(launcher, data_dir, 19391, {manager.normalized(manager.APP_PYTHON)})
    assert actual is (child if mismatch is None else None)
    child.terminate.assert_not_called()


def test_foreign_listener_never_receives_stop_signal(identity, tmp_path, monkeypatch):
    data_dir, _, _ = identity
    monkeypatch.setattr(manager, "port_busy", lambda port: True)
    signal = Mock()
    monkeypatch.setattr(manager, "request_break", signal)
    args = SimpleNamespace(data_dir=data_dir, port=19391, dry_run=False, timeout=1)
    with pytest.raises(manager.RuntimeErrorDetail, match="unverified_port_owner"):
        manager.stop(args, tmp_path / "state.json", None, set())
    signal.assert_not_called()


@pytest.mark.parametrize("exited", [False, True])
@pytest.mark.parametrize("busy", [False, True])
def test_inspect_rechecks_launcher_that_exits_during_child_enumeration(identity, monkeypatch, exited, busy):
    data_dir, state, launcher = identity
    launcher.children.side_effect = psutil.NoSuchProcess(launcher.pid)
    if exited:
        launcher.create_time.side_effect = [100.0, psutil.NoSuchProcess(launcher.pid)]
    monkeypatch.setattr(manager, "port_busy", lambda port: busy)
    result = manager.inspect(state, data_dir, 19391, set())
    expected = "unverified_port_owner" if busy else "stopped" if exited else "starting_or_stopping"
    assert result["status"] == expected
    assert result["launcher_pid"] == (None if exited else launcher.pid)
    assert result["server_pid"] is None


@pytest.mark.parametrize("error", [psutil.NoSuchProcess, psutil.ZombieProcess])
def test_inspect_rechecks_server_that_exits_during_identity_validation(identity, monkeypatch, error):
    data_dir, state, launcher = identity
    server = Mock(spec=PROCESS_CLASS)
    server.pid = 12346
    server.create_time.side_effect = error(server.pid)
    state.update(server_pid=server.pid, server_created=101.0, server_command=["known-python"])
    monkeypatch.setattr(manager, "server_process", lambda *args: server)
    monkeypatch.setattr(manager, "port_busy", lambda port: False)
    result = manager.inspect(state, data_dir, 19391, set())
    assert result["status"] == "starting_or_stopping"
    assert result["launcher_pid"] == launcher.pid
    assert result["server_pid"] is None


@pytest.mark.parametrize("phase", ["children", "server_identity"])
def test_inspect_never_treats_access_denied_as_an_exited_process(identity, monkeypatch, phase):
    data_dir, state, launcher = identity
    if phase == "children":
        launcher.children.side_effect = psutil.AccessDenied(launcher.pid)
    else:
        server = Mock(spec=PROCESS_CLASS)
        server.pid = 12346
        server.create_time.side_effect = psutil.AccessDenied(server.pid)
        state.update(server_pid=server.pid, server_created=101.0, server_command=["known-python"])
        monkeypatch.setattr(manager, "server_process", lambda *args: server)
    with pytest.raises(psutil.AccessDenied):
        manager.inspect(state, data_dir, 19391, set())


def test_stop_succeeds_when_launcher_exits_between_post_signal_checks(identity, tmp_path, monkeypatch):
    data_dir, state, launcher = identity
    child = Mock(spec=PROCESS_CLASS)
    child.pid = 12346
    child.cmdline.return_value = [str(manager.APP_PYTHON), "-m", "scenerecall.main", "--data-dir", str(data_dir),
                                 "--port", "19391"]
    child.username.return_value = "test-user"
    child.exe.return_value = str(manager.APP_PYTHON)
    child.cwd.return_value = str(manager.PROJECT_ROOT)
    child.net_connections.return_value = [SimpleNamespace(
        status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.1", port=19391))]
    launcher.children.side_effect = [[child], psutil.NoSuchProcess(launcher.pid)]
    launcher.create_time.side_effect = [100.0, 100.0, psutil.NoSuchProcess(launcher.pid)]
    monkeypatch.setattr(manager, "port_busy", lambda port: False)
    signal = Mock()
    monkeypatch.setattr(manager, "request_break", signal)
    path = tmp_path / "state.json"
    path.write_text("synthetic state", encoding="utf-8")
    args = SimpleNamespace(data_dir=data_dir, port=19391, dry_run=False, timeout=1)
    result = manager.stop(args, path, state, {manager.normalized(manager.APP_PYTHON)})
    assert result["status"] == "stopped" and result["signal_sent"]
    assert not path.exists()
    signal.assert_called_once_with(state, data_dir, 19391, {manager.normalized(manager.APP_PYTHON)})
    launcher.terminate.assert_not_called()


def test_graceful_stop_timeout_never_escalates(identity, tmp_path, monkeypatch):
    data_dir, state, launcher = identity
    monkeypatch.setattr(manager, "inspect", lambda *args: {"status": "running"})
    signal = Mock()
    monkeypatch.setattr(manager, "request_break", signal)
    args = SimpleNamespace(data_dir=data_dir, port=19391, dry_run=False, timeout=0.000001)
    with pytest.raises(manager.RuntimeErrorDetail, match="no force stop"):
        manager.stop(args, tmp_path / "state.json", state, set())
    signal.assert_called_once()
    launcher.terminate.assert_not_called()
    launcher.kill.assert_not_called()


@pytest.mark.parametrize("mismatch", ["pid", "created", "arguments"])
def test_listener_identity_is_checked_again_before_signalling(identity, mismatch):
    _, state, _ = identity
    state.update(server_pid=12346, server_created=101.0, server_command=["known-python", "known-module"])
    server = Mock(spec=PROCESS_CLASS)
    server.pid = 12346
    server.create_time.return_value = 101.0
    server.cmdline.return_value = list(state["server_command"])
    if mismatch == "pid":
        server.pid += 1
    elif mismatch == "created":
        server.create_time.return_value += 1
    else:
        server.cmdline.return_value += ["unrelated"]
    with pytest.raises(manager.RuntimeErrorDetail, match="Server identity changed"):
        manager.check_server_identity(state, server)
    server.terminate.assert_not_called()


@pytest.mark.skipif(sys.platform != "win32", reason="real native Windows console regression")
def test_real_hidden_start_status_refusal_and_graceful_stop(tmp_path):
    if not manager.APP_PYTHON.is_file() or not (manager.PROJECT_ROOT / "frontend/dist/index.html").is_file():
        pytest.skip("native Windows development runtime is not installed")
    data_dir = (tmp_path / "Windows 中文 library").resolve()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def invoke(action, *extra, library=data_dir, check=True):
        result = subprocess.run([str(manager.APP_PYTHON), str(SCRIPT), action, "--data-dir", str(library),
                                 "--port", str(port), "--timeout", "20", *extra],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        if check:
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)
        return result

    started = invoke("start")
    try:
        assert started["status"] == "running"
        assert invoke("start")["server_pid"] == started["server_pid"]
        assert invoke("status")["server_pid"] == started["server_pid"]
        rejected = invoke("stop", library=tmp_path / "another-library", check=False)
        assert rejected.returncode == 1 and "no signal was sent" in rejected.stderr
        assert psutil.Process(started["server_pid"]).is_running()

        # A matching port and edited PID file still cannot authorize a signal to pytest.
        path = manager.state_path(data_dir, port)
        original = path.read_bytes()
        state = json.loads(original)
        state["launcher_pid"] = psutil.Process().pid
        state["launcher_created"] = psutil.Process().create_time()
        path.write_text(json.dumps(state), encoding="utf-8")
        try:
            rejected = invoke("stop", check=False)
            assert rejected.returncode == 1 and "identity changed" in rejected.stderr
        finally:
            path.write_bytes(original)
        verified = invoke("stop", "--dry-run")
        assert verified["stop_verified"] and not verified["signal_sent"]
        assert manager.health_ready(port)
    finally:
        stopped = invoke("stop")
    assert stopped["status"] == "stopped" and stopped["signal_sent"]
    assert not manager.health_ready(port)
    assert invoke("stop")["status"] == "stopped"
    log = Path(started["log_path"]).read_text(encoding="utf-8")
    assert str(data_dir) in log
    assert "Application shutdown complete." in log
