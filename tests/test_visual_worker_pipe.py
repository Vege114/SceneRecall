"""Exercise the actual subprocess pipe protocol without model dependencies."""
import subprocess
import sys

import psutil
import pytest

from scenerecall.visual_identity import IdentityUnavailable, ImgutilsBackend


def worker(monkeypatch, source, *, timeout_s=5):
    spawn = subprocess.Popen

    def start(_command, **options):
        return spawn([sys.executable, "-u", "-c", source], **options)

    monkeypatch.setattr(subprocess, "Popen", start)
    return ImgutilsBackend("pipe-test", timeout_s=timeout_s)


def test_resident_worker_uses_subprocess_pipe_for_multiple_replies(monkeypatch):
    backend = worker(monkeypatch, """
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({'result': request['action']}), flush=True)
""")
    try:
        assert backend._request("first") == "first"
        process = backend._process
        assert backend._request("second") == "second"
        assert backend._process is process
    finally:
        backend.close()


def test_partial_worker_reply_obeys_deadline_and_stops_process(monkeypatch):
    backend = worker(monkeypatch, """
import sys, time
sys.stdin.readline()
sys.stdout.write('{')
sys.stdout.flush()
time.sleep(30)
""", timeout_s=.2)
    try:
        with pytest.raises(IdentityUnavailable, match="超时"):
            backend._request("partial")
        assert backend._process is None
    finally:
        backend.close()


def test_closed_worker_pipe_reports_unavailability(monkeypatch):
    backend = worker(monkeypatch, "import sys; sys.stdin.readline()")
    try:
        with pytest.raises(IdentityUnavailable, match="worker 不可用"):
            backend._request("closed")
        assert backend._process is None
    finally:
        backend.close()


def test_worker_unicode_reply_has_explicit_matching_utf8_encoding(monkeypatch):
    backend = worker(monkeypatch, """
import json, os, sys
sys.stdin.readline()
assert os.environ['PYTHONIOENCODING'] == 'utf-8'
print(json.dumps({'result': '\u4e2d\u6587\u6a21\u578b\u8def\u5f84'}, ensure_ascii=False), flush=True)
""")
    try:
        assert backend._request("unicode") == "中文模型路径"
    finally:
        backend.close()


@pytest.mark.parametrize("timeout", [False, True])
def test_close_reaps_actual_worker_behind_python_venv_launcher(monkeypatch, timeout):
    backend = worker(monkeypatch, """
import json, os, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if request['action'] == 'stall':
        sys.stdout.write('{')
        sys.stdout.flush()
        time.sleep(30)
    print(json.dumps({'result': os.getpid()}), flush=True)
""", timeout_s=.3 if timeout else 5)
    actual_worker = None
    try:
        actual_worker = psutil.Process(backend._request("pid"))
        if timeout:
            with pytest.raises(IdentityUnavailable, match="超时"):
                backend._request("stall")
        else:
            backend.close()
        assert backend._process is None
        try:
            actual_worker.wait(timeout=1)
        except psutil.TimeoutExpired:
            pytest.fail("Closing the backend left its actual Python worker running")
    finally:
        backend.close()
        if actual_worker is not None and actual_worker.is_running():
            actual_worker.kill()
            actual_worker.wait(timeout=2)
