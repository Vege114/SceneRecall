"""Real pipe lifecycle tests without optional models or downloaded weights."""
import subprocess
import sys
import threading

import pytest

from scenerecall import visual_identity
from scenerecall.visual_identity import IdentityUnavailable, ImgutilsBackend


@pytest.fixture
def worker(monkeypatch):
    original = subprocess.Popen
    processes = []
    backends = []

    def create(script, *, timeout=5):
        def start(_args, **kwargs):
            process = original([sys.executable, "-u", "-c", script], **kwargs)
            processes.append(process)
            return process

        monkeypatch.setattr(visual_identity.subprocess, "Popen", start)
        backend = ImgutilsBackend(timeout_s=timeout)
        backends.append(backend)
        return backend, processes

    yield create
    for backend in backends:
        backend.close()


def test_worker_reuses_real_pipes_and_preserves_unicode(worker):
    backend, processes = worker("""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({'result': request}, ensure_ascii=False), flush=True)
""")
    assert backend._request("detect", path="C:/动画 视频/第一集.png")["path"] == "C:/动画 视频/第一集.png"
    assert backend._request("feature", path="C:/动画 视频/第二集.png")["action"] == "feature"
    assert len(processes) == 1
    backend.close()
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed
    assert backend._reader is None
    backend.close()


def test_worker_partial_line_timeout_reaps_process_and_reader(worker):
    backend, processes = worker("""
import sys, time
sys.stdin.readline()
sys.stdout.write('{"result":')
sys.stdout.flush()
time.sleep(60)
""", timeout=.2)
    with pytest.raises(IdentityUnavailable, match="超时"):
        backend._request("metadata")
    assert processes[0].poll() is not None
    assert backend._process is None and backend._reader is None
    assert not any(thread.name == "scenerecall-visual-output" for thread in threading.enumerate())


@pytest.mark.parametrize("output", ["not-json", ""])
def test_worker_invalid_output_or_eof_closes_pipes(worker, output):
    backend, processes = worker(f"import sys; sys.stdin.readline(); print({output!r}, flush=True)")
    with pytest.raises(IdentityUnavailable, match="worker 不可用"):
        backend._request("metadata")
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_worker_restarts_after_exited_process(worker):
    backend, processes = worker("""
import json, sys
for line in sys.stdin:
    print(json.dumps({'result': json.loads(line)['action']}), flush=True)
""")
    assert backend._request("first") == "first"
    processes[0].terminate()
    processes[0].wait(timeout=5)
    assert backend._request("second") == "second"
    assert len(processes) == 2
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_worker_broken_pipe_cleanup_preserves_user_error(worker):
    backend, processes = worker("""
import json, sys
for line in sys.stdin:
    print(json.dumps({'result': 'ready'}), flush=True)
""")
    assert backend._request("first") == "ready"
    original = processes[0].stdin

    class BrokenInput:
        def write(self, _value):
            raise BrokenPipeError("worker closed input")

        def close(self):
            original.close()
            raise BrokenPipeError("buffered input failed again during close")

    processes[0].stdin = BrokenInput()
    with pytest.raises(IdentityUnavailable, match="worker 不可用"):
        backend._request("second")
    assert processes[0].poll() is not None
    assert original.closed and processes[0].stdout.closed
    assert backend._process is None and backend._reader is None
