"""Failure classifications used by adaptive visual preparation, without model calls."""
import asyncio
import json
import os
import sys

import httpx
import pytest
from PIL import Image

from scenerecall import codex_cli
from scenerecall.providers import ProviderError, ProviderManager


@pytest.fixture
def codex_manager(tmp_path):
    providers = ProviderManager(tmp_path / "settings")
    providers.upsert({"id": "codex", "name": "local inference", "provider_type": "codex_cli",
                      "capabilities": ["preparation", "query"]})
    return providers


def evidence_frames(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "blue").save(path)
    return [{"id": f"f{index}", "at_ms": index * 1000, "path": str(path)} for index in range(2)]


@pytest.mark.asyncio
async def test_visual_invalid_partition_stops_after_one_cli_turn_and_preserves_usage(
        tmp_path, monkeypatch, codex_manager):
    calls = []

    async def complete(*args, **kwargs):
        calls.append(args)
        return codex_cli.CodexCLIResult(
            content=json.dumps({"chunks": [{"frame_ids": ["invented-frame"], "summary": "无效分块",
                                            "representative_frame_ids": ["invented-frame"],
                                            "needs_deep": False}], "unresolved": []}),
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    monkeypatch.setattr(codex_cli, "complete", complete)
    with pytest.raises(ProviderError) as caught:
        await codex_manager.prepare_visual("codex", evidence_frames(tmp_path), [], None, None)
    assert caught.value.code == "invalid_output"
    assert caught.value.request_count == len(calls) == 1
    assert caught.value.usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert caught.value.cost is None


@pytest.mark.parametrize("code", ["timeout", None])
@pytest.mark.asyncio
async def test_cli_failure_classification_and_partial_usage_reach_provider(
        tmp_path, monkeypatch, codex_manager, code):
    calls = []

    async def complete(*args, **kwargs):
        calls.append(args)
        raise codex_cli.CodexCLIError("推理失败", code=code, request_count=1,
                                    usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})

    monkeypatch.setattr(codex_cli, "complete", complete)
    with pytest.raises(ProviderError) as caught:
        await codex_manager.prepare_visual("codex", evidence_frames(tmp_path), [], None, None)
    assert caught.value.code == code
    assert caught.value.request_count == len(calls) == 1
    assert caught.value.usage["total_tokens"] == 6
    assert caught.value.cost is None


@pytest.mark.asyncio
async def test_complete_preserves_inner_process_timeout_classification_and_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_cli, "_executable", lambda: sys.executable)

    async def inspect(*args):
        return {"compatible": True, "authenticated": True}

    async def run(*args, on_line=None, **kwargs):
        on_line(json.dumps({"type": "turn.completed",
                            "usage": {"input_tokens": 5, "output_tokens": 3}}).encode())
        raise codex_cli.CodexCLIError("请求超时", code="timeout")

    monkeypatch.setattr(codex_cli, "_inspect", inspect)
    monkeypatch.setattr(codex_cli, "_run", run)
    with pytest.raises(codex_cli.CodexCLIError) as caught:
        await codex_cli.complete("Return JSON", "data", "", 5)
    assert caught.value.code == "timeout"
    assert caught.value.request_count == 1
    assert caught.value.usage == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}


@pytest.mark.asyncio
async def test_complete_deadline_during_preflight_has_no_model_request(monkeypatch):
    monkeypatch.setattr(codex_cli, "_executable", lambda: sys.executable)

    async def inspect(*args):
        await asyncio.Event().wait()

    async def run(*args, **kwargs):
        pytest.fail("preflight deadline must not start a model turn")

    monkeypatch.setattr(codex_cli, "_inspect", inspect)
    monkeypatch.setattr(codex_cli, "_run", run)
    with pytest.raises(codex_cli.CodexCLIError) as caught:
        await codex_cli.complete("Return JSON", "data", "", 0.01)
    assert caught.value.code == "timeout"
    assert caught.value.request_count == 0
    assert caught.value.usage["total_tokens"] == 0


@pytest.mark.asyncio
async def test_real_child_timeout_is_classified_and_reaped(tmp_path):
    pid_path = tmp_path / "child.pid"
    script = ("import os,time\nfrom pathlib import Path\n"
              f"Path({str(pid_path)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n")
    with pytest.raises(codex_cli.CodexCLIError) as caught:
        await codex_cli._run([sys.executable, "-c", script], directory=tmp_path, timeout_s=0.5)
    assert caught.value.code == "timeout"
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_path.read_text()), 0)


@pytest.mark.parametrize("failure,code,attempts", [
    ("timeout", "timeout", 3), ("connect_timeout", None, 3), ("network", None, 3), ("authentication", None, 1),
    ("invalid_then_authentication", None, 2), ("invalid_output", "invalid_output", 3),
])
@pytest.mark.asyncio
async def test_api_exhaustion_classifies_only_current_adaptable_failure(tmp_path, monkeypatch, failure, code, attempts):
    calls = []

    async def no_wait(*args):
        pass

    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_wait)

    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("deadline", request=request)
        if failure == "connect_timeout":
            raise httpx.ConnectTimeout("offline", request=request)
        if failure == "network":
            raise httpx.ConnectError("offline", request=request)
        if failure == "authentication" or (failure == "invalid_then_authentication" and len(calls) == 2):
            return httpx.Response(403)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}],
                                        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                                  "total_tokens": 15}})

    providers = ProviderManager(tmp_path, transport=httpx.MockTransport(handler))
    providers.upsert({"id": "api", "name": "fixture", "base_url": "https://fixture.test/v1",
                      "model": "fixture", "capabilities": ["query"], "secret_mode": "none"})
    with pytest.raises(ProviderError) as caught:
        await providers.interpret("api", "find a scene")
    assert caught.value.code == code
    assert caught.value.request_count == len(calls) == attempts
    expected_usage = 15 * (attempts if failure == "invalid_output" else int(failure == "invalid_then_authentication"))
    assert caught.value.usage["total_tokens"] == expected_usage
