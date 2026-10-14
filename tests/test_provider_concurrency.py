"""Manual concurrency has no application ceiling; connectors still bound work."""
import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from scenerecall import codex_cli
from scenerecall.execution import execution_width
from scenerecall.models import ExecutionOptions
from scenerecall.providers import Profile, ProviderManager


def connection(provider_type="openai_compatible", **changes):
    return {"id": "manual", "name": "explicit synthetic connector", "provider_type": provider_type,
            "base_url": "https://fixture.invalid/v1", "model": "synthetic", "capabilities": ["query"],
            "secret_mode": "none", **changes}


@pytest.mark.parametrize("provider_type", ["openai_compatible", "codex_cli"])
def test_fresh_connector_defaults_to_sixteen_and_legacy_serial_requests_stay_compatible(tmp_path, provider_type):
    task = ExecutionOptions()
    profile = Profile(**connection(provider_type))
    assert task.max_concurrency == 3 and profile.max_concurrency == 16
    assert execution_width({"execution": task.model_dump()}, "transcript", profile.model_dump()) == 1
    assert execution_width({"execution": {"transcript": "parallel"}}, "transcript") == 4
    # Missing execution in old queued jobs still means serial scheduling.
    assert execution_width({}, "transcript", profile.model_dump()) == 1
    providers = ProviderManager(tmp_path)
    saved = providers.upsert(connection(provider_type))
    assert saved["max_concurrency"] == 16
    assert ProviderManager(tmp_path).get("manual")["max_concurrency"] == 16


@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("provider_type", ["openai_compatible", "codex_cli"])
def test_existing_task_and_connector_values_survive_factory_default_change(tmp_path, width, provider_type):
    task = ExecutionOptions(transcript="parallel", max_concurrency=width)
    assert ExecutionOptions.model_validate(task.model_dump()).max_concurrency == width
    assert execution_width({"execution": task.model_dump()}, "transcript", {"max_concurrency": 4}) == width
    providers = ProviderManager(tmp_path)
    saved = providers.upsert(connection(provider_type, max_concurrency=width))
    assert saved["max_concurrency"] == width
    renamed = providers.upsert({"id": "manual", "name": "renamed fixture"})
    assert renamed["max_concurrency"] == width
    assert ProviderManager(tmp_path).get("manual")["max_concurrency"] == width


@pytest.mark.parametrize("width", [9, 32, 64, 128])
@pytest.mark.parametrize("provider_type", ["openai_compatible", "codex_cli"])
def test_positive_manual_values_above_previous_caps_are_preserved(width, provider_type):
    task = ExecutionOptions(transcript="parallel", max_concurrency=width)
    profile = Profile(**connection(provider_type, max_concurrency=width))
    assert task.max_concurrency == profile.max_concurrency == width
    assert execution_width({"execution": task.model_dump()}, "transcript", profile.model_dump()) == width
    assert execution_width({"execution": task.model_dump()}, "transcript", {"max_concurrency": 5}) == 5


@pytest.mark.parametrize("invalid", [0, -1, True, False, "3", 3.0, 1.5, float("inf"), float("nan"), None])
def test_task_and_connection_concurrency_require_strict_positive_integer(invalid):
    with pytest.raises(ValidationError):
        ExecutionOptions(max_concurrency=invalid)
    for provider_type in ("openai_compatible", "codex_cli"):
        with pytest.raises(ValidationError):
            Profile(**connection(provider_type, max_concurrency=invalid))


@pytest.mark.parametrize("provider_type", ["openai_compatible", "codex_cli"])
@pytest.mark.asyncio
async def test_connector_semaphore_above_sixteen_runs_selected_width(tmp_path, monkeypatch, provider_type):
    """Neither transport contacts the network or launches a real Codex process."""
    entered, active, peak = 0, 0, 0
    thirty_two_started, release = asyncio.Event(), asyncio.Event()

    async def hold():
        nonlocal entered, active, peak
        entered += 1
        active += 1
        peak = max(peak, active)
        if entered == 32:
            thirty_two_started.set()
        await release.wait()
        active -= 1

    async def transport(request):
        await hold()
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"search_text": "合成结果"})},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    async def complete(*args):
        await hold()
        return codex_cli.CodexCLIResult(content=json.dumps({"search_text": "合成结果"}),
                                        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    monkeypatch.setattr(codex_cli, "complete", complete)
    providers = ProviderManager(tmp_path, transport=httpx.MockTransport(transport))
    saved = providers.upsert(connection(provider_type, max_concurrency=32))
    assert saved["max_concurrency"] == 32
    pending = [asyncio.create_task(providers.interpret("manual", f"合成问题 {index}")) for index in range(40)]
    try:
        await asyncio.wait_for(thirty_two_started.wait(), 2)
        await asyncio.sleep(0)
        assert active == entered == peak == 32
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*pending), 2)
    finally:
        release.set()
        await asyncio.gather(*pending, return_exceptions=True)
    assert entered == 40 and peak == 32 and active == 0
    assert all(result.data["search_text"] == "合成结果" for result in results)
    assert providers._limiters["manual"].available == 32


@pytest.mark.parametrize("provider_type", ["openai_compatible", "codex_cli"])
@pytest.mark.asyncio
async def test_live_cap_changes_share_existing_calls_and_wake_waiters(tmp_path, monkeypatch, provider_type):
    active, peak, entered = 0, 0, 0
    gates = [asyncio.Event() for _ in range(5)]
    starts = {count: asyncio.Event() for count in range(1, 6)}
    finished = {count: asyncio.Event() for count in range(1, 6)}
    completed = 0

    async def hold():
        nonlocal active, peak, entered, completed
        index = entered
        entered += 1
        active += 1
        peak = max(peak, active)
        starts[entered].set()
        await gates[index].wait()
        active -= 1
        completed += 1
        finished[completed].set()

    async def transport(request):
        await hold()
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"search_text":"合成结果"}'}}],
                                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    async def complete(*args):
        await hold()
        return codex_cli.CodexCLIResult('{"search_text":"合成结果"}')

    monkeypatch.setattr(codex_cli, "complete", complete)
    providers = ProviderManager(tmp_path, transport=httpx.MockTransport(transport))
    providers.upsert(connection(provider_type, max_concurrency=2))
    pending = [asyncio.create_task(providers.interpret("manual", f"合成问题 {index}")) for index in range(5)]
    try:
        await asyncio.wait_for(starts[2].wait(), 2)
        assert active == entered == 2
        # Raising the cap admits one waiting call, counting the two already running.
        providers.upsert({"id": "manual", "max_concurrency": 3})
        await asyncio.wait_for(starts[3].wait(), 2)
        assert active == entered == 3
        # Lowering cannot revoke paid calls; it blocks admission until all three drain.
        providers.upsert({"id": "manual", "max_concurrency": 1})
        for index in range(2):
            gates[index].set()
            await asyncio.wait_for(finished[index + 1].wait(), 2)
            await asyncio.sleep(0)
            assert entered == 3
        gates[2].set()
        await asyncio.wait_for(starts[4].wait(), 2)
        assert active == 1 and entered == 4
        gates[3].set()
        await asyncio.wait_for(starts[5].wait(), 2)
        assert active == 1 and entered == 5
        gates[4].set()
        await asyncio.wait_for(asyncio.gather(*pending), 2)
    finally:
        for gate in gates:
            gate.set()
        await asyncio.gather(*pending, return_exceptions=True)
    assert active == 0 and peak == 3
    assert providers._limiters["manual"].available == 1
