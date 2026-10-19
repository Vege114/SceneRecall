"""Authentication preflight concurrency; every CLI child is a fixture."""
import asyncio
from pathlib import Path

import pytest

from scenerecall import codex_cli
from test_codex_cli import cli  # noqa: F401


def ready():
    return {"installed": True, "compatible": True, "authenticated": True,
            "auth_method": "chatgpt", "version": "0.155.1", "message": "ready"}


@pytest.fixture(autouse=True)
def clear_status_cache():
    codex_cli._STATUS_CACHE.clear()
    codex_cli._STATUS_INFLIGHT.clear()
    yield
    codex_cli._STATUS_CACHE.clear()
    assert not codex_cli._STATUS_INFLIGHT


@pytest.mark.asyncio
async def test_each_preflight_command_has_its_own_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_cli, "STATUS_TIMEOUT_S", .05)
    calls = []

    async def run(args, *, timeout_s, **kwargs):
        calls.append(args[1:])
        async with asyncio.timeout(timeout_s):
            await asyncio.sleep(.02)
        if args[1:] == ["--version"]:
            return 0, b"codex-cli 0.155.1", b""
        if args[1:] == ["exec", "--help"]:
            return 0, " ".join(codex_cli._REQUIRED_FLAGS).encode(), b""
        if args[1:] == ["features", "list"]:
            return 0, "\n".join((*codex_cli._DISABLED_FEATURES, "skip_host_skill_discovery")).encode(), b""
        return 0, b"", b"Logged in using ChatGPT"

    monkeypatch.setattr(codex_cli, "_run", run)
    checked = await codex_cli._inspect("fixture", tmp_path)
    # Four individually valid calls take > .05 seconds in total. The old shared
    # deadline rejected this successfully authenticated installation.
    assert checked["authenticated"] and checked["compatible"] and len(calls) == 4


@pytest.mark.asyncio
async def test_parallel_callers_share_owned_check_and_cancellation_does_not_abort_siblings(tmp_path, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    directories = []

    async def inspect(executable, directory):
        directories.append(directory)
        started.set()
        await release.wait()
        assert directory.exists()
        return ready()

    monkeypatch.setattr(codex_cli, "_inspect", inspect)
    tasks = [asyncio.create_task(codex_cli._checked_status(str(tmp_path / "fixture"))) for _ in range(12)]
    await started.wait()
    await asyncio.sleep(.02)
    tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    assert len(directories) == 1 and directories[0].exists()
    release.set()
    results = await asyncio.gather(*tasks[1:])
    assert all(result["authenticated"] for result in results)
    assert len(directories) == 1 and not directories[0].exists()
    results[0]["authenticated"] = False
    assert (await codex_cli._checked_status(str(tmp_path / "fixture")))["authenticated"]


@pytest.mark.asyncio
async def test_success_cache_expires_and_environment_or_binary_change_rechecks(tmp_path, monkeypatch):
    executable = tmp_path / "fixture"
    executable.write_text("version1")
    monkeypatch.setattr(codex_cli, "STATUS_CACHE_TTL_S", .2)
    calls = []

    async def inspect(*args):
        calls.append(1)
        return ready()

    monkeypatch.setattr(codex_cli, "_inspect", inspect)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "private-auth-home"))
    await codex_cli._checked_status(str(executable))
    await codex_cli._checked_status(str(executable))
    assert len(calls) == 1
    await asyncio.sleep(.25)
    await codex_cli._checked_status(str(executable))
    assert len(calls) == 2
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "another-private-auth-home"))
    await codex_cli._checked_status(str(executable))
    assert len(calls) == 3
    executable.write_text("different CLI version")
    await codex_cli._checked_status(str(executable))
    assert len(calls) == 4
    assert all(len(key) == 64 and set(key) <= set("0123456789abcdef") for key in codex_cli._STATUS_CACHE)


@pytest.mark.asyncio
async def test_failed_checks_are_not_cached_and_explicit_status_refreshes(tmp_path, monkeypatch):
    executable = str(tmp_path / "fixture")
    monkeypatch.setattr(codex_cli, "_executable", lambda: executable)
    calls = []

    async def inspect(*args):
        calls.append(1)
        return {**ready(), "authenticated": len(calls) != 1}

    monkeypatch.setattr(codex_cli, "_inspect", inspect)
    assert not (await codex_cli._checked_status(executable))["authenticated"]
    assert (await codex_cli._checked_status(executable))["authenticated"]
    assert len(calls) == 2
    assert (await codex_cli.status())["authenticated"]
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_failed_exec_invalidates_successful_auth_cache(cli, monkeypatch):  # noqa: F811
    configure, calls = cli
    checks = []

    async def inspect(*args):
        checks.append(1)
        return ready()

    monkeypatch.setattr(codex_cli, "_inspect", inspect)
    configure(scenario="failure")
    with pytest.raises(codex_cli.CodexCLIError, match="推理失败"):
        await codex_cli.complete("fixture", "fixture", "", 5)
    configure()
    result = await codex_cli.complete("fixture", "fixture", "", 5)
    assert result.request_count == 1 and len(calls()) == 2 and len(checks) == 2


@pytest.mark.asyncio
async def test_preflight_timeout_reports_only_safe_stage(tmp_path, monkeypatch):
    async def run(args, **kwargs):
        if args[1:] == ["--version"]:
            raise codex_cli.CodexCLIError("sensitive raw diagnostic", code="timeout")
        pytest.fail("Must stop after the failed status command")

    monkeypatch.setattr(codex_cli, "_run", run)
    result = await codex_cli._inspect("fixture", Path(tmp_path))
    assert not result["authenticated"] and result["check_stage"] == "version"
    assert "sensitive" not in str(result)


@pytest.mark.asyncio
async def test_preflight_failure_stage_is_preserved_without_admitting_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_cli, "_executable", lambda: str(tmp_path / "fixture"))

    async def inspect(*args):
        return {**ready(), "authenticated": False, "check_stage": "features", "message": "safe preflight failure"}

    async def forbidden(*args, **kwargs):
        pytest.fail("Failed preflight must not admit an inference process")

    monkeypatch.setattr(codex_cli, "_inspect", inspect)
    monkeypatch.setattr(codex_cli, "_run", forbidden)
    with pytest.raises(codex_cli.CodexCLIError) as caught:
        await codex_cli.complete("fixture", "fixture", "", 5)
    assert caught.value.code == "preflight_features" and caught.value.request_count == 0
