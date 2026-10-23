"""Opening the app is independent of owned, revision-fenced background work."""
import asyncio
import threading

import httpx
import pytest

from scenerecall.main import create_app


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(.01)
    await asyncio.wait_for(wait(), 5)


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost",
                             headers={"authorization": "Bearer startup-test"})


@pytest.mark.asyncio
async def test_slow_index_does_not_delay_lifespan_bootstrap_or_library(tmp_path, monkeypatch):
    app = create_app(tmp_path, start_worker=False, token="startup-test")
    started, release = threading.Event(), threading.Event()
    calls = []

    def source(*args, **kwargs):
        calls.append(kwargs)
        started.set()
        assert release.wait(5)
        return []

    monkeypatch.setattr(app.state.library, "records", source)
    monkeypatch.setattr(app.state.search.visual_encoder, "settings", lambda: {"available": True})
    try:
        async with app.router.lifespan_context(app), client_for(app) as client:
            assert await asyncio.to_thread(started.wait, 2)
            for route in ("/api/health", "/api/bootstrap", "/api/assets", "/api/jobs"):
                response = await asyncio.wait_for(client.get(route), 1)
                assert response.status_code == 200
            bootstrap = (await client.get("/api/bootstrap")).json()
            assert bootstrap["stats"] == {}
            assert bootstrap["startup"]["index"]["status"] == "running"
            response = await client.post("/api/search", json={"query": "杯子"})
            assert response.status_code == 503 and response.headers["retry-after"] == "2"
            release.set()
            await wait_until(lambda: app.state.preparation.states["encoder"]["status"] == "ready")
            ready = (await client.get("/api/bootstrap")).json()
            assert ready["startup"]["index"]["status"] == "ready"
            assert ready["stats"] == {"assets": 0, "records": 0, "favorites": 0}
            assert calls == [{"compact_subtitles": False}], "startup statistics must reuse the source projection"
            assert (await client.get("/api/index/status")).json()["lexical_ready"]
            assert (await client.post("/api/search", json={"query": "杯子"})).status_code == 200
    finally:
        release.set()


@pytest.mark.asyncio
async def test_index_failure_is_visible_and_retry_recovers_without_restart(tmp_path, monkeypatch):
    app = create_app(tmp_path, start_worker=False, token="startup-test")
    calls = 0

    def source(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("bad projection")
        return []

    monkeypatch.setattr(app.state.library, "records", source)
    async with app.router.lifespan_context(app), client_for(app) as client:
        await wait_until(lambda: app.state.preparation.task.done())
        assert (await client.get("/api/bootstrap")).json()["startup"]["index"]["status"] == "error"
        assert (await client.get("/api/assets")).status_code == 200
        assert (await client.post("/api/search", json={"query": "杯子"})).status_code == 503
        assert (await client.post("/api/startup/retry")).status_code == 200
        await wait_until(lambda: app.state.preparation.states["statistics"]["status"] == "ready")
        assert calls == 2
        assert (await client.post("/api/search", json={"query": "杯子"})).status_code == 200


@pytest.mark.asyncio
async def test_mutation_during_startup_retries_instead_of_publishing_old_source(tmp_path, monkeypatch):
    app = create_app(tmp_path, start_worker=False, token="startup-test")
    started, release = threading.Event(), threading.Event()
    calls = 0
    original = app.state.search._prepare_rebuild

    def pause(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(app.state.search, "_prepare_rebuild", pause)
    try:
        async with app.router.lifespan_context(app), client_for(app) as client:
            assert await asyncio.to_thread(started.wait, 2)
            response = await client.post("/api/folders", json={"name": "新文件夹"})
            assert response.status_code == 200
            release.set()
            await wait_until(lambda: app.state.preparation.states["index"]["status"] == "ready")
            assert calls >= 2
            assert len((await client.get("/api/folders")).json()) == 1
    finally:
        release.set()


@pytest.mark.asyncio
async def test_statistics_and_encoder_are_serial_cached_and_do_not_gate_search(tmp_path, monkeypatch):
    app = create_app(tmp_path, start_worker=False, token="startup-test")
    started, release = threading.Event(), threading.Event()
    stats_calls, encoder_calls = [], []

    def stats(*args, **kwargs):
        stats_calls.append(1)
        started.set()
        assert release.wait(5)
        return {"assets": 0, "records": 0, "favorites": 0}

    monkeypatch.setattr(app.state.library, "statistics", stats)
    monkeypatch.setattr(app.state.search.visual_encoder, "settings",
                        lambda: encoder_calls.append(1) or {"available": True})
    try:
        async with app.router.lifespan_context(app), client_for(app) as client:
            assert await asyncio.to_thread(started.wait, 2)
            assert not encoder_calls
            assert (await client.post("/api/search", json={"query": "杯子"})).status_code == 200
            for _ in range(3):
                result = (await client.get("/api/bootstrap")).json()
                assert result["startup"]["statistics"]["status"] == "running"
            assert stats_calls == [1]
            release.set()
            await wait_until(lambda: app.state.preparation.task.done())
            for _ in range(3):
                await client.get("/api/bootstrap")
            assert stats_calls == encoder_calls == [1]
    finally:
        release.set()


@pytest.mark.asyncio
async def test_shutdown_drains_preparation_before_closing_encoder(tmp_path, monkeypatch):
    app = create_app(tmp_path, start_worker=False, token="startup-test")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    closed = []

    def source(*args, **kwargs):
        started.set()
        assert release.wait(5)
        finished.set()
        return []

    async def close():
        assert finished.is_set()
        closed.append(True)

    monkeypatch.setattr(app.state.library, "records", source)
    monkeypatch.setattr(app.state.search.visual_encoder, "close", close)
    context = app.router.lifespan_context(app)
    await context.__aenter__()
    try:
        assert await asyncio.to_thread(started.wait, 2)
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        await asyncio.sleep(.05)
        assert not shutdown.done() and not closed
        release.set()
        await asyncio.wait_for(shutdown, 2)
        assert closed and app.state.preparation.task.done()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_source_disappearing_during_mutation_is_retried(tmp_path, monkeypatch):
    app = create_app(tmp_path, start_worker=False, token="startup-test")
    started, release = threading.Event(), threading.Event()
    calls = []

    def source(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
            raise FileNotFoundError("source was deleted during the projection")
        return []

    monkeypatch.setattr(app.state.library, "records", source)
    try:
        async with app.router.lifespan_context(app), client_for(app) as client:
            assert await asyncio.to_thread(started.wait, 2)
            assert (await client.post("/api/folders", json={"name": "资料已更新"})).status_code == 200
            release.set()
            await wait_until(lambda: app.state.preparation.states["index"]["status"] == "ready")
            assert len(calls) == 2
    finally:
        release.set()
