"""Large synchronous library reads must not stall the local HTTP event loop."""
import asyncio
import threading

import httpx
import pytest

from scenerecall.main import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize("route,store,method", [
    ("/api/bootstrap", "library", "records"),
    ("/api/assets", "library", "list_assets"),
    ("/api/projects", "organization", "projects"),
    ("/api/anime", "organization", "anime"),
    ("/api/folders", "organization", "folders"),
    ("/api/collections", "library", "collections"),
    ("/api/export/citations", "library", "collections"),
    ("/api/assets/asset_one/characters", "characters", "catalog"),
])
async def test_health_responds_while_file_projection_is_waiting(tmp_path, monkeypatch, route, store, method):
    app = create_app(tmp_path / "library", start_worker=False, token="responsiveness-test")
    target = app.state.library.organization if store == "organization" else getattr(app.state, store)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_projection(*args, **kwargs):
        started.set()
        # Bound the wait even when a regression accidentally runs it in the
        # event loop; the assertion still sees that health could not overlap it.
        release.wait(timeout=2)
        finished.set()
        return {"profiles": []} if store == "characters" else []

    monkeypatch.setattr(target, method, slow_projection)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        pending = asyncio.create_task(client.get(route, headers={"authorization": "Bearer responsiveness-test"}))
        try:
            assert await asyncio.to_thread(started.wait, 1), "the file read never started"
            health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
            assert health.status_code == 200 and health.json()["status"] == "ok"
            assert not finished.is_set(), "health was blocked behind the synchronous file projection"
        finally:
            release.set()
            response = await pending
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_responds_while_task_continuation_reads_character_references(tmp_path, monkeypatch):
    app = create_app(tmp_path / "library", start_worker=False, token="responsiveness-test")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_continuation(*args, **kwargs):
        started.set()
        release.wait(timeout=2)
        finished.set()
        return {"id": "job", "type": "analysis", "config": {}}

    monkeypatch.setattr(app.state.jobs, "control", slow_continuation)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        pending = asyncio.create_task(client.post("/api/jobs/job/continue",
            headers={"authorization": "Bearer responsiveness-test"}))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
            assert health.status_code == 200
            assert not finished.is_set(), "health was blocked while continuing a task"
        finally:
            release.set()
            response = await pending
        assert response.status_code == 200
