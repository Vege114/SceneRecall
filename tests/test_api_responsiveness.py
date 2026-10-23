"""Large synchronous library reads must not stall the local HTTP event loop."""
import asyncio
import threading

import httpx
import pytest

from scenerecall.main import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_reviews", [False, True])
async def test_detail_json_serialization_does_not_block_health(tmp_path, monkeypatch, identity_reviews):
    import scenerecall.main as main
    import scenerecall.visual_identity_api as identity_api

    app = create_app(tmp_path / "library", start_worker=False, token="responsiveness-test")
    library = app.state.library
    monkeypatch.setattr(library, "get_asset", lambda *args, **kwargs: {"id": "asset_one"})
    for name in ("records", "observations", "subtitles", "annotations", "runs"):
        monkeypatch.setattr(library, name, lambda *args, **kwargs: [])
    monkeypatch.setattr(app.state.visual_identity, "reviews", lambda *args, **kwargs: {"regions": []})
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    module = identity_api if identity_reviews else main
    original = module.JSONResponse

    def slow_json(content, *args, **kwargs):
        if isinstance(content, dict) and ("asset" in content or "regions" in content):
            started.set()
            release.wait(2)
            finished.set()
        return original(content, *args, **kwargs)

    monkeypatch.setattr(module, "JSONResponse", slow_json)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        route = "/api/assets/asset_one" + ("/identity/reviews" if identity_reviews else "")
        pending = asyncio.create_task(client.get(route,
            headers={"authorization": "Bearer responsiveness-test"}))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            assert (await asyncio.wait_for(client.get("/api/health"), 1)).status_code == 200
            assert not finished.is_set(), "large episode serialization blocked the event loop"
        finally:
            release.set()
            response = await pending
        assert response.status_code == 200
        if identity_reviews:
            assert response.json()["regions"] == []
        else:
            assert response.json()["asset"]["id"] == "asset_one"


@pytest.mark.asyncio
@pytest.mark.parametrize("route,store,method", [
    ("/api/bootstrap", "providers", "list"),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,projection,payload", [
    ("post", "/api/projects", "save_project", {"name": "项目"}),
    ("post", "/api/folders", "save_folder", {"name": "文件夹"}),
    ("patch", "/api/projects/project_one", "save_project", {"name": "改名"}),
    ("patch", "/api/folders/folder_one", "save_folder", {"name": "改名"}),
    ("post", "/api/projects/assign", "assign", {"asset_ids": ["asset_one"], "project_id": None}),
])
async def test_organization_edit_waiting_for_background_library_lock_keeps_http_responsive(
        tmp_path, monkeypatch, method, path, projection, payload):
    app = create_app(tmp_path, start_worker=False, token="responsiveness-test")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        started.set()
        release.wait(2)
        finished.set()
        return {"id": "test"}

    monkeypatch.setattr(app.state.library.organization, projection, blocked)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        pending = asyncio.create_task(client.request(method, path, json=payload,
            headers={"authorization": "Bearer responsiveness-test"}))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            assert (await asyncio.wait_for(client.get("/api/health"), 1)).status_code == 200
            assert not finished.is_set()
        finally:
            release.set()
            response = await pending
        assert response.status_code == 200
