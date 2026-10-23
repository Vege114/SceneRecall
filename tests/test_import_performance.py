"""Import workers remain responsive and compact only disposable search evidence."""
import asyncio
import threading

import pytest
from fastapi.testclient import TestClient
from startup_helpers import wait_for_search

from scenerecall.anime_profiles import AnimeProfileStore
from scenerecall.jobs import JobQueue
from scenerecall.library import Library, atomic_json, atomic_jsonl, digest, read_json
from scenerecall.main import create_app
from scenerecall.models import ImportInput
from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine
from test_organization import seed


@pytest.mark.asyncio
async def test_import_phases_run_off_event_loop_and_pause_reuses_committed_asset(tmp_path, monkeypatch):
    library = Library(tmp_path / "library")
    providers = ProviderManager(library.root / "private")
    search = SearchEngine(library.root / "indexes", providers)
    queue = JobQueue(library, providers, search)
    request = ImportInput(request_id="responsive", items=[{
        "item_id": "one", "video_path": str(tmp_path / "synthetic.mp4"), "title": "合成动画",
        "anime_profile": {"mode": "scratch"}}])
    job = queue.add_import(request)
    main_thread = threading.get_ident()
    phases = []
    registered = []
    entered, release = threading.Event(), threading.Event()

    def register(payload, *, import_item_key=None):
        assert threading.get_ident() != main_thread
        phases.append(queue.get(job["id"])["stage"])
        asset = seed(library, "asset_committed", title=payload.title, series="", episode=1)
        path = library.asset_dir(asset["id"]) / "manifest.json"
        atomic_json(path, {**read_json(path), "import_item_key": import_item_key})
        registered.append(asset["id"])
        entered.set()
        assert release.wait(5), "test did not release the registration worker"
        return library.get_asset(asset["id"])

    monkeypatch.setattr(library, "register", register)
    original_apply = AnimeProfileStore.apply_asset

    def apply_profile(*args, **kwargs):
        assert threading.get_ident() != main_thread
        phases.append(queue.get(job["id"])["stage"])
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(AnimeProfileStore, "apply_asset", apply_profile)
    original_sync = search.sync_asset

    def sync(records, asset_id=None):
        assert threading.get_ident() != main_thread
        phases.append(queue.get(job["id"])["stage"])
        return original_sync(records, asset_id)

    monkeypatch.setattr(search, "sync_asset", sync)
    task = asyncio.create_task(queue.execute(job["id"]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await asyncio.wait_for(asyncio.sleep(0), timeout=1)
        active = queue.get(job["id"])
        assert active["stage"] == "import"
        assert active["items"][0]["status"] == "running"
        assert active["items"][0]["phase"] == "import"
        assert active["completed"] == 0
        assert queue.control(job["id"], "pause")["status"] == "pausing"
    finally:
        release.set()
        await task
    assert queue.get(job["id"])["status"] == "paused"
    assert queue.get(job["id"])["items"][0]["asset_id"] == "asset_committed"
    queue.control(job["id"], "resume")
    await queue.execute(job["id"])
    final = queue.get(job["id"])
    assert final["status"] == "completed" and final["completed"] == 1
    assert registered == ["asset_committed"]
    assert phases[:3] == ["import", "import_profile", "import_index"]
    assert set(final["items"][0]["timings_ms"]) == {"import", "import_profile", "import_index"}
    assert final["request_count"] == 0


@pytest.mark.asyncio
async def test_import_index_does_not_block_event_loop_or_report_premature_completion(tmp_path, monkeypatch):
    app = create_app(tmp_path / "library", start_worker=False, token="index-thread")
    library, queue, search = app.state.library, app.state.jobs, app.state.search
    job = queue.add_import(ImportInput(request_id="index-thread", items=[{
        "item_id": "one", "video_path": "already-committed.mp4"}]))
    asset = seed(library)
    path = library.asset_dir(asset["id"]) / "manifest.json"
    atomic_json(path, {**read_json(path), "import_item_key": digest([job["id"], "one"])})
    entered, release = threading.Event(), threading.Event()
    original = search.sync_asset
    errors = []

    def slow_index(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            errors.append("event loop failed to release index worker")
        return original(*args, **kwargs)

    monkeypatch.setattr(search, "sync_asset", slow_index)
    task = asyncio.create_task(queue.execute(job["id"]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await asyncio.wait_for(asyncio.sleep(0), timeout=1)
        active = queue.get(job["id"])
        assert active["stage"] == "import_index" and active["completed"] == 0
        assert active["items"][0]["status"] == "running"
        assert active["items"][0]["asset_id"] == asset["id"]
    finally:
        release.set()
        await task
    assert not errors
    assert queue.get(job["id"])["status"] == "completed"


def test_compacted_search_preserves_source_cues_favorites_and_annotation_updates(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, token="compact-test")
    library, search = app.state.library, app.state.search
    asset = seed(library, "asset_effect", series="", title="合成字幕")
    directory = library.asset_dir(asset["id"])
    manifest = directory / "manifest.json"
    atomic_json(manifest, {**read_json(manifest), "subtitle_mode": "external"})
    cues = [{"schema_version": "1.0", "id": "cue_" + identity, "asset_id": asset["id"],
             "start_ms": start, "end_ms": end, "text": "合成测试字幕", "language": "zh",
             "source": "external", "evidence_frame_ids": []}
            for identity, start, end in (("a", 0, 900), ("b", 0, 900), ("c", 0, 900), ("d", 1500, 2500))]
    path = directory / "subtitles" / "external.jsonl"
    atomic_jsonl(path, cues)
    original = path.read_bytes()
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "compact-test"})
        projected = library.records(asset["id"])
        assert len(projected) == 2
        merged = next(row for row in projected if row["start_ms"] == 0)
        assert set(merged["source_record_ids"]) == {"cue_a", "cue_b", "cue_c"}
        detail = client.get(f"/api/assets/{asset['id']}").json()
        assert len(detail["subtitles"]) == 4
        for identity in ("b", "c"):
            response = client.post(f"/api/assets/{asset['id']}/annotations",
                                   json={"record_id": "cue_" + identity, "favorite": True})
            assert response.status_code == 200, response.text
        assert {row["record_id"] for row in client.get("/api/collections").json()} == {"cue_b", "cue_c"}
        assert len(library.records(asset["id"])) == 4
        for identity in ("b", "c"):
            response = client.post(f"/api/assets/{asset['id']}/annotations",
                                   json={"record_id": "cue_" + identity, "favorite": False})
            assert response.status_code == 200, response.text
        assert len(library.records(asset["id"])) == 2
        with search._db() as db:
            indexed = {row[0] for row in db.execute("SELECT id FROM records WHERE is_context=0")}
        assert indexed == {"cue_a", "cue_d"}, "retired effect projections must leave no stale hits"
        wait_for_search(client)
        response = client.post("/api/search", json={"query": "合成测试字幕"})
        assert response.status_code == 200, response.text
        assert {row["record_id"] for row in response.json()["results"]} == {"cue_a", "cue_d"}
    assert path.read_bytes() == original
    assert len(library.subtitles(asset["id"])) == 4
