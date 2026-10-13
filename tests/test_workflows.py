"""Multi-episode workflows use generated media and an explicit model substitute."""
import asyncio
import subprocess

import pytest
from fastapi.testclient import TestClient

from scenerecall.jobs import JobQueue
from scenerecall.library import digest, read_json
from scenerecall.main import create_app
from scenerecall.models import AssetInput, ImportInput, ImportRetryInput, LocalFilesInput
from scenerecall.providers import AIResult
from scenerecall.workflow_api import list_local_files


class SyntheticProvider:
    def __init__(self):
        self.calls = []

    def get(self, profile_id):
        assert profile_id == "synthetic"
        return {"id": profile_id, "name": "test only", "model": "synthetic", "base_url": "http://test.invalid",
                "capabilities": ["vision", "subtitle"], "input_price_per_million": 1, "output_price_per_million": 1}

    def list(self):
        return [self.get("synthetic")]

    async def analyze(self, profile_id, frames, start, end, character_context=None):
        self.calls.append([frame["id"] for frame in frames])
        return AIResult({"summary": "明确的合成测试画面", "entities": [], "events": [],
                         "evidence_frame_ids": [frame["id"] for frame in frames]}, {}, 1, .01)


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    root = tmp_path_factory.mktemp("batch-synthetic")
    paths = []
    for i, color in enumerate(("red", "blue", "green")):
        path = root / f"合成 episode {i + 1}.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        f"color=c={color}:s=96x64:r=12:d=1.2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
        paths.append(path)
    return paths


@pytest.fixture
def client(tmp_path):
    provider = SyntheticProvider()
    app = create_app(tmp_path / "library", providers=provider, start_worker=False, token="test")
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "test"})
        response = client.put("/api/settings", json={"bindings": {"vision": "synthetic", "subtitle": "synthetic"}})
        assert response.status_code == 200, response.text
        yield client


def run(client, job_id):
    asyncio.run(client.app.state.jobs.execute(job_id))
    return client.get(f"/api/jobs/{job_id}").json()


def register(client, path, **kwargs):
    response = client.post("/api/assets", json={"video_path": str(path), "subtitle_mode": "embedded", **kwargs})
    assert response.status_code == 200, response.text
    return response.json()


def prepare(client, asset_id):
    result = client.post("/api/keyframe-plans", json={"asset_id": asset_id})
    assert result.status_code == 200, result.text
    plan = result.json()
    done = run(client, plan["job_id"])
    assert done["status"] == "completed", done
    return client.get(f"/api/keyframe-plans/{plan['id']}").json()


def test_local_listing_natural_order_pagination_and_selection_boundaries(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("Episode 10.MP4", "Episode 2.mp4", "字幕.ass", "secret.txt", ".hidden.mp4"):
        (tmp_path / name).write_bytes(b"listing does not probe media")
    (tmp_path / "alias.mp4").symlink_to(tmp_path / "Episode 2.mp4")
    result = list_local_files(LocalFilesInput(path=str(tmp_path), limit=2))
    assert result["entries"][0]["type"] == "directory"
    assert result["next_offset"] == 2
    rest = list_local_files(LocalFilesInput(path=str(tmp_path), offset=2))
    assert [entry["name"] for entry in rest["entries"]] == ["Episode 2.mp4", "Episode 10.MP4"]
    assert result["entries"][1]["path"] == str((tmp_path / "Episode 2.mp4").resolve())
    subtitles = list_local_files(LocalFilesInput(path=str(tmp_path), kind="subtitle"))
    assert [entry["name"] for entry in subtitles["entries"]] == ["sub", "字幕.ass"]


@pytest.mark.parametrize("path", ["relative/path", "https://example.invalid/video", "bad\x00path"])
def test_local_listing_rejects_nonlocal_paths(path):
    with pytest.raises(ValueError):
        list_local_files(LocalFilesInput(path=path))


def test_new_routes_use_existing_session_and_origin_protection(client, tmp_path):
    client.cookies.clear()
    assert client.post("/api/local-files/list", json={"path": str(tmp_path)}).status_code == 401
    assert client.post("/api/keyframe-plans", json={"asset_id": "missing"}).status_code == 401
    client.post("/api/session", json={"token": "test"})
    assert client.post("/api/local-files/list", json={"path": str(tmp_path)},
                       headers={"origin": "https://evil.invalid"}).status_code == 403


def test_import_partial_retry_duplicate_and_no_models(client, videos, tmp_path):
    project = client.post("/api/projects", json={"name": "合成动画", "kind": "animation"}).json()
    payload = {"request_id": "batch-import", "project_id": project["id"], "items": [
        {"item_id": "one", "video_path": str(videos[0]), "episode": 1},
        {"item_id": "two", "video_path": str(tmp_path / "missing.mp4"), "episode": 2},
        {"item_id": "dup", "video_path": str(videos[0]), "episode": 3}]}
    first = client.post("/api/imports", json=payload)
    assert first.status_code == 200, first.text
    job = first.json()
    assert client.post("/api/imports", json=payload).json()["id"] == job["id"]
    changed = {**payload, "project_id": None}
    assert client.post("/api/imports", json=changed).status_code == 409
    done = run(client, job["id"])
    assert done["status"] == "partial", done
    assert [item["status"] for item in done["items"]] == ["completed", "failed", "existing"]
    assert str(tmp_path) not in str(done) and str(videos[0].parent) not in str(done)
    assert len(client.get("/api/assets").json()) == 1
    config = client.get(f"/api/imports/{job['id']}").json()
    assert config["items"][1]["video_path"].endswith("missing.mp4")
    retried = client.post(f"/api/imports/{job['id']}/retry", json={"items": [
        {"item_id": "two", "video_path": str(videos[1]), "episode": 2}]})
    assert retried.status_code == 200, retried.text
    assert run(client, job["id"])["status"] == "completed"
    assets = client.get("/api/assets").json()
    assert len(assets) == 2 and all(asset["project_id"] == project["id"] for asset in assets)
    assert client.app.state.providers.calls == []


def test_import_recovers_commit_before_checkpoint(client, videos, monkeypatch):
    queue = client.app.state.jobs
    request = ImportInput(request_id="crash", items=[{"item_id": "one", "video_path": str(videos[0])}])
    job = queue.add_import(request)
    # Simulate process death after the atomically published asset, before item progress.
    asset = queue.library.register(AssetInput(video_path=str(videos[0])), import_item_key=digest([job["id"], "one"]))
    queue.update(job["id"], status="running")
    restored = JobQueue(queue.library, queue.providers, queue.search)
    def no_second_import(*args, **kwargs):
        raise AssertionError("must recognize the committed import item")
    monkeypatch.setattr(queue.library, "register", no_second_import)
    asyncio.run(restored.execute(job["id"]))
    final = restored.get(job["id"])
    assert final["status"] == "completed"
    assert final["items"][0]["asset_id"] == asset["id"]
    assert final["items"][0]["status"] == "completed"
    with pytest.raises(ValueError, match="只能"):
        restored.retry_import(job["id"], ImportRetryInput(items=request.items))


def test_import_retry_cannot_relabel_a_committed_item(client, videos, monkeypatch):
    queue = client.app.state.jobs
    request = ImportInput(request_id="index-failure", items=[{"item_id": "one", "video_path": str(videos[0])}])
    job = queue.add_import(request)
    original_upsert = queue.search.upsert

    def fail_index(*args, **kwargs):
        raise OSError("synthetic indexing failure after asset commit")

    monkeypatch.setattr(queue.search, "upsert", fail_index)
    assert run(client, job["id"])["status"] == "partial"
    imported = queue.library.list_assets()
    assert len(imported) == 1
    with pytest.raises(ValueError, match="已经导入"):
        queue.retry_import(job["id"], ImportRetryInput(items=[{"item_id": "one", "video_path": str(videos[1])}]))
    assert queue.get(job["id"])["config"]["items"][0]["video_path"] == str(videos[0])
    monkeypatch.setattr(queue.search, "upsert", original_upsert)
    queue.retry_import(job["id"], ImportRetryInput(items=request.items))
    assert run(client, job["id"])["status"] == "completed"
    assert [asset["id"] for asset in queue.library.list_assets()] == [imported[0]["id"]]


def test_scan_estimate_analysis_and_batch_idempotency(client, videos):
    first, second = [register(client, path, episode=i + 1) for i, path in enumerate(videos[:2])]
    assert client.post("/api/jobs/estimate", json={"asset_id": first["id"]}).status_code == 400
    plans = [prepare(client, asset["id"]) for asset in (first, second)]
    assert client.app.state.providers.calls == []
    assert all(plan["frame_count"] == 1 and plan["batch_count"] == 1 for plan in plans)
    for asset, plan in zip((first, second), plans):
        assert client.get(plan["frames"][0]["thumbnail_url"]).status_code == 200
        assert client.post("/api/keyframe-plans", json={"asset_id": asset["id"]}).json()["id"] == plan["id"]
    items = [{"asset_id": asset["id"], "keyframe_plan_id": plan["id"], "max_requests": 10}
             for asset, plan in zip((first, second), plans)]
    estimate = client.post("/api/jobs/batch/estimate", json={"items": items}).json()
    assert estimate["estimated_frames"] == 2 and estimate["estimated_requests"] == 2
    batch = client.post("/api/jobs/batch", json={"request_id": "analyze", "items": items})
    assert batch.status_code == 200, batch.text
    batch = batch.json()
    again = client.post("/api/jobs/batch", json={"request_id": "analyze", "items": items}).json()
    assert [job["id"] for job in again["jobs"]] == [job["id"] for job in batch["jobs"]]
    assert all(job["sampling_strategy"] == "content-keyframes-v1" for job in batch["jobs"])
    paused = client.post(f"/api/jobs/batches/{batch['batch_id']}/pause").json()
    assert all(job["status"] == "paused" for job in paused["jobs"])
    client.post(f"/api/jobs/batches/{batch['batch_id']}/resume")
    for job in batch["jobs"]:
        assert run(client, job["id"])["status"] == "completed"
    assert len(client.app.state.providers.calls) == 2
    observation = client.app.state.library.observations(first["id"])[0]
    assert observation["provenance"]["sampling_strategy"] == "content-keyframes-v1"
    assert observation["provenance"]["selected_frames"][0]["pts"] == plans[0]["frames"][0]["pts"]
    # A fresh job with the same plan and character revision reuses successful paid output.
    cached = client.post("/api/jobs", json=items[0]).json()
    assert run(client, cached["id"])["status"] == "completed"
    assert len(client.app.state.providers.calls) == 2


def test_wrong_plan_and_failed_batch_validation_enqueue_nothing(client, videos):
    first, second = [register(client, path) for path in videos[:2]]
    plan = prepare(client, first["id"])
    before = len(client.app.state.jobs.list())
    bad = client.post("/api/jobs/batch", json={"request_id": "wrong", "items": [
        {"asset_id": first["id"], "keyframe_plan_id": plan["id"]},
        {"asset_id": second["id"], "keyframe_plan_id": plan["id"]}]})
    assert bad.status_code == 400
    assert len(client.app.state.jobs.list()) == before
    wrong_range = client.post("/api/jobs/estimate", json={"asset_id": first["id"], "start_ms": 100,
                                                         "keyframe_plan_id": plan["id"]})
    assert wrong_range.status_code == 400
    assert client.app.state.providers.calls == []


def test_new_api_without_plan_still_uses_content_not_legacy_intervals(client, videos):
    asset = register(client, videos[0])
    result = client.post("/api/jobs", json={"asset_id": asset["id"], "window_ms": 1000, "frames_per_window": 12})
    assert result.status_code == 200, result.text
    job = result.json()
    assert job["sampling_strategy"] == "content-keyframes-v1"
    assert any("已弃用" in warning for warning in job["warnings"])
    assert run(client, job["id"])["status"] == "completed"
    assert len(client.app.state.providers.calls) == 1
    assert len(client.app.state.providers.calls[0]) == 1
    saved = read_json(next(client.app.state.library.asset_dir(asset["id"]).glob("keyframe-plans/*.json")))
    assert saved["algorithm"] == "content-keyframes-v1"
