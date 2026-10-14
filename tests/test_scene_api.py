"""Scene HTTP contracts use synthetic records and never contact a model service."""
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scenerecall.library import atomic_json, atomic_jsonl, source_snapshot
from scenerecall.main import create_app
from scenerecall.scenes import SceneStore


class SceneAPIProvider:
    def __init__(self):
        self.provider_type = "openai_compatible"
        self.capabilities = ["vision", "preparation"]

    def get(self, profile_id):
        assert profile_id == "synthetic"
        return {"id": profile_id, "name": "Scene API test double", "model": "synthetic",
                "provider_type": self.provider_type, "base_url": "https://test.invalid/v1",
                "capabilities": self.capabilities,
                "input_price_per_million": 1, "output_price_per_million": 1}

    def list(self):
        return [self.get("synthetic")]


@pytest.fixture
def scene_app(tmp_path):
    provider = SceneAPIProvider()
    app = create_app(tmp_path / "library", start_worker=False, providers=provider, token="scenes-api-test")
    library = app.state.library
    source = tmp_path / "synthetic.mp4"
    source.write_bytes(b"Synthetic source; no media decoding or model calls.")
    asset_id = "scene_asset"
    directory = library.asset_dir(asset_id)
    atomic_json(directory / "manifest.json", {
        "schema_version": "1.0", "id": asset_id, "work_id": "scene_work", "title": "合成站台场景",
        "kind": "animation", "series": "合成剧集", "season": 1, "episode": 1, "version": "test",
        "duration_ms": 10000, "width": 32, "height": 32, "fingerprint": "synthetic-scene",
        "subtitle_mode": "external", "created_at": "2026-10-13T00:00:00Z", "status": "ready",
    })
    atomic_json(library.root / "private/media-locations.json", {asset_id: source_snapshot(source)})
    frame_path = directory / "frames/frame_original.jpg"
    frame_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "green").save(frame_path)
    library.register_frame(asset_id, {"id": "frame_original", "path": str(frame_path), "at_ms": 0})
    atomic_jsonl(directory / "subtitles/external.jsonl", [{
        "id": "cue_original", "start_ms": 1000, "end_ms": 3000, "text": "他已经离开了。",
        "language": "zh", "source": "external", "review_status": "confirmed",
    }])
    library.save_observation(asset_id, {
        "id": "old_observation", "asset_id": asset_id, "run_id": "old_run", "window_id": "window",
        "shot_id": "shot", "start_ms": 0, "end_ms": 10000, "summary": "旧版画面记录",
        "entities": [], "events": [], "evidence_frame_ids": ["frame_original"], "uncertainties": [],
    })
    store = SceneStore(library)
    scene = store.publish(asset_id, "initial_run", [{
        "start_ms": 0, "end_ms": 10000, "summary": "人物在站台谈论告别",
        "frame_ids": ["frame_original"], "evidence_frame_ids": ["frame_original"],
        "representative_frame_ids": ["frame_original"], "subtitle_ids": ["cue_original"],
        "shot_ids": ["shot"], "entities": [], "events": [], "uncertainties": [], "needs_deep": True,
        "visual_facts": [{"text": "人物站在站台", "frame_ids": ["frame_original"]}],
        "dialogue_facts": [{"text": "台词提到离开", "cue_ids": ["cue_original"]}],
        "inferences": [{"text": "可能在告别", "frame_ids": ["frame_original"], "cue_ids": []}],
    }], 0, 10000)[0]
    app.state.search.sync_asset(library.records(asset_id))
    with TestClient(app) as client:
        assert client.get(f"/api/assets/{asset_id}").status_code == 401
        client.post("/api/session", json={"token": "scenes-api-test"})
        assert client.put("/api/settings", json={"bindings": {"vision": "synthetic"}}).status_code == 200
        yield app, client, provider, scene


def test_scene_search_and_detail_only_project_current_records_without_model_work(scene_app):
    app, client, _, scene = scene_app
    detail = client.get("/api/assets/scene_asset").json()
    assert detail["observations"] == []
    assert len(detail["scenes"]) == 1
    current = detail["scenes"][0]
    assert current["chunk_id"] == scene["id"]
    assert current["understanding_level"] == "initial" and not current["context_stale"]
    assert current["visual_facts"][0]["frame_ids"] == ["frame_original"]
    assert current["dialogue_facts"][0]["cue_ids"] == ["cue_original"]
    response = client.post("/api/search", json={"query": "站台", "filters": {"kind": "scene"}})
    assert response.status_code == 200, response.text
    assert [row["record_id"] for row in response.json()["results"]] == [scene["id"]]
    assert app.state.jobs.list() == []


def test_scene_deepen_estimate_is_read_only_and_submit_is_durable_budgeted_and_deduplicated(scene_app):
    app, client, _, scene = scene_app
    base = f"/api/assets/scene_asset/chunks/{scene['id']}/deepen"
    budget = {"max_requests": 9, "max_cost": 2}
    estimate = client.post(base + "/estimate", json=budget)
    assert estimate.status_code == 200, estimate.text
    assert estimate.json()["estimated_requests"] == 1
    assert app.state.jobs.list() == []
    created = client.post(base, json=budget)
    assert created.status_code == 200, created.text
    job = created.json()
    assert job["status"] == "queued" and job["chunk_id"] == scene["id"]
    assert job["budget"] == budget
    assert "config" not in job and "pending_scene_receipt" not in job and "scene_receipts" not in job
    config = app.state.jobs.get(job["id"])["config"]
    assert config["analysis_strategy"] == "scene-chunks-v1"
    assert config["deepen_chunk_id"] == scene["id"]
    assert set(config["profile_snapshots"]) == {"vision"}
    assert config["start_ms"] == 0 and config["end_ms"] == 10000
    repeated = client.post(base, json=budget)
    assert repeated.status_code == 200 and repeated.json()["id"] == job["id"]
    assert len(app.state.jobs.list()) == 1


@pytest.mark.parametrize("budget", [{"max_requests": 0}, {"max_requests": 1.5}, {"max_cost": 0}, {"max_cost": -1}])
def test_invalid_deepen_budgets_do_not_create_tasks(scene_app, budget):
    app, client, _, scene = scene_app
    base = f"/api/assets/scene_asset/chunks/{scene['id']}/deepen"
    assert client.post(base, json=budget).status_code == 422
    assert app.state.jobs.list() == []


def test_unknown_chunks_and_codex_dollar_budgets_are_rejected_before_enqueuing(scene_app):
    app, client, provider, scene = scene_app
    assert client.post("/api/assets/scene_asset/chunks/missing/deepen", json={}).status_code == 404
    provider.provider_type = "codex_cli"
    base = f"/api/assets/scene_asset/chunks/{scene['id']}/deepen"
    assert client.post(base, json={"max_cost": 1}).status_code == 400
    assert app.state.jobs.list() == []
    response = client.post(base, json={"max_requests": 12})
    assert response.status_code == 200, response.text
    assert response.json()["budget"] == {"max_requests": 12, "max_cost": None}


def test_deep_revision_preserves_favorite_and_export_identity(scene_app):
    app, client, _, scene = scene_app
    response = client.post("/api/assets/scene_asset/annotations", json={
        "record_id": scene["id"], "favorite": True, "note": "核对原片后引用",
    })
    assert response.status_code == 200, response.text
    store = SceneStore(app.state.library)
    upgraded = store.upgrade("scene_asset", "deep_run", {**scene, "summary": "站台交谈；台词提到离开，画面未显示离开动作"})
    collection = client.get("/api/collections").json()
    assert len(collection) == 1
    assert collection[0]["record_id"] == upgraded["id"] == scene["id"]
    assert collection[0]["understanding_level"] == "deep"
    assert collection[0]["note"] == "核对原片后引用"
    exported = client.get("/api/export/citations?format=json").json()
    assert exported[0]["record_id"] == scene["id"] and exported[0]["kind"] == "scene"
    assert app.state.jobs.list() == []
