"""Character workflows use fabricated assets and pictures; no model calls or user media."""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scenerecall.library import atomic_json, read_json, source_snapshot
from scenerecall.main import create_app


class CharacterAPIProvider:
    def get(self, profile_id):
        assert profile_id == "synthetic"
        return {"id": profile_id, "name": "API test double", "model": "synthetic",
                "base_url": "http://test.invalid/v1", "capabilities": ["vision"],
                "input_price_per_million": 1, "output_price_per_million": 1}

    def list(self):
        return [self.get("synthetic")]


def seed_episode(library, asset_id, episode, series="合成剧集"):
    directory = library.asset_dir(asset_id)
    source = library.root.parent / f"{asset_id}.mp4"
    source.write_bytes(f"synthetic video fixture {asset_id}".encode())
    asset = {"schema_version": "1.0", "id": asset_id, "work_id": f"work_{asset_id}",
             "title": f"合成第 {episode} 集", "kind": "animation", "series": series,
             "season": 1, "episode": episode, "version": "test", "duration_ms": 6000,
             "width": 32, "height": 32, "fingerprint": "test_" + asset_id,
             "subtitle_mode": "embedded", "created_at": "2026-09-29T00:00:00+00:00", "status": "ready"}
    atomic_json(directory / "manifest.json", asset)
    atomic_json(directory / "active-analysis.json", {"schema_version": "1.0", "observations": {}, "subtitle_runs": {}})
    locations = read_json(library.root / "private" / "media-locations.json", {})
    locations[asset_id] = source_snapshot(source)
    atomic_json(library.root / "private" / "media-locations.json", locations)
    frame = directory / "frames" / "frame_test.jpg"
    frame.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "blue").save(frame)
    library.register_frame(asset_id, {"id": "frame_test", "path": str(frame), "at_ms": 0})
    return asset


def seed_appearance(app, asset_id, character_id=None, second_person=False):
    entities = [{"id": "person1", "kind": "person", "description": "戴红围巾的短发人物",
                 "name": None, "character_id": character_id,
                 "match_confidence": 0.95 if character_id else 0,
                 "match_reason": "红围巾和面部外观一致" if character_id else "首次出现",
                 "portrait_frame_id": "frame_test", "portrait_box": [0.25, 0, 0.5, 1],
                 "evidence_frame_ids": ["frame_test"]}]
    if second_person:
        entities.append({**entities[0], "id": "person2", "description": "戴圆眼镜的长发人物",
                         "character_id": None, "match_confidence": 0})
    record = {"id": f"obs_{asset_id}", "asset_id": asset_id, "run_id": "run_test",
              "window_id": "window_test", "shot_id": "shot_test", "start_ms": 0, "end_ms": 6000,
              "summary": "人物在街边交谈", "entities": entities, "evidence_frame_ids": ["frame_test"]}
    with app.state.library.lock:
        record = app.state.characters.apply_observation(asset_id, record)
        app.state.library.save_observation(asset_id, record)
    return record


@pytest.fixture
def character_app(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, providers=CharacterAPIProvider(), token="characters-test")
    seed_episode(app.state.library, "asset_first", 1)
    seed_episode(app.state.library, "asset_second", 2)
    seed_episode(app.state.library, "asset_other", 1, series="另一部合成剧")
    original = seed_appearance(app, "asset_first", second_person=True)
    character_id = original["entities"][0]["character_id"]
    seed_appearance(app, "asset_second", character_id)
    with TestClient(app) as client:
        assert client.get("/api/assets/asset_first/characters").status_code == 401
        client.post("/api/session", json={"token": "characters-test"})
        client.put("/api/settings", json={"bindings": {"vision": "synthetic"}})
        yield app, client, original


def test_edit_search_merge_delete_and_immutable_original(character_app):
    app, client, original = character_app
    base = "/api/assets/asset_first/characters"
    catalog = client.get(base).json()
    first, second = [entity["character_id"] for entity in original["entities"]]
    assert len(catalog["profiles"]) == 2 and len(catalog["assets"]) == 2
    assert client.get("/api/assets/asset_other/characters").json()["profiles"] == []
    edit = client.patch(f"{base}/{first}", json={"name": "阿岚", "aliases": ["小红巾"],
        "description": "红围巾，短黑发", "notes": "同一人物换装后仍沿用该档案", "appearance": ["短黑发"],
        "expected_revision": catalog["revision"]})
    assert edit.status_code == 200, edit.text
    assert edit.json()["appearance"] == ["短黑发"]
    detail = client.get("/api/assets/asset_second").json()
    assert detail["observations"][0]["entities"][0]["character_name"] == "阿岚"
    results = client.post("/api/search", json={"query": "小红巾"}).json()["results"]
    assert {row["asset_id"] for row in results} == {"asset_first", "asset_second"}
    stale = client.patch(f"{base}/{first}", json={"name": "过期覆盖", "expected_revision": catalog["revision"]})
    assert stale.status_code == 409
    assert client.patch(f"/api/assets/asset_other/characters/{first}", json={"name": "错误作用域"}).status_code in (400, 404)
    revision = client.get(base).json()["revision"]
    merge = client.post(f"{base}/{second}/merge", json={"target_id": first, "expected_revision": revision})
    assert merge.status_code == 200, merge.text
    merged = client.get(base).json()
    assert len(merged["profiles"]) == 1
    assert merged["profiles"][0]["appearance_count"] == 3
    assert merged["profiles"][0]["name"] == "阿岚"
    assert {row["character_id"] for row in client.get("/api/assets/asset_first").json()["observations"][0]["entities"]} == {first}
    deletion = client.delete(f"{base}/{first}?expected_revision={merged['revision']}")
    assert deletion.status_code == 200, deletion.text
    assert client.get(base).json()["profiles"] == []
    assert not client.get("/api/assets/asset_second").json()["observations"][0]["entities"][0].get("character_id")
    assert client.post("/api/search", json={"query": "小红巾"}).json()["results"] == []
    raw = app.state.library.observations("asset_first")[0]
    assert raw["entities"] == original["entities"]


def test_reanalyze_uses_latest_revision_and_orders_whole_series(character_app):
    app, client, original = character_app
    first = original["entities"][0]["character_id"]
    client.patch(f"/api/assets/asset_first/characters/{first}", json={"name": "新名字"})
    base = "/api/assets/asset_second/characters"
    revision = client.get(base).json()["revision"]
    response = client.post(f"{base}/reanalyze", json={"scope": "series", "max_requests": 25,
        "max_cost": 2, "expected_revision": revision})
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["revision"] == revision
    assert [job["asset_id"] for job in value["jobs"]] == ["asset_first", "asset_second"]
    for public_job in value["jobs"]:
        config = app.state.jobs.get(public_job["id"])["config"]
        assert config["force"] and config["stages"] == ["vision"]
        assert config["start_ms"] == 0 and config["end_ms"] == 6000
        assert config["max_requests"] == 25 and config["max_cost"] == 2
        assert config["character_snapshot"]["revision"] == revision
        assert "config" not in public_job


def test_reanalysis_validates_entire_batch_before_enqueuing(character_app):
    app, client, original = character_app
    base = "/api/assets/asset_first/characters"
    revision = client.get(base).json()["revision"]
    client.patch(f"{base}/{original['entities'][0]['character_id']}", json={"name": "已修改"})
    assert client.post(f"{base}/reanalyze", json={"expected_revision": revision}).status_code == 409
    assert app.state.jobs.list() == []
    Path(app.state.library.source_path("asset_second")).unlink()
    response = client.post(f"{base}/reanalyze", json={"scope": "series"})
    assert response.status_code == 400 and "第 2 集" in response.json()["detail"]
    assert app.state.jobs.list() == []
    assert client.post(f"{base}/reanalyze", json={"scope": "asset"}).status_code == 200
    assert len(app.state.jobs.list()) == 1


def test_portrait_uses_current_real_frame_crop_without_changing_evidence(character_app):
    app, client, original = character_app
    character_id = original["entities"][0]["character_id"]
    base = f"/api/assets/asset_first/characters/{character_id}"
    catalog = client.get("/api/assets/asset_first/characters").json()
    current = next(profile for profile in catalog["profiles"] if profile["id"] == character_id)
    assert current["representative"]["asset_id"] == "asset_second"
    response = client.get(f"{base}/portrait")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(response.content)) as portrait:
        assert portrait.size == (16, 32)
        assert portrait.getpixel((8, 16))[2] > 200
    with Image.open(app.state.library.frame_path("asset_second", "frame_test")) as evidence:
        assert evidence.size == (32, 32)
    assert client.get(f"/api/assets/asset_other/characters/{character_id}/portrait").status_code in (400, 404)
    assert client.delete(base).status_code == 200
    assert client.get(f"{base}/portrait").status_code in (400, 404)


@pytest.mark.parametrize("payload", [{}, {"name": None}, {"aliases": [""]}, {"aliases": ["x" * 201]},
                                      {"notes": "x" * 100001}, {"unexpected": "field"}])
def test_character_edit_rejects_invalid_payload(character_app, payload):
    _, client, original = character_app
    character_id = original["entities"][0]["character_id"]
    assert client.patch(f"/api/assets/asset_first/characters/{character_id}", json=payload).status_code == 422
