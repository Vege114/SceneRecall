import asyncio

from scenerecall.retrieval_store import RetrievalStore, STRATEGY
from test_retrieval_jobs import Encoder
from test_scene_api import scene_app  # noqa: F401


def test_captionless_visual_hit_survives_canonical_api_and_seeks_real_frame(scene_app):  # noqa: F811
    app, client, _provider, _scene = scene_app
    library = app.state.library
    asset_id = "scene_asset"
    record = {"id": "retrieval_window", "asset_id": asset_id, "source_fingerprint": "synthetic-scene",
              "strategy": STRATEGY, "evidence_type": "retrieval_window", "kind": "visual",
              "start_ms": 0, "end_ms": 6000, "summary": "", "text": "", "entities": [], "events": [],
              "frame_ids": ["frame_original"], "evidence_frame_ids": ["frame_original"],
              "ordered_frames": [{"id": "frame_original", "at_ms": 0, "pts": 0, "time_base": [1, 1000]}]}
    RetrievalStore(library).publish(asset_id, "retrieval_run", record)
    encoder = Encoder()
    app.state.search.visual_encoder = encoder
    # Keep lifespan cleanup on the production adapter, not on this small double.
    async def close():
        return None
    encoder.close = close
    encoder.settings = lambda: {"available": True}
    projection = library.records(asset_id)
    selected = [r for r in projection if r["id"] == record["id"]]
    frame = {**record["ordered_frames"][0], "path": str(library.frame_path(asset_id, "frame_original")),
             "asset_id": asset_id, "source_fingerprint": "synthetic-scene"}
    asyncio.run(app.state.search.index_visual(selected, [frame]))
    response = client.post("/api/search", json={"query": "没有台词的背景", "filters": {"kind": "visual"}})
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["id"] == record["id"] and result["summary"] == ""
    assert result["hit_frame_id"] == "frame_original" and result["seek_ms"] == 0
    assert result["thumbnail_url"].endswith("/frames/frame_original")
    assert app.state.jobs.list() == []
    detail = client.get(f"/api/assets/{asset_id}").json()
    assert detail["retrieval_windows"][0]["evidence_type"] == "retrieval_window"
    # The old scene projection does not suppress independent retrieval evidence.
    assert detail["scenes"] and detail["retrieval_windows"]


def test_api_revalidates_held_frame_merge_against_current_sources(scene_app):  # noqa: F811
    app, client, _provider, _scene = scene_app
    library = app.state.library
    store = RetrievalStore(library)
    for number, (start, stop) in enumerate(((0, 6000), (6000, 10000))):
        row = {"id": f"held_window_{number}", "asset_id": "scene_asset", "source_fingerprint": "synthetic-scene",
               "strategy": STRATEGY, "evidence_type": "retrieval_window", "kind": "visual",
               "start_ms": start, "end_ms": stop, "summary": "", "text": "", "entities": [], "events": [],
               "frame_ids": ["frame_original"], "evidence_frame_ids": ["frame_original"],
               "ordered_frames": [{"id": "frame_original", "at_ms": 0, "pts": 0, "time_base": [1, 1000]}],
               "held_frame_spans": [{"frame_id": "frame_original", "start_ms": start, "end_ms": stop}] if start else []}
        store.publish("scene_asset", "held_run", row)
    encoder = Encoder()
    encoder.settings = lambda: {"available": True}
    async def close():
        return None
    encoder.close = close
    app.state.search.visual_encoder = encoder
    records = [r for r in library.records("scene_asset") if r.get("evidence_type") == "retrieval_window"]
    frame = {**records[0]["ordered_frames"][0], "path": str(library.frame_path("scene_asset", "frame_original")),
             "asset_id": "scene_asset", "source_fingerprint": "synthetic-scene"}
    asyncio.run(app.state.search.index_visual(records, [frame]))
    result = client.post("/api/search", json={"query": "静止的背景", "filters": {"kind": "visual"}}).json()["results"]
    assert len(result) == 1 and set(result[0]["member_ids"]) == {"held_window_0", "held_window_1"}
    assert (result[0]["start_ms"], result[0]["end_ms"], result[0]["seek_ms"]) == (0, 10000, 0)
    assert len(result[0]["source_spans"]) == 2


def test_api_uses_canonical_event_onset_without_replacing_real_visual_hit(scene_app, monkeypatch):  # noqa: F811
    from PIL import Image

    app, client, _provider, _scene = scene_app
    library = app.state.library
    frames = [{"id": "frame_original", "at_ms": 0, "pts": 0, "time_base": [1, 1000]}]
    for timestamp in (1000, 2000):
        fid = f"event_frame_{timestamp}"
        path = library.asset_dir("scene_asset") / "frames" / f"{fid}.jpg"
        Image.new("RGB", (32, 32), "blue").save(path)
        library.register_frame("scene_asset", {"id": fid, "path": str(path), "at_ms": timestamp})
        frames.append({"id": fid, "at_ms": timestamp, "pts": timestamp, "time_base": [1, 1000]})
    row = {"id": "timed_window", "asset_id": "scene_asset", "source_fingerprint": "synthetic-scene",
           "strategy": STRATEGY, "evidence_type": "retrieval_window", "kind": "visual", "start_ms": 0, "end_ms": 6000,
           "summary": "", "text": "", "entities": [], "events": [],
           "frame_ids": [f["id"] for f in frames], "evidence_frame_ids": [f["id"] for f in frames], "ordered_frames": frames}
    store = RetrievalStore(library)
    store.publish("scene_asset", "timed_run", row)
    store.publish_observation("scene_asset", "timed_run", {"unit_id": row["id"], "summary": "红衣人物转身",
        "identity_policy": "identity-binding-v2", "provenance": {"evidence_kind": "current_visual"},
        "evidence_frame_ids": row["frame_ids"], "uncertainties": [], "spatial_observations": [],
        "entities": [{"id": "p1", "kind": "person", "description": "红衣人物", "evidence_frame_ids": row["frame_ids"]}],
        "events": [{"action": "转身", "actor_ids": ["p1"], "target_ids": [], "temporal": True, "polarity": "positive",
                    "uncertainty": "", "start_ms": 1000, "end_ms": 2000,
                    "evidence_frame_ids": ["event_frame_1000", "event_frame_2000"]}]})

    async def search(*args, **kwargs):
        return {"results": [{**row, "hit_frame_id": "frame_original", "seek_ms": 9999, "match_type": "partial"}],
                "degraded": [], "explanation": ""}

    # Real search results have an indexed owner; preserve that boundary in this
    # ranking double while testing authoritative event timing independently.
    app.state.search.sync_asset(library.records("scene_asset"), "scene_asset")
    monkeypatch.setattr(app.state.search, "search", search)
    result = client.post("/api/search", json={"query": "红衣人物转身"}).json()["results"][0]
    assert result["seek_ms"] == 1000 and result["seek_basis"] == "local_event"
    assert result["event_evidence"]["evidence_frame_ids"] == ["event_frame_1000", "event_frame_2000"]
    assert result["hit_frame_id"] == "frame_original" and result["match_type"] == "partial"
    refused = client.post("/api/search", json={"query": "红衣人物没有转身"}).json()["results"][0]
    assert refused["seek_ms"] == 0 and refused["seek_basis"] == "visual_frame"
    assert "event_evidence" not in refused
