"""Character shorthand keeps the same evidence ranking through the HTTP boundary."""
from copy import deepcopy

import pytest
from PIL import Image

from scenerecall.library import atomic_jsonl
from scenerecall.retrieval_store import RetrievalStore, STRATEGY
from test_scene_api import scene_app  # noqa: F401


SHORT_QUERY = "华恋与光站在东京塔上"
FULL_QUERY = "爱城华恋与神乐光站在东京塔上"


@pytest.fixture
def character_search_api(scene_app, monkeypatch):  # noqa: F811
    app, client, _provider, _scene = scene_app
    library, engine = app.state.library, app.state.search
    asset_id = "scene_asset"
    app.state.characters.create(asset_id, {"name": "爱城华恋"})
    app.state.characters.create(asset_id, {"name": "神乐光"})

    second_path = library.asset_dir(asset_id) / "frames" / "frame_second.jpg"
    Image.new("RGB", (32, 32), "blue").save(second_path)
    library.register_frame(asset_id, {"id": "frame_second", "path": str(second_path), "at_ms": 6000})
    store = RetrievalStore(library)
    for identity, text, frame_id, at_ms, end_ms in (
        ("short_caption", SHORT_QUERY, "frame_original", 0, 5000),
        ("full_caption", FULL_QUERY, "frame_second", 6000, 10000),
    ):
        store.publish(asset_id, "alias_equivalence_run", {
            "id": identity, "asset_id": asset_id, "source_fingerprint": "synthetic-scene",
            "strategy": STRATEGY, "evidence_type": "retrieval_window", "kind": "visual",
            "start_ms": at_ms, "end_ms": end_ms, "summary": text, "text": text,
            "entities": [], "events": [], "frame_ids": [frame_id], "evidence_frame_ids": [frame_id],
            "ordered_frames": [{"id": frame_id, "at_ms": at_ms, "pts": at_ms, "time_base": [1, 1000]}],
        })
    records = [row for row in library.records(asset_id) if row.get("evidence_type") == "retrieval_window"]
    engine.sync_asset(records, asset_id)
    windows = {row["id"]: row for row in records}
    visual_queries = []

    async def visual(query, filters, warnings):
        # Deterministic image evidence; lexical recall and reciprocal ranking
        # still run normally, including their different original-query hits.
        visual_queries.append(query)
        return [{**deepcopy(windows[identity]), "hit_frame_id": windows[identity]["frame_ids"][0],
                 "seek_ms": windows[identity]["start_ms"], "retrieval_branches": ["visual"]}
                for identity in ("full_caption", "short_caption")]

    monkeypatch.setattr(engine, "_visual", visual)
    return app, client, visual_queries


@pytest.mark.parametrize("kind", ["visual", None], ids=["visual", "all-evidence"])
@pytest.mark.parametrize("verification", [False, True], ids=["without-verification", "with-verification"])
def test_shorthand_and_full_names_have_identical_final_evidence(
    character_search_api, monkeypatch, kind, verification,
):
    app, client, visual_queries = character_search_api
    from scenerecall import retrieval_events, retrieval_verification

    event_queries, verification_queries = [], []
    original_event_seek = retrieval_events.event_seek

    def event_seek(query, record):
        event_queries.append(query)
        return original_event_seek(query, record)

    async def verify_candidates(library, providers, profile_id, query, response):
        verification_queries.append(query)
        return {**response, "verification": {"status": "completed", "candidate_count": len(response["results"])}}

    monkeypatch.setattr(retrieval_events, "event_seek", event_seek)
    monkeypatch.setattr(retrieval_verification, "verify_candidates", verify_candidates)
    filters = {"asset_id": "scene_asset", **({"kind": kind} if kind else {})}
    results = []
    for query in (SHORT_QUERY, FULL_QUERY):
        response = client.post("/api/search", json={
            "query": query, "filters": filters, "visual_verification": verification,
        })
        assert response.status_code == 200, response.text
        rows = response.json()["results"]
        assert {row["id"] for row in rows} == {"short_caption", "full_caption"}
        results.append([{field: row.get(field) for field in (
            "id", "score", "hit_frame_id", "seek_ms", "seek_basis", "match_type", "thumbnail_url",
        )} for row in rows])

    assert results[0] == results[1]
    assert visual_queries == [FULL_QUERY, FULL_QUERY]
    assert event_queries == [FULL_QUERY] * 4
    assert verification_queries == ([FULL_QUERY, FULL_QUERY] if verification else [])
    assert [(row["hit_frame_id"], row["seek_ms"]) for row in results[0]] == [
        ("frame_second", 6000), ("frame_original", 0),
    ]
    assert all(row["match_type"] == "partial" for row in results[0])
    assert app.state.jobs.list() == []


@pytest.fixture
def subtitle_only_api(scene_app, monkeypatch):  # noqa: F811
    app, client, _provider, _scene = scene_app
    for name, aliases in (("爱城华恋", []), ("神乐光", []), ("大场奈奈", ["香蕉"])):
        app.state.characters.create("scene_asset", {"name": name, "aliases": aliases})

    async def no_visual_candidates(query, filters, warnings):
        return []

    monkeypatch.setattr(app.state.search, "_visual", no_visual_candidates)
    return app, client


def publish_subtitles_only(app, lines):
    """Leave only authentic subtitle projections in the searchable index."""
    cues = [{"id": f"dialogue_{index}", "start_ms": index * 3500, "end_ms": index * 3500 + 1000,
             "text": text, "language": "zh", "source": "external", "review_status": "confirmed"}
            for index, text in enumerate(lines)]
    atomic_jsonl(app.state.library.asset_dir("scene_asset") / "subtitles" / "external.jsonl", cues)
    records = [row for row in app.state.library.records("scene_asset") if row["kind"] == "subtitle"]
    app.state.search.sync_asset(records, "scene_asset")
    assert {row["kind"] for row in app.state.search.filtered_records()} == {"subtitle"}
    return {row["id"]: row["text"] for row in cues}


@pytest.mark.parametrize("verification", [False, True], ids=["without-verification", "with-verification"])
def test_subtitle_only_index_resolves_names_and_preserves_dialogue(subtitle_only_api, monkeypatch, verification):
    app, client = subtitle_only_api
    from scenerecall import retrieval_events, retrieval_verification

    originals = publish_subtitles_only(app, [SHORT_QUERY, FULL_QUERY])
    event_queries, verification_queries = [], []

    def event_seek(query, record):
        event_queries.append(query)
        return None

    async def verify_candidates(library, providers, profile_id, query, response):
        verification_queries.append(query)
        return response

    monkeypatch.setattr(retrieval_events, "event_seek", event_seek)
    monkeypatch.setattr(retrieval_verification, "verify_candidates", verify_candidates)
    results = []
    for query in (SHORT_QUERY, FULL_QUERY):
        response = client.post("/api/search", json={
            "query": query, "filters": {"asset_id": "scene_asset", "kind": "subtitle"},
            "visual_verification": verification,
        })
        assert response.status_code == 200, response.text
        rows = response.json()["results"]
        assert {row["id"] for row in rows} == set(originals)
        assert all(row["text"] == row["subtitle_text"] == originals[row["id"]] for row in rows)
        assert all(row.get("hit_frame_id") is None and row.get("seek_ms") is None for row in rows)
        results.append(rows)

    assert results[0] == results[1]
    assert not event_queries
    assert verification_queries == ([FULL_QUERY, FULL_QUERY] if verification else [])
    assert app.state.jobs.list() == []


def test_subtitle_partial_terms_recall_the_original_dialogue(subtitle_only_api):
    app, client = subtitle_only_api
    dialogue = "华恋，光在东京塔上等你。"
    originals = publish_subtitles_only(app, [dialogue, "森林里有一只鸟。"])
    for query in ("东京塔 等你", "京塔"):
        response = client.post("/api/search", json={
            "query": query, "filters": {"asset_id": "scene_asset", "kind": "subtitle"},
        })
        assert response.status_code == 200, response.text
        rows = response.json()["results"]
        assert [row["id"] for row in rows] == ["dialogue_0"]
        assert rows[0]["text"] == rows[0]["subtitle_text"] == originals["dialogue_0"]


def test_registered_nickname_recalls_subtitles_without_shared_name_characters(subtitle_only_api):
    app, client = subtitle_only_api
    originals = publish_subtitles_only(app, ["香蕉，轮到你了。", "今天大场奈奈来了。", "森林里有一只鸟。"])
    results = []
    for query in ("香蕉", "大场奈奈"):
        response = client.post("/api/search", json={
            "query": query, "filters": {"asset_id": "scene_asset", "kind": "subtitle"},
        })
        assert response.status_code == 200, response.text
        rows = response.json()["results"]
        assert {row["id"] for row in rows} == {"dialogue_0", "dialogue_1"}
        assert all(row["text"] == row["subtitle_text"] == originals[row["id"]] for row in rows)
        results.append(rows)
    assert results[0] == results[1]
