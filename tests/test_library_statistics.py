"""Startup statistics reuse evidence and empty collections avoid projecting it."""
import copy

from scenerecall.library import Library, atomic_json, atomic_jsonl
from test_storage_security import ASSET_ID, observation, seed_asset


def test_empty_collection_uses_effective_annotations_without_projecting_evidence(tmp_path, monkeypatch):
    library = Library(tmp_path / "library")
    seed_asset(library)
    annotations = library.asset_dir(ASSET_ID) / "annotations"
    for identity, value in (
        ("first", {"updated_at": "1", "favorite": True}),
        ("second", {"updated_at": "2", "favorite": False}),
        ("note", {"updated_at": "3", "note": "保留笔记"}),
    ):
        atomic_json(annotations / f"{identity}.json", {"record_id": "retired", **value})

    def unexpected_projection(*_args, **_kwargs):
        raise AssertionError("an empty collection must not project evidence")

    monkeypatch.setattr(library, "records", unexpected_projection)
    monkeypatch.setattr(library, "annotations", unexpected_projection)
    assert library.collections() == []


def test_collection_projects_only_assets_with_effective_favorites(tmp_path, monkeypatch):
    library = Library(tmp_path / "library")
    seed_asset(library)
    seed_asset(library, "asset_uncollected")
    library.save_observation(ASSET_ID, observation())
    library.annotate(ASSET_ID, {"record_id": "obs_test", "favorite": True})
    original = library.records
    calls = []

    def project(asset_id=None, *, compact_subtitles=True):
        calls.append((asset_id, compact_subtitles))
        return original(asset_id, compact_subtitles=compact_subtitles)

    monkeypatch.setattr(library, "records", project)
    assert [row["record_id"] for row in library.collections()] == ["obs_test"]
    assert calls == [(ASSET_ID, False)]


def test_statistics_reuse_one_raw_projection_for_compaction_and_historical_favorites(tmp_path, monkeypatch):
    library = Library(tmp_path / "library")
    seed_asset(library)
    seed_asset(library, "asset_empty")
    library.save_observation(ASSET_ID, observation())
    library.annotate(ASSET_ID, {"record_id": "obs_test", "favorite": True, "note": "旧版本证据"})
    library.save_observation(ASSET_ID, observation("obs_new", "run_new", summary="当前证据"))
    atomic_jsonl(library.asset_dir(ASSET_ID) / "subtitles" / "external.jsonl", [
        {"id": identity, "asset_id": ASSET_ID, "start_ms": 0, "end_ms": 1000,
         "text": "重复字幕", "language": "zh", "source": "external"}
        for identity in ("cue_a", "cue_b", "cue_favorite")
    ])
    library.annotate(ASSET_ID, {"record_id": "cue_favorite", "favorite": True})
    original = library.records
    calls = []

    def project(asset_id=None, *, compact_subtitles=True):
        calls.append((asset_id, compact_subtitles))
        return original(asset_id, compact_subtitles=compact_subtitles)

    monkeypatch.setattr(library, "records", project)
    assert library.statistics() == {"assets": 2, "records": 3, "favorites": 2}
    assert calls == [(None, False)]
    raw = original(compact_subtitles=False)
    before = copy.deepcopy(raw)
    calls.clear()
    assert library.statistics(records=raw) == {"assets": 2, "records": 3, "favorites": 2}
    collected = library.collections(records=raw)
    assert {row["id"] for row in collected} == {"obs_test", "cue_favorite"}
    assert next(row for row in collected if row["id"] == "obs_test")["is_historical"]
    assert calls == []
    assert raw == before


def test_statistics_and_collections_exclude_deleted_assets(tmp_path):
    library = Library(tmp_path / "library")
    seed_asset(library)
    library.save_observation(ASSET_ID, observation())
    library.annotate(ASSET_ID, {"record_id": "obs_test", "favorite": True})
    assert library.statistics() == {"assets": 1, "records": 1, "favorites": 1}
    library.deletions.delete(kind="assets", asset_ids=[ASSET_ID])
    assert library.statistics() == {"assets": 0, "records": 0, "favorites": 0}
    assert library.collections() == []
