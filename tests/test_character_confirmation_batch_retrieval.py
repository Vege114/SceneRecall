"""Batch confirmations publish exact lexical identities without rewriting evidence."""
from copy import deepcopy

import pytest

from scenerecall.character_confirmation import CharacterConfirmation
from scenerecall.identity_bindings import POLICY_VERSION
from scenerecall.library import read_json
from scenerecall.retrieval_identity import observation
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.search import SearchEngine
from test_identity_bindings import calibrated  # noqa: F401
from test_retrieval_identity import current_window
from test_visual_identity import frame, visual  # noqa: F401


@pytest.mark.asyncio
async def test_batch_publishes_three_exact_regions_once_and_preserves_unselected_evidence(calibrated, monkeypatch):  # noqa: F811
    library, _characters, gallery, pipeline, backend, identity, _other, _refs, raw = calibrated
    extra_frames = [frame(library, "frame_c", red=87, at_ms=1000),
                    frame(library, "frame_d", red=88, at_ms=1400)]
    pipeline.run("asset_one", extra_frames, {**pipeline.settings("asset_one"), "identity_policy": POLICY_VERSION})
    source_frames = [*raw["ordered_frames"], *({**row, "shot_id": "shot_one"} for row in extra_frames)]
    store = RetrievalStore(library)
    visual_result = pipeline.reviews("asset_one")
    windows = []
    for index, source_frame in enumerate(source_frames):
        window = current_window(library, raw)
        frame_id = source_frame["id"]
        window.update(id=f"rw_batch_{index}", start_ms=index * 400, end_ms=(index + 1) * 400,
                      ordered_frames=[deepcopy(source_frame)], evidence_frame_ids=[frame_id], local_frame_ids=[frame_id])
        window = store.publish("asset_one", "run_before_batch", window)
        store.publish_identity("asset_one", "run_local_batch", observation(
            window, visual_result, run_id="run_local_batch", signature="synthetic-batch-confirmation"))
        windows.append(window)
    immutable_paths = [*library.asset_dir("asset_one").glob("analyses/**/*.json"),
                       *(pipeline._directory("asset_one") / "regions").glob("*.json")]
    immutable = {path: path.read_bytes() for path in immutable_paths}
    projected = sorted(library.records("asset_one"), key=lambda row: row["start_ms"])
    assert len(projected) == 4
    assert all(row["entities"][0]["binding_status"] == "candidate" for row in projected)
    search = SearchEngine(library.root / "indexes", None)
    search.sync_asset(projected, "asset_one")
    assert (await search.search("Aoi", {"character_id": identity}))["results"] == []
    assert len((await search.search("Aoi"))["results"]) == 4
    sync_calls = []
    original_sync = search.sync_asset

    def counted_sync(records, asset_id):
        sync_calls.append(asset_id)
        return original_sync(records, asset_id)

    monkeypatch.setattr(search, "sync_asset", counted_sync)
    service = CharacterConfirmation(library, pipeline, search)
    selected_ids = {row["id"] for row in projected[:3]}
    selected_regions = {row["entities"][0]["identity_region_id"] for row in projected[:3]}
    candidates = [{"source_asset_id": "asset_one", "region_id": row["entities"][0]["identity_region_id"]}
                  for row in projected[:3]]
    model_calls = backend.feature_calls, backend.detection_calls

    result = service.confirm_batch("asset_one", identity, candidates=candidates,
                                   single_character=True, use_as_reference=True)

    assert result["summary"] == {"requested": 3, "confirmed": 3, "failed": 0}
    assert all(row["status"] == "confirmed" and not row["recovery_required"] for row in result["results"])
    assert {row["region_id"] for row in result["results"]} == selected_regions
    assert sync_calls == ["asset_one"]
    for row in library.records("asset_one"):
        entity = row["entities"][0]
        if row["id"] in selected_ids:
            assert entity["character_id"] == identity and entity["binding_status"] == "human_confirmed"
        else:
            assert entity["character_id"] is None and entity["binding_status"] == "candidate"
            assert entity["candidate_character_id"] == identity
    for filters in ({"character_id": identity}, {"character": "Aoi"}):
        assert {row["id"] for row in (await search.search("Aoi", filters))["results"]} == selected_ids
    assert len((await search.search("Aoi"))["results"]) == 4
    assert (backend.feature_calls, backend.detection_calls) == model_calls
    assert store.windows("asset_one") == windows
    assert all(path.read_bytes() == content for path, content in immutable.items())
    enrolled = [row for row in gallery.trusted("asset_one") if row["source"]["kind"] == "frame_crop"]
    assert len(enrolled) == 3
    assert {row["source"]["region_id"] for row in enrolled} == selected_regions
    review_revision = read_json(pipeline._directory("asset_one") / "decisions.json")["revision"]

    # A lost response can safely retry the same batch: no duplicate references
    # or human revisions, and one current search publication for the source.
    repeated = service.confirm_batch("asset_one", identity, candidates=candidates,
                                     single_character=True, use_as_reference=True)
    assert repeated["summary"] == result["summary"]
    assert sync_calls == ["asset_one", "asset_one"]
    assert len([row for row in gallery.trusted("asset_one") if row["source"]["kind"] == "frame_crop"]) == 3
    assert read_json(pipeline._directory("asset_one") / "decisions.json")["revision"] == review_revision
    assert all(path.read_bytes() == content for path, content in immutable.items())
    assert {row["id"] for row in (await search.search("Aoi", {"character_id": identity}))["results"]} == selected_ids
