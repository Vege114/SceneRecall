"""The preflight confirmation updates exact retrieval evidence and lexical filters."""
from copy import deepcopy

import pytest

from scenerecall.character_confirmation import CharacterConfirmation
from scenerecall.retrieval_identity import observation
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.search import SearchEngine
from test_identity_bindings import calibrated  # noqa: F401
from test_retrieval_identity import current_window
from test_visual_identity import visual  # noqa: F401


@pytest.mark.asyncio
async def test_confirmation_updates_exact_published_region_and_confirmed_character_filter(calibrated):  # noqa: F811
    library, _characters, gallery, pipeline, backend, identity, _other, _refs, raw = calibrated
    store = RetrievalStore(library)
    visual_result = pipeline.reviews("asset_one")
    windows = []
    for index, source_frame in enumerate(raw["ordered_frames"]):
        window = current_window(library, raw)
        frame_id = source_frame["id"]
        window.update(id=f"rw_confirm_{index}", start_ms=index * 400, end_ms=(index + 1) * 400,
                      ordered_frames=[deepcopy(source_frame)], evidence_frame_ids=[frame_id], local_frame_ids=[frame_id])
        window = store.publish("asset_one", "run_before_confirmation", window)
        store.publish_identity("asset_one", "run_local_confirmation", observation(
            window, visual_result, run_id="run_local_confirmation", signature="synthetic-confirmation"))
        windows.append(window)
    immutable = {path: path.read_bytes() for path in library.asset_dir("asset_one").glob("analyses/**/*.json")}
    projected = library.records("asset_one")
    assert len(projected) == 2
    assert all(row["entities"][0]["binding_status"] == "candidate" for row in projected)
    search = SearchEngine(library.root / "indexes", None)
    search.sync_asset(projected, "asset_one")
    assert (await search.search("Aoi", {"character_id": identity}))["results"] == []
    assert (await search.search("Aoi", {"character": "Aoi"}))["results"] == []
    assert len((await search.search("Aoi"))["results"]) == 2
    original_features = backend.feature_calls
    selected = projected[0]
    region_id = selected["entities"][0]["identity_region_id"]

    service = CharacterConfirmation(library, pipeline, search)
    service.decide("asset_one", identity, action="confirm", source_asset_id="asset_one", region_id=region_id,
                   single_character=True, use_as_reference=True)

    updated = library.records("asset_one")
    confirmed = next(row for row in updated if row["id"] == selected["id"])["entities"][0]
    unconfirmed = next(row for row in updated if row["id"] != selected["id"])["entities"][0]
    assert confirmed["character_id"] == identity and confirmed["binding_status"] == "human_confirmed"
    assert unconfirmed["character_id"] is None and unconfirmed["binding_status"] == "candidate"
    assert unconfirmed["candidate_character_id"] == identity
    for filters in ({"character_id": identity}, {"character": "Aoi"}):
        assert [row["id"] for row in (await search.search("Aoi", filters))["results"]] == [selected["id"]]
    assert backend.feature_calls == original_features
    assert store.windows("asset_one") == windows
    assert all(path.read_bytes() == content for path, content in immutable.items())
    enrolled = [row for row in gallery.trusted("asset_one") if row["source"]["kind"] == "frame_crop"]
    assert len(enrolled) == 1 and enrolled[0]["source"]["region_id"] == region_id
    # Retrying an already completed operation updates neither the evidence nor
    # the trusted exemplar count, and the same search projection remains usable.
    service.decide("asset_one", identity, action="confirm", source_asset_id="asset_one", region_id=region_id,
                   single_character=True, use_as_reference=True)
    assert len([row for row in gallery.trusted("asset_one") if row["source"]["kind"] == "frame_crop"]) == 1
    assert [row["id"] for row in (await search.search("Aoi", {"character_id": identity}))["results"]] == [selected["id"]]
