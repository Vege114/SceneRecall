"""Appearance counts reuse current retrieval bindings without skipping legacy decisions."""
from copy import deepcopy

from scenerecall.identity_bindings import BindingResolver
from scenerecall.retrieval_identity import observation
from scenerecall.retrieval_store import RetrievalStore
from test_identity_bindings import calibrated  # noqa: F401
from test_retrieval_identity import current_window
from test_visual_identity import visual  # noqa: F401


def test_catalog_resolves_each_retrieval_once_and_preserves_legacy_human_decisions(calibrated, monkeypatch):  # noqa: F811
    library, characters, _gallery, pipeline, _backend, identity, _other, _refs, raw = calibrated
    store = RetrievalStore(library)
    window = store.publish("asset_one", "run_window", current_window(library, raw))
    local = observation(window, pipeline.reviews("asset_one"), run_id="run_identity", signature="fixture")
    store.publish_identity("asset_one", "run_identity", local)
    region = local["entities"][0]["identity_region_id"]
    pipeline.review("asset_one", region, status="confirmed", character_id=identity, reason="Synthetic human decision")
    legacy = {"id": "legacy", "asset_id": "asset_one", "run_id": "run_legacy", "start_ms": 0, "end_ms": 1000,
              "evidence_frame_ids": list(local["entities"][0]["evidence_frame_ids"]),
              "entities": [{**deepcopy(local["entities"][0]), "character_id": None}]}
    original = deepcopy(legacy)
    monkeypatch.setattr(library, "observations", lambda _: [legacy])
    resolve = BindingResolver.resolve
    resolved = []

    def counted(self, row):
        resolved.append(row["id"])
        return resolve(self, row)

    monkeypatch.setattr(BindingResolver, "resolve", counted)
    profile = next(row for row in characters.catalog("asset_one")["profiles"] if row["id"] == identity)
    assert resolved == [window["id"]]
    assert {row["record_id"] for row in profile["appearances"]} == {window["id"], "legacy"}
    assert profile["appearance_count"] == 2
    assert legacy == original

    pipeline.review("asset_one", region, status="unknown", reason="Synthetic human revocation")
    resolved.clear()
    profile = next(row for row in characters.catalog("asset_one")["profiles"] if row["id"] == identity)
    assert resolved == [window["id"]]
    assert profile["appearance_count"] == 0
    assert legacy == original
