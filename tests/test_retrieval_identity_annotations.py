"""Human annotation integration over local identity evidence, using synthetic input."""
import pytest

from scenerecall.library import read_json
from scenerecall.retrieval_identity import observation
from scenerecall.retrieval_store import RetrievalStore
from test_identity_bindings import calibrated  # noqa: F401
from test_retrieval_identity import current_window
from test_visual_identity import visual  # noqa: F401

def test_region_annotation_confirms_stable_identity_and_never_overrides_rename_or_later_review(calibrated):  # noqa: F811
    library, characters, _gallery, pipeline, backend, identity, other, _refs, raw_observation = calibrated
    store = RetrievalStore(library)
    window = store.publish("asset_one", "run_before", current_window(library, raw_observation))
    local = observation(window, pipeline.reviews("asset_one"), run_id="run_local", signature="fixture")
    store.publish_identity("asset_one", "run_local", local)
    person = store.records("asset_one")[0]["entities"][0]
    assert person["binding_status"] == "candidate"
    characters.update("asset_one", identity, {"aliases": ["小葵"]})
    original_window = store.windows("asset_one")
    features = backend.feature_calls
    annotation = library.annotate("asset_one", {"record_id": window["id"], "entity_id": person["id"],
        "character_name": "小葵", "aliases": ["仅保留在这次笔记的旧别名"], "note": "核对这位人物"})
    assert annotation["identity_review"] == {"region_id": person["identity_region_id"], "character_id": identity,
                                           "version": 1, "status": "confirmed", "source": "human"}
    bound = next(row for row in library.records("asset_one") if row["id"] == window["id"])
    assert bound["entities"][0]["character_id"] == identity
    assert bound["entities"][0]["binding_status"] == "human_confirmed"
    assert bound["entities"][0]["character_name"] == "Aoi"
    assert bound["review_status"] == "user_confirmed"
    characters.update("asset_one", identity, {"name": "新角色名", "aliases": ["Aoi"]})
    renamed = next(row for row in library.records("asset_one") if row["id"] == window["id"])
    assert renamed["entities"][0]["character_name"] == "新角色名"
    assert "仅保留在这次笔记的旧别名" not in renamed["character"]
    pipeline.review("asset_one", person["identity_region_id"], status="unknown", reason="撤销此前确认")
    revoked = next(row for row in library.records("asset_one") if row["id"] == window["id"])
    assert revoked["entities"][0]["character_id"] is None
    assert revoked["entities"][0]["character_name"] is None
    assert revoked["entities"][0]["binding_status"] == "unknown"
    assert revoked["review_status"] == "unreviewed"
    pipeline.review("asset_one", person["identity_region_id"], status="confirmed", character_id=other,
                    reason="重新核对为另一个角色")
    rebound = next(row for row in library.records("asset_one") if row["id"] == window["id"])
    assert rebound["entities"][0]["character_id"] == other
    assert rebound["entities"][0]["character_name"] == "Other"
    assert backend.feature_calls == features and store.windows("asset_one") == original_window
    saved = read_json(library.asset_dir("asset_one") / "annotations" / f"{annotation['id']}.json")
    assert saved["character_name"] == "小葵" and saved["identity_review"]["version"] == 1


def test_region_annotation_rejects_unknown_and_ambiguous_names_without_writing_decisions(calibrated):  # noqa: F811
    library, characters, _gallery, pipeline, _backend, _identity, _other, _refs, raw_observation = calibrated
    store = RetrievalStore(library)
    window = store.publish("asset_one", "run_before", current_window(library, raw_observation))
    store.publish_identity("asset_one", "run_local", observation(window, pipeline.reviews("asset_one"),
                                                               run_id="run_local", signature="fixture"))
    person = store.records("asset_one")[0]["entities"][0]
    characters.create("asset_one", {"name": "Brenda", "aliases": ["Aoi"]})
    for requested in ("Aoi", "不属于当前作品的人物"):
        with pytest.raises(ValueError, match="无法唯一对应当前作品"):
            library.annotate("asset_one", {"record_id": window["id"], "entity_id": person["id"],
                                           "character_name": requested})
    assert read_json(pipeline._directory("asset_one") / "decisions.json") is None
    assert library.annotations("asset_one") == []
