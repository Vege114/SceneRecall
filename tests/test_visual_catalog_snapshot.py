"""A visual projection reuses one complete catalog without retaining it later."""
from copy import deepcopy

from scenerecall.characters import CharacterStore
from scenerecall.scene_jobs import canonical
from scenerecall.visual_identity import VisualIdentityPipeline
from test_visual_frame_projection import region
from test_visual_identity import add_reference, add_role, visual  # noqa: F401


def legacy_projection(pipeline, result):
    """The preceding three-catalog algorithm, using public/default read paths."""
    asset_id = "asset_one"
    projected = deepcopy(result)
    projected["regions"] = pipeline._project(asset_id, projected.get("regions", []))
    profiles = {row["id"]: row for row in CharacterStore(pipeline.library).catalog(asset_id)["profiles"]}
    selected = list(dict.fromkeys(value for row in projected["regions"] for value in [
        row["match"].get("character_id"), row["match"].get("candidate_character_id"),
        *[candidate["character_id"] for candidate in row["match"].get("candidates", [])]]
        if row["match"].get("status") in {"candidate", "confirmed"} and value in profiles))
    projected["selected_profiles"] = [profiles[value] for value in selected]
    used = {identity for row in projected["regions"] if row["match"].get("status") in {"candidate", "confirmed"}
            for candidate in row["match"].get("candidates", []) for identity in candidate.get("reference_ids", [])}
    projected["references"] = [{"character_id": row["character_id"], "reference_id": row["id"],
                                "path": row["path"], "reference_only": True,
                                "reference_status": row["status"], "human_confirmed": row["status"] == "confirmed"}
                               for row in pipeline.gallery.matching(asset_id) if row["character_id"] in selected
                               and (row["status"] == "confirmed" or row["id"] in used)]
    return projected


def test_projection_catalog_snapshot_is_canonical_identical_and_rebuilt_after_human_changes(visual, monkeypatch):  # noqa: F811
    library, characters, gallery = visual
    first, second = add_role(characters, "Aoi"), add_role(characters, "Akane")
    references = [add_reference(gallery, first), add_reference(gallery, second, red=120)]
    result = {"status": "completed", "regions": [region("region_a", "frame_a", first, references[0]["id"]),
                                                region("region_b", "frame_b", second, references[1]["id"])]}
    original = deepcopy(result)
    pipeline = VisualIdentityPipeline(library)
    catalog = CharacterStore.catalog
    calls = []

    def counted(self, asset_id):
        calls.append(asset_id)
        return catalog(self, asset_id)

    monkeypatch.setattr(CharacterStore, "catalog", counted)
    old = legacy_projection(pipeline, result)
    assert calls == ["asset_one"] * 3
    calls.clear()
    current = pipeline.project_result("asset_one", result)
    assert calls == ["asset_one"]
    assert canonical(current) == canonical(old) and result == original
    # The snapshot belongs to one projection, never a process-wide cache.
    characters.update("asset_one", first, {"name": "Human corrected Aoi"})
    calls.clear()
    renamed = pipeline.project_result("asset_one", result)
    assert calls == ["asset_one"]
    assert next(row for row in renamed["selected_profiles"] if row["id"] == first)["name"] == "Human corrected Aoi"
    gallery.update("asset_one", first, references[0]["id"], status="revoked", reason="Human rejected reference")
    calls.clear()
    rejected = pipeline.project_result("asset_one", result)
    assert calls == ["asset_one"]
    assert rejected["regions"][0]["match"]["status"] == "unknown"
    assert [row["reference_id"] for row in rejected["references"]] == [references[1]["id"]]
