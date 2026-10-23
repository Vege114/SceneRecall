"""Visual identity reads current dossiers without scanning appearance history."""
from copy import deepcopy

import pytest

from scenerecall.characters import CharacterStore
from scenerecall.identity_jobs import IdentityJobMixin
from scenerecall.library import atomic_json
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.scene_jobs import canonical
from scenerecall.scenes import SceneStore
from scenerecall.visual_identity import VisualIdentityPipeline
from test_visual_frame_projection import region
from test_visual_identity import add_reference, add_role, visual  # noqa: F401


def legacy_projection(pipeline, result):
    """The preceding three-catalog algorithm, using public/default read paths."""
    asset_id = "asset_one"
    projected = deepcopy(result)
    projected["regions"] = pipeline._project(asset_id, projected.get("regions", []),
                                              _catalog=CharacterStore(pipeline.library).catalog(asset_id))
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
                               for row in pipeline.gallery.matching(asset_id,
                                   _catalog=CharacterStore(pipeline.library).catalog(asset_id)) if row["character_id"] in selected
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
    catalog, roster = CharacterStore.catalog, CharacterStore.roster
    calls, roster_calls = [], []

    def counted(self, asset_id):
        calls.append(asset_id)
        return catalog(self, asset_id)

    monkeypatch.setattr(CharacterStore, "catalog", counted)
    def counted_roster(self, asset_id):
        roster_calls.append(asset_id)
        return roster(self, asset_id)
    monkeypatch.setattr(CharacterStore, "roster", counted_roster)
    old = legacy_projection(pipeline, result)
    assert calls == ["asset_one"] * 3
    calls.clear()
    current = pipeline.project_result("asset_one", result)
    assert calls == [] and roster_calls == ["asset_one"]
    # Appearance history is presentation metadata, not identity evidence. All
    # dossier fields and region/reference decisions remain identical.
    old["selected_profiles"] = [{key: value for key, value in profile.items()
                                if key not in {"appearances", "appearance_count", "representative"}}
                               for profile in old["selected_profiles"]]
    assert canonical(current) == canonical(old) and result == original
    # The snapshot belongs to one projection, never a process-wide cache.
    characters.update("asset_one", first, {"name": "Human corrected Aoi"})
    calls.clear()
    roster_calls.clear()
    renamed = pipeline.project_result("asset_one", result)
    assert calls == [] and roster_calls == ["asset_one"]
    assert next(row for row in renamed["selected_profiles"] if row["id"] == first)["name"] == "Human corrected Aoi"
    gallery.update("asset_one", first, references[0]["id"], status="revoked", reason="Human rejected reference")
    calls.clear()
    roster_calls.clear()
    rejected = pipeline.project_result("asset_one", result)
    assert calls == [] and roster_calls == ["asset_one"]
    assert rejected["regions"][0]["match"]["status"] == "unknown"
    assert [row["reference_id"] for row in rejected["references"]] == [references[1]["id"]]


def test_scene_request_identity_refresh_never_scans_episode_appearance_history(visual, monkeypatch):  # noqa: F811
    library, characters, gallery = visual
    identity = add_role(characters)
    reference = add_reference(gallery, identity)
    saved = {"visual": {"status": "completed", "regions": [
        region("region_a", "frame_a", identity, reference["id"]),
        region("region_b", "frame_b", identity, reference["id"])]}}
    job = {"asset_id": "asset_one", "run_id": "run_refresh", "config": {"identity_settings": {}}}
    queue = IdentityJobMixin()
    queue.library, queue.get = library, lambda _: job
    atomic_json(queue._identity_path(job), saved)

    def forbidden(*args, **kwargs):
        pytest.fail("refreshing current-frame identity must not scan all episode evidence")

    monkeypatch.setattr(CharacterStore, "catalog", forbidden)
    monkeypatch.setattr(SceneStore, "list", forbidden)
    monkeypatch.setattr(RetrievalStore, "records", forbidden)
    monkeypatch.setattr(library, "observations", forbidden)
    # No time range means this fixture exercises the visual refresh only.
    args = [[{"id": "frame_a"}], [], {}, {}, {"profiles": []}]
    refreshed = queue.refresh_scene_request_identity("job_one", "preparation", args)
    assert [row["id"] for row in refreshed[4]["identity_regions"]] == ["region_a"]
    assert refreshed[4]["profiles"][0]["name"] == "Aoi"
    assert args[4] == {"profiles": []}
    # Fresh reads still observe current names and revoked references.
    catalog = characters.roster("asset_one")
    registry = characters._read(catalog["scope_id"])
    registry["profiles"][identity]["name"] = "Human corrected Aoi"
    registry["revision"] += 1
    atomic_json(characters._path(catalog["scope_id"]), registry)
    assert queue.refresh_scene_request_identity("job_one", "preparation", args)[4]["profiles"][0]["name"] == "Human corrected Aoi"
    gallery.update("asset_one", identity, reference["id"], status="revoked", reason="Human rejected reference")
    revoked = queue.refresh_scene_request_identity("job_one", "preparation", args)
    assert revoked[4]["identity_regions"][0]["match"]["status"] == "unknown"
    assert revoked[4]["profiles"] == []
