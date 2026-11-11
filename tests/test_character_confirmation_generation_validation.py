"""Bounded cross-asset sampling survives backup without trusting saved plans."""
from copy import deepcopy

import pytest

from scenerecall.character_confirmation_validation import validate_sampling_v2
from scenerecall.library import Library, atomic_json, digest, now, read_json
from scenerecall.visual_identity import validate_visual_artifacts
from test_character_confirmation_validation import confirmation_archive  # noqa: F401
from test_visual_identity import visual  # noqa: F401


def sign_plan(sample):
    plan = [{key: row[key] for key in ("asset_id", "source_fingerprint", "at_ms", "frame_id")}
            for row in sample["plan"]]
    sample["id"] = digest([sample["base_id"], sample["round"], plan])


@pytest.fixture
def generation_archive(confirmation_archive):  # noqa: F811
    library, characters, gallery, directory, key, state, _legacy = confirmation_archive
    second = {**library.get_asset("asset_one"), "id": "asset_two", "episode": 2, "fingerprint": "second-video"}
    atomic_json(library.asset_dir("asset_two") / "manifest.json", second)
    base_id = digest(["generation-backup-fixture"])
    plan = [{"asset_id": asset, "source_fingerprint": fingerprint, "at_ms": at,
             "frame_id": "frame_" + digest([base_id, 0, asset, at]), "status": status}
            for asset, fingerprint, at, status in (("asset_one", "synthetic-video", 199, "completed"),
                                                   ("asset_two", "second-video", 6000, "failed"))]
    library.register_sample_alias("asset_one", plan[0]["frame_id"], "frame_one")
    plan[0]["actual_at_ms"] = 200
    plan[1]["error"] = "Synthetic unavailable decoder"
    sample = {"schema_version": "1.0", "sampling_version": "2.0", "base_id": base_id, "round": 0,
              "asset_id": "asset_one", "scope_id": state["scope_id"], "request_scope": "work",
              "selected_character_ids": [row["id"] for row in characters.roster("asset_one")["profiles"]],
              "budget": {"max_frames": 120, "max_assets": 24, "frames_per_asset": 12},
              "status": "partial", "plan": plan, "total": 2, "completed": 1, "failed_frames": 1,
              "attempted": 2, "requested_times": [199, 6000], "frame_ids": [plan[0]["frame_id"]],
              "completed_frames": [{"asset_id": "asset_one", "frame_id": plan[0]["frame_id"],
                                    "at_ms": 200, "source_fingerprint": "synthetic-video"}],
              "assets": [{"asset_id": "asset_one", "source_fingerprint": "synthetic-video", "total": 1,
                          "completed": 1, "status": "completed"},
                         {"asset_id": "asset_two", "source_fingerprint": "second-video", "total": 1,
                          "completed": 0, "status": "failed", "error": "Synthetic unavailable decoder"}],
              "updated_at": now(), "reused_frames": 1, "detected_regions": 1, "accepted_regions": 1}
    sign_plan(sample)
    atomic_json(directory / "sampling.json", sample)
    return library, characters, gallery, directory, key, state, sample


def test_work_sampling_backup_restores_exact_progress_and_human_decisions(generation_archive, tmp_path):
    library, _characters, _gallery, directory, _key, state, sample = generation_archive
    validate_visual_artifacts(library)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    target = restored.root / directory.relative_to(library.root)
    assert read_json(target / "state.json") == state
    assert read_json(target / "sampling.json") == sample
    validate_visual_artifacts(restored)


@pytest.mark.parametrize("path,value", [
    (("status",), []), (("status",), "silently_accepted"), (("request_scope",), []),
    (("budget", "max_frames"), True), (("budget", "max_frames"), 121),
    (("budget", "max_assets"), 1), (("budget", "frames_per_asset"), 0),
    (("budget", "max_assets"), 25), (("budget", "frames_per_asset"), 25),
    (("total",), 3), (("attempted",), 1), (("attempted",), True), (("failed_frames",), 0),
    (("round",), -1), (("id",), "tampered_generation"), (("schema_version",), "3.0"),
    (("sampling_version",), "3.0"),
    (("scope_id",), "other_scope"), (("asset_id",), "../outside"),
    (("selected_character_ids",), []), (("selected_character_ids",), ["../outside"]),
    (("frame_ids",), ["../outside"]), (("requested_times",), [True, 6000]),
    (("completed_frames", 0, "at_ms"), 199), (("completed_frames", 0, "source_fingerprint"), "changed"),
    (("plan", 0, "actual_at_ms"), 199), (("plan", 0, "frame_id"), "frame_one"),
    (("plan", 0, "source_fingerprint"), "changed"), (("plan", 1, "at_ms"), 10000),
    (("plan", 1, "status"), []), (("plan", 1, "error"), {}),
    (("assets", 0, "completed"), 0), (("assets", 0, "total"), 2),
    (("assets", 0, "source_fingerprint"), "changed"), (("assets", 0, "status"), "trusted"),
    (("assets", 0, "error"), []), (("reused_frames",), 2), (("accepted_regions",), 2),
])
def test_generation_rejects_corrupt_sources_budgets_and_progress(generation_archive, path, value):
    library, _characters, _gallery, _directory, _key, state, original = generation_archive
    sample = deepcopy(original)
    target = sample
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_sampling_v2(library, state["scope_id"], sample)


def test_generation_rejects_duplicate_plan_and_completed_frames(generation_archive):
    library, _characters, _gallery, _directory, _key, state, original = generation_archive
    duplicate = deepcopy(original)
    duplicate["plan"][1] = deepcopy(duplicate["plan"][0])
    sign_plan(duplicate)
    with pytest.raises(ValueError):
        validate_sampling_v2(library, state["scope_id"], duplicate)
    duplicate = deepcopy(original)
    duplicate["completed_frames"].append(deepcopy(duplicate["completed_frames"][0]))
    duplicate.update(completed=2, failed_frames=0, status="completed")
    with pytest.raises(ValueError):
        validate_sampling_v2(library, state["scope_id"], duplicate)


def test_generation_rejects_other_work_even_with_valid_local_identifiers(generation_archive):
    library, _characters, _gallery, _directory, _key, state, sample = generation_archive
    project = library.organization.save_project({"name": "Another Series"})
    library.organization.assign(["asset_two"], project["id"])
    with pytest.raises(ValueError, match="来源"):
        validate_sampling_v2(Library(library.root), state["scope_id"], sample)


def test_generation_accepts_cancellation_pending_frames_and_zero_plan_asset(generation_archive):
    library, _characters, _gallery, _directory, _key, state, sample = generation_archive
    sample["plan"][1].update(status="pending")
    sample["plan"][1].pop("error")
    sample.update(status="cancelled", attempted=1, failed_frames=0)
    sample["assets"][1].update(status="cancelled")
    third = {**library.get_asset("asset_two"), "id": "asset_three", "episode": 3}
    atomic_json(library.asset_dir("asset_three") / "manifest.json", third)
    sample["assets"].append({"asset_id": "asset_three", "source_fingerprint": third["fingerprint"],
                             "total": 0, "completed": 0, "status": "unavailable"})
    validate_sampling_v2(library, state["scope_id"], sample)


def test_generation_does_not_invalidate_history_when_selected_role_is_disabled_or_deleted(generation_archive):
    library, characters, _gallery, _directory, _key, state, sample = generation_archive
    characters.delete("asset_one", sample["selected_character_ids"][0])
    roster = characters.roster("asset_one")
    characters.set_recognition_selection("asset_one", [], expected_revision=roster["revision"], expected_scope=state["scope_id"])
    validate_sampling_v2(library, state["scope_id"], sample)


def test_generation_preserves_retry_failures_and_warning_audit(generation_archive):
    library, _characters, _gallery, _directory, _key, state, sample = generation_archive
    failure = sample["plan"][1]
    sample["warnings"] = ["Synthetic history write was unavailable"]
    sample["attempt_history"] = [{"status": "partial", "completed": 1, "failed_frames": 1,
                                   "error": None, "updated_at": now(),
                                   "failures": [{"asset_id": failure["asset_id"], "frame_id": failure["frame_id"],
                                                 "error": failure["error"]}]}]
    validate_sampling_v2(library, state["scope_id"], sample)
    sample["attempt_history"][0]["failures"][0]["asset_id"] = "../outside"
    with pytest.raises(ValueError):
        validate_sampling_v2(library, state["scope_id"], sample)


def test_asset_sampling_version_two_retains_single_asset_budget(generation_archive):
    library, _characters, _gallery, _directory, _key, state, sample = generation_archive
    sample.update(request_scope="asset", status="completed", total=1, completed=1, attempted=1, failed_frames=0)
    sample["plan"] = sample["plan"][:1]
    sample["requested_times"] = sample["requested_times"][:1]
    sample["assets"] = sample["assets"][:1]
    sample["budget"].update(max_frames=24, max_assets=1)
    sign_plan(sample)
    validate_sampling_v2(library, state["scope_id"], sample)
    sample["budget"]["max_frames"] = 25
    with pytest.raises(ValueError):
        validate_sampling_v2(library, state["scope_id"], sample)


def test_bad_work_sampling_backup_never_publishes_imported_library(generation_archive, tmp_path):
    library, _characters, _gallery, directory, _key, _state, sample = generation_archive
    sample["budget"]["max_frames"] = 121
    atomic_json(directory / "sampling.json", sample)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    with pytest.raises(ValueError, match="跨集采样"):
        restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert restored.list_assets() == []
