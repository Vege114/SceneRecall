"""Confirmation backup graphs retain human choices without trusting stored paths."""
from copy import deepcopy

import pytest

from scenerecall.character_confirmation_validation import validate_character_confirmation_artifacts
from scenerecall.library import Library, atomic_json, digest, now, read_json
from scenerecall.visual_identity import VisualIdentityPipeline, validate_visual_artifacts
from test_visual_identity import FakeCCIP, add_role, frame, visual  # noqa: F401


@pytest.fixture
def confirmation_archive(visual):  # noqa: F811
    library, characters, gallery = visual
    character_id = add_role(characters)
    other_id = add_role(characters, "Other")
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    region = pipeline.run("asset_one", [frame(library)], {"enabled": True})["regions"][0]
    scope = characters.catalog("asset_one")["scope_id"]
    key = digest(["candidate", character_id, "asset_one", region["id"]])
    source = {"asset_id": "asset_one", "frame_id": region["frame_id"], "box": region["box"],
              "region_id": region["id"], "source_fingerprint": "synthetic-video", "confirmation_id": key}
    reference = gallery.add("asset_one", character_id, library.frame_path("asset_one", "frame_one").read_bytes(),
                            source_kind="frame_crop", source=source, confirmed=True, single_character=True)
    decision = {"action": "confirm", "character_id": character_id, "source_asset_id": "asset_one",
                "region_id": region["id"], "source_fingerprint": "synthetic-video", "frame_id": region["frame_id"],
                "box": region["box"], "reference_id": reference["id"], "updated_at": now()}
    other_key = digest(["candidate", other_id, "asset_one", region["id"]])
    rejected = {**decision, "action": "reject", "character_id": other_id}
    rejected.pop("reference_id")
    state = {"schema_version": "1.0", "scope_id": scope, "revision": 3,
             "decisions": {key: decision, other_key: rejected,
                           digest(["role", other_id]): {"action": "skip", "character_id": other_id, "updated_at": now()}},
             "audit": [{"after": deepcopy(decision)}, {"after": deepcopy(rejected)}]}
    directory = library.root / "works" / scope / "confirmation"
    atomic_json(directory / "state.json", state)
    sample = {"id": digest(["sample", "asset_one"]), "scope_id": scope, "asset_id": "asset_one",
              "source_fingerprint": "synthetic-video", "status": "completed", "total": 1, "completed": 1,
              "frame_ids": ["frame_one"], "requested_times": [200], "updated_at": now()}
    atomic_json(directory / "sampling.json", sample)
    return library, characters, gallery, directory, key, state, sample


def test_confirmation_backup_restores_choices_sources_and_sampling_without_models(confirmation_archive, tmp_path):
    library, _characters, _gallery, directory, _key, state, sample = confirmation_archive
    validate_visual_artifacts(library)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    restored_directory = restored.root / directory.relative_to(library.root)
    assert read_json(restored_directory / "state.json") == state
    assert read_json(restored_directory / "sampling.json") == sample
    validate_visual_artifacts(restored)


def test_service_confirmation_survives_backup_with_its_exact_human_label(visual, tmp_path):  # noqa: F811
    from scenerecall.character_confirmation import CharacterConfirmation
    from test_visual_identity import add_reference

    library, characters, gallery = visual
    character_id = add_role(characters)
    add_reference(gallery, character_id)
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    region = pipeline.run("asset_one", [frame(library)], {"enabled": True})["regions"][0]
    service = CharacterConfirmation(library, pipeline)
    service.decide("asset_one", character_id, action="confirm", source_asset_id="asset_one",
                   region_id=region["id"], single_character=True, use_as_reference=True)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    queue = CharacterConfirmation(restored).queue("asset_one")
    assert queue["summary"]["confirmed"] == 1
    assert queue["roles"][0]["trusted_reference_count"] == 1
    saved = VisualIdentityPipeline(restored).reviews("asset_one")["regions"][0]["match"]
    assert saved["status"] == "confirmed" and saved["source"] == "human"
    assert saved["character_id"] == character_id


@pytest.mark.parametrize("field,value", [
    ("source_asset_id", "../outside"),
    ("region_id", "missing_region"),
    ("frame_id", "missing_frame"),
    ("source_fingerprint", "different-video"),
    ("box", [0, 0, .5, .5]),
    ("reference_id", "missing_reference"),
    ("character_id", "missing_character"),
    ("action", "automatically_trust_everyone"),
])
def test_confirmation_restore_rejects_broken_evidence_graph(confirmation_archive, field, value):
    library, _characters, _gallery, directory, key, state, _sample = confirmation_archive
    state["decisions"][key][field] = value
    if field in {"source_asset_id", "region_id", "character_id"}:
        row = state["decisions"].pop(key)
        state["decisions"][digest(["candidate", row["character_id"], row["source_asset_id"], row["region_id"]])] = row
    atomic_json(directory / "state.json", state)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_character_confirmation_artifacts(library)


def test_confirmation_restoration_rejects_reference_from_another_region(confirmation_archive):
    library, _characters, gallery, directory, key, state, _sample = confirmation_archive
    saved = gallery._read(state["scope_id"])
    saved["references"][0]["source"]["region_id"] = "another_region"
    atomic_json(gallery._directory(state["scope_id"]) / "gallery.json", saved)
    with pytest.raises(ValueError, match="人物来源"):
        validate_character_confirmation_artifacts(library)
    assert state["decisions"][key]["reference_id"] == saved["references"][0]["id"]


def test_confirmation_backup_retains_revoked_references_and_deleted_roles(confirmation_archive):
    library, characters, gallery, _directory, key, state, _sample = confirmation_archive
    row = state["decisions"][key]
    gallery.update("asset_one", row["character_id"], row["reference_id"], status="revoked", reason="Incorrect crop")
    characters.delete("asset_one", row["character_id"])
    validate_visual_artifacts(library)


def test_confirmation_backup_accepts_interrupted_confirmation_for_idempotent_recovery(confirmation_archive):
    library, _characters, _gallery, directory, key, state, _sample = confirmation_archive
    state["decisions"][key].update(action="confirming")
    state["decisions"][key].pop("reference_id")
    atomic_json(directory / "state.json", state)
    validate_visual_artifacts(library)


@pytest.mark.parametrize("changes", [
    {"source_fingerprint": "different-video"},
    {"frame_ids": ["../outside"]},
    {"frame_ids": [["frame_one"]]},
    {"frame_ids": ["frame_one", "frame_one"]},
    {"requested_times": [-1]},
    {"requested_times": [10000]},
    {"requested_times": [True]},
    {"completed": 2},
    {"status": "ready_without_sampling"},
    {"status": []},
    {"total": 25, "requested_times": list(range(25))},
])
def test_confirmation_backup_rejects_invalid_sampling_progress(confirmation_archive, changes):
    library, _characters, _gallery, directory, _key, _state, sample = confirmation_archive
    atomic_json(directory / "sampling.json", {**sample, **changes})
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_character_confirmation_artifacts(library)


def test_invalid_confirmation_backup_does_not_publish_imported_library(confirmation_archive, tmp_path):
    library, _characters, _gallery, directory, key, state, _sample = confirmation_archive
    state["decisions"][key]["box"] = [0, 0, .5, .5]
    atomic_json(directory / "state.json", state)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    with pytest.raises(ValueError, match="原始人物区域"):
        restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert restored.list_assets() == []
    assert not (restored.root / "assets" / "asset_one").exists()
