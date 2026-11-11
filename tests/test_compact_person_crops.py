import importlib.util
from pathlib import Path

import pytest
from scenerecall.library import Library, atomic_json, read_json
from scenerecall.library_activity import library_activity
from scenerecall.visual_identity import VisualIdentityPipeline
from test_visual_identity import FakeCCIP, add_reference, add_role, frame
from test_visual_identity import visual as visual_fixture


spec = importlib.util.spec_from_file_location("compact_crops", Path(__file__).parents[1] / "scripts/compact_person_crops.py")
compact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compact)


@pytest.fixture
def visual(tmp_path):
    return visual_fixture.__wrapped__(tmp_path)


def test_legacy_crop_measure_compact_preview_and_backup_without_video(visual, tmp_path):
    library, characters, gallery = visual
    person = add_role(characters)
    reference = Path(add_reference(gallery, person)["path"])
    reference_bytes = reference.read_bytes()
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    source = frame(library)
    row = pipeline.run("asset_one", [source], {"enabled": True, "crop_retention": "retain"})["regions"][0]
    pipeline.review("asset_one", row["id"], status="confirmed", character_id=person, reason="review")
    crop = Path(row["crop_path"])
    expected = crop.read_bytes()
    raw_path = crop.parent.parent / "regions" / f"{row['id']}.json"
    raw = read_json(raw_path)
    for key in ("crop_retention", "crop_sha256", "source_frame_sha256", "crop_box_pixels", "feature"):
        raw.pop(key)
    atomic_json(raw_path, raw)
    old_metadata = raw_path.read_bytes()
    assert library.source_path("asset_one") is None
    assert compact.run(library.root)["totals"]["candidate_bytes"] == len(expected)
    assert compact.run(library.root, mode="verify")["totals"]["verified_bytes"] == len(expected)
    assert raw_path.read_bytes() == old_metadata and crop.read_bytes() == expected

    with library_activity(library.root), pytest.raises(RuntimeError, match="资料库正在使用"):
        compact.run(library.root, mode="apply")
    with library_activity(library.root, maintenance=True), pytest.raises(RuntimeError, match="资料库正在使用"):
        from scenerecall.main import create_app
        create_app(library.root, start_worker=False)
    result = compact.run(library.root, mode="apply")["totals"]
    assert result["removed_count"] == 1 and result["removed_bytes"] == len(expected)
    assert not crop.exists() and reference.read_bytes() == reference_bytes
    assert pipeline.crop_bytes("asset_one", row["id"]) == expected
    assert pipeline.reviews("asset_one")["regions"][0]["match"]["character_id"] == person
    assert compact.run(library.root, mode="apply")["totals"]["removed_count"] == 0
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert VisualIdentityPipeline(restored).crop_bytes("asset_one", row["id"]) == expected


@pytest.mark.parametrize("damage", ["missing_frame", "different_bytes", "history", "registered_crop",
                                    "scene-inputs", "scene-input-history"])
def test_skip_when_copy_is_not_proven_redundant(visual, damage):
    library, _, _ = visual
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    source = frame(library)
    row = pipeline.run("asset_one", [source], {"enabled": True, "crop_retention": "retain"})["regions"][0]
    crop = Path(row["crop_path"])
    if damage == "missing_frame":
        Path(source["path"]).unlink()
    elif damage == "different_bytes":
        crop.write_bytes(b"different")
    elif damage == "history":
        history = next((crop.parent.parent / "region-history" / row["id"]).glob("*.json"))
        atomic_json(history, {**read_json(history), "at_ms": 999})
    elif damage == "registered_crop":
        library.register_frame("asset_one", {"id": "another_frame", "path": str(crop), "at_ms": 0})
    else:
        inputs = library.asset_dir("asset_one") / "analyses" / "existing" / damage / "old.json"
        atomic_json(inputs, {"args": [[], None, {"region_images": [{"path": str(crop)}]}], "images": 1})
    before = crop.read_bytes()
    result = compact.run(library.root, mode="apply")["totals"]
    assert result["removed_count"] == 0 and result["retained_count"] == 1
    assert crop.read_bytes() == before
