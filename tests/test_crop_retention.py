"""Crop lifetime/provenance regressions; synthetic data are not accuracy claims."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import shutil

from PIL import Image
import pytest

from scenerecall.crop_artifacts import configured_crop_retention, crop_retention
from scenerecall.identity_jobs import IdentityJobMixin
from scenerecall.library import Library, atomic_json, digest, read_json
from scenerecall.visual_identity import VisualIdentityPipeline, recognition_settings, validate_visual_artifacts
from test_visual_identity import FakeCCIP, add_reference, add_role, frame, picture
from test_visual_identity import visual as visual_fixture


@pytest.fixture
def visual(tmp_path):
    return visual_fixture.__wrapped__(tmp_path)


def test_development_and_production_defaults_and_explicit_policy(visual, monkeypatch):
    library, _characters, _gallery = visual
    monkeypatch.delenv("SCENERECALL_CROP_RETENTION", raising=False)
    monkeypatch.delenv("SCENERECALL_STAGE", raising=False)
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    assert pipeline.settings("asset_one")["crop_retention"] == "default"
    assert pipeline.settings("asset_one")["crop_retention_effective"] == "retain"
    source = frame(library)
    retained = pipeline.run("asset_one", [source], {"enabled": True})["regions"][0]
    assert Path(retained["crop_path"]).is_file()
    monkeypatch.setenv("SCENERECALL_STAGE", "production")
    assert pipeline.settings("asset_one")["crop_retention_effective"] == "discard"
    discarded = pipeline.run("asset_one", [source], {"enabled": True})["regions"][0]
    assert retained["id"] == discarded["id"]
    assert not Path(discarded["crop_path"]).exists()
    monkeypatch.setenv("SCENERECALL_CROP_RETENTION", "retain")
    assert crop_retention() == "retain"
    assert crop_retention("discard") == "discard"
    assert pipeline.configure("asset_one", {"crop_retention": "discard"})["crop_retention_effective"] == "discard"
    assert configured_crop_retention({"crop_retention": "default", "crop_retention_effective": "discard"}) == "discard"
    assert recognition_settings({"enabled": True, "crop_retention": "default", "crop_retention_effective": "discard"}) == {"enabled": True}
    with pytest.raises(ValueError):
        pipeline.configure("asset_one", {"crop_retention": "unknown"})


def test_discard_preserves_features_hashes_and_rebuilds_without_files(visual):
    library, characters, gallery = visual
    reference = add_reference(gallery, add_role(characters), red=40)
    backend = FakeCCIP(boxes=[[8, 0, 120, 192]])
    pipeline = VisualIdentityPipeline(library, backend)
    source = frame(library)
    result = pipeline.run("asset_one", [source], {"enabled": True, "crop_retention": "discard"})
    row = result["regions"][0]
    assert row["feature"] and row["crop_sha256"] and row["source_frame_sha256"]
    assert row["crop_box_pixels"] == [8, 0, 120, 192]
    assert result["metrics"]["crop_encoded_bytes"] > 0
    assert result["metrics"]["crop_retained_bytes"] == 0
    assert not list((library.asset_dir("asset_one") / "identity").rglob("*.png"))
    first = pipeline.crop_bytes("asset_one", row["id"])
    assert hashlib.sha256(first).hexdigest() == row["crop_sha256"]
    assert Image.open(io.BytesIO(first)).size == (112, 192)
    assert all(pipeline.crop_bytes("asset_one", row) == first for _ in range(3))
    assert not Path(row["crop_path"]).exists()
    assert Path(reference["path"]).is_file() and Path(source["path"]).is_file()
    calls = backend.feature_calls
    shutil.rmtree(library.root / "cache" / "identity-features")
    # Raw region features survive even an evicted cache or a metadata backup.
    rerun = pipeline.run("asset_one", [source], {"enabled": True, "crop_retention": "discard"})
    assert backend.feature_calls == calls + 1  # only the independent reference
    assert rerun["regions"][0]["feature"] == row["feature"]


def test_missing_development_crop_rebuild_checks_source_pixels_and_timestamp(visual):
    library, _characters, _gallery = visual
    source = frame(library)
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    row = pipeline.run("asset_one", [source], {"enabled": True})["regions"][0]
    encoded = Path(row["crop_path"]).read_bytes()
    Path(row["crop_path"]).unlink()
    assert pipeline.crop_available("asset_one", row)
    assert pipeline.crop_bytes("asset_one", row) == encoded
    Path(source["path"]).write_bytes(picture(200))
    with pytest.raises(ValueError, match="source frame pixels changed"):
        pipeline.crop_bytes("asset_one", row)
    Path(source["path"]).write_bytes(picture())
    tampered = {**row, "at_ms": row["at_ms"] + 1}
    assert not pipeline.crop_available("asset_one", tampered)
    with pytest.raises(ValueError, match="timestamp changed"):
        pipeline.crop_bytes("asset_one", tampered)


def test_retained_preview_works_with_temporarily_missing_registered_frame(visual):
    library, _characters, _gallery = visual
    source = frame(library)
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    row = pipeline.run("asset_one", [source], {"enabled": True})["regions"][0]
    expected = pipeline.crop_bytes("asset_one", row)
    Path(source["path"]).unlink()
    assert pipeline.crop_available("asset_one", row)
    assert pipeline.crop_bytes("asset_one", row) == expected
    assert not pipeline.crop_available("asset_one", {**row, "at_ms": 0})
    Path(row["crop_path"]).unlink()
    assert not pipeline.crop_available("asset_one", row)
    with pytest.raises(FileNotFoundError):
        pipeline.crop_bytes("asset_one", row)


@pytest.mark.parametrize("x0,x1,end_pixel", [(11, 75.5, 76), (42, 106.5, 106)])
def test_fractional_detector_bounds_rebuild_exact_recorded_pixels(visual, x0, x1, end_pixel):
    library, _characters, _gallery = visual
    pipeline = VisualIdentityPipeline(library, FakeCCIP(boxes=[[x0, 0, x1, 192]]))
    row = pipeline.run("asset_one", [frame(library, size=(127, 192))],
                       {"enabled": True, "crop_retention": "discard"})["regions"][0]
    assert row["crop_box_pixels"] == [x0, 0, end_pixel, 192]
    encoded = pipeline.crop_bytes("asset_one", row)
    assert Image.open(io.BytesIO(encoded)).size == (end_pixel - x0, 192)
    assert hashlib.sha256(encoded).hexdigest() == row["crop_sha256"]
    with pytest.raises(ValueError, match="source bounds changed"):
        pipeline.crop_bytes("asset_one", {**row, "crop_box_pixels": [x0 + 2, 0, end_pixel, 192]})


def test_discard_backup_roundtrip_retains_decisions_features_and_reference_pixels(visual, tmp_path):
    library, characters, gallery = visual
    character = add_role(characters)
    reference = add_reference(gallery, character)
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    row = pipeline.run("asset_one", [frame(library)], {"enabled": True, "crop_retention": "discard"})["regions"][0]
    pipeline.review("asset_one", row["id"], status="confirmed", character_id=character, reason="synthetic review")
    expected = pipeline.crop_bytes("asset_one", row)
    # Backup validation may decode absent crops but must not materialize caches
    # or change raw history/decisions while inspecting them.
    before = {str(path.relative_to(library.root)): path.read_bytes()
              for path in library.root.rglob("*") if path.is_file()}
    validate_visual_artifacts(library)
    assert {str(path.relative_to(library.root)): path.read_bytes()
            for path in library.root.rglob("*") if path.is_file()} == before
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    service = VisualIdentityPipeline(restored, FakeCCIP())
    restored_row = service.reviews("asset_one")["regions"][0]
    assert restored_row["match"]["character_id"] == character
    assert restored_row["feature"] == row["feature"]
    assert service.crop_bytes("asset_one", restored_row) == expected
    assert not Path(restored_row["crop_path"]).exists()
    assert Path(service.gallery.trusted("asset_one")[0]["path"]).read_bytes() == Path(reference["path"]).read_bytes()


def test_saved_identity_prelude_applies_discard_without_rerunning_models(visual):
    library, characters, gallery = visual
    reference = add_reference(gallery, add_role(characters))
    source = frame(library)
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    result = pipeline.run("asset_one", [source], {"enabled": True})
    row = result["regions"][0]
    unrelated = pipeline.run("asset_one", [frame(library, "frame_unrelated", at_ms=900)],
                             {"enabled": True})["regions"][0]
    # An obsolete absolute path from another computer is never a deletion target.
    row["crop_path"] = reference["path"]
    settings = {"visual": {"enabled": True, "crop_retention": "discard"}, "audio": {"enabled": False}}
    job = {"asset_id": "asset_one", "run_id": "run_one", "config": {
        "start_ms": 0, "end_ms": 1000, "identity_settings": settings}}

    class Manager(IdentityJobMixin):
        def checkpoint(self, _job_id):
            return deepcopy(job)

        def identity_settings(self, _asset_id):
            current = deepcopy(settings)
            current["visual"].update(crop_retention="default", crop_retention_effective="retain")
            return current

    manager = Manager()
    manager.library = library
    header = digest([library.get_asset("asset_one")["fingerprint"], 0, 1000, settings, [source["id"]]])
    atomic_json(manager._identity_path(job), {"header": header, "visual": result, "audio": {}})
    # A saved prelude returns synchronously, without scheduling model workers.
    with pytest.raises(StopIteration):
        manager.prepare_scene_identity("job_one", [source], []).send(None)
    assert not (pipeline._directory("asset_one") / "crops" / f"{row['id']}.png").exists()
    assert Path(reference["path"]).is_file() and Path(unrelated["crop_path"]).is_file()
    assert pipeline.crop_bytes("asset_one", row)
    assert read_json(pipeline._directory("asset_one") / "regions" / f"{row['id']}.json")["crop_retention"] == "discard"


def test_failed_feature_call_cleans_operation_temporary_png(visual):
    library, _characters, _gallery = visual

    class Failing(FakeCCIP):
        temporary = None

        def feature(self, path):
            self.temporary = Path(path)
            assert self.temporary.is_file()
            raise RuntimeError("synthetic failure")

    backend = Failing()
    pipeline = VisualIdentityPipeline(library, backend)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        pipeline.run("asset_one", [frame(library)], {"enabled": True, "crop_retention": "discard"})
    assert backend.temporary is not None and not backend.temporary.exists()
    assert not list((library.asset_dir("asset_one") / "identity").rglob("*.png"))


@pytest.mark.parametrize("alias_kind", ["file", "directory"])
def test_crop_alias_cannot_redirect_discard_to_another_artifact(visual, alias_kind):
    library, _characters, _gallery = visual
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    row = pipeline.run("asset_one", [frame(library)], {"enabled": True})["regions"][0]
    original = Path(row["crop_path"])
    expected = original.read_bytes()
    if alias_kind == "file":
        protected = original.with_name("protected_other_region.png")
        protected.write_bytes(expected)
        original.unlink()
        alias, target = original, protected
    else:
        # A directory alias may resolve to sibling artifacts while remaining
        # within visual/, which ordinary subtree confinement would permit.
        target = original.parent.with_name("protected_other_crops")
        original.parent.rename(target)
        protected = target / original.name
        alias = original.parent
    try:
        alias.symlink_to(target, target_is_directory=alias_kind == "directory")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Filesystem symlinks are unavailable: {exc}")
    with pytest.raises(ValueError, match="filesystem alias"):
        pipeline.apply_crop_retention("asset_one", [row], {"crop_retention": "discard"})
    assert protected.read_bytes() == expected
    assert alias.is_symlink()
