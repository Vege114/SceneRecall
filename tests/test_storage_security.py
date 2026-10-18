"""Storage regression cases use only fabricated metadata, never user video files."""
from __future__ import annotations

import json
import os
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from scenerecall.library import Library, atomic_json, atomic_jsonl, confined_path


ASSET_ID = "asset_synthetic"


def seed_asset(library: Library, asset_id=ASSET_ID) -> dict:
    asset = {
        "schema_version": "1.0", "id": asset_id, "work_id": "work_synthetic", "title": "合成存储测试",
        "kind": "movie", "series": "", "season": None, "episode": None, "version": "test",
        "duration_ms": 6000, "width": 24, "height": 12, "video_codec": "h264", "audio_codec": None,
        "format_name": "mp4", "streams": [], "fingerprint": "a" * 64, "file_name": "synthetic.mp4",
        "subtitle_mode": "embedded", "subtitle_offset_ms": 0,
        "created_at": "2026-09-22T00:00:00+00:00", "status": "ready",
    }
    directory = library.asset_dir(asset_id)
    atomic_json(directory / "manifest.json", asset)
    atomic_json(directory / "active-analysis.json", {"schema_version": "1.0", "observations": {}, "subtitle_runs": {}})
    return asset


@pytest.fixture
def library(tmp_path):
    result = Library(tmp_path / "library")
    seed_asset(result)
    return result


def observation(record_id="obs_test", run_id="run_original", start=0, end=6000, summary="原始模型描述"):
    return {
        "id": record_id, "asset_id": ASSET_ID, "run_id": run_id,
        "window_id": f"window_{start}_{end}", "shot_id": "shot_test", "start_ms": start, "end_ms": end,
        "summary": summary, "entities": [{"id": "person_1", "appearance": "穿红衣的人"}],
        "evidence_frame_ids": [], "review_status": "unreviewed",
    }


def test_historical_favorite_remains_available_after_resegmentation(library):
    library.save_observation(ASSET_ID, observation())
    library.annotate(ASSET_ID, {"record_id": "obs_test", "favorite": True, "note": "历史证据笔记"})
    library.save_observation(ASSET_ID, observation("obs_left", "run_new", 0, 3000))
    library.save_observation(ASSET_ID, observation("obs_right", "run_new", 3000, 6000))
    collection = library.collections()
    assert len(collection) == 1 and collection[0]["record_id"] == "obs_test"
    assert collection[0]["is_historical"] and collection[0]["note"] == "历史证据笔记"
    library.annotate(ASSET_ID, {"record_id": "obs_test", "note": "更新历史笔记"})
    assert library.collections()[0]["note"] == "更新历史笔记"
    library.annotate(ASSET_ID, {"record_id": "obs_test", "favorite": False})
    assert library.collections() == []


def write_zip(path: Path, files: dict[str, object]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("library.json", json.dumps({"schema_version": "1.0", "id": "library_import"}))
        for name, value in files.items():
            if isinstance(value, bytes):
                encoded = value
            elif name.endswith(".jsonl"):
                encoded = "".join(json.dumps(line) + "\n" for line in value)
            else:
                encoded = json.dumps(value)
            if "\\" in name:
                # ZipInfo normalizes native separators on Windows. Preserve
                # the malicious member spelling to exercise the reader guard.
                info = zipfile.ZipInfo(name)
                info.filename = name
                archive.writestr(info, encoded)
            else:
                archive.writestr(name, encoded)
    return path


def test_metadata_backup_roundtrip_keeps_corrections_and_excludes_secrets(library, tmp_path):
    library.save_observation(ASSET_ID, observation())
    library.annotate(ASSET_ID, {"record_id": "obs_test", "summary": "人工纠错描述", "note": "影评笔记", "favorite": True})
    atomic_json(library.root / "private" / "settings.json", {"private_marker": "never-export-me"})
    backup = library.backup(False)
    path = library.root / "runtime" / "backups" / (backup["id"] + ".zip")
    with zipfile.ZipFile(path) as archive:
        assert not any(name.startswith(("private/", "runtime/", "indexes/", "media/")) for name in archive.namelist())
        assert all(b"never-export-me" not in archive.read(name) for name in archive.namelist())
    restored = Library(tmp_path / "restored")
    restored.restore(path)
    record = restored.records()[0]
    assert record["summary"] == "人工纠错描述" and record["note"] == "影评笔记" and record["favorite"]
    assert restored.observations(ASSET_ID)[0]["summary"] == "原始模型描述"
    assert not restored.get_asset(ASSET_ID)["source_available"]


@pytest.mark.skipif(os.name != "nt", reason="Windows writes native separators in stored relative references")
def test_windows_observation_and_project_backup_restore_preserves_native_references(library, tmp_path):
    project = library.organization.save_project({"name": "Windows synthetic project", "kind": "animation"})
    library.save_observation(ASSET_ID, observation())
    reference = library.active(ASSET_ID)["observations"]["obs_test"]
    assert "\\" in reference
    assert library.observations(ASSET_ID)[0]["summary"] == "原始模型描述"
    backup = library.backup()
    archive_path = library.root / "runtime" / "backups" / (backup["id"] + ".zip")
    with zipfile.ZipFile(archive_path) as archive:
        assert all("\\" not in info.filename for info in archive.infolist())
        assert all(info.create_system == 0 for info in archive.infolist())
    restored = Library(tmp_path / "windows-restored")
    restored.restore(archive_path)
    assert restored.observations(ASSET_ID)[0]["summary"] == "原始模型描述"
    assert project["id"] in restored.organization._read()["projects"]
    assert not restored.organization.journal.exists()


@pytest.mark.skipif(os.name != "nt", reason="Only Windows permits native backslashes in internal references")
def test_windows_relative_reference_with_unicode_and_spaces_stays_inside_owner(tmp_path):
    target = tmp_path / "中文 文件夹" / "evidence.json"
    assert confined_path(tmp_path, "中文 文件夹\\evidence.json") == target.resolve()
    assert confined_path(tmp_path, "中文 文件夹/evidence.json") == target.resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows path forms must never override the owner directory")
@pytest.mark.parametrize("relative", [
    r"..\outside.json", r"nested/..\..\outside.json", r"C:\outside.json", r"C:outside.json",
    r"\outside.json", r"\\server\share\outside.json", r"\\?\C:\outside.json", "/outside.json",
])
def test_windows_native_reference_rejects_drives_roots_and_parent_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        confined_path(tmp_path, relative)


@pytest.mark.parametrize("bad_name", ["../escaped.json", "/absolute.json", "assets/../../escaped.json", "assets\\escaped.json"])
def test_zip_member_traversal_rejected_without_writes(library, tmp_path, bad_name):
    path = write_zip(tmp_path / "traversal.zip", {bad_name: {"marker": "bad"}})
    before = sorted(str(p.relative_to(library.root)) for p in library.root.rglob("*") if p.is_file())
    with pytest.raises(ValueError):
        library.restore(path)
    after = sorted(str(p.relative_to(library.root)) for p in library.root.rglob("*") if p.is_file())
    assert before == after


def test_zip_symlink_rejected(library, tmp_path):
    path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("assets/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../outside")
    with pytest.raises(ValueError):
        library.restore(path)


@pytest.mark.parametrize("kind", ["observations", "subtitle_runs", "frame_registry"])
@pytest.mark.parametrize("reference", ["../../private/secret.json", "/tmp/scenerecall-never-read.json"])
def test_restore_rejects_embedded_path_escape_before_merging(tmp_path, kind, reference):
    target = Library(tmp_path / "target")
    source = Library(tmp_path / "source")
    manifest = seed_asset(source)
    active = {"schema_version": "1.0", "observations": {}, "subtitle_runs": {}}
    files = {f"assets/{ASSET_ID}/manifest.json": manifest,
             f"assets/{ASSET_ID}/active-analysis.json": active}
    if kind == "observations":
        active["observations"] = {"obs_escape": reference}
    elif kind == "subtitle_runs":
        active["subtitle_runs"] = {"run_escape": {"path": reference}}
    else:
        files[f"assets/{ASSET_ID}/frames/registry.json"] = {"frame_escape": {"path": reference, "at_ms": 0}}
    archive = write_zip(tmp_path / f"unsafe-{kind}.zip", files)
    with pytest.raises(ValueError):
        target.restore(archive)
    assert target.list_assets() == []


@pytest.mark.parametrize("kind", ["observations", "subtitle_runs"])
def test_active_references_are_confined_even_without_restore(library, tmp_path, kind):
    outside = tmp_path / ("outside.json" if kind == "observations" else "outside.jsonl")
    if kind == "observations":
        atomic_json(outside, observation(summary="must not read outside data root"))
        active = {"observations": {"obs_escape": str(outside)}, "subtitle_runs": {}}
    else:
        atomic_jsonl(outside, [{"id": "cue", "start_ms": 0, "end_ms": 500, "text": "outside"}])
        active = {"observations": {}, "subtitle_runs": {"run_escape": {"path": str(outside)}}}
    atomic_json(library.asset_dir(ASSET_ID) / "active-analysis.json", active)
    reader = library.observations if kind == "observations" else library.subtitles
    with pytest.raises((ValueError, FileNotFoundError)):
        reader(ASSET_ID)


def test_frame_registry_cannot_escape_asset(library, tmp_path):
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (2, 2)).save(outside)
    atomic_json(library.asset_dir(ASSET_ID) / "frames" / "registry.json", {
        "escape": {"path": str(outside), "at_ms": 0},
    })
    with pytest.raises((ValueError, FileNotFoundError)):
        library.frame_path(ASSET_ID, "escape")


def test_restore_conflict_is_all_or_nothing(library, tmp_path):
    old_manifest = (library.asset_dir(ASSET_ID) / "manifest.json").read_bytes()
    archive = write_zip(tmp_path / "conflict.zip", {
        "works/work_new/work.json": {"id": "work_new"},
        f"assets/{ASSET_ID}/manifest.json": {"id": ASSET_ID, "title": "overwriting is forbidden"},
    })
    with pytest.raises(ValueError):
        library.restore(archive)
    assert (library.asset_dir(ASSET_ID) / "manifest.json").read_bytes() == old_manifest
    assert not (library.root / "works" / "work_new").exists()


def test_favorite_and_note_do_not_confirm_ai_facts(library):
    library.save_observation(ASSET_ID, observation())
    library.annotate(ASSET_ID, {"record_id": "obs_test", "favorite": True, "note": "需要核对"})
    record = library.records()[0]
    assert record["review_status"] == "unreviewed"


def test_description_correction_survives_new_run_without_mutating_old_analysis(library):
    library.save_observation(ASSET_ID, observation())
    original_path = library.asset_dir(ASSET_ID) / "analyses" / "run_original" / "observations" / "obs_test.json"
    original = original_path.read_bytes()
    library.annotate(ASSET_ID, {"record_id": "obs_test", "summary": "人工纠错保留", "favorite": True})
    library.save_observation(ASSET_ID, observation(run_id="run_reanalysis", summary="模型重跑描述"))
    assert original_path.read_bytes() == original
    assert library.records()[0]["summary"] == "人工纠错保留"
    assert library.records()[0]["favorite"]
    assert library.observations(ASSET_ID)[0]["summary"] == "模型重跑描述"


def test_entity_identity_does_not_transfer_to_reused_local_id_on_new_run(library):
    library.save_observation(ASSET_ID, observation())
    library.annotate(ASSET_ID, {"record_id": "obs_test", "entity_id": "person_1",
                              "character_name": "小红", "aliases": ["红衣角色"]})
    updated = observation(run_id="run_reanalysis")
    updated["entities"] = [{"id": "person_1", "appearance": "穿蓝衣的人"}]
    library.save_observation(ASSET_ID, updated)
    # Retain the original annotation for review, but don't claim the new entity is named.
    assert library.annotations(ASSET_ID)[0]["character_name"] == "小红"
    assert "小红" not in library.records()[0]["character"]


def test_observation_run_file_cannot_be_overwritten(library):
    first = observation()
    library.save_observation(ASSET_ID, first)
    path = library.asset_dir(ASSET_ID) / "analyses" / "run_original" / "observations" / "obs_test.json"
    original = path.read_bytes()
    with pytest.raises(ValueError):
        library.save_observation(ASSET_ID, observation(summary="different data under the same run ID"))
    assert path.read_bytes() == original


def test_resegmentation_removes_fully_covered_stale_window(library):
    library.save_observation(ASSET_ID, observation())
    library.save_observation(ASSET_ID, observation("obs_left", "run_split", 0, 3000))
    library.save_observation(ASSET_ID, observation("obs_right", "run_split", 3000, 6000))
    assert {r["id"] for r in library.observations(ASSET_ID)} == {"obs_left", "obs_right"}


def test_unreadable_visible_ocr_text_retains_review_marker(library):
    library.save_subtitles(ASSET_ID, "run_ocr", [{
        "id": "cue_unreadable", "start_ms": 1000, "end_ms": 1500, "text": "", "language": "und",
        "source": "embedded", "review_status": "needs_review", "evidence_frame_ids": [],
    }], 0, 6000)
    cue = library.subtitles(ASSET_ID)[0]
    assert cue["text"] == "" and cue["review_status"] == "needs_review"


def test_subtitle_save_cannot_write_cues_outside_requested_replacement_span(library):
    with pytest.raises(ValueError):
        library.save_subtitles(ASSET_ID, "run_wrong_span", [{
            "id": "cue_outside", "start_ms": 0, "end_ms": 500, "text": "超出本次窗口", "language": "zh",
            "source": "embedded", "review_status": "unreviewed", "evidence_frame_ids": [],
        }], 1000, 2000)
    assert library.subtitles(ASSET_ID) == []
