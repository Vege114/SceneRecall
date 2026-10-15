"""Storage regression cases use only fabricated metadata, never user video files."""
from __future__ import annotations

import json
import os
import errno
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from scenerecall import file_io
from scenerecall.library import Library, atomic_json, atomic_jsonl, confined_path, read_json, safe_id
from scenerecall.providers import _atomic_json as provider_atomic_json


ASSET_ID = "asset_synthetic"


def sharing_error(code):
    error = PermissionError(errno.EACCES, "synthetic Windows sharing conflict")
    error.winerror = code
    return error


@pytest.mark.parametrize("writer,value", [(atomic_json, {"new": True}), (atomic_jsonl, [{"new": True}]),
                                          (provider_atomic_json, {"new": True})])
def test_atomic_publication_retries_transient_windows_locks(tmp_path, monkeypatch, writer, value):
    target = tmp_path / "state.json"
    original = b'{"old": true}\n'
    target.write_bytes(original)
    replace = os.replace
    calls, delays = [], []
    failures = iter([5, 32, 33, None])

    def locked_then_publish(source, destination):
        calls.append((source, destination))
        assert target.read_bytes() == original
        code = next(failures)
        if code is not None:
            raise sharing_error(code)
        return replace(source, destination)

    monkeypatch.setattr(file_io, "_WINDOWS", True)
    monkeypatch.setattr(file_io, "sleep", delays.append)
    monkeypatch.setattr(file_io.os, "replace", locked_then_publish)
    writer(target, value)
    assert len(calls) == 4 and delays == list(file_io._REPLACE_DELAYS[:3])
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("writer,value", [(atomic_json, {"new": True}), (atomic_jsonl, [{"new": True}]),
                                          (provider_atomic_json, {"new": True})])
def test_atomic_publication_exhausted_windows_lock_preserves_old_file(tmp_path, monkeypatch, writer, value):
    target = tmp_path / "state.json"
    original = b'{"old": true}\n'
    target.write_bytes(original)
    calls, delays = [], []
    failure = sharing_error(5)

    def locked(source, destination):
        calls.append((source, destination))
        assert target.read_bytes() == original
        raise failure

    monkeypatch.setattr(file_io, "_WINDOWS", True)
    monkeypatch.setattr(file_io, "sleep", delays.append)
    monkeypatch.setattr(file_io.os, "replace", locked)
    with pytest.raises(PermissionError) as caught:
        writer(target, value)
    assert caught.value is failure
    assert len(calls) == len(file_io._REPLACE_DELAYS) + 1
    assert delays == list(file_io._REPLACE_DELAYS)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("windows,code", [(False, 5), (False, 32), (True, 2), (True, 3), (True, None)])
def test_atomic_publication_other_errors_are_not_retried(tmp_path, monkeypatch, windows, code):
    target = tmp_path / "state.json"
    target.write_bytes(b"original")
    calls, delays = [], []
    failure = sharing_error(code)

    def failed(source, destination):
        calls.append((source, destination))
        raise failure

    monkeypatch.setattr(file_io, "_WINDOWS", windows)
    monkeypatch.setattr(file_io, "sleep", delays.append)
    monkeypatch.setattr(file_io.os, "replace", failed)
    with pytest.raises(PermissionError) as caught:
        atomic_json(target, {"new": True})
    assert caught.value is failure
    assert len(calls) == 1 and delays == []
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


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
            # ZipInfo normalizes the host's separator on Windows. Preserve the
            # supplied archive name so malicious backslash fixtures stay raw.
            member = zipfile.ZipInfo(name)
            member.filename = name
            archive.writestr(member, encoded)
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


@pytest.mark.parametrize("subtitle_source", ["ocr", "container"])
def test_portable_references_and_backup_roundtrip(library, tmp_path, subtitle_source):
    """Exercise native Windows paths through writes, reads and restore journals."""
    directory = library.asset_dir(ASSET_ID)
    frame = directory / "frames" / "frame_portable.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(frame)
    library.register_frame(ASSET_ID, {"id": "frame_portable", "path": str(frame), "at_ms": 1000})
    record = observation()
    record["evidence_frame_ids"] = ["frame_portable"]
    library.save_observation(ASSET_ID, record)
    cues = [{"id": "cue_portable", "start_ms": 500, "end_ms": 1500, "text": "合成字幕", "language": "zh"}]
    if subtitle_source == "ocr":
        library.save_subtitles(ASSET_ID, "run_portable", cues, 0, 6000)
    else:
        asset = library.get_asset(ASSET_ID)
        pointer = library._write_container(asset, {
            "track": {"index": 1, "codec": "subrip"}, "suffix": ".srt", "cues": cues,
            "original": b"1\n00:00:00,500 --> 00:00:01,500\nSynthetic\n"}, 0)
        active = library.active(ASSET_ID)
        active["container_subtitles"] = pointer
        atomic_json(directory / "active-analysis.json", active)
    proxy = directory / "proxy" / "browser.mp4"
    proxy.parent.mkdir()
    proxy.write_bytes(b"rebuildable proxy")
    active = library.active(ASSET_ID)
    references = [*active["observations"].values(),
                  *(value["path"] for value in active["subtitle_runs"].values()),
                  read_json(directory / "frames" / "registry.json")["frame_portable"]["path"]]
    if subtitle_source == "container":
        references += [active["container_subtitles"]["path"], active["container_subtitles"]["original_path"]]
    assert all("/" in reference and "\\" not in reference for reference in references)
    assert library.frame_path(ASSET_ID, "frame_portable") == frame.resolve()
    assert library.subtitles(ASSET_ID)[0]["text"] == "合成字幕"
    backup = library.backup()
    archive_path = library.root / "runtime" / "backups" / f"{backup['id']}.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert all("\\" not in name and "/proxy/" not in name for name in archive.namelist())
    restored = Library(tmp_path / "restored portable library")
    restored.restore(archive_path)
    assert restored.subtitles(ASSET_ID) == library.subtitles(ASSET_ID)
    assert restored.observations(ASSET_ID) == library.observations(ASSET_ID)
    assert restored.frame_path(ASSET_ID, "frame_portable").read_bytes() == frame.read_bytes()
    assert not restored.organization.journal.exists()


@pytest.mark.parametrize("bad_name", ["../escaped.json", "/absolute.json", "assets/../../escaped.json", "assets\\escaped.json",
    *(pytest.param(name, marks=pytest.mark.skipif(os.name != "nt", reason="Windows filename normalization"))
      for name in ("assets/file.json:stream", "assets/NUL.json", "assets/ending. /manifest.json"))])
def test_zip_member_traversal_rejected_without_writes(library, tmp_path, bad_name):
    path = write_zip(tmp_path / "traversal.zip", {bad_name: {"marker": "bad"}})
    before = sorted(str(p.relative_to(library.root)) for p in library.root.rglob("*") if p.is_file())
    with pytest.raises(ValueError):
        library.restore(path)
    after = sorted(str(p.relative_to(library.root)) for p in library.root.rglob("*") if p.is_file())
    assert before == after


@pytest.mark.parametrize("identity", ["..", ".", "name:stream", "trailing.", "CON", "nul.json", "LPT1", "COM9.data"])
@pytest.mark.skipif(os.name != "nt", reason="Windows filename restrictions preserve existing POSIX libraries")
def test_identifiers_reject_windows_aliases_and_streams(identity):
    with pytest.raises(ValueError, match="记录 ID"):
        safe_id(identity)


@pytest.mark.parametrize("relative", ["frames/image.png:stream", "frames/NUL.png", "frames/ending./image.png",
                                       "frames/ending /image.png", "frames/a?b.png", "frames/a\x00b.png"])
@pytest.mark.skipif(os.name != "nt", reason="Windows filename restrictions preserve existing POSIX libraries")
def test_relative_references_reject_windows_aliases_before_access(tmp_path, relative):
    with pytest.raises(ValueError, match="文件引用"):
        confined_path(tmp_path, relative)


@pytest.mark.skipif(os.name == "nt", reason="POSIX libraries retain their existing legal filenames")
def test_legacy_posix_identifiers_and_paths_remain_compatible(tmp_path):
    assert safe_id("asset:legacy") == "asset:legacy"
    path = confined_path(tmp_path, "legacy:cache/result.json")
    atomic_json(path, {"preserved": True})
    assert read_json(path) == {"preserved": True}


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("identity", ["entity:legacy", "CON", "trailing."])
def test_platform_filename_rules_preserve_legacy_macos_ids(monkeypatch, windows, identity):
    monkeypatch.setattr("scenerecall.library._WINDOWS_PATHS", windows)
    if windows:
        with pytest.raises(ValueError, match="记录 ID"):
            safe_id(identity)
    else:
        assert safe_id(identity) == identity


@pytest.mark.parametrize("windows", [False, True])
def test_platform_filename_rules_preserve_legacy_macos_references(tmp_path, monkeypatch, windows):
    monkeypatch.setattr("scenerecall.library._WINDOWS_PATHS", windows)
    if windows:
        with pytest.raises(ValueError, match="文件引用"):
            confined_path(tmp_path, "legacy:cache/result.json")
    else:
        assert confined_path(tmp_path, "legacy:cache/result.json") == (tmp_path / "legacy:cache/result.json").resolve()


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
