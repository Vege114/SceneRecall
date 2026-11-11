import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

from scenerecall.library import atomic_json, digest, read_json
from scenerecall.library_activity import library_activity


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("sample_cleanup", SCRIPTS / "compact_sample_frames.py")
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "library"
    asset_id = "asset_sample"
    asset = root / "assets" / asset_id
    (asset / "frames").mkdir(parents=True)
    (root / "runtime").mkdir()
    signature = "1" * 64
    atomic_json(root / "library.json", {"id": "test"})
    atomic_json(asset / "manifest.json", {"id": asset_id, "fingerprint": signature})
    registry, original, frames = {}, {}, []
    for ms in (0, 100, 200):
        frame_id = "kf_" + digest([asset_id, signature, ms, [1, 1000]])
        frame = {"id": frame_id, "at_ms": ms, "pts": ms, "time_base": [1, 1000],
                 "source_pts": ms, "source_time_base": [1, 1000]}
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (ms, 80, 40)).save(buffer, "JPEG")
        original[frame_id] = buffer.getvalue()
        (asset / "frames" / f"{frame_id}.jpg").write_bytes(buffer.getvalue())
        registry[frame_id] = {"path": f"frames/{frame_id}.jpg", "at_ms": ms}
        frames.append(frame)
    atomic_json(asset / "frames/registry.json", registry)
    atomic_json(asset / "retrieval/plans/rp_test.json", {"id": "rp_test", "source_fingerprint": signature, "frames": frames})
    monkeypatch.setattr(cleanup, "check_legacy_services", lambda root: None)
    monkeypatch.setattr(cleanup.sample_frames, "verify_source", lambda *args: ("source", {}))

    def regenerate(source, metadata, requests, directory):
        for fid, entry in requests:
            assert entry["sample"]["jpeg_sha256"] == hashlib.sha256(original[fid]).hexdigest()
            (directory / (fid + ".jpg")).write_bytes(original[fid])
    monkeypatch.setattr(cleanup.sample_frames, "restore_batch", regenerate)
    return root, asset, registry, original


def test_migration_keeps_frozen_paths_pins_aliases_and_can_resume(archive, monkeypatch):
    root, asset, registry, original = archive
    pinned, frozen, sampled = original
    registry["human_frame_alias"] = {**registry[pinned]}
    atomic_json(asset / "frames/registry.json", registry)
    atomic_json(asset / "analyses/run_one/scene-input-history/old.json",
                {"args": [{"path": str(asset / registry[frozen]["path"])}]})
    old = (asset / "frames/registry.json").read_bytes()
    estimated = cleanup.run(root)["totals"]
    assert estimated["pinned_count"] == estimated["frozen_path_count"] == estimated["sample_count"] == 1
    assert cleanup.run(root, mode="verify")["totals"]["verified_count"] == 1
    assert (asset / "frames/registry.json").read_bytes() == old
    with library_activity(root), pytest.raises(RuntimeError, match="资料库正在使用"):
        cleanup.run(root, mode="apply")

    # Simulate an interruption immediately after the durable registry switch.
    old_unlink = Path.unlink
    def interrupted(path, *args, **kwargs):
        if path.name == f"{pinned}.jpg" and path.parent == asset / "frames":
            raise OSError("simulated interruption")
        return old_unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted)
        with pytest.raises(OSError, match="interruption"):
            cleanup.run(root, mode="apply")
    reg = read_json(asset / "frames/registry.json")
    assert reg[pinned]["path"] == reg["human_frame_alias"]["path"] == f"preserved-frames/{pinned}.jpg"
    assert (asset / reg[pinned]["path"]).read_bytes() == original[pinned]
    assert (asset / registry[pinned]["path"]).exists()

    final = cleanup.run(root, mode="apply")["totals"]
    assert final["removed_count"] == 2 and final["verified_count"] == 1
    reg = read_json(asset / "frames/registry.json")
    assert not (asset / registry[pinned]["path"]).exists()
    assert (asset / registry[frozen]["path"]).read_bytes() == original[frozen]
    assert reg[sampled]["sample"]["jpeg_sha256"] == hashlib.sha256(original[sampled]).hexdigest()
    assert cleanup.run(root, mode="apply")["totals"].get("removed_count", 0) == 0


def test_annotations_pin_all_of_their_saved_evidence_and_failed_regeneration_keeps_jpegs(archive, monkeypatch):
    root, asset, registry, original = archive
    first, second, sampled = original
    atomic_json(asset / "annotations/note.json", {"record_id": "window_1", "favorite": True})
    atomic_json(asset / "analyses/run_1/retrieval-units/window_1.json", {"evidence_frame_ids": [first, second]})
    def failed(*args):
        raise ValueError("different original pixels")
    monkeypatch.setattr(cleanup.sample_frames, "restore_batch", failed)
    with pytest.raises(ValueError, match="different"):
        cleanup.run(root, mode="apply")
    updated = read_json(asset / "frames/registry.json")
    assert all((asset / updated[f]["path"]).read_bytes() == original[f] for f in (first, second, sampled))
    assert all("pinned_sha256" in updated[f] for f in (first, second))
    assert updated[sampled] == registry[sampled]


def test_subtitle_favorites_only_pin_the_selected_or_explicit_evidence(archive):
    root, asset, registry, _ = archive
    first, second, _ = registry
    for index, fields in enumerate([{}, {}, {"preview_frame_id": first}]):
        atomic_json(asset / "annotations" / f"{index}.json", {"record_id": f"cue_{index}", "favorite": True, **fields})
    old = asset / "analyses" / "history" / "subtitles.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"id": "cue_0", "text": "line"}) + "\n"
                   + json.dumps({"id": "cue_1", "evidence_frame_ids": [second]}) + "\n", encoding="utf-8")
    assert cleanup.run(root)["totals"]["pinned_count"] == 2
    assert cleanup.run(root)["totals"]["sample_count"] == 1


def test_explicit_favorite_cover_pins_only_that_hit_in_a_multi_frame_record(archive):
    root, asset, registry, _ = archive
    first, second, selected = registry
    atomic_json(asset / "annotations/cover.json", {"record_id": "window_1", "favorite": True,
                                                    "preview_frame_id": selected, "preview_at_ms": 200})
    atomic_json(asset / "analyses/run_1/retrieval-units/window_1.json", {"evidence_frame_ids": [first, second, selected]})
    result = cleanup.run(root)["totals"]
    assert result["pinned_count"] == 1 and result["sample_count"] == 2
