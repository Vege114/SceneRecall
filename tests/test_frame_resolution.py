"""Synthetic large registries stay linear without weakening confined paths."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenerecall import library as library_module
from scenerecall import scene_jobs, scenes, workflows
from scenerecall.library import Library, atomic_json, digest
from scenerecall.scenes import SceneStore


def synthetic_library(tmp_path, count):
    library = Library(tmp_path / "library")
    asset_id, duration = "asset_bulk", count * 10
    asset = {"schema_version": "1.0", "id": asset_id, "work_id": "work_bulk", "title": "Synthetic frame registry",
             "kind": "animation", "series": "Synthetic only", "season": 1, "episode": 1,
             "version": "fixture", "duration_ms": duration, "fingerprint": "synthetic_bulk",
             "subtitle_mode": "embedded", "created_at": "2026-10-13T00:00:00Z"}
    atomic_json(library.asset_dir(asset_id) / "manifest.json", asset)
    directory = library.asset_dir(asset_id) / "frames"
    directory.mkdir()
    registry, frames = {}, []
    for index in range(count):
        frame_id, at = f"frame_{index}", index * 10
        (directory / f"{frame_id}.jpg").write_bytes(b"Synthetic bytes; no media or image decoding.")
        registry[frame_id] = {"path": f"frames/{frame_id}.jpg", "at_ms": at}
        frames.append({"id": frame_id, "at_ms": at, "shot_id": "shot_bulk", "pts": at, "time_base": [1, 1000],
                       "source_pts": at, "source_time_base": [1, 1000], "reason": "synthetic"})
    atomic_json(directory / "registry.json", registry)
    plan_id = "plan_" + digest([asset_id, asset["fingerprint"], 0, duration, workflows.KEYFRAME_ALGORITHM])
    plan = {"id": plan_id, "plan_id": plan_id, "asset_id": asset_id, "fingerprint": asset["fingerprint"],
            "start_ms": 0, "end_ms": duration, "algorithm": workflows.KEYFRAME_ALGORITHM, "status": "completed",
            "shots": [{"id": "shot_bulk", "start_ms": 0, "end_ms": duration}], "frames": frames,
            "batches": [{"id": f"batch_{offset}", "shot_id": "shot_bulk", "start_ms": offset * 10,
                         "end_ms": min(offset + 8, count) * 10,
                         "frame_ids": [frame["id"] for frame in frames[offset:offset + 8]]}
                        for offset in range(0, count, 8)]}
    return library, asset_id, registry, plan


def registry_reads(monkeypatch):
    reads = []
    read = library_module.read_json

    def counted(path, *args, **kwargs):
        path = Path(path)
        if path.name == "registry.json" and path.parent.name == "frames":
            reads.append(path)
        return read(path, *args, **kwargs)

    for module in (library_module, scene_jobs, scenes, workflows):
        monkeypatch.setattr(module, "read_json", counted)
    return reads


@pytest.mark.parametrize("operation", ["batch", "snapshot", "plan", "scene", "job_frames"])
def test_two_thousand_frames_use_one_registry_read_per_operation(tmp_path, monkeypatch, operation):
    library, asset_id, registry, plan = synthetic_library(tmp_path, 2048)
    ids = list(registry)
    reads = registry_reads(monkeypatch)
    if operation in {"batch", "snapshot"}:
        paths = library.frame_paths(asset_id, ids, **({"registry": registry} if operation == "snapshot" else {}))
        assert list(paths) == ids and all(path.is_file() and path.is_absolute() for path in paths.values())
    elif operation == "plan":
        assert library.validate_keyframe_plan(asset_id, plan, expected_id=plan["id"]) == plan
    elif operation == "scene":
        row = {"run_id": "run_bulk", "start_ms": 0, "end_ms": plan["end_ms"], "summary": "Synthetic scene",
               "frame_ids": ids, "evidence_frame_ids": ids, "representative_frame_ids": [ids[0], ids[-1]],
               "shot_ids": ["shot_bulk"], "visual_facts": [{"text": "Synthetic evidence", "frame_ids": [ids[0]]}]}
        assert SceneStore(library).validate(asset_id, row)["frame_ids"] == ids
    else:
        frames = scene_jobs.SceneJobMixin._scene_frames(SimpleNamespace(library=library), asset_id, ids)
        assert [frame["id"] for frame in frames] == ids
        assert all(Path(frame["path"]).is_file() for frame in frames)
    assert len(reads) == (0 if operation == "snapshot" else 1)


def test_scalar_path_remains_compatible_and_next_batch_observes_registry_change(tmp_path, monkeypatch):
    library, asset_id, registry, _ = synthetic_library(tmp_path, 2)
    reads = registry_reads(monkeypatch)
    original = library.frame_path(asset_id, "frame_0")
    assert original.name == "frame_0.jpg" and len(reads) == 1
    registry["frame_0"]["path"] = registry["frame_1"]["path"]
    atomic_json(library.asset_dir(asset_id) / "frames/registry.json", registry)
    assert library.frame_paths(asset_id, ["frame_0"])["frame_0"].name == "frame_1.jpg"
    assert len(reads) == 2
    del registry["frame_0"]
    atomic_json(library.asset_dir(asset_id) / "frames/registry.json", registry)
    with pytest.raises(FileNotFoundError):
        library.frame_paths(asset_id, ["frame_0"])
    assert len(reads) == 3


@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("defect", ["missing_id", "missing_file", "traversal", "absolute", "symlink", "unsafe_id"])
def test_batch_resolution_preserves_path_safety_with_or_without_snapshot(tmp_path, defect, snapshot):
    library, asset_id, registry, _ = synthetic_library(tmp_path, 2)
    directory = library.asset_dir(asset_id)
    invalid_id = "frame_0"
    if defect == "missing_id":
        del registry[invalid_id]
    elif defect == "missing_file":
        (directory / registry[invalid_id]["path"]).unlink()
    elif defect == "traversal":
        (directory.parent / "outside.jpg").write_bytes(b"Outside asset")
        registry[invalid_id]["path"] = "../outside.jpg"
    elif defect == "absolute":
        registry[invalid_id]["path"] = str(directory / "frames/frame_0.jpg")
    elif defect == "symlink":
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(b"Outside library")
        link = directory / "frames/linked.jpg"
        link.symlink_to(outside)
        registry[invalid_id]["path"] = "frames/linked.jpg"
    else:
        invalid_id = "../frame_0"
    atomic_json(directory / "frames/registry.json", registry)
    with pytest.raises((ValueError, FileNotFoundError)):
        library.frame_paths(asset_id, ["frame_1", invalid_id], **({"registry": registry} if snapshot else {}))


def test_completed_plan_reuse_reads_registry_once_and_never_schedules_scan(tmp_path, monkeypatch):
    library, asset_id, _, plan = synthetic_library(tmp_path, 2048)
    path = library.asset_dir(asset_id) / "keyframe-plans" / (plan["id"] + ".json")
    atomic_json(path, plan)
    source = tmp_path / "synthetic-source.txt"
    source.write_text("Synthetic source only; never opened by media tools.")
    monkeypatch.setattr(library, "require_source", lambda _asset: source)

    def unexpected(*args, **kwargs):
        pytest.fail("A completed plan must not schedule or inspect scan jobs")

    queue = SimpleNamespace(library=library, _plan_path=lambda _asset, _plan: path,
                            list=unexpected, add=unexpected, control=unexpected)
    reads = registry_reads(monkeypatch)
    assert workflows.WorkflowQueueMixin.ensure_plan(queue, asset_id) == plan
    assert len(reads) == 1


@pytest.mark.parametrize("count", [8, 1024])
def test_backup_restore_reuses_asset_registry_for_all_observation_frames(tmp_path, monkeypatch, count):
    library, asset_id, registry, plan = synthetic_library(tmp_path, count)
    library.save_observation(asset_id, {"id": "observation_bulk", "asset_id": asset_id, "run_id": "run_bulk",
        "window_id": "window_bulk", "shot_id": "shot_bulk", "start_ms": 0, "end_ms": plan["end_ms"],
        "summary": "Synthetic observation", "evidence_frame_ids": list(registry), "entities": [], "events": []})
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    reads = registry_reads(monkeypatch)
    restored.restore(library.root / "runtime/backups" / (backup["id"] + ".zip"))
    # One archive JSON sanity pass plus one asset reference-graph snapshot.
    assert len(reads) == 2
    assert restored.observations(asset_id)[0]["evidence_frame_ids"] == list(registry)
