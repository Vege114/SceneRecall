"""Promotion checks use generated images and local metadata only."""
import copy
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from scripts import promote_keyframe_benchmark as promote
from scenerecall import media
from scenerecall.keyframes import ALGORITHM_PARAMETERS
from scenerecall.library import atomic_json, digest, read_json
from test_scene_jobs import scene_worker  # noqa: F401


def trial_for(worker, directory):
    asset = worker.asset
    source = worker.library.require_source(asset["id"])
    data = media.scan_keyframes(source, 0, asset["duration_ms"])
    data["algorithm_parameters"] = dict(ALGORITHM_PARAMETERS)
    rename = {}
    for frame in data["frames"]:
        identity = "kf_" + digest([asset["id"], asset["fingerprint"], frame["pts"], frame["time_base"]])
        rename[frame["id"]] = identity
        frame["id"] = identity
    for batch in data["batches"]:
        batch["frame_ids"] = [rename[identity] for identity in batch["frame_ids"]]
    outputs = [directory / "library" / "assets" / asset["id"] / "frames" / f"{frame['id']}.jpg"
               for frame in data["frames"]]
    media.extract_keyframes(source, data["frames"], outputs)
    location = read_json(worker.library.root / "private" / "media-locations.json")[asset["id"]]
    atomic_json(directory / "library" / "private" / "media-locations.json", {asset["id"]: location})
    identity = worker.queue._keyframe_plan_id(asset, 0, asset["duration_ms"], "ffmpeg-v1")
    plan = {**data, "id": identity, "plan_id": identity, "asset_id": asset["id"],
            "fingerprint": asset["fingerprint"], "start_ms": 0, "end_ms": asset["duration_ms"],
            "status": "completed"}
    inputs = {key: plan[key] for key in ("asset_id", "fingerprint", "start_ms", "end_ms")}
    inputs["source_file_bytes"] = location["size"]
    metrics = {"benchmark": "keyframe-scan", "status": "completed", "variant": "C", "remote_model_calls": 0,
               "source_metadata_unchanged": True, "input": inputs, "input_digest": digest(inputs),
               "frame_count": len(data["frames"]), "shot_count": len(data["shots"]),
               "batch_count": len(data["batches"]), "images": []}
    for frame, path in zip(data["frames"], outputs, strict=True):
        metrics["images"].append({**{key: frame[key] for key in ("id", "at_ms", "source_pts", "source_time_base")},
                                  "path": path.relative_to(directory).as_posix(),
                                  "sha256": promote.file_hash(path), "bytes": path.stat().st_size})
    atomic_json(directory / "plan.json", plan)
    atomic_json(directory / "metrics.json", metrics)
    return plan, metrics


def snapshot(root):
    files = [path for path in root.rglob("*") if path.is_file()]
    databases = set()
    for path in files:
        if path.suffix in {".sqlite", ".sqlite3", ".db"}:
            with path.open("rb") as source:
                if source.read(16) == b"SQLite format 3\0":
                    databases.add(path)
    result = {}
    for path in files:
        # Connection cleanup may checkpoint a WAL or remove SHM between the two
        # snapshots. Compare committed schema/data through a read-only SQLite
        # transaction, including WAL contents, rather than physical page layout.
        if path.name.endswith(("-wal", "-shm")) and path.with_name(path.name[:-4]) in databases:
            continue
        if path in databases:
            checksum = hashlib.sha256()
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                db.execute("BEGIN")
                for pragma in ("user_version", "application_id"):
                    checksum.update(f"{pragma}={db.execute('PRAGMA ' + pragma).fetchone()[0]}\n".encode())
                for statement in db.iterdump():
                    checksum.update(statement.encode("utf-8"))
                    checksum.update(b"\n")
            result[path.relative_to(root).as_posix()] = checksum.hexdigest()
        else:
            result[path.relative_to(root).as_posix()] = promote.file_hash(path)
    return result


def test_snapshot_ignores_checkpoints_but_detects_database_and_file_changes(tmp_path):
    database = tmp_path / "fixture.sqlite"
    document = tmp_path / "metadata.json"
    document.write_text('{"version": 1}', encoding="utf-8")
    with closing(sqlite3.connect(database)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value BLOB)")
        writer.execute("INSERT INTO sample VALUES (1,?)", (b"before",))
        writer.commit()
        before = snapshot(tmp_path)
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert snapshot(tmp_path) == before
        writer.execute("UPDATE sample SET value=?", (b"after",))
        writer.commit()
        changed = snapshot(tmp_path)
        assert changed[database.name] != before[database.name]
        writer.execute("CREATE INDEX sample_value ON sample(value)")
        writer.commit()
        schema_changed = snapshot(tmp_path)
        assert schema_changed[database.name] != changed[database.name]
        document.write_text('{"version": 2}', encoding="utf-8")
        file_changed = snapshot(tmp_path)
        assert file_changed[document.name] != schema_changed[document.name]
        assert file_changed[database.name] == schema_changed[database.name]
    # Closing the last writer may merge/remove WAL and SHM without altering data.
    assert snapshot(tmp_path) == file_changed


def test_dry_run_validates_without_writing_library(scene_worker, tmp_path):  # noqa: F811
    worker = scene_worker(3)
    trial = tmp_path / "candidate"
    trial_for(worker, trial)
    before = snapshot(worker.library.root)
    checked = promote.validate_candidate(worker.library.root, worker.asset["id"], trial, "pyav-libavfilter-v1")
    assert checked.summary["verified_image_count"] == 3
    assert checked.summary["decoder_profile"] == "pyav-libavfilter-v1"
    assert checked.summary["existing_images"] == 0
    assert snapshot(worker.library.root) == before


@pytest.mark.parametrize("fault", ["image", "snapshot", "profile", "fingerprint"])
def test_invalid_candidate_is_rejected_before_production_writes(scene_worker, tmp_path, fault):  # noqa: F811
    worker = scene_worker(2)
    trial = tmp_path / "candidate"
    _plan, metrics = trial_for(worker, trial)
    if fault == "image":
        (trial / metrics["images"][0]["path"]).write_bytes(b"changed generated image")
    elif fault == "snapshot":
        path = trial / "library" / "private" / "media-locations.json"
        locations = read_json(path)
        locations[worker.asset["id"]]["size"] += 1
        atomic_json(path, locations)
    elif fault == "profile":
        metrics["variant"] = "B"
        atomic_json(trial / "metrics.json", metrics)
    else:
        metrics["input"]["fingerprint"] = "different-source"
        metrics["input_digest"] = digest(metrics["input"])
        atomic_json(trial / "metrics.json", metrics)
    before = snapshot(worker.library.root)
    with pytest.raises(ValueError):
        promote.validate_candidate(worker.library.root, worker.asset["id"], trial, "pyav-libavfilter-v1")
    assert snapshot(worker.library.root) == before


def test_evidence_directory_cannot_escape_asset(scene_worker, tmp_path):  # noqa: F811
    worker = scene_worker(2)
    trial = tmp_path / "candidate"
    trial_for(worker, trial)
    frames = worker.library.asset_dir(worker.asset["id"]) / "frames"
    outside = tmp_path / "outside-evidence"
    outside.mkdir()
    if frames.exists():
        frames.rmdir()
    try:
        frames.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a directory symlink is unavailable on this host")
    before = snapshot(worker.library.root)
    with pytest.raises(ValueError, match="越界"):
        promote.validate_candidate(worker.library.root, worker.asset["id"], trial, "pyav-libavfilter-v1")
    assert snapshot(worker.library.root) == before
    assert list(outside.iterdir()) == []


@pytest.mark.asyncio
async def test_source_change_after_preflight_is_rejected_before_writes(scene_worker, tmp_path, monkeypatch):  # noqa: F811
    worker = scene_worker(2)
    trial = tmp_path / "candidate"
    trial_for(worker, trial)
    candidate = promote.validate_candidate(worker.library.root, worker.asset["id"], trial, "pyav-libavfilter-v1")
    monkeypatch.setattr(promote, "require_stopped_worker", lambda *_args: None)
    candidate.source.write_bytes(b"changed synthetic source after preflight")
    before = snapshot(worker.library.root)
    with pytest.raises(ValueError, match="Source metadata changed"):
        await promote.apply_candidate(candidate)
    assert snapshot(worker.library.root) == before


@pytest.mark.asyncio
async def test_apply_reuses_candidate_images_and_publishes_via_scan_job(scene_worker, tmp_path, monkeypatch):  # noqa: F811
    worker = scene_worker(3)
    old = worker.queue.ensure_plan(worker.asset["id"], decoder_profile="ffmpeg-v1")
    worker.queue.control(old["job_id"], "cancel")
    cancelled = copy.deepcopy(worker.queue.get(old["job_id"]))
    trial = tmp_path / "candidate"
    plan, metrics = trial_for(worker, trial)
    candidate = promote.validate_candidate(worker.library.root, worker.asset["id"], trial, "pyav-libavfilter-v1")
    monkeypatch.setattr(promote, "require_stopped_worker", lambda *_args: None)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Promotion must reuse its verified selection and images without decoding")

    monkeypatch.setattr(media, "scan_keyframes", forbidden)
    monkeypatch.setattr(media, "extract_keyframes", forbidden)
    result = await promote.apply_candidate(candidate)
    assert result["status"] == "promoted"
    assert result["plan_id"] != plan["id"]
    assert result["job_id"] != old["job_id"]
    assert worker.queue.get(old["job_id"]) == cancelled
    assert result["keyframe_metrics"]["selection_cache_hit"] is True
    assert result["keyframe_metrics"]["reused_frames"] == 3
    assert result["keyframe_metrics"]["extracted_frames"] == 0
    for image in metrics["images"]:
        path = worker.library.frame_path(worker.asset["id"], image["id"])
        assert promote.file_hash(Path(path)) == image["sha256"]
    assert worker.queue.get(result["job_id"])["status"] == "completed"
    assert worker.provider.calls == []
