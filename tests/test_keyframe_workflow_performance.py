"""Batch persistence and scan recovery use synthetic metadata and generated images."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scenerecall import library as library_module, media
from scenerecall.library import Library, atomic_json, digest, read_json
from scenerecall.models import AnalysisInput
from test_scene_jobs import scene_worker  # noqa: F401


def pictures(library, count, prefix="frame"):
    directory = library.asset_dir("asset_test") / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for number in range(count):
        path = directory / f"{prefix}_{number}.jpg"
        path.write_bytes(b"explicit generated persistence fixture")
        result.append({"id": path.stem, "path": str(path), "at_ms": number})
    return result


def test_batch_registration_reads_and_syncs_once_and_preserves_single_frame_api(tmp_path, monkeypatch):
    library = Library(tmp_path / "library")
    existing = pictures(library, 1, "existing")[0]
    assert library.register_frame("asset_test", existing) is existing
    frames = pictures(library, 256)
    registry_path = library.asset_dir("asset_test") / "frames" / "registry.json"
    reads, writes = [], []
    original_read, original_write = library_module.read_json, library_module.atomic_json

    def read(path, *args, **kwargs):
        if path == registry_path:
            reads.append(path)
        return original_read(path, *args, **kwargs)

    def write(path, data):
        if path == registry_path:
            writes.append(path)
        return original_write(path, data)

    monkeypatch.setattr(library_module, "read_json", read)
    monkeypatch.setattr(library_module, "atomic_json", write)
    assert library.register_frames("asset_test", frames) is frames
    assert len(reads) == len(writes) == 1
    registry = original_read(registry_path)
    assert len(registry) == 257
    assert registry[existing["id"]]["at_ms"] == 0
    assert all(row["path"].startswith("frames/") for row in registry.values())
    library.register_frames("asset_test", frames)
    assert len(writes) == 1  # Identical retries do not sync the unchanged registry.
    replacement = {**existing, "at_ms": 42}
    assert library.register_frame("asset_test", replacement) is replacement
    assert original_read(registry_path)[existing["id"]]["at_ms"] == 42


@pytest.mark.parametrize("fault", ["outside", "missing", "duplicate", "timestamp", "id"])
def test_batch_registration_validates_every_item_before_committing(tmp_path, fault):
    library = Library(tmp_path / "library")
    existing = pictures(library, 1, "existing")[0]
    library.register_frame("asset_test", existing)
    frames = pictures(library, 2)
    if fault == "outside":
        path = tmp_path / "outside.jpg"
        path.write_bytes(b"outside")
        frames[1]["path"] = str(path)
    elif fault == "missing":
        Path(frames[1]["path"]).unlink()
    elif fault == "duplicate":
        frames[1]["id"] = frames[0]["id"]
    elif fault == "timestamp":
        frames[1]["at_ms"] = True
    else:
        frames[1]["id"] = "../invalid"
    registry_path = library.asset_dir("asset_test") / "frames" / "registry.json"
    before = registry_path.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        library.register_frames("asset_test", frames)
    assert registry_path.read_bytes() == before


def test_concurrent_batches_preserve_both_writers(tmp_path):
    library = Library(tmp_path / "library")
    batches = [pictures(library, 32, prefix) for prefix in ("first", "second")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda batch: library.register_frames("asset_test", batch), batches))
    registry = read_json(library.asset_dir("asset_test") / "frames" / "registry.json")
    assert set(registry) == {frame["id"] for batch in batches for frame in batch}


def test_internal_probe_identity_is_not_persisted_or_returned_by_asset_api(scene_worker, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient
    from scenerecall.main import create_app

    original_probe = media.probe

    def probe(path):
        return {**original_probe(path), "_source_identity": {"path_digest": "local-only-fixture"},
                "_future_internal_probe_field": "not portable"}

    monkeypatch.setattr(media, "probe", probe)
    worker = scene_worker(2)
    manifest = read_json(worker.library.asset_dir(worker.asset["id"]) / "manifest.json")
    assert not any(key.startswith("_") for key in manifest)
    assert manifest["video_codec"] == "synthetic"
    assert not any(key.startswith("_") for key in worker.asset)
    app = create_app(worker.library.root, providers=worker.provider, start_worker=False, token="test")
    with TestClient(app) as client:
        assert client.post("/api/session", json={"token": "test"}).status_code == 200
        response = client.get("/api/assets")
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert not any(key.startswith("_") for key in response.json()[0])


@pytest.mark.asyncio
async def test_preview_pause_resumes_selection_and_only_missing_batches(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(260)
    plan = worker.queue.ensure_plan(worker.asset["id"])
    job_id = plan["job_id"]
    extracted, probes, updates = [], [], []
    original_extract, original_probe = media.extract_keyframes, media.probe
    original_update = worker.queue.update
    original_register = worker.library.register_frames
    batches = []

    def probe(path):
        probes.append(path)
        return original_probe(path)

    def extract(path, frames, outputs, checkpoint=None, *, metadata=None):
        assert metadata["video_codec"] == "synthetic"
        extracted.append([frame["id"] for frame in frames])
        if len(extracted) == 2:
            worker.queue.update(job_id, status="paused")
            checkpoint()
        return original_extract(path, frames, outputs, checkpoint, metadata=metadata)

    def register(asset_id, frames):
        batches.append(len(frames))
        return original_register(asset_id, frames)

    def update(identity, **fields):
        updates.append(copy.deepcopy(fields))
        if fields.get("progress") == 1:
            assert read_json(worker.queue._plan_path(worker.asset["id"], plan["id"]))["status"] == "completed"
        return original_update(identity, **fields)

    monkeypatch.setattr(media, "probe", probe)
    monkeypatch.setattr(media, "extract_keyframes", extract)
    monkeypatch.setattr(worker.library, "register_frames", register)
    monkeypatch.setattr(worker.queue, "update", update)
    await worker.queue.execute(job_id)
    assert worker.queue.get(job_id)["status"] == "paused"
    assert len(worker.scans) == 1 and batches == [128]
    assert worker.queue.load_plan(plan["id"])["status"] == "pending"
    assert not any(fields.get("progress") == 1 for fields in updates)
    assert worker.queue.get(job_id)["keyframe_metrics"]["extracted_frames"] == 128
    worker.queue.control(job_id, "resume")
    await worker.queue.execute(job_id)
    completed = worker.queue.get(job_id)
    assert completed["status"] == "completed", completed
    assert len(worker.scans) == 1 and len(probes) == 2
    assert batches == [128, 128, 4]
    assert set(extracted[0]).isdisjoint({identity for batch in extracted[2:] for identity in batch})
    metrics = completed["keyframe_metrics"]
    assert metrics["selection_cache_hit"] and metrics["selection_ms"] == 0
    assert metrics["reused_frames"] == 128 and metrics["extracted_frames"] == 132
    assert metrics["selected_frames"] == 260
    assert all(metrics[name] >= 0 for name in ("elapsed_ms", "extraction_ms", "registration_ms", "validation_ms"))
    assert any("正在保存关键帧预览" in fields.get("message", "") for fields in updates)
    assert any("正在验证并发布" in fields.get("message", "") for fields in updates)
    worker.library.validate_keyframe_plan(worker.asset["id"], worker.queue.load_plan(plan["id"]), plan["id"])


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["identity", "pts", "json"])
async def test_invalid_selection_cache_is_rebuilt(scene_worker, monkeypatch, fault):  # noqa: F811
    worker = scene_worker(3)
    plan = worker.queue.ensure_plan(worker.asset["id"])
    original_extract = media.extract_keyframes

    def fail(*args, **kwargs):
        raise ValueError("explicit synthetic extraction failure")

    monkeypatch.setattr(media, "extract_keyframes", fail)
    await worker.queue.execute(plan["job_id"])
    assert worker.queue.get(plan["job_id"])["status"] == "failed"
    cache_path = worker.library.asset_dir(worker.asset["id"]) / "keyframe-work" / plan["id"] / "selection.json"
    cached = read_json(cache_path)
    if fault == "identity":
        cached["identity"] = "obsolete parameters"
    elif fault == "pts":
        cached["selection"]["frames"][0]["source_pts"] += 1
        cached["selection_digest"] = digest(cached["selection"])
    if fault == "json":
        cache_path.write_text("{bad json", encoding="utf-8")
    else:
        atomic_json(cache_path, cached)
    monkeypatch.setattr(media, "extract_keyframes", original_extract)
    worker.queue.control(plan["job_id"], "retry")
    await worker.queue.execute(plan["job_id"])
    assert worker.queue.get(plan["job_id"])["status"] == "completed"
    assert len(worker.scans) == 2


@pytest.mark.asyncio
async def test_completed_plan_repairs_missing_preview_without_rescanning(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(3)
    plan = worker.queue.ensure_plan(worker.asset["id"], decoder_profile="ffmpeg-v1")
    await worker.queue.execute(plan["job_id"])
    saved = worker.queue.load_plan(plan["id"])
    saved.pop("decoder_profile")
    atomic_json(worker.queue._plan_path(worker.asset["id"], plan["id"]), saved)
    # A completed plan from before selection caching was introduced must also repair.
    cache_path = worker.library.asset_dir(worker.asset["id"]) / "keyframe-work" / plan["id"] / "selection.json"
    cache_path.unlink()
    missing = saved["frames"][1]
    worker.library.frame_path(worker.asset["id"], missing["id"]).unlink()
    extracted = []
    original_extract = media.extract_keyframes

    def extract(path, frames, outputs, checkpoint=None, *, metadata=None):
        extracted.extend(frame["id"] for frame in frames)
        return original_extract(path, frames, outputs, checkpoint, metadata=metadata)

    monkeypatch.setattr(media, "extract_keyframes", extract)
    monkeypatch.setattr("scenerecall.workflows.DEFAULT_KEYFRAME_DECODER_PROFILE", "pyav-libavfilter-v1")
    repaired = worker.queue.ensure_plan(worker.asset["id"])
    assert repaired["id"] == plan["id"]
    assert repaired["decoder_profile"] == "ffmpeg-v1"
    assert cache_path.is_file()
    assert read_json(cache_path)["selection"]["frames"] == saved["frames"]
    await worker.queue.execute(repaired["job_id"])
    assert worker.queue.get(repaired["job_id"])["status"] == "completed"
    assert len(worker.scans) == 1 and extracted == [missing["id"]]
    assert worker.queue.ensure_plan(worker.asset["id"])["id"] == plan["id"]
    assert worker.queue.get(repaired["job_id"])["keyframe_metrics"]["reused_frames"] == 2


@pytest.mark.asyncio
async def test_profiles_keep_old_plan_identity_and_explicit_pyav_is_independent(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(2)
    decoders = []
    original_scan = media.scan_keyframes

    def scan(*args, **kwargs):
        decoders.append(kwargs["decoder"])
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(media, "scan_keyframes", scan)
    old = worker.queue.ensure_plan(worker.asset["id"], decoder_profile="ffmpeg-v1")
    legacy_id = "plan_" + digest([worker.asset["id"], worker.asset["fingerprint"], 0,
                                   worker.asset["duration_ms"], "content-keyframes-v1"])
    assert old["id"] == legacy_id
    await worker.queue.execute(old["job_id"])
    saved = worker.queue.load_plan(old["id"])
    saved.pop("decoder_profile")
    atomic_json(worker.queue._plan_path(worker.asset["id"], old["id"]), saved)
    old_config = worker.queue.get(old["job_id"])["config"]
    old_config.pop("keyframe_decoder_profile")
    worker.queue.update(old["job_id"], config=old_config)
    monkeypatch.setattr("scenerecall.workflows.DEFAULT_KEYFRAME_DECODER_PROFILE", "pyav-libavfilter-v1")
    assert worker.queue.ensure_plan(worker.asset["id"])["id"] == old["id"]
    new = worker.queue.ensure_plan(worker.asset["id"], decoder_profile="pyav-libavfilter-v1")
    assert new["id"] != old["id"] and new["job_id"] != old["job_id"]
    assert worker.queue.get(old["job_id"])["config"] == old_config
    assert worker.queue.get(new["job_id"])["config"]["keyframe_decoder_profile"] == "pyav-libavfilter-v1"
    await worker.queue.execute(new["job_id"])
    done = worker.queue.get(new["job_id"])
    assert done["status"] == "completed", done
    assert decoders == ["ffmpeg", "pyav"]
    assert done["keyframe_metrics"]["reused_frames"] == 2
    plans = [worker.queue.validated_plan(worker.asset["id"], 0, worker.asset["duration_ms"], item["id"])
             for item in (old, new)]
    assert worker.queue._plan_decoder_profile(plans[0]) == "ffmpeg-v1"
    assert plans[1]["decoder_profile"] == "pyav-libavfilter-v1"
    caches = [read_json(worker.library.asset_dir(worker.asset["id"]) / "keyframe-work" /
                        item["id"] / "selection.json") for item in (old, new)]
    assert caches[0]["identity"] != caches[1]["identity"]
    changed = {**plans[1], "decoder_profile": "ffmpeg-v1"}
    atomic_json(worker.queue._plan_path(worker.asset["id"], new["id"]), changed)
    with pytest.raises(ValueError, match="不一致"):
        worker.queue.validated_plan(worker.asset["id"], 0, worker.asset["duration_ms"], new["id"])


@pytest.mark.asyncio
async def test_new_analysis_freezes_decoder_but_missing_legacy_setting_uses_ffmpeg(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(2)
    original_scan = media.scan_keyframes
    decoders = []

    def scan(*args, **kwargs):
        decoders.append(kwargs["decoder"])
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(media, "scan_keyframes", scan)
    monkeypatch.setattr("scenerecall.workflows.DEFAULT_KEYFRAME_DECODER_PROFILE", "pyav-libavfilter-v1")
    _, config = worker.queue.validate(AnalysisInput(asset_id=worker.asset["id"],
        analysis_strategy="content-keyframes-v1"), {"bindings": {"vision": "deep"}})
    assert config["keyframe_decoder_profile"] == "pyav-libavfilter-v1"
    job = worker.queue.add("analysis", config, worker.asset["id"])
    monkeypatch.setattr("scenerecall.workflows.DEFAULT_KEYFRAME_DECODER_PROFILE", "ffmpeg-v1")
    pyav_plan = await worker.queue.scan_plan(job["id"])
    assert pyav_plan["decoder_profile"] == "pyav-libavfilter-v1"
    _, supplied_config = worker.queue.validate(AnalysisInput(asset_id=worker.asset["id"],
        analysis_strategy="content-keyframes-v1", keyframe_plan_id=pyav_plan["id"]),
        {"bindings": {"vision": "deep"}})
    assert supplied_config["keyframe_decoder_profile"] == "pyav-libavfilter-v1"
    legacy = worker.queue.add("keyframe_scan", {"start_ms": 0, "end_ms": worker.asset["duration_ms"],
        "keyframe_decoder_profile": "ffmpeg-v1"}, worker.asset["id"])
    legacy_config = dict(legacy["config"])
    legacy_config.pop("keyframe_decoder_profile")
    worker.queue.update(legacy["id"], config=legacy_config)
    monkeypatch.setattr("scenerecall.workflows.DEFAULT_KEYFRAME_DECODER_PROFILE", "pyav-libavfilter-v1")
    legacy_plan = await worker.queue.scan_plan(legacy["id"])
    assert legacy_plan["decoder_profile"] == "ffmpeg-v1"
    assert decoders == ["pyav", "ffmpeg"]
    _, supplied_legacy_config = worker.queue.validate(AnalysisInput(asset_id=worker.asset["id"],
        analysis_strategy="content-keyframes-v1", keyframe_plan_id=legacy_plan["id"]),
        {"bindings": {"vision": "deep"}})
    assert supplied_legacy_config["keyframe_decoder_profile"] == "ffmpeg-v1"
    retrieval = worker.queue.add("analysis", {"analysis_strategy": "retrieval-first-v1", "stages": []},
                                 worker.asset["id"])
    assert retrieval["config"]["keyframe_decoder_profile"] == "pyav-libavfilter-v1"


def test_explicit_new_profile_does_not_rewrite_cancelled_old_scan(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(2)
    old = worker.queue.ensure_plan(worker.asset["id"], decoder_profile="ffmpeg-v1")
    worker.queue.control(old["job_id"], "cancel")
    before = worker.queue.get(old["job_id"])
    new = worker.queue.ensure_plan(worker.asset["id"], decoder_profile="pyav-libavfilter-v1")
    assert new["job_id"] != old["job_id"]
    assert worker.queue.get(old["job_id"]) == before
    count = len(worker.queue.list())
    with pytest.raises(ValueError, match="解码配置"):
        worker.queue.ensure_plan(worker.asset["id"], decoder_profile="unknown-profile")
    assert len(worker.queue.list()) == count
