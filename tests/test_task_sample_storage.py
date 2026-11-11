"""Exercise sampling publication, frozen filenames, and post-job storage on a real video."""
import json
from pathlib import Path
import shutil

import pytest

from scenerecall.library import atomic_json, read_json
from scenerecall.main import create_app
from scenerecall.models import AnalysisInput, AssetInput
from scenerecall.scene_frame_inputs import freeze_frame_paths, resolve_frame_paths
from scenerecall.task_sample_storage import finish_task_samples
from scenerecall.visual_identity import VisualIdentityPipeline
from test_retrieval_frame_capture import many_targets
from test_retrieval_jobs import Encoder
from test_scene_jobs import SyntheticSceneProvider
from test_visual_identity import FakeCCIP


@pytest.fixture
def live_queue(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required")
    movie, _, _ = many_targets(tmp_path, count=5)
    app = create_app(tmp_path / "library", start_worker=False)
    queue = app.state.jobs
    asset = queue.library.register(AssetInput(video_path=str(movie), title="sampled test video"))
    return queue, asset["id"]


@pytest.mark.asyncio
async def test_retrieval_publishes_then_retires_only_source_bound_jpegs_and_replays(live_queue):
    queue, aid = live_queue
    encoder = Encoder()
    queue.search.visual_encoder = encoder
    _, config = queue.validate(AnalysisInput(asset_id=aid, analysis_strategy="retrieval-first-v1", retrieval_mode="core"),
                               {"bindings": {}})
    job = queue.add("analysis", config, aid)
    await queue.execute(job["id"])
    done = queue.get(job["id"])
    assert done["status"] == "completed", done.get("error")
    folder = queue.library.asset_dir(aid) / "frames"
    registry = read_json(folder / "registry.json")
    assert registry and done["sample_frame_storage"]["released_count"] == len(registry)
    assert not list(folder.glob("*.jpg"))
    fid = next(iter(registry))
    path = queue.library.frame_path(aid, fid)
    original = path.read_bytes()
    assert original.startswith(b"\xff\xd8")
    frozen = json.loads(json.dumps(freeze_frame_paths([{"id": fid, "path": str(path)}])))
    assert str(path) not in json.dumps(frozen)
    assert Path(resolve_frame_paths(queue.library, frozen)[0]["path"]).read_bytes() == original
    before = list(encoder.images)
    queue.update(job["id"], status="failed")
    queue.control(job["id"], "retry")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "completed"
    assert encoder.images == before and not list(folder.glob("*.jpg"))


@pytest.mark.asyncio
async def test_standalone_scan_and_uniform_sample_keep_frozen_paths_and_human_evidence(live_queue):
    queue, aid = live_queue
    library = queue.library
    asset = library.asset_dir(aid)
    request = queue.ensure_plan(aid)
    await queue.execute(request["job_id"])
    completed = queue.get(request["job_id"])
    assert completed["status"] == "completed", completed.get("error")
    assert completed["sample_frame_storage"]["released_count"] > 0
    assert not list((asset / "frames").glob("*.jpg"))
    # Legacy OCR/interval sampling follows the original requested time and crop.
    frames = await queue.frames(aid, library.require_source(aid), [0, 100, 200])
    expected = {row["id"]: Path(row["path"]).read_bytes() for row in frames}
    keep, pinned, evict = (row["id"] for row in frames)
    atomic_json(asset / "analyses/test/scene-inputs/old.json", {"args": [{"path": frames[0]["path"]}]})
    atomic_json(library.root / "collections/test.json", {"evidence_frame_ids": [pinned]})
    result = finish_task_samples(library, aid, lambda: None)
    assert result["released_count"] == 2 and result["preserved_count"] == 1
    assert (asset / "frames" / f"{keep}.jpg").is_file()
    assert not (asset / "frames" / f"{evict}.jpg").exists()
    assert (asset / "preserved-frames" / f"{pinned}.jpg").read_bytes() == expected[pinned]
    for fid, jpeg in expected.items():
        assert library.frame_bytes(aid, fid) == jpeg


@pytest.mark.asyncio
async def test_scene_receipts_replay_in_a_new_service_without_retaining_temporary_paths(live_queue):
    queue, aid = live_queue
    provider = SyntheticSceneProvider()
    queue.providers = provider
    request = AnalysisInput(asset_id=aid, analysis_strategy="scene-chunks-v1", stages=["vision"], max_requests=20)
    settings = {"bindings": {"preparation": "fast", "vision": "deep"}}

    async def analyze(queue):
        _, config = queue.validate(request, settings)
        job = queue.add("analysis", config, aid)
        await queue.execute(job["id"])
        end = queue.get(job["id"])
        assert end["status"] == "completed", end.get("error")
        return end

    await analyze(queue)
    asset = queue.library.asset_dir(aid)
    frozen = [read_json(path) for path in asset.glob("analyses/*/scene-inputs/*.json")]
    assert any("sample_frame_id" in json.dumps(value) for value in frozen)
    assert all("scenerecall-frame-" not in json.dumps(value) for value in frozen)
    assert provider.calls and not list((asset / "frames").glob("*.jpg"))
    # A new service instance has an independent frame cache. Its temporary
    # directory may change; its receipt and original pixels stay valid.
    restarted = create_app(queue.library.root, providers=provider, start_worker=False).state.jobs
    calls = len(provider.calls)
    await analyze(restarted)
    assert len(provider.calls) == calls


@pytest.mark.asyncio
async def test_successful_job_retains_existing_jpeg_if_source_or_file_changed(live_queue):
    queue, aid = live_queue
    library = queue.library
    source = library.require_source(aid)
    frame, = await queue.frames(aid, source, [0])
    path = Path(frame["path"])
    original = path.read_bytes()
    offline = source.with_suffix(".offline")
    source.rename(offline)
    counts = finish_task_samples(library, aid, lambda: None)
    assert counts["released_count"] == 0 and path.read_bytes() == original
    offline.rename(source)
    path.write_bytes(original + b"new independent content")
    counts = finish_task_samples(library, aid, lambda: None)
    assert counts["released_count"] == 0 and path.read_bytes() != original


@pytest.mark.asyncio
async def test_identity_task_reads_a_batch_of_retired_frames_together(live_queue, monkeypatch):
    from scenerecall import keyframes
    queue, aid = live_queue
    request = queue.ensure_plan(aid)
    await queue.execute(request["job_id"])
    plan = queue.load_plan(request["id"])
    assert len(plan["frames"]) > 1 and not list((queue.library.asset_dir(aid) / "frames").glob("*.jpg"))
    batches = []
    extract = keyframes.extract_keyframes

    def count(source, frames, *args, **kwargs):
        batches.append([frame["id"] for frame in frames])
        return extract(source, frames, *args, **kwargs)

    monkeypatch.setattr(keyframes, "extract_keyframes", count)
    result = VisualIdentityPipeline(queue.library, FakeCCIP()).run(aid, plan["frames"], {"enabled": True})
    assert result["status"] == "completed"
    assert batches == [[frame["id"] for frame in plan["frames"]]]
