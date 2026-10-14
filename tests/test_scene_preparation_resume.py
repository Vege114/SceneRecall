"""Published preparation resumes from a complete plan, without batch replay."""
import asyncio
import copy

import pytest

from scenerecall.library import atomic_json, read_json
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


@pytest.mark.parametrize("action", ["pause", "cancel"])
@pytest.mark.asyncio
async def test_stopping_during_deep_analysis_resumes_after_preparation(scene_worker, monkeypatch, action):
    worker = scene_worker(30, with_subtitles=True)
    worker.provider.deep_times = {250}
    job = enqueue(worker)
    started, release = asyncio.Event(), asyncio.Event()
    analyze = worker.provider.analyze

    async def blocked(*args, **kwargs):
        started.set()
        await release.wait()
        return await analyze(*args, **kwargs)

    monkeypatch.setattr(worker.provider, "analyze", blocked)
    running = asyncio.create_task(worker.queue.execute(job["id"]))
    await asyncio.wait_for(started.wait(), 5)
    worker.queue.control(job["id"], action)
    release.set()
    await running
    stopped = worker.queue.get(job["id"])
    assert stopped["status"] == {"pause": "paused", "cancel": "cancelled"}[action]
    scene_call = worker.queue._scene_call

    async def reject_preparation_replay(job_id, stage, *args, **kwargs):
        assert stage not in {"transcript", "preparation"}
        return await scene_call(job_id, stage, *args, **kwargs)

    monkeypatch.setattr(worker.queue, "_scene_call", reject_preparation_replay)
    worker.queue.control(job["id"], "resume")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["request_count"] == 4
    assert counts(worker) == {"transcript": 1, "preparation": 1, "deep_analysis": 2}
    assert completed["stage_metrics"]["preparation"] == stopped["stage_metrics"]["preparation"]


@pytest.mark.asyncio
async def test_unpublished_preparation_snapshot_finishes_publication_without_replay(scene_worker, monkeypatch):
    worker = scene_worker(60)
    job = enqueue(worker)
    def interrupted(*args, **kwargs):
        raise ValueError("合成发布中断")

    with monkeypatch.context() as fault:
        fault.setattr(type(worker.store), "publish", interrupted)
        failed = await execute(worker, job)
    assert failed["status"] == "failed", failed
    directory = worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"]
    assert (directory / "scene-prepared.json").exists()
    assert not read_json(directory / "scene-state.json", {}).get("published")
    assert worker.store.list(worker.asset["id"]) == []
    worker.queue.control(job["id"], "retry")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["request_count"] == 2 and counts(worker) == {"preparation": 2}
    assert completed["stage_metrics"]["preparation"]["cache_hits"] == 0
    assert read_json(directory / "scene-state.json")["published"]
    assert len(worker.store.list(worker.asset["id"])) == 2


@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.asyncio
async def test_published_preparation_skips_all_receipts_after_deep_failure(scene_worker, monkeypatch, mode, legacy):
    worker = scene_worker(112, with_subtitles=True)
    worker.provider.chunk_size = 28
    worker.provider.deep_times = {250, 28250}
    worker.provider.fail_deep_at = 28250
    execution = {"transcript": mode, "preparation": mode, "subtitle": mode, "max_concurrency": 2}
    failed = await execute(worker, enqueue(worker, execution=execution))
    assert failed["status"] == "failed", failed
    directory = worker.library.asset_dir(worker.asset["id"]) / "analyses" / failed["run_id"]
    assert read_json(directory / "scene-state.json")["published"]
    snapshot_path = directory / "scene-prepared.json"
    assert snapshot_path.exists()
    initial_frames = [frame_id for chunk in worker.store.list(worker.asset["id"])
                      for frame_id in chunk["frame_ids"]]
    prep_metrics = copy.deepcopy(failed["stage_metrics"]["preparation"])
    transcript_metrics = copy.deepcopy(failed["stage_metrics"]["transcript"])
    if legacy:
        snapshot_path.unlink()
    scene_call = worker.queue._scene_call
    visited = []

    async def reject_preparation_replay(job_id, stage, *args, **kwargs):
        visited.append(stage)
        assert stage not in {"transcript", "preparation"}, "completed stage replayed its receipts"
        return await scene_call(job_id, stage, *args, **kwargs)

    monkeypatch.setattr(worker.queue, "_scene_call", reject_preparation_replay)
    worker.queue.control(failed["id"], "retry")
    completed = await execute(worker, failed)
    assert completed["status"] == "completed", completed
    assert completed["stage_progress"]["transcript"] == {"completed": 1, "total": 1, "status": "completed"}
    assert completed["stage_progress"]["preparation"] == {"completed": 2, "total": 2, "status": "completed"}
    assert completed["stage_progress"]["deep_analysis"] == {"completed": 4, "total": 4, "status": "completed"}
    assert completed["stage_metrics"]["preparation"] == prep_metrics
    assert completed["stage_metrics"]["transcript"] == transcript_metrics
    assert counts(worker) == {"transcript": 1, "preparation": 2, "deep_analysis": 5, "context_refresh": 2}
    assert visited == ["deep_analysis", "deep_analysis", "context_refresh", "context_refresh"]
    assert [frame_id for chunk in worker.store.list(worker.asset["id"]) for frame_id in chunk["frame_ids"]] == initial_frames
    assert snapshot_path.exists()


@pytest.mark.asyncio
async def test_invalid_prepared_snapshot_cannot_resume_or_spend(scene_worker):
    worker = scene_worker(30)
    worker.provider.deep_times = {250}
    worker.provider.fail_deep_at = 24250
    failed = await execute(worker, enqueue(worker))
    assert failed["status"] == "failed", failed
    path = worker.library.asset_dir(worker.asset["id"]) / "analyses" / failed["run_id"] / "scene-prepared.json"
    saved = read_json(path)
    saved["chunks"][0]["frame_ids"].pop()
    atomic_json(path, saved)
    before = list(worker.provider.calls)
    worker.queue.control(failed["id"], "retry")
    refused = await execute(worker, failed)
    assert refused["status"] == "failed", refused
    assert worker.provider.calls == before


@pytest.mark.asyncio
async def test_saved_preparation_still_rejects_changed_subtitles(scene_worker):
    worker = scene_worker(30, with_subtitles=True)
    worker.provider.deep_times = {250}
    worker.provider.fail_deep_at = 24250
    failed = await execute(worker, enqueue(worker))
    assert failed["status"] == "failed", failed
    cue = worker.library.subtitles(worker.asset["id"])[0]
    worker.library.annotate(worker.asset["id"], {"record_id": cue["id"], "summary": "人工修改后的台词"})
    before = list(worker.provider.calls)
    worker.queue.control(failed["id"], "retry")
    refused = await execute(worker, failed)
    assert refused["status"] == "failed", refused
    assert worker.provider.calls == before
