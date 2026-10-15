"""Synthetic episode scheduling proves overlap, barriers and durable recovery."""
import asyncio
import threading

import pytest

from scenerecall.jobs import JobQueue
from scenerecall.library import read_json
from scenerecall.models import AnalysisInput, AssetInput, BatchAnalysisInput
from test_scene_jobs import scene_worker as scene_worker


def episode_batch(worker, count=3, *, stages=None, concurrency=4):
    source = worker.library.source_path(worker.asset["id"])
    assets = [worker.library.register(AssetInput(video_path=str(source), title="合成连续动画",
        kind="animation", series="合成连续动画", season=1, episode=index + 1, subtitle_mode="embedded"))
        for index in range(count)]
    for index, asset in enumerate(assets):
        worker.library.save_subtitles(asset["id"], f"fixture_episode_{index + 1}", [
            {"start_ms": 0, "end_ms": 1000, "text": f"原始台词第{index + 1}集", "language": "zh"}],
            0, asset["duration_ms"])
    result = worker.queue.add_batch(BatchAnalysisInput(request_id="episode_batch", items=[
        AnalysisInput(asset_id=asset["id"], stages=stages or ["vision"],
            analysis_strategy="scene-chunks-v1", max_requests=500,
            execution={"transcript": "parallel", "preparation": "parallel", "subtitle": "parallel",
                       "max_concurrency": concurrency}) for asset in assets]),
        {"bindings": {"preparation": "fast", "vision": "deep", "subtitle": "ocr"}})
    return assets, result["jobs"]


async def wait_for_jobs(queue, jobs):
    async def finished():
        while any(queue.get(job["id"])["status"] in {"queued", "running", "pausing", "cancelling"} for job in jobs):
            await asyncio.sleep(.01)
    await asyncio.wait_for(finished(), 5)


@pytest.mark.asyncio
async def test_twelve_keyframe_scans_use_bounded_local_pool(scene_worker, monkeypatch):
    worker = scene_worker(2)
    assets, _jobs = episode_batch(worker, 12)
    # This test measures scan workers only, with no queued paid analysis work.
    for job in _jobs:
        worker.queue.control(job["id"], "cancel")
    queue = worker.queue
    queue.scan_concurrency = 2
    queue._scan_slots = asyncio.Semaphore(2)
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = __import__("scenerecall.media", fromlist=["scan_keyframes"]).scan_keyframes
    lock = threading.Lock()
    active, peak = 0, 0

    def blocked(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(3), "second local scan was never admitted"
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("scenerecall.media.scan_keyframes", blocked)
    plans = [queue.ensure_plan(asset["id"]) for asset in assets]
    jobs = [queue.get(plan["job_id"]) for plan in plans]
    await queue.start()
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert sum(queue.get(job["id"])["status"] == "running" for job in jobs) == 2
        release.set()
        await wait_for_jobs(queue, jobs)
        assert peak == 2
        assert all(queue.get(job["id"])["status"] == "completed" for job in jobs)
    finally:
        release.set()
        await queue.stop()


@pytest.mark.asyncio
async def test_task_internal_and_standalone_scans_share_local_slots(scene_worker, monkeypatch):
    worker = scene_worker(2)
    _assets, jobs = episode_batch(worker, 1)
    plan = worker.queue.ensure_plan(worker.asset["id"])
    worker.queue.scan_concurrency = 1
    worker.queue._scan_slots = asyncio.Semaphore(1)
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = __import__("scenerecall.media", fromlist=["scan_keyframes"]).scan_keyframes
    active, peak, calls = 0, 0, 0

    def blocked(*args, **kwargs):
        nonlocal active, peak, calls
        calls += 1
        active += 1
        peak = max(active, peak)
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(3)
            return original(*args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr("scenerecall.media.scan_keyframes", blocked)
    first = asyncio.create_task(worker.queue.execute(plan["job_id"]))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(worker.queue.execute(jobs[0]["id"]))
        await asyncio.sleep(.03)
        assert calls == 1 and active == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 3)
        assert calls == 2 and peak == 1
        assert worker.queue.get(jobs[0]["id"])["status"] == "completed"
    finally:
        release.set()
        await asyncio.gather(first, *(task for task in [second] if task), return_exceptions=True)


@pytest.mark.asyncio
async def test_parallel_episode_dialogue_barrier_preserves_chronological_plot(scene_worker, monkeypatch):
    worker = scene_worker(2)
    assets, jobs = episode_batch(worker)
    entered, later_done, release_first = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_text, original_visual = worker.provider.prepare_subtitles, worker.provider.prepare_visual
    active, peak = 0, 0
    finished, visuals = set(), []

    async def prepare(identity, cues, story):
        nonlocal active, peak
        episode = next(index + 1 for index in range(3) if str(index + 1) in cues[0]["text"])
        active += 1
        peak = max(peak, active)
        if active == 2:
            entered.set()
        await asyncio.wait_for(entered.wait(), 2)
        if episode == 1:
            await asyncio.wait_for(release_first.wait(), 3)
        result = await original_text(identity, cues, story)
        assert story["prior"]["prior"] == {}  # No inferred earlier-episode plot.
        active -= 1
        finished.add(episode)
        if finished >= {2, 3}:
            later_done.set()
        return result

    async def visual(identity, frames, boundary, subtitles, story, characters=None):
        assert finished == {1, 2, 3}, "visual inference crossed the batch dialogue barrier"
        visuals.append(story)
        return await original_visual(identity, frames, boundary, subtitles, story, characters)

    monkeypatch.setattr(worker.provider, "prepare_subtitles", prepare)
    monkeypatch.setattr(worker.provider, "prepare_visual", visual)
    await worker.queue.start()
    try:
        await asyncio.wait_for(later_done.wait(), 3)
        assert peak == 2 and visuals == []
        release_first.set()
        await wait_for_jobs(worker.queue, jobs)
        assert all(worker.queue.get(job["id"])["status"] == "completed" for job in jobs)
        assert [row["asset_id"] for row in visuals[1]["prior"]["summaries"]] == [assets[0]["id"]]
        assert {row["asset_id"] for row in visuals[2]["prior"]["summaries"]} == {
            assets[0]["id"], assets[1]["id"]}
        assert len([call for call in worker.provider.calls if call["stage"] == "transcript"]) == 3
    finally:
        release_first.set()
        await worker.queue.stop()


@pytest.mark.asyncio
async def test_prefetch_shutdown_drains_receipts_and_restart_reuses_dialogue(scene_worker, monkeypatch):
    worker = scene_worker(2)
    _assets, jobs = episode_batch(worker, 2)
    entered, release = asyncio.Event(), asyncio.Event()
    original = worker.provider.prepare_subtitles
    calls = 0

    async def blocked(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(worker.provider, "prepare_subtitles", blocked)
    await worker.queue.start()
    try:
        await asyncio.wait_for(entered.wait(), 3)
        stopping = asyncio.create_task(worker.queue.stop())
        await asyncio.sleep(.02)
        assert not stopping.done()
        release.set()
        await asyncio.wait_for(stopping, 3)
        for job in jobs:
            saved = worker.queue.get(job["id"])
            assert saved["request_count"] == 1 and not saved["inflight_calls"]
            assert read_json(worker.library.asset_dir(saved["asset_id"]) / "analyses" /
                             saved["run_id"] / "scene-transcript.json")["complete"]
        restarted = JobQueue(worker.library, worker.provider, worker.queue.search)
        await restarted.start()
        try:
            await wait_for_jobs(restarted, jobs)
            assert all(restarted.get(job["id"])["status"] == "completed" for job in jobs)
            assert calls == 2
        finally:
            await restarted.stop()
    finally:
        release.set()
        await worker.queue.stop()


@pytest.mark.asyncio
async def test_ocr_overlaps_between_files_before_any_visual_understanding(scene_worker, monkeypatch):
    worker = scene_worker(2)
    _assets, jobs = episode_batch(worker, 2, stages=["vision", "subtitle"])
    entered, release = asyncio.Event(), asyncio.Event()
    original = worker.provider.recognize_subtitles
    calls, active, peak = 0, 0, 0

    async def blocked(*args):
        nonlocal calls, active, peak
        calls += 1
        active += 1
        peak = max(peak, active)
        if calls == 2:
            entered.set()
        await asyncio.wait_for(release.wait(), 3)
        result = await original(*args)
        active -= 1
        return result

    monkeypatch.setattr(worker.provider, "recognize_subtitles", blocked)
    await worker.queue.start()
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert peak == 2
        assert not any(call["stage"] == "preparation" for call in worker.provider.calls)
        release.set()
        await wait_for_jobs(worker.queue, jobs)
        assert all(worker.queue.get(job["id"])["status"] == "completed" for job in jobs)
    finally:
        release.set()
        await worker.queue.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("preparation_cap", [1, 2])
async def test_narrow_ocr_connection_allows_next_file_while_first_transcript_runs(
        scene_worker, monkeypatch, preparation_cap):
    worker = scene_worker(2)
    _assets, jobs = episode_batch(worker, 2, stages=["vision", "subtitle"])
    second_ocr, first_transcript, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    get_profile = worker.provider.get
    original_ocr, original_text = worker.provider.recognize_subtitles, worker.provider.prepare_subtitles
    ocr_calls, text_calls = 0, 0
    ocr_limiter = asyncio.Semaphore(1)

    def profiles(identity):
        return {**get_profile(identity), "max_concurrency": 1 if identity == "ocr" else preparation_cap}

    async def ocr(*args):
        nonlocal ocr_calls
        async with ocr_limiter:
            ocr_calls += 1
            if ocr_calls == 2:
                second_ocr.set()
            return await original_ocr(*args)

    async def transcript(*args):
        nonlocal text_calls
        text_calls += 1
        if text_calls == 1:
            first_transcript.set()
            await release.wait()
        return await original_text(*args)

    monkeypatch.setattr(worker.provider, "get", profiles)
    monkeypatch.setattr(worker.provider, "recognize_subtitles", ocr)
    monkeypatch.setattr(worker.provider, "prepare_subtitles", transcript)
    await worker.queue.start()
    try:
        await asyncio.wait_for(first_transcript.wait(), 2)
        await asyncio.wait_for(second_ocr.wait(), 2)
        assert not any(call["stage"] == "preparation" for call in worker.provider.calls)
        release.set()
        await wait_for_jobs(worker.queue, jobs)
        assert all(worker.queue.get(job["id"])["status"] == "completed" for job in jobs)
    finally:
        release.set()
        await worker.queue.stop()


@pytest.mark.parametrize("action", ["pause", "cancel"])
@pytest.mark.asyncio
async def test_batch_controls_drain_cross_file_prefetch_then_reuse_success(scene_worker, monkeypatch, action):
    worker = scene_worker(2)
    _assets, jobs = episode_batch(worker, 2)
    entered, release = asyncio.Event(), asyncio.Event()
    original = worker.provider.prepare_subtitles
    calls = 0

    async def blocked(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(worker.provider, "prepare_subtitles", blocked)
    await worker.queue.start()
    try:
        await asyncio.wait_for(entered.wait(), 2)
        worker.queue.control_batch(jobs[0]["batch_id"], action)
        assert all(worker.queue.get(job["id"])["status"] in {"pausing", "cancelling"} for job in jobs)
        release.set()
        await wait_for_jobs(worker.queue, jobs)
        expected = "paused" if action == "pause" else "cancelled"
        assert all(worker.queue.get(job["id"])["status"] == expected for job in jobs)
        assert not any(call["stage"] == "preparation" for call in worker.provider.calls)
        worker.queue.control_batch(jobs[0]["batch_id"], "resume" if action == "pause" else "retry")
        await wait_for_jobs(worker.queue, jobs)
        assert calls == 2
        assert all(worker.queue.get(job["id"])["status"] == "completed" for job in jobs)
    finally:
        release.set()
        await worker.queue.stop()


def test_batch_eta_counts_whole_shared_connection_waves():
    from scenerecall.timing import batch_timing
    estimates = [{"parallel_episode_preparation": True, "stage_estimates": {
        "transcript": {"requests_min": 1, "requests_max": 1, "concurrency": 1,
            "connection_concurrency": 5, "profile_id": "shared", "waves_min": 1, "waves_max": 1,
            "request_wall_ms_min": 15000, "request_wall_ms_max": 90000,
            "estimated_wall_ms_min": 15000, "estimated_wall_ms_max": 90000}}} for _index in range(12)]
    result = batch_timing(estimates)
    assert result["estimated_wall_ms_min"] == 3 * 15000
    assert result["estimated_wall_ms_max"] == 3 * 90000
    assert result["stage_estimates"]["transcript"]["concurrency"] == 5
    assert result["stage_estimates"]["transcript"]["waves_min"] == 3


@pytest.mark.parametrize("file_width,expected", [(1, 24), (2, 12)])
def test_batch_eta_respects_file_worker_capacity_across_distinct_connections(file_width, expected):
    from scenerecall.timing import batch_timing
    stage = {"requests_min": 1, "requests_max": 1, "concurrency": 1,
        "connection_concurrency": 16, "waves_min": 1, "waves_max": 1,
        "request_wall_ms_min": 1000, "request_wall_ms_max": 1000,
        "estimated_wall_ms_min": 1000, "estimated_wall_ms_max": 1000}
    estimates = [{"parallel_episode_preparation": True, "episode_preparation_concurrency": file_width,
        "stage_estimates": {"subtitle": {**stage, "profile_id": "ocr"},
                            "transcript": {**stage, "profile_id": "text"}}} for _index in range(12)]
    result = batch_timing(estimates)
    assert result["estimated_wall_ms_min"] == result["estimated_wall_ms_max"] == expected * 1000
    assert result["stage_estimates"]["transcript"]["concurrency"] == file_width


@pytest.mark.asyncio
async def test_eta_uses_live_connection_cap_and_matching_stage_history(scene_worker):
    worker = scene_worker(120)
    _assets, jobs = episode_batch(worker, 1)
    config = worker.queue.get(jobs[0]["id"])["config"]
    config["keyframe_plan_id"] = (await worker.queue.scan_plan(jobs[0]["id"], asset_id=config["asset_id"],
        start_ms=config["start_ms"], end_ms=config["end_ms"]))["id"]
    plan = worker.queue.validated_plan(config["asset_id"], config["start_ms"], config["end_ms"], config["keyframe_plan_id"])
    serial = {**config, "execution": {**config["execution"], "preparation": "serial"}}
    cold = worker.queue.with_timing(config, worker.queue.estimate_scenes(config, plan))
    baseline = worker.queue.with_timing(serial, worker.queue.estimate_scenes(serial, plan))
    assert cold["stage_estimates"]["preparation"]["concurrency"] == 2
    assert cold["stage_estimates"]["preparation"]["waves_min"] == 2
    assert baseline["stage_estimates"]["preparation"]["waves_min"] == 3
    assert cold["estimated_wall_ms_min"] < baseline["estimated_wall_ms_min"]
    worker.queue.update(jobs[0]["id"], stage_metrics={"preparation": {
        "request_count": 4, "elapsed_ms": 4000, "cache_hits": 20, "usage": {}}})
    warm = worker.queue.with_timing(config, worker.queue.estimate_scenes(config, plan))
    stage = warm["stage_estimates"]["preparation"]
    assert stage["latency_source"] == "history"
    assert stage["estimated_wall_ms_min"] == 2 * 700
    assert stage["estimated_wall_ms_max"] == stage["waves_max"] * 1500


@pytest.mark.asyncio
async def test_deleted_preparation_connection_fails_own_jobs_without_stopping_queue(scene_worker, monkeypatch):
    from scenerecall.providers import ProviderError
    worker = scene_worker(2)
    _assets, jobs = episode_batch(worker, 2)
    scan = worker.queue.ensure_plan(worker.asset["id"])
    indexing = worker.queue.add("index", {})
    original = worker.provider.get

    def deleted(identity):
        if identity == "fast":
            raise ProviderError("模型连接已删除")
        return original(identity)

    monkeypatch.setattr(worker.provider, "get", deleted)
    all_jobs = [*jobs, worker.queue.get(scan["job_id"]), indexing]
    await worker.queue.start()
    try:
        await wait_for_jobs(worker.queue, all_jobs)
        assert all(worker.queue.get(job["id"])["status"] == "failed" for job in jobs)
        assert worker.queue.get(scan["job_id"])["status"] == "completed"
        assert worker.queue.get(indexing["id"])["status"] == "completed"
        assert not worker.queue.task.done()
    finally:
        await worker.queue.stop()
