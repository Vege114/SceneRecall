"""Independent scheduling regressions using fabricated media and model results."""
import asyncio
from collections import Counter

import pytest

from scenerecall.deletions import DeletionStore
from scenerecall.jobs import JobQueue, StopJob
from scenerecall.organization import OrganizationConflict
from scenerecall.providers import ProviderError
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


def dense_dialogue(worker, count=301):
    duration = worker.asset["duration_ms"]
    interval = max(10, duration // (count + 1))
    worker.library.save_subtitles(worker.asset["id"], "independent_dialogue", [
        {"start_ms": i * interval, "end_ms": i * interval + interval - 1,
         "text": f"明确的合成台词 {i}", "language": "zh"}
        for i in range(count)], 0, duration)


@pytest.mark.asyncio
async def test_reverse_completion_keeps_cross_batch_continuations_and_exact_coverage(scene_worker, monkeypatch):
    worker = scene_worker(120)
    dense_dialogue(worker)
    worker.provider.continue_batches = True
    original = worker.provider._record
    active = Counter()
    peaks = Counter()
    entered = {stage: asyncio.Event() for stage in ("transcript", "preparation")}
    later_finished = {stage: asyncio.Event() for stage in entered}
    completion_order = {stage: [] for stage in entered}

    async def record(stage, **inputs):
        if stage not in entered:
            return await original(stage, **inputs)
        first = (inputs["cues"][0]["start_ms"] == 0 if stage == "transcript"
                 else inputs["frames"][0]["at_ms"] == 250)
        active[stage] += 1
        peaks[stage] = max(peaks[stage], active[stage])
        if active[stage] == 2:
            entered[stage].set()
        if first:
            await asyncio.wait_for(entered[stage].wait(), 2)
            await asyncio.wait_for(later_finished[stage].wait(), 2)
        value = await original(stage, **inputs)
        completion_order[stage].append(first)
        active[stage] -= 1
        if not first:
            later_finished[stage].set()
        return value

    monkeypatch.setattr(worker.provider, "_record", record)
    final = await execute(worker, enqueue(worker, execution={
        "transcript": "parallel", "preparation": "parallel", "max_concurrency": 4}))
    assert final["status"] == "completed", final
    # The provider's limit of two is stricter than the requested task width.
    assert peaks == {"transcript": 2, "preparation": 2}
    assert all(order[0] is False for order in completion_order.values())
    scenes = worker.store.list(worker.asset["id"])
    assert len(scenes) == 1
    merged = scenes[0]
    assert (merged["start_ms"], merged["end_ms"]) == (0, 120000)
    assert len(merged["frame_ids"]) == len(set(merged["frame_ids"])) == 120
    assert len(merged["shot_ids"]) == 120
    assert len(merged["subtitle_ids"]) == 301
    transcript = [call for call in worker.provider.calls if call["stage"] == "transcript"]
    cue_ids = [cue["id"] for call in transcript for cue in call["cues"]]
    assert len(cue_ids) == len(set(cue_ids)) == 301
    assert all(call["story_context"]["previous_segments"] == [] for call in transcript)
    assert all(call["story_context"]["execution_mode"] == "parallel" for call in transcript)


@pytest.mark.asyncio
async def test_multiple_durable_receipts_recover_after_base_exception_without_repaying(scene_worker, monkeypatch):
    worker = scene_worker(4)
    dense_dialogue(worker)
    job = enqueue(worker, execution={"transcript": "parallel", "max_concurrency": 2})

    class ProcessExit(BaseException):
        pass

    def interrupt(*args):
        raise ProcessExit()

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "_account_scene_receipt", interrupt)
        with pytest.raises(ProcessExit):
            await worker.queue.execute(job["id"])
    interrupted = worker.queue.get(job["id"])
    assert len(interrupted["inflight_calls"]) == 2
    assert interrupted["inflight_attempts"] == 6
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    recovered = worker.queue.get(job["id"])
    assert recovered["request_count"] == 2
    assert len(recovered["scene_receipts"]) == 2
    assert recovered["inflight_attempts"] == 0
    assert recovered.get("uncertain_request_count", 0) == 0
    assert recovered["usage"] == {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}
    final = await execute(worker, job)
    assert final["status"] == "completed", final
    assert final["request_count"] == 5
    assert counts(worker) == {"transcript": 4, "preparation": 1}


@pytest.mark.asyncio
async def test_parallel_ocr_failure_keeps_active_track_and_retries_only_failed_batch(scene_worker, monkeypatch):
    worker = scene_worker(12, with_subtitles=True)
    original_track = worker.library.subtitles(worker.asset["id"])
    original = worker.provider.recognize_subtitles
    failed = False

    async def fail_second(identity, frames):
        nonlocal failed
        result = await original(identity, frames)
        if frames[0]["at_ms"] == 4000 and not failed:
            failed = True
            raise ProviderError("明确的合成 OCR 故障", request_count=1, cost=.001)
        return result

    monkeypatch.setattr(worker.provider, "recognize_subtitles", fail_second)
    job = enqueue(worker, stages=["subtitle"], execution={"subtitle": "parallel", "max_concurrency": 2})
    partial = await execute(worker, job)
    assert partial["status"] == "partial", partial
    assert partial["request_count"] == 3
    assert worker.library.subtitles(worker.asset["id"]) == original_track
    assert counts(worker) == {"ocr": 3}
    worker.queue.control(job["id"], "retry")
    final = await execute(worker, job)
    assert final["status"] == "completed", final
    assert final["request_count"] == 4
    assert counts(worker) == {"ocr": 4}
    assert worker.library.subtitles(worker.asset["id"]) != original_track


@pytest.mark.asyncio
async def test_parallel_story_inputs_stay_parallel_when_provider_limit_changes_on_resume(scene_worker, monkeypatch):
    worker = scene_worker(4)
    dense_dialogue(worker, 201)
    original = worker.provider.get
    provider_limit = 1

    def profile(identity):
        return {**original(identity), "max_concurrency": provider_limit}

    monkeypatch.setattr(worker.provider, "get", profile)
    job = enqueue(worker, max_requests=3, execution={"transcript": "parallel", "max_concurrency": 4})
    paused = await execute(worker, job)
    assert paused["status"] == "paused", paused
    assert counts(worker) == {"transcript": 1}
    provider_limit = 2
    worker.queue.control(job["id"], "resume", {"max_requests": 100})
    final = await execute(worker, job)
    assert final["status"] == "completed", final
    transcripts = [call for call in worker.provider.calls if call["stage"] == "transcript"]
    assert len(transcripts) == 3
    assert all(call["story_context"]["execution_mode"] == "parallel" for call in transcripts)
    assert all(call["story_context"]["previous_segments"] == [] for call in transcripts)


@pytest.mark.parametrize("action,draining,terminal", [
    ("pause", "pausing", "paused"), ("cancel", "cancelling", "cancelled")])
@pytest.mark.asyncio
async def test_controls_keep_asset_busy_until_sibling_receipt_finishes(
        scene_worker, monkeypatch, action, draining, terminal):
    worker = scene_worker(4)
    dense_dialogue(worker)
    destination = worker.library.organization.save_project({"name": "明确的合成目标项目"})
    job = enqueue(worker, execution={"transcript": "parallel", "max_concurrency": 2})
    original = worker.provider.prepare_subtitles
    both_started, release_sibling, draining_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    checkpoint = worker.queue.checkpoint

    def observed_checkpoint(identity):
        try:
            return checkpoint(identity)
        except StopJob:
            draining_started.set()
            raise

    async def controlled(identity, cues, story):
        if cues[0]["start_ms"] == 0:
            await asyncio.wait_for(both_started.wait(), 2)
            worker.queue.control(job["id"], action)
        else:
            both_started.set()
            await release_sibling.wait()
        return await original(identity, cues, story)

    monkeypatch.setattr(worker.queue, "checkpoint", observed_checkpoint)
    monkeypatch.setattr(worker.provider, "prepare_subtitles", controlled)
    running = asyncio.create_task(worker.queue.execute(job["id"]))
    try:
        await asyncio.wait_for(draining_started.wait(), 2)
        state = worker.queue.get(job["id"])
        assert not running.done()
        assert state["status"] == draining
        assert len(state["inflight_calls"]) == 1
        if action == "cancel":
            # Repeated cancellation must not expose a stopped job prematurely.
            worker.queue.control(job["id"], "cancel")
            assert worker.queue.get(job["id"])["status"] == "cancelling"
        with pytest.raises(OrganizationConflict, match="正在分析"):
            worker.library.organization.assign([worker.asset["id"]], destination["id"], worker.queue)
        with pytest.raises(ValueError, match="当前任务状态"):
            worker.queue.control(job["id"], "resume")
    finally:
        release_sibling.set()
        await asyncio.wait_for(running, 2)
    final = worker.queue.get(job["id"])
    assert final["status"] == terminal
    assert final["request_count"] == 2
    assert final["inflight_calls"] == {}
    assert counts(worker) == {"transcript": 2}


@pytest.mark.asyncio
async def test_budget_pause_keeps_asset_busy_while_sibling_frames_are_extracted(scene_worker, monkeypatch):
    worker = scene_worker(12)
    job = enqueue(worker, stages=["subtitle"], max_requests=1,
                  execution={"subtitle": "parallel", "max_concurrency": 2})
    sibling_started, release_sibling, draining_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_frames = worker.queue.frames
    original_before = worker.queue.before_call

    async def controlled_frames(asset_id, source, times, crop):
        if times[0] == 0:
            await asyncio.wait_for(sibling_started.wait(), 2)
        else:
            sibling_started.set()
            await release_sibling.wait()
        return await original_frames(asset_id, source, times, crop)

    def observed_before(*args, **kwargs):
        try:
            return original_before(*args, **kwargs)
        except StopJob:
            draining_started.set()
            raise

    monkeypatch.setattr(worker.queue, "frames", controlled_frames)
    monkeypatch.setattr(worker.queue, "before_call", observed_before)
    running = asyncio.create_task(worker.queue.execute(job["id"]))
    try:
        await asyncio.wait_for(draining_started.wait(), 2)
        state = worker.queue.get(job["id"])
        assert not running.done()
        assert state.get("inflight_calls", {}) == {}
        assert state["status"] == "pausing"
        deletions = DeletionStore(worker.library, worker.queue, worker.queue.search)
        preview = deletions.preview(kind="assets", asset_ids=[worker.asset["id"]])
        assert [item["id"] for item in preview["blocked_jobs"]] == [job["id"]]
        with pytest.raises(OrganizationConflict, match="正在运行"):
            deletions.delete(kind="assets", asset_ids=[worker.asset["id"]])
    finally:
        release_sibling.set()
        await asyncio.wait_for(running, 2)
    final = worker.queue.get(job["id"])
    assert final["status"] == "paused"
    assert final["request_count"] == 0
    assert not worker.provider.calls
