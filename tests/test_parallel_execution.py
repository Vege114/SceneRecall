"""Synthetic-only concurrency, ordering, budgets and durable call lifetimes."""
import asyncio
import copy

import pytest

from scenerecall.execution import bounded_map, execution_width
from scenerecall.jobs import JobQueue, StopJob
from scenerecall.providers import AIResult, ProviderError
from test_jobs_review import worker as worker
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


def options(**changes):
    return {"transcript": "parallel", "preparation": "parallel", "subtitle": "parallel",
            "max_concurrency": 3, **changes}


def dense_dialogue(worker, count=205):
    worker.library.save_subtitles(worker.asset["id"], "explicit_dense_fixture", [
        {"start_ms": index * 50, "end_ms": index * 50 + 30,
         "text": f"合成原始台词 {index}", "language": "zh"}
        for index in range(count)], 0, worker.asset["duration_ms"])


@pytest.mark.asyncio
async def test_transcript_overlaps_and_merges_in_chronological_order(scene_worker, monkeypatch):
    worker = scene_worker(12)
    dense_dialogue(worker)
    started, second_finished = asyncio.Event(), asyncio.Event()
    original = worker.provider.prepare_subtitles
    active, peak, entered = 0, 0, 0

    async def delayed(identity, cues, story_context):
        nonlocal active, peak, entered
        index = entered
        entered += 1
        active += 1
        peak = max(peak, active)
        if entered == 2:
            started.set()
        await asyncio.wait_for(started.wait(), 2)
        if index == 0:
            await asyncio.wait_for(second_finished.wait(), 2)
        result = await original(identity, cues, story_context)
        result.data["segments"][0]["summary"] = f"本组摘要 {index}"
        if index == 1:
            second_finished.set()
        active -= 1
        return result

    monkeypatch.setattr(worker.provider, "prepare_subtitles", delayed)
    completed = await execute(worker, enqueue(worker, execution=options(preparation="serial")))
    assert completed["status"] == "completed", completed
    assert peak == 2  # Live profile cap, although the task requested three.
    visual = next(call for call in worker.provider.calls if call["stage"] == "preparation")
    assert [segment["summary"] for segment in visual["subtitle_context"]["segments"]] == [
        "本组摘要 0", "本组摘要 1", "本组摘要 2"]
    transcript = [call for call in worker.provider.calls if call["stage"] == "transcript"]
    all_ids = [cue["id"] for cue in worker.library.subtitles(worker.asset["id"])]
    assert sorted(cue["id"] for call in transcript for cue in call["cues"]) == sorted(all_ids)
    stories = [call["story_context"] for call in transcript]
    assert all(story["previous_segments"] == [] and story["unresolved"] == [] for story in stories)
    assert stories[0]["prior"] == stories[1]["prior"] == stories[2]["prior"]
    assert all(len(story["previous_cues"]) <= 20 for story in stories)
    assert completed["request_count"] == 4 and completed["usage"]["total_tokens"] == 120


@pytest.mark.parametrize("continue_batches", [False, True])
@pytest.mark.asyncio
async def test_parallel_visual_reverse_completion_preserves_boundaries_and_continuation(
        scene_worker, monkeypatch, continue_batches):
    worker = scene_worker(120)
    worker.provider.continue_batches = continue_batches
    original = worker.provider.prepare_visual
    second_done = asyncio.Event()
    memories, contexts = [], []

    async def delayed(identity, frames, context_frames, subtitles, memory, characters):
        memories.append(copy.deepcopy(memory))
        contexts.append(copy.deepcopy(context_frames))
        if frames[0]["at_ms"] == 250:
            await asyncio.wait_for(second_done.wait(), 2)
        result = await original(identity, frames, context_frames, subtitles, memory, characters)
        if frames[0]["at_ms"] == 56250:
            second_done.set()
        return result

    monkeypatch.setattr(worker.provider, "prepare_visual", delayed)
    result = await execute(worker, enqueue(worker, execution=options()))
    assert result["status"] == "completed", result
    assert len(memories) == 3 and all(memory == memories[0] for memory in memories)
    assert memories[0]["recent"] == []
    assert [len(context) for context in contexts] == [0, 8, 8]
    scenes = worker.store.list(worker.asset["id"])
    assert len(scenes) == (1 if continue_batches else 3)
    assert scenes[0]["start_ms"] == 0 and scenes[-1]["end_ms"] == 120000
    assert sum(len(scene["frame_ids"]) for scene in scenes) == 120
    # Published provenance must describe the actual frozen model input. The
    # final memory is assembled chronologically for later sequential work.
    assert all(scene["memory_before"] == memories[0] for scene in scenes)
    assert [entry["chunk_id"] for entry in scenes[-1]["memory_after"]["recent"]] == [
        scene["id"] for scene in scenes]
    assert result["request_count"] == 3 and len(result["scene_receipts"]) == 3


@pytest.mark.parametrize("action", ["pause", "cancel"])
@pytest.mark.asyncio
async def test_controls_drain_started_calls_and_resume_without_repaying(scene_worker, monkeypatch, action):
    worker = scene_worker(170)
    job = enqueue(worker, execution=options())
    both_started, release = asyncio.Event(), asyncio.Event()
    original = worker.provider.prepare_visual
    entered = 0

    async def blocked(*args):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_started.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(worker.provider, "prepare_visual", blocked)
    running = asyncio.create_task(worker.queue.execute(job["id"]))
    await asyncio.wait_for(both_started.wait(), 2)
    assert len(worker.queue.get(job["id"])["inflight_calls"]) == 2
    worker.queue.control(job["id"], action)
    release.set()
    await running
    stopped = worker.queue.get(job["id"])
    assert stopped["status"] == {"pause": "paused", "cancel": "cancelled"}[action]
    assert entered == 2 and stopped["request_count"] == 2
    assert stopped["inflight_calls"] == {} and stopped["inflight_attempts"] == 0
    assert worker.store.list(worker.asset["id"]) == []
    worker.queue.control(job["id"], "resume")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["request_count"] == 4 and entered == 4
    assert completed["stage_metrics"]["preparation"]["cache_hits"] == 2


@pytest.mark.asyncio
async def test_shutdown_drains_success_and_does_not_start_budget_waiter(scene_worker, monkeypatch):
    worker = scene_worker(120)
    job = enqueue(worker, execution=options(), max_requests=4)
    started, release = asyncio.Event(), asyncio.Event()
    original = worker.provider.prepare_visual
    entered = 0

    async def blocked(*args):
        nonlocal entered
        entered += 1
        started.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(worker.provider, "prepare_visual", blocked)
    worker.queue.task = asyncio.create_task(worker.queue.execute(job["id"]))
    await asyncio.wait_for(started.wait(), 2)
    stopping = asyncio.create_task(worker.queue.stop())
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(stopping, 2)
    stopped = worker.queue.get(job["id"])
    assert entered == 1 and stopped["request_count"] == 1
    assert stopped["inflight_calls"] == {} and len(stopped["scene_receipts"]) == 1
    restarted = JobQueue(worker.library, worker.provider, worker.queue.search)
    assert restarted.get(job["id"])["request_count"] == 1
    assert restarted.get(job["id"]).get("uncertain_request_count", 0) == 0


@pytest.mark.parametrize("action", ["pause", "cancel"])
@pytest.mark.asyncio
async def test_terminal_control_status_waits_until_all_paid_calls_drain(scene_worker, monkeypatch, action):
    fixture = scene_worker(120)
    job = enqueue(fixture, execution=options())
    started, accounted = asyncio.Event(), asyncio.Event()
    gates = [asyncio.Event(), asyncio.Event()]
    original = fixture.provider.prepare_visual
    account = fixture.queue._account_scene_receipt
    entered = 0

    async def blocked(*args):
        nonlocal entered
        index = entered
        entered += 1
        if entered == 2:
            started.set()
        await gates[index].wait()
        return await original(*args)

    def signal_account(*args):
        account(*args)
        accounted.set()

    monkeypatch.setattr(fixture.provider, "prepare_visual", blocked)
    monkeypatch.setattr(fixture.queue, "_account_scene_receipt", signal_account)
    running = asyncio.create_task(fixture.queue.execute(job["id"]))
    await asyncio.wait_for(started.wait(), 2)
    fixture.queue.control(job["id"], action)
    gates[0].set()
    await asyncio.wait_for(accounted.wait(), 2)
    await asyncio.sleep(0)
    state = fixture.queue.get(job["id"])
    assert state["status"] == {"pause": "pausing", "cancel": "cancelling"}[action]
    assert state["request_count"] == 1 and state["inflight_attempts"] == 3
    with pytest.raises(ValueError, match="不支持"):
        fixture.queue.control(job["id"], "resume")
    gates[1].set()
    await running
    assert fixture.queue.get(job["id"])["status"] == {"pause": "paused", "cancel": "cancelled"}[action]


@pytest.mark.asyncio
async def test_tight_request_budget_waits_for_reservations_then_pauses(scene_worker):
    worker = scene_worker(120)
    worker.provider.delays["preparation"] = .02
    result = await execute(worker, enqueue(worker, execution=options(), max_requests=4))
    assert result["status"] == "paused", result
    assert result["request_count"] == 2 and counts(worker) == {"preparation": 2}
    assert result["inflight_calls"] == {} and result["inflight_attempts"] == 0


def test_atomic_request_and_cost_reservations_include_siblings(worker):
    queue, _asset, _ = worker
    profile = queue.providers.get("test")
    job = queue.add("index", {"max_requests": 6, "max_cost": .03})
    first = queue.before_call(job["id"], profile)
    with pytest.raises(StopJob):
        queue.before_call(job["id"], profile)
    state = queue.get(job["id"])
    assert "费用预算" in state["message"] and state["request_count"] == 0
    assert list(state["inflight_calls"]) == [first]
    queue.account(job["id"], AIResult({}, {"input_tokens": 10}, 1, .001), first)
    assert queue.get(job["id"])["inflight_attempts"] == 0


@pytest.mark.asyncio
async def test_parallel_cost_budget_waits_and_keeps_all_active_reserves_within_limit(worker):
    queue, _asset, _ = worker
    profile = queue.providers.get("test")
    job = queue.add("index", {"max_requests": 100, "max_cost": .025})
    active, peak = 0, 0

    async def provider(identity, value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        state = queue.get(job["id"])
        assert state["known_cost"] + sum(call["cost"] for call in state["inflight_calls"].values()) <= .025
        await asyncio.sleep(.01)
        active -= 1
        return AIResult({"value": value}, {"input_tokens": 10}, 1, .001)

    async def operation(index, value):
        return await queue.call(job["id"], profile, provider, value)

    results = await bounded_map(range(4), operation, 4, lambda: queue.checkpoint(job["id"]))
    assert results == [{"value": value} for value in range(4)]
    assert peak == 1  # Each new call reserves its worst-case three retries.
    final = queue.get(job["id"])
    assert final["request_count"] == 4 and final["cost"] == pytest.approx(.004)
    assert final["usage"] == {"input_tokens": 40} and final["inflight_calls"] == {}


@pytest.mark.asyncio
async def test_multiple_scene_receipts_recover_exactly_once_after_process_exit(scene_worker, monkeypatch):
    worker = scene_worker(120)
    job = enqueue(worker, execution=options())
    worker.provider.delays["preparation"] = .01

    class ProcessExit(BaseException):
        pass

    def interrupt_account(*args):
        raise ProcessExit()

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "_account_scene_receipt", interrupt_account)
        with pytest.raises(ProcessExit):
            await execute(worker, job)
    interrupted = worker.queue.get(job["id"])
    assert interrupted["request_count"] == 0 and interrupted["inflight_attempts"] == 6
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    recovered = worker.queue.get(job["id"])
    assert recovered["request_count"] == 2 and recovered["cost"] == pytest.approx(.002)
    assert recovered.get("uncertain_request_count", 0) == 0 and recovered["inflight_calls"] == {}
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["request_count"] == 3 and counts(worker) == {"preparation": 3}


@pytest.mark.asyncio
async def test_ocr_overlaps_and_durable_raw_receipts_resume_without_repaying(scene_worker, monkeypatch):
    worker = scene_worker(12)
    job = enqueue(worker, stages=["subtitle"], execution=options())
    original = worker.provider.recognize_subtitles
    both_started = asyncio.Event()
    entered = 0

    async def delayed(*args):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), 2)
        return await original(*args)

    class ProcessExit(BaseException):
        pass

    monkeypatch.setattr(worker.provider, "recognize_subtitles", delayed)
    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "_account_scene_receipt", lambda *args: (_ for _ in ()).throw(ProcessExit()))
        with pytest.raises(ProcessExit):
            await execute(worker, job)
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    assert worker.queue.get(job["id"])["request_count"] == 2
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["request_count"] == 3 and entered == 3
    cues = worker.library.subtitles(worker.asset["id"])
    assert cues[0]["start_ms"] == 0 and cues[-1]["end_ms"] == 12000


@pytest.mark.asyncio
async def test_error_drains_sibling_success_and_retry_only_repeats_failed_unit(scene_worker, monkeypatch):
    worker = scene_worker(120)
    original = worker.provider.prepare_visual
    second_started = asyncio.Event()
    failed = False

    async def failure(identity, frames, *args):
        nonlocal failed
        if frames[0]["at_ms"] == 56250:
            second_started.set()
            await asyncio.sleep(.01)
        if frames[0]["at_ms"] == 250 and not failed:
            failed = True
            await second_started.wait()
            raise ProviderError("合成失败", request_count=1, cost=.001)
        return await original(identity, frames, *args)

    monkeypatch.setattr(worker.provider, "prepare_visual", failure)
    job = enqueue(worker, execution=options())
    failed_job = await execute(worker, job)
    assert failed_job["status"] == "failed", failed_job
    assert failed_job["request_count"] == 2 and failed_job["inflight_calls"] == {}
    worker.queue.control(job["id"], "retry")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["request_count"] == 4 and counts(worker) == {"preparation": 3}


@pytest.mark.asyncio
async def test_parallel_context_is_frozen_even_when_live_cap_is_one(scene_worker, monkeypatch):
    worker = scene_worker(12)
    dense_dialogue(worker)
    get = worker.provider.get
    monkeypatch.setattr(worker.provider, "get", lambda identity: {**get(identity), "max_concurrency": 1})
    result = await execute(worker, enqueue(worker, execution=options()))
    assert result["status"] == "completed", result
    transcript = [call for call in worker.provider.calls if call["stage"] == "transcript"]
    assert all(call["story_context"]["execution_mode"] == "parallel" for call in transcript)
    assert all(call["story_context"]["previous_segments"] == [] for call in transcript)
    assert execution_width(result["config"], "transcript", worker.provider.get("fast")) == 1


@pytest.mark.asyncio
async def test_controlled_latency_parallel_benchmark_reports_wall_time(scene_worker):
    serial, parallel = scene_worker(170), scene_worker(170)
    for fixture in (serial, parallel):
        fixture.provider.delays["preparation"] = .08
    baseline = await execute(serial, enqueue(serial))
    accelerated = await execute(parallel, enqueue(parallel, execution=options()))
    assert baseline["status"] == accelerated["status"] == "completed"
    serial_wall = baseline["stage_metrics"]["preparation"]["wall_elapsed_ms"]
    parallel_wall = accelerated["stage_metrics"]["preparation"]["wall_elapsed_ms"]
    assert parallel_wall < serial_wall * .8
    assert baseline["request_count"] == accelerated["request_count"] == 4
    assert accelerated["stage_metrics"]["preparation"]["elapsed_ms"] >= parallel_wall
