"""Visual overload recovery uses synthetic media, owned receipts and real validators."""
import asyncio
import os
from collections import Counter

import pytest

from scenerecall.jobs import JobQueue
from scenerecall.providers import ProviderError
from scenerecall.scene_jobs import split_visual_args, visual_fallback_calls, visual_split_index
from test_scene_jobs import enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


def overload(monkeypatch, worker, limit, *, code="timeout", pause_after_first=False):
    original = worker.provider.prepare_visual
    calls = []

    async def prepare(identity, frames, *args):
        calls.append(tuple(frame["id"] for frame in frames))
        if len(frames) > limit:
            raise ProviderError("明确的合成视觉请求失败", code=code, request_count=2,
                                usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}, cost=.002)
        result = await original(identity, frames, *args)
        if pause_after_first and len(calls) == 2:
            active = next(job for job in worker.queue.list() if job["status"] == "running")
            worker.queue.control(active["id"], "pause")
        return result

    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    return calls


@pytest.mark.parametrize("provider_type,size,requests", [("codex_cli", 24, 3), ("openai_compatible", 56, 1)])
@pytest.mark.asyncio
async def test_new_tasks_freeze_visual_batch_size_and_estimate_fallback_bounds(
        scene_worker, monkeypatch, provider_type, size, requests):
    worker = scene_worker(56)
    get = worker.provider.get
    monkeypatch.setattr(worker.provider, "get", lambda identity: {**get(identity), "provider_type": provider_type})
    job = enqueue(worker)
    assert job["config"]["visual_batch_size"] == size
    done = await execute(worker, job)
    assert done["status"] == "completed", done
    visuals = [call for call in worker.provider.calls if call["stage"] == "preparation"]
    assert [len(call["frames"]) for call in visuals] == ([24, 24, 8] if size == 24 else [56])
    assert done["request_count"] == requests
    assert done["config"]["scene_dependencies"].get("visual_batch_size", 56) == size
    plan = worker.queue.validated_plan(worker.asset["id"], 0, 56000, done["config"]["keyframe_plan_id"])
    estimate = worker.queue.estimate_scenes(done["config"], plan)
    assert estimate["stage_estimates"]["preparation"] == {
        "requests_min": requests, "requests_max": 7 if size == 24 else 15}


@pytest.mark.parametrize("provider_type", ["codex_cli", "openai_compatible"])
@pytest.mark.asyncio
async def test_legacy_tasks_keep_56_frame_units_and_original_dependency_shape(scene_worker, monkeypatch, provider_type):
    worker = scene_worker(60)
    get = worker.provider.get
    monkeypatch.setattr(worker.provider, "get", lambda identity: {**get(identity), "provider_type": provider_type})
    job = enqueue(worker, max_requests=3)
    config = dict(job["config"])
    config.pop("visual_batch_size")
    worker.queue.update(job["id"], config=config)
    paused = await execute(worker, job)
    assert paused["status"] == "paused", paused
    assert "visual_batch_size" not in paused["config"]["scene_dependencies"]
    worker.queue.control(job["id"], "resume", {"max_requests": 20})
    first = await execute(worker, job)
    assert first["status"] == "completed" and first["request_count"] == 2
    assert first["stage_metrics"]["preparation"]["cache_hits"] == 0
    assert [len(call["frames"]) for call in worker.provider.calls] == [56, 4]


@pytest.mark.parametrize("code", ["timeout", "invalid_output"])
@pytest.mark.parametrize("continues", [True, False])
@pytest.mark.asyncio
async def test_recursive_fallback_preserves_all_frames_boundaries_continuation_and_accounting(
        scene_worker, monkeypatch, code, continues):
    worker = scene_worker(56)
    worker.provider.continue_batches = continues
    calls = overload(monkeypatch, worker, 7, code=code)
    done = await execute(worker, enqueue(worker))
    assert done["status"] == "completed", done
    assert Counter(map(len, calls)) == {56: 1, 28: 2, 14: 4, 7: 8}
    assert done["request_count"] == 22  # Seven exhausted parents at two attempts, eight leaves.
    assert done["known_cost"] == pytest.approx(.022)
    assert done["usage"] == {"input_tokens": 188, "output_tokens": 94, "total_tokens": 282}
    assert done["stage_metrics"]["preparation"]["request_count"] == 22
    assert done["inflight_calls"] == {} and done["inflight_attempts"] == 0
    scenes = worker.store.list(worker.asset["id"])
    assert len(scenes) == (1 if continues else 8)
    assert (scenes[0]["start_ms"], scenes[-1]["end_ms"]) == (0, 56000)
    frame_ids = [frame_id for scene in scenes for frame_id in scene["frame_ids"]]
    assert len(frame_ids) == len(set(frame_ids)) == 56
    assert all(scene["frame_ids"][0] in scene["representative_frame_ids"]
               and scene["frame_ids"][-1] in scene["representative_frame_ids"] for scene in scenes)
    assert all(left["end_ms"] == right["start_ms"] for left, right in zip(scenes, scenes[1:]))
    # Rebuilding another identical run reuses the aggregate, without repaying leaves.
    again = await execute(worker, enqueue(worker))
    assert again["status"] == "completed" and again["request_count"] == 0
    assert len(calls) == 15


@pytest.mark.parametrize("action", ["pause", "cancel"])
@pytest.mark.asyncio
async def test_pause_and_restart_reuse_split_decision_and_successful_child_receipt(scene_worker, monkeypatch, action):
    worker = scene_worker(56)
    calls = overload(monkeypatch, worker, 28, pause_after_first=True)
    control = worker.queue.control
    monkeypatch.setattr(worker.queue, "control", lambda job_id, requested, *rest:
                        control(job_id, action if requested == "pause" else requested, *rest))
    job = enqueue(worker)
    paused = await execute(worker, job)
    assert paused["status"] == {"pause": "paused", "cancel": "cancelled"}[action], paused
    assert list(map(len, calls)) == [56, 28] and paused["request_count"] == 3
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    worker.queue.control(job["id"], "resume")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert list(map(len, calls)) == [56, 28, 28]
    assert completed["request_count"] == 4 and completed["known_cost"] == pytest.approx(.004)
    assert completed["stage_metrics"]["preparation"]["cache_hits"] == 1


@pytest.mark.asyncio
async def test_budget_pause_admits_no_new_child_and_retry_does_not_repay_parent(scene_worker, monkeypatch):
    worker = scene_worker(56)
    calls = overload(monkeypatch, worker, 28)
    job = enqueue(worker, max_requests=5)
    paused = await execute(worker, job)
    assert paused["status"] == "paused", paused
    assert list(map(len, calls)) == [56, 28] and paused["request_count"] == 3
    worker.queue.control(job["id"], "resume", {"max_requests": 20})
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert list(map(len, calls)) == [56, 28, 28] and completed["request_count"] == 4


@pytest.mark.parametrize("code", [None, "authentication", "rate_limit", "network"])
@pytest.mark.asyncio
async def test_unrecoverable_errors_do_not_expand_paid_work(scene_worker, monkeypatch, code):
    worker = scene_worker(56)
    calls = overload(monkeypatch, worker, 28, code=code)
    done = await execute(worker, enqueue(worker))
    assert done["status"] == "failed" and list(map(len, calls)) == [56]
    assert done["request_count"] == 2
    run = worker.library.asset_dir(worker.asset["id"]) / "analyses" / done["run_id"]
    assert not list((run / "scene-splits").glob("*.json"))


@pytest.mark.asyncio
async def test_small_terminal_failure_keeps_successful_left_leaf_for_retry(scene_worker, monkeypatch):
    worker = scene_worker(28)
    original = worker.provider.prepare_visual
    calls = []
    fail_right = True

    async def prepare(identity, frames, *args):
        calls.append((len(frames), frames[0]["at_ms"]))
        if len(frames) > 14 or (frames[0]["at_ms"] == 14250 and fail_right):
            raise ProviderError("合成超时", code="timeout", request_count=1, cost=.001)
        return await original(identity, frames, *args)

    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    job = enqueue(worker)
    failed = await execute(worker, job)
    assert failed["status"] == "failed", failed
    assert calls == [(28, 250), (14, 250), (14, 14250), (7, 14250)]
    assert failed["request_count"] == 4
    fail_right = False
    worker.queue.control(job["id"], "retry")
    done = await execute(worker, job)
    assert done["status"] == "completed", done
    assert calls == [(28, 250), (14, 250), (14, 14250), (7, 14250), (7, 14250), (7, 21250)]
    assert done["request_count"] == 6


def test_split_keeps_same_timestamp_frames_together_and_clips_subtitle_evidence():
    frames = [{"id": f"frame-{index}", "at_ms": index // 2 * 1000} for index in range(28)]
    index = visual_split_index(frames)
    assert index == 14 and frames[index - 1]["at_ms"] < frames[index]["at_ms"]
    assert visual_split_index([{**frame, "at_ms": 0} for frame in frames]) is None
    cues = [{"id": "left", "start_ms": 0, "end_ms": 1000, "text": "左"},
            {"id": "cross", "start_ms": 6500, "end_ms": 7500, "text": "跨界"},
            {"id": "right", "start_ms": 8000, "end_ms": 9000, "text": "右"}]
    args = [frames, [], {"cues": cues, "start_ms": 0, "end_ms": 14000,
             "segments": [{"cue_ids": ["left", "cross", "right"], "summary": "原始摘要"}]}, {}]
    left, right = split_visual_args(args, index)
    assert [cue["id"] for cue in left[2]["cues"]] == ["left", "cross"]
    assert [cue["id"] for cue in right[2]["cues"]] == ["cross", "right"]
    assert left[2]["segments"][0]["cue_ids"] == ["left", "cross"]
    assert right[2]["segments"][0]["cue_ids"] == ["cross", "right"]
    assert left[2]["end_ms"] == right[2]["start_ms"] == 7000
    assert len(right[1]) == 8


def test_fallback_estimate_follows_timestamp_boundaries_instead_of_count_halves():
    frames = [{"id": str(index), "at_ms": 0 if index < 7 else 1000 if index < 14 else 2000}
              for index in range(24)]
    assert visual_fallback_calls(frames) == 5
    assert visual_fallback_calls([{**frame, "at_ms": index} for index, frame in enumerate(frames)]) == 3


@pytest.mark.asyncio
async def test_aggregate_receipt_does_not_release_parallel_sibling_reservation(scene_worker, monkeypatch):
    worker = scene_worker(120)
    original = worker.provider.prepare_visual
    sibling_started, release_sibling, aggregate_saved = asyncio.Event(), asyncio.Event(), asyncio.Event()
    observed = []
    # NTFS publication of 120 frame records and fallback receipts can exceed
    # two seconds on CI. The event and reservation assertions prove ordering.
    io_timeout = 30 if os.name == "nt" else 2

    async def prepare(identity, frames, *args):
        if frames[0]["at_ms"] == 56250:
            sibling_started.set()
            await release_sibling.wait()
        elif len(frames) == 56 and frames[0]["at_ms"] == 250:
            await sibling_started.wait()
            raise ProviderError("合成超时", code="timeout", request_count=1, cost=.001)
        return await original(identity, frames, *args)

    account = worker.queue._account_scene_receipt

    def record(identity, receipt):
        account(identity, receipt)
        if receipt.get("split"):
            state = worker.queue.get(identity)
            observed.append((len(state["inflight_calls"]), state["inflight_attempts"]))
            aggregate_saved.set()

    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    monkeypatch.setattr(worker.queue, "_account_scene_receipt", record)
    job = enqueue(worker, execution={"preparation": "parallel", "max_concurrency": 2})
    running = asyncio.create_task(worker.queue.execute(job["id"]))
    try:
        await asyncio.wait_for(aggregate_saved.wait(), io_timeout)
        assert observed == [(1, 3)]
    finally:
        # Release both fake-provider gates if setup or an assertion failed.
        sibling_started.set()
        release_sibling.set()
        await asyncio.wait_for(running, io_timeout)
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed" and done["request_count"] == 5
    assert done["inflight_calls"] == {} and done["inflight_attempts"] == 0


@pytest.mark.parametrize("large_field", ["summary", "visual_facts"])
@pytest.mark.asyncio
async def test_continuation_keeps_two_chunks_when_merging_would_drop_text_or_facts(
        scene_worker, monkeypatch, large_field):
    worker = scene_worker(28)
    worker.provider.continue_batches = True
    original = worker.provider.prepare_visual

    async def prepare(identity, frames, *args):
        if len(frames) > 14:
            raise ProviderError("合成超时", code="timeout", request_count=1, cost=.001)
        result = await original(identity, frames, *args)
        if large_field == "summary":
            result.data["chunks"][0]["summary"] = "完整合成摘要" * 200
        else:
            result.data["chunks"][0]["visual_facts"] = [
                {"text": f"完整合成画面事实{index}", "frame_ids": [frames[0]["id"]]} for index in range(40)]
        return result

    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    done = await execute(worker, enqueue(worker))
    assert done["status"] == "completed", done
    scenes = worker.store.list(worker.asset["id"])
    assert len(scenes) == 2 and sum(len(scene["frame_ids"]) for scene in scenes) == 28
    if large_field == "summary":
        assert all(scene["summary"] == "完整合成摘要" * 200 for scene in scenes)
    else:
        assert all(len(scene["visual_facts"]) == 40 for scene in scenes)


@pytest.mark.asyncio
async def test_preflight_timeout_with_zero_attempts_does_not_split(scene_worker, monkeypatch):
    worker = scene_worker(56)
    calls = []

    async def prepare(*args):
        calls.append(True)
        raise ProviderError("合成预检超时", code="timeout", request_count=0)

    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    done = await execute(worker, enqueue(worker))
    assert done["status"] == "failed" and done["request_count"] == 0 and len(calls) == 1
