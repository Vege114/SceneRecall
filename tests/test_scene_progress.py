"""Stage progress is rebuilt from durable work without changing paid receipts."""
import copy
import math

import pytest

from scenerecall.jobs import JobQueue
from scenerecall.models import DeepenInput
from scenerecall.providers import ProviderError
from scenerecall.scene_jobs import deep_batch_size
from test_scene_deep_orchestration import select_frames
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


def progress(completed, total, status="completed"):
    return {"completed": completed, "total": total, "status": status}


def capture_calls(worker, monkeypatch, job):
    snapshots = []
    record = worker.provider._record

    async def capture(stage, **inputs):
        snapshots.append((stage, copy.deepcopy(worker.queue.get(job["id"])["stage_progress"])))
        await record(stage, **inputs)

    monkeypatch.setattr(worker.provider, "_record", capture)
    return snapshots


def capture_updates(worker, monkeypatch):
    snapshots = []
    update = worker.queue.update

    def capture(job_id, **changes):
        if "stage_progress" in changes:
            snapshots.append(copy.deepcopy(changes["stage_progress"]))
        return update(job_id, **changes)

    monkeypatch.setattr(worker.queue, "update", capture)
    return snapshots


@pytest.mark.asyncio
async def test_stage_totals_and_completed_batches_follow_actual_pipeline(scene_worker, monkeypatch):
    worker = scene_worker(12)
    worker.provider.chunk_size = 4
    worker.provider.deep_times = {250}
    worker.library.save_subtitles(worker.asset["id"], "dense_subtitles", [
        {"start_ms": i * 50, "end_ms": i * 50 + 30, "text": f"合成台词{i}", "language": "zh"}
        for i in range(205)], 0, 12000)
    job = enqueue(worker)
    snapshots = capture_calls(worker, monkeypatch, job)
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["stage_progress"] == {"transcript": progress(3, 3), "preparation": progress(1, 1),
                                           "deep_analysis": progress(1, 1), "context_refresh": progress(2, 2)}
    transcript = [state for stage, state in snapshots if stage == "transcript"]
    assert [state["transcript"] for state in transcript] == [progress(i, 3, "running") for i in range(3)]
    assert all(state["deep_analysis"] == progress(0, None, "pending") for state in transcript)
    prep = next(state for stage, state in snapshots if stage == "preparation")
    assert prep["transcript"] == progress(3, 3) and prep["preparation"] == progress(0, 1, "running")
    assert prep["context_refresh"] == progress(0, None, "pending")
    deep = next(state for stage, state in snapshots if stage == "deep_analysis")
    assert deep["deep_analysis"] == progress(0, 1, "running")
    assert deep["context_refresh"] == progress(0, 2, "pending")
    refreshed = [state["context_refresh"] for stage, state in snapshots if stage == "context_refresh"]
    assert refreshed == [progress(0, 2, "running"), progress(1, 2, "running")]
    assert counts(worker) == {"transcript": 3, "preparation": 1, "deep_analysis": 1, "context_refresh": 2}


@pytest.mark.parametrize("provider_type", ["codex_cli", "openai_compatible"])
@pytest.mark.asyncio
async def test_deep_stage_total_uses_selected_frames_and_profile_batch_size(scene_worker, monkeypatch, provider_type):
    worker = scene_worker(40)
    worker.provider.deep_times = {250}
    get = worker.provider.get
    monkeypatch.setattr(worker.provider, "get", lambda identity: {
        **get(identity), "provider_type": provider_type if identity == "deep" else "openai_compatible"})
    select_frames(worker, monkeypatch, lambda frames: frames[:16] + [frames[-1]])
    parts = math.ceil(17 / deep_batch_size(worker.provider.get("deep")))
    job = enqueue(worker)
    snapshots = capture_calls(worker, monkeypatch, job)
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["stage_progress"]["deep_analysis"] == progress(parts, parts)
    assert [state["deep_analysis"] for stage, state in snapshots if stage == "deep_analysis"] == [
        progress(index, parts, "running") for index in range(parts)]
    assert completed["stage_progress"]["context_refresh"] == progress(0, 0)


@pytest.mark.asyncio
async def test_pause_resume_rebuilds_cached_preparation_progress_without_accumulating(scene_worker, monkeypatch):
    worker = scene_worker(60)
    paused = await execute(worker, enqueue(worker, max_requests=3))
    assert paused["status"] == "paused", paused
    assert paused["stage_progress"]["preparation"] == progress(1, 2, "running")
    snapshots = capture_updates(worker, monkeypatch)
    worker.queue.control(paused["id"], "resume", {"max_requests": 100})
    completed = await execute(worker, paused)
    assert completed["status"] == "completed", completed
    assert snapshots[0] == paused["stage_progress"]
    assert completed["stage_progress"] == {"transcript": progress(0, 0), "preparation": progress(2, 2),
                                           "deep_analysis": progress(0, 0), "context_refresh": progress(0, 0)}
    assert counts(worker) == {"preparation": 2} and completed["request_count"] == 2
    assert completed["stage_metrics"]["preparation"]["cache_hits"] == 0
    assert all(item["completed"] <= item["total"] for state in snapshots for item in state.values()
               if item["total"] is not None)
    cached = await execute(worker, enqueue(worker))
    assert cached["status"] == "completed" and cached["request_count"] == 0
    assert cached["stage_progress"] == completed["stage_progress"]
    assert counts(worker) == {"preparation": 2}


@pytest.mark.parametrize("recover", ["part", "target"])
@pytest.mark.asyncio
async def test_failed_deep_replays_saved_parts_or_targets_exactly_once(scene_worker, monkeypatch, recover):
    worker = scene_worker(30 if recover == "part" else 8)
    worker.provider.chunk_size = 56 if recover == "part" else 4
    worker.provider.deep_times = {250, 4250}
    worker.provider.fail_deep_at = 24250 if recover == "part" else 4250
    failed = await execute(worker, enqueue(worker))
    assert failed["status"] == "failed", failed
    assert failed["stage_progress"]["deep_analysis"] == progress(1, 2, "running")
    directory = worker.library.asset_dir(worker.asset["id"]) / "analyses" / failed["run_id"]
    assert bool(list((directory / "scene-deep").glob("*.json"))) == (recover == "target")
    snapshots = capture_calls(worker, monkeypatch, failed)
    worker.queue.control(failed["id"], "retry")
    completed = await execute(worker, failed)
    assert completed["status"] == "completed", completed
    assert snapshots[0][1]["deep_analysis"] == progress(1, 2, "running")
    assert completed["stage_progress"]["deep_analysis"] == progress(2, 2)
    assert completed["request_count"] == 4 and counts(worker) == {"preparation": 1, "deep_analysis": 3}
    for folder in ("scene-inputs", "scene-receipts", "scenes"):
        assert all('"stage_progress"' not in path.read_text(encoding="utf-8")
                   for path in (directory / folder).glob("*.json"))


@pytest.mark.asyncio
async def test_receipt_recovered_after_process_exit_rebuilds_progress_without_new_call(scene_worker, monkeypatch):
    worker = scene_worker(8)
    job = enqueue(worker)

    class ProcessExit(BaseException):
        pass

    def interrupt(job_id, receipt):
        raise ProcessExit()

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "_account_scene_receipt", interrupt)
        with pytest.raises(ProcessExit):
            await execute(worker, job)
    assert worker.queue.get(job["id"])["stage_progress"]["preparation"] == progress(0, 1, "running")
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["stage_progress"]["preparation"] == progress(1, 1)
    assert completed["request_count"] == 1 and counts(worker) == {"preparation": 1}


@pytest.mark.asyncio
async def test_failed_context_refresh_replays_prior_receipt_without_double_progress(scene_worker, monkeypatch):
    worker = scene_worker(12)
    worker.provider.chunk_size = 4
    worker.provider.deep_times = {250}
    reconcile = worker.provider.reconcile_scene
    attempts = 0

    async def fail_second(*args):
        nonlocal attempts
        result = await reconcile(*args)
        attempts += 1
        if attempts == 2:
            raise ProviderError("合成上下文失败", request_count=1, cost=.001)
        return result

    monkeypatch.setattr(worker.provider, "reconcile_scene", fail_second)
    failed = await execute(worker, enqueue(worker))
    assert failed["status"] == "failed", failed
    assert failed["stage_progress"]["context_refresh"] == progress(1, 2, "running")
    snapshots = capture_calls(worker, monkeypatch, failed)
    worker.queue.control(failed["id"], "retry")
    completed = await execute(worker, failed)
    assert completed["status"] == "completed", completed
    assert snapshots == [("context_refresh", {"transcript": progress(0, 0), "preparation": progress(1, 1),
                                               "deep_analysis": progress(1, 1),
                                               "context_refresh": progress(1, 2, "running")})]
    assert completed["stage_progress"]["context_refresh"] == progress(2, 2)
    assert counts(worker) == {"preparation": 1, "deep_analysis": 1, "context_refresh": 3}


@pytest.mark.asyncio
async def test_manual_deep_progress_omits_preparation_work_and_survives_budget_pause(scene_worker):
    worker = scene_worker(30)
    await execute(worker, enqueue(worker))
    scene = worker.store.list(worker.asset["id"])[0]
    config, _ = worker.queue.deepen_config(worker.asset["id"], scene["id"], DeepenInput(max_requests=2),
                                         {"bindings": {"vision": "deep"}})
    paused = await execute(worker, worker.queue.add("analysis", config, worker.asset["id"]))
    assert paused["status"] == "paused", paused
    assert paused["stage_progress"] == {"transcript": progress(0, 0), "preparation": progress(0, 0),
                                        "deep_analysis": progress(0, 2, "running"), "context_refresh": progress(0, 0)}
    worker.queue.control(paused["id"], "resume", {"max_requests": 20})
    completed = await execute(worker, paused)
    assert completed["status"] == "completed", completed
    assert completed["stage_progress"]["deep_analysis"] == progress(2, 2)
    assert completed["request_count"] == 2 and completed["stage_progress"]["preparation"] == progress(0, 0)


@pytest.mark.asyncio
async def test_legacy_analysis_does_not_add_scene_progress(scene_worker):
    worker = scene_worker(4)
    completed = await execute(worker, enqueue(worker, analysis_strategy="content-keyframes-v1"))
    assert completed["status"] == "completed", completed
    assert "stage_progress" not in completed
