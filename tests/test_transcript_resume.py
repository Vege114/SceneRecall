"""Transcript checkpoints restore all completed units before missing dispatch."""
import asyncio
import copy
import os

import pytest

from scenerecall.jobs import JobQueue
from scenerecall.library import atomic_json, digest, read_json
from scenerecall.providers import ProviderError
from scenerecall.scene_preparation import SUBTITLE_PREPARATION_PROMPT_VERSION
from test_parallel_execution import dense_dialogue, options
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


def checkpoint_path(worker, job):
    return worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"] / "scene-transcript.json"


def reject_finished_units(worker, monkeypatch, job, units):
    original = worker.queue._scene_call
    resumed = []

    async def observe(job_id, stage, unit, *args, **kwargs):
        if stage == "transcript":
            assert unit not in units, "completed transcript unit entered the model admission path"
            resumed.append((unit, copy.deepcopy(worker.queue.get(job_id)["stage_progress"]["transcript"])))
        return await original(job_id, stage, unit, *args, **kwargs)

    monkeypatch.setattr(worker.queue, "_scene_call", observe)
    return resumed


@pytest.mark.asyncio
async def test_serial_transcript_restores_partial_results_before_next_call(scene_worker, monkeypatch):
    worker = scene_worker(12)
    dense_dialogue(worker)
    paused = await execute(worker, enqueue(worker, max_requests=3))
    assert paused["status"] == "paused", paused
    saved = read_json(checkpoint_path(worker, paused))
    assert [result is not None for result in saved["results"]] == [True, False, False]
    resumed = reject_finished_units(worker, monkeypatch, paused, {"0"})
    worker.queue.control(paused["id"], "resume", {"max_requests": 100})
    completed = await execute(worker, paused)
    assert completed["status"] == "completed", completed
    assert resumed[0] == ("1", {"completed": 1, "total": 3, "status": "running"})
    assert [unit for unit, _ in resumed] == ["1", "2"]
    assert counts(worker) == {"transcript": 3, "preparation": 1}
    assert completed["stage_metrics"]["transcript"]["cache_hits"] == 0
    assert read_json(checkpoint_path(worker, paused))["complete"]


@pytest.mark.parametrize("action", ["pause", "cancel"])
@pytest.mark.asyncio
async def test_parallel_transcript_drains_and_only_dispatches_missing_batches(scene_worker, monkeypatch, action):
    worker = scene_worker(20)
    dense_dialogue(worker, 305)
    job = enqueue(worker, execution=options())
    original = worker.provider.prepare_subtitles
    started, release = asyncio.Event(), asyncio.Event()
    entered = 0
    # Admission includes durable frame and transcript preparation on Windows;
    # the gate proves overlap without imposing a disk-throughput requirement.
    io_timeout = 30 if os.name == "nt" else 2

    async def blocked(*args):
        nonlocal entered
        entered += 1
        if entered == 2:
            started.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(worker.provider, "prepare_subtitles", blocked)
    running = asyncio.create_task(worker.queue.execute(job["id"]))
    try:
        await asyncio.wait_for(started.wait(), io_timeout)
        worker.queue.control(job["id"], action)
    finally:
        release.set()
        await asyncio.wait_for(running, io_timeout)
    stopped = worker.queue.get(job["id"])
    assert stopped["status"] == {"pause": "paused", "cancel": "cancelled"}[action]
    saved = read_json(checkpoint_path(worker, stopped))
    assert [result is not None for result in saved["results"]] == [True, True, False, False]
    resumed = reject_finished_units(worker, monkeypatch, stopped, {"0", "1"})
    worker.queue.control(job["id"], "resume")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert resumed[0] == ("2", {"completed": 2, "total": 4, "status": "running"})
    assert entered == 4 and counts(worker) == {"transcript": 4, "preparation": 1}
    assert completed["inflight_calls"] == {} and completed["request_count"] == 5


@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.asyncio
async def test_complete_transcript_skips_stage_and_preserves_story_semantics(scene_worker, monkeypatch, mode):
    worker = scene_worker(12)
    dense_dialogue(worker)
    original = worker.provider.prepare_subtitles
    initial_summaries = []

    async def distinct(identity, cues, story):
        result = await original(identity, cues, story)
        summary = "本批摘要 " + cues[0]["id"]
        result.data["segments"][0]["summary"] = summary
        result.data["unresolved"] = ["本批悬念 " + cues[0]["id"]]
        initial_summaries.append(summary)
        return result

    monkeypatch.setattr(worker.provider, "prepare_subtitles", distinct)
    visual = worker.provider.prepare_visual
    failed_once = False

    async def fail_preparation(*args):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise ProviderError("合成分块失败", request_count=1, cost=.001)
        return await visual(*args)

    monkeypatch.setattr(worker.provider, "prepare_visual", fail_preparation)
    failed = await execute(worker, enqueue(worker, execution=options(transcript=mode, preparation="serial")))
    assert failed["status"] == "failed", failed
    saved = read_json(checkpoint_path(worker, failed))
    assert saved["complete"]
    before_metrics = copy.deepcopy(failed["stage_metrics"]["transcript"])
    resumed = reject_finished_units(worker, monkeypatch, failed, {"0", "1", "2"})
    worker.queue.control(failed["id"], "retry")
    completed = await execute(worker, failed)
    assert completed["status"] == "completed", completed
    assert resumed == []
    assert completed["stage_metrics"]["transcript"] == before_metrics
    segments, unresolved = worker.queue._transcript_outputs(saved["results"], mode == "parallel")
    assert [segment["summary"] for segment in segments] == initial_summaries
    assert unresolved == ([entry for result in saved["results"] for entry in result["unresolved"]]
                          if mode == "parallel" else saved["results"][-1]["unresolved"])
    stories = [call["story_context"] for call in worker.provider.calls if call["stage"] == "transcript"]
    if mode == "serial":
        assert stories[1]["previous_segments"] == saved["results"][0]["segments"]
        assert stories[2]["unresolved"] == saved["results"][1]["unresolved"]
    else:
        assert all(story["previous_segments"] == [] and story["unresolved"] == [] for story in stories)


@pytest.mark.asyncio
async def test_old_transcript_receipts_migrate_without_call_replay(scene_worker, monkeypatch):
    worker = scene_worker(12)
    dense_dialogue(worker)
    paused = await execute(worker, enqueue(worker, max_requests=3))
    assert paused["status"] == "paused", paused
    checkpoint_path(worker, paused).unlink()
    resumed = reject_finished_units(worker, monkeypatch, paused, {"0"})
    worker.queue.control(paused["id"], "resume", {"max_requests": 100})
    completed = await execute(worker, paused)
    assert completed["status"] == "completed", completed
    assert resumed[0][1]["completed"] == 1
    assert counts(worker) == {"transcript": 3, "preparation": 1}
    assert completed["stage_metrics"]["transcript"]["cache_hits"] == 1


@pytest.mark.asyncio
async def test_transcript_receipt_after_process_exit_is_accounted_and_restored_once(scene_worker, monkeypatch):
    worker = scene_worker(12)
    dense_dialogue(worker)
    job = enqueue(worker)

    class ProcessExit(BaseException):
        pass

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "_account_scene_receipt", lambda *args: (_ for _ in ()).throw(ProcessExit()))
        with pytest.raises(ProcessExit):
            await execute(worker, job)
    assert worker.queue.get(job["id"])["request_count"] == 0
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    assert worker.queue.get(job["id"])["request_count"] == 1
    resumed = reject_finished_units(worker, monkeypatch, job, {"0"})
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert resumed[0][1]["completed"] == 1
    assert completed["request_count"] == 4 and counts(worker) == {"transcript": 3, "preparation": 1}


@pytest.mark.parametrize("corruption", ["profile", "batch", "frozen"])
@pytest.mark.asyncio
async def test_invalid_transcript_checkpoint_or_frozen_input_cannot_spend(scene_worker, corruption):
    worker = scene_worker(12)
    dense_dialogue(worker)
    paused = await execute(worker, enqueue(worker, max_requests=3))
    assert paused["status"] == "paused", paused
    path = checkpoint_path(worker, paused)
    if corruption == "frozen":
        path.unlink()
        path = path.parent / "scene-inputs" / f"{digest(['transcript', SUBTITLE_PREPARATION_PROMPT_VERSION, '0'])}.json"
        saved = read_json(path)
        saved["args"][1]["prior"] = {"changed": True}
    else:
        saved = read_json(path)
        if corruption == "profile":
            saved["header"]["profile"]["model"] = "unrelated-model"
        else:
            saved["results"][0]["segments"][0]["cue_ids"].pop()
    atomic_json(path, saved)
    before = list(worker.provider.calls)
    worker.queue.control(paused["id"], "resume", {"max_requests": 100})
    refused = await execute(worker, paused)
    assert refused["status"] == "failed", refused
    assert worker.provider.calls == before
