"""Durable scene receipts, bounded dialogue and human-corrected memory regressions."""
import json

import pytest

from scenerecall import scene_preparation
from scenerecall.jobs import JobQueue
from scenerecall.library import read_json
from scenerecall.scene_jobs import encoded_size
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


class SyntheticProcessExit(BaseException):
    """Bypass normal task exception handling, as an interrupted process would."""


@pytest.mark.parametrize("interrupted_status", ["running", "pausing", "cancelling"])
@pytest.mark.asyncio
async def test_durable_receipt_recovers_exact_usage_before_resume_without_repaying(scene_worker, monkeypatch,
                                                                               interrupted_status):
    worker = scene_worker(8)
    job = enqueue(worker)

    def exit_before_account(job_id, receipt):
        pending = worker.queue.get(job_id)["pending_scene_receipt"]
        assert read_json(worker.library.asset_dir(worker.asset["id"]) / pending) == receipt
        raise SyntheticProcessExit()

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "_account_scene_receipt", exit_before_account)
        with pytest.raises(SyntheticProcessExit):
            await worker.queue.execute(job["id"])

    interrupted = worker.queue.get(job["id"])
    assert interrupted["request_count"] == 0 and interrupted["inflight_attempts"] == 3
    assert interrupted["pending_scene_receipt"]
    worker.queue.update(job["id"], status=interrupted_status)
    restarted = JobQueue(worker.library, worker.provider, worker.queue.search)
    recovered = restarted.get(job["id"])
    assert recovered["status"] == {"running": "queued", "pausing": "paused", "cancelling": "cancelled"}[interrupted_status]
    assert recovered["request_count"] == 1
    assert recovered.get("uncertain_request_count", 0) == 0
    assert recovered["inflight_attempts"] == 0 and recovered["pending_scene_receipt"] is None
    assert recovered["cost"] == pytest.approx(.001) and not recovered["cost_incomplete"]
    assert recovered["usage"] == {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
    assert len(recovered["scene_receipts"]) == 1
    worker.queue = restarted
    if recovered["status"] != "queued":
        worker.queue.control(job["id"], "resume")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert completed["request_count"] == 1 and len(completed["scene_receipts"]) == 1
    assert completed["stage_metrics"]["preparation"]["request_count"] == 1
    assert completed["stage_metrics"]["preparation"]["cache_hits"] == 1
    assert counts(worker) == {"preparation": 1}


@pytest.mark.asyncio
async def test_prompt_upgrade_rejects_old_paused_job_and_new_run_does_not_reuse_old_receipts(scene_worker, monkeypatch):
    worker = scene_worker(60)
    old = await execute(worker, enqueue(worker, max_requests=3))
    assert old["status"] == "paused" and old["request_count"] == 1, old
    before = scene_preparation.VISUAL_PREPARATION_PROMPT_VERSION
    assert old["config"]["scene_dependencies"]["prompt_versions"]["preparation"] == before
    monkeypatch.setattr(scene_preparation, "VISUAL_PREPARATION_PROMPT_VERSION", before + "-test-upgrade")
    worker.queue.control(old["id"], "resume", {"max_requests": 100})
    refused = await execute(worker, old)
    assert refused["status"] == "failed", refused
    assert refused["request_count"] == 1 and counts(worker) == {"preparation": 1}
    new = await execute(worker, enqueue(worker))
    assert new["status"] == "completed", new
    assert new["request_count"] == 2 and counts(worker) == {"preparation": 3}
    assert new["config"]["scene_dependencies"]["prompt_versions"]["preparation"] == before + "-test-upgrade"


@pytest.mark.asyncio
async def test_long_static_scene_prepares_every_subtitle_while_visual_context_stays_bounded(scene_worker, monkeypatch):
    worker = scene_worker(600)

    def static_scan(path, start, end, checkpoint=None, progress=None):
        shot = {"id": "synthetic_held_scene", "start_ms": start, "end_ms": end}
        frame = {"id": "synthetic_held_frame", "shot_id": shot["id"], "at_ms": start + 250,
                 "pts": start + 250, "time_base": [1, 1000], "source_pts": start + 250,
                 "source_time_base": [1, 1000], "reason": "stable_pose"}
        if checkpoint:
            checkpoint()
        return {"algorithm": "content-keyframes-v1", "shots": [shot], "frames": [frame],
                "batches": [{"id": "synthetic_held_batch", "shot_id": shot["id"], "start_ms": start,
                             "end_ms": end, "frame_ids": [frame["id"]]}]}

    monkeypatch.setattr("scenerecall.media.scan_keyframes", static_scan)
    cues = [{"start_ms": number * 2000, "end_ms": (number + 1) * 2000,
             "text": f"第 {number} 条独立合成台词", "language": "zh", "review_status": "unreviewed"}
            for number in range(300)]
    worker.library.save_subtitles(worker.asset["id"], "synthetic_long_transcript", cues, 0, 600000)
    expected = [cue["id"] for cue in worker.library.subtitles(worker.asset["id"])]
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    transcript_calls = [call for call in worker.provider.calls if call["stage"] == "transcript"]
    assert len(transcript_calls) == 3 and all(len(call["cues"]) == 100 for call in transcript_calls)
    assert [cue["id"] for call in transcript_calls for cue in call["cues"]] == expected
    visual = next(call for call in worker.provider.calls if call["stage"] == "preparation")
    context = visual["subtitle_context"]
    assert len(visual["frames"]) == 1 and len(context["cues"]) <= 100
    assert context["omitted_cues"] == 300 - len(context["cues"])
    assert encoded_size(context) < 24000
    assert (context["start_ms"], context["end_ms"]) == (0, 600000)
    scene = worker.store.list(worker.asset["id"])[0]
    assert scene["subtitle_ids"] == expected
    assert {cue_id for fact in scene["dialogue_facts"] for cue_id in fact["cue_ids"]} == set(expected)
    assert len(worker.library.subtitles(worker.asset["id"])) == 300
    assert counts(worker) == {"transcript": 3, "preparation": 1}


@pytest.mark.asyncio
async def test_human_scene_revision_projects_into_later_memory_without_rewriting_source(scene_worker):
    worker = scene_worker(8)
    first = await execute(worker, enqueue(worker, end_ms=4000))
    assert first["status"] == "completed", first
    source = worker.store.list(worker.asset["id"])[0]
    source_path = worker.library.asset_dir(worker.asset["id"]) / worker.library.active(worker.asset["id"])["scenes"][source["id"]]
    original = read_json(source_path)
    before = worker.store.prior_memory(worker.asset["id"], 4000)
    corrected = "人工核实：这一段角色在等朋友，尚未离开。"
    worker.library.annotate(worker.asset["id"], {"record_id": source["id"], "summary": corrected})
    projected = worker.store.prior_memory(worker.asset["id"], 4000)
    assert projected["digest"] != before["digest"]
    assert projected["summaries"][0]["summary"] == corrected
    assert read_json(source_path) == original
    later = await execute(worker, enqueue(worker, start_ms=4000, end_ms=8000))
    assert later["status"] == "completed", later
    subsequent_input = [call for call in worker.provider.calls if call["stage"] == "preparation"][-1]["story_context"]
    assert corrected in json.dumps(subsequent_input, ensure_ascii=False)
    scenes = worker.store.list(worker.asset["id"])
    assert len(scenes) == 2 and all(not scene["stale"] for scene in scenes)
    assert scenes[0]["summary"] == corrected
    worker.library.annotate(worker.asset["id"], {"record_id": source["id"], "note": "仅供我收藏时参考", "favorite": True})
    after_note = worker.store.prior_memory(worker.asset["id"], 4000)
    assert after_note["digest"] == projected["digest"]
    assert all(not scene["stale"] for scene in worker.store.list(worker.asset["id"]))


@pytest.mark.parametrize("requested_stages", [["vision"], ["vision", "subtitle"]])
@pytest.mark.asyncio
async def test_queued_partial_range_is_rechecked_before_ocr_or_visual_requests(scene_worker, requested_stages):
    worker = scene_worker(8)
    full = enqueue(worker)
    partial = enqueue(worker, start_ms=0, end_ms=4000, stages=requested_stages)
    finished = await execute(worker, full)
    assert finished["status"] == "completed", finished
    existing = worker.store.list(worker.asset["id"])
    assert len(existing) == 1 and existing[0]["end_ms"] == 8000
    prior_calls = len(worker.provider.calls)
    rejected = await execute(worker, partial)
    assert rejected["status"] == "failed" and "0–8 秒" in rejected["error"], rejected
    assert rejected["request_count"] == 0
    assert len(worker.provider.calls) == prior_calls
    assert worker.store.list(worker.asset["id"]) == existing
