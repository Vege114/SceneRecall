"""Local visual stage visibility during real synthetic queue prefetch/resume."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scenerecall.audio_identity import AudioIdentityService
from scenerecall.visual_identity import VisualIdentityPipeline
from test_scene_jobs import counts, enqueue, set_subtitles
from test_scene_jobs import scene_worker as scene_worker  # noqa: F401


@pytest.mark.asyncio
async def test_prefetched_transcript_stays_complete_during_throttled_visual_prelude(scene_worker, monkeypatch):
    worker = scene_worker(32, with_subtitles=True)
    VisualIdentityPipeline(worker.library).configure(worker.asset["id"], {"enabled": True})
    job = enqueue(worker)
    await worker.queue.execute(job["id"], prepare_only=True)
    prefetched = worker.queue.get(job["id"])
    assert prefetched["transcript_prefetched"] and not prefetched["config"].get("scene_dependencies")
    assert prefetched["stage_progress"]["transcript"] == {"completed": 1, "total": 1, "status": "completed"}
    seen, frame_writes = [], []
    update = worker.queue.update

    def tracked(job_id, **changes):
        if changes.get("stage") == "visual_identity":
            frame_writes.append(changes["completed"])
        return update(job_id, **changes)

    def visual(_self, asset_id, frames, **kwargs):
        current = deepcopy(worker.queue.get(job["id"]))
        seen.append(current)
        assert current["stage"] == "visual_identity"
        assert "本机人物检测与 CCIP 匹配" in current["message"]
        assert current["total"] == len(frames) and current["completed"] == 0
        assert current["stage_progress"]["transcript"] == prefetched["stage_progress"]["transcript"]
        for completed in range(1, len(frames) + 1):
            kwargs["checkpoint"]()
            kwargs["progress"](completed, len(frames))
        return {"status": "completed", "regions": [], "references": [], "selected_profiles": []}

    monkeypatch.setattr(worker.queue, "update", tracked)
    monkeypatch.setattr("scenerecall.identity_jobs.time", SimpleNamespace(monotonic=lambda: 0))
    monkeypatch.setattr(VisualIdentityPipeline, "run", visual)
    monkeypatch.setattr(AudioIdentityService, "run", lambda *args, **kwargs: {"status": "disabled", "records": []})
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done.get("error")
    assert len(seen) == 1 and frame_writes == [0, 32]
    assert done["stage_progress"]["visual_identity"] == {"completed": 32, "total": 32, "status": "completed"}
    assert done["stage_progress"]["transcript"] == prefetched["stage_progress"]["transcript"]
    assert counts(worker)["transcript"] == 1


@pytest.mark.asyncio
async def test_changed_prefetched_dialogue_is_rejected_before_visual_processing(scene_worker, monkeypatch):
    worker = scene_worker(2, with_subtitles=True)
    job = enqueue(worker)
    await worker.queue.execute(job["id"], prepare_only=True)
    set_subtitles(worker, "依赖已经变化的新字幕")
    calls = []
    monkeypatch.setattr(VisualIdentityPipeline, "run", lambda *args, **kwargs: calls.append(args))
    await worker.queue.execute(job["id"])
    failed = worker.queue.get(job["id"])
    assert failed["status"] == "failed" and "台词或来源已改变" in failed["error"]
    assert calls == [] and counts(worker)["transcript"] == 1


@pytest.mark.asyncio
async def test_visual_failure_does_not_claim_unprocessed_frames_complete(scene_worker, monkeypatch):
    worker = scene_worker(3, with_subtitles=True)
    VisualIdentityPipeline(worker.library).configure(worker.asset["id"], {"enabled": True})
    job = enqueue(worker)
    clock = [0]
    audio_drained = []

    def visual(_self, asset_id, frames, **kwargs):
        clock[0] = 2
        kwargs["progress"](1, len(frames))
        raise ValueError("合成人物帧处理失败")

    def audio(*args, **kwargs):
        audio_drained.append(True)
        return {"status": "disabled", "records": []}

    monkeypatch.setattr("scenerecall.identity_jobs.time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(VisualIdentityPipeline, "run", visual)
    monkeypatch.setattr(AudioIdentityService, "run", audio)
    await worker.queue.execute(job["id"])
    failed = worker.queue.get(job["id"])
    assert failed["status"] == "failed" and "合成人物帧处理失败" in failed["error"]
    assert failed["stage_progress"]["visual_identity"] == {"completed": 1, "total": 3, "status": "running"}
    assert failed["stage_progress"]["transcript"]["status"] == "completed"
    assert audio_drained == [True]
