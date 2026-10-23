"""Source duration is a performance target, never a task execution deadline."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scenerecall.retrieval_store import RetrievalStore
from test_retrieval_complete import complete_worker  # noqa: F401
from test_retrieval_jobs import enqueue
from test_scene_jobs import scene_worker  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["core", "complete"])
async def test_retrieval_finishes_when_processing_exceeds_source_duration(
        complete_worker, monkeypatch, mode):  # noqa: F811
    from scenerecall import jobs, keyframes, retrieval_jobs

    worker = complete_worker(frame_count=12)
    source_duration_ms = worker.asset["duration_ms"]
    elapsed_seconds = 0.0
    origin = datetime.now(timezone.utc)

    def monotonic():
        return elapsed_seconds

    def now():
        return (origin + timedelta(seconds=elapsed_seconds)).isoformat()

    # Replace only the application modules' clocks. In particular, asyncio and
    # the thread pool keep their real clocks and no real long sleep is needed.
    clock = SimpleNamespace(monotonic=monotonic, perf_counter=monotonic)
    monkeypatch.setattr(jobs, "time", clock)
    monkeypatch.setattr(jobs, "now", now)
    monkeypatch.setattr(retrieval_jobs, "time", clock)
    monkeypatch.setattr(retrieval_jobs, "now", now)
    scan = keyframes.scan_keyframes
    encode = worker.queue.search.visual_encoder.encode_images

    def slow_scan(*args, **kwargs):
        nonlocal elapsed_seconds
        elapsed_seconds += source_duration_ms / 1000 * 10
        # The synthetic scanner still invokes production task checkpoints.
        return scan(*args, **kwargs)

    async def slow_encode(items):
        nonlocal elapsed_seconds
        elapsed_seconds += source_duration_ms / 1000 * 10
        return await encode(items)

    monkeypatch.setattr(keyframes, "scan_keyframes", slow_scan)
    monkeypatch.setattr(worker.queue.search.visual_encoder, "encode_images", slow_encode)
    job = enqueue(worker, mode=mode)
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])

    assert done["status"] == "completed", done
    assert done["error"] is None and done["failures"] == []
    for stage in ("retrieval_sampling", "visual_index"):
        assert done["stage_metrics"][stage]["elapsed_ms"] >= source_duration_ms * 10
    wall_elapsed = datetime.fromisoformat(done["updated_at"]) - datetime.fromisoformat(done["created_at"])
    wall_ms = wall_elapsed.total_seconds() * 1000
    assert wall_ms >= source_duration_ms * 20
    assert done["retrieval"]["status"] == ("completed" if mode == "complete" else "core_ready")
    assert bool(done["retrieval"]["default_completed_at"]) == (mode == "complete")
    assert done["request_count"] == (1 if mode == "complete" else 0)
    assert (await worker.queue.search.search("蓝色天空"))["results"]
    RetrievalStore(worker.library).validate_all()
