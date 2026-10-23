"""Cached scene assembly must allow the local HTTP event loop to run."""
import asyncio

import pytest

from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


@pytest.mark.asyncio
async def test_cached_parallel_preparation_yields_between_assembly_batches(scene_worker, monkeypatch):
    worker = scene_worker(112)
    execution = {"transcript": "parallel", "preparation": "parallel", "subtitle": "parallel",
                 "max_concurrency": 2}
    first = await execute(worker, enqueue(worker, execution=execution))
    assert first["status"] == "completed", first
    assert counts(worker) == {"preparation": 2}
    expected = [frame_id for chunk in worker.store.list(worker.asset["id"])
                for frame_id in chunk["frame_ids"]]
    requests_before = list(worker.provider.calls)
    loop = asyncio.get_running_loop()
    serviced = []
    snapshots = []
    update = worker.queue.update

    def observe_assembly(job_id, **changes):
        if changes.get("message", "").startswith("整理已完成的视觉分块："):
            snapshots.append(bool(serviced))
            if len(snapshots) == 1:
                # This callback stands for other ready work, such as a local
                # HTTP request. It must run before the next assembly batch.
                loop.call_soon_threadsafe(serviced.append, True)
        return update(job_id, **changes)

    monkeypatch.setattr(worker.queue, "update", observe_assembly)
    replayed = await execute(worker, enqueue(worker, execution=execution))

    assert replayed["status"] == "completed", replayed
    assert snapshots == [False, True]
    assert replayed["request_count"] == 0
    assert worker.provider.calls == requests_before
    assert replayed["stage_metrics"]["preparation"]["cache_hits"] == 2
    actual = [frame_id for chunk in worker.store.list(worker.asset["id"])
              for frame_id in chunk["frame_ids"]]
    assert actual == expected
    assert len(actual) == len(set(actual)) == worker.frame_count
