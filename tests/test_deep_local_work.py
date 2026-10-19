"""Owned dependency/commit threads keep paid deep parts durable on controls."""
import asyncio
import threading

import pytest

from scenerecall.jobs import JobQueue
from scenerecall.library import read_json
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker  # noqa: F401


async def entered_worker(event):
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 2), 3)


@pytest.mark.asyncio
async def test_scene_call_dependency_projection_does_not_block_loop(scene_worker, monkeypatch):
    worker = scene_worker(2)
    original = worker.queue._check_scene_dependencies
    entered, release = threading.Event(), threading.Event()
    threads = []

    def check(*args):
        threads.append(threading.get_ident())
        if len(threads) == 1:
            entered.set()
            assert release.wait(4)
        return original(*args)

    monkeypatch.setattr(worker.queue, "_check_scene_dependencies", check)
    job = enqueue(worker)
    task = asyncio.create_task(worker.queue.execute(job["id"]))
    try:
        await entered_worker(entered)
        assert threads[0] != threading.get_ident() and not task.done()
        assert worker.queue.get(job["id"])["status"] == "running"
        assert not worker.provider.calls
    finally:
        release.set()
        await task
    assert worker.queue.get(job["id"])["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("point", ["part", "scene"])
@pytest.mark.parametrize("control", ["pause", "shutdown"])
async def test_paid_deep_commit_is_owned_off_loop_and_resume_never_repays(
        scene_worker, monkeypatch, point, control):
    worker = scene_worker(2)
    worker.provider.deep_times = {250}
    original = worker.queue.characters.apply_observation
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    threads = []
    job = enqueue(worker)

    def apply(asset_id, record, revision):
        result = original(asset_id, record, revision)
        target = bool(record.get("input_frame_ids")) if point == "part" else record.get("understanding_level") == "deep"
        if target and not threads:
            threads.append(threading.get_ident())
            if control == "pause":
                worker.queue.control(job["id"], "pause")
            entered.set()
            assert release.wait(4)
            finished.set()
        return result

    monkeypatch.setattr(worker.queue.characters, "apply_observation", apply)
    task = asyncio.create_task(worker.queue.execute(job["id"]))
    try:
        await entered_worker(entered)
        assert threads[0] != threading.get_ident()
        if control == "shutdown":
            task.cancel()
            for _ in range(5):
                await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
        assert counts(worker) == {"preparation": 1, "deep_analysis": 1}
    finally:
        release.set()
    if control == "shutdown":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
        assert worker.queue.get(job["id"])["status"] == "paused"
    assert finished.is_set()
    run = worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"]
    parts = list((run / "scene-deep" / "parts").glob("*/*.json"))
    assert len(parts) == 1 and read_json(parts[0])["input_frame_ids"]
    if point == "scene":
        assert len(list((run / "scene-deep").glob("*.json"))) == 1
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    if control == "pause":
        worker.queue.control(job["id"], "resume")
    done = await execute(worker, job)
    assert done["status"] == "completed", done.get("error")
    assert done["request_count"] == 2 and counts(worker) == {"preparation": 1, "deep_analysis": 1}
    scenes = worker.store.list(worker.asset["id"])
    assert len(scenes) == 1 and scenes[0]["understanding_level"] == "deep"
    assert len(scenes[0]["frame_ids"]) == 2
