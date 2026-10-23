"""Local scene preparation stays responsive without changing paid inputs."""
import asyncio
import json
import threading
from copy import deepcopy
from pathlib import Path

import pytest

from scenerecall import providers as provider_module
from scenerecall.execution import ReservationBusy
from scenerecall.jobs import StopJob
from scenerecall.library import atomic_json
from scenerecall.providers import AIResult, ProviderError
from scenerecall.scene_jobs import SceneJobMixin
from test_identity_scenes import region
from test_scene_jobs import enqueue
from test_scene_jobs import scene_worker as scene_worker  # noqa: F401
from test_scene_providers import chunk, images, manager, reply


async def worker_entered(event):
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 2), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["status", "reservation", "receipt", "cache", "split", "progress"])
async def test_scene_call_lock_contention_keeps_request_loop_responsive(scene_worker, monkeypatch, operation):
    worker = scene_worker(60 if operation == "progress" else 2)
    if operation == "cache":
        initial = enqueue(worker)
        await worker.queue.execute(initial["id"])
        assert worker.queue.get(initial["id"])["status"] == "completed"
    if operation == "split":
        prepare = worker.provider.prepare_visual

        async def split_large(identity, frames, *args):
            if len(frames) > 1:
                raise ProviderError("Synthetic context overflow", code="character_context_too_large", request_count=0)
            return await prepare(identity, frames, *args)

        monkeypatch.setattr(worker.provider, "prepare_visual", split_large)
    name = {"status": "update", "reservation": "before_call", "receipt": "_account_scene_receipt",
            "cache": "_scene_metric", "split": "update", "progress": "_scene_progress"}[operation]
    original = getattr(worker.queue, name)
    entered, proceed, attempting = threading.Event(), threading.Event(), threading.Event()
    locked, release = threading.Event(), threading.Event()
    calling_threads = []

    def guarded(*args, **kwargs):
        selected = (kwargs.get("pending_scene_receipt") is not None if operation == "status" else
                    "细分" in kwargs.get("message", "") if operation == "split" else
                    args[1] == "preparation" and kwargs.get("completed", 0) > 0 if operation == "progress" else
                    kwargs.get("cache") == 1 if operation == "cache" else True)
        if selected and not calling_threads:
            calling_threads.append(threading.get_ident())
            entered.set()
            assert proceed.wait(4), "scene metadata blocked the request event loop"
            attempting.set()
        return original(*args, **kwargs)

    def hold_library_lock():
        with worker.library.lock:
            locked.set()
            assert release.wait(4), "request loop could not release the contended library lock"

    monkeypatch.setattr(worker.queue, name, guarded)
    job = enqueue(worker, **({"execution": {"preparation": "parallel"}} if operation == "progress" else {}))
    task = asyncio.create_task(worker.queue.execute(job["id"]))
    holder = None
    try:
        await worker_entered(entered)
        assert len(calling_threads) == 1 and calling_threads[0] != threading.get_ident()
        holder = asyncio.create_task(asyncio.to_thread(hold_library_lock))
        await worker_entered(locked)
        proceed.set()
        await worker_entered(attempting)
        # A request-like loop operation still executes while the scene job is
        # waiting on the same lock as local character/identity work.
        assert worker.queue.get(job["id"])["status"] == "running"
        assert not task.done()
    finally:
        proceed.set()
        release.set()
        if holder is not None:
            await holder
        await task
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done.get("error")
    assert done.get("inflight_calls", {}) == {}
    assert done["request_count"] == (0 if operation == "cache" else 2 if operation in {"split", "progress"} else 1)


@pytest.mark.asyncio
async def test_cancel_during_reservation_drains_worker_and_releases_unstarted_call(scene_worker, monkeypatch):
    worker = scene_worker(2)
    job = enqueue(worker)
    original = worker.queue.before_call
    entered, release = threading.Event(), threading.Event()

    def reserve(*args, **kwargs):
        call_id = original(*args, **kwargs)
        entered.set()
        assert release.wait(4)
        return call_id

    monkeypatch.setattr(worker.queue, "before_call", reserve)
    task = asyncio.create_task(worker.queue.reserve_call(job["id"], worker.provider.get("fast")))
    try:
        await worker_entered(entered)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert len(worker.queue.get(job["id"])["inflight_calls"]) == 1
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    state = worker.queue.get(job["id"])
    assert state["inflight_calls"] == {} and state["inflight_attempts"] == 0
    assert state["request_count"] == 0 and not worker.provider.calls


@pytest.mark.asyncio
async def test_worker_accounting_wakes_budget_waiter_on_its_event_loop(scene_worker, monkeypatch):
    worker = scene_worker(2)
    job = enqueue(worker, max_requests=3)
    profile = worker.provider.get("fast")
    first = await worker.queue.reserve_call(job["id"], profile)
    original = worker.queue.before_call
    waiting = threading.Event()

    def reserve(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ReservationBusy:
            waiting.set()
            raise

    monkeypatch.setattr(worker.queue, "before_call", reserve)
    loop = asyncio.get_running_loop()
    previous_debug = loop.get_debug()
    loop.set_debug(True)
    waiter = asyncio.create_task(worker.queue.reserve_call(job["id"], profile))
    try:
        await worker_entered(waiting)
        await asyncio.to_thread(worker.queue.account, job["id"], AIResult({}, {}, 1, 0), first)
        # Once the first call spent one request, this tight budget can no
        # longer cover three retries. The awakened waiter must pause normally.
        with pytest.raises(StopJob):
            await asyncio.wait_for(waiter, 3)
    finally:
        if not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        loop.set_debug(previous_debug)
    state = worker.queue.get(job["id"])
    assert state["status"] == "paused" and state["request_count"] == 1
    assert state["inflight_calls"] == {}


@pytest.mark.asyncio
async def test_late_busy_result_keeps_wakeup_when_another_waiter_exhausts_budget(scene_worker, monkeypatch):
    worker = scene_worker(2)
    job = enqueue(worker, max_requests=3)
    profile = worker.provider.get("fast")
    first = await worker.queue.reserve_call(job["id"], profile)
    original = worker.queue.before_call
    busy, deliver_busy = threading.Event(), threading.Event()

    def delayed_busy(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ReservationBusy:
            busy.set()
            assert deliver_busy.wait(4)
            raise

    monkeypatch.setattr(worker.queue, "before_call", delayed_busy)
    waiter = asyncio.create_task(worker.queue.reserve_call(job["id"], profile))
    try:
        await worker_entered(busy)
        generation = worker.queue._reservation_events[job["id"]]
        await asyncio.to_thread(worker.queue.account, job["id"], AIResult({}, {}, 1, 0), first)
        await asyncio.wait_for(generation.wait(), 3)
        # A new admission discovers the permanently exhausted budget while
        # the first worker's old ReservationBusy result is still in transit.
        with pytest.raises(StopJob):
            await worker.queue.reserve_call(job["id"], profile)
        deliver_busy.set()
        with pytest.raises(StopJob):
            await asyncio.wait_for(waiter, 3)
    finally:
        deliver_busy.set()
        if not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
    assert worker.queue.get(job["id"])["inflight_calls"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["serial", "parallel"])
async def test_preparation_context_and_refresh_run_off_loop(scene_worker, monkeypatch, mode):
    worker = scene_worker(2)
    job = enqueue(worker, execution={"preparation": mode, "max_concurrency": 2})
    loop_thread = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = worker.queue.scene_identity_context

    def context(*args):
        calls.append(threading.get_ident())
        if len(calls) == 1:
            entered.set()
            assert release.wait(4), "event loop did not release local work"
        return original(*args)

    monkeypatch.setattr(worker.queue, "scene_identity_context", context)
    task = asyncio.create_task(worker.queue.execute(job["id"]))
    try:
        await worker_entered(entered)
        # This request-like loop operation can run while projection is blocked.
        assert worker.queue.get(job["id"])["status"] == "running"
        assert not task.done() and not worker.provider.calls
    finally:
        release.set()
        await task
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done.get("error")
    assert len(calls) == 2  # Initial context and fresh-request human mapping check.
    assert all(identity != loop_thread for identity in calls)
    assert done["request_count"] == 1


@pytest.mark.asyncio
async def test_offloaded_projection_keeps_bounded_candidates_and_inputs(scene_worker, monkeypatch):
    worker = scene_worker(2)
    job = enqueue(worker)
    value = {"status": "completed", "regions": [region(), region("r2", "f2", "c23")],
             "selected_profiles": [{"id": "c19", "name": "先验角色"}, {"id": "c23", "name": "另帧角色"}],
             "references": [{"character_id": "c19", "reference_id": "sample", "path": "/reference.png"}]}
    monkeypatch.setattr("scenerecall.visual_identity.VisualIdentityPipeline.project_result",
                        lambda self, asset_id, saved: deepcopy(saved))
    path = worker.queue._identity_path(job)
    atomic_json(path, {"visual": value})
    before_file = path.read_bytes()
    context = {"profiles": [{"id": "c19", "name": "人工编辑名称"}], "reference_frames": []}
    before_context = deepcopy(context)
    args = [job["id"], context, [{"id": "f1"}]]
    expected = worker.queue.scene_identity_context(*args)
    actual = await worker.queue._scene_local_context(job["id"], worker.queue.scene_identity_context, *args)
    assert actual == expected
    assert [row["id"] for row in actual["identity_regions"]] == ["r1"]
    assert actual["identity_regions"][0]["match"]["status"] == "candidate"
    assert actual["profiles"][0]["name"] == "人工编辑名称"
    assert context == before_context and path.read_bytes() == before_file


@pytest.mark.asyncio
async def test_pause_during_local_projection_drains_then_prevents_model_call(scene_worker, monkeypatch):
    worker = scene_worker(2)
    job = enqueue(worker)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = worker.queue.scene_identity_context

    def context(*args):
        entered.set()
        assert release.wait(4)
        try:
            return original(*args)
        finally:
            finished.set()

    monkeypatch.setattr(worker.queue, "scene_identity_context", context)
    task = asyncio.create_task(worker.queue.execute(job["id"]))
    try:
        await worker_entered(entered)
        worker.queue.control(job["id"], "pause")
        assert not task.done() and not finished.is_set()
    finally:
        release.set()
        await task
    assert finished.is_set()
    assert worker.queue.get(job["id"])["status"] == "paused"
    assert not worker.provider.calls


@pytest.mark.asyncio
async def test_shutdown_cancellation_waits_for_owned_context_thread():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    checkpoints = []

    class LocalQueue(SceneJobMixin):
        def checkpoint(self, identity):
            checkpoints.append(identity)

    def local():
        entered.set()
        try:
            assert release.wait(4)
            return {"same": "result"}
        finally:
            finished.set()

    task = asyncio.create_task(LocalQueue()._scene_local_context("fixture", local))
    await worker_entered(entered)
    task.cancel()
    # Poll event-loop scheduling without sleeping or ending ownership early.
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done() and not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set() and checkpoints == ["fixture"]


@pytest.mark.asyncio
async def test_visual_content_matches_inline_messages_including_image_bytes(tmp_path, monkeypatch):
    frames = images(tmp_path, 3)
    requests = []
    response = {"chunks": [chunk(["frame-1", "frame-2"])], "unresolved": []}
    providers = manager(tmp_path, lambda request: requests.append(json.loads(request.content)) or reply(response))
    context = {"profiles": [{"id": "c19", "name": "先验"}],
               "reference_frames": [{**frames[0], "character_id": "c19", "box": [0, 0, .5, 1]}],
               "identity_regions": [region(frame="frame-1")],
               "region_images": [{**frames[1], "local_id": "P1", "frame_id": "frame-1"}]}

    async def inline(method, *args, **kwargs):
        return method(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(provider_module.asyncio, "to_thread", inline)
        await providers.prepare_visual("scene", frames[1:], frames[:1], None, {"summary": "先前"}, context)
    await providers.prepare_visual("scene", frames[1:], frames[:1], None, {"summary": "先前"}, context)
    assert requests[0]["messages"] == requests[1]["messages"]
    parts = requests[1]["messages"][1]["content"]
    assert len([part for part in parts if part["type"] == "image_url"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_slow_image_build_yields_and_owns_temporary_directory(tmp_path, monkeypatch, cancel):
    frames = images(tmp_path, 1)
    requests = []
    response = {"chunks": [chunk(["frame-0"])], "unresolved": []}
    providers = manager(tmp_path, lambda request: requests.append(request) or reply(response))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    directories, worker_threads = [], []
    original = provider_module.contact_sheets

    def sheets(frames, context, directory):
        directories.append(Path(directory))
        worker_threads.append(threading.get_ident())
        assert Path(directory).exists()
        entered.set()
        try:
            assert release.wait(4)
            return original(frames, context, directory)
        finally:
            finished.set()

    monkeypatch.setattr(provider_module, "contact_sheets", sheets)
    task = asyncio.create_task(providers.prepare_visual("scene", frames, [], None, None))
    await worker_entered(entered)
    try:
        assert len(worker_threads) == 1 and worker_threads[0] != threading.get_ident()
        assert not task.done() and not requests and directories[0].exists()
        if cancel:
            task.cancel()
            for _ in range(5):
                await asyncio.sleep(0)
            assert not task.done() and directories[0].exists()
    finally:
        release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not requests
    else:
        result = await task
        assert result.data == response and len(requests) == 1
    assert finished.is_set() and not directories[0].exists()
