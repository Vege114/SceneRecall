"""Immediate cache hits must leave the service loop responsive and owned."""
import asyncio
import threading

import pytest

from scenerecall import execution


@pytest.mark.asyncio
async def test_failure_during_async_checkpoint_prevents_admission_and_drains_first_unit():
    checking, release_check, failure_signalled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_call = asyncio.Event()
    checks, calls = [], []
    error = RuntimeError("failure during control check")

    async def checkpoint():
        checks.append(True)
        if len(checks) == 2:
            checking.set()
            await release_check.wait()

    async def operation(index, value):
        calls.append(index)
        await checking.wait()
        execution.stop_admission(error)
        failure_signalled.set()
        await release_call.wait()
        raise error

    task = asyncio.create_task(execution.bounded_map(range(3), operation, 2, checkpoint))
    try:
        await asyncio.wait_for(failure_signalled.wait(), 3)
        release_check.set()
        await asyncio.sleep(.01)
        assert calls == [0] and not task.done()
    finally:
        release_check.set()
        release_call.set()
        with pytest.raises(RuntimeError) as caught:
            await task
    assert caught.value is error


@pytest.mark.asyncio
async def test_terminal_failure_stops_admission_while_accounting_drains():
    sibling_started = asyncio.Event()
    failure_signalled = asyncio.Event()
    sibling_finished = asyncio.Event()
    accounting_started, release_accounting = threading.Event(), threading.Event()
    started = []
    error = RuntimeError("original provider failure")

    def account():
        accounting_started.set()
        assert release_accounting.wait(4)

    async def operation(index, value):
        started.append(index)
        if index == 0:
            await sibling_started.wait()
            execution.stop_admission(error)
            failure_signalled.set()
            await execution.run_in_thread(account)
            raise error
        sibling_started.set()
        await failure_signalled.wait()
        sibling_finished.set()
        return value

    task = asyncio.create_task(execution.bounded_map(range(3), operation, 2, lambda: None))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(accounting_started.wait, 2), 3)
        await asyncio.wait_for(sibling_finished.wait(), 3)
        # Give the scheduler a chance to observe the completed sibling. It
        # must drain the failed unit, not fill the available slot with unit 3.
        await asyncio.sleep(.01)
        assert started == [0, 1] and not task.done()
    finally:
        release_accounting.set()
        with pytest.raises(RuntimeError) as caught:
            await task
    assert caught.value is error

    async def normal(index, value):
        return value

    # The failure belongs only to its original batch; retry/new work proceeds.
    assert await execution.bounded_map(range(3), normal, 2, lambda: None) == [0, 1, 2]


@pytest.mark.asyncio
async def test_immediate_cache_operations_allow_heartbeat_between_admissions():
    events = []
    loop = asyncio.get_running_loop()

    async def cached(index, value):
        events.append(("cached", index))
        loop.call_soon(events.append, ("heartbeat", index))
        return value * 2

    results = await execution.bounded_map(range(20), cached, 16, lambda: None)

    assert results == [value * 2 for value in range(20)]
    for index in range(19):
        assert events.index(("cached", index)) < events.index(("heartbeat", index))
        assert events.index(("heartbeat", index)) < events.index(("cached", index + 1))


@pytest.mark.parametrize("operation_fails", [False, True])
@pytest.mark.asyncio
async def test_cancellation_during_admission_drains_owned_call_and_stops_new_work(monkeypatch, operation_fails):
    admission = asyncio.Event()
    started = asyncio.Event()
    release = asyncio.Event()
    cancellation_drained = asyncio.Event()
    calls = []
    receipts = []

    async def admission_pause():
        admission.set()
        # Leave cancellation at the admission await, with a call already owned.
        await asyncio.Event().wait()

    async def operation(index, value):
        calls.append(index)
        started.set()
        await release.wait()
        receipts.append(value)
        if operation_fails:
            raise RuntimeError("provider failure accounted by the owned call")
        return value

    monkeypatch.setattr(execution, "yield_to_io", admission_pause)
    running = asyncio.create_task(execution.bounded_map(range(5), operation, 4, lambda: None))
    await asyncio.wait_for(admission.wait(), 1)
    await asyncio.wait_for(started.wait(), 1)
    running.cancel()
    # Allow the cancellation handler and its shielded drain to run first.
    asyncio.get_running_loop().call_soon(cancellation_drained.set)
    await cancellation_drained.wait()
    assert not running.done()
    assert calls == [0] and receipts == []

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 1)
    assert calls == [0] and receipts == [0]
