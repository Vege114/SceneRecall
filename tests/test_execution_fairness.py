"""Immediate cache hits must leave the service loop responsive and owned."""
import asyncio

import pytest

from scenerecall import execution


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
