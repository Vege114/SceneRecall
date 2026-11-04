"""Detailed timing stays bounded, request-local and independent of behavior."""
import asyncio
import json
import threading

import pytest

from scenerecall.execution import run_in_thread
from scenerecall.search_timing import aggregate_operation, timed_lock, timing_count, trace_request, trace_span


def test_operations_are_local_to_nearest_span_and_globally_counted_once():
    with trace_request() as trace:
        with trace_span("parent"):
            with aggregate_operation("read", identity="private/file.json", chars=10):
                pass
            with trace_span("child"):
                for _ in range(3):
                    with aggregate_operation("read", identity="private/file.json") as operation:
                        operation.set(chars=20, forbidden_text="private source contents")
                    timing_count("records", 2)
        snapshot = trace.snapshot()
    parent, child = snapshot["spans"]
    assert len(snapshot["spans"]) == 2
    assert parent["attributes"]["read_count"] == 1
    assert child["attributes"]["read_count"] == 3
    assert child["attributes"]["records"] == 6
    assert child["attributes"]["read_unique_count"] == 1
    assert child["attributes"]["read_repeated_count"] == 2
    total = snapshot["operations"]["read"]
    assert total["count"] == 4 and total["chars"] == 70
    assert total["unique_count"] == 1 and total["repeated_count"] == 3
    assert total["elapsed_ms"] >= 0
    assert "private" not in json.dumps(snapshot)


@pytest.mark.asyncio
async def test_overlapping_request_threads_do_not_share_counters():
    barrier = threading.Barrier(2)

    async def request(count):
        with trace_request() as trace:
            with trace_span("request"):
                def worker():
                    barrier.wait(timeout=5)
                    with trace_span("work"):
                        for _ in range(count):
                            with aggregate_operation("read", identity="same-private-path"):
                                pass
                            timing_count("records")
                await run_in_thread(worker)
            return trace.snapshot()

    first, second = await asyncio.gather(request(2), request(5))
    assert first["request_id"] != second["request_id"]
    for snapshot, count in ((first, 2), (second, 5)):
        assert snapshot["operations"]["read"]["count"] == count
        assert snapshot["operations"]["read"]["repeated_count"] == count - 1
        parent, worker = snapshot["spans"]
        assert worker["parent_id"] == parent["id"]
        assert "read_count" not in parent["attributes"]
        assert worker["attributes"]["records"] == count


def test_many_operations_create_no_extra_spans_and_bound_identity_storage(monkeypatch):
    monkeypatch.setattr("scenerecall.search_timing._IDENTITY_LIMIT", 2)
    with trace_request() as trace:
        with trace_span("batch"):
            for index in range(1000):
                with aggregate_operation("read", identity=str(index)):
                    pass
        snapshot = trace.snapshot()
    assert len(snapshot["spans"]) == 1
    operation = snapshot["operations"]["read"]
    assert operation["count"] == 1000
    assert operation["unique_count"] == 2
    assert operation["identity_limit_reached"] is True


def test_errors_and_reentrant_locks_keep_original_semantics():
    lock = threading.RLock()
    with trace_request() as trace:
        with trace_span("work"):
            with pytest.raises(ValueError, match="original error"):
                with timed_lock(lock):
                    with timed_lock(lock):
                        with aggregate_operation("parse"):
                            raise ValueError("original error")
        snapshot = trace.snapshot()
    assert snapshot["operations"]["parse"]["error_count"] == 1
    assert snapshot["operations"]["lock_wait"]["count"] == 2
    acquired = []
    def acquire_from_worker():
        success = lock.acquire(timeout=1)
        acquired.append(success)
        if success:
            lock.release()
    thread = threading.Thread(target=acquire_from_worker)
    thread.start()
    thread.join(timeout=2)
    assert acquired == [True]


def test_no_trace_and_count_units_do_not_change_operation_results():
    with aggregate_operation("parse", identity="not-published", count=3):
        result = json.loads('[1, 2, 3]')
    timing_count("ignored", 5)
    assert result == [1, 2, 3]
    with trace_request() as trace:
        with aggregate_operation("parse", identity="file", count=3):
            pass
        snapshot = trace.snapshot()
    assert snapshot["spans"] == []
    assert snapshot["operations"]["parse"]["count"] == 3
    assert snapshot["operations"]["parse"]["identity_calls"] == 1
