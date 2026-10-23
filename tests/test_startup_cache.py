"""Background readiness never publishes statistics from an obsolete snapshot."""
import asyncio
import threading

import pytest

from scenerecall.startup import StartupPreparation


class LibraryDouble:
    def __init__(self):
        self.records_calls = 0
        self.statistics_inputs = []

    def records(self, *, compact_subtitles=False):
        assert compact_subtitles is False
        self.records_calls += 1
        return [{"id": "cue", "record_id": "cue", "asset_id": "asset", "kind": "subtitle",
                 "text": "当前证据", "start_ms": 0, "end_ms": 1000}]

    def statistics(self, records=None):
        self.statistics_inputs.append(records)
        rows = self.records(compact_subtitles=False) if records is None else records
        return {"assets": 1, "records": len(rows), "favorites": 0}


class EncoderDouble:
    def __init__(self):
        self.available = False
        self.calls = 0

    def settings(self):
        self.calls += 1
        return {"available": self.available, "offline": True}


class SearchDouble:
    def __init__(self):
        self._generation = 0
        self.visual_encoder = EncoderDouble()
        self.fail = False

    async def rebuild_from_source(self, source, *, publication_guard):
        if self.fail:
            raise ValueError("synthetic invalid source")
        rows = await asyncio.to_thread(source)
        publication_guard()
        self._generation += 1
        return {"records": len(rows)}


class QueueDouble:
    def __init__(self):
        self.starts = 0

    async def start(self):
        self.starts += 1


@pytest.fixture
def preparation():
    return StartupPreparation(LibraryDouble(), SearchDouble(), QueueDouble())


@pytest.mark.asyncio
async def test_statistics_changed_during_computation_keep_last_good_counts(preparation, monkeypatch):
    initial = {"assets": 9, "records": 42, "favorites": 2}
    preparation.stats = initial.copy()
    started, release = threading.Event(), threading.Event()
    original = preparation.library.statistics

    def statistics(records=None):
        started.set()
        assert release.wait(5), "test did not release statistics worker"
        return original(records=records)

    monkeypatch.setattr(preparation.library, "statistics", statistics)
    pending = asyncio.create_task(preparation._statistics())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        preparation.begin_mutation()
        preparation.end_mutation()
    finally:
        release.set()
        await pending
    assert preparation.stats == initial
    assert preparation.states["statistics"]["status"] == "pending"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["revision", "generation"])
async def test_statistics_reject_obsolete_raw_startup_projection(preparation, changed):
    old_rows = preparation.library.records(compact_subtitles=False)
    snapshot = (old_rows, preparation.revision, preparation.search._generation)
    if changed == "revision":
        preparation.begin_mutation()
        preparation.end_mutation()
    else:
        preparation.search._generation += 1
    await preparation._statistics(snapshot)
    assert preparation.library.statistics_inputs == [None]
    assert preparation.library.records_calls == 2
    assert preparation.states["statistics"]["status"] == "ready"


@pytest.mark.asyncio
async def test_index_failure_defers_queue_until_successful_retry_and_starts_it_once(preparation):
    preparation.search.fail = True
    preparation.start()
    try:
        await preparation.task
        assert preparation.states["index"]["status"] == "error"
        assert preparation.queue.starts == 0
        preparation.search.fail = False
        preparation.retry()
        await preparation.task
        assert preparation.states["index"]["status"] == "ready"
        assert preparation.states["statistics"]["status"] == "ready"
        assert preparation.queue.starts == 1
        assert preparation.library.records_calls == 1, "startup statistics must reuse the index source projection"
        preparation.retry()
        await preparation.task
        assert preparation.queue.starts == 1
    finally:
        await preparation.stop()


@pytest.mark.asyncio
async def test_encoder_retry_rechecks_normal_unavailable_result_after_configuration_repair(preparation):
    preparation.start()
    try:
        await preparation.task
        assert preparation.states["encoder"]["status"] == "ready"
        assert preparation.encoder["available"] is False
        preparation.search.visual_encoder.available = True
        preparation.retry()
        await preparation.task
        assert preparation.encoder["available"] is True
        assert preparation.search.visual_encoder.calls == 2
    finally:
        await preparation.stop()


@pytest.mark.asyncio
async def test_expired_encoder_status_refreshes_without_rebuilding_evidence(preparation):
    preparation.start()
    try:
        await preparation.task
        assert preparation.encoder["available"] is False
        preparation.search.visual_encoder.available = True
        preparation.encoder_checked = float("-inf")
        preparation.request_refresh()
        await preparation.task
        assert preparation.encoder["available"] is True
        assert preparation.search.visual_encoder.calls == 2
        assert preparation.library.records_calls == 1
        assert preparation.queue.starts == 1
    finally:
        await preparation.stop()
