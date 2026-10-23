"""Startup rebuild work yields the request loop and cannot outlive cancellation."""
import asyncio
import threading

import pytest

from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine, SearchGenerationConflict
import scenerecall.search as search_module


def record(identity, text):
    return {"id": identity, "asset_id": "asset", "kind": "subtitle", "text": text,
            "start_ms": 0, "end_ms": 900}


@pytest.fixture
def engine(tmp_path):
    return SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "settings"))


def worker_gate():
    loop = asyncio.get_running_loop()
    owner = threading.get_ident()
    started, release = asyncio.Event(), threading.Event()

    def block():
        assert threading.get_ident() != owner, "rebuild blocked the request event loop"
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5), "worker release timed out"

    return block, started, release


async def wait_for_worker(started, rebuilding):
    waiting = asyncio.create_task(started.wait())
    try:
        done, _ = await asyncio.wait({waiting, rebuilding}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        if rebuilding in done:
            rebuilding.result()
        assert started.is_set(), "build did not reach the worker gate"
        assert not rebuilding.done()
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["validate", "contexts", "digest", "checkpoint"])
async def test_rebuild_preparation_keeps_event_loop_available(engine, monkeypatch, phase):
    rows = [record("kept", "retainedword")]
    await engine.rebuild(rows)
    block, started, release = worker_gate()
    if phase == "checkpoint":
        original = engine._db

        def tracked_db():
            block()
            return original()

        monkeypatch.setattr(engine, "_db", tracked_db)
    else:
        target, name = ((search_module, "_context_records") if phase == "contexts" else
                        (engine, "_validate_records" if phase == "validate" else "_records_digest"))
        original = getattr(target, name)

        def tracked(records):
            block()
            return original(records)

        monkeypatch.setattr(target, name, tracked)
    rebuilding = asyncio.create_task(engine.rebuild(rows))
    try:
        await wait_for_worker(started, rebuilding)
        # A request coroutine can run while any expensive cached-build phase waits.
        assert await asyncio.wait_for(asyncio.sleep(0, result="request served"), 1) == "request served"
    finally:
        release.set()
    assert (await rebuilding)["resumed"] is True


@pytest.mark.asyncio
async def test_edit_during_background_preparation_is_retryable_and_preserved(engine, monkeypatch):
    old = record("kept", "oldword")
    await engine.rebuild([old])
    block, started, release = worker_gate()
    original = search_module._context_records

    def tracked(records):
        block()
        return original(records)

    monkeypatch.setattr(search_module, "_context_records", tracked)
    rebuilding = asyncio.create_task(engine.rebuild([old]))
    try:
        await wait_for_worker(started, rebuilding)
        # Restore the incremental context builder so only the old build is paused.
        monkeypatch.setattr(search_module, "_context_records", original)
        await asyncio.to_thread(engine.sync_asset, [{**old, "text": "newword"}])
    finally:
        release.set()
    with pytest.raises(SearchGenerationConflict, match="索引构建期间资料发生变化"):
        await rebuilding
    assert engine.filtered_records()[0]["text"] == "newword"


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_publication_guard_rejects_source_change_even_without_index_edit(engine, cached):
    old = record("kept", "oldword")
    await engine.rebuild([old])

    def reject():
        raise SearchGenerationConflict("source mutation in progress")

    with pytest.raises(SearchGenerationConflict, match="source mutation"):
        await engine.rebuild_from_source(lambda: [old if cached else record("new", "newword")],
                                         publication_guard=reject)
    assert [row["id"] for row in engine.filtered_records()] == ["kept"]
    assert engine._generation == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("from_source", [False, True])
async def test_waiting_for_publication_lock_does_not_block_event_loop(engine, monkeypatch, from_source):
    rows = [record("kept", "retainedword")]
    await engine.rebuild(rows)
    block, started, release = worker_gate()
    original = engine._lock

    class BusyLock:
        def __enter__(self):
            block()
            return original.__enter__()

        def __exit__(self, *args):
            return original.__exit__(*args)

    monkeypatch.setattr(engine, "_lock", BusyLock())
    rebuilding = asyncio.create_task(engine.rebuild_from_source(lambda: rows) if from_source else engine.rebuild(rows))
    try:
        await wait_for_worker(started, rebuilding)
        assert await asyncio.wait_for(asyncio.sleep(0, result=True), 1)
    finally:
        release.set()
    assert (await rebuilding)["resumed"] is True


@pytest.mark.asyncio
async def test_cancelling_snapshot_waits_for_source_and_never_publishes(engine):
    block, started, release = worker_gate()

    def source():
        block()
        return [record("new", "newword")]

    rebuilding = asyncio.create_task(engine.rebuild_from_source(source))
    try:
        await wait_for_worker(started, rebuilding)
        rebuilding.cancel()
        await asyncio.sleep(0)
        rebuilding.cancel()  # Repeated shutdown cancellation must also drain.
        await asyncio.sleep(0)
        assert not rebuilding.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await rebuilding
    assert engine.status()["records"] == 0


@pytest.mark.asyncio
async def test_cancelling_publication_waits_for_atomic_commit(engine, monkeypatch):
    block, started, release = worker_gate()
    original = engine._write_record

    def tracked(db, row, **kwargs):
        block()
        original(db, row, **kwargs)

    monkeypatch.setattr(engine, "_write_record", tracked)
    rebuilding = asyncio.create_task(engine.rebuild([record("new", "newword")]))
    try:
        await wait_for_worker(started, rebuilding)
        rebuilding.cancel()
        await asyncio.sleep(0)
        assert not rebuilding.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await rebuilding
    # Completion of the cancelled task is a safe boundary for lifespan closure.
    assert engine.status()["records"] == 1
    assert engine._generation == 1
    with engine._db() as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM build_checkpoints").fetchone()[0] == 1
