import asyncio
import threading

import pytest

from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine, SearchSnapshotChanged
from test_search import record, vector_provider


@pytest.fixture
def engine(tmp_path):
    return SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "private"))


@pytest.mark.asyncio
async def test_branches_overlap_and_completion_order_does_not_change_ties(engine, monkeypatch):
    engine.visual_encoder = object()
    engine.concept_lookup = object()
    rows = [record("one", "内容", kind="visual"), record("two", "内容", kind="visual")]
    engine.upsert(rows)
    barrier = threading.Barrier(4, timeout=5)
    completion_order = ["semantic", "visual", "concept"]
    completed = []
    gates = {name: asyncio.Event() for name in completion_order}
    gates[completion_order[0]].set()

    def lexical(*args):
        barrier.wait()
        return rows

    def recall(name):
        async def run(*args):
            await asyncio.to_thread(barrier.wait)
            await gates[name].wait()
            completed.append(name)
            position = completion_order.index(name)
            if position + 1 < len(completion_order):
                gates[completion_order[position + 1]].set()
            # Two branches rank each candidate first, producing a true RRF tie.
            return list(rows if name == "visual" else reversed(rows))
        return run

    monkeypatch.setattr(engine, "_lexical", lexical)
    monkeypatch.setattr(engine, "_semantic", recall("semantic"))
    monkeypatch.setattr(engine, "_visual", recall("visual"))
    monkeypatch.setattr(engine, "_concept", recall("concept"))
    first = await asyncio.wait_for(engine.search("内容", embedding_profile_id="fake"), 10)
    assert completed == ["semantic", "visual", "concept"]
    completion_order = ["concept", "visual", "semantic"]
    completed = []
    gates = {name: asyncio.Event() for name in completion_order}
    gates[completion_order[0]].set()
    second = await asyncio.wait_for(engine.search("内容", embedding_profile_id="fake"), 10)
    assert completed == ["concept", "visual", "semantic"]
    assert first == second
    assert [row["id"] for row in first["results"]] == ["one", "two"]
    assert first["results"][0]["score"] == first["results"][1]["score"]
    assert set(engine.query_diagnostics["branches_ms"]) == {"lexical", "semantic", "visual", "concept"}


@pytest.mark.asyncio
async def test_changed_snapshot_retries_once_and_returns_latest(engine, monkeypatch):
    engine.upsert([record("one", "原始内容")])
    original = engine._lexical
    calls = 0

    def lexical(*args):
        nonlocal calls
        calls += 1
        rows = original(*args)
        if calls == 1:
            engine.upsert([record("one", "最新内容")])
        return rows

    monkeypatch.setattr(engine, "_lexical", lexical)
    result = await engine.search("内容")
    assert result["results"][0]["text"] == "最新内容"
    assert calls == 2 and engine.query_diagnostics["retries"] == 1


@pytest.mark.asyncio
async def test_real_semantic_retry_reuses_query_embedding(tmp_path, monkeypatch):
    providers, calls = vector_provider(tmp_path)
    engine = SearchEngine(tmp_path / "index", providers)
    await engine.rebuild([record("cup", "人物拿着杯子", kind="visual"),
                          record("bird", "树上飞鸟", kind="visual")], "embed")
    calls.clear()
    original = engine._semantic
    recalls = 0

    async def changed_after_first_recall(*args):
        nonlocal recalls
        result = await original(*args)
        recalls += 1
        if recalls == 1:
            engine.upsert([record("unrelated", "无关的新记录", kind="visual")])
        return result

    monkeypatch.setattr(engine, "_semantic", changed_after_first_recall)
    result = await engine.search("喝水", embedding_profile_id="embed")
    assert [row["id"] for row in result["results"]] == ["cup"]
    assert recalls == 2 and engine.query_diagnostics["retries"] == 1
    assert len(calls) == 1 and calls[0][1]["input"] == ["喝水"]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["_semantic_snapshot", "_semantic_candidates"])
async def test_real_semantic_blocking_io_overlaps_visual_and_drains_on_cancel(tmp_path, monkeypatch, blocked_phase):
    providers, calls = vector_provider(tmp_path)
    engine = SearchEngine(tmp_path / "index", providers)
    await engine.rebuild([record("cup", "人物拿着杯子", kind="visual")], "embed")
    calls.clear()
    engine.visual_encoder = object()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    visual_started = asyncio.Event()
    original = getattr(engine, blocked_phase)

    def blocking_io(*args):
        entered.set()
        try:
            assert release.wait(5)
            return original(*args)
        finally:
            finished.set()

    async def visual(*args):
        visual_started.set()
        return []

    monkeypatch.setattr(engine, blocked_phase, blocking_io)
    monkeypatch.setattr(engine, "_visual", visual)
    task = asyncio.create_task(engine.search("喝水", embedding_profile_id="embed"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await asyncio.wait_for(visual_started.wait(), 1)
        assert not finished.is_set()
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done(), "cancellation must wait for owned SQLite/Lance work"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert len(calls) == (0 if blocked_phase == "_semantic_snapshot" else 1)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_continuous_updates_raise_retryable_error(engine, monkeypatch):
    original = engine._lexical
    count = 0

    def lexical(*args):
        nonlocal count
        count += 1
        rows = original(*args)
        engine.upsert([record("one", f"内容{count}")])
        return rows

    monkeypatch.setattr(engine, "_lexical", lexical)
    with pytest.raises(SearchSnapshotChanged, match="正在更新"):
        await engine.search("内容")
    assert count == 2


@pytest.mark.asyncio
async def test_expected_failure_retains_other_branches(engine, monkeypatch):
    engine.upsert([record("one", "内容")])
    engine.visual_encoder = object()

    async def fail(*args):
        raise ValueError("坏掉的视觉索引")

    monkeypatch.setattr(engine, "_visual", fail)
    result = await engine.search("内容")
    assert [row["id"] for row in result["results"]] == ["one"]
    assert result["degraded"] == ["直接视觉检索不可用，已保留其他召回结果：坏掉的视觉索引"]


@pytest.mark.asyncio
async def test_cancellation_drains_blocking_recall(engine, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def lexical(*args):
        entered.set()
        release.wait(5)
        finished.set()
        return []

    monkeypatch.setattr(engine, "_lexical", lexical)
    task = asyncio.create_task(engine.search("内容"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_unexpected_error_cancels_other_recall(engine, monkeypatch):
    started, cleaned = asyncio.Event(), asyncio.Event()
    engine.visual_encoder = object()

    async def visual(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    def lexical(*args):
        raise AssertionError("unexpected")

    monkeypatch.setattr(engine, "_visual", visual)
    monkeypatch.setattr(engine, "_lexical", lexical)
    with pytest.raises(AssertionError, match="unexpected"):
        await engine.search("内容")
    assert started.is_set() and cleaned.is_set()


def test_candidate_owners_include_members_and_subtitle_sources(engine):
    engine.upsert([record("one", "内容"), record("two", "别处", asset_id="second")])
    assert engine.result_asset_ids([{"id": "invented", "asset_id": "untrusted", "member_ids": ["one"],
                                     "source_ids": ["two"]}]) == ["film-one", "second"]


@pytest.mark.asyncio
async def test_unversioned_provider_cannot_supply_knowledge_hits(engine):
    engine.upsert([record("one", "内容")])

    class BrokenRevision:
        def revision(self, *args):
            raise ValueError("unavailable")

        def search(self, *args):
            raise AssertionError("unversioned knowledge must not be accepted")

    engine.knowledge_provider = BrokenRevision()
    result = await engine.search("内容")
    assert result["results"][0]["id"] == "one"
    assert any("知识版本不可用" in message for message in result["degraded"])


@pytest.mark.asyncio
async def test_completed_query_detects_late_changes(engine):
    engine.upsert([record("one", "内容")])
    await engine.search("内容")
    engine.remove_assets(["film-one"])
    with pytest.raises(SearchSnapshotChanged):
        await engine.check_completed_query()
