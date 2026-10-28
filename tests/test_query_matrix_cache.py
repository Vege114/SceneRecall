"""Query snapshots preserve exact scores while tracking independent DB writers."""
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from scenerecall.providers import ProviderManager
from scenerecall.query_cache import MatrixCache, query_revision
from scenerecall.search import SearchEngine
from scenerecall.visual_index import evidence_digest, normalized_vectors


@pytest.fixture
def seeded(tmp_path):
    engine = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "private"))
    rng = np.random.default_rng(87)
    vectors = normalized_vectors(rng.normal(size=(720, 64)), 720)
    # Exact ties cross matrix-block boundaries and two frames within a window.
    vectors[1] = vectors[0].copy()
    vectors[515] = vectors[0].copy()
    rows, links, cache, manifests = [], [], [], []
    for index in range(240):
        identity = f"window-{index:04}"
        asset = "small" if index == 239 else "large"
        frames = [{"id": f"frame-{index * 3 + frame:04}", "at_ms": index * 1000 + frame * 100,
                   "pts": index * 1000 + frame * 100, "time_base": [1, 1000]} for frame in range(3)]
        record = {"id": identity, "asset_id": asset, "kind": "visual", "text": "画面",
                  "evidence_type": "retrieval_window", "source_fingerprint": "source-" + asset,
                  "start_ms": index * 1000, "end_ms": index * 1000 + 1000,
                  "frame_ids": [frame["id"] for frame in frames],
                  "evidence_frame_ids": [frame["id"] for frame in frames], "ordered_frames": frames}
        rows.append(record)
        manifests.append(("fixture", identity, asset, evidence_digest(record)))
        for offset, frame in enumerate(frames):
            links.append(("fixture", identity, frame["id"], frame["id"], frame["at_ms"]))
            cache.append(("fixture", frame["id"], 64, vectors[index * 3 + offset].tobytes()))
    engine.upsert(rows)
    with engine._db() as db:
        db.execute("INSERT INTO visual_namespaces VALUES ('fixture','{}',64,0)")
        db.executemany("INSERT INTO visual_records VALUES (?,?,?,?)", manifests)
        db.executemany("INSERT INTO visual_links VALUES (?,?,?,?,?)", links)
        db.executemany("INSERT INTO visual_cache VALUES (?,?,?,?)", cache)
    return engine, rows, vectors


def rank(engine, vector, filters=None, limit=50):
    where, params = engine._where(filters or {})
    return engine.visual_index._rank("fixture", vector, 64, where, params, limit)


def revision(engine):
    with engine._db() as db:
        return query_revision(db)


def test_cached_and_streamed_rankings_are_identical_over_multiple_blocks(seeded):
    engine, _, vectors = seeded
    cache = engine.visual_index.matrix_cache
    for filters in ({}, {"asset_id": "small"}, {"asset_id": "absent"}):
        for vector in (vectors[0], vectors[2], -vectors[0]):
            for limit in (5, 50):
                cached = rank(engine, vector, filters, limit)
                with engine._db() as db:
                    db.execute("BEGIN")
                    where, params = engine._where(filters)
                    streamed = engine.visual_index._rank_stream(db, "fixture", vector, 64, where, params, limit)
                assert cached == streamed
    assert cache.stats()["builds"] == 3
    assert cache.stats()["hits"] == 15
    assert cache.stats()["bytes"] <= cache.max_bytes


def test_hot_queries_do_not_read_vectors_or_parse_all_records(seeded, monkeypatch):
    engine, _, vectors = seeded
    first = rank(engine, vectors[0])
    statements = []
    original = engine._db

    def traced():
        db = original()
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(engine, "_db", traced)
    second = rank(engine, vectors[1], limit=5)
    assert second == first[:5]
    assert not any("visual_cache" in sql or "FROM records" in sql for sql in statements)
    # Public result objects must not expose shared snapshot state.
    second[0]["ordered_frames"][0]["at_ms"] = -999
    assert rank(engine, vectors[0], limit=5) == first[:5]


def test_vector_cache_writes_do_not_invalidate_but_independent_writers_do(seeded, tmp_path):
    engine, records, vectors = seeded
    rank(engine, vectors[0])
    before = revision(engine)
    with engine._db() as db:
        db.execute("INSERT INTO visual_query_cache VALUES ('fixture','question',64,?)", (vectors[0].tobytes(),))
    assert revision(engine) == before
    rank(engine, vectors[0])
    assert engine.visual_index.matrix_cache.stats()["builds"] == 1
    other = SearchEngine(engine.index_dir, ProviderManager(tmp_path / "other-private"))
    assert revision(engine) == before
    changed = copy.deepcopy(records[0])
    changed["text"] = "最新人工名称"
    other.upsert([changed])
    assert revision(engine) != before
    result = rank(engine, vectors[0])
    assert next(row for row in result if row["id"] == changed["id"])["text"] == changed["text"]
    assert engine.visual_index.matrix_cache.stats()["builds"] == 2
    other.remove_assets(["large"])
    assert all(row["asset_id"] == "small" for row in rank(engine, vectors[0]))


@pytest.mark.parametrize("mutation", ["source", "vector", "link", "namespace", "fts"])
def test_direct_sql_changes_invalidate_cache_and_rollback_does_not(seeded, mutation):
    engine, records, vectors = seeded
    rank(engine, vectors[0])
    initial = revision(engine)
    with engine._db() as db:
        db.execute("BEGIN")
        db.execute("DELETE FROM visual_records")
        assert query_revision(db) != initial
        db.rollback()
    assert revision(engine) == initial
    with engine._db() as db:
        if mutation == "source":
            changed = {**records[0], "source_fingerprint": "replacement"}
            db.execute("UPDATE records SET data=? WHERE id=?", (json.dumps(changed), changed["id"]))
        elif mutation == "vector":
            db.execute("UPDATE visual_cache SET vector=? WHERE cache_key='frame-0000'", ((-vectors[0]).tobytes(),))
        elif mutation == "link":
            db.execute("DELETE FROM visual_links WHERE frame_id='frame-0000'")
        elif mutation == "namespace":
            db.execute("UPDATE visual_namespaces SET identity='changed'")
        else:
            db.execute("DELETE FROM evidence_fts")
    assert revision(engine) != initial
    cached = rank(engine, vectors[0])
    with engine._db() as db:
        expected = engine.visual_index._rank_stream(db, "fixture", vectors[0], 64, "1=1", [], 50)
    assert cached == expected
    assert engine.visual_index.matrix_cache.stats()["builds"] == 2


def test_disabled_over_budget_and_allocation_failure_preserve_results(seeded, monkeypatch):
    engine, _, vectors = seeded
    expected = rank(engine, vectors[0])
    cache = engine.visual_index.matrix_cache
    for budget in (0, 1, 4096):
        cache.max_bytes = budget
        assert rank(engine, vectors[0]) == expected
        assert cache.stats()["bytes"] == 0
    cache.max_bytes = 256 * 1024 * 1024

    def exhausted(*args):
        raise MemoryError

    monkeypatch.setattr(engine.visual_index, "_matrix_snapshot", exhausted)
    assert rank(engine, vectors[0]) == expected
    assert cache.stats()["bytes"] == 0


def test_lru_retained_budget_evicts_old_scope():
    cache = MatrixCache(10)
    a, b = object(), object()
    assert cache.get_or_build("a", lambda budget: (a, 7)) is a
    assert cache.get_or_build("b", lambda budget: (b, 7)) is b
    assert cache.stats()["bytes"] == 7 and cache.stats()["evictions"] == 1
    assert cache.get_or_build("b", lambda _: pytest.fail("must hit")) is b
    cache.clear()
    assert cache.stats()["bytes"] == 0 and cache.stats()["entries"] == 0


def test_concurrent_same_scope_constructs_once(seeded, monkeypatch):
    engine, _, vectors = seeded
    entered, release = threading.Event(), threading.Event()
    original = engine.visual_index._matrix_snapshot
    builds = []

    def blocked(*args):
        builds.append(1)
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(engine.visual_index, "_matrix_snapshot", blocked)
    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(rank, engine, vectors[0])
        assert entered.wait(5)
        others = [pool.submit(rank, engine, vectors[0]) for _ in range(3)]
        release.set()
        expected = first.result(timeout=5)
        assert all(future.result(timeout=5) == expected for future in others)
    assert len(builds) == 1
