"""Fallback diagnostics report observed causes without changing cache policy."""
import json

import numpy as np
import pytest

from scenerecall.providers import ProviderManager
from scenerecall.query_cache import MatrixCache, note_matrix_build
from scenerecall.search import SearchEngine
from scenerecall.search_timing import trace_request
from scenerecall.visual_index import evidence_digest


def test_disabled_generic_decline_and_allocation_error_are_distinct_and_memoized():
    cache = MatrixCache(0)
    info = {}
    assert cache.get_or_build("private-key", lambda _: pytest.fail("disabled"), diagnostics=info) is None
    assert info["fallback_reason"] == "cache_disabled"
    assert info["cache_budget_bytes"] == 0
    assert info["build_attempted"] is False
    assert info["memoized_fallback"] is False
    for reason in ("builder_declined", "allocation_error"):
        cache.max_bytes = 10
        calls = []

        def build(budget):
            calls.append(budget)
            if reason == "allocation_error":
                note_matrix_build(attempted_retained_bytes=6, retained_bytes_phase="fixture")
                raise MemoryError
            return None

        first, second = {}, {}
        assert cache.get_or_build("private-key", build, diagnostics=first) is None
        assert cache.get_or_build("private-key", build, diagnostics=second) is None
        assert first["fallback_reason"] == second["fallback_reason"] == reason
        assert first["memoized_fallback"] is False
        assert second["memoized_fallback"] is True
        assert first["build_attempted"] is True
        assert second["build_attempted"] is False
        assert second["attempted_retained_bytes"] == (6 if reason == "allocation_error" else None)
        assert calls == [10]
        assert "private-key" not in json.dumps([first, second])


def test_retained_budget_detail_survives_negative_cache_and_clear_retries():
    cache = MatrixCache(10)
    calls = []

    def build(budget):
        calls.append(budget)
        note_matrix_build(fallback_reason="retained_budget_exceeded",
                          attempted_retained_bytes=11, retained_bytes_phase="vectors")
        return None

    for memoized in (False, True):
        info = {}
        assert cache.get_or_build("key", build, diagnostics=info) is None
        assert info["fallback_reason"] == "retained_budget_exceeded"
        assert info["attempted_retained_bytes"] == 11
        assert info["retained_bytes_phase"] == "vectors"
        assert info["memoized_fallback"] is memoized
        assert info["cache_budget_bytes"] == 10
    assert calls == [10]
    cache.clear()
    info = {}
    cache.get_or_build("key", build, diagnostics=info)
    assert calls == [10, 10] and info["memoized_fallback"] is False


def test_budget_diagnostics_come_from_actual_matrix_accounting_and_preserve_streamed_results(tmp_path, monkeypatch):
    engine = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "private"))
    vector = np.array([1., 0.], dtype="<f4")
    frame = {"id": "frame", "at_ms": 200, "pts": 200, "time_base": [1, 1000]}
    record = {"id": "window", "asset_id": "asset", "kind": "visual", "text": "",
              "evidence_type": "retrieval_window", "source_fingerprint": "source",
              "start_ms": 0, "end_ms": 1000, "frame_ids": ["frame"], "evidence_frame_ids": ["frame"],
              "ordered_frames": [frame]}
    engine.upsert([record])
    with engine._db() as db:
        db.execute("INSERT INTO visual_namespaces VALUES ('fixture','{}',2,0)")
        db.execute("INSERT INTO visual_records VALUES (?,?,?,?)", ("fixture", "window", "asset", evidence_digest(record)))
        db.execute("INSERT INTO visual_links VALUES ('fixture','window','frame','frame',200)")
        db.execute("INSERT INTO visual_cache VALUES ('fixture','frame',2,?)", (vector.tobytes(),))

    def rank():
        with trace_request() as trace:
            rows = engine.visual_index._rank("fixture", vector, 2, "1=1", [], 50)
        info = next(span["attributes"] for span in trace.snapshot()["spans"] if span["name"] == "visual.matrix_cache")
        return rows, info

    cache = engine.visual_index.matrix_cache
    expected, built = rank()
    assert built["cache_status"] == "built"
    full_size = built["attempted_retained_bytes"]
    _, hit = rank()
    assert hit["cache_status"] == "hit" and hit["cached_retained_bytes"] == full_size
    for budget, stage in ((1, "records"), (full_size - 1, "vectors")):
        cache.max_bytes = budget
        for memoized in (False, True):
            rows, info = rank()
            assert rows == expected
            assert info["fallback_reason"] == "retained_budget_exceeded"
            assert info["attempted_retained_bytes"] > budget
            assert info["retained_bytes_phase"] == stage
            assert info["memoized_fallback"] is memoized
            assert info["matrix_built"] is not memoized
            assert info["streaming_fallback"] is True
        assert cache.stats()["bytes"] == 0
    cache.max_bytes = 256 * 1024 * 1024

    def exhausted(*args):
        raise MemoryError

    monkeypatch.setattr(engine.visual_index, "_matrix_snapshot", exhausted)
    for memoized in (False, True):
        rows, info = rank()
        assert rows == expected
        assert info["fallback_reason"] == "allocation_error"
        assert info["attempted_retained_bytes"] is None
        assert info["memoized_fallback"] is memoized
