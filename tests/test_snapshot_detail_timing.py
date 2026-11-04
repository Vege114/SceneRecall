# ruff: noqa: F811
"""Version-detail tracing keeps freshness checks and cache behavior authoritative."""
import json

import pytest

from scenerecall.library import atomic_json, read_json
from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine, SearchSnapshotChanged
from scenerecall.search_timing import trace_request, trace_span
from test_knowledge_service import service  # noqa: F401
from test_work_entities import work  # noqa: F401


def counters(snapshot, name):
    return sum(row["attributes"].get(name, 0) for row in snapshot["spans"])


def test_knowledge_revision_details_keep_tokens_cache_and_full_scope_identity(service):
    svc, _, engine, _, _, _, _ = service
    filters = {"asset_id": "episode-one"}
    expected = svc.revision(filters, engine)
    svc.baseline.clear()
    snapshots = []
    for _ in range(2):
        with trace_request() as trace:
            with trace_span("test.revision"):
                assert svc.revision(filters, engine) == expected
        snapshots.append(trace.snapshot())
    cold, warm = snapshots
    assert counters(cold, "knowledge_cache_miss") == 2
    assert counters(warm, "knowledge_cache_miss") == 0
    assert counters(warm, "knowledge_cache_hit") == 3
    assert counters(cold, "selected_assets") == counters(warm, "selected_assets") == 1
    # E01 filtering does not narrow the authoritative identity revision to E01.
    assert counters(cold, "scope_assets") == counters(warm, "scope_assets") == 2
    names = {row["name"] for row in warm["spans"]}
    assert {"knowledge.scopes", "knowledge.projection", "knowledge.revision.runtime",
            "knowledge.revision.identity", "knowledge.revision.baseline", "knowledge.baseline.snapshot"} <= names
    assert len(cold["spans"]) == len(warm["spans"])
    assert "episode-one" not in json.dumps(warm)

    manifest = svc.library.asset_dir("episode-two") / "manifest.json"
    changed = read_json(manifest)
    changed["fingerprint"] += "-changed"
    atomic_json(manifest, changed)
    with trace_request():
        assert svc.revision(filters, engine) != expected


def test_repeated_source_validation_aggregates_without_paths_or_per_file_spans(service):
    svc, _, _, scope, source, _, _ = service
    saved = svc.knowledge.scope_catalog(scope)
    expected = svc.knowledge._source_text(saved, source["source_id"])
    with trace_request() as trace:
        with trace_span("test.sources"):
            for _ in range(25):
                assert svc.knowledge._source_text(saved, source["source_id"]) == expected
    snapshot = trace.snapshot()
    reads = snapshot["operations"]["source_read"]
    assert reads["count"] == 25
    assert reads["unique_count"] == 1
    assert reads["repeated_count"] == 24
    assert reads["bytes"] == len(expected[1].encode("utf-8")) * 25
    assert snapshot["operations"]["source_sha256"]["count"] == 25
    assert snapshot["operations"]["source_decode"]["count"] == 25
    assert len(snapshot["spans"]) == 1
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert str(svc.library.root) not in serialized
    assert source["source_id"] not in serialized
    assert expected[1] not in serialized


def test_source_integrity_failure_is_preserved_with_detail_tracing(service):
    svc, _, _, scope, source, _, _ = service
    saved = svc.knowledge.scope_catalog(scope)
    row = saved["source_documents"][source["source_id"]]
    path = svc.store._path(scope).parent / row["path"]
    path.write_bytes(b"tampered source")
    with trace_request() as trace:
        with trace_span("test.sources"):
            with pytest.raises(ValueError, match="哈希校验失败"):
                svc.knowledge._source_text(saved, source["source_id"])
    operations = trace.snapshot()["operations"]
    assert operations["source_read"]["count"] == operations["source_sha256"]["count"] == 1
    assert "source_decode" not in operations


@pytest.mark.asyncio
async def test_normal_search_revision_passes_and_final_freshness_fence(tmp_path):
    engine = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "private"))
    await engine.rebuild([])
    with trace_request() as trace:
        result = await engine.search("PRIVATE-QUERY-SENTINEL")
        await engine.check_completed_query()
    assert result["results"] == []
    snapshot = trace.snapshot()
    passes = [row["attributes"] for row in snapshot["spans"]
              if row["name"] in {"snapshot.acquire", "snapshot.validate"}]
    assert [(row["phase"], row["pass"]) for row in passes] == [
        ("initial", 1), ("after_recall", 2), ("after_canonical", 3)]
    assert snapshot["attributes"]["snapshot_passes"] == 3
    assert "PRIVATE-QUERY-SENTINEL" not in json.dumps(snapshot)
    engine.upsert([{"id": "new", "asset_id": "asset", "kind": "subtitle", "text": "new",
                    "start_ms": 0, "end_ms": 1000}])
    with trace_request() as trace:
        with pytest.raises(SearchSnapshotChanged):
            await engine.check_completed_query()
    fence = next(row for row in trace.snapshot()["spans"] if row["name"] == "snapshot.validate")
    assert fence["attributes"]["phase"] == "after_canonical"
    assert fence["status"] == "error"
