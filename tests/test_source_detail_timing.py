"""Source-read instrumentation keeps evidence, errors and IO behavior unchanged."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scenerecall.identity_bindings import resolve_observations
from scenerecall.library import Library, atomic_json, digest, read_json, read_jsonl
from scenerecall.search_timing import trace_request, trace_span
from test_identity_bindings import calibrated  # noqa: F401
from test_scene_api import scene_app  # noqa: F401
from test_search_timing import assert_tree
from test_storage_security import ASSET_ID, seed_asset
from test_visual_identity import visual  # noqa: F401


def test_file_metrics_count_actual_reads_without_exposing_paths_or_content(tmp_path, monkeypatch):
    source = tmp_path / "PRIVATE-PATH.json"
    lines = tmp_path / "PRIVATE-PATH.jsonl"
    value = {"schema_version": "1.0", "text": "PRIVATE-CONTENT 中文", "values": [1, None, True]}
    source.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    lines.write_text(json.dumps(value) + "\n \n" + json.dumps(value) + "\n", encoding="utf-8")
    calls = []
    original = Path.read_text

    def counted(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    expected_digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    with trace_request() as trace:
        with trace_span("files"):
            assert read_json(source) == value
            assert read_json(source) == value
            assert read_jsonl(lines) == [value, value]
            fallback = {"same": "object"}
            assert read_json(tmp_path / "missing.json", fallback) is fallback
            assert read_jsonl(tmp_path / "missing.jsonl") == []
            assert digest(value) == expected_digest
    snapshot = trace.snapshot()
    assert calls == [source, source, lines]
    assert len(snapshot["spans"]) == 1
    metrics = snapshot["spans"][0]["attributes"]
    assert metrics["fs_exists_count"] == 5
    assert metrics["file_read_text_count"] == metrics["json_parse_count"] == 3
    assert metrics["file_read_text_unique_count"] == 2
    assert metrics["file_read_text_repeated_count"] == 1
    assert metrics["file_missing_count"] == metrics["jsonl_record_count"] == 2
    assert metrics["digest_serialize_count"] == metrics["digest_sha256_count"] == 1
    assert "PRIVATE-" not in json.dumps(snapshot)
    assert str(tmp_path) not in json.dumps(snapshot)
    assert_tree(snapshot)


@pytest.mark.parametrize("reader", [read_json, read_jsonl])
@pytest.mark.parametrize("content,error", [("{bad", json.JSONDecodeError),
                                          ('{"schema_version":"future"}', ValueError)])
def test_file_metrics_preserve_parse_and_schema_errors(tmp_path, reader, content, error):
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(error):
        reader(path)
    with trace_request() as trace:
        with pytest.raises(error):
            with trace_span("invalid"):
                reader(path)
    row = trace.snapshot()["spans"][0]
    assert row["status"] == "error"
    assert row["attributes"]["file_read_text_count"] == 1
    assert row["attributes"]["json_parse_count"] == 1


def test_whole_asset_source_projection_is_unchanged_and_spans_do_not_scale_per_file(tmp_path):
    library = Library(tmp_path / "library")
    seed_asset(library)
    root = library.asset_dir(ASSET_ID)
    active = {"units": {}}
    for number in range(80):
        record = {"id": f"window_{number}", "asset_id": ASSET_ID, "start_ms": number * 10,
                  "end_ms": number * 10 + 10, "summary": "PRIVATE-SOURCE-CONTENT", "entities": [],
                  "events": [], "evidence_type": "retrieval_window", "evidence_frame_ids": []}
        relative = f"retrieval/units/{record['id']}.json"
        atomic_json(root / relative, record)
        active["units"][record["id"]] = {"path": relative, "digest": digest(record)}
    atomic_json(root / "retrieval/active.json", active)
    expected = library.records(ASSET_ID)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")}
    with trace_request() as trace:
        with trace_span("canonical.asset_records"):
            actual = library.records(ASSET_ID)
    snapshot = trace.snapshot()
    assert actual == expected and len(actual) == 80
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")}
    phases = {row["name"]: row for row in snapshot["spans"]}
    assert phases["records.retrieval"]["attributes"]["record_count"] == 80
    loading = phases["retrieval.load_validate"]["attributes"]
    assert loading["window_count"] == loading["digest_sha256_count"] == 80
    assert loading["file_read_text_count"] == 81
    assert phases["records.assemble"]["attributes"]["output_count"] == 80
    assert len(snapshot["spans"]) < 25
    assert "PRIVATE-SOURCE-CONTENT" not in json.dumps(snapshot)
    assert_tree(snapshot)


def test_identity_batch_keeps_decisions_and_reuses_region_reference_reads(calibrated):  # noqa: F811
    library, _characters, _gallery, _pipeline, _backend, _identity, _other, _refs, observation = calibrated
    source = [copy.deepcopy(observation) for _ in range(20)]
    before = copy.deepcopy(source)
    expected = resolve_observations(library, "asset_one", source)
    with trace_request() as trace:
        actual = resolve_observations(library, "asset_one", source)
    snapshot = trace.snapshot()
    assert actual == expected and source == before
    assert len([row for row in snapshot["spans"] if row["name"] == "identity.resolver_init"]) == 1
    assert len(snapshot["spans"]) == 2
    batch = next(row for row in snapshot["spans"] if row["name"] == "identity.resolve_batch")
    assert batch["attributes"]["record_count"] == 20
    assert batch["attributes"]["region_cache_miss_count"] == 2
    assert batch["attributes"]["region_cache_hit_count"] > 0
    assert batch["attributes"]["reference_hash_cache_hit_count"] > 0
    assert_tree(snapshot)


def test_http_source_timing_reports_requested_and_whole_asset_counts(scene_app):  # noqa: F811
    _app, client, _provider, scene = scene_app
    response = client.post("/api/search", json={"query": "站台", "filters": {"kind": "scene"}})
    assert response.status_code == 200, response.text
    assert [row["record_id"] for row in response.json()["results"]] == [scene["id"]]
    snapshot = json.loads(response.headers["X-Search-Timing"])
    lookup = next(row for row in snapshot["spans"] if row["name"] == "canonical.owner_lookup")
    assert lookup["attributes"]["owner_count"] == 1
    owner = next(row for row in snapshot["spans"] if row["name"] == "canonical.asset_records")
    assert owner["attributes"]["requested_record_count"] == 1
    assert owner["attributes"]["produced_count"] > owner["attributes"]["requested_record_count"]
    assert owner["attributes"]["produced_scene_count"] == 1
    assert owner["attributes"]["produced_subtitle_count"] == 1
    assert_tree(snapshot)
