"""Correlated search timings preserve parallelism, cache truth and privacy."""
import asyncio
import json
import sys
import threading

import httpx
import pytest
from PIL import Image

from scenerecall.main import create_app
from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine
from scenerecall.search_timing import current_trace, trace_request, trace_span
from scenerecall.search_timing_encoder import TimedVisualEncoder
from scenerecall.visual_encoder import VisualEncoder


def assert_tree(trace):
    spans = {row["id"]: row for row in trace["spans"]}
    assert spans
    for row in spans.values():
        assert 0 <= row["start_ms"] <= row["end_ms"] <= trace["total_ms"] + .01
        assert row["elapsed_ms"] >= 0
        if row["parent_id"] is not None:
            parent = spans[row["parent_id"]]
            assert parent["start_ms"] <= row["start_ms"]
            assert parent["end_ms"] >= row["end_ms"]


@pytest.mark.asyncio
async def test_search_traces_keep_parallel_children_and_concurrent_requests_isolated(tmp_path, monkeypatch):
    engine = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "private"))
    await engine.rebuild([])
    engine.visual_encoder = object()

    async def visual(query, filters, degraded):
        with trace_span("test.encoder"):
            await asyncio.sleep(.02)
        return []

    monkeypatch.setattr(engine, "_visual", visual)

    async def query(text):
        with trace_request() as trace:
            await engine.search(text)
            return trace.snapshot()

    left, right = await asyncio.gather(query("private left query"), query("private right query"))
    assert left["request_id"] != right["request_id"]
    assert current_trace() is None
    for trace in (left, right):
        assert_tree(trace)
        assert "private" not in json.dumps(trace)
        parallel = next(row for row in trace["spans"] if row["name"] == "recall.parallel")
        branches = [row for row in trace["spans"] if row["parent_id"] == parallel["id"]]
        assert {row["name"] for row in branches} == {"recall.lexical", "recall.semantic", "recall.visual", "recall.concept"}
        lexical = next(row for row in branches if row["name"] == "recall.lexical")
        visual = next(row for row in branches if row["name"] == "recall.visual")
        assert lexical["start_ms"] < visual["end_ms"]
        assert visual["start_ms"] < lexical["end_ms"]


@pytest.mark.asyncio
async def test_http_search_timing_includes_json_and_does_not_record_query(tmp_path):
    app = create_app(tmp_path, start_worker=False, token="test-token")
    # An unconfigured local encoder must degrade normally, without loading models.
    app.state.search.visual_encoder = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/api/search", headers={"Authorization": "Bearer test-token"},
                                     json={"query": "PRIVATE-QUERY-SENTINEL", "filters": {"kind": "subtitle"}})
    assert response.status_code == 200, response.text
    trace = json.loads(response.headers["X-Search-Timing"])
    assert trace["request_id"] == response.headers["X-Search-Request-ID"]
    assert "search;dur=" in response.headers["Server-Timing"]
    assert trace["attributes"]["query_kind"] == "subtitle"
    assert trace["attributes"]["result_count"] == 0
    assert "PRIVATE-QUERY" not in json.dumps(trace)
    assert "test-token" not in json.dumps(trace)
    assert_tree(trace)
    serialization = next(row for row in trace["spans"] if row["name"] == "response.json")
    assert serialization["attributes"]["bytes"] == len(response.content)
    assert response.json()["results"] == []


def test_disabled_spans_are_noops_and_error_span_is_closed():
    with trace_span("outside"):
        assert current_trace() is None
    with trace_request() as trace:
        with pytest.raises(ValueError):
            with trace_span("failed"):
                raise ValueError("private value")
        snapshot = trace.snapshot()
    assert snapshot["spans"][0]["status"] == "error"
    assert "private" not in json.dumps(snapshot)
    assert_tree(snapshot)


def test_encoder_timing_adapter_preserves_model_signature_and_real_weight_changes(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture weights")
    (tmp_path / "retrieval-encoder.json").write_text(json.dumps({
        "encoder_id": "siglip2-base-p16-256", "model_dir": str(model), "revision": "a" * 40, "python": sys.executable,
    }), encoding="utf-8")
    monkeypatch.setattr(VisualEncoder, "_runtime_versions", lambda self, config: {"torch": "fixture"})
    base, measured = VisualEncoder(tmp_path), TimedVisualEncoder(tmp_path)
    original = base.identity()
    assert measured.identity() == original
    with trace_request() as trace:
        assert measured.identity() == original
    assert trace.snapshot()["spans"][0]["name"] == "encoder.identity.work"
    (model / "model.safetensors").write_bytes(b"changed real weights")
    assert measured.identity()["signature"] != original["signature"]
    assert measured.identity() == base.identity()


def test_encoder_records_this_request_load_once_and_keeps_worker_durations_nested(tmp_path, monkeypatch):
    def start(self, config, identity):
        if self._process is None:
            self._process = object()
        return self._process

    def request(self, action, values, cancelled):
        self._start({}, {})
        self.metadata = {"model_load_ms": 12, "inference_ms": 3}
        return self._vectors([[1., 0.]], 1)

    monkeypatch.setattr(VisualEncoder, "_start", start)
    monkeypatch.setattr(VisualEncoder, "_request", request)
    encoder = TimedVisualEncoder(tmp_path)
    for cold in (True, False):
        with trace_request() as trace:
            assert encoder._request("texts", ["private query"], threading.Event()) == [[1., 0.]]
        snapshot = trace.snapshot()
        assert_tree(snapshot)
        phases = {row["name"]: row for row in snapshot["spans"]}
        assert phases["encoder.worker_start"]["attributes"]["worker_started"] is cold
        metadata = phases["encoder.worker_exchange"]["attributes"]
        assert metadata["worker_model_load_ms"] == (12 if cold else 0)
        assert metadata["worker_inference_ms"] == 3
        assert "private" not in json.dumps(snapshot)


@pytest.mark.asyncio
async def test_real_visual_cache_hits_skip_encoding_and_matrix_build(tmp_path):
    class Encoder:
        batch_size = 16
        calls = 0

        def identity(self):
            return {"model": "test-only"}

        async def encode_images(self, frames):
            return [[1., 0.] for _ in frames]

        async def encode_texts(self, texts):
            self.calls += 1
            return [[1., 0.] for _ in texts]

    engine = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "private"))
    encoder = engine.visual_encoder = Encoder()
    picture = tmp_path / "frame.png"
    Image.new("RGB", (12, 8), "red").save(picture)
    frame = {"id": "frame", "at_ms": 200, "pts": 200, "time_base": [1, 1000], "path": str(picture)}
    record = {"id": "window", "asset_id": "asset", "kind": "visual", "text": "",
              "evidence_type": "retrieval_window", "source_fingerprint": "source",
              "start_ms": 0, "end_ms": 1000, "frame_ids": ["frame"], "evidence_frame_ids": ["frame"],
              "ordered_frames": [{key: frame[key] for key in ("id", "at_ms", "pts", "time_base")}]}
    await engine.index_visual([record], [frame])
    for cached in (False, True):
        with trace_request() as trace:
            result = await engine.search("private visual query")
        assert len(result["results"]) == 1
        snapshot = trace.snapshot()
        assert_tree(snapshot)
        phases = {row["name"]: row for row in snapshot["spans"]}
        assert phases["visual.query_cache_lookup"]["attributes"]["query_cache_hit"] is cached
        assert phases["visual.matrix_cache"]["attributes"]["matrix_cache_hit"] is cached
        assert phases["visual.matrix_cache"]["attributes"]["matrix_built"] is not cached
        assert ("visual.query_encode" in phases) is not cached
        assert ("visual.matrix_build.work" in phases) is not cached
        assert "private" not in json.dumps(snapshot)
    assert encoder.calls == 1
