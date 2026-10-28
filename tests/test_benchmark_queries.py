"""Benchmark boundaries must prevent source writes and expose result drift."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("benchmark_queries", Path(__file__).parents[1] / "scripts/benchmark_queries.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_query_benchmark_rejects_source_overlap_and_existing_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for destination in (source, source / "child", tmp_path):
        with pytest.raises(ValueError):
            benchmark.safe_destination(source, destination)
    benchmark.safe_destination(source, tmp_path / "new-results")


def test_query_summary_detects_score_and_order_drift(tmp_path):
    response = {"results": [{"id": "one", "score": .8}, {"id": "two", "score": .7}]}
    for label in ("baseline", "current"):
        path = tmp_path / "pair-1" / label / "result.json"
        row = {"status": "completed", "implementation": label, "trial": 1, "source_hashes": {},
               "samples": [{"mode": "native", "case": "warm", "question_id": 0, "repetition": 0,
                            "elapsed_ms": 10 if label == "baseline" else 5,
                            "response_sha256": benchmark.digest(response)}]}
        benchmark.write(path, row)
    assert benchmark.summarize(tmp_path)["full_responses_equal"]
    path = tmp_path / "pair-1/current/result.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["samples"][0]["response_sha256"] = benchmark.digest({"results": list(reversed(response["results"]))})
    benchmark.write(path, row)
    result = benchmark.summarize(tmp_path)
    assert not result["full_responses_equal"] and len(result["mismatches"]) == 1


def test_p95_interpolates_without_discarding_tail():
    assert benchmark.percentile([1, 2, 3, 4, 100], .95) == pytest.approx(80.8)
