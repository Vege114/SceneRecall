"""Experiment accounting must distinguish evidence, unavailable gold and cache fallback."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_knowledge.py"
    spec = importlib.util.spec_from_file_location("knowledge_benchmark_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(expected=None):
    return {"work_scope_id": "work-a", "queries": [{"id": "q1", "query": "暂停时间", "split": "held_out",
            "expected": expected or {"knowledge_ids": None, "target_refs": None, "record_ids": None}}]}


def response():
    return {"knowledge_hits": [{"card_id": "card-a", "work_scope_id": "work-a", "target_ref": {
        "kind": "concept", "concept_id": "concept-a"}, "sources": [{"source_id": "source-a", "snapshot_id": "snapshot-a",
        "start": 1, "end": 3, "quote": "时间", "url": "https://example.org/manual"}]}],
        "results": [{"id": "record-a", "asset_id": "asset-a", "start_ms": 100, "end_ms": 300,
                     "match_basis": "setting_concept_appearance", "match_type": "partial", "event_verified": False}]}


def sources():
    return {"source-a": {"snapshot_id": "snapshot-a", "text": "停时间！", "url": "https://example.org/manual"}}


def scored(benchmark, expected=None, actual=None):
    return benchmark.score(manifest(expected)["queries"][0], actual if actual is not None else response(),
                           "work-a", {"asset-a": "work-a"}, sources())


def trial_row(benchmark, mode, *, used=True, calls=None):
    expected = {"knowledge_ids": ["card-a"], "target_refs": [{"kind": "concept", "concept_id": "concept-a"}],
                "record_ids": ["record-a"]}
    return {"query_id": "q1", "query": "暂停时间", "split": "held_out", "mode": mode, "regime": "warm",
            "http_status": 200, "warmup": False, "end_to_end_ms": 12,
            "knowledge_diagnostics": {"generation": "g1", "calls": calls or []} if used else {"degraded": ["not built"]},
            "http_diagnostics": {}, "lightrag_used": mode != "baseline" and used,
            "degraded": [], "scores": scored(benchmark, expected)}


def run(rows, signature="same"):
    return {"rows": rows, "formal_signatures_before": {"work-a": signature},
            "formal_signatures_after": {"work-a": signature}, "formal_inputs_unchanged": True,
            "disk": {}, "memory": {}}


def test_validation_and_scoring_distinguish_pending_from_confirmed_empty(benchmark):
    pending = manifest()
    assert benchmark.validate(pending)["queries_with_any_gold"] == 0
    scores = scored(benchmark)
    assert scores["knowledge"] == {"annotated": False}
    assert scores["has_any_gold"] is False and scores["gold_complete"] is False
    empty = {"knowledge_ids": [], "target_refs": [], "record_ids": []}
    assert benchmark.validate(manifest(empty))["queries_with_any_gold"] == 1
    scores = scored(benchmark, empty, {})
    assert scores["gold_complete"] is True
    assert all(scores[layer]["empty_correct"] for layer in ("knowledge", "targets", "film"))
    broken = manifest()
    broken["work_scope_id"] = None
    with pytest.raises(ValueError, match="work_scope_id"):
        benchmark.validate(broken, run=True)


def test_source_provenance_and_scope_are_checked_from_authoritative_snapshot(benchmark):
    result = scored(benchmark)
    assert result["source_trace_failures"] == 0 and result["cross_work_hits"] == 0
    altered = response()
    altered["knowledge_hits"][0]["sources"][0]["quote"] = "模型重写的原文"
    altered["knowledge_hits"][0]["work_scope_id"] = "work-b"
    altered["results"][0]["asset_id"] = "asset-b"
    result = scored(benchmark, actual=altered)
    assert result["source_trace_failures"] == 1 and result["cross_work_hits"] == 2
    altered["knowledge_hits"][0]["sources"] = []
    assert scored(benchmark, actual=altered)["source_trace_failures"] == 1


def test_partial_appearance_does_not_count_as_verified_event(benchmark):
    expected = {"knowledge_ids": ["card-a"], "target_refs": [{"kind": "concept", "concept_id": "concept-a"}],
                "record_ids": ["record-a"], "film_spans": [{"asset_id": "asset-a", "start_ms": 100, "end_ms": 300}],
                "required_evidence_type": "event"}
    result = scored(benchmark, expected)
    assert result["film"]["hit_at"]["5"] == 0 and result["film_spans"]["hit_at"]["5"] == 0
    assert result["ineligible_film_results"] == 1
    asserted = response()
    asserted["results"][0]["event_verified"] = True
    assert scored(benchmark, expected, asserted)["film"]["hit_at"]["5"] == 1
    negative = {"record_ids": [], "event_assertion_allowed": False}
    assert scored(benchmark, negative)["wrong_event_assertions"] == 0
    assert scored(benchmark, negative, asserted)["wrong_event_assertions"] == 1
    assert scored(benchmark, negative)["film"]["empty_correct"] is False  # related appearance is still a returned candidate


def test_degraded_or_unpaired_or_incomplete_gold_cannot_claim_improvement(benchmark):
    baseline = trial_row(benchmark, "baseline")
    experiment = trial_row(benchmark, "lightrag")
    assert benchmark.summarize([run([baseline, experiment])], manifest())["ready_for_effect_claims"] is True
    degraded = trial_row(benchmark, "lightrag", used=False)
    report = benchmark.summarize([run([baseline, degraded])], manifest())
    assert report["ready_for_effect_claims"] is False and report["lightrag_missing_or_degraded"] is True
    assert benchmark.summarize([run([baseline])], manifest())["ready_for_effect_claims"] is False
    incomplete = copy.deepcopy(experiment)
    incomplete["scores"] = scored(benchmark, {"record_ids": ["record-a"]})
    assert benchmark.summarize([run([baseline, incomplete])], manifest())["quality_annotations_pending"] is True
    changed = [run([baseline], "before"), run([experiment], "after")]
    assert benchmark.summarize(changed, manifest())["ready_for_effect_claims"] is False


def test_unknown_attempts_tokens_and_cost_are_not_recorded_as_zero(benchmark):
    unknown = {"event": "model_finished", "purpose": "knowledge_keyword", "request_count": None,
               "usage": None, "cost": None}
    measured = {"event": "model_finished", "purpose": "knowledge_embed_query", "request_count": 2,
                "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}, "cost": .0001}
    row = trial_row(benchmark, "lightrag", calls=[unknown])
    report = benchmark.summarize([run([row])], manifest())
    accounting = report["groups"][0]["calls"]
    assert accounting["known_attempts"] is None and accounting["known_cost"] is None
    assert accounting["known_tokens"]["total_tokens"] is None
    assert not accounting["attempts_complete"] and not accounting["tokens_complete"] and not accounting["cost_complete"]
    row["knowledge_diagnostics"]["calls"].append(measured)
    accounting = benchmark.summarize([run([row])], manifest())["groups"][0]["calls"]
    assert accounting["known_attempts"] == 2 and accounting["known_cost"] == .0001
    assert accounting["known_tokens"]["total_tokens"] == 10 and not accounting["tokens_complete"]
    assert benchmark.known_sum([]) == 0 and benchmark.known_sum([0]) == 0


def test_warmup_cost_is_included_but_not_latency_or_quality_count(benchmark):
    measured = trial_row(benchmark, "lightrag")
    warmup = copy.deepcopy(measured)
    warmup.update(warmup=True, end_to_end_ms=9999)
    warmup["knowledge_diagnostics"]["calls"] = [{"event": "model_finished", "purpose": "knowledge_keyword", "cost": .01}]
    report = benchmark.summarize([run([warmup, measured])], manifest())
    assert report["groups"][0]["latency"]["n"] == 1
    assert report["groups"][0]["latency"]["median_ms"] == 12
    assert report["experiment_calls_including_warmup"]["known_cost"] == .01


def test_progress_is_complete_jsonl_before_final_result(benchmark, tmp_path):
    path = tmp_path / "progress.partial.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        benchmark.append_progress(stream, {"query_id": "q1", "response": {"text": "原文"}})
        assert json.loads(path.read_text(encoding="utf-8"))["response"]["text"] == "原文"
        benchmark.append_progress(stream, {"query_id": "q2"})
    assert [json.loads(line)["query_id"] for line in path.read_text(encoding="utf-8").splitlines()] == ["q1", "q2"]


def test_stop_only_explicit_worker_and_descendant(benchmark):
    import psutil
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    child_id = int(process.stdout.readline().strip())
    try:
        benchmark.stop_worker_tree(process)
        assert process.poll() is not None
        assert not psutil.pid_exists(child_id) or psutil.Process(child_id).status() == psutil.STATUS_ZOMBIE
    finally:
        if process.poll() is None:
            benchmark.stop_worker_tree(process)
        process.stdout.close()
