#!/usr/bin/env python3
"""Paired knowledge experiments on an explicitly supplied isolated library.

Queries may call configured text and embedding providers. This script never
creates reviewed facts, gold labels or film bindings. Each measured run imports
the real app in a fresh process, keeps persistent query caches, and records
degradation so an unavailable backend cannot masquerade as a LightRAG result.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_progress(stream, row):
    """Keep completed queries inspectable after an interrupted long run."""
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def stop_worker_tree(process):
    """Terminate only the explicitly launched benchmark child and descendants."""
    import psutil
    try:
        parent = psutil.Process(process.pid)
        targets = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        targets = []
    for target in reversed(targets):
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=3)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=3)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_worker(command, log_path):
    process = subprocess.Popen(command, text=True, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        stop_worker_tree(process)
        stdout, stderr = process.communicate()
        Path(log_path).write_text(stdout + stderr + "\nBenchmark interrupted; child tree stopped.\n", encoding="utf-8")
        raise
    Path(log_path).write_text(stdout + stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Experiment worker failed; inspect {log_path}")


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def target_key(ref):
    if ref["kind"] == "concept":
        return "concept:" + ref["concept_id"]
    return "character:" + ref["character_scope_id"] + ":" + ref["character_id"]


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    offset = (len(values) - 1) * fraction
    low = int(offset)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (offset - low)


def distribution(values):
    return {"n": len(values), "median_ms": statistics.median(values) if values else None,
            "p95_ms": percentile(values, .95), "min_ms": min(values) if values else None,
            "max_ms": max(values) if values else None}


def layer_metrics(actual, expected):
    if expected is None:
        return {"annotated": False}
    expected = set(expected)
    if not expected:
        return {"annotated": True, "expected_empty": True, "empty_correct": not actual}
    result = {"annotated": True, "expected_empty": False, "hit_at": {}, "recall_at": {}}
    for k in (1, 5, 10):
        count = len(set(actual[:k]) & expected)
        result["hit_at"][str(k)] = int(count > 0)
        result["recall_at"][str(k)] = count / len(expected)
    result["reciprocal_rank"] = next((1 / rank for rank, value in enumerate(actual, 1) if value in expected), 0)
    return result


def known_sum(values):
    """An absent measurement is unknown; an empty set of calls truly costs 0."""
    values = list(values)
    measured = [value for value in values if value is not None]
    return sum(measured) if measured or not values else None


def score(question, response, scope, memberships, source_lookup):
    expected = question.get("expected", {})
    knowledge = response.get("knowledge_hits", [])
    film = response.get("results", [])
    cards = list(dict.fromkeys(row["card_id"] for row in knowledge))
    targets = list(dict.fromkeys(target_key(row["target_ref"]) for row in knowledge))
    requirement = expected.get("required_evidence_type")
    eligible = film
    if requirement == "event":
        eligible = [row for row in film if row.get("event_verified") is True]
    elif requirement == "subtitle_mention":
        eligible = [row for row in film if row.get("kind") == "subtitle"]
    records = [row["id"] for row in eligible]
    target_gold = expected.get("target_refs")
    trace_failures = 0
    for hit in knowledge:
        if not hit.get("sources"):
            trace_failures += 1
        for source in hit.get("sources", []):
            current = source_lookup.get(source.get("source_id"))
            if (not current or current["snapshot_id"] != source.get("snapshot_id")
                    or current["text"][source["start"]:source["end"]] != source.get("quote")
                    or current["url"] != source.get("url")):
                trace_failures += 1
    cross_work = sum(hit.get("work_scope_id") != scope for hit in knowledge)
    cross_work += sum(memberships.get(hit.get("asset_id")) != scope for hit in film)
    forbidden = {target_key(ref) for ref in expected.get("forbidden_confirmed_target_refs", [])}
    asserted = {"concept:" + concept for hit in film for concept in hit.get("matched_concept_ids", [])}
    asserted.update("character:" + hit.get("matched_character_scope_id", "") + ":" + character
                    for hit in film for character in hit.get("matched_character_ids", []))
    results = {"knowledge": layer_metrics(cards, expected.get("knowledge_ids")),
               "targets": layer_metrics(targets, [target_key(ref) for ref in target_gold] if target_gold is not None else None),
               "film": layer_metrics(records, expected.get("record_ids")),
               "source_trace_failures": trace_failures, "cross_work_hits": cross_work,
               "wrong_identity_confirmations": len(forbidden & asserted) if "forbidden_confirmed_target_refs" in expected else None,
               "wrong_event_assertions": sum(bool(hit.get("event_verified")) for hit in film)
               if expected.get("event_assertion_allowed") is False else None}
    if requirement:
        results["required_evidence_type"] = requirement
        results["ineligible_film_results"] = len(film) - len(eligible)
    if expected.get("film_spans") is not None:
        gold = expected["film_spans"]
        matched = []
        for hit in eligible:
            matched.append(any(hit.get("asset_id") == span["asset_id"]
                               and max(hit["start_ms"], span["start_ms"]) < min(hit["end_ms"], span["end_ms"])
                               for span in gold))
        results["film_spans"] = {"annotated": True, "expected_empty": not gold,
            "empty_correct": not eligible if not gold else None,
            "hit_at": {str(k): int(any(matched[:k])) for k in (1, 5, 10)} if gold else None}
    results["has_any_gold"] = any(results[layer].get("annotated") for layer in ("knowledge", "targets", "film")) or expected.get("film_spans") is not None
    results["gold_complete"] = (results["knowledge"]["annotated"] and results["targets"]["annotated"]
                                and (results["film"]["annotated"] or expected.get("film_spans") is not None))
    return results


def disk_bytes(library, scopes):
    def size(path):
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) if path.exists() else 0
    result = {}
    for scope in sorted(scopes):
        entity = library / "works" / "entities" / scope
        result[scope] = {"source_snapshots_bytes": size(entity / "knowledge"),
                         "catalog_bytes": (entity / "catalog.json").stat().st_size if (entity / "catalog.json").exists() else 0,
                         "derived_index_bytes": size(library / "works" / "knowledge-indexes" / scope)}
    return {"by_scope": result, "shared_receipts_bytes": size(library / "works" / "knowledge-indexes" / "receipts"),
            "excluded": ["original videos", "model files", "existing film indexes", "reference images"],
            "catalog_note": "Whole shared catalog size; use before/after delta for new formal metadata overhead."}


class MemorySampler:
    def __init__(self):
        self.values = defaultdict(int)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        import psutil
        process = psutil.Process()
        while not self.stop.is_set():
            try:
                main = process.memory_info().rss
                knowledge = other = 0
                for child in process.children(recursive=True):
                    try:
                        rss = child.memory_info().rss
                        if "knowledge_worker" in " ".join(child.cmdline()):
                            knowledge += rss
                        else:
                            other += rss
                    except psutil.Error:
                        pass
                for key, value in (("main_rss_peak_bytes", main), ("knowledge_worker_rss_peak_bytes", knowledge),
                                   ("other_children_rss_peak_bytes", other), ("process_tree_rss_peak_bytes", main + knowledge + other)):
                    self.values[key] = max(self.values[key], value)
            except psutil.Error:
                pass
            self.stop.wait(.02)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        return {**self.values, "sampling_interval_ms": 20, "method": "sampled process-tree RSS; not allocator/GPU peak"}


def validate(manifest, *, run=False):
    questions = [row for row in manifest.get("queries", []) if row.get("enabled", True)]
    if not questions or len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Manifest needs uniquely identified enabled queries")
    if any(not row.get("query", "").strip() or row.get("split") not in {"development", "held_out"} for row in questions):
        raise ValueError("Every query needs text and development/held_out split")
    if run and any(not (row.get("work_scope_id") or manifest.get("work_scope_id")) for row in questions):
        raise ValueError("Assign real work_scope_id values before running the experiment")
    return {"queries": len(questions), "disabled_scenarios": len(manifest["queries"]) - len(questions),
            "split": {split: sum(row["split"] == split for row in questions) for split in ("development", "held_out")},
            "queries_with_any_gold": sum(any(row.get("expected", {}).get(key) is not None for key in
                                             ("knowledge_ids", "target_refs", "record_ids", "film_spans")) for row in questions)}


async def worker(args):
    sys.path.insert(0, str(ROOT / "backend"))
    import httpx
    from scenerecall.main import create_app
    from scenerecall.providers import ProviderManager
    providers = ProviderManager(args.profiles_dir) if args.profiles_dir else None
    app = create_app(Path(args.library), start_worker=False, token="knowledge-benchmark-session", providers=providers)
    runtime = app.state.knowledge_runtime
    if args.worker_mode == "naive":
        original = runtime.query

        async def naive(projection, query):
            return await original(projection, query, mode="naive")
        runtime.query = naive
    if args.disable_visual:
        app.state.search.visual_encoder = None
    manifest = read(args.manifest)
    questions = [row for row in manifest["queries"] if row.get("enabled", True)]
    if args.split != "all":
        questions = [row for row in questions if row["split"] == args.split]
    knowledge = app.state.knowledge
    scopes = {row.get("work_scope_id") or manifest["work_scope_id"] for row in questions}
    projections = {scope: knowledge.projection(scope) for scope in scopes}
    before = {scope: projection["signature"] for scope, projection in projections.items()}
    source_lookup = {source["source_id"]: knowledge.source_content(scope, source["source_id"])
                     for scope in scopes for source in knowledge.sources(scope) if source["status"] == "active"}
    memberships = knowledge.store._memberships()["assets"]
    memory = MemorySampler()
    memory.start()
    rows = []
    progress_path = Path(args.output_dir) / f"{args.worker_mode}-{args.run_kind}-{args.run_id}.partial.jsonl"
    progress = progress_path.open("x", encoding="utf-8")
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765",
                                   headers={"authorization": "Bearer knowledge-benchmark-session"}, timeout=None) as client:
            repeats = 1 if args.run_kind == "cold" else args.repeats
            # Warm-up responses are saved but excluded from measurement. Each
            # query is warmed once; persistent provider/RAG caches are preserved.
            schedules = [(-1, questions)] if args.run_kind == "warm" else []
            schedules.extend((repeat, questions) for repeat in range(repeats))
            for repeat, batch in schedules:
                for number, question in enumerate(batch):
                    scope = question.get("work_scope_id") or manifest["work_scope_id"]
                    if args.run_kind == "cold":
                        await runtime.close()
                        app.state.search.knowledge_provider.baseline.clear()
                        cache = getattr(app.state.search.visual_index, "matrix_cache", None)
                        if cache:
                            cache.clear()
                    payload = {"query": question["query"], "filters": question.get("filters", {}), "limit": 50,
                               "knowledge_scope_id": scope, "knowledge_mode": "baseline" if args.worker_mode == "baseline" else "lightrag",
                               "enhancements": False, "visual_verification": False}
                    if question.get("target_ref"):
                        payload["knowledge_target"] = question["target_ref"]
                    started = time.perf_counter()
                    response = await client.post("/api/search", json=payload)
                    duration = (time.perf_counter() - started) * 1000
                    value = response.json()
                    diagnostics = value.get("knowledge_diagnostics", {})
                    if args.worker_mode != "baseline" and "calls" not in diagnostics:
                        diagnostics = {**diagnostics, "calls": copy.deepcopy(runtime._ledger), "partial_receipt": True}
                    row = {"query_id": question["id"], "query": question["query"], "split": question["split"],
                           "category": question.get("category"), "mode": args.worker_mode, "regime": args.run_kind,
                           "round": args.run_id, "repeat": repeat, "warmup": repeat == -1,
                           "http_status": response.status_code, "end_to_end_ms": duration,
                           "knowledge_diagnostics": diagnostics,
                           "http_diagnostics": getattr(app.state, "search_http_diagnostics", {}),
                           "degraded": value.get("degraded", []),
                           "lightrag_used": bool(diagnostics.get("generation")) if args.worker_mode != "baseline" else False,
                           "cache_condition": "persistent text/keyword caches preserved; OS file cache uncontrolled",
                           "process_first_query": repeat == (-1 if args.run_kind == "warm" else 0) and number == 0,
                           "response": value}
                    row["scores"] = score(question, value, scope, memberships, source_lookup) if response.is_success else None
                    rows.append(row)
                    append_progress(progress, row)
                    print(json.dumps({"query_id": row["query_id"], "mode": row["mode"], "repeat": repeat,
                                      "http_status": row["http_status"], "end_to_end_ms": duration}), flush=True)
        after = {scope: knowledge.projection(scope)["signature"] for scope in scopes}
        result = {"rows": rows, "formal_signatures_before": before, "formal_signatures_after": after,
                  "formal_inputs_unchanged": before == after, "disk": disk_bytes(Path(args.library), scopes)}
    finally:
        progress.close()
        await runtime.close()
        if app.state.search.visual_encoder:
            await app.state.search.visual_encoder.close()
        result_memory = memory.finish()
    result["memory"] = result_memory
    write(Path(args.output_dir) / (args.worker_mode + "-" + args.run_kind + "-" + str(args.run_id) + ".json"), result)


def summarize(runs, manifest):
    grouped = defaultdict(list)
    for run in runs:
        for row in run["rows"]:
            if not row["warmup"]:
                grouped[(row["mode"], row["regime"], row["split"])].append(row)
    groups = []
    for (mode, regime, split), rows in sorted(grouped.items()):
        group = {"mode": mode, "regime": regime, "split": split,
                 "distinct_queries": len({row["query_id"] for row in rows}),
                 "latency": distribution([row["end_to_end_ms"] for row in rows]),
                 "http_failures": sum(row["http_status"] != 200 for row in rows),
                 "lightrag_used_trials": sum(row["lightrag_used"] for row in rows),
                 "degraded_trials": sum(bool(row["degraded"]) or bool(row["knowledge_diagnostics"].get("degraded")) for row in rows)}
        scored = [row["scores"] for row in rows if row["scores"]]
        group["safety"] = {key: {"observed_errors": sum(row[key] for row in scored if row.get(key) is not None),
                                 "annotated_trials": sum(row.get(key) is not None for row in scored)} for key in
                           ("source_trace_failures", "cross_work_hits", "wrong_event_assertions", "wrong_identity_confirmations")}
        phase_values = defaultdict(list)
        for row in rows:
            for source, values in (("knowledge", row["knowledge_diagnostics"]), ("http", row["http_diagnostics"])):
                for name, value in values.items():
                    if name.endswith("_ms") and isinstance(value, (int, float)):
                        phase_values[source + "." + name].append(value)
        group["phases"] = {key: distribution(values) for key, values in phase_values.items()}
        for layer in ("knowledge", "targets", "film", "film_spans"):
            values = [row[layer] for row in scored if row.get(layer) and row[layer].get("annotated")]
            positive = [row for row in values if not row.get("expected_empty")]
            negatives = [row for row in values if row.get("expected_empty")]
            group[layer] = {"annotated_trials": len(values), "positive_trials": len(positive),
                            "empty_answer_trials": len(negatives),
                            "empty_answer_accuracy": statistics.mean(row["empty_correct"] for row in negatives) if negatives else None,
                            "hit_at": {str(k): statistics.mean(row["hit_at"][str(k)] for row in positive) if positive else None
                                       for k in (1, 5, 10)}}
        calls = [call for row in rows for call in row["knowledge_diagnostics"].get("calls", [])
                 if call.get("event") == "model_finished"]
        group["calls"] = {"logical_callbacks": len(calls),
                          "known_attempts": known_sum(call.get("request_count") for call in calls),
                          "attempts_complete": all(call.get("request_count") is not None for call in calls),
                          "known_cost": known_sum(call.get("cost") for call in calls),
                          "cost_complete": all(call.get("cost") is not None for call in calls),
                          "by_purpose": {purpose: sum(call.get("purpose") == purpose for call in calls)
                                         for purpose in sorted({call.get("purpose", "unknown") for call in calls})}}
        token_fields = ("input_tokens", "output_tokens", "total_tokens")
        group["calls"]["known_tokens"] = {
            key: sum((call.get("usage") or {}).get(key) or 0 for call in calls)
            if not calls or any((call.get("usage") or {}).get(key) is not None for call in calls) else None
            for key in token_fields}
        group["calls"]["tokens_complete"] = all(all((call.get("usage") or {}).get(key) is not None
                                                       for key in token_fields) for call in calls)
        groups.append(group)
    pending = any(not row.get("scores") or not row["scores"].get("gold_complete", False)
                  for run in runs for row in run["rows"] if not row["warmup"])
    backend_missing = any(row["mode"] != "baseline" and not row["lightrag_used"]
                          for run in runs for row in run["rows"] if not row["warmup"])
    changed = any(not run["formal_inputs_unchanged"] for run in runs)
    signatures = {identity(run["formal_signatures_before"]) for run in runs}
    modes = {row["mode"] for run in runs for row in run["rows"]}
    all_calls = [call for run in runs for row in run["rows"]
                 for call in row["knowledge_diagnostics"].get("calls", []) if call.get("event") == "model_finished"]
    return {"manifest_sha256": identity(manifest), "manifest_status": validate(manifest), "groups": groups,
            "quality_annotations_pending": pending, "lightrag_missing_or_degraded": backend_missing,
            "formal_inputs_changed": changed, "paired_formal_inputs_identical": len(signatures) == 1,
            "ready_for_effect_claims": not pending and not backend_missing and not changed and len(signatures) == 1
            and {"baseline", "lightrag"} <= modes,
            "experiment_calls_including_warmup": {"callbacks": len(all_calls),
                "known_cost": known_sum(call.get("cost") for call in all_calls),
                "cost_complete": all(call.get("cost") is not None for call in all_calls)},
            "scope": "Small-sample evidence; repeat count is not independent query count. Inspect paired raw responses.",
            "memory_runs": [{"memory": run["memory"]} for run in runs], "disk": runs[-1]["disk"] if runs else {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--profiles-dir", type=Path, help="Read existing connection/keyring namespace; do not copy secrets")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", dest="modes", action="append", choices=("baseline", "lightrag", "naive"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--split", choices=("all", "development", "held_out"), default="all")
    parser.add_argument("--disable-visual", action="store_true", help="Knowledge-only diagnostic; apply equally to every mode")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--worker-mode", choices=("baseline", "lightrag", "naive"), help=argparse.SUPPRESS)
    parser.add_argument("--run-kind", choices=("cold", "warm"), help=argparse.SUPPRESS)
    parser.add_argument("--run-id", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = read(args.manifest)
    if args.validate_only:
        print(json.dumps(validate(manifest), ensure_ascii=False))
        return
    validate(manifest, run=True)
    if not args.library or not args.output_dir or args.repeats < 1 or args.cold_runs < 0:
        parser.error("--library, --output-dir, repeats>=1 and cold-runs>=0 are required")
    if args.worker_mode:
        asyncio.run(worker(args))
        return
    library, output = args.library.resolve(strict=True), args.output_dir.resolve()
    if output.exists() or output == library or library in output.parents or output in library.parents:
        raise ValueError("Output must be a new directory outside the library; existing output is never replaced")
    output.mkdir(parents=True)
    write(output / "manifest.json", manifest)
    runs = []
    for mode in args.modes or ("baseline", "lightrag"):
        for kind, count in (("cold", args.cold_runs), ("warm", 1)):
            for number in range(count):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker-mode", mode, "--run-kind", kind,
                           "--run-id", str(number), "--library", str(library), "--manifest", str(args.manifest.resolve()),
                           "--output-dir", str(output), "--repeats", str(args.repeats), "--split", args.split]
                if args.disable_visual:
                    command.append("--disable-visual")
                if args.profiles_dir:
                    command.extend(["--profiles-dir", str(args.profiles_dir.resolve(strict=True))])
                run_worker(command, output / f"{mode}-{kind}-{number}.log")
                runs.append(read(output / f"{mode}-{kind}-{number}.json"))
    with (output / "queries.jsonl").open("w", encoding="utf-8") as stream:
        for run in runs:
            for row in run["rows"]:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = summarize(runs, manifest)
    write(output / "summary.json", report)
    print(json.dumps({"output_dir": str(output), "ready_for_effect_claims": report["ready_for_effect_claims"],
                      "groups": len(report["groups"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
