#!/usr/bin/env python3
"""Compare query implementations against copied indexes, without rebuilding images.

Run with explicit baseline/current roots, an isolated source library, encoder
configuration and the original query manifest. Every worker imports exactly one
implementation and disables remote providers. Raw responses are content-addressed
so each timed request can be checked without duplicating large frame manifests.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    offset = (len(values) - 1) * fraction
    left = int(offset)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (offset - left)


def metrics(values):
    return {"count": len(values), "median_ms": statistics.median(values),
            "p95_ms": percentile(values, .95), "min_ms": min(values), "max_ms": max(values)}


def safe_destination(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Benchmark source and output must be separate directory trees")
    if destination.exists():
        raise ValueError(f"Refusing to replace benchmark output: {destination}")


def configure_imports(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / "backend"))
    import scenerecall
    if root not in Path(scenerecall.__file__).resolve().parents:
        raise RuntimeError("Worker imported the wrong application implementation")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return root


def matrix_clear(engine):
    cache = getattr(engine.visual_index, "matrix_cache", None)
    if cache is not None:
        cache.clear()


def matrix_stats(engine):
    cache = getattr(engine.visual_index, "matrix_cache", None)
    if cache is None:
        return {"available": False}
    stats = getattr(cache, "stats", {})
    stats = stats() if callable(stats) else stats
    return {"available": True, **stats}


def clear_text(engine):
    with engine._db() as db:
        db.execute("DELETE FROM visual_query_cache")


def implementation_hashes(root):
    names = ("search.py", "visual_index.py", "query_cache.py", "knowledge_recall.py",
             "work_entities.py", "main.py", "visual_encoder.py", "visual_encoder_worker.py")
    return {name: hashlib.sha256((root / "backend/scenerecall" / name).read_bytes()).hexdigest()
            for name in names if (root / "backend/scenerecall" / name).is_file()}


async def worker(args):
    root = configure_imports(args.implementation_root)
    import httpx
    import numpy as np
    import psutil
    from scenerecall.main import create_app
    from scenerecall.providers import ProviderManager

    class LocalOnlyProviders(ProviderManager):
        async def _request(self, *values, **kwargs):
            raise RuntimeError("Query benchmark prohibits remote model calls")

        async def _chat(self, *values, **kwargs):
            raise RuntimeError("Query benchmark prohibits generative model calls")

    destination = Path(args.output_dir).resolve()
    safe_destination(args.source_library, destination)
    library = destination / "library"
    shutil.copytree(args.source_library, library)
    shutil.copy2(args.encoder_config, library / "private/retrieval-encoder.json")
    manifest = read(args.manifest)
    asset_id = manifest["clips"][0]["asset_id"]
    questions = [row["query"] for row in manifest["clips"][0]["queries"]]
    app = create_app(library, start_worker=False, providers=LocalOnlyProviders(library / "private"),
                     token="query-benchmark-local")
    engine = app.state.search
    encoder = engine.visual_encoder
    report = {"status": "running", "implementation": args.label, "trial": args.trial,
              "source_hashes": implementation_hashes(root), "queries": questions, "samples": [], "responses": {},
              "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              "source_index_sha256": hashlib.sha256(
                  (Path(args.source_library) / "indexes/search.sqlite").read_bytes()).hexdigest(),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "sqlite": sqlite3.sqlite_version, "platform": platform.platform()},
              "model_warmed": True, "remote_model_calls_allowed": False,
              "http_boundary": "ASGI transport includes middleware, actual route and JSON; excludes network/UI",
              "timing_boundary": "per-request wall clock; excludes library copy, imports and model warmup"}
    gpu_tool = shutil.which("nvidia-smi")
    if gpu_tool:
        gpu = subprocess.run([gpu_tool, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        report["environment"]["gpu"] = gpu.stdout.strip() if not gpu.returncode else "unavailable"
    process = psutil.Process()
    result_path = destination / "result.json"
    started = time.perf_counter()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765",
                              headers={"authorization": "Bearer query-benchmark-local"})
    try:
        before = time.perf_counter()
        await encoder.encode_texts(["benchmark warmup"])
        report["model_warmup_ms"] = (time.perf_counter() - before) * 1000
        report["encoder_runtime"] = dict(encoder.metadata)
        with engine._db() as db:
            report["index"] = {"vectors": db.execute("SELECT COUNT(*) FROM visual_cache").fetchone()[0],
                               "windows": db.execute("SELECT COUNT(*) FROM visual_records").fetchone()[0]}

        async def invoke(mode, question):
            if mode == "http":
                response = await client.post("/api/search", json={"query": question,
                    "filters": {"asset_id": asset_id}, "limit": 5, "enhancements": False,
                    "visual_verification": False})
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
                return response.json()
            if mode == "native":
                return await engine.search(question, filters={"asset_id": asset_id}, limit=5)
            degraded = []
            rows = await engine.visual_index.search(question, {"asset_id": asset_id}, encoder, degraded, limit=5)
            return {"results": rows, "degraded": degraded}

        async def measure(mode, question_id, case, repetition):
            before = time.perf_counter()
            response = await invoke(mode, questions[question_id])
            elapsed = (time.perf_counter() - before) * 1000
            identity = digest(response)
            report["responses"].setdefault(identity, response)
            report["samples"].append({"mode": mode, "question_id": question_id, "case": case,
                "repetition": repetition, "elapsed_ms": elapsed, "response_sha256": identity,
                "engine_diagnostics": json.loads(json.dumps(getattr(engine, "query_diagnostics", {})))
                    if mode != "visual_only" else {},
                "http_diagnostics": dict(getattr(app.state, "search_http_diagnostics", {}))
                    if mode == "http" else {}})

        for mode in ("native", "visual_only", "http"):
            for question_id in range(len(questions)):
                matrix_clear(engine)
                clear_text(engine)
                await measure(mode, question_id, "cold_matrix_uncached_text", 0)
                clear_text(engine)
                await measure(mode, question_id, "warm_matrix_uncached_text", 0)
                for repetition in range(args.repeats):
                    await measure(mode, question_id, "warm_matrix_cached_text", repetition)
                write(result_path, report)
            print(json.dumps({"event": "mode_complete", "implementation": args.label,
                              "trial": args.trial, "mode": mode}), flush=True)
        report["matrix_cache"] = matrix_stats(engine)
        report["process_memory"] = process.memory_info()._asdict()
        report["status"] = "completed"
    finally:
        await client.aclose()
        await encoder.close()
        report["worker_closed"] = True
        report["total_worker_ms"] = (time.perf_counter() - started) * 1000
        write(result_path, report)
    return report


def summarize(output):
    output = Path(output)
    reports = [read(path) for path in sorted(output.glob("pair-*/*/result.json"))]
    if not reports or any(row["status"] != "completed" for row in reports):
        raise ValueError("All requested benchmark workers must finish before comparison")
    grouped, per_pair, fingerprints = {}, {}, {}
    mismatches = []
    for report in reports:
        label = report["implementation"]
        for sample in report["samples"]:
            key = (sample["mode"], sample["case"])
            grouped.setdefault((label, *key), []).append(sample["elapsed_ms"])
            per_pair.setdefault((report["trial"], label, *key), []).append(sample["elapsed_ms"])
            response_key = (sample["mode"], sample["question_id"])
            previous = fingerprints.setdefault(response_key, sample["response_sha256"])
            if previous != sample["response_sha256"]:
                mismatches.append({"trial": report["trial"], "implementation": label, **sample})
    comparisons = []
    for mode, case in sorted({key[1:] for key in grouped}):
        baseline = metrics(grouped[("baseline", mode, case)])
        current = metrics(grouped[("current", mode, case)])
        pairs = []
        for trial in sorted({row["trial"] for row in reports}):
            before = metrics(per_pair[(trial, "baseline", mode, case)])
            after = metrics(per_pair[(trial, "current", mode, case)])
            pairs.append({"trial": trial, "baseline": before, "current": after,
                          "median_reduction_percent": 100 * (1 - after["median_ms"] / before["median_ms"])})
        comparisons.append({"mode": mode, "case": case, "baseline": baseline, "current": current,
                            "median_reduction_percent": 100 * (1 - current["median_ms"] / baseline["median_ms"]),
                            "paired_trials": pairs})
    stable_sources = all(len({digest(row["source_hashes"]) for row in reports if row["implementation"] == label}) == 1
                         for label in ("baseline", "current"))
    result = {"schema_version": 1, "status": "passed" if not mismatches else "result_mismatch",
              "pairs": len(reports) // 2, "requests_compared": sum(len(row["samples"]) for row in reports),
              "queries": len({sample["question_id"] for sample in reports[0]["samples"]}),
              "query_texts": reports[0].get("queries", []), "comparisons": comparisons, "mismatches": mismatches,
              "full_responses_equal": not mismatches, "source_files_stable_between_trials": stable_sources,
              "comparison_definition": "All public fields, scores, ordering and evidence references; timing stored separately",
              "baseline_source_hashes": reports[0]["source_hashes"],
              "current_source_hashes": next(row["source_hashes"] for row in reports if row["implementation"] == "current")}
    write(output / "comparison.json", result)
    if not stable_sources:
        raise RuntimeError("Implementation source changed between trials; rerun the final paired experiment")
    return result


def run_pairs(args):
    output = Path(args.output_dir).resolve()
    safe_destination(args.source_library, output)
    output.mkdir(parents=True)
    source = Path(__file__).resolve()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1")
    for trial in range(1, args.pairs + 1):
        order = ("baseline", "current") if trial % 2 else ("current", "baseline")
        for label in order:
            destination = output / f"pair-{trial}" / label
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(source), "--worker", "--label", label, "--trial", str(trial),
                       "--implementation-root", str(getattr(args, label + "_root")),
                       "--source-library", str(args.source_library), "--encoder-config", str(args.encoder_config),
                       "--manifest", str(args.manifest), "--output-dir", str(destination),
                       "--repeats", str(args.repeats)]
            with (destination.parent / f"{label}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            if completed.returncode:
                raise RuntimeError(f"{label} trial {trial} failed; inspect {destination.parent / (label + '.log')}")
            print(json.dumps({"event": "worker_complete", "trial": trial, "label": label}), flush=True)
    summary = summarize(output)
    print(json.dumps({key: summary[key] for key in ("status", "pairs", "requests_compared", "full_responses_equal")}),
          flush=True)
    if not summary["full_responses_equal"]:
        raise RuntimeError("Query results differ; inspect comparison.json before accepting performance")


async def scale_worker(args):
    """Synthetic repeated vectors measure mechanics, never semantic quality."""
    root = configure_imports(args.implementation_root)
    import numpy as np
    import psutil
    from scenerecall.search import SearchEngine
    from scenerecall.visual_index import evidence_digest

    destination = Path(args.output_dir).resolve()
    safe_destination(args.source_library, destination)
    destination.mkdir(parents=True)
    with sqlite3.connect(f"file:{(Path(args.source_library) / 'indexes/search.sqlite').as_posix()}?mode=ro",
                         uri=True) as db:
        signature, identity_text, dimensions = db.execute(
            "SELECT signature,identity,dimensions FROM visual_namespaces LIMIT 1").fetchone()
        source_vectors = [row[0] for row in db.execute(
            "SELECT vector FROM visual_cache WHERE signature=? ORDER BY cache_key", (signature,))]
        query_vector = db.execute("SELECT vector FROM visual_query_cache WHERE signature=? LIMIT 1",
                                  (signature,)).fetchone()
    if not source_vectors:
        raise ValueError("Scaling requires a previously measured visual index")
    query_vector = np.frombuffer(query_vector[0] if query_vector else source_vectors[0], dtype="<f4")

    class FixedEncoder:
        def identity(self):
            return json.loads(identity_text)

        async def encode_texts(self, texts):
            return [query_vector.tolist() for _ in texts]

    engine = SearchEngine(destination / "indexes", None)
    engine.visual_encoder = FixedEncoder()
    rows = []
    with engine._db() as db:
        db.execute("INSERT INTO visual_namespaces VALUES (?,?,?,?)", (signature, identity_text, dimensions, 0))
        for start in range(0, args.vectors, 20):
            frames = [{"id": f"frame-{number:07d}", "at_ms": number * 100, "pts": number,
                       "time_base": [1, 10]} for number in range(start, min(start + 20, args.vectors))]
            record_id = f"window-{start // 20:07d}"
            row = {"id": record_id, "asset_id": "synthetic-scale", "kind": "visual",
                   "evidence_type": "retrieval_window", "source_fingerprint": "synthetic-only",
                   "text": "", "start_ms": start * 100, "end_ms": (start + len(frames)) * 100,
                   "frame_ids": [frame["id"] for frame in frames],
                   "evidence_frame_ids": [frame["id"] for frame in frames], "ordered_frames": frames}
            engine._write_record(db, row)
            db.execute("INSERT INTO visual_records VALUES (?,?,?,?)",
                       (signature, record_id, row["asset_id"], evidence_digest(row)))
            for offset, frame in enumerate(frames):
                cache_key = frame["id"]
                db.execute("INSERT INTO visual_cache VALUES (?,?,?,?)",
                           (signature, cache_key, dimensions, source_vectors[(start + offset) % len(source_vectors)]))
                db.execute("INSERT INTO visual_links VALUES (?,?,?,?,?)",
                           (signature, record_id, frame["id"], cache_key, frame["at_ms"]))
            rows.append(record_id)
    report = {"implementation": args.label, "vectors": args.vectors, "windows": len(rows),
              "dimensions": dimensions, "synthetic": True, "quality_claim": False,
              "source_hashes": implementation_hashes(root), "samples": [], "responses": {}}
    try:
        for repetition in range(args.repeats + 1):
            started = time.perf_counter()
            response = await engine.search("synthetic scaling query", {"asset_id": "synthetic-scale"}, limit=50)
            elapsed = (time.perf_counter() - started) * 1000
            identity = digest(response)
            report["responses"].setdefault(identity, response)
            report["samples"].append({"case": "cold" if repetition == 0 else "warm",
                                      "elapsed_ms": elapsed, "response_sha256": identity})
        report["matrix_cache"] = matrix_stats(engine)
        report["process_memory"] = psutil.Process().memory_info()._asdict()
        report["status"] = "completed"
    finally:
        write(destination / "result.json", report)
    print(json.dumps({"event": "scale_complete", "implementation": args.label, "vectors": args.vectors,
                      "warm": metrics([sample["elapsed_ms"] for sample in report["samples"][1:]])}), flush=True)


def entity_benchmark(args):
    """Compare authoritative legacy traversal and the occurrence provider in one fixture."""
    root = configure_imports(args.implementation_root)
    import copy
    from PIL import Image
    from scenerecall.knowledge_recall import WorkEntityRecall
    from scenerecall.library import Library, atomic_json
    from scenerecall.retrieval_store import RetrievalStore
    from scenerecall.search import SearchEngine
    from scenerecall.work_entities import WorkEntityStore

    destination = Path(args.output_dir).resolve()
    if destination.exists():
        raise ValueError("Entity benchmark output already exists")
    library = Library(destination / "library")
    asset_id = "entity-benchmark"
    atomic_json(library.asset_dir(asset_id) / "manifest.json", {
        "schema_version": "1.0", "id": asset_id, "work_id": "entity-work", "title": "Synthetic entity fixture",
        "kind": "animation", "created_at": "2026-10-28T00:00:00Z", "duration_ms": 5000,
        "version": "test", "fingerprint": "entity-fixture-source"})
    store = WorkEntityStore(library)
    store.create_scope("Synthetic entity fixture", "test", [asset_id])
    image_path = library.asset_dir(asset_id) / "frames/registered-frame.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 64), (90, 30, 150)).save(image_path)
    frame = {"id": "registered-frame", "path": str(image_path), "at_ms": 200,
             "pts": 200, "time_base": [1, 1000]}
    library.register_frame(asset_id, frame)
    row = {"schema_version": "1.0", "id": "window-one", "asset_id": asset_id,
           "source_fingerprint": "entity-fixture-source", "strategy": "retrieval-first-v1", "kind": "visual",
           "evidence_type": "retrieval_window", "text": "圆形金色物体", "start_ms": 0, "end_ms": 1000,
           "frame_ids": [frame["id"]], "evidence_frame_ids": [frame["id"]],
           "ordered_frames": [{key: frame[key] for key in ("id", "at_ms", "pts", "time_base")}],
           "entities": [{"id": "local-object", "name": "圆形物体", "description": "金色表盘"}]}
    RetrievalStore(library).publish(asset_id, "entity-run", row)
    concept = store.create(asset_id, {"type": "prop", "name": "时间道具", "original_name": "時の道具",
        "aliases": ["秒表"], "appearance": "金色圆形表盘",
        "sources": [{"id": "manual", "kind": "manual", "title": "Synthetic fixture"}],
        "facts": [{"id": "stop", "kind": "function", "text": "能停止时间", "source_ids": ["manual"]}]})
    engine = SearchEngine(destination / "indexes", None)
    records = [row]
    for number in range(300):
        unrelated = copy.deepcopy(row)
        unrelated["id"] = f"unbound-{number}"
        records.append(unrelated)
    engine.upsert(records)
    store.bind(asset_id, {"record_id": row["id"], "entity_id": "local-object", "concept_id": concept["concept_id"],
        "status": "confirmed", "evidence_frame_ids": [frame["id"]], "reason": "Synthetic confirmed binding"})
    provider = WorkEntityRecall(store)
    filters = {"asset_id": asset_id}
    report = {"synthetic": True, "quality_claim": False, "eligible_records": len(records), "confirmed_bindings": 1,
              "fixture": "Real registered image and validated binding; 300 unbound index distractors",
              "source_hashes": implementation_hashes(root), "samples": [], "responses": {}}
    for question in ("停止时间", "秒表", "金色表盘", "一只恐龙"):
        provider.clear()
        for repetition in range(args.repeats + 1):
            order = ("legacy", "provider") if repetition % 2 == 0 else ("provider", "legacy")
            for mode in order:
                started = time.perf_counter()
                if mode == "legacy":
                    response = store.search_occurrences(question, filters, eligible_records=engine.filtered_records(filters))
                else:
                    response = provider.search(question, filters, engine)
                elapsed = (time.perf_counter() - started) * 1000
                identity = digest(response)
                report["responses"].setdefault(identity, response)
                report["samples"].append({"query": question, "mode": mode,
                    "case": "cold" if repetition == 0 else "warm", "repetition": repetition,
                    "elapsed_ms": elapsed, "response_sha256": identity})
    fingerprints = {}
    report["full_responses_equal"] = True
    for sample in report["samples"]:
        previous = fingerprints.setdefault(sample["query"], sample["response_sha256"])
        report["full_responses_equal"] &= previous == sample["response_sha256"]
    report["provider_stats"] = dict(provider.stats)
    report["summary"] = {mode: metrics([row["elapsed_ms"] for row in report["samples"]
                        if row["mode"] == mode and row["case"] == "warm"]) for mode in ("legacy", "provider")}
    report["status"] = "passed" if report["full_responses_equal"] else "result_mismatch"
    write(destination / "result.json", report)
    print(json.dumps({key: report[key] for key in ("status", "full_responses_equal", "summary")}), flush=True)
    if not report["full_responses_equal"]:
        raise RuntimeError("Occurrence provider differs from legacy traversal")


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--baseline-root", type=Path)
    value.add_argument("--current-root", type=Path)
    value.add_argument("--source-library", type=Path)
    value.add_argument("--encoder-config", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--pairs", type=int, default=5)
    value.add_argument("--repeats", type=int, default=20)
    value.add_argument("--worker", action="store_true")
    value.add_argument("--scale-worker", action="store_true")
    value.add_argument("--entity-benchmark", action="store_true")
    value.add_argument("--vectors", type=int, default=10000)
    value.add_argument("--summarize", action="store_true")
    value.add_argument("--implementation-root", type=Path)
    value.add_argument("--label", choices=("baseline", "current"))
    value.add_argument("--trial", type=int, default=0)
    return value


if __name__ == "__main__":
    options = parser().parse_args()
    if options.summarize:
        summarize(options.output_dir)
    elif options.entity_benchmark:
        entity_benchmark(options)
    elif options.scale_worker:
        asyncio.run(scale_worker(options))
    elif options.worker:
        asyncio.run(worker(options))
    else:
        run_pairs(options)
