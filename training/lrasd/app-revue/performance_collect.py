"""Collect report evidence without app startup, models, APIs or library writes.

Run again after the target episode finishes. Only performance_* files beside this
script are written. Timestamps from file mtimes are evidence publication times,
not exact worker start/end times. No source-media locations or credentials are
read or emitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


ASSETS = {
    "asset_49d289a62d9740babce8": 1,
    "asset_b0b26aad786d4cb2b620": 2,
    "asset_fb9f73883b304d3c92d3": 3,
}
HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = "scenerecall-performance-v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def seconds(start, end):
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def source(path, *, kind="json", selector=None):
    if path.suffix == ".sqlite" and kind == "json":
        kind = "sqlite"
    result = {"path": str(path.resolve()), "kind": kind}
    if selector:
        result["selector"] = selector
    return result


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default


def file_evidence(path):
    if not path.is_file():
        return {"source": source(path), "exists": False}
    stat = path.stat()
    return {"source": source(path), "exists": True, "bytes": stat.st_size,
            "last_write_utc": iso(stat.st_mtime), "mtime_unix_ns": stat.st_mtime_ns,
            "time_semantics": "last filesystem write; atomic replacement is not stage creation"}


def metric(value, unit, semantics, path, selector, *, limitations=()):
    return {"value": value, "unit": unit, "semantics": semantics,
            "source": source(path, selector=selector), "limitations": list(limitations)}


def jobs_snapshot(library):
    path = library / "runtime/jobs.sqlite"
    # No app imports, constructors, journal_mode changes, migrations or writes.
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5) as database:
        database.execute("PRAGMA query_only=ON")
        database.execute("BEGIN")
        return [row for (raw,) in database.execute("SELECT data FROM jobs ORDER BY rowid")
                if (row := json.loads(raw)).get("asset_id") in ASSETS]


def compact_job(job):
    fields = ("id", "asset_id", "run_id", "type", "status", "stage", "progress", "completed", "total",
              "created_at", "updated_at", "stage_progress", "stage_metrics", "request_count", "error", "error_code")
    result = {key: job.get(key) for key in fields if key in job}
    result["inflight_count"] = len(job.get("inflight_calls", {}))
    return result


def receipt_evidence(directory, job):
    entries, totals, seen = [], defaultdict(Counter), set()
    accounted = set(job.get("scene_receipts", []))
    for path in sorted((directory / "scene-receipts").glob("*.json")):
        raw = read(path, {})
        entry = file_evidence(path)
        for key in ("receipt_id", "call_id", "job_id", "stage", "request_count", "elapsed_ms", "split"):
            if key in raw:
                entry[key] = raw[key]
        entry["owned_by_current_job"] = raw.get("job_id") == job["id"]
        entry["accounted_in_current_job"] = raw.get("receipt_id") in accounted
        entry["provider_success_receipt"] = not raw.get("split") and (raw.get("request_count") or 0) > 0
        entry["interval_estimate"] = None
        if entry["provider_success_receipt"] and raw.get("elapsed_ms") is not None:
            end = path.stat().st_mtime
            entry["interval_estimate"] = {
                "start_utc": iso(end - raw["elapsed_ms"] / 1000), "end_utc": iso(end),
                "semantics": "receipt mtime minus recorded call elapsed; approximate only",
                "limitations": ["Unknown publication/fsync delay; never exact admission time.",
                                "May cover provider retries; only successful receipts are represented."]}
        entries.append(entry)
        identity = raw.get("receipt_id") or path.stem
        if identity in seen:
            entry["duplicate_receipt_id"] = True
            continue
        seen.add(identity)
        aggregate = totals[raw.get("stage", "unknown")]
        aggregate["unique_receipt_count"] += 1
        aggregate["aggregate_split_receipt_count"] += bool(raw.get("split"))
        aggregate["successful_provider_receipt_count"] += entry["provider_success_receipt"]
        if entry["owned_by_current_job"] and entry["provider_success_receipt"]:
            aggregate["owned_success_request_attempts"] += raw.get("request_count") or 0
            aggregate["owned_success_accumulated_elapsed_ms"] += raw.get("elapsed_ms") or 0
    return {"entries": entries, "totals_by_stage": dict(totals),
            "limitations": ["Elapsed sums overlap under concurrency and must not be plotted as elapsed wall time.",
                            "Failed calls without a success receipt are absent; stage_metrics may be larger.",
                            "Zero-call split aggregate receipts are counted but excluded from provider elapsed sums."]}


def cache_evidence(directory):
    paths = list(directory.glob("*.json"))
    times = [path.stat().st_mtime for path in paths]
    return {"source": source(directory, kind="directory", selector="*.json"), "files": len(paths),
            "first_current_file_write_utc": iso(min(times)) if times else None,
            "last_current_file_write_utc": iso(max(times)) if times else None,
            "semantics": "write envelope of currently surviving cache files; includes reuse/repair, not active duration"}


def analysis_evidence(library, job):
    asset_dir = library / "assets" / job["asset_id"]
    directory = asset_dir / "analyses" / job["run_id"]
    db_path = library / "runtime/jobs.sqlite"
    config = job.get("config", {})
    result = {"run_source": source(directory, kind="directory"), "job": compact_job(job),
              "job_source": source(db_path, kind="sqlite", selector=f"jobs[id={job['id']}].data"),
              "configured_media_range_ms": {key: config.get(key) for key in ("start_ms", "end_ms")},
              "stage_timings": {}, "receipts": receipt_evidence(directory, job)}
    endpoint = job.get("updated_at") if job.get("status") in {"completed", "cancelled", "canceled", "failed"} else utc()
    result["lifecycle_envelope"] = {"start_utc": job.get("created_at"), "end_utc": endpoint,
        "seconds": seconds(job.get("created_at"), endpoint),
        "semantics": "created-to-observation (or terminal last-update); includes queue, pause, repair and execution"}
    result["status_consistency_notes"] = []
    if job.get("status") != "running" and any(row.get("status") == "running"
                                               for row in job.get("stage_progress", {}).values()):
        result["status_consistency_notes"].append(
            "Nested running progress is retained from an earlier invocation; job status controls current activity.")
    for stage, values in job.get("stage_metrics", {}).items():
        selector = f"jobs[id={job['id']}].stage_metrics.{stage}"
        result["stage_timings"][stage] = {
            "accumulated_elapsed": metric(values.get("elapsed_ms"), "ms", "accumulated instrumented elapsed",
                db_path, selector + ".elapsed_ms", limitations=["Concurrent/nested calls can overlap.",
                "Includes instrumented failed attempts where persisted; not active CPU or GPU time."]),
            "latest_recorded_wall": metric(values.get("wall_elapsed_ms"), "ms", "latest recorded stage-invocation wall",
                db_path, selector + ".wall_elapsed_ms", limitations=["Assignment, not accumulation: earlier resume invocations may be absent.",
                "Null means not recorded, never zero."]),
            "request_attempts": values.get("request_count", 0), "cache_hits": values.get("cache_hits", 0),
            "failure_counts": values.get("failure_counts", {}), "fallback_count": values.get("fallback_count", 0)}
    artifacts = ("identity-prelude.json", "scene-transcript.json", "scene-preparation-results.json",
                 "scene-preparation.json", "scene-state.json", "lrasd-scene-reviews.json")
    result["artifacts"] = {name: file_evidence(directory / name) for name in artifacts}
    prelude_path = directory / "identity-prelude.json"
    prelude = read(prelude_path, {})
    visual, audio = prelude.get("visual", {}), prelude.get("audio", {})
    result["visual_identity"] = {"status": visual.get("status", "not_persisted"),
        "regions": len(visual.get("regions", [])), "metrics": visual.get("metrics", {}),
        "source": source(prelude_path, selector="/visual"),
        "timing_semantics": "last persisted run result; cached work from earlier attempts is not recomputed",
        "cache": {name: cache_evidence(asset_dir / "identity/visual" / name) for name in ("detections", "regions")}}
    result["audio_identity"] = {"status": audio.get("status", "not_persisted"),
        "record_count": len(audio.get("records", [])), "metrics": audio.get("metrics", {}),
        "source": source(prelude_path, selector="/audio")}
    asd_path = directory / "lrasd-scene-reviews.json"
    asd = read(asd_path, {})
    reviews = [{key: row.get(key) for key in ("scene_id", "start_ms", "end_ms", "status", "cache_hit", "timings")}
               | {"tracks": len(row.get("tracks", []))} for row in asd.get("reviews", [])]
    result["asd_reviews"] = {"status": asd.get("status", "not_started"), "created_at": asd.get("created_at"),
        "updated_at": asd.get("updated_at"), "coverage": asd.get("coverage", {}), "reviews": reviews,
        "status_counts": dict(Counter(row["status"] for row in reviews)),
        "timing_sum_ms": {key: sum((row.get("timings") or {}).get(key, 0) or 0 for row in reviews)
                          for key in sorted({key for row in reviews for key in (row.get("timings") or {})
                                             if key.endswith("_ms")})},
        "source": source(asd_path), "timing_semantics": "sum of persisted sample timings; cache hits may retain original timing",
        "limitations": ["Sampled windows, never full-episode ASD coverage.",
                        "No_tracks is a completed unsuccessful sample, not scored evidence.",
                        "Created-to-updated envelope can include paused time across resumed runs."]}
    result["frozen_input_files"] = len(list((directory / "scene-inputs").glob("*.json")))
    result["split_codes"] = dict(Counter(read(path, {}).get("code", "unknown")
                                       for path in (directory / "scene-splits").glob("*.json")))
    result["provider_models"] = {key: {field: value.get(field) for field in ("provider_type", "model")}
                                  for key, value in config.get("profile_snapshots", {}).items()}
    return result


def historical_audits():
    result = []
    for name in ("preflight-audit.json", "preparation-drain-audit.json", "formal-run-audit.json"):
        path = HERE / name
        raw = read(path)
        if raw is None:
            continue
        entry = {"observed_at_utc": raw.get("checked_at_utc"), "source": source(path), "jobs": []}
        for job in raw.get("jobs", []):
            if isinstance(job, dict):
                entry["jobs"].append({key: job.get(key) for key in
                    ("job_id", "asset_id", "run_id", "status", "stage", "stage_metrics", "asd_status")})
        if "job_status" in raw:
            entry["unscoped_job_observation"] = {key: raw.get(key) for key in
                ("job_status", "stage_progress", "metrics", "inflight_count", "new_runtime_cutoff_utc",
                 "paid_receipt_counts", "split_plan_code_counts")}
            entry["limitations"] = ["Audit file has no job_id; do not assign it automatically to a job."]
        result.append(entry)
    return result


def target_complete(episodes, targets):
    selected = [ep for ep in episodes if ep["episode"] in targets]
    return bool(targets) and len(selected) == len(targets) and all(
        max(ep.get("analysis_runs", []), key=lambda row: row["job"].get("created_at", ""), default={})
        .get("job", {}).get("status") == "completed" for ep in selected)


def run(library):
    started = utc()
    jobs = jobs_snapshot(library)
    annotations = read(HERE / "performance_annotations.json", {})
    scope = annotations.get("completion_scope", {"target_episodes": [1], "excluded_episodes": [2, 3]})
    targets = set(scope["target_episodes"])
    episodes = []
    for asset_id, episode in ASSETS.items():
        manifest_path = library / "assets" / asset_id / "manifest.json"
        manifest = read(manifest_path, {})
        own_jobs = [job for job in jobs if job.get("asset_id") == asset_id]
        analyses = [analysis_evidence(library, job) for job in own_jobs if job.get("type") == "analysis"]
        scans = [compact_job(job) | {"created_to_last_update_seconds": seconds(job.get("created_at"), job.get("updated_at")),
                 "timing_semantics": "job lifecycle envelope includes any queue/pause; active scan time is unknown"}
                 for job in own_jobs if job.get("type") == "keyframe_scan"]
        episodes.append({"episode": episode, "asset_id": asset_id, "title": manifest.get("title"),
                         "completion_role": "target" if episode in targets else "excluded_by_user",
                         "exclusion_reason": None if episode in targets else "用户缩小范围，不再继续；仅保留已发生历史开销",
                         "media_duration_ms": manifest.get("duration_ms"), "manifest_source": source(manifest_path),
                         "scan_jobs": scans, "analysis_runs": analyses})
    after = {row["id"]: row.get("updated_at") for row in jobs_snapshot(library)}
    return {"schema_version": SCHEMA_VERSION, "capture_started_utc": started, "capture_finished_utc": utc(),
        "completion_scope": scope,
        "report_state": "final" if target_complete(episodes, targets) else "partial_requires_final_refresh",
        "timezone_display": "Asia/Shanghai", "library_read_only": True,
        "snapshot_history_source": source(HERE / "performance_snapshots.jsonl", kind="jsonl"),
        "job_snapshot_changed_during_capture": [row["id"] for row in jobs if after.get(row["id"]) != row.get("updated_at")],
        "episodes": episodes, "historical_audit_observations": historical_audits(),
        "annotations": annotations,
        "timeline_policy": {"queue_wait_ms": None, "paused_ms": None, "repair_ms": None, "retry_wall_ms": None,
            "reason": "No durable complete status-transition history; current job timestamps cannot separate these durations.",
            "events_source": source(HERE / "performance_annotations.json", selector="/timeline_events"),
            "rules": ["Use observed snapshot timestamps as transition bounds only, not exact transitions.",
                      "Use receipt intervals only as approximate call intervals.",
                      "Do not add per-episode walls when scans/transcripts overlap.",
                      "Never sum concurrent receipt elapsed to label end-to-end wall time.",
                      "Do not subtract counters from differently scoped measurements to invent repair time."]},
        "instrumentation_sources": [
            {"path": str((HERE.parents[2] / "backend/scenerecall/scene_jobs.py").resolve()),
             "selectors": ["_scene_metric", "_account_scene_receipt", "_scene_call"],
             "explains": "elapsed accumulation versus latest wall assignment and success receipt persistence"}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("D:/SceneRecallLibrary"))
    args = parser.parse_args()
    value = run(args.library.resolve())
    output = HERE / "performance_data.json"
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    snapshot = {"observed_at_utc": value["capture_finished_utc"], "report_state": value["report_state"],
                "jobs": [row["job"] for ep in value["episodes"] for row in ep["analysis_runs"]]}
    with (HERE / "performance_snapshots.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "state": value["report_state"],
                      "episodes": [{"episode": ep["episode"], "jobs": [{"status": row["job"]["status"],
                         "stage_progress": row["job"].get("stage_progress"),
                         "receipt_count": len(row["receipts"]["entries"]), "asd": row["asd_reviews"]["status_counts"]}
                         for row in ep["analysis_runs"]]} for ep in value["episodes"]]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
