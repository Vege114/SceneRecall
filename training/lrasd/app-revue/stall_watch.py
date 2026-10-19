"""Read-only job/artifact liveness checks; write only this workspace's monitor state.

No app initialization, media/model execution, network, process control or retries.
Input files and in-flight calls indicate activity, not successful durable progress.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
import time


HERE = Path(__file__).resolve().parent
VERSION = "scenerecall-stall-watch-v1"
DEFAULT_JOB = "job_da9fda9cf49243589c95"
ARTIFACTS = {"receipt": ("scene-receipts", "*.json"), "part": ("scene-deep/parts", "*/*.json"),
             "input": ("scene-inputs", "*.json"), "split": ("scene-splits", "*.json")}


def number(value):
    return value if type(value) in {int, float} and math.isfinite(value) and value >= 0 else None


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


def load_state(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_job(library, job_id):
    # In particular do not use JobQueue/Library constructors: they recover and
    # migrate state. Do not set journal_mode on this read-only connection.
    database = library / "runtime/jobs.sqlite"
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=3) as connection:
        connection.execute("PRAGMA query_only=ON")
        row = connection.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise LookupError("job_missing")
    value = json.loads(row[0])
    if not isinstance(value, dict):
        raise ValueError("invalid_job_snapshot")
    return value


def artifact_metadata(library, job):
    asset, run = job.get("asset_id"), job.get("run_id")
    if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (asset, run)):
        raise ValueError("missing_or_invalid_asset_run")
    root = library / "assets" / asset / "analyses" / run
    if not root.resolve().is_relative_to(library.resolve()):
        raise ValueError("run_outside_library")
    result = {}
    for kind, (directory, pattern) in ARTIFACTS.items():
        rows = {}
        for path in (root / directory).glob(pattern):
            try:
                if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                    continue
                stat = path.stat()
                rows[path.relative_to(root).as_posix()] = [stat.st_mtime_ns, stat.st_size]
            except FileNotFoundError:  # Atomic replacement during a snapshot.
                continue
        result[kind] = rows
    return result


def progress_counts(job):
    stages = job.get("stage_progress")
    result = {}
    if isinstance(stages, dict):
        for stage, row in stages.items():
            count = number(row.get("completed")) if isinstance(row, dict) else None
            if count is not None:
                result[stage] = count
    count = number(job.get("completed"))
    if count is not None:
        result["job_completed"] = count
    return result


def changed_files(current, previous):
    return [name for name, metadata in current.items() if name not in previous or metadata != previous[name]]


def evaluate(job_id, job, artifacts, previous, now, *, stall_seconds=480):
    """Pure classification from snapshots; counters alone never imply a paid success."""
    warnings, reasons = [], []
    same_run = (previous.get("schema_version") == VERSION and previous.get("job_id") == job_id
                and previous.get("run_id") == job.get("run_id"))
    old = previous if same_run else {}
    first = not old
    if first:
        warnings.append("first_snapshot_history_unknown; artifact timestamps are inferred publication times")
    counts = progress_counts(job)
    old_counts = old.get("progress_counts", {})
    old_counts = old_counts if isinstance(old_counts, dict) else {}
    old_artifacts = old.get("artifacts", {})
    old_artifacts = old_artifacts if isinstance(old_artifacts, dict) else {}
    durable_changes = []
    activity_changes = []
    if not first:
        for kind in ("receipt", "part"):
            durable_changes.extend(f"{kind}:{name}" for name in changed_files(artifacts[kind], old_artifacts.get(kind, {})))
        for stage, count in counts.items():
            if stage in old_counts and count > old_counts[stage]:
                durable_changes.append(f"progress:{stage}")
        for kind in ("input", "split"):
            activity_changes.extend(f"{kind}:{name}" for name in changed_files(artifacts[kind], old_artifacts.get(kind, {})))
    request_count = number(job.get("request_count"))
    if not first and request_count is not None and request_count != old.get("request_count"):
        activity_changes.append("request_accounting_changed")
    inflight = job.get("inflight_calls")
    inflight_ids = sorted(inflight) if isinstance(inflight, dict) else []
    if inflight is not None and not isinstance(inflight, dict):
        warnings.append("invalid_inflight_fields")
    latest = {}
    for kind, files in artifacts.items():
        name = max(files, key=lambda key: files[key][0]) if files else None
        stamp = files[name][0] / 1e9 if name is not None else None
        if stamp is not None and stamp > now + 5:
            warnings.append(f"future_artifact_clock:{kind}")
        latest[kind] = {"count": len(files), "path": name, "mtime_utc": utc(stamp)}
    checked_before = number(old.get("checked_at"))
    clock_backwards = checked_before is not None and now < checked_before
    if clock_backwards:
        warnings.append("wall_clock_moved_backwards; no negative elapsed time was credited")
    last_durable = number(old.get("last_durable_at"))
    observed_without = number(old.get("observed_without_durable_s")) or 0
    status = job.get("status")
    if first:
        stamps = [metadata[0] / 1e9 for kind in ("receipt", "part") for metadata in artifacts[kind].values()
                  if metadata[0] / 1e9 <= now + 5]
        last_durable = min(max(stamps), now) if stamps else None
        # A historical file can predate a deliberate pause. Its age is reported,
        # but only observed continuous running time can establish a stall.
        observed_without = 0
    elif durable_changes:
        # Observed publication is authoritative even if file clocks are skewed.
        last_durable, observed_without = now, 0
    elif status == "running" and old.get("status") == "running":
        observed_without += max(0, now - checked_before) if checked_before is not None else 0
    else:
        observed_without = 0
        if status == "running":
            warnings.append("running_observation_baseline_started; paused/queued time excluded")
    missing = not isinstance(status, str) or not job.get("stage") or request_count is None
    if missing:
        warnings.append("missing_status_stage_or_request_count")
    classification, alert = "healthy", False
    if status in {"failed", "partial"}:
        classification, alert = "failed", True
        reasons.append("job reports failure; inspect the saved error and last frozen unit before any retry")
    elif status in {"completed", "complete"}:
        classification = "completed"
        reasons.append("job reports completion")
    elif status in {"paused", "pausing", "cancelled", "canceled", "cancelling", "queued", "pending"}:
        classification = status
        reasons.append("job is not actively running; do not resume or retry automatically")
    elif status != "running":
        classification, alert = "unknown", True
        reasons.append("job state is missing or unrecognized")
    elif observed_without >= stall_seconds:
        classification, alert = "suspected_stall", True
        reasons.append(f"no durable forward progress for at least {round(observed_without)} seconds")
        reasons.append("inflight IDs and new input/split files do not reset the durable-progress timer")
    elif clock_backwards or missing or first:
        classification = "unknown"
        reasons.append("insufficient reliable history; preserve this baseline and check again")
    elif observed_without >= 120:
        classification = "slow_active"
        reasons.append("no recent durable result yet; still below the eight-minute stall threshold")
    else:
        reasons.append("durable progress was observed" if durable_changes else
                       "no durable change observed in this short running interval")
    if classification in {"failed", "suspected_stall", "unknown"}:
        recommendation = ("Diagnose the pending unit, CLI age and configured timeout/grace, local IO and service "
                          "responsiveness. Preserve paid receipts. Never auto-kill, blindly retry or change evidence/model settings.")
    elif classification == "completed":
        recommendation = "Verify final outputs and prepare the requested report."
    else:
        recommendation = "Check again at the scheduled interval; no automatic task mutation."
    result = {"schema_version": VERSION, "job_id": job_id, "run_id": job.get("run_id"),
        "checked_utc": utc(now), "classification": classification, "alert": alert, "first_snapshot": first,
        "status": status, "stage": job.get("stage"), "stage_progress": job.get("stage_progress"),
        "completed": job.get("completed"), "total": job.get("total"), "request_count": request_count,
        "inflight_ids": inflight_ids, "latest_artifacts": latest,
        "last_durable_progress_utc": utc(last_durable),
        "last_durable_progress_age_s": round(max(0, now - last_durable), 1) if last_durable is not None else None,
        "observed_without_durable_s": round(observed_without, 1), "durable_changes": durable_changes[:12],
        "durable_change_count": len(durable_changes), "activity_changes": activity_changes[:12],
        "error": str(job.get("error"))[:300] if job.get("error") else None,
        "reasons": reasons, "warnings": warnings, "recommendation": recommendation}
    state = {"schema_version": VERSION, "job_id": job_id, "run_id": job.get("run_id"),
        "checked_at": max(now, checked_before or 0), "status": status,
        "last_durable_at": last_durable, "observed_without_durable_s": observed_without,
        "progress_counts": counts, "request_count": request_count, "artifacts": copy.deepcopy(artifacts)}
    return result, state


def watch(library, job_id, state_path, *, now=None):
    now = time.time() if now is None else now
    previous = load_state(state_path)
    try:
        job = read_job(library, job_id)
        try:
            artifacts = artifact_metadata(library, job)
        except (OSError, ValueError):
            if job.get("status") not in {"failed", "partial"}:
                raise
            # An already reported failure must alert even when an old job lacks
            # asset/run metadata. The status has precedence over artifact reads.
            artifacts = {kind: {} for kind in ARTIFACTS}
        result, state = evaluate(job_id, job, artifacts, previous, now)
    except (OSError, ValueError, sqlite3.Error, LookupError) as exc:
        # Leave the last good durable baseline intact after a transient read
        # failure. Do not report a busy/unreadable library as healthy silence.
        sqlite_details = ({"sqlite_errorcode": getattr(exc, "sqlite_errorcode", None),
                           "sqlite_errorname": getattr(exc, "sqlite_errorname", None)}
                          if isinstance(exc, sqlite3.Error) else {})
        return {"schema_version": VERSION, "job_id": job_id, "checked_utc": utc(now),
                "classification": "unavailable", "alert": True, "read_error": type(exc).__name__,
                **sqlite_details,
                "recommendation": "Diagnose read availability; preserve the prior baseline. Do not kill or retry tasks."}
    pending = state_path.with_suffix(state_path.suffix + ".tmp")
    pending.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(state_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(r"D:\SceneRecallLibrary"))
    parser.add_argument("--job", default=DEFAULT_JOB)
    options = parser.parse_args()
    result = watch(options.library, options.job, HERE / "monitor_state.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["alert"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
