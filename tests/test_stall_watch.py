"""Offline monitor fixtures: no app, network, models or production library."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sqlite3

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "training/lrasd/app-revue/stall_watch.py"
spec = importlib.util.spec_from_file_location("stall_watch", SCRIPT)
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def job(**changes):
    return {"status": "running", "stage": "deep_analysis", "asset_id": "asset_fixture", "run_id": "run_fixture",
            "request_count": 363, "inflight_calls": {}, "completed": 12, "total": 88,
            "stage_progress": {"deep_analysis": {"completed": 50, "total": 223, "status": "running"}}, **changes}


def artifacts():
    return {kind: {} for kind in watcher.ARTIFACTS}


def evaluate(value, files, state=None, now=1000):
    return watcher.evaluate("job_fixture", value, files, state or {}, now)


def test_new_child_receipt_is_durable_progress_even_when_parent_and_request_count_unchanged():
    files = artifacts()
    files["part"] = {"scene-deep/parts/scene/0.json": [900 * 10**9, 100]}
    first, state = evaluate(job(), files)
    assert first["first_snapshot"] and first["classification"] == "unknown"
    assert first["last_durable_progress_age_s"] == 100
    _, state = evaluate(job(), files, state, now=1479)
    files["receipt"]["scene-receipts/paid-child.json"] = [1480 * 10**9, 200]
    result, state = evaluate(job(), files, state, now=1480)
    assert result["classification"] == "healthy" and not result["alert"]
    assert result["stage_progress"]["deep_analysis"]["completed"] == 50
    assert result["observed_without_durable_s"] == 0
    assert result["durable_changes"] == ["receipt:scene-receipts/paid-child.json"]


@pytest.mark.parametrize("status", ["failed", "partial"])
def test_failure_alerts_immediately_on_first_snapshot_even_without_optional_fields(status):
    result, _ = evaluate({"status": status, "error": "character_context_too_large"}, artifacts())
    assert result["classification"] == "failed" and result["alert"]
    assert result["error"] == "character_context_too_large"


def test_input_split_inflight_and_request_activity_never_extend_durable_deadline():
    files, value = artifacts(), job()
    _, state = evaluate(value, files)
    for now in (1120, 1240, 1360, 1480):
        value["inflight_calls"] = {f"call_{now}": {"attempts": 3}}
        value["request_count"] += 1
        files["input"][f"scene-inputs/{now}.json"] = [now * 10**9, 100]
        files["split"][f"scene-splits/{now}.json"] = [now * 10**9, 40]
        result, state = evaluate(value, files, state, now=now)
    assert result["classification"] == "suspected_stall" and result["alert"]
    assert result["observed_without_durable_s"] == 480
    assert result["last_durable_progress_age_s"] is None
    assert result["inflight_ids"] == ["call_1480"] and result["activity_changes"]
    assert "Never auto-kill" in result["recommendation"]


def test_stage_completion_advances_without_a_receipt_and_partial_interval_is_slow():
    files = artifacts()
    _, state = evaluate(job(), files)
    result, state = evaluate(job(), files, state, now=1120)
    assert result["classification"] == "slow_active" and not result["alert"]
    newer = job(stage_progress={"deep_analysis": {"completed": 51, "total": 223, "status": "running"}})
    result, _ = evaluate(newer, files, state, now=1240)
    assert result["classification"] == "healthy" and result["observed_without_durable_s"] == 0


def test_backwards_clock_then_recovery_does_not_count_the_same_elapsed_interval_twice():
    files = artifacts()
    _, state = evaluate(job(), files, now=1000)
    backwards, state = evaluate(job(), files, state, now=500)
    assert backwards["classification"] == "unknown" and state["checked_at"] == 1000
    recovered, state = evaluate(job(), files, state, now=1000)
    assert recovered["observed_without_durable_s"] == 0
    result, _ = evaluate(job(), files, state, now=1480)
    assert result["classification"] == "suspected_stall" and result["observed_without_durable_s"] == 480


def test_paused_and_queued_time_is_excluded_when_running_resumes():
    files = artifacts()
    _, state = evaluate(job(), files)
    _, state = evaluate(job(status="paused"), files, state, now=1400)
    result, state = evaluate(job(status="queued"), files, state, now=2400)
    assert result["classification"] == "queued"
    result, state = evaluate(job(), files, state, now=2500)
    assert result["observed_without_durable_s"] == 0 and not result["alert"]
    result, _ = evaluate(job(), files, state, now=2980)
    assert result["classification"] == "suspected_stall"


def test_future_file_clock_and_missing_fields_are_not_reported_as_healthy():
    files = artifacts()
    files["receipt"] = {"scene-receipts/future.json": [2000 * 10**9, 50]}
    first, state = evaluate(job(), files)
    assert first["classification"] == "unknown" and first["last_durable_progress_age_s"] is None
    assert "future_artifact_clock:receipt" in first["warnings"]
    value = job()
    del value["request_count"]
    result, _ = evaluate(value, files, state, now=1100)
    assert result["classification"] == "unknown" and "missing_status_stage_or_request_count" in result["warnings"]


def test_new_run_cannot_inherit_previous_stall_and_completion_is_terminal():
    files = artifacts()
    _, state = evaluate(job(), files)
    stalled, state = evaluate(job(), files, state, now=1500)
    assert stalled["alert"]
    result, _ = evaluate(job(run_id="run_new"), files, state, now=1600)
    assert result["first_snapshot"] and not result["alert"]
    result, _ = evaluate(job(status="completed"), files, state, now=1700)
    assert result["classification"] == "completed" and not result["alert"]


def write_job(library, value):
    database = library / "runtime/jobs.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT)")
        connection.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)", ("job_fixture", json.dumps(value)))


def test_real_readonly_sqlite_and_artifact_snapshot_write_only_external_monitor_state(tmp_path):
    library = tmp_path / "library"
    write_job(library, job())
    receipt = library / "assets/asset_fixture/analyses/run_fixture/scene-receipts/child.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"not_read_by_monitor":true}', encoding="utf-8")
    os.utime(receipt, (900, 900))
    before = {str(path.relative_to(library)): path.read_bytes() for path in library.rglob("*") if path.is_file()}
    state_path = tmp_path / "monitor_state.json"
    result = watcher.watch(library, "job_fixture", state_path, now=1000)
    after = {str(path.relative_to(library)): path.read_bytes() for path in library.rglob("*") if path.is_file()}
    assert before == after and state_path.is_file()
    assert result["latest_artifacts"]["receipt"]["path"].endswith("child.json")
    assert set(path.name for path in tmp_path.iterdir()) == {"library", "monitor_state.json"}
    assert not state_path.with_suffix(".json.tmp").exists()


def test_unreadable_database_preserves_baseline_and_failed_legacy_job_still_alerts(tmp_path):
    state_path = tmp_path / "monitor_state.json"
    original = {"preserve": "last known good baseline"}
    state_path.write_text(json.dumps(original), encoding="utf-8")
    result = watcher.watch(tmp_path / "missing", "job_fixture", state_path, now=1000)
    assert result["classification"] == "unavailable" and result["alert"]
    assert json.loads(state_path.read_text()) == original
    library = tmp_path / "library"
    write_job(library, {"status": "failed", "error": "saved_failure"})
    result = watcher.watch(library, "job_fixture", state_path, now=1000)
    assert result["classification"] == "failed" and result["alert"]


def test_evaluation_does_not_mutate_its_inputs():
    files, value = artifacts(), job()
    _, state = evaluate(value, files)
    original = copy.deepcopy((files, value, state))
    evaluate(value, files, state, now=1120)
    assert (files, value, state) == original
