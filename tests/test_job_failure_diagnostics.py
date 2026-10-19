"""Persist bounded provider failure codes without raw diagnostics or new calls."""
import json

import pytest

from scenerecall.jobs import JobQueue
from scenerecall.providers import ProviderError
from test_scene_jobs import enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


@pytest.mark.asyncio
async def test_preflight_failure_survives_job_reload_and_keeps_paid_receipt(scene_worker, monkeypatch):
    worker = scene_worker(60)
    prepare = worker.provider.prepare_visual
    calls = 0

    async def fail_after_paid_success(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProviderError("无法检查 Codex CLI 状态", request_count=0, code="preflight_features")
        return await prepare(*args)

    monkeypatch.setattr(worker.provider, "prepare_visual", fail_after_paid_success)
    failed = await execute(worker, enqueue(worker))
    assert failed["status"] == "failed" and failed["error_code"] == "preflight_features"
    assert failed["error"] == "无法检查 Codex CLI 状态"
    metric = failed["stage_metrics"]["preparation"]
    assert metric["last_failure_code"] == "preflight_features"
    assert metric["failure_counts"] == {"preflight_features": 1}
    assert metric["request_count"] == 1
    assert "timeout_count" not in metric and "invalid_output_count" not in metric
    directory = worker.library.asset_dir(worker.asset["id"]) / "analyses" / failed["run_id"]
    assert json.loads((directory / "run.json").read_text(encoding="utf-8"))["error_code"] == "preflight_features"
    receipts = list((directory / "scene-receipts").glob("*.json"))
    assert len(receipts) == 1
    paid_receipt = receipts[0].read_bytes()
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    restored = worker.queue.get(failed["id"])
    assert restored["error_code"] == "preflight_features"
    assert restored["stage_metrics"]["preparation"] == metric
    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    worker.queue.control(failed["id"], "retry")
    done = await execute(worker, failed)
    assert done["status"] == "completed" and done["error_code"] is None
    assert done["stage_metrics"]["preparation"]["failure_counts"] == {"preflight_features": 1}
    assert done["request_count"] == 2
    assert receipts[0].read_bytes() == paid_receipt
    assert json.loads((directory / "run.json").read_text(encoding="utf-8"))["error_code"] is None


@pytest.mark.parametrize("code", ["timeout", "invalid_output", "preflight_login_status"])
def test_stage_diagnostics_keep_known_codes_and_existing_counters(scene_worker, code):
    worker = scene_worker(1)
    job = enqueue(worker)
    worker.queue._scene_metric(job["id"], "preparation", failure_code=code)
    worker.queue._scene_metric(job["id"], "preparation", failure_code=code)
    metric = worker.queue.get(job["id"])["stage_metrics"]["preparation"]
    assert metric["last_failure_code"] == code and metric["failure_counts"] == {code: 2}
    if code in {"timeout", "invalid_output"}:
        assert metric[code + "_count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["raw command --token=synthetic-secret\nENV=synthetic-secret", ["synthetic-secret"]])
async def test_unrecognized_failure_code_is_not_persisted(scene_worker, monkeypatch, code):
    worker = scene_worker(1)

    async def fail(*args):
        raise ProviderError("连接暂时不可用", request_count=0, code=code)

    monkeypatch.setattr(worker.provider, "prepare_visual", fail)
    failed = await execute(worker, enqueue(worker))
    assert failed["status"] == "failed" and failed["error_code"] is None
    assert failed["error"] == "连接暂时不可用"
    metric = failed["stage_metrics"]["preparation"]
    assert "last_failure_code" not in metric and "failure_counts" not in metric
    assert "synthetic-secret" not in json.dumps(failed, ensure_ascii=False)
