"""Submitted tasks own local planning and human review, without paid test calls."""
import asyncio
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from scenerecall.jobs import JobQueue
from scenerecall.main import create_app
from test_character_api import CharacterAPIProvider, seed_episode


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    app = create_app(tmp_path / "library", start_worker=False, providers=CharacterAPIProvider(), token="lifecycle")
    for identity, episode in (("first", 1), ("second", 2)):
        seed_episode(app.state.library, identity, episode)
    queue = app.state.jobs
    calls = []

    async def scan(job_id, **kwargs):
        calls.append(("scan", job_id))
        queue.update(job_id, stage="keyframes", progress=.5)
        return {"id": "plan_test", "frames": [{"id": "frame_test"}]}

    async def analyze(job_id):
        calls.append(("analyze", job_id))

    monkeypatch.setattr(queue, "scan_plan", scan)
    monkeypatch.setattr(queue, "analyze", analyze)
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "lifecycle"})
        client.put("/api/settings", json={"bindings": {"vision": "synthetic"}})
        yield app, client, queue, calls


def submit(client, **changes):
    response = client.post("/api/jobs", json={"asset_id": "first", "analysis_strategy": "content-keyframes-v1",
                                               "task_pipeline": "unified-v1", **changes})
    assert response.status_code == 200, response.text
    return response.json()


def run(queue, job, **kwargs):
    asyncio.run(queue.execute(job["id"], **kwargs))
    return queue.get(job["id"])


def test_start_creates_one_job_without_preparation_and_auto_finishes_without_roles(lifecycle):
    _app, client, queue, calls = lifecycle
    job = submit(client)
    assert job["status"] == "queued" and calls == []
    assert all(value["status"] == "pending" for value in job["workflow_progress"].values())
    assert len(queue.list()) == 1
    result = run(queue, job)
    assert result["status"] == "completed"
    assert [kind for kind, _ in calls] == ["scan", "analyze"]
    assert all(value["status"] == "completed" for value in result["workflow_progress"].values())
    assert result["config"]["keyframe_plan_id"] == "plan_test"


def test_role_gate_requires_explicit_decisions_and_refreshes_character_snapshot(lifecycle):
    app, client, queue, calls = lifecycle
    role = app.state.characters.create("first", {"name": "合成角色"})
    job = submit(client)
    original = queue.get(job["id"])["config"]["character_snapshot"]
    waiting = run(queue, job)
    assert waiting["status"] == "waiting_input"
    assert waiting["stage"] == "character_confirmation"
    assert waiting["input_required"]["asset_id"] == "first"
    assert waiting["workflow_progress"]["keyframes"]["status"] == "completed"
    assert [kind for kind, _ in calls] == ["scan"]
    assert client.post(f"/api/jobs/{job['id']}/continue").status_code == 400
    assert run(queue, job)["status"] == "waiting_input"
    app.state.characters.create("first", {"name": "新角色"})
    review = app.state.character_confirmation.queue("first")
    for item in review["roles"]:
        app.state.character_confirmation.decide("first", item["character_id"], action="skip")
    resumed = client.post(f"/api/jobs/{job['id']}/continue")
    assert resumed.status_code == 200, resumed.text
    frozen = queue.get(job["id"])["config"]["character_snapshot"]
    assert frozen["revision"] > original["revision"]
    assert run(queue, job)["status"] == "completed"
    assert [kind for kind, _ in calls] == ["scan", "analyze"]
    assert role["id"] in {item["character_id"] for item in review["roles"]}


def test_waiting_pause_cancel_restart_and_retry_do_not_bypass_review(lifecycle):
    app, client, queue, calls = lifecycle
    app.state.characters.create("first", {"name": "待核对角色"})
    job = submit(client)
    run(queue, job)
    assert queue.control(job["id"], "pause")["status"] == "paused"
    queue.control(job["id"], "resume")
    assert run(queue, job)["status"] == "waiting_input"
    restarted = JobQueue(app.state.library, app.state.providers, app.state.search)
    assert restarted.get(job["id"])["status"] == "waiting_input"
    assert restarted.control(job["id"], "cancel")["status"] == "cancelled"
    assert restarted.control(job["id"], "retry")["status"] == "queued"
    assert run(queue, job)["status"] == "waiting_input"
    assert [kind for kind, _ in calls] == ["scan"]


def test_scene_dialogue_prefetch_cannot_run_before_role_gate(lifecycle, monkeypatch):
    app, client, queue, calls = lifecycle
    app.state.characters.create("first", {"name": "待核对角色"})
    job = submit(client)

    async def prepare(job_id):
        calls.append(("dialogue", job_id))

    monkeypatch.setattr(queue, "prepare_episode_inputs", prepare)
    assert run(queue, job, prepare_only=True)["status"] == "waiting_input"
    assert [kind for kind, _ in calls] == ["scan"]


def test_batch_continue_releases_shared_work_without_repeating_questions(lifecycle):
    app, client, queue, calls = lifecycle
    app.state.characters.create("first", {"name": "同作品角色"})
    payload = {"request_id": "unified-batch", "items": [
        {"asset_id": identity, "task_pipeline": "unified-v1", "analysis_strategy": "content-keyframes-v1"}
        for identity in ("first", "second")]}
    response = client.post("/api/jobs/batch", json=payload)
    assert response.status_code == 200, response.text
    jobs = response.json()["jobs"]
    assert all(run(queue, job)["status"] == "waiting_input" for job in jobs)
    for item in app.state.character_confirmation.queue("first")["roles"]:
        app.state.character_confirmation.decide("first", item["character_id"], action="skip")
    assert client.post(f"/api/jobs/{jobs[0]['id']}/continue").status_code == 200
    assert all(queue.get(job["id"])["status"] == "queued" for job in jobs)
    assert all(run(queue, job)["status"] == "completed" for job in jobs)
    repeated = client.post("/api/jobs/batch", json=payload).json()
    assert repeated["batch_id"] == response.json()["batch_id"]
    changed = deepcopy(payload)
    changed["items"][0]["task_pipeline"] = None
    assert client.post("/api/jobs/batch", json=changed).status_code == 409
    assert len([kind for kind, _ in calls if kind == "scan"]) == 2


def test_legacy_tasks_keep_previous_execution_without_new_human_gate(lifecycle):
    app, client, queue, calls = lifecycle
    app.state.characters.create("first", {"name": "既有接口角色"})
    job = submit(client, task_pipeline=None)
    assert "workflow_progress" not in job
    assert run(queue, job)["status"] == "completed"
    assert [kind for kind, _ in calls] == ["analyze"]


def test_review_continuation_rejects_changed_source_and_active_sampling(lifecycle, monkeypatch):
    app, client, queue, _calls = lifecycle
    role = app.state.characters.create("first", {"name": "合成角色"})
    job = submit(client)
    run(queue, job)
    app.state.character_confirmation.decide("first", role["id"], action="skip")
    monkeypatch.setattr(app.state.character_confirmation, "sampling", lambda _: {"status": "running"})
    result = client.post(f"/api/jobs/{job['id']}/continue")
    assert result.status_code == 400 and "采样" in result.json()["detail"]
    assert queue.get(job["id"])["status"] == "waiting_input"
    source = app.state.library.require_source("first")
    source.write_bytes(b"changed source")
    assert client.post(f"/api/jobs/{job['id']}/continue").status_code == 400


def test_automatic_gate_waits_for_running_sampler_and_same_work_continue(lifecycle, monkeypatch):
    app, client, queue, calls = lifecycle
    role = app.state.characters.create("first", {"name": "同作品角色"})
    first = submit(client)
    run(queue, first)
    app.state.character_confirmation.decide("first", role["id"], action="skip")
    second = submit(client, asset_id="second")
    assert run(queue, second)["status"] == "waiting_input"
    assert [kind for kind, _ in calls] == ["scan", "scan"]
    queue.control(first["id"], "cancel")
    monkeypatch.setattr(app.state.character_confirmation, "sampling", lambda _: {"status": "running"})
    queue.control(second["id"], "pause")
    queue.control(second["id"], "resume")
    assert run(queue, second)["status"] == "waiting_input"
    assert [kind for kind, _ in calls] == ["scan", "scan"]


def test_keyframe_failure_retries_inside_same_task_before_any_inference(lifecycle, monkeypatch):
    _app, client, queue, calls = lifecycle
    original = queue.scan_plan

    async def fail(*_args, **_kwargs):
        raise ValueError("合成解码失败")

    monkeypatch.setattr(queue, "scan_plan", fail)
    job = submit(client)
    assert run(queue, job)["status"] == "failed"
    assert calls == []
    monkeypatch.setattr(queue, "scan_plan", original)
    queue.control(job["id"], "retry")
    assert run(queue, job)["status"] == "completed"
    assert [kind for kind, _ in calls] == ["scan", "analyze"]
