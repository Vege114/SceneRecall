"""Execution choices are validated and frozen without calling any model."""
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from scenerecall.main import create_app
from scenerecall.models import AnalysisInput, ExecutionOptions
from test_character_api import CharacterAPIProvider, seed_episode


PARALLEL = {"transcript": "parallel", "preparation": "parallel", "subtitle": "parallel", "max_concurrency": 4}
SERIAL = ExecutionOptions().model_dump()


@pytest.fixture
def execution_api(tmp_path):
    app = create_app(tmp_path / "library", providers=CharacterAPIProvider(), start_worker=False, token="execution-test")
    seed_episode(app.state.library, "first", 1)
    seed_episode(app.state.library, "second", 2)
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "execution-test"})
        client.put("/api/settings", json={"bindings": {"vision": "synthetic"}})
        yield app, client


def test_defaults_can_be_saved_and_old_api_requests_retain_serial_semantics(execution_api):
    app, client = execution_api
    assert client.get("/api/settings").json()["defaults"]["execution"] == PARALLEL
    choices = {**PARALLEL, "transcript": "serial", "max_concurrency": 4}
    saved = client.put("/api/settings", json={"defaults": {"execution": choices}})
    assert saved.status_code == 200, saved.text
    assert client.get("/api/bootstrap").json()["settings"]["defaults"]["execution"] == choices
    job = client.post("/api/jobs", json={"asset_id": "first"}).json()
    assert job["execution"] == SERIAL
    assert app.state.jobs.get(job["id"])["config"]["execution"] == SERIAL
    assert AnalysisInput(asset_id="first").execution.model_dump() == SERIAL


def test_partial_execution_settings_preserve_other_defaults_and_invalid_choices_do_not_persist(execution_api):
    _, client = execution_api
    changed = client.put("/api/settings", json={"defaults": {"execution": {"transcript": "serial"}}})
    assert changed.status_code == 200, changed.text
    expected = {**PARALLEL, "transcript": "serial"}
    assert changed.json()["defaults"]["execution"] == expected
    bad = client.put("/api/settings", json={"defaults": {"execution": {"max_concurrency": 0}}})
    assert bad.status_code == 400, bad.text
    assert client.get("/api/settings").json()["defaults"]["execution"] == expected


def test_execution_default_patch_preserves_hidden_modes_and_unrelated_settings_on_restart(execution_api):
    app, client = execution_api
    existing = {**PARALLEL, "transcript": "serial", "preparation": "serial", "max_concurrency": 3}
    client.put("/api/settings", json={"defaults": {
        "execution": existing, "subtitle_fps": 1.5, "max_requests": 77, "max_cost": .5}})
    before = client.get("/api/settings").json()
    response = client.put("/api/settings", json={"defaults": {
        "execution": {"subtitle": "parallel", "max_concurrency": 6}}})
    assert response.status_code == 200, response.text
    expected = {**before, "defaults": {**before["defaults"], "execution": {
        **existing, "subtitle": "parallel", "max_concurrency": 6}}}
    assert response.json() == expected
    restarted = create_app(app.state.library.root, providers=CharacterAPIProvider(),
                           start_worker=False, token="restarted-defaults")
    with TestClient(restarted) as other:
        other.post("/api/session", json={"token": "restarted-defaults"})
        assert other.get("/api/settings").json() == expected
    job = client.post("/api/jobs", json={"asset_id": "first", "execution": expected["defaults"]["execution"]}).json()
    client.put("/api/settings", json={"defaults": {"execution": {"max_concurrency": 8}}})
    assert client.get(f"/api/jobs/{job['id']}").json()["execution"]["max_concurrency"] == 6


@pytest.mark.parametrize("choice", [
    {"max_concurrency": 0}, {"max_concurrency": -1}, {"max_concurrency": True},
    {"max_concurrency": 2.5}, {"max_concurrency": "3"}, {"transcript": "automatic"},
    {"deep_analysis": "parallel"}, {"subtitle": False}, None,
])
def test_invalid_execution_rejected_before_enqueueing(execution_api, choice):
    app, client = execution_api
    response = client.post("/api/jobs", json={"asset_id": "first", "execution": choice})
    assert response.status_code == 422, response.text
    assert app.state.jobs.list() == []


@pytest.mark.parametrize("concurrency", [9, 17, 32, 128, 100000])
def test_user_defined_concurrency_above_old_caps_is_saved_without_clamping(execution_api, concurrency):
    app, client = execution_api
    choices = {**PARALLEL, "max_concurrency": concurrency}
    saved = client.put("/api/settings", json={"defaults": {"execution": choices}})
    assert saved.status_code == 200, saved.text
    assert saved.json()["defaults"]["execution"] == choices
    response = client.post("/api/jobs", json={"asset_id": "first", "execution": choices})
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["execution"] == choices
    assert app.state.jobs.get(job["id"])["config"]["execution"] == choices
    assert job["request_count"] == 0


def test_job_snapshot_resume_and_public_projection(execution_api):
    app, client = execution_api
    choices = {**PARALLEL, "subtitle": "serial", "max_concurrency": 5}
    response = client.post("/api/jobs", json={"asset_id": "first", "execution": choices})
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["execution"] == choices and job["stages"] == ["vision"]
    assert job["analysis_strategy"] == "content-keyframes-v1"
    app.state.jobs.update(job["id"], status="paused", inflight_calls={}, inflight_attempts=0)
    client.put("/api/settings", json={"defaults": {"execution": SERIAL}})
    resumed = client.post(f"/api/jobs/{job['id']}/resume", json={}).json()
    assert resumed["execution"] == choices
    assert not {"config", "inflight_calls", "inflight_attempts", "pending_scene_receipt"} & resumed.keys()
    config = app.state.jobs.get(job["id"])["config"]
    config.pop("execution")
    app.state.jobs.update(job["id"], config=config)
    assert client.get(f"/api/jobs/{job['id']}").json()["execution"] == SERIAL


def test_batch_idempotency_and_independent_execution_snapshots(execution_api):
    app, client = execution_api
    payload = {"request_id": "execution-batch", "items": [
        {"asset_id": "first", "execution": PARALLEL},
        {"asset_id": "second", "execution": SERIAL},
    ]}
    first = client.post("/api/jobs/batch", json=payload)
    assert first.status_code == 200, first.text
    repeated = client.post("/api/jobs/batch", json=payload)
    assert repeated.json()["batch_id"] == first.json()["batch_id"]
    by_asset = {job["asset_id"]: job for job in first.json()["jobs"]}
    assert by_asset["first"]["execution"] == PARALLEL
    assert by_asset["second"]["execution"] == SERIAL
    changed = deepcopy(payload)
    changed["items"][0]["execution"]["max_concurrency"] = 5
    assert client.post("/api/jobs/batch", json=changed).status_code == 409
    assert len(app.state.jobs.list()) == 2


def test_omitted_batch_execution_matches_explicit_legacy_defaults(execution_api):
    _, client = execution_api
    payload = {"request_id": "serial-batch", "items": [{"asset_id": "first"}]}
    first = client.post("/api/jobs/batch", json=payload)
    assert first.status_code == 200, first.text
    payload["items"][0]["execution"] = SERIAL
    assert client.post("/api/jobs/batch", json=payload).json()["batch_id"] == first.json()["batch_id"]


def test_character_reanalysis_inherits_defaults_or_uses_explicit_choices(execution_api):
    app, client = execution_api
    endpoint = "/api/assets/first/characters/reanalyze"
    inherited = client.post(endpoint, json={"scope": "asset"})
    assert inherited.status_code == 200, inherited.text
    assert inherited.json()["jobs"][0]["execution"] == PARALLEL
    choices = {**SERIAL, "preparation": "parallel", "max_concurrency": 2}
    explicit = client.post(endpoint, json={"scope": "series", "execution": choices})
    assert explicit.status_code == 200, explicit.text
    assert len(explicit.json()["jobs"]) == 2
    assert all(job["execution"] == choices for job in explicit.json()["jobs"])
    assert all(app.state.jobs.get(job["id"])["config"]["execution"] == choices
               for job in explicit.json()["jobs"])
