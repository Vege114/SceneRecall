"""Shared task preferences persist without starting media or model work."""
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from scenerecall.main import create_app
from scenerecall.models import AnalysisInput


@pytest.fixture
def settings_api(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    app = create_app(tmp_path / "library", start_worker=False, token="analysis-defaults")
    app.state.providers.upsert({"id": "api", "name": "Local API", "provider_type": "openai_compatible",
        "base_url": "http://127.0.0.1:9999/v1", "model": "small-vision", "secret_mode": "none",
        "capabilities": ["vision", "preparation"]})
    app.state.providers.upsert({"id": "codex", "name": "CLI", "provider_type": "codex_cli",
        "model": "connection-model", "capabilities": ["preparation", "vision", "subtitle"]})
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "analysis-defaults"})
        yield app, client


def test_shared_defaults_are_complete_and_persist_independent_stage_connections(settings_api):
    app, client = settings_api
    original = client.get("/api/settings").json()
    assert original["defaults"]["analysis_strategy"] == "scene-chunks-v1"
    assert original["defaults"]["retrieval_mode"] == "core"
    assert original["defaults"]["stages"] == ["vision", "subtitle"]
    assert original["defaults"]["model_bindings"] == {}
    edited = {**original["defaults"], "analysis_strategy": "retrieval-first-v1", "retrieval_mode": "complete",
              "stages": ["vision"], "model_bindings": {"vision": "api", "preparation": "codex"},
              "model_overrides": {"preparation": {"provider_type": "codex_cli", "profile_id": "codex",
                                                   "reasoning_effort": "high"}},
              "subtitle_fps": 1.5, "subtitle_crop": [.1, .7, .8, .2], "max_requests": 77,
              "execution": {"transcript": "serial", "preparation": "parallel", "subtitle": "serial",
                            "max_concurrency": 6}}
    response = client.put("/api/settings", json={"defaults": edited})
    assert response.status_code == 200, response.text
    assert response.json() == {**original, "defaults": edited}
    assert app.state.jobs.list() == []
    restarted = create_app(app.state.library.root, start_worker=False, token="restarted-defaults")
    with TestClient(restarted) as other:
        other.post("/api/session", json={"token": "restarted-defaults"})
        assert other.get("/api/bootstrap").json()["settings"] == response.json()


def test_model_defaults_can_be_cleared_and_do_not_materialize_omitted_cli_model(settings_api):
    _, client = settings_api
    response = client.put("/api/settings", json={"defaults": {
        "model_bindings": {"vision": "codex"},
        "model_overrides": {"vision": {"provider_type": "codex_cli", "profile_id": "codex"}}}})
    assert response.status_code == 200, response.text
    assert "model" not in response.json()["defaults"]["model_overrides"]["vision"]
    assert client.put("/api/settings", json={"defaults": {"model_overrides": {
        "vision": {"provider_type": "codex_cli", "profile_id": "codex", "model": ""}}}}).status_code == 200
    assert client.get("/api/settings").json()["defaults"]["model_overrides"]["vision"]["model"] == ""
    response = client.put("/api/settings", json={"defaults": {"model_bindings": {}, "model_overrides": {}}})
    assert response.status_code == 200, response.text
    assert response.json()["defaults"]["model_bindings"] == response.json()["defaults"]["model_overrides"] == {}


@pytest.mark.parametrize("changes", [
    {"analysis_strategy": "unknown"}, {"retrieval_mode": "unknown"}, {"stages": []},
    {"execution": None}, {"execution": []}, {"execution": "parallel"},
    {"execution": 3}, {"execution": False},
    {"stages": ["vision", "vision"]}, {"model_bindings": {"subtitle": "api"}},
    {"model_bindings": {"query": "api"}}, {"model_bindings": {"vision": ""}},
    {"model_overrides": {"vision": {"provider_type": "codex_cli", "profile_id": "codex"}}},
    {"model_bindings": {"vision": "api"}, "model_overrides": {
        "vision": {"provider_type": "codex_cli", "profile_id": "api"}}},
])
def test_invalid_default_choices_do_not_write_partial_settings(settings_api, changes):
    _, client = settings_api
    before = deepcopy(client.get("/api/settings").json())
    response = client.put("/api/settings", json={"defaults": changes})
    assert response.status_code == 400, response.text
    assert client.get("/api/settings").json() == before


def test_connections_used_by_defaults_require_reassignment_before_removal(settings_api):
    _, client = settings_api
    assert client.put("/api/settings", json={"defaults": {"model_bindings": {"vision": "api"}}}).status_code == 200
    assert client.delete("/api/profiles/api").status_code == 400
    response = client.post("/api/profiles", json={"id": "api", "capabilities": ["preparation"]})
    assert response.status_code == 400
    assert "任务默认设置" in response.json()["detail"]
    assert client.put("/api/settings", json={"defaults": {"model_bindings": {}}}).status_code == 200
    assert client.delete("/api/profiles/api").status_code == 200


def test_legacy_requests_keep_existing_semantics_and_unified_pipeline_is_explicit():
    legacy = AnalysisInput(asset_id="asset")
    assert legacy.task_pipeline is None and legacy.analysis_strategy is None
    assert legacy.stages == ["vision"] and legacy.model_bindings == {}
    assert legacy.execution.transcript == "serial"
    assert AnalysisInput(asset_id="asset", task_pipeline="unified-v1").task_pipeline == "unified-v1"
