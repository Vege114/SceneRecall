"""Central concurrency saves preserve secrets, budgets and recover publication."""
import json
import os
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from scenerecall import concurrency
from scenerecall.main import create_app
from scenerecall.providers import ProviderManager


@pytest.fixture
def central(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, token="preferences-test")
    manager = app.state.providers
    manager.upsert({"id": "api", "name": "API fixture", "base_url": "https://fixture.invalid/v1",
                    "model": "fixture", "capabilities": ["preparation", "vision"], "max_concurrency": 8,
                    "api_key": "synthetic-session-key", "timeout_s": 123,
                    "input_price_per_million": 1, "output_price_per_million": 2})
    manager.upsert({"id": "codex", "name": "CLI fixture", "provider_type": "codex_cli",
                    "capabilities": ["subtitle", "query"], "max_concurrency": 3})
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "preferences-test"})
        client.put("/api/settings", json={"bindings": {"preparation": "api", "vision": "api", "subtitle": "codex"},
                                         "defaults": {"max_requests": 88, "subtitle_fps": 1.5}})
        yield app, client


def test_one_save_changes_only_execution_and_caps_and_survives_restart(central):
    app, client = central
    before = deepcopy(client.get("/api/settings").json())
    profiles = deepcopy(app.state.providers.list())
    execution = {"transcript": "parallel", "preparation": "parallel", "subtitle": "serial", "max_concurrency": 16}
    response = client.put("/api/concurrency", json={"execution": execution, "profiles": {"api": 16, "codex": 16}})
    assert response.status_code == 200, response.text
    after = response.json()
    assert after["settings"] == {**before, "defaults": {**before["defaults"], "execution": execution}}
    assert after["provider_profiles"] == [{**profile, "max_concurrency": 16} for profile in profiles]
    assert app.state.providers._key(app.state.providers.get("api")) == "synthetic-session-key"
    root = app.state.library.root / "private"
    assert not (root / "concurrency-pending.json").exists()
    assert "synthetic-session-key" not in (root / "profiles.json").read_text()
    if os.name == "posix":
        assert (root / "profiles.json").stat().st_mode & 0o777 == 0o600
        assert (root / "settings.json").stat().st_mode & 0o777 == 0o600
    restarted = create_app(app.state.library.root, start_worker=False, token="restarted")
    with TestClient(restarted) as other:
        other.post("/api/session", json={"token": "restarted"})
        assert other.get("/api/settings").json() == after["settings"]
        assert other.get("/api/profiles").json() == after["provider_profiles"]


def test_partial_edits_keep_modes_unselected_connections_and_large_caps(central):
    app, client = central
    before = client.get("/api/settings").json()
    response = client.put("/api/concurrency", json={"profiles": {"api": 128}})
    assert response.status_code == 200
    assert response.json()["settings"] == before
    assert app.state.providers.get("api")["max_concurrency"] == 128
    assert app.state.providers.get("codex")["max_concurrency"] == 3
    response = client.put("/api/concurrency", json={"execution": {"max_concurrency": 32}})
    assert response.status_code == 200
    assert response.json()["settings"]["defaults"]["execution"] == {
        **before["defaults"]["execution"], "max_concurrency": 32}
    # Existing endpoints remain valid and visible to the central panel.
    client.post("/api/profiles", json={"id": "api", "max_concurrency": 64})
    assert client.get("/api/bootstrap").json()["provider_profiles"][0]["max_concurrency"] == 64


@pytest.mark.parametrize("payload", [
    {"profiles": {"api": 0}}, {"profiles": {"api": -1}}, {"profiles": {"api": True}},
    {"profiles": {"api": "16"}}, {"profiles": {"api": 1.5}}, {"profiles": None},
    {"execution": {"max_concurrency": False}}, {"execution": {"preparation": "automatic"}},
    {"execution": {"deep_analysis": "parallel"}}, {"bindings": {"query": "api"}},
    {"profiles": {"missing-profile": 16}},
])
def test_invalid_batch_never_partially_writes_a_valid_sibling(central, payload):
    app, client = central
    root = app.state.library.root / "private"
    before = {name: (root / name).read_bytes() for name in ("settings.json", "profiles.json")}
    response = client.put("/api/concurrency", json={"execution": {"max_concurrency": 32}, **payload})
    assert response.status_code in (400, 422)
    assert {name: (root / name).read_bytes() for name in before} == before
    assert not (root / "concurrency-pending.json").exists()


@pytest.mark.parametrize("failed_file", ["profiles.json", "settings.json"])
def test_interrupted_publish_replays_before_profile_edits_and_startup(central, monkeypatch, failed_file):
    app, client = central
    publish = concurrency.publish_json
    failed = False

    def fail_once(path, data):
        nonlocal failed
        if path.name == failed_file and not failed:
            failed = True
            raise OSError("synthetic write interruption")
        return publish(path, data)

    monkeypatch.setattr(concurrency, "publish_json", fail_once)
    response = client.put("/api/concurrency", json={"execution": {"max_concurrency": 16},
                                                    "profiles": {"api": 16, "codex": 16}})
    assert response.status_code == 400 and "尚未完整保存" in response.text
    root = app.state.library.root / "private"
    pending = root / "concurrency-pending.json"
    assert pending.exists() and "synthetic-session-key" not in pending.read_text()
    if failed_file == "profiles.json":
        # A normal profile edit must replay first, then keep the newer edit.
        app.state.providers.upsert({"id": "api", "max_concurrency": 24, "name": "newer edit"})
    else:
        create_app(app.state.library.root, start_worker=False)
    assert not pending.exists()
    expected = 24 if failed_file == "profiles.json" else 16
    assert app.state.providers.get("api")["max_concurrency"] == expected
    assert client.get("/api/settings").json()["defaults"]["execution"]["max_concurrency"] == 16
    assert app.state.providers.get("codex")["max_concurrency"] == 16
    assert app.state.providers._key(app.state.providers.get("api")) == "synthetic-session-key"


def test_failure_before_journal_commit_changes_nothing(central, monkeypatch):
    app, client = central
    root = app.state.library.root / "private"
    before = {name: (root / name).read_bytes() for name in ("settings.json", "profiles.json")}

    def fail(path, data):
        raise OSError("synthetic journal failure")

    monkeypatch.setattr(concurrency, "publish_json", fail)
    response = client.put("/api/concurrency", json={"profiles": {"api": 16}})
    assert response.status_code == 400
    assert {name: (root / name).read_bytes() for name in before} == before
    assert not (root / "concurrency-pending.json").exists()


def test_corrupt_journal_is_not_published(central):
    app, _ = central
    root = app.state.library.root / "private"
    previous = (root / "profiles.json").read_bytes()
    (root / "concurrency-pending.json").write_text(json.dumps({"schema_version": "1.0", "settings": "invalid"}))
    with pytest.raises(ValueError, match="恢复记录损坏"):
        ProviderManager(root).list()
    assert (root / "profiles.json").read_bytes() == previous


def test_central_update_requires_existing_local_session(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False)
    with TestClient(app) as client:
        assert client.put("/api/concurrency", json={"profiles": {}}).status_code == 401
