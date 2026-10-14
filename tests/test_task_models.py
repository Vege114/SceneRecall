"""Task model choices use fabricated media and stubbed Codex, never paid calls."""
import asyncio
import json
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from scenerecall import codex_cli
from scenerecall.jobs import JobQueue, StopJob, model_identity
from scenerecall.main import create_app
from scenerecall.models import AnalysisInput
from scenerecall.providers import ProviderManager
from scenerecall.scenes import SceneStore
from test_character_api import seed_episode
from test_scene_jobs import scene_worker as scene_worker
from test_scene_providers import chunk


def selection(model="task-model", effort="high", **changes):
    return {"provider_type": "codex_cli", "profile_id": "codex", "model": model,
            "reasoning_effort": effort, **changes}


def connect(providers):
    return providers.upsert({"id": "codex", "name": "合成 Codex 连接", "provider_type": "codex_cli",
        "model": "connection-model", "max_concurrency": 3,
        "capabilities": ["preparation", "vision", "subtitle", "query"]})


def response(data):
    return codex_cli.CodexCLIResult(json.dumps(data), {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3})


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    app = create_app(tmp_path / "library", start_worker=False, token="task-selection-test")
    connect(app.state.providers)
    seed_episode(app.state.library, "first", 1)
    seed_episode(app.state.library, "second", 2)
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "task-selection-test"})
        client.put("/api/settings", json={"bindings": {"vision": "codex", "subtitle": "codex"}})
        yield app, client


def test_task_selection_freezes_effective_and_source_profiles_and_resume_preserves_them(api):
    app, client = api
    original = app.state.providers.get("codex")
    submitted = client.post("/api/jobs", json={"asset_id": "first", "model_overrides": {"vision": selection()}})
    assert submitted.status_code == 200, submitted.text
    public = submitted.json()
    assert public["task_models"]["vision"] == selection()
    assert "config" not in public and "source_profile_snapshots" not in public
    saved = app.state.jobs.get(public["id"])["config"]
    assert saved["profile_snapshots"]["vision"]["model"] == "task-model"
    assert saved["profile_snapshots"]["vision"]["reasoning_effort"] == "high"
    assert saved["source_profile_snapshots"]["vision"] == original
    assert app.state.providers.get("codex") == original
    app.state.jobs.update(public["id"], status="paused")
    app.state.jobs = JobQueue(app.state.library, app.state.providers, app.state.search)
    resumed = app.state.jobs.control(public["id"], "resume")
    assert resumed["config"] == saved
    app.state.jobs.before_call(public["id"], saved["profile_snapshots"]["vision"])
    app.state.providers.upsert({"id": "codex", "model": "changed-connection"})
    with pytest.raises(StopJob):
        app.state.jobs.before_call(public["id"], saved["profile_snapshots"]["vision"])
    assert app.state.jobs.get(public["id"])["status"] == "paused"


def test_omitted_and_explicit_default_model_are_distinct_and_reasoning_identity_is_compatible(api):
    app, client = api
    inherit = {"provider_type": "codex_cli", "profile_id": "codex", "reasoning_effort": "low"}
    first = client.post("/api/jobs", json={"asset_id": "first", "model_overrides": {"vision": inherit}}).json()
    second = client.post("/api/jobs", json={"asset_id": "first", "model_overrides": {"vision": selection("", "low")}}).json()
    assert first["task_models"]["vision"]["model"] == "connection-model"
    assert second["task_models"]["vision"]["model"] == ""
    original = app.state.providers.get("codex")
    legacy = {key: original.get(key) for key in ("protocol", "base_url", "model", "provider_type")}
    assert model_identity(original) == model_identity({**original, "reasoning_effort": None}) == legacy
    assert model_identity({**original, "reasoning_effort": "low"}) != model_identity({**original, "reasoning_effort": "high"})


@pytest.mark.parametrize("overrides,status", [
    ({"preparation": selection()}, 400),
    ({"vision": selection(profile_id="old-binding")}, 400),
    ({"query": selection()}, 422),
    ({"vision": selection(provider_type="openai_compatible")}, 422),
    ({"vision": selection(model="bad\nmodel")}, 422),
    ({"vision": selection(model="bad\x85model")}, 422),
    ({"vision": selection(model="bad model")}, 422),
    ({"vision": selection(model="--help")}, 422),
    ({"vision": selection(profile_id=" ")}, 422),
    ({"vision": selection(effort="unlimited")}, 422),
])
def test_invalid_or_stale_choices_are_rejected_before_any_job_is_created(api, overrides, status):
    app, client = api
    answer = client.post("/api/jobs", json={"asset_id": "first", "model_overrides": overrides})
    assert answer.status_code == status, answer.text
    assert app.state.jobs.list() == []


def test_api_source_cannot_accept_codex_selection(api):
    app, client = api
    app.state.providers.upsert({"id": "codex", "provider_type": "openai_compatible", "model": "api-model",
                               "base_url": "https://api.example.invalid/v1", "secret_mode": "none"})
    response = client.post("/api/jobs", json={"asset_id": "first", "model_overrides": {"vision": selection()}})
    assert response.status_code == 400 and "来源" in response.json()["detail"]
    assert app.state.jobs.list() == []


def test_batch_and_character_reanalysis_retain_choices_and_batch_idempotency_distinguishes_default(api):
    app, client = api
    inherit = {"provider_type": "codex_cli", "profile_id": "codex"}
    payload = {"request_id": "explicit-models", "items": [
        {"asset_id": "first", "model_overrides": {"vision": inherit}},
        {"asset_id": "second", "model_overrides": {"vision": selection("second-model", "low")}},
    ]}
    result = client.post("/api/jobs/batch", json=payload)
    assert result.status_code == 200, result.text
    repeated = client.post("/api/jobs/batch", json=payload)
    assert repeated.json()["batch_id"] == result.json()["batch_id"]
    changed = deepcopy(payload)
    changed["items"][0]["model_overrides"]["vision"]["model"] = ""
    assert client.post("/api/jobs/batch", json=changed).status_code == 409
    assert {job["task_models"]["vision"]["model"] for job in result.json()["jobs"]} == {"connection-model", "second-model"}
    reanalysis = client.post("/api/assets/first/characters/reanalyze", json={
        "scope": "series", "model_overrides": {"vision": selection("reanalysis-model", "medium")}})
    assert reanalysis.status_code == 200, reanalysis.text
    assert len(reanalysis.json()["jobs"]) == 2
    assert all(job["task_models"]["vision"]["model"] == "reanalysis-model" for job in reanalysis.json()["jobs"])


def test_deepening_estimate_and_start_keep_only_vision_and_do_not_silently_reuse_other_choices(api):
    app, client = api
    scene = SceneStore(app.state.library).publish("first", "initial", [{
        **chunk(["frame_test"]), "start_ms": 0, "end_ms": 6000,
        "evidence_frame_ids": ["frame_test"], "shot_ids": ["shot"], "subtitle_ids": [],
    }], 0, 6000)[0]
    url = f"/api/assets/first/chunks/{scene['id']}/deepen"
    payload = {"model_overrides": {"vision": selection("deep-model", "high")}}
    estimate = client.post(url + "/estimate", json=payload)
    assert estimate.status_code == 200 and estimate.json()["estimated_requests"] == 1
    assert app.state.jobs.list() == []
    started = client.post(url, json=payload)
    assert started.status_code == 200, started.text
    assert started.json()["task_models"] == {"vision": selection("deep-model", "high")}
    assert client.post(url, json=payload).json()["id"] == started.json()["id"]
    assert client.post(url, json={"model_overrides": {"vision": selection("other-model")}}).status_code == 400
    assert client.post(url + "/estimate", json={"model_overrides": {"preparation": selection()}}).status_code == 400


def test_catalog_route_is_authenticated_and_no_store(api, monkeypatch):
    _, client = api
    calls = []

    async def catalog():
        calls.append(True)
        return {"models": [{"id": "fixture-model", "reasoning_efforts": ["low", "high"]}]}

    monkeypatch.setattr(codex_cli, "model_catalog", catalog)
    got = client.get("/api/codex/models")
    assert got.status_code == 200 and got.headers["cache-control"] == "no-store"
    assert got.json()["models"][0]["id"] == "fixture-model"
    client.cookies.clear()
    assert client.get("/api/codex/models").status_code == 401
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_concurrent_task_choices_reach_codex_and_never_leak_to_unscoped_requests(api, monkeypatch):
    app, _ = api
    queue, providers = app.state.jobs, app.state.providers
    calls = []
    barrier = asyncio.Event()

    async def complete(instruction, content, model, timeout_s, reasoning_effort=None):
        calls.append((content, model, reasoning_effort))
        if len(calls) == 3:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), 2)
        return response({"search_text": content})

    monkeypatch.setattr(codex_cli, "complete", complete)
    jobs = []
    for asset_id, model, effort in [("first", "task-a", "low"), ("second", "task-b", "high")]:
        _, config = queue.validate(AnalysisInput(asset_id=asset_id,
            model_overrides={"vision": selection(model, effort)}), {"bindings": {"vision": "codex"}})
        jobs.append(queue.add("analysis", config, asset_id))
    await asyncio.gather(*(queue.call(job["id"], job["config"]["profile_snapshots"]["vision"],
                                     providers.interpret, job["asset_id"]) for job in jobs),
                         providers.interpret("codex", "outside"))
    assert set(calls) == {("first", "task-a", "low"), ("second", "task-b", "high"),
                          ("outside", "connection-model", None)}
    assert providers._profile("codex", "vision")["model"] == "connection-model"


@pytest.mark.asyncio
async def test_retry_uses_repaired_timeout_but_keeps_original_model_and_effort(api, monkeypatch):
    app, _ = api
    queue, providers = app.state.jobs, app.state.providers
    _, config = queue.validate(AnalysisInput(asset_id="first", model_overrides={"vision": selection()}),
                               {"bindings": {"vision": "codex"}})
    job = queue.add("analysis", config, "first")
    calls = []

    async def complete(instruction, content, model, timeout_s, reasoning_effort=None):
        calls.append((model, timeout_s, reasoning_effort))
        if len(calls) == 1:
            raise codex_cli.CodexCLIError("合成超时", request_count=1)
        return response({"search_text": "已恢复"})

    monkeypatch.setattr(codex_cli, "complete", complete)
    with pytest.raises(ValueError, match="合成超时"):
        await queue.call(job["id"], config["profile_snapshots"]["vision"], providers.interpret, "query")
    assert providers._profile("codex", "vision")["model"] == "connection-model"
    queue.update(job["id"], status="failed")
    providers.upsert({"id": "codex", "timeout_s": 300})
    queue.control(job["id"], "retry")
    await queue.call(job["id"], config["profile_snapshots"]["vision"], providers.interpret, "query")
    assert calls == [("task-model", 90, "high"), ("task-model", 300, "high")]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["content-keyframes-v1", "scene-chunks-v1"])
async def test_full_worker_uses_stage_specific_frozen_models_for_legacy_and_scenes(scene_worker, tmp_path, monkeypatch, strategy):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    worker = scene_worker(2)
    providers = ProviderManager(tmp_path / "providers")
    connect(providers)
    worker.queue.providers = providers
    calls = []

    async def complete(instruction, content, model, timeout_s, reasoning_effort=None, output_schema=None):
        calls.append((model, reasoning_effort))
        mapping = next((part["text"].removeprefix("Original frame map: ") for part in content
                        if part.get("type") == "text" and part["text"].startswith("Original frame map: ")), None)
        if mapping:
            ids = [frame["frame_id"] for frame in json.loads(mapping) if not frame["reference_only"]]
            return response({"chunks": [chunk(ids, needs_deep=True, reason="合成复杂动作")], "unresolved": []})
        ids = [json.loads(part["text"])["frame_id"] for part in content if part.get("type") == "text"
               and part["text"].startswith('{"frame_id":')]
        return response({"summary": "合成画面", "entities": [], "events": [], "evidence_frame_ids": ids})

    monkeypatch.setattr(codex_cli, "complete", complete)
    overrides = {"vision": selection("vision-choice", "high")}
    if strategy == "scene-chunks-v1":
        overrides["preparation"] = selection("prepare-choice", "low")
    _, config = worker.queue.validate(AnalysisInput(asset_id=worker.asset["id"],
        analysis_strategy=strategy, model_overrides=overrides), {"bindings": {"vision": "codex", "preparation": "codex"}})
    job = worker.queue.add("analysis", config, worker.asset["id"])
    await worker.queue.execute(job["id"])
    finished = worker.queue.get(job["id"])
    assert finished["status"] == "completed", finished
    assert ("vision-choice", "high") in calls
    assert ("prepare-choice", "low") in calls if strategy == "scene-chunks-v1" else len(calls) == 2
    assert providers.get("codex")["model"] == "connection-model"
