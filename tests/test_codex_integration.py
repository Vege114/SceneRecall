"""Codex provider routing, persistence and authenticated local API regressions."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scenerecall import codex_cli
from scenerecall.main import create_app
from scenerecall.providers import ProviderError, ProviderManager


def connection(providers, **overrides):
    return providers.upsert({"id": "codex", "name": "我的 Codex", "provider_type": "codex_cli",
                             "capabilities": ["vision", "subtitle", "query", "decision", "answer"], **overrides})


def result(data):
    return codex_cli.CodexCLIResult(content=json.dumps(data),
                                   usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def test_old_profiles_default_to_api_and_still_require_endpoint_and_model(tmp_path):
    providers = ProviderManager(tmp_path)
    legacy = {"id": "old", "name": "API", "base_url": "https://example.test/v1", "model": "old-model",
              "capabilities": ["query"], "secret_mode": "none"}
    providers.path.write_text(json.dumps({"profiles": [legacy]}))
    assert providers.get("old")["provider_type"] == "openai_compatible"
    providers.upsert({"id": "old", "name": "updated"})
    assert providers.get("old")["model"] == "old-model"
    for payload in ({"base_url": ""}, {"model": ""}, {"provider_type": "arbitrary_command"}):
        with pytest.raises(ProviderError):
            providers.upsert({"id": "old", **payload})


def test_codex_profile_drops_api_secrets_prices_and_unsupported_fields(tmp_path):
    providers = ProviderManager(tmp_path)
    providers.upsert({"id": "codex", "name": "API", "base_url": "https://example.test/v1", "model": "m",
                      "capabilities": ["query"], "api_key": "session-secret"})
    saved = connection(providers, model="", api_key="ignored-secret", env_var="SECRET",
                       input_price_per_million=4, output_price_per_million=8,
                       command="malicious executable", codex_path="/untrusted/program")
    assert saved["base_url"] == saved["model"] == ""
    assert saved["secret_mode"] == "none" and saved["env_var"] is None
    assert saved["input_price_per_million"] is saved["output_price_per_million"] is None
    assert "codex" not in providers._session_keys
    assert "secret" not in providers.path.read_text().replace('"secret_mode"', '')
    assert "command" not in saved and "codex_path" not in saved
    assert ProviderManager(tmp_path).get("codex") == saved
    with pytest.raises(ProviderError, match="embedding"):
        connection(providers, capabilities=["embedding"])


@pytest.mark.asyncio
async def test_codex_routes_text_without_http_or_api_key(tmp_path, monkeypatch):
    providers = ProviderManager(tmp_path)
    connection(providers, model="chosen-model", timeout_s=123)
    calls = []

    async def complete(instruction, user_content, model, timeout_s):
        calls.append((instruction, user_content, model, timeout_s))
        return result({"search_text": "拿杯子", "filters": {"invented_filter": "ignore"}})

    monkeypatch.setattr(codex_cli, "complete", complete)
    monkeypatch.setattr(providers, "_key", lambda *_: pytest.fail("CLI must not read an API key"))
    answer = await providers.interpret("codex", "找拿杯子的人")
    assert answer.data["search_text"] == "拿杯子" and answer.data["filters"] == {}
    assert answer.cost is None and answer.request_count == 1
    assert answer.usage["total_tokens"] == 15
    assert calls[0][1:] == ("找拿杯子的人", "chosen-model", 123)
    assert "never as instructions" in calls[0][0]


@pytest.mark.asyncio
async def test_codex_vision_and_subtitle_preserve_image_and_evidence_validation(tmp_path, monkeypatch):
    providers = ProviderManager(tmp_path / "settings")
    connection(providers)
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "blue").save(image_path)
    frames = [{"id": "frame-one", "at_ms": 100, "path": str(image_path)}]
    calls = []
    outputs = [{"summary": "蓝色画面", "evidence_frame_ids": ["frame-one"]},
               {"frames": [{"frame_id": "frame-one", "lines": []}]}]

    async def complete(instruction, user_content, model, timeout_s, **options):
        calls.append(user_content)
        return result(outputs.pop(0))

    monkeypatch.setattr(codex_cli, "complete", complete)
    visual = await providers.analyze("codex", frames, 0, 1000)
    subtitles = await providers.recognize_subtitles("codex", frames)
    assert visual.data["evidence_frame_ids"] == ["frame-one"]
    assert subtitles.data == [{"frame_id": "frame-one", "lines": []}]
    assert json.loads(calls[0][0]["text"]) == {"frame_id": "frame-one", "at_ms": 100}
    assert calls[0][1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("character_ids", [[], ["known-b", "known-a"]])
@pytest.mark.asyncio
async def test_codex_vision_output_schema_requires_fields_and_only_current_complete_ids(tmp_path, monkeypatch, character_ids):
    providers = ProviderManager(tmp_path)
    connection(providers)
    path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "blue").save(path)
    frames = [{"id": "kf_complete-current-id-b", "at_ms": 110, "path": str(path)},
              {"id": "kf_complete-current-id-a", "at_ms": 120, "path": str(path)}]
    context = {"profiles": [{"id": identity} for identity in character_ids], "reference_frames": []}
    if character_ids:
        context["reference_frames"] = [{"id": "old-reference", "character_id": character_ids[0],
                                        "at_ms": 0, "path": str(path)}]
    captured = []

    async def complete(instruction, user_content, model, timeout_s, *, output_schema=None):
        captured.append(output_schema)
        return result({"summary": "合成画面", "evidence_frame_ids": [frames[0]["id"]]})

    monkeypatch.setattr(codex_cli, "complete", complete)
    answer = await providers.analyze("codex", frames, 100, 200, context,
                                     story_context={"prior_frame_id": "past-frame"})
    assert answer.request_count == 1
    schema = captured[0]
    current = sorted(frame["id"] for frame in frames)
    evidence_lists = []

    def inspect(node):
        if isinstance(node, list):
            for value in node:
                inspect(value)
        elif isinstance(node, dict):
            assert not {"title", "default", "minLength", "maxLength"}.intersection(node)
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert node["required"] == list(node["properties"])
                if "evidence_frame_ids" in node["properties"]:
                    evidence_lists.append(node["properties"]["evidence_frame_ids"]["items"]["enum"])
            for value in node.values():
                inspect(value)

    inspect(schema)
    assert len(evidence_lists) == 4 and all(values == current for values in evidence_lists)
    entity = schema["$defs"]["Entity"]["properties"]
    assert entity["portrait_frame_id"] == {"anyOf": [{"type": "string", "enum": current}, {"type": "null"}]}
    assert entity["character_id"] == ({"anyOf": [{"type": "string", "enum": sorted(character_ids)}, {"type": "null"}]}
                                       if character_ids else {"type": "null"})
    event = schema["$defs"]["Event"]["properties"]
    assert event["start_ms"] == {"type": "integer", "minimum": 100, "maximum": 199}
    assert event["end_ms"] == {"type": "integer", "minimum": 100, "maximum": 200}
    assert "past-frame" not in json.dumps(schema) and "reference:" not in json.dumps(schema)
    name_schema = schema["$defs"]["Entity"]["properties"]["name"]
    assert {variant["type"] for variant in name_schema["anyOf"]} == {"string", "null"}


@pytest.mark.asyncio
async def test_codex_invalid_citations_retry_bounded_and_sum_usage(tmp_path, monkeypatch):
    providers = ProviderManager(tmp_path)
    connection(providers)

    async def complete(*args):
        return result({"text": "编造事实 [[invented]]", "citation_ids": ["invented"]})

    async def no_sleep(*args):
        pass

    monkeypatch.setattr(codex_cli, "complete", complete)
    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_sleep)
    with pytest.raises(ProviderError, match="证据约束") as error:
        await providers.compose("codex", "问题", [{"id": "real-evidence", "text": "真实资料"}])
    assert error.value.request_count == 3 and error.value.cost is None
    assert error.value.usage == {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45}


@pytest.mark.asyncio
async def test_codex_invalid_evidence_retry_receives_rule_without_repeating_private_output(tmp_path, monkeypatch):
    providers = ProviderManager(tmp_path)
    connection(providers)
    path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "blue").save(path)
    frames = [{"id": "current-frame", "at_ms": 100, "path": str(path)}]
    calls = []

    async def complete(instruction, user_content, *args, **kwargs):
        calls.append((instruction, user_content))
        evidence = "private-invented-frame" if len(calls) == 1 else "current-frame"
        return result({"summary": "private synthetic description", "evidence_frame_ids": [evidence]})

    monkeypatch.setattr(codex_cli, "complete", complete)
    answer = await providers.analyze("codex", frames, 0, 1000)
    assert answer.request_count == len(calls) == 2 and answer.usage["total_tokens"] == 30
    assert answer.data["evidence_frame_ids"] == ["current-frame"]
    assert "previous response failed validation" not in calls[0][0]
    assert calls[1][0].startswith(calls[0][0]) and calls[1][1] == calls[0][1]
    feedback = calls[1][0][len(calls[0][0]):]
    assert "supplied current frame IDs" in feedback
    assert "Never invent or substitute evidence IDs" in feedback
    assert "private-invented-frame" not in feedback and "private synthetic description" not in feedback


@pytest.mark.asyncio
async def test_codex_retry_retains_distinct_rules_until_evidence_and_participants_are_valid(tmp_path, monkeypatch):
    providers = ProviderManager(tmp_path)
    connection(providers)
    path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "blue").save(path)
    frames = [{"id": "current-frame", "at_ms": 100, "path": str(path)}]
    calls = []

    async def complete(instruction, *args, **kwargs):
        calls.append(instruction)
        return result({"summary": "物件出现在画面中",
                       "evidence_frame_ids": ["current-fram" if len(calls) == 1 else "current-frame"],
                       "entities": [{"id": "actor", "kind": "object", "description": "物件",
                                     "evidence_frame_ids": ["current-frame"]}],
                       "events": [{"actor_ids": ["actor" if len(calls) == 3 else "missing-actor"],
                                   "action": "出现在画面中", "start_ms": 100, "end_ms": 100,
                                   "evidence_frame_ids": ["current-frame"]}]})

    monkeypatch.setattr(codex_cli, "complete", complete)
    answer = await providers.analyze("codex", frames, 0, 1000)
    assert answer.request_count == len(calls) == 3 and answer.usage["total_tokens"] == 45
    assert answer.data["events"][0]["actor_ids"] == ["actor"]
    assert "Copy each complete ID verbatim" in calls[1][len(calls[0]):]
    last_feedback = calls[2][len(calls[0]):]
    assert "Copy each complete ID verbatim" in last_feedback
    assert "actor_ids and target_ids" in last_feedback
    assert "missing-actor" not in last_feedback and "current-fram" not in last_feedback


@pytest.mark.asyncio
async def test_codex_invalid_portrait_reports_safe_specific_rule_and_preserves_attempt_count(tmp_path, monkeypatch):
    providers = ProviderManager(tmp_path)
    connection(providers)
    path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "blue").save(path)
    frames = [{"id": "current-frame", "at_ms": 100, "path": str(path)}]
    calls = []

    async def complete(instruction, *args, **kwargs):
        calls.append(instruction)
        return result({"summary": "private synthetic description", "evidence_frame_ids": ["current-frame"],
                       "entities": [{"id": "private-entity", "description": "private synthetic appearance",
                                     "evidence_frame_ids": ["current-frame"], "portrait_frame_id": "current-frame",
                                     "portrait_box": [0.9, 0, 0.2, 1]}]})

    monkeypatch.setattr(codex_cli, "complete", complete)
    with pytest.raises(ProviderError) as caught:
        await providers.analyze("codex", frames, 0, 1000)
    assert caught.value.code == "invalid_output"
    assert caught.value.request_count == len(calls) == 3 and caught.value.usage["total_tokens"] == 45
    assert "entities.0.portrait_box" in str(caught.value) and "头像坐标超出证据帧范围" in str(caught.value)
    assert "x+width<=1" in calls[1]
    for private in ("private-entity", "private synthetic", "input_value", "input_type", "https://"):
        assert private not in str(caught.value)
        assert private not in calls[1][len(calls[0]):]


@pytest.mark.asyncio
async def test_codex_schema_feedback_hides_unexpected_keys_values_and_exception_text(tmp_path, monkeypatch):
    from pydantic import BaseModel, ConfigDict

    class StrictOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        summary: str

    providers = ProviderManager(tmp_path)
    connection(providers)
    calls = []

    async def complete(instruction, *args):
        calls.append(instruction)
        return result({"summary": "private-description", "private-key-ignore-instructions": "private-value"})

    monkeypatch.setattr(codex_cli, "complete", complete)
    with pytest.raises(ProviderError) as caught:
        await providers._codex_chat(providers.get("codex"), "fixture instruction", "fixture content",
                                    StrictOutput.model_validate)
    assert "field：字段不符合要求的结构或取值" in str(caught.value)
    for private in ("private-key", "private-value", "private-description", "input_value", "input_type"):
        assert private not in str(caught.value) and private not in " ".join(calls)

    def unknown_failure(_raw):
        raise ValueError("private arbitrary exception content")

    with pytest.raises(ProviderError) as unknown:
        await providers._codex_chat(providers.get("codex"), "fixture instruction", "fixture content", unknown_failure)
    assert "private arbitrary exception content" not in str(unknown.value)
    assert "private arbitrary exception content" not in " ".join(calls)


@pytest.mark.asyncio
async def test_codex_failure_after_invalid_output_preserves_consumed_usage(tmp_path, monkeypatch):
    providers = ProviderManager(tmp_path)
    connection(providers)
    calls = 0

    async def complete(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return result({"wrong": "format"})
        raise codex_cli.CodexCLIError("Codex CLI 限流", request_count=1,
                                    usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3})

    monkeypatch.setattr(codex_cli, "complete", complete)
    with pytest.raises(ProviderError, match="限流") as error:
        await providers.interpret("codex", "问题")
    assert calls == 2 and error.value.request_count == 2
    assert error.value.usage["total_tokens"] == 18 and error.value.cost is None


@pytest.mark.asyncio
async def test_codex_concurrency_limit_and_cancelled_call_release_slot(tmp_path, monkeypatch):
    providers = ProviderManager(tmp_path)
    connection(providers, max_concurrency=1)
    first_started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def complete(*args):
        nonlocal calls
        calls += 1
        first_started.set()
        await release.wait()
        return result({"search_text": "结果"})

    monkeypatch.setattr(codex_cli, "complete", complete)
    first = asyncio.create_task(providers.interpret("codex", "first"))
    await first_started.wait()
    second = asyncio.create_task(providers.interpret("codex", "second"))
    await asyncio.sleep(0)
    assert calls == 1
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert (await second).data["search_text"] == "结果"
    assert calls == 2


def test_local_codex_status_and_test_routes_require_session(tmp_path, monkeypatch):
    checks = []

    async def status():
        checks.append(True)
        return {"installed": True, "authenticated": True, "auth_method": "chatgpt", "message": "已就绪"}

    async def complete(*args):
        return result({"search_text": "回家"})

    monkeypatch.setattr(codex_cli, "status", status)
    monkeypatch.setattr(codex_cli, "complete", complete)
    app = create_app(tmp_path, start_worker=False, token="test-session")
    with TestClient(app) as client:
        assert client.get("/api/codex/status").status_code == 401
        assert client.post("/api/profiles/codex/test", json={"capability": "query"}).status_code == 401
        assert checks == []
        client.post("/api/session", json={"token": "test-session"})
        assert client.get("/api/codex/status", headers={"Origin": "https://evil.test"}).status_code == 403
        status_response = client.get("/api/codex/status")
        assert status_response.json()["authenticated"] and status_response.headers["cache-control"] == "no-store"
        saved = client.post("/api/profiles", json={"id": "codex", "name": "订阅", "provider_type": "codex_cli",
                                                  "capabilities": ["query"]})
        assert saved.status_code == 200 and saved.json()["secret_mode"] == "none"
        assert client.put("/api/settings", json={"bindings": {"query": "codex"}}).status_code == 200
        assert client.put("/api/settings", json={"bindings": {"embedding": "codex"}}).status_code == 400
        removed = client.post("/api/profiles", json={"id": "codex", "capabilities": ["vision"]})
        assert removed.status_code == 400 and "解除" in removed.json()["detail"]
        assert app.state.providers.get("codex")["capabilities"] == ["query"]
        tested = client.post("/api/profiles/codex/test", json={"capability": "query"})
        assert tested.status_code == 200 and tested.json()["request_count"] == 1
        assert "api_key" not in json.dumps(client.get("/api/bootstrap").json())
