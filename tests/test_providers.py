import json

import httpx
import pytest
from PIL import Image
from scenerecall.providers import MAX_OUTPUT_TOKENS, ProviderError, ProviderManager


@pytest.fixture
def frames(tmp_path):
    result = []
    for index in range(2):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (32, 32), "red" if index else "blue").save(path)
        result.append({"id": f"frame-{index}", "path": str(path), "at_ms": index * 500})
    return result


def manager(tmp_path, handler, **profile):
    providers = ProviderManager(tmp_path / "settings", transport=httpx.MockTransport(handler))
    connection = providers.upsert({
        "id": "test", "name": "测试模型", "base_url": "https://example.test/v1", "model": "vision-test",
        "capabilities": ["vision", "subtitle", "embedding", "query", "decision", "answer"],
        "secret_mode": "session", "api_key": "top-secret-123",
        "input_price_per_million": 1, "output_price_per_million": 2, **profile,
    })
    return providers, connection


def chat(data, **kwargs):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(data, ensure_ascii=False)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, **kwargs,
    })


def observation():
    return {
        "summary": "左侧的人拿起杯子", "evidence_frame_ids": ["frame-0", "frame-1"],
        "entities": [
            {"id": "person1", "kind": "person", "description": "穿红衣服", "position": "left",
             "name": None, "evidence_frame_ids": ["frame-0"]},
            {"id": "cup", "kind": "object", "description": "白色杯子", "evidence_frame_ids": ["frame-1"]},
        ],
        "events": [{"actor_ids": ["person1"], "action": "拿起", "target_ids": ["cup"],
                    "start_ms": 0, "end_ms": 500, "evidence_frame_ids": ["frame-0", "frame-1"]}],
        "spatial_observations": [{"subject_id": "person1", "relation": "left",
                                  "object_id": "cup", "evidence_frame_ids": ["frame-0"]}],
        "uncertainties": ["镜头外动作未知"],
    }


def test_secret_storage_never_serializes_key_and_empty_preserves(tmp_path, monkeypatch):
    captured = []
    providers, connection = manager(tmp_path, lambda request: captured.append(request) or chat({}))
    assert "api_key" not in connection
    providers.upsert({"id": "test", "name": "新名字", "api_key": ""})
    assert providers._key(providers.get("test")) == "top-secret-123"
    assert "top-secret" not in providers.path.read_text(encoding="utf-8")
    assert "top-secret" not in json.dumps(providers.list())
    restarted = ProviderManager(providers.settings_dir)
    with pytest.raises(ProviderError, match="尚无 API Key"):
        restarted._key(restarted.get("test"))
    monkeypatch.setenv("SCENERECALL_TEST_KEY", "env-only-secret")
    connection = providers.upsert({"id": "test", "secret_mode": "env", "env_var": "SCENERECALL_TEST_KEY"})
    assert providers._key(connection) == "env-only-secret"
    assert "env-only-secret" not in providers.path.read_text(encoding="utf-8")


def test_keyring_mode_separates_libraries_and_deletes(tmp_path, monkeypatch):
    secrets = {}
    monkeypatch.setattr("scenerecall.providers.keyring.set_password", lambda service, key, value: secrets.__setitem__((service, key), value))
    monkeypatch.setattr("scenerecall.providers.keyring.get_password", lambda service, key: secrets.get((service, key)))
    monkeypatch.setattr("scenerecall.providers.keyring.delete_password", lambda service, key: secrets.pop((service, key), None))
    providers, profile = manager(tmp_path, lambda request: chat({}), secret_mode="keyring")
    other, other_profile = manager(tmp_path / "other", lambda request: chat({}), secret_mode="keyring", api_key="other-secret")
    assert providers._key(profile) == "top-secret-123"
    assert other._key(other_profile) == "other-secret"
    providers.delete("test")
    assert len(secrets) == 1


@pytest.mark.asyncio
async def test_multiframe_vision_sends_actual_images_and_validates_entities(tmp_path, frames):
    captured = []
    providers, _ = manager(tmp_path, lambda request: captured.append(request) or chat(observation()))
    result = await providers.analyze("test", frames, 0, 1000)
    assert result.data["entities"][0]["name"] is None
    assert result.request_count == 1 and result.cost == pytest.approx(0.00002)
    payload = json.loads(captured[0].content)
    assert payload["max_tokens"] == MAX_OUTPUT_TOKENS
    assert len([x for x in payload["messages"][1]["content"] if x["type"] == "image_url"]) == 2
    assert captured[0].headers["authorization"] == "Bearer top-secret-123"


@pytest.mark.parametrize("defect", ["evidence", "participant", "time", "name"])
@pytest.mark.asyncio
async def test_invalid_observation_retries_bounded_and_accounts_usage(tmp_path, frames, monkeypatch, defect):
    data = observation()
    if defect == "evidence":
        data["events"][0]["evidence_frame_ids"] = ["invented-frame"]
    elif defect == "participant":
        data["events"][0]["actor_ids"] = ["invented-person"]
    elif defect == "time":
        data["events"][0]["end_ms"] = 2000
    else:
        data["entities"][0]["name"] = "无依据的人名"
    async def no_sleep(*args):
        pass
    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_sleep)
    providers, _ = manager(tmp_path, lambda request: chat(data))
    with pytest.raises(ProviderError) as error:
        await providers.analyze("test", frames, 0, 1000)
    assert error.value.request_count == 3
    assert error.value.usage == {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45}
    assert error.value.cost == pytest.approx(0.00006)
    assert "top-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_auth_failure_does_not_retry_or_echo_provider_error(tmp_path):
    providers, _ = manager(tmp_path, lambda request: httpx.Response(401, json={"error": "top-secret-123"}))
    with pytest.raises(ProviderError, match="认证") as error:
        await providers.interpret("test", "拿杯子")
    assert error.value.request_count == 1
    assert "top-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_rate_limit_then_success_and_missing_usage_is_unknown_cost(tmp_path, monkeypatch):
    attempts = []
    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return chat({"search_text": "拿杯子", "filters": {}}, usage=None)
    async def no_sleep(*args):
        pass
    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_sleep)
    providers, _ = manager(tmp_path, handler)
    result = await providers.interpret("test", "拿杯子")
    assert result.request_count == 2 and result.cost is None


@pytest.mark.asyncio
async def test_subtitles_require_every_frame_preserve_languages_and_empty_frames(tmp_path, frames):
    output = {"frames": [
        {"frame_id": "frame-1", "lines": []},
        {"frame_id": "frame-0", "lines": [{"text": "回家", "language": "zh", "uncertain": False},
                                           {"text": "帰ろう", "language": "ja", "uncertain": True}]},
    ]}
    providers, _ = manager(tmp_path, lambda request: chat(output))
    result = await providers.recognize_subtitles("test", frames)
    assert result.data[0]["frame_id"] == "frame-0"
    assert result.data[0]["lines"][1]["text"] == "帰ろう"
    assert result.data[1]["lines"] == []


@pytest.mark.asyncio
async def test_embedding_reorders_response_and_accounts_usage(tmp_path):
    providers, _ = manager(tmp_path, lambda request: httpx.Response(200, json={
        "data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}],
        "usage": {"prompt_tokens": 12, "total_tokens": 12},
    }))
    result = await providers.embed("test", ["左边", "右边"])
    assert result.data == [[1.0, 0.0], [0.0, 1.0]]
    assert result.usage["input_tokens"] == 12


@pytest.mark.asyncio
async def test_decision_and_answer_reject_invented_citations(tmp_path, monkeypatch):
    async def no_sleep(*args):
        pass
    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_sleep)
    providers, _ = manager(tmp_path, lambda request: chat({"candidates": [
        {"id": "candidate", "score": 1, "match_type": "full", "evidence_ids": ["fake"]}]}))
    candidates = [{"id": "candidate", "record_id": "candidate", "text": "真实资料"}]
    with pytest.raises(ProviderError):
        await providers.decide("test", "问题", candidates)
    providers._transport = httpx.MockTransport(lambda request: chat({"text": "发生了事情 [[fake]]", "citation_ids": ["fake"]}))
    with pytest.raises(ProviderError):
        await providers.compose("test", "问题", candidates)


@pytest.mark.asyncio
async def test_explicit_keyless_local_service_and_unsupported_capability(tmp_path):
    captured = []
    providers, _ = manager(tmp_path, lambda request: captured.append(request) or chat({"search_text": "回家"}),
                           secret_mode="none", capabilities=["query"])
    await providers.interpret("test", "回家")
    assert "authorization" not in captured[0].headers
    with pytest.raises(ProviderError, match="未启用"):
        await providers.embed("test", ["text"])


def test_rejects_url_embedded_credentials(tmp_path):
    with pytest.raises(ValueError):
        manager(tmp_path, lambda request: chat({}), base_url="https://user:secret@example.test/v1")


def test_profile_validation_errors_never_echo_secret_input(tmp_path):
    providers = ProviderManager(tmp_path)
    with pytest.raises(ProviderError) as error:
        providers.upsert({"api_key": "secret-in-invalid-profile"})
    assert "secret-in-invalid-profile" not in str(error.value)
    assert "name" in str(error.value)


@pytest.mark.asyncio
async def test_malformed_usage_and_content_remain_bounded(tmp_path, monkeypatch):
    async def no_sleep(*args):
        pass
    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_sleep)
    providers, _ = manager(tmp_path, lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": 123}}], "usage": "invalid-usage",
    }))
    with pytest.raises(ProviderError) as error:
        await providers.interpret("test", "text")
    assert error.value.request_count == 3 and error.value.cost is None


@pytest.mark.asyncio
async def test_partial_usage_does_not_claim_zero_cost(tmp_path):
    providers, _ = manager(tmp_path, lambda request: chat({"search_text": "text"}, usage={"total_tokens": 25}))
    result = await providers.interpret("test", "text")
    assert result.usage["total_tokens"] == 25 and result.cost is None
