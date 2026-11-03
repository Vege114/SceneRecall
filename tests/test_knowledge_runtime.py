"""Independent knowledge RPC and the pinned SDK's actual storage contract."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import httpx
import pytest

from scenerecall.knowledge_runtime import KnowledgeRuntime, KnowledgeUnavailable
from scenerecall.library import atomic_json
from scenerecall.providers import AIResult, ProviderError, ProviderManager


class Providers:
    def __init__(self):
        self.calls = []
        self.profiles = {
            "text": {"id": "text", "provider_type": "codex_cli", "capabilities": ["query"],
                     "model": "stub-model", "reasoning_effort": "low"},
            "embed": {"id": "embed", "provider_type": "openai_compatible", "capabilities": ["embedding"],
                      "model": "stub-embedding", "base_url": "http://unused.invalid"},
        }

    def get(self, identity):
        return self.profiles[identity]

    @contextmanager
    def use_profile(self, profile):
        yield

    async def complete_text(self, identity, **kwargs):
        self.calls.append(kwargs["purpose"])
        if kwargs["purpose"] == "knowledge_keyword":
            data = {"high_level_keywords": ["time control"], "low_level_keywords": ["Clock"]}
        elif kwargs["purpose"] == "knowledge_merge_summary":
            return AIResult("Clock stops time.", {"total_tokens": 5}, 1, None)
        else:
            data = {"entities": [{"name": "Clock", "type": "Object", "description": "Clock stops time."}],
                    "relationships": []}
        return AIResult(json.dumps(data), {"total_tokens": 10}, 1, None)

    async def embed(self, identity, texts):
        self.calls.append("embed")
        # Deterministic nonzero embeddings; they exercise the stores, never
        # claim semantic quality or incur an external provider request.
        return AIResult([[1.0, 0.1, 0.2, 0.3] for text in texts], {"total_tokens": len(texts)}, 1, 0)


def runtime(tmp_path):
    library = type("Library", (), {"root": tmp_path})()
    value = KnowledgeRuntime(library, Providers())
    value.configure({"enabled": True, "python": sys.executable,
                     "text_profile_id": "text", "embedding_profile_id": "embed", "embedding_backend": "api"})
    return value


def projection():
    return {"work_scope_id": "work-one", "work_version_id": "edition-one", "signature": "knowledge-rev",
            "cards": [{"card_id": "card-one", "text": "Clock stops time.",
                       "target_ref": {"kind": "concept", "concept_id": "clock"}, "source_refs": []}]}


def ready(value):
    config = value._configuration(require_preflight=False)
    config["identity"]["embedding_dim"] = 4
    atomic_json(value.path, {**value.settings(), "embedding_dim": 4,
                            "preflight_signature": __import__("scenerecall.library", fromlist=["digest"]).digest(config["identity"])})


def test_configuration_requires_explicit_model_and_invalidates_preflight(tmp_path):
    value = runtime(tmp_path)
    value.providers.profiles["text"]["model"] = ""
    assert value.status()["status"] == "configuration"
    value.configure({"text_model": "explicit-model"})
    ready(value)
    assert value.status()["configured"] is True
    old = value.signature(projection())
    value.configure({"text_model": "another-model"})
    with pytest.raises(KnowledgeUnavailable):
        value.signature(projection())
    ready(value)
    assert value.signature(projection()) != old


@pytest.mark.asyncio
async def test_busy_and_stale_fallback_do_not_start_process(tmp_path, monkeypatch):
    value = runtime(tmp_path)
    ready(value)

    async def no_start(*args):
        pytest.fail("must not spawn for a busy or absent index")

    monkeypatch.setattr(value, "_start", no_start)
    value._busy = True
    with pytest.raises(KnowledgeUnavailable, match="其他任务"):
        await value.query(projection(), "Clock")
    value._busy = False
    with pytest.raises(KnowledgeUnavailable) as error:
        await value.query(projection(), "Clock")
    assert error.value.code == "stale"
    assert value._busy is False


@pytest.mark.asyncio
async def test_duplex_reader_services_model_callback_before_operation_reply(tmp_path, monkeypatch):
    value = runtime(tmp_path)
    ready(value)
    # The child blocks its final response until receiving the embedding
    # callback result. A parent doing request/read serially would deadlock.
    child = tmp_path / "rpc_child.py"
    child.write_text('''import json,sys
r=json.loads(sys.stdin.readline())
print(json.dumps({"type":"callback","id":"cb","method":"embed","params":{"texts":["x"],"purpose":"knowledge_embed_query"}}),flush=True)
c=json.loads(sys.stdin.readline())
print(json.dumps({"id":r["id"],"result":{"callback":c["result"]}}),flush=True)
''', encoding="utf-8")
    import asyncio
    process = await asyncio.create_subprocess_exec(sys.executable, "-u", str(child),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    value._process = process
    value._enter("query")
    value._reader = asyncio.create_task(value._read(value._configuration()))
    result = await value._call("query", {}, 5)
    assert result["callback"] == [[1.0, 0.1, 0.2, 0.3]]
    assert any(row.get("purpose") == "knowledge_embed_query" and row["event"] == "model_finished"
               for row in value._ledger)
    await value._stop()


@pytest.mark.asyncio
async def test_timeout_kills_child_and_drains_callbacks(tmp_path):
    value = runtime(tmp_path)
    import asyncio
    value._process = await asyncio.create_subprocess_exec(sys.executable, "-u", "-c", "import time; time.sleep(20)",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    process = value._process
    with pytest.raises(KnowledgeUnavailable) as error:
        await value._call("query", {}, .05)
    assert error.value.code == "timeout" and process.returncode is not None
    assert not value._pending and not value._callbacks


@pytest.mark.asyncio
async def test_failed_model_attempts_remain_in_final_receipt(tmp_path, monkeypatch):
    value = runtime(tmp_path)
    ready(value)
    value._enter("query")

    async def failed(*args):
        raise ProviderError("bounded retry failed", request_count=3,
                            usage={"input_tokens": 17, "output_tokens": 2, "total_tokens": 19}, cost=.02)

    async def sent(*args):
        pass

    monkeypatch.setattr(value.providers, "embed", failed)
    monkeypatch.setattr(value, "_send", sent)
    await value._callback({"id": "cb", "method": "embed", "params": {"texts": ["x"],
        "purpose": "knowledge_embed_query"}}, value._configuration())
    row = next(row for row in value._ledger if row["event"] == "model_finished")
    assert row["request_count"] == 3 and row["usage"]["total_tokens"] == 19 and row["cost"] == .02
    assert row["usage_incomplete"] and row["error"]


@pytest.mark.asyncio
async def test_profile_edit_before_callback_never_calls_different_live_model(tmp_path, monkeypatch):
    value = runtime(tmp_path)
    ready(value)
    config = value._configuration()
    value._enter("query")
    value.providers.profiles["embed"]["model"] = "changed-live-model"

    async def sent(*args):
        pass

    monkeypatch.setattr(value, "_send", sent)
    await value._callback({"id": "cb", "method": "embed", "params": {"texts": ["x"]}}, config)
    assert not value.providers.calls
    assert next(row for row in value._ledger if row["event"] == "model_finished")["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [None, "enabled", "disabled"])
async def test_generic_text_adapter_preserves_order_and_raw_response(tmp_path, thinking):
    seen = []

    def transport(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "  raw\n{text}  "}}],
                                         "usage": {"prompt_tokens": 2, "completion_tokens": 3}})

    providers = ProviderManager(tmp_path, transport=httpx.MockTransport(transport))
    profile = providers.upsert({"name": "text", "base_url": "http://model.invalid/v1", "model": "text",
                                "capabilities": ["query"], "secret_mode": "none"})
    result = await providers.complete_text(profile["id"], system_prompt="sys", prompt="last",
        history_messages=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "prior"}],
        thinking_mode=thinking)
    assert result.data == "  raw\n{text}  "
    assert [message["content"] for message in seen[0]["messages"]] == ["sys", "first", "prior", "last"]
    if thinking is None:
        assert "thinking" not in seen[0]
    else:
        assert seen[0]["thinking"] == {"type": thinking}


@pytest.mark.asyncio
async def test_codex_rejects_api_thinking_without_starting_cli(tmp_path):
    providers = ProviderManager(tmp_path)
    profile = providers.upsert({"name": "codex", "provider_type": "codex_cli", "model": "stub-model",
                                "capabilities": ["query"]})
    with pytest.raises(ProviderError, match="API thinking"):
        await providers.complete_text(profile["id"], system_prompt="sys", prompt="x", thinking_mode="disabled")


def test_thinking_setting_is_explicit_and_invalidates_model_fingerprint(tmp_path):
    value = runtime(tmp_path)
    value.providers.profiles["text"].update(provider_type="openai_compatible", base_url="http://unused.invalid")
    ready(value)
    original = value.signature(projection())
    assert value.settings()["text_thinking"] is None
    value.configure({"text_thinking": "disabled"})
    with pytest.raises(KnowledgeUnavailable):
        value.signature(projection())
    ready(value)
    assert value.signature(projection()) != original
    assert value._configuration()["identity"]["text"]["thinking"] == "disabled"


SDK_PYTHON = Path(__file__).resolve().parents[1] / ".cache" / "knowledge-runtime" / "Scripts" / "python.exe"


@pytest.mark.asyncio
@pytest.mark.skipif(not SDK_PYTHON.exists(), reason="run setup_knowledge.ps1 for pinned SDK contract")
async def test_pinned_sdk_build_query_and_scope_reload_with_stub_models(tmp_path):
    value = runtime(tmp_path)
    value.configure({"python": str(SDK_PYTHON)})
    try:
        result = await value.preflight()
        assert result["runtime"]["version"] == "1.5.7"
        try:
            built = await value.build(projection(), is_current=lambda: True)
        except KnowledgeUnavailable:
            pytest.fail(value._protocol_error + "\n" + value._stderr_tail)
        assert built["chunk_count"] >= 1
        queried = await value.query(projection(), "Clock stops time", mode="mix")
        assert queried["data"]["chunks"]
        chunks = queried["mappings"]["chunks"]
        assert all(chunk["chunk_id"] in chunks for chunk in queried["data"]["chunks"])
        assert {card for mapping in chunks.values() for card in mapping["card_ids"]} == {"card-one"}
        candidates = json.loads((value._scope_root(projection()) / built["generation"] / "graph-candidates.json").read_text())
        assert candidates["entities"][0]["alignment_status"] == "unresolved"
        assert candidates["entities"][0]["supporting_chunk_ids"]
        # Loading persisted storage in a new process also has to preserve IDs.
        await value.close()
        reloaded = await value.query(projection(), "Clock stops time", mode="naive")
        assert reloaded["data"]["chunks"][0]["chunk_id"] in chunks
        other = {**projection(), "work_scope_id": "work-two"}
        with pytest.raises(KnowledgeUnavailable):
            await value.query(other, "Clock")
        # An edit racing with a full build cannot publish stale inputs.
        with pytest.raises(KnowledgeUnavailable) as error:
            await value.build(projection(), is_current=lambda: False)
        assert error.value.code == "stale"
        assert value.status(projection())["generation"] == built["generation"]
    finally:
        await value.close()


LOCAL_DIRECTORY = Path(__file__).resolve().parents[1] / ".cache" / "knowledge-models" / "multilingual-minilm-l12-v2"


@pytest.mark.asyncio
@pytest.mark.skipif(not SDK_PYTHON.exists() or not (LOCAL_DIRECTORY / "manifest.json").exists(),
                    reason="explicit local model setup required")
async def test_local_embedding_is_offline_windowed_and_traced(tmp_path):
    value = runtime(tmp_path)
    value.configure({"python": str(SDK_PYTHON), "embedding_backend": "local",
                     "local_model_dir": str(LOCAL_DIRECTORY)})
    try:
        preflight = await value.preflight()
        assert preflight["embedding_dim"] == 384
        built = await value.build(projection(), is_current=lambda: True)
        result = await value.query(projection(), "Clock stops time", mode="mix")
        assert result["generation"] == built["generation"]
        calls = result["diagnostics"]["calls"]
        assert any(call.get("backend") == "local" and call.get("purpose") == "knowledge_embed_query" for call in calls)
        assert "embed" not in value.providers.calls
        long_text = "保留全文的多语言知识。" * 150
        windowed = await value._call("local_embed", {"local_model_dir": str(LOCAL_DIRECTORY),
                                                     "texts": [long_text]}, 30)
        assert len(windowed["vectors"][0]) == 384
        assert windowed["diagnostics"]["window_count"] > 1
    finally:
        await value.close()
