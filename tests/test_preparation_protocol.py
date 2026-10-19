"""Codex preparation receives typed output constraints, with evidence guards intact."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scenerecall import codex_cli
from scenerecall import providers as provider_module
from scenerecall.providers import Entity, Event, ProviderError
from scenerecall.scene_preparation import VisualPreparation
from test_codex_cli import cli as cli  # noqa: F401
from test_scene_jobs import enqueue, execute
from test_scene_jobs import scene_worker as scene_worker  # noqa: F401
from test_scene_providers import chunk, images, manager


KINDS = ["person", "object", "animal", "character", "anthropomorphic_character", "other"]


def prepared():
    person = Entity(id="P1", kind="character", description="Animated person turns",
                    evidence_frame_ids=["frame-1"], is_character=True, character_id=None,
                    match_confidence=0, match_reason="Candidate only; identity remains uncertain").model_dump()
    event = Event(actor_ids=["P1"], action="Turns", start_ms=1000, end_ms=1000,
                  evidence_frame_ids=["frame-1"]).model_dump()
    value = {"chunks": [chunk(["frame-1", "frame-2"], entities=[person], events=[event],
                              dialogue_facts=[{"text": "A supplied line", "cue_ids": ["cue-full-id"]}])],
             "unresolved": []}
    return VisualPreparation.model_validate(value).model_dump()


def context():
    return {"profiles": [{"id": "known-character", "name": "Prior only"}], "reference_frames": [],
            "cast_policy": {"mode": "bangumi"}}


def subtitles():
    return {"start_ms": 1000, "end_ms": 3000,
            "cues": [{"id": "cue-full-id", "start_ms": 1000, "end_ms": 2000, "text": "A supplied line"}]}


def inspect_strict_schema(schema):
    def inspect(node):
        if isinstance(node, list):
            for value in node:
                inspect(value)
        elif isinstance(node, dict):
            assert not {"default", "title", "minLength", "maxLength"}.intersection(node)
            if node.get("type") == "object":
                assert node["required"] == list(node["properties"])
                assert node["additionalProperties"] is False
            for value in node.values():
                inspect(value)
    inspect(schema)
    definitions = schema["$defs"]
    entity = definitions["Entity"]["properties"]
    assert entity["kind"]["enum"] == KINDS and "human" not in entity["kind"]["enum"]
    assert entity["evidence_frame_ids"]["items"]["enum"] == ["frame-1", "frame-2"]
    assert entity["portrait_frame_id"]["anyOf"][0]["enum"] == ["frame-1", "frame-2"]
    assert entity["character_id"]["anyOf"][0]["enum"] == ["known-character"]
    assert entity["name"] == {"type": "null"}
    for name in ("PreparedChunk", "VisualFact", "SceneInference"):
        properties = definitions[name]["properties"]
        assert properties["frame_ids"]["items"]["enum"] == ["frame-1", "frame-2"]
    for field in ("representative_frame_ids", "deep_frame_ids"):
        assert definitions["PreparedChunk"]["properties"][field]["items"]["enum"] == ["frame-1", "frame-2"]
    assert definitions["PreparedChunk"]["properties"]["entities"]["items"] == {"$ref": "#/$defs/Entity"}
    assert definitions["PreparedChunk"]["properties"]["events"]["items"] == {"$ref": "#/$defs/Event"}
    for name in ("DialogueFact", "SceneInference"):
        assert definitions[name]["properties"]["cue_ids"]["items"]["enum"] == ["cue-full-id"]
    event = definitions["Event"]["properties"]
    assert event["evidence_frame_ids"]["items"]["enum"] == ["frame-1", "frame-2"]
    assert event["start_ms"]["minimum"] == 1000 and event["start_ms"]["maximum"] == 2999
    assert event["end_ms"]["maximum"] == 3000
    assert "frame-0" not in json.dumps(schema) and "prior-frame" not in json.dumps(schema)


@pytest.mark.asyncio
async def test_preparation_schema_reaches_actual_cli_private_file_and_keeps_kind_enum(cli, tmp_path):
    configure, calls = cli
    value = prepared()
    configure(events=[{"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(value)}},
                      {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 9}}])
    real = manager(tmp_path, lambda request: pytest.fail("Codex must not call HTTP"), provider_type="codex_cli")
    frames = images(tmp_path, 3)
    result = await real.prepare_visual("scene", frames[1:], frames[:1], subtitles(),
                                       {"prior_frame_id": "prior-frame"}, context())
    [call] = calls()
    inspect_strict_schema(call["output_schema"])
    assert "Entity kind must be exactly one of person, object, animal, character, anthropomorphic_character, other" in call["instructions"]
    assert result.request_count == 1 and result.data["chunks"][0]["entities"][0]["kind"] == "character"
    assert result.data["chunks"][0]["entities"][0]["character_id"] is None
    schema_path = Path(call["args"][call["args"].index("--output-schema") + 1])
    assert not schema_path.exists() and not Path(call["cwd"]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["kind", "context_frame", "foreign_cue", "foreign_actor"])
async def test_preparation_adapter_still_rejects_invalid_kind_and_evidence_without_extra_attempts(
        tmp_path, monkeypatch, defect):
    value = prepared()
    if defect == "kind":
        value["chunks"][0]["entities"][0]["kind"] = "human"
    elif defect == "context_frame":
        value["chunks"][0]["entities"][0]["evidence_frame_ids"] = ["frame-0"]
    elif defect == "foreign_cue":
        value["chunks"][0]["dialogue_facts"][0]["cue_ids"] = ["made-up-cue"]
    else:
        value["chunks"][0]["events"][0]["actor_ids"] = ["absent-entity"]
    captured = []

    async def complete(instruction, content, model, timeout_s, *, output_schema):
        inspect_strict_schema(output_schema)
        captured.append(deepcopy(output_schema))
        # Simulate corrupt output bypassing generation constraints. Runtime
        # validation must still reject it, never coerce a kind or trust citations.
        return codex_cli.CodexCLIResult(json.dumps(value), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    monkeypatch.setattr(codex_cli, "complete", complete)
    real = manager(tmp_path, lambda request: pytest.fail("Unexpected HTTP"), provider_type="codex_cli")
    frames = images(tmp_path, 3)
    with pytest.raises(ProviderError) as caught:
        await real.prepare_visual("scene", frames[1:], frames[:1], subtitles(), None, context())
    assert caught.value.code == "invalid_output" and caught.value.request_count == 1 and len(captured) == 1
    if defect == "kind":
        assert "kind" in str(caught.value) and "取值不在允许范围" in str(caught.value)


@pytest.mark.asyncio
async def test_preparation_without_cues_uses_empty_lists_instead_of_empty_enum(tmp_path, monkeypatch):
    captured = []

    async def complete(instruction, content, model, timeout_s, *, output_schema):
        captured.append(output_schema)
        value = VisualPreparation.model_validate({"chunks": [chunk(["frame-0"])], "unresolved": []}).model_dump()
        return codex_cli.CodexCLIResult(json.dumps(value))

    monkeypatch.setattr(codex_cli, "complete", complete)
    real = manager(tmp_path, lambda request: pytest.fail("Unexpected HTTP"), provider_type="codex_cli")
    result = await real.prepare_visual("scene", images(tmp_path, 1), [], None, None)
    assert result.request_count == 1
    for name in ("DialogueFact", "SceneInference"):
        assert captured[0]["$defs"][name]["properties"]["cue_ids"]["maxItems"] == 0
        assert captured[0]["$defs"][name]["properties"]["cue_ids"]["minItems"] == 0
    assert captured[0]["$defs"]["PreparedChunk"]["properties"]["dialogue_facts"]["maxItems"] == 0
    assert '"enum": []' not in json.dumps(captured[0])


@pytest.mark.asyncio
async def test_successful_paid_preparation_from_before_schema_is_reused(scene_worker, tmp_path, monkeypatch):
    worker = scene_worker(2)
    original_get = worker.provider.get
    monkeypatch.setattr(worker.provider, "get", lambda identity: {**original_get(identity), "provider_type": "codex_cli"})
    calls = []

    async def complete(instruction, content, model, timeout_s, **options):
        calls.append(options)
        mapping = json.loads(next(part["text"].removeprefix("Original frame map: ") for part in content
                                  if part.get("text", "").startswith("Original frame map: ")))
        value = {"chunks": [chunk([row["frame_id"] for row in mapping if not row["reference_only"]])],
                 "unresolved": []}
        return codex_cli.CodexCLIResult(json.dumps(value), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    monkeypatch.setattr(codex_cli, "complete", complete)
    real = manager(tmp_path, lambda request: pytest.fail("Unexpected HTTP"), provider_type="codex_cli")

    async def adapter(identity, *args):
        return await real.prepare_visual("scene", *args)

    monkeypatch.setattr(worker.provider, "prepare_visual", adapter)
    # Simulate the previous protocol, which paid successfully without a schema.
    with monkeypatch.context() as old:
        old.setattr(provider_module, "_preparation_output_schema", lambda *args, **kwargs: None)
        first = await execute(worker, enqueue(worker))
    assert first["status"] == "completed" and first["request_count"] == 1 and calls == [{}], first
    second = await execute(worker, enqueue(worker))
    assert second["status"] == "completed" and second["request_count"] == 0, second
    assert calls == [{}]
