import base64
import io
import json
from copy import deepcopy

import httpx
import pytest
from PIL import Image

from scenerecall import codex_cli
from scenerecall.models import Observation
from scenerecall.providers import MAX_OUTPUT_TOKENS, ProviderError, ProviderManager
from scenerecall.scene_preparation import (
    CONTEXTUAL_VISION_PROMPT_VERSION, MAX_PREPARATION_OUTPUT_TOKENS, MAX_SCENE_CONTEXT_BYTES,
    RECONCILIATION_PROMPT_VERSION, VISUAL_PREPARATION_PROMPT_VERSION, checked_frames, contact_sheets,
    validate_visual_preparation,
)


def images(tmp_path, count=3):
    result = []
    for number in range(count):
        path = tmp_path / f"frame-{number}.png"
        Image.new("RGB", (64, 36), (number * 3 % 256, 30, 80)).save(path)
        result.append({"id": f"frame-{number}", "path": str(path), "at_ms": number * 1000})
    return result


def cues():
    return [{"id": f"cue-{number}", "start_ms": number * 1000, "end_ms": (number + 1) * 1000,
             "text": text} for number, text in enumerate(["我们回家", "好的"])]


def chunk(frame_ids, **changes):
    return {"frame_ids": frame_ids, "summary": "角色站在门前", "visual_facts": [
                {"text": "一名角色在门前", "frame_ids": [frame_ids[0]]}],
            "dialogue_facts": [], "inferences": [],
            "representative_frame_ids": list(dict.fromkeys([frame_ids[0], frame_ids[-1]])),
            "needs_deep": False, "reason": "", "continues_previous": False, **changes}


def reply(data):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(data)},
                                                  "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})


def manager(tmp_path, handler, **kwargs):
    providers = ProviderManager(tmp_path / "settings", transport=httpx.MockTransport(handler))
    providers.upsert({"id": "scene", "name": "scene fixture", "base_url": "https://fixture.test/v1",
                      "model": "scene-test", "capabilities": ["preparation", "vision"], "secret_mode": "none",
                      "input_price_per_million": 1, "output_price_per_million": 2, **kwargs})
    return providers


@pytest.fixture(autouse=True)
def immediate_retries(monkeypatch):
    async def no_wait(_delay):
        pass
    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_wait)


def test_contact_sheets_retain_every_original_and_distinguish_context(tmp_path):
    frames = images(tmp_path, 64)
    sheets, mapping = contact_sheets(frames[8:], frames[:8], tmp_path / "sheets")
    assert len(sheets) == 4
    assert [item["frame_id"] for item in mapping] == [frame["id"] for frame in frames]
    assert [item["reference_only"] for item in mapping] == [True] * 8 + [False] * 56
    assert [item["label"] for item in mapping] == list(range(1, 65))
    for sheet in sheets:
        with Image.open(sheet["path"]) as picture:
            assert picture.size == (1280, 800)


@pytest.mark.asyncio
async def test_subtitle_preparation_transports_only_text_and_accounts_usage(tmp_path):
    requests = []
    data = {"segments": [{"cue_ids": ["cue-0", "cue-1"], "summary": "对话提议回家"}], "unresolved": []}
    providers = manager(tmp_path, lambda request: requests.append(request) or reply(data))
    result = await providers.prepare_subtitles("scene", cues(), {"summary": "之前在讨论去向"})
    assert result.data == data and result.request_count == 1
    assert result.cost == pytest.approx(0.00004)
    payload = json.loads(requests[0].content)
    assert payload["max_tokens"] == MAX_PREPARATION_OUTPUT_TOKENS
    assert json.loads(payload["messages"][1]["content"])["cues"] == cues()
    assert "image_url" not in requests[0].content.decode()


@pytest.mark.parametrize("partition", [["cue-1", "cue-0"], ["cue-0"], ["cue-0", "cue-0"], ["missing"]])
@pytest.mark.asyncio
async def test_subtitle_partition_rejects_missing_duplicate_reordered_and_unknown_cues(tmp_path, partition):
    providers = manager(tmp_path, lambda request: reply({
        "segments": [{"cue_ids": partition, "summary": "对话"}], "unresolved": []}))
    with pytest.raises(ProviderError) as caught:
        await providers.prepare_subtitles("scene", cues())
    assert caught.value.request_count == 3 and caught.value.usage["input_tokens"] == 60


@pytest.mark.asyncio
async def test_visual_preparation_uses_sheets_with_original_evidence_and_no_silent_scene_loss(tmp_path):
    frames = images(tmp_path)
    data = {"chunks": [chunk(["frame-1", "frame-2"], continues_previous=True)], "unresolved": []}
    requests = []
    providers = manager(tmp_path, lambda request: requests.append(request) or reply(data))
    result = await providers.prepare_visual("scene", frames[1:], frames[:1], None, {"summary": "过去的场景"})
    assert result.data == data
    payload = json.loads(requests[0].content)
    parts = payload["messages"][1]["content"]
    mapping = json.loads(next(part["text"].split(": ", 1)[1] for part in parts
                              if part.get("text", "").startswith("Original frame map:")))
    assert mapping[0]["reference_only"] is True
    picture = next(part["image_url"]["url"] for part in parts if part["type"] == "image_url")
    with Image.open(io.BytesIO(base64.b64decode(picture.split(",", 1)[1]))) as image:
        assert image.size == (1280, 800)
    assert result.data["chunks"][0]["frame_ids"] == ["frame-1", "frame-2"]


@pytest.mark.parametrize("defect", ["context", "sheet", "duplicate", "order", "cross_chunk", "cue", "empty_inference",
                                  "representative", "string_bool", "continuation", "source_type"])
@pytest.mark.asyncio
async def test_visual_output_rejects_evidence_and_partition_defects(tmp_path, defect):
    frames = images(tmp_path, 4)
    data = {"chunks": [chunk(["frame-1", "frame-2"]), chunk(["frame-3"])], "unresolved": []}
    first = data["chunks"][0]
    if defect == "context":
        first["visual_facts"][0]["frame_ids"] = ["frame-0"]
    elif defect == "sheet":
        first["visual_facts"][0]["frame_ids"] = ["contact-sheet-1"]
    elif defect == "duplicate":
        first["frame_ids"] = ["frame-1", "frame-1"]
    elif defect == "order":
        first["frame_ids"].reverse()
    elif defect == "cross_chunk":
        first["visual_facts"][0]["frame_ids"] = ["frame-3"]
    elif defect == "cue":
        first["dialogue_facts"] = [{"text": "他说了", "cue_ids": ["invented-cue"]}]
    elif defect == "empty_inference":
        first["inferences"] = [{"text": "推测", "frame_ids": [], "cue_ids": []}]
    elif defect == "representative":
        first["representative_frame_ids"] = ["frame-1"]
    elif defect == "string_bool":
        first["needs_deep"] = "false"
    elif defect == "continuation":
        data["chunks"][1]["continues_previous"] = True
    elif defect == "source_type":
        first["visual_facts"][0] = {"text": "从台词得知", "cue_ids": ["cue-0"]}
    providers = manager(tmp_path, lambda request: reply(data))
    with pytest.raises(ProviderError) as caught:
        await providers.prepare_visual("scene", frames[1:], frames[:1], cues(), None)
    assert caught.value.request_count == 3


def test_visual_facts_allow_typed_dialogue_and_marked_inference(tmp_path):
    frames = images(tmp_path, 2)
    data = {"chunks": [chunk(["frame-0", "frame-1"],
        dialogue_facts=[{"text": "台词提议回家", "cue_ids": ["cue-0"]}],
        inferences=[{"text": "可能打算离开", "frame_ids": ["frame-1"], "cue_ids": ["cue-0"]}],
        needs_deep=True, reason="需要核对动作")], "unresolved": []}
    assert validate_visual_preparation(data, frames, {"cue-0"}, has_context=False) == data


def test_same_timestamp_frames_stay_in_one_chunk_and_blank_text_is_rejected(tmp_path):
    frames = images(tmp_path, 2)
    for frame in frames:
        frame["at_ms"] = 1000
    split = {"chunks": [chunk(["frame-0"]), chunk(["frame-1"])], "unresolved": []}
    with pytest.raises(ValueError, match="相同时间"):
        validate_visual_preparation(split, frames, set(), has_context=False, start_ms=0, end_ms=2000)
    merged = {"chunks": [chunk(["frame-0", "frame-1"])], "unresolved": []}
    assert validate_visual_preparation(merged, frames, set(), has_context=False, start_ms=0, end_ms=2000) == merged
    merged["chunks"][0]["summary"] = "  "
    with pytest.raises(ValueError, match="空白"):
        validate_visual_preparation(merged, frames, set(), has_context=False, start_ms=0, end_ms=2000)


@pytest.mark.parametrize("field", ["dialogue_facts", "inferences"])
@pytest.mark.asyncio
async def test_batch_known_cue_cannot_be_cited_by_a_nonoverlapping_chunk(tmp_path, field):
    frames = images(tmp_path, 2)
    fact = {"text": "这一句出现在下一段", "cue_ids": ["cue-1"]}
    if field == "inferences":
        fact["frame_ids"] = ["frame-0"]
    data = {"chunks": [chunk(["frame-0"], **{field: [fact]}), chunk(["frame-1"])], "unresolved": []}
    providers = manager(tmp_path, lambda request: reply(data))
    with pytest.raises(ProviderError) as caught:
        await providers.prepare_visual("scene", frames, [],
                                       {"cues": cues(), "start_ms": 0, "end_ms": 2000}, None)
    assert caught.value.request_count == 3


@pytest.mark.asyncio
async def test_chunk_timing_accepts_overlapping_cues_and_supplied_outer_coverage(tmp_path):
    frames = images(tmp_path, 3)[1:]
    earlier_cue = {"id": "early", "start_ms": 100, "end_ms": 500, "text": "画面前的同段台词"}
    spanning_cue = {"id": "spanning", "start_ms": 1500, "end_ms": 2500, "text": "跨过分块边界"}
    data = {"chunks": [chunk(["frame-1"], dialogue_facts=[{"text": "先前台词", "cue_ids": ["early"]},
                              {"text": "跨界台词", "cue_ids": ["spanning"]}]),
                       chunk(["frame-2"], dialogue_facts=[{"text": "跨界台词", "cue_ids": ["spanning"]}])],
            "unresolved": []}
    providers = manager(tmp_path, lambda request: reply(data))
    result = await providers.prepare_visual("scene", frames, [],
        {"cues": [earlier_cue, spanning_cue], "start_ms": 0, "end_ms": 3000}, None)
    assert result.data == data


@pytest.mark.asyncio
async def test_oversized_or_invalid_inputs_fail_before_model_request(tmp_path):
    requests = []
    providers = manager(tmp_path, lambda request: requests.append(request) or reply({}))
    frames = images(tmp_path, 57)
    with pytest.raises(ValueError, match="56"):
        await providers.prepare_visual("scene", frames, [], None, None)
    with pytest.raises(ValueError, match="上下文超过"):
        await providers.prepare_subtitles("scene", cues(), {"summary": "x" * MAX_SCENE_CONTEXT_BYTES})
    with pytest.raises(ValueError, match="唯一"):
        checked_frames(frames[:2], frames[:1])
    assert not requests


@pytest.mark.asyncio
async def test_contextual_deep_vision_keeps_old_output_and_strict_current_evidence(tmp_path):
    frames = images(tmp_path, 2)
    data = {"summary": "一名角色", "entities": [], "events": [], "spatial_observations": [],
            "uncertainties": [], "evidence_frame_ids": ["frame-0"]}
    requests = []
    providers = manager(tmp_path, lambda request: requests.append(request) or reply(data))
    legacy = await providers.analyze("scene", frames, 0, 2000)
    contextual = await providers.analyze("scene", frames, 0, 2000, None,
                                        {"cues": cues()}, {"prior_frame_id": "past-frame"})
    assert legacy.data == contextual.data
    payloads = [json.loads(request.content) for request in requests]
    assert payloads[0]["max_tokens"] == MAX_OUTPUT_TOKENS
    assert CONTEXTUAL_VISION_PROMPT_VERSION not in payloads[0]["messages"][0]["content"]
    assert CONTEXTUAL_VISION_PROMPT_VERSION in payloads[1]["messages"][0]["content"]
    legacy_prompt = payloads[0]["messages"][0]["content"].split("\nReturn JSON only.")[0]
    contextual_prompt = payloads[1]["messages"][0]["content"]
    assert contextual_prompt.startswith(legacy_prompt)
    assert "at most 5 short observed_traits" not in legacy_prompt
    assert "at most 5 short observed_traits" in contextual_prompt
    assert "every visible character and important change" in contextual_prompt
    assert "smallest sufficient set" in contextual_prompt
    assert "repeated stationary poses" in contextual_prompt
    assert "distinctive static props and details" in contextual_prompt
    assert "grouped object entities" in contextual_prompt
    data["evidence_frame_ids"] = ["past-frame"]
    with pytest.raises(ProviderError):
        await providers.analyze("scene", frames, 0, 2000, None, {"cues": cues()}, {"prior_frame_id": "past-frame"})


def compact_vision_observation():
    return {"summary": "角色们在控制室内，一人按下控制台按钮后转身。", "entities": [
        {"id": f"actor-{number}", "kind": "character", "description": f"可见角色{number}",
         "observed_traits": ["圆脸"], "evidence_frame_ids": ["frame-0"]}
        for number in range(10)],
        "events": [
            {"actor_ids": ["actor-0"], "action": "按下按钮", "start_ms": 0, "end_ms": 1000,
             "evidence_frame_ids": ["frame-0", "frame-1"]},
            {"actor_ids": ["actor-0"], "action": "转身", "start_ms": 1000, "end_ms": 2000,
             "evidence_frame_ids": ["frame-1", "frame-2"]}],
        "spatial_observations": [{"subject_id": "actor-0", "relation": "面向", "object_id": "actor-1",
                                  "evidence_frame_ids": ["frame-2"]}],
        "uncertainties": ["台词无法确定是谁按下按钮。"], "evidence_frame_ids": ["frame-0", "frame-2"]}


@pytest.mark.asyncio
async def test_compact_contextual_observation_accepts_all_characters_changes_and_selective_original_evidence(tmp_path):
    data = compact_vision_observation()
    providers = manager(tmp_path, lambda request: reply(data))
    result = await providers.analyze("scene", images(tmp_path, 3), 0, 3000,
                                     subtitle_context={"cues": cues()}, story_context={})
    assert len(result.data["entities"]) == 10
    assert result.data["events"] == [{"target_ids": [], **event} for event in data["events"]]
    assert result.data["spatial_observations"] == data["spatial_observations"]
    assert result.data["uncertainties"] == data["uncertainties"]
    assert result.data["evidence_frame_ids"] == ["frame-0", "frame-1", "frame-2"]
    assert data["evidence_frame_ids"] == ["frame-0", "frame-2"]


@pytest.mark.asyncio
async def test_compact_contextual_observation_retains_searchable_grouped_static_props(tmp_path):
    data = compact_vision_observation()
    data["entities"].append({"id": "static-details", "kind": "object", "is_character": False,
        "description": "窗边排列着两盆绿植和一本展开的日历", "evidence_frame_ids": ["frame-1"]})
    providers = manager(tmp_path, lambda request: reply(data))
    result = await providers.analyze("scene", images(tmp_path, 3), 0, 3000, story_context={})
    group = result.data["entities"][-1]
    assert group["description"] == data["entities"][-1]["description"]
    assert group["evidence_frame_ids"] == ["frame-1"]
    assert not group["is_character"] and group["character_id"] is None


@pytest.mark.parametrize("part", ["entities", "events", "spatial_observations"])
@pytest.mark.parametrize("provider_type", ["openai_compatible", "codex_cli"])
@pytest.mark.asyncio
async def test_sparse_item_evidence_is_preserved_in_durable_provider_observation(
        tmp_path, monkeypatch, part, provider_type):
    data = {"summary": "画面中有一盏灯", "entities": [], "events": [], "spatial_observations": [],
            "uncertainties": [], "evidence_frame_ids": ["frame-0"]}
    if part == "entities":
        data[part] = [{"id": "lamp", "kind": "object", "description": "灯",
                       "evidence_frame_ids": ["frame-1"]}]
    elif part == "events":
        data[part] = [{"action": "灯亮起", "start_ms": 1000, "end_ms": 1000,
                       "evidence_frame_ids": ["frame-1"]}]
    else:
        data["entities"] = [{"id": "lamp", "kind": "object", "description": "灯",
                              "evidence_frame_ids": ["frame-0"]}]
        data[part] = [{"subject_id": "lamp", "relation": "位于左侧",
                       "evidence_frame_ids": ["frame-1"]}]
    original = deepcopy(data)
    calls = []

    async def complete(*args, **kwargs):
        calls.append(args)
        return codex_cli.CodexCLIResult(json.dumps(data),
                                        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    monkeypatch.setattr(codex_cli, "complete", complete)
    providers = manager(tmp_path, lambda request: reply(data), provider_type=provider_type)
    result = await providers.analyze("scene", images(tmp_path, 3), 0, 3000, story_context={})
    assert result.data["evidence_frame_ids"] == ["frame-0", "frame-1"]
    assert result.data[part][0]["evidence_frame_ids"] == ["frame-1"]
    assert result.request_count == 1 and len(calls) == int(provider_type == "codex_cli")
    assert result.usage["total_tokens"] == (15 if provider_type == "codex_cli" else 30)
    assert data == original
    durable = Observation.model_validate({**result.data, "id": "observation", "asset_id": "asset",
        "run_id": "run", "window_id": "window", "shot_id": "shot", "start_ms": 0, "end_ms": 3000})
    assert durable.evidence_frame_ids == ["frame-0", "frame-1"]


@pytest.mark.parametrize("part", ["entities", "events", "spatial_observations"])
@pytest.mark.parametrize("foreign_id", ["cue-0", "past-frame", "reference:known-character"])
@pytest.mark.asyncio
async def test_compact_contextual_items_reject_dialogue_memory_and_reference_image_evidence(tmp_path, part, foreign_id):
    data = compact_vision_observation()
    data[part][0]["evidence_frame_ids"] = [foreign_id]
    providers = manager(tmp_path, lambda request: reply(data))
    with pytest.raises(ProviderError) as caught:
        await providers.analyze("scene", images(tmp_path, 3), 0, 3000,
            subtitle_context={"cues": cues()}, story_context={"prior_frame_id": "past-frame"})
    assert caught.value.request_count == 3


@pytest.mark.asyncio
async def test_preparation_capability_works_for_codex_and_connection_test(tmp_path, monkeypatch):
    captured = []

    async def complete(instruction, content, model, timeout_s, **kwargs):
        captured.append((instruction, deepcopy(content)))
        assert kwargs["output_schema"]["$defs"]["Entity"]["properties"]["kind"]["enum"] == [
            "person", "object", "animal", "character", "anthropomorphic_character", "other"]
        mapping = json.loads(next(part["text"].split(": ", 1)[1] for part in content
                                  if part.get("text", "").startswith("Original frame map:")))
        data = {"chunks": [chunk([item["frame_id"] for item in mapping if not item["reference_only"]])],
                "unresolved": []}
        return codex_cli.CodexCLIResult(json.dumps(data), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    monkeypatch.setattr(codex_cli, "complete", complete)
    providers = manager(tmp_path, lambda request: pytest.fail("Codex must not use HTTP"), provider_type="codex_cli")
    result = await providers.test("scene", "preparation")
    assert result.request_count == 1 and result.cost is None
    assert result.data["chunks"][0]["frame_ids"] == ["test-0", "test-1"]
    assert VISUAL_PREPARATION_PROMPT_VERSION in captured[0][0]
    assert any(part["type"] == "image_url" for part in captured[0][1])


def scene_candidate():
    return {"id": "scene-1", "record_id": "scene-1", "kind": "scene", "chunk_id": "scene-1",
            "summary": "可能要回家", "evidence_frame_ids": ["frame-0", "frame-1"],
            "subtitle_ids": ["cue-0", "cue-1"],
            "visual_facts": [{"text": "角色在门前", "frame_ids": ["frame-0"]}],
            "dialogue_facts": [{"text": "台词说我们回家", "cue_ids": ["cue-0"]}],
            "inferences": [{"text": "可能刚刚离家", "frame_ids": ["frame-1"], "cue_ids": ["cue-1"]}]}


@pytest.mark.parametrize("evidence", [["frame-0"], ["cue-0"], ["scene-1", "frame-0", "cue-0"]])
@pytest.mark.asyncio
async def test_scene_full_match_accepts_only_fact_bound_frame_or_subtitle_sources(tmp_path, evidence):
    data = {"candidates": [{"id": "scene-1", "score": .8, "match_type": "full", "evidence_ids": evidence}]}
    providers = manager(tmp_path, lambda request: reply(data), capabilities=["decision"])
    result = await providers.decide("scene", "找提议回家的片段", [scene_candidate()])
    assert result.data["candidates"][0]["evidence_ids"] == evidence


@pytest.mark.parametrize("evidence", [["scene-1"], ["frame-1"], ["cue-1"], ["frame-0", "cue-1"], ["other-cue"]])
@pytest.mark.asyncio
async def test_scene_full_match_rejects_summary_only_and_inference_only_evidence(tmp_path, evidence):
    data = {"candidates": [{"id": "scene-1", "score": .8, "match_type": "full", "evidence_ids": evidence}]}
    providers = manager(tmp_path, lambda request: reply(data), capabilities=["decision"])
    with pytest.raises(ProviderError) as caught:
        await providers.decide("scene", "找刚刚离家的片段", [scene_candidate()])
    assert caught.value.request_count == 3


@pytest.mark.asyncio
async def test_scene_partial_can_use_summary_and_answer_keeps_owner_citation(tmp_path):
    requests = []
    decisions = {"candidates": [{"id": "scene-1", "score": .5, "match_type": "partial",
                                "evidence_ids": ["scene-1"]}]}
    answer = {"text": "台词提议回家；是否已经离开无法从画面确认。[[scene-1]]", "citation_ids": ["scene-1"]}
    responses = iter([decisions, answer])
    providers = manager(tmp_path, lambda request: requests.append(request) or reply(next(responses)),
                        capabilities=["decision", "answer"])
    result = await providers.decide("scene", "离家", [scene_candidate()])
    assert result.data["candidates"][0]["match_type"] == "partial"
    result = await providers.compose("scene", "离家", [scene_candidate()])
    assert result.data == answer
    prompt = json.loads(requests[-1].content)["messages"][0]["content"]
    assert "not visual proof" in prompt and "owning [[candidate-id]]" in prompt


@pytest.mark.asyncio
async def test_reconciliation_is_text_only_and_uses_current_fact_sources(tmp_path):
    requests = []
    data = {"summary": "角色站在门前，台词提议回家。", "inferences": [
        {"text": "可能准备离开", "frame_ids": ["frame-0"], "cue_ids": ["cue-0"]}], "unresolved": []}
    providers = manager(tmp_path, lambda request: requests.append(request) or reply(data))
    scene = scene_candidate()
    scene["path"] = "/private/source/should-not-be-sent"
    result = await providers.reconcile_scene("scene", scene, {"summary": "此前正在商量去哪儿"})
    assert result.data == data and result.request_count == 1
    payload = json.loads(requests[0].content)
    sent = json.loads(payload["messages"][1]["content"])
    assert set(sent["current_scene"]) == {"chunk_id", "visual_facts", "dialogue_facts"}
    assert "image_url" not in requests[0].content.decode()
    assert "should-not-be-sent" not in requests[0].content.decode()
    assert RECONCILIATION_PROMPT_VERSION in payload["messages"][0]["content"]


@pytest.mark.parametrize("reference", ["old-frame", "frame-1"])
@pytest.mark.asyncio
async def test_reconciliation_rejects_memory_and_nonfactual_current_references(tmp_path, reference):
    data = {"summary": "推测", "inferences": [{"text": "发生过离家", "frame_ids": [reference], "cue_ids": []}],
            "unresolved": []}
    providers = manager(tmp_path, lambda request: reply(data))
    with pytest.raises(ProviderError) as caught:
        await providers.reconcile_scene("scene", scene_candidate(), {"frame_ids": ["old-frame"]})
    assert caught.value.request_count == 3
