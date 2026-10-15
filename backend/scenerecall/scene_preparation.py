"""Bounded scene preparation inputs, contact sheets and evidence validation.

Contact sheets are disposable previews. Every durable reference continues to use
an original frame ID; previous-window pictures are context, never new evidence.
"""
from __future__ import annotations

import json
import unicodedata
from typing import Literal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator

PREPARATION_PROMPT_VERSION = "scene-preparation-v1"
SUBTITLE_PREPARATION_PROMPT_VERSION = "scene-subtitle-preparation-v2"
VISUAL_PREPARATION_PROMPT_VERSION = "scene-visual-preparation-v5"
CONTEXTUAL_VISION_PROMPT_VERSION = "scene-contextual-vision-v7"
RECONCILIATION_PROMPT_VERSION = "scene-reconciliation-v3"
MAX_CORE_FRAMES = 56
MAX_CONTEXT_FRAMES = 8
MAX_SUBTITLE_CUES = 256
MAX_SCENE_CONTEXT_BYTES = 64_000
MAX_PREPARATION_OUTPUT_TOKENS = 8192
TILE_WIDTH, TILE_HEIGHT, LABEL_HEIGHT = 320, 180, 20


def normalize_scene_key(value: str) -> str:
    """A bounded visual-similarity hint, never an assertion of scene identity."""
    if not isinstance(value, str) or len(value) > 120 or any(unicodedata.category(char).startswith("C") for char in value):
        raise ValueError("相似场景标签必须是 120 字以内的单行文字")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    if len(normalized) > 120:
        raise ValueError("相似场景标签超过长度上限")
    return normalized


class SceneValue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("summary", "text", check_fields=False)
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("片段文字不能为空白")
        return value


class SubtitleSegment(SceneValue):
    content_kind: Literal["dialogue", "credits"] = "dialogue"
    cue_ids: list[str] = Field(min_length=1, max_length=MAX_SUBTITLE_CUES)
    summary: str = Field(min_length=1, max_length=2000)


class SubtitlePreparation(SceneValue):
    segments: list[SubtitleSegment] = Field(min_length=1, max_length=MAX_SUBTITLE_CUES)
    unresolved: list[str] = Field(max_length=64)


class VisualFact(SceneValue):
    text: str = Field(min_length=1, max_length=1000)
    frame_ids: list[str] = Field(min_length=1, max_length=MAX_CORE_FRAMES)


class DialogueFact(SceneValue):
    text: str = Field(min_length=1, max_length=1000)
    cue_ids: list[str] = Field(min_length=1, max_length=MAX_SUBTITLE_CUES)


class SceneInference(SceneValue):
    text: str = Field(min_length=1, max_length=1000)
    frame_ids: list[str] = Field(max_length=MAX_CORE_FRAMES)
    cue_ids: list[str] = Field(max_length=MAX_SUBTITLE_CUES)


class PreparedChunk(SceneValue):
    content_kind: Literal["narrative", "credits"] = "narrative"
    entities: list[dict] = Field(default_factory=list, max_length=128)
    events: list[dict] = Field(default_factory=list, max_length=128)
    frame_ids: list[str] = Field(min_length=1, max_length=MAX_CORE_FRAMES)
    summary: str = Field(min_length=1, max_length=2000)
    scene_key: str = Field(default="", max_length=120)
    visual_facts: list[VisualFact] = Field(max_length=64)
    dialogue_facts: list[DialogueFact] = Field(max_length=64)
    inferences: list[SceneInference] = Field(max_length=64)
    representative_frame_ids: list[str] = Field(min_length=1, max_length=8)
    deep_frame_ids: list[str] = Field(default_factory=list, max_length=MAX_CORE_FRAMES)
    needs_deep: bool
    needs_identity_review: bool = False
    needs_action_deep: bool | None = None
    reason: str = Field(max_length=2000)
    continues_previous: bool

    @field_validator("scene_key")
    @classmethod
    def visual_similarity_key(cls, value: str) -> str:
        return normalize_scene_key(value)


class VisualPreparation(SceneValue):
    chunks: list[PreparedChunk] = Field(min_length=1, max_length=MAX_CORE_FRAMES)
    unresolved: list[str] = Field(max_length=64)


class SceneReconciliation(SceneValue):
    summary: str = Field(min_length=1, max_length=2000)
    inferences: list[SceneInference] = Field(max_length=64)
    unresolved: list[str] = Field(max_length=64)


def bounded_json(value, *, limit: int = MAX_SCENE_CONTEXT_BYTES) -> str:
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(text.encode("utf-8")) > limit:
        raise ValueError("片段上下文超过单次请求上限，请缩小准备批次")
    return text


def checked_cues(cues: list[dict], *, allow_empty: bool = False) -> list[dict]:
    if not isinstance(cues, list) or len(cues) > MAX_SUBTITLE_CUES or (not cues and not allow_empty):
        raise ValueError("字幕准备批次必须包含 1–256 条字幕")
    result, seen, previous = [], set(), -1
    for cue in cues:
        if not isinstance(cue, dict):
            raise ValueError("字幕上下文格式无效")
        cue_id, start, end, text = (cue.get(key) for key in ("id", "start_ms", "end_ms", "text"))
        if (not isinstance(cue_id, str) or not cue_id or cue_id in seen or len(cue_id) > 200
                or type(start) is not int or type(end) is not int or not 0 <= start < end
                or start < previous or not isinstance(text, str)):
            raise ValueError("字幕必须按时间排列并有唯一 ID、有效时间和文字")
        result.append({"id": cue_id, "start_ms": start, "end_ms": end, "text": text})
        previous = start
        seen.add(cue_id)
    bounded_json(result)
    return result


def subtitle_context_data(context) -> tuple[object, set[str]]:
    if context is None:
        return None, set()
    if isinstance(context, list):
        cues = checked_cues(context, allow_empty=True)
        return cues, {cue["id"] for cue in cues}
    if not isinstance(context, dict):
        raise ValueError("字幕上下文应为字幕列表或含 cues 的对象")
    cues = checked_cues(context.get("cues", []), allow_empty=True)
    data = {**context, "cues": cues}
    bounded_json(data)
    return data, {cue["id"] for cue in cues}


def checked_frames(frames: list[dict], context_frames: list[dict]) -> None:
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_CORE_FRAMES:
        raise ValueError("视觉准备批次必须包含 1–56 张当前关键帧")
    if not isinstance(context_frames, list) or len(context_frames) > MAX_CONTEXT_FRAMES:
        raise ValueError("边界上下文最多包含 8 张关键帧")
    seen, previous = set(), -1
    for frame in [*context_frames, *frames]:
        if not isinstance(frame, dict):
            raise ValueError("关键帧上下文格式无效")
        frame_id, at = frame.get("id"), frame.get("at_ms")
        if (not isinstance(frame_id, str) or not frame_id or len(frame_id) > 200 or frame_id in seen
                or type(at) is not int or at < previous or at < 0):
            raise ValueError("关键帧必须按时间排列且 ID 唯一；上下文帧须位于当前帧之前")
        seen.add(frame_id)
        previous = at


def contact_sheets(frames: list[dict], context_frames: list[dict], directory: Path) -> tuple[list[dict], list[dict]]:
    """Create at most four 1280×800 sheets and their explicit original-ID map."""
    checked_frames(frames, context_frames)
    directory.mkdir(parents=True, exist_ok=True)
    sheets, mapping = [], []
    ordered = [(frame, True) for frame in context_frames] + [(frame, False) for frame in frames]
    font = ImageFont.load_default(size=13)
    for offset in range(0, len(ordered), 16):
        sheet_index = offset // 16
        sheet = Image.new("RGB", (TILE_WIDTH * 4, (TILE_HEIGHT + LABEL_HEIGHT) * 4), "#181818")
        draw = ImageDraw.Draw(sheet)
        for position, (frame, reference_only) in enumerate(ordered[offset:offset + 16]):
            x = position % 4 * TILE_WIDTH
            y = position // 4 * (TILE_HEIGHT + LABEL_HEIGHT)
            path = Path(frame["path"])
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("关键帧超过 16 MB，请减小图像尺寸")
            with Image.open(path) as original:
                picture = ImageOps.contain(original.convert("RGB"), (TILE_WIDTH, TILE_HEIGHT))
                sheet.paste(picture, (x + (TILE_WIDTH - picture.width) // 2, y + (TILE_HEIGHT - picture.height) // 2))
            label = offset + position + 1
            kind = "CONTEXT" if reference_only else "CORE"
            draw.text((x + 4, y + TILE_HEIGHT + 2), f"{label:02d}  {kind}  {frame['at_ms']} ms", font=font, fill="white")
            mapping.append({"label": label, "sheet": sheet_index + 1, "tile": position + 1,
                            "frame_id": frame["id"], "at_ms": frame["at_ms"], "reference_only": reference_only})
        output = directory / f"sheet-{sheet_index + 1}.jpg"
        sheet.save(output, format="JPEG", quality=88)
        sheets.append({"id": f"contact-sheet-{sheet_index + 1}", "path": str(output), "at_ms": 0})
    return sheets, mapping


def validate_subtitle_preparation(raw, cues: list[dict]) -> dict:
    result = SubtitlePreparation.model_validate(raw)
    if [cue_id for segment in result.segments for cue_id in segment.cue_ids] != [cue["id"] for cue in cues]:
        raise ValueError("字幕分段必须按顺序恰好覆盖每条字幕一次")
    value = result.model_dump(exclude_unset=True)
    for segment in value["segments"]:
        if segment.get("content_kind") == "credits":
            segment["summary"] = "片头／片尾演职员信息"
    return value


def _references(values: list[str], allowed: set[str]) -> None:
    if len(values) != len(set(values)) or not set(values).issubset(allowed):
        raise ValueError("片段事实引用了未知、重复或其他片段的证据")


def reconciliation_scene_data(scene: dict) -> tuple[dict, set[str], set[str]]:
    if not isinstance(scene, dict):
        raise ValueError("场景理解输入格式无效")
    data = {key: scene[key] for key in ("chunk_id", "start_ms", "end_ms") if key in scene}
    allowed = {"frame_ids": set(), "cue_ids": set()}
    for field, reference_key in (("visual_facts", "frame_ids"), ("dialogue_facts", "cue_ids")):
        facts = scene.get(field, [])
        if not isinstance(facts, list):
            raise ValueError("场景事实输入格式无效")
        data[field] = []
        for fact in facts:
            references = fact.get(reference_key) if isinstance(fact, dict) else None
            if (not isinstance(fact, dict) or not isinstance(fact.get("text"), str) or not fact["text"].strip()
                    or not isinstance(references, list) or not references
                    or any(not isinstance(value, str) or not value for value in references)
                    or len(references) != len(set(references))):
                raise ValueError("场景事实必须有文字与当前来源引用")
            data[field].append({"text": fact["text"], reference_key: list(references)})
            allowed[reference_key].update(references)
    bounded_json(data)
    return data, allowed["frame_ids"], allowed["cue_ids"]


def validate_scene_reconciliation(raw, frame_ids: set[str], cue_ids: set[str]) -> dict:
    result = SceneReconciliation.model_validate(raw)
    for inference in result.inferences:
        _references(inference.frame_ids, frame_ids)
        _references(inference.cue_ids, cue_ids)
        if not inference.frame_ids and not inference.cue_ids:
            raise ValueError("场景推断必须引用当前事实的证据")
    return result.model_dump()


def validate_visual_preparation(raw, frames: list[dict], cue_ids: set[str], *, has_context: bool,
                                cues: list[dict] | None = None, start_ms: int | None = None,
                                end_ms: int | None = None) -> dict:
    result = VisualPreparation.model_validate(raw)
    if [frame_id for chunk in result.chunks for frame_id in chunk.frame_ids] != [frame["id"] for frame in frames]:
        raise ValueError("视觉分块必须按顺序恰好覆盖每张当前关键帧一次")
    times = {frame["id"]: frame["at_ms"] for frame in frames}
    cue_lookup = {cue["id"]: cue for cue in cues or []}
    start_ms = frames[0]["at_ms"] if start_ms is None else start_ms
    end_ms = max([frames[-1]["at_ms"] + 1, *(cue["end_ms"] for cue in cues or [])]) if end_ms is None else end_ms
    if (type(start_ms) is not int or type(end_ms) is not int
            or not 0 <= start_ms <= frames[0]["at_ms"] <= frames[-1]["at_ms"] < end_ms):
        raise ValueError("视觉分块的覆盖时间与当前帧不匹配")
    for index, chunk in enumerate(result.chunks):
        own = set(chunk.frame_ids)
        chunk_start = start_ms if index == 0 else times[chunk.frame_ids[0]]
        chunk_end = (times[result.chunks[index + 1].frame_ids[0]]
                     if index + 1 < len(result.chunks) else end_ms)
        if chunk_start >= chunk_end or any(not chunk_start <= times[fid] < chunk_end for fid in chunk.frame_ids):
            raise ValueError("相同时间的关键帧不能拆成零时长片段")
        own_cues = ({cue_id for cue_id, cue in cue_lookup.items()
                     if cue["start_ms"] < chunk_end and cue["end_ms"] > chunk_start}
                    if cues is not None else cue_ids)
        _references(chunk.representative_frame_ids, own)
        if not {chunk.frame_ids[0], chunk.frame_ids[-1]}.issubset(chunk.representative_frame_ids):
            raise ValueError("代表帧必须保留片段首尾")
        if chunk.deep_frame_ids:
            _references(chunk.deep_frame_ids, own)
            selected = set(chunk.deep_frame_ids)
            if not set(chunk.representative_frame_ids).issubset(selected):
                raise ValueError("深入理解帧必须包含全部代表帧")
            if chunk.deep_frame_ids != [frame_id for frame_id in chunk.frame_ids if frame_id in selected]:
                raise ValueError("深入理解帧必须按原始关键帧时间顺序排列")
        if chunk.continues_previous and (index or not has_context):
            raise ValueError("只有有边界上下文的首个片段可以延续前段")
        identities = [entity.get("id") for entity in chunk.entities]
        if any(not isinstance(identity, str) or not identity for identity in identities) or len(set(identities)) != len(identities):
            raise ValueError("片段角色实体 ID 缺失或重复")
        for entity in chunk.entities:
            evidence = entity.get("evidence_frame_ids")
            if not isinstance(evidence, list) or not evidence or any(not isinstance(value, str) for value in evidence):
                raise ValueError("片段角色实体必须引用当前画面")
            _references(evidence, own)
        from .providers import Event
        for raw_event in chunk.events:
            event = Event.model_validate(raw_event)
            _references(event.evidence_frame_ids, own)
            if not set(event.actor_ids + event.target_ids).issubset(set(identities)):
                raise ValueError("unknown participant")
            if not chunk_start <= event.start_ms <= event.end_ms <= chunk_end or event.start_ms >= chunk_end:
                raise ValueError("event outside window")
        for fact in chunk.visual_facts:
            _references(fact.frame_ids, own)
        for fact in chunk.dialogue_facts:
            _references(fact.cue_ids, own_cues & cue_ids)
        for inference in chunk.inferences:
            _references(inference.frame_ids, own)
            _references(inference.cue_ids, own_cues & cue_ids)
            if not inference.frame_ids and not inference.cue_ids:
                raise ValueError("推断必须引用至少一项输入证据")
        if (chunk.needs_deep if chunk.needs_action_deep is None else chunk.needs_action_deep) and not chunk.reason.strip():
            raise ValueError("深入理解标记必须说明原因")
    # Older cached/fixture responses may omit optional navigation metadata.
    values = result.model_dump(exclude_unset=True)
    for chunk in values["chunks"]:
        if chunk.get("needs_action_deep") is not None:
            chunk["needs_deep"] = chunk["needs_action_deep"]
        if chunk.get("content_kind") == "credits":
            # Keep all original frames and subtitle references. Credits are
            # episode metadata, not story facts, characters, or deep work.
            chunk.update(summary="片头／片尾演职员信息", visual_facts=[], dialogue_facts=[],
                         inferences=[], entities=[], events=[], needs_deep=False,
                         needs_action_deep=False, needs_identity_review=False, reason="", scene_key="",
                         continues_previous=False)
    return values
