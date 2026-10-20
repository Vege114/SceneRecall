"""Small, independently versioned local visual facts; no episode story inference."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VERSION = "retrieval-local-observation-v1"
TEMPORAL_ACTIONS = ("递交", "递给", "递出", "转身", "进入", "走进", "走入", "离开", "拿起", "放下")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LocalEntity(Strict):
    id: str = Field(min_length=1, max_length=100)
    kind: Literal["person", "character", "animal", "object", "vehicle", "place", "symbol", "other"]
    description: str = Field(min_length=1, max_length=1000)
    observed_traits: list[str] = Field(default_factory=list, max_length=12)
    evidence_frame_ids: list[str] = Field(min_length=1, max_length=12)
    identity_region_id: str | None = None
    identity_support_region_ids: list[str] = Field(default_factory=list, max_length=6)
    candidate_character_ids: list[str] = Field(default_factory=list, max_length=5)
    candidate_concept_ids: list[str] = Field(default_factory=list, max_length=5)


class LocalEvent(Strict):
    action: str = Field(min_length=1, max_length=500)
    actor_ids: list[str] = Field(default_factory=list)
    target_ids: list[str] = Field(default_factory=list)
    object_id: str | None = None
    recipient_id: str | None = None
    polarity: Literal["positive", "negative", "uncertain"] = "positive"
    temporal: bool = False
    start_ms: int = Field(ge=0, strict=True)
    end_ms: int = Field(ge=0, strict=True)
    evidence_frame_ids: list[str] = Field(min_length=1, max_length=12)
    uncertainty: str = Field(default="", max_length=1000)


class LocalRelation(Strict):
    subject_id: str
    relation: str = Field(min_length=1, max_length=200)
    object_id: str | None = None
    evidence_frame_ids: list[str] = Field(min_length=1, max_length=12)


class LocalObservation(Strict):
    unit_id: str
    summary: str = Field(min_length=1, max_length=1500)
    entities: list[LocalEntity] = Field(default_factory=list, max_length=40)
    events: list[LocalEvent] = Field(default_factory=list, max_length=30)
    spatial_observations: list[LocalRelation] = Field(default_factory=list, max_length=30)
    uncertainties: list[str] = Field(default_factory=list, max_length=12)
    evidence_frame_ids: list[str] = Field(min_length=1, max_length=12)


class LocalBatch(Strict):
    observations: list[LocalObservation] = Field(min_length=1, max_length=2)


def output_schema(units, character_ids=(), concept_ids=()):
    def strict(node):
        if isinstance(node, list):
            return [strict(value) for value in node]
        if not isinstance(node, dict):
            return node
        value = {key: strict(item) for key, item in node.items() if key not in {"default", "title"}}
        if isinstance(value.get("properties"), dict):
            value["required"] = list(value["properties"])
            value["additionalProperties"] = False
            if "evidence_frame_ids" in value["properties"]:
                value["properties"]["evidence_frame_ids"]["items"] = {
                    "type": "string", "enum": sorted({f for u in units for f in u["local_frame_ids"]})}
        return value
    schema = strict(LocalBatch.model_json_schema())
    props = schema["$defs"]["LocalEntity"]["properties"]
    for field, ids in (("candidate_character_ids", character_ids), ("candidate_concept_ids", concept_ids)):
        props[field]["items"] = {"type": "string", "enum": sorted(ids)} if ids else {"type": "string"}
        if not ids:
            props[field]["maxItems"] = 0
    schema["$defs"]["LocalObservation"]["properties"]["unit_id"] = {
        "type": "string", "enum": [u["id"] for u in units]}
    return schema


def validate_batch(data, units, frames, character_ids=(), concept_ids=(), region_ids=()):
    value = LocalBatch.model_validate(data).model_dump()
    by_id = {unit["id"]: unit for unit in units}
    observations = value["observations"]
    if len(observations) != len(units) or {o["unit_id"] for o in observations} != set(by_id):
        raise ValueError("局部记录遗漏或重复单元")
    times = {f["id"]: f["at_ms"] for f in frames}
    for obs in observations:
        unit = by_id[obs["unit_id"]]
        allowed = set(unit["local_frame_ids"])
        entities = {e["id"] for e in obs["entities"]}
        if len(entities) != len(obs["entities"]):
            raise ValueError("局部实体 ID 重复")
        for item in [obs, *obs["entities"], *obs["events"], *obs["spatial_observations"]]:
            if not set(item["evidence_frame_ids"]).issubset(allowed):
                raise ValueError("局部事实引用上下文或未提供的帧")
        for entity in obs["entities"]:
            if not set(entity["candidate_character_ids"]).issubset(character_ids):
                raise ValueError("角色候选不在本次作品名单内")
            if not set(entity["candidate_concept_ids"]).issubset(concept_ids):
                raise ValueError("概念候选不在本次相关资料内")
            if entity["identity_region_id"] and entity["identity_region_id"] not in region_ids:
                raise ValueError("身份区域不属于当前观察")
            if not set(entity["identity_support_region_ids"]).issubset(region_ids):
                raise ValueError("身份支持区域不属于当前观察")
        for event in obs["events"]:
            refs = event["actor_ids"] + event["target_ids"] + [event[k] for k in ("object_id", "recipient_id") if event[k]]
            if not set(refs).issubset(entities):
                raise ValueError("事件引用不存在的局部实体")
            if not unit["start_ms"] <= event["start_ms"] <= event["end_ms"] <= unit["end_ms"] or event["start_ms"] >= unit["end_ms"]:
                raise ValueError("事件时间超出当前核心范围")
            ev_times = [times[f] for f in event["evidence_frame_ids"]]
            if ev_times != sorted(ev_times):
                raise ValueError("事件时序证据顺序无效")
            requires_order = event["temporal"] or (event["polarity"] == "positive"
                and any(action in event["action"] for action in TEMPORAL_ACTIONS))
            if requires_order and (len(set(ev_times)) < 2 or event["start_ms"] == event["end_ms"]):
                raise ValueError("时间动作需要至少两帧不同时间的有序证据")
        for relation in obs["spatial_observations"]:
            if relation["subject_id"] not in entities or relation["object_id"] and relation["object_id"] not in entities:
                raise ValueError("空间关系引用不存在的局部实体")
    return value
