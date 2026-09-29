from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetInput(Contract):
    video_path: str
    title: str = ""
    kind: Literal["movie", "animation", "series"] = "movie"
    series: str = ""
    season: int | None = Field(default=None, ge=0)
    episode: int | None = Field(default=None, ge=0)
    version: str = "原始版本"
    subtitle_mode: Literal["embedded", "external"]
    subtitle_path: str | None = None
    subtitle_offset_ms: int = 0

    @model_validator(mode="after")
    def subtitles_required(self):
        if self.subtitle_mode == "external" and not self.subtitle_path:
            raise ValueError("外挂字幕模式必须提供 SRT、VTT 或 ASS 字幕文件")
        return self


class AnalysisInput(Contract):
    asset_id: str
    stages: list[Literal["vision", "subtitle"]] = Field(default_factory=lambda: ["vision"])
    start_ms: int = Field(default=0, ge=0)
    end_ms: int | None = Field(default=None, gt=0)
    window_ms: int = Field(default=6000, ge=1000, le=30000)
    frames_per_window: int = Field(default=4, ge=2, le=12)
    subtitle_fps: float = Field(default=2, ge=0.2, le=10)
    subtitle_crop: list[float] = Field(default_factory=lambda: [0, .65, 1, .35], min_length=4, max_length=4)
    max_requests: int = Field(default=1000, ge=1, le=100000)
    max_cost: float | None = Field(default=None, gt=0)
    force: bool = False

    @model_validator(mode="after")
    def validate_options(self):
        if not self.stages or len(self.stages) != len(set(self.stages)):
            raise ValueError("识别阶段不能为空或重复")
        if self.end_ms is not None and self.end_ms <= self.start_ms:
            raise ValueError("结束时间必须晚于开始时间")
        x, y, w, h = self.subtitle_crop
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError("字幕区域须位于画面内，使用归一化 x/y/宽/高")
        return self


class AnnotationInput(Contract):
    record_id: str
    summary: str | None = Field(default=None, max_length=20000)
    note: str | None = Field(default=None, max_length=20000)
    character_name: str | None = Field(default=None, max_length=200)
    entity_id: str | None = None
    aliases: list[str] | None = None
    favorite: bool | None = None


class SearchInput(Contract):
    query: str = Field(min_length=1, max_length=4000)
    filters: dict = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=50)


class CharacterEditInput(Contract):
    name: str | None = Field(default=None, max_length=200)
    aliases: list[str] | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=10000)
    notes: str | None = Field(default=None, max_length=100000)
    appearance: list[str] | None = Field(default=None, max_length=64)
    expected_revision: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_edit(self):
        editable = self.model_fields_set - {"expected_revision"}
        if not editable or any(getattr(self, field) is None for field in editable):
            raise ValueError("请提供要修改的角色档案字段；清空内容请使用空字符串或空列表")
        if self.aliases and any(not alias.strip() or len(alias) > 200 for alias in self.aliases):
            raise ValueError("角色别名不能为空，且每个别名不能超过 200 字")
        if self.appearance and any(not trait.strip() or len(trait) > 2000 for trait in self.appearance):
            raise ValueError("角色外观特征不能为空，且每条不能超过 2000 字")
        return self


class CharacterMergeInput(Contract):
    target_id: str = Field(min_length=1, max_length=180)
    expected_revision: int | None = Field(default=None, ge=0, strict=True)


class CharacterReanalysisInput(Contract):
    scope: Literal["asset", "series"] = "asset"
    max_requests: int = Field(default=1000, ge=1, le=100000, strict=True)
    max_cost: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    expected_revision: int | None = Field(default=None, ge=0, strict=True)


class Observation(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = "1.0"
    id: str
    asset_id: str
    run_id: str
    window_id: str
    shot_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    summary: str = Field(min_length=1)
    entities: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    spatial_observations: list[dict] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    evidence_frame_ids: list[str] = Field(default_factory=list)
    review_status: str = "unreviewed"

    @model_validator(mode="after")
    def check_evidence(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("识别记录时间无效")
        entity_ids = [x.get("id") for x in self.entities]
        if any(not x for x in entity_ids) or len(set(entity_ids)) != len(entity_ids):
            raise ValueError("实体 ID 缺失或重复")
        for event in self.events:
            if not set(event.get("actor_ids", []) + event.get("target_ids", [])).issubset(entity_ids):
                raise ValueError("事件引用不存在的实体")
            for key in ("subject_id", "object_id", "recipient_id"):
                if event.get(key) and event[key] not in entity_ids:
                    raise ValueError(f"事件引用不存在的实体: {key}")
            a, b = event.get("start_ms", self.start_ms), event.get("end_ms", self.end_ms)
            if not self.start_ms <= a < b <= self.end_ms:
                raise ValueError("事件超出观察时间范围")
            if not set(event.get("evidence_frame_ids", [])).issubset(self.evidence_frame_ids):
                raise ValueError("事件引用不存在的证据帧")
        for position in self.spatial_observations:
            for key in ("subject_id", "object_id"):
                if position.get(key) and position[key] not in entity_ids:
                    raise ValueError("空间描述引用不存在的实体")
            if not set(position.get("evidence_frame_ids", [])).issubset(self.evidence_frame_ids):
                raise ValueError("空间描述引用不存在的证据帧")
            if position.get("entity_id") and position["entity_id"] not in entity_ids:
                raise ValueError("空间描述引用不存在的实体")
            if position.get("frame_id") and position["frame_id"] not in self.evidence_frame_ids:
                raise ValueError("空间描述引用不存在的证据帧")
        return self
