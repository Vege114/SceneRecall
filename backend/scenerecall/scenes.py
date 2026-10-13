"""Immutable, evidence-bound scene revisions and bounded chronological memory."""
from __future__ import annotations

import copy
import json

from .library import SCHEMA, atomic_json, confined_path, digest, read_json, read_jsonl, safe_id


class SceneStore:
    def __init__(self, library):
        self.library = library

    @staticmethod
    def chunk_id(asset_id: str, start_ms: int, end_ms: int) -> str:
        return "scene_" + digest([asset_id, start_ms, end_ms])

    def subtitle_digest(self, asset_id: str) -> str:
        asset = self.library.get_asset(asset_id, include_deleted=True)
        return digest({"mode": asset.get("subtitle_mode"), "stream": asset.get("subtitle_stream_index"),
                       "offset": asset.get("subtitle_offset_ms", 0),
                       "cues": [{key: cue.get(key) for key in ("id", "start_ms", "end_ms", "text", "language")}
                                for cue in self.subtitle_records(asset_id)]})

    def subtitle_records(self, asset_id: str) -> list[dict]:
        """Current timed cues with user text corrections, leaving source files intact."""
        corrections = {row["record_id"]: row["summary"] for row in self._annotations(asset_id)
                       if row.get("summary") is not None}
        return [{**cue, "text": corrections.get(cue["id"], cue.get("text"))}
                for cue in self.library.subtitles(asset_id)]

    def _annotations(self, asset_id: str) -> list[dict]:
        # Library.annotations itself projects scenes: read raw overlays here to
        # avoid a recursion, and keep notes/favorites outside semantic memory.
        return sorted((read_json(path) for path in (self.library.asset_dir(asset_id) / "annotations").glob("*.json")),
                      key=lambda row: row.get("updated_at", ""))

    def _project_semantics(self, asset_id: str, chunk: dict, annotations=None) -> dict:
        row = copy.deepcopy(chunk)
        for annotation in self._annotations(asset_id) if annotations is None else annotations:
            if annotation["record_id"] != row["id"]:
                continue
            if annotation.get("summary") is not None:
                row["summary"] = annotation["summary"]
            if annotation.get("character_name") and annotation.get("base_run_id") == row.get("run_id"):
                for entity in row.get("entities", []):
                    if entity.get("id") == annotation.get("entity_id"):
                        entity.update(name=annotation["character_name"], character_name=annotation["character_name"],
                                      aliases=annotation.get("aliases") or [])
        return row

    def dependencies(self, asset_id: str) -> dict:
        from .characters import CharacterStore, scope_id
        asset = self.library.get_asset(asset_id, include_deleted=True)
        return {"fingerprint": asset["fingerprint"], "subtitle_digest": self.subtitle_digest(asset_id),
                "character_revision": CharacterStore(self.library)._read(scope_id(asset))["revision"],
                "membership_revision": asset.get("membership_revision", 0)}

    def _raw(self, asset_id: str) -> list[dict]:
        result = []
        for identity, relative in self.library.active(asset_id).get("scenes", {}).items():
            safe_id(identity)
            path = confined_path(self.library.asset_dir(asset_id), relative)
            row = read_json(path)
            if (not row or row.get("id") != identity or row.get("asset_id") != asset_id
                    or relative != self._relative(row)):
                raise ValueError("场景活动版本引用无效")
            result.append(row)
        return sorted(result, key=lambda row: (row["start_ms"], row["end_ms"]))

    def list(self, asset_id: str, include_stale: bool = True) -> list[dict]:
        self.library.get_asset(asset_id, include_deleted=True)
        rows = self._raw(asset_id)
        if not rows:
            return []
        annotations = self._annotations(asset_id)
        cache = {(asset_id, "annotations"): annotations}
        result = []
        for row in rows:
            row = self._project_semantics(asset_id, row, annotations)
            reasons = self._stale_reasons(asset_id, row, cache, set())
            value = {**row, "stale": bool(reasons), "context_stale": bool(reasons), "stale_reasons": reasons}
            if include_stale or not reasons:
                result.append(value)
        return result

    def _stale_reasons(self, asset_id, row, cache, visiting):
        identity = (asset_id, row["id"])
        if identity in cache:
            return cache[identity]
        if identity in visiting:
            return ["连续上下文引用存在循环"]
        visiting = visiting | {identity}
        if (asset_id, "dependencies") not in cache:
            cache[asset_id, "dependencies"] = self.dependencies(asset_id)
        current = cache[asset_id, "dependencies"]
        marked = self.library.active(asset_id).get("scene_staleness", {})
        labels = {"fingerprint": "视频来源已变化", "subtitle_digest": "字幕来源或时间轴已变化",
                  "character_revision": "角色人工档案已变化", "membership_revision": "所属项目已变化"}
        reasons = list(marked.get(row["id"], []))
        for key, expected in row.get("dependencies", {}).items():
            if key in current and expected != current[key]:
                reasons.append(labels[key])
        memory = row.get("memory_before", {})
        for dependency in memory.get("provenance", []) if isinstance(memory, dict) else []:
            source_asset, source_id = dependency.get("asset_id"), dependency.get("chunk_id")
            if not source_asset or not source_id:
                continue
            try:
                if (source_asset, "rows") not in cache:
                    cache[source_asset, "rows"] = {item["id"]: item for item in self._raw(source_asset)}
                sources = cache[source_asset, "rows"]
                source = sources.get(source_id)
                if source is not None:
                    if (source_asset, "annotations") not in cache:
                        cache[source_asset, "annotations"] = self._annotations(source_asset)
                    source = self._project_semantics(source_asset, source, cache[source_asset, "annotations"])
                source_info = self.library.get_asset(source_asset)
                target_info = self.library.get_asset(asset_id, include_deleted=True)
                if (source is None or (dependency.get("revision_digest") and dependency["revision_digest"] != self.revision_digest(source))
                        or source_info.get("project_id") != target_info.get("project_id")
                        or self._stale_reasons(source_asset, source, cache, visiting)):
                    reasons.append("前序场景上下文已变化")
            except (ValueError, FileNotFoundError):
                reasons.append("前序场景上下文已不可用")
        cache[identity] = list(dict.fromkeys(reasons))
        return cache[identity]

    def get(self, asset_id: str, chunk_id: str) -> dict:
        return next((row for row in self.list(asset_id) if row["id"] == chunk_id), None) or self._missing()

    @staticmethod
    def _missing():
        raise FileNotFoundError("场景块不存在")

    @staticmethod
    def revision_digest(chunk: dict) -> str:
        # Related IDs are navigation only. The visual-similarity key is retained
        # in bounded memory, so changes to that hint do invalidate its consumers.
        fields = ("id", "asset_id", "start_ms", "end_ms", "summary", "visual_facts", "dialogue_facts", "inferences",
                  "frame_ids", "evidence_frame_ids", "subtitle_ids", "entities", "events", "spatial_observations",
                  "uncertainties", "understanding_level", "dependencies")
        return digest({**{key: chunk.get(key) for key in fields},
                       **({"scene_key": chunk["scene_key"]} if chunk.get("scene_key") else {})})

    @staticmethod
    def _relative(chunk: dict) -> str:
        suffix = ".deep" if chunk.get("understanding_level") == "deep" else ""
        return f"analyses/{safe_id(chunk['run_id'])}/scenes/{safe_id(chunk['id'])}{suffix}.json"

    def validate(self, asset_id: str, chunk: dict, *, historical: bool = False) -> dict:
        asset = self.library.get_asset(asset_id, include_deleted=True)
        if not isinstance(chunk, dict):
            raise ValueError("场景块必须是对象")
        value = copy.deepcopy(chunk)
        start, end = value.get("start_ms"), value.get("end_ms")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= asset["duration_ms"]:
            raise ValueError("场景块时间范围无效")
        identity = self.chunk_id(asset_id, start, end)
        if value.get("id", identity) != identity or value.get("chunk_id", identity) != identity:
            raise ValueError("场景块 ID 与时间范围不一致")
        if value.get("asset_id", asset_id) != asset_id:
            raise ValueError("场景块不属于当前视频")
        from .scene_preparation import normalize_scene_key
        value["scene_key"] = normalize_scene_key(value.get("scene_key", ""))
        related = value.setdefault("related_chunk_ids", [])
        if (not isinstance(related, list) or any(not isinstance(item, str) for item in related)
                or len(set(related)) != len(related) or identity in related):
            raise ValueError("相似场景关联必须是唯一且不包含自身的 ID 列表")
        for item in related:
            safe_id(item)
        safe_id(value.get("run_id"))
        value.update(id=identity, chunk_id=identity, asset_id=asset_id, schema_version=SCHEMA)
        for flag in ("stale", "context_stale", "stale_reasons"):
            value.pop(flag, None)
        if not isinstance(value.get("summary"), str) or not value["summary"].strip():
            raise ValueError("场景块摘要不能为空")
        if value.get("understanding_level", "initial") not in {"initial", "deep"}:
            raise ValueError("场景理解层级无效")
        value.setdefault("understanding_level", "initial")
        registry = read_json(self.library.asset_dir(asset_id) / "frames" / "registry.json", {})
        for field in ("frame_ids", "evidence_frame_ids", "subtitle_ids", "representative_frame_ids", "deep_frame_ids", "shot_ids"):
            ids = value.setdefault(field, [])
            if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
                raise ValueError("场景证据引用列表无效")
            for item in ids:
                safe_id(item)
        frames = set(value["frame_ids"])
        self.library.frame_paths(asset_id, value["frame_ids"], registry=registry)
        for frame_id in frames:
            if not start <= registry[frame_id]["at_ms"] < end:
                raise ValueError("场景帧超出当前块时间范围，历史上下文不能作为当前证据")
        if not set(value["evidence_frame_ids"] + value["representative_frame_ids"]).issubset(frames):
            raise ValueError("场景引用了未提供的画面")
        deep_frames = set(value["deep_frame_ids"])
        if deep_frames and (not deep_frames.issubset(frames)
                or not set(value["representative_frame_ids"]).issubset(deep_frames)
                or value["deep_frame_ids"] != [frame_id for frame_id in value["frame_ids"] if frame_id in deep_frames]):
            raise ValueError("深入理解帧必须按本块原帧顺序排列并包含全部代表帧")
        cues = {cue["id"]: cue for cue in self.library.subtitles(asset_id)}
        if historical:
            for path in (self.library.asset_dir(asset_id) / "analyses").glob("*/subtitles.jsonl"):
                cues.update({cue["id"]: cue for cue in read_jsonl(path)})
            cues.update({cue["id"]: cue for cue in read_jsonl(self.library.asset_dir(asset_id) / "subtitles" / "external.jsonl")})
        for cue_id in value["subtitle_ids"]:
            cue = cues.get(cue_id)
            if cue is None or cue["start_ms"] >= end or cue["end_ms"] <= start:
                raise ValueError("场景字幕引用缺失或超出当前块时间范围")
        for field, allowed_fields in (("visual_facts", ("frame_ids",)), ("dialogue_facts", ("cue_ids",)),
                                      ("inferences", ("frame_ids", "cue_ids"))):
            facts = value.setdefault(field, [])
            if not isinstance(facts, list):
                raise ValueError("场景事实列表无效")
            for fact in facts:
                if not isinstance(fact, dict) or not isinstance(fact.get("text"), str) or not fact["text"].strip():
                    raise ValueError("场景事实文字无效")
                refs = []
                for key in ("frame_ids", "cue_ids"):
                    ids = fact.get(key, [])
                    allowed = set(value["evidence_frame_ids"]) if key == "frame_ids" else set(value["subtitle_ids"])
                    if (not isinstance(ids, list) or any(not isinstance(item, str) for item in ids)
                            or not set(ids).issubset(allowed) or (key not in allowed_fields and ids)):
                        raise ValueError("场景事实引用了错误类型或不存在的证据")
                    refs.extend(ids)
                if not refs:
                    raise ValueError("场景事实必须引用当前块证据")
        for key in ("dependencies", "provenance"):
            if not isinstance(value.setdefault(key, {}), dict):
                raise ValueError("场景来源或依赖信息无效")
        if "deep_analyzed_frame_ids" in value["provenance"]:
            analyzed = value["provenance"]["deep_analyzed_frame_ids"]
            if (not isinstance(analyzed, list) or not analyzed or any(not isinstance(item, str) for item in analyzed)
                    or len(set(analyzed)) != len(analyzed) or not set(analyzed).issubset(frames)
                    or analyzed != [frame_id for frame_id in value["frame_ids"] if frame_id in set(analyzed)]):
                raise ValueError("深入理解实际输入记录必须引用按原帧顺序排列的当前场景帧")
        for key in ("memory_before", "memory_after"):
            memory = value.setdefault(key, {})
            if not isinstance(memory, (dict, list, str)) or len(json.dumps(memory, ensure_ascii=False).encode()) > 128000:
                raise ValueError("场景连续上下文无效或超过上限")
            if isinstance(memory, dict) and "provenance" in memory:
                references = memory["provenance"]
                if not isinstance(references, list):
                    raise ValueError("连续上下文来源列表无效")
                for reference in references:
                    if not isinstance(reference, dict):
                        raise ValueError("连续上下文来源无效")
                    safe_id(reference.get("asset_id"))
                    safe_id(reference.get("chunk_id"))
        # Existing event/entity validators retain their evidence and time invariants.
        if any(value.get(key) for key in ("entities", "events", "spatial_observations")):
            from .models import Observation
            Observation.model_validate({**value, "window_id": identity, "shot_id": identity})
        return value

    def _write(self, asset_id: str, run_id: str, chunk: dict) -> tuple[dict, str]:
        value = self.validate(asset_id, {**chunk, "run_id": run_id})
        relative = self._relative(value)
        path = self.library.asset_dir(asset_id) / relative
        previous = read_json(path)
        if previous is not None and previous != value:
            raise ValueError("已提交场景版本不可覆盖，请创建新的分析运行")
        if previous is None:
            atomic_json(path, value)
        return value, relative

    def publish(self, asset_id: str, run_id: str, chunks: list[dict], start_ms: int, end_ms: int) -> list[dict]:
        with self.library.lock:
            asset = self.library.get_asset(asset_id)
            if not chunks or not 0 <= start_ms < end_ms <= asset["duration_ms"]:
                raise ValueError("场景发布范围无效或没有完整结果")
            ordered = sorted(chunks, key=lambda row: row["start_ms"])
            boundary = start_ms
            for row in ordered:
                if row["start_ms"] != boundary:
                    raise ValueError("场景发布必须连续覆盖完整范围")
                boundary = row["end_ms"]
                self.validate(asset_id, {**row, "run_id": run_id})
            if boundary != end_ms:
                raise ValueError("场景发布没有覆盖完整范围")
            old = self._raw(asset_id)
            if any(row["start_ms"] < end_ms and row["end_ms"] > start_ms
                   and not start_ms <= row["start_ms"] < row["end_ms"] <= end_ms for row in old):
                raise ValueError("重新分块范围须覆盖相交旧块的完整范围")
            active = self.library.active(asset_id)
            pointers = active.setdefault("scenes", {})
            stale = active.setdefault("scene_staleness", {})
            for row in old:
                if row["start_ms"] < end_ms and row["end_ms"] > start_ms:
                    pointers.pop(row["id"], None)
                    stale.pop(row["id"], None)
            values = []
            defaults = self.dependencies(asset_id)
            for row in ordered:
                row = {**row, "dependencies": {**defaults, **row.get("dependencies", {})}}
                value, relative = self._write(asset_id, run_id, row)
                previous = next((item for item in old if item["id"] == value["id"]), None)
                if (previous and previous["run_id"] == run_id and previous.get("understanding_level") == "deep"
                        and value["understanding_level"] == "initial"):
                    value, relative = previous, self._relative(previous)
                pointers[value["id"]] = relative
                stale.pop(value["id"], None)
                values.append(value)
            atomic_json(self.library.asset_dir(asset_id) / "active-analysis.json", active)
            return values

    def upgrade(self, asset_id: str, run_id: str, chunk: dict, *, invalidate_dependents: bool = True) -> dict:
        with self.library.lock:
            current = self.get(asset_id, chunk.get("id", chunk.get("chunk_id")))
            if (current["start_ms"], current["end_ms"]) != (chunk["start_ms"], chunk["end_ms"]):
                raise ValueError("深入理解不能改变已有场景块边界")
            metadata = {"scene_key": current.get("scene_key", ""),
                        "related_chunk_ids": current.get("related_chunk_ids", []),
                        "deep_frame_ids": current.get("deep_frame_ids", [])}
            value = self.publish(asset_id, run_id, [{**metadata, **chunk, "understanding_level": "deep"}],
                                 current["start_ms"], current["end_ms"])[0]
            if invalidate_dependents and self.revision_digest(value) != self.revision_digest(current):
                self.mark_dependents_stale(asset_id, current["start_ms"], "前序场景已深入理解，连续上下文待刷新")
            return value

    @staticmethod
    def _episode(asset: dict):
        if type(asset.get("season")) is not int or type(asset.get("episode")) is not int:
            return None
        return asset["season"], asset["episode"]

    def prior_memory(self, asset_id: str, start_ms: int) -> dict:
        asset = self.library.get_asset(asset_id)
        episode = self._episode(asset)
        sources = []
        if episode is not None and asset.get("project_id"):
            sources = [item for item in self.library.list_assets() if item["id"] != asset_id
                       and item.get("project_id") == asset["project_id"] and self._episode(item) is not None
                       and self._episode(item) < episode]
            sources.sort(key=lambda item: (*self._episode(item), item["id"]))
        sources.append(asset)
        summaries, provenance, gaps = [], [], []
        if episode is not None:
            numbered = {}
            for source in sources:
                position = self._episode(source)
                if position is not None:
                    numbered.setdefault(position[0], set()).add(position[1])
            for season, present in sorted(numbered.items()):
                stop = episode[1] if season == episode[0] else max(present) + 1
                missing_start = None
                for number in range(1, stop + 1):
                    missing = number < stop and number not in present
                    if missing and missing_start is None:
                        missing_start = number
                    elif not missing and missing_start is not None:
                        gaps.append({"kind": "missing_episode", "season": season,
                                     "start_episode": missing_start, "end_episode": number - 1})
                        missing_start = None
        for source in sources:
            same_asset = source["id"] == asset_id
            limit = start_ms if same_asset else source["duration_ms"]
            boundary = 0
            accepted, source_gaps = [], []
            for row in self.list(source["id"]):
                if row["end_ms"] > limit:
                    continue
                if row["start_ms"] > boundary:
                    source_gaps.append({"asset_id": source["id"], "start_ms": boundary, "end_ms": row["start_ms"]})
                boundary = row["end_ms"]
                if row.get("needs_deep"):
                    source_gaps.append({"asset_id": source["id"], "start_ms": row["start_ms"],
                                        "end_ms": row["end_ms"], "reason": "pending_deep_analysis"})
                    continue
                if row["context_stale"]:
                    source_gaps.append({"asset_id": source["id"], "start_ms": row["start_ms"], "end_ms": row["end_ms"], "reason": "stale"})
                    continue
                accepted.append(row)
            if boundary < limit:
                source_gaps.append({"asset_id": source["id"], "start_ms": boundary, "end_ms": limit})
            gaps.extend(source_gaps)
            # Later episodes may use only completed, current prior episodes.
            # Same-episode earlier blocks remain useful before the asset finishes.
            if source_gaps and not same_asset:
                continue
            for row in accepted:
                summaries.append({"chunk_id": row["id"], "asset_id": source["id"],
                                  "start_ms": row["start_ms"], "end_ms": row["end_ms"],
                                  "summary": row["summary"][:2000],
                                  "scene_key": row.get("scene_key", ""),
                                  **{field: [{**fact, "text": fact["text"][:500]} for fact in row.get(field, [])[:3]]
                                     for field in ("visual_facts", "dialogue_facts", "inferences")}})
                provenance.append({"chunk_id": row["id"], "asset_id": source["id"], "run_id": row["run_id"],
                                   "revision_digest": self.revision_digest(row)})
        # A fixed recent-history budget keeps prompt growth independent of episode count.
        omitted = max(0, len(summaries) - 12)
        summaries, provenance = summaries[-12:], provenance[-12:]
        while summaries and len(json.dumps(summaries, ensure_ascii=False).encode()) > 24000:
            summaries.pop(0)
            provenance.pop(0)
            omitted += 1
        value = {"summaries": summaries, "provenance": provenance, "gaps": gaps[-24:], "omitted_count": omitted}
        return {**value, "digest": digest(value)}

    def mark_dependents_stale(self, asset_id: str, start_ms: int, reason: str) -> None:
        asset = self.library.get_asset(asset_id)
        episode = self._episode(asset)
        with self.library.lock:
            for target in self.library.list_assets():
                same = target["id"] == asset_id
                later = (episode is not None and self._episode(target) is not None and asset.get("project_id")
                         and target.get("project_id") == asset["project_id"] and self._episode(target) > episode)
                if not same and not later:
                    continue
                active = self.library.active(target["id"])
                marked = active.setdefault("scene_staleness", {})
                changed = False
                for row in self._raw(target["id"]):
                    if same and row["start_ms"] <= start_ms:
                        continue
                    marked[row["id"]] = list(dict.fromkeys([*marked.get(row["id"], []), reason]))
                    changed = True
                if changed:
                    atomic_json(self.library.asset_dir(target["id"]) / "active-analysis.json", active)

    def validate_all(self, asset_id: str) -> None:
        rows = self._raw(asset_id)
        boundary = 0
        for row in rows:
            if row["start_ms"] < boundary:
                raise ValueError("有效场景块时间范围重叠")
            boundary = row["end_ms"]
        for path in (self.library.asset_dir(asset_id) / "analyses").glob("*/scenes/*.json"):
            row = self.validate(asset_id, read_json(path), historical=True)
            if path != self.library.asset_dir(asset_id) / self._relative(row):
                raise ValueError("历史场景文件与版本身份不一致")
            memory = row.get("memory_before", {})
            for reference in memory.get("provenance", []) if isinstance(memory, dict) else []:
                source_asset = reference["asset_id"]
                source_id = reference["chunk_id"]
                self.library.get_asset(source_asset, include_deleted=True)
                directory = self.library.asset_dir(source_asset) / "analyses"
                revisions = [*directory.glob(f"*/scenes/{source_id}.json"), *directory.glob(f"*/scenes/{source_id}.deep.json")]
                if not revisions:
                    raise ValueError("连续上下文引用了不存在的历史场景")
                if source_asset == asset_id and any(read_json(source)["end_ms"] > row["start_ms"] for source in revisions):
                    raise ValueError("连续上下文不能引用当前或未来场景")
