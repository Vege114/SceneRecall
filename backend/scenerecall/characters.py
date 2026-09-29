"""Editable cast registries over immutable, per-shot model observations.

Identity belongs to a series (including every season), or to one work when no
series was supplied. Redirects keep historical evidence readable after merges
and deletions without changing the original model output.
"""
from __future__ import annotations

import copy
import math
import unicodedata
from pathlib import Path

from .library import SCHEMA, atomic_json, digest, now, read_json, safe_id
from .models import Observation


EDITABLE_FIELDS = {"name", "aliases", "description", "notes", "appearance"}
MAX_NOTES_LENGTH = 100000


class StaleCharacterRevision(ValueError):
    """A human changed the cast while an analysis response was in flight."""


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def scope_id(asset: dict) -> str:
    series = normalized(asset.get("series") or "")
    if series:
        return "cast_" + digest([asset.get("kind", "movie"), series])
    return safe_id(asset["work_id"])


def unique(values: list[str], limit: int = 100) -> list[str]:
    result, seen = [], set()
    for value in values:
        clean = value.strip()
        key = normalized(clean)
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
        if len(result) >= limit:
            break
    return result


def checked_id(value) -> str:
    if not isinstance(value, str):
        raise ValueError("角色档案 ID 无效")
    return safe_id(value)


def checked_box(value) -> list[float] | None:
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 4
            or any(type(number) not in {int, float} or not math.isfinite(number) for number in value)):
        raise ValueError("角色图片区域须为归一化 x/y/宽/高")
    x, y, width, height = value
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 or y + height > 1:
        raise ValueError("角色图片区域必须位于证据帧内")
    return list(value)


class CharacterStore:
    def __init__(self, library):
        self.library = library

    def _path(self, scope: str) -> Path:
        return self.library.root / "works" / checked_id(scope) / "characters.json"

    def _read(self, scope: str) -> dict:
        registry = read_json(self._path(scope), {
            "schema_version": SCHEMA, "scope_id": scope, "revision": 0,
            "profiles": {}, "redirects": {},
        })
        self._validate_structure(registry, scope)
        return registry

    @staticmethod
    def _validate_structure(registry: dict, scope: str) -> None:
        if (not isinstance(registry, dict) or registry.get("scope_id") != scope
                or type(registry.get("revision")) is not int or registry["revision"] < 0
                or not isinstance(registry.get("profiles"), dict)
                or not isinstance(registry.get("redirects"), dict)):
            raise ValueError("角色档案格式或所属作品无效")
        for character_id, profile in registry["profiles"].items():
            checked_id(character_id)
            if not isinstance(profile, dict) or profile.get("id") != character_id:
                raise ValueError("角色档案 ID 不一致")
            for field in ("name", "description", "notes", "created_at", "updated_at"):
                if not isinstance(profile.get(field), str):
                    raise ValueError("角色档案文字字段无效")
            if not profile["name"].strip():
                raise ValueError("角色名称不能为空")
            for field in ("aliases", "appearance", "edited_fields"):
                if not isinstance(profile.get(field), list) or any(not isinstance(v, str) for v in profile[field]):
                    raise ValueError("角色档案列表字段无效")
            if not set(profile["edited_fields"]).issubset(EDITABLE_FIELDS) or type(profile.get("user_edited")) is not bool:
                raise ValueError("角色档案人工编辑标记无效")
            representative = profile.get("representative")
            if representative is not None:
                if not isinstance(representative, dict):
                    raise ValueError("角色证据引用无效")
                checked_id(representative.get("asset_id"))
                checked_id(representative.get("frame_id"))
                checked_box(representative.get("box"))
        for source, target in registry["redirects"].items():
            checked_id(source)
            if source in registry["profiles"]:
                raise ValueError("角色档案与合并记录冲突")
            if target is not None:
                checked_id(target)
            CharacterStore._resolve(registry, source, strict=True)

    @staticmethod
    def _resolve(registry: dict, character_id: str | None, *, strict=False) -> str | None:
        if character_id is None:
            return None
        checked_id(character_id)
        current, visited = character_id, set()
        while current in registry["redirects"]:
            if current in visited:
                raise ValueError("角色档案合并记录存在循环")
            visited.add(current)
            current = registry["redirects"][current]
            if current is None:
                return None
        if current not in registry["profiles"]:
            if strict:
                raise ValueError("角色档案引用不存在或不属于当前作品")
            return None
        return current

    @staticmethod
    def _check_revision(registry: dict, expected_revision: int | None) -> None:
        if expected_revision is not None and (
                type(expected_revision) is not int or expected_revision != registry["revision"]):
            raise StaleCharacterRevision("角色档案已更新，请刷新后重试；识别任务需使用最新版档案重新开始")

    def _assets(self, scope: str) -> list[dict]:
        return sorted([asset for asset in self.library.list_assets() if scope_id(asset) == scope],
                      key=lambda a: (a.get("season") or 0, a.get("episode") or 0, a["created_at"], a["id"]))

    def catalog(self, asset_id: str) -> dict:
        with self.library.lock:
            scope = scope_id(self.library.get_asset(asset_id))
            registry = self._read(scope)
            assets = self._assets(scope)
            profiles = copy.deepcopy(registry["profiles"])
            current_representatives = {}
            for profile in profiles.values():
                profile["appearances"] = []
            for asset in assets:
                for record in self.library.observations(asset["id"]):
                    for entity in record.get("entities", []):
                        resolved = self._resolve(registry, entity.get("character_id"))
                        if not resolved:
                            continue
                        frames = entity.get("evidence_frame_ids") or record.get("evidence_frame_ids", [])
                        profiles[resolved]["appearances"].append({
                            "asset_id": asset["id"], "asset_title": asset["title"],
                            "record_id": record["id"], "entity_id": entity["id"],
                            "start_ms": record["start_ms"], "end_ms": record["end_ms"],
                            "frame_ids": frames,
                        })
                        representative = self._representative(asset["id"], entity, record)
                        if representative:
                            current_representatives[resolved] = representative
            for profile in profiles.values():
                profile["appearance_count"] = len(profile["appearances"])
                # Episode/shot order is authoritative; a reanalysis can remove the
                # latest old association, in which case an earlier active image wins.
                if profile["id"] in current_representatives:
                    profile["representative"] = current_representatives[profile["id"]]
            return {"scope_id": scope, "revision": registry["revision"],
                    "profiles": list(profiles.values()), "assets": assets}

    def context(self, asset_id: str) -> dict:
        with self.library.lock:
            catalog = self.catalog(asset_id)
            fields = ("id", "name", "aliases", "description", "notes", "appearance",
                      "user_edited", "edited_fields", "representative", "updated_at", "created_at")
            return {"scope_id": catalog["scope_id"], "revision": catalog["revision"], "profiles": [
                {**{key: copy.deepcopy(profile[key]) for key in fields},
                 "observed_traits": list(profile["appearance"])}
                for profile in catalog["profiles"]
            ]}

    @staticmethod
    def _representative(asset_id: str, entity: dict, record: dict) -> dict | None:
        frames = entity.get("evidence_frame_ids") or record.get("evidence_frame_ids", [])
        portrait_frame = entity.get("portrait_frame_id")
        box = checked_box(entity.get("portrait_box"))
        if portrait_frame is not None:
            checked_id(portrait_frame)
            if portrait_frame not in frames or portrait_frame not in record.get("evidence_frame_ids", []):
                raise ValueError("角色图片必须引用该人物的证据帧")
        elif box is not None:
            raise ValueError("角色图片区域必须指定对应的证据帧")
        frame = portrait_frame or (frames[-1] if frames else None)
        if frame is None:
            return None
        return {"asset_id": asset_id, "frame_id": frame, **({"box": box} if box is not None else {})}

    def portrait_reference(self, asset_id: str, character_id: str) -> dict:
        with self.library.lock:
            scope = scope_id(self.library.get_asset(asset_id))
            resolved = self._resolve(self._read(scope), character_id, strict=True)
            if resolved is None:
                raise FileNotFoundError("角色档案已删除")
            profile = next(profile for profile in self.catalog(asset_id)["profiles"] if profile["id"] == resolved)
            representative = profile.get("representative")
            if not representative:
                raise FileNotFoundError("角色尚无可用图片")
            self.library.frame_path(representative["asset_id"], representative["frame_id"])
            return copy.deepcopy(representative)

    def apply_observation(self, asset_id: str, record: dict, expected_revision: int | None = None) -> dict:
        """Associate validated evidence; callers hold library.lock through saving it."""
        with self.library.lock:
            asset = self.library.get_asset(asset_id)
            scope = scope_id(asset)
            registry = self._read(scope)
            self._check_revision(registry, expected_revision)
            result = Observation.model_validate(copy.deepcopy(record)).model_dump()
            if result["asset_id"] != asset_id or result["end_ms"] > asset["duration_ms"]:
                raise ValueError("角色证据不属于当前视频时间轴")
            for frame_id in result["evidence_frame_ids"]:
                self.library.frame_path(asset_id, frame_id)
            for entity in result["entities"]:
                checked_id(entity["id"])
                if not set(entity.get("evidence_frame_ids", [])).issubset(result["evidence_frame_ids"]):
                    raise ValueError("角色证据引用不存在的画面")
                self._representative(asset_id, entity, result)
                explicit = entity.get("character_id")
                if explicit is not None:
                    self._resolve(registry, explicit, strict=True)
                    confidence = entity.get("match_confidence")
                    if confidence is not None and (
                            type(confidence) not in {int, float} or not math.isfinite(confidence)
                            or not 0.65 <= confidence <= 1):
                        raise ValueError("角色匹配可信度不足，请保留为待确认人物")
            for entity in result["entities"]:
                is_character = entity.get("is_character", entity.get("kind", "person") in {
                    "person", "character", "anthropomorphic_character",
                })
                if type(is_character) is not bool:
                    raise ValueError("角色类型标记无效")
                if not is_character or entity.get("kind") == "object":
                    entity["character_id"] = None
                    continue
                explicit = entity.get("character_id")
                character_id = self._resolve(registry, explicit, strict=True) if explicit else None
                if explicit and character_id is None:
                    # Human deletion remains effective even for delayed model responses.
                    entity["character_id"] = None
                    continue
                if character_id is None:
                    character_id = "character_" + digest([
                        scope, asset_id, result["run_id"], result["id"], entity["id"],
                    ])
                    if character_id in registry["redirects"]:
                        character_id = self._resolve(registry, character_id)
                        if character_id is None:
                            entity["character_id"] = None
                            continue
                profile = registry["profiles"].get(character_id)
                description = str(entity.get("description") or entity.get("appearance") or "").strip()
                traits = entity.get("observed_traits") or []
                if not isinstance(traits, list) or any(not isinstance(value, str) for value in traits):
                    raise ValueError("角色外观特征格式无效")
                if entity.get("appearance") and isinstance(entity["appearance"], str):
                    traits = [*traits, entity["appearance"]]
                if profile is None:
                    label = str(entity.get("label") or description or f"未命名角色 {len(registry['profiles']) + 1}")
                    timestamp = now()
                    profile = {"id": character_id, "name": label[:80], "aliases": [],
                               "description": description[:8000], "notes": "", "appearance": unique(traits, 64),
                               "user_edited": False, "edited_fields": [], "created_at": timestamp,
                               "updated_at": timestamp, "representative": None}
                    registry["profiles"][character_id] = profile
                locked = set(profile["edited_fields"])
                if description and "description" not in locked:
                    profile["description"] = "\n".join(unique(profile["description"].splitlines() + [description]))[:8000]
                if "appearance" not in locked:
                    profile["appearance"] = unique(profile["appearance"] + traits, 64)
                representative = self._representative(asset_id, entity, result)
                if representative:
                    profile["representative"] = representative
                profile["updated_at"] = now()
                entity["character_id"] = character_id
            result["character_scope_id"] = scope
            result["character_revision"] = registry["revision"]
            atomic_json(self._path(scope), registry)
            return result

    def update(self, asset_id: str, character_id: str, payload: dict) -> dict:
        with self.library.lock:
            scope = scope_id(self.library.get_asset(asset_id))
            registry = self._read(scope)
            payload = dict(payload)
            self._check_revision(registry, payload.pop("expected_revision", None))
            if not payload or set(payload) - EDITABLE_FIELDS:
                raise ValueError("角色档案包含不可编辑字段")
            resolved = self._resolve(registry, character_id, strict=True)
            if resolved is None:
                raise ValueError("角色档案已删除")
            for field, value in payload.items():
                if field in {"aliases", "appearance"}:
                    if (not isinstance(value, list) or len(value) > 100
                            or any(not isinstance(v, str) or len(v) > 2000 for v in value)):
                        raise ValueError("角色别名或外观特征无效")
                    payload[field] = unique(value)
                else:
                    limit = 200 if field == "name" else MAX_NOTES_LENGTH if field == "notes" else 20000
                    if not isinstance(value, str) or len(value) > limit or (field == "name" and not value.strip()):
                        raise ValueError("角色名称或说明无效")
                    payload[field] = value.strip()
            profile = registry["profiles"][resolved]
            profile.update(payload)
            profile["edited_fields"] = sorted(set(profile["edited_fields"]) | payload.keys())
            profile["user_edited"] = True
            profile["updated_at"] = now()
            registry["revision"] += 1
            atomic_json(self._path(scope), registry)
            return next(profile for profile in self.catalog(asset_id)["profiles"] if profile["id"] == resolved)

    def delete(self, asset_id: str, character_id: str, *, expected_revision: int | None = None) -> dict:
        with self.library.lock:
            scope = scope_id(self.library.get_asset(asset_id))
            registry = self._read(scope)
            self._check_revision(registry, expected_revision)
            resolved = self._resolve(registry, character_id, strict=True)
            if resolved is None:
                raise ValueError("角色档案已删除")
            del registry["profiles"][resolved]
            registry["redirects"][resolved] = None
            registry["revision"] += 1
            atomic_json(self._path(scope), registry)
            return {"deleted_id": resolved, "scope_id": scope, "revision": registry["revision"]}

    def merge(self, asset_id: str, source_id: str, target_id: str, *, expected_revision: int | None = None) -> dict:
        with self.library.lock:
            scope = scope_id(self.library.get_asset(asset_id))
            registry = self._read(scope)
            self._check_revision(registry, expected_revision)
            source = self._resolve(registry, source_id, strict=True)
            target = self._resolve(registry, target_id, strict=True)
            if source is None or target is None or source == target:
                raise ValueError("请选择两个不同且未删除的角色档案")
            removed, kept = registry["profiles"][source], registry["profiles"][target]
            aliases = unique(kept["aliases"] + [removed["name"]] + removed["aliases"], 202)
            if len([alias for alias in aliases if normalized(alias) != normalized(kept["name"])]) > 100:
                raise ValueError("合并后的别名超过 100 条，请先精简别名后再合并")
            kept["aliases"] = aliases
            kept["aliases"] = [alias for alias in kept["aliases"] if normalized(alias) != normalized(kept["name"])]
            locked = set(kept["edited_fields"])
            merged_notes = []
            for field in ("description", "notes"):
                if not kept[field] and field not in locked:
                    kept[field] = removed[field]
                    if field in removed["edited_fields"]:
                        locked.add(field)
                elif removed[field] and normalized(removed[field]) != normalized(kept[field]):
                    label = "描述" if field == "description" else "备注"
                    merged_notes.append(f"合并自「{removed['name']}」的{label}：\n{removed[field]}")
            if "appearance" not in locked:
                appearance = unique(kept["appearance"] + removed["appearance"], 200)
                kept["appearance"] = appearance[:64]
                if appearance[64:]:
                    merged_notes.append(f"合并自「{removed['name']}」的其他外观记录：\n" + "；".join(appearance[64:]))
            else:
                existing = {normalized(value) for value in kept["appearance"]}
                additional = [value for value in removed["appearance"] if normalized(value) not in existing]
                if additional:
                    merged_notes.append(f"合并自「{removed['name']}」的外观记录：\n" + "；".join(additional))
            if merged_notes:
                kept["notes"] = "\n\n".join([value for value in [kept["notes"], *merged_notes] if value])
                locked.add("notes")
            if len(kept["notes"]) > MAX_NOTES_LENGTH:
                raise ValueError("合并后的备注超过 100000 字，请先精简备注后再合并")
            kept["representative"] = kept.get("representative") or removed.get("representative")
            kept["edited_fields"] = sorted(locked | {"name", "aliases"})
            kept["user_edited"] = True
            kept["updated_at"] = now()
            del registry["profiles"][source]
            registry["redirects"][source] = target
            registry["revision"] += 1
            atomic_json(self._path(scope), registry)
            return next(profile for profile in self.catalog(asset_id)["profiles"] if profile["id"] == target)

    def decorate_observations(self, asset_id: str, records: list[dict]) -> list[dict]:
        scope = scope_id(self.library.get_asset(asset_id))
        with self.library.lock:
            registry = self._read(scope)
            result = copy.deepcopy(records)
            for record in result:
                for entity in record.get("entities", []):
                    original = entity.get("character_id")
                    resolved = self._resolve(registry, original)
                    if resolved:
                        profile = registry["profiles"][resolved]
                        entity.update(character_id=resolved, character_name=profile["name"], name=profile["name"],
                                      aliases=list(profile["aliases"]), character_user_edited=profile["user_edited"])
                    elif original:
                        entity.update(character_id=None, character_name=None, name=None, aliases=[])
            return result

    def validate_all(self) -> None:
        """Verify registry and evidence references before restoring any backup file."""
        assets = {asset["id"]: asset for asset in self.library.list_assets()}
        registries = {}
        for path in (self.library.root / "works").glob("*/characters.json"):
            scope = checked_id(path.parent.name)
            registry = self._read(scope)
            if not any(scope_id(asset) == scope for asset in assets.values()):
                raise ValueError("角色档案没有对应的视频作品")
            registries[scope] = registry
            for profile in registry["profiles"].values():
                representative = profile.get("representative")
                if representative:
                    asset = assets.get(representative["asset_id"])
                    if not asset or scope_id(asset) != scope:
                        raise ValueError("角色证据引用了其他作品")
                    self.library.frame_path(asset["id"], representative["frame_id"])
        for asset in assets.values():
            scope = scope_id(asset)
            registry = registries.get(scope, {"profiles": {}, "redirects": {}})
            # Historical immutable records also need valid cross-work references.
            for path in self.library.asset_dir(asset["id"]).glob("analyses/*/observations/*.json"):
                record = read_json(path)
                if record.get("character_scope_id") not in {None, scope}:
                    raise ValueError("识别记录引用了其他作品的角色档案")
                for entity in record.get("entities", []):
                    self._representative(asset["id"], entity, record)
                    if entity.get("character_id"):
                        self._resolve(registry, entity["character_id"], strict=True)
