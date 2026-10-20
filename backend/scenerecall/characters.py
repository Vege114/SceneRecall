"""Editable cast registries over immutable, per-shot model observations.

Identity belongs to a series (including every season), or to one work when no
series was supplied. Redirects keep historical evidence readable after merges
and deletions without changing the original model output.
"""
from __future__ import annotations

import copy
import math
import re
import unicodedata
from pathlib import Path

from .library import SCHEMA, atomic_json, digest, now, read_json, safe_id
from .models import Observation


EDITABLE_FIELDS = {"name", "aliases", "description", "notes", "appearance"}
MAX_NOTES_LENGTH = 100000
BANGUMI_REFERENCE_POLICY = (
    "Bangumi identities, biographies and images are prior reference only. "
    "They do not establish current presence, appearance, actions or relationships; "
    "current observations must be supported by the supplied video frames."
)
CAST_POLICY = {
    "mode": "named_only",
    "allow_new_profiles": True,
    "instruction": "Create dossiers only for named narrative characters with supported identity; anonymous people, appearance labels and temporary entity IDs remain observations only.",
}


class StaleCharacterRevision(ValueError):
    """A human changed the cast while an analysis response was in flight."""


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def scope_id(asset: dict) -> str:
    if asset.get("character_scope_id"):
        return safe_id(asset["character_scope_id"])
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


def bangumi_id(value) -> int:
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit() and int(value) > 0:
        return int(value)
    raise ValueError("Bangumi 条目或角色 ID 无效")


def external_sources(profile: dict) -> list[dict]:
    return [source for source in [profile.get("external_source"), *profile.get("external_sources", [])]
            if source is not None]


def named_identity(value) -> str | None:
    """A supplied identity must be a name, never an appearance/temporary label."""
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        return None
    name = value.strip()
    if normalized(name) in {"unknown", "unidentified", "unnamed", "anonymous", "none", "null", "n/a", "未知", "不明", "待确认"}:
        return None
    if re.fullmatch(
        r"(?:未命名|未知|不明|待确认|匿名|unknown[_\s-]*|unidentified[_\s-]*|unnamed[_\s-]*|anonymous[_\s-]*)?"
        r"(?:人物|角色|路人|女孩|男孩|女人|男人|女子|男子|学生|老师|观众|工作人员|人|少女|少年|person|character|entity|woman|man|girl|boy|student|teacher|bystander|extra|crowd|people|female|male|child|teenager)"
        r"(?:[\s_#-]*[A-Za-z\d一二三四五六七八九十甲乙丙丁]+)?", name, flags=re.I
    ) or re.fullmatch(
        r".*(?:学生|老师|教师|男生|女生|观众|工作人员|少女|少年|女子|男子|女孩|男孩|女人|男人|女性|男性|同学|人物|的人|青年|小孩|儿童)", name
    ) or re.fullmatch(
        r"(?:red|blue|black|white|short|long|left|right|background|foreground|young|old|tall|small|unidentified|unknown|anonymous|wearing|glasses|cap|hat|uniformed|stage|at|on|in|a|the)[\s_-]+[\w\s-]*(?:person|woman|man|girl|boy|student|teacher|character|child|teenager)",
        name, flags=re.I
    ):
        return None
    return name


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
    def cast_policy(registry: dict, archive: dict | None = None) -> dict:
        if archive and archive.get("mode") == "scratch":
            return copy.deepcopy(CAST_POLICY)
        if archive and archive.get("mode") == "bangumi":
            dossier = archive.get("dossier") or {}
            return {"mode": "bangumi", "allow_new_profiles": False,
                    "subject_id": bangumi_id(dossier.get("subject", {}).get("id")),
                    "character_ids": sorted({bangumi_id(row.get("id")) for row in dossier.get("characters", [])}),
                    "instruction": "The supplied Bangumi cast is the complete dossier roster. Never add another dossier; unmatched people remain observations with character_id=null."}
        return copy.deepcopy(registry.get("cast_policy") or CAST_POLICY)

    @staticmethod
    def profile_allowed(profile: dict, policy: dict) -> bool:
        if policy["mode"] == "bangumi":
            return any(bangumi_id(source["subject_id"]) == policy["subject_id"]
                       and bangumi_id(source["character_id"]) in policy["character_ids"]
                       for source in external_sources(profile))
        # Preserve deliberate human naming. Old machine dossiers named with their
        # own appearance description are hidden without deleting raw evidence.
        return ("name" in profile.get("edited_fields", [])
                or (named_identity(profile.get("name")) is not None
                    and normalized(profile["name"]) not in {
                        normalized(line) for line in profile.get("description", "").splitlines()}))

    @staticmethod
    def _validate_structure(registry: dict, scope: str) -> None:
        if (not isinstance(registry, dict) or registry.get("scope_id") != scope
                or type(registry.get("revision")) is not int or registry["revision"] < 0
                or not isinstance(registry.get("profiles"), dict)
                or not isinstance(registry.get("redirects"), dict)):
            raise ValueError("角色档案格式或所属作品无效")
        if (type(registry.get("semantic_revision", registry["revision"])) is not int
                or not 0 <= registry.get("semantic_revision", registry["revision"]) <= registry["revision"]):
            raise ValueError("角色语义修订版本无效")
        policy = registry.get("cast_policy")
        if policy is not None:
            if (not isinstance(policy, dict) or policy.get("mode") != "bangumi"
                    or policy.get("allow_new_profiles") is not False
                    or not isinstance(policy.get("character_ids"), list)
                    or not isinstance(policy.get("instruction"), str)):
                raise ValueError("角色建档策略无效")
            bangumi_id(policy.get("subject_id"))
            for identity in policy["character_ids"]:
                bangumi_id(identity)
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
            if not isinstance(profile.get("identity_origins", []), list):
                raise ValueError("角色身份来源无效")
            origins = [*profile.get("identity_origins", [])]
            if profile.get("imported_from") is not None:
                origins.append(profile["imported_from"])
            for origin in origins:
                if not isinstance(origin, dict):
                    raise ValueError("角色身份来源无效")
                checked_id(origin.get("scope_id"))
                checked_id(origin.get("character_id"))
            if not isinstance(profile.get("external_sources", []), list):
                raise ValueError("角色外部来源无效")
            for source in external_sources(profile):
                if not isinstance(source, dict) or source.get("source") != "bangumi":
                    raise ValueError("角色外部来源无效")
                bangumi_id(source.get("subject_id"))
                bangumi_id(source.get("character_id"))
                for field in ("url", "fetched_at"):
                    if not isinstance(source.get(field), str):
                        raise ValueError("角色外部来源字段无效")
                if source.get("image_url") is not None and not isinstance(source["image_url"], str):
                    raise ValueError("角色外部来源图片无效")
            if not isinstance(profile.get("reference_description", ""), str):
                raise ValueError("角色参考说明无效")
            for field in ("name_cn", "name_ja", "name_en"):
                if not isinstance(profile.get(field, ""), str):
                    raise ValueError("角色多语言名称无效")
            variants = profile.get("name_variants", [])
            if (not isinstance(variants, list) or any(not isinstance(row, dict)
                    or any(not isinstance(row.get(field), str)
                           for field in ("name", "language", "kind", "source", "source_url")) for row in variants)):
                raise ValueError("角色多语言名称来源无效")
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
        imports = registry.get("imports", {})
        if not isinstance(imports, dict):
            raise ValueError("角色来源映射无效")
        for source_scope, mapping in imports.items():
            checked_id(source_scope)
            if not isinstance(mapping, dict):
                raise ValueError("角色来源映射无效")
            for source_id, target in mapping.items():
                checked_id(source_id)
                CharacterStore._resolve(registry, target, strict=True)
        bindings = registry.get("external_bindings", {})
        if not isinstance(bindings, dict) or set(bindings) - {"bangumi"}:
            raise ValueError("角色外部身份映射无效")
        for mapping in bindings.values():
            if not isinstance(mapping, dict):
                raise ValueError("角色外部身份映射无效")
            for source_id, target in mapping.items():
                bangumi_id(source_id)
                CharacterStore._resolve(registry, target, strict=True)

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

    @staticmethod
    def _remember_external_identity(registry: dict, profile: dict) -> None:
        """Retain external identity decisions even after a profile is deleted."""
        for source in external_sources(profile):
            mapping = registry.setdefault("external_bindings", {}).setdefault("bangumi", {})
            mapping.setdefault(str(bangumi_id(source["character_id"])), profile["id"])

    @staticmethod
    def seed_registry(registry: dict, dossier: dict) -> dict:
        """Add a Bangumi cast without rewriting local evidence or human decisions.

        This returns a copy and performs no I/O, so the project dossier and cast
        can be published together by OrganizationStore's durable transaction.
        """
        if not isinstance(registry, dict):
            raise ValueError("角色档案格式无效")
        scope = checked_id(registry.get("scope_id"))
        CharacterStore._validate_structure(registry, scope)
        if (not isinstance(dossier, dict) or dossier.get("source") != "bangumi"
                or not isinstance(dossier.get("subject"), dict)
                or not isinstance(dossier.get("characters"), list)):
            raise ValueError("Bangumi 档案格式无效")
        subject_id = bangumi_id(dossier["subject"].get("id"))
        fetched_at = dossier.get("fetched_at")
        if fetched_at is None:
            fetched_at = now()
        if not isinstance(fetched_at, str) or not fetched_at.strip() or len(fetched_at) > 128:
            raise ValueError("Bangumi 档案获取时间无效")
        result = copy.deepcopy(registry)
        result["cast_policy"] = CharacterStore.cast_policy(registry, {"mode": "bangumi", "dossier": dossier})
        timestamp = fetched_at
        for character in dossier["characters"]:
            if not isinstance(character, dict):
                raise ValueError("Bangumi 角色档案无效")
            source_id = bangumi_id(character.get("id"))
            for field in ("name", "name_cn", "name_ja", "name_en", "description", "url", "image_url", "role"):
                if character.get(field) is not None and not isinstance(character[field], str):
                    raise ValueError("Bangumi 角色文字字段无效")
            raw_aliases = character.get("aliases", [])
            if raw_aliases is None:
                raw_aliases = []
            if not isinstance(raw_aliases, list) or any(not isinstance(value, str) for value in raw_aliases):
                raise ValueError("Bangumi 角色别名无效")
            name = ((character.get("name_cn") or "").strip()
                    or (character.get("name") or "").strip())[:200]
            if not name:
                raise ValueError("Bangumi 角色名称不能为空")
            aliases = unique([value[:2000] for value in [character.get("name") or "",
                             character.get("name_cn") or "", *raw_aliases]])
            aliases = [alias for alias in aliases if normalized(alias) != normalized(name)]
            language_names = {field: (character.get(field) or "").strip()[:200]
                              for field in ("name_cn", "name_ja", "name_en")}
            language_names["name_ja"] = language_names["name_ja"] or (character.get("name") or "").strip()[:200]
            variants = character.get("name_variants", [])
            if (not isinstance(variants, list) or any(not isinstance(row, dict)
                    or any(not isinstance(row.get(field), str)
                           for field in ("name", "language", "kind", "source", "source_url")) for row in variants)):
                raise ValueError("Bangumi 角色多语言名称来源无效")
            aliases = unique(aliases + [value for value in language_names.values()
                                      if value and normalized(value) != normalized(name)])
            description = (character.get("description") or "").strip()[:8000]
            source = {"source": "bangumi", "subject_id": subject_id, "character_id": source_id,
                      "url": (character.get("url") or f"https://bgm.tv/character/{source_id}")[:2000],
                      "image_url": (character.get("image_url") or "")[:2000] or None,
                      "fetched_at": fetched_at}
            if character.get("role"):
                source["role"] = character["role"][:200]
            mapping = result.setdefault("external_bindings", {}).setdefault("bangumi", {})
            identity = str(source_id)
            if identity in mapping:
                character_id = CharacterStore._resolve(result, mapping[identity], strict=True)
                if character_id is None:
                    continue
            else:
                # Imported profiles can retain an older local ID. Provenance,
                # rather than a name guess, proves their external identity.
                character_id = next((profile["id"] for profile in result["profiles"].values()
                    if any(origin["source"] == "bangumi" and bangumi_id(origin["character_id"]) == source_id
                           for origin in external_sources(profile))), None)
                deterministic = "character_bgm_" + digest([scope, "bangumi", source_id])
                if character_id is None and deterministic in result["redirects"]:
                    character_id = CharacterStore._resolve(result, deterministic, strict=True)
                    mapping[identity] = deterministic
                    if character_id is None:
                        continue
                character_id = character_id or deterministic
                mapping.setdefault(identity, character_id)
            profile = result["profiles"].get(character_id)
            if profile is None:
                profile = {"id": character_id, "name": name, "aliases": aliases,
                           "description": description, "reference_description": description,
                           "notes": "", "appearance": [], "user_edited": False, "edited_fields": [],
                           "created_at": timestamp, "updated_at": timestamp, "representative": None,
                           "external_source": source, **language_names,
                           "name_variants": copy.deepcopy(variants)}
                result["profiles"][character_id] = profile
                continue
            before = copy.deepcopy(profile)
            known_sources = external_sources(profile)
            matching_source = next((origin for origin in known_sources
                if bangumi_id(origin["character_id"]) == source_id
                and bangumi_id(origin["subject_id"]) == subject_id), None)
            if not profile.get("external_source"):
                profile["external_source"] = source
            elif matching_source is None:
                profile.setdefault("external_sources", []).append(source)
            else:
                for field, value in source.items():
                    if not matching_source.get(field) and value:
                        matching_source[field] = value
            primary = profile["external_source"]
            is_primary = (bangumi_id(primary["subject_id"]) == subject_id
                          and bangumi_id(primary["character_id"]) == source_id)
            if is_primary:
                if "name" not in profile["edited_fields"] and profile["name"] != name:
                    if "aliases" not in profile["edited_fields"]:
                        profile["aliases"] = unique([*profile["aliases"], profile["name"]])
                    profile["name"] = name
                for field, value in language_names.items():
                    if value:
                        profile[field] = value
                profile["reference_description"] = description
            else:
                aliases = unique([name, *aliases])
            for variant in variants:
                if variant not in profile.setdefault("name_variants", []):
                    profile["name_variants"].append(copy.deepcopy(variant))
            locked = set(profile["edited_fields"])
            if "aliases" not in locked:
                profile["aliases"] = unique(profile["aliases"] + aliases)
                profile["aliases"] = [alias for alias in profile["aliases"]
                                      if normalized(alias) != normalized(profile["name"])]
            if is_primary and "description" not in locked:
                # Official biographies remain separate from observed appearance.
                # This also repairs old machine-appended prose without changing
                # the immutable observations or a human's description edits.
                profile["description"] = profile.get("reference_description", "")
            if profile != before:
                profile["updated_at"] = timestamp
        subjects = {bangumi_id(source["subject_id"]) for profile in result["profiles"].values()
                    for source in external_sources(profile)}
        if len(subjects) > 1:
            # A series can share role identities across several works. The
            # concrete asset archive supplies the roster policy; retaining a
            # last-viewed work policy here would mutate revisions on every view.
            result.pop("cast_policy", None)
        if result != registry:
            result["semantic_revision"] = registry.get("semantic_revision", registry["revision"]) + 1
            result["revision"] += 1
        CharacterStore._validate_structure(result, scope)
        return result

    def seed_bangumi(self, scope: str, dossier: dict) -> dict:
        with self.library.lock:
            previous = self._read(scope)
            registry = self.seed_registry(previous, dossier)
            if registry != previous:
                atomic_json(self._path(scope), registry)
            return registry

    def _assets(self, scope: str, *, catalog=None) -> list[dict]:
        return sorted([asset for asset in self.library.list_assets(_catalog=catalog) if scope_id(asset) == scope],
                      key=lambda a: (a.get("season") or 0, a.get("episode") or 0, a["created_at"], a["id"]))

    def catalog(self, asset_id: str) -> dict:
        with self.library.lock:
            organization = self.library.organization._read()
            asset = self.library.get_asset(asset_id, _catalog=organization)
            membership_revision = asset.get("membership_revision", 0)
            scope = scope_id(asset)
            registry = self._read(scope)
            archive = asset.get("anime_profile")
            if archive and archive.get("mode") == "bangumi":
                # Upgrade old offline snapshots locally; no remote API is needed
                # while viewing a cast or resuming recognition.
                from .bangumi import normalize_dossier_names
                upgraded = self.seed_registry(registry, normalize_dossier_names(archive["dossier"]))
                if upgraded != registry:
                    atomic_json(self._path(scope), upgraded)
                    registry = upgraded
            assets = self._assets(scope, catalog=organization)
            policy = self.cast_policy(registry, asset.get("anime_profile"))
            profiles = {identity: copy.deepcopy(profile) for identity, profile in registry["profiles"].items()
                        if self.profile_allowed(profile, policy)}
            current_representatives = {}
            for profile in profiles.values():
                profile["appearances"] = []
            for asset in assets:
                from .scenes import SceneStore
                scenes = SceneStore(self.library).list(asset["id"])
                deep_coverage = []
                for scene in sorted(scenes, key=lambda row: (row["start_ms"], row["end_ms"])):
                    if scene.get("understanding_level") != "deep":
                        continue
                    start, end = scene["start_ms"], scene["end_ms"]
                    if deep_coverage and start <= deep_coverage[-1][1]:
                        deep_coverage[-1][1] = max(deep_coverage[-1][1], end)
                    else:
                        deep_coverage.append([start, end])
                observations = [row for row in self.library.observations(asset["id"])
                                if not any(start <= row["start_ms"] and row["end_ms"] <= end
                                           for start, end in deep_coverage)]
                from .retrieval_store import RetrievalStore
                retrieval = RetrievalStore(self.library).records(asset["id"])
                from .visual_identity import decorate_identity_records
                active_records = decorate_identity_records(self.library, asset["id"], sorted(
                    observations + scenes + retrieval, key=lambda row: (row["start_ms"], row["end_ms"], row["id"])))
                for record in active_records:
                    for entity in record.get("entities", []):
                        resolved = self.library.organization.resolve_reference(
                            asset["id"], record, entity.get("character_id"), registry, catalog=organization)
                        if resolved not in profiles:
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
                profile["representative"] = current_representatives.get(profile["id"])
            return {"scope_id": scope, "revision": registry["revision"],
                    "profiles": list(profiles.values()), "assets": assets,
                    "cast_policy": policy,
                    "membership_revision": membership_revision}

    def context(self, asset_id: str) -> dict:
        with self.library.lock:
            catalog = self.catalog(asset_id)
            fields = ("id", "name", "aliases", "description", "notes", "appearance",
                      "user_edited", "edited_fields", "representative", "updated_at", "created_at")
            context = {"scope_id": catalog["scope_id"], "revision": catalog["revision"],
                       "cast_policy": copy.deepcopy(catalog["cast_policy"]),
                       "membership_revision": catalog["membership_revision"], "profiles": [
                {**{key: copy.deepcopy(profile[key]) for key in fields},
                 "observed_traits": list(profile["appearance"]),
                 **({"external_source": copy.deepcopy(profile.get("external_source")),
                     "external_sources": copy.deepcopy(profile.get("external_sources", [])),
                     "reference_description": profile.get("reference_description", ""),
                     **{key: copy.deepcopy(profile.get(key, [] if key == "name_variants" else ""))
                        for key in ("name_cn", "name_ja", "name_en", "name_variants")},
                     "reference_policy": BANGUMI_REFERENCE_POLICY}
                    if external_sources(profile) else {})}
                for profile in catalog["profiles"] if (profile["appearance_count"] > 0 or external_sources(profile)
                                                       or profile.get("user_edited"))
            ]}
            anime_profile = self.library.get_asset(asset_id).get("anime_profile") or {}
            if anime_profile.get("mode") == "bangumi":
                dossier = anime_profile.get("dossier") or {}
                subject = dossier.get("subject") or {}
                context["work_summary"] = {
                    "source": "bangumi", "id": subject.get("id"),
                    **{field: str(subject.get(field) or "")[:limit]
                       for field, limit in (("name", 200), ("name_cn", 200), ("url", 2000), ("summary", 4000))},
                    "reference_policy": BANGUMI_REFERENCE_POLICY,
                }
            return context

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
            profile = next((profile for profile in self.catalog(asset_id)["profiles"] if profile["id"] == resolved), None)
            if profile is None:
                raise FileNotFoundError("角色不在当前作品的具名角色名单中")
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
            policy = self.cast_policy(registry, asset.get("anime_profile"))
            from .visual_identity import decorate_identity_records
            result = Observation.model_validate(decorate_identity_records(
                self.library, asset_id, [copy.deepcopy(record)], strict=True)[0]).model_dump()
            if result["asset_id"] != asset_id or result["end_ms"] > asset["duration_ms"]:
                raise ValueError("角色证据不属于当前视频时间轴")
            if result["evidence_frame_ids"]:
                # Validate every path against one operation's registry snapshot.
                self.library.frame_paths(asset_id, result["evidence_frame_ids"])
            from .identity_bindings import uses_policy
            for entity in result["entities"]:
                checked_id(entity["id"])
                if not set(entity.get("evidence_frame_ids", [])).issubset(result["evidence_frame_ids"]):
                    raise ValueError("角色证据引用不存在的画面")
                self._representative(asset_id, entity, result)
                identity_status = entity.get("identity_status", entity.get("association_status"))
                if identity_status in {"candidate", "unknown"}:
                    # Retrieval suggestions must not become facts merely because
                    # a provider copied the suggested dossier ID into its output.
                    if identity_status == "candidate" and entity.get("character_id"):
                        entity.setdefault("candidate_character_id", entity["character_id"])
                    entity["character_id"] = None
                explicit = entity.get("character_id")
                if explicit is not None:
                    self._resolve(registry, explicit, strict=True)
                    confidence = entity.get("match_confidence")
                    if confidence is not None and (
                            type(confidence) not in {int, float} or not math.isfinite(confidence)
                            or not 0.65 <= confidence <= 1):
                        raise ValueError("角色匹配可信度不足，请保留为待确认人物")
            if uses_policy(result):
                # An observation decision must never silently enrich the
                # reference biography/appearance used for future recognition.
                result["character_scope_id"] = scope
                result["character_revision"] = registry["revision"]
                return result
            for entity in result["entities"]:
                if entity.get("identity_status", entity.get("association_status")) in {"candidate", "unknown"}:
                    entity["character_id"] = None
                    continue
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
                if character_id and not self.profile_allowed(registry["profiles"][character_id], policy):
                    entity["character_id"] = None
                    continue
                if explicit and character_id is None:
                    # Human deletion remains effective even for delayed model responses.
                    entity["character_id"] = None
                    continue
                if character_id is None:
                    identity_name = named_identity(entity.get("name"))
                    confidence = entity.get("match_confidence")
                    reason = str(entity.get("match_reason") or "").strip()
                    if (not policy["allow_new_profiles"] or identity_name is None or not reason
                            or type(confidence) not in {int, float} or not math.isfinite(confidence)
                            or confidence < 0.65
                            or normalized(identity_name) not in normalized(reason + " " + str(entity.get("description") or ""))):
                        entity["character_id"] = None
                        continue
                    matching = [profile["id"] for profile in registry["profiles"].values()
                                if self.profile_allowed(profile, policy)
                                and normalized(identity_name) in {normalized(value) for value in
                                    [profile["name"], *profile["aliases"]]}]
                    if len(matching) == 1:
                        character_id = matching[0]
                    elif len(matching) > 1:
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
                    timestamp = now()
                    profile = {"id": character_id, "name": identity_name, "aliases": [],
                               "description": description[:8000], "notes": "", "appearance": unique(traits, 64),
                               "user_edited": False, "edited_fields": [], "created_at": timestamp,
                               "updated_at": timestamp, "representative": None}
                    registry["profiles"][character_id] = profile
                locked = set(profile["edited_fields"])
                if description and "description" not in locked and not external_sources(profile):
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

    def create(self, asset_id: str, payload: dict) -> dict:
        """Create a human-supplied identity before its first video appearance."""
        from .library import uid
        with self.library.lock:
            asset = self.library.get_asset(asset_id)
            scope = scope_id(asset)
            registry = self._read(scope)
            if not self.cast_policy(registry, asset.get("anime_profile"))["allow_new_profiles"]:
                raise ValueError("已绑定角色名单的作品请使用现有角色档案；未知人物保留为局部实体")
            name = named_identity(payload.get("name"))
            aliases = payload.get("aliases", [])
            description = payload.get("description", "")
            if (name is None or not isinstance(aliases, list) or len(aliases) > 100
                    or any(not isinstance(value, str) or len(value) > 2000 for value in aliases)
                    or not isinstance(description, str) or len(description) > 20000):
                raise ValueError("请填写真实角色名称和有效的档案字段")
            timestamp = now()
            identity = uid("character")
            profile = {"id": identity, "name": name, "aliases": unique(aliases), "description": description.strip(),
                       "notes": "", "appearance": [], "representative": None, "user_edited": True,
                       "edited_fields": ["aliases", "description", "name"], "created_at": timestamp,
                       "updated_at": timestamp}
            registry["profiles"][identity] = profile
            registry["semantic_revision"] = registry.get("semantic_revision", registry["revision"]) + 1
            registry["revision"] += 1
            atomic_json(self._path(scope), registry)
            return next(row for row in self.catalog(asset_id)["profiles"] if row["id"] == identity)

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
            if not self.profile_allowed(registry["profiles"][resolved], self.cast_policy(
                    registry, self.library.get_asset(asset_id).get("anime_profile"))):
                raise FileNotFoundError("角色不在当前作品的具名角色名单中")
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
            # Display names/aliases are resolved at read/index time; changing
            # them is not a new observation or a reason to re-run plot models.
            registry["semantic_revision"] = registry.get("semantic_revision", registry["revision"]) + (
                0 if set(payload).issubset({"name", "aliases"}) else 1)
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
            if not self.profile_allowed(registry["profiles"][resolved], self.cast_policy(
                    registry, self.library.get_asset(asset_id).get("anime_profile"))):
                raise FileNotFoundError("角色不在当前作品的具名角色名单中")
            self._remember_external_identity(registry, registry["profiles"][resolved])
            del registry["profiles"][resolved]
            registry["redirects"][resolved] = None
            registry["semantic_revision"] = registry.get("semantic_revision", registry["revision"]) + 1
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
            policy = self.cast_policy(registry, self.library.get_asset(asset_id).get("anime_profile"))
            if not self.profile_allowed(kept, policy):
                raise FileNotFoundError("角色不在当前作品的具名角色名单中")
            self._remember_external_identity(registry, removed)
            self._remember_external_identity(registry, kept)
            kept_sources = external_sources(kept)
            for origin in external_sources(removed):
                if not kept_sources:
                    kept["external_source"] = copy.deepcopy(origin)
                elif origin not in kept_sources:
                    kept.setdefault("external_sources", []).append(copy.deepcopy(origin))
                kept_sources = external_sources(kept)
            if not kept.get("reference_description") and removed.get("reference_description"):
                kept["reference_description"] = removed["reference_description"]
            for field in ("name_cn", "name_ja", "name_en"):
                if not kept.get(field) and removed.get(field):
                    kept[field] = removed[field]
            for variant in removed.get("name_variants", []):
                if variant not in kept.setdefault("name_variants", []):
                    kept["name_variants"].append(copy.deepcopy(variant))
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
            registry["semantic_revision"] = registry.get("semantic_revision", registry["revision"]) + 1
            registry["revision"] += 1
            atomic_json(self._path(scope), registry)
            return next(profile for profile in self.catalog(asset_id)["profiles"] if profile["id"] == target)

    def decorate_observations(self, asset_id: str, records: list[dict]) -> list[dict]:
        with self.library.lock:
            organization = self.library.organization._read()
            asset = self.library.get_asset(asset_id, _catalog=organization)
            scope = scope_id(asset)
            registry = self._read(scope)
            policy = self.cast_policy(registry, asset.get("anime_profile"))
            from .visual_identity import decorate_identity_records
            result = decorate_identity_records(self.library, asset_id, copy.deepcopy(records))
            for record in result:
                for entity in record.get("entities", []):
                    original = entity.get("character_id")
                    resolved = self.library.organization.resolve_reference(asset_id, record, original, registry,
                                                                           catalog=organization)
                    if resolved and not self.profile_allowed(registry["profiles"][resolved], policy):
                        resolved = None
                    if resolved:
                        profile = registry["profiles"][resolved]
                        entity.update(character_id=resolved, character_name=profile["name"], name=profile["name"],
                                      aliases=list(profile["aliases"]), character_user_edited=profile["user_edited"])
                    elif original:
                        entity.update(character_id=None, character_name=None, name=None, aliases=[])
            from .identity_scene import project_actions
            return [project_actions(record) for record in result]

    def validate_all(self) -> None:
        """Verify registry and evidence references before restoring any backup file."""
        assets = {asset["id"]: asset for asset in self.library.list_assets(include_deleted=True)}
        organization = self.library.organization.validate()
        historical_scopes = {scope for membership in organization["assets"].values()
                             for scope in [membership["legacy_character_scope_id"],
                                           membership["character_scope_id"], *membership["bindings"],
                                           *membership.get("previous_scopes", [])]}
        # A project can acquire a Bangumi cast before its first video is imported.
        historical_scopes.update(project["character_scope_id"] for project in organization["projects"].values())
        registries = {}
        for path in (self.library.root / "works").glob("*/characters.json"):
            scope = checked_id(path.parent.name)
            registry = self._read(scope)
            if scope not in historical_scopes:
                raise ValueError("角色档案没有对应的视频作品")
            registries[scope] = registry
            for profile in registry["profiles"].values():
                representative = profile.get("representative")
                if representative:
                    asset = assets.get(representative["asset_id"])
                    membership = organization["assets"].get(representative["asset_id"], {})
                    allowed = {membership.get("legacy_character_scope_id"),
                               membership.get("character_scope_id"), *membership.get("bindings", {}),
                               *membership.get("previous_scopes", [])}
                    if not asset or scope not in allowed:
                        raise ValueError("角色证据引用了其他作品")
                    self.library.frame_path(asset["id"], representative["frame_id"])
        for registry in registries.values():
            for profile in registry["profiles"].values():
                origins = [*profile.get("identity_origins", [])]
                if profile.get("imported_from"):
                    origins.append(profile["imported_from"])
                for origin in origins:
                    source = registries.get(origin["scope_id"])
                    if source is None:
                        raise ValueError("角色档案缺少历史来源")
                    self._resolve(source, origin["character_id"], strict=True)
            for source_scope, mapping in registry.get("imports", {}).items():
                source = registries.get(source_scope)
                if source is None:
                    raise ValueError("角色映射缺少历史来源")
                for character_id in mapping:
                    self._resolve(source, character_id, strict=True)
        for asset in assets.values():
            scope = scope_id(asset)
            registry = registries.get(scope, {"profiles": {}, "redirects": {}})
            # Historical immutable records also need valid cross-work references.
            evidence_dir = self.library.asset_dir(asset["id"])
            for path in [*evidence_dir.glob("analyses/*/observations/*.json"),
                         *evidence_dir.glob("analyses/*/scenes/*.json")]:
                record = read_json(path)
                source_scope = record.get("character_scope_id") or organization["assets"][asset["id"]]["legacy_character_scope_id"]
                source_registry = registries.get(source_scope, {"profiles": {}, "redirects": {}})
                for entity in record.get("entities", []):
                    self._representative(asset["id"], entity, record)
                    if entity.get("character_id"):
                        self._resolve(source_registry, entity["character_id"], strict=True)
                        self.library.organization.resolve_reference(
                            asset["id"], record, entity["character_id"], registry, strict=True, catalog=organization)
