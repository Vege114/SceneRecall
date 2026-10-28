"""Small, explicitly scoped work dossiers; facts never assert on-screen events.

All writes are local and audited. References are uploaded bytes or registered
source frames. Attribution URLs are inert metadata, never download instructions.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .characters import CharacterStore, checked_box
from .library import SCHEMA, atomic_json, confined_path, digest, now, read_json, safe_id, uid
from .models import Contract
from .visual_identity import MAX_IMAGE_BYTES, _image, _png, _write_bytes

VERSION = "work-entities-v1"
MAX_PACKAGE_BYTES = 32 * 1024 * 1024


class EntitySource(Contract):
    id: str = Field(min_length=1, max_length=180)
    kind: Literal["manual", "official", "reference", "episode"] = "manual"
    title: str = Field(default="", max_length=1000)
    url: str = Field(default="", max_length=2000)
    note: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def check_source(self):
        safe_id(self.id)
        if self.url and urlsplit(self.url).scheme not in {"https", "http"}:
            raise ValueError("来源链接只接受 HTTP(S) 出处，不会自动下载")
        return self


class SettingFact(Contract):
    id: str = Field(min_length=1, max_length=180)
    kind: Literal["function", "ownership", "setting"]
    text: str = Field(min_length=1, max_length=2000)
    source_ids: list[str] = Field(min_length=1, max_length=16)
    applicability: str = Field(default="", max_length=1000)


class EntityForm(Contract):
    name: str = Field(min_length=1, max_length=200)
    appearance: str = Field(min_length=1, max_length=2000)


class CharacterReference(Contract):
    asset_id: str
    character_scope_id: str
    character_id: str


class EntityRelation(Contract):
    predicate: str = Field(min_length=1, max_length=200)
    target_concept_id: str | None = None
    character_ref: CharacterReference | None = None
    source_ids: list[str] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def one_target(self):
        if (self.target_concept_id is None) == (self.character_ref is None):
            raise ValueError("实体关系须且只能引用一个作品概念或现有角色")
        return self


class EntityInput(Contract):
    type: Literal["prop", "mecha", "place", "organization_symbol"]
    name: str = Field(min_length=1, max_length=200)
    original_name: str = Field(default="", max_length=200)
    translated_names: list[str] = Field(default_factory=list, max_length=32)
    aliases: list[str] = Field(default_factory=list, max_length=64)
    appearance: str = Field(default="", max_length=6000)
    forms: list[EntityForm] = Field(default_factory=list, max_length=16)
    facts: list[SettingFact] = Field(default_factory=list, max_length=64)
    relations: list[EntityRelation] = Field(default_factory=list, max_length=32)
    sources: list[EntitySource] = Field(default_factory=list, max_length=64)
    applicability: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def evidence_sources(self):
        if not self.name.strip():
            raise ValueError("实体名称不能为空")
        ids = [row.id for row in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("实体资料来源 ID 重复")
        fact_ids = [safe_id(row.id) for row in self.facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("设定事实 ID 重复")
        if any(not set(row.source_ids) <= set(ids) for row in [*self.facts, *self.relations]):
            raise ValueError("设定事实或关系引用了不存在的资料来源")
        if any(not name.strip() or len(name) > 200 for name in [*self.aliases, *self.translated_names]):
            raise ValueError("译名或别名无效")
        return self


class BindingInput(Contract):
    record_id: str
    entity_id: str | None = None
    concept_id: str | None = None
    status: Literal["confirmed", "candidate", "unknown", "revoked"]
    evidence_frame_ids: list[str] = Field(min_length=1, max_length=64)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def identity_state(self):
        if self.status in {"confirmed", "candidate"} and not self.concept_id:
            raise ValueError("确认或候选绑定必须指定作品概念")
        if self.status == "unknown" and self.concept_id:
            raise ValueError("未知观察不能同时绑定作品概念")
        if not self.reason.strip() or len(set(self.evidence_frame_ids)) != len(self.evidence_frame_ids):
            raise ValueError("绑定原因或证据帧无效")
        if self.box is not None:
            checked_box(self.box)
        return self


def observation_digest(record):
    return digest({key: record.get(key) for key in (
        "id", "asset_id", "start_ms", "end_ms", "source_fingerprint", "evidence_frame_ids", "frame_ids")})


class WorkEntityStore:
    def __init__(self, library):
        self.library = library
        self.root = library.root / "works" / "entities"
        self.membership_path = library.root / "works" / "entity-membership.json"

    def _memberships(self):
        result = read_json(self.membership_path, {"schema_version": SCHEMA, "version": VERSION,
                                                 "assets": {}, "audit": []})
        if result.get("version") != VERSION or not isinstance(result.get("assets"), dict):
            raise ValueError("作品实体归属数据无效")
        for asset_id, scope in result["assets"].items():
            safe_id(asset_id)
            safe_id(scope)
        return result

    def _path(self, scope):
        return self.root / safe_id(scope) / "catalog.json"

    def _read(self, scope):
        value = read_json(self._path(scope))
        if value is None:
            raise FileNotFoundError("作品实体资料库不存在")
        if (value.get("version") != VERSION or value.get("work_scope_id") != scope
                or not isinstance(value.get("entities"), dict) or not isinstance(value.get("bindings"), dict)
                or type(value.get("revision")) is not int):
            raise ValueError("作品实体资料库结构无效")
        return value

    def scope(self, asset_id):
        self.library.get_asset(asset_id)
        identity = self._memberships()["assets"].get(asset_id)
        return self._read(identity) if identity else None

    def create_scope(self, title, version="", asset_ids=()):
        if not isinstance(title, str) or not title.strip() or len(title) > 500 or len(version) > 200:
            raise ValueError("作品实体资料库标题或版本无效")
        with self.library.lock:
            memberships = self._memberships()
            for asset_id in asset_ids:
                self.library.get_asset(asset_id)
                if asset_id in memberships["assets"]:
                    raise ValueError("素材已有作品实体归属，不能隐式跨作品重绑定")
            identity = uid("workscope")
            value = {"schema_version": SCHEMA, "version": VERSION, "work_scope_id": identity,
                     "title": title.strip(), "work_version": version, "revision": 0,
                     "entities": {}, "bindings": {}, "audit": [{"action": "create_scope", "at": now()}]}
            atomic_json(self._path(identity), value)
            for asset_id in asset_ids:
                memberships["assets"][asset_id] = identity
                memberships["audit"].append({"at": now(), "asset_id": asset_id, "work_scope_id": identity})
            atomic_json(self.membership_path, memberships)
        return self.scope_view(identity)

    def attach(self, asset_id, scope):
        self.library.get_asset(asset_id)
        self._read(scope)
        with self.library.lock:
            memberships = self._memberships()
            current = memberships["assets"].get(asset_id)
            if current and current != scope:
                raise ValueError("素材已有另一作品实体归属，不能隐式跨作品重绑定")
            if current is None:
                memberships["assets"][asset_id] = scope
                memberships["audit"].append({"at": now(), "asset_id": asset_id, "work_scope_id": scope})
                atomic_json(self.membership_path, memberships)
        return self.scope_view(scope)

    def scope_view(self, scope):
        saved = self._read(scope)
        assets = [asset for asset, identity in self._memberships()["assets"].items() if identity == scope]
        return {key: copy.deepcopy(saved[key]) for key in ("work_scope_id", "title", "work_version", "revision")} | {
            "asset_ids": assets, "entity_count": len([row for row in saved["entities"].values() if row["status"] == "active"])}

    def _required(self, asset_id):
        saved = self.scope(asset_id)
        if saved is None:
            raise ValueError("请先为素材关联作品实体资料库")
        return saved

    def _check_relations(self, saved, data, *, allow_historical=False):
        memberships = self._memberships()["assets"]
        for relation in data["relations"]:
            target = relation.get("target_concept_id")
            if target and target not in saved["entities"]:
                raise ValueError("关系目标不在同一作品实体资料库")
            character = relation.get("character_ref")
            if character:
                if memberships.get(character["asset_id"]) != saved["work_scope_id"]:
                    raise ValueError("角色关系来源不属于当前作品")
                registry = CharacterStore(self.library)._read(character["character_scope_id"])
                resolved = CharacterStore._resolve(registry, character["character_id"], strict=True)
                if resolved not in registry["profiles"]:
                    raise ValueError("角色外键不存在")
                if not allow_historical:
                    catalog = CharacterStore(self.library).catalog(character["asset_id"])
                    if catalog["scope_id"] != character["character_scope_id"]:
                        raise ValueError("角色关系必须引用素材当前的角色资料库")

    @staticmethod
    def _public(row, scope):
        result = copy.deepcopy(row)
        for reference in result.get("references", []):
            reference.pop("path", None)
            reference["image_url"] = (f"/api/work-entity-scopes/{scope}/entities/{row['concept_id']}"
                                      f"/references/{reference['id']}/image")
        return result

    def list(self, asset_id, *, include_revoked=False):
        saved = self.scope(asset_id)
        if saved is None:
            return {"work_scope_id": None, "entities": [], "revision": 0}
        return {"work_scope_id": saved["work_scope_id"], "revision": saved["revision"],
                "entities": [self._public(row, saved["work_scope_id"]) for row in saved["entities"].values()
                             if include_revoked or row["status"] == "active"]}

    def get(self, asset_id, concept_id):
        saved = self._required(asset_id)
        row = saved["entities"].get(safe_id(concept_id))
        if row is None:
            raise FileNotFoundError("作品实体不存在或属于另一作品")
        return self._public(row, saved["work_scope_id"])

    def _commit(self, saved, action, identity, before=None, reason=""):
        saved["revision"] += 1
        saved["audit"].append({"at": now(), "action": action, "id": identity, "reason": reason,
                               "before": copy.deepcopy(before), "revision": saved["revision"]})
        atomic_json(self._path(saved["work_scope_id"]), saved)

    def create(self, asset_id, payload):
        data = EntityInput.model_validate(payload).model_dump()
        with self.library.lock:
            saved = self._required(asset_id)
            self._check_relations(saved, data)
            identity = uid("concept")
            row = {**data, "concept_id": identity, "work_scope_id": saved["work_scope_id"],
                   "status": "active", "version": 1, "references": [], "created_at": now(), "updated_at": now()}
            saved["entities"][identity] = row
            self._commit(saved, "create", identity)
            return self._public(row, saved["work_scope_id"])

    def update(self, asset_id, concept_id, payload, *, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("修改作品设定须填写原因")
        if set(payload) - EntityInput.model_fields.keys():
            raise ValueError("作品实体修改包含未声明字段")
        with self.library.lock:
            saved = self._required(asset_id)
            row = saved["entities"].get(safe_id(concept_id))
            if row is None:
                raise FileNotFoundError("作品实体不存在或属于另一作品")
            data = EntityInput.model_validate({**{key: row[key] for key in EntityInput.model_fields},
                                              **payload}).model_dump()
            self._check_relations(saved, data)
            before = copy.deepcopy(row)
            row.update(data, version=row["version"] + 1, updated_at=now())
            self._commit(saved, "update", concept_id, before, reason)
            return self._public(row, saved["work_scope_id"])

    def revoke(self, asset_id, concept_id, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("撤销作品实体须填写原因")
        with self.library.lock:
            saved = self._required(asset_id)
            row = saved["entities"].get(safe_id(concept_id))
            if row is None:
                raise FileNotFoundError("作品实体不存在")
            before = copy.deepcopy(row)
            row.update(status="revoked", version=row["version"] + 1, updated_at=now())
            self._commit(saved, "revoke", concept_id, before, reason)
            return self._public(row, saved["work_scope_id"])

    def add_reference(self, asset_id, concept_id, content, *, source_url="", confirmed=False,
                      view="whole_scene", source=None):
        EntitySource(id="reference", url=source_url)
        if view not in {"whole_scene", "landmark", "object_region"}:
            raise ValueError("参考图视角类型无效")
        picture = _image(content)
        if min(picture.size) < 16:
            raise ValueError("作品实体参考图尺寸过小")
        encoded = _png(picture)
        with self.library.lock:
            saved = self._required(asset_id)
            row = saved["entities"].get(safe_id(concept_id))
            if row is None or row["status"] != "active":
                raise FileNotFoundError("作品实体不存在或已撤销")
            if row["type"] == "place" and view == "object_region":
                raise ValueError("地点参考使用完整场景或明确地标，不使用物体裁剪规则")
            before = copy.deepcopy(row)
            identity = uid("entityref")
            relative = f"references/{identity}.png"
            _write_bytes(confined_path(self._path(saved["work_scope_id"]).parent, relative), encoded)
            reference = {"id": identity, "path": relative, "sha256": hashlib.sha256(encoded).hexdigest(),
                         "view": view, "status": "confirmed" if confirmed else "candidate",
                         "confirmation_source": "human" if confirmed else None, "version": 1,
                         "source": source or {"kind": "upload", "url": source_url},
                         "width": picture.width, "height": picture.height, "reference_only": True}
            row["references"].append(reference)
            row.update(version=row["version"] + 1, updated_at=now())
            self._commit(saved, "add_reference", concept_id, before)
            return self._public(row, saved["work_scope_id"])["references"][-1]

    def reference_from_frame(self, asset_id, concept_id, frame_id, *, box=None, confirmed=False, view="whole_scene"):
        asset = self.library.get_asset(asset_id)
        path = self.library.frame_path(asset_id, frame_id)
        picture = _image(path.read_bytes())
        if box is not None:
            box = checked_box(box)
            x, y, width, height = box
            picture = picture.crop((round(x * picture.width), round(y * picture.height),
                                    round((x + width) * picture.width), round((y + height) * picture.height)))
            if view == "whole_scene":
                view = "object_region"
        return self.add_reference(asset_id, concept_id, _png(picture), confirmed=confirmed, view=view,
                                  source={"kind": "source_frame", "asset_id": asset_id, "frame_id": frame_id,
                                          "source_fingerprint": asset["fingerprint"], "box": box})

    def reference_path(self, scope, concept_id, reference_id):
        saved = self._read(scope)
        row = saved["entities"].get(safe_id(concept_id))
        reference = next((item for item in row.get("references", []) if item["id"] == safe_id(reference_id)), None) if row else None
        if reference is None:
            raise FileNotFoundError("作品实体参考图不存在")
        path = confined_path(self._path(scope).parent, reference["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("作品实体参考图缺失或校验失败")
        return path

    def update_reference(self, asset_id, concept_id, reference_id, *, status, reason):
        if status not in {"confirmed", "candidate", "revoked"} or not isinstance(reason, str) or not reason.strip():
            raise ValueError("参考图更新须指定有效状态和原因")
        with self.library.lock:
            saved = self._required(asset_id)
            row = saved["entities"].get(safe_id(concept_id))
            reference = next((ref for ref in row.get("references", []) if ref["id"] == safe_id(reference_id)), None) if row else None
            if reference is None:
                raise FileNotFoundError("作品实体参考图不存在")
            self.reference_path(saved["work_scope_id"], concept_id, reference_id)
            before = copy.deepcopy(row)
            reference.update(status=status, confirmation_source="human" if status == "confirmed" else None,
                             version=reference["version"] + 1)
            row.update(version=row["version"] + 1, updated_at=now())
            self._commit(saved, "reference_" + status, concept_id, before, reason)
            return next(ref for ref in self._public(row, saved["work_scope_id"])["references"] if ref["id"] == reference_id)

    def bind(self, asset_id, payload):
        data = BindingInput.model_validate(payload).model_dump()
        with self.library.lock:
            saved = self._required(asset_id)
            concept = saved["entities"].get(data["concept_id"]) if data["concept_id"] else None
            if data["concept_id"] and (concept is None or concept["status"] != "active"):
                raise ValueError("绑定目标不属于当前作品或已撤销")
            record = next((row for row in self.library.records(asset_id) if row["id"] == data["record_id"]), None)
            if record is None or (record["kind"] not in {"visual", "scene"}
                                  and record.get("evidence_type") != "retrieval_window"):
                raise ValueError("实体绑定必须指向现有画面观察")
            if not set(data["evidence_frame_ids"]) <= set(record.get("evidence_frame_ids", [])):
                raise ValueError("实体绑定证据必须属于当前画面观察")
            if data["entity_id"] and data["entity_id"] not in {row["id"] for row in record.get("entities", [])}:
                raise ValueError("镜头局部实体 ID 不存在")
            self.library.frame_paths(asset_id, data["evidence_frame_ids"])
            key = digest([asset_id, data["record_id"], data["entity_id"], data["box"]])
            before = saved["bindings"].get(key)
            binding = {**data, "id": key, "asset_id": asset_id, "work_scope_id": saved["work_scope_id"],
                       "observation_digest": observation_digest(record),
                       "source_fingerprint": self.library.get_asset(asset_id)["fingerprint"],
                       "decision_source": "human", "updated_at": now()}
            saved["bindings"][key] = binding
            self._commit(saved, "binding", key, before, data["reason"])
            return copy.deepcopy(binding)

    def project_record(self, asset_id, record):
        if record.get("asset_id") != asset_id:
            raise ValueError("作品实体投影的素材归属不一致")
        return self.project_records([record])[0]

    def _projection_state(self, records):
        """One membership/catalog/fingerprint read per relevant owner, not cue."""
        memberships = self._memberships()["assets"]
        assets = {row["asset_id"] for row in records if row.get("asset_id") in memberships
                  and (row.get("kind") in {"visual", "scene"} or row.get("evidence_type") == "retrieval_window")}
        catalogs, fingerprints, asset_scopes, bindings = {}, {}, {}, {}
        with self.library.lock:
            for asset_id in assets:
                try:
                    fingerprint = self.library.get_asset(asset_id)["fingerprint"]
                except (FileNotFoundError, ValueError):
                    # Indexed evidence can outlive a deletion until its removal
                    # transaction commits; it cannot acquire a fresh binding.
                    continue
                scope = memberships[asset_id]
                if scope not in catalogs:
                    saved = catalogs[scope] = self._read(scope)
                    for binding in saved["bindings"].values():
                        bindings.setdefault((scope, binding["asset_id"], binding["record_id"]), []).append(binding)
                asset_scopes[asset_id] = scope
                fingerprints[asset_id] = fingerprint
        return catalogs, fingerprints, asset_scopes, bindings

    @staticmethod
    def _project(record, saved, fingerprint, candidates):
        projected = copy.deepcopy(record)
        bindings = []
        observation = observation_digest(record)
        for binding in candidates:
            if (binding["observation_digest"] != observation
                    or binding["source_fingerprint"] != fingerprint):
                continue
            concept = saved["entities"].get(binding["concept_id"])
            if binding["status"] != "confirmed" or not concept or concept["status"] != "active":
                continue
            bindings.append({**copy.deepcopy(binding), "concept_name": concept["name"],
                             "concept_type": concept["type"], "aliases": concept["aliases"]})
        projected["concept_bindings"] = bindings
        return projected

    def project_records(self, records):
        """Project a caller's evidence snapshot without recursively reading it."""
        catalogs, fingerprints, asset_scopes, bindings = self._projection_state(records)
        if not catalogs:
            return records
        result = []
        for record in records:
            scope = asset_scopes.get(record.get("asset_id"))
            if scope is None or not (record.get("kind") in {"visual", "scene"}
                                     or record.get("evidence_type") == "retrieval_window"):
                result.append(record)
                continue
            result.append(self._project(record, catalogs[scope], fingerprints[record["asset_id"]],
                bindings.get((scope, record["asset_id"], record["id"]), ())))
        return result

    @staticmethod
    def _rank_concepts(saved, query, concept_ids=()):
        if len(concept_ids) > 5 or any(identity not in saved["entities"] for identity in concept_ids):
            raise ValueError("按需档案最多五个，且必须属于当前作品")
        from .search import normalize, tokenize
        terms = set(tokenize(query))
        ranked = []
        for row in saved["entities"].values():
            if row["status"] != "active":
                continue
            text = " ".join([row["name"], row["original_name"], *row["translated_names"], *row["aliases"],
                             row["appearance"], *(fact["text"] for fact in row["facts"])])
            score = len(terms & set(tokenize(text)))
            if query and normalize(query) in normalize(text):
                score += 10
            if row["concept_id"] in concept_ids:
                score += 100
            if score:
                ranked.append((score, row))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["concept_id"]))
        return ranked[:5]

    def context(self, asset_id, query="", concept_ids=()):
        saved = self._required(asset_id)
        ranked = self._rank_concepts(saved, query, concept_ids)
        refs_left, profiles = 8, []
        for _, row in ranked:
            public = self._public(row, saved["work_scope_id"])
            public["references"] = [ref for ref in public["references"] if ref["status"] == "confirmed"][:refs_left]
            refs_left -= len(public["references"])
            profiles.append(public)
        return {"work_scope_id": saved["work_scope_id"], "revision": saved["revision"], "profiles": profiles,
                "reference_only": True, "facts_are_not_current_events": True}

    @staticmethod
    def _matches_filters(record, filters):
        from .search import ALLOWED_FILTERS, normalize
        for field, value in filters.items():
            if field not in ALLOWED_FILTERS or value is None or value == "":
                continue
            if field == "character":
                names = []
                for entity in record.get("entities", []):
                    names.extend([entity.get("name", ""), entity.get("character_name", ""), *entity.get("aliases", [])])
                if normalize(str(value)) not in {normalize(name) for name in names if name}:
                    return False
            elif field == "character_id":
                if value not in {entity.get("character_id") for entity in record.get("entities", [])}:
                    return False
            elif field == "concept_id":
                if value not in {binding["concept_id"] for binding in record.get("concept_bindings", [])
                                 if binding["status"] == "confirmed"}:
                    return False
            elif field in {"tag", "folder_id"}:
                if value not in record.get("tags" if field == "tag" else "folder_ids", []):
                    return False
            elif record.get(field) != value:
                return False
        return True

    def search_occurrences(self, query, filters=None, eligible_records=None):
        """Concept recall over a caller-owned, already scoped index snapshot.

        Never calls library.records(): that method may itself use project_records.
        A matching setting function retrieves confirmed appearances only and does
        not establish that the depicted entity is currently using that function.
        """
        records = list(eligible_records or [])
        if not records or not isinstance(query, str) or not query.strip():
            return []
        catalogs, fingerprints, asset_scopes, bindings = self._projection_state(records)
        selected = {scope: {row["concept_id"]: score for score, row in self._rank_concepts(saved, query)}
                    for scope, saved in catalogs.items()}
        results = []
        for record in records:
            scope = asset_scopes.get(record.get("asset_id"))
            if scope is None or not selected.get(scope):
                continue
            projected = self._project(record, catalogs[scope], fingerprints[record["asset_id"]],
                bindings.get((scope, record["asset_id"], record["id"]), ()))
            if not self._matches_filters(projected, filters or {}):
                continue
            match = self._occurrence_match(projected, selected[scope])
            if match is not None:
                results.append(match)
        results.sort(key=lambda pair: (-pair[0], pair[1]["asset_id"], pair[1]["start_ms"], pair[1]["id"]))
        return [row for _, row in results[:50]]

    @staticmethod
    def _occurrence_match(projected, selected):
        """Shared ranking/output contract for traversal and indexed recall."""
        matched = [binding for binding in projected["concept_bindings"] if binding["concept_id"] in selected]
        if not matched:
            return None
        score = max(selected[binding["concept_id"]] for binding in matched)
        return score, {**projected, "matched_concept_ids": [binding["concept_id"] for binding in matched],
            "match_basis": "setting_concept_appearance", "event_verified": False,
            "retrieval_branches": ["concept"], "match_type": "partial",
            "evidence_ids": list(dict.fromkeys(frame for binding in matched for frame in binding["evidence_frame_ids"]))}

    def occurrences(self, asset_id, concept_id):
        concept = self.get(asset_id, concept_id)
        if concept["status"] != "active":
            return []
        rows = []
        for projected in self.project_records(self.library.records(asset_id)):
            matches = [binding for binding in projected.get("concept_bindings", []) if binding["concept_id"] == concept_id]
            if matches:
                rows.append({"record_id": projected["id"], "asset_id": asset_id, "start_ms": projected["start_ms"],
                             "end_ms": projected["end_ms"], "bindings": matches, "event_verified": False,
                             "match_basis": "confirmed_concept_appearance"})
        return rows

    def import_package(self, asset_id, content):
        """Import a small uploaded ZIP atomically; never follow package URLs."""
        if not content or len(content) > MAX_PACKAGE_BYTES:
            raise ValueError("作品实体导入包不能超过 32 MiB")
        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as exc:
            raise ValueError("导入文件必须为包含 manifest.json 的 ZIP 包") from exc
        with archive:
            entries = archive.infolist()
            names = [item.filename for item in entries]
            if (len(entries) > 64 or len(set(names)) != len(names)
                    or sum(item.file_size for item in entries) > MAX_PACKAGE_BYTES):
                raise ValueError("导入包文件数量、大小或重复名称无效")
            for item in entries:
                name = PurePosixPath(item.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in item.filename or ":" in item.filename
                        or item.is_dir() or (item.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError("导入包包含越界路径或链接")
            if "manifest.json" not in names or archive.getinfo("manifest.json").file_size > 1024 * 1024:
                raise ValueError("导入包须包含不超过 1 MiB 的 manifest.json")
            manifest = json.loads(archive.read("manifest.json"))
            if (not isinstance(manifest, dict) or set(manifest) != {"entities"} or not isinstance(manifest["entities"], list)
                    or not 1 <= len(manifest["entities"]) <= 20):
                raise ValueError("导入清单须包含 1–20 项 entities")
            prepared = []
            for item in manifest["entities"]:
                if not isinstance(item, dict):
                    raise ValueError("导入实体必须为对象")
                data = EntityInput.model_validate({key: value for key, value in item.items() if key != "reference_files"}).model_dump()
                refs = []
                reference_files = item.get("reference_files", [])
                if not isinstance(reference_files, list) or len(reference_files) > 8:
                    raise ValueError("每个导入实体最多八张参考图")
                for reference in reference_files:
                    if not isinstance(reference, dict) or set(reference) - {"file", "source_url", "view", "confirmed"}:
                        raise ValueError("导入参考图包含未声明字段")
                    filename = reference.get("file")
                    if filename not in names or filename == "manifest.json" or archive.getinfo(filename).file_size > MAX_IMAGE_BYTES:
                        raise ValueError("导入参考图文件缺失或过大")
                    picture = _image(archive.read(filename))
                    if min(picture.size) < 16:
                        raise ValueError("导入参考图尺寸过小")
                    EntitySource(id="import", url=reference.get("source_url", ""))
                    view = reference.get("view", "whole_scene")
                    if view not in {"whole_scene", "landmark", "object_region"} or data["type"] == "place" and view == "object_region":
                        raise ValueError("导入参考图视角类型无效")
                    refs.append((reference, _png(picture)))
                prepared.append((data, refs))
            with self.library.lock:
                saved = self._required(asset_id)
                for data, _ in prepared:
                    self._check_relations(saved, data)
                identifiers = []
                for data, refs in prepared:
                    identity = uid("concept")
                    row = {**data, "concept_id": identity, "work_scope_id": saved["work_scope_id"],
                           "status": "active", "version": 1, "references": [], "created_at": now(), "updated_at": now()}
                    for reference, encoded in refs:
                        ref_id = uid("entityref")
                        relative = f"references/{ref_id}.png"
                        _write_bytes(confined_path(self._path(saved["work_scope_id"]).parent, relative), encoded)
                        row["references"].append({"id": ref_id, "path": relative,
                            "sha256": hashlib.sha256(encoded).hexdigest(), "view": reference.get("view", "whole_scene"),
                            "status": "candidate", "confirmation_source": None, "version": 1,
                            "source": {"kind": "local_package", "url": reference.get("source_url", "")},
                            "reference_only": True})
                    saved["entities"][identity] = row
                    identifiers.append(identity)
                self._commit(saved, "import_package", hashlib.sha256(content).hexdigest(),
                             reason="受限本地导入；参考图待人工确认")
                return [self._public(saved["entities"][identity], saved["work_scope_id"]) for identity in identifiers]

    def validate_all(self):
        """Backup/restore validation, with no model calls or network access."""
        memberships = self._memberships()["assets"]
        for asset_id, scope in memberships.items():
            self.library.get_asset(asset_id, include_deleted=True)
            self._read(scope)
        for path in self.root.glob("*/catalog.json"):
            saved = self._read(path.parent.name)
            for concept_id, row in saved["entities"].items():
                if (safe_id(concept_id) != row.get("concept_id") or row.get("work_scope_id") != saved["work_scope_id"]
                        or row.get("status") not in {"active", "revoked"}):
                    raise ValueError("作品实体身份或状态无效")
                data = EntityInput.model_validate({key: row[key] for key in EntityInput.model_fields}).model_dump()
                self._check_relations(saved, data, allow_historical=True)
                for reference in row["references"]:
                    if reference.get("status") not in {"confirmed", "candidate", "revoked"}:
                        raise ValueError("作品实体参考图状态无效")
                    if (reference.get("view") not in {"whole_scene", "landmark", "object_region"}
                            or row["type"] == "place" and reference["view"] == "object_region"
                            or reference.get("status") == "confirmed" and reference.get("confirmation_source") != "human"):
                        raise ValueError("作品实体参考图类别或确认来源无效")
                    self.reference_path(saved["work_scope_id"], concept_id, reference["id"])
                    source = reference["source"]
                    if source.get("kind") not in {"upload", "source_frame", "local_package"}:
                        raise ValueError("作品实体参考图来源类型无效")
                    EntitySource(id="reference", url=source.get("url", ""))
                    if source["kind"] == "source_frame":
                        if memberships.get(source["asset_id"]) != saved["work_scope_id"]:
                            raise ValueError("作品实体参考图来自另一作品")
                        self.library.frame_path(source["asset_id"], source["frame_id"])
            for key, binding in saved["bindings"].items():
                BindingInput.model_validate({field: binding[field] for field in BindingInput.model_fields})
                if (key != binding.get("id") or memberships.get(binding["asset_id"]) != saved["work_scope_id"]
                        or binding["work_scope_id"] != saved["work_scope_id"]
                        or binding["concept_id"] and binding["concept_id"] not in saved["entities"]
                        or binding.get("decision_source") != "human"):
                    raise ValueError("作品实体绑定归属或证据状态无效")
                self.library.frame_paths(binding["asset_id"], binding["evidence_frame_ids"])
