"""Authoritative, source-anchored knowledge cards independent of a RAG index.

Source text is uploaded UTF-8 data, never a request to download a URL. Reviewed
cards are the common input to lexical recall and rebuildable LightRAG indexes.
All catalog mutations share WorkEntityStore's lock, atomic writes and audit.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .characters import CharacterStore, scope_id as character_scope_id
from .library import confined_path, digest, now, safe_id, uid
from .models import Contract
from .search_timing import aggregate_operation, timed, timed_lock, timing_count
from .visual_identity import _write_bytes

VERSION = "knowledge-cards-v1"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
REVIEW_STATES = {"candidate", "reviewed", "rejected", "revoked"}


class KnowledgeTarget(Contract):
    kind: Literal["concept", "character"]
    concept_id: str | None = None
    asset_id: str | None = None
    character_scope_id: str | None = None
    character_id: str | None = None

    @model_validator(mode="after")
    def valid_target(self):
        if self.kind == "concept":
            if not self.concept_id or any((self.asset_id, self.character_scope_id, self.character_id)):
                raise ValueError("概念引用仅填写 concept_id")
        elif not all((self.asset_id, self.character_scope_id, self.character_id)) or self.concept_id:
            raise ValueError("角色引用须填写现有素材、角色资料库与角色 ID")
        for identity in (self.concept_id, self.asset_id, self.character_scope_id, self.character_id):
            if identity is not None:
                safe_id(identity)
        return self


class SourceReference(Contract):
    source_id: str
    snapshot_id: str
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def valid_range(self):
        safe_id(self.source_id)
        safe_id(self.snapshot_id)
        if self.end <= self.start:
            raise ValueError("原文范围须为非空的 Unicode 字符区间 [start,end)")
        return self


class KnowledgeCardInput(Contract):
    title: str = Field(default="", max_length=500)
    kind: Literal["definition", "fact", "relation"] = "fact"
    text: str = Field(min_length=1, max_length=12000)
    target_ref: KnowledgeTarget
    predicate: str = Field(default="", max_length=200)
    object_ref: KnowledgeTarget | None = None
    source_refs: list[SourceReference] = Field(min_length=1, max_length=32)
    applicability: str = Field(default="", max_length=2000)
    conditions: list[str] = Field(default_factory=list, max_length=32)
    costs: list[str] = Field(default_factory=list, max_length=32)
    exceptions: list[str] = Field(default_factory=list, max_length=32)
    review_status: Literal["candidate", "reviewed", "rejected", "revoked"] = "candidate"
    reason: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def valid_card(self):
        if not self.text.strip():
            raise ValueError("知识资料卡文字不能为空")
        if self.kind == "relation" and (not self.predicate.strip() or self.object_ref is None):
            raise ValueError("关系资料卡须填写有方向的谓词和目标")
        if self.kind != "relation" and (self.predicate or self.object_ref is not None):
            raise ValueError("关系端点只用于关系资料卡")
        for values in (self.conditions, self.costs, self.exceptions):
            if any(not value.strip() or len(value) > 2000 for value in values):
                raise ValueError("条件、代价和例外须为非空短文本列表")
        return self


def target_key(target):
    """An identity key, never a display-name join."""
    if target["kind"] == "concept":
        return "concept:" + target["concept_id"]
    return "character:" + target["character_scope_id"] + ":" + target["character_id"]


class KnowledgeStore:
    def __init__(self, work_entity_store):
        self.store = work_entity_store
        self.library = work_entity_store.library

    def scope_catalog(self, scope):
        saved = self.store._read(safe_id(scope))
        for field in ("source_documents", "knowledge_cards", "knowledge_alignments"):
            saved.setdefault(field, {})
            if not isinstance(saved[field], dict):
                raise ValueError("作品知识资料结构无效")
        return saved

    def _source(self, saved, source_id):
        source = saved["source_documents"].get(safe_id(source_id))
        if source is None:
            raise FileNotFoundError("资料来源不存在或属于另一作品")
        return source

    @staticmethod
    def _public_source(source):
        return {key: copy.deepcopy(value) for key, value in source.items() if key != "path"}

    def sources(self, scope):
        return [self._public_source(row) for row in self.scope_catalog(scope)["source_documents"].values()]

    def import_source(self, scope, title, url, content, filename="source.txt"):
        if not isinstance(title, str) or not title.strip() or len(title) > 1000:
            raise ValueError("请填写资料标题（最多 1000 字符）")
        if not isinstance(url, str) or len(url) > 2000 or url and urlsplit(url).scheme not in {"https", "http"}:
            raise ValueError("资料出处只接受 HTTP(S) URL，不会自动下载")
        if not isinstance(content, bytes) or not content or len(content) > MAX_SOURCE_BYTES:
            raise ValueError("知识资料须为不超过 2 MiB 的 UTF-8 文本")
        if not isinstance(filename, str) or "\\" in filename or "/" in filename:
            raise ValueError("资料文件名无效")
        if PurePosixPath(filename).suffix.lower() not in {".txt", ".md", ".markdown"}:
            raise ValueError("首轮支持 Markdown 或纯文本资料")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("知识资料必须使用 UTF-8 编码") from exc
        if not text.strip() or "\x00" in text:
            raise ValueError("知识资料不是有效的非空文本")
        sha = hashlib.sha256(content).hexdigest()
        with self.library.lock:
            saved = self.scope_catalog(scope)
            # Idempotent duplicate upload in this work; different works never share IDs.
            for row in saved["source_documents"].values():
                if row["status"] == "active" and row["sha256"] == sha and row["title"] == title.strip() and row["url"] == url:
                    return self._public_source(row)
            identity = uid("source")
            relative = f"knowledge/snapshots/{sha}.txt"
            path = confined_path(self.store._path(scope).parent, relative)
            if not path.exists():
                _write_bytes(path, content)
            elif hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError("原文快照已损坏，不能覆盖同名快照")
            source = {"source_id": identity, "work_scope_id": scope, "work_version_id": saved["work_version_id"],
                      "title": title.strip(), "url": url, "filename": filename, "sha256": sha,
                      "content_hash": sha, "snapshot_id": "snapshot_" + sha, "path": relative,
                      "status": "active", "import_status": "imported", "error": None,
                      "character_count": len(text), "byte_count": len(content), "created_at": now()}
            saved["source_documents"][identity] = source
            self.store._commit(saved, "knowledge_source_import", identity)
            return self._public_source(source)

    def _source_text(self, saved, source_id):
        source = self._source(saved, source_id)
        path = confined_path(self.store._path(saved["work_scope_id"]).parent, source["path"])
        try:
            with aggregate_operation("source_read", identity=str(path)) as operation:
                content = path.read_bytes()
                operation.set(bytes=len(content))
        except FileNotFoundError as exc:
            raise ValueError("资料原文快照缺失") from exc
        with aggregate_operation("source_sha256", identity=str(path), bytes=len(content)):
            content_hash = hashlib.sha256(content).hexdigest()
        if content_hash != source["sha256"]:
            raise ValueError("资料原文快照哈希校验失败")
        with aggregate_operation("source_decode", bytes=len(content)):
            text = content.decode("utf-8-sig")
        return source, text

    def source_content(self, scope, source_id):
        saved = self.scope_catalog(scope)
        source, text = self._source_text(saved, source_id)
        return {**self._public_source(source), "text": text}

    def revoke_source(self, scope, source_id, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("撤销资料须填写原因")
        with self.library.lock:
            saved = self.scope_catalog(scope)
            row = self._source(saved, source_id)
            before = copy.deepcopy(row)
            row.update(status="revoked", revoked_at=now())
            self.store._commit(saved, "knowledge_source_revoke", source_id, before, reason)
            return self._public_source(row)

    def _target(self, saved, reference):
        timing_count("knowledge_target")
        reference = KnowledgeTarget.model_validate(reference).model_dump(exclude_none=True)
        if reference["kind"] == "concept":
            row = saved["entities"].get(reference["concept_id"])
            if not row or row["status"] != "active":
                raise ValueError("知识目标概念不属于当前作品或已撤销")
            return reference, {"name": row["name"], "aliases": [row.get("original_name", ""),
                *row.get("translated_names", []), *row.get("aliases", [])], "type": row["type"],
                "definition": row.get("definition", ""), "applicability": row.get("applicability", "")}
        memberships = self.store._memberships()["assets"]
        asset = self.library.get_asset(reference["asset_id"])
        if memberships.get(reference["asset_id"]) != saved["work_scope_id"]:
            raise ValueError("知识角色不属于当前作品")
        if character_scope_id(asset) != reference["character_scope_id"]:
            raise ValueError("知识角色引用不是素材当前角色资料库")
        characters = CharacterStore(self.library)
        registry = characters._read(reference["character_scope_id"])
        resolved = characters._resolve(registry, reference["character_id"], strict=True)
        profile = registry["profiles"].get(resolved)
        policy = characters.cast_policy(registry, asset.get("anime_profile"))
        if profile is None or not characters.profile_allowed(profile, policy):
            raise ValueError("知识角色档案不存在或不在允许名单中")
        reference["character_id"] = resolved
        return reference, {"name": profile["name"], "aliases": profile.get("aliases", []), "type": "character"}

    def _references(self, saved, references, *, allow_revoked=False):
        result = []
        for reference in references:
            timing_count("knowledge_source_reference")
            reference = SourceReference.model_validate(reference).model_dump()
            source, text = self._source_text(saved, reference["source_id"])
            if source["snapshot_id"] != reference["snapshot_id"] or reference["end"] > len(text):
                raise ValueError("资料卡原文范围或快照 ID 无效")
            if not allow_revoked and source["status"] != "active":
                raise ValueError("资料卡引用的来源已撤销")
            quote = text[reference["start"]:reference["end"]]
            if not quote.strip():
                raise ValueError("资料卡引用的原文不能为空白")
            result.append({**reference, "title": source["title"], "url": source["url"], "quote": quote,
                           "status": source["status"]})
        return result

    def save_card(self, scope, payload, card_id=None):
        data = KnowledgeCardInput.model_validate(payload).model_dump(exclude_none=True)
        with self.library.lock:
            saved = self.scope_catalog(scope)
            target, _ = self._target(saved, data["target_ref"])
            data["target_ref"] = target
            if data.get("object_ref"):
                data["object_ref"], _ = self._target(saved, data["object_ref"])
            self._references(saved, data["source_refs"])
            identity = safe_id(card_id) if card_id else uid("card")
            before = saved["knowledge_cards"].get(identity)
            if card_id and before is None:
                raise FileNotFoundError("知识资料卡不存在或属于另一作品")
            reason = data.pop("reason", "")
            row = {**data, "card_id": identity, "work_scope_id": scope,
                   "work_version_id": saved["work_version_id"], "version": before["version"] + 1 if before else 1,
                   "created_at": before["created_at"] if before else now(), "updated_at": now(),
                   "review_source": "human" if data["review_status"] == "reviewed" else None}
            # Import provenance is server-owned and survives a human review or
            # correction; it cannot be supplied through KnowledgeCardInput.
            if before and "origin" in before:
                row["origin"] = copy.deepcopy(before["origin"])
            saved["knowledge_cards"][identity] = row
            self.store._commit(saved, "knowledge_card_save", identity, before, reason)
            return self._public_card(saved, row)

    def revoke_card(self, scope, card_id, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("撤销知识资料卡须填写原因")
        with self.library.lock:
            saved = self.scope_catalog(scope)
            row = saved["knowledge_cards"].get(safe_id(card_id))
            if row is None:
                raise FileNotFoundError("知识资料卡不存在")
            before = copy.deepcopy(row)
            row.update(review_status="revoked", version=row["version"] + 1, updated_at=now())
            self.store._commit(saved, "knowledge_card_revoke", card_id, before, reason)
            return self._public_card(saved, row)

    def _public_card(self, saved, card):
        result = copy.deepcopy(card)
        result["effective_status"] = card["review_status"]
        try:
            result["sources"] = self._references(saved, card["source_refs"], allow_revoked=True)
            result["target_ref"], target = self._target(saved, card["target_ref"])
            result["target_name"] = target["name"]
            result["target_names"] = [value for value in [target["name"], *target["aliases"]] if value]
            if card.get("object_ref"):
                result["object_ref"], obj = self._target(saved, card["object_ref"])
                result["object_name"] = obj["name"]
            if any(source["status"] != "active" for source in result["sources"]):
                result["effective_status"] = "unavailable"
        except (ValueError, FileNotFoundError) as exc:
            result.update(effective_status="unavailable", error=str(exc), sources=[])
        return result

    def cards(self, scope):
        saved = self.scope_catalog(scope)
        return [self._public_card(saved, row) for row in saved["knowledge_cards"].values()]

    def alignments(self, scope):
        saved = self.scope_catalog(scope)
        rows = copy.deepcopy(list(saved["knowledge_alignments"].values()))
        for row in rows:
            versions = row.get("supporting_card_versions", {})
            if row["status"] == "reviewed" and any(
                    card_id not in saved["knowledge_cards"]
                    or versions.get(card_id) != saved["knowledge_cards"][card_id]["version"]
                    or self._public_card(saved, saved["knowledge_cards"][card_id])["effective_status"] != "reviewed"
                    for card_id in row["supporting_card_ids"]):
                row.update(status="unresolved", previous_status="reviewed", error="支持资料卡已变化，请重新核对身份映射")
        return rows

    def _active_alignments(self, saved, cards):
        active = {card["card_id"] for card in cards}
        result = []
        for alignment in saved["knowledge_alignments"].values():
            if alignment["status"] != "reviewed" or not set(alignment["supporting_card_ids"]) <= active:
                continue
            try:
                targets = [self._target(saved, target)[0] for target in alignment["target_refs"]]
            except (ValueError, FileNotFoundError):
                continue
            result.append({**copy.deepcopy(alignment), "target_refs": targets})
        return result

    def save_alignment(self, scope, upstream_key, target_refs, supporting_card_ids, status, reason):
        if not isinstance(upstream_key, str) or not upstream_key.strip() or len(upstream_key) > 4000:
            raise ValueError("上游图谱键无效")
        if status not in {"reviewed", "ambiguous", "unresolved", "revoked"} or not reason.strip():
            raise ValueError("图谱映射核对须填写有效状态和原因")
        if not isinstance(target_refs, list) or len(target_refs) > 32:
            raise ValueError("图谱映射目标无效")
        if not isinstance(supporting_card_ids, list) or not 1 <= len(supporting_card_ids) <= 64:
            raise ValueError("图谱映射须提供支持资料卡")
        with self.library.lock:
            saved = self.scope_catalog(scope)
            targets = [self._target(saved, target)[0] for target in target_refs]
            for card_id in supporting_card_ids:
                card = saved["knowledge_cards"].get(safe_id(card_id))
                if not card or self._public_card(saved, card)["effective_status"] != "reviewed":
                    raise ValueError("图谱映射的支持资料卡未经核对或已失效")
            if status == "reviewed" and len({target_key(target) for target in targets}) != 1:
                raise ValueError("已核对图谱映射须有唯一内部目标")
            identity = "alignment_" + digest(upstream_key)
            before = saved["knowledge_alignments"].get(identity)
            row = {"alignment_id": identity, "upstream_key": upstream_key, "target_refs": targets,
                   "supporting_card_ids": sorted(set(supporting_card_ids)), "status": status,
                   "supporting_card_versions": {key: saved["knowledge_cards"][key]["version"]
                                                for key in supporting_card_ids},
                   "version": before["version"] + 1 if before else 1}
            saved["knowledge_alignments"][identity] = row
            self.store._commit(saved, "knowledge_alignment", identity, before, reason)
            return copy.deepcopy(row)

    @timed("knowledge.projection")
    def projection(self, scope):
        """Snapshot only reviewed, current cards. Never include film bindings.

        Reads and byte validation share the application's mutation lock; callers
        must release it before any worker or model await, and recheck signature
        before publishing a derived generation.
        """
        with timed_lock(self.library.lock, "library_lock_wait"):
            saved = self.scope_catalog(scope)
            timing_count("knowledge_cards", len(saved["knowledge_cards"]))
            timing_count("knowledge_sources", len(saved["source_documents"]))
            cards = []
            for card in sorted(saved["knowledge_cards"].values(), key=lambda row: row["card_id"]):
                public = self._public_card(saved, card)
                if public["effective_status"] != "reviewed":
                    continue
                timing_count("knowledge_reviewed_cards")
                target_ref, target = self._target(saved, card["target_ref"])
                parts = ["作品 " + saved["title"], "版本 " + saved["work_version"],
                         "主目标 " + target["name"], "别名 " + " / ".join(filter(None, target["aliases"])),
                         "类型 " + target["type"], "定义 " + target.get("definition", ""),
                         "适用阶段 " + target.get("applicability", ""), card.get("title", ""), card["text"]]
                if card.get("object_ref"):
                    object_ref, obj = self._target(saved, card["object_ref"])
                    parts.extend(["关系 " + target["name"] + " " + card["predicate"] + " " + obj["name"],
                                  "目标别名 " + " / ".join(filter(None, obj["aliases"]))])
                else:
                    object_ref = None
                parts.append("适用阶段 " + card.get("applicability", ""))
                for field, label in (("conditions", "条件"), ("costs", "代价"), ("exceptions", "例外")):
                    parts.extend(label + " " + value for value in card.get(field, []))
                for source in public["sources"]:
                    parts.extend(["来源 " + source["title"], "原文 " + source["quote"]])
                text = "\n".join(part for part in parts if part.strip())
                with aggregate_operation("projection_sha256"):
                    projection_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                cards.append({**public, "text": text, "statement_text": card["text"], "target_ref": target_ref,
                              "object_ref": object_ref, "target_key": target_key(target_ref),
                              "projection_hash": projection_hash})
            source_state = [{key: row[key] for key in ("source_id", "snapshot_id", "sha256", "title", "url", "status")}
                            for row in sorted(saved["source_documents"].values(), key=lambda row: row["source_id"])]
            signature_cards = [{key: card.get(key) for key in (
                "card_id", "version", "projection_hash", "target_ref", "object_ref", "source_refs")} for card in cards]
            signature = digest([VERSION, scope, saved["work_version_id"], saved["work_version"], source_state,
                                signature_cards, saved["knowledge_alignments"]])
            return {"version": VERSION, "work_scope_id": scope, "work_version_id": saved["work_version_id"],
                    "work_version": saved["work_version"], "title": saved["title"], "signature": signature,
                    "cards": cards, "alignments": self._active_alignments(saved, cards)}

    def lexical_hits(self, scope, query, *, limit=50):
        from .search import normalize, tokenize
        if not isinstance(query, str) or not query.strip():
            return []
        terms = set(tokenize(query))
        hits = []
        for card in self.projection(scope)["cards"]:
            score = len(terms & set(tokenize(card["text"])))
            if normalize(query) in normalize(card["text"]):
                score += 10
            if score:
                hits.append({**card, "score": score, "retrieval_source": "lexical", "mapping_status": "reviewed",
                             "has_film_evidence": False, "evidence_type": "knowledge_source"})
        hits.sort(key=lambda row: (-row["score"], row["card_id"]))
        return hits[:limit]
