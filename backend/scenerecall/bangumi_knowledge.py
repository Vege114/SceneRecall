"""Turn an already fetched Bangumi dossier into source-backed review candidates.

This adapter performs no network or model calls. It never creates a character,
formal concept, appearance binding, or reviewed knowledge assertion.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone

from .bangumi import _NAME_KEYS, _infobox_entries
from .characters import CharacterStore, external_sources, scope_id as character_scope_id
from .knowledge_store import KnowledgeCardInput, KnowledgeStore, MAX_SOURCE_BYTES
from .library import confined_path, digest, now, read_json, uid
from .visual_identity import _write_bytes
from .work_entities import WorkEntityStore

VERSION = "bangumi-knowledge-v1"
MAX_CHARACTER_CARDS = 64
_CREDIT_FIELDS = {"声优", "聲優", "CV", "cv", "配音", "出演", "演员", "演員"}
_ROSTER_REMOVED = "角色已不在当前 Bangumi 作品名单中"


def _text(value):
    return value if isinstance(value, str) else ""


def _subject_id(asset):
    profile = asset.get("anime_profile") or {}
    value = profile.get("dossier", {}).get("subject", {}).get("id")
    return value if profile.get("mode") == "bangumi" and type(value) is int and value > 0 else None


def _timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _version_identity(asset, subject_id):
    return {"subject_id": subject_id, "season": asset.get("season"),
            "media": "episodes" if asset.get("episode") is not None else "feature"}


def _stored_version_identity(entities, saved, origin):
    if isinstance(origin.get("version_identity"), dict):
        return origin["version_identity"]
    # Upgrade an early auto-scope using its original import, whose version
    # hints survive a project move. Project IDs and folder names are not part
    # of a work edition's identity.
    from .organization import anime_metadata
    memberships = entities._memberships()
    original = next((row["asset_id"] for row in memberships.get("audit", [])
                     if row.get("work_scope_id") == saved["work_scope_id"]), None)
    manifest = read_json(entities.library.asset_dir(original) / "manifest.json") if original else None
    if not manifest:
        raise ValueError("旧自动知识库缺少可核对的原始作品版本信息")
    return _version_identity(anime_metadata(manifest, profile={"mode": "bangumi"}), origin["subject_id"])


def _scope(entities, asset, subject_id, requested):
    current = entities.scope(asset["id"])
    if requested is not None:
        if current is None or current["work_scope_id"] != requested:
            raise ValueError("请先将素材关联到指定作品知识库，不能隐式更换作品范围")
    if current is not None:
        origin = current.get("bangumi_knowledge", {})
        if origin.get("subject_id", subject_id) != subject_id:
            raise ValueError("当前知识库已有另一 Bangumi 作品的资料，不能混合导入")
        if origin.get("auto_scope"):
            version_identity = _stored_version_identity(entities, current, origin)
            if version_identity != _version_identity(asset, subject_id):
                raise ValueError("当前自动知识库属于另一动画版本，不能混合导入")
            if "version_identity" not in origin:
                origin["version_identity"] = version_identity
                entities._commit(current, "bangumi_knowledge_version_identity", str(subject_id))
        for source in current.get("source_documents", {}).values():
            provenance = source.get("origin", {})
            if provenance.get("provider") == "bangumi" and provenance.get("subject_id") != subject_id:
                raise ValueError("当前知识库已有另一 Bangumi 作品的资料，不能混合导入")
        return current

    # Names and shared character registries cannot establish work identity.
    anime_id = asset.get("anime_id")
    members = ([member for member in entities.library.list_assets()
                if member.get("anime_id") == anime_id and _subject_id(member) == subject_id]
               if anime_id else [asset])
    memberships = entities._memberships()["assets"]
    reusable = None
    for member in members:
        identity = memberships.get(member["id"])
        if not identity:
            continue
        saved = entities._read(identity)
        origin = saved.get("bangumi_knowledge", {})
        if (origin.get("auto_scope") and origin.get("subject_id") == subject_id
                and _stored_version_identity(entities, saved, origin) == _version_identity(asset, subject_id)):
            reusable = saved
            break
    if reusable is None:
        subject = asset["anime_profile"]["dossier"]["subject"]
        title = _text(subject.get("name_cn") or subject.get("name") or asset["title"])[:500]
        created = entities.create_scope(title, f"Bangumi #{subject_id} · 当前导入版本", [asset["id"]])
        reusable = entities._read(created["work_scope_id"])
        reusable["bangumi_knowledge"] = {"version": VERSION, "subject_id": subject_id,
                                         "anime_id": asset.get("anime_id"), "auto_scope": True,
                                         "version_identity": _version_identity(asset, subject_id),
                                         "documents": {}}
        entities._commit(reusable, "bangumi_knowledge_scope", str(subject_id))
    for member in members:
        if member["id"] not in entities._memberships()["assets"]:
            entities.attach(member["id"], reusable["work_scope_id"])
    return entities._read(reusable["work_scope_id"])


def _document(raw, kind, identifier):
    """Render received fields, retaining nested infobox values in the snapshot."""
    display = _text(raw.get("name_cn") or raw.get("name")) or str(identifier)
    summary = _text(raw.get("description")) if kind == "character" else _text(raw.get("summary"))
    parts = [f"# Bangumi {kind} #{identifier}: {display}\n", "\n## 名称与别名\n"]
    for key in ("name", "name_cn", "name_ja", "name_en"):
        if _text(raw.get(key)).strip():
            parts.append(f"{key}: {raw[key]}\n")
    aliases = raw.get("aliases", [])
    if isinstance(aliases, list):
        parts.extend(f"别名: {value}\n" for value in aliases if isinstance(value, str) and value.strip())
    statements = []

    def add(field, label, text, *, candidate=True):
        if not text.strip():
            return
        parts.append(f"\n## {label}\n")
        start = sum(len(part) for part in parts)
        parts.append(text)
        parts.append("\n")
        if candidate:
            # A long biography is several exact ranges, not a truncated quote.
            for offset in range(0, len(text), 10000):
                piece = text[offset:offset + 10000]
                if piece.strip():
                    statements.append({"field": field, "title": label, "text": piece,
                                       "start": start + offset, "end": start + offset + len(piece)})

    add("description" if kind == "character" else "summary", "角色简介" if kind == "character" else "作品简介",
        summary, candidate=kind == "character")
    for index, (key, label, value) in enumerate(_infobox_entries(raw.get("infobox"))):
        add(f"infobox:{index}:{key}:{label}", label, value,
            candidate=kind == "character" and key not in _NAME_KEYS and label not in _NAME_KEYS
            and key not in _CREDIT_FIELDS and label not in _CREDIT_FIELDS)
    # This is a received API field snapshot, not a downloaded page or inferred fact.
    fields = ("id", "name", "name_cn", "name_ja", "name_en", "aliases", "name_variants", "summary",
              "description", "infobox", "url", "role", "relation_type", "detail_error", "type", "date")
    received = {key: raw[key] for key in fields if key in raw}
    parts.extend(["\n## 接收到的 API 字段快照\n```json\n",
                  json.dumps(received, ensure_ascii=False, sort_keys=True, indent=2), "\n```\n"])
    return display, "".join(parts), statements[:MAX_CHARACTER_CARDS], len(statements) > MAX_CHARACTER_CARDS


def _retire(saved, entry, reason):
    source = saved["source_documents"].get(entry.get("source_id"))
    if source and source["status"] == "active":
        source.update(status="revoked", revoked_at=now(), superseded_reason=reason,
                      revoked_by_bangumi_refresh=True)
        return True
    return False


def _denied_statements(saved, entries):
    denied = set()
    for entry in entries:
        for statement in entry.get("statements", []):
            old_card = saved["knowledge_cards"].get(statement["card_id"], {})
            if old_card.get("review_status") in {"rejected", "revoked"}:
                denied.add((statement["field"], statement["statement_hash"]))
    return denied


def _candidate_cards(saved, source, statements, display, target, provenance, covered):
    new_cards, tracking = {}, []
    if target is None:
        return new_cards, tracking
    stamp = now()
    for statement in statements:
        statement_hash = digest([statement["field"], statement["text"]])
        if (statement["field"], statement_hash) in covered:
            continue
        payload = KnowledgeCardInput.model_validate({
            "title": f"{display} · {statement['title']}"[:500], "text": statement["text"],
            "kind": "fact", "target_ref": target, "source_refs": [{"source_id": source["source_id"],
                "snapshot_id": source["snapshot_id"], "start": statement["start"], "end": statement["end"]}],
            "review_status": "candidate"}).model_dump(exclude_none=True)
        payload.pop("reason", None)
        card_id = uid("card")
        new_cards[card_id] = {**payload, "card_id": card_id,
            "work_scope_id": saved["work_scope_id"], "work_version_id": saved["work_version_id"],
            "version": 1, "created_at": stamp, "updated_at": stamp, "review_source": None,
            "origin": {**provenance, "field": statement["field"], "statement_hash": statement_hash}}
        tracking.append({"card_id": card_id, "field": statement["field"], "statement_hash": statement_hash})
    return new_cards, tracking


def _fill_missing_candidates(knowledge, saved, previous, source, statements, display, target, stats):
    if not source or source["status"] != "active":
        return False  # A source withdrawal is a durable human decision.
    knowledge._source_text(saved, source["source_id"])
    covered = _denied_statements(saved, [*previous.get("history", []), previous])
    covered.update((row["field"], row["statement_hash"]) for row in previous.get("statements", []))
    for card in saved["knowledge_cards"].values():
        references = [reference for reference in card.get("source_refs", [])
                      if reference["source_id"] == source["source_id"]
                      and reference["snapshot_id"] == source["snapshot_id"]]
        if not references:
            continue
        try:
            knowledge._target(saved, card["target_ref"])
        except (ValueError, FileNotFoundError):
            stats["warnings"].append(f"资料卡 {card['card_id']} 的内部对象已失效，请编辑资料卡并重新选择内部对象")
        # Manual cards own their quoted interpretation too. Do not create an
        # automatic duplicate merely because they are absent from our ledger.
        for statement in statements:
            if any(reference["start"] == statement["start"] and reference["end"] == statement["end"]
                   for reference in references):
                covered.add((statement["field"], digest([statement["field"], statement["text"]])))
    new_cards, tracking = _candidate_cards(saved, source, statements, display, target,
                                           source.get("origin", {}), covered)
    if not new_cards:
        return False
    saved["knowledge_cards"].update(new_cards)
    previous.setdefault("card_ids", []).extend(new_cards)
    previous.setdefault("statements", []).extend(tracking)
    stats["cards_added"] += len(new_cards)
    return True


def _save_document(knowledge, saved, document_key, raw, kind, identifier, subject_id,
                   fetched_at, target, stats):
    display, text, statements, limited = _document(raw, kind, identifier)
    content = text.encode("utf-8")
    if len(content) > MAX_SOURCE_BYTES or "\x00" in text:
        raise ValueError("Bangumi 资料超过文本快照限制或包含无效字符")
    if limited:
        stats["warnings"].append(f"角色 #{identifier} 的资料卡超过单次上限，其余信息已保留在原文快照")
    content_hash = hashlib.sha256(content).hexdigest()
    documents = saved["bangumi_knowledge"]["documents"]
    previous = documents.get(document_key)
    old_source = saved["source_documents"].get(previous.get("source_id")) if previous else None
    roster_returned = (old_source and old_source.get("revoked_by_bangumi_refresh")
                       and old_source.get("superseded_reason") == _ROSTER_REMOVED)
    if previous and previous.get("content_hash") == content_hash and not roster_returned:
        # A human rejection/revocation is durable even when an import is replayed.
        stats["sources_reused"] += 1
        stats["cards_reused"] += len(previous.get("card_ids", []))
        return _fill_missing_candidates(knowledge, saved, previous, old_source, statements, display, target, stats)
    if previous and raw.get("detail_error"):
        stats["warnings"].append(f"角色 #{identifier} 详情读取失败，保留此前资料，未用空白覆盖")
        return False
    source_id = uid("source")
    relative = f"knowledge/snapshots/{content_hash}.txt"
    path = confined_path(knowledge.store._path(saved["work_scope_id"]).parent, relative)
    provenance = {"provider": "bangumi", "version": VERSION, "subject_id": subject_id,
                  "document_key": document_key, "record_kind": kind, "record_id": identifier,
                  "fetched_at": fetched_at}
    stamp = now()
    source = {"source_id": source_id, "work_scope_id": saved["work_scope_id"],
              "work_version_id": saved["work_version_id"], "title": f"Bangumi · {display}"[:1000],
              "url": f"https://bgm.tv/{'character' if kind == 'character' else 'subject'}/{identifier}",
              "filename": f"bangumi-{kind}-{identifier}.md", "sha256": content_hash,
              "content_hash": content_hash, "snapshot_id": "snapshot_" + content_hash, "path": relative,
              "status": "active", "import_status": "imported", "error": None,
              "character_count": len(text), "byte_count": len(content), "created_at": stamp,
              "origin": provenance}
    history = copy.deepcopy(previous.get("history", [])) if previous else []
    if previous:
        history.append({key: copy.deepcopy(value) for key, value in previous.items() if key != "history"})
    denied = _denied_statements(saved, [*history, *([previous] if previous else [])])
    new_cards, tracking = _candidate_cards(saved, source, statements, display, target, provenance, denied)
    card_ids = list(new_cards)
    # Validate every candidate before mutating the publication. A failed write
    # or oversized field cannot withdraw the previous good document.
    if not path.exists():
        _write_bytes(path, content)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != content_hash:
        raise ValueError("已有 Bangumi 原文快照损坏，不能覆盖")
    saved["source_documents"][source_id] = source
    saved["knowledge_cards"].update(new_cards)
    if previous:
        _retire(saved, previous, "Bangumi 来源内容已更新，请核对新资料")
    documents[document_key] = {"source_id": source_id, "content_hash": content_hash,
                               "card_ids": card_ids, "statements": tracking, "history": history,
                               "fetched_at": fetched_at}
    stats["sources_added"] += 1
    stats["cards_added"] += len(card_ids)
    return True


def sync_bangumi_knowledge(library, asset_id, *, scope_id=None):
    """Import an asset's current server-owned dossier, tolerating partial entries.

    Work selection errors are raised to the caller. Individual character failures
    are reported as warnings so knowledge enrichment cannot abort video import.
    """
    with library.lock:
        asset = library.get_asset(asset_id)
        subject_id = _subject_id(asset)
        if subject_id is None:
            raise ValueError("此素材尚未关联 Bangumi 作品档案")
        dossier = asset["anime_profile"]["dossier"]
        snapshot = dossier.get("knowledge_snapshot") or dossier
        if snapshot.get("subject", {}).get("id") != subject_id:
            raise ValueError("Bangumi 知识快照与当前作品不一致")
        entities = WorkEntityStore(library)
        saved = _scope(entities, asset, subject_id, scope_id)
        scope = saved["work_scope_id"]
        knowledge = KnowledgeStore(entities)
        saved = knowledge.scope_catalog(scope)
        before = copy.deepcopy(saved)
        origin = saved.setdefault("bangumi_knowledge", {"version": VERSION, "subject_id": subject_id,
                                                       "anime_id": asset.get("anime_id"), "auto_scope": False,
                                                       "documents": {}})
        origin.setdefault("documents", {})
        stats = {"status": "completed", "work_scope_id": scope, "subject_id": subject_id,
                 "sources_added": 0, "cards_added": 0, "sources_reused": 0, "cards_reused": 0,
                 "warnings": list(snapshot.get("warnings", dossier.get("warnings", [])))}
        fetched_at = snapshot.get("fetched_at", dossier.get("fetched_at", ""))
        previous_time, incoming_time = _timestamp(origin.get("latest_snapshot_at")), _timestamp(fetched_at)
        if previous_time is not None and (incoming_time is None or incoming_time < previous_time):
            # An older episode, import retry or moved asset must not replace a
            # newly fetched roster or restore superseded source assertions.
            stats["warnings"].append("此素材携带较旧或无有效日期的 Bangumi 快照，保留知识库中较新的资料")
            stats.update(status="partial", created_sources=0, created_cards=0,
                         sources_reused=len(origin["documents"]),
                         cards_reused=sum(len(row.get("card_ids", [])) for row in origin["documents"].values()),
                         source_count=len(saved["source_documents"]), card_count=len(saved["knowledge_cards"]))
            return stats
        try:
            _save_document(knowledge, saved, "subject", snapshot["subject"], "subject", subject_id,
                           subject_id, fetched_at, None, stats)
        except (ValueError, TypeError, KeyError) as exc:
            stats["warnings"].append(f"作品资料未导入：{exc}")
        # Read the current authoritative registry without appearance scans or
        # new cast creation; imported external identities were already seeded.
        characters = CharacterStore(library)
        registry = characters._read(character_scope_id(asset))
        policy = characters.cast_policy(registry, asset.get("anime_profile"))
        targets = {}
        for profile in registry["profiles"].values():
            if not characters.profile_allowed(profile, policy):
                continue
            for external in external_sources(profile):
                if external.get("source") == "bangumi" and external.get("subject_id") == subject_id:
                    targets.setdefault(external.get("character_id"), []).append(profile["id"])
        seen = {"subject"}
        for character in snapshot.get("characters", []):
            identifier = character.get("id") if isinstance(character, dict) else None
            if type(identifier) is not int or identifier <= 0:
                stats["warnings"].append("跳过缺少有效 Bangumi ID 的角色资料")
                continue
            key = f"character:{identifier}"
            seen.add(key)
            matches = sorted(set(targets.get(identifier, [])))
            target = None
            if len(matches) == 1:
                target = {"kind": "character", "asset_id": asset_id,
                          "character_scope_id": registry["scope_id"], "character_id": matches[0]}
                try:
                    target = knowledge._target(saved, target)[0]
                except (ValueError, FileNotFoundError):
                    target = None
            if target is None:
                stats["warnings"].append(f"角色 #{identifier} 无唯一有效内部身份，仅保存来源，不猜测名称关联")
            try:
                _save_document(knowledge, saved, key, character, "character", identifier,
                               subject_id, fetched_at, target, stats)
            except (ValueError, TypeError, KeyError) as exc:
                stats["warnings"].append(f"角色 #{identifier} 资料未导入：{exc}")
        for key, entry in origin["documents"].items():
            if key not in seen:
                _retire(saved, entry, _ROSTER_REMOVED)
        if incoming_time is not None:
            origin["latest_snapshot_at"] = fetched_at
        if saved != before:
            entities._commit(saved, "bangumi_knowledge_sync", str(subject_id),
                             reason="保存 Bangumi 来源与待核对资料，不确认设定或影片出场")
        stats.update(created_sources=stats["sources_added"], created_cards=stats["cards_added"],
                     source_count=len(saved["source_documents"]), card_count=len(saved["knowledge_cards"]))
        if stats["warnings"]:
            stats["status"] = "partial"
        return stats
