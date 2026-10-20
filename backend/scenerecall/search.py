"""Disposable hybrid indexes over local evidence records, never over source videos."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jieba

from .providers import ProviderError, ProviderManager

jieba.setLogLevel(logging.WARNING)
ALLOWED_FILTERS = {"asset_id", "work_id", "kind", "season", "episode", "character", "character_id",
                   "concept_id", "project_id", "anime_id", "folder_id", "tag"}
STOP_WORDS = {
    "的", "了", "在", "是", "有", "和", "与", "及", "或", "吗", "呢", "啊", "吧", "着", "地", "得",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "那个", "这个", "这些", "那些", "哪个",
    "什么", "哪里", "有没有", "是否", "请", "帮我", "一下", "查找", "搜索", "找到", "寻找",
    "片段", "场景", "画面", "镜头", "视频", "电影", "一段", "一个", "中", "里", "中有", "中的",
    "the", "a", "an", "is", "are", "of", "in", "and", "or", "to", "find", "scene", "movie",
}


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = [word.strip() for word in jieba.cut_for_search(normalized) if normalize(word)]
    # Character bigrams improve short names and phrases unknown to jieba, consistently
    # applied at indexing and query time. Do not strip spaces between Latin words.
    for run in re.findall(r"[\u3400-\u9fff]+", normalized):
        words.extend(run[i:i + 2] for i in range(len(run) - 1))
    return [word for word in dict.fromkeys(words) if word not in STOP_WORDS]


def _search_text(record: dict) -> str:
    pieces = [record.get("text", "")]
    # User-confirmed names and aliases are searchable; no entity linking is inferred.
    for entity in record.get("entities", []) or []:
        pieces.extend(str(entity[key]) for key in ("name", "character_name", "description") if entity.get(key))
        pieces.extend(entity.get("aliases", []) or [])
        # Candidate labels help find evidence, but never enter the confirmed
        # character filter or establish visible identity.
        if entity.get("candidate_character_id") and not entity.get("character_id"):
            pieces.append(entity.get("candidate_character_name", ""))
            pieces.extend(entity.get("candidate_character_aliases", []) or [])
    for binding in record.get("concept_bindings", []) or []:
        if binding.get("status") == "confirmed":
            pieces.append(binding.get("concept_name", ""))
            pieces.extend(binding.get("aliases", []) or [])
    pieces.extend(record.get("aliases", []) or [])
    return " ".join(str(part) for part in pieces if part)


def unverified_character_names(query: str, record: dict) -> list[str]:
    """Keep an explicitly queried, candidate-only identity from becoming fact."""
    question = normalize(query)
    confirmed, candidates = set(), []
    for entity in record.get("entities", []) or []:
        if entity.get("character_id"):
            confirmed.update(normalize(str(value)) for value in (
                entity.get("character_name"), entity.get("name"), *(entity.get("aliases") or [])) if value)
        elif entity.get("candidate_character_id"):
            candidates.extend(value for value in (
                entity.get("candidate_character_name"), *(entity.get("candidate_character_aliases") or [])) if value)
    return list(dict.fromkeys(str(name) for name in candidates
                             if normalize(str(name)) and normalize(str(name)) in question
                             and normalize(str(name)) not in confirmed))


def _text_hash(record: dict) -> str:
    return hashlib.sha256(_search_text(record).encode("utf-8")).hexdigest()


def _context_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("kind") == "subtitle":
            groups[(record["asset_id"], record.get("language", "und"))].append(record)
    result = []
    for group in groups.values():
        group.sort(key=lambda row: (row["start_ms"], row["end_ms"]))
        for start in range(len(group) - 1):
            for length in (2, 3):
                rows = group[start:start + length]
                if len(rows) < length or any(rows[i + 1]["start_ms"] - rows[i]["end_ms"] > 2000
                                             for i in range(len(rows) - 1)):
                    continue
                ids = [row["id"] for row in rows]
                context = {**rows[0], "id": "context_" + hashlib.sha256("|".join(ids).encode()).hexdigest()[:24],
                           "source_ids": ids, "start_ms": min(row["start_ms"] for row in rows),
                           "end_ms": max(row["end_ms"] for row in rows),
                           "text": "\n".join(row["text"] for row in rows), "is_context": True,
                           "evidence_frame_ids": list(dict.fromkeys(
                               frame for row in rows for frame in row.get("evidence_frame_ids", [])))}
                context["subtitle_text"] = context["text"]
                result.append(context)
    return result


def _visual_merge_keys(record: dict) -> set[tuple]:
    if (record.get("kind") != "visual" or record.get("evidence_type") != "retrieval_window"
            or not record.get("asset_id") or not record.get("source_fingerprint")
            or len(record.get("member_ids", [])) > 1):
        return set()
    keys = set()
    frame_id = record.get("hit_frame_id")
    if frame_id and frame_id in record.get("frame_ids", []):
        keys.add(("frame", frame_id))
    stable = {}
    for entity in record.get("entities", []):
        identity = entity.get("character_id")
        status = entity.get("identity_status", entity.get("association_status"))
        if identity and status in {None, "confirmed", "auto_bound", "manual_confirmed"}:
            stable.setdefault(entity["id"], set()).add(("character", identity))
    for binding in record.get("concept_bindings", []):
        if binding.get("status") == "confirmed" and binding.get("entity_id") and binding.get("concept_id"):
            stable.setdefault(binding["entity_id"], set()).add(("concept", binding["concept_id"]))
    for event in record.get("events", []):
        action = unicodedata.normalize("NFKC", str(event.get("action") or "")).strip().casefold()
        polarity = event.get("polarity", "positive")
        if not action or polarity not in {"positive", "negative"} or event.get("uncertainty"):
            continue
        roles = []
        has_participants = False
        for field in ("actor_ids", "target_ids", "subject_id", "object_id", "recipient_id"):
            ids = event.get(field) or []
            ids = [ids] if isinstance(ids, str) else ids
            if not isinstance(ids, list) or any(len(stable.get(identity, ())) != 1 for identity in ids):
                break
            has_participants |= bool(ids)
            roles.append((field, tuple(sorted(next(iter(stable[identity])) for identity in ids))))
        else:
            if has_participants:
                keys.add(("event", action, polarity, bool(event.get("temporal")), tuple(roles)))
    return keys


def merge_visual_candidates(candidates: list[dict]) -> list[dict]:
    """Collapse bounded presentation duplicates without synthesizing events.

    Each group retains the highest-ranked representative and all original source
    spans. Adjacent pictures alone never establish an action or identity. The
    common key must hold across every member, so a chain cannot join unrelated
    events through a middle record with several different events.
    """
    groups = []
    for candidate in candidates:
        keys = _visual_merge_keys(candidate)
        selected = None
        for group in groups:
            representative = group["rows"][0]
            if (not keys or not group["keys"].intersection(keys)
                    or any(candidate.get(field) != representative.get(field) for field in
                           ("asset_id", "source_fingerprint", "work_id", "work_scope_id"))):
                continue
            start = min(group["start"], candidate["start_ms"])
            end = max(group["end"], candidate["end_ms"])
            gap = max(candidate["start_ms"] - group["end"], group["start"] - candidate["end_ms"], 0)
            if gap > 1000 or end - start > 12000:
                continue
            # A frame ID must continue to mean precisely the same source PTS.
            frames = {frame["id"]: frame for row in group["rows"] for frame in row.get("ordered_frames", [])}
            if any(frame["id"] in frames and frames[frame["id"]] != frame for frame in candidate.get("ordered_frames", [])):
                continue
            selected = group
            break
        if selected is None:
            groups.append({"rows": [candidate], "keys": keys,
                           "start": candidate["start_ms"], "end": candidate["end_ms"]})
        else:
            selected["rows"].append(candidate)
            selected["keys"].intersection_update(keys)
            selected["start"], selected["end"] = start, end
    result = []
    for group in groups:
        rows = group["rows"]
        if len(rows) == 1:
            result.append(rows[0])
            continue
        temporal = sorted(rows, key=lambda row: (row["start_ms"], row["end_ms"], row["id"]))
        frames = {frame["id"]: frame for row in temporal for frame in row.get("ordered_frames", [])}
        ordered_frames = sorted(frames.values(), key=lambda frame: (frame["at_ms"], frame["id"]))
        frame_order = {frame["id"]: index for index, frame in enumerate(ordered_frames)}
        merged = {**rows[0], "start_ms": group["start"], "end_ms": group["end"],
                  "member_ids": [row.get("record_id") or row["id"] for row in temporal],
                  "source_spans": [{"record_id": row.get("record_id") or row["id"],
                                    "start_ms": row["start_ms"], "end_ms": row["end_ms"]} for row in temporal],
                  "ordered_frames": [dict(frame) for frame in ordered_frames],
                  "merge_basis": ("same_hit_frame" if any(key[0] == "frame" for key in group["keys"])
                                  else "same_stable_event")}
        for field in ("frame_ids", "evidence_frame_ids", "evidence_ids"):
            values = list(dict.fromkeys(value for row in temporal for value in row.get(field, [])))
            merged[field] = sorted(values, key=lambda value: frame_order.get(value, len(frame_order)))
        merged["retrieval_branches"] = list(dict.fromkeys(
            branch for row in rows for branch in row.get("retrieval_branches", [])))
        # Keep the representative's local entities/events untouched: their IDs
        # are local to that observation, and grouping is no new continuous event.
        result.append(merged)
    return result


class SearchEngine:
    def __init__(self, index_dir: Path, providers: ProviderManager):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.index_dir / "search.sqlite"
        self.providers = providers
        self._lock = threading.RLock()
        self._generation = 0
        self.visual_encoder = None
        self.concept_lookup = None
        self._initialize()
        from .visual_index import VisualIndex
        self.visual_index = VisualIndex(self)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def _initialize(self) -> None:
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, work_id TEXT,
                    kind TEXT NOT NULL, season INTEGER, episode INTEGER,
                    normalized TEXT NOT NULL, characters TEXT NOT NULL,
                    text_hash TEXT NOT NULL, data TEXT NOT NULL, is_context INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS records_asset ON records(asset_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(id UNINDEXED, tokens);
                CREATE TABLE IF NOT EXISTS namespaces (
                    signature TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
                    model TEXT NOT NULL, table_name TEXT NOT NULL, dimensions INTEGER NOT NULL,
                    record_count INTEGER NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vector_cache (
                    signature TEXT NOT NULL, text_hash TEXT NOT NULL, vector TEXT NOT NULL,
                    PRIMARY KEY(signature,text_hash)
                );
                CREATE TABLE IF NOT EXISTS build_checkpoints (
                    kind TEXT PRIMARY KEY, records_digest TEXT NOT NULL
                );
            """)
            self._migrate_fts_rowids(db)

    @staticmethod
    def _migrate_fts_rowids(db: sqlite3.Connection) -> None:
        """Align legacy FTS rows with the indexed records.id -> records.rowid map.

        Existing tokens and vector manifests remain usable. The rename and version
        marker commit together, so an interrupted upgrade leaves the old index intact.
        """
        db.execute("BEGIN IMMEDIATE")
        if db.execute("PRAGMA user_version").fetchone()[0] >= 1:
            return
        db.execute("CREATE VIRTUAL TABLE evidence_fts_rowids USING fts5(id UNINDEXED, tokens)")
        db.execute("""INSERT OR REPLACE INTO evidence_fts_rowids(rowid,id,tokens)
                      SELECT r.rowid,r.id,f.tokens FROM evidence_fts f JOIN records r ON r.id=f.id
                      ORDER BY f.rowid""")
        # Normally every record already has a token row. Repair a missing legacy
        # row without re-tokenizing the rest of a potentially large local index.
        missing = db.execute("""SELECT r.rowid,r.data FROM records r
                                LEFT JOIN evidence_fts_rowids f ON f.rowid=r.rowid
                                WHERE f.rowid IS NULL""").fetchall()
        for rowid, data in missing:
            record = json.loads(data)
            db.execute("INSERT INTO evidence_fts_rowids(rowid,id,tokens) VALUES (?,?,?)",
                       (rowid, record["id"], " ".join(tokenize(_search_text(record)))))
        db.execute("DROP TABLE evidence_fts")
        db.execute("ALTER TABLE evidence_fts_rowids RENAME TO evidence_fts")
        db.execute("PRAGMA user_version=1")

    @staticmethod
    def _validate_records(records: list[dict]) -> list[dict]:
        seen = set()
        for record in records:
            if not isinstance(record.get("id"), str) or not record["id"] or record["id"] in seen:
                raise ValueError("搜索记录 ID 缺失或重复")
            if not record.get("asset_id") or record.get("kind") not in {"visual", "subtitle", "scene", "audio"}:
                raise ValueError("搜索记录缺少媒体或证据类型")
            if not isinstance(record.get("text"), str):
                raise ValueError("搜索记录正文无效")  # noqa: TRY004 - API exposes validation errors consistently
            if (type(record.get("start_ms")) is not int or type(record.get("end_ms")) is not int
                    or not 0 <= record["start_ms"] < record["end_ms"]):
                raise ValueError("搜索记录时间范围无效")
            seen.add(record["id"])
        return records

    @staticmethod
    def _write_record(db: sqlite3.Connection, record: dict, *, replace_existing: bool = True) -> None:
        text = _search_text(record)
        names = []
        for entity in record.get("entities", []) or []:
            names.extend(str(entity[key]) for key in ("name", "character_name") if entity.get(key))
            names.extend(entity.get("aliases", []) or [])
        names.extend(record.get("aliases", []) or [])
        # core may explicitly project an annotation's confirmed character name
        if record.get("character_name"):
            names.append(record["character_name"])
        previous = (db.execute("SELECT rowid FROM records WHERE id=?", (record["id"],)).fetchone()
                    if replace_existing else None)
        if previous:
            db.execute("DELETE FROM evidence_fts WHERE rowid=?", (previous[0],))
        written = db.execute("""INSERT INTO records
                   (id,asset_id,work_id,kind,season,episode,normalized,characters,text_hash,data,is_context)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     asset_id=excluded.asset_id,work_id=excluded.work_id,kind=excluded.kind,
                     season=excluded.season,episode=excluded.episode,normalized=excluded.normalized,
                     characters=excluded.characters,text_hash=excluded.text_hash,data=excluded.data,
                     is_context=excluded.is_context""", (
            record["id"], record["asset_id"], record.get("work_id"), record["kind"],
            record.get("season"), record.get("episode"), normalize(text),
            json.dumps([normalize(name) for name in names], ensure_ascii=False), _text_hash(record),
            json.dumps(record, ensure_ascii=False, allow_nan=False), int(record.get("is_context", False)),
        ))
        rowid = previous[0] if previous else written.lastrowid
        db.execute("INSERT INTO evidence_fts(rowid,id,tokens) VALUES (?,?,?)",
                   (rowid, record["id"], " ".join(tokenize(text))))

    @staticmethod
    def _delete_record(db: sqlite3.Connection, identity: str) -> None:
        db.execute("DELETE FROM evidence_fts WHERE rowid=(SELECT rowid FROM records WHERE id=?)", (identity,))
        db.execute("DELETE FROM records WHERE id=?", (identity,))

    def _sync_records(self, db: sqlite3.Connection, records: list[dict], owner: str, *, retire: bool) -> dict:
        previous = {row["id"]: json.loads(row["data"]) for row in db.execute(
            "SELECT id,data FROM records WHERE asset_id=? AND is_context=0", (owner,))}
        incoming = {record["id"]: record for record in records}
        if not retire:
            incoming = {**previous, **incoming}
        removed = previous.keys() - incoming.keys()
        for identity in removed:
            self._delete_record(db, identity)
        changed_count = context_count = 0
        for identity, record in incoming.items():
            if previous.get(identity) != record:
                self._write_record(db, record)
                changed_count += 1
        old_subtitles = {identity: row for identity, row in previous.items() if row["kind"] == "subtitle"}
        new_subtitles = {identity: row for identity, row in incoming.items() if row["kind"] == "subtitle"}
        if old_subtitles != new_subtitles:
            old_contexts = {row["id"]: json.loads(row["data"]) for row in db.execute(
                "SELECT id,data FROM records WHERE asset_id=? AND is_context=1", (owner,))}
            new_contexts = {row["id"]: row for row in _context_records(list(new_subtitles.values()))}
            for identity in old_contexts.keys() - new_contexts.keys():
                self._delete_record(db, identity)
                context_count += 1
            for identity, row in new_contexts.items():
                if old_contexts.get(identity) != row:
                    self._write_record(db, row)
                    context_count += 1
        return {"upserted": changed_count, "removed": len(removed), "contexts_changed": context_count}

    def upsert(self, records: list[dict]) -> None:
        self._validate_records(records)
        assets = defaultdict(list)
        for record in records:
            assets[record["asset_id"]].append(record)
        with self._lock, self._db() as db:
            changed = False
            for owner, rows in assets.items():
                changed |= any(self._sync_records(db, rows, owner, retire=False).values())
            if changed:
                db.execute("DELETE FROM build_checkpoints")
                self._generation += 1

    def remove_assets(self, asset_ids):
        """Remove evidence immediately and invalidate any in-flight publication."""
        asset_ids = list(asset_ids)
        with self._lock, self._db() as db:
            self.visual_index.remove_assets(db, asset_ids)
            for asset_id in asset_ids:
                db.execute("DELETE FROM evidence_fts WHERE rowid IN (SELECT rowid FROM records WHERE asset_id=?)", (asset_id,))
                db.execute("DELETE FROM records WHERE asset_id=?", (asset_id,))
            db.execute("DELETE FROM build_checkpoints")
            self._generation += 1

    def sync_asset(self, records: list[dict], asset_id: str | None = None) -> dict:
        """Publish one asset projection, touching only changed rows and contexts.

        Unlike upsert this removes records retired by resegmentation. The optional
        asset ID permits publishing an empty projection without guessing ownership.
        """
        self._validate_records(records)
        assets = {record["asset_id"] for record in records}
        if asset_id is not None:
            assets.add(asset_id)
        if len(assets) != 1:
            raise ValueError("增量同步必须指定单个视频资产")
        owner = next(iter(assets))
        with self._lock, self._db() as db:
            result = self._sync_records(db, records, owner, retire=True)
            if any(result.values()):
                db.execute("DELETE FROM build_checkpoints")
                self._generation += 1
        return result

    def _signature(self, profile_id: str) -> str:
        profile = self.providers.get(profile_id)
        # Storage mode, keys and prices do not affect the mathematical vector space.
        return hashlib.sha256(json.dumps({key: profile[key] for key in ("base_url", "model")},
                                         sort_keys=True).encode()).hexdigest()[:32]

    @staticmethod
    async def _callback(callback: Callable | None, value: Any) -> None:
        if callback:
            result = callback(value)
            if inspect.isawaitable(result):
                await result

    async def rebuild_from_source(self, source: Callable[[], list[dict]],
                                  embedding_profile_id: str | None = None, **kwargs) -> dict:
        """Fence the source snapshot as well as publication against concurrent edits."""
        with self._lock:
            generation = self._generation
        records = await asyncio.to_thread(source)
        return await self.rebuild(records, embedding_profile_id, expected_generation=generation, **kwargs)

    async def rebuild(self, records: list[dict], embedding_profile_id: str | None = None,
                      *, before_batch: Callable | None = None, on_batch: Callable | None = None,
                      embed_batch: Callable | None = None, expected_generation: int | None = None) -> dict:
        with self._lock:
            generation = self._generation if expected_generation is None else expected_generation
            self._check_generation(generation)
        self._validate_records(records)
        all_records = records + _context_records(records)
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        count, cost, cost_known, cache_hits = 0, 0.0, True, 0
        namespace = None
        records_digest = self._records_digest(all_records)
        signature = self._signature(embedding_profile_id) if embedding_profile_id else None
        with self._lock, self._db() as db:
            self._check_generation(generation)
            completed = {row[0]: row[1] for row in db.execute("SELECT kind,records_digest FROM build_checkpoints")}
            lexical_ready = (completed.get("lexical") == records_digest
                and db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == len(all_records)
                and db.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0] == len(all_records))
            vector_ready = bool(signature and completed.get("vector:" + signature) == records_digest
                and db.execute("SELECT 1 FROM namespaces WHERE signature=?", (signature,)).fetchone())
            if lexical_ready and (not signature or vector_ready):
                return {"records": len(records), "contexts": len(all_records) - len(records),
                        "embedding_profile_id": embedding_profile_id,
                        "cache_hits": len({_text_hash(record) for record in all_records}) if signature else 0,
                        "request_count": 0, "usage": usage, "cost": 0.0, "resumed": True}
        if embedding_profile_id and all_records:
            import lancedb
            profile = self.providers.get(embedding_profile_id)
            vector_by_hash = {}
            with self._db() as db:
                for row in db.execute("SELECT text_hash,vector FROM vector_cache WHERE signature=?", (signature,)):
                    vector_by_hash[row[0]] = json.loads(row[1])
            unique = {_text_hash(record): _search_text(record) for record in all_records}
            missing = [(key, text) for key, text in unique.items() if key not in vector_by_hash]
            cache_hits = len(unique) - len(missing)
            for offset in range(0, len(missing), 64):
                batch = missing[offset:offset + 64]
                await self._callback(before_batch, len(batch))
                try:
                    texts = [text for _, text in batch]
                    result = (await embed_batch(texts) if embed_batch else
                              await self.providers.embed(embedding_profile_id, texts))
                except ProviderError as exc:
                    await self._callback(on_batch, exc)
                    raise
                await self._callback(on_batch, result)
                count += result.request_count
                for key in usage:
                    usage[key] += result.usage.get(key, 0)
                if result.cost is None:
                    cost_known = False
                else:
                    cost += result.cost
                if len(result.data) != len(batch):
                    raise ValueError("向量数量与记录不一致")
                with self._db() as db:
                    for (text_hash, _), vector in zip(batch, result.data):
                        vector_by_hash[text_hash] = vector
                        db.execute("INSERT OR REPLACE INTO vector_cache VALUES (?,?,?)",
                                   (signature, text_hash, json.dumps(vector)))
            dimensions = {len(vector_by_hash[key]) for key in unique}
            if len(dimensions) != 1:
                raise ValueError("同一模型的向量维度发生变化；请使用新模型名称创建连接")
            dimension = dimensions.pop()
            name = "vectors_" + signature + "_" + uuid.uuid4().hex[:12]
            data = [{"id": record["id"], "asset_id": record["asset_id"],
                     "text_hash": _text_hash(record), "vector": vector_by_hash[_text_hash(record)]}
                    for record in all_records]
            # A fresh table is published only after all embeddings succeed. An interrupted
            # build never overwrites the table currently referenced by the SQLite manifest.
            lance = lancedb.connect(str(self.index_dir / "vectors"))
            await asyncio.to_thread(lance.create_table, name, data=data)
            namespace = (signature, embedding_profile_id, profile["model"], name, dimension,
                         len(all_records), time.time())
        await asyncio.to_thread(self._publish, all_records, namespace, generation)
        return {"records": len(records), "contexts": len(all_records) - len(records),
                "embedding_profile_id": embedding_profile_id, "cache_hits": cache_hits,
                "request_count": count, "usage": usage, "cost": cost if cost_known else None}

    @staticmethod
    def _records_digest(records: list[dict]) -> str:
        return hashlib.sha256(json.dumps(sorted(records, key=lambda row: row["id"]),
            ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _check_generation(self, generation: int) -> None:
        if self._generation != generation:
            raise ValueError("索引构建期间资料发生变化，请重试重建；原索引已保留")

    def _publish(self, records: list[dict], namespace: tuple | None, generation: int) -> None:
        with self._lock, self._db() as db:
            self._check_generation(generation)
            records_digest = self._records_digest(records)
            checkpoint = db.execute("SELECT records_digest FROM build_checkpoints WHERE kind='lexical'").fetchone()
            if (not checkpoint or checkpoint[0] != records_digest
                    or db.execute("SELECT COUNT(*) FROM records").fetchone()[0] != len(records)
                    or db.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0] != len(records)):
                db.execute("DELETE FROM records")
                db.execute("DELETE FROM evidence_fts")
                for record in records:
                    # Both tables are empty, so inserts need no prior-row lookup.
                    self._write_record(db, record, replace_existing=False)
                db.execute("DELETE FROM build_checkpoints")
                db.execute("INSERT INTO build_checkpoints VALUES ('lexical',?)", (records_digest,))
            if namespace:
                db.execute("INSERT OR REPLACE INTO namespaces VALUES (?,?,?,?,?,?,?)", namespace)
                db.execute("INSERT OR REPLACE INTO build_checkpoints VALUES (?,?)",
                           ("vector:" + namespace[0], records_digest))
            self._generation += 1

    def status(self) -> dict:
        with self._db() as db:
            records = db.execute("SELECT COUNT(*) FROM records WHERE is_context=0").fetchone()[0]
            contexts = db.execute("SELECT COUNT(*) FROM records WHERE is_context=1").fetchone()[0]
            namespaces = [dict(row) for row in db.execute("SELECT * FROM namespaces ORDER BY updated_at DESC")]
        return {"records": records, "contexts": contexts, "embedding_namespaces": namespaces,
                "visual_namespaces": self.visual_index.status(), "lexical_ready": True}

    async def index_visual(self, records: list[dict], frames: list[dict], encoder=None, progress=None) -> dict:
        return await self.visual_index.build(records, frames, encoder or self.visual_encoder, progress)

    async def match_visual_references(self, records: list[dict], references: list[dict], encoder=None) -> dict:
        return await self.visual_index.match_references(records, references, encoder or self.visual_encoder)

    async def _visual(self, query: str, filters: dict, degraded: list[str]) -> list[dict]:
        if self.visual_encoder is None:
            return []
        return await self.visual_index.search(query, filters, self.visual_encoder, degraded)

    def filtered_records(self, filters: dict | None = None) -> list[dict]:
        """Canonical current index rows, filtered before concept lookup."""
        where, params = self._where(filters or {})
        with self._db() as db:
            return [json.loads(row["data"]) for row in db.execute(
                f"SELECT r.data FROM records r WHERE {where} AND r.is_context=0", params)]

    async def _concept(self, query: str, filters: dict) -> list[dict]:
        from .visual_index import evidence_digest
        candidates = await asyncio.to_thread(self.concept_lookup, query, filters)
        if not isinstance(candidates, list):
            raise ValueError("作品概念召回须返回记录列表")
        # The callback receives original constraints; its output is fenced again
        # against deletion, stale source evidence and the authoritative SQL scope.
        current = {row["id"]: row for row in await asyncio.to_thread(self.filtered_records, filters)}
        selected, seen = [], set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("作品概念召回包含无效记录")
            row = current.get(candidate.get("id"))
            if row is None or row["id"] in seen or evidence_digest(row) != evidence_digest(candidate):
                continue
            if (candidate.get("match_basis") != "setting_concept_appearance"
                    or not candidate.get("concept_bindings")
                    or any(binding.get("status") != "confirmed" for binding in candidate["concept_bindings"])):
                continue
            selected.append({**row, **{field: candidate[field] for field in (
                "concept_bindings", "matched_concept_ids", "evidence_ids") if field in candidate},
                "match_basis": "setting_concept_appearance", "event_verified": False, "retrieval_branches": ["concept"]})
            seen.add(row["id"])
            if len(selected) == 50:
                break
        return selected

    @staticmethod
    def _where(filters: dict) -> tuple[str, list]:
        clauses, params = [], []
        for key, value in filters.items():
            if key not in ALLOWED_FILTERS or value is None or value == "":
                continue
            if key == "character":
                clauses.append("EXISTS (SELECT 1 FROM json_each(r.characters) WHERE value=?)")
                params.append(normalize(str(value)))
            elif key == "character_id":
                clauses.append("""(json_extract(r.data, '$.character_id')=? OR EXISTS
                    (SELECT 1 FROM json_each(r.data, '$.entities')
                     WHERE json_extract(value, '$.character_id')=?))""")
                params.extend([value, value])
            elif key == "concept_id":
                clauses.append("""EXISTS (SELECT 1 FROM json_each(r.data, '$.concept_bindings')
                    WHERE json_extract(value, '$.concept_id')=? AND json_extract(value, '$.status')='confirmed')""")
                params.append(value)
            elif key in {"project_id", "anime_id"}:
                clauses.append(f"json_extract(r.data, '$.{key}')=?")
                params.append(value)
            elif key in {"folder_id", "tag"}:
                field = "folder_ids" if key == "folder_id" else "tags"
                clauses.append(f"EXISTS (SELECT 1 FROM json_each(r.data, '$.{field}') WHERE value=?)")
                params.append(value)
            else:
                clauses.append(f"r.{key}=?")
                params.append(value)
        return (" AND ".join(clauses) or "1=1"), params

    def _lexical(self, query: str, filters: dict, limit: int = 50) -> list[dict]:
        where, params = self._where(filters)
        normalized = normalize(query)
        if not normalized:
            return []
        with self._db() as db:
            exact = db.execute(f"SELECT r.data FROM records r WHERE {where} AND instr(r.normalized,?)>0 "
                               "ORDER BY r.is_context ASC, length(r.normalized) ASC LIMIT ?",
                               [*params, normalized, limit]).fetchall()
            candidates = {json.loads(row[0])["id"]: json.loads(row[0]) for row in exact}
            tokens = tokenize(query)
            # Tokens quoted as literal FTS terms; no user-controlled FTS query language.
            fts_query = " OR ".join('"' + term.replace('"', '""') + '"' for term in tokens[:100])
            if fts_query:
                rows = db.execute(f"""SELECT r.data FROM evidence_fts JOIN records r ON r.rowid=evidence_fts.rowid
                                    WHERE evidence_fts MATCH ? AND {where}
                                    ORDER BY bm25(evidence_fts) LIMIT ?""", [fts_query, *params, limit]).fetchall()
                for row in rows:
                    record = json.loads(row[0])
                    candidates.setdefault(record["id"], record)
        return list(candidates.values())[:limit]

    async def _semantic(self, query: str, filters: dict, profile_id: str,
                        degraded: list[str]) -> list[dict]:
        import lancedb
        signature = self._signature(profile_id)
        with self._db() as db:
            namespace = db.execute("SELECT * FROM namespaces WHERE signature=?", (signature,)).fetchone()
            if not namespace:
                degraded.append("当前向量模型尚未建立索引；本次使用字面检索")
                return []
            where, params = self._where(filters)
            valid = {row["id"]: (json.loads(row["data"]), row["text_hash"]) for row in db.execute(
                f"SELECT r.id,r.data,r.text_hash FROM records r WHERE {where}", params)}
        if not valid:
            return []
        result = await self.providers.embed(profile_id, [query])
        if len(result.data[0]) != namespace["dimensions"]:
            degraded.append("查询向量维度与索引不一致，请重建当前模型索引")
            return []
        table = lancedb.connect(str(self.index_dir / "vectors")).open_table(namespace["table_name"])
        # Pre-filter IDs so a small filtered work isn't excluded by a global nearest-50.
        id_filter = "id IN (" + ",".join("'" + value.replace("'", "''") + "'" for value in valid) + ")"
        search = table.search(result.data[0]).distance_type("cosine")
        if filters:
            search = search.where(id_filter, prefilter=True)
        rows = await asyncio.to_thread(search.limit(50).to_list)
        records, stale = [], False
        for row in rows:
            current = valid.get(row["id"])
            if not current:
                continue
            if current[1] != row["text_hash"]:
                stale = True
                continue
            # Similarity alone never establishes a complete match. A conservative
            # candidate floor also prevents arbitrary nearest neighbors on empty queries.
            if row["_distance"] <= 0.45:
                records.append(current[0])
        if stale:
            degraded.append("部分资料已更新，旧向量已跳过；请重建向量索引")
        return records

    async def search(self, query: str, filters: dict | None = None, embedding_profile_id: str | None = None,
                     query_profile_id: str | None = None, decision_profile_id: str | None = None,
                     answer_profile_id: str | None = None, limit: int = 10) -> dict:
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("检索问题长度须为 1–2000 个字符")
        limit = max(1, min(int(limit), 50))
        filters = {key: value for key, value in (filters or {}).items() if key in ALLOWED_FILTERS}
        degraded = []
        search_text = query
        if query_profile_id:
            try:
                interpreted = await self.providers.interpret(query_profile_id, query)
                search_text = interpreted.data["search_text"]
                filters = {**interpreted.data.get("filters", {}), **filters}
            except ProviderError as exc:
                degraded.append("查询理解失败，已保留原始问题：" + str(exc))
        lexical = await asyncio.to_thread(self._lexical, query, filters)
        if search_text != query:
            seen = {row["id"] for row in lexical}
            expanded = await asyncio.to_thread(self._lexical, search_text, filters)
            lexical.extend(row for row in expanded if row["id"] not in seen)
            lexical = lexical[:50]
        semantic = []
        if embedding_profile_id:
            try:
                semantic = await self._semantic(search_text, filters, embedding_profile_id, degraded)
            except (ProviderError, OSError, ValueError, RuntimeError) as exc:
                message = str(exc) if isinstance(exc, ProviderError) else "索引读取失败，请重建索引"
                degraded.append("语义检索不可用：" + message)
        elif self.visual_encoder is None and self.concept_lookup is None:
            degraded.append("未配置向量模型；本次仅使用字面检索")
        visual = []
        if self.visual_encoder is not None:
            try:
                # Preserve the original actor/object/action/negation constraints.
                visual = await self._visual(query, filters, degraded)
            except (ProviderError, OSError, ValueError, RuntimeError) as exc:
                degraded.append("直接视觉检索不可用，已保留其他召回结果：" + str(exc))
        concept = []
        if self.concept_lookup is not None:
            try:
                concept = await self._concept(query, filters)
            except (OSError, ValueError, RuntimeError) as exc:
                degraded.append("作品概念检索不可用，已保留其他召回结果：" + str(exc))
        records, scores = {}, defaultdict(float)
        for ranking in (lexical, semantic, visual, concept):
            for rank, record in enumerate(ranking, 1):
                previous = records.get(record["id"], {})
                records[record["id"]] = {**previous, **record,
                    "retrieval_branches": list(dict.fromkeys(previous.get("retrieval_branches", [])
                                                             + record.get("retrieval_branches", [])))}
                if previous.get("evidence_ids") or record.get("evidence_ids"):
                    records[record["id"]]["evidence_ids"] = list(dict.fromkeys(
                        previous.get("evidence_ids", []) + record.get("evidence_ids", [])))
                scores[record["id"]] += 1 / (60 + rank)
        ordered = sorted(records.values(), key=lambda record: scores[record["id"]], reverse=True)
        candidates = []
        for record in ordered:
            exact = record["kind"] == "subtitle" and normalize(query) in normalize(record.get("text", ""))
            candidates.append({**record, "score": scores[record["id"]],
                               "match_type": "full" if exact else "partial",
                               "match_reason": ("设定匹配该实体；画面证据仅确认出场，未证明当前功能或动作"
                                                if record.get("match_basis") == "setting_concept_appearance" else
                                                "台词原文包含查询文本" if exact else "包含相关证据，组合条件尚待核对")})
        if decision_profile_id and candidates:
            try:
                decision = await self.providers.decide(decision_profile_id, query, candidates[:20])
                judgments = decision.data["candidates"]
                if decision.data.get("expand_search") and len(candidates) > 20:
                    expanded = await self.providers.decide(decision_profile_id, query, candidates[20:40])
                    judgments.extend(expanded.data["candidates"])
                by_id = {row["id"]: row for row in candidates}
                reranked = []
                for judgment in judgments:
                    if judgment["id"] not in by_id or judgment["match_type"] == "none":
                        continue
                    candidate = by_id[judgment["id"]]
                    match_type = judgment["match_type"]
                    if (candidate.get("context_stale") or candidate.get("match_basis") == "setting_concept_appearance") and match_type == "full":
                        match_type = "partial"
                    reranked.append({**candidate, "score": judgment["score"],
                                     "match_type": match_type,
                                     "match_reason": "依据记录核对全部条件" if match_type == "full"
                                     else "记录仅支持部分查询条件"})
                candidates = sorted(reranked, key=lambda row: row["score"], reverse=True)
            except ProviderError as exc:
                degraded.append("候选核对失败，已保留召回结果：" + str(exc))
        # Suppress a context hit when one of its shorter source cues already contains
        # the complete literal quote. Context remains when the query spans cue boundaries.
        chosen = []
        for candidate in merge_visual_candidates(candidates):
            uncertain_names = unverified_character_names(query, candidate)
            if uncertain_names:
                candidate.update(match_type="partial", unverified_character_names=uncertain_names,
                                 match_reason="匹配到候选角色，画面人物的实名尚未确认")
            if candidate.get("is_context") and any(
                normalize(query) in normalize(records.get(source, {}).get("text", ""))
                for source in candidate.get("source_ids", [])
            ):
                continue
            chosen.append(candidate)
            if len(chosen) == limit:
                break
        explanation = (f"找到 {len(chosen)} 个候选片段；请结合原片核对标记为部分匹配的结果。"
                       if chosen else "未找到有足够证据支持的片段。")
        if answer_profile_id and chosen:
            try:
                answer = await self.providers.compose(answer_profile_id, query, chosen)
                explanation = answer.data["text"]
            except ProviderError as exc:
                degraded.append("解释生成失败，已使用证据摘要：" + str(exc))
        return {"results": chosen, "degraded": list(dict.fromkeys(degraded)), "explanation": explanation}
