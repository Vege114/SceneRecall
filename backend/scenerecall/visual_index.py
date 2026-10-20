"""Disposable, model-isolated image vectors and source-timed window manifests.

SQLite commits each finished window together with its lexical projection. Cached
pixels are reusable after an interrupted build; unfinished windows are never
advertised as indexed. Similarity is candidate evidence, not an identity or event.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "direct-visual-index-v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def evidence_digest(record: dict) -> str:
    """Names, prose and human display corrections do not change source pixels."""
    identity = {key: record.get(key) for key in (
        "id", "asset_id", "kind", "evidence_type", "source_fingerprint",
        "start_ms", "end_ms", "frame_ids", "evidence_frame_ids", "ordered_frames")}
    if record.get("held_frame_spans"):
        identity["held_frame_spans"] = record["held_frame_spans"]
    return digest(identity)


def normalized_vectors(values, expected: int, dimensions: int | None = None) -> list[np.ndarray]:
    try:
        vectors = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("视觉编码器返回了无效向量") from exc
    if (vectors.ndim != 2 or vectors.shape[0] != expected or vectors.shape[1] < 1
            or (dimensions is not None and vectors.shape[1] != dimensions)
            or not np.isfinite(vectors).all()):
        raise ValueError("视觉编码器向量数量、维度或数值无效")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("视觉编码器返回了零长度或无效向量")
    return [np.asarray(row / norm, dtype="<f4") for row, norm in zip(vectors, norms)]


class VisualIndex:
    def __init__(self, engine):
        self.engine = engine
        with engine._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS visual_namespaces (
                    signature TEXT PRIMARY KEY, identity TEXT NOT NULL,
                    dimensions INTEGER NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS visual_cache (
                    signature TEXT NOT NULL, cache_key TEXT NOT NULL,
                    dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
                    PRIMARY KEY(signature,cache_key)
                );
                CREATE TABLE IF NOT EXISTS visual_records (
                    signature TEXT NOT NULL, record_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL, evidence_digest TEXT NOT NULL,
                    PRIMARY KEY(signature,record_id)
                );
                CREATE INDEX IF NOT EXISTS visual_records_asset ON visual_records(asset_id);
                CREATE TABLE IF NOT EXISTS visual_links (
                    signature TEXT NOT NULL, record_id TEXT NOT NULL,
                    frame_id TEXT NOT NULL, cache_key TEXT NOT NULL, at_ms INTEGER NOT NULL,
                    PRIMARY KEY(signature,record_id,frame_id)
                );
                CREATE TABLE IF NOT EXISTS visual_query_cache (
                    signature TEXT NOT NULL, query_hash TEXT NOT NULL,
                    dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
                    PRIMARY KEY(signature,query_hash)
                );
            """)

    @staticmethod
    def namespace(encoder) -> tuple[str, dict]:
        identity = encoder.identity()
        if not isinstance(identity, dict) or not identity:
            raise ValueError("视觉编码器必须提供冻结的模型、权重与预处理身份")
        # Round-trip makes a detached, JSON-only snapshot, including every field
        # supplied by the encoder, rather than guessing interchangeable models.
        identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
        return digest([SCHEMA_VERSION, identity]), identity

    @staticmethod
    def _prepare(records, frames):
        by_id = {}
        for frame in frames:
            if not isinstance(frame.get("id"), str) or not frame["id"] or frame["id"] in by_id:
                raise ValueError("视觉源帧 ID 缺失或重复")
            by_id[frame["id"]] = dict(frame)
        owners = {record["asset_id"] for record in records}
        if len(owners) != 1:
            raise ValueError("视觉索引批次必须属于一个视频资产")
        keys, items, stats = {}, {}, {}
        pixel_hashes = {}
        for record in records:
            if (record["kind"] != "visual" or record.get("evidence_type") != "retrieval_window"
                    or not record.get("source_fingerprint")):
                raise ValueError("视觉索引只接受带原片身份的检索窗口")
            ids = record.get("frame_ids")
            ordered = record.get("ordered_frames")
            if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                    or record.get("evidence_frame_ids") != ids
                    or not isinstance(ordered, list) or [row.get("id") for row in ordered] != ids):
                raise ValueError("视觉窗口的有序源帧和证据引用不一致")
            held_spans = record.get("held_frame_spans") or []
            held_span = None
            if held_spans:
                if (len(ids) != 1 or not isinstance(held_spans, list) or len(held_spans) != 1
                        or not isinstance(held_spans[0], dict)
                        or held_spans[0].get("frame_id") != ids[0]
                        or type(held_spans[0].get("start_ms")) is not int
                        or type(held_spans[0].get("end_ms")) is not int
                        or not held_spans[0]["start_ms"] <= record["start_ms"] < record["end_ms"] <= held_spans[0]["end_ms"]):
                    raise ValueError("持帧声明必须由单一真实前帧覆盖整个窗口")
                held_span = held_spans[0]
            previous = -1
            for row in ordered:
                frame = by_id.get(row["id"])
                if frame is None:
                    raise ValueError("视觉窗口引用了不存在的源帧")
                at = frame.get("at_ms")
                base = frame.get("time_base")
                if held_span:
                    in_window = type(at) is int and 0 <= at <= held_span["start_ms"] and at < record["start_ms"]
                else:
                    in_window = type(at) is int and record["start_ms"] <= at < record["end_ms"]
                if (not in_window or at < previous
                        or type(frame.get("pts")) is not int
                        or not isinstance(base, (list, tuple)) or len(base) != 2
                        or any(type(value) is not int or value <= 0 for value in base)
                        or any(row.get(key) != frame.get(key) for key in ("at_ms", "pts", "time_base"))):
                    raise ValueError("视觉窗口的源帧时间证据无效")
                previous = at
                if frame.get("source_fingerprint", record["source_fingerprint"]) != record["source_fingerprint"]:
                    raise ValueError("视觉源帧不属于窗口的原片版本")
                path = Path(frame.get("path", ""))
                if not path.is_file():
                    raise ValueError("视觉源帧文件不存在")
                stat = path.stat()
                stamp = (stat.st_size, stat.st_mtime_ns)
                if path not in pixel_hashes:
                    with path.open("rb") as source:
                        pixel_hashes[path] = hashlib.file_digest(source, "sha256").hexdigest()
                    current = path.stat()
                    if (current.st_size, current.st_mtime_ns) != stamp:
                        raise ValueError("视觉源帧在读取时发生变化")
                bbox = frame.get("bbox", frame.get("box"))
                if bbox is not None and (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox)):
                    raise ValueError("视觉源帧区域无效")
                key = digest({"source": record["source_fingerprint"], "pixels_sha256": pixel_hashes[path],
                              "pts": frame["pts"], "time_base": list(base), "bbox": bbox})
                identity = (record["id"], row["id"])
                keys[identity] = key
                items.setdefault(key, frame)
                stats[key] = (path, stamp)
        return keys, items, stats

    @staticmethod
    def _check_files(keys, stats):
        for key in keys:
            path, stamp = stats[key]
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != stamp:
                raise ValueError("视觉源帧在编码时发生变化，未发布旧结果")

    async def build(self, records, frames, encoder, progress=None):
        engine = self.engine
        if encoder is None:
            raise ValueError("请先配置本地视觉编码器")
        engine._validate_records(records)
        signature, identity = self.namespace(encoder)
        if not records:
            return {"signature": signature, "records": 0, "frames": 0, "encoded_frames": 0,
                    "cache_hits": 0, "batches": 0, "status": "completed"}
        batch_size = getattr(encoder, "batch_size", 16)
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError("视觉编码批次必须为 1–256 的整数")
        # Freeze caller-owned inputs before yielding; nothing below edits source
        # records or relies on an asynchronously mutated prompt/evidence object.
        records = json.loads(json.dumps(records, ensure_ascii=False, allow_nan=False))
        frames = [dict(frame) for frame in frames]
        with engine._lock:
            generation = engine._generation
        keys, items, stats = await asyncio.to_thread(self._prepare, records, frames)
        with engine._lock, engine._db() as db:
            engine._check_generation(generation)
            namespace = db.execute("SELECT dimensions FROM visual_namespaces WHERE signature=?", (signature,)).fetchone()
            dimensions = namespace["dimensions"] if namespace else None
            cached = {row["cache_key"] for row in db.execute(
                "SELECT cache_key FROM visual_cache WHERE signature=?", (signature,))}
        available = set(items) & cached
        cache_hits = len(available)
        pending = [key for key in items if key not in available]
        published, batches, encoded = set(), 0, 0

        def publish():
            nonlocal generation
            ready = [row for row in records if row["id"] not in published
                     and all(keys[(row["id"], frame)] in available for frame in row["frame_ids"])]
            if not ready:
                return
            self._check_files({keys[(row["id"], frame)] for row in ready for frame in row["frame_ids"]}, stats)
            with engine._lock, engine._db() as db:
                engine._check_generation(generation)
                # Lexical rows and the complete visual manifest activate together.
                engine._sync_records(db, ready, ready[0]["asset_id"], retire=False)
                for row in ready:
                    db.execute("INSERT OR REPLACE INTO visual_records VALUES (?,?,?,?)",
                               (signature, row["id"], row["asset_id"], evidence_digest(row)))
                    db.execute("DELETE FROM visual_links WHERE signature=? AND record_id=?", (signature, row["id"]))
                    db.executemany("INSERT INTO visual_links VALUES (?,?,?,?,?)", [
                        (signature, row["id"], frame["id"], keys[(row["id"], frame["id"])], frame["at_ms"])
                        for frame in row["ordered_frames"]])
                db.execute("DELETE FROM build_checkpoints")
                engine._generation += 1
                generation = engine._generation
            published.update(row["id"] for row in ready)

        async def report():
            await engine._callback(progress, {"completed": len(available), "total": len(items),
                                             "published_records": len(published), "cache_hits": cache_hits,
                                             "batch_size": batch_size})

        publish()
        await report()
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            with engine._lock:
                engine._check_generation(generation)
            self._check_files(batch, stats)
            values = await encoder.encode_images([items[key] for key in batch])
            vectors = normalized_vectors(values, len(batch), dimensions)
            dimensions = len(vectors[0])
            self._check_files(batch, stats)
            if self.namespace(encoder)[0] != signature:
                raise ValueError("视觉模型或预处理身份在编码期间发生变化")
            with engine._lock, engine._db() as db:
                engine._check_generation(generation)
                db.executemany("INSERT OR REPLACE INTO visual_cache VALUES (?,?,?,?)",
                               [(signature, key, dimensions, vector.tobytes()) for key, vector in zip(batch, vectors)])
                db.execute("INSERT OR REPLACE INTO visual_namespaces VALUES (?,?,?,?)",
                           (signature, json.dumps(identity, ensure_ascii=False), dimensions, time.time()))
            available.update(batch)
            batches += 1
            encoded += len(batch)
            publish()
            await report()
        with engine._lock:
            engine._check_generation(generation)
        return {"signature": signature, "records": len(published), "frames": len(items),
                "encoded_frames": encoded, "cache_hits": cache_hits, "batches": batches, "status": "completed"}

    async def search(self, query, filters, encoder, degraded, limit=50):
        engine = self.engine
        signature, _identity = self.namespace(encoder)
        where, params = engine._where(filters)
        with engine._db() as db:
            namespace = db.execute("SELECT dimensions FROM visual_namespaces WHERE signature=?", (signature,)).fetchone()
            if not namespace:
                degraded.append("当前图文编码器尚未建立视觉索引")
                return []
            if not db.execute(f"""SELECT 1 FROM records r JOIN visual_records v ON v.record_id=r.id
                                  WHERE v.signature=? AND {where} LIMIT 1""", [signature, *params]).fetchone():
                return []
            query_hash = digest(query)
            cached = db.execute("SELECT vector,dimensions FROM visual_query_cache WHERE signature=? AND query_hash=?",
                                (signature, query_hash)).fetchone()
        dimensions = namespace["dimensions"]
        if cached and cached["dimensions"] == dimensions:
            vector = np.frombuffer(cached["vector"], dtype="<f4")
        else:
            vector = normalized_vectors(await encoder.encode_texts([query]), 1, dimensions)[0]
            if self.namespace(encoder)[0] != signature:
                raise ValueError("视觉查询编码器身份在运行期间发生变化")
            with engine._db() as db:
                db.execute("INSERT OR REPLACE INTO visual_query_cache VALUES (?,?,?,?)",
                           (signature, query_hash, dimensions, vector.tobytes()))
        return await asyncio.to_thread(self._rank, signature, vector, dimensions, where, params, limit)

    def _rank(self, signature, vector, dimensions, where, params, limit):
        matches = {}
        # Filter in SQL before scoring; a small selected work must not lose its
        # nearest candidates to frames from a larger unrelated work. Read current
        # evidence after query encoding, in one snapshot, and parse each window
        # once rather than once per source image. Matrix batches bound memory.
        with self.engine._db() as db:
            db.execute("BEGIN")
            valid = {}
            for row in db.execute(f"""SELECT r.id,r.data,v.evidence_digest FROM records r
                    JOIN visual_records v ON v.record_id=r.id
                    WHERE v.signature=? AND {where}""", [signature, *params]):
                record = json.loads(row["data"])
                if evidence_digest(record) == row["evidence_digest"]:
                    valid[row["id"]] = record
            if not valid:
                return []
            rows = db.execute(f"""SELECT l.record_id,l.frame_id,l.at_ms,c.vector,c.dimensions
                FROM visual_links l JOIN visual_cache c
                  ON c.signature=l.signature AND c.cache_key=l.cache_key
                JOIN records r ON r.id=l.record_id
                JOIN visual_records v ON v.record_id=r.id AND v.signature=l.signature
                WHERE l.signature=? AND {where}""", [signature, *params])
            while batch := rows.fetchmany(512):
                accepted, vectors = [], []
                for row in batch:
                    if row["record_id"] not in valid or row["dimensions"] != dimensions:
                        continue
                    image = np.frombuffer(row["vector"], dtype="<f4")
                    if image.shape == vector.shape and np.isfinite(image).all():
                        accepted.append(row)
                        vectors.append(image)
                if not accepted:
                    continue
                scores = np.stack(vectors) @ vector
                for row, value in zip(accepted, scores):
                    score = float(value)
                    # Positive scores are uncalibrated candidates, never "full".
                    if not math.isfinite(score) or score <= 0:
                        continue
                    old = matches.get(row["record_id"])
                    if old is None or score > old["visual_similarity"]:
                        matches[row["record_id"]] = {**valid[row["record_id"]], "visual_similarity": score,
                            "visual_signature": signature, "hit_frame_id": row["frame_id"], "seek_ms": row["at_ms"],
                            "evidence_ids": [row["frame_id"]], "retrieval_branches": ["visual"]}
        ordered = sorted(matches.values(), key=lambda row: (-row["visual_similarity"], row["asset_id"],
                                                            row["start_ms"], row["id"]))
        return ordered[:min(50, max(1, limit))]

    @staticmethod
    def _prepare_references(references):
        items, stats, concepts = {}, {}, {}
        ids = set()
        for reference in references:
            if reference.get("status", "confirmed") != "confirmed":
                continue
            identifier, concept = reference.get("id"), reference.get("concept_id")
            scope = reference.get("work_scope_id") or reference.get("asset_id")
            if (not isinstance(identifier, str) or not identifier or identifier in ids
                    or not isinstance(concept, str) or not concept or not isinstance(scope, str) or not scope):
                raise ValueError("作品参考图须具有唯一 ID、概念 ID 和作品作用域")
            ids.add(identifier)
            path = Path(reference.get("path", ""))
            if not path.is_file():
                raise ValueError("作品参考图文件不存在")
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
            with path.open("rb") as source:
                sha256 = hashlib.file_digest(source, "sha256").hexdigest()
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != stamp:
                raise ValueError("作品参考图在读取期间发生变化")
            if reference.get("source_fingerprint") != sha256:
                raise ValueError("作品参考图像素与声明的 SHA256 不一致")
            bbox = reference.get("bbox", reference.get("box"))
            if bbox is not None and (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                    or any(type(value) not in {int, float} or not math.isfinite(value) for value in bbox)
                    or not 0 <= bbox[0] < bbox[2] <= 1 or not 0 <= bbox[1] < bbox[3] <= 1):
                raise ValueError("作品参考图区域须为归一化 xyxy")
            if reference.get("pts") != 0 or reference.get("time_base") != [1, 1]:
                raise ValueError("作品参考图不能伪装成当前原片时间证据")
            key = digest({"reference_only": True, "work_scope_id": scope, "pixels_sha256": sha256,
                          "source": reference["source_fingerprint"], "pts": 0, "time_base": [1, 1], "bbox": bbox})
            items.setdefault(key, dict(reference))
            stats[key] = (path, stamp)
            concepts.setdefault(key, set()).add(concept)
        return items, stats, concepts

    async def match_references(self, records, references, encoder):
        """Select relevant dossiers from pixels, never confirm their identity.

        The caller supplies only references from the work's confirmed bank.
        Explicit candidate/revoked references are excluded defensively. Current
        pictures come exclusively from validated, published window manifests;
        reference features are cached in the same model space but have a distinct
        reference-only source identity and can never become frame evidence.
        """
        if not records or not references:
            return {}
        if encoder is None:
            raise ValueError("作品图像档案选择需要已配置的视觉编码器")
        signature, _ = self.namespace(encoder)
        records = json.loads(json.dumps(records, ensure_ascii=False, allow_nan=False))
        with self.engine._lock, self.engine._db() as db:
            generation = self.engine._generation
            namespace = db.execute("SELECT dimensions FROM visual_namespaces WHERE signature=?", (signature,)).fetchone()
            if namespace is None:
                return {}
        dimensions = namespace["dimensions"]
        items, stats, concepts = await asyncio.to_thread(self._prepare_references, [dict(ref) for ref in references])
        if not items:
            return {}
        batch_size = getattr(encoder, "batch_size", 16)
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError("视觉编码批次必须为 1–256 的整数")
        with self.engine._lock, self.engine._db() as db:
            self.engine._check_generation(generation)
            cached = {}
            item_keys = list(items)
            for offset in range(0, len(item_keys), 256):
                group = item_keys[offset:offset + 256]
                placeholders = ",".join("?" for _ in group)
                cached.update({row["cache_key"]: row for row in db.execute(
                    f"""SELECT cache_key,dimensions,vector FROM visual_cache
                        WHERE signature=? AND cache_key IN ({placeholders})""", [signature, *group])})
        vectors = {}
        for key, row in cached.items():
            vector = np.frombuffer(row["vector"], dtype="<f4")
            if row["dimensions"] == dimensions and vector.shape == (dimensions,) and np.isfinite(vector).all():
                vectors[key] = vector
        missing = [key for key in items if key not in vectors]
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset:offset + batch_size]
            self._check_files(batch, stats)
            with self.engine._lock:
                self.engine._check_generation(generation)
            encoded = normalized_vectors(await encoder.encode_images([items[key] for key in batch]), len(batch), dimensions)
            self._check_files(batch, stats)
            if self.namespace(encoder)[0] != signature:
                raise ValueError("视觉模型或预处理身份在参考图编码期间发生变化")
            with self.engine._lock, self.engine._db() as db:
                self.engine._check_generation(generation)
                db.executemany("INSERT OR REPLACE INTO visual_cache VALUES (?,?,?,?)",
                    [(signature, key, dimensions, vector.tobytes()) for key, vector in zip(batch, encoded)])
            vectors.update(zip(batch, encoded))
        self._check_files(items, stats)
        with self.engine._lock:
            self.engine._check_generation(generation)
        return await asyncio.to_thread(self._match_reference_vectors, records, signature, dimensions, vectors, concepts)

    def _match_reference_vectors(self, records, signature, dimensions, references, concepts):
        supplied = {row["id"]: evidence_digest(row) for row in records}
        best = {}
        keys = list(references)
        with self.engine._db() as db:
            db.execute("BEGIN")
            valid = set()
            identities = list(supplied)
            for offset in range(0, len(identities), 256):
                group = identities[offset:offset + 256]
                placeholders = ",".join("?" for _ in group)
                for row in db.execute(f"""SELECT r.id,r.data,v.evidence_digest FROM records r
                        JOIN visual_records v ON v.record_id=r.id
                        WHERE v.signature=? AND r.id IN ({placeholders})""", [signature, *group]):
                    if supplied[row["id"]] == row["evidence_digest"] == evidence_digest(json.loads(row["data"])):
                        valid.add(row["id"])
            if not valid:
                return {}
            identities = list(valid)
            for position in range(0, len(identities), 256):
                group = identities[position:position + 256]
                placeholders = ",".join("?" for _ in group)
                rows = db.execute(f"""SELECT l.record_id,c.vector,c.dimensions FROM visual_links l
                    JOIN visual_cache c ON c.signature=l.signature AND c.cache_key=l.cache_key
                    JOIN records r ON r.id=l.record_id
                    WHERE l.signature=? AND l.record_id IN ({placeholders})""", [signature, *group])
                while batch := rows.fetchmany(512):
                    vectors, positions = [], {}
                    for row in batch:
                        if row["dimensions"] != dimensions:
                            continue
                        vector = np.frombuffer(row["vector"], dtype="<f4")
                        if vector.shape != (dimensions,) or not np.isfinite(vector).all():
                            continue
                        positions.setdefault(row["record_id"], []).append(len(vectors))
                        vectors.append(vector)
                    if not vectors:
                        continue
                    matrix = np.stack(vectors)
                    for offset in range(0, len(keys), 128):
                        ref_keys = keys[offset:offset + 128]
                        scores = matrix @ np.stack([references[key] for key in ref_keys]).T
                        for record_id, indices in positions.items():
                            maxima = scores[indices].max(axis=0)
                            for key, value in zip(ref_keys, maxima):
                                score = float(value)
                                if not math.isfinite(score) or score <= 0:
                                    continue
                                current = best.setdefault(record_id, {})
                                for concept in concepts[key]:
                                    current[concept] = max(current.get(concept, -1.), score)
        return {record_id: [{"concept_id": concept, "similarity": similarity}
                            for concept, similarity in sorted(values.items(), key=lambda pair: (-pair[1], pair[0]))[:5]]
                for record_id, values in best.items()}

    @staticmethod
    def remove_assets(db, asset_ids):
        for asset_id in asset_ids:
            db.execute("""DELETE FROM visual_links WHERE record_id IN
                          (SELECT record_id FROM visual_records WHERE asset_id=?)""", (asset_id,))
            db.execute("DELETE FROM visual_records WHERE asset_id=?", (asset_id,))

    def status(self):
        with self.engine._db() as db:
            return [{**dict(row), "identity": json.loads(row["identity"])} for row in
                    db.execute("""SELECT n.*,COUNT(v.record_id) AS records FROM visual_namespaces n
                                  LEFT JOIN visual_records v ON n.signature=v.signature
                                  GROUP BY n.signature ORDER BY n.updated_at DESC""")]
