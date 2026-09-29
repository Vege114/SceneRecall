from __future__ import annotations

import hashlib
import json
import copy
import os
import re
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .models import AssetInput, Observation

SCHEMA = "1.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,180}", value) or value in {".", ".."}:
        raise ValueError("无效的记录 ID")
    return value


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    result = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(result, dict) and result.get("schema_version", SCHEMA) != SCHEMA:
        raise ValueError(f"不支持的数据格式版本: {path.name}")
    return result


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(r.get("schema_version", SCHEMA) != SCHEMA for r in records):
        raise ValueError("不支持的数据格式版本")
    return records


def confined_path(base: Path, relative: str) -> Path:
    """File references are data, never permission to traverse outside an asset."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("资料包含非法文件引用")
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("资料包含越界文件引用")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("资料包含越界文件引用")
    return path


def same_contents(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while chunk := left.read(1024 * 1024):
            if chunk != right.read(len(chunk)):
                return False
    return True


def source_snapshot(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class Library:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.lock = threading.RLock()
        for directory in ("assets", "works", "collections", "indexes", "runtime", "private"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / "private").chmod(0o700)
        if not (self.root / "library.json").exists():
            atomic_json(self.root / "library.json", {"schema_version": SCHEMA, "id": uid("library"), "created_at": now()})
        read_json(self.root / "library.json")

    def asset_dir(self, asset_id: str) -> Path:
        return self.root / "assets" / safe_id(asset_id)

    def get_asset(self, asset_id: str) -> dict:
        result = read_json(self.asset_dir(asset_id) / "manifest.json")
        if result is None:
            raise FileNotFoundError("未找到视频资产")
        required = {"id", "work_id", "title", "created_at", "duration_ms", "version", "fingerprint"}
        if not isinstance(result, dict) or required - result.keys() or result["id"] != asset_id:
            raise ValueError("资产清单不完整或 ID 与目录不一致")
        location = read_json(self.root / "private" / "media-locations.json", {}).get(asset_id)
        path = self.source_path(asset_id)
        exists = bool(path and path.is_file())
        result["source_changed"] = bool(exists and (not isinstance(location, dict) or source_snapshot(path) != location))
        result["source_available"] = exists and not result["source_changed"]
        return result

    def source_path(self, asset_id: str) -> Path | None:
        item = read_json(self.root / "private" / "media-locations.json", {}).get(safe_id(asset_id))
        return Path(item["path"] if isinstance(item, dict) else item) if item else None

    def require_source(self, asset_id: str) -> Path:
        if not self.get_asset(asset_id)["source_available"]:
            raise ValueError("原视频离线或文件已变化，请先重新定位并校验原片")
        return self.source_path(asset_id)

    def list_assets(self) -> list[dict]:
        return sorted([self.get_asset(p.parent.name) for p in (self.root / "assets").glob("*/manifest.json")],
                      key=lambda a: a["created_at"], reverse=True)

    def register(self, payload: AssetInput) -> dict:
        from .media import fingerprint, probe
        from .subtitles import parse_subtitles

        source = Path(payload.video_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("视频路径必须指向文件")
        snapshot = source_snapshot(source)
        metadata = probe(source)
        cues = []
        sub_path = None
        if payload.subtitle_mode == "external":
            sub_path = Path(payload.subtitle_path).expanduser().resolve(strict=True)
            cues = parse_subtitles(sub_path, metadata["duration_ms"], payload.subtitle_offset_ms)
            if not cues:
                raise ValueError("字幕文件没有有效字幕")
        checksum = fingerprint(source)
        if source_snapshot(source) != snapshot:
            raise ValueError("视频在导入期间被修改，请重新导入")
        with self.lock:
            for asset in self.list_assets():
                if asset["fingerprint"] == checksum:
                    raise ValueError("该视频已在资料库中，可重新定位原文件或重跑识别")
            asset_id = uid("asset")
            title = payload.title.strip() or source.stem
            identity = [title, payload.kind, payload.series, payload.season, payload.episode]
            work_id = "work_" + digest(identity)
            work = {"schema_version": SCHEMA, "id": work_id, "title": title, "kind": payload.kind,
                    "series": payload.series, "season": payload.season, "episode": payload.episode}
            asset = {**work, **metadata, "id": asset_id, "work_id": work_id,
                     "version": payload.version, "fingerprint": checksum, "file_name": source.name,
                     "subtitle_mode": payload.subtitle_mode, "subtitle_offset_ms": payload.subtitle_offset_ms,
                     "created_at": now(), "status": "ready"}
            atomic_json(self.root / "works" / work_id / "work.json", work)
            directory = self.asset_dir(asset_id)
            directory.mkdir(parents=True)
            if sub_path:
                originals = directory / "subtitles" / "originals"
                originals.mkdir(parents=True)
                shutil.copy2(sub_path, originals / ("original" + sub_path.suffix.lower()))
                cues = [{**c, "schema_version": SCHEMA, "asset_id": asset_id,
                         "id": "cue_" + digest([asset_id, c["id"]]), "source": "external"} for c in cues]
                atomic_jsonl(directory / "subtitles" / "external.jsonl", cues)
            atomic_json(directory / "active-analysis.json", {"schema_version": SCHEMA, "observations": {}, "subtitle_runs": {}})
            locations = read_json(self.root / "private" / "media-locations.json", {})
            locations[asset_id] = snapshot
            atomic_json(self.root / "private" / "media-locations.json", locations)
            atomic_json(directory / "manifest.json", asset)
        return self.get_asset(asset_id)

    def relocate(self, asset_id: str, path: str) -> dict:
        from .media import fingerprint
        asset = self.get_asset(asset_id)
        new_path = Path(path).expanduser().resolve(strict=True)
        snapshot = source_snapshot(new_path)
        if fingerprint(new_path) != asset["fingerprint"]:
            raise ValueError("文件内容与原视频不一致；不同剪辑版应单独导入")
        if source_snapshot(new_path) != snapshot:
            raise ValueError("视频在校验期间被修改，请重新定位")
        with self.lock:
            locations = read_json(self.root / "private" / "media-locations.json", {})
            locations[asset_id] = snapshot
            atomic_json(self.root / "private" / "media-locations.json", locations)
        return self.get_asset(asset_id)

    def active(self, asset_id: str) -> dict:
        return read_json(self.asset_dir(asset_id) / "active-analysis.json", {"schema_version": SCHEMA, "observations": {}, "subtitle_runs": {}})

    def observations(self, asset_id: str) -> list[dict]:
        directory = self.asset_dir(asset_id)
        return sorted([read_json(confined_path(directory, p)) for p in self.active(asset_id)["observations"].values()],
                      key=lambda r: r["start_ms"])

    def subtitles(self, asset_id: str) -> list[dict]:
        directory = self.asset_dir(asset_id)
        result = read_jsonl(directory / "subtitles" / "external.jsonl")
        for value in self.active(asset_id).get("subtitle_runs", {}).values():
            cues = read_jsonl(confined_path(directory, value["path"]))
            result.extend(cues)
        return sorted(result, key=lambda c: (c["start_ms"], c.get("language", "und")))

    def save_observation(self, asset_id: str, value: dict) -> dict:
        asset = self.get_asset(asset_id)
        record = Observation.model_validate(value).model_dump()
        if record["end_ms"] > asset["duration_ms"] or record["asset_id"] != asset_id:
            raise ValueError("记录不属于当前视频时间轴")
        for frame_id in record["evidence_frame_ids"]:
            self.frame_path(asset_id, frame_id)
        relative = Path("analyses") / safe_id(record["run_id"]) / "observations" / f"{safe_id(record['id'])}.json"
        with self.lock:
            existing = read_json(self.asset_dir(asset_id) / relative)
            if existing is not None and existing != record:
                raise ValueError("已提交的运行结果不可覆盖；请创建新的分析运行")
            atomic_json(self.asset_dir(asset_id) / relative, record)
            active = self.active(asset_id)
            # New segmentation replaces only fully covered old records, avoiding silent mixing.
            for old_id, path in list(active["observations"].items()):
                old = read_json(confined_path(self.asset_dir(asset_id), path))
                if record["start_ms"] <= old["start_ms"] and old["end_ms"] <= record["end_ms"]:
                    del active["observations"][old_id]
            active["observations"][record["id"]] = str(relative)
            entries = {rid: read_json(confined_path(self.asset_dir(asset_id), ref))
                       for rid, ref in active["observations"].items()}
            covered = sorted((r["start_ms"], r["end_ms"]) for r in entries.values() if r["run_id"] == record["run_id"])
            merged = []
            for start, end in covered:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            for rid, old in entries.items():
                if old["run_id"] != record["run_id"] and any(a <= old["start_ms"] and old["end_ms"] <= b for a, b in merged):
                    active["observations"].pop(rid, None)
            atomic_json(self.asset_dir(asset_id) / "active-analysis.json", active)
        return record

    def save_subtitles(self, asset_id: str, run_id: str, cues: list[dict], start_ms: int, end_ms: int) -> None:
        asset = self.get_asset(asset_id)
        if not 0 <= start_ms < end_ms <= asset["duration_ms"]:
            raise ValueError("字幕替换范围无效")
        for cue in cues:
            empty_unreadable = not cue["text"].strip() and cue.get("review_status") == "needs_review"
            if not 0 <= cue["start_ms"] < cue["end_ms"] <= asset["duration_ms"] or (not cue["text"].strip() and not empty_unreadable):
                raise ValueError("字幕内容或时间范围无效")
            if not start_ms <= cue["start_ms"] < cue["end_ms"] <= end_ms:
                raise ValueError("字幕超出本次替换范围")
        with self.lock:
            active = self.active(asset_id)
            # Retain unaffected spans from previous OCR runs, including partially overlapped cues.
            previous = [c for c in self.subtitles(asset_id) if c.get("source") != "external"]
            retained = []
            for cue in previous:
                if cue["end_ms"] <= start_ms or cue["start_ms"] >= end_ms:
                    retained.append(cue)
                else:
                    if cue["start_ms"] < start_ms:
                        retained.append({**cue, "id": cue["id"] + "_left", "end_ms": start_ms})
                    if cue["end_ms"] > end_ms:
                        retained.append({**cue, "id": cue["id"] + "_right", "start_ms": end_ms})
            final = retained + [{**c, "schema_version": SCHEMA, "asset_id": asset_id, "run_id": run_id,
                                 "id": "cue_" + digest([asset_id, c.get("language"), c["start_ms"], c["end_ms"], c["text"]]),
                                 "source": "ocr"} for c in cues]
            relative = Path("analyses") / safe_id(run_id) / "subtitles.jsonl"
            atomic_jsonl(self.asset_dir(asset_id) / relative, final)
            active["subtitle_runs"] = {run_id: {"path": str(relative)}}
            atomic_json(self.asset_dir(asset_id) / "active-analysis.json", active)

    def register_frame(self, asset_id: str, frame: dict) -> dict:
        path = Path(frame["path"]).resolve()
        directory = self.asset_dir(asset_id).resolve()
        relative = path.relative_to(directory)
        with self.lock:
            registry_path = directory / "frames" / "registry.json"
            registry = read_json(registry_path, {})
            registry[safe_id(frame["id"])] = {"path": str(relative), "at_ms": frame["at_ms"]}
            atomic_json(registry_path, registry)
        return frame

    def frame_path(self, asset_id: str, frame_id: str) -> Path:
        registry = read_json(self.asset_dir(asset_id) / "frames" / "registry.json", {})
        value = registry.get(safe_id(frame_id))
        if not value:
            raise FileNotFoundError("证据帧不存在")
        path = confined_path(self.asset_dir(asset_id), value["path"])
        if not path.is_file():
            raise FileNotFoundError("证据帧文件不存在")
        return path

    def annotations(self, asset_id: str) -> list[dict]:
        annotations = sorted([read_json(p) for p in (self.asset_dir(asset_id) / "annotations").glob("*.json")], key=lambda a: a["updated_at"])
        records = {r["id"]: r for r in self.observations(asset_id) + self.subtitles(asset_id)}
        for annotation in annotations:
            record = records.get(annotation["record_id"])
            annotation["needs_reassociation"] = record is None or bool(annotation.get("character_name") and annotation.get("base_run_id") != record.get("run_id"))
        return annotations

    def annotate(self, asset_id: str, payload: dict) -> dict:
        records = {r["id"]: r for r in self.observations(asset_id) + self.subtitles(asset_id)}
        if payload["record_id"] not in records:
            historical = next((r for r in self.collections() if r["asset_id"] == asset_id and r["record_id"] == payload["record_id"]), None)
            if not historical or set(payload) - {"record_id", "favorite", "note"}:
                raise ValueError("历史记录只允许修改收藏与笔记；身份和描述请在当前有效记录中确认")
            records[payload["record_id"]] = historical
        record = records[payload["record_id"]]
        if payload.get("character_name"):
            if payload.get("entity_id") not in {e["id"] for e in record.get("entities", [])}:
                raise ValueError("确认角色姓名前请选择记录中的人物实体")
        with self.lock:
            annotation_id = "ann_" + digest([payload["record_id"], payload.get("entity_id")])
            path = self.asset_dir(asset_id) / "annotations" / f"{annotation_id}.json"
            previous = read_json(path, {})
            annotation = {**previous, **payload, "schema_version": SCHEMA, "id": annotation_id,
                          "asset_id": asset_id, "updated_at": now(), "base_run_id": record.get("run_id")}
            atomic_json(path, annotation)
            return annotation

    def records(self, asset_id: str | None = None) -> list[dict]:
        from .characters import CharacterStore
        characters = CharacterStore(self)
        result = []
        for asset in [self.get_asset(asset_id)] if asset_id else self.list_assets():
            annotations = self.annotations(asset["id"])
            observations = characters.decorate_observations(asset["id"], self.observations(asset["id"]))
            for record in observations + self.subtitles(asset["id"]):
                is_visual = "summary" in record
                related = [a for a in annotations if a["record_id"] == record["id"]]
                text = record.get("summary", record.get("text", ""))
                favorite, note, user_confirmed = False, "", False
                names = []
                entities = copy.deepcopy(record.get("entities", []))
                for annotation in related:
                    if annotation.get("summary") is not None:
                        text = annotation["summary"]
                        user_confirmed = True
                    if "favorite" in annotation:
                        favorite = annotation["favorite"]
                    if annotation.get("note") is not None:
                        note = annotation["note"]
                    if annotation.get("character_name") and not annotation.get("needs_reassociation"):
                        user_confirmed = True
                        names += [annotation["character_name"], *(annotation.get("aliases") or [])]
                        for entity in entities:
                            if entity["id"] == annotation.get("entity_id"):
                                entity["name"] = annotation["character_name"]
                                entity["character_name"] = annotation["character_name"]
                                entity["aliases"] = annotation.get("aliases") or []
                description = text
                # An unreadable cue remains visible for repair but is not searchable content.
                if not text.strip():
                    continue
                if is_visual:
                    for entity in entities:
                        description += " " + " ".join(str(entity.get(k, "")) for k in ("appearance", "label", "name", "description"))
                        if entity.get("character_id"):
                            names.extend([entity.get("character_name", ""), *(entity.get("aliases") or [])])
                    for event in record.get("events", []):
                        description += " " + str(event.get("action", ""))
                    description += " " + " ".join(names)
                frames = record.get("evidence_frame_ids", [])
                result.append({"id": record["id"], "record_id": record["id"], "asset_id": asset["id"],
                               "work_id": asset["work_id"], "title": asset["title"], "series": asset.get("series", ""),
                               "season": asset.get("season"), "episode": asset.get("episode"), "version": asset["version"],
                               "kind": "visual" if is_visual else "subtitle", "start_ms": record["start_ms"],
                               "end_ms": record["end_ms"], "text": description.strip(), "summary": text,
                               "subtitle_text": text if not is_visual else None, "language": record.get("language"),
                               "thumbnail": frames[0] if frames else None, "evidence_frame_ids": frames,
                               "source": record.get("source", "vision"), "review_status": "user_confirmed" if user_confirmed else record.get("review_status", "unreviewed"),
                               "entities": entities, "events": record.get("events", []),
                               "character": " ".join(dict.fromkeys(names)), "favorite": favorite, "note": note,
                               "run_id": record.get("run_id"), "uncertainties": record.get("uncertainties", [])})
        return result

    def save_run(self, asset_id: str, run_id: str, data: dict):
        atomic_json(self.asset_dir(asset_id) / "analyses" / safe_id(run_id) / "run.json",
                    {**data, "schema_version": SCHEMA, "id": run_id, "asset_id": asset_id})

    def runs(self, asset_id: str) -> list[dict]:
        return [read_json(p) for p in (self.asset_dir(asset_id) / "analyses").glob("*/run.json")]

    def collections(self) -> list[dict]:
        current = {r["id"]: r for r in self.records()}
        result = [r for r in current.values() if r["favorite"]]
        for asset in self.list_assets():
            states = {}
            for annotation in self.annotations(asset["id"]):
                rid = annotation["record_id"]
                states[rid] = {**states.get(rid, {}), **annotation}
            for rid, annotation in states.items():
                if rid in current or not annotation.get("favorite"):
                    continue
                run_id = annotation.get("base_run_id")
                record = None
                if run_id:
                    base = self.asset_dir(asset["id"]) / "analyses" / safe_id(run_id)
                    record = read_json(base / "observations" / f"{safe_id(rid)}.json")
                    if record is None:
                        record = next((c for c in read_jsonl(base / "subtitles.jsonl") if c["id"] == rid), None)
                if record is None:
                    continue
                text = annotation.get("summary", record.get("summary", record.get("text", "")))
                frames = record.get("evidence_frame_ids", [])
                result.append({"id": rid, "record_id": rid, "asset_id": asset["id"], "work_id": asset["work_id"],
                               "title": asset["title"], "series": asset.get("series"), "season": asset.get("season"),
                               "episode": asset.get("episode"), "version": asset["version"], "start_ms": record["start_ms"],
                               "end_ms": record["end_ms"], "kind": "visual" if "summary" in record else "subtitle",
                               "text": text, "summary": text, "note": annotation.get("note", ""), "favorite": True,
                               "source": record.get("source", "vision"), "review_status": "historical",
                               "is_historical": True, "run_id": run_id, "evidence_frame_ids": frames,
                               "thumbnail": frames[0] if frames else None})
        return result

    def backup(self, include_media: bool = False) -> dict:
        backup_id = uid("backup")
        target = self.root / "runtime" / "backups" / f"{backup_id}.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for top in ("library.json", "works", "assets", "collections"):
                item = self.root / top
                for path in [item] if item.is_file() else item.rglob("*"):
                    if path.is_file() and not path.is_symlink() and path.suffix != ".tmp" and "/proxy/" not in str(path):
                        archive.write(path, path.relative_to(self.root))
            if include_media:
                for asset in self.list_assets():
                    path = self.require_source(asset["id"])
                    archive.write(path, f"media/{asset['id']}{path.suffix}", compress_type=zipfile.ZIP_STORED)
        return {"id": backup_id, "download_url": f"/api/backups/{backup_id}", "includes_media": include_media}

    def restore(self, archive_path: Path) -> dict:
        staging = self.root / "runtime" / uid("restore")
        staging.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                if len({i.filename for i in members}) != len(members):
                    raise ValueError("备份包含重复路径")
                if len(members) > 300000 or sum(i.file_size for i in members) > 200 * 1024**3:
                    raise ValueError("备份文件超出恢复上限")
                for info in members:
                    relative = PurePosixPath(info.filename)
                    if relative.is_absolute() or ".." in relative.parts or "\\" in info.filename or not relative.parts:
                        raise ValueError("备份含非法路径")
                    if relative.parts[0] not in {"library.json", "works", "assets", "collections", "media"}:
                        raise ValueError("备份包含非资料文件")
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("备份不允许符号链接")
                    archive.extract(info, staging)
            read_json(staging / "library.json") or (_ for _ in ()).throw(ValueError("缺少资料库清单"))
            for file in staging.rglob("*.json"):
                read_json(file)
            for file in staging.rglob("*.jsonl"):
                read_jsonl(file)
            # Validate the entire reference graph before committing any imported file.
            restored = Library(staging)
            for manifest in (staging / "assets").glob("*/manifest.json"):
                if read_json(manifest).get("id") != manifest.parent.name:
                    raise ValueError("资产清单 ID 与目录不一致")
            for asset in restored.list_assets():
                asset_id = safe_id(asset["id"])
                if not (staging / "assets" / asset_id / "manifest.json").is_file():
                    raise ValueError("资产清单 ID 与目录不一致")
                for observation in restored.observations(asset_id):
                    checked = Observation.model_validate(observation)
                    if checked.asset_id != asset_id or checked.end_ms > asset["duration_ms"]:
                        raise ValueError("备份识别记录时间轴无效")
                    for frame_id in checked.evidence_frame_ids:
                        restored.frame_path(asset_id, frame_id)
                for cue in restored.subtitles(asset_id):
                    if not 0 <= cue["start_ms"] < cue["end_ms"] <= asset["duration_ms"]:
                        raise ValueError("备份字幕时间轴无效")
                registry = read_json(restored.asset_dir(asset_id) / "frames" / "registry.json", {})
                for frame_id in registry:
                    restored.frame_path(asset_id, frame_id)
            from .characters import CharacterStore
            CharacterStore(restored).validate_all()
            paths = [p for p in staging.rglob("*") if p.is_file() and p.name != "library.json"]
            from .media import fingerprint
            for path in (staging / "media").glob("*"):
                asset = restored.get_asset(path.stem)
                if fingerprint(path) != asset["fingerprint"]:
                    raise ValueError("备份中的视频内容与资产指纹不一致")
            with self.lock:
                for path in paths:
                    target = self.root / path.relative_to(staging)
                    if target.exists() and not same_contents(target, path):
                        raise ValueError(f"恢复冲突，未覆盖现有文件: {path.relative_to(staging)}")
                for path in paths:
                    target = self.root / path.relative_to(staging)
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(path, target)
                locations = read_json(self.root / "private" / "media-locations.json", {})
                for path in (self.root / "media").glob("*"):
                    if path.stem.startswith("asset_"):
                        locations[path.stem] = source_snapshot(path)
                atomic_json(self.root / "private" / "media-locations.json", locations)
            return {"restored_files": len(paths), "message": "资料已恢复；元数据备份中的原视频可能需要重新定位"}
        finally:
            shutil.rmtree(staging, ignore_errors=True)
