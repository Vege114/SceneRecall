from __future__ import annotations

import hashlib
import json
import copy
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .models import AssetInput, Observation

SCHEMA = "1.0"


class DeletedAssetError(ValueError):
    def __init__(self):
        super().__init__("该视频已在回收站，请先从回收站恢复")


class DuplicateAssetError(ValueError):
    def __init__(self, asset_id):
        super().__init__("该视频已在资料库中，可重新定位原文件或重跑识别")
        self.asset_id = asset_id


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,180}", value) or value in {".", ".."}:
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


def recommended_subtitle_track(tracks: list[dict]) -> dict | None:
    supported = [track for track in tracks if track["supported"]]
    return min(supported, key=lambda track: (not track["default"], track["forced"], track["index"])) if supported else None


def subtitle_track_info(path: Path) -> dict:
    from .media import probe, subtitle_tracks

    metadata = probe(path)
    tracks = subtitle_tracks(metadata)
    recommended = recommended_subtitle_track(tracks)
    return {"tracks": tracks, "recommended_stream_index": recommended["index"] if recommended else None,
            "duration_ms": metadata["duration_ms"]}


def select_subtitle_track(tracks: list[dict], stream_index: int | None = None) -> dict:
    if stream_index is not None:
        track = next((track for track in tracks if track["index"] == stream_index), None)
        if track is None:
            raise ValueError("所选字幕轨不存在，请重新检测视频中的字幕轨")
        if not track["supported"]:
            raise ValueError("所选字幕轨无法提取文字：" + track.get("reason", "不支持的字幕格式"))
        return track
    track = recommended_subtitle_track(tracks)
    if track is not None:
        return track
    if tracks:
        raise ValueError("容器中只有位图或不支持的字幕轨，请提供外挂文字字幕；画面 OCR 仅适用于已经烧录在视频画面中的文字，不能读取软字幕轨")
    raise ValueError("容器中没有字幕轨，请提供外挂字幕或使用画面烧录字幕 OCR")


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
        from .organization import OrganizationStore
        self.organization = OrganizationStore(self)
        from .deletions import DeletionStore
        self.deletions = DeletionStore(self)
        self.deletions.reconcile_sources()

    def asset_dir(self, asset_id: str) -> Path:
        return self.root / "assets" / safe_id(asset_id)

    def get_asset(self, asset_id: str, *, include_deleted=False) -> dict:
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
        # The atomic active pointer is the source of truth during track switches.
        # A crash between publishing it and refreshing the manifest is recoverable.
        container = self._container_pointer(asset_id)
        if container:
            result.update(subtitle_mode="container", subtitle_stream_index=container["stream_index"],
                          subtitle_track=container["track"], subtitle_offset_ms=container["offset_ms"],
                          subtitle_import_warnings=container.get("warnings", []))
            result.pop("subtitle_fallback_reason", None)
        return self.organization.project_asset(result, include_deleted=include_deleted)

    def source_path(self, asset_id: str) -> Path | None:
        item = read_json(self.root / "private" / "media-locations.json", {}).get(safe_id(asset_id))
        return Path(item["path"] if isinstance(item, dict) else item) if item else None

    def require_source(self, asset_id: str) -> Path:
        if not self.get_asset(asset_id)["source_available"]:
            raise ValueError("原视频离线或文件已变化，请先重新定位并校验原片")
        return self.source_path(asset_id)

    def list_assets(self, *, include_deleted=False) -> list[dict]:
        assets = [self.get_asset(p.parent.name, include_deleted=True) for p in (self.root / "assets").glob("*/manifest.json")]
        return sorted([asset for asset in assets if include_deleted or not asset["deleted"]],
                      key=lambda a: a["created_at"], reverse=True)

    def register(self, payload: AssetInput, *, import_item_key: str | None = None) -> dict:
        from .media import fingerprint, probe, subtitle_tracks
        from .subtitles import import_warnings, parse_subtitles

        if import_item_key:
            for asset in self.list_assets(include_deleted=True):
                if asset.get("import_item_key") == import_item_key:
                    if asset["deleted"]:
                        raise DeletedAssetError()
                    return asset
        explicit_project = "project_id" in payload.model_fields_set
        if explicit_project and payload.project_id is not None:
            if not any(project["id"] == payload.project_id for project in self.organization.projects()):
                raise ValueError("项目不存在")
        source = Path(payload.video_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("视频路径必须指向文件")
        snapshot = source_snapshot(source)
        metadata = probe(source)
        tracks = subtitle_tracks(metadata)
        cues = []
        subtitle_diagnostics = {}
        sub_path = None
        container = None
        mode = payload.subtitle_mode
        fallback = None
        if mode in {"auto", "container"}:
            if mode == "auto" and recommended_subtitle_track(tracks) is None and payload.subtitle_stream_index is None:
                from .subtitle_detection import detect_subtitle_sources

                detection = detect_subtitle_sources(source, metadata=metadata)
                if detection["recommended_mode"] == "external":
                    mode = "external"
                    sub_path = Path(detection["recommended_subtitle_path"])
                elif tracks:
                    # Bitmap soft tracks cannot be read by OCR of the video
                    # frame; do not silently select that unrelated workflow.
                    select_subtitle_track(tracks)
                else:
                    if payload.subtitle_offset_ms:
                        raise ValueError("未检测到可用文字字幕，无法应用字幕时间偏移")
                    mode = "embedded"
                    fallback = detection["reason"] + "；仅在画面有烧录字幕时，可绑定字幕视觉模型后运行 OCR。音轨转写暂不支持。"
                    if detection["warnings"]:
                        fallback += " " + " ".join(detection["warnings"])
            else:
                track = select_subtitle_track(tracks, payload.subtitle_stream_index)
                container = self._extract_container(source, metadata, track, payload.subtitle_offset_ms)
                mode = "container"
        if mode == "external":
            sub_path = sub_path or Path(payload.subtitle_path).expanduser().resolve(strict=True)
            cues = parse_subtitles(sub_path, metadata["duration_ms"], payload.subtitle_offset_ms,
                                   diagnostics=subtitle_diagnostics)
            if not cues:
                raise ValueError("字幕文件没有有效字幕")
        checksum = fingerprint(source)
        if source_snapshot(source) != snapshot:
            raise ValueError("视频在导入期间被修改，请重新导入")
        with self.lock:
            if explicit_project and payload.project_id is not None:
                self.organization.get_project(payload.project_id)
            for asset in self.list_assets(include_deleted=True):
                if import_item_key and asset.get("import_item_key") == import_item_key:
                    if asset["deleted"]:
                        raise DeletedAssetError()
                    return asset
                if asset["fingerprint"] == checksum:
                    if asset["deleted"]:
                        raise DeletedAssetError()
                    raise DuplicateAssetError(asset["id"])
            asset_id = uid("asset")
            title = payload.title.strip() or source.stem
            identity = [title, payload.kind, payload.series, payload.season, payload.episode]
            work_id = "work_" + digest(identity)
            work = {"schema_version": SCHEMA, "id": work_id, "title": title, "kind": payload.kind,
                    "series": payload.series, "season": payload.season, "episode": payload.episode}
            asset = {**work, **metadata, "id": asset_id, "work_id": work_id,
                     "version": payload.version, "fingerprint": checksum, "file_name": source.name,
                     "subtitle_mode": mode, "subtitle_import_mode": payload.subtitle_mode,
                     "subtitle_offset_ms": payload.subtitle_offset_ms, "subtitle_tracks": tracks,
                     "subtitle_import_warnings": container["warnings"] if container else import_warnings(subtitle_diagnostics),
                     "created_at": now(), "status": "ready"}
            if explicit_project:
                asset["project_id"] = payload.project_id
                if payload.project_id is None:
                    asset["character_scope_id"] = "cast_" + asset_id
            if import_item_key:
                asset["import_item_key"] = import_item_key
            if fallback:
                asset["subtitle_fallback_reason"] = fallback
            # Complete all asset files in staging before the asset becomes
            # visible; failed extraction/storage must not publish a partial asset.
            with tempfile.TemporaryDirectory(prefix="import-", dir=self.root / "runtime") as temporary:
                directory = Path(temporary)
                if sub_path:
                    originals = directory / "subtitles" / "originals"
                    originals.mkdir(parents=True)
                    shutil.copy2(sub_path, originals / ("original" + sub_path.suffix.lower()))
                    cues = [{**c, "schema_version": SCHEMA, "asset_id": asset_id,
                             "id": "cue_" + digest([asset_id, c["id"]]), "source": "external"} for c in cues]
                    atomic_jsonl(directory / "subtitles" / "external.jsonl", cues)
                    atomic_json(directory / "subtitles" / "import-diagnostics.json", subtitle_diagnostics)
                active = {"schema_version": SCHEMA, "observations": {}, "subtitle_runs": {}}
                if container:
                    pointer = self._write_container(asset, container, payload.subtitle_offset_ms, directory=directory)
                    active["container_subtitles"] = pointer
                    asset.update(subtitle_stream_index=pointer["stream_index"], subtitle_track=pointer["track"])
                atomic_json(directory / "active-analysis.json", active)
                atomic_json(directory / "manifest.json", asset)
                atomic_json(self.root / "works" / work_id / "work.json", work)
                locations = read_json(self.root / "private" / "media-locations.json", {})
                locations[asset_id] = snapshot
                atomic_json(self.root / "private" / "media-locations.json", locations)
                os.replace(directory, self.asset_dir(asset_id))
        return self.get_asset(asset_id)

    @staticmethod
    def _extract_container(source: Path, metadata: dict, track: dict, offset_ms: int) -> dict:
        from .media import extract_subtitle_track
        from .subtitles import import_warnings, parse_container_subtitles

        with tempfile.TemporaryDirectory(prefix="scenerecall-subtitles-") as temporary:
            output = extract_subtitle_track(source, track["index"], Path(temporary) / "track", metadata=metadata)
            diagnostics = {}
            cues = parse_container_subtitles(output, metadata["duration_ms"], offset_ms, diagnostics=diagnostics)
            if not cues:
                raise ValueError("所选容器字幕轨没有可用的字幕文字")
            return {"track": track, "cues": cues, "suffix": output.suffix.lower(), "original": output.read_bytes(),
                    "diagnostics": diagnostics, "warnings": import_warnings(diagnostics)}

    def _write_container(self, asset: dict, extracted: dict, offset_ms: int, directory: Path | None = None) -> dict:
        asset_id, track, run_id = asset["id"], extracted["track"], uid("run")
        directory = directory or self.asset_dir(asset_id)
        relative = Path("analyses") / run_id / "subtitles.jsonl"
        original = Path("subtitles") / "originals" / (run_id + extracted["suffix"])
        original_path = directory / original
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_bytes(extracted["original"])
        cues = [{**cue, "schema_version": SCHEMA, "asset_id": asset_id, "run_id": run_id,
                 "id": "cue_" + digest([asset_id, "container", track["index"], cue["id"]]),
                 "source": "container", "source_file": original.name, "source_stream_index": track["index"],
                 "source_codec": track["codec"], "source_language": track.get("language"),
                 "source_title": track.get("title"), "offset_ms": offset_ms}
                for cue in extracted["cues"]]
        if any(not 0 <= cue["start_ms"] < cue["end_ms"] <= asset["duration_ms"] for cue in cues):
            raise ValueError("提取字幕超出资产原有时间轴，请检查视频版本")
        atomic_jsonl(directory / relative, cues)
        pointer = {"path": str(relative), "original_path": str(original), "run_id": run_id,
                   "stream_index": track["index"], "track": track, "offset_ms": offset_ms,
                   "warnings": extracted.get("warnings", []),
                   "extraction_version": 1}
        atomic_json(directory / "analyses" / run_id / "subtitle-import-diagnostics.json", extracted.get("diagnostics", {}))
        atomic_json(directory / "analyses" / run_id / "run.json", {
            "schema_version": SCHEMA, "id": run_id, "asset_id": asset_id,
            "stage": "subtitle", "source": "container", "status": "completed",
            "created_at": now(), "request_count": 0, "cost": 0,
            "subtitle_track": track, "subtitle_offset_ms": offset_ms, "subtitle_count": len(cues),
            "subtitle_import_warnings": extracted.get("warnings", [])})
        return pointer

    def extract_subtitles(self, asset_id: str, stream_index: int | None = None, offset_ms: int = 0) -> dict:
        from .media import probe, subtitle_tracks

        asset = self.get_asset(asset_id)
        if asset["subtitle_mode"] == "external":
            raise ValueError("外挂字幕资产请保留原字幕来源；提取容器轨适用于画面 OCR 或容器字幕资产")
        source = self.require_source(asset_id)
        snapshot = source_snapshot(source)
        metadata = probe(source)
        tracks = subtitle_tracks(metadata)
        track = select_subtitle_track(tracks, stream_index)
        extracted = self._extract_container(source, metadata, track, offset_ms)
        with self.lock:
            self.require_source(asset_id)
            if source_snapshot(source) != snapshot:
                raise ValueError("视频在提取期间被修改，请重新定位并校验原片")
            asset = self.get_asset(asset_id)
            pointer = self._write_container(asset, extracted, offset_ms)
            active = self.active(asset_id)
            # Keep old OCR runs and their annotations as history, but expose one
            # selected text track. Publish only after extraction has fully passed.
            active["container_subtitles"] = pointer
            atomic_json(self.asset_dir(asset_id) / "active-analysis.json", active)
            manifest = read_json(self.asset_dir(asset_id) / "manifest.json")
            manifest.update(subtitle_mode="container", subtitle_stream_index=track["index"], subtitle_track=track,
                            subtitle_tracks=tracks, subtitle_offset_ms=offset_ms,
                            subtitle_import_warnings=extracted.get("warnings", []))
            manifest.pop("subtitle_fallback_reason", None)
            atomic_json(self.asset_dir(asset_id) / "manifest.json", manifest)
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

    def _container_pointer(self, asset_id: str) -> dict | None:
        pointer = self.active(asset_id).get("container_subtitles")
        if pointer is None:
            return None
        required = {"path", "original_path", "run_id", "stream_index", "track", "offset_ms"}
        if (not isinstance(pointer, dict) or required - pointer.keys()
                or not isinstance(pointer["run_id"], str)
                or type(pointer["stream_index"]) is not int or pointer["stream_index"] < 0
                or type(pointer["offset_ms"]) is not int or not isinstance(pointer["track"], dict)
                or pointer["track"].get("index") != pointer["stream_index"]):
            raise ValueError("容器字幕轨来源信息无效")
        run_id = safe_id(pointer["run_id"])
        directory = self.asset_dir(asset_id)
        path = confined_path(directory, pointer["path"])
        original = confined_path(directory, pointer["original_path"])
        if (pointer["path"] != f"analyses/{run_id}/subtitles.jsonl"
                or pointer["original_path"] != f"subtitles/originals/{run_id}{original.suffix}"
                or original.suffix not in {".ass", ".srt"}
                or not path.is_file() or not original.is_file()):
            raise ValueError("容器字幕记录或原始提取文件缺失，或引用无效")
        return pointer

    def observations(self, asset_id: str) -> list[dict]:
        directory = self.asset_dir(asset_id)
        return sorted([read_json(confined_path(directory, p)) for p in self.active(asset_id)["observations"].values()],
                      key=lambda r: r["start_ms"])

    def subtitles(self, asset_id: str) -> list[dict]:
        directory = self.asset_dir(asset_id)
        container = self._container_pointer(asset_id)
        if container:
            path = confined_path(directory, container["path"])
            if not path.is_file():
                raise ValueError("容器字幕记录文件缺失")
            return sorted(read_jsonl(path), key=lambda cue: (cue["start_ms"], cue.get("language", "und")))
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
        self.frame_paths(asset_id, record["evidence_frame_ids"])
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
            if self.get_asset(asset_id)["subtitle_mode"] != "embedded":
                raise ValueError("字幕来源已切换为文字字幕，旧画面 OCR 任务不能覆盖当前字幕轨")
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
        return self.frame_paths(asset_id, [frame_id])[frame_id]

    def frame_paths(self, asset_id: str, frame_ids: list[str], *, registry: dict | None = None) -> dict[str, Path]:
        """Resolve one operation's frames using one registry snapshot, never a persistent cache."""
        directory = self.asset_dir(asset_id)
        if registry is None:
            registry = read_json(directory / "frames" / "registry.json", {})
        paths = {}
        for frame_id in frame_ids:
            value = registry.get(safe_id(frame_id))
            if not value:
                raise FileNotFoundError("证据帧不存在")
            path = confined_path(directory, value["path"])
            if not path.is_file():
                raise FileNotFoundError("证据帧文件不存在")
            paths[frame_id] = path
        return paths

    def validate_keyframe_plan(self, asset_id: str, plan: dict, expected_id: str | None = None) -> dict:
        """Validate saved plans without opening the source video or invoking a model."""
        from fractions import Fraction
        asset = self.get_asset(asset_id, include_deleted=True)
        try:
            if (not isinstance(plan, dict) or plan.get("asset_id") != asset_id
                    or plan.get("fingerprint") != asset["fingerprint"]
                    or (expected_id is not None and plan.get("id") != expected_id)
                    or plan.get("plan_id", plan.get("id")) != plan.get("id")
                    or not isinstance(plan.get("algorithm"), str) or not plan["algorithm"]):
                raise ValueError("关键帧计划身份或来源无效")
            safe_id(plan["id"])
            start, end = plan["start_ms"], plan["end_ms"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= asset["duration_ms"]:
                raise ValueError("关键帧计划时间范围无效")
            if plan.get("status") == "pending":
                if any(plan.get(key) for key in ("shots", "frames", "batches")):
                    raise ValueError("未完成的关键帧计划不能包含有效帧引用")
                return plan
            if plan.get("status") != "completed":
                raise ValueError("关键帧计划状态无效")
            if any(not isinstance(plan.get(key), list) or not plan[key] for key in ("shots", "frames", "batches")):
                raise ValueError("关键帧计划缺少镜头、帧或分析批次")
            shots = {}
            last_end = start
            for shot in plan["shots"]:
                identity = safe_id(shot["id"])
                a, b = shot["start_ms"], shot["end_ms"]
                if (identity in shots or type(a) is not int or type(b) is not int
                        or not last_end <= a < b <= end):
                    raise ValueError("关键帧镜头范围无效")
                shots[identity] = shot
                last_end = b
            frames = {}
            previous_time = None
            stored_origin = asset.get("start_time_seconds", str(asset.get("start_time_ms", 0) / 1000))
            origin = Fraction(plan.get("source_start_time_seconds", stored_origin))
            if "start_time_seconds" in asset:
                if origin != Fraction(asset["start_time_seconds"]):
                    raise ValueError("关键帧计划与视频的精确起始时间不一致")
            elif round(origin * 1000) != asset.get("start_time_ms", 0):
                raise ValueError("关键帧计划与视频的起始时间不一致")
            registry = read_json(self.asset_dir(asset_id) / "frames" / "registry.json", {})
            self.frame_paths(asset_id, [frame["id"] for frame in plan["frames"]], registry=registry)
            for frame in plan["frames"]:
                identity = safe_id(frame["id"])
                if identity in frames or frame["shot_id"] not in shots:
                    raise ValueError("关键帧 ID 重复或镜头引用无效")
                for field in ("time_base", "source_time_base"):
                    base = frame[field]
                    if not isinstance(base, list) or len(base) != 2 or any(type(number) is not int or number <= 0 for number in base):
                        raise ValueError("关键帧时间基无效")
                if any(type(frame[field]) is not int for field in ("pts", "source_pts", "at_ms")):
                    raise ValueError("关键帧时间戳无效")
                timestamp = frame["pts"] * Fraction(*frame["time_base"])
                source_time = frame["source_pts"] * Fraction(*frame["source_time_base"])
                shot = shots[frame["shot_id"]]
                if (timestamp != source_time - origin or round(timestamp * 1000) != frame["at_ms"]
                        or not Fraction(start, 1000) <= timestamp < Fraction(end, 1000)
                        or not shot["start_ms"] <= frame["at_ms"] < shot["end_ms"]
                        or (previous_time is not None and timestamp <= previous_time)):
                    raise ValueError("关键帧时间戳与镜头不一致")
                if registry[identity].get("at_ms") != frame["at_ms"]:
                    raise ValueError("关键帧计划与已保存证据帧不一致")
                frames[identity] = frame
                previous_time = timestamp
            referenced, batch_ids = [], set()
            last_end = start
            for batch in plan["batches"]:
                identity = safe_id(batch["id"])
                a, b = batch["start_ms"], batch["end_ms"]
                shot = shots.get(batch["shot_id"])
                ids = batch["frame_ids"]
                if (identity in batch_ids or type(a) is not int or type(b) is not int or not shot
                        or not last_end <= a < b <= end
                        or not shot["start_ms"] <= a < b <= shot["end_ms"]
                        or not isinstance(ids, list) or not 1 <= len(ids) <= 8):
                    raise ValueError("关键帧分析批次无效")
                for frame_id in ids:
                    frame = frames.get(frame_id)
                    if frame is None or frame["shot_id"] != batch["shot_id"] or not a <= frame["at_ms"] < b:
                        raise ValueError("关键帧批次引用了无效证据")
                referenced.extend(ids)
                batch_ids.add(identity)
                last_end = b
            if referenced != list(frames):
                raise ValueError("关键帧批次遗漏、重复或打乱了证据帧")
            return plan
        except (KeyError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError("关键帧计划格式无效") from exc

    def annotations(self, asset_id: str) -> list[dict]:
        from .scenes import SceneStore
        annotations = sorted([read_json(p) for p in (self.asset_dir(asset_id) / "annotations").glob("*.json")], key=lambda a: a["updated_at"])
        records = {r["id"]: r for r in self.observations(asset_id) + self.subtitles(asset_id) + SceneStore(self).list(asset_id)}
        for annotation in annotations:
            record = records.get(annotation["record_id"])
            annotation["needs_reassociation"] = record is None or bool(annotation.get("character_name") and annotation.get("base_run_id") != record.get("run_id"))
        return annotations

    def annotate(self, asset_id: str, payload: dict) -> dict:
        from .scenes import SceneStore
        self.get_asset(asset_id)
        records = {r["id"]: r for r in self.observations(asset_id) + self.subtitles(asset_id) + SceneStore(self).list(asset_id)}
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
            def semantic(value):
                return (value.get("summary"), (value.get("entity_id"), value.get("character_name"), value.get("aliases"))
                        if value.get("character_name") else None)
            if semantic(previous) != semantic(annotation):
                SceneStore(self).mark_dependents_stale(asset_id, record["start_ms"], "前序证据的人工解释已变化")
            return annotation

    def records(self, asset_id: str | None = None) -> list[dict]:
        from .characters import CharacterStore
        from .scenes import SceneStore
        characters = CharacterStore(self)
        result = []
        for asset in [self.get_asset(asset_id)] if asset_id else self.list_assets():
            annotations = self.annotations(asset["id"])
            observations = characters.decorate_observations(asset["id"], self.observations(asset["id"]))
            scenes = characters.decorate_observations(asset["id"], SceneStore(self).list(asset["id"]))
            # Original observations stay immutable and collectible, but are not
            # duplicated as search hits where a new scene plan replaced them.
            observations = [row for row in observations if not any(
                scene["start_ms"] < row["end_ms"] and scene["end_ms"] > row["start_ms"] for scene in scenes)]
            for record in observations + scenes + self.subtitles(asset["id"]):
                is_visual = "summary" in record
                is_scene = "chunk_id" in record
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
                if is_scene:
                    description += " " + " ".join(fact["text"] for field in ("visual_facts", "dialogue_facts", "inferences")
                                                  for fact in record.get(field, []))
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
                               **{key: asset.get(key) for key in ("project_id", "project_name", "folder_id", "folder_ids", "tags")},
                               "work_id": asset["work_id"], "title": asset["title"], "series": asset.get("series", ""),
                               "season": asset.get("season"), "episode": asset.get("episode"), "version": asset["version"],
                               "kind": "scene" if is_scene else "visual" if is_visual else "subtitle", "start_ms": record["start_ms"],
                               "end_ms": record["end_ms"], "text": description.strip(), "summary": text,
                               "subtitle_text": text if not is_visual else None, "language": record.get("language"),
                               "thumbnail": frames[0] if frames else None, "evidence_frame_ids": frames,
                               **{key: record[key] for key in ("source_stream_index", "source_codec", "source_language",
                                   "source_title", "offset_ms", "timing_clipped", "source_start_ms", "source_end_ms")
                                  if key in record},
                               "source": record.get("source", "scene" if is_scene else "vision"), "review_status": "user_confirmed" if user_confirmed else record.get("review_status", "unreviewed"),
                               "entities": entities, "events": record.get("events", []),
                               "character": " ".join(dict.fromkeys(names)), "favorite": favorite, "note": note,
                               "run_id": record.get("run_id"), "uncertainties": record.get("uncertainties", []),
                               **({key: record.get(key) for key in ("chunk_id", "shot_ids", "frame_ids", "subtitle_ids",
                                   "visual_facts", "dialogue_facts", "inferences", "representative_frame_ids",
                                   "understanding_level", "needs_deep", "reason", "provenance", "dependencies",
                                   "context_stale", "stale_reasons")} if is_scene else {})})
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
                        record = read_json(base / "scenes" / f"{safe_id(rid)}.deep.json")
                    if record is None:
                        record = read_json(base / "scenes" / f"{safe_id(rid)}.json")
                    if record is None:
                        record = next((c for c in read_jsonl(base / "subtitles.jsonl") if c["id"] == rid), None)
                if record is None:
                    continue
                if "summary" in record:
                    from .characters import CharacterStore
                    record = CharacterStore(self).decorate_observations(asset["id"], [record])[0]
                text = annotation.get("summary", record.get("summary", record.get("text", "")))
                frames = record.get("evidence_frame_ids", [])
                result.append({"id": rid, "record_id": rid, "asset_id": asset["id"], "work_id": asset["work_id"],
                               **{key: asset.get(key) for key in ("project_id", "project_name", "folder_id", "folder_ids", "tags")},
                               "title": asset["title"], "series": asset.get("series"), "season": asset.get("season"),
                               "episode": asset.get("episode"), "version": asset["version"], "start_ms": record["start_ms"],
                               "end_ms": record["end_ms"], "kind": "scene" if "chunk_id" in record else "visual" if "summary" in record else "subtitle",
                               "text": text, "summary": text, "note": annotation.get("note", ""), "favorite": True,
                               "source": record.get("source", "vision"), "review_status": "historical",
                               "is_historical": True, "run_id": run_id, "evidence_frame_ids": frames,
                               "entities": record.get("entities", []), "events": record.get("events", []),
                               "thumbnail": frames[0] if frames else None,
                               **({key: record.get(key) for key in ("chunk_id", "visual_facts", "dialogue_facts", "inferences",
                                   "subtitle_ids", "frame_ids", "understanding_level", "provenance")}
                                  if "chunk_id" in record else {})})
        return result

    def backup(self, include_media: bool = False) -> dict:
        backup_id = uid("backup")
        target = self.root / "runtime" / "backups" / f"{backup_id}.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            self.organization.migrate()
            for top in ("library.json", "works", "assets", "collections", "organization"):
                item = self.root / top
                for path in [item] if item.is_file() else item.rglob("*"):
                    if path.is_file() and not path.is_symlink() and path.suffix != ".tmp" and "/proxy/" not in str(path):
                        archive.write(path, path.relative_to(self.root))
            if include_media:
                for asset in self.list_assets(include_deleted=True):
                    if asset["deleted"] and not asset["source_available"]:
                        continue
                    path = self.source_path(asset["id"]) if asset["deleted"] else self.require_source(asset["id"])
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
                    if relative.parts[0] not in {"library.json", "works", "assets", "collections", "media", "organization"}:
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
            restored.organization.validate()
            for manifest in (staging / "assets").glob("*/manifest.json"):
                if read_json(manifest).get("id") != manifest.parent.name:
                    raise ValueError("资产清单 ID 与目录不一致")
            for asset in restored.list_assets(include_deleted=True):
                asset_id = safe_id(asset["id"])
                if not (staging / "assets" / asset_id / "manifest.json").is_file():
                    raise ValueError("资产清单 ID 与目录不一致")
                registry = read_json(restored.asset_dir(asset_id) / "frames" / "registry.json", {})
                restored.frame_paths(asset_id, list(registry), registry=registry)
                for observation in restored.observations(asset_id):
                    checked = Observation.model_validate(observation)
                    if checked.asset_id != asset_id or checked.end_ms > asset["duration_ms"]:
                        raise ValueError("备份识别记录时间轴无效")
                    restored.frame_paths(asset_id, checked.evidence_frame_ids, registry=registry)
                for cue in restored.subtitles(asset_id):
                    if not 0 <= cue["start_ms"] < cue["end_ms"] <= asset["duration_ms"]:
                        raise ValueError("备份字幕时间轴无效")
                container = restored._container_pointer(asset_id)
                if container:
                    cues = restored.subtitles(asset_id)
                    if not cues or any(cue.get("asset_id") != asset_id or cue.get("source") != "container"
                                       or cue.get("run_id") != container["run_id"]
                                       or cue.get("source_stream_index") != container["stream_index"]
                                       or not isinstance(cue.get("text"), str) or not cue["text"].strip()
                                       for cue in cues):
                        raise ValueError("备份容器字幕与所选来源不一致")
                # Retired OCR pointers remain in the graph after switching to a
                # text track; they must still be confined before any merge.
                for value in restored.active(asset_id).get("subtitle_runs", {}).values():
                    old_path = confined_path(restored.asset_dir(asset_id), value["path"])
                    for cue in read_jsonl(old_path):
                        if not 0 <= cue["start_ms"] < cue["end_ms"] <= asset["duration_ms"]:
                            raise ValueError("备份历史字幕时间轴无效")
                for path in restored.asset_dir(asset_id).glob("keyframe-plans/*.json"):
                    restored.validate_keyframe_plan(asset_id, read_json(path), path.stem)
                from .scenes import SceneStore
                SceneStore(restored).validate_all(asset_id)
            from .characters import CharacterStore
            CharacterStore(restored).validate_all()
            paths = [p for p in staging.rglob("*") if p.is_file() and p.name != "library.json"
                     and p != restored.organization.path]
            from .media import fingerprint
            for path in (staging / "media").glob("*"):
                asset = restored.get_asset(path.stem, include_deleted=True)
                if fingerprint(path) != asset["fingerprint"]:
                    raise ValueError("备份中的视频内容与资产指纹不一致")
            with self.lock:
                merged_organization = self.organization.merge_catalog(restored.organization._read())
                for path in paths:
                    target = confined_path(self.root, str(path.relative_to(staging)))
                    if target.exists() and (not target.is_file() or not same_contents(target, path)):
                        raise ValueError(f"恢复冲突，未覆盖现有文件: {path.relative_to(staging)}")
                locations = read_json(self.root / "private" / "media-locations.json", {})
                for path in (self.root / "media").glob("*"):
                    if path.stem.startswith("asset_"):
                        locations[path.stem] = source_snapshot(path)
                for source in (staging / "media").glob("*"):
                    target = self.root / "media" / source.name
                    snapshot = source_snapshot(target if target.exists() else source)
                    locations[source.stem] = {**snapshot, "path": str(target)}
                self.organization.commit_restore(staging, paths, merged_organization, locations)
            return {"restored_files": len(paths), "message": "资料已恢复；元数据备份中的原视频可能需要重新定位"}
        finally:
            pending = read_json(self.organization.journal, {})
            if pending.get("staging") != str(staging.relative_to(self.root)):
                shutil.rmtree(staging, ignore_errors=True)
