"""Automatic timeline previews and independently preserved human evidence."""
from collections import OrderedDict
import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
import threading
import uuid

from PIL import Image

from .file_io import atomic_replace
from .library import atomic_json, confined_path, read_json, safe_id
from .media import MediaError, extract_frame


# Only small timeline thumbnails live here. Exact sampled frames have their
# own resolver. Keep expensive FFmpeg invocations out of the library lock.
_decode_slots = threading.BoundedSemaphore(2)
_thumbnail_lock = threading.Lock()
_thumbnails = OrderedDict()
_THUMBNAIL_LIMIT = 32


def _publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _timeline_path(library, asset_id, at_ms):
    return confined_path(library.asset_dir(asset_id), f"preserved-frames/time-{at_ms}.jpg")


def timeline_preview(library, asset_id, at_ms):
    from .sample_frames import source_identity, verify_source
    asset = library.get_asset(asset_id)
    if type(at_ms) is not int or not 0 <= at_ms < asset["duration_ms"]:
        raise ValueError("预览时间超出素材范围")
    pinned = _timeline_path(library, asset_id, at_ms)
    if pinned.is_file():
        return pinned.read_bytes()
    path, metadata = verify_source(library, asset_id)
    stamp = metadata["_sample_identity"]
    key = (stamp, asset["fingerprint"], metadata["start_time_ms"], metadata["video_stream_index"], at_ms)
    with _decode_slots:
        if source_identity(path) != stamp:
            raise ValueError("原片在生成预览期间发生变化")
        with _thumbnail_lock:
            if key in _thumbnails:
                _thumbnails.move_to_end(key)
                return _thumbnails[key]
        with tempfile.TemporaryDirectory(prefix="scenerecall-preview-") as work:
            image_path = Path(work) / "preview.jpg"
            extract_frame(path, at_ms, image_path, metadata=metadata)
            with Image.open(image_path) as source, BytesIO() as stream:
                source.thumbnail((640, 360))
                source.convert("RGB").save(stream, format="JPEG", quality=85)
                data = stream.getvalue()
        if source_identity(path) != stamp:
            raise ValueError("原片在生成预览期间发生变化")
        with _thumbnail_lock:
            _thumbnails[key] = data
            _thumbnails.move_to_end(key)
            if len(_thumbnails) > _THUMBNAIL_LIMIT:
                _thumbnails.popitem(last=False)
    return data


def preserve_frames(library, asset_id, identities):
    """Preserve registered evidence in its original encoding before acknowledging a human edit."""
    directory = library.asset_dir(asset_id)
    registry_path = directory / "frames" / "registry.json"
    with library.lock:
        registry = read_json(registry_path, {})
        selected = []
        for identity in dict.fromkeys(identities):
            entry = registry.get(safe_id(identity))
            if entry and Path(entry["path"]).suffix.lower() in {".jpg", ".jpeg"}:
                selected.append(identity)
        if not selected:
            return
        paths = library.frame_paths(asset_id, selected, registry=registry)
        changed = False
        for identity in selected:
            entry = registry[identity]
            data = paths[identity].read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if entry.get("pinned_sha256") == sha:
                continue
            target = confined_path(directory, f"preserved-frames/{identity}{Path(entry['path']).suffix}")
            _publish(target, data)
            entry["pinned_sha256"] = sha
            changed = True
        if changed:
            atomic_json(registry_path, registry)


def preserve_record(library, asset_id, record, preview_frame=None):
    frames = record.get("evidence_frame_ids") or record.get("frame_ids") or []
    cover = preview_frame or record.get("thumbnail") or next(iter(frames), None)
    if cover:
        preserve_frames(library, asset_id, [cover])
        entry = library.frame_entry(asset_id, cover)
        if entry:
            return {"preview_frame_id": cover, "preview_at_ms": entry["at_ms"]}
    elif type(record.get("start_ms")) is int:
        at_ms = record["start_ms"]
        target = _timeline_path(library, asset_id, at_ms)
        if not target.is_file():
            try:
                data = timeline_preview(library, asset_id, at_ms)
            except (FileNotFoundError, MediaError):
                # Offline text annotations still carry their original cue and time.
                return {}
            _publish(target, data)
    return {}


def bookmark_frame(record, frame_id, member_ids, records):
    """Bind a bookmark cover to an actual source of this result, including a verified merged window."""
    def contains(row):
        return frame_id in (row.get("evidence_frame_ids") or row.get("frame_ids") or [])

    if contains(record):
        return frame_id
    if member_ids and record["id"] in member_ids and all(mid in records for mid in member_ids):
        from .search import merge_visual_candidates
        members = [records[mid] for mid in dict.fromkeys(member_ids)]
        merged = merge_visual_candidates(members)
        if len(merged) == 1 and set(merged[0].get("member_ids", [])) == set(member_ids) and contains(merged[0]):
            return frame_id
    raise ValueError("收藏画面已不属于当前记录，请刷新结果")
