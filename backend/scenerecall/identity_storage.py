"""Lossless whitespace compaction for offline visual snapshots and retrieval receipts."""
import io
import json
import os
import stat
import uuid

from .file_io import atomic_replace


def compact_identity_snapshot(path, asset_id, *, apply=False):
    """Return the removable byte count for one file while holding the offline maintenance lock on writes."""
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("快照不是独立普通文件")
    original = path.read_bytes()
    value = json.loads(original.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("人物快照结构无效")
    visual = value.get("visual") if path.name == "retrieval-local-identity.json" else value
    if (not isinstance(visual, dict) or visual.get("asset_id") != asset_id
            or not isinstance(visual.get("regions"), list)):
        raise ValueError("人物快照与素材不匹配")
    # A valid JSON string cannot contain a literal newline. Trimming only the
    # edges of physical lines preserves every token, including number spelling,
    # Unicode escapes, duplicate object keys and whitespace inside strings.
    with io.BytesIO() as output:
        for line in io.BytesIO(original):
            output.write(line.strip(b" \t\r\n"))
        packed = output.getvalue()
    saved = len(original) - len(packed)
    if not saved:
        return 0
    if json.loads(packed) != value:
        raise ValueError("整理前后的 JSON 内容不一致")
    if apply:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(packed)
                handle.flush()
                os.fsync(handle.fileno())
            before_publish = path.stat()
            if (before_publish.st_ino, before_publish.st_size, before_publish.st_mtime_ns) != (
                    info.st_ino, info.st_size, info.st_mtime_ns):
                raise ValueError("快照已变化，请重新核验")
            atomic_replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return saved
