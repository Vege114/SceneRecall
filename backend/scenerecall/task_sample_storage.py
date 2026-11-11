"""Release reproducible sampling outputs after a successful analysis or scan."""
from collections import Counter
import hashlib
from pathlib import Path
import stat
import tempfile

from .frame_retention import protected_sample_frames
from .library import confined_path, read_json, safe_id
from .record_previews import preserve_frames
from .sample_frames import restore_batch, source_identity, validate_recipe, verify_source


def _plain(info):
    return not (stat.S_ISLNK(info.st_mode) or
                getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def finish_task_samples(library, asset_id, checkpoint):
    """Only the sampling producer can mark a new file eligible for this lifecycle.

    Exact PTS images have a source-bound recipe and a hash from the encoder's
    original output. Uniform/OCR sampling can depend on nearby seeks in a
    batch, so prove that *independent* reconstruction matches before unlinking.
    """
    asset = library.asset_dir(asset_id)
    directory = asset / "frames"
    registry_path = directory / "registry.json"
    result = {"candidate_count": 0, "released_count": 0, "released_bytes": 0, "preserved_count": 0}
    registry = read_json(registry_path, {})
    candidates = {}
    for identity, entry in registry.items():
        if (entry.get("sample_generated") is True and entry.get("sample") and
                entry.get("path") == f"frames/{safe_id(identity)}.jpg"):
            path = directory / f"{identity}.jpg"
            if path.is_file():
                candidates[identity] = path
    if not candidates:
        return result
    result["candidate_count"] = len(candidates)
    try:
        if not all(_plain(p.lstat()) for p in (library.root / "assets", asset, directory, registry_path)):
            return result
        source, metadata = verify_source(library, asset_id)
        # A legacy/custom media adapter may return probe data without identifying
        # the file or selected video stream. Its cached images are independent
        # evidence; never re-extract or remove them as part of task completion.
        if not metadata.get("_source_identity") or type(metadata.get("video_stream_index")) is not int:
            return result
        # A second registry identity can use this exact file (including via a
        # symlink). Preserve it even if the second identity lacks a recipe.
        aliases = Counter(confined_path(asset, row["path"]) for row in registry.values() if not row.get("alias_of"))
        verified = {}
        for index, (identity, path) in enumerate(candidates.items()):
            if index % 64 == 0:
                checkpoint()
            try:
                entry = registry[identity]
                info = path.lstat()
                if not _plain(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or aliases[path] != 1:
                    continue
                validate_recipe(asset_id, metadata["_sample_fingerprint"], identity, entry)
                stamp = source_identity(path)
                with path.open("rb") as archive:
                    if hashlib.file_digest(archive, "sha256").hexdigest() != entry["sample"]["jpeg_sha256"]:
                        continue
                if entry["sample"].get("kind") == "uniform":
                    with tempfile.TemporaryDirectory(prefix="scenerecall-verify-sample-") as work:
                        restore_batch(source, metadata, [(identity, entry)], Path(work))
                verified[identity] = (stamp, info.st_size)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if not verified:
            return result
        with library.lock:
            checkpoint()
            if source_identity(source) != metadata["_sample_identity"] or library.require_source(asset_id) != source:
                return result
            current = read_json(registry_path, {})
            if current != registry:
                # New aliases, pins, or registrations may have appeared while verifying.
                return result
            reasons = protected_sample_frames(library, asset_id)
            for identity, entry in registry.items():
                if entry.get("alias_of") and identity in reasons:
                    reasons.setdefault(entry["alias_of"], set()).update(reasons[identity])
            manual = [identity for identity in verified if reasons.get(identity, set()) - {"frozen_path"}]
            if manual:
                preserve_frames(library, asset_id, manual)
                result["preserved_count"] = len(manual)
            for index, (identity, (stamp, size)) in enumerate(verified.items()):
                if index % 64 == 0:
                    checkpoint()
                if "frozen_path" in reasons.get(identity, ()):
                    continue
                path = candidates[identity]
                try:
                    if not _plain(path.lstat()) or source_identity(path) != stamp:
                        continue
                    path.unlink()
                except OSError:
                    continue
                result["released_count"] += 1
                result["released_bytes"] += size
    except (OSError, ValueError, KeyError, TypeError):
        # An unavailable source or unreadable archive must retain independent
        # files. Successfully published index receipts remain valid.
        return result
    return result
