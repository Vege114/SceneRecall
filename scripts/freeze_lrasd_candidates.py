#!/usr/bin/env python3
"""Freeze validated candidate snapshots without overwriting an existing output.

Only Python's standard library is required. Concurrent appenders are supported
through complete-byte snapshots; malformed complete JSON lines always fail.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import wave
import zipfile

ASSET_FIELDS = {"visual_path": "visual", "teacher_visual_path": "teacher_visual", "audio_path": "audio"}
SEMANTIC_FIELDS = (
    "sample_id", "window_id", "split", "group", "evaluation_group", "movie", "clip", "track",
    "start_frame", "end_frame", "fps", "frames", "labels", "mask", "audio_start_seconds",
    "preprocessing", "crop_type", "teacher_crop_type", "normalized_body_boxes_xyxy",
    "normalized_face_boxes_xyxy",
)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def snapshot(path):
    for _ in range(4):
        data = path.read_bytes()
        if not data or data.endswith(b"\n"):
            return data
        time.sleep(.05)
    raise ValueError(f"Manifest has an unfinished final line after retries: {path}")


def npy_shape(stream):
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("Not a NumPy array")
    version = stream.read(2)
    size = 2 if version[0] == 1 else 4
    length = struct.unpack("<H" if size == 2 else "<I", stream.read(size))[0]
    if length > 65536:
        raise ValueError("Oversized NumPy header")
    header = ast.literal_eval(stream.read(length).decode("utf-8" if version[0] == 3 else "latin1").strip())
    return tuple(header["shape"]), header["descr"]


def verify_visual(path, frames):
    with zipfile.ZipFile(path) as archive:
        with archive.open("visual.npy") as stream:
            shape, dtype = npy_shape(stream)
        with archive.open("boxes_xyxy_normalized.npy") as stream:
            box_shape, _ = npy_shape(stream)
    if shape != (frames, 112, 112) or dtype not in ("|u1", "<u1") or box_shape != (frames, 4):
        raise ValueError(f"Visual shape/dtype/boxes mismatch in {path}: {shape}, {dtype}, {box_shape}")


def verify_audio(path, frames, offset):
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError(f"Audio is not mono 16kHz PCM16: {path}")
        wanted = frames * 640
        if stream.getnframes() < round(offset * 16000) + wanted or (offset == 0 and stream.getnframes() != wanted):
            raise ValueError(f"Audio sample count does not match the window: {path}")


def exclusive_write(path, data):
    # O_EXCL remains safe if another process creates the output after preflight.
    owned = False
    try:
        with path.open("xb") as stream:
            owned = True
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if owned:
            path.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    inputs = list(dict.fromkeys(path.resolve() for path in args.manifest))
    output = args.output.resolve()
    audit_path = args.audit.resolve() if args.audit else output.with_suffix(".audit.json")
    if output == audit_path or output.exists() or audit_path.exists():
        raise FileExistsError("Frozen output/audit already exists, or both paths are identical")
    if output in inputs or audit_path in inputs:
        raise ValueError("Output paths cannot be an input manifest")
    kept, signatures, origins = {}, {}, {}
    asset_cache, visual_cache, audio_cache = {}, set(), set()
    group_splits, source_splits = defaultdict(set), defaultdict(set)
    input_audit, duplicates = [], []
    frames_expected = None
    for manifest in inputs:
        data = snapshot(manifest)
        text = data.decode("utf-8-sig")
        input_record = {"path": str(manifest), "sha256": digest_bytes(data), "bytes": len(data),
                        "rows": 0, "retained_first_occurrences": 0, "consistent_duplicate_rows": 0}
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)  # Never discard a malformed complete line.
            location = f"{manifest}:{line_number}"
            required = set(ASSET_FIELDS) | {"sample_id", "split", "group", "movie", "clip", "track", "start_frame",
                                         "end_frame", "fps", "labels", "audio_start_seconds", "source_video_path", "source_start_seconds", "asset_sha256"}
            if not isinstance(row, dict) or required - row.keys():
                raise ValueError(f"Missing candidate fields: {location}")
            sample_id = row["sample_id"]
            if any(not isinstance(row[key], str) or not row[key].strip() for key in ("sample_id", "group", "movie", "clip", "track", "source_video_path")):
                raise ValueError(f"Empty/non-string candidate identity: {location}")
            if row["split"] not in ("train", "val", "test") or row["fps"] != 25:
                raise ValueError(f"Invalid split or fps: {location}")
            if any(type(row[key]) is not int or row[key] < 0 for key in ("start_frame", "end_frame")):
                raise ValueError(f"Invalid frame coordinates: {location}")
            frames = row["end_frame"] - row["start_frame"]
            if frames not in (50, 100) or len(row["labels"]) != frames or row.get("frames", frames) != frames:
                raise ValueError(f"Invalid 50/100-frame candidate: {location}")
            if frames_expected is None:
                frames_expected = frames
            if frames != frames_expected:
                raise ValueError("Cannot freeze different window lengths together")
            if any(type(label) is not int or label not in (-1, 0, 1) for label in row["labels"]):
                raise ValueError(f"Invalid labels: {location}")
            start = row["source_start_seconds"]
            offset = row["audio_start_seconds"]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in (start, offset)):
                raise ValueError(f"Invalid source/audio time: {location}")
            if abs(start - row["start_frame"] / 25) > 1e-7:
                raise ValueError(f"Source time disagrees with global frame coordinate: {location}")
            for field in ("normalized_body_boxes_xyxy", "normalized_face_boxes_xyxy"):
                boxes = row.get(field)
                if not isinstance(boxes, list) or len(boxes) != frames or any(not isinstance(box, list) or len(box) != 4 for box in boxes):
                    raise ValueError(f"Invalid per-frame boxes: {location}")
            group_splits[row["group"]].add(row["split"])
            if row.get("evaluation_group", row["group"]) != row["group"]:
                raise ValueError(f"Evaluation group disagrees with group: {location}")
            source_path = str(Path(row["source_video_path"]).resolve())
            source_hash = row.get("source_sha256")
            source_keys = ["path:" + source_path]
            if source_hash:
                source_keys.append("sha256:" + source_hash)
            for source_key in source_keys:
                source_splits[source_key].add(row["split"])
            if len(group_splits[row["group"]]) != 1 or any(len(source_splits[key]) != 1 for key in source_keys):
                raise ValueError(f"Source or group leaks across splits: {location}")
            actual_hashes = {}
            for field, hash_key in ASSET_FIELDS.items():
                asset = Path(row[field])
                asset = (asset if asset.is_absolute() else manifest.parent / asset).resolve(strict=True)
                if asset not in asset_cache:
                    digest = hashlib.sha256()
                    with asset.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    asset_cache[asset] = {"sha256": digest.hexdigest(), "bytes": asset.stat().st_size}
                actual = asset_cache[asset]["sha256"]
                if row["asset_sha256"].get(hash_key) != actual:
                    raise ValueError(f"Asset SHA256 mismatch ({field}): {location}")
                actual_hashes[hash_key] = actual
                row[field] = str(asset)
                if field == "audio_path":
                    key = (asset, frames, offset)
                    if key not in audio_cache:
                        verify_audio(asset, frames, offset)
                        audio_cache.add(key)
                elif (asset, frames) not in visual_cache:
                    verify_visual(asset, frames)
                    visual_cache.add((asset, frames))
            semantics = {field: row.get(field) for field in SEMANTIC_FIELDS}
            semantics.update({"frames": frames, "fps": 25, "source_identity": "sha256:" + source_hash if source_hash else "path:" + source_path,
                              "source_start_seconds": round(start, 8), "asset_sha256": actual_hashes})
            signature = canonical(semantics)
            input_record["rows"] += 1
            if sample_id in kept:
                if signatures[sample_id] != signature:
                    previous = json.loads(signatures[sample_id])
                    changed = [field for field in semantics if previous.get(field) != semantics[field]]
                    raise ValueError(f"Conflicting duplicate {sample_id} at {location}; first {origins[sample_id]}; fields {changed}")
                duplicates.append({"sample_id": sample_id, "first": origins[sample_id], "duplicate": location,
                                   "semantic_sha256": digest_bytes(signature.encode("utf-8"))})
                input_record["consistent_duplicate_rows"] += 1
            else:
                kept[sample_id], signatures[sample_id], origins[sample_id] = row, signature, location
                input_record["retained_first_occurrences"] += 1
        input_audit.append(input_record)
    if not kept:
        raise ValueError("No complete candidate rows to freeze")
    split_order = {"train": 0, "val": 1, "test": 2}
    rows = sorted(kept.values(), key=lambda row: (split_order[row["split"]], row["group"], row["source_start_seconds"], row["track"], row["sample_id"]))
    output_bytes = ("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)).encode("utf-8")
    split_counts = {}
    for split in split_order:
        selected = [row for row in rows if row["split"] == split]
        split_counts[split] = {"tracks": len(selected), "distinct_windows": len({row.get("window_id", row["sample_id"]) for row in selected}),
                               "groups": sorted({row["group"] for row in selected}),
                               "positive_frames": sum(row["labels"].count(1) for row in selected),
                               "negative_frames": sum(row["labels"].count(0) for row in selected),
                               "unlabeled_frames": sum(row["labels"].count(-1) for row in selected)}
    audit = {"schema_version": "lrasd-frozen-candidates-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
             "output": str(output), "output_sha256": digest_bytes(output_bytes), "output_bytes": len(output_bytes),
             "frames": frames_expected, "fps": 25, "window_seconds": frames_expected / 25,
             "inputs": input_audit, "tracks": len(rows), "consistent_duplicate_rows": len(duplicates),
             "duplicates": duplicates, "split_counts": split_counts,
             "source_split_keys": {key: sorted(values) for key, values in source_splits.items()},
             "assets_verified": len(asset_cache), "asset_file_bytes": sum(item["bytes"] for item in asset_cache.values()),
             "asset_sha256_list_hash": digest_bytes(canonical(sorted((str(path), item["sha256"]) for path, item in asset_cache.items())).encode("utf-8")),
             "validation": "uniform real NPZ dimensions and PCM audio; actual asset SHA256; fixed source/group split; duplicate semantic equality",
             "label_policy": "Input labels preserved. No teacher score or test performance used to select rows."}
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    exclusive_write(output, output_bytes)
    exclusive_write(audit_path, (json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({"output": str(output), "audit": str(audit_path), "sha256": audit["output_sha256"],
                      "tracks": len(rows), "duplicates": len(duplicates), "splits": split_counts}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
