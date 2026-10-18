"""Export saved human ASD reviews without reading original media or model predictions.

Input is the standalone annotator's clips/ directory, not its JSON download.
Without --plan every window remains train-only. An explicit, frozen scene plan
can assign within-title validation/test; this script never starts training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall.lrasd_training_data import canonical_hash, load_annotation_plan  # noqa: E402


ANNOTATION_VERSION = "scenerecall-human-asd-v1"
EXPORT_VERSION = "scenerecall-human-asd-export-v1"
GRID_VERSION = "presentation-rgb480x270-bilinear25-pcm16-mono16k-v1"
PREPROCESSING = "lrasd-gray112-pcm16-mfcc13-25fps-fullbody-v1"
HUMAN_LABEL_SOURCE = "human_frame_asd"
FRAMES = 50
WIDTH, HEIGHT, FPS, SAMPLES_PER_FRAME = 480, 270, 25, 640
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked_id(value, name):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"Invalid {name}: only safe local IDs are allowed")
    return value


def confined(root, path):
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError(f"Annotation path escapes its selected root: {path.name}")
    return resolved


def read_clip(path):
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError(f"Missing or oversized clip.json: {path.parent.name}")
    raw = path.read_bytes()
    try:
        def reject_constant(value):
            raise ValueError(f"Nonfinite JSON number: {value}")
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result
        clip = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid clip.json in {path.parent.name}: {exc}") from exc
    if not isinstance(clip, dict):
        raise ValueError(f"clip.json must be an object: {path.parent.name}")
    return clip, hashlib.sha256(raw).hexdigest()


def server_field(clip, name):
    server = clip.get("server", {})
    if not isinstance(server, dict):
        raise ValueError("clip.server must be an object")
    if name in clip and name in server and clip[name] != server[name]:
        raise ValueError(f"Conflicting source metadata: {name}")
    return server.get(name, clip.get(name))


def checked_snapshot(root, directory, clip):
    count = clip["frame_count"]
    files = server_field(clip, "files")
    if not isinstance(files, dict):
        raise ValueError(f"Missing source media hashes: {clip['id']}")
    saved = {}
    for key, filename in (("frames_rgb", "frames.rgb"), ("audio_wav", "audio.wav")):
        declared = files.get(key, {})
        if (not isinstance(declared, dict) or declared.get("relative_path") != filename
                or not isinstance(declared.get("sha256"), str) or not SHA256.fullmatch(declared["sha256"])
                or type(declared.get("bytes")) is not int):
            raise ValueError(f"Invalid fixed media declaration: {clip['id']}/{key}")
        path = confined(root, directory / filename)
        if not path.is_file() or path.stat().st_size != declared["bytes"]:
            raise ValueError(f"Source media size mismatch: {clip['id']}/{filename}")
        digest = sha256_file(path)
        if digest != declared["sha256"]:
            raise ValueError(f"Source media SHA256 mismatch: {clip['id']}/{filename}")
        saved[key] = {"relative_path": filename, "sha256": digest, "bytes": path.stat().st_size}
    if saved["frames_rgb"]["bytes"] != count * WIDTH * HEIGHT * 3:
        raise ValueError(f"RGB size does not match the 25fps frame grid: {clip['id']}")
    try:
        with wave.open(str(directory / "audio.wav"), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                    audio.getcomptype(), audio.getnframes()) != (1, 2, 16000, "NONE", count * SAMPLES_PER_FRAME):
                raise ValueError(f"Audio must be exact aligned mono PCM16 16kHz: {clip['id']}")
            if len(audio.readframes(count * SAMPLES_PER_FRAME)) != count * SAMPLES_PER_FRAME * 2:
                raise ValueError(f"Truncated PCM data: {clip['id']}")
    except wave.Error as exc:
        raise ValueError(f"Invalid audio.wav: {clip['id']}") from exc
    return saved


def checked_clip(directory, clip):
    identity = checked_id(clip.get("id"), "clip ID")
    if directory.name != identity:
        raise ValueError("clip.json ID does not match its local directory")
    if clip.get("schema_version") != "1.0" or clip.get("annotation_version") != ANNOTATION_VERSION:
        raise ValueError(f"Unsupported annotation schema: {identity}")
    checked_id(clip.get("asset_id"), "asset ID")
    revision, count = clip.get("revision"), clip.get("frame_count")
    start, end = clip.get("start_ms"), clip.get("end_ms")
    if type(revision) is not int or revision < 0:
        raise ValueError(f"Invalid saved revision: {identity}")
    if (type(count) is not int or not 16 <= count <= 200 or clip.get("fps") != FPS
            or type(start) is not int or type(end) is not int or start < 0
            or start % 40 or end - start != count * 40):
        raise ValueError(f"Invalid continuous 25fps annotation frame grid: {identity}")
    fingerprint, source_sha = clip.get("source_fingerprint"), server_field(clip, "source_sha256")
    if (not isinstance(fingerprint, str) or not SHA256.fullmatch(fingerprint)
            or not isinstance(source_sha, str) or not SHA256.fullmatch(source_sha)):
        raise ValueError(f"Missing full source fingerprint/SHA256: {identity}")
    basis = server_field(clip, "source_sha256_basis")
    if basis != "library_full_file_sha256" or source_sha != fingerprint:
        raise ValueError(f"Unrecognized source SHA256 provenance: {identity}")
    stat = clip.get("source_stat")
    if (not isinstance(stat, dict) or any(type(stat.get(key)) is not int or stat[key] < 0
                                         for key in ("size", "mtime_ns"))):
        raise ValueError(f"Missing source stat binding: {identity}")
    if server_field(clip, "preprocessing") != GRID_VERSION:
        raise ValueError(f"Unrecognized RGB/audio preprocessing: {identity}")
    if type(clip.get("audio_stream_index")) is not int or clip["audio_stream_index"] < 0:
        raise ValueError(f"Missing selected source audio stream: {identity}")
    if not isinstance(clip.get("tracks"), list) or len(clip["tracks"]) > 32:
        raise ValueError(f"Invalid annotation track list: {identity}")


def checked_track(track, count, revision):
    if not isinstance(track, dict):
        raise ValueError("Track must be an object")
    checked_id(track.get("id"), "track ID")
    if type(track.get("reviewed")) is not bool or type(track.get("excluded")) is not bool:
        raise ValueError("reviewed/excluded must be explicit booleans")
    if track["excluded"] or not track["reviewed"]:
        return "excluded" if track["excluded"] else "not_reviewed"
    if revision < 1:
        raise ValueError("Reviewed tracks require an explicitly saved revision >= 1")
    provenance = track.get("provenance", {})
    if (not track.get("reviewed_at") or track.get("reviewer") != {"kind": "human"}
            or not isinstance(provenance, dict) or provenance.get("boxes_reviewed") is not True
            or provenance.get("labels_source") != "human_input"
            or provenance.get("model_scores_copied") is not False):
        raise ValueError("Reviewed tracks require saved human-input provenance and review time")
    labels, boxes = track.get("labels"), track.get("boxes")
    if (not isinstance(labels, list) or len(labels) != count
            or any(type(value) is not int or value not in (-1, 0, 1) for value in labels)):
        raise ValueError("Labels must contain exactly one -1/0/1 integer per frame")
    if not isinstance(boxes, list) or len(boxes) != count:
        raise ValueError("Boxes must contain one explicit normalized xywh box or null per frame")
    for index, (label, box) in enumerate(zip(labels, boxes)):
        if box is None:
            if label != -1:
                raise ValueError(f"Frame {index}: a known label requires a real box")
            continue
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
            raise ValueError(f"Frame {index}: malformed normalized xywh box")
        x, y, width, height = box
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
            raise ValueError(f"Frame {index}: box is empty or outside the canvas")
        if round((x + width) * WIDTH) <= round(x * WIDTH) or round((y + height) * HEIGHT) <= round(y * HEIGHT):
            raise ValueError(f"Frame {index}: box is empty after pixel rounding")
    character = track.get("character_id")
    if character is not None:
        checked_id(character, "character ID")
    return "all_unknown" if not any(label >= 0 for label in labels) else None


def crop_visual(frames, boxes):
    """Use the app's RGB480x270 -> body crop -> gray112 bilinear convention."""
    import cv2
    import numpy as np

    result = np.zeros((len(boxes), 112, 112), dtype=np.uint8)
    for index, box in enumerate(boxes):
        if box is None:
            continue
        x, y, width, height = box
        left, top = max(0, round(x * WIDTH)), max(0, round(y * HEIGHT))
        right, bottom = min(WIDTH, round((x + width) * WIDTH)), min(HEIGHT, round((y + height) * HEIGHT))
        gray = cv2.cvtColor(frames[index, top:bottom, left:right], cv2.COLOR_RGB2GRAY)
        result[index] = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_LINEAR)
    return result


def write_npz(path, arrays):
    """Fixed ZIP timestamps/order keep equivalent exports byte reproducible."""
    import numpy as np

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key, array in sorted(arrays.items()):
            content = io.BytesIO()
            np.lib.format.write_array(content, array, allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content.getvalue(), compresslevel=6)


def export_annotations(root, output, *, movie=None, plan_path=None):
    """Return an auditable export summary; never overwrite a prior export."""
    import numpy as np

    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Annotation clips root does not exist or is not a directory")
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output must be a new directory separate from the annotation clips root")
    plan = load_annotation_plan(plan_path) if plan_path else None
    planned = {item["clip_id"]: item for item in plan["items"]} if plan else {}
    directories = sorted(path for path in root.iterdir() if (path.is_dir() or path.is_symlink())
                         and (not plan or path.name in planned))
    if not directories:
        raise ValueError("No annotation clip directories found")
    snapshots, rejected, skipped, tails = [], [], [], []
    binding, movie_name = None, movie
    for candidate in directories:
        checked_id(candidate.name, "clip directory")
        directory = confined(root, candidate)
        path = confined(root, directory / "clip.json")
        clip, annotation_sha = read_clip(path)
        if (clip.get("id") != candidate.name or clip.get("schema_version") != "1.0"
                or clip.get("annotation_version") != ANNOTATION_VERSION
                or clip.get("status") not in ("ready", "preparing", "failed")):
            raise ValueError(f"Invalid annotation clip schema/status/ID: {candidate.name}")
        if clip.get("status") != "ready":
            skipped.append({"clip": candidate.name, "reason": "clip_not_ready", "status": clip.get("status")})
            continue
        checked_clip(directory, clip)
        if plan:
            item = planned[clip["id"]]
            if (clip["source_fingerprint"] != plan["source_fingerprint"]
                    or (plan.get("asset_id") and clip["asset_id"] != plan["asset_id"])
                    or (clip["start_ms"], clip["end_ms"]) != (item["start_ms"], item["end_ms"])):
                raise ValueError(f"Clip source/range differs from its fixed annotation plan: {clip['id']}")
        current = (clip["asset_id"], clip["source_fingerprint"], canonical_json(clip["source_stat"]))
        if binding is not None and binding != current:
            raise ValueError("One export must contain one unchanged original asset/source fingerprint")
        binding = current
        source_movie = clip.get("movie") or clip.get("asset_title") or clip["asset_id"]
        if not isinstance(source_movie, str) or not source_movie.strip():
            raise ValueError("Missing original work title")
        movie_name = movie_name or source_movie
        if movie is None and source_movie != movie_name:
            raise ValueError("Conflicting original work titles in one annotation root")
        saved = checked_snapshot(root, directory, clip)
        tracks, track_ids = [], set()
        for track in clip["tracks"]:
            identity = track.get("id") if isinstance(track, dict) else None
            if isinstance(identity, str) and identity in track_ids:
                raise ValueError(f"Duplicate annotation track ID in {clip['id']}")
            if isinstance(identity, str):
                track_ids.add(identity)
            try:
                reason = checked_track(track, clip["frame_count"], clip["revision"])
            except ValueError as exc:
                rejected.append({"clip": clip["id"], "track": identity, "reason": str(exc)})
                continue
            if reason:
                skipped.append({"clip": clip["id"], "track": identity, "reason": reason})
            else:
                tracks.append(track)
        tail = clip["frame_count"] % FRAMES
        if tail:
            tails.append({"clip": clip["id"], "frames": tail,
                          "start_frame": clip["frame_count"] - tail, "reason": "incomplete_2s_window_discarded"})
        if clip["frame_count"] < FRAMES:
            skipped.append({"clip": clip["id"], "reason": "fewer_than_50_frames"})
        snapshots.append((directory, clip, annotation_sha, saved, tracks))
    if not snapshots:
        raise ValueError("No ready annotation clips found; no training export was written")
    if not isinstance(movie_name, str) or not movie_name.strip():
        raise ValueError("movie must be a nonempty original work name")
    output.mkdir(parents=True)
    rows, input_records = [], []
    evaluation_group = "human-work-" + hashlib.sha256(canonical_json(list(binding[:2])).encode()).hexdigest()[:24]
    for directory, clip, annotation_sha, saved, tracks in snapshots:
        item = planned.get(clip["id"])
        clip_group = ("human-scene-" + canonical_hash([clip["source_fingerprint"], item["scene_id"]])[:24]
                      if item else evaluation_group)
        input_records.append({"clip": clip["id"], "revision": clip["revision"], "annotation_sha256": annotation_sha,
                              "source_fingerprint": clip["source_fingerprint"], "files": saved})
        frames = np.memmap(directory / "frames.rgb", dtype=np.uint8, mode="r",
                           shape=(clip["frame_count"], HEIGHT, WIDTH, 3))
        try:
            with wave.open(str(directory / "audio.wav"), "rb") as stream:
                pcm = stream.readframes(clip["frame_count"] * SAMPLES_PER_FRAME)
            for offset in range(0, clip["frame_count"] - FRAMES + 1, FRAMES):
                active = [track for track in tracks if any(value >= 0 for value in track["labels"][offset:offset + FRAMES])]
                for track in tracks:
                    if track not in active:
                        skipped.append({"clip": clip["id"], "track": track["id"], "start_frame": offset,
                                        "reason": "window_all_unknown"})
                if not active:
                    continue
                destination = output / "assets" / clip["id"] / f"window-{offset:06d}"
                destination.mkdir(parents=True)
                audio_path = destination / "audio.wav"
                with wave.open(str(audio_path), "wb") as stream:
                    stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    stream.writeframes(pcm[offset * SAMPLES_PER_FRAME * 2:(offset + FRAMES) * SAMPLES_PER_FRAME * 2])
                audio_sha = sha256_file(audio_path)
                for track in active:
                    boxes, labels = track["boxes"][offset:offset + FRAMES], track["labels"][offset:offset + FRAMES]
                    visual = crop_visual(frames[offset:offset + FRAMES], boxes)
                    box_array = np.zeros((FRAMES, 4), dtype=np.float32)
                    for index, box in enumerate(boxes):
                        if box is not None:
                            x, y, width, height = box
                            box_array[index] = (x, y, x + width, y + height)
                    visual_path = destination / (hashlib.sha256(track["id"].encode()).hexdigest()[:24] + ".npz")
                    write_npz(visual_path, {"visual": visual, "boxes_xyxy_normalized": box_array,
                                          "boxes_valid_mask": np.asarray([box is not None for box in boxes], dtype=np.bool_)})
                    sample_digest = hashlib.sha256(canonical_json([ANNOTATION_VERSION, clip["asset_id"],
                        clip["source_fingerprint"], clip["id"], annotation_sha, clip["revision"], track["id"], offset]).encode()).hexdigest()
                    rows.append({"schema_version": 1, "sample_id": "human-" + sample_digest[:32], "split": item["split"] if item else "train",
                        "movie": movie_name, "clip": clip["id"], "track": track["id"], "character_id": track.get("character_id"),
                        "evaluation_group": clip_group, "group": clip_group, "start_frame": offset,
                        "end_frame": offset + FRAMES, "fps": FPS, "frame_coordinate_domain": "annotation_clip_25fps_half_open",
                        "visual_path": visual_path.relative_to(output).as_posix(), "audio_path": audio_path.relative_to(output).as_posix(),
                        "audio_start_seconds": 0, "labels": labels, "mask": [value >= 0 for value in labels],
                        "preprocessing": PREPROCESSING, "canvas_width": WIDTH, "canvas_height": HEIGHT,
                        "crop_type": "full_body_bbox", "synthetic": False, "machine_generated": False,
                        "human_annotated": True, "human_frame_asd_ground_truth": True, "label_source": HUMAN_LABEL_SOURCE,
                        "annotation_granularity": "human_reviewed_frame_asd_and_body_boxes",
                        "asset_sha256": {"visual": sha256_file(visual_path), "audio": audio_sha},
                        "source_start_ms": clip["start_ms"] + offset * 40,
                        "source_end_ms": clip["start_ms"] + (offset + FRAMES) * 40,
                        "source_fingerprint": clip["source_fingerprint"], "source_video_sha256": server_field(clip, "source_sha256"),
                        "source_sha256_basis": server_field(clip, "source_sha256_basis"), "source_stat": clip["source_stat"],
                        "audio_stream_index": clip["audio_stream_index"],
                        "human_annotation": {"annotation_version": ANNOTATION_VERSION, "clip_id": clip["id"],
                            "revision": clip["revision"], "annotation_sha256": annotation_sha, "track_id": track["id"],
                            "reviewed": True, "excluded": False, "reviewed_at": track.get("reviewed_at"),
                            "reviewer": track.get("reviewer"), "track_provenance": track.get("provenance"),
                            "note": track.get("note", "")},
                        "source_snapshot": saved, "frame_grid_provenance": {
                            "version": server_field(clip, "preprocessing"), "clip_start_ms": clip["start_ms"],
                            "clip_end_ms": clip["end_ms"], "clip_frame_count": clip["frame_count"],
                            "window_start_frame": offset, "window_end_frame": offset + FRAMES,
                            "fps": FPS, "samples_per_frame": SAMPLES_PER_FRAME,
                            "preparation": clip.get("server", {}).get("preparation")}})
                    if item:
                        rows[-1]["annotation_plan"] = {"sha256": plan["plan_sha256"], "item_id": item["id"],
                            "scene_id": item["scene_id"], "split": item["split"],
                            "minimum_cross_split_gap_ms": plan["minimum_cross_split_gap_ms"],
                            "validation_scope": "within_title_scene_holdout", "cross_title_validation": False}
        finally:
            frames._mmap.close()
        if (sha256_file(directory / "clip.json") != annotation_sha
                or checked_snapshot(root, directory, clip) != saved):
            raise ValueError(f"Annotations/media changed during export; discard this incomplete output: {clip['id']}")
    for directory, clip, annotation_sha, saved, _ in snapshots:
        if (sha256_file(directory / "clip.json") != annotation_sha
                or checked_snapshot(root, directory, clip) != saved):
            raise ValueError(f"Annotations/media changed before manifest publication: {clip['id']}")
    manifest = output / "labeled.jsonl"
    manifest.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    split_counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    missing = sorted(set(planned) - {clip["id"] for _, clip, _, _, _ in snapshots})
    blocked = ([{"clip": identity, "reason": "missing_or_not_ready_planned_clip"} for identity in missing]
               + rejected + [entry for entry in skipped if entry["reason"] in {"not_reviewed", "fewer_than_50_frames"}])
    empty_clips = [clip["id"] for _, clip, _, _, tracks in snapshots if not tracks]
    blocked.extend({"clip": identity, "reason": "no_usable_reviewed_track"} for identity in empty_clips)
    readiness = bool(plan) and not blocked and all(split_counts[split] >= minimum for split, minimum in (("train", 20), ("val", 5), ("test", 5)))
    readiness = readiness and all(any(label == value for row in rows if row["split"] == split for label in row["labels"])
                                  for split in split_counts for value in (0, 1))
    summary = {"schema_version": EXPORT_VERSION, "annotation_version": ANNOTATION_VERSION,
        "label_source": HUMAN_LABEL_SOURCE, "export_ready": bool(rows), "training_ready": False,
        "training_limitation": "Same-title scene holdout only; no cross-title validation. Baseline may have prior weak-label exposure to this film." if plan else "Train-only export; provide a frozen scene plan for validation/test before training.",
        "human_annotated": True, "human_frame_asd_ground_truth": True, "machine_generated": False,
        "movie": movie_name, "evaluation_group": evaluation_group, "frames": FRAMES, "fps": FPS,
        "accepted_windows": len(rows), "split_counts": split_counts,
        "annotation_plan_sha256": plan["plan_sha256"] if plan else None,
        "split_strategy": "scene_holdout_same_title" if plan else "train_only", "blocked": blocked,
        "positive_frames": sum(row["labels"].count(1) for row in rows),
        "negative_frames": sum(row["labels"].count(0) for row in rows),
        "masked_frames": sum(row["labels"].count(-1) for row in rows),
        "inputs": input_records, "skipped": skipped, "rejected_tracks": rejected, "discarded_tails": tails,
        "labeled_manifest_sha256": sha256_file(manifest), "writes_original_media": False,
        "starts_training": False, "identity_verification_claimed": False}
    summary["training_ready"] = readiness
    summary_path = output / "label-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {**summary, "manifest_path": str(manifest), "summary_path": str(summary_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="将独立小程序已人工复核的 ASD 轨迹导出为两秒训练资产；不启动训练")
    parser.add_argument("--annotations-root", required=True, type=Path, help="独立标注数据下的 clips 目录")
    parser.add_argument("--output", required=True, type=Path, help="新的导出目录，禁止覆盖已有结果")
    parser.add_argument("--movie", help="固定原作品名称；默认读取 clip.json 中的 movie/asset_title")
    parser.add_argument("--plan", type=Path, help="预先固定的场景 train/val/test 划分；不提供时仍全部 train")
    args = parser.parse_args(argv)
    try:
        summary = export_annotations(args.annotations_root, args.output, movie=args.movie, plan_path=args.plan)
    except (ValueError, OSError, ImportError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    result = {key: summary[key] for key in ("export_ready", "accepted_windows", "manifest_path", "summary_path")}
    result.update(rejected_tracks=len(summary["rejected_tracks"]), skipped=len(summary["skipped"]),
                  discarded_tails=len(summary["discarded_tails"]))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if summary["export_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
