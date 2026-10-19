#!/usr/bin/env python3
"""Prepare real, unlabeled anime ASD windows from authorized local videos.

Subtitles select candidate times only. This script never infers speaking labels.
Run with .venv-visual; local ONNX weights and ffmpeg must already be present.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import wave

import cv2
import numpy as np

from probe_lrasd_detectors import LocalDetector

PREPROCESSING = "lrasd-gray112-pcm16-mfcc13-25fps-fullbody-v1"


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(item):
    return (float(item["start"]), float(item["end"])) if isinstance(item, dict) else tuple(map(float, item[:2]))


def merge_intervals(items):
    merged = []
    for start, end in sorted(interval(item) for item in items):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + .5:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def source_descriptors(args):
    descriptors = []
    entries = list(args.source)
    if args.sources:
        collection = read_json(args.sources)
        if isinstance(collection, dict):
            collection = collection.get("sources", collection.get("episodes", []))
        for item in collection:
            if isinstance(item, str):
                path = Path(item)
                entries.append(path if path.is_absolute() else args.sources.parent / path)
            elif isinstance(item, dict):
                if "video_path" in item:
                    descriptors.append(item)
                else:
                    path = Path(item.get("source_path", item.get("path", item.get("source", ""))))
                    entries.append(path if path.is_absolute() else args.sources.parent / path)
    for path in entries:
        descriptor = read_json(path)
        descriptor["descriptor_path"] = str(path.resolve())
        descriptors.append(descriptor)
    if not descriptors:
        raise ValueError("Provide --source source.json or --sources sources.json")
    return descriptors


def candidate_times(source, args):
    intervals_path = source.get("dialogue_intervals_path")
    if not intervals_path:
        descriptor = Path(source.get("descriptor_path", "/mnt/d/SceneRecall/training/lrasd/kaguya/source.json"))
        intervals_path = descriptor.parent / "dialogue-intervals.json"
    data = read_json(intervals_path)
    music = merge_intervals(data.get("music_intervals", []))
    dialogues = data.get("clean_dialogue_intervals", data.get("dialogue_intervals", []))
    duration = args.frames / 25
    long_cue_duration = duration + .4
    times, long_times, short_times = [], [], []
    for item in sorted(dialogues, key=lambda item: (interval(item)[1] - interval(item)[0] < long_cue_duration, interval(item)[0])):
        left, right = interval(item)
        start = max(0, round(((left + right) / 2 - duration / 2) * 25) / 25)
        if right - left >= long_cue_duration:
            earliest = float(np.ceil((left + .2) * 25) / 25)
            latest = float(np.floor((right - duration - .2) * 25) / 25)
            if earliest > latest:
                continue
            start = min(max(start, earliest), latest)
        if any(abs(start - selected) < duration for selected in times) or any(a - .5 < start + duration and b + .5 > start for a, b in music):
            continue
        if "split" not in source and not 10 <= start % 360 <= 350 - duration:
            continue
        times.append(start)
        (long_times if right - left >= long_cue_duration else short_times).append(start)
    gap_starts = []
    if args.include_dialogue_gaps:
        merged_dialogues = merge_intervals(data.get("dialogue_intervals", []))
        for previous, following in zip(merged_dialogues, merged_dialogues[1:]):
            if following[0] - previous[1] < duration + 2:
                continue
            start = round(((previous[1] + following[0]) / 2 - duration / 2) * 25) / 25
            if start < 0 or any(a - .5 < start + duration and b + .5 > start for a, b in music):
                continue
            if "split" not in source and not 10 <= start % 360 <= 350 - duration:
                continue
            if any(abs(start - selected) < duration for selected in times + gap_starts):
                continue
            gap_starts.append(start)
    source["_gap_starts"] = set(gap_starts)
    # Prioritize cues whose complete window fits inside with >=.2s margins.
    # Gap windows remain unlabeled and are interleaved, never assumed silent.
    rng = random.Random(args.seed + int(hashlib.sha256(source["video_path"].encode()).hexdigest()[:8], 16))
    for selected in (long_times, short_times, gap_starts):
        rng.shuffle(selected)
    priority = long_times + short_times
    ordered = []
    gaps = deque(gap_starts)
    for number, start in enumerate(priority, 1):
        ordered.append(start)
        if number % 3 == 0 and gaps:
            ordered.append(gaps.popleft())
    ordered.extend(gaps)
    return deque(ordered)


def ffmpeg_capture(video, start, args):
    command = [args.ffmpeg, "-v", "error", "-nostdin", "-threads", str(args.decode_threads),
               "-ss", f"{start:.6f}", "-i", str(video), "-map", "0:v:0", "-an", "-sn",
               "-vf", "fps=25,scale=480:270:flags=bilinear", "-frames:v", str(args.frames),
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, timeout=args.decode_timeout)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace")[-1000:])
    expected = args.frames * 270 * 480 * 3
    if len(completed.stdout) != expected:
        raise ValueError(f"Expected {args.frames} frames, got {len(completed.stdout) / (270 * 480 * 3):.2f}")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(args.frames, 270, 480, 3)


def xyxy(item):
    x, y, width, height = item["box_xywh"]
    return np.array([x, y, x + width, y + height], dtype=np.float64)


def iou(left, right):
    overlap = np.maximum(0, np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]))
    intersection = float(overlap.prod())
    areas = np.maximum(0, left[2:] - left[:2]).prod() + np.maximum(0, right[2:] - right[:2]).prod()
    return intersection / max(float(areas) - intersection, 1e-9)


def unique_face_body_pairs(people, faces, args):
    """Choose the smallest containing person box, without sharing it across faces."""
    pairs = []
    for face_number, detection in enumerate(faces):
        face = xyxy(detection)
        if np.any(face[2:] - face[:2] < args.min_face_size):
            continue
        center = (face[:2] + face[2:]) / 2
        containing = []
        for body_number, person in enumerate(people):
            body = xyxy(person)
            if np.all(center >= body[:2] - 2) and np.all(center <= body[2:] + 2):
                intersection = np.maximum(0, np.minimum(face[2:], body[2:]) - np.maximum(face[:2], body[:2])).prod()
                coverage = intersection / max(float(np.maximum(0, face[2:] - face[:2]).prod()), 1e-9)
                if coverage >= .85:
                    containing.append((float(np.maximum(0, body[2:] - body[:2]).prod()), body_number, body))
        if not containing:
            continue
        containing.sort(key=lambda item: item[0])
        # Near-identical competing detections provide no clear face/body association.
        if len(containing) > 1 and containing[1][0] < containing[0][0] * 1.1:
            continue
        _, body_number, body = containing[0]
        pairs.append({"body": body, "face": face, "body_index": body_number, "face_index": face_number,
                      "body_score": float(people[body_number]["score"]), "face_score": float(detection["score"])})
    counts = Counter(pair["body_index"] for pair in pairs)
    return [pair for pair in pairs if counts[pair["body_index"]] == 1]


def track_boxes(frames, face_detector, person_detector, args):
    # Reject visible cuts before matching. This is conservative filtering, not shot truth.
    small = np.stack([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (64, 36)) for frame in frames])
    differences = np.abs(np.diff(small.astype(np.float32), axis=0)).mean(axis=(1, 2))
    if differences.max() > args.cut_threshold:
        raise ValueError(f"shot_cut_difference:{differences.max():.2f}")
    tracks = []
    for index in args.keyframes:
        detected_people = person_detector(frames[index])
        detected_faces = face_detector(frames[index])
        if args.max_tracks_per_window == 1 and (len(detected_people) != 1 or len(detected_faces) != 1):
            raise ValueError(f"not_unique_person_face:keyframe={index}:persons={len(detected_people)}:faces={len(detected_faces)}")
        pairs = unique_face_body_pairs(detected_people, detected_faces, args)
        if index == 0:
            pairs.sort(key=lambda pair: float(pair["face"][0]))
            tracks = [{"track_index": number, "bodies": [pair["body"]], "faces": [pair["face"]],
                       "scores": [{"frame": 0, "body": pair["body_score"], "face": pair["face_score"]}]}
                      for number, pair in enumerate(pairs[:args.max_tracks_per_window])]
        else:
            proposed = []
            for track in tracks:
                options = []
                for number, pair in enumerate(pairs):
                    body_iou = iou(track["bodies"][-1], pair["body"])
                    face_iou = iou(track["faces"][-1], pair["face"])
                    if body_iou >= .25 and face_iou >= .1:
                        options.append((.7 * face_iou + .3 * body_iou, number))
                options.sort(reverse=True)
                if not options or (len(options) > 1 and options[0][0] - options[1][0] < .08):
                    continue
                proposed.append((options[0][0], track, options[0][1]))
            # A pair claimed by multiple prior tracks is ambiguous: reject those tracks.
            claims = Counter(number for _, _, number in proposed)
            continued = []
            for _, track, number in proposed:
                if claims[number] != 1:
                    continue
                pair = pairs[number]
                track["bodies"].append(pair["body"])
                track["faces"].append(pair["face"])
                track["scores"].append({"frame": int(index), "body": pair["body_score"], "face": pair["face_score"]})
                continued.append(track)
            tracks = continued
        if not tracks:
            raise ValueError(f"no_unambiguous_continuous_face_person_track:keyframe={index}:persons={len(detected_people)}:faces={len(detected_faces)}")
    positions = np.arange(args.frames)
    for track in tracks:
        track["body_boxes"] = np.stack([np.interp(positions, args.keyframes, np.array(track["bodies"])[:, c]) for c in range(4)], axis=1)
        track["face_boxes"] = np.stack([np.interp(positions, args.keyframes, np.array(track["faces"])[:, c]) for c in range(4)], axis=1)
    return tracks, float(differences.max())


def crop_visual(frames, boxes):
    crops = []
    for frame, box in zip(frames, boxes):
        # This matches App canonical RGB frame -> rounded crop -> grayscale -> resize.
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = max(0, min(479, x1)), max(1, min(480, x2))
        y1, y2 = max(0, min(269, y1)), max(1, min(270, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("empty_crop")
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        crops.append(cv2.resize(gray, (112, 112), interpolation=cv2.INTER_LINEAR))
    return np.stack(crops)


def save_npz(path, visual, boxes):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, visual=visual, boxes_xyxy_normalized=boxes / [480, 270, 480, 270])
    temporary.replace(path)


def save_audio(source, start, path, args):
    temporary = path.with_suffix(".tmp.wav")
    samples = args.frames * 640
    command = [args.ffmpeg, "-v", "error", "-nostdin", "-y", "-ss", f"{start:.6f}",
               "-i", source["video_path"], "-map", f"0:{source.get('audio_stream_index', 1)}",
               "-t", str(args.frames / 25), "-vn", "-sn", "-ac", "1",
               "-af", f"aresample=16000,asetpts=PTS-STARTPTS,apad,atrim=end_sample={samples}",
               "-ar", "16000", "-c:a", "pcm_s16le", str(temporary)]
    completed = subprocess.run(command, capture_output=True, timeout=args.decode_timeout)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace")[-1000:])
    with wave.open(str(temporary), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("Decoded candidate audio must be mono 16kHz PCM16")
        actual_samples = audio.getnframes()
        correction = samples - actual_samples
        if abs(correction) > 32:
            raise ValueError(f"Decoded candidate audio differs from {samples} samples by {correction}, exceeding 32-sample tail limit")
        pcm = audio.readframes(actual_samples)
    if len(pcm) != actual_samples * 2:
        raise ValueError("Decoded candidate audio has a truncated PCM payload")
    if correction:
        pcm = pcm[:samples * 2]
        pcm += b"\x00" * (samples * 2 - len(pcm))
        with wave.open(str(temporary), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(pcm)
    with wave.open(str(temporary), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) != (1, 2, 16000, samples):
            raise ValueError(f"Candidate audio tail correction did not produce exactly {samples} samples")
    temporary.replace(path)
    return correction


def prepare_one(source, start, sample_id, face_detector, person_detector, args):
    frames = ffmpeg_capture(source["video_path"], start, args)
    tracks, cut_difference = track_boxes(frames, face_detector, person_detector, args)
    window_folder = args.output / "assets" / sample_id
    window_folder.mkdir(parents=True, exist_ok=True)
    audio_path = window_folder / "audio.wav"
    audio_tail_correction = save_audio(source, start, audio_path, args)
    audio_hash = sha256(audio_path)
    block = int(start // 360)
    split = source.get("split", "train" if block % 10 < 8 else "val" if block % 10 == 8 else "test")
    group = source.get("group", source.get("evaluation_group", f"kaguya-block-{block:02d}"))
    rows = []
    for track in sorted(tracks, key=lambda item: item["track_index"]):
        track_index = track["track_index"]
        track_id = sample_id if args.max_tracks_per_window == 1 else f"{sample_id}-t{track_index}"
        folder = window_folder if args.max_tracks_per_window == 1 else args.output / "assets" / track_id
        folder.mkdir(parents=True, exist_ok=True)
        body_path, face_path = folder / "body.npz", folder / "face.npz"
        body, face = track["body_boxes"], track["face_boxes"]
        save_npz(body_path, crop_visual(frames, body), body)
        save_npz(face_path, crop_visual(frames, face), face)
        midpoint = args.frames // 2
        preview = frames[midpoint].copy()
        for boxes, color in ((body, (0, 255, 0)), (face, (255, 0, 0))):
            x1, y1, x2, y2 = np.rint(boxes[midpoint]).astype(int)
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.imwrite(str(folder / "preview.jpg"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        gap = start in source.get("_gap_starts", set())
        row = {"schema_version": 1, "sample_id": track_id, "window_id": sample_id, "split": split,
           "movie": source.get("movie", source.get("target_movie", "Cosmic Princess Kaguya")),
           "clip": source.get("clip", "kaguya-local"), "track": f"person-{track_index}", "group": group,
           "evaluation_group": group, "start_frame": int(round(start * 25)),
           "end_frame": int(round(start * 25)) + args.frames, "fps": 25,
           "frames": args.frames, "window_seconds": args.frames / 25,
           "visual_path": str(body_path.relative_to(args.output)),
           "teacher_visual_path": str(face_path.relative_to(args.output)),
           "audio_path": str(audio_path.relative_to(args.output)), "audio_start_seconds": 0,
           "labels": [-1] * args.frames, "synthetic": False, "label_status": "unlabeled_candidate",
           "audio_sample_count": args.frames * 640,
           "audio_tail_correction_samples": audio_tail_correction,
           "preprocessing": PREPROCESSING, "crop_type": "fullbody", "teacher_crop_type": "detected_face",
           "source_video_path": source["video_path"], "source_start_seconds": start,
           "source_sha256": source.get("sha256", source.get("video_sha256", source.get("source_sha256"))),
           "candidate_selection": "subtitle_gap_timing_only_not_verified_silence" if gap else "subtitle_dialogue_timing_only_music_excluded",
           "tracking": "nine_keyframe_unique_face_person_pair_iou_match_linear_box_interpolation",
           "tracking_configuration": args.tracking_configuration,
           "tracks_retained_in_window": len(tracks),
           "normalized_body_boxes_xyxy": (body / [480, 270, 480, 270]).round(6).tolist(),
           "normalized_face_boxes_xyxy": (face / [480, 270, 480, 270]).round(6).tolist(),
           "detection_scores": track["scores"], "maximum_frame_difference": cut_difference,
           "quality_limitations": ["machine detections are not verified identities", "undetected background people may remain",
                                   "interpolated boxes may drift between keyframes", "subtitles are not speaker labels"],
           "asset_sha256": {"visual": sha256(body_path), "teacher_visual": sha256(face_path), "audio": audio_hash}}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/mnt/d/SceneRecall/training/lrasd/candidates"))
    parser.add_argument("--models", type=Path, default=Path("/mnt/d/SceneRecall/training/lrasd/models"))
    parser.add_argument("--max-windows", type=int, default=120)
    parser.add_argument("--frames", type=int, choices=(50, 100), default=100)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--decode-timeout", type=int, default=45)
    parser.add_argument("--cut-threshold", type=float, default=32)
    parser.add_argument("--min-face-size", type=float, default=12)
    parser.add_argument("--max-tracks-per-window", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--face-threshold", type=float, default=.30)
    parser.add_argument("--person-threshold", type=float, default=.35)
    parser.add_argument("--include-dialogue-gaps", action="store_true",
                        help="Also sample subtitle gaps; these remain unlabeled and are not assumed silent")
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    args.keyframes = np.unique(np.rint(np.linspace(0, args.frames - 1, 9)).astype(int))
    if not 0 < args.face_threshold <= 1 or not 0 < args.person_threshold <= 1:
        parser.error("Detection thresholds must be in (0,1]")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, decisions = args.output / "manifest.jsonl", args.output / "decisions.jsonl"
    existing = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()] if manifest.exists() else []
    if any(len(row["labels"]) != args.frames or row["end_frame"] - row["start_frame"] != args.frames for row in existing):
        raise ValueError("Output manifest contains a different window length; use a separate output directory")
    existing_ids = Counter(row["sample_id"] for row in existing)
    duplicates = [sample_id for sample_id, count in existing_ids.items() if count > 1]
    if duplicates:
        raise ValueError("Existing manifest contains duplicate sample IDs; repair before resume: " + ", ".join(duplicates[:10]))
    seen = set(existing_ids)
    # A crash can leave some track rows durable before the window decision is
    # written. Skip that entire window on resume rather than append its tracks
    # twice; losing the remaining tracks in that one window is acceptable.
    existing_windows = {row.get("window_id", row["sample_id"]) for row in existing}
    seen.update(existing_windows)
    if decisions.exists():
        seen.update(json.loads(line)["sample_id"] for line in decisions.read_text().splitlines() if line.strip())
    descriptors = source_descriptors(args)
    queues = [(source, candidate_times(source, args)) for source in descriptors]
    face_detector = LocalDetector(args.models / "face.onnx", threshold=args.face_threshold)
    person_detector = LocalDetector(args.models / "person.onnx", threshold=args.person_threshold)
    args.tracking_configuration = {
        "face_threshold": args.face_threshold, "person_threshold": args.person_threshold,
        "face_model_sha256": sha256(args.models / "face.onnx"),
        "person_model_sha256": sha256(args.models / "person.onnx"),
        "maximum_tracks_per_window": args.max_tracks_per_window,
        "minimum_face_size_pixels": args.min_face_size, "keyframes": args.keyframes.tolist(),
        "frames": args.frames, "window_seconds": args.frames / 25, "audio_samples": args.frames * 640,
        "minimum_body_iou": .25, "minimum_face_iou": .1,
        "matching_ambiguity_score_margin": .08, "minimum_face_coverage_by_body": .85,
        "maximum_frame_difference": args.cut_threshold, "include_dialogue_gaps": args.include_dialogue_gaps,
        "long_cue_priority_seconds": args.frames / 25 + .4, "seed": args.seed,
        "frame_decode": "ffmpeg fps=25,scale=480:270:flags=bilinear RGB24",
        "audio_decode": f"aresample=16000,asetpts=PTS-STARTPTS,apad,atrim=end_sample={args.frames * 640}; output -t {args.frames / 25}",
        "audio_tail_correction_limit_samples": 32,
        "audio_tail_correction_method": "end only; positive=zero-padded samples, negative=trimmed samples; first sample unchanged",
        "crop": "rounded xyxy pixel crop -> RGB2GRAY -> INTER_LINEAR 112x112",
    }
    stats = {"existing_candidates": len(existing), "accepted": 0, "attempted": 0,
             "existing_distinct_windows": len(existing_windows),
             "source_count": len(descriptors), "remaining_timed_windows": sum(len(queue) for _, queue in queues),
             "preprocessing": PREPROCESSING, "labels": "all -1, no speaking labels inferred"}
    stats["tracking_configuration"] = args.tracking_configuration
    started = time.perf_counter()
    emit("started", **stats, output=str(args.output))
    rejection_counts = Counter()
    while any(queue for _, queue in queues) and len(existing) + stats["accepted"] < args.max_windows and stats["attempted"] < args.max_attempts:
        for source, queue in queues:
            if not queue or len(existing) + stats["accepted"] >= args.max_windows or stats["attempted"] >= args.max_attempts:
                continue
            start = queue.popleft()
            identity = source["video_path"] + ":" + f"{start:.2f}"
            if args.frames != 100:
                identity += f":frames={args.frames}"
            sample_id = "anime-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            if sample_id in seen:
                continue
            stats["attempted"] += 1
            try:
                rows = prepare_one(source, start, sample_id, face_detector, person_detector, args)
                rows = rows[:args.max_windows - len(existing) - stats["accepted"]]
                for row in rows:
                    if row["sample_id"] in seen:
                        raise ValueError("Refusing duplicate candidate sample ID: " + row["sample_id"])
                    append_json(manifest, row)
                    seen.add(row["sample_id"])
                    stats["accepted"] += 1
                    emit("accepted", sample_id=row["sample_id"], split=row["split"], start_seconds=start,
                         candidates=len(existing) + stats["accepted"], manifest=str(manifest),
                         tracks_in_window=len(rows), selection=row["candidate_selection"])
                decision = {"sample_id": sample_id, "accepted": True, "source": source["video_path"],
                            "start": start, "track_sample_ids": [row["sample_id"] for row in rows]}
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                reason = str(error)[:500]
                rejection_counts[reason.split(":")[0]] += 1
                decision = {"sample_id": sample_id, "accepted": False, "source": source["video_path"], "start": start, "reason": reason}
                if stats["attempted"] % 10 == 0:
                    emit("progress", **{**stats, "rejection_counts": dict(rejection_counts),
                                        "elapsed_seconds": time.perf_counter() - started})
            append_json(decisions, decision)
            seen.add(sample_id)
            stats["elapsed_seconds"] = time.perf_counter() - started
            stats["rejection_counts"] = dict(rejection_counts)
            write_json(args.output / "preparation-summary.json", stats)
    emit("finished", **stats, total_candidates=len(existing) + stats["accepted"])


if __name__ == "__main__":
    main()
