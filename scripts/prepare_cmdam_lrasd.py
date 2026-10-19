#!/usr/bin/env python3
"""Convert official CMD-AM annotations into conservative, derived full-body ASD labels.

Only local files are read. No downloads or training occur. Sentence-level speaker
truth is not frame-level mouth-motion truth; blank/ambiguous time is masked.
NumPy/OpenCV are imported only while extracting verified local video assets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import threading
import wave

PROJECT = Path(__file__).resolve().parents[1]
POLICY = "cmdam-official-concordant-sentence-body-v1"
PROBE_SCHEMA = "cmdam-native-video-probe-v2"
PREPROCESSING = "lrasd-gray112-pcm16-mfcc13-25fps-fullbody-v1"
LABEL_SOURCE = "cmdam_official_sentence_alignment"
VIDEO_SUFFIXES = {".mkv", ".mp4", ".webm"}
TITLE_ALIASES = {"Spider Man Into the Spider-Verse": "Spider Man: Into the Spider-Verse",
                 "Spider-Man: Into the Spider-Verse": "Spider Man: Into the Spider-Verse"}
CHARACTER_ALIASES = {
    "Despicable Me": {"Felonius Gru": "Gru", "Felonious Gru": "Gru", "Agnes Gru": "Agnes",
                       "Margo Gru": "Margo", "Edith Gru": "Edith", "Kyle Gru": "Kyle"},
    "Ice Age": {"Roshan": "Roshan, the Infant"},
    "Spider Man: Into the Spider-Verse": {"Wilson Fick": "Wilson Fisk", "Richard Fick": "Richard Fisk"},
    "Sing": {"Big daddy": "Big Daddy"},
}
# Collective names do not identify an individual audible speaker, even when
# only one member happens to be visible. They can never generate positives.
COLLECTIVE_NAMES = {"Minion", "Bears", "Squids", "Rhino Officers", "Shrimp", "Frogs", "Sheep",
                    "Rosita & Norman's children", "The Q-Teez", "Turtles", "Bunnies"}
_EMIT_LOCK = threading.Lock()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(event, **fields):
    with _EMIT_LOCK:
        print(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def canonical_movie(name):
    return TITLE_ALIASES.get(name, name)


def canonical_character(name, movie):
    # Exact literal Other is the open-set label. Other Father/Other Mother are
    # real character names and must survive this normalization.
    name = " ".join(str(name).split())
    return CHARACTER_ALIASES.get(canonical_movie(movie), {}).get(name, name)


def local_path(value, selection):
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (PROJECT / path, Path(selection).parent / path):
        if candidate.exists():
            return candidate
    return PROJECT / path


def selection_records(path):
    data = read_json(path)
    rows = data.get("clips") if isinstance(data, dict) else data
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("The official selection must contain all 40 annotated clips")
    result, seen = [], set()
    for row in rows:
        clip = row.get("clip_id")
        if not isinstance(clip, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", clip) or clip in seen:
            raise ValueError("Selection clip IDs must be unique original 11-character IDs")
        seen.add(clip)
        movie = canonical_movie(row.get("movie", ""))
        if not movie or not row.get("annotation_path"):
            raise ValueError("Selection requires movie and annotation_path")
        result.append({**row, "movie": movie})
    if len({row["movie"] for row in result}) != 8:
        raise ValueError("All eight original movies must be present; missing movies cannot be silently replaced")
    return sorted(result, key=lambda row: row["clip_id"])


def locate_video(directory, clip):
    matches = sorted(path for path in Path(directory).rglob(clip + ".*")
                     if path.is_file() and path.stem == clip and path.suffix.lower() in VIDEO_SUFFIXES)
    if len(matches) != 1:
        raise ValueError(f"{clip}: expected one local .mkv/.mp4/.webm, found {len(matches)}")
    return matches[0]


def resource_records(path):
    records = {}
    for row in read_json(path):
        clip = row["clip_idx"]
        if clip in records:
            raise ValueError(f"Duplicate official resource clip: {clip}")
        dimensions = {tuple(frame["resolution"]) for frame in row["key_frames"]}
        if len(dimensions) != 1:
            raise ValueError(f"{clip}: inconsistent/missing official annotation canvas")
        width, height = dimensions.pop()
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError("Invalid official annotation canvas")
        records[clip] = {"movie": canonical_movie(row["movie_title"]), "width": width, "height": height,
                         "keyframe_indices": [frame["frame_idx"] for frame in row["key_frames"]]}
    return records


def alignment_overrides(path, selection):
    if path is None:
        return {}
    data = read_json(path)
    if not isinstance(data, dict) or data.get("schema_version") != "cmdam-alignment-overrides-v1" or not isinstance(data.get("clips"), dict):
        raise ValueError("Unsupported explicit alignment-override schema")
    known = {row["clip_id"] for row in selection}
    if set(data["clips"]) - known:
        raise ValueError("Alignment override references a clip outside the frozen 40-clip selection")
    for clip, override in data["clips"].items():
        if (not isinstance(override, dict) or override.get("verified") is not True
                or not re.fullmatch(r"[0-9a-f]{64}", str(override.get("source_sha256", "")))
                or not isinstance(override.get("evidence"), list) or not override["evidence"]
                or any(not isinstance(value, str) or not value.strip() for value in override["evidence"])):
            raise ValueError(f"{clip}: overrides require verified=true, exact source SHA-256, and explicit review evidence")
        if set(override) - {"verified", "source_sha256", "evidence", "crop", "annotation_prefix_frames", "notes"}:
            raise ValueError("Alignment overrides cannot change FPS, first PTS, or add guessed time offsets")
    return data["clips"]


def dense_frames(path, movie, width, height):
    """Return unique-name candidate regions; invalid/duplicate names stay ambiguous."""
    raw = read_json(path)
    ordered, stats = {}, Counter()
    for entry in raw.values():
        filename = entry.get("filename", "")
        if not re.fullmatch(r"\d+\.jpg", filename):
            raise ValueError("Dense frame names must be native zero-based integer JPG names")
        index = int(filename[:-4])
        if index in ordered:
            raise ValueError("Duplicate dense native frame")
        regions = entry.get("regions", [])
        regions = list(regions.values()) if isinstance(regions, dict) else regions
        by_name = defaultdict(list)
        for region in regions:
            shape, attributes = region.get("shape_attributes", {}), region.get("region_attributes", {})
            name = canonical_character(attributes.get("name", ""), movie)
            box = None
            try:
                x, y, w, h = [float(shape[key]) for key in ("x", "y", "width", "height")]
                if shape.get("name") != "rect" or not name or not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError("Invalid rectangle")
                if w <= 0 or h <= 0:
                    raise ValueError("Empty rectangle")
                # A few official rectangles overshoot by 1-3 pixels. Larger
                # disagreement rejects the clip instead of guessing its canvas.
                spill = max(0, -x, -y, x + w - width, y + h - height)
                if spill > 4:
                    raise RuntimeError("Annotation coordinates disagree with official resource canvas")
                box = (max(0., x), max(0., y), min(float(width), x + w), min(float(height), y + h))
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("Empty clipped rectangle")
                stats["clamped_boxes"] += spill > 0
            except (KeyError, ValueError, TypeError):
                stats["invalid_boxes"] += 1
            by_name[name].append(box)
            stats["boxes"] += 1
        ordered[index] = dict(by_name)
    if sorted(ordered) != list(range(len(ordered))):
        raise ValueError("Dense native frames must cover 0..N-1 without gaps")
    stats["frames"] = len(ordered)
    return [ordered[index] for index in range(len(ordered))], dict(stats)


def load_speakers(path, selection):
    expected = {row["clip_id"]: row["movie"] for row in selection}
    result = {}
    for movie, clips in read_json(path).items():
        movie = canonical_movie(movie)
        for clip, segments in clips.items():
            if clip not in expected or movie != expected[clip] or clip in result:
                raise ValueError("Speaker annotations do not uniquely match the selected movie/clip IDs")
            converted = []
            for segment in segments:
                left, right = segment["temporal_range"]
                if not all(type(v) in (int, float) and math.isfinite(v) for v in (left, right)) or not 0 <= left < right:
                    raise ValueError("Invalid official speaker time interval")
                converted.append((float(left), float(right), canonical_character(segment["character"], movie)))
            result[clip] = sorted(converted)
    if set(result) != set(expected):
        raise ValueError("Speaker annotations must cover all selected 40 clips")
    return result


def agreed_speaker(at, ap, der, visible, boundary_seconds=.08):
    versions = []
    for segments in (ap, der):
        if any(abs(at - edge) <= boundary_seconds for left, right, _ in segments for edge in (left, right)):
            return None
        names = {name for left, right, name in segments if left <= at < right}
        if len(names) != 1:
            return None
        versions.append(next(iter(names)))
    name = versions[0]
    if (versions[1] != name or not name or name == "Other" or name in COLLECTIVE_NAMES
            or len(visible.get(name, [])) != 1 or visible[name][0] is None):
        return None
    return name


def speaker_extent_audit(probe, ap, der):
    """Check trainable named speakers; masked open-set tails cannot add labels."""
    end = probe["video_offset_seconds"] + probe["native_duration_seconds"]
    tolerance = 2 / float(Fraction(probe["fps"]))
    trainable, ignored = [], []
    for version, segments in (("ap", ap), ("der", der)):
        for left, right, name in segments:
            record = {"version": version, "start_seconds": left, "end_seconds": right, "character": name}
            (trainable if name and name != "Other" and name not in COLLECTIVE_NAMES else ignored).append(record)
    known_end = max((row["end_seconds"] for row in trainable), default=0.)
    if known_end > end + tolerance:
        raise ValueError("Official named-speaker timestamps extend beyond the verified annotation-prefix video clock")
    ignored_end = max((row["end_seconds"] for row in ignored), default=0.)
    overflow = [row for row in ignored if row["end_seconds"] > end]
    return {"verified_prefix_end_seconds": end, "named_speaker_tolerance_seconds": tolerance,
            "maximum_named_speaker_end_seconds": known_end,
            "ignored_nontraining_annotation_extent": {
                "policy": "literal_Other_empty_and_collective_speakers_are_always_masked",
                "segments": len(ignored), "maximum_end_seconds": ignored_end,
                "seconds_beyond_prefix": max(0., ignored_end - end), "overflow_segments": overflow}}


def round_near_nonnegative(value):
    value = Fraction(value)
    if value < 0:
        raise ValueError("Resampling timestamps must be nonnegative")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def resampled_indices(native_pts, time_base, output_frames):
    """Mirror explicit FFmpeg fps round=near: quantize input PTS, then dup/drop."""
    if not native_pts:
        raise ValueError("No native frame PTS")
    base = Fraction(time_base)
    origin = native_pts[0]
    grid = [round_near_nonnegative((pts - origin) * base * 25) for pts in native_pts]
    if any(b < a for a, b in zip(grid, grid[1:])):
        raise ValueError("Nonmonotonic native frame timestamps")
    source, result = 0, []
    for output in range(output_frames):
        while source + 1 < len(grid) and grid[source + 1] <= output:
            source += 1
        result.append(source)
    return result


def probe_video(video, expected_frames, canvas, ffprobe, timeout, override=None):
    entries = ("stream=index,codec_type,width,height,avg_frame_rate,r_frame_rate,time_base,start_time,start_pts:"
               "stream_side_data=rotation:format=start_time,duration")
    command = [ffprobe, "-v", "error", "-show_entries", entries, "-of", "json", str(video)]
    metadata = json.loads(subprocess.check_output(command, timeout=timeout))
    videos = [stream for stream in metadata["streams"] if stream["codec_type"] == "video"]
    audios = [stream for stream in metadata["streams"] if stream["codec_type"] == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError("Official source must have one unambiguous video and audio stream")
    stream, audio = videos[0], audios[0]
    crop = override.get("crop") if override else None
    if crop is not None:
        if (not isinstance(crop, list) or len(crop) != 4 or any(type(value) is not int for value in crop)
                or min(crop[:2]) < 0 or min(crop[2:]) <= 0
                or crop[0] + crop[2] > stream["width"] or crop[1] + crop[3] > stream["height"]
                or tuple(crop[2:]) != (canvas["width"], canvas["height"])):
            raise ValueError("Verified static crop must fit source pixels and equal the official annotation canvas")
    elif (stream["width"], stream["height"]) != (canvas["width"], canvas["height"]):
        raise ValueError("Video dimensions differ from official annotation canvas; no scaling/letterbox guess permitted")
    if any(float(side.get("rotation", 0)) != 0 for side in stream.get("side_data_list", [])):
        raise ValueError("Rotated video requires an explicit annotation-coordinate contract")
    fps, nominal = Fraction(stream["avg_frame_rate"]), Fraction(stream["r_frame_rate"])
    base = Fraction(stream["time_base"])
    if not 0 < fps <= 120 or abs(float(fps - nominal)) > 1e-5:
        raise ValueError("A verified constant source frame rate is required")
    frame_command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries",
                     "frame=best_effort_timestamp", "-of", "json", str(video)]
    decoded = json.loads(subprocess.check_output(frame_command, timeout=timeout))["frames"]
    all_pts = [int(frame["best_effort_timestamp"]) for frame in decoded]
    prefix = override.get("annotation_prefix_frames") if override else None
    if prefix is not None and (type(prefix) is not int or prefix != expected_frames):
        raise ValueError("annotation_prefix_frames must equal the exact dense annotation count")
    if not all_pts or len(all_pts) < expected_frames or (len(all_pts) != expected_frames and prefix is None):
        raise ValueError(f"Decoded native frames {len(all_pts)} do not equal dense annotation frames {expected_frames}; no verified prefix override")
    # Official Matroska sources contain native 41/42 ms quantization, a small
    # initial timestamp displacement, and occasional doubled tail intervals.
    # Native annotation index still identifies each decoded frame exactly; do
    # not replace its timestamp with i/fps or reject that ordinary container
    # timing. Reject nonmonotonic or implausible intervals instead, and retain
    # the actual integer PTS for every downstream frame/audio/label mapping.
    period = 1 / fps
    intervals = [(right - left) * base for left, right in zip(all_pts, all_pts[1:])]
    minimum_interval = max(base, period / 4 - 2 * base)
    maximum_interval = 3 * period + 2 * base
    if any(not minimum_interval <= interval <= maximum_interval for interval in intervals):
        raise ValueError("Native frame PTS are nonmonotonic or contain implausible timestamp gaps")
    drifts = [(pts - all_pts[0]) * base - Fraction(index, 1) / fps for index, pts in enumerate(all_pts)]
    format_origin = Fraction(str(metadata.get("format", {}).get("start_time", "0")))
    video_origin = all_pts[0] * base
    offset = video_origin - format_origin
    if offset < 0:
        raise ValueError("Video begins before container origin; explicit source timing review required")
    pts = all_pts[:expected_frames]
    observed_span = (pts[-1] - pts[0]) * base
    # With no invented duration for the final decoded frame, output timestamps
    # strictly before its rounded PTS are guaranteed to have real lookahead in
    # FFmpeg's fps filter. Discard only the incomplete two-second tail later.
    guaranteed_output_frames = round_near_nonnegative(observed_span * 25)
    return {"schema_version": PROBE_SCHEMA, "width": canvas["width"], "height": canvas["height"],
            "source_width": stream["width"], "source_height": stream["height"], "static_crop": crop,
            "source_native_frames": len(all_pts), "source_tail_frames_dropped": len(all_pts) - expected_frames,
            "source_full_pts_sha256": value_hash(all_pts), "alignment_override_sha256": value_hash(override) if override else None,
            "native_frames": len(pts), "fps": str(fps), "time_base": str(base), "native_pts": pts,
            "video_stream_index": stream["index"], "audio_stream_index": audio["index"],
            "format_origin": str(format_origin), "video_first_pts": pts[0],
            "video_offset_seconds": float(offset), "video_offset_fraction": str(offset),
            "native_duration_seconds": float(observed_span),
            "native_observed_span_fraction": str(observed_span),
            "guaranteed_canonical_frames": guaranteed_output_frames,
            "pts_audit": {"policy": "strict_monotonic_bounded_native_intervals_actual_pts_mapping",
                          "minimum_interval_seconds": float(min(intervals)) if intervals else None,
                          "maximum_interval_seconds": float(max(intervals)) if intervals else None,
                          "allowed_interval_seconds": [float(minimum_interval), float(maximum_interval)],
                          "minimum_nominal_drift_seconds": float(min(drifts)),
                          "maximum_nominal_drift_seconds": float(max(drifts)),
                          "final_nominal_drift_seconds": float(drifts[-1])}}


def movie_split_plan(movies, seed=114):
    movies = sorted(set(movies))
    if len(movies) != 8 or seed != 114:
        raise ValueError("This official experiment fixes all eight movies and seed 114")
    random.Random(seed).shuffle(movies)
    return {movie: "test" if index == 0 else "val" if index == 1 else "train" for index, movie in enumerate(movies)}


def read_exact(stream, count):
    chunks, remaining = [], count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ValueError("Continuous video decoder ended before the verified output frame count")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def extract_aligned_audio(video, target, probe, output_frames, args):
    offset = float(Fraction(probe["video_offset_fraction"]))
    duration = output_frames / 25
    origin = float(Fraction(probe["format_origin"]))
    filters = (f"asetpts=PTS-({origin:.12f})/TB,aresample=16000:async=1:first_pts=0,"
               f"atrim=start={offset:.12f}:end={offset + duration:.12f},asetpts=PTS-STARTPTS")
    command = [args.ffmpeg, "-nostdin", "-v", "error", "-y", "-copyts", "-i", str(video),
               "-map", f"0:{probe['audio_stream_index']}", "-vn", "-sn", "-af", filters,
               "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)]
    subprocess.run(command, check=True, capture_output=True, timeout=args.timeout)
    with wave.open(str(target), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("Decoded audio is not mono PCM16 at 16 kHz")
        samples = wav.getnframes()
    difference = samples - output_frames * 640
    return {"decoded_samples": samples, "expected_samples": output_frames * 640,
            "tail_difference_samples": difference,
            "tail_adjustment": "drop_complete_tail_windows" if difference < -32 else "zero_pad" if difference < 0 else "trim" if difference else "none",
            "video_offset_seconds_applied": offset, "format_origin_seconds": origin,
            "method": "preserve_common_PTS_then_crop_at_video_offset"}


def common_av_window_audit(video_frames, decoded_samples, window_frames=50, tail_allowance_samples=32):
    """Retain only complete windows supported by real video and aligned PCM."""
    if (type(video_frames) is not int or video_frames < 0
            or type(decoded_samples) is not int or decoded_samples < 0
            or type(window_frames) is not int or window_frames < 1
            or video_frames % window_frames
            or type(tail_allowance_samples) is not int or not 0 <= tail_allowance_samples <= 32):
        raise ValueError("Common A/V support requires nonnegative real counts and complete windows")
    samples_per_window = window_frames * 640
    supported_audio_frames = ((decoded_samples + tail_allowance_samples) // samples_per_window) * window_frames
    retained = min(video_frames, supported_audio_frames)
    return {"policy": "drop_whole_2s_tail_windows_to_common_real_AV_support_max_32_sample_rounding",
            "video_supported_frames": video_frames, "decoded_audio_samples": decoded_samples,
            "retained_canonical_frames": retained, "discarded_video_frames": video_frames - retained,
            "discarded_video_seconds": (video_frames - retained) / 25,
            "unused_decoded_audio_tail_samples": max(0, decoded_samples - retained * 640),
            "retained_audio_rounding_pad_samples": max(0, retained * 640 - decoded_samples),
            "audio_tail_allowance_samples": tail_allowance_samples}


def window_audio(source, output, start_sample, count):
    with wave.open(str(source), "rb") as wav:
        wav.setpos(min(start_sample, wav.getnframes()))
        pcm = wav.readframes(count)
    missing = count - len(pcm) // 2
    if missing > 32:
        raise ValueError("Window audio is missing more than the audited tail allowance")
    pcm += b"\0\0" * missing
    temporary = output.with_name(output.stem + ".tmp.wav")
    with wave.open(str(temporary), "wb") as wav:
        wav.setparams((1, 2, 16000, count, "NONE", "not compressed"))
        wav.writeframes(pcm)
    temporary.replace(output)


def crop_track(frames, boxes, width, height):
    import cv2
    import numpy as np
    normalized = np.asarray(boxes, dtype=np.float64) / [width, height, width, height]
    canonical = normalized * [480, 270, 480, 270]
    crops = []
    for frame, box in zip(frames, canonical):
        left, top, right, bottom = np.rint(box).astype(int)
        if right <= left or bottom <= top:
            raise ValueError("Canonical body crop is smaller than one pixel")
        gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
        crops.append(cv2.resize(gray, (112, 112), interpolation=cv2.INTER_LINEAR))
    return np.stack(crops), normalized


def prepare_clip(row, probe, canvas, ap, der, split, config_hash, args):
    import numpy as np
    clip, movie = row["clip_id"], row["movie"]
    video = locate_video(args.video_dir, clip)
    annotations = local_path(row["annotation_path"], args.selection)
    dense, dense_stats = dense_frames(annotations, movie, canvas["width"], canvas["height"])
    video_complete = probe["guaranteed_canonical_frames"] // args.frames * args.frames
    if video_complete < args.frames:
        raise ValueError("Clip has no complete real two-second window")
    clip_rows, counts = [], Counter()
    work = args.output / "_work"
    work.mkdir(exist_ok=True)
    audio_cache = work / (clip + ".wav")
    try:
        audio_audit = extract_aligned_audio(video, audio_cache, probe, video_complete, args)
        common_av = common_av_window_audit(video_complete, audio_audit["decoded_samples"], args.frames)
        audio_audit["common_av_support"] = common_av
        complete = common_av["retained_canonical_frames"]
        if complete < args.frames:
            raise ValueError("Clip has no complete real two-second window with common video/audio support")
        mapping = resampled_indices(probe["native_pts"], probe["time_base"], complete)
    except Exception:
        audio_cache.unlink(missing_ok=True)
        raise
    if common_av["discarded_video_frames"]:
        emit("clip_common_av_tail_trimmed", clip=clip, **common_av)
    command = [args.ffmpeg, "-nostdin", "-v", "error", "-threads", str(args.decode_threads), "-noautorotate",
               "-copyts", "-i", str(video), "-map", f"0:{probe['video_stream_index']}", "-an", "-sn", "-vf",
               ("crop={2}:{3}:{0}:{1}:exact=1,".format(*probe["static_crop"]) if probe.get("static_crop") else "")
               + f"setpts=PTS-({probe['video_first_pts']}),fps=fps=25:start_time=0:round=near,scale=480:270:flags=bilinear",
               "-frames:v", str(complete), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    annotation_hash = file_hash(annotations)
    offset = float(Fraction(probe["video_offset_fraction"]))
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
            try:
                for start in range(0, complete, args.frames):
                    frames = np.frombuffer(read_exact(process.stdout, args.frames * 270 * 480 * 3), dtype=np.uint8).reshape(args.frames, 270, 480, 3)
                    visible = [dense[mapping[index]] for index in range(start, start + args.frames)]
                    names = set.intersection(*(set(frame) for frame in visible))
                    candidates = sorted(name for name in names if name and name != "Other"
                                        and all(len(frame[name]) == 1 and frame[name][0] is not None for frame in visible))
                    speakers = [agreed_speaker(offset + (index + .5) / 25, ap, der, frame, args.boundary_ms / 1000)
                                for index, frame in zip(range(start, start + args.frames), visible)]
                    counts["video_windows"] += 1
                    counts["candidate_tracks"] += len(candidates)
                    audio_path = args.output / "assets" / clip / f"window-{start:06d}" / "audio.wav"
                    for name in candidates:
                        labels = [-1 if speaker is None else int(speaker == name) for speaker in speakers]
                        if sum(label >= 0 for label in labels) < args.min_valid_frames:
                            counts["insufficient_label_tracks"] += 1
                            continue
                        folder = audio_path.parent
                        folder.mkdir(parents=True, exist_ok=True)
                        if not audio_path.exists():
                            window_audio(audio_cache, audio_path, start * 640, args.frames * 640)
                        sample_id = "cmdam-" + value_hash([POLICY, clip, start, name])[:20]
                        visual_path = folder / (sample_id + ".npz")
                        try:
                            visual, boxes = crop_track(frames, [frame[name][0] for frame in visible], canvas["width"], canvas["height"])
                        except ValueError:
                            counts["subpixel_crop_tracks"] += 1
                            continue
                        with visual_path.with_suffix(".tmp").open("wb") as stream:
                            np.savez_compressed(stream, visual=visual, boxes_xyxy_normalized=boxes)
                        visual_path.with_suffix(".tmp").replace(visual_path)
                        clip_rows.append({"schema_version": 1, "sample_id": sample_id, "split": split,
                            "movie": movie, "clip": clip, "track": name + ":" + str(start), "group": movie,
                            "evaluation_group": movie, "start_frame": start, "end_frame": start + args.frames,
                            "fps": 25, "visual_path": str(visual_path.relative_to(args.output)),
                            "audio_path": str(audio_path.relative_to(args.output)), "audio_start_seconds": 0,
                            "labels": labels, "mask": [label >= 0 for label in labels], "synthetic": False,
                            "machine_generated": True, "human_annotated": False, "official_annotations": True,
                            "label_source": LABEL_SOURCE, "annotation_granularity": "sentence_speaker_and_character_boxes",
                            "official_eval_annotations_repurposed_for_training": True,
                            "preprocessing": PREPROCESSING, "crop_type": "full_body_bbox",
                            "source_native_frame_indices": mapping[start:start + args.frames],
                            "source_format_start_seconds": offset + start / 25,
                            "source_video_offset_seconds": offset, "source_video_sha256": probe["video_sha256"],
                            "source_alignment_override_sha256": probe.get("alignment_override_sha256"),
                            "source_static_crop": probe.get("static_crop"),
                            "source_tail_frames_dropped": probe.get("source_tail_frames_dropped", 0),
                            "source_common_av_tail_discarded_frames": common_av["discarded_video_frames"],
                            "source_annotation_sha256": annotation_hash, "preparation_config_hash": config_hash,
                            "asset_sha256": {"visual": file_hash(visual_path), "audio": file_hash(audio_path)},
                            "quality_limitations": ["sentence speaker alignment is not mouth-motion ground truth",
                                "voiceover is not explicitly annotated and may remain", "no cross-movie benchmark claim on repurposed evaluation data"]})
                        counts["accepted_tracks"] += 1
                    if counts["video_windows"] % 20 == 0:
                        emit("clip_progress", clip=clip, windows=counts["video_windows"], accepted=counts["accepted_tracks"])
                if process.stdout.read(1):
                    raise ValueError("Decoder exceeded the requested canonical frame count")
                if process.wait(timeout=args.timeout):
                    raise ValueError("Continuous video decoding failed")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
    finally:
        audio_cache.unlink(missing_ok=True)
    return clip_rows, {"clip": clip, "movie": movie, "status": "completed", "dense": dense_stats,
                       "canonical_frames": complete, "video_supported_canonical_frames": video_complete,
                       "discarded_tail_seconds": probe["native_duration_seconds"] - complete / 25,
                       "audio_alignment": audio_audit, "counts": dict(counts)}


def training_summary(rows, total_clips):
    splits, reasons = {}, []
    for split, minimum in (("train", 20), ("val", 5), ("test", 5)):
        selected = [row for row in rows if row["split"] == split]
        counts = Counter(label for row in selected for label in row["labels"])
        splits[split] = {"windows": len(selected), "positive_frames": counts[1], "negative_frames": counts[0],
                         "masked_frames": counts[-1], "movies": sorted({row["movie"] for row in selected}),
                         "sha256": value_hash(selected)}
        if len(selected) < minimum or not counts[0] or not counts[1]:
            reasons.append(f"{split}: needs >= {minimum} windows and both nonmasked classes")
    if total_clips != 40:
        reasons.append("All 40 official annotated clips must complete preprocessing")
    return {"schema_version": "cmdam-derived-asd-summary-v1", "training_ready": not reasons,
            "refusal_reasons": reasons, "label_source": LABEL_SOURCE, "human_ground_truth": False,
            "official_annotations": True, "annotation_granularity": "sentence_speaker_and_character_boxes",
            "official_eval_annotations_repurposed_for_training": True,
            "evaluation_interpretation": "Movie-heldout performance on derived speaker/body alignment labels; not an official CMD-AM benchmark or mouth-motion accuracy",
            "processed_clips": total_clips, "accepted_windows": len(rows), "splits": splits}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=PROJECT / "training/lrasd/cmdam/download-selection.json")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--resource", type=Path, default=PROJECT / "training/lrasd/cmdam/cmdam_boxes.json")
    parser.add_argument("--alignment-overrides", type=Path,
                        help="Explicit source-hash-bound static crops/native prefix trims, after manual alignment review")
    parser.add_argument("--ap", type=Path, default=PROJECT / "test-results/lrasd-200step-20261015/dataset-source/cmd_test_audio_recognition_annotations_ap.json")
    parser.add_argument("--der", type=Path, default=PROJECT / "test-results/lrasd-200step-20261015/dataset-source/cmd_test_closed_set_annotations_der_gt.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "training/lrasd/cmdam/prepared-v1")
    parser.add_argument("--frames", type=int, choices=(50,), default=50)
    parser.add_argument("--min-valid-frames", type=int, default=20)
    parser.add_argument("--boundary-ms", type=int, default=80)
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--prepare-workers", type=int, choices=range(1, 5), default=1,
                        help="Independent per-clip extraction workers; aggregation remains selection ordered")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args(argv)
    if not 20 <= args.min_valid_frames <= args.frames or args.boundary_ms < 80 or min(args.decode_threads, args.timeout) < 1:
        parser.error("Keep >=20 valid frames, >=80 ms boundary masks, and positive decoder limits")
    args.output.mkdir(parents=True, exist_ok=True)
    selection, resource = selection_records(args.selection), resource_records(args.resource)
    overrides = alignment_overrides(args.alignment_overrides, selection)
    ap, der = load_speakers(args.ap, selection), load_speakers(args.der, selection)
    probes, failures = {}, []
    probe_dir = args.output / "probes"
    probe_dir.mkdir(exist_ok=True)
    for row in selection:
        clip = row["clip_id"]
        try:
            canvas = resource[clip]
            if canvas["movie"] != row["movie"]:
                raise ValueError("Resource and selection movies disagree")
            annotation = local_path(row["annotation_path"], args.selection)
            if file_hash(annotation) != row["annotation_sha256"]:
                raise ValueError("Dense annotation hash differs from the audited selection")
            dense, _ = dense_frames(annotation, row["movie"], canvas["width"], canvas["height"])
            if len(dense) != row["annotation_frames"] or any(not 0 <= frame < len(dense) for frame in canvas["keyframe_indices"]):
                raise ValueError("Official native frame counts/indices disagree")
            video = locate_video(args.video_dir, clip)
            source_stat = video.stat()
            source_hash = file_hash(video)
            if (video.stat().st_size, video.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
                raise ValueError("Source video changed while hashing; wait for its verified download to finish")
            override = overrides.get(clip)
            if override and override["source_sha256"] != source_hash:
                raise ValueError("Verified alignment override belongs to different source video bytes")
            cache = probe_dir / (clip + ".json")
            probe = read_json(cache) if cache.exists() else None
            if (not isinstance(probe, dict) or probe.get("schema_version") != PROBE_SCHEMA
                    or probe.get("video_sha256") != source_hash or probe.get("native_frames") != len(dense)
                    or (probe.get("width"), probe.get("height")) != (canvas["width"], canvas["height"])
                    or probe.get("alignment_override_sha256") != (value_hash(override) if override else None)):
                probe = probe_video(video, len(dense), canvas, args.ffprobe, args.timeout, override)
                probe["video_sha256"] = source_hash
                write_json(cache, probe)
            if (video.stat().st_size, video.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
                raise ValueError("Source video changed during native-frame probing")
            # Literal Other/collective tails generate only masked labels. Their
            # extent can outlive the annotated native video prefix without
            # inventing speech labels; retain the discrepancy as audit data.
            probe["speaker_annotation_extent_audit"] = speaker_extent_audit(probe, ap[clip], der[clip])
            write_json(cache, probe)
            probes[clip] = probe
            emit("clip_probe_verified", clip=clip, movie=row["movie"], native_frames=probe["native_frames"],
                 canvas=[probe["width"], probe["height"]], source_fps=probe["fps"], video_offset_seconds=probe["video_offset_seconds"])
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as error:
            failures.append({"clip": clip, "error": f"{type(error).__name__}: {error}"})
            emit("clip_probe_rejected", clip=clip, reason=str(error))
    write_json(args.output / "probe-summary.json", {"verified_clips": len(probes), "required_clips": 40,
               "failures": failures, "split_plan_frozen": False})
    if failures or len(probes) != 40:
        emit("blocked_missing_or_invalid_sources", verified=len(probes), failures=len(failures))
        return 2
    if args.probe_only:
        return 0
    plan = movie_split_plan([row["movie"] for row in selection])
    config = {"policy": POLICY, "selection_sha256": file_hash(args.selection), "resource_sha256": file_hash(args.resource),
              "ap_sha256": file_hash(args.ap), "der_sha256": file_hash(args.der), "frames": args.frames,
              "min_valid_frames": args.min_valid_frames, "boundary_ms": args.boundary_ms,
              "source_video_hashes": {clip: probe["video_sha256"] for clip, probe in probes.items()},
              "alignment_overrides_sha256": file_hash(args.alignment_overrides) if args.alignment_overrides else None,
              "alignment_override_hashes": {clip: value_hash(value) for clip, value in overrides.items()},
              "movie_split": plan, "seed": 114, "preprocessing": PREPROCESSING,
              "character_aliases": CHARACTER_ALIASES, "collective_names_masked_as_speaker": sorted(COLLECTIVE_NAMES),
              "video_resampling": "actual_native_integer_PTS_quantized_like_ffmpeg_fps_25_round_near",
              "native_probe_schema": PROBE_SCHEMA,
              "video_tail_boundary": "full_2s_windows_strictly_before_last_known_native_frame_rounded_pts",
              "common_av_tail_policy": "drop_whole_2s_tail_windows_to_common_real_AV_support_max_32_sample_rounding",
              "audio_tail_allowance_samples": 32, "official_eval_annotations_repurposed_for_training": True}
    config_hash = value_hash(config)
    config_path = args.output / "preparation-config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing preparation belongs to different immutable sources/policy; use a new output directory")
    write_json(config_path, config)
    write_json(args.output / "split-plan.json", {"strategy": "movie_holdout_6_1_1", "seed": 114, "movies": plan,
               "official_eval_annotations_repurposed_for_training": True})
    all_rows, clip_audits = [], []
    per_clip = args.output / "clip-manifests"
    per_clip.mkdir(exist_ok=True)
    def prepare_one(row):
        clip = row["clip_id"]
        rows_path, audit_path = per_clip / (clip + ".jsonl"), per_clip / (clip + ".audit.json")
        if rows_path.exists() and audit_path.exists() and read_json(audit_path).get("config_hash") == config_hash:
            audit = read_json(audit_path)
            if file_hash(rows_path) != audit["rows_sha256"]:
                raise ValueError("Cached clip manifest hash mismatch")
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows, audit = prepare_clip(row, probes[clip], resource[clip], ap[clip], der[clip], plan[row["movie"]], config_hash, args)
            temporary = rows_path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in rows), encoding="utf-8")
            temporary.replace(rows_path)
            audit.update(config_hash=config_hash, rows_sha256=file_hash(rows_path))
            write_json(audit_path, audit)
        return clip, rows, audit

    with ThreadPoolExecutor(max_workers=args.prepare_workers) as executor:
        for clip, rows, audit in executor.map(prepare_one, selection):
            all_rows.extend(rows)
            clip_audits.append(audit)
            emit("clip_completed", clip=clip, accepted_windows=len(rows), completed_clips=len(clip_audits))
            write_json(args.output / "progress.json", {"status": "preparing", "completed_clips": len(clip_audits),
                       "accepted_windows": len(all_rows), "config_hash": config_hash,
                       "prepare_workers": args.prepare_workers})
    target = args.output / "labeled.jsonl"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in all_rows), encoding="utf-8")
    temporary.replace(target)
    summary = training_summary(all_rows, len(clip_audits))
    summary.update(config_hash=config_hash, labeled_manifest_sha256=file_hash(target), clip_audits=clip_audits)
    write_json(args.output / "label-summary.json", summary)
    write_json(args.output / "progress.json", {"status": "completed" if summary["training_ready"] else "insufficient_labels_do_not_train",
               "training_ready": summary["training_ready"], "accepted_windows": len(all_rows), "completed_clips": len(clip_audits)})
    emit("preparation_completed", training_ready=summary["training_ready"], accepted_windows=len(all_rows),
         refusal_reasons=summary["refusal_reasons"])
    return 0 if summary["training_ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
