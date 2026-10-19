#!/usr/bin/env python3
"""Inspect only the user's explicitly authorized Revue Starlight TV files.

External ASS timing/style/actor fields are source evidence, not frame-level
active-speaker ground truth. Empty actor fields never become speaker labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall.lrasd_training_data import atomic_json  # noqa: E402

AUTHORIZED_GLOB = "*Shoujo Kageki Revue Starlight*"
EPISODE = re.compile(r"\[(\d{2})\]\[Ma10p_1080p\]")
DIALOGUE_STYLE = re.compile(r"^(?:default\d*|main|dialogue|dialog|对白|對白)$", re.I)
MUSIC_STYLE = re.compile(r"^(?:op(?:[-\d]|$)|ed(?:[-\d]|$)|o-|song|lyric|karaoke|music)|歌词|歌詞|插曲", re.I)
KARAOKE = re.compile(r"\\[kK](?:f|o)?\d")


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def discover(directory):
    # Do not enumerate unrelated Downloads files, recurse, or inspect other media.
    selected = sorted(path for path in directory.glob(AUTHORIZED_GLOB) if path.is_file())
    videos, subtitles = {}, {}
    for path in selected:
        match = EPISODE.search(path.name)
        if not match:
            continue
        episode = int(match[1])
        bucket = videos if path.suffix.lower() == ".mkv" else subtitles if path.name.endswith(".sc.ass") else None
        if bucket is not None:
            if episode in bucket:
                raise ValueError(f"Ambiguous matching source for episode {episode:02d}")
            bucket[episode] = path.resolve()
    expected = set(range(1, 13))
    if set(videos) != expected or set(subtitles) != expected:
        raise ValueError(f"Expected episodes 01-12 with .mkv and .sc.ass: videos={sorted(videos)}, subtitles={sorted(subtitles)}")
    return [(episode, videos[episode], subtitles[episode]) for episode in sorted(expected)]


def seconds(value):
    hours, minutes, tail = value.strip().split(":")
    return round(int(hours) * 3600 + int(minutes) * 60 + float(tail), 3)


def merge_intervals(intervals, gap=.08):
    merged = []
    for start, end in sorted(intervals):
        if start < 0 or end <= start:
            raise ValueError("Invalid subtitle presentation interval")
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [[round(start, 3), round(end, 3)] for start, end in merged]


def subtract_intervals(intervals, excluded, margin=.5):
    result = []
    for start, end in intervals:
        pieces = [[start, end]]
        for left, right in excluded:
            left, right = max(0, left - margin), right + margin
            next_pieces = []
            for a, b in pieces:
                if right <= a or left >= b:
                    next_pieces.append([a, b])
                else:
                    if a < left:
                        next_pieces.append([a, left])
                    if right < b:
                        next_pieces.append([right, b])
            pieces = next_pieces
        result.extend(pieces)
    return [[round(start, 3), round(end, 3)] for start, end in result]


def parse_ass(path):
    section, columns = "", []
    styles, actors, dialogue_actors = Counter(), Counter(), Counter()
    dialogue, music, other = [], [], []
    events, malformed = [], 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.casefold()
        elif section == "[events]" and stripped.startswith("Format:"):
            columns = [part.strip().casefold() for part in stripped[7:].split(",")]
        elif section == "[events]" and stripped.startswith("Dialogue:"):
            if not columns:
                raise ValueError(f"ASS events have no Format: {path.name}")
            values = stripped[9:].lstrip().split(",", len(columns) - 1)
            if len(values) != len(columns):
                malformed += 1
                continue
            row = dict(zip(columns, values))
            try:
                start, end = seconds(row["start"]), seconds(row["end"])
            except (KeyError, ValueError):
                malformed += 1
                continue
            if start < 0 or end <= start:
                malformed += 1
                continue
            style, actor = row.get("style", ""), row.get("name", row.get("actor", "")).strip()
            styles[style] += 1
            actors[actor or "<empty>"] += 1
            text = row.get("text", "")
            is_music = bool(MUSIC_STYLE.search(style) or KARAOKE.search(text))
            is_dialogue = bool(DIALOGUE_STYLE.fullmatch(style)) and not is_music
            kind = "music" if is_music else "dialogue" if is_dialogue else "other"
            {"music": music, "dialogue": dialogue, "other": other}[kind].append([start, end])
            if is_dialogue:
                dialogue_actors[actor or "<empty>"] += 1
            events.append({"start_seconds": start, "end_seconds": end, "style": style,
                           "actor_name": actor or None, "kind": kind})
    merged_dialogue, merged_music = merge_intervals(dialogue), merge_intervals(music)
    return {"schema_version": "lrasd-ass-intervals-v1", "time_unit": "seconds", "interval_format": "half_open_start_end_pairs",
            "subtitle_path": str(path.resolve()), "subtitle_sha256": sha256(path),
            "dialogue_intervals": merged_dialogue, "music_intervals": merged_music,
            "clean_dialogue_intervals": subtract_intervals(merged_dialogue, merged_music),
            "other_intervals": merge_intervals(other), "music_exclusion_margin_seconds": .5,
            "actor_name_counts": dict(actors), "dialogue_actor_name_counts": dict(dialogue_actors),
            "style_counts": dict(styles), "dialogue_styles": sorted({row["style"] for row in events if row["kind"] == "dialogue"}),
            "music_styles": sorted({row["style"] for row in events if row["kind"] == "music"}),
            "unclassified_styles": sorted({row["style"] for row in events if row["kind"] == "other"}),
            "events": events, "malformed_events": malformed,
            "has_nonempty_actor_fields": any(name != "<empty>" for name in actors),
            "has_nonempty_dialogue_actor_fields": any(name != "<empty>" for name in dialogue_actors),
            "human_active_speaker_ground_truth": False,
            "interpretation": "ASS dialogue styles provide timing candidates only. Empty actor Name fields provide no speaker labels. "
                              "Music detection is a conservative style/karaoke heuristic, not a complete singing annotation."}


def probe(path, executable):
    result = subprocess.run([executable, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                            text=True, capture_output=True, check=True, timeout=120)
    value = json.loads(result.stdout)
    audio = [stream for stream in value.get("streams", []) if stream.get("codec_type") == "audio"]
    japanese = [stream for stream in audio if stream.get("tags", {}).get("language", "").casefold() in {"jpn", "ja", "japanese"}]
    if not japanese:
        raise ValueError(f"No Japanese audio language tag verified for {path.name}")
    selected = sorted(japanese, key=lambda row: (not row.get("disposition", {}).get("default"), row["index"]))[0]
    video = next(stream for stream in value["streams"] if stream.get("codec_type") == "video"
                 and not stream.get("disposition", {}).get("attached_pic"))
    return value, selected, video


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, default=Path("training/lrasd/revue"))
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--skip-video-hash", action="store_true", help="Write explicitly pending hash state for fast source preparation")
    return parser.parse_args()


def main(args):
    sources = discover(args.directory)
    parsed = [(episode, video, subtitle, parse_ass(subtitle)) for episode, video, subtitle in sources]
    actor_totals, dialogue_actor_totals, style_totals = Counter(), Counter(), Counter()
    for _, _, _, data in parsed:
        actor_totals.update(data["actor_name_counts"])
        dialogue_actor_totals.update(data["dialogue_actor_name_counts"])
        style_totals.update(data["style_counts"])
    inspection = {"episodes": len(sources), "episode_numbers": [row[0] for row in sources],
                  "actor_name_counts": dict(actor_totals), "dialogue_actor_name_counts": dict(dialogue_actor_totals),
                  "style_counts": dict(style_totals),
                  "episodes_with_nonempty_actor_fields": [episode for episode, _, _, data in parsed if data["has_nonempty_actor_fields"]],
                  "human_active_speaker_ground_truth": False}
    print(json.dumps({"event": "source_discovery", **inspection}, ensure_ascii=False), flush=True)
    if args.inspect_only:
        return inspection
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "source-inspection.json", inspection)
    descriptors = []
    for episode, video_path, subtitle_path, intervals in parsed:
        directory = args.output / f"episode-{episode:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        intervals_path = directory / "dialogue-intervals.json"
        atomic_json(intervals_path, intervals)
        metadata, audio, video = probe(video_path, args.ffprobe)
        probe_path = directory / "probe.json"
        atomic_json(probe_path, metadata)
        stat = video_path.stat()
        descriptor = {"schema_version": "lrasd-local-source-v1", "video_path": str(video_path),
                      "subtitle_path": str(subtitle_path), "movie": "Shoujo Kageki Revue Starlight TV",
                      "subtitle_offset_seconds": 0,
                      "clip": f"revue-episode-{episode:02d}", "episode": episode,
                      "evaluation_group": f"revue-episode-{episode:02d}", "group": f"revue-episode-{episode:02d}",
                      "split": "train" if episode <= 8 else "val" if episode <= 10 else "test",
                      "audio_stream_index": audio["index"], "audio_language": audio["tags"]["language"],
                      "audio_language_verification": "ffprobe_stream_language_tag",
                      "video_stream_index": video["index"], "width": video["width"], "height": video["height"],
                      "duration_seconds": float(metadata["format"]["duration"]),
                      "presentation_origin_seconds": float(metadata["format"].get("start_time", 0)),
                      "dialogue_intervals_path": str(intervals_path.resolve()), "probe_path": str(probe_path.resolve()),
                      "subtitle_sha256": intervals["subtitle_sha256"], "video_sha256": None,
                      "video_hash_status": "pending", "video_size_bytes": stat.st_size, "video_mtime_ns": stat.st_mtime_ns,
                      "dataset_domain": "Modern Japanese 2D animated television; local user-authorized source",
                      "has_nonempty_ass_actor_fields": intervals["has_nonempty_actor_fields"],
                      "has_nonempty_dialogue_actor_fields": intervals["has_nonempty_dialogue_actor_fields"],
                      "human_active_speaker_ground_truth": False,
                      "label_source": "unlabeled_candidate_source",
                      "source_authorization": "User explicitly authorized these 12 local videos and external simplified-Chinese ASS subtitles"}
        source_path = args.output / f"source-{episode:02d}.json"
        atomic_json(source_path, descriptor)
        descriptors.append((source_path, descriptor, video_path))
        print(json.dumps({"event": "episode_prepared", "episode": episode, "split": descriptor["split"],
                          "japanese_audio_stream_index": audio["index"], "source": str(source_path),
                          "dialogue_intervals": len(intervals["dialogue_intervals"]),
                          "music_intervals": len(intervals["music_intervals"])}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "all_source_descriptors_ready", "directory": str(args.output.resolve())}), flush=True)
    if not args.skip_video_hash:
        for source_path, descriptor, video_path in descriptors:
            descriptor["video_sha256"] = sha256(video_path)
            descriptor["video_hash_status"] = "complete"
            atomic_json(source_path, descriptor)
            print(json.dumps({"event": "video_hash_completed", "episode": descriptor["episode"]}), flush=True)
    atomic_json(args.output / "sources.json", {"schema_version": "lrasd-local-source-catalog-v1",
                "sources": [str(path.resolve()) for path, _, _ in descriptors], "inspection": inspection,
                "split_policy": "episodes 01-08 train, 09-10 validation, 11-12 test; episode identities never cross splits",
                "human_active_speaker_ground_truth": False,
                "evaluation_scope": "Episode holdout on machine weak labels; no independent human ASD validation"})
    return inspection


if __name__ == "__main__":
    main(parse_args())
