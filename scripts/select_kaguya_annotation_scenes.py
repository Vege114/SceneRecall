#!/usr/bin/env python3
"""Propose separated manual-annotation scenes using subtitle timing only.

This never produces ASD labels or speaker identities. Candidates need visual
inspection before freezing the final annotation plan and its scene groups.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


DIALOGUE_STYLES = frozenset({"Default", "Default-UP", "Default2"})


def ass_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_text(value: str) -> str:
    value = re.sub(r"\{[^}]*\}", "", value)
    return re.sub(r"\s+", " ", value.replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")).strip()


def parse_ass(path: Path) -> list[dict]:
    result = []
    seen = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        fields = line.split(":", 1)[1].lstrip().split(",", 9)
        if len(fields) != 10:
            continue
        _, start, end, style, name, _, _, _, _, raw_text = fields
        start, end = ass_seconds(start), ass_seconds(end)
        text = clean_text(raw_text)
        key = (start, end, style, text)
        if not text or end <= start or key in seen:
            continue
        seen.add(key)
        result.append({"start": start, "end": end, "style": style, "name": name, "text": text})
    return sorted(result, key=lambda event: (event["start"], event["end"], event["text"]))


def merge_intervals(intervals: list[tuple[float, float]], gap: float = 2) -> list[list[float]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def overlaps(start: float, end: float, intervals: list[list[float]], margin: float = 0) -> bool:
    return any(start < right + margin and end > left - margin for left, right in intervals)


def propose_scenes(events: list[dict], count: int = 36, start: float = 180, end: float = 7800,
                   duration: float = 6, minimum_separation: float = 140) -> tuple[list[dict], list[list[float]]]:
    if count < 1 or duration <= 0 or end <= start or minimum_separation < duration:
        raise ValueError("invalid candidate-selection configuration")
    dialogue = [event for event in events if event["style"] in DIALOGUE_STYLES]
    # ASS karaoke is often many overlapping rendered fragments. Merge by time,
    # not by the count or text of these duplicated animation events.
    songs = merge_intervals([(event["start"], event["end"]) for event in events
                             if event["style"].startswith(("IN", "ED-"))])
    candidates = []
    for event in dialogue:
        center = (event["start"] + event["end"]) / 2
        left = round((center - duration / 2) * 25) / 25
        right = left + duration
        if left < start or right > end or overlaps(left, right, songs, margin=2):
            continue
        nearby = [item for item in dialogue if item["start"] < right and item["end"] > left]
        covered = merge_intervals([(max(left, item["start"]), min(right, item["end"]))
                                  for item in nearby], gap=0)
        coverage = sum(b - a for a, b in covered) / duration
        # Prefer multiple short subtitle turns with audible pauses. This is a
        # selection heuristic, not evidence of who is visible or speaking.
        score = min(len(nearby), 3) * 2 + 2 * (1 - abs(coverage - .75))
        if len(nearby) > 5:
            score -= len(nearby) - 5
        candidates.append({"center_seconds": round(left + duration / 2, 2),
                           "start_ms": round(left * 1000), "end_ms": round(right * 1000),
                           "subtitle_turns": len(nearby), "subtitle_coverage": round(coverage, 3),
                           "score": round(score, 3), "dialogue": nearby,
                           "selection_basis": "dialogue_timing_only_music_styles_excluded",
                           "human_active_speaker_ground_truth": False,
                           "requires_visual_review": True})
    selected = []
    width = (end - start) / count
    # One candidate per equally sized temporal band prevents dense early
    # dialogue from displacing later scenes. Empty music bands remain empty.
    for index in range(count):
        band_start, band_end = start + index * width, start + (index + 1) * width
        band_center = (band_start + band_end) / 2
        eligible = [item for item in candidates if band_start <= item["center_seconds"] < band_end
                    and all(abs(item["center_seconds"] - chosen["center_seconds"]) >= minimum_separation
                            for chosen in selected)]
        if not eligible:
            continue
        chosen = max(eligible, key=lambda item: (item["score"] - abs(item["center_seconds"] - band_center) / width,
                                               -item["center_seconds"]))
        selected.append(chosen)
    for number, item in enumerate(selected, 1):
        item["candidate_id"] = f"kaguya-scene-{number:02d}"
        seconds = int(item["center_seconds"])
        item["timecode"] = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
        item["proposed_scene_neighborhood_ms"] = [item["start_ms"] - 45000, item["end_ms"] + 45000]
    return selected, songs


def prior_training_ranges(manifests: list[Path], source_sha256: str) -> list[dict]:
    """Read exact-movie train windows; validation/test rows never count as training."""
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("source video SHA256 is required for the previous-training audit")
    seen = {}
    for filename in manifests:
        for number, line in enumerate(filename.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("split") != "train" or item.get("source_sha256") != source_sha256:
                continue
            start, fps = item.get("source_start_seconds"), item.get("fps")
            labels = item.get("labels")
            if (isinstance(start, bool) or not isinstance(start, (float, int)) or not math.isfinite(start)
                    or start < 0 or isinstance(fps, bool) or not isinstance(fps, (float, int))
                    or not math.isfinite(fps) or fps <= 0 or not isinstance(labels, list) or not labels):
                raise ValueError(f"invalid source timing at {filename}:{number}")
            end = start + len(labels) / fps
            key = (start, end)
            reference = {"manifest": str(filename), "sample_id": item.get("sample_id"), "line": number}
            if key in seen:
                seen[key]["training_examples"].append(reference)
            else:
                seen[key] = {"start_seconds": start, "end_seconds": end, "training_examples": [reference]}
    return sorted(seen.values(), key=lambda item: (item["start_seconds"], item["end_seconds"]))


def audit_holdout_neighborhoods(candidates: list[dict], training_ranges: list[dict], margin: float = 120) -> None:
    """Temporal screening alone cannot establish independent semantic scenes."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("holdout exclusion margin must be finite and nonnegative")
    for candidate in candidates:
        start, end = candidate["start_ms"] / 1000, candidate["end_ms"] / 1000
        intersections = [item for item in training_ranges
                         if start < item["end_seconds"] + margin and end > item["start_seconds"] - margin]
        candidate["prior_training_neighborhood_overlap"] = intersections
        candidate["holdout_eligible_after_temporal_audit"] = not intersections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--start", type=float, default=180)
    parser.add_argument("--end", type=float, default=7800)
    parser.add_argument("--duration", type=float, default=6)
    parser.add_argument("--minimum-separation", type=float, default=140)
    parser.add_argument("--source", type=Path, default=Path("training/lrasd/kaguya/source.json"),
                        help="Source metadata carrying the exact video_sha256, not a movie-title match")
    parser.add_argument("--prior-training-manifest", type=Path, action="append",
                        help="Repeat for every earlier training manifest; default: the two existing JP2D label manifests")
    parser.add_argument("--holdout-exclusion-margin", type=float, default=120)
    args = parser.parse_args()
    events = parse_ass(args.subtitles)
    selected, songs = propose_scenes(events, args.count, args.start, args.end, args.duration,
                                    args.minimum_separation)
    source = json.loads(args.source.read_text(encoding="utf-8-sig"))
    manifests = args.prior_training_manifest or [Path("training/lrasd/labels-2s-v1/labeled.jsonl"),
                                                Path("training/lrasd/teacher-multi-pilot2/labels/labeled.jsonl")]
    training_ranges = prior_training_ranges(manifests, source.get("video_sha256", ""))
    audit_holdout_neighborhoods(selected, training_ranges, args.holdout_exclusion_margin)
    result = {"schema_version": "kaguya-annotation-candidates-v1",
              "human_active_speaker_ground_truth": False,
              "subtitle_sha256": hashlib.sha256(args.subtitles.read_bytes()).hexdigest(),
              "subtitle_style_counts": dict(Counter(event["style"] for event in events)),
              "configuration": {"count": args.count, "start": args.start, "end": args.end,
                                "duration": args.duration, "minimum_separation": args.minimum_separation},
              "excluded_song_intervals_seconds": songs,
              "source_video_sha256": source["video_sha256"],
              "previous_machine_training_source_intervals": training_ranges,
              "holdout_exclusion_margin_seconds": args.holdout_exclusion_margin,
              "prior_training_manifest_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                 for path in manifests},
              "candidates": selected,
              "final_plan_policy": "Visually review candidates, freeze 24 scene groups, assign 16 train/4 val/4 test; "
                                   "all tracks and windows from the same scene must share a split. "
                                   "Validation/test scenes must pass the previous-training neighborhood audit. "
                                   "No labels or speaker identities may be copied from subtitles."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(selected), "song_ranges": len(songs), "output": str(args.output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
