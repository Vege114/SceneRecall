#!/usr/bin/env python3
"""Validate local ASS container tracks without retaining media or subtitle text.

Run from an installed development environment, for example:
    python scripts/verify_container_subtitles.py --media-dir /path/to/media \
        --glob '*.mkv' --expected-episodes 12 --output /tmp/subtitle-verification.json

Every selectable text track is extracted and parsed by the application, then
compared with independently read ffprobe packet text and presentation timestamps.
The packet text reference currently supports ASS/SSA; other codecs are reported
as blocked, never silently passed. ASS timestamps have centisecond precision, so
the default accepted timestamp difference is 10 ms (the actual maximum is saved).
An isolated, temporary library imports every episode in auto mode and performs
at least three uniformly distributed literal searches through the local index.
No provider is configured; attempts to invoke one fail. Temporary subtitle files,
the library, and its index are removed at the end. Reports contain no cue text,
queries, full local paths, or source media. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import pysubs2
from pysubs2.formats.substation import parse_tags

from scenerecall import media, subtitles
from scenerecall.library import Library
from scenerecall.models import AssetInput
from scenerecall.search import SearchEngine, normalize


class NoProviders:
    """An executable guarantee that this verification cannot invoke a model."""

    def __getattr__(self, name: str):
        raise AssertionError("Provider access is forbidden during local verification")


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str):
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs):
        if tag == "br":
            self.parts.append("\n")


def language(text: str) -> str:
    for pattern, code in ((r"[\u3040-\u30ff]", "ja"), (r"[\uac00-\ud7af]", "ko"),
                          (r"[\u3400-\u9fff]", "zh")):
        if re.search(pattern, text):
            return code
    return "und"


def displayed_groups(event: pysubs2.SSAEvent) -> list[str]:
    """Independent reference cleaning; never call the application's parser here."""
    fragments = parse_tags(event.text)
    displayed = "".join(text for text, style in fragments if not style.drawing)
    displayed = displayed.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
    html = PlainText()
    html.feed(displayed)
    groups: list[tuple[str, str]] = []
    for line in "".join(html.parts).splitlines():
        value = line.strip()
        if not value:
            continue
        code = language(value)
        if groups and groups[-1][1] == code:
            previous, _ = groups.pop()
            groups.append((previous + "\n" + value, code))
        else:
            groups.append((value, code))
    return [value for value, _ in groups]


def packet_bytes(dump: str) -> bytes:
    # ffprobe -show_data places a 39-character hexadecimal column before ASCII.
    # Never consume the printable column: it may itself contain hexadecimal text.
    chunks = []
    for line in dump.splitlines():
        if not line:
            continue
        address, hexadecimal = line.split(": ", 1)
        if not re.fullmatch(r"[0-9a-fA-F]+", address):
            raise ValueError("Invalid packet dump")
        chunks.append(bytes.fromhex(hexadecimal[:39]))
    return b"".join(chunks)


def packet_reference(path: Path, stream: int, metadata: dict) -> tuple[list[dict], dict]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required")
    process = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", str(stream), "-show_packets", "-show_data",
         "-show_entries", "packet=pts_time,duration_time,data", "-of", "json", str(path)],
        check=True, capture_output=True, timeout=600,
    )
    packets = json.loads(process.stdout).get("packets", [])
    counts = {"packets": len(packets), "filtered_drawing_packets": 0,
              "filtered_empty_packets": 0, "filtered_outside_packets": 0,
              "clipped_packets": 0, "packet_reference_errors": 0}
    reference = []
    for packet in packets:
        try:
            # Matroska ASS payload: ReadOrder,Layer,Style,Name,MarginL,MarginR,
            # MarginV,Effect,Text. Text can contain arbitrary additional commas.
            fields = packet_bytes(packet["data"]).decode("utf-8").split(",", 8)
            if len(fields) != 9:
                raise ValueError("Invalid Matroska ASS packet")
            event = pysubs2.SSAEvent(text=fields[8])
            groups = displayed_groups(event)
            if not groups:
                counts["filtered_drawing_packets" if event.is_drawing else "filtered_empty_packets"] += 1
                continue
            start = round(float(packet["pts_time"]) * 1000) - metadata["start_time_ms"]
            end = start + round(float(packet["duration_time"]) * 1000)
            clipped_start, clipped_end = max(0, start), min(metadata["duration_ms"], end)
            if clipped_end <= clipped_start:
                counts["filtered_outside_packets"] += 1
                continue
            if (start, end) != (clipped_start, clipped_end):
                counts["clipped_packets"] += 1
            reference.extend({"text": value, "start_ms": clipped_start, "end_ms": clipped_end}
                             for value in groups)
        except (KeyError, ValueError, UnicodeDecodeError):
            counts["packet_reference_errors"] += 1
    # ASS comments are not muxed subtitle packets and cannot be counted here.
    return reference, counts


def compare_cues(cues: list[dict], reference: list[dict], tolerance_ms: int) -> dict:
    by_text = defaultdict(list)
    for item in reference:
        by_text[item["text"]].append(item)
    deltas = []
    unexpected = 0
    timing_failures = 0
    for cue in cues:
        options = by_text[cue["text"]]
        if not options:
            unexpected += 1
            continue
        expected = min(options, key=lambda item: abs(item["start_ms"] - cue["start_ms"])
                       + abs(item["end_ms"] - cue["end_ms"]))
        options.remove(expected)
        delta = max(abs(cue["start_ms"] - expected["start_ms"]),
                    abs(cue["end_ms"] - expected["end_ms"]))
        deltas.append(delta)
        timing_failures += int(delta > tolerance_ms)
    missing = sum(len(items) for items in by_text.values())
    return {"parsed_cues": len(cues), "expected_cues": len(reference),
            "matched_cues": len(deltas), "unexpected_cues": unexpected, "missing_cues": missing,
            "timing_failures": timing_failures, "max_timestamp_delta_ms": max(deltas, default=0),
            "first_start_ms": min((cue["start_ms"] for cue in cues), default=None),
            "last_end_ms": max((cue["end_ms"] for cue in cues), default=None),
            "failure_count": unexpected + missing + timing_failures}


def search_samples(cues: list[dict], count: int) -> list[dict]:
    # Prefer unique dialogue so a recurring opening lyric cannot displace the
    # intended timestamp from the first 50 results. No query text is reported.
    occurrences = Counter(normalize(cue["text"]) for cue in cues)
    eligible = [cue for cue in cues if 4 <= len(normalize(cue["text"]))
                and len(cue["text"]) <= 2000 and occurrences[normalize(cue["text"])] == 1]
    if len(eligible) < count:
        raise ValueError("Not enough distinct subtitle cues for search sampling")
    return [eligible[(2 * index + 1) * len(eligible) // (2 * count)] for index in range(count)]


async def verify_import(path: Path, episode: int | None, library: Library, engine: SearchEngine,
                        track_cues: dict[int, list[dict]], samples: int) -> dict:
    asset = library.register(AssetInput(video_path=str(path), subtitle_mode="auto", episode=episode))
    cues = library.subtitles(asset["id"])
    selected = asset.get("subtitle_stream_index")
    expected = track_cues.get(selected, [])
    def projection(values):
        return Counter((cue["start_ms"], cue["end_ms"], cue["text"]) for cue in values)
    imported_matches = bool(expected) and projection(cues) == projection(expected)
    # This is the same local index refresh used by the asset registration route;
    # no worker run, OCR frame, or embedding operation is needed.
    await engine.rebuild(library.records())
    searches = []
    for sample in search_samples(cues, samples):
        result = await engine.search(sample["text"], filters={"asset_id": asset["id"]}, limit=50)
        hits = [row for row in result["results"] if row["asset_id"] == asset["id"]
                and row["start_ms"] == sample["start_ms"] and row["end_ms"] == sample["end_ms"]
                and row["match_type"] == "full" and not row.get("is_context")]
        searches.append({"target_start_ms": sample["start_ms"], "target_end_ms": sample["end_ms"],
                         "passed": bool(hits)})
    failures = int(not imported_matches) + sum(not item["passed"] for item in searches)
    return {"status": "passed" if not failures else "failed", "selected_stream_index": selected,
            "imported_cues": len(cues), "matches_extracted_track": imported_matches,
            "searches": searches, "searches_passed": sum(item["passed"] for item in searches),
            "searches_failed": sum(not item["passed"] for item in searches), "failure_count": failures}


async def verify(args: argparse.Namespace) -> dict:
    paths = sorted(path for path in args.media_dir.expanduser().glob(args.glob) if path.is_file())
    episode_pattern = re.compile(args.episode_pattern)
    episodes = [(path, int(match[1]) if (match := episode_pattern.search(path.name)) else None) for path in paths]
    found = [episode for _, episode in episodes]
    inventory_passed = bool(paths) and (args.expected_episodes is None or
                                      (len(paths) == args.expected_episodes and
                                       sorted(episode for episode in found if episode is not None)
                                       == list(range(1, args.expected_episodes + 1))))
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "inventory": {"files": len(paths), "episodes": found,
                            "expected_episodes": args.expected_episodes, "passed": inventory_passed},
              "tolerance_ms": args.tolerance_ms, "samples_per_episode": args.samples,
              "limits": ["Packet text baseline supports ASS/SSA only.",
                         "ASS comments are not muxed packets and have no packet-level count.",
                         "Literal search is sampled; every parsed cue is checked for text and timing.",
                         "No OCR, paid provider, external upload, or semantic search is exercised."],
              "episodes": []}
    with tempfile.TemporaryDirectory(prefix="scenerecall-subtitle-validation-") as temporary:
        root = Path(temporary)
        library = Library(root / "library")
        engine = SearchEngine(root / "index", NoProviders())
        for number, (path, episode) in enumerate(episodes):
            item = {"file": path.name, "episode": episode, "tracks": []}
            track_cues = {}
            try:
                metadata = media.probe(path)
                item.update({key: metadata[key] for key in ("duration_ms", "start_time_ms")})
                tracks = media.subtitle_tracks(metadata)
                for track in tracks:
                    if not track["supported"]:
                        continue
                    row = {key: track.get(key) for key in ("index", "codec", "language", "title")}
                    item["tracks"].append(row)
                    if track["codec"] not in {"ass", "ssa"}:
                        row.update(status="blocked", reason="unsupported_packet_reference_codec", failure_count=1)
                        continue
                    try:
                        reference, counts = packet_reference(path, track["index"], metadata)
                        extracted = media.extract_subtitle_track(
                            path, track["index"], root / f"episode-{number}-stream-{track['index']}.ass", metadata)
                        cues = subtitles.parse_container_subtitles(extracted, metadata["duration_ms"])
                        track_cues[track["index"]] = cues
                        row.update(counts)
                        row.update(compare_cues(cues, reference, args.tolerance_ms))
                        row["failure_count"] += counts["packet_reference_errors"]
                        row["status"] = "passed" if not row["failure_count"] and cues else "failed"
                    except Exception as exc:
                        row.update(status="failed", error_type=type(exc).__name__, failure_count=1)
                if args.skip_library:
                    item["import"] = {"status": "not_run", "reason": "explicit_skip_library"}
                else:
                    try:
                        item["import"] = await verify_import(path, episode, library, engine, track_cues, args.samples)
                    except Exception as exc:
                        item["import"] = {"status": "failed", "error_type": type(exc).__name__, "failure_count": 1}
            except Exception as exc:
                item.update(status="failed", error_type=type(exc).__name__, failure_count=1)
            item["status"] = ("passed" if item["tracks"] and
                              all(row["status"] == "passed" for row in item["tracks"]) and
                              item.get("import", {}).get("status") == "passed" else "incomplete")
            report["episodes"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    tracks = [track for item in report["episodes"] for track in item["tracks"]]
    report["summary"] = {
        "status": "passed" if inventory_passed and all(item["status"] == "passed" for item in report["episodes"])
        else "incomplete", "episodes_passed": sum(item["status"] == "passed" for item in report["episodes"]),
        "tracks_checked": len(tracks), "tracks_passed": sum(track["status"] == "passed" for track in tracks),
        "cues_checked": sum(track.get("parsed_cues", 0) for track in tracks),
        "max_timestamp_delta_ms": max((track.get("max_timestamp_delta_ms", 0) for track in tracks), default=0),
        "searches_passed": sum(item.get("import", {}).get("searches_passed", 0) for item in report["episodes"]),
        "searches_failed": sum(item.get("import", {}).get("searches_failed", 0) for item in report["episodes"]),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--media-dir", required=True, type=Path)
    parser.add_argument("--glob", default="*.mkv")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--episode-pattern", default=r" - (\d{1,3})(?:v\d+)?(?:[ ._\[]|$)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tolerance-ms", type=int, default=10)
    parser.add_argument("--skip-library", action="store_true", help="Debug extraction only; final status stays incomplete")
    args = parser.parse_args()
    if args.samples < 3 or args.tolerance_ms < 0 or (args.expected_episodes is not None and args.expected_episodes < 1):
        parser.error("Require samples >= 3, tolerance >= 0 and expected episodes >= 1")
    report = asyncio.run(verify(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if report["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
