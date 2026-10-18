#!/usr/bin/env python3
"""Repair only <=32-sample endpoint rounding in candidate 2s/4s PCM WAV files.

Does not shift audio timestamps. Validates every input before modifying files,
then atomically rewrites affected WAVs and refreshes specified manifest hashes.
Run while candidate extraction and labeling are stopped.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import wave

MAXIMUM_DIFFERENCE = 32


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="*")
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifests = list(dict.fromkeys(path.resolve() for path in args.manifests + args.manifest))
    if not manifests:
        parser.error("Provide at least one candidate manifest")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report or manifests[0].parent / f"audio-repair-{stamp}.json"
    report = {"schema_version": "lrasd-candidate-audio-repair-v1", "started_utc": stamp,
              "status": "validating", "expected_samples": None,
              "expected_samples_rule": "manifest frames (50 or 100) * 640; exact 25fps",
              "maximum_allowed_difference_samples": MAXIMUM_DIFFERENCE,
              "method": "zero-pad or truncate tail only; preserve first sample and time origin",
              "manifests": [], "audio_files": []}
    loaded, assets = {}, {}
    try:
        for manifest in manifests:
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            loaded[manifest] = rows
            report["manifests"].append({"path": str(manifest), "sha256_before": sha256(manifest), "rows": len(rows)})
            for row in rows:
                frames = len(row["labels"])
                if frames not in (50, 100) or row.get("fps") != 25 or row["end_frame"] - row["start_frame"] != frames:
                    raise ValueError(f"Reject invalid 50/100-frame 25fps window: {row['sample_id']}")
                expected = frames * 640
                path = Path(row["audio_path"])
                path = (path if path.is_absolute() else manifest.parent / path).resolve()
                if path in assets:
                    if assets[path]["expected_samples"] != expected:
                        raise ValueError(f"Reject shared audio with conflicting window lengths: {path}")
                    continue
                with wave.open(str(path), "rb") as audio:
                    actual = audio.getnframes()
                    if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
                        raise ValueError(f"Reject non-mono-16k-PCM16 audio: {path}")
                    if abs(actual - expected) > MAXIMUM_DIFFERENCE:
                        raise ValueError(f"Reject {path}: {actual} samples differ by more than {MAXIMUM_DIFFERENCE}")
                    pcm = audio.readframes(actual)
                if len(pcm) != actual * 2:
                    raise ValueError(f"Reject truncated PCM payload: {path}")
                assets[path] = {"path": str(path), "samples_before": actual,
                                "expected_samples": expected, "frames": frames,
                                "delta_samples": expected - actual,
                                "sha256_before": sha256(path), "pcm": pcm}
        report["status"] = "repairing"
        expected_counts = sorted({asset["expected_samples"] for asset in assets.values()})
        report["expected_samples"] = expected_counts[0] if len(expected_counts) == 1 else None
        report["expected_sample_counts"] = expected_counts
        for path, asset in assets.items():
            actual = asset["samples_before"]
            expected = asset["expected_samples"]
            if actual != expected:
                pcm = asset["pcm"][:expected * 2]
                pcm += b"\x00" * (expected * 2 - len(pcm))
                temporary = path.with_name(path.name + ".repair.tmp.wav")
                with wave.open(str(temporary), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(pcm)
                temporary.replace(path)
            asset["sha256_after"] = sha256(path)
            asset["samples_after"] = expected
            asset["changed"] = actual != expected
            report["audio_files"].append({key: value for key, value in asset.items() if key != "pcm"})
        for manifest, rows in loaded.items():
            for row in rows:
                path = Path(row["audio_path"])
                path = (path if path.is_absolute() else manifest.parent / path).resolve()
                row.setdefault("asset_sha256", {})["audio"] = assets[path]["sha256_after"]
                row["audio_sample_count"] = assets[path]["expected_samples"]
                if assets[path]["changed"]:
                    row["audio_endpoint_repair"] = {
                        "delta_samples": assets[path]["delta_samples"], "report": str(report_path.resolve()),
                        "method": "tail_zero_pad" if assets[path]["delta_samples"] > 0 else "tail_truncate"}
            temporary = manifest.with_name(manifest.name + ".audio-repair.tmp")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
            temporary.replace(manifest)
            next(item for item in report["manifests"] if item["path"] == str(manifest))["sha256_after"] = sha256(manifest)
        report["status"] = "complete"
        report["changed_audio_files"] = sum(item["changed"] for item in report["audio_files"])
        report["unique_audio_files"] = len(assets)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        write_json(report_path, report)
        raise
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "changed_audio_files": report["changed_audio_files"],
                      "unique_audio_files": len(assets), "report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
