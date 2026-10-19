"""Inspect authorized local media and subtitle speaker fields without uploading it."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8-sig"))
    video = source["video_path"]
    destination = args.source.parent
    result = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", video],
                            capture_output=True, check=True, text=True, timeout=60)
    metadata = json.loads(result.stdout)
    (destination / "probe.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"duration_seconds": float(metadata["format"]["duration"]), "streams": [], "subtitles": []}
    for stream in metadata["streams"]:
        summary["streams"].append({key: stream.get(key) for key in
                                   ("index", "codec_type", "codec_name", "width", "height", "r_frame_rate",
                                    "sample_rate", "channels", "tags", "disposition")})
        if stream.get("codec_type") != "subtitle" or stream.get("codec_name") not in ("ass", "ssa", "subrip", "webvtt"):
            continue
        path = destination / ("subtitle-%s.ass" % stream["index"])
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", video, "-map", "0:%s" % stream["index"],
                        "-c:s", "ass", str(path)], check=True, timeout=180)
        names, styles = Counter(), Counter()
        examples = []
        columns = []
        events = False
        count = 0
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line == "[Events]":
                events = True
            elif events and line.startswith("Format:"):
                columns = [part.strip() for part in line[7:].split(",")]
            elif events and line.startswith("Dialogue:") and columns:
                fields = dict(zip(columns, line[9:].lstrip().split(",", len(columns) - 1)))
                count += 1
                names[fields.get("Name", "")] += 1
                styles[fields.get("Style", "")] += 1
                if len(examples) < 12:
                    examples.append({key: fields.get(key) for key in ("Start", "End", "Style", "Name", "Text")})
        summary["subtitles"].append({"path": str(path), "events": count, "name_fields": dict(names),
                                     "styles": dict(styles), "examples": examples})
    (destination / "source-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
