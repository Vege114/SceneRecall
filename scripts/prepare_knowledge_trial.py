"""Create an isolated trial from published evidence; never re-encode a film.

Only image bytes are hardlinked. Mutable JSON and the selected SQLite evidence
are copied, so all later knowledge edits stay inside the new trial directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall.library import Library, atomic_json, safe_id  # noqa: E402
from scenerecall.providers import ProviderManager  # noqa: E402
from scenerecall.search import SearchEngine  # noqa: E402


def copy_file(source, destination):
    if Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        try:
            os.link(source, destination)
            return destination
        except OSError:
            pass
    return shutil.copy2(source, destination)


def prepare(source, asset_ids, output):
    source, output = source.resolve(), output.resolve()
    if output.exists() or source == output or source in output.parents and output.parts[:len(source.parts)] == source.parts:
        raise ValueError("输出目录须为新的隔离目录，且不能位于正式资料库中")
    output.mkdir(parents=True)
    for asset in asset_ids:
        origin = source / "assets" / safe_id(asset)
        destination = output / "assets" / asset
        destination.mkdir(parents=True)
        for name in ("manifest.json", "active-analysis.json"):
            if (origin / name).is_file():
                shutil.copy2(origin / name, destination / name)
        for name in ("frames", "retrieval", "identity", "analyses", "subtitles", "keyframe-plans"):
            if (origin / name).is_dir():
                shutil.copytree(origin / name, destination / name, copy_function=copy_file)
    # Character registries and references retain their stable IDs. Existing work
    # entity scopes are copied too, but membership is narrowed to selected assets.
    if (source / "works").is_dir():
        shutil.copytree(source / "works", output / "works", copy_function=copy_file,
                        ignore=shutil.ignore_patterns("knowledge-indexes"))
    membership = output / "works" / "entity-membership.json"
    if membership.is_file():
        saved = json.loads(membership.read_text(encoding="utf-8"))
        saved["assets"] = {key: value for key, value in saved["assets"].items() if key in asset_ids}
        atomic_json(membership, saved)
    library = Library(output)
    search = SearchEngine(output / "indexes", ProviderManager(output / "private"))
    original = sqlite3.connect((source / "indexes" / "search.sqlite").as_uri() + "?mode=ro", uri=True)
    original.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in asset_ids)
    rows = list(original.execute(f"SELECT data FROM records WHERE asset_id IN ({placeholders})", asset_ids))
    search.upsert([json.loads(row["data"]) for row in rows])
    with search._db() as target:
        target.execute("BEGIN IMMEDIATE")
        for row in original.execute("SELECT * FROM visual_namespaces"):
            target.execute("INSERT OR REPLACE INTO visual_namespaces VALUES (?,?,?,?)", tuple(row))
        for table in ("visual_records", "visual_links"):
            query = (f"SELECT v.* FROM {table} v JOIN records r ON r.id=v.record_id "
                     f"WHERE r.asset_id IN ({placeholders})")
            for row in original.execute(query, asset_ids):
                target.execute(f"INSERT OR REPLACE INTO {table} VALUES ({','.join('?' for _ in row)})", tuple(row))
        for row in original.execute(f"""SELECT DISTINCT c.* FROM visual_cache c
                JOIN visual_links l ON l.signature=c.signature AND l.cache_key=c.cache_key
                JOIN records r ON r.id=l.record_id WHERE r.asset_id IN ({placeholders})""", asset_ids):
            target.execute("INSERT OR REPLACE INTO visual_cache VALUES (?,?,?,?)", tuple(row))
    original.close()
    for name in ("retrieval-encoder.json", "media-locations.json"):
        if (source / "private" / name).is_file():
            shutil.copy2(source / "private" / name, output / "private" / name)
    result = {"source_library": str(source), "library": str(output), "asset_ids": asset_ids,
              "indexed_records": len(rows), "visual": search.visual_index.status(),
              "frames": "hardlinked when possible; read-only inputs", "original_modified": False}
    atomic_json(output / "trial.json", result)
    # Validate canonical source evidence before a knowledge trial can start.
    for asset in asset_ids:
        library.records(asset)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-library", type=Path, required=True)
    parser.add_argument("--asset-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_library, args.asset_id, args.output_dir), ensure_ascii=False))
