#!/usr/bin/env python3
"""Verify one local benchmark trial and optionally import it through the scan queue.

Dry-run is the default and performs no library construction or writes. Stop the
application worker before --apply; this tool executes only its keyframe job and
never starts the queue loop or model tasks. Existing cancelled jobs under a
different decoder profile remain unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from scripts.benchmark_keyframes import file_hash, read_json, verify_images  # noqa: E402
from scenerecall.file_io import atomic_replace  # noqa: E402
from scenerecall.keyframes import ALGORITHM_PARAMETERS  # noqa: E402
from scenerecall.library import Library, confined_path, digest, safe_id, source_snapshot  # noqa: E402
from scenerecall.workflows import (  # noqa: E402
    KEYFRAME_ALGORITHM, KEYFRAME_DECODER_PROFILES, WorkflowQueueMixin,
)


class ReadOnlySelectionLibrary(Library):
    """Use the production metadata validator without initializing a library."""

    def __init__(self, root, manifest):
        self.root = root
        self.manifest = manifest

    def get_asset(self, asset_id, **_kwargs):
        if asset_id != self.manifest["id"]:
            raise ValueError("Candidate asset differs from the requested asset")
        return self.manifest


class LocalOnlyServices:
    def __getattr__(self, _name):
        raise RuntimeError("Benchmark promotion cannot access models or search services")


def validate_candidate(library_root, asset_id, trial_dir, decoder_profile):
    """Read and verify every input before constructing the writable library."""
    root = Path(library_root).expanduser().resolve(strict=True)
    trial = Path(trial_dir).expanduser().resolve(strict=True)
    asset_id = safe_id(asset_id)
    if root == trial or root.is_relative_to(trial) or trial.is_relative_to(root):
        raise ValueError("The benchmark trial must be independent of the production library")
    inspector = WorkflowQueueMixin()
    profile = inspector._keyframe_decoder_profile(decoder_profile)
    metrics, original = read_json(trial / "metrics.json"), read_json(trial / "plan.json")
    expected_variant = {"ffmpeg-v1": {"A", "B"}, "pyav-libavfilter-v1": {"C"}}[profile]
    if (metrics.get("benchmark") != "keyframe-scan" or metrics.get("status") != "completed"
            or metrics.get("variant") not in expected_variant or metrics.get("remote_model_calls") != 0
            or metrics.get("source_metadata_unchanged") is not True):
        raise ValueError("Only completed, unchanged local trials of the requested decoder can be imported")
    if original.get("status") != "completed" or original.get("algorithm") != KEYFRAME_ALGORITHM:
        raise ValueError("The candidate is not a completed content-keyframe plan")
    if original.get("algorithm_parameters") != ALGORITHM_PARAMETERS:
        raise ValueError("The candidate selection parameters differ from the current implementation")
    directory = confined_path(root, f"assets/{asset_id}")
    frames_directory = confined_path(directory, "frames")
    if frames_directory != directory / "frames":
        raise ValueError("Benchmark promotion requires the asset's canonical evidence directory")
    manifest_path = confined_path(directory, "manifest.json")
    locations_path = root / "private" / "media-locations.json"
    manifest = read_json(manifest_path)
    if manifest.get("id") != asset_id or not manifest.get("fingerprint"):
        raise ValueError("The requested source manifest is invalid")
    inputs = metrics.get("input", {})
    if (inputs.get("asset_id") != asset_id or inputs.get("fingerprint") != manifest["fingerprint"]
            or metrics.get("input_digest") != digest(inputs)):
        raise ValueError("The trial is not bound to this source asset and fingerprint")
    if any(original.get(key) != inputs.get(key) for key in ("asset_id", "fingerprint", "start_ms", "end_ms")):
        raise ValueError("The trial report and selected plan describe different inputs")
    location = read_json(locations_path).get(asset_id)
    trial_location = read_json(trial / "library" / "private" / "media-locations.json").get(asset_id)
    if not isinstance(location, dict) or not location.get("path") or location != trial_location:
        raise ValueError("The source location or file snapshot changed after the trial")
    source = Path(location["path"]).resolve(strict=True)
    if source_snapshot(source) != location or inputs.get("source_file_bytes") != location.get("size"):
        raise ValueError("The source file snapshot no longer matches the trial")
    catalog_path = root / "organization" / "catalog.json"
    if catalog_path.is_file():
        membership = read_json(catalog_path).get("assets", {}).get(asset_id, {})
        if membership.get("trash_id"):
            raise ValueError("Cannot import a plan into a deleted asset")
    inspector.library = ReadOnlySelectionLibrary(root, manifest)
    plan_id = inspector._keyframe_plan_id(manifest, original["start_ms"], original["end_ms"], profile)
    data = inspector._keyframe_selection_data(copy.deepcopy(original))
    data["decoder_profile"] = profile
    validated_plan = {**original, **data, "id": plan_id, "plan_id": plan_id}
    inspector.library.validate_keyframe_selection(asset_id, validated_plan, plan_id)
    for frame in data["frames"]:
        expected = "kf_" + digest([asset_id, manifest["fingerprint"], frame["pts"], frame["time_base"]])
        if frame["id"] != expected:
            raise ValueError("A candidate frame does not have the canonical source timestamp identity")
    count = verify_images(trial, original, metrics)
    if (metrics.get("frame_count") != count or metrics.get("shot_count") != len(data["shots"])
            or metrics.get("batch_count") != len(data["batches"])):
        raise ValueError("The trial's recorded selection counts are inconsistent")
    images = {image["id"]: image for image in metrics["images"]}
    registry_path = confined_path(frames_directory, "registry.json")
    registry = read_json(registry_path) if registry_path.is_file() else {}
    copies = []
    for frame in data["frames"]:
        image = images[frame["id"]]
        input_path = (trial / image["path"]).resolve(strict=True)
        if input_path.stat().st_size != image.get("bytes"):
            raise ValueError("The trial's recorded preview size is inconsistent")
        output = confined_path(frames_directory, f"{frame['id']}.jpg")
        if output != frames_directory / f"{frame['id']}.jpg":
            raise ValueError("An evidence image redirects away from its canonical destination")
        expected_entry = {"path": output.relative_to(directory).as_posix(), "at_ms": frame["at_ms"]}
        if frame["id"] in registry and registry[frame["id"]] != expected_entry:
            raise ValueError("An existing evidence registration conflicts with the candidate")
        if output.exists() and (not output.is_file() or file_hash(output) != image["sha256"]):
            raise ValueError("An existing evidence image differs; no image will be overwritten")
        copies.append((frame, input_path, output, image["sha256"]))
    destination_plan = confined_path(directory, f"keyframe-plans/{plan_id}.json")
    confined_path(directory, f"keyframe-work/{plan_id}/selection.json")
    if destination_plan.is_file():
        existing = read_json(destination_plan)
        if existing.get("status") == "completed":
            existing_data = inspector._keyframe_selection_data(existing)
            existing_data.setdefault("decoder_profile", "ffmpeg-v1")
            if existing_data != data:
                raise ValueError("An existing completed plan differs from this candidate")
    summary = {"status": "verified", "asset_id": asset_id, "plan_id": plan_id,
               "decoder_profile": profile, "benchmark_variant": metrics["variant"],
               "start_ms": original["start_ms"], "end_ms": original["end_ms"],
               "frame_count": count, "verified_image_count": count,
               "existing_images": sum(output.is_file() for _, _, output, _ in copies),
               "input_digest": metrics["input_digest"], "remote_model_calls": 0}
    return SimpleNamespace(root=root, trial=trial, asset_id=asset_id, manifest=manifest, data=data,
                           plan_id=plan_id, profile=profile, copies=copies, source=source, location=location,
                           directory=directory, frames_directory=frames_directory,
                           summary=summary, manifest_path=manifest_path, locations_path=locations_path,
                           source_metadata_hashes=(file_hash(manifest_path), file_hash(locations_path)))


def require_stopped_worker(root, port):
    from scenerecall.service import find_service

    if find_service(root, port) is not None:
        raise ValueError("Stop the application worker before applying a benchmark import")
    database = root / "runtime" / "jobs.sqlite"
    if database.is_file():
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            jobs = [json.loads(row[0]) for row in connection.execute("SELECT data FROM jobs")]
        if any(job.get("status") in {"running", "pausing", "cancelling"} for job in jobs):
            raise ValueError("Active job states remain; finish the normal worker shutdown before importing")


def copy_verified(source, destination, expected_hash):
    """Recheck bytes while copying and atomically publish only matching images."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".benchmark-", suffix=".jpg", dir=destination.parent,
                                     delete=False) as output:
        temporary = Path(output.name)
        checksum = hashlib.sha256()
        try:
            with source.open("rb") as incoming:
                while block := incoming.read(1024 * 1024):
                    checksum.update(block)
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            output.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if checksum.hexdigest() != expected_hash:
            raise ValueError("Benchmark image changed during import")
        atomic_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


async def apply_candidate(candidate, *, port=8765):
    from scenerecall.jobs import JobQueue

    require_stopped_worker(candidate.root, port)
    if (source_snapshot(candidate.source) != candidate.location
            or (file_hash(candidate.manifest_path), file_hash(candidate.locations_path))
            != candidate.source_metadata_hashes):
        raise ValueError("Source metadata changed after preflight; rerun verification")
    if (confined_path(candidate.root, f"assets/{candidate.asset_id}") != candidate.directory
            or confined_path(candidate.directory, "frames") != candidate.frames_directory):
        raise ValueError("The evidence destination changed after preflight; rerun verification")
    library = Library(candidate.root)
    queue = JobQueue(library, LocalOnlyServices(), LocalOnlyServices())
    plan = queue.ensure_plan(candidate.asset_id, candidate.summary["start_ms"], candidate.summary["end_ms"],
                             decoder_profile=candidate.profile)
    if plan["id"] != candidate.plan_id:
        raise ValueError("The formal workflow selected an unexpected plan identity")
    if plan["status"] == "completed":
        return {**candidate.summary, "status": "already_available", "job_id": plan.get("job_id")}
    job_id = plan["job_id"]
    if queue.get(job_id)["type"] != "keyframe_scan":
        raise ValueError("Promotion may execute only a local keyframe job")
    try:
        for offset in range(0, len(candidate.copies), 128):
            library.require_source(candidate.asset_id)
            current = candidate.copies[offset:offset + 128]
            frames = []
            for frame, source, output, checksum in current:
                if confined_path(candidate.frames_directory, f"{frame['id']}.jpg") != output:
                    raise ValueError("The evidence destination changed after preflight")
                if not output.is_file():
                    copy_verified(source, output, checksum)
                elif file_hash(output) != checksum:
                    raise ValueError("Existing evidence changed after preflight")
                frames.append({**frame, "path": str(output)})
            library.register_frames(candidate.asset_id, frames)
        queue._save_keyframe_selection(candidate.manifest, candidate.plan_id, candidate.summary["start_ms"],
                                       candidate.summary["end_ms"], candidate.data)
        await queue.execute(job_id)
        finished = queue.get(job_id)
        if finished["status"] != "completed":
            raise ValueError("Formal keyframe publication failed: " + (finished.get("error") or finished["status"]))
        published = queue.validated_plan(candidate.asset_id, candidate.summary["start_ms"],
                                         candidate.summary["end_ms"], candidate.plan_id)
        if queue._keyframe_selection_data(published) != candidate.data:
            raise ValueError("Published selection differs from the verified benchmark candidate")
        return {**candidate.summary, "status": "promoted", "job_id": job_id,
                "keyframe_metrics": finished.get("keyframe_metrics", {})}
    except BaseException:
        if queue.get(job_id)["status"] == "queued":
            queue.control(job_id, "pause")
        raise


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--library", type=Path, required=True)
    cli.add_argument("--asset-id", required=True)
    cli.add_argument("--trial", type=Path, required=True)
    cli.add_argument("--decoder-profile", choices=KEYFRAME_DECODER_PROFILES, required=True)
    cli.add_argument("--port", type=int, default=8765, help="Stopped application worker's port")
    cli.add_argument("--apply", action="store_true", help="Copy verified previews and execute only their keyframe job")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1024 <= args.port <= 65535:
        raise ValueError("Port must be between 1024 and 65535")
    candidate = validate_candidate(args.library, args.asset_id, args.trial, args.decoder_profile)
    result = asyncio.run(apply_candidate(candidate, port=args.port)) if args.apply else candidate.summary
    print(json.dumps({**result, "dry_run": not args.apply}, ensure_ascii=False))


if __name__ == "__main__":
    main()
