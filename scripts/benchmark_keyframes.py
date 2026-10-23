#!/usr/bin/env python3
"""Run one local keyframe A/B/C trial in a fresh process and isolated library.

A imports an entire frozen backend; B uses the current FFmpeg implementation;
C uses the current PyAV implementation. No models or source-library constructors
are invoked. Run trials sequentially with alternating order and fresh output dirs.
OS file caches are uncontrolled; an empty application cache is not a cold disk.

Run: benchmark_keyframes.py --library ... --asset-id ... --start-ms 0
     --end-ms 60000 --variant A --baseline-root ... --output-dir ...
Compare: benchmark_keyframes.py compare --baseline RUN_A --candidate RUN_B
         --output comparison.json [--verify-images] [--contact-sheet differences.jpg]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from bisect import bisect_left
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

PLAN_FIELDS = ("algorithm", "algorithm_parameters", "source_start_time_seconds", "id", "plan_id", "asset_id", "fingerprint",
               "start_ms", "end_ms", "shots", "frames", "batches")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def dependency_versions():
    versions = {}
    for name in ("numpy", "scenedetect", "av", "psutil", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    executable = shutil.which("ffmpeg")
    versions["ffmpeg"] = None
    if executable:
        try:
            process = subprocess.run([executable, "-version"], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=10, check=False)
            if process.returncode == 0:
                versions["ffmpeg"] = process.stdout.splitlines()[0]
        except (OSError, subprocess.TimeoutExpired, IndexError):
            pass
    return versions


def load_runtime(args):
    """Select the complete package before importing any scenerecall modules."""
    if any(name == "scenerecall" or name.startswith("scenerecall.") for name in sys.modules):
        raise RuntimeError("Each benchmark variant requires a fresh Python process")
    if args.variant == "A":
        if args.baseline_root is None:
            raise ValueError("Variant A requires --baseline-root with a frozen complete backend")
        root = args.baseline_root.resolve(strict=True)
        if (root / "backend" / "scenerecall").is_dir():
            root /= "backend"
    else:
        root = Path(__file__).resolve().parents[1] / "backend"
    if not all((root / "scenerecall" / name).is_file() for name in ("keyframes.py", "library.py", "media.py")):
        raise ValueError("Runtime root must contain the complete scenerecall package")
    sys.path.insert(0, str(root))
    library = importlib.import_module("scenerecall.library")
    keyframes = importlib.import_module("scenerecall.keyframes")
    media = importlib.import_module("scenerecall.media")
    for module in (library, keyframes, media):
        if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError("Benchmark imported a module outside its selected backend")
    return SimpleNamespace(library=library, keyframes=keyframes, media=media,
                           cv2=importlib.import_module("cv2"), psutil=importlib.import_module("psutil"), root=root)


def validate_inputs(args, runtime):
    source_library = args.library.resolve(strict=True)
    output = args.output_dir.resolve()
    if (output == source_library or output.is_relative_to(source_library)
            or source_library.is_relative_to(output)):
        raise ValueError("Output directory must be independent of the source library")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Each trial requires a new or empty output directory")
    asset_id = runtime.library.safe_id(args.asset_id)
    manifest_path = source_library / "assets" / asset_id / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("id") != asset_id or not manifest.get("fingerprint"):
        raise ValueError("Invalid source manifest identity")
    if not 0 <= args.start_ms < args.end_ms <= manifest["duration_ms"]:
        raise ValueError("Benchmark range is outside the source duration")
    locations_path = source_library / "private" / "media-locations.json"
    location = read_json(locations_path)[asset_id]
    if not isinstance(location, dict) or "path" not in location:
        raise ValueError("Source must have a current file snapshot")
    source = Path(location["path"]).resolve(strict=True)
    if output == source or source.is_relative_to(output):
        raise ValueError("Output directory cannot contain the original media")
    if runtime.library.source_snapshot(source) != location:
        raise ValueError("Source file snapshot has changed")
    if args.variant == "A" and args.opencv_threads is not None:
        raise ValueError("Variant A must retain its default OpenCV thread count")
    if args.variant != "A" and not hasattr(runtime.library.Library, "register_frames"):
        raise ValueError("Candidate runtime does not yet provide register_frames")
    return SimpleNamespace(source_library=source_library, output=output, manifest=manifest,
                           manifest_path=manifest_path, locations_path=locations_path,
                           location=location, source=source,
                           source_metadata_hashes={"manifest": file_hash(manifest_path),
                                                   "locations": file_hash(locations_path)})


class ResourceSampler:
    """Sample process-tree resources; exited children's last interval may be missed."""
    def __init__(self, psutil):
        self.psutil = psutil
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.child_cpu = {}
        self.peak_rss = 0
        self.samples = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        total_rss = 0
        try:
            processes = [self.process, *self.process.children(recursive=True)]
        except self.psutil.Error:
            processes = [self.process]
        for process in processes:
            try:
                total_rss += process.memory_info().rss
                if process.pid != self.process.pid:
                    value = process.cpu_times()
                    identity = (process.pid, process.create_time())
                    self.child_cpu[identity] = max(self.child_cpu.get(identity, 0), value.user + value.system)
            except self.psutil.Error:
                continue
        self.peak_rss = max(self.peak_rss, total_rss)
        self.samples += 1

    def _loop(self):
        while not self.stop_event.wait(.1):
            self._sample()

    def start(self):
        self.started_cpu = time.process_time()
        self._sample()
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self._sample()
        return {"parent_cpu_seconds": round(time.process_time() - self.started_cpu, 6),
                "sampled_child_cpu_seconds": round(sum(self.child_cpu.values()), 6),
                "sampled_peak_tree_rss_bytes": self.peak_rss, "resource_samples": self.samples,
                "resource_sampling_interval_ms": 100,
                "resource_limitations": "Child CPU may omit the final interval of exited processes; "
                    "RSS is the maximum sampled sum, not a kernel peak or unique physical memory."}


class DecodeCounter:
    def __init__(self, module):
        self.module = module
        self.originals = {}
        self.frames = 0
        self.wait_seconds = 0

    def __enter__(self):
        # Both decoders yield actual source frames, including any seek preroll.
        for name in ("_decode", "_decode_pyav"):
            original = getattr(self.module, name, None)
            if original is None:
                continue
            self.originals[name] = original

            def counted(*args, _original=original, **kwargs):
                iterator = _original(*args, **kwargs)
                try:
                    while True:
                        started = time.perf_counter()
                        try:
                            item = next(iterator)
                        except StopIteration:
                            self.wait_seconds += time.perf_counter() - started
                            return
                        self.wait_seconds += time.perf_counter() - started
                        self.frames += 1
                        yield item
                finally:
                    iterator.close()

            setattr(self.module, name, counted)
        return self

    def __exit__(self, *_args):
        for name, original in self.originals.items():
            setattr(self.module, name, original)


def execute(args, runtime=None):
    runtime = runtime or load_runtime(args)
    inputs = validate_inputs(args, runtime)
    library_module = runtime.library
    if args.variant != "A":
        runtime.cv2.setNumThreads(args.opencv_threads or 1)
    inputs.output.mkdir(parents=True, exist_ok=True)
    timings = {}
    report = {"schema_version": "1.0", "benchmark": "keyframe-scan", "status": "running",
              "variant": args.variant,
              "baseline_expected_revision": getattr(args, "baseline_revision", None) if args.variant == "A" else None,
              "baseline_revision_status": "operator_provided_not_verified" if args.variant == "A"
                  and getattr(args, "baseline_revision", None) else "not_provided",
              "started_at": datetime.now(timezone.utc).isoformat(), "opencv_threads": runtime.cv2.getNumThreads(),
              "opencv_version": runtime.cv2.__version__, "python_version": platform.python_version(),
              "platform": platform.platform(), "remote_model_calls": 0,
              "input": {"asset_id": args.asset_id, "fingerprint": inputs.manifest["fingerprint"],
                        "start_ms": args.start_ms, "end_ms": args.end_ms,
                        "source_file_bytes": inputs.location["size"]},
              "runtime_file_hashes": {name: file_hash(runtime.root / "scenerecall" / name)
                                      for name in ("keyframes.py", "media.py", "library.py")},
              "dependency_versions": dependency_versions(),
              "cache_policy": "New process and empty isolated application outputs; OS file cache uncontrolled.",
              "timing_definition": "processing_wall_ms includes probe, scan, preview extraction, registry writes, "
                  "plan validation/publication; excludes imports, isolated-library setup, verification hashes "
                  "and report serialization. trial_wall_ms additionally includes setup and verification hashes. "
                  "No job-queue progress/checkpoint persistence or remote models are included.",
              "resource_timing_definition": "CPU and RSS sampling cover the trial, including isolated-library "
                  "setup and verification hashes; they do not share the narrower processing_wall_ms boundary.",
              "decode_measurement_definition": "decode_iterator_wait is elapsed time spent requesting generator "
                  "items, not decoder CPU time. FFmpeg decoding/filtering overlaps Python analysis; PyAV is "
                  "synchronous, so these values cannot attribute decoder cost across variants. "
                  "yielded_frames_including_preroll counts only generator outputs, excluding frames discarded "
                  "inside the decoder while seeking from an earlier GOP.",
              "stage_ms": timings, "quality_acceptance": "not_compared"}
    report["input_digest"] = digest(report["input"])
    sampler = ResourceSampler(runtime.psutil)
    sampler.start()
    trial_started = time.perf_counter()
    try:
        started = time.perf_counter()
        destination = inputs.output / "library"
        manifest = copy.deepcopy(inputs.manifest)
        manifest.update(project_id=None, subtitle_mode="external")
        library_module.atomic_json(destination / "assets" / args.asset_id / "manifest.json", manifest)
        library_module.atomic_json(destination / "private" / "media-locations.json", {args.asset_id: inputs.location})
        library = library_module.Library(destination)
        library.require_source(args.asset_id)
        timings["setup"] = (time.perf_counter() - started) * 1000
        processing_started = time.perf_counter()
        metadata = None
        if args.variant != "A":
            started = time.perf_counter()
            metadata = runtime.media.probe(inputs.source)
            timings["probe"] = (time.perf_counter() - started) * 1000
        else:
            timings["probe"] = None  # Original probes are included in scan/extract.
        started = time.perf_counter()
        options = {} if args.variant == "A" else {"metadata": metadata,
                    "decoder": "pyav" if args.variant == "C" else "ffmpeg"}
        with DecodeCounter(runtime.keyframes) as decoded:
            data = runtime.keyframes.scan_keyframes(inputs.source, args.start_ms, args.end_ms, **options)
        timings["scan"] = (time.perf_counter() - started) * 1000
        timings["decode_iterator_wait"] = decoded.wait_seconds * 1000
        report["yielded_frames_including_preroll"] = decoded.frames
        rename = {}
        for frame in data["frames"]:
            frame_id = "kf_" + library_module.digest([args.asset_id, manifest["fingerprint"],
                                                       frame["pts"], frame["time_base"]])
            rename[frame["id"]] = frame_id
            frame["id"] = frame_id
        for batch in data["batches"]:
            batch["frame_ids"] = [rename[value] for value in batch["frame_ids"]]
        outputs = [library.asset_dir(args.asset_id) / "frames" / f"{frame['id']}.jpg" for frame in data["frames"]]
        timings.update(extract=0.0, register=0.0)
        report["extraction_batches"] = 0
        report["registry_write_calls"] = 0
        for offset in range(0, len(outputs), 128):
            library.require_source(args.asset_id)
            started = time.perf_counter()
            options = {} if args.variant == "A" else {"metadata": metadata}
            extracted = runtime.keyframes.extract_keyframes(
                inputs.source, data["frames"][offset:offset + 128], outputs[offset:offset + 128], **options)
            timings["extract"] += (time.perf_counter() - started) * 1000
            report["extraction_batches"] += 1
            started = time.perf_counter()
            if args.variant == "A":
                for frame in extracted:
                    library.register_frame(args.asset_id, frame)
                report["registry_write_calls"] += len(extracted)
            else:
                library.register_frames(args.asset_id, extracted)
                report["registry_write_calls"] += 1
            timings["register"] += (time.perf_counter() - started) * 1000
        plan_id = "plan_" + library_module.digest([args.asset_id, manifest["fingerprint"], args.start_ms,
                                                  args.end_ms, data["algorithm"]])
        plan = {**data, "id": plan_id, "plan_id": plan_id, "asset_id": args.asset_id,
                "fingerprint": manifest["fingerprint"], "start_ms": args.start_ms, "end_ms": args.end_ms,
                "status": "completed", "created_at": datetime.now(timezone.utc).isoformat()}
        started = time.perf_counter()
        library.require_source(args.asset_id)
        library.validate_keyframe_plan(args.asset_id, plan, expected_id=plan_id)
        library_module.atomic_json(library.asset_dir(args.asset_id) / "keyframe-plans" / f"{plan_id}.json", plan)
        library_module.atomic_json(inputs.output / "plan.json", plan)
        timings["validate_and_publish"] = (time.perf_counter() - started) * 1000
        report["processing_wall_ms"] = (time.perf_counter() - processing_started) * 1000
        started = time.perf_counter()
        report["images"] = [{"id": frame["id"], "at_ms": frame["at_ms"],
                             "source_pts": frame["source_pts"], "source_time_base": frame["source_time_base"],
                             "path": path.relative_to(inputs.output).as_posix(), "sha256": file_hash(path),
                             "bytes": path.stat().st_size}
                            for frame, path in zip(plan["frames"], outputs, strict=True)]
        if (file_hash(inputs.manifest_path) != inputs.source_metadata_hashes["manifest"]
                or file_hash(inputs.locations_path) != inputs.source_metadata_hashes["locations"]
                or library_module.source_snapshot(inputs.source) != inputs.location):
            raise ValueError("Source metadata or media changed during the trial")
        timings["verification_hashes"] = (time.perf_counter() - started) * 1000
        report.update(status="completed", frame_count=len(plan["frames"]), shot_count=len(plan["shots"]),
                      batch_count=len(plan["batches"]), source_metadata_unchanged=True,
                      preview_bytes=sum(image["bytes"] for image in report["images"]),
                      isolated_output_bytes_before_metrics=sum(path.stat().st_size for path in inputs.output.rglob("*")
                                                              if path.is_file()))
        return report
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["trial_wall_ms"] = (time.perf_counter() - trial_started) * 1000
        report.update(sampler.finish())
        report["stage_ms"] = {key: round(value, 3) if value is not None else None for key, value in timings.items()}
        for key in ("trial_wall_ms", "processing_wall_ms"):
            if key in report:
                report[key] = round(report[key], 3)
        library_module.atomic_json(inputs.output / "metrics.json", report)


def canonical_plan(plan):
    """Ignore execution identity only; retain every substantive selected-frame field."""
    return {field: plan.get(field) for field in PLAN_FIELDS}


def exact_time(frame):
    return frame["source_pts"] * Fraction(*frame["source_time_base"])


def nearest_distances(left, right):
    values = sorted(exact_time(frame) for frame in right)
    result = []
    if not values:
        return result
    for frame in left:
        stamp = exact_time(frame)
        position = bisect_left(values, stamp)
        result.append(float(min(abs(stamp - values[i]) for i in (position - 1, position)
                                if 0 <= i < len(values))) * 1000)
    return result


def distance_summary(values, tolerance_ms):
    if not values:
        return {"count": 0, "max_ms": None, "median_ms": None, "within_tolerance": 0}
    return {"count": len(values), "max_ms": max(values), "median_ms": statistics.median(values),
            "within_tolerance": sum(value <= tolerance_ms for value in values)}


def verify_images(directory, plan, metrics):
    """Rehash existing evidence before acceptance instead of trusting recorded hashes."""
    directory = Path(directory).resolve()
    expected = {frame["id"]: frame for frame in plan["frames"]}
    images = metrics["images"]
    if len(images) != len(expected) or {image["id"] for image in images} != set(expected):
        raise ValueError("Recorded previews do not cover the selected frames exactly")
    for image in images:
        frame = expected[image["id"]]
        if any(image.get(field) != frame.get(field) for field in ("at_ms", "source_pts", "source_time_base")):
            raise ValueError("Recorded preview identity differs from its selected frame")
        path = (directory / image["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Preview reference escapes its benchmark output directory")
        if not path.is_file():
            raise ValueError("Missing preview: " + image["id"])
        if file_hash(path) != image["sha256"]:
            raise ValueError("Preview hash mismatch: " + image["id"])
    return len(images)


def compare_runs(baseline, candidate, tolerance_ms=0.0, *, verify_image_files=False):
    baseline, candidate = Path(baseline), Path(candidate)
    a, b = read_json(baseline / "plan.json"), read_json(candidate / "plan.json")
    am, bm = read_json(baseline / "metrics.json"), read_json(candidate / "metrics.json")
    if am["status"] != "completed" or bm["status"] != "completed":
        raise ValueError("Only completed trials can be compared")
    if am["input_digest"] != bm["input_digest"] or am["input"] != bm["input"]:
        raise ValueError("Trials refer to different source identities or ranges")
    verified_counts = None
    if verify_image_files:
        verified_counts = {"baseline": verify_images(baseline, a, am), "candidate": verify_images(candidate, b, bm)}
    left, right = canonical_plan(a), canonical_plan(b)
    differences = [field for field in PLAN_FIELDS if left[field] != right[field]]
    ah = {image["id"]: image["sha256"] for image in am["images"]}
    bh = {image["id"]: image["sha256"] for image in bm["images"]}
    image_equal = ah == bh
    at, bt = am.get("processing_wall_ms"), bm.get("processing_wall_ms")
    return {"schema_version": "1.0", "benchmark": "keyframe-comparison", "status": "compared",
            "input": am["input"], "baseline_variant": am["variant"], "candidate_variant": bm["variant"],
            "plan_equal": not differences, "different_plan_fields": differences,
            "image_hashes_equal": image_equal, "exact_equivalence": not differences and image_equal,
            "image_verification": "existing_files_rehashed" if verify_image_files else "recorded_hashes_only",
            "verified_image_count": verified_counts,
            "baseline_frame_count": len(a["frames"]), "candidate_frame_count": len(b["frames"]),
            "baseline_shot_count": len(a["shots"]), "candidate_shot_count": len(b["shots"]),
            "common_frame_ids": len(set(ah) & set(bh)),
            "different_common_image_hashes": sum(ah[key] != bh[key] for key in set(ah) & set(bh)),
            "time_tolerance_ms": tolerance_ms,
            "baseline_to_candidate_nearest": distance_summary(nearest_distances(a["frames"], b["frames"]),
                                                                tolerance_ms),
            "candidate_to_baseline_nearest": distance_summary(nearest_distances(b["frames"], a["frames"]),
                                                                tolerance_ms),
            "tolerance_note": "Nearest-time statistics are diagnostic only, not one-to-one matching or acceptance.",
            "baseline_processing_wall_ms": at, "candidate_processing_wall_ms": bt,
            "processing_reduction_fraction": 1 - bt / at if at and bt is not None else None,
            "acceptance": "requires_repeated_speed_measurements" if not differences and image_equal
                          else "not_equivalent_requires_review"}


def contact_sheet(baseline, candidate, output, limit=24):
    """Display the first differing selected timestamps; analysis artifact, never auto-acceptance."""
    from PIL import Image, ImageDraw, ImageOps

    runs = [Path(baseline), Path(candidate)]
    reports = [read_json(run / "metrics.json") for run in runs]
    maps = [{exact_time(frame): frame for frame in report["images"]} for report in reports]
    stamps = [stamp for stamp in sorted(set(maps[0]) | set(maps[1]))
              if stamp not in maps[0] or stamp not in maps[1]
              or maps[0][stamp]["sha256"] != maps[1][stamp]["sha256"]][:limit]
    if not stamps:
        stamps = list(sorted(maps[0]))[:min(4, limit)]
    canvas = Image.new("RGB", (660, max(1, len(stamps)) * 210 + 36), "#202124")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Baseline / candidate: differing timestamps (or equal-result sample)", fill="white")
    for row, stamp in enumerate(stamps):
        for column, mapping in enumerate(maps):
            x, y = 10 + column * 330, 36 + row * 210
            frame = mapping.get(stamp)
            draw.text((x, y), f"{reports[column]['variant']}  source={float(stamp):.6f}s", fill="white")
            if frame is None:
                draw.text((x, y + 50), "No selected frame at this exact timestamp", fill="#ffb0a0")
                continue
            path = (runs[column] / frame["path"]).resolve()
            if not path.is_relative_to(runs[column].resolve()):
                raise ValueError("Image reference escapes its benchmark output directory")
            with Image.open(path) as image:
                tile = ImageOps.contain(image.convert("RGB"), (310, 180))
                canvas.paste(tile, (x, y + 20))
    output = Path(output)
    if output.exists():
        raise ValueError("Contact sheet already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {"rows": len(stamps), "row_limit": limit, "truncated": len(set(maps[0]) | set(maps[1])) > limit}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--library", type=Path, required=True)
    result.add_argument("--asset-id", required=True)
    result.add_argument("--start-ms", type=int, required=True)
    result.add_argument("--end-ms", type=int, required=True)
    result.add_argument("--variant", choices=("A", "B", "C"), required=True)
    result.add_argument("--opencv-threads", type=int, choices=(1, 2, 4))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--baseline-root", type=Path)
    result.add_argument("--baseline-revision", help="Optional operator-provided frozen commit; file hashes remain authoritative")
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "compare":
        compare = argparse.ArgumentParser(description="Compare completed local keyframe trials")
        compare.add_argument("--baseline", type=Path, required=True)
        compare.add_argument("--candidate", type=Path, required=True)
        compare.add_argument("--output", type=Path, required=True)
        compare.add_argument("--tolerance-ms", type=float, default=0)
        compare.add_argument("--contact-sheet", type=Path)
        compare.add_argument("--verify-images", action="store_true", help="Rehash every existing preview; reject missing or changed files")
        args = compare.parse_args(argv[1:])
        if args.tolerance_ms < 0 or args.output.exists():
            raise ValueError("Tolerance must be nonnegative and the report path must be new")
        report = compare_runs(args.baseline, args.candidate, args.tolerance_ms, verify_image_files=args.verify_images)
        if args.contact_sheet:
            report["contact_sheet"] = contact_sheet(args.baseline, args.candidate, args.contact_sheet)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
    else:
        report = execute(parser().parse_args(argv))
        print(json.dumps({key: report[key] for key in ("status", "variant", "frame_count", "shot_count",
                         "processing_wall_ms", "stage_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
