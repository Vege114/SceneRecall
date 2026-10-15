#!/usr/bin/env python3
"""Compare both real analysis strategies using an existing scan and Codex binding.

This explicitly invokes the configured Codex subscription. The source library is
read-only; each strategy runs sequentially in a separate scratch library with the
same frames, subtitles, configured model profile and empty initial character
memory. No rescan occurs.
The report contains aggregate measurements; scratch libraries contain local media
evidence and model outputs for human review. Reusing --work-dir resumes saved jobs.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import inspect
import json
import shutil
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from scenerecall.jobs import JobQueue, model_identity
from scenerecall.library import Library, atomic_json, atomic_jsonl, digest, read_json, read_jsonl
from scenerecall.models import AnalysisInput
from scenerecall.providers import ProviderManager
from scenerecall.scene_jobs import canonical, encoded_size
from scenerecall.search import SearchEngine


class MeasuredProviders(ProviderManager):
    def __init__(self, directory, measurements):
        super().__init__(directory)
        self.measurements = measurements
        self._measurement = ContextVar("benchmark_provider_measurement", default=None)

    async def _chat(self, id, capability, instruction, user_content, validator, **kwargs):
        measurement = self._measurement.get()
        if measurement is not None:
            parts = user_content if isinstance(user_content, list) else []
            measurement["image_count_per_attempt"] = sum(part.get("type") == "image_url" for part in parts)
            measurement["prompt_text_bytes"] = len(instruction.encode("utf-8")) + (
                sum(len(part.get("text", "").encode("utf-8")) for part in parts if part.get("type") == "text")
                if isinstance(user_content, list) else len(str(user_content).encode("utf-8")))
        return await super()._chat(id, capability, instruction, user_content, validator, **kwargs)

    def __getattribute__(self, name):
        method = super().__getattribute__(name)
        if name not in {"analyze", "prepare_subtitles", "prepare_visual", "reconcile_scene"}:
            return method

        async def measured(*args, **kwargs):
            started = time.monotonic()
            bound = inspect.signature(method).bind(*args, **kwargs)
            inputs = bound.arguments
            row = {"call_id": uuid.uuid4().hex, "method": name, "started_at": utc_now(), "status": "running",
                   "elapsed_ms": None, "request_count": None, "usage": {}, "failure": None,
                   "input_metadata_bytes": encoded_size(canonical([args, kwargs])),
                   "core_frame_count": len(inputs.get("frames", [])),
                   "context_frame_count": len(inputs.get("context_frames", [])),
                   "character_reference_count": (len((inputs.get("character_context") or {}).get("reference_frames", []))
                                                 if name == "analyze" else 0),
                   "image_count_per_attempt": None, "prompt_text_bytes": None}
            values = read_json(self.measurements, [])
            values.append(copy.deepcopy(row))
            atomic_json(self.measurements, values)
            token = self._measurement.set(row)
            result = None
            try:
                result = await method(*args, **kwargs)
                row["status"] = "completed"
                return result
            except BaseException as exc:
                result, row["failure"] = exc, type(exc).__name__
                row["status"] = "interrupted" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)) else "failed"
                raise
            finally:
                self._measurement.reset(token)
                row.update(elapsed_ms=round((time.monotonic() - started) * 1000), finished_at=utc_now(),
                           request_count=getattr(result, "request_count", None), usage=getattr(result, "usage", {}))
                try:
                    values = read_json(self.measurements, [])
                    values = [copy.deepcopy(row) if item.get("call_id") == row["call_id"] else item for item in values]
                    atomic_json(self.measurements, values)
                except (OSError, ValueError):
                    # The persisted running entry makes incompleteness observable;
                    # never hide a paid result or replace its original exception.
                    print(json.dumps({"event": "measurement_write_failed", "call_id": row["call_id"]}),
                          file=sys.stderr, flush=True)
                print(json.dumps({"call": name, **{k: row[k] for k in
                    ("elapsed_ms", "request_count", "failure")}}), flush=True)
        return measured


class MeasuredSearch(SearchEngine):
    def __init__(self, directory, provider):
        super().__init__(directory, provider)
        self.elapsed_ms = 0
        self.depth = 0

    def __getattribute__(self, name):
        method = super().__getattribute__(name)
        if name not in {"sync_asset", "upsert", "rebuild"}:
            return method

        if inspect.iscoroutinefunction(method):
            async def measured_async(*args, **kwargs):
                started = time.monotonic()
                self.depth += 1
                try:
                    return await method(*args, **kwargs)
                finally:
                    self.depth -= 1
                    if not self.depth:
                        self.elapsed_ms += round((time.monotonic() - started) * 1000)
            return measured_async

        def measured(*args, **kwargs):
            started = time.monotonic()
            self.depth += 1
            try:
                return method(*args, **kwargs)
            finally:
                self.depth -= 1
                if not self.depth:
                    self.elapsed_ms += round((time.monotonic() - started) * 1000)
        return measured


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def measurement_summary(measurements, job):
    """Keep unknown/interrupted work visible; never manufacture zero timings."""
    complete = all(isinstance(row.get("elapsed_ms"), (int, float))
                   and row.get("status", "legacy_finished") not in {"running", "interrupted"}
                   and isinstance(row.get("request_count"), int) for row in measurements)
    measured_requests = sum(row.get("request_count") or 0 for row in measurements)
    complete &= measured_requests == job["request_count"] and not job.get("uncertain_request_count", 0)
    known_ms = sum(row.get("elapsed_ms") or 0 for row in measurements)
    return {"model_ms": known_ms if complete else None, "observed_model_ms": known_ms,
            "model_timing_complete": bool(complete), "measured_request_count": measured_requests,
            "unmeasured_or_uncertain_requests": max(0, job["request_count"] - measured_requests),
            "image_metrics_complete": all("image_count_per_attempt" in row and row["image_count_per_attempt"] is not None
                                          for row in measurements)}


def execution_ledger(path, previous, had_previous_work):
    ledger = read_json(path)
    if ledger is None:
        # Reports from the old runner can preserve known totals. An absent total
        # after prior work is unknown, not zero elapsed time.
        ledger = {"legacy_wall_ms": previous.get("wall_ms", None if had_previous_work else 0),
                  # The original runner timed coroutine creation rather than the
                  # awaited rebuild. Its index total cannot be migrated as known.
                  "legacy_index_ms": (previous.get("index_ms") if previous.get("index_timing_complete")
                                      else None if had_previous_work else 0), "sessions": []}
    for session in ledger["sessions"]:
        if session["status"] == "running":
            session.update(status="interrupted", wall_ms=None, index_ms=None)
    atomic_json(path, ledger)
    return ledger


def execution_totals(ledger):
    result = {}
    for field in ("wall_ms", "index_ms"):
        values = [ledger.get("legacy_" + field), *(session.get(field) for session in ledger["sessions"])]
        known = sum(value for value in values if isinstance(value, (int, float)))
        complete = all(isinstance(value, (int, float)) for value in values)
        result.update({field: known if complete else None, "observed_" + field: known,
                       field.removesuffix("_ms") + "_timing_complete": complete})
    return result


def update_comparison(report):
    for key in ("model_reduction_fraction", "timing_target_met"):
        report.pop(key, None)
    baseline, candidate = [report["strategies"].get(s) for s in ("content-keyframes-v1", "scene-chunks-v1")]
    if (baseline and candidate and baseline.get("model_ms")
            and all(value["status"] == "completed" and value.get("model_timing_complete") for value in (baseline, candidate))):
        report["model_reduction_fraction"] = 1 - candidate["model_ms"] / baseline["model_ms"]
        report["timing_target_met"] = report["model_reduction_fraction"] >= .5


def selected_subtitles(args, plan):
    asset_root = args.library / "assets" / args.asset
    active = read_json(asset_root / "active-analysis.json", {})
    container = active.get("container_subtitles")
    cues = read_jsonl(asset_root / container["path"]) if container else read_jsonl(asset_root / "subtitles/external.jsonl")
    if not container:
        for value in active.get("subtitle_runs", {}).values():
            cues.extend(read_jsonl(asset_root / value["path"]))
    annotations = [read_json(p) for p in (asset_root / "annotations").glob("*.json")]
    overrides = {a["record_id"]: a["summary"] for a in sorted(annotations, key=lambda a: a.get("updated_at", ""))
                 if a.get("summary") is not None}
    cues = [{**cue, "text": overrides.get(cue["id"], cue["text"])} for cue in cues
            if cue["end_ms"] > plan["start_ms"] and cue["start_ms"] < plan["end_ms"]]
    return sorted(cues, key=lambda cue: (cue["start_ms"], cue["end_ms"], cue["id"]))


def input_identity(args, plan, profile):
    asset_root = args.library / "assets" / args.asset
    registry = read_json(asset_root / "frames/registry.json")
    picture_hashes = {frame["id"]: hashlib.sha256((asset_root / registry[frame["id"]]["path"]).read_bytes()).hexdigest()
                      for frame in plan["frames"]}
    cues = selected_subtitles(args, plan)
    value = {"asset_id": args.asset, "fingerprint": plan["fingerprint"], "start_ms": plan["start_ms"],
             "end_ms": plan["end_ms"], "plan_id": plan["id"], "frames_digest": digest(plan["frames"]),
             "picture_hashes": picture_hashes, "subtitle_digest": digest(cues),
             "model_identity": model_identity(profile), "initial_character_memory": "empty"}
    return {**value, "digest": digest(value)}


def validate_seed(destination, args, plan, profile, expected):
    target = destination / "assets" / args.asset
    stored = read_json(destination / "benchmark-input.json")
    if stored is not None and stored != expected:
        raise ValueError("Benchmark input identity changed; use a new work directory")
    manifest = read_json(target / "manifest.json")
    copied_plan = read_json(target / "keyframe-plans" / (plan["id"] + ".json"))
    cues = read_jsonl(target / "subtitles/external.jsonl")
    cues.sort(key=lambda cue: (cue["start_ms"], cue["end_ms"], cue["id"]))
    profiles = read_json(destination / "private/profiles.json", {}).get("profiles", [])
    copied_profile = next((item for item in profiles if item["id"] == profile["id"]), None)
    if (not manifest or manifest["fingerprint"] != expected["fingerprint"]
            or not copied_plan or digest(copied_plan["frames"]) != expected["frames_digest"]
            or copied_plan["start_ms"] != expected["start_ms"] or copied_plan["end_ms"] != expected["end_ms"]
            or digest(cues) != expected["subtitle_digest"] or copied_profile is None
            or model_identity(copied_profile) != expected["model_identity"]):
        raise ValueError("Existing benchmark library does not match requested frames, subtitles, range or model")
    registry = read_json(target / "frames/registry.json")
    for frame_id, expected_hash in expected["picture_hashes"].items():
        if hashlib.sha256((target / registry[frame_id]["path"]).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Benchmark image bytes changed; use a new work directory")
    atomic_json(destination / "benchmark-input.json", expected)


def seed(args, destination, plan, profile, expected=None):
    expected = expected or input_identity(args, plan, profile)
    asset_root = args.library / "assets" / args.asset
    target = destination / "assets" / args.asset
    if (destination / "library.json").exists():
        validate_seed(destination, args, plan, profile, expected)
        return Library(destination)
    manifest = read_json(asset_root / "manifest.json")
    manifest.update(subtitle_mode="external", project_id=None)
    atomic_json(target / "manifest.json", manifest)
    cues = selected_subtitles(args, plan)
    atomic_jsonl(target / "subtitles/external.jsonl", cues)
    registry = read_json(asset_root / "frames/registry.json")
    selected = {}
    for frame in plan["frames"]:
        entry = registry[frame["id"]]
        path = target / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset_root / entry["path"], path)
        selected[frame["id"]] = entry
    atomic_json(target / "frames/registry.json", selected)
    atomic_json(target / "keyframe-plans" / (plan["id"] + ".json"), plan)
    locations = read_json(args.library / "private/media-locations.json")
    atomic_json(destination / "private/media-locations.json", {args.asset: locations[args.asset]})
    atomic_json(destination / "private/profiles.json", {"profiles": [profile]})
    library = Library(destination)
    validate_seed(destination, args, plan, profile, expected)
    return library


async def run(args):
    if not 1 <= args.max_requests <= 100000:
        raise ValueError("max-requests must be between 1 and 100000")
    original = read_json(args.library / "assets" / args.asset / "keyframe-plans" / (args.plan + ".json"))
    shots = [s for s in original["shots"] if args.start_ms <= s["start_ms"] and s["end_ms"] <= args.end_ms]
    if not shots or shots[0]["start_ms"] != args.start_ms or shots[-1]["end_ms"] != args.end_ms:
        raise ValueError("Choose exact whole-shot boundaries from the saved plan")
    shot_ids = {s["id"] for s in shots}
    plan = {**original, "shots": shots, "frames": [f for f in original["frames"] if f["shot_id"] in shot_ids],
            "batches": [b for b in original["batches"] if b["shot_id"] in shot_ids],
            "start_ms": args.start_ms, "end_ms": args.end_ms}
    plan["id"] = plan["plan_id"] = "plan_" + digest([original["id"], args.start_ms, args.end_ms])
    settings = read_json(args.library / "private/settings.json")
    profile_id = settings["bindings"]["vision"]
    profiles = json.loads((args.library / "private/profiles.json").read_text(encoding="utf-8"))
    profile = next(p for p in profiles["profiles"] if p["id"] == profile_id)
    if profile.get("provider_type") != "codex_cli":
        raise ValueError("This benchmark only supports the explicitly authorized current Codex connection")
    profile = {**profile, "capabilities": list(set(profile["capabilities"]) | {"preparation", "vision"})}
    expected_inputs = input_identity(args, plan, profile)
    report = read_json(args.output, {"scan_ms": 0, "scan_reused": True, "frames": len(plan["frames"]),
        "shots": len(shots), "start_ms": args.start_ms, "end_ms": args.end_ms,
        "model": profile.get("model") or "Codex CLI configured default", "provider_type": "codex_cli",
        "quality_status": "Human review required; timing alone is not acceptance", "strategies": {}})
    if report.get("input_digest") not in {None, expected_inputs["digest"]}:
        raise ValueError("Report belongs to different benchmark inputs; choose a new report and work directory")
    if any(report.get(key) != value for key, value in
           (("frames", len(plan["frames"])), ("start_ms", args.start_ms), ("end_ms", args.end_ms))):
        raise ValueError("Report range or frame selection differs from the requested benchmark")
    report.update(runner_schema_version=2, input_digest=expected_inputs["digest"],
                  model_resolution="explicit_profile_model" if profile.get("model") else "unresolved_cli_default",
                  timing_definition="Provider-method wall time includes local image preparation, CLI startup and model retries; "
                                    "execution wall time excludes scratch seeding and the reused scan. prompt_text_bytes measures "
                                    "stage instructions and user text before generic transport wrappers; it is not a token count.",
                  fairness="Same copied source images, timed subtitles and configured model profile; empty initial character "
                           "memory. Strategies run sequentially, so service load/order can affect timing. "
                           "Layered analysis also interprets subtitles; compare quality separately.")
    for strategy in args.strategies:
        directory = args.work_dir / strategy
        library = seed(args, directory, plan, profile, expected_inputs)
        provider = MeasuredProviders(directory / "private", directory / "measurements.json")
        search = MeasuredSearch(directory / "indexes", provider)
        queue = JobQueue(library, provider, search)
        jobs = queue.list()
        if any(job["type"] != "analysis" or job.get("asset_id") != args.asset
               or job["config"].get("analysis_strategy") != strategy
               or job["config"].get("keyframe_plan_id") != plan["id"]
               or job["config"]["start_ms"] != args.start_ms or job["config"]["end_ms"] != args.end_ms for job in jobs):
            raise ValueError("Scratch directory contains a job for different benchmark inputs")
        if len(jobs) > 1:
            raise ValueError("Benchmark scratch directory contains multiple jobs; choose a clean directory")
        job = jobs[0] if jobs else None
        if not job:
            _, config = queue.validate(AnalysisInput(asset_id=args.asset, start_ms=args.start_ms, end_ms=args.end_ms,
                stages=["vision"], analysis_strategy=strategy, keyframe_plan_id=plan["id"],
                max_requests=args.max_requests), {"bindings": {"preparation": profile_id, "vision": profile_id}})
            job = queue.add("analysis", config, args.asset)
        ledger_path = directory / "execution-sessions.json"
        prior = report["strategies"].get(strategy, {})
        ledger = execution_ledger(ledger_path, prior,
                                  bool(read_json(directory / "measurements.json", [])) or job["request_count"] > 0)
        if job["status"] != "completed":
            if job["status"] in {"paused", "cancelled", "failed", "partial"}:
                queue.control(job["id"], "resume" if job["status"] == "paused" else "retry",
                              {"max_requests": args.max_requests})
            elif job["status"] == "queued":
                queue.update(job["id"], config={**job["config"], "max_requests": args.max_requests})
            else:
                raise ValueError("Benchmark job is already active")
            print(json.dumps({"strategy": strategy, "job": job["id"], "event": "starting"}), flush=True)
            session = {"id": uuid.uuid4().hex, "started_at": utc_now(), "status": "running",
                       "wall_ms": None, "index_ms": None}
            ledger["sessions"].append(session)
            atomic_json(ledger_path, ledger)
            started = time.monotonic()
            try:
                await queue.execute(job["id"])
                session["status"] = "finished"
            except BaseException:
                session["status"] = "interrupted"
                raise
            finally:
                session.update(finished_at=utc_now(), wall_ms=round((time.monotonic() - started) * 1000),
                               index_ms=search.elapsed_ms, job_status=queue.get(job["id"])["status"])
                atomic_json(ledger_path, ledger)
        job = queue.get(job["id"])
        measurements = read_json(directory / "measurements.json", [])
        report["strategies"][strategy] = {"status": job["status"], "error": job.get("error"),
            "failures": job.get("failures", []), **execution_totals(ledger), **measurement_summary(measurements, job),
            "request_count": job["request_count"], "usage": job.get("usage"), "cost": job.get("cost"),
            "stage_metrics": job.get("stage_metrics"), "calls": copy.deepcopy(measurements), "execution_sessions": ledger,
            "legacy_measurement_rows": sum("call_id" not in row for row in measurements),
            "record_counts": {kind: sum(r["kind"] == kind for r in library.records(args.asset))
                              for kind in ("scene", "visual", "subtitle")}}
        update_comparison(report)
        atomic_json(args.output, report)
        print(json.dumps({"strategy": strategy, **{k: report["strategies"][strategy][k]
              for k in ("status", "model_ms", "request_count", "error")}}), flush=True)
        if job["status"] != "completed":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=60)
    parser.add_argument("--strategies", nargs="+", choices=["content-keyframes-v1", "scene-chunks-v1"],
                        default=["content-keyframes-v1", "scene-chunks-v1"])
    asyncio.run(run(parser.parse_args()))
