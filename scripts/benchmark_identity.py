#!/usr/bin/env python3
"""Three-strategy short-clip benchmark. Provider calls require explicit opt-in.

Without --allow-provider-calls this writes a six-cell not_measured report only.
Each strategy executes in a fresh process, then compares an empty local feature
cache with retained features/resident models. Original media are never copied or
uploaded by this runner; configured vision providers receive normal frame inputs
only after opt-in. No weights are downloaded. Use a reviewed short saved scan.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from scenerecall.characters import scope_id
from scenerecall.jobs import JobQueue
from scenerecall.library import Library, atomic_json, atomic_jsonl, digest, read_json
from scenerecall.models import AnalysisInput


_spec = importlib.util.spec_from_file_location("identity_scene_benchmark_base", Path(__file__).with_name("benchmark_scenes.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
GROUPS = ("old", "new_visual", "new_audiovisual")
METRICS = ("total_ms", "first_searchable_ms", "first_visual_searchable_ms", "visual_calls", "fallback_count",
           "timeout_count", "invalid_output_count", "role_coverage", "role_error_rate",
           "dialogue_speaker_accuracy", "diarization_error_rate", "audio_total_ms", "visual_model_load_ms")


def report_template(reason):
    return {"schema_version": "1.0", "benchmark": "character-centric-av", "status": "not_measured",
            "measurement_policy": "No missing measurement is represented by zero. Synthetic tests are not model accuracy.",
            "timing_definition": "total_ms includes model loading, audio extraction/diarization/features, visual identity, "
                                 "provider calls and indexing; excludes source cloning and the reused local keyframe scan. "
                                 "Cold means new process and empty application feature caches (OS/HF file cache uncontrolled). "
                                 "Warm means fresh analysis/index with retained local features and resident models; "
                                 "provider result caches are not copied. Provider service latency is uncontrolled. "
                                 "visual_model_load_ms is visual only; audio loading remains in diarization/embedding totals.",
            "baseline_definition": "old uses scene-chunks-v1 with frozen pre-identity prompts and no identity stage; "
                                   "it is a legacy-prompt baseline in the current executable, not the old commit binary.",
            "groups": {group: {phase: {"status": "not_measured", "reason": reason,
                                      **dict.fromkeys(METRICS)} for phase in ("cold", "warm")} for group in GROUPS}}


def _assignment(weights):
    """Maximum-weight bipartite assignment, including unmatched dummy nodes.

    Hungarian shortest-augmenting-path formulation; O(n^3), no ML extra needed.
    """
    rows, cols = len(weights), max((len(row) for row in weights), default=0)
    n = max(rows, cols)
    if not n:
        return {}
    maximum = max((max(row) for row in weights if row), default=0)
    cost = [[maximum - (weights[i][j] if i < rows and j < len(weights[i]) else 0)
             for j in range(n)] for i in range(n)]
    u, v, p, previous = [0] * (n + 1), [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = [float("inf")] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    value = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if value < minimum[j]:
                        minimum[j], previous[j] = value, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            prior = previous[j0]
            p[j0], j0 = p[prior], prior
    return {p[j] - 1: j - 1 for j in range(1, n + 1) if 0 < p[j] <= rows and j <= cols}


def diarization_metrics(reference, hypothesis):
    """Overlap-aware DER, zero collar, globally optimal speaker permutation."""
    if reference is None:
        return {"diarization_error_rate": None, "der_reason": "no_diarization_labels"}
    times = sorted({row[key] for row in [*reference, *hypothesis] for key in ("start_ms", "end_ms")})
    reference_ids = sorted({row["speaker_id"] for row in reference})
    hypothesis_ids = sorted({row["speaker_id"] for row in hypothesis})
    weights = [[0 for _ in hypothesis_ids] for _ in reference_ids]
    intervals = []
    for start, end in zip(times, times[1:]):
        truth = {row["speaker_id"] for row in reference if row["start_ms"] < end and row["end_ms"] > start}
        predicted = {row["speaker_id"] for row in hypothesis if row["start_ms"] < end and row["end_ms"] > start}
        duration = end - start
        intervals.append((duration, truth, predicted))
        for i, truth_id in enumerate(reference_ids):
            for j, hypothesis_id in enumerate(hypothesis_ids):
                if truth_id in truth and hypothesis_id in predicted:
                    weights[i][j] += duration
    mapping = {hypothesis_ids[j]: reference_ids[i] for i, j in _assignment(weights).items()}
    missed = false_alarm = confused = denominator = 0
    for duration, truth, predicted in intervals:
        correct = len(truth & {mapping[value] for value in predicted if value in mapping})
        missed += max(0, len(truth) - len(predicted)) * duration
        false_alarm += max(0, len(predicted) - len(truth)) * duration
        confused += (min(len(truth), len(predicted)) - correct) * duration
        denominator += len(truth) * duration
    return {"diarization_error_rate": (missed + false_alarm + confused) / denominator if denominator else None,
            "der_speaker_ms": denominator, "der_missed_ms": missed, "der_false_alarm_ms": false_alarm,
            "der_confused_ms": confused, "der_collar_ms": 0, "der_overlap_included": True,
            "der_reason": None if denominator else "no_labelled_speech", "speaker_permutation": mapping}


def speaker_accuracy(reference, records):
    if reference is None:
        return {"dialogue_speaker_accuracy": None, "speaker_accuracy_reason": "no_character_speech_labels"}
    reference = [row for row in reference if row.get("character_id")]
    times = sorted({row[key] for row in [*reference, *records] for key in ("start_ms", "end_ms")})
    correct = total = 0
    for start, end in zip(times, times[1:]):
        truth = Counter(row["character_id"] for row in reference if row["start_ms"] < end and row["end_ms"] > start)
        predicted = Counter(row["character_id"] for row in records
                            if row.get("character_id") and row["start_ms"] < end and row["end_ms"] > start)
        total += sum(truth.values()) * (end - start)
        correct += sum((truth & predicted).values()) * (end - start)
    return {"dialogue_speaker_accuracy": correct / total if total else None,
            "speaker_correct_ms": correct, "speaker_labelled_ms": total,
            "speaker_accuracy_definition": "duration-weighted character speech recall; unknown/candidate are abstentions"}


def _iou(a, b):
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection) if intersection else 0


def visual_accuracy(reference, predictions):
    linked = sum(bool(row.get("character_id")) for row in predictions)
    result = {"role_coverage": linked / len(predictions) if predictions else None,
              "role_error_rate": None, "visible_regions": len(predictions), "linked_regions": linked,
              "role_coverage_definition": "confirmed links / observed visible regions", "role_labelled_count": None}
    if reference is None:
        return {**result, "role_error_reason": "no_visual_identity_labels"}
    weights = [[_iou(truth["box"], row["box"]) if truth["frame_id"] == row["frame_id"] and row.get("box") else 0
                for row in predictions] for truth in reference]
    assignments = {i: j for i, j in _assignment(weights).items() if weights[i][j] >= .5}
    labelled_linked = wrong = 0
    for i, j in assignments.items():
        if predictions[j].get("character_id"):
            labelled_linked += 1
            wrong += predictions[j]["character_id"] != reference[i].get("character_id")
    result.update(role_coverage=labelled_linked / len(reference) if reference else None,
                  role_error_rate=wrong / labelled_linked if labelled_linked else None,
                  role_labelled_count=len(reference), role_wrong_count=wrong, role_labelled_linked_count=labelled_linked,
                  role_coverage_definition="confirmed links / labelled regions; one-to-one frame-box IoU >= 0.5")
    return result


class Providers(base.MeasuredProviders):
    def __getattribute__(self, name):
        method = super().__getattribute__(name)
        if name not in {"analyze", "prepare_subtitles", "prepare_visual", "reconcile_scene"}:
            return method
        async def measured(*args, **kwargs):
            try:
                return await method(*args, **kwargs)
            except BaseException as exc:
                path = self.measurements.with_name("errors.json")
                rows = read_json(path, [])
                rows.append({"method": name, "type": type(exc).__name__, "code": getattr(exc, "code", None)})
                atomic_json(path, rows)
                raise
        return measured


class Search(base.MeasuredSearch):
    def __init__(self, directory, provider):
        super().__init__(directory, provider)
        self.started = None
        self.first_searchable_ms = None
        self.first_visual_searchable_ms = None

    def upsert(self, records, *args, **kwargs):
        records = list(records)
        value = super().upsert(records, *args, **kwargs)
        if self.started is not None and records:
            elapsed = round((time.monotonic() - self.started) * 1000)
            if self.first_searchable_ms is None:
                self.first_searchable_ms = elapsed
            if self.first_visual_searchable_ms is None and any(row.get("kind") in {"scene", "visual"} for row in records):
                self.first_visual_searchable_ms = elapsed
        return value


def clone_source(args, destination, plan, profile):
    """Copy only local evidence/reference inputs, never a source video or tokens."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Benchmark phase already exists; use a new work directory to avoid mixing runs")
    original_root = args.library / "assets" / args.asset
    manifest = read_json(original_root / "manifest.json")
    catalog = read_json(args.library / "organization/catalog.json", {"assets": {}})
    member = catalog["assets"].get(args.asset, {})
    cast_scope = member.get("character_scope_id") or scope_id(manifest)
    for original_asset in (args.library / "assets").glob("*/manifest.json"):
        value = read_json(original_asset)
        membership = catalog["assets"].get(value["id"], {})
        if value["id"] != args.asset and membership.get("character_scope_id") != cast_scope:
            continue
        if membership.get("trash_id"):
            continue
        value.update({key: copy.deepcopy(membership[key]) for key in
                      ("anime_profile", "anime_title", "season", "episode", "kind") if key in membership})
        value.update(project_id=None, character_scope_id=cast_scope)
        if value["id"] == args.asset:
            value.update(subtitle_mode="external")
        atomic_json(destination / "assets" / value["id"] / "manifest.json", value)
    source_cast = args.library / "works" / cast_scope
    if source_cast.is_dir():
        shutil.copytree(source_cast, destination / "works" / cast_scope,
                        ignore=shutil.ignore_patterns("voice-features"), dirs_exist_ok=True)
    target = destination / "assets" / args.asset
    registry = read_json(original_root / "frames/registry.json")
    selected = {}
    for frame in plan["frames"]:
        entry = registry[frame["id"]]
        path = target / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_root / entry["path"], path)
        selected[frame["id"]] = entry
    atomic_json(target / "frames/registry.json", selected)
    atomic_json(target / "keyframe-plans" / f"{plan['id']}.json", plan)
    atomic_jsonl(target / "subtitles/external.jsonl", base.selected_subtitles(args, plan))
    locations = read_json(args.library / "private/media-locations.json")
    atomic_json(destination / "private/media-locations.json", {args.asset: locations[args.asset]})
    atomic_json(destination / "private/profiles.json", {"profiles": [profile]})
    for relative in ("identity/visual/settings.json", "audio/settings.json"):
        source = original_root / relative
        if source.is_file():
            atomic_json(target / relative, read_json(source))
    return Library(destination)


def retain_feature_cache(cold, warm, asset_id):
    for relative in ("cache", f"assets/{asset_id}/audio/cache", f"assets/{asset_id}/audio/features",
                     f"assets/{asset_id}/identity/visual/detections", f"assets/{asset_id}/identity/visual/crops"):
        source = cold / relative
        if source.is_dir():
            shutil.copytree(source, warm / relative, dirs_exist_ok=True)
    for source in (cold / "works").glob("*/voice-features"):
        shutil.copytree(source, warm / source.relative_to(cold), dirs_exist_ok=True)


def read_inputs(args):
    if args.end_ms <= args.start_ms or args.end_ms - args.start_ms > 120000:
        raise ValueError("Benchmark is restricted to an explicitly selected clip of at most 120 seconds")
    original = read_json(args.library / "assets" / args.asset / "keyframe-plans" / f"{args.plan}.json")
    if not original:
        raise ValueError("Saved local keyframe plan not found")
    shots = [row for row in original["shots"] if args.start_ms <= row["start_ms"] and row["end_ms"] <= args.end_ms]
    if not shots or shots[0]["start_ms"] != args.start_ms or shots[-1]["end_ms"] != args.end_ms:
        raise ValueError("Select exact whole-shot boundaries from the saved keyframe plan")
    ids = {row["id"] for row in shots}
    plan = {**original, "start_ms": args.start_ms, "end_ms": args.end_ms, "shots": shots,
            "frames": [row for row in original["frames"] if row["shot_id"] in ids],
            "batches": [row for row in original["batches"] if row["shot_id"] in ids]}
    plan["id"] = plan["plan_id"] = "plan_" + digest([original["id"], args.start_ms, args.end_ms])
    settings = read_json(args.library / "private/settings.json", {})
    profile_id = settings.get("bindings", {}).get("vision")
    profile = next((row for row in read_json(args.library / "private/profiles.json", {}).get("profiles", [])
                    if row["id"] == profile_id), None)
    if not profile or profile.get("provider_type") != "codex_cli":
        raise ValueError("Use an explicitly selected Codex CLI profile; the runner never copies API secrets")
    profile = {**profile, "capabilities": sorted(set(profile["capabilities"]) | {"preparation", "vision"})}
    return plan, profile


async def execute_group(args):
    from scenerecall.audio_identity import AudioIdentityService
    from scenerecall.visual_identity import VisualIdentityPipeline
    plan, profile = read_inputs(args)
    labels = read_json(args.labels) if args.labels else {}
    report = report_template("phase_not_started")["groups"][args.execute_group]
    for phase in ("cold", "warm"):
        directory = args.work_dir / args.execute_group / phase
        library = clone_source(args, directory, plan, profile)
        if phase == "warm":
            retain_feature_cache(directory.parent / "cold", directory, args.asset)
        VisualIdentityPipeline(library).configure(args.asset, {"enabled": args.execute_group != "old"})
        AudioIdentityService(library).configure(args.asset, {"enabled": args.execute_group == "new_audiovisual"})
        providers = Providers(directory / "private", directory / "measurements.json")
        search = Search(directory / "indexes", providers)
        queue = JobQueue(library, providers, search)
        strategy = "scene-chunks-v1"
        _, config = queue.validate(AnalysisInput(asset_id=args.asset, start_ms=args.start_ms, end_ms=args.end_ms,
            stages=["vision"], analysis_strategy=strategy, keyframe_plan_id=plan["id"], max_requests=args.max_requests),
            {"bindings": {"vision": profile["id"], "preparation": profile["id"]}})
        if args.execute_group == "old":
            from scenerecall.scene_jobs import PRE_IDENTITY_PROMPT_VERSIONS
            config["scene_prompt_versions"] = copy.deepcopy(PRE_IDENTITY_PROMPT_VERSIONS)
            config.pop("identity_settings", None)
        if args.visual_batch_size is not None:
            config["visual_batch_size"] = args.visual_batch_size
        job = queue.add("analysis", config, args.asset)
        search.started = time.monotonic()
        await queue.execute(job["id"])
        total_ms = round((time.monotonic() - search.started) * 1000)
        job = queue.get(job["id"])
        calls = read_json(directory / "measurements.json", [])
        errors = read_json(directory / "errors.json", [])
        raw = read_json(queue._identity_path(job), {}) if hasattr(queue, "_identity_path") else {}
        visual = raw.get("visual", {})
        audio = raw.get("audio", {})
        regions = [{"frame_id": row["frame_id"], "box": row["box"], "character_id": row["match"].get("character_id")}
                   for row in visual.get("regions", [])]
        if args.execute_group == "old":
            regions = [{"frame_id": entity.get("portrait_frame_id") or (entity.get("evidence_frame_ids") or [None])[0],
                        "box": entity.get("portrait_box"), "character_id": entity.get("character_id")}
                       for record in library.records(args.asset) if record.get("kind") in {"visual", "scene"}
                       for entity in record.get("entities", []) if entity.get("kind") != "object"]
        splits = list((library.asset_dir(args.asset) / "analyses" / job["run_id"] / "scene-splits").glob("*.json"))
        row = {"status": "measured" if job["status"] == "completed" else "incomplete", "job_status": job["status"],
               "error": job.get("error"), "total_ms": total_ms, "first_searchable_ms": search.first_searchable_ms,
               "first_visual_searchable_ms": search.first_visual_searchable_ms,
               "visual_calls": sum(call.get("request_count") or 0 for call in calls
                                   if call["method"] in {"analyze", "prepare_visual"}),
               "fallback_count": len(splits), "timeout_count": sum(row["code"] == "timeout" for row in errors),
               "invalid_output_count": sum(row["code"] == "invalid_output" for row in errors),
               "failure_count_definition": "failed provider-method calls; transport attempts remain in request_count",
               "request_count": job["request_count"], "stage_metrics": job.get("stage_metrics"),
               "audio_total_ms": round(audio.get("timings", {}).get("total_s", 0) * 1000) if audio else None,
               "visual_model_load_ms": visual.get("metrics", {}).get("model_load_ms"),
               "visual_batch_size": config.get("visual_batch_size"), "prompt_versions": config.get("scene_prompt_versions"),
               "visual_identity": {key: value for key, value in visual.items() if key in {"status", "metrics", "degraded_reason"}},
               "audio_identity": {key: value for key, value in audio.items() if key in {"status", "timings", "degraded", "cache_hits"}},
               **visual_accuracy(labels.get("visual"), regions),
               **speaker_accuracy(labels.get("speech"), audio.get("records", [])),
               **diarization_metrics(labels.get("diarization"), audio.get("records", []))}
        required = [visual] if args.execute_group == "new_visual" else [visual, audio] if args.execute_group == "new_audiovisual" else []
        if any(value.get("status") not in {"completed", "partial"} for value in required):
            row.update(status="degraded_not_comparable", reason="Required local identity stage did not complete")
        report[phase] = row
        atomic_json(args.work_dir / args.execute_group / "report.json", report)
        if job["status"] != "completed":
            break
    return report


def run(args):
    report = report_template("Provider calls were not authorized; pass --allow-provider-calls for a reviewed short clip")
    report["source"] = {"asset_id": args.asset, "plan_id": args.plan, "start_ms": args.start_ms, "end_ms": args.end_ms}
    if not args.allow_provider_calls:
        atomic_json(args.output, report)
        return report
    read_inputs(args)  # Validate before creating workers or making any calls.
    if args.execute_group:
        return asyncio.run(execute_group(args))
    for group in GROUPS:
        command = [sys.executable, str(Path(__file__).resolve()), "--library", str(args.library), "--asset", args.asset,
                   "--plan", args.plan, "--start-ms", str(args.start_ms), "--end-ms", str(args.end_ms),
                   "--work-dir", str(args.work_dir), "--output", str(args.output), "--max-requests", str(args.max_requests),
                   "--allow-provider-calls", "--execute-group", group]
        if args.labels:
            command.extend(["--labels", str(args.labels)])
        if args.visual_batch_size is not None:
            command.extend(["--visual-batch-size", str(args.visual_batch_size)])
        outcome = subprocess.run(command, check=False)
        measured = read_json(args.work_dir / group / "report.json")
        if measured:
            report["groups"][group] = measured
        if outcome.returncode:
            report["groups"][group].setdefault("failure", f"runner exited {outcome.returncode}")
        report["status"] = "measured_with_limitations"
        atomic_json(args.output, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--library", type=Path, required=True)
    result.add_argument("--asset", required=True)
    result.add_argument("--plan", required=True)
    result.add_argument("--start-ms", type=int, required=True)
    result.add_argument("--end-ms", type=int, required=True)
    result.add_argument("--work-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-requests", type=int, choices=range(1, 101), default=20)
    result.add_argument("--labels", type=Path, help="JSON {visual:[frame_id,box,character_id], speech:[times,character_id], diarization:[times,speaker_id]}")
    result.add_argument("--visual-batch-size", type=int, choices=range(1, 17), default=None,
                        help="Freeze a 1–16 frame preparation batch for controlled batch-size comparisons")
    result.add_argument("--allow-provider-calls", action="store_true")
    result.add_argument("--execute-group", choices=GROUPS, help=argparse.SUPPRESS)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
