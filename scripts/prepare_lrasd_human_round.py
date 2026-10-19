"""Check saved human reviews, then prepare a frozen scene-heldout training round.

The default command only checks readiness. --export explicitly writes new,
derived training assets. Neither mode starts training or edits annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall.lrasd_training_data import (  # noqa: E402
    HUMAN_LABEL_SOURCE, HUMAN_REPLAY_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE,
    atomic_json, load_annotation_plan, load_manifest, resolve_asset,
    sha256_file, split_summary, supervision_summary,
)
from export_lrasd_annotations import (  # noqa: E402
    checked_clip, checked_id, checked_snapshot, checked_track, confined,
    export_annotations, read_clip,
)


def check_readiness(root, plan):
    root = Path(root).resolve()
    blocked, ready, snapshots = [], [], []
    for item in plan["items"]:
        identity = checked_id(item["clip_id"], "planned clip ID")
        directory = confined(root, root / identity)
        path = directory / "clip.json"
        context = {"item_id": item["id"], "clip_id": identity, "split": item["split"],
                   "title": item.get("title", ""), "start_ms": item["start_ms"]}
        if not path.is_file():
            blocked.append({**context, "reason": "clip_missing"})
            continue
        try:
            clip, digest = read_clip(path)
            if clip.get("status") != "ready":
                raise ValueError("clip_not_ready")
            checked_clip(directory, clip)
            if (clip["source_fingerprint"] != plan["source_fingerprint"]
                    or (plan.get("asset_id") and clip["asset_id"] != plan["asset_id"])
                    or (clip["start_ms"], clip["end_ms"]) != (item["start_ms"], item["end_ms"])):
                raise ValueError("source_or_range_does_not_match_plan")
            checked_snapshot(root, directory, clip)
            usable = 0
            for track in clip["tracks"]:
                reason = checked_track(track, clip["frame_count"], clip["revision"])
                # A validated, reviewed all-unknown track supplies no loss or
                # metric frames. Keep its saved annotation and skip it just as
                # the exporter does; the clip still needs a supervised track.
                if reason in {"excluded", "all_unknown"}:
                    continue
                if reason:
                    blocked.append({**context, "track_id": track.get("id"), "reason": reason})
                else:
                    usable += 1
            if not usable:
                blocked.append({**context, "reason": "no_usable_reviewed_track"})
            if not any(entry["clip_id"] == identity for entry in blocked):
                ready.append({**context, "usable_tracks": usable})
            snapshots.append({"clip_id": identity, "annotation_sha256": digest, "revision": clip["revision"]})
        except (ValueError, OSError, KeyError) as exc:
            blocked.append({**context, "reason": str(exc)})
    return {"ready": not blocked, "planned_clips": len(plan["items"]), "ready_clips": len(ready),
            "blocked": blocked, "clips": ready, "annotation_snapshots": snapshots,
            "annotation_plan_sha256": plan["plan_sha256"],
            "validation_scope": "within_title_scene_holdout", "cross_title_validation": False,
            "baseline_prior_film_exposure": "previous_weak_adaptation_not_pristine",
            "starts_training": False, "writes_annotations": False}


def check_baseline_exposure(plan, prior_manifest):
    """Record known prior film windows; reject nearby new val/test scenes."""
    if prior_manifest is None or not Path(prior_manifest).is_file():
        return {"status": "unavailable", "pristine_claimed": False, "known_windows": [], "conflicts": []}
    known, conflicts = [], []
    for line in Path(prior_manifest).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source = row.get("source_sha256", row.get("source_video_sha256", row.get("source_fingerprint")))
        if source != plan["source_fingerprint"]:
            continue
        if isinstance(row.get("source_start_seconds"), (int, float)):
            start = round(row["source_start_seconds"] * 1000)
        else:
            start = row.get("source_start_ms")
        if type(start) is not int:
            return {"status": "source_timeline_unavailable", "pristine_claimed": False,
                    "known_windows": known, "conflicts": [{"reason": "cannot_locate_prior_target_film_window"}]}
        end = start + len(row["labels"]) * 40
        known.append({"sample_id": row["sample_id"], "split": row["split"], "start_ms": start, "end_ms": end})
        for item in plan["items"]:
            if item["split"] == "train":
                continue
            distance = max(item["start_ms"] - end, start - item["end_ms"])
            if distance < plan["minimum_cross_split_gap_ms"]:
                conflicts.append({"item_id": item["id"], "clip_id": item["clip_id"], "reason": "near_prior_baseline_film_window",
                                  "prior_sample_id": row["sample_id"], "distance_ms": distance})
    return {"status": "known_manifest_checked_other_exposure_not_ruled_out", "pristine_claimed": False,
            "manifest_sha256": sha256_file(prior_manifest), "known_windows": known, "conflicts": conflicts}


def append_official_train(manifest, replay_manifest):
    """Keep only official train replay, retaining absolute asset paths and provenance."""
    rows = load_manifest(manifest, frames=50, allow_within_show_group_split=True)
    replay = load_manifest(replay_manifest, frames=50, allow_within_show_group_split=True)
    selected = [dict(row) for row in replay if row["split"] == "train" and row.get("label_source") == OFFICIAL_LABEL_SOURCE]
    if not selected:
        raise ValueError("Replay manifest contains no official training rows")
    supervision_summary(selected, OFFICIAL_LABEL_SOURCE)
    # Official replay cannot silently introduce another view of this movie.
    human_sources = {row["source_fingerprint"] for row in rows}
    for row in selected:
        if row.get("source_fingerprint", row.get("source_video_sha256", row.get("source_sha256"))) in human_sources:
            raise ValueError("Official replay source overlaps the target film")
        for key in ("visual_path", "audio_path"):
            row[key] = str(resolve_asset(replay_manifest, row[key]).resolve())
    rows.extend(selected)
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                                for row in rows), encoding="utf-8")
    load_manifest(manifest, frames=50, allow_within_show_group_split=True)
    supervision_summary(rows, HUMAN_REPLAY_LABEL_SOURCE)
    summary_path = manifest.parent / "label-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(label_source=HUMAN_REPLAY_LABEL_SOURCE, labeled_manifest_sha256=sha256_file(manifest),
                   human_frame_asd_ground_truth=False, human_frame_asd_ground_truth_present=True,
                   machine_generated=any(row.get("machine_generated") is True for row in rows),
                   official_replay={"manifest_sha256": sha256_file(replay_manifest), "train_windows": len(selected),
                                    "validation_or_test_replayed": False}, split_counts={split: data["windows"] for split, data in split_summary(rows).items()})
    atomic_json(summary_path, summary)


def commands(output, baseline_bundle, *, replay=False):
    def q(path):
        return shlex.quote(str(path))
    python = "/mnt/d/SceneRecall-LR-ASD/.venv/bin/python"
    manifest = output / "dataset/labeled.jsonl"
    common = f"--manifest {q(manifest)} --baseline-bundle {q(baseline_bundle)}"
    return {
        "baseline_validation": f"{python} scripts/evaluate_lrasd_human.py {common} --output {q(output / 'baseline-validation.json')}",
        "finetune": (f"{python} scripts/train_lrasd.py --manifest {q(manifest)} --output {q(output / 'finetune')} "
                     f"--parent-bundle {q(baseline_bundle)} --stage-name kaguya-human-scene-v1 --frames 50 --batch-size 4 "
                     f"--epochs 6 --patience 3 --lr 0.00002 --label-source {HUMAN_REPLAY_LABEL_SOURCE if replay else HUMAN_LABEL_SOURCE} "
                     "--allow-within-show-group-split --split-strategy scene_holdout_same_title --defer-final-test "
                     "--dataset-name Kaguya-human-ASD --dataset-version scene-v1 "
                     "--source-domain 'Japanese 2D film; human reviewed frame ASD' "
                     "--target-domain 'Kaguya within-title scene holdout; cross-title unverified'"),
        "final_comparison": (f"{python} scripts/evaluate_lrasd_human.py {common} "
                             f"--candidate-bundle {q(output / 'finetune/bundle')} --final-test "
                             f"--baseline-report {q(output / 'baseline-validation.json')} --output {q(output / 'comparison.json')}")}


def prepare_round(root, plan_path, output, baseline_bundle, *, export=False, replay_manifest=None, prior_manifest=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new round output directory; prior reports and datasets are immutable")
    root = Path(root).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Round output must be separate from original annotation clips")
    plan = load_annotation_plan(plan_path)
    report = check_readiness(root, plan)
    exposure = check_baseline_exposure(plan, prior_manifest)
    report["baseline_exposure_check"] = exposure
    if exposure["conflicts"]:
        report["blocked"].extend(exposure["conflicts"])
        report["ready"] = False
    report.update(export_requested=export, dataset_exported=False)
    output.mkdir(parents=True)
    atomic_json(output / "annotation-plan.json", {key: value for key, value in plan.items() if key != "plan_sha256"})
    if export and report["ready"]:
        summary = export_annotations(root, output / "dataset", plan_path=plan_path)
        report.update(dataset_exported=True, training_ready=summary["training_ready"],
                      manifest_sha256=summary["labeled_manifest_sha256"], split_counts=summary["split_counts"])
        if not summary["training_ready"]:
            report["blocked"].append({"reason": "insufficient_windows_or_missing_class",
                                      "required": "train>=20, val>=5, test>=5; both classes in every split"})
            report["ready"] = False
        elif replay_manifest:
            append_official_train(output / "dataset/labeled.jsonl", replay_manifest)
            report["manifest_sha256"] = sha256_file(output / "dataset/labeled.jsonl")
        if report["ready"]:
            report["commands"] = commands(output, baseline_bundle, replay=bool(replay_manifest))
            atomic_json(output / "commands.json", report["commands"])
    elif not export:
        report["next_action"] = "Finish/review/save the listed clips, then rerun with --export and a new output directory."
    atomic_json(output / "readiness.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-bundle", type=Path, default=Path("training/lrasd/runs/cmdam-jp2d-20261015-v1/bundle"))
    parser.add_argument("--official-replay-manifest", type=Path)
    parser.add_argument("--prior-adaptation-manifest", type=Path, default=Path("training/lrasd/labels-2s-v1/labeled.jsonl"))
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = prepare_round(args.annotations_root, args.plan, args.output, args.baseline_bundle,
                               export=args.export, replay_manifest=args.official_replay_manifest,
                               prior_manifest=args.prior_adaptation_manifest)
    except (OSError, ValueError, ImportError) as exc:
        print(f"Human round preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("ready", "planned_clips", "ready_clips", "blocked", "dataset_exported", "starts_training")}, ensure_ascii=False))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
