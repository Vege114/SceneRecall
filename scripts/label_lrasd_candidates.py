#!/usr/bin/env python3
"""Create conservative MACHINE weak labels from two frozen LR-ASD teachers.

This never creates human ground truth. Held-out metrics on these labels measure
teacher agreement within one source movie, not Japanese anime ASD accuracy.
Run using the independent LR-ASD environment; weights must already be local.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall.av_sync_worker import load_lrasd_model  # noqa: E402
from scenerecall.lrasd_training_data import (  # noqa: E402
    atomic_json, canonical_hash, load_sample, resolve_asset, sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/d/SceneRecall-LR-ASD"))
    parser.add_argument("--ava-checkpoint", type=Path)
    parser.add_argument("--talkset-checkpoint", type=Path)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, choices=(50, 100), default=100)
    parser.add_argument("--min-valid-frames", type=int, default=None,
                        help="Minimum retained consensus frames; default is 40%% of T (20/40)")
    parser.add_argument("--positive-threshold", type=float, default=.95)
    parser.add_argument("--negative-threshold", type=float, default=.05)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not 1 <= args.max_candidates <= 1000:
        parser.error("At most 1000 candidates may be labeled per experiment")
    if not 1 <= args.batch_size <= 4 or args.batch_size * args.frames > 400:
        parser.error("Use at most 4 windows and 400 total visual frames per batch")
    minimum = math.ceil(.4 * args.frames)
    if args.min_valid_frames is None:
        args.min_valid_frames = minimum
    if not minimum <= args.min_valid_frames <= args.frames or args.boundary_radius < 2 or args.cpu_threads < 1:
        parser.error(f"Require {minimum}..{args.frames} valid frames, boundary radius >=2 and positive CPU threads")
    if not .95 <= args.positive_threshold <= 1 or not 0 <= args.negative_threshold <= .05:
        parser.error("Conservative consensus requires positive >=.95 and negative <=.05")
    args.ava_checkpoint = args.ava_checkpoint or args.source_root / "weight/pretrain_AVA.model"
    args.talkset_checkpoint = args.talkset_checkpoint or args.source_root / "weight/finetuning_TalkSet.model"
    return args


def emit(event, **fields):
    print(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
                     ensure_ascii=False, allow_nan=False), flush=True)


def candidate_rows(path, limit, frames=100):
    """Preserve the preparer's time-block splits; never randomly split windows."""
    rows, seen, groups = [], set(), {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        fields = {"sample_id", "split", "movie", "clip", "track", "start_frame", "end_frame", "fps",
                  "visual_path", "audio_path", "audio_start_seconds"}
        if not isinstance(row, dict) or fields - row.keys():
            raise ValueError(f"Candidate line {number} is missing required manifest fields")
        if row["sample_id"] in seen or row.get("synthetic") is True:
            raise ValueError("Duplicate or synthetic candidate rejected")
        seen.add(row["sample_id"])
        if row["split"] not in {"train", "val", "test"} or row["fps"] != 25:
            raise ValueError("Candidates require an explicit original split and exact 25 fps")
        if row["end_frame"] - row["start_frame"] != frames:
            raise ValueError(f"Candidates must contain T={frames} continuous frames")
        group = row.get("group", row["movie"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError("The same group/time block leaks across splits")
        groups[group] = row["split"]
        if any(not resolve_asset(path, row[key]).is_file() for key in ("visual_path", "audio_path")):
            raise ValueError(f"Candidate assets are missing: {row['sample_id']}")
        row["labels"] = [-1] * frames
        rows.append(row)
    if not rows:
        raise ValueError("No real candidate windows are available")
    # Stratification is for processing coverage only; existing split membership
    # and group identity remain untouched. It cannot create held-out evidence.
    selected, buckets = [], {split: [row for row in rows if row["split"] == split]
                             for split in ("train", "val", "test")}
    positions = Counter()
    while len(selected) < min(len(rows), limit):
        for split in ("train", "val", "test"):
            if positions[split] < len(buckets[split]) and len(selected) < limit:
                selected.append(buckets[split][positions[split]])
                positions[split] += 1
    return selected, len(rows)


def consensus_labels(ava, talkset, *, high=.95, low=.05, radius=2):
    """Threshold both teachers independently; mask transitions and clip edges."""
    if len(ava) != len(talkset) or not ava:
        raise ValueError("Teacher probabilities must have equal nonzero lengths")
    if any(not 0 <= value <= 1 for value in [*ava, *talkset]):
        raise ValueError("Nonfinite or out-of-range teacher probability")
    raw = [1 if a >= high and b >= high else 0 if a <= low and b <= low else -1
           for a, b in zip(ava, talkset)]
    labels = list(raw)
    mask = set(range(min(radius, len(raw)))) | set(range(max(0, len(raw) - radius), len(raw)))
    for index in range(1, len(raw)):
        if raw[index] != raw[index - 1]:
            mask.update(range(max(0, index - radius), min(len(raw), index + radius + 1)))
    for index in mask:
        labels[index] = -1
    return labels, raw


def teacher_inputs(row, manifest):
    selected = dict(row)
    selected["visual_path"] = row.get("teacher_visual_path") or row["visual_path"]
    # The loader validates uint8[T,112,112], exact audio start and 13-D MFCCs.
    audio, visual, _ = load_sample(selected, manifest)
    return audio, visual


def label_record(row, ava, talkset, args, teachers):
    labels, raw = consensus_labels(ava, talkset, high=args.positive_threshold,
                                   low=args.negative_threshold, radius=args.boundary_radius)
    result = dict(row)
    for field in ("visual_path", "audio_path", "teacher_visual_path"):
        if result.get(field):
            result[field] = str(resolve_asset(args.manifest, result[field]).resolve())
    valid = sum(value >= 0 for value in labels)
    result.update({"labels": labels, "machine_generated": True, "human_annotated": False,
                   "mask": [value >= 0 for value in labels],
                   "label_source": "ensemble_teacher_weak", "label_accepted": valid >= args.min_valid_frames,
                   "label_rejection_reason": None if valid >= args.min_valid_frames
                                             else f"fewer_than_{args.min_valid_frames}_consensus_frames",
                   "teacher_probabilities": {"ava": ava, "talkset": talkset,
                                             "ensemble_mean": [(a + b) / 2 for a, b in zip(ava, talkset)]},
                   "weak_label_metadata": {"schema_version": "lrasd-machine-consensus-v1",
                       "source": "frozen_public_LR-ASD_AVA_and_TalkSet_models", "teachers": teachers,
                       "teacher_crop_type": row.get("teacher_crop_type", "unspecified_external_crop"
                                                    if row.get("teacher_visual_path") else "full_body_bbox"),
                       "teacher_visual_path": str(resolve_asset(args.manifest,
                                                  row.get("teacher_visual_path") or row["visual_path"]).resolve()),
                       "probabilities_calibrated": False,
                       "positive_threshold": args.positive_threshold,
                       "negative_threshold": args.negative_threshold,
                       "decision": "both_teachers_must_cross_the_same_threshold",
                       "boundary_radius_frames": args.boundary_radius, "valid_frames": valid,
                       "raw_consensus_positive_frames": raw.count(1), "raw_consensus_negative_frames": raw.count(0),
                       "evaluation_scope": "teacher_agreement_only_not_human_accuracy"}})
    return result


def build_summary(records, config, total, elapsed):
    split_summaries, reasons = {}, []
    for split, minimum in (("train", 20), ("val", 5), ("test", 5)):
        rows = [row for row in records if row["split"] == split]
        accepted = [row for row in rows if row["label_accepted"]]
        counts = Counter(value for row in accepted for value in row["labels"])
        all_frames = [(a, b) for row in rows for a, b in zip(row["teacher_probabilities"]["ava"],
                                                           row["teacher_probabilities"]["talkset"])]
        n = len(all_frames)
        split_summaries[split] = {
            "candidates": len(rows), "accepted_windows": len(accepted), "minimum_windows": minimum,
            "positive_frames": counts[1], "negative_frames": counts[0], "masked_frames": counts[-1],
            "positive_frame_fraction": counts[1] / (counts[1] + counts[0]) if counts[1] + counts[0] else None,
            "pure_positive_windows": sum(0 not in row["labels"] for row in accepted),
            "pure_negative_windows": sum(1 not in row["labels"] for row in accepted),
            "groups": sorted({row.get("group", row["movie"]) for row in rows}),
            "baseline_ensemble_vs_single": {
                "frame_count": n,
                "ava_high_positive_frames": sum(a >= config.get("positive_threshold", .95) for a, _ in all_frames),
                "talkset_high_positive_frames": sum(b >= config.get("positive_threshold", .95) for _, b in all_frames),
                "ava_high_negative_frames": sum(a <= config.get("negative_threshold", .05) for a, _ in all_frames),
                "talkset_high_negative_frames": sum(b <= config.get("negative_threshold", .05) for _, b in all_frames),
                "teacher_binary_agreement_at_0_5": sum((a >= .5) == (b >= .5) for a, b in all_frames) / n if n else None,
                "mean_absolute_teacher_probability_difference": sum(abs(a - b) for a, b in all_frames) / n if n else None,
                "ensemble_changed_vs_ava_at_0_5": sum(((a + b) / 2 >= .5) != (a >= .5) for a, b in all_frames) / n if n else None,
                "ensemble_changed_vs_talkset_at_0_5": sum(((a + b) / 2 >= .5) != (b >= .5) for a, b in all_frames) / n if n else None,
                "interpretation": "prediction_disagreement_not_accuracy"}}
        if len(accepted) < minimum:
            reasons.append(f"{split}: {len(accepted)} accepted windows, requires >= {minimum}")
        if not counts[1] or not counts[0]:
            reasons.append(f"{split}: both positive and negative machine-labeled frames are required")
    return {"schema_version": "lrasd-machine-label-summary-v1", "training_ready": not reasons,
            "refusal_reasons": reasons, "human_ground_truth": False,
            "evaluation_interpretation": "Weak-label heldout metrics measure teacher agreement within heldout time blocks; "
                                         "they do not establish true active-speaker accuracy or cross-movie generalization.",
            "training_class_balance": "Training runner must balance unmasked frame classes; pure-class windows remain explicit.",
            "config": config, "total_input_candidates": total, "processed_candidates": len(records),
            "accepted_candidates": sum(row["label_accepted"] for row in records),
            "elapsed_seconds": elapsed, "splits": split_summaries}


def run(args):
    import numpy as np
    import torch
    args.output.mkdir(parents=True, exist_ok=True)
    rows, total = candidate_rows(args.manifest, args.max_candidates, frames=args.frames)
    teachers = {name: {"path": str(path.resolve()), "sha256": sha256_file(path), "architecture": "lr-asd-original-168-v1"}
                for name, path in (("ava", args.ava_checkpoint), ("talkset", args.talkset_checkpoint))}
    config = {"manifest_sha256": sha256_file(args.manifest), "teachers": teachers,
              "candidate_ids_sha256": canonical_hash([row["sample_id"] for row in rows]),
              "positive_threshold": args.positive_threshold, "negative_threshold": args.negative_threshold,
              "boundary_radius": args.boundary_radius, "min_valid_frames": args.min_valid_frames,
              "frames": args.frames, "device": args.device, "batch_size": args.batch_size,
              "label_script_sha256": sha256_file(Path(__file__)),
              "human_ground_truth": False, "teacher_score_domain": "active_speaker_probability"}
    config_path, all_path = args.output / "label-config.json", args.output / "candidate-probabilities.jsonl"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output belongs to another manifest/teacher/threshold configuration")
    atomic_json(config_path, config)
    records = []
    if all_path.exists():
        for line in all_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    done = {row["sample_id"] for row in records}
    if len(done) != len(records) or not done.issubset({row["sample_id"] for row in rows}):
        raise ValueError("Cached predictions are not a unique subset of this candidate selection")
    pending = [row for row in rows if row["sample_id"] not in done]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the independent LR-ASD environment")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = [load_lrasd_model(args.source_root, path, args.device)
              for path in (args.ava_checkpoint, args.talkset_checkpoint)]
    started = time.perf_counter()
    emit("weak_labeling_started", candidates=len(rows), cached=len(done), config=config)
    with all_path.open("a", encoding="utf-8") as output, torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            inputs = [teacher_inputs(row, args.manifest) for row in batch]
            audio = torch.from_numpy(np.stack([pair[0] for pair in inputs])).to(args.device)
            visual = torch.from_numpy(np.stack([pair[1] for pair in inputs])).to(args.device)
            predictions = []
            for model in models:
                av, _ = model.model(audio, visual)
                probability = torch.softmax(model.lossAV.FC(av), dim=-1)[:, 1].reshape(len(batch), args.frames)
                if not torch.isfinite(probability).all():
                    raise ValueError("A teacher returned nonfinite probabilities")
                predictions.append(probability.cpu().tolist())
            for index, row in enumerate(batch):
                record = label_record(row, predictions[0][index], predictions[1][index], args, teachers)
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                records.append(record)
            output.flush()
            elapsed = time.perf_counter() - started
            progress = {"status": "labeling", "processed": len(records), "candidates": len(rows),
                        "accepted": sum(row["label_accepted"] for row in records), "elapsed_seconds": elapsed,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0}
            atomic_json(args.output / "label-progress.json", progress)
            if start == 0 or len(records) % 40 < args.batch_size or len(records) == len(rows):
                emit("weak_labeling_progress", **progress)
    summary = build_summary(records, config, total, time.perf_counter() - started)
    summary["runtime"] = {"python_version": sys.version, "torch_version": torch.__version__,
                          "numpy_version": np.__version__, "cached_candidates": len(done),
                          "newly_processed_candidates": len(pending),
                          "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0}
    target = args.output / "labeled.jsonl"
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            if record["label_accepted"]:
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(target)
    atomic_json(args.output / "label-summary.json", summary)
    atomic_json(args.output / "label-progress.json", {"status": "ready_for_weak_training" if summary["training_ready"]
                                                     else "insufficient_labels_do_not_train",
               "processed": len(records), "accepted": summary["accepted_candidates"],
               "training_ready": summary["training_ready"], "refusal_reasons": summary["refusal_reasons"]})
    emit("weak_labeling_completed", training_ready=summary["training_ready"],
         accepted=summary["accepted_candidates"], summary=str(args.output / "label-summary.json"),
         refusal_reasons=summary["refusal_reasons"])
    return summary


if __name__ == "__main__":
    sys.exit(0 if run(parse_args())["training_ready"] else 2)
