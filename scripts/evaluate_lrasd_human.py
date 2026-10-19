"""Evaluate frozen LR-ASD bundles on the fixed human scene holdout.

Default: baseline validation only. --candidate-bundle compares validation.
--final-test additionally compares test once, with thresholds chosen on val.
There is no optimizer, training, application activation or test threshold tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall.lrasd_training_data import (  # noqa: E402
    HUMAN_LABEL_SOURCE, HUMAN_REPLAY_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE,
    atomic_json, classification_metrics, label_semantics, load_manifest,
    select_threshold, sha256_file, split_summary, supervision_summary,
    validation_selection_report,
)
from train_lrasd import build_model, inspect_parent_bundle, predict_models  # noqa: E402


def inspect_inputs(manifest, baseline_bundle, candidate_bundle=None, *, final_test=False, baseline_report=None):
    rows = load_manifest(manifest, frames=50, allow_within_show_group_split=True)
    mode = HUMAN_REPLAY_LABEL_SOURCE if any(row.get("label_source") == OFFICIAL_LABEL_SOURCE for row in rows) else HUMAN_LABEL_SOURCE
    supervision = supervision_summary(rows, mode)
    if any(row.get("label_source") != HUMAN_LABEL_SOURCE for row in rows if row["split"] != "train"):
        raise ValueError("This evaluation accepts only human-reviewed ASD on val/test; official replay is train-only")
    summary = split_summary(rows)
    for split in ("val", "test"):
        if not summary[split]["positive_frames"] or not summary[split]["negative_frames"]:
            raise ValueError(f"{split} needs both reviewed ASD classes")
    baseline = inspect_parent_bundle(Path(baseline_bundle), Path(baseline_bundle) / "model.pt")
    candidate = inspect_parent_bundle(Path(candidate_bundle), Path(candidate_bundle) / "model.pt") if candidate_bundle else None
    manifest_sha = sha256_file(manifest)
    if candidate and candidate["manifest_sha256"] != manifest_sha:
        raise ValueError("Candidate was not trained with this exact frozen human round manifest")
    candidate_metadata = json.loads((Path(candidate_bundle) / "metadata.json").read_text(encoding="utf-8")) if candidate else None
    if candidate and candidate_metadata.get("training", {}).get("pretrained_sha256") != baseline["checkpoint_sha256"]:
        raise ValueError("Candidate initialization differs from the frozen baseline checkpoint")
    if final_test:
        if not candidate or not baseline_report:
            raise ValueError("Final test requires a candidate bundle and its earlier baseline-validation report")
        if candidate_metadata.get("heldout_test", {}).get("test_inference_performed") is not False:
            raise ValueError("Candidate must defer its training-run final test before this fixed comparison")
        report = json.loads(Path(baseline_report).read_text(encoding="utf-8"))
        if (report.get("manifest_sha256") != manifest_sha
                or report.get("baseline", {}).get("checkpoint_sha256") != baseline["checkpoint_sha256"]
                or report.get("test_inference_performed") is not False):
            raise ValueError("Earlier baseline report differs from this checkpoint/manifest or already consumed test")
    return rows, supervision, baseline, candidate


def evaluate_frozen(torch, models, rows, args, *, final_test=False):
    """Never even pass test rows to the predictor before explicit final comparison."""
    validation = [row for row in rows if row["split"] == "val"]
    labels, predictions = predict_models(torch, models, validation, args)
    results = []
    for scores in predictions:
        threshold = select_threshold(labels, scores)
        report = validation_selection_report(validation, labels, scores, threshold)
        results.append({"validation": report["metrics"], "validation_by_label_source": {
            source: {**metrics, "semantics": label_semantics(source)} for source, metrics in report["by_label_source"].items()},
            "threshold": threshold, "threshold_selected_on": "val"})
    if final_test:
        testing = [row for row in rows if row["split"] == "test"]
        labels, predictions = predict_models(torch, models, testing, args)
        for result, scores in zip(results, predictions):
            result["test"] = classification_metrics(labels, scores, result["threshold"])
            result["test_semantics"] = label_semantics(HUMAN_LABEL_SOURCE)
    return results


def run(args):
    if args.output.exists():
        raise ValueError("Evaluation output already exists; preserve the prior report and use a new path")
    rows, supervision, baseline, candidate = inspect_inputs(args.manifest, args.baseline_bundle, args.candidate_bundle,
                                                            final_test=args.final_test, baseline_report=args.baseline_report)
    # Import the heavyweight runtime only after provenance/split checks pass.
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable in this evaluation runtime")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    bundles = [args.baseline_bundle] + ([args.candidate_bundle] if candidate else [])
    models = [build_model(torch, args.source_root, bundle / "model.pt", args.device) for bundle in bundles]
    loader_args = SimpleNamespace(manifest=args.manifest, seed=114, batch_size=args.batch_size, workers=0, device=args.device)
    metrics = evaluate_frozen(torch, models, rows, loader_args, final_test=args.final_test)
    report = {"schema_version": "scenerecall-human-asd-evaluation-v1", "manifest_sha256": sha256_file(args.manifest),
              "annotation_plan_sha256": next(row["annotation_plan"]["sha256"] for row in rows if row.get("annotation_plan")),
              "validation_scope": "within_title_scene_holdout", "cross_title_validation": False,
              "baseline_prior_film_exposure": "previous_weak_adaptation_not_pristine",
              "threshold_selection": "each_frozen_model_validation_only", "masked_frames_in_metrics": False,
              "test_inference_performed": args.final_test, "split_summary": split_summary(rows), "supervision": supervision,
              "baseline": {**baseline, **metrics[0]}, "starts_training": False, "activates_model": False,
              "limitation": "Same-film sampled scenes only. Prior weak-label exposure of the baseline is not ruled out. Test must not be used for repeated tuning or model/threshold selection."}
    if candidate:
        report["candidate"] = {**candidate, **metrics[1]}
        report["comparison"] = {f"{split}_ap_delta": metrics[1][split]["ap"] - metrics[0][split]["ap"]
                                for split in ("validation", "test") if split in metrics[0]}
        report["automatic_deployment_approved"] = False
    atomic_json(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/d/SceneRecall-LR-ASD"))
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size * 50 > 400 or args.cpu_threads < 1:
        parser.error("Positive counts required; evaluation batch capped at 400 visual frames")
    try:
        report = run(args)
    except (ValueError, OSError, ImportError) as exc:
        print(f"Human evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"report": str(args.output), "test_inference_performed": report["test_inference_performed"],
                      "baseline_validation_ap": report["baseline"]["validation"]["ap"],
                      "comparison": report.get("comparison")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
