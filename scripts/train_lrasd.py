#!/usr/bin/env python3
"""Train original LR-ASD on real, grouped, explicitly labelled character windows.

Run with the independent LR-ASD Python environment, not the app's audio venv.
No dataset downloads, synthetic data generation, or application activation occur.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall.lrasd_training_data import (  # noqa: E402
    ARCHITECTURE, BUNDLE_SCHEMA, PREPROCESSING, SCORE_DOMAIN, atomic_json,
    canonical_hash, classification_metrics, load_manifest, load_sample,
    select_threshold, sha256_file, split_summary, supervision_summary,
    TEACHER_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE, MIXED_LABEL_SOURCE,
    HUMAN_LABEL_SOURCE, HUMAN_REPLAY_LABEL_SOURCE, label_semantics, validation_selection_report,
)


class ManifestDataset:
    def __init__(self, rows, manifest, *, train=False, seed=114, epoch=0):
        self.rows, self.manifest = rows, manifest
        self.train, self.seed, self.epoch = train, seed, epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return load_sample(self.rows[index], self.manifest, train=self.train, seed=self.seed, epoch=self.epoch)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/d/SceneRecall-LR-ASD"))
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--stage-name", default="single_stage")
    parser.add_argument("--parent-bundle", type=Path,
                        help="Completed prior-stage bundle; initialize its model with a fresh optimizer, not --resume")
    parser.add_argument("--defer-final-test", action="store_true",
                        help="Export a validated intermediate parent bundle without test inference")
    parser.add_argument("--previously-evaluated-test-source", action="append", default=[],
                        choices=(TEACHER_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE, HUMAN_LABEL_SOURCE),
                        help="Declare historical evaluation of this test-label source; never describe it as pristine")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, choices=(50, 100), default=100,
                        help="Fixed window length for this run; supports T=50 or T=100 without changing the architecture")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="Save resumable state and stop; do not export a completed bundle")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dataset-name", default="CMD-AM")
    parser.add_argument("--dataset-version", default="2025-weak-speaker-alignment-v1")
    parser.add_argument("--source-domain", default="English animated feature films; full-body boxes and sentence speaker labels")
    parser.add_argument("--target-domain", default="Japanese 2D anime series; not validated by this source dataset")
    parser.add_argument("--label-source", choices=(TEACHER_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE, MIXED_LABEL_SOURCE,
                                                 HUMAN_LABEL_SOURCE, HUMAN_REPLAY_LABEL_SOURCE),
                        default=TEACHER_LABEL_SOURCE)
    parser.add_argument("--class-balance", choices=("inverse-frequency", "none"), default="inverse-frequency")
    parser.add_argument("--batchnorm-mode", choices=("update", "frozen"), default="update",
                        help="Update BatchNorm running statistics or retain the initialization's statistics")
    parser.add_argument("--trainable-scope", choices=("all", "heads"), default="all",
                        help="Optimize all parameters or only lossAV.FC and lossV.FC classification heads")
    parser.add_argument("--validation-selection", choices=("pooled_ap", "macro_source_ap"), default="pooled_ap",
                        help="Choose checkpoints by pooled AP or equal-weight mean AP across validation label sources")
    parser.add_argument("--allow-within-show-group-split", "--allow-within-movie-block-split",
                        dest="allow_within_show_group_split", action="store_true",
                        help="Explicitly permit held-out episodes from one show; never claim cross-title generalization")
    parser.add_argument("--split-strategy", choices=("movie_holdout", "episode_holdout_same_show", "scene_holdout_same_title"), default=None)
    parser.add_argument("--max-gpu-temperature", type=float, default=88)
    parser.add_argument("--resume-gpu-temperature", type=float, default=84)
    parser.add_argument("--cooldown-seconds", type=float, default=10)
    parser.add_argument("--max-cooldown-checks", type=int, default=6)
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size, args.frames, args.cpu_threads,
           args.checkpoint_every, args.log_every) < 1 or args.workers < 0 or args.max_steps < 0:
        parser.error("Counts must be positive, with nonnegative workers/max-steps")
    if args.batch_size * args.frames > 400:
        parser.error("FP32 training is capped at 400 visual frames per step for this 8 GB GPU")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.lr_decay) or not 0 < args.lr_decay <= 1:
        parser.error("Invalid learning rate or decay")
    if (not all(math.isfinite(value) for value in (args.max_gpu_temperature, args.resume_gpu_temperature, args.cooldown_seconds))
            or not 0 < args.resume_gpu_temperature < args.max_gpu_temperature <= 100
            or not 0 < args.cooldown_seconds <= 30 or args.max_cooldown_checks < 1):
        parser.error("Invalid thermal guard settings")
    if args.split_strategy in {"episode_holdout_same_show", "scene_holdout_same_title"} and not args.allow_within_show_group_split:
        parser.error("Episode/scene holdout requires --allow-within-show-group-split")
    args.split_strategy = args.split_strategy or ("episode_holdout_same_show" if args.allow_within_show_group_split else "movie_holdout")
    args.stage_name = args.stage_name.strip()
    if not args.stage_name:
        parser.error("stage-name must be nonempty")
    args.pretrained = args.pretrained or (args.parent_bundle / "model.pt" if args.parent_bundle
                                         else args.source_root / "weight" / "pretrain_AVA.model")
    return args


def inspect_parent_bundle(directory, pretrained):
    if directory is None:
        return None
    directory = directory.resolve()
    metadata_path, checkpoint = directory / "metadata.json", directory / "model.pt"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != BUNDLE_SCHEMA or metadata.get("architecture") != ARCHITECTURE
            or metadata.get("preprocessing") != PREPROCESSING or metadata.get("checkpoint") != "model.pt"
            or metadata.get("training", {}).get("status") != "completed"):
        raise ValueError("Parent bundle must be a completed compatible LR-ASD export")
    checkpoint_sha = sha256_file(checkpoint)
    if metadata.get("checkpoint_sha256") != checkpoint_sha or sha256_file(pretrained) != checkpoint_sha:
        raise ValueError("Parent bundle checkpoint SHA does not match metadata or --pretrained")
    dataset = metadata.get("dataset", {})
    return {"bundle_path": str(directory), "metadata_sha256": sha256_file(metadata_path),
            "checkpoint_sha256": checkpoint_sha, "stage_name": metadata["training"].get("stage_name", "unspecified_legacy_stage"),
            "manifest_sha256": dataset.get("split_manifest_sha256"), "label_source": dataset.get("label_source"),
            "split_groups": dataset.get("split_groups", {}), "split_clips": dataset.get("split_clips", {})}


def validate_parent_holdout_isolation(rows, parent):
    if parent is None:
        return "not_applicable"
    groups = parent.get("split_groups", {})
    clips = parent.get("split_clips", {})
    if not groups and not clips:
        return "unavailable_for_legacy_parent_metadata"
    heldout_groups = set(groups.get("val", [])) | set(groups.get("test", []))
    heldout_clips = {tuple(item) for split in ("val", "test") for item in clips.get(split, [])}
    for row in rows:
        group = row.get("evaluation_group", row.get("group", row["movie"]))
        if row["split"] == "train" and (group in heldout_groups or (row["movie"], row["clip"]) in heldout_clips):
            raise ValueError("Prior-stage held-out group/clip cannot enter next-stage training replay")
    return "prior_stage_validation_and_test_excluded_from_current_training"


def build_model(torch, source_root, pretrained, device):
    """Use upstream architecture and head names so all 168 checkpoint keys match."""
    sys.path.insert(0, str(source_root.resolve()))
    original_model = importlib.import_module("model.Model").ASD_Model
    losses = importlib.import_module("loss")

    class OriginalLRASD(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = original_model()
            self.lossAV = losses.lossAV()
            self.lossV = losses.lossV()

    model = OriginalLRASD().to(device)
    initial = torch.load(pretrained, map_location="cpu", weights_only=True)
    if not isinstance(initial, dict) or len(initial) != 168:
        raise ValueError("Initialization needs a raw 168-key model state_dict (bundle/model.pt), not best.pt/latest.pt resume state")
    model.load_state_dict(initial, strict=True)
    if len(model.state_dict()) != 168:
        raise ValueError("Expected the original 168-key LR-ASD architecture")
    return model


def training_control_config(args):
    """Hash nondefault controls while preserving legacy update/all resume hashes."""
    controls = {}
    for name, default, allowed in (("batchnorm_mode", "update", {"update", "frozen"}),
                                    ("trainable_scope", "all", {"all", "heads"})):
        value = getattr(args, name)
        if value not in allowed:
            raise ValueError(f"Invalid {name}: {value}")
        if value != default:
            controls[name] = value
    return controls


def configure_trainable_scope(model, scope):
    """Freeze parameters independently of BatchNorm's running-buffer policy."""
    if scope not in {"all", "heads"}:
        raise ValueError(f"Invalid trainable_scope: {scope}")
    total = trainable = 0
    for name, parameter in model.named_parameters():
        enabled = scope == "all" or name.startswith(("lossAV.FC.", "lossV.FC."))
        parameter.requires_grad_(enabled)
        total += parameter.numel()
        if enabled:
            trainable += parameter.numel()
    if not trainable:
        raise ValueError("Selected trainable scope contains no parameters")
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def set_training_mode(torch, model, batchnorm_mode):
    """Reapply the buffer policy after every transition back from evaluation."""
    if batchnorm_mode not in {"update", "frozen"}:
        raise ValueError(f"Invalid batchnorm_mode: {batchnorm_mode}")
    model.train()
    if batchnorm_mode == "frozen":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def make_loader(torch, rows, args, *, train=False, epoch=0):
    dataset = ManifestDataset(rows, args.manifest, train=train, seed=args.seed, epoch=epoch)
    # DataLoader creates a base seed even with zero workers. A private generator
    # prevents restarting an iterator from advancing the model/dropout RNG.
    generator = torch.Generator().manual_seed(args.seed + epoch + (0 if train else 1_000_000))
    options = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                   pin_memory=args.device == "cuda", drop_last=False, generator=generator)
    if args.workers:
        options["prefetch_factor"] = 1
    return torch.utils.data.DataLoader(dataset, **options)


def predict_models(torch, models, rows, args):
    """Evaluate frozen models in one data pass; masked frames never enter metrics."""
    for model in models:
        model.eval()
    labels, score_sets = [], [[] for _ in models]
    with torch.inference_mode():
        for audio, visual, target in make_loader(torch, rows, args):
            audio = audio.to(args.device, non_blocking=True)
            visual = visual.to(args.device, non_blocking=True)
            target = target.to(args.device).reshape(-1)
            mask = target >= 0
            labels.extend(target[mask].cpu().tolist())
            for index, model in enumerate(models):
                av, _ = model.model(audio, visual)
                logits = model.lossAV.FC(av)
                scores = torch.softmax(logits, dim=-1)[:, 1][mask]
                if not torch.isfinite(scores).all():
                    raise ValueError("Nonfinite validation probability")
                score_sets[index].extend(scores.cpu().tolist())
    return labels, score_sets


def evaluate_final_heldout(torch, model, rows, args, *, threshold, baseline_validation,
                           best_sha, supervision, progress_callback):
    """An intermediate stage returns before constructing a model or loading test assets."""
    sources = set(supervision["splits"]["test"])
    historical = set(args.previously_evaluated_test_source)
    if historical - sources:
        raise ValueError("Previously evaluated test sources must occur in this test split")
    policy = {source: {"history": "previously_evaluated" if source in historical else "history_not_declared",
                       "pristine_claimed": False} for source in sorted(sources)}
    common = {"best_checkpoint_sha256": best_sha,
              "label_semantics": supervision["split_semantics"]["test"],
              "label_sources": supervision["splits"]["test"], "evaluation_history": policy}
    if args.defer_final_test:
        return {**common, "status": "deferred_not_evaluated", "test_inference_performed": False,
                "artifact_purpose": "intermediate_stage_initialization"}
    baseline = build_model(torch, args.source_root, args.pretrained, args.device)
    progress_callback("final_heldout_test")
    labels, predictions = predict_models(torch, [model, baseline], rows, args)
    indices, position = {source: [] for source in sources}, 0
    for row in rows:
        count = sum(label >= 0 for label in row["labels"])
        indices[row["label_source"]].extend(range(position, position + count))
        position += count
    if position != len(labels):
        raise ValueError("Test provenance does not align with unmasked prediction frames")
    by_source = {}
    for source, selected in indices.items():
        targets = [labels[index] for index in selected]
        by_source[source] = {
            "trained": classification_metrics(targets, [predictions[0][index] for index in selected], threshold),
            "baseline": classification_metrics(targets, [predictions[1][index] for index in selected], baseline_validation["threshold"]),
            "evaluation_history": policy[source],
            "label_semantics": label_semantics(source)}
    return {**common, "status": "evaluated", "test_inference_performed": True,
            "trained": classification_metrics(labels, predictions[0], threshold),
            "baseline": classification_metrics(labels, predictions[1], baseline_validation["threshold"]),
            "by_label_source": by_source}


def pack_rng(torch, np):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"kind": state[0], "keys": state[1].tolist(),
            "position": state[2], "has_gauss": state[3], "cached_gauss": state[4]},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(torch, np, state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns["kind"], np.asarray(ns["keys"], dtype=np.uint32), ns["position"],
                         ns["has_gauss"], ns["cached_gauss"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_checkpoint(torch, path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_resume_checkpoint(state, config_hash):
    """Check provenance/progress before any model, optimizer, or RNG is restored."""
    if (not isinstance(state, dict) or state.get("schema_version") != "scenerecall-lrasd-resume-v1"
            or state.get("config_hash") != config_hash or not isinstance(state.get("config"), dict)
            or canonical_hash(state["config"]) != config_hash):
        raise ValueError("Resume state is incompatible with this manifest, initialization, or training configuration")
    for name in ("model", "optimizer", "scheduler", "rng"):
        if not isinstance(state.get(name), dict):
            raise ValueError(f"Resume checkpoint has no valid {name} state")
    if len(state["model"]) != 168:
        raise ValueError("Resume checkpoint must retain all original 168 model keys")
    for name in ("epoch", "batch_offset", "steps", "epochs_completed", "best_epoch", "stale_epochs"):
        if type(state.get(name)) is not int or state[name] < 0:
            raise ValueError(f"Invalid checkpoint progress: {name}")
    if state["epoch"] != state["epochs_completed"] or state["best_epoch"] > state["epochs_completed"]:
        raise ValueError("Checkpoint epoch and best-selection progress disagree")
    if (type(state.get("best_ap")) not in (float, int) or not math.isfinite(state["best_ap"])
            or not -1 <= state["best_ap"] <= 1):
        raise ValueError("Invalid checkpoint best AP")
    metric = state.get("best_selection_metric", "pooled_ap")
    value = state.get("best_selection_value", state["best_ap"])
    if (metric not in {"pooled_ap", "macro_source_ap"}
            or metric != state["config"].get("validation_selection", "pooled_ap")
            or type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1
            or (metric == "pooled_ap" and abs(value - state["best_ap"]) > 1e-12)):
        raise ValueError("Invalid checkpoint validation selection metric/value")
    if state["steps"] and not isinstance(state.get("baseline_validation"), dict):
        raise ValueError("Trained checkpoint is missing its original pretrained baseline validation")
    return state


def validate_best_checkpoint(state, config_hash, *, expected_epoch=None, expected_ap=None,
                             expected_selection_value=None):
    validate_resume_checkpoint(state, config_hash)
    if (not state["best_epoch"] or state["best_epoch"] != state["epochs_completed"]
            or state["batch_offset"] != 0 or state["stale_epochs"] != 0):
        raise ValueError("best.pt is not a completed, validation-selected epoch")
    if ((expected_epoch is not None and state["best_epoch"] != expected_epoch)
            or (expected_ap is not None and abs(state["best_ap"] - expected_ap) > 1e-12)
            or (expected_selection_value is not None and abs(
                state.get("best_selection_value", state["best_ap"]) - expected_selection_value) > 1e-12)):
        raise ValueError("best.pt does not match the validation selection in the resumed checkpoint")
    return state


def emit(event, **fields):
    print(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
                     allow_nan=False), flush=True)


def gpu_temperature():
    """Read telemetry only; never modify clocks, power, fans, or driver settings."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", "--id=0"],
                                check=False, capture_output=True, text=True, timeout=5)
        value = float(result.stdout.strip().splitlines()[0]) if result.returncode == 0 else None
        return value if value is not None and math.isfinite(value) else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def process_peak_rss_bytes(status_path=None):
    """Linux process high-water RSS; unavailable platforms report null."""
    try:
        path = Path(status_path) if status_path is not None else Path("/proc/self/status")
        for line in path.read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) == 3 and parts[2].lower() == "kb":
                    value = int(parts[1])
                    return value * 1024 if value >= 0 else None
    except (OSError, UnicodeError, ValueError):
        pass
    return None


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import torch.nn.functional as functional

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "bundle" / "metadata.json").exists():
        raise ValueError("Completed bundle already exists; use a new output directory for another experiment")
    rows = load_manifest(args.manifest, frames=args.frames,
                         allow_within_show_group_split=args.allow_within_show_group_split)
    if any(row.get("synthetic") is True for row in rows):
        raise ValueError("Synthetic windows are not accepted by this real-data training runner")
    supervision = supervision_summary(rows, args.label_source)
    if HUMAN_LABEL_SOURCE in supervision["sources"]:
        if args.split_strategy != "scene_holdout_same_title":
            raise ValueError("This human round requires explicit scene_holdout_same_title split strategy")
        # Official data is replay for training only. Human val/test remain the
        # fixed target-domain checkpoint-selection and evaluation sets.
        if any(row["label_source"] != HUMAN_LABEL_SOURCE and row["split"] != "train" for row in rows):
            raise ValueError("Human round accepts official replay only in train; val/test must remain human ASD")
    if args.validation_selection == "macro_source_ap" and any(
            not counts["positive_frames"] or not counts["negative_frames"]
            for counts in supervision["splits"]["val"].values()):
        raise ValueError("macro_source_ap requires both validation classes for every label source")
    if set(args.previously_evaluated_test_source) - set(supervision["splits"]["test"]):
        raise ValueError("Previously evaluated test sources must occur in this test split")
    parent = inspect_parent_bundle(args.parent_bundle, args.pretrained)
    parent_holdout_guard = validate_parent_holdout_isolation(rows, parent)
    summary = split_summary(rows)
    for name, minimum in (("train", 20), ("val", 5), ("test", 5)):
        if summary[name]["windows"] < minimum:
            raise ValueError(f"{name} needs at least {minimum} usable windows; received {summary[name]['windows']}")
    label_summary_path = args.manifest.parent / "label-summary.json"
    if args.label_source == TEACHER_LABEL_SOURCE and label_summary_path.exists():
        label_summary = json.loads(label_summary_path.read_text(encoding="utf-8"))
        if label_summary.get("training_ready") is not True:
            raise ValueError("The teacher-label quality gate did not mark this dataset training_ready")
    if HUMAN_LABEL_SOURCE in supervision["sources"]:
        if not label_summary_path.exists():
            raise ValueError("Human round requires an exported label-summary.json readiness gate")
        label_summary = json.loads(label_summary_path.read_text(encoding="utf-8"))
        if (label_summary.get("training_ready") is not True
                or label_summary.get("labeled_manifest_sha256") != sha256_file(args.manifest)):
            raise ValueError("Human review/scene-split readiness gate failed or manifest changed")
    for name, split in summary.items():
        if not split["positive_frames"] or not split["negative_frames"]:
            raise ValueError(f"{name} must contain both positive and negative unmasked frames")
    split_rows = {name: [row for row in rows if row["split"] == name] for name in summary}
    manifest_sha = sha256_file(args.manifest)
    config = {"manifest_sha256": manifest_sha, "pretrained_sha256": sha256_file(args.pretrained),
              "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING, "batch_size": args.batch_size,
              "frames": args.frames, "lr": args.lr, "lr_decay": args.lr_decay, "seed": args.seed,
              "dataset_name": args.dataset_name, "dataset_version": args.dataset_version,
              "source_domain": args.source_domain, "target_domain": args.target_domain}
    config.update(label_source=args.label_source, class_balance=args.class_balance,
                  split_strategy=args.split_strategy, allow_within_show_group_split=args.allow_within_show_group_split,
                  patience=args.patience, gradient_clip_norm=5.0,
                  stage_name=args.stage_name, parent_bundle=parent, supervision=supervision,
                  parent_holdout_guard=parent_holdout_guard,
                  defer_final_test=args.defer_final_test,
                  previously_evaluated_test_sources=sorted(set(args.previously_evaluated_test_source)))
    train_supervision = supervision["splits"]["train"]
    forgetting_control = {
        "method": ("official_train_replay_with_target_human_labels" if HUMAN_LABEL_SOURCE in train_supervision
                   else "official_train_replay_with_target_teacher_labels") if parent and OFFICIAL_LABEL_SOURCE in train_supervision
                   and any(source in train_supervision for source in (HUMAN_LABEL_SOURCE, TEACHER_LABEL_SOURCE)) else "none",
        "train_windows_by_label_source": {source: counts["windows"] for source, counts in train_supervision.items()},
        "distillation_or_parameter_anchor": False, "source_domain_forgetting_evaluated": False,
        "limitation": "Replay is recorded as a training method; it is not evidence that source-domain performance was preserved."}
    config["forgetting_control"] = forgetting_control
    # Keep legacy pooled-AP checkpoint hashes compatible; absence means pooled_ap.
    if args.validation_selection != "pooled_ap":
        config["validation_selection"] = args.validation_selection
    config.update(training_control_config(args))
    config_hash = canonical_hash(config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected LR-ASD environment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_model(torch, args.source_root, args.pretrained, args.device)
    training_controls = {"batchnorm_mode": args.batchnorm_mode, "trainable_scope": args.trainable_scope,
                         "parameter_counts": configure_trainable_scope(model, args.trainable_scope)}
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                 lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    class_weights = None
    if args.class_balance == "inverse-frequency":
        positives = summary["train"]["positive_frames"]
        negatives = summary["train"]["negative_frames"]
        total = positives + negatives
        class_weights = torch.tensor([total / (2 * negatives), total / (2 * positives)],
                                     device=args.device, dtype=torch.float32)
    epoch, batch_offset, steps, epochs_completed = 0, 0, 0, 0
    best_ap, stale, best_epoch = -1.0, 0, 0
    best_selection_value = -1.0
    baseline_validation = None
    elapsed_prior = 0.0
    started = time.perf_counter()
    latest_path, best_path = args.output / "latest.pt", args.output / "best.pt"
    stopping = {"requested": False}

    def request_stop(signum, frame):
        stopping["requested"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    if args.resume:
        state = validate_resume_checkpoint(torch.load(args.resume, map_location="cpu", weights_only=True), config_hash)
        if best_path.exists():
            saved_best = validate_best_checkpoint(torch.load(best_path, map_location="cpu", weights_only=True), config_hash)
            # best.pt and latest.pt are each atomic. Recover the small crash
            # interval after writing a newer best but before updating latest.
            if args.resume.resolve() == latest_path.resolve() and saved_best["steps"] > state["steps"]:
                state = saved_best
                emit("recovered_newer_best_checkpoint", checkpoint=str(best_path), steps=state["steps"])
            if state["best_epoch"]:
                validate_best_checkpoint(saved_best, config_hash, expected_epoch=state["best_epoch"],
                                         expected_ap=state["best_ap"],
                                         expected_selection_value=state.get("best_selection_value", state["best_ap"]))
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch, batch_offset, steps = state["epoch"], state["batch_offset"], state["steps"]
        epochs_completed, best_ap = state["epochs_completed"], state["best_ap"]
        best_selection_value = state.get("best_selection_value", state["best_ap"])
        stale, best_epoch = state["stale_epochs"], state["best_epoch"]
        baseline_validation = state["baseline_validation"]
        elapsed_prior = state["elapsed_seconds"]
        restore_rng(torch, np, state["rng"])
        if best_epoch and not best_path.exists():
            raise ValueError("Resume requires the best.pt artifact from the same run")
    elif latest_path.exists() or best_path.exists():
        raise ValueError("Existing training state requires --resume; refusing to overwrite it")

    def save_state(path):
        atomic_checkpoint(torch, path, {
            "schema_version": "scenerecall-lrasd-resume-v1", "config": config, "config_hash": config_hash,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": pack_rng(torch, np), "epoch": epoch, "batch_offset": batch_offset, "steps": steps,
            "epochs_completed": epochs_completed, "best_ap": best_ap, "best_epoch": best_epoch,
            "best_selection_metric": args.validation_selection, "best_selection_value": best_selection_value,
            "stale_epochs": stale, "baseline_validation": baseline_validation,
            "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
        })

    def progress(status, **extra):
        atomic_json(args.output / "progress.json", {"status": status, "epoch": epoch + 1,
                    "epochs_completed": epochs_completed, "steps": steps, "best_epoch": best_epoch,
                    "best_validation_ap": best_ap if best_ap >= 0 else None, "config_hash": config_hash,
                    "best_selection_metric": args.validation_selection,
                    "best_selection_value": best_selection_value if best_selection_value >= 0 else None,
                    "elapsed_seconds": elapsed_prior + time.perf_counter() - started, **extra})

    def thermal_guard():
        if args.device != "cuda":
            return True
        temperature = gpu_temperature()
        if temperature is None:
            emit("thermal_telemetry_unavailable")
            return True
        if temperature < args.max_gpu_temperature:
            return True
        save_state(latest_path)
        for attempt in range(args.max_cooldown_checks):
            progress("thermal_cooldown", gpu_temperature_c=temperature, cooldown_check=attempt + 1,
                     cooldown_seconds=args.cooldown_seconds)
            emit("thermal_cooldown", temperature_c=temperature, check=attempt + 1,
                 resume_below_c=args.resume_gpu_temperature)
            time.sleep(args.cooldown_seconds)
            temperature = gpu_temperature()
            if stopping["requested"]:
                progress("stopped_resumable", reason="stop_requested_during_cooldown")
                return False
            if temperature is not None and temperature <= args.resume_gpu_temperature:
                progress("training", gpu_temperature_c=temperature)
                emit("thermal_resumed", temperature_c=temperature)
                return True
        progress("thermal_paused_resumable", gpu_temperature_c=temperature, checkpoint=str(latest_path))
        emit("thermal_paused_resumable", temperature_c=temperature, checkpoint=str(latest_path))
        return False

    atomic_json(args.output / "run_config.json", {**config, **training_controls, "max_epochs": args.epochs,
                "validation_selection": args.validation_selection,
                "validation_sources": supervision["splits"]["val"],
                "per_source_validation_metrics": "history.jsonl validation_by_label_source; bundle/metadata.json validation.by_label_source",
                "patience": args.patience, "split_summary": summary, "precision": "float32",
                "optimizer": "AdamW", "weight_decay": 0.01, "loss": "CE(audio_visual) + 0.5 * CE(visual)",
                "class_balance": args.class_balance,
                "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
                "thermal_guard": {"max_temperature_c": args.max_gpu_temperature,
                                  "resume_temperature_c": args.resume_gpu_temperature,
                                  "cooldown_seconds": args.cooldown_seconds,
                                  "max_cooldown_checks": args.max_cooldown_checks},
                "visual_canvas": {"width": 480, "height": 270, "resize": "ffmpeg-bilinear"}})
    try:
        if baseline_validation is None:
            progress("baseline_validation")
            labels, predictions = predict_models(torch, [model], split_rows["val"], args)
            threshold = select_threshold(labels, predictions[0])
            report = validation_selection_report(split_rows["val"], labels, predictions[0], threshold, args.validation_selection)
            baseline_validation = {**report["metrics"], "by_label_source": report["by_label_source"],
                                   "selection_metric": report["selection_metric"], "selection_value": report["selection_value"]}
            atomic_json(args.output / "baseline_validation.json", baseline_validation)
            save_state(latest_path)
            emit("baseline_validation", metrics=baseline_validation)
        progress("training")
        emit("training_started", resumed=bool(args.resume), config=config, split_summary=summary,
             training_controls=training_controls)
        if not thermal_guard():
            return
        if stopping["requested"] or (args.max_steps and steps >= args.max_steps):
            save_state(latest_path)
            progress("stopped_resumable")
            return
        while epoch < args.epochs and stale < args.patience:
            order = list(range(len(split_rows["train"])))
            random.Random(args.seed + epoch).shuffle(order)
            remaining = [split_rows["train"][i] for i in order[batch_offset * args.batch_size:]]
            set_training_mode(torch, model, args.batchnorm_mode)
            recent_started, recent_frames, recent_steps = time.perf_counter(), 0, 0
            epoch_losses = []
            for audio, visual, target in make_loader(torch, remaining, args, train=True, epoch=epoch):
                audio = audio.to(args.device, non_blocking=True)
                visual = visual.to(args.device, non_blocking=True)
                target = target.to(args.device).reshape(-1)
                mask = target >= 0
                optimizer.zero_grad(set_to_none=True)
                av, video = model.model(audio, visual)
                # Divide by frame count rather than weight sum, so rare-class-only
                # batches retain their intended inverse-frequency contribution.
                denominator = mask.sum()
                loss_av = functional.cross_entropy(model.lossAV.FC(av)[mask], target[mask],
                                                   weight=class_weights, reduction="sum") / denominator
                loss_v = functional.cross_entropy(model.lossV.FC(video)[mask], target[mask],
                                                  weight=class_weights, reduction="sum") / denominator
                loss = loss_av + 0.5 * loss_v
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss; latest valid checkpoint remains available")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                if not math.isfinite(float(norm)):
                    raise ValueError("Nonfinite gradient norm")
                epoch_losses.append(loss_value)
                steps += 1
                batch_offset += 1
                recent_steps += 1
                recent_frames += int(visual.shape[0] * visual.shape[1])
                if steps % args.log_every == 0:
                    duration = time.perf_counter() - recent_started
                    emit("training_step", epoch=epoch + 1, step=steps, batch=batch_offset, loss=loss_value,
                         lr=optimizer.param_groups[0]["lr"], seconds_per_step=duration / recent_steps,
                         visual_frames_per_second=recent_frames / duration,
                         cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                         cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == "cuda" else 0,
                         process_peak_rss_bytes=process_peak_rss_bytes())
                    recent_started, recent_frames, recent_steps = time.perf_counter(), 0, 0
                    progress("training", last_loss=loss_value)
                    if not thermal_guard():
                        return
                if steps % args.checkpoint_every == 0:
                    save_state(latest_path)
                if stopping["requested"] or (args.max_steps and steps >= args.max_steps):
                    save_state(latest_path)
                    progress("stopped_resumable")
                    emit("stopped_resumable", step=steps, checkpoint=str(latest_path))
                    return
            labels, predictions = predict_models(torch, [model], split_rows["val"], args)
            threshold = select_threshold(labels, predictions[0])
            report = validation_selection_report(split_rows["val"], labels, predictions[0], threshold, args.validation_selection)
            metrics = report["metrics"]
            epochs_completed = epoch + 1
            improved = report["selection_value"] > best_selection_value + 1e-8
            if improved:
                best_ap, best_epoch, stale = metrics["ap"], epochs_completed, 0
                best_selection_value = report["selection_value"]
            else:
                stale += 1
            scheduler.step()
            epoch += 1
            batch_offset = 0
            record = {"epoch": epochs_completed, "steps": steps, "validation": metrics,
                      "validation_by_label_source": report["by_label_source"],
                      "selection_metric": report["selection_metric"], "selection_value": report["selection_value"],
                      "mean_train_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else None,
                      "best": improved}
            with (args.output / "history.jsonl").open("a", encoding="utf-8") as history:
                history.write(json.dumps(record, allow_nan=False) + "\n")
            if improved:
                save_state(best_path)
            save_state(latest_path)
            progress("training", validation=metrics)
            emit("epoch_completed", **record)
            if stopping["requested"]:
                progress("stopped_resumable")
                return

        if not epochs_completed or not best_path.exists():
            raise ValueError("Training has no completed validated epoch")
        best = validate_best_checkpoint(torch.load(best_path, map_location="cpu", weights_only=True), config_hash,
                                        expected_epoch=best_epoch, expected_ap=best_ap,
                                        expected_selection_value=best_selection_value)
        model.load_state_dict(best["model"], strict=True)
        labels, predictions = predict_models(torch, [model], split_rows["val"], args)
        threshold = select_threshold(labels, predictions[0])
        validation_report = validation_selection_report(split_rows["val"], labels, predictions[0], threshold, args.validation_selection)
        validation = validation_report["metrics"]
        # The held-out test is consulted only after training/early stopping and threshold selection.
        final_test_path = args.output / "final_test_results.json"
        best_sha = sha256_file(best_path)
        if final_test_path.exists() and not args.defer_final_test:
            final_test = json.loads(final_test_path.read_text(encoding="utf-8"))
            if final_test["best_checkpoint_sha256"] != best_sha or final_test.get("status") == "deferred_not_evaluated":
                raise ValueError("Held-out test was already evaluated for a different checkpoint")
        else:
            final_test = evaluate_final_heldout(
                torch, model, split_rows["test"], args, threshold=threshold,
                baseline_validation=baseline_validation, best_sha=best_sha,
                supervision=supervision, progress_callback=progress)
            atomic_json(final_test_path, final_test)
        bundle = args.output / "bundle"
        bundle.mkdir(exist_ok=True)
        atomic_checkpoint(torch, bundle / "model.pt", {key: value.detach().cpu() for key, value in model.state_dict().items()})
        try:
            upstream_commit = subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            upstream_commit = "unknown"
        metadata = {
            "schema_version": BUNDLE_SCHEMA, "model_type": "lrasd", "architecture": ARCHITECTURE,
            "checkpoint": "model.pt", "checkpoint_sha256": sha256_file(bundle / "model.pt"),
            "preprocessing": PREPROCESSING, "score_domain": SCORE_DOMAIN,
            "visual_canvas": {"width": 480, "height": 270, "resize": "ffmpeg-bilinear"},
            "dataset": {"name": args.dataset_name, "version": args.dataset_version,
                        "crop_type": "full_body_bbox", "split_manifest_sha256": manifest_sha,
                        "split_hashes": {name: data["sha256"] for name, data in summary.items()},
                        "split_groups": {name: data["groups"] for name, data in summary.items()},
                        "split_clips": {name: sorted({(row["movie"], row["clip"]) for row in selected})
                                        for name, selected in split_rows.items()},
                        "source_domain": args.source_domain, "target_domain": args.target_domain},
            "training": {"status": "completed", "epochs_completed": epochs_completed, "steps": steps,
                         **training_controls,
                         "best_epoch": best_epoch, "seed": args.seed, "precision": "float32",
                         "window_frames": args.frames, "batch_size": args.batch_size,
                         "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                         "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else 0,
                         "process_peak_rss_bytes": process_peak_rss_bytes(),
                         "exported_checkpoint_epoch": best["epochs_completed"], "exported_checkpoint_steps": best["steps"],
                         "early_stopped": stale >= args.patience, "pretrained_sha256": config["pretrained_sha256"],
                         "upstream_commit": upstream_commit, "stage_name": args.stage_name,
                         "parent_bundle": parent, "initialization": "parent_bundle_fresh_optimizer" if parent else "raw_pretrained_state_dict",
                         "parent_holdout_guard": parent_holdout_guard,
                         "completion_scope": "training_and_validation_only" if args.defer_final_test else "training_validation_and_final_test",
                         "artifact_purpose": "intermediate_stage_initialization" if args.defer_final_test else "completed_stage_model",
                         "forgetting_control": {**forgetting_control,
                             "source_domain_forgetting_evaluated": bool(parent and not args.defer_final_test
                                 and OFFICIAL_LABEL_SOURCE in final_test.get("by_label_source", {}))}},
            "validation": {"status": "completed", "version": "group-heldout-supervision-v2", "split": "val",
                           "metrics": validation, "threshold": threshold,
                           "threshold_selected_on": "val", "threshold_scope": "unmasked_frame_probability",
                           "semantics": supervision["split_semantics"]["val"],
                           "label_sources": supervision["splits"]["val"],
                           "by_label_source": validation_report["by_label_source"],
                           "selection_metric": args.validation_selection,
                           "selection_value": validation_report["selection_value"]},
            "calibration": {"status": "uncalibrated"},
            "domain_validation": {"status": "unverified", "label_quality": supervision["label_quality"],
                                  "human_frame_ground_truth": supervision["human_frame_ground_truth"],
                                  "human_frame_ground_truth_present": supervision.get("human_frame_ground_truth_present", False),
                                  "source_domain": args.source_domain,
                                  "target_domain": args.target_domain,
                                  "limitation": supervision["limitation"]},
            "supervision": supervision,
            "baseline_validation": baseline_validation, "heldout_test": final_test,
            "final_acceptance": {"status": "deferred_not_evaluated" if args.defer_final_test else "evaluated_against_declared_supervision",
                                 "human_frame_asd_verified": False,
                                 "test_sources_history": final_test.get("evaluation_history", {})},
            "comparison": {"validation_ap_delta": validation["ap"] - baseline_validation["ap"],
                           "validation_selection_metric": args.validation_selection,
                           "validation_selection_value_delta": validation_report["selection_value"]
                               - baseline_validation.get("selection_value", baseline_validation["ap"])},
        }
        if not args.defer_final_test:
            metadata["comparison"]["test_ap_delta"] = final_test["trained"]["ap"] - final_test["baseline"]["ap"]
        metadata["dataset"].update(split_strategy=args.split_strategy, label_source=args.label_source,
                                   machine_generated_labels=any(row.get("machine_generated") is True for row in rows),
                                   teacher_generated_labels=supervision["teacher_labels_present"],
                                   official_annotations=supervision["official_annotations_present"],
                                   human_annotated=supervision.get("human_frame_ground_truth_present", False),
                                   human_frame_ground_truth=supervision["human_frame_ground_truth"])
        atomic_json(bundle / "metadata.json", metadata)
        progress("completed", bundle=str(bundle), heldout_test=final_test)
        emit("training_completed", bundle=str(bundle), validation=validation, heldout_test=final_test)
    except BaseException as error:
        progress("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    run(parse_args())
