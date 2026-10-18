"""Real-data pipeline gates using temporary synthetic fixtures; never train models."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from scenerecall.lrasd_training_data import (  # noqa: E402
    HUMAN_LABEL_SOURCE, HUMAN_REPLAY_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE,
    canonical_hash, load_annotation_plan, load_manifest, supervision_summary,
)
import evaluate_lrasd_human as evaluation  # noqa: E402
import prepare_lrasd_human_round as preparation  # noqa: E402


def human_row(split="train", **overrides):
    source = "a" * 64
    scene = f"scene-{split}"
    group = "human-scene-" + canonical_hash([source, scene])[:24]
    return {"sample_id": f"sample-{split}", "split": split, "movie": "Synthetic work", "clip": f"clip-{split}",
            "track": "track-1", "start_frame": 0, "end_frame": 50, "fps": 25,
            "visual_path": "visual.npz", "audio_path": "audio.wav", "audio_start_seconds": 0,
            "labels": [1, 0] * 24 + [-1, -1], "label_source": HUMAN_LABEL_SOURCE,
            "synthetic": False, "human_annotated": True, "human_frame_asd_ground_truth": True, "machine_generated": False,
            "annotation_granularity": "human_reviewed_frame_asd_and_body_boxes",
            "source_fingerprint": source, "source_video_sha256": source,
            "source_start_ms": {"train": 0, "val": 120000, "test": 240000}[split],
            "source_end_ms": {"train": 2000, "val": 122000, "test": 242000}[split],
            "evaluation_group": group,
            "annotation_plan": {"sha256": "b" * 64, "scene_id": scene, "split": split, "minimum_cross_split_gap_ms": 90000},
            "human_annotation": {"revision": 1, "reviewed": True, "excluded": False, "reviewed_at": "2026-10-17T00:00:00Z",
                                 "reviewer": {"kind": "human"}, "annotation_sha256": "c" * 64,
                                 "track_provenance": {"boxes_reviewed": True, "labels_source": "human_input", "model_scores_copied": False}},
            **overrides}


class HumanRoundTests(unittest.TestCase):
    def test_human_mode_has_true_frame_semantics_and_keeps_unknown_mask(self):
        rows = [human_row(split) for split in ("train", "val", "test")]
        report = supervision_summary(rows, HUMAN_LABEL_SOURCE)
        self.assertTrue(report["human_frame_ground_truth"])
        self.assertEqual(report["split_semantics"]["val"], "human_reviewed_frame_asd_agreement")
        self.assertEqual(report["splits"]["test"][HUMAN_LABEL_SOURCE]["masked_frames"], 2)

    def test_generated_or_unreviewed_cannot_be_called_human(self):
        for change in ({"machine_generated": True}, {"synthetic": True}, {"human_annotated": False},
                       {"human_annotation": {"reviewed": False}}, {"source_video_sha256": "d" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                supervision_summary([human_row(**change)], HUMAN_LABEL_SOURCE)

    def test_official_replay_preserves_separate_semantics(self):
        official = {**human_row(), "sample_id": "official", "label_source": OFFICIAL_LABEL_SOURCE,
                    "human_annotated": False, "human_frame_asd_ground_truth": False,
                    "official_annotations": True, "annotation_granularity": "sentence_speaker_and_character_boxes"}
        report = supervision_summary([human_row(), human_row("val"), official], HUMAN_REPLAY_LABEL_SOURCE)
        self.assertFalse(report["human_frame_ground_truth"])
        self.assertTrue(report["human_frame_ground_truth_present"])
        self.assertEqual(report["split_semantics"]["val"], "human_reviewed_frame_asd_agreement")
        self.assertEqual(set(report["sources"]), {HUMAN_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE})

    def test_no_test_inference_during_baseline_validation(self):
        rows = [human_row(split) for split in ("train", "val", "test")]
        def predictions(torch, models, selected, args):
            self.assertEqual({row["split"] for row in selected}, {"val"})
            labels = [label for row in selected for label in row["labels"] if label >= 0]
            return labels, [[0.9 if label else 0.1 for label in labels]]
        with patch.object(evaluation, "predict_models", side_effect=predictions):
            result = evaluation.evaluate_frozen(None, [object()], rows, None)
        self.assertNotIn("test", result[0])

    def test_final_test_uses_validation_threshold_without_test_tuning(self):
        rows = [human_row(split) for split in ("train", "val", "test")]
        calls = []
        def predictions(torch, models, selected, args):
            calls.append(selected[0]["split"])
            labels = [label for row in selected for label in row["labels"] if label >= 0]
            return labels, [[0.9 if label else 0.1 for label in labels] for _ in models]
        with patch.object(evaluation, "predict_models", side_effect=predictions):
            report = evaluation.evaluate_frozen(None, [object(), object()], rows, None, final_test=True)
        self.assertEqual(calls, ["val", "test"])
        self.assertEqual(report[0]["test"]["threshold"], report[0]["validation"]["threshold"])

    def test_cli_accepts_human_mode_and_explicit_scene_split(self):
        from train_lrasd import parse_args
        args = parse_args(["--manifest", "x", "--output", "y", "--frames", "50", "--label-source", HUMAN_LABEL_SOURCE,
                           "--allow-within-show-group-split", "--split-strategy", "scene_holdout_same_title"])
        self.assertEqual(args.label_source, HUMAN_LABEL_SOURCE)
        self.assertEqual(args.split_strategy, "scene_holdout_same_title")


def test_scene_plan_rejects_scene_or_temporal_leakage(tmp_path):
    path = tmp_path / "plan.json"
    plan = {"source_fingerprint": "a" * 64, "minimum_cross_split_gap_ms": 90000,
            "items": [{"id": "one", "clip_id": "one", "scene_id": "one", "split": "train", "start_ms": 0, "end_ms": 4000},
                      {"id": "two", "clip_id": "two", "scene_id": "two", "split": "val", "start_ms": 120000, "end_ms": 124000}]}
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert load_annotation_plan(path)["minimum_cross_split_gap_ms"] == 90000
    plan["items"][1]["scene_id"] = "one"
    path.write_text(json.dumps(plan), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="scene"):
        load_annotation_plan(path)
    plan["items"][1].update(scene_id="two", start_ms=3000, end_ms=7000)
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap or are too close"):
        load_annotation_plan(path)


def test_human_manifest_plan_and_temporal_guard_cannot_be_bypassed(tmp_path):
    path = tmp_path / "labeled.jsonl"
    records = [human_row(split) for split in ("train", "val", "test")]
    def save():
        path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    save()
    loaded = load_manifest(path, frames=50, check_assets=False, allow_within_show_group_split=True)
    assert len(loaded) == 3
    records[1].update(source_start_ms=4000, source_end_ms=6000)
    save()
    import pytest
    with pytest.raises(ValueError, match="overlap or are too close"):
        load_manifest(path, frames=50, check_assets=False, allow_within_show_group_split=True)
    records[1] = human_row("val")
    records[1].pop("annotation_plan")
    save()
    with pytest.raises(ValueError, match="fixed scene"):
        load_manifest(path, frames=50, check_assets=False, allow_within_show_group_split=True)


def test_unannotated_round_writes_blocked_report_without_dataset(tmp_path):
    plan = {"source_fingerprint": "a" * 64, "items": [
        {"id": "missing", "clip_id": "asd_missing", "scene_id": "scene", "split": "train", "start_ms": 0, "end_ms": 4000}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    root = tmp_path / "clips"
    root.mkdir()
    report = preparation.prepare_round(root, path, tmp_path / "round", tmp_path / "bundle", export=True)
    assert not report["ready"] and not report["dataset_exported"]
    assert report["blocked"][0]["reason"] == "clip_missing"
    assert not (tmp_path / "round/dataset").exists()
    assert not report["starts_training"]


def test_prior_baseline_exposure_rejects_nearby_holdout(tmp_path):
    prior = tmp_path / "prior.jsonl"
    prior.write_text(json.dumps({"sample_id": "old", "split": "train", "source_sha256": "a" * 64,
                                 "source_start_seconds": 100, "labels": [1] * 50}), encoding="utf-8")
    plan = {"source_fingerprint": "a" * 64, "minimum_cross_split_gap_ms": 90000,
            "items": [{"id": "val", "clip_id": "clip-val", "split": "val", "start_ms": 120000, "end_ms": 124000}]}
    report = preparation.check_baseline_exposure(plan, prior)
    assert not report["pristine_claimed"]
    assert report["conflicts"][0]["reason"] == "near_prior_baseline_film_window"


def test_frozen_comparison_requires_matching_parent_manifest_and_deferred_test(tmp_path):
    from scenerecall.lrasd_training_data import ARCHITECTURE, BUNDLE_SCHEMA, PREPROCESSING, sha256_file
    import pytest
    for filename in ("visual.npz", "audio.wav"):
        (tmp_path / filename).write_bytes(b"hash-bound-test-only")
    assets = {"visual": sha256_file(tmp_path / "visual.npz"), "audio": sha256_file(tmp_path / "audio.wav")}
    records = [human_row(split, asset_sha256=assets) for split in ("train", "val", "test")]
    manifest = tmp_path / "labeled.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for folder, data in ((baseline, b"baseline"), (candidate, b"candidate")):
        folder.mkdir()
        (folder / "model.pt").write_bytes(data)
    def metadata(folder, **training):
        return {"schema_version": BUNDLE_SCHEMA, "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING,
                "checkpoint": "model.pt", "checkpoint_sha256": sha256_file(folder / "model.pt"),
                "dataset": {"split_manifest_sha256": sha256_file(manifest)}, "training": {"status": "completed", **training},
                "heldout_test": {"test_inference_performed": False}}
    (baseline / "metadata.json").write_text(json.dumps(metadata(baseline)), encoding="utf-8")
    state = metadata(candidate, pretrained_sha256="f" * 64)
    (candidate / "metadata.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization differs"):
        evaluation.inspect_inputs(manifest, baseline, candidate)
    state["training"]["pretrained_sha256"] = sha256_file(baseline / "model.pt")
    state["heldout_test"]["test_inference_performed"] = True
    (candidate / "metadata.json").write_text(json.dumps(state), encoding="utf-8")
    previous = tmp_path / "baseline-validation.json"
    previous.write_text(json.dumps({"manifest_sha256": sha256_file(manifest),
                                    "baseline": {"checkpoint_sha256": sha256_file(baseline / "model.pt")},
                                    "test_inference_performed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="must defer"):
        evaluation.inspect_inputs(manifest, baseline, candidate, final_test=True, baseline_report=previous)
    state["heldout_test"]["test_inference_performed"] = False
    (candidate / "metadata.json").write_text(json.dumps(state), encoding="utf-8")
    rows, _, _, _ = evaluation.inspect_inputs(manifest, baseline, candidate, final_test=True, baseline_report=previous)
    assert len(rows) == 3
    (tmp_path / "audio.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset SHA256"):
        evaluation.inspect_inputs(manifest, baseline)


if __name__ == "__main__":
    unittest.main()
