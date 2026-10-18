"""Data-integrity and probability-metric checks; these tests never start training."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import pytest

from scenerecall.lrasd_training_data import (
    assign_group_splits, classification_metrics, load_manifest, load_sample,
    ranking_metrics, select_threshold, split_summary, supervision_summary,
    TEACHER_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE, MIXED_LABEL_SOURCE,
    validation_selection_report,
)


def row(**overrides):
    return {"schema_version": 1, "sample_id": "sample", "split": "train", "movie": "Movie A",
            "clip": "clip-a", "track": "character-a", "start_frame": 0, "end_frame": 100, "fps": 25,
            "visual_path": "visual.npz", "audio_path": "audio.wav", "audio_start_seconds": 0,
            "labels": [0, 1] * 50, **overrides}


def manifest(tmp_path, rows):
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in rows), encoding="utf-8")
    return path


def test_movie_and_show_leakage_is_rejected(tmp_path):
    path = manifest(tmp_path, [row(), row(sample_id="heldout", split="val")])
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(path, check_assets=False)
    path = manifest(tmp_path, [row(group="same-show"), row(sample_id="heldout", movie="Episode 2", group="same-show", split="test")])
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(path, check_assets=False)


def test_masks_and_split_hash_capture_label_changes(tmp_path):
    path = manifest(tmp_path, [row(mask=[False] + [True] * 99)])
    loaded = load_manifest(path, check_assets=False)
    assert loaded[0]["labels"][0] == -1
    summary = split_summary(loaded)["train"]
    assert summary["positive_frames"] == 50
    assert summary["negative_frames"] == 49
    assert summary["masked_frames"] == 1
    before = summary["sha256"]
    loaded[0]["labels"][1] = 0
    assert split_summary(loaded)["train"]["sha256"] != before


def test_within_show_episode_holdout_needs_opt_in_and_never_splits_a_clip(tmp_path):
    rows = [row(evaluation_group="episode-01"),
            row(sample_id="val", split="val", clip="episode-09", evaluation_group="episode-09")]
    path = manifest(tmp_path, rows)
    with pytest.raises(ValueError, match="opt-in"):
        load_manifest(path, check_assets=False)
    loaded = load_manifest(path, check_assets=False, allow_within_show_group_split=True)
    assert split_summary(loaded)["val"]["groups"] == ["episode-09"]
    rows[1]["clip"] = "clip-a"
    with pytest.raises(ValueError, match="Source clip leakage"):
        load_manifest(manifest(tmp_path, rows), check_assets=False, allow_within_show_group_split=True)
    rows[1]["clip"] = "episode-09"
    rows[1]["evaluation_group"] = "episode-01"
    with pytest.raises(ValueError, match="group leakage"):
        load_manifest(manifest(tmp_path, rows), check_assets=False, allow_within_show_group_split=True)


@pytest.mark.parametrize("overrides", [
    {"fps": 23.98}, {"end_frame": 99}, {"labels": [True] * 100},
    {"labels": [-1] * 100}, {"audio_start_seconds": -0.01},
])
def test_manifest_rejects_incompatible_preprocessing(tmp_path, overrides):
    with pytest.raises(ValueError):
        load_manifest(manifest(tmp_path, [row(**overrides)]), check_assets=False)


def test_tie_aware_ap_auc_and_threshold_are_probabilities():
    assert ranking_metrics([1, 0], [0.5, 0.5]) == {"ap": 0.5, "auc": 0.5}
    labels, scores = [0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]
    threshold = select_threshold(labels, scores)
    assert 0.2 < threshold <= 0.3
    result = classification_metrics(labels, scores, threshold)
    assert result["ap"] == result["auc"] == result["f1"] == result["balanced_accuracy"] == 1
    assert result["fp"] == result["fn"] == 0
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics([0, 1], [-1.0, 2.0])
    with pytest.raises(ValueError, match="both validation classes"):
        select_threshold([1, 1], [0.2, 0.7])


def test_group_split_is_deterministic_and_has_all_three_splits():
    first = assign_group_splits(["B", "C", "A", "D", "E"], 114)
    assert first == assign_group_splits(["E", "A", "C", "B", "D"], 114)
    assert set(first.values()) == {"train", "val", "test"}


def test_wav_offset_and_uint8_shape_are_verified(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    np.savez_compressed(tmp_path / "visual.npz", visual=np.zeros((100, 112, 112), dtype=np.uint8))
    # The second four-second window has a distinguishable PCM value.
    samples = np.concatenate([np.zeros(64000, dtype="<i2"), np.full(64000, 123, dtype="<i2")])
    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())
    observed = {}

    def mfcc(signal, sample_rate, **kwargs):
        observed.update(first=int(signal[0]), length=len(signal), sample_rate=sample_rate, **kwargs)
        return np.ones((398, 13))

    monkeypatch.setitem(sys.modules, "python_speech_features", SimpleNamespace(mfcc=mfcc))
    path = manifest(tmp_path, [row(audio_start_seconds=4)])
    loaded = load_manifest(path)[0]
    audio, visual, target = load_sample(loaded, path)
    assert observed["first"] == 123 and observed["length"] == 64000
    assert audio.shape == (400, 13) and audio.dtype == np.float32
    assert visual.shape == (100, 112, 112) and target.shape == (100,)
    loaded["audio_start_seconds"] = 4.1
    with pytest.raises(ValueError, match="exceeds WAV"):
        load_sample(loaded, path)
    loaded["audio_start_seconds"] = 0
    np.savez_compressed(tmp_path / "visual.npz", visual=np.zeros((100, 112, 112), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        load_sample(loaded, path)


def test_resume_loader_does_not_consume_model_random_generator():
    """A reconstructed loader must receive its own seed, independent of dropout RNG."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_lrasd.py"
    spec = importlib.util.spec_from_file_location("lrasd_runner_for_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = {}

    class Generator:
        def manual_seed(self, seed):
            self.seed = seed
            return self

    fake_torch = SimpleNamespace(Generator=Generator, utils=SimpleNamespace(data=SimpleNamespace(
        DataLoader=lambda dataset, **kwargs: received.update(kwargs))))
    args = SimpleNamespace(manifest="unused", seed=114, batch_size=4, workers=0, device="cpu")
    module.make_loader(fake_torch, [], args, train=True, epoch=3)
    assert received["generator"].seed == 117
    assert received["shuffle"] is False


def test_thermal_telemetry_is_read_only_and_finite(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_lrasd.py"
    spec = importlib.util.spec_from_file_location("lrasd_runner_thermal_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="89\n")

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.gpu_temperature() == 89
    assert seen == [["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", "--id=0"]]
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="nan\n"))
    assert module.gpu_temperature() is None


def test_export_checkpoint_must_match_manifest_and_validation_selection():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_lrasd.py"
    spec = importlib.util.spec_from_file_location("lrasd_checkpoint_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from scenerecall.lrasd_training_data import canonical_hash

    config = {"manifest_sha256": "dataset-a", "seed": 114}
    config_hash = canonical_hash(config)
    state = {"schema_version": "scenerecall-lrasd-resume-v1", "config": config, "config_hash": config_hash,
             "model": {f"parameter-{i}": 0 for i in range(168)}, "optimizer": {}, "scheduler": {}, "rng": {},
             "epoch": 3, "epochs_completed": 3, "batch_offset": 0, "steps": 30,
             "best_epoch": 3, "best_ap": .8, "stale_epochs": 0, "baseline_validation": {"ap": .7}}
    assert module.validate_best_checkpoint(state, config_hash, expected_epoch=3, expected_ap=.8) is state
    with pytest.raises(ValueError, match="validation selection"):
        module.validate_best_checkpoint(state, config_hash, expected_epoch=2, expected_ap=.8)
    with pytest.raises(ValueError, match="incompatible"):
        module.validate_best_checkpoint(state, canonical_hash({"manifest_sha256": "dataset-b", "seed": 114}))
    state["stale_epochs"] = 1
    with pytest.raises(ValueError, match="validation-selected"):
        module.validate_best_checkpoint(state, config_hash)


def test_two_second_window_uses_200_mfcc_frames_and_preserves_mask(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    np.savez_compressed(tmp_path / "visual.npz", visual=np.zeros((50, 112, 112), dtype=np.uint8))
    samples = np.concatenate([np.full(32000, 17, dtype="<i2"), np.full(32000, 123, dtype="<i2")])
    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())

    def mfcc(signal, sample_rate, **kwargs):
        assert sample_rate == 16000 and len(signal) == 32000 and int(signal[0]) == 123
        assert kwargs == {"numcep": 13, "winlen": .025, "winstep": .010}
        # A two-second PCM signal produces 199 MFCC frames before upstream wrap padding.
        return np.repeat(np.arange(199, dtype=np.float64)[:, None], 13, axis=1)

    monkeypatch.setitem(sys.modules, "python_speech_features", SimpleNamespace(mfcc=mfcc))
    path = manifest(tmp_path, [row(start_frame=50, end_frame=100, labels=[0, 1] * 25,
                                  mask=[False] * 4 + [True] * 46, audio_start_seconds=2)])
    loaded = load_manifest(path, frames=50)
    audio, visual, target = load_sample(loaded[0], path)
    assert audio.shape == (200, 13) and audio.dtype == np.float32
    assert audio[-1, 0] == audio[0, 0] == 0
    assert visual.shape == (50, 112, 112) and target.shape == (50,)
    assert target.tolist()[:4] == [-1] * 4
    assert split_summary(loaded)["train"]["track_frames"] == 50
    assert split_summary(loaded)["train"]["masked_frames"] == 4
    loaded[0]["audio_start_seconds"] = 2.04
    with pytest.raises(ValueError, match="exceeds WAV"):
        load_sample(loaded[0], path)


def test_t50_batch_eight_obeys_existing_frame_budget_and_rss_falls_back(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_lrasd.py"
    spec = importlib.util.spec_from_file_location("lrasd_t50_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["--manifest", "manifest.jsonl", "--output", "run", "--frames", "50", "--batch-size", "8"])
    assert args.frames == 50 and args.batch_size == 8
    with pytest.raises(SystemExit):
        module.parse_args(["--manifest", "manifest.jsonl", "--output", "run", "--frames", "50", "--batch-size", "9"])
    status = tmp_path / "status"
    status.write_text("Name:\ttest\nVmHWM:\t12345 kB\n", encoding="ascii")
    assert module.process_peak_rss_bytes(status) == 12345 * 1024
    assert module.process_peak_rss_bytes(tmp_path / "missing-status") is None


def supervised_row(source, **overrides):
    fields = {"label_source": source, "human_annotated": False}
    if source == TEACHER_LABEL_SOURCE:
        fields["machine_generated"] = True
    else:
        fields.update(official_annotations=True, annotation_granularity="sentence_speaker_and_character_boxes")
    return row(**{**fields, **overrides})


def test_official_sentence_annotations_are_not_teacher_or_human_frame_truth():
    record = supervised_row(OFFICIAL_LABEL_SOURCE)
    result = supervision_summary([record], OFFICIAL_LABEL_SOURCE)
    assert result["official_annotations_present"] is True
    assert result["teacher_labels_present"] is False
    assert result["human_frame_ground_truth"] is False
    assert result["semantics"] == "official_sentence_speaker_alignment_agreement"
    assert "projected to frames" in result["limitation"]
    assert result["sources"][OFFICIAL_LABEL_SOURCE]["positive_frames"] == 50
    with pytest.raises(ValueError, match="granularity"):
        supervision_summary([supervised_row(OFFICIAL_LABEL_SOURCE, official_annotations=False)], OFFICIAL_LABEL_SOURCE)
    for source in (TEACHER_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE):
        with pytest.raises(ValueError, match="frame-level ASD"):
            supervision_summary([supervised_row(source, human_annotated=True)], source)
        with pytest.raises(ValueError, match="frame-level ASD"):
            supervision_summary([supervised_row(source, human_ground_truth=True)], source)


def test_mixed_supervision_requires_both_sources_and_records_split_counts():
    records = [supervised_row(OFFICIAL_LABEL_SOURCE),
               supervised_row(TEACHER_LABEL_SOURCE, sample_id="target-train"),
               supervised_row(TEACHER_LABEL_SOURCE, sample_id="target-val", split="val")]
    result = supervision_summary(records, MIXED_LABEL_SOURCE)
    assert result["sources"][TEACHER_LABEL_SOURCE]["windows"] == 2
    assert result["splits"]["train"][OFFICIAL_LABEL_SOURCE]["windows"] == 1
    assert result["splits"]["val"][TEACHER_LABEL_SOURCE]["negative_frames"] == 50
    assert OFFICIAL_LABEL_SOURCE not in result["splits"]["val"]
    assert result["split_semantics"]["train"] == "mixed_official_alignment_and_teacher_agreement"
    assert result["split_semantics"]["val"] == "teacher_consensus_agreement"
    with pytest.raises(ValueError, match="mixed requires both"):
        supervision_summary(records[:1], MIXED_LABEL_SOURCE)
    with pytest.raises(ValueError, match="Every row label_source"):
        supervision_summary(records, OFFICIAL_LABEL_SOURCE)
    with pytest.raises(ValueError, match="Unsupported label_source"):
        supervision_summary(records, "anything_bypassing_validation")
    with pytest.raises(ValueError, match="machine_generated"):
        supervision_summary([supervised_row(TEACHER_LABEL_SOURCE, machine_generated=False)], TEACHER_LABEL_SOURCE)


def test_parent_bundle_initialization_verifies_sha_and_resets_stage_arguments(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_lrasd.py"
    spec = importlib.util.spec_from_file_location("lrasd_stage_init_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    checkpoint = bundle / "model.pt"
    checkpoint.write_bytes(b"test-state-dict-content")
    metadata = {"schema_version": module.BUNDLE_SCHEMA, "architecture": module.ARCHITECTURE,
                "preprocessing": module.PREPROCESSING, "checkpoint": "model.pt",
                "checkpoint_sha256": module.sha256_file(checkpoint),
                "training": {"status": "completed", "stage_name": "source-supervision",
                             "artifact_purpose": "intermediate_stage_initialization"},
                "heldout_test": {"status": "deferred_not_evaluated"},
                "dataset": {"split_groups": {"val": ["heldout-source"], "test": []},
                            "split_clips": {"val": [["Movie A", "heldout-clip"]]}}}
    (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    args = module.parse_args(["--manifest", "data.jsonl", "--output", "stage2", "--parent-bundle", str(bundle),
                              "--stage-name", "target-adaptation", "--label-source", MIXED_LABEL_SOURCE])
    assert args.pretrained == checkpoint and args.resume is None
    parent = module.inspect_parent_bundle(bundle, args.pretrained)
    assert parent["stage_name"] == "source-supervision"
    assert parent["checkpoint_sha256"] == module.sha256_file(checkpoint)
    with pytest.raises(ValueError, match="held-out"):
        module.validate_parent_holdout_isolation([row(group="heldout-source")], parent)
    with pytest.raises(ValueError, match="held-out"):
        module.validate_parent_holdout_isolation([row(clip="heldout-clip")], parent)
    assert module.validate_parent_holdout_isolation([row(group="source-training-group")], parent).startswith("prior_stage")
    checkpoint.write_bytes(b"corrupted-content")
    with pytest.raises(ValueError, match="SHA"):
        module.inspect_parent_bundle(bundle, checkpoint)


def test_deferred_stage_never_builds_or_infers_a_test_model(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_lrasd.py"
    spec = importlib.util.spec_from_file_location("lrasd_deferred_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        pytest.fail("Deferred final test must not build a baseline or load/infer test assets")

    monkeypatch.setattr(module, "build_model", forbidden)
    monkeypatch.setattr(module, "predict_models", forbidden)
    records = [supervised_row(TEACHER_LABEL_SOURCE, split="test")]
    supervision = supervision_summary(records, TEACHER_LABEL_SOURCE)
    args = SimpleNamespace(defer_final_test=True, previously_evaluated_test_source=[TEACHER_LABEL_SOURCE])
    progress = []
    result = module.evaluate_final_heldout(
        None, None, records, args, threshold=.5, baseline_validation=None,
        best_sha="unread-checkpoint", supervision=supervision, progress_callback=progress.append)
    assert result["status"] == "deferred_not_evaluated"
    assert result["test_inference_performed"] is False and "trained" not in result and "baseline" not in result
    assert result["artifact_purpose"] == "intermediate_stage_initialization"
    assert result["evaluation_history"][TEACHER_LABEL_SOURCE]["history"] == "previously_evaluated"
    assert result["evaluation_history"][TEACHER_LABEL_SOURCE]["pristine_claimed"] is False
    assert progress == []


def test_macro_source_ap_prevents_larger_source_dominating_checkpoint_selection():
    records = [{"label_source": OFFICIAL_LABEL_SOURCE, "labels": [1] * 20 + [0] * 20},
               {"label_source": TEACHER_LABEL_SOURCE, "labels": [1, 0]}]
    labels = [label for record in records for label in record["labels"]]
    # Candidate A wins the larger official source, but reverses target labels.
    first = [.9] * 20 + [.1] * 20 + [.01, .95]
    second = [.9] * 10 + [.1] * 10 + [.5] * 20 + [.99, .0]
    a = validation_selection_report(records, labels, first, .5, "macro_source_ap")
    b = validation_selection_report(records, labels, second, .5, "macro_source_ap")
    assert a["metrics"]["ap"] > b["metrics"]["ap"]
    assert a["selection_value"] < b["selection_value"]
    assert a["selection_value"] == .75
    assert b["selection_value"] == .875
    legacy = validation_selection_report(records, labels, first, .5)
    assert legacy["selection_metric"] == "pooled_ap" and legacy["selection_value"] == legacy["metrics"]["ap"]
    records[1]["labels"] = [1, 1]
    with pytest.raises(ValueError, match="both validation classes"):
        validation_selection_report(records, labels[:-2] + [1, 1], first, .5, "macro_source_ap")


def test_macro_checkpoint_keeps_selection_value_separate_from_pooled_ap():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_lrasd.py"
    spec = importlib.util.spec_from_file_location("lrasd_macro_checkpoint_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from scenerecall.lrasd_training_data import canonical_hash
    config = {"manifest_sha256": "mixed-dataset", "validation_selection": "macro_source_ap"}
    config_hash = canonical_hash(config)
    state = {"schema_version": "scenerecall-lrasd-resume-v1", "config": config, "config_hash": config_hash,
             "model": {f"parameter-{i}": 0 for i in range(168)}, "optimizer": {}, "scheduler": {}, "rng": {},
             "epoch": 3, "epochs_completed": 3, "batch_offset": 0, "steps": 30,
             "best_epoch": 3, "best_ap": .92, "best_selection_metric": "macro_source_ap",
             "best_selection_value": .81, "stale_epochs": 0, "baseline_validation": {"ap": .7}}
    assert module.validate_best_checkpoint(state, config_hash, expected_ap=.92, expected_selection_value=.81) is state
    with pytest.raises(ValueError, match="validation selection"):
        module.validate_best_checkpoint(state, config_hash, expected_selection_value=.92)
    state["best_selection_metric"] = "pooled_ap"
    with pytest.raises(ValueError, match="selection metric"):
        module.validate_resume_checkpoint(state, config_hash)
