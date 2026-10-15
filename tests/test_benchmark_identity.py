"""Benchmark arithmetic and permission gates; never calls live models."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenerecall.library import atomic_json, read_json
from test_scene_jobs import enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


spec = importlib.util.spec_from_file_location("identity_benchmark_tests", Path(__file__).parents[1] / "scripts/benchmark_identity.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_default_report_has_three_cold_warm_groups_and_unknown_is_never_zero(tmp_path, monkeypatch):
    args = SimpleNamespace(asset="missing", plan="missing", start_ms=0, end_ms=1000,
                           allow_provider_calls=False, output=tmp_path / "report.json")
    def forbidden(*args, **kwargs):
        raise AssertionError("No local model or paid calls may run by default")
    monkeypatch.setattr(benchmark, "read_inputs", forbidden)
    result = benchmark.run(args)
    assert set(result["groups"]) == {"old", "new_visual", "new_audiovisual"}
    for group in result["groups"].values():
        for phase in ("cold", "warm"):
            assert group[phase]["status"] == "not_measured"
            assert all(group[phase][metric] is None for metric in benchmark.METRICS)
    assert read_json(args.output) == result


def test_der_optimal_permutation_overlap_misses_false_alarm_and_confusion():
    truth = [{"start_ms": 0, "end_ms": 1000, "speaker_id": "A"},
             {"start_ms": 500, "end_ms": 1500, "speaker_id": "B"}]
    swapped = [{"start_ms": 0, "end_ms": 1000, "speaker_id": "speaker_7"},
               {"start_ms": 500, "end_ms": 1500, "speaker_id": "speaker_1"}]
    result = benchmark.diarization_metrics(truth, swapped)
    assert result["diarization_error_rate"] == 0
    assert result["der_speaker_ms"] == 2000 and result["der_overlap_included"]
    assert result["speaker_permutation"] == {"speaker_7": "A", "speaker_1": "B"}
    missing = benchmark.diarization_metrics(truth, swapped[:1])
    assert missing["der_missed_ms"] == 1000 and missing["diarization_error_rate"] == .5
    false_alarm = benchmark.diarization_metrics(truth, swapped + [{"start_ms": 1600, "end_ms": 1800, "speaker_id": "speaker_7"}])
    assert false_alarm["der_false_alarm_ms"] == 200 and false_alarm["diarization_error_rate"] == .1
    confused = benchmark.diarization_metrics(
        [{"start_ms": 0, "end_ms": 1000, "speaker_id": "A"}, {"start_ms": 1000, "end_ms": 2000, "speaker_id": "B"}],
        [{"start_ms": 0, "end_ms": 2000, "speaker_id": "one"}])
    assert confused["der_confused_ms"] == 1000 and confused["diarization_error_rate"] == .5
    assert benchmark.diarization_metrics(None, swapped)["diarization_error_rate"] is None


def test_assignment_finds_global_optimum_instead_of_greedy_and_handles_rectangles():
    assert benchmark._assignment([[9, 8], [8, 0]]) == {0: 1, 1: 0}
    assert benchmark._assignment([[2, 5, 3]]) == {0: 1}
    assert benchmark._assignment([]) == {}


def test_character_speech_accuracy_keeps_overlap_and_unknown_abstentions():
    truth = [{"start_ms": 0, "end_ms": 1000, "character_id": "A"},
             {"start_ms": 500, "end_ms": 1000, "character_id": "B"}]
    hypothesis = [{"start_ms": 0, "end_ms": 1000, "character_id": "A"},
                  {"start_ms": 500, "end_ms": 1000, "character_id": None, "candidate_character_id": "B"}]
    result = benchmark.speaker_accuracy(truth, hypothesis)
    assert result["dialogue_speaker_accuracy"] == pytest.approx(2 / 3)
    assert result["speaker_labelled_ms"] == 1500
    assert benchmark.speaker_accuracy(None, hypothesis)["dialogue_speaker_accuracy"] is None


def test_visual_labels_are_paired_by_frame_and_region_not_actor_name():
    labels = [{"frame_id": "F1", "box": [0, 0, .5, 1], "character_id": "A"},
              {"frame_id": "F1", "box": [.5, 0, .5, 1], "character_id": "B"},
              {"frame_id": "F2", "box": [0, 0, .5, 1], "character_id": "C"}]
    predictions = [{"frame_id": "F1", "box": [0, 0, .5, 1], "character_id": "A"},
                   {"frame_id": "F1", "box": [.5, 0, .5, 1], "character_id": "wrong"},
                   {"frame_id": "F2", "box": [0, 0, .5, 1], "character_id": None}]
    result = benchmark.visual_accuracy(labels, predictions)
    assert result["role_coverage"] == pytest.approx(2 / 3)
    assert result["role_error_rate"] == .5
    unknown = benchmark.visual_accuracy(None, predictions)
    assert unknown["role_error_rate"] is None and unknown["role_coverage"] == pytest.approx(2 / 3)


def test_warm_copy_retains_features_not_previous_analysis_or_identity_decisions(tmp_path):
    cold, warm = tmp_path / "cold", tmp_path / "warm"
    for path in ("cache/identity-features/feature.json", "assets/a/audio/features/embedding.json",
                 "assets/a/audio/cache/key/turns.json", "assets/a/identity/visual/detections/detection.json",
                 "assets/a/identity/visual/decisions.json", "assets/a/analyses/run/scene.json",
                 "runtime/call-receipts/job/receipt.json"):
        atomic_json(cold / path, {"test": True})
    benchmark.retain_feature_cache(cold, warm, "a")
    assert (warm / "cache/identity-features/feature.json").is_file()
    assert (warm / "assets/a/audio/features/embedding.json").is_file()
    assert not (warm / "assets/a/identity/visual/decisions.json").exists()
    assert not (warm / "assets/a/analyses/run/scene.json").exists()
    assert not (warm / "runtime/call-receipts/job/receipt.json").exists()


@pytest.mark.asyncio
async def test_clone_keeps_same_frames_reference_roster_and_source_immutable(scene_worker, tmp_path):
    worker = scene_worker(4, with_subtitles=True)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job.get("error")
    plan = worker.queue.load_plan(job["config"]["keyframe_plan_id"])
    args = SimpleNamespace(library=worker.library.root, asset=worker.asset["id"])
    profile = {**worker.provider.get("deep"), "provider_type": "codex_cli", "secret_mode": "none"}
    before = {path: path.read_bytes() for path in worker.library.root.rglob("*.json")}
    cloned = benchmark.clone_source(args, tmp_path / "copied", plan, profile)
    assert cloned.get_asset(args.asset)["fingerprint"] == worker.asset["fingerprint"]
    assert cloned.subtitles(args.asset)
    for item in plan["frames"]:
        assert cloned.frame_path(args.asset, item["id"]).read_bytes() == worker.library.frame_path(args.asset, item["id"]).read_bytes()
    assert all(path.read_bytes() == content for path, content in before.items())
    with pytest.raises(ValueError, match="already exists"):
        benchmark.clone_source(args, tmp_path / "copied", plan, profile)
