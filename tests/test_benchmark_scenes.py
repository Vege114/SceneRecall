"""Measurement/report integrity using local synthetic fixtures, never live models."""
import asyncio
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scenerecall.library import atomic_json, read_json
from scenerecall.providers import AIResult, ProviderError, ProviderManager
from scenerecall.search import SearchEngine
from test_scene_jobs import enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


spec = importlib.util.spec_from_file_location("scene_benchmark_test_module", Path(__file__).parents[1] / "scripts/benchmark_scenes.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def measured_provider(tmp_path):
    provider = benchmark.MeasuredProviders(tmp_path / "private", tmp_path / "measurements.json")
    provider.upsert({"id": "test", "name": "Explicit synthetic transport", "base_url": "https://fixture.invalid/v1",
                     "model": "synthetic", "capabilities": ["preparation", "vision"], "secret_mode": "none"})
    return provider


def synthetic_frames(tmp_path, count):
    result = []
    for number in range(count):
        path = tmp_path / f"synthetic-{number}.png"
        Image.new("RGB", (32, 18), (number, 60, 80)).save(path)
        result.append({"id": f"frame-{number}", "path": str(path), "at_ms": number * 1000})
    return result


@pytest.mark.asyncio
async def test_wire_image_measurements_count_sheets_and_character_references_not_core_frames(tmp_path, monkeypatch):
    async def local_chat(self, *args, **kwargs):
        return AIResult({}, {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}, 1, .001)

    monkeypatch.setattr(ProviderManager, "_chat", local_chat)
    provider = measured_provider(tmp_path)
    frames = synthetic_frames(tmp_path, 64)
    await provider.prepare_visual("test", frames[8:], frames[:8], None, None)
    character = {"scope_id": "test", "revision": 0, "profiles": [{"id": "character-one"}],
                 "reference_frames": [{**frames[0], "character_id": "character-one"}]}
    await provider.analyze("test", frames[:2], 0, 2000, character)
    rows = read_json(provider.measurements)
    assert (rows[0]["core_frame_count"], rows[0]["context_frame_count"], rows[0]["image_count_per_attempt"]) == (56, 8, 4)
    assert (rows[1]["core_frame_count"], rows[1]["character_reference_count"], rows[1]["image_count_per_attempt"]) == (2, 1, 3)
    assert all(row["status"] == "completed" and row["request_count"] == 1 for row in rows)
    assert all(row["prompt_text_bytes"] > 0 for row in rows)
    assert all("current_images" not in row for row in rows)


@pytest.mark.asyncio
async def test_failed_model_attempts_keep_usage_and_failure_and_are_not_successful_calls(tmp_path, monkeypatch):
    async def local_chat(self, *args, **kwargs):
        raise ProviderError("explicit synthetic failure", request_count=2,
                            usage={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50})

    monkeypatch.setattr(ProviderManager, "_chat", local_chat)
    provider = measured_provider(tmp_path)
    frames = synthetic_frames(tmp_path, 1)
    with pytest.raises(ProviderError):
        await provider.analyze("test", frames, 0, 1000)
    row = read_json(provider.measurements)[0]
    assert row["status"] == "failed" and row["failure"] == "ProviderError"
    assert row["request_count"] == 2 and row["usage"]["total_tokens"] == 50
    assert row["image_count_per_attempt"] == 1 and row["elapsed_ms"] is not None


@pytest.mark.asyncio
async def test_cancelled_provider_call_persists_unknown_attempt_count_not_zero(tmp_path, monkeypatch):
    async def interrupted(self, id, frames, start_ms, end_ms, character_context=None):
        pending = read_json(self.measurements)[0]
        assert pending["status"] == "running" and pending["elapsed_ms"] is None
        raise asyncio.CancelledError()

    monkeypatch.setattr(ProviderManager, "analyze", interrupted)
    provider = measured_provider(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await provider.analyze("test", [], 0, 1000)
    row = read_json(provider.measurements)[0]
    assert row["status"] == "interrupted" and row["request_count"] is None
    result = benchmark.measurement_summary([row], {"request_count": 3, "uncertain_request_count": 3})
    assert result["model_ms"] is None and not result["model_timing_complete"]
    assert result["unmeasured_or_uncertain_requests"] == 3


@pytest.mark.asyncio
async def test_search_timer_awaits_rebuild_and_does_not_double_count_nested_upsert(tmp_path, monkeypatch):
    def upsert(self, records):
        time.sleep(.02)

    async def rebuild(self, records, *args, **kwargs):
        await asyncio.sleep(.02)
        self.upsert(records)
        return {"completed": True}

    monkeypatch.setattr(SearchEngine, "upsert", upsert)
    monkeypatch.setattr(SearchEngine, "rebuild", rebuild)
    search = benchmark.MeasuredSearch(tmp_path / "index", None)
    started = time.monotonic()
    assert await search.rebuild([]) == {"completed": True}
    elapsed = (time.monotonic() - started) * 1000
    assert 35 <= search.elapsed_ms <= elapsed + 3
    assert search.depth == 0


def test_legacy_image_counts_remain_untrusted_and_request_gaps_disable_speed_claims():
    legacy = [{"method": "prepare_visual", "elapsed_ms": 100, "request_count": 1, "current_images": 56}]
    original = dict(legacy[0])
    known = benchmark.measurement_summary(legacy, {"request_count": 1})
    assert known["model_ms"] == 100 and not known["image_metrics_complete"]
    missing = benchmark.measurement_summary(legacy, {"request_count": 2})
    assert missing["model_ms"] is None and missing["observed_model_ms"] == 100
    assert missing["unmeasured_or_uncertain_requests"] == 1
    assert legacy[0] == original
    report = {"model_reduction_fraction": .9, "timing_target_met": True, "strategies": {
        "content-keyframes-v1": {"status": "completed", **known},
        "scene-chunks-v1": {"status": "completed", **missing}}}
    benchmark.update_comparison(report)
    assert "model_reduction_fraction" not in report and "timing_target_met" not in report


def test_interrupted_execution_ledger_keeps_known_subtotals_without_fabricating_wall_time(tmp_path):
    path = tmp_path / "execution-sessions.json"
    atomic_json(path, {"legacy_wall_ms": 10, "legacy_index_ms": 2, "sessions": [
        {"id": "one", "status": "finished", "wall_ms": 40, "index_ms": 5},
        {"id": "two", "status": "running", "wall_ms": None, "index_ms": None}]})
    ledger = benchmark.execution_ledger(path, {}, True)
    assert ledger["sessions"][-1]["status"] == "interrupted"
    summary = benchmark.execution_totals(ledger)
    assert summary["wall_ms"] is None and summary["observed_wall_ms"] == 50
    assert summary["index_ms"] is None and summary["observed_index_ms"] == 7
    assert not summary["wall_timing_complete"] and not summary["index_timing_complete"]
    assert benchmark.execution_ledger(tmp_path / "unmeasured.json", {}, True)["legacy_wall_ms"] is None
    assert benchmark.execution_totals(benchmark.execution_ledger(tmp_path / "fresh.json", {}, False))["wall_ms"] == 0
    migrated = benchmark.execution_ledger(tmp_path / "old-report.json", {"wall_ms": 40, "index_ms": 0}, True)
    assert migrated["legacy_wall_ms"] == 40 and migrated["legacy_index_ms"] is None
    verified = benchmark.execution_ledger(tmp_path / "verified-report.json",
        {"wall_ms": 40, "index_ms": 3, "index_timing_complete": True}, True)
    assert verified["legacy_index_ms"] == 3


@pytest.mark.asyncio
async def test_seed_preserves_equal_inputs_and_empty_memory_and_rejects_tampering(scene_worker, tmp_path):
    worker = scene_worker(4, with_subtitles=True)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    plan = worker.queue.load_plan(job["config"]["keyframe_plan_id"])
    args = SimpleNamespace(library=worker.library.root, asset=worker.asset["id"])
    profile = {**worker.provider.get("deep"), "provider_type": "codex_cli", "base_url": "", "secret_mode": "none"}
    expected = benchmark.input_identity(args, plan, profile)
    first = benchmark.seed(args, tmp_path / "baseline", plan, profile, expected)
    second = benchmark.seed(args, tmp_path / "candidate", plan, profile, expected)
    assert read_json(first.root / "benchmark-input.json") == read_json(second.root / "benchmark-input.json")
    assert not first.observations(args.asset) and not second.observations(args.asset)
    assert not first.active(args.asset).get("scenes") and not second.active(args.asset).get("scenes")
    assert first.subtitles(args.asset) == second.subtitles(args.asset)
    original_path = worker.library.frame_path(args.asset, plan["frames"][0]["id"])
    original_bytes = original_path.read_bytes()
    second.frame_path(args.asset, plan["frames"][0]["id"]).write_bytes(b"tampered synthetic image")
    with pytest.raises(ValueError, match="image bytes changed"):
        benchmark.seed(args, second.root, plan, profile, expected)
    assert original_path.read_bytes() == original_bytes
    changed = {**expected, "model_identity": {**expected["model_identity"], "model": "different"}}
    with pytest.raises(ValueError, match="input identity changed"):
        benchmark.seed(args, first.root, plan, profile, changed)
