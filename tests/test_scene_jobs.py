"""Scene jobs against synthetic media/model adapters, never user films or paid AI.

The benchmark measures controlled fixture delays and local test bookkeeping only.
It is not an estimate of real provider throughput or recognition quality.
"""
import asyncio
import json
import time
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from scenerecall.jobs import JobQueue
from scenerecall.library import Library
from scenerecall.models import AnalysisInput, AssetInput, DeepenInput
from scenerecall.providers import AIResult, ProviderError
from scenerecall.scene_preparation import (
    reconciliation_scene_data, validate_scene_reconciliation, validate_subtitle_preparation,
    validate_visual_preparation,
)
from scenerecall.scenes import SceneStore
from scenerecall.search import SearchEngine


class SyntheticSceneProvider:
    """A deterministic paid-call double with genuine preparation contracts."""

    def __init__(self):
        self.calls = []
        self.chunk_size = 56
        self.continue_batches = False
        self.deep_times = set()
        self.fail_deep_at = None
        self.failed_once = False
        self.delays = {}
        self.nominal_delay_s = 0.0

    def get(self, identity):
        return {"id": identity, "name": "explicit synthetic provider", "base_url": "https://fixture.invalid/v1",
                "model": "synthetic-" + identity, "provider_type": "openai_compatible", "secret_mode": "none",
                "capabilities": ["preparation", "vision", "subtitle"], "max_concurrency": 2,
                "input_price_per_million": 1, "output_price_per_million": 1}

    async def _record(self, stage, **inputs):
        self.calls.append({"stage": stage, **deepcopy(inputs)})
        delay = self.delays.get(stage, 0)
        self.nominal_delay_s += delay
        if delay:
            await asyncio.sleep(delay)

    @staticmethod
    def _result(value):
        return AIResult(value, {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}, 1, .001)

    async def prepare_subtitles(self, identity, cues, story_context=None):
        await self._record("transcript", cues=cues, story_context=story_context)
        value = {"segments": [{"cue_ids": [cue["id"] for cue in cues], "summary": "合成台词提到回家"}],
                 "unresolved": []}
        return self._result(validate_subtitle_preparation(value, cues))

    async def prepare_visual(self, identity, frames, context_frames, subtitle_context, story_context,
                             character_context=None):
        await self._record("preparation", frames=frames, context_frames=context_frames,
                           subtitle_context=subtitle_context, story_context=story_context)
        chunks = []
        for offset in range(0, len(frames), self.chunk_size):
            selected = frames[offset:offset + self.chunk_size]
            ids = [frame["id"] for frame in selected]
            deep = selected[0]["at_ms"] in self.deep_times
            chunks.append({"frame_ids": ids, "summary": f"合成静态门前画面 {selected[0]['at_ms']}",
                           "visual_facts": [{"text": "合成画面中有门", "frame_ids": [ids[0]]}],
                           "dialogue_facts": [], "inferences": [],
                           "representative_frame_ids": list(dict.fromkeys([ids[0], ids[-1]])),
                           "needs_deep": deep, "reason": "合成测试复杂互动" if deep else "",
                           "continues_previous": bool(offset == 0 and context_frames and self.continue_batches)})
        context = subtitle_context or {}
        cues = context.get("cues", [])
        value = validate_visual_preparation({"chunks": chunks, "unresolved": []}, frames,
            {cue["id"] for cue in cues}, has_context=bool(context_frames), cues=cues,
            start_ms=context.get("start_ms"), end_ms=context.get("end_ms"))
        return self._result(value)

    async def analyze(self, identity, frames, start_ms, end_ms, character_context=None,
                      subtitle_context=None, story_context=None):
        await self._record("deep_analysis", frames=frames, start_ms=start_ms, end_ms=end_ms,
                           subtitle_context=subtitle_context, story_context=story_context)
        if start_ms == self.fail_deep_at and not self.failed_once:
            self.failed_once = True
            raise ProviderError("显式合成模型故障", request_count=1, cost=.001,
                                usage={"input_tokens": 20, "output_tokens": 0, "total_tokens": 20})
        return self._result({"summary": f"深入核实：角色递出杯子 {start_ms}", "entities": [], "events": [],
                             "spatial_observations": [], "uncertainties": [],
                             "evidence_frame_ids": [frame["id"] for frame in frames]})

    async def reconcile_scene(self, identity, scene, story_context):
        await self._record("context_refresh", scene=scene, story_context=story_context)
        _, frame_ids, cue_ids = reconciliation_scene_data(scene)
        value = {"summary": "合成场景结合前序深入结果更新理解", "inferences": [], "unresolved": []}
        return self._result(validate_scene_reconciliation(value, frame_ids, cue_ids))

    async def recognize_subtitles(self, identity, frames):
        await self._record("ocr", frames=frames)
        return self._result([{"frame_id": frame["id"], "lines": [
            {"text": "合成台词我们回家", "language": "zh", "uncertain": False}]} for frame in frames])


@pytest.fixture
def scene_worker(tmp_path, monkeypatch):
    sources = {}
    scans = []

    def probe(path):
        duration = sources[str(path)]["duration_ms"]
        return {"duration_ms": duration, "width": 32, "height": 32, "video_codec": "synthetic", "streams": []}

    def extract_frames(path, times_ms, output_paths, crop=None):
        result = []
        for at, output in zip(times_ms, output_paths, strict=True):
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), ((at // 100) % 255, (at // 10) % 255, 20)).save(output)
            result.append({"id": output.stem, "path": str(output), "at_ms": at})
        return result

    def scan(path, start_ms, end_ms, checkpoint=None, progress=None, *, metadata=None, decoder="ffmpeg"):
        scans.append((str(path), start_ms, end_ms))
        frames, shots, batches = [], [], []
        for at in range(start_ms, end_ms, 1000):
            stop = min(at + 1000, end_ms)
            when = min(at + 250, stop - 1)
            shot = {"id": f"synthetic_shot_{at}", "start_ms": at, "end_ms": stop}
            frame = {"id": f"synthetic_frame_{when}", "shot_id": shot["id"], "at_ms": when,
                     "pts": when, "time_base": [1, 1000], "source_pts": when,
                     "source_time_base": [1, 1000], "reason": "stable_pose"}
            shots.append(shot)
            frames.append(frame)
            batches.append({"id": f"synthetic_batch_{at}", "shot_id": shot["id"],
                            "start_ms": at, "end_ms": stop, "frame_ids": [frame["id"]]})
        if checkpoint:
            checkpoint()
        if progress:
            progress(end_ms - start_ms, end_ms - start_ms)
        return {"algorithm": "content-keyframes-v1", "shots": shots, "frames": frames, "batches": batches}

    def extract_keyframes(path, frames, output_paths, checkpoint=None, *, metadata=None):
        if checkpoint:
            checkpoint()
        actual = extract_frames(path, [frame["at_ms"] for frame in frames], output_paths)
        return [{**planned, **image} for planned, image in zip(frames, actual, strict=True)]

    monkeypatch.setattr("scenerecall.media.probe", probe)
    monkeypatch.setattr("scenerecall.media.extract_frames", extract_frames)
    monkeypatch.setattr("scenerecall.media.scan_keyframes", scan)
    monkeypatch.setattr("scenerecall.media.extract_keyframes", extract_keyframes)

    def extract_retrieval_frames(path, frames, output_paths, checkpoint=None, *, metadata=None, metrics=None):
        images = extract_keyframes(path, frames, output_paths, checkpoint, metadata=metadata)
        if metrics is not None:
            metrics.update(decode_passes=1, extraction_processes=1, frames=len(images))
        return images

    monkeypatch.setattr("scenerecall.keyframes.extract_keyframes_stream", extract_retrieval_frames)

    def create(frame_count=60, with_subtitles=False):
        identity = len(sources)
        directory = tmp_path / f"worker-{identity}"
        directory.mkdir()
        source = directory / "explicit-synthetic.mp4"
        source.write_bytes(f"No real media. Synthetic fixture {identity}.".encode())
        sources[str(source)] = {"duration_ms": frame_count * 1000}
        library = Library(directory / "library")
        asset = library.register(AssetInput(video_path=str(source), title="Explicit synthetic test", subtitle_mode="embedded"))
        provider = SyntheticSceneProvider()
        queue = JobQueue(library, provider, SearchEngine(library.root / "indexes", provider))
        result = SimpleNamespace(queue=queue, library=library, asset=asset, provider=provider,
                                 store=SceneStore(library), scans=scans, frame_count=frame_count)
        if with_subtitles:
            set_subtitles(result, "最初合成台词")
        return result

    return create


def set_subtitles(worker, text):
    duration = worker.asset["duration_ms"]
    worker.library.save_subtitles(worker.asset["id"], "fixture_subtitles_" + str(time.time_ns()),
        [{"start_ms": 0, "end_ms": min(duration, 1500), "text": text, "language": "zh", "review_status": "unreviewed"}],
        0, duration)


def enqueue(worker, **options):
    arguments = {"asset_id": worker.asset["id"], "analysis_strategy": "scene-chunks-v1", "stages": ["vision"],
                 "max_requests": 500, **options}
    _, config = worker.queue.validate(AnalysisInput(**arguments), {
        "bindings": {"preparation": "fast", "vision": "deep", "subtitle": "ocr"}})
    return worker.queue.add("analysis", config, worker.asset["id"])


async def execute(worker, job):
    await worker.queue.execute(job["id"])
    return worker.queue.get(job["id"])


def counts(worker):
    return Counter(call["stage"] for call in worker.provider.calls)


@pytest.mark.asyncio
async def test_silent_video_keeps_all_frames_with_56_core_and_eight_reference_boundary_frames(scene_worker):
    worker = scene_worker(60)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    preparation = [call for call in worker.provider.calls if call["stage"] == "preparation"]
    assert [(len(call["frames"]), len(call["context_frames"])) for call in preparation] == [(56, 0), (4, 8)]
    expected = [frame["id"] for call in preparation for frame in call["frames"]]
    assert [frame["id"] for frame in preparation[1]["context_frames"]] == expected[48:56]
    scenes = worker.store.list(worker.asset["id"])
    assert [frame_id for scene in scenes for frame_id in scene["frame_ids"]] == expected
    assert len(set(expected)) == 60 and len(scenes) == 2
    assert scenes[0]["start_ms"] == 0 and scenes[-1]["end_ms"] == 60000
    assert all(not scene["stale"] for scene in scenes)
    assert counts(worker) == {"preparation": 2}
    assert job["request_count"] == 2
    assert len([record for record in worker.library.records(worker.asset["id"]) if record["kind"] == "scene"]) == 2


@pytest.mark.asyncio
async def test_continuation_merges_across_preparation_batch_without_losing_shots_or_frames(scene_worker):
    worker = scene_worker(60)
    worker.provider.continue_batches = True
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    scenes = worker.store.list(worker.asset["id"])
    assert len(scenes) == 1
    scene = scenes[0]
    assert (scene["start_ms"], scene["end_ms"]) == (0, 60000)
    assert len(scene["frame_ids"]) == len(scene["shot_ids"]) == 60
    assert scene["frame_ids"][0] in scene["representative_frame_ids"]
    assert scene["frame_ids"][-1] in scene["representative_frame_ids"]
    assert len(scene["visual_facts"]) == 2 and counts(worker) == {"preparation": 2}


@pytest.mark.asyncio
async def test_complex_scene_deepens_then_later_simple_scenes_refresh_text_memory_only(scene_worker):
    worker = scene_worker(12)
    worker.provider.chunk_size = 4
    worker.provider.deep_times = {250}
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    assert counts(worker) == {"preparation": 1, "deep_analysis": 1, "context_refresh": 2}
    scenes = worker.store.list(worker.asset["id"])
    assert [scene["understanding_level"] for scene in scenes] == ["deep", "initial", "initial"]
    assert all(not scene["stale"] for scene in scenes)
    for call in (call for call in worker.provider.calls if call["stage"] == "context_refresh"):
        assert "深入核实" in json.dumps(call["story_context"], ensure_ascii=False)
        assert "frames" not in call
    assert [frame for scene in scenes for frame in scene["frame_ids"]] == [
        frame["id"] for frame in worker.provider.calls[0]["frames"]]


@pytest.mark.parametrize("run_ocr", [False, True])
@pytest.mark.asyncio
async def test_transcript_preparation_precedes_visual_and_requested_ocr_completes_first(scene_worker, run_ocr):
    worker = scene_worker(4, with_subtitles=not run_ocr)
    job = await execute(worker, enqueue(worker, stages=["vision", "subtitle"] if run_ocr else ["vision"]))
    assert job["status"] == "completed", job
    stages = [call["stage"] for call in worker.provider.calls]
    assert stages == (["ocr", "transcript", "preparation"] if run_ocr else ["transcript", "preparation"])
    transcript = next(call for call in worker.provider.calls if call["stage"] == "transcript")
    actual_ids = {cue["id"] for cue in worker.library.subtitles(worker.asset["id"])}
    assert {cue["id"] for cue in transcript["cues"]} == actual_ids
    visual = worker.provider.calls[-1]
    assert visual["subtitle_context"]["segments"]
    assert all(scene["dialogue_facts"] for scene in worker.store.list(worker.asset["id"]))


@pytest.mark.asyncio
async def test_failed_deep_retry_reuses_preparation_and_successful_deep_receipts(scene_worker):
    worker = scene_worker(30)
    worker.provider.deep_times = {250}
    worker.provider.fail_deep_at = 24250
    first = await execute(worker, enqueue(worker))
    assert first["status"] == "failed", first
    assert first["request_count"] == 3
    assert counts(worker) == {"preparation": 1, "deep_analysis": 2}
    assert worker.store.list(worker.asset["id"])[0]["understanding_level"] == "initial"
    worker.queue.control(first["id"], "retry")
    resumed = await execute(worker, first)
    assert resumed["status"] == "completed", resumed
    assert resumed["request_count"] == 4
    assert counts(worker) == {"preparation": 1, "deep_analysis": 3}
    assert [call["start_ms"] for call in worker.provider.calls if call["stage"] == "deep_analysis"] == [0, 24250, 24250]
    assert worker.store.list(worker.asset["id"])[0]["understanding_level"] == "deep"
    # The published stage is restored as a whole; its receipts are not replayed.
    assert resumed["stage_metrics"]["preparation"] == first["stage_metrics"]["preparation"]


@pytest.mark.asyncio
async def test_identical_new_run_reuses_paid_preparation_without_adding_request_usage(scene_worker):
    worker = scene_worker(60)
    first = await execute(worker, enqueue(worker))
    assert first["status"] == "completed", first
    old_ids = [scene["id"] for scene in worker.store.list(worker.asset["id"])]
    second = await execute(worker, enqueue(worker))
    assert second["status"] == "completed", second
    assert second["run_id"] != first["run_id"]
    assert counts(worker) == {"preparation": 2}
    assert second["request_count"] == 0 and second["usage"] == {}
    assert second["stage_metrics"]["preparation"]["cache_hits"] == 2
    assert [scene["id"] for scene in worker.store.list(worker.asset["id"])] == old_ids


@pytest.mark.asyncio
async def test_manual_deepen_respects_budget_and_preserves_scene_identity_favorite_and_frames(scene_worker):
    worker = scene_worker(8)
    initial = await execute(worker, enqueue(worker))
    assert initial["status"] == "completed", initial
    original = worker.store.list(worker.asset["id"])[0]
    worker.library.annotate(worker.asset["id"], {"record_id": original["id"], "favorite": True, "note": "保留此引用"})
    config, _ = worker.queue.deepen_config(worker.asset["id"], original["id"],
        DeepenInput(max_requests=2), {"bindings": {"vision": "deep"}})
    job = worker.queue.add("analysis", config, worker.asset["id"])
    paused = await execute(worker, job)
    assert paused["status"] == "paused" and paused["request_count"] == 0, paused
    assert counts(worker) == {"preparation": 1}
    worker.queue.control(job["id"], "resume", {"max_requests": 10})
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    result = worker.store.list(worker.asset["id"])[0]
    assert result["id"] == original["id"] and result["frame_ids"] == original["frame_ids"]
    assert result["understanding_level"] == "deep"
    record = next(record for record in worker.library.records(worker.asset["id"]) if record["id"] == original["id"])
    assert record["favorite"] and record["note"] == "保留此引用"
    assert counts(worker) == {"preparation": 1, "deep_analysis": 1}
    assert len(worker.scans) == 1


@pytest.mark.asyncio
async def test_changed_subtitles_mark_scenes_stale_and_prevent_resuming_old_dependencies(scene_worker):
    worker = scene_worker(30, with_subtitles=True)
    worker.provider.deep_times = {250}
    worker.provider.fail_deep_at = 24250
    first = await execute(worker, enqueue(worker))
    assert first["status"] == "failed", first
    old_count = len(worker.provider.calls)
    set_subtitles(worker, "修正后完全不同的台词")
    scenes = worker.store.list(worker.asset["id"])
    assert scenes and all(scene["stale"] for scene in scenes)
    assert any("字幕" in reason for scene in scenes for reason in scene["stale_reasons"])
    worker.queue.control(first["id"], "retry")
    retried = await execute(worker, first)
    assert retried["status"] == "failed" and "字幕" in retried["error"], retried
    assert len(worker.provider.calls) == old_count


@pytest.mark.asyncio
async def test_synthetic_benchmark_reports_request_counts_and_controlled_model_delay(scene_worker, record_property):
    reports = {}
    for label, strategy in (("legacy", "content-keyframes-v1"), ("layered", "scene-chunks-v1")):
        worker = scene_worker(60)
        worker.provider.delays = {"preparation": .001, "deep_analysis": .004}
        started = time.perf_counter()
        job = await execute(worker, enqueue(worker, analysis_strategy=strategy))
        elapsed = time.perf_counter() - started
        assert job["status"] == "completed", job
        reports[label] = {"requests": job["request_count"], "stages": dict(counts(worker)),
                          "controlled_model_delay_ms": round(worker.provider.nominal_delay_s * 1000),
                          "observed_test_elapsed_ms": round(elapsed * 1000)}
    assert reports["legacy"]["requests"] == 60
    assert reports["layered"]["requests"] == 2
    assert reports["legacy"]["controlled_model_delay_ms"] == 240
    assert reports["layered"]["controlled_model_delay_ms"] == 2
    report = {"kind": "synthetic fixture only; not a real-media speed claim", "frames": 60, **reports}
    record_property("synthetic_scene_benchmark", json.dumps(report, ensure_ascii=False))
    print("SYNTHETIC_SCENE_BENCHMARK=" + json.dumps(report, ensure_ascii=False))
