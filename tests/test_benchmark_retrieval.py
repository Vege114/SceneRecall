"""New-flow experiment guardrails with generated fixtures, no model inference."""
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenerecall.library import atomic_json, read_json
from test_retrieval_jobs import Encoder
from test_scene_jobs import scene_worker  # noqa: F401


@pytest.fixture
def benchmark(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("retrieval_benchmark_test_module", scripts / "benchmark_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(directory):
    # Fixture-created sqlite connections may be awaiting cyclic GC. Settle their
    # WAL checkpoint before taking a physical byte snapshot of the read-only source.
    gc.collect()
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


@pytest.fixture
def benchmark_inputs(scene_worker, tmp_path, monkeypatch, benchmark):  # noqa: F811
    from scenerecall import media

    worker = scene_worker(frame_count=120, with_subtitles=True)
    clip = {"id": "clip_one", "source_library": str(worker.library.root), "asset_id": worker.asset["id"],
            "start_ms": 0, "end_ms": 12000, "queries": ["蓝色天空"]}
    manifest = tmp_path / "manifest.json"
    config = tmp_path / "encoder.json"
    atomic_json(manifest, {"clips": [clip]})
    atomic_json(config, {"encoder_id": "siglip2-base-p16-256"})
    args = SimpleNamespace(manifest=manifest, encoder_config=config, encoder="siglip2-base-p16-256",
        work_dir=tmp_path / "isolated-experiment", output=tmp_path / "new-report.json",
        run_mode="cold-content", allow_local_inference=True)
    encoders = []

    class SyntheticEncoder(Encoder):
        def __init__(self, _settings_dir):
            super().__init__()
            self.metadata = {"provider": "test-double", "model_load_ms": 0}
            self.closed = False
            encoders.append(self)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(benchmark, "VisualEncoder", SyntheticEncoder)
    original_scan = media.scan_keyframes
    monkeypatch.setattr("scenerecall.keyframes.scan_keyframes", lambda *a, **kw: original_scan(
        *a, **{key: value for key, value in kw.items() if key != "sample_collector"}))
    monkeypatch.setattr("scenerecall.keyframes.extract_keyframes", media.extract_keyframes)
    return worker, clip, args, encoders


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_inference", [False, True])
@pytest.mark.parametrize("unsafe_target", ["work", "output"])
async def test_formal_source_is_untouched_even_when_unsafe_arguments_are_rejected(
        benchmark, benchmark_inputs, allow_inference, unsafe_target):
    worker, _clip, args, encoders = benchmark_inputs
    args.allow_local_inference = allow_inference
    if unsafe_target == "work":
        args.work_dir = worker.library.root / "unsafe-experiment"
    else:
        args.output = worker.library.root / "unsafe-report.json"
    original = snapshot(worker.library.root)
    with pytest.raises(ValueError, match="独立"):
        await benchmark.execute(args)
    assert snapshot(worker.library.root) == original
    assert encoders == []


@pytest.mark.asyncio
async def test_without_explicit_inference_flag_no_worker_or_runtime_is_created(benchmark, benchmark_inputs):
    worker, _clip, args, encoders = benchmark_inputs
    args.allow_local_inference = False
    original = snapshot(worker.library.root)
    result = await benchmark.execute(args)
    assert result["status"] == "not_measured"
    assert encoders == [] and not args.work_dir.exists()
    assert snapshot(worker.library.root) == original


@pytest.mark.asyncio
async def test_full_episode_range_is_rejected_before_runtime_or_inference(benchmark, benchmark_inputs):
    worker, clip, args, encoders = benchmark_inputs
    clip["end_ms"] = 120000
    atomic_json(args.manifest, {"clips": [clip]})
    original = snapshot(worker.library.root)
    with pytest.raises(ValueError, match="九十秒"):
        await benchmark.execute(args)
    assert encoders == [] and not args.work_dir.exists()
    assert snapshot(worker.library.root) == original


@pytest.mark.asyncio
async def test_core_experiment_and_replay_are_labeled_and_keep_source_library_read_only(
        benchmark, benchmark_inputs):
    worker, _clip, args, encoders = benchmark_inputs
    original = snapshot(worker.library.root)
    report = await benchmark.execute(args)
    assert report["status"] == "measured_core_only"
    row = report["rows"][0]
    assert row["status"] == "completed", row
    assert row["request_count"] == 0 and row["default_complete"] is False
    assert row["default_completed_ms"] is None
    assert row["retrieval"]["status"] == "core_ready"
    assert row["queries"] and row["queries"][0]["results"]
    assert report["quality_acceptance"] == "not_evaluated"
    assert all(encoder.closed for encoder in encoders)
    assert any(encoder.images for encoder in encoders)
    previous = len(encoders)
    args.run_mode = "receipt-replay"
    args.output = args.output.with_name("replay-report.json")
    replay = await benchmark.execute(args)
    assert replay["run_mode"] == "receipt-replay"
    assert replay["rows"][0]["status"] == "completed"
    assert not any(encoder.images for encoder in encoders[previous:])
    assert all(encoder.closed for encoder in encoders)
    assert snapshot(worker.library.root) == original


@pytest.mark.asyncio
async def test_replay_refuses_changed_range_without_marking_existing_task_failed(benchmark, benchmark_inputs):
    _worker, clip, args, _encoders = benchmark_inputs
    report = await benchmark.execute(args)
    assert report["rows"][0]["status"] == "completed"
    clip["end_ms"] = 6000
    atomic_json(args.manifest, {"clips": [clip]})
    args.run_mode = "receipt-replay"
    args.output = args.output.with_name("wrong-range-report.json")
    with pytest.raises(ValueError, match="相同原片范围"):
        await benchmark.execute(args)
    library = benchmark.Library(args.work_dir / clip["id"] / "library")
    providers = benchmark.LocalOnlyProviders(library.root / "private")
    queue = benchmark.JobQueue(library, providers, benchmark.MeasuredSearch(library.root / "indexes", providers))
    assert all(job["status"] == "completed" for job in queue.list())


@pytest.mark.asyncio
async def test_encoder_failure_closes_all_workers_and_reports_only_failed_core(
        benchmark, benchmark_inputs, monkeypatch):
    _worker, _clip, args, encoders = benchmark_inputs

    async def fail(_self, _items):
        raise ValueError("Synthetic encoding failure")

    monkeypatch.setattr(benchmark.VisualEncoder, "encode_images", fail)
    report = await benchmark.execute(args)
    assert report["status"] == "failed"
    assert report["rows"][0]["status"] == "failed"
    assert report["rows"][0]["default_complete"] is False
    assert all(encoder.closed for encoder in encoders)
    assert read_json(args.output)["rows"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_remote_provider_entrypoints_are_disabled(benchmark, tmp_path):
    provider = benchmark.LocalOnlyProviders(tmp_path / "private")
    for method in (provider._request, provider._chat):
        with pytest.raises(RuntimeError, match="禁止"):
            await method()
