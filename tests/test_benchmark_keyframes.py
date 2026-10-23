"""Keyframe benchmark isolation, honest comparison and variant execution contracts."""
import copy
import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from PIL import Image

from scenerecall import library as library_module


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_keyframes.py"
    spec = importlib.util.spec_from_file_location("keyframe_benchmark_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def synthetic_plan(count=2):
    frames = [{"id": f"frame_{index}", "source_pts": index * 10, "source_time_base": [1, 1000],
               "pts": index * 10, "time_base": [1, 1000], "at_ms": index * 10,
               "shot_id": "shot_one", "reason": "representative_pose"} for index in range(count)]
    batches = []
    for offset in range(0, count, 8):
        end = min(offset + 8, count)
        batches.append({"id": f"batch_{offset}", "shot_id": "shot_one", "start_ms": offset * 10,
                        "end_ms": end * 10 if end < count else 2000,
                        "frame_ids": [frame["id"] for frame in frames[offset:end]]})
    return {"algorithm": "content-keyframes-v1", "algorithm_parameters": {"synthetic_fixture": True},
            "source_start_time_seconds": "0", "shots": [{"id": "shot_one", "start_ms": 0, "end_ms": 2000}],
            "frames": frames, "batches": batches}


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "original.bin"
    source.write_bytes(b"synthetic file; the media functions are replaced in these tests")
    source_library = tmp_path / "source-library"
    manifest = {"id": "asset_one", "work_id": "work_one", "title": "Synthetic fixture",
                "created_at": "2026-10-23T00:00:00+00:00", "duration_ms": 2000, "version": "1",
                "fingerprint": "fixture-fingerprint", "start_time_seconds": "0", "start_time_ms": 0}
    library_module.atomic_json(source_library / "assets" / "asset_one" / "manifest.json", manifest)
    library_module.atomic_json(source_library / "private" / "media-locations.json",
                               {"asset_one": library_module.source_snapshot(source)})
    args = SimpleNamespace(library=source_library, asset_id="asset_one", start_ms=0, end_ms=2000,
                           variant="A", opencv_threads=None, output_dir=tmp_path / "trial-A", baseline_root=None)
    return args


@pytest.fixture
def runtime(monkeypatch):
    calls = {"scan": [], "extract": [], "threads": [], "register": [], "probe": []}
    plan = synthetic_plan()

    def scan(*args, **kwargs):
        calls["scan"].append((args, kwargs))
        return copy.deepcopy(plan)

    def extract(source, frames, outputs, **kwargs):
        calls["extract"].append((source, copy.deepcopy(frames), list(outputs), kwargs))
        for index, path in enumerate(outputs):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (index * 10, 20, 30)).save(path)
        return [{**frame, "path": str(path)} for frame, path in zip(frames, outputs, strict=True)]

    def probe(source):
        calls["probe"].append(source)
        return {"duration_ms": 2000, "test_metadata": True}

    original = library_module.Library.register_frames

    def register_frames(self, asset_id, frames):
        calls["register"].append(len(frames))
        return original(self, asset_id, frames)

    monkeypatch.setattr(library_module.Library, "register_frames", register_frames)
    cv2 = SimpleNamespace(__version__="fixture", getNumThreads=lambda: calls["threads"][-1] if calls["threads"] else 32,
                          setNumThreads=calls["threads"].append)
    value = SimpleNamespace(library=library_module, keyframes=SimpleNamespace(scan_keyframes=scan,
                            extract_keyframes=extract), media=SimpleNamespace(probe=probe), cv2=cv2, psutil=psutil,
                            root=Path(__file__).parents[1] / "backend", calls=calls, plan=plan)
    return value


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_variants_keep_source_read_only_and_publish_valid_precise_plans(benchmark, inputs, runtime, variant):
    inputs.variant = variant
    before = snapshot(inputs.library)
    result = benchmark.execute(inputs, runtime)
    assert result["status"] == "completed"
    assert snapshot(inputs.library) == before
    assert result["source_metadata_unchanged"]
    assert result["frame_count"] == 2 and result["remote_model_calls"] == 0
    assert result["registry_write_calls"] == (2 if variant == "A" else 1)
    assert runtime.calls["register"] == ([1, 1] if variant == "A" else [2])
    assert runtime.calls["threads"] == ([] if variant == "A" else [1])
    assert len(runtime.calls["probe"]) == (0 if variant == "A" else 1)
    options = runtime.calls["scan"][0][1]
    assert options == ({} if variant == "A" else {"metadata": {"duration_ms": 2000, "test_metadata": True},
                                                  "decoder": "pyav" if variant == "C" else "ffmpeg"})
    plan = benchmark.read_json(inputs.output_dir / "plan.json")
    library = library_module.Library(inputs.output_dir / "library")
    assert library.validate_keyframe_plan(inputs.asset_id, plan) == plan
    assert all(image["sha256"] == benchmark.file_hash(inputs.output_dir / image["path"])
               for image in result["images"])
    assert benchmark.read_json(inputs.output_dir / "metrics.json") == result
    assert result["processing_wall_ms"] <= result["trial_wall_ms"]


@pytest.mark.parametrize("unsafe", ["inside", "ancestor", "existing"])
def test_unsafe_outputs_fail_before_writes(benchmark, inputs, runtime, unsafe):
    before = snapshot(inputs.library)
    if unsafe == "inside":
        inputs.output_dir = inputs.library / "trial"
    elif unsafe == "ancestor":
        inputs.output_dir = inputs.library.parent
    else:
        inputs.output_dir.mkdir()
        (inputs.output_dir / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="independent|new or empty"):
        benchmark.execute(inputs, runtime)
    assert snapshot(inputs.library) == before
    assert not runtime.calls["scan"] and not runtime.calls["threads"]


@pytest.mark.parametrize("failure", ["changed_source", "range", "a_threads"])
def test_invalid_input_never_starts_scan(benchmark, inputs, runtime, failure):
    if failure == "changed_source":
        (inputs.library.parent / "original.bin").write_bytes(b"changed fixture")
    elif failure == "range":
        inputs.end_ms = 2001
    else:
        inputs.opencv_threads = 1
    with pytest.raises(ValueError):
        benchmark.execute(inputs, runtime)
    assert not inputs.output_dir.exists() and not runtime.calls["scan"]


def test_extraction_failure_records_failure_without_completed_plan(benchmark, inputs, runtime):
    before = snapshot(inputs.library)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic extraction failure")

    runtime.keyframes.extract_keyframes = fail
    with pytest.raises(RuntimeError, match="synthetic extraction"):
        benchmark.execute(inputs, runtime)
    result = benchmark.read_json(inputs.output_dir / "metrics.json")
    assert result["status"] == "failed" and result["error_type"] == "RuntimeError"
    assert "processing_wall_ms" not in result
    assert not (inputs.output_dir / "plan.json").exists()
    assert snapshot(inputs.library) == before


def test_extraction_chunks_and_registration_are_bounded(benchmark, inputs, runtime):
    inputs.variant = "B"
    runtime.plan.update(synthetic_plan(129))
    result = benchmark.execute(inputs, runtime)
    assert [len(call[1]) for call in runtime.calls["extract"]] == [128, 1]
    assert runtime.calls["register"] == [128, 1]
    assert result["frame_count"] == 129 and result["extraction_batches"] == 2


def test_compare_preserves_substantive_fields_and_image_hashes(benchmark, inputs, runtime):
    first = inputs.output_dir
    benchmark.execute(inputs, runtime)
    inputs.variant = "B"
    inputs.output_dir = first.with_name("trial-B")
    benchmark.execute(inputs, runtime)
    second = inputs.output_dir
    result = benchmark.compare_runs(first, second, tolerance_ms=40)
    assert result["exact_equivalence"]
    assert result["acceptance"] == "requires_repeated_speed_measurements"
    plan = benchmark.read_json(second / "plan.json")
    plan["frames"][0]["reason"] = "changed-selection-reason"
    library_module.atomic_json(second / "plan.json", plan)
    result = benchmark.compare_runs(first, second, tolerance_ms=40)
    assert result["different_plan_fields"] == ["frames"]
    assert not result["exact_equivalence"]
    assert result["baseline_to_candidate_nearest"]["max_ms"] == 0
    assert result["acceptance"] == "not_equivalent_requires_review"
    plan["frames"][0]["reason"] = "representative_pose"
    library_module.atomic_json(second / "plan.json", plan)
    metrics = benchmark.read_json(second / "metrics.json")
    metrics["images"][0]["sha256"] = "different-bytes"
    library_module.atomic_json(second / "metrics.json", metrics)
    result = benchmark.compare_runs(first, second)
    assert result["plan_equal"] and not result["image_hashes_equal"] and not result["exact_equivalence"]
    metrics["input"]["end_ms"] -= 1
    library_module.atomic_json(second / "metrics.json", metrics)
    with pytest.raises(ValueError, match="different source"):
        benchmark.compare_runs(first, second)


def test_time_diagnostics_use_exact_pts_and_do_not_claim_one_to_one(benchmark):
    frames = [{"source_pts": 1, "source_time_base": [1, 30]},
              {"source_pts": 2, "source_time_base": [1, 30]}]
    assert benchmark.exact_time(frames[0]) == Fraction(1, 30)
    values = benchmark.nearest_distances(frames, [frames[0]])
    assert values == pytest.approx([0, 1000 / 30])
    assert benchmark.distance_summary(values, 34)["within_tolerance"] == 2


def test_final_comparison_rehashes_existing_images_and_rejects_changed_or_missing_files(benchmark, inputs, runtime):
    report = benchmark.execute(inputs, runtime)
    comparison = benchmark.compare_runs(inputs.output_dir, inputs.output_dir, verify_image_files=True)
    assert comparison["exact_equivalence"]
    assert comparison["image_verification"] == "existing_files_rehashed"
    assert comparison["verified_image_count"] == {"baseline": 2, "candidate": 2}
    path = inputs.output_dir / report["images"][0]["path"]
    path.write_bytes(b"explicit changed preview fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        benchmark.compare_runs(inputs.output_dir, inputs.output_dir, verify_image_files=True)
    path.unlink()
    with pytest.raises(ValueError, match="Missing preview"):
        benchmark.compare_runs(inputs.output_dir, inputs.output_dir, verify_image_files=True)


def test_contact_sheet_reads_only_confined_trial_files(benchmark, inputs, runtime):
    benchmark.execute(inputs, runtime)
    output = inputs.output_dir.parent / "sheet.jpg"
    report = benchmark.contact_sheet(inputs.output_dir, inputs.output_dir, output)
    assert report["rows"] == 2
    with Image.open(output) as image:
        assert image.size == (660, 456)
    metrics = benchmark.read_json(inputs.output_dir / "metrics.json")
    metrics["images"][0]["path"] = "../outside.jpg"
    library_module.atomic_json(inputs.output_dir / "metrics.json", metrics)
    with pytest.raises(ValueError, match="escapes"):
        benchmark.contact_sheet(inputs.output_dir, inputs.output_dir, output.with_name("unsafe.jpg"))


def test_decoder_counter_restores_functions_after_error(benchmark):
    closed = []

    def decode():
        try:
            yield "one"
            raise RuntimeError("decoder failed")
        finally:
            closed.append(True)

    module = SimpleNamespace(_decode=decode)
    with pytest.raises(RuntimeError, match="decoder failed"):
        with benchmark.DecodeCounter(module) as counter:
            list(module._decode())
    assert module._decode is decode and counter.frames == 1 and closed == [True]


def test_frozen_baseline_selects_entire_package_in_fresh_process(benchmark, tmp_path):
    root = tmp_path / "frozen" / "backend"
    package = root / "scenerecall"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("library", "keyframes", "media"):
        (package / f"{name}.py").write_text(f"MARKER = 'frozen-{name}'\n", encoding="utf-8")
    code = "\n".join([
        "import sys,json", "from types import SimpleNamespace", "from pathlib import Path",
        "sys.path.insert(0, sys.argv[1])", "import benchmark_keyframes as benchmark",
        "runtime=benchmark.load_runtime(SimpleNamespace(variant='A', baseline_root=Path(sys.argv[2])))",
        "print(json.dumps([runtime.library.MARKER,runtime.media.MARKER,runtime.keyframes.MARKER]))",
    ])
    result = subprocess.run([sys.executable, "-c", code, str(Path(benchmark.__file__).parent), str(root.parent)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["frozen-library", "frozen-media", "frozen-keyframes"]


def test_runtime_refuses_mixed_variant_imports(benchmark, inputs):
    with pytest.raises(RuntimeError, match="fresh Python process"):
        benchmark.load_runtime(inputs)
