"""Import-time playback origins survive small FFmpeg probe changes."""
import copy
import shutil
import subprocess
from fractions import Fraction

import pytest

from scenerecall import keyframes, media
from scenerecall.jobs import JobQueue
from scenerecall.library import Library, atomic_json, read_json
from scenerecall.models import AnalysisInput, AssetInput
from scenerecall.search import SearchEngine
from test_retrieval_jobs import Encoder
from test_scene_jobs import SyntheticSceneProvider


@pytest.fixture
def shifted_movie(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for synthetic timeline regressions")
    path = tmp_path / "shifted.mkv"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=128x72:rate=10:duration=2", "-c:v", "libx264",
        "-output_ts_offset", "0.005", str(path),
    ], check=True, capture_output=True)
    current = media.probe(path)
    assert current["start_time_ms"] == 5
    legacy = {**current, "start_time_ms": 0, "start_time_seconds": "0.0",
              "duration_ms": current["duration_ms"] + 5}
    return path, legacy


@pytest.mark.parametrize("decoder", ["ffmpeg", "pyav"])
def test_imported_origin_preserves_full_range_source_pts_and_exact_extraction(shifted_movie, tmp_path, decoder):
    path, legacy = shifted_movie
    current = media.probe(path)
    before = copy.deepcopy(legacy)
    metadata = media.probe_for_asset(path, legacy)
    assert legacy == before
    assert metadata["_source_identity"] == current["_source_identity"]
    assert metadata["streams"] == current["streams"]
    assert metadata["duration_ms"] == legacy["duration_ms"]
    assert Fraction(metadata["start_time_seconds"]) == 0
    # The old code failed before decoding this unchanged full-file range.
    with pytest.raises(media.MediaError, match="结束位置.*可播放范围"):
        media.scan_keyframes(path, 0, legacy["duration_ms"])
    plan = media.scan_keyframes(path, 0, legacy["duration_ms"], metadata=metadata, decoder=decoder)
    assert plan["shots"][0]["start_ms"] == 0
    assert plan["shots"][-1]["end_ms"] == legacy["duration_ms"]
    assert Fraction(plan["source_start_time_seconds"]) == 0
    for frame in plan["frames"]:
        absolute = frame["source_pts"] * Fraction(*frame["source_time_base"])
        assert frame["at_ms"] == round(absolute * 1000)
        assert frame["pts"] * Fraction(*frame["time_base"]) == absolute
    # Narrow slices retain the first/last actual source pictures, including the
    # held last picture. The imported origin must not move either by five ms.
    for start, end, expected in [(0, 10, 5), (1900, legacy["duration_ms"], 1905)]:
        partial = media.scan_keyframes(path, start, end, metadata=metadata, decoder=decoder)
        assert partial["frames"][0]["at_ms"] == expected
        frames = partial["frames"]
        extracted = media.extract_keyframes(path, frames, [tmp_path / f"{decoder}-{start}-{i}.jpg"
            for i in range(len(frames))], metadata=metadata)
        assert [f["source_pts"] for f in extracted] == [f["source_pts"] for f in frames]


@pytest.mark.parametrize("change", ["different_end", "large_shift", "wrong_format"])
def test_timeline_compatibility_does_not_extend_different_media(shifted_movie, change):
    path, legacy = shifted_movie
    if change == "different_end":
        legacy["duration_ms"] += 2
    elif change == "large_shift":
        legacy.update(start_time_ms=-1000, start_time_seconds="-1.0", duration_ms=legacy["duration_ms"] + 1000)
    else:
        # A non-Matroska file must not receive this container-specific fix.
        path = path.with_suffix(".mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(shifted_movie[0]),
                        "-c", "copy", str(path)], check=True, capture_output=True)
    assert media.probe_for_asset(path, legacy) == media.probe(path)


def test_range_and_source_validation_remain_strict(shifted_movie, tmp_path):
    path, legacy = shifted_movie
    metadata = media.probe_for_asset(path, legacy)
    with pytest.raises(media.MediaError, match="结束位置.*可播放范围"):
        media.scan_keyframes(path, 0, legacy["duration_ms"] + 1, metadata=metadata)
    with pytest.raises(media.MediaError):
        media.scan_keyframes(path, legacy["duration_ms"], legacy["duration_ms"] + 1, metadata=metadata)
    with pytest.raises(media.MediaError, match="没有实际起始时间"):
        media.scan_keyframes(path, 0, 1, metadata=metadata)
    other = tmp_path / "other.mkv"
    shutil.copy2(path, other)
    with pytest.raises(media.MediaError, match="不属于当前文件"):
        media.scan_keyframes(other, 0, legacy["duration_ms"], metadata=metadata)


def test_legacy_rounded_origin_does_not_replace_precise_plan_timestamps(monkeypatch, tmp_path):
    current = {"format_name": "matroska,webm", "start_time_seconds": "0.0054",
               "start_time_ms": 5, "duration_ms": 2000}
    monkeypatch.setattr(media, "probe", lambda _path: copy.deepcopy(current))
    legacy = {"start_time_ms": 5, "duration_ms": 2000}
    assert media.probe_for_asset(tmp_path / "synthetic.mkv", legacy) == current


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["keyframes", "retrieval"])
async def test_failed_full_movie_job_retries_without_changing_its_timeline(shifted_movie, tmp_path, monkeypatch, kind):
    path, legacy = shifted_movie
    library = Library(tmp_path / "library")
    asset = library.register(AssetInput(video_path=str(path), title="Synthetic origin drift", subtitle_mode="embedded"))
    manifest = library.asset_dir(asset["id"]) / "manifest.json"
    asset = read_json(manifest)
    asset.update({k: legacy[k] for k in ("duration_ms", "start_time_ms", "start_time_seconds")})
    atomic_json(manifest, asset)
    original_manifest = manifest.read_bytes()
    provider = SyntheticSceneProvider()
    search = SearchEngine(library.root / "indexes", provider)
    search.visual_encoder = Encoder()
    queue = JobQueue(library, provider, search)
    if kind == "keyframes":
        plan = queue.ensure_plan(asset["id"])
        job = queue.get(plan["job_id"])
    else:
        _, config = queue.validate(AnalysisInput(asset_id=asset["id"], task_pipeline="unified-v1",
            analysis_strategy="retrieval-first-v1", retrieval_mode="core"), {"bindings": {}})
        job = queue.add("analysis", config, asset["id"])
    original_config = copy.deepcopy(job["config"])
    with monkeypatch.context() as old:
        old.setattr(media, "probe_for_asset", lambda source, _asset: media.probe(source))
        await queue.execute(job["id"])
    failed = queue.get(job["id"])
    assert failed["status"] == "failed"
    assert "结束位置" in failed["error"]
    queue.control(job["id"], "retry")
    await queue.execute(job["id"])
    completed = queue.get(job["id"])
    assert completed["status"] == "completed", completed
    assert {k: completed["config"][k] for k in original_config} == original_config
    assert provider.calls == []
    if kind == "keyframes":
        plan = queue.validated_plan(asset["id"], 0, asset["duration_ms"], plan["id"])
    else:
        plan = await queue._retrieval_plan(job["id"])
        assert completed["retrieval"]["coverage"]["end_ms"] == asset["duration_ms"]
    assert all(frame["at_ms"] == round(frame["source_pts"] * Fraction(*frame["source_time_base"]) * 1000)
               for frame in plan["frames"])
    assert manifest.read_bytes() == original_manifest
    # A repaired job continues to reuse its existing plan and exact frame IDs.
    with monkeypatch.context() as cached:
        cached.setattr(keyframes, "scan_keyframes", lambda *a, **kw: pytest.fail("valid plan should be reused"))
        if kind == "keyframes":
            assert queue.ensure_plan(asset["id"])["id"] == plan["id"]
        else:
            assert (await queue._retrieval_plan(job["id"]))["frames"] == plan["frames"]
