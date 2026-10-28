"""Execution changes must preserve dense evidence, cache identities and publication."""
# ruff: noqa: F811
import asyncio
import copy
from pathlib import Path
import threading
import time

import pytest

from scenerecall import visual_encoder_compat as compat
from scenerecall.library import atomic_json, read_json
from scenerecall.visual_index import digest
from test_retrieval_jobs import Encoder as RetrievalEncoder, enqueue
from test_scene_jobs import scene_worker  # noqa: F401
from test_visual_retrieval import Encoder, engine, evidence  # noqa: F401
from test_visual_encoder import configured, fake_command, ECHO_WORKER  # noqa: F401


@pytest.mark.asyncio
async def test_retrieval_uses_pyav_and_one_extraction_for_more_than_128_dense_frames(scene_worker, monkeypatch):
    from scenerecall import keyframes, media
    worker = scene_worker(frame_count=260)
    decoder_calls, extraction_calls = [], []
    original_scan, original_extract = media.scan_keyframes, keyframes.extract_keyframes_stream

    def scan(*args, sample_collector=None, **kwargs):
        decoder_calls.append(kwargs["decoder"])
        return original_scan(*args, **kwargs)

    def extract(*args, **kwargs):
        extraction_calls.append(len(args[1]))
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(keyframes, "scan_keyframes", scan)
    monkeypatch.setattr(keyframes, "extract_keyframes_stream", extract)
    worker.queue.search.visual_encoder = RetrievalEncoder()
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done
    assert decoder_calls == ["pyav"]
    assert extraction_calls == [260]
    assert len(worker.queue.search.visual_encoder.images) == 260
    plan = read_json(next((worker.library.asset_dir(worker.asset["id"]) / "retrieval/plans").glob("*.json")))
    assert plan["execution"]["extraction"]["decode_passes"] == 1
    assert plan["sampling_version"] == "retrieval-sampling-v1"
    assert len(plan["frames"]) == 260


class PrefetchEncoder(Encoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefetch_calls = []

    async def encode_images_prefetched(self, frames, next_frames):
        self.prefetch_calls.append(([f["id"] for f in frames], [f["id"] for f in next_frames]))
        return await self.encode_images(frames)


@pytest.mark.asyncio
async def test_index_prefetch_has_only_one_lookahead_and_keeps_incremental_resume(engine, tmp_path):
    pairs = [evidence(tmp_path, f"item-{i}", at=i * 1000 + 100) for i in range(3)]
    encoder = PrefetchEncoder()
    encoder.batch_size, encoder.fail_call = 1, 2
    with pytest.raises(RuntimeError, match="fixture encoder"):
        await engine.index_visual([r for r, _ in pairs], [f for _, f in pairs], encoder=encoder)
    assert encoder.prefetch_calls == [([pairs[0][1]["id"]], [pairs[1][1]["id"]]),
                                     ([pairs[1][1]["id"]], [pairs[2][1]["id"]])]
    encoder.fail_call = None
    done = await engine.index_visual([r for r, _ in pairs], [f for _, f in pairs], encoder=encoder)
    assert done["cache_hits"] == 1 and done["encoded_frames"] == 2 and done["records"] == 3
    assert encoder.prefetch_calls[-1] == ([pairs[2][1]["id"]], [])


@pytest.mark.asyncio
async def test_prefetched_file_mutation_is_rejected_before_cached_pixels_publish(engine, tmp_path):
    pairs = [evidence(tmp_path, f"changing-{i}", at=i * 1000 + 100) for i in range(2)]
    encoder = PrefetchEncoder()
    encoder.batch_size = 1
    encoder.on_images = lambda: Path(pairs[1][1]["path"]).write_bytes(b"changed source image")
    with pytest.raises(ValueError, match="发生变化"):
        await engine.index_visual([r for r, _ in pairs], [f for _, f in pairs], encoder=encoder)
    assert len(encoder.image_calls) == 1


def identity(implementation_hash, **changes):
    value = {"implementation": "retrieval-encoder-v1", "implementation_hash": implementation_hash,
             "weights_sha256": "same-weights", "preprocessing": "same-numerical-operations", **changes}
    return {**value, "signature": digest(value)}


class IdentityEncoder(Encoder):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def identity(self):
        return copy.deepcopy(self.value)


@pytest.mark.asyncio
async def test_verified_execution_upgrade_reads_existing_namespace_without_reencoding(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(compat, "EQUIVALENT_IMPLEMENTATION_HASHES", ())
    old = identity("old-source")
    new = identity("new-source")
    row, frame = evidence(tmp_path, "upgrade")
    await engine.index_visual([row], [frame], encoder=IdentityEncoder(old))
    monkeypatch.setattr(compat, "EQUIVALENT_IMPLEMENTATION_HASHES", ("old-source", "new-source"))
    current = IdentityEncoder(new)
    engine.visual_encoder = current
    assert (await engine.search("same picture"))["results"][0]["id"] == row["id"]
    done = await engine.index_visual([row], [frame], encoder=current)
    assert done["cache_hits"] == 1 and done["encoded_frames"] == 0
    assert not current.image_calls
    assert compat.compatible_identities(old, new)
    # Unknown source changes and different weights must still isolate caches.
    for changed in (identity("unknown-source"), identity("new-source", weights_sha256="different")):
        engine.visual_encoder = IdentityEncoder(changed)
        assert not (await engine.search("same picture"))["results"]
        assert not compat.compatible_identities(old, changed)
    assert not compat.compatible_identities(old, {**new, "signature": "tampered"})


@pytest.mark.asyncio
async def test_model_preparation_overlaps_scan_without_dummy_image_or_query(scene_worker, monkeypatch):
    from scenerecall import keyframes, media
    worker = scene_worker(frame_count=2)
    preparing = threading.Event()
    scanned = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_scan = media.scan_keyframes

    class WarmEncoder(RetrievalEncoder):
        async def prepare(self):
            preparing.set()
            await scanned.wait()

    def scan(*args, sample_collector=None, **kwargs):
        assert preparing.wait(3), "Model preparation did not overlap the scan"
        result = original_scan(*args, **kwargs)
        loop.call_soon_threadsafe(scanned.set)
        return result

    monkeypatch.setattr(keyframes, "scan_keyframes", scan)
    encoder = WarmEncoder()
    worker.queue.search.visual_encoder = encoder
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    assert worker.queue.get(job["id"])["status"] == "completed"
    assert len(encoder.images) == 2


@pytest.mark.asyncio
async def test_scan_failure_cancels_and_drains_concurrent_model_preparation(scene_worker, monkeypatch):
    from scenerecall import keyframes
    worker = scene_worker(frame_count=2)
    preparing, stopped = threading.Event(), threading.Event()

    class WarmEncoder(RetrievalEncoder):
        async def prepare(self):
            preparing.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    def fail_scan(*args, **kwargs):
        assert preparing.wait(3)
        raise ValueError("explicit scan failure")

    monkeypatch.setattr(keyframes, "scan_keyframes", fail_scan)
    worker.queue.search.visual_encoder = WarmEncoder()
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    assert worker.queue.get(job["id"])["status"] == "failed"
    assert stopped.is_set()
    assert not worker.queue.search.visual_encoder.images


@pytest.mark.asyncio
async def test_model_failure_stops_and_joins_concurrent_scan(scene_worker, monkeypatch):
    from scenerecall import keyframes
    worker = scene_worker(frame_count=2)
    loop = asyncio.get_running_loop()
    scanning, stopped = asyncio.Event(), threading.Event()

    class FailedEncoder(RetrievalEncoder):
        async def prepare(self):
            await scanning.wait()
            raise RuntimeError("explicit model preparation failure")

    def scan(*args, checkpoint=None, **kwargs):
        loop.call_soon_threadsafe(scanning.set)
        try:
            while True:
                checkpoint()
                time.sleep(.001)
        finally:
            stopped.set()

    monkeypatch.setattr(keyframes, "scan_keyframes", scan)
    worker.queue.search.visual_encoder = FailedEncoder()
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    assert worker.queue.get(job["id"])["status"] == "failed"
    assert stopped.is_set()
    assert not worker.queue.search.visual_encoder.images


@pytest.mark.asyncio
async def test_parent_prepare_loads_worker_with_no_dummy_vectors(configured, monkeypatch):
    encoder, _, _, _ = configured
    fake_command(monkeypatch, ECHO_WORKER)
    try:
        await encoder.prepare()
        assert encoder.metadata["worker_pid"]
    finally:
        await encoder.close()


def test_runtime_warmup_only_prepares_model():
    from scenerecall.visual_encoder_worker import EncoderRuntime
    runtime = EncoderRuntime.__new__(EncoderRuntime)
    runtime._prepare_images = lambda values: pytest.fail("Warmup must not encode dummy images")
    assert runtime.encode("warmup", []) == []


def test_new_retrieval_defaults_to_pyav_and_keeps_explicit_old_decoder(scene_worker):
    worker = scene_worker(frame_count=2)
    new = worker.queue.add("analysis", {"analysis_strategy": "retrieval-first-v1", "stages": []}, worker.asset["id"])
    old = worker.queue.add("analysis", {"analysis_strategy": "retrieval-first-v1", "stages": [],
                                      "keyframe_decoder_profile": "ffmpeg-v1"}, worker.asset["id"])
    assert new["config"]["keyframe_decoder_profile"] == "pyav-libavfilter-v1"
    assert old["config"]["keyframe_decoder_profile"] == "ffmpeg-v1"


@pytest.mark.asyncio
async def test_retrieval_plan_cache_isolates_decoders_and_restores_unlabelled_legacy_plan(scene_worker, monkeypatch):
    from scenerecall import keyframes, media
    worker = scene_worker(frame_count=2)
    worker.queue.search.visual_encoder = RetrievalEncoder()
    decoders = []
    original = media.scan_keyframes

    def scan(*args, sample_collector=None, **kwargs):
        decoders.append(kwargs["decoder"])
        return original(*args, **kwargs)

    monkeypatch.setattr(keyframes, "scan_keyframes", scan)
    new = enqueue(worker)
    old = worker.queue.add("analysis", {**new["config"], "keyframe_decoder_profile": "ffmpeg-v1"}, worker.asset["id"])
    pyav_plan = await worker.queue._retrieval_plan(new["id"])
    legacy_plan = await worker.queue._retrieval_plan(old["id"])
    assert pyav_plan["id"] != legacy_plan["id"]
    assert decoders == ["pyav", "ffmpeg"]
    directory = worker.library.asset_dir(worker.asset["id"]) / "retrieval/plans"
    legacy_plan.pop("decoder_profile")
    atomic_json(directory / (legacy_plan["id"] + ".json"), legacy_plan)
    assert await worker.queue._retrieval_plan(old["id"]) == legacy_plan
    assert decoders == ["pyav", "ffmpeg"]
    atomic_json(directory / (pyav_plan["id"] + ".json"), {**pyav_plan, "decoder_profile": "ffmpeg-v1"})
    with pytest.raises(ValueError, match="身份不匹配"):
        await worker.queue._retrieval_plan(new["id"])


def test_continuous_extraction_allows_long_progress_but_still_times_out_stalls(monkeypatch):
    from scenerecall import keyframes
    clock = [0]
    monkeypatch.setattr(keyframes.time, "monotonic", lambda: clock[0])
    continuous = keyframes._ExtractionWatchdog(continuous=True)
    bounded = keyframes._ExtractionWatchdog(continuous=False)
    for count, at in enumerate((300, 600, 900), start=1):
        clock[0] = at
        assert not continuous.expired(count)
    assert bounded.expired(3)
    clock[0] = 1501
    assert continuous.expired(3)
