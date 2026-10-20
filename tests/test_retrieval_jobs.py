import pytest

import scenerecall.keyframes  # noqa: F401 - load native media aliases before synthetic fixtures patch media
from scenerecall.models import AnalysisInput
from scenerecall.retrieval_store import RetrievalStore
from test_scene_jobs import scene_worker  # noqa: F401


class Encoder:
    batch_size = 16

    def __init__(self):
        self.images = []

    def identity(self):
        return {"repo": "explicit-test-double", "revision": "test", "weights": "test", "device": "test"}

    async def encode_images(self, items):
        self.images.extend(item["id"] for item in items)
        return [[1., 0.] for _ in items]

    async def encode_texts(self, texts):
        return [[1., 0.] for _ in texts]


def enqueue(worker, mode="core"):
    _, config = worker.queue.validate(AnalysisInput(asset_id=worker.asset["id"],
        analysis_strategy="retrieval-first-v1", retrieval_mode=mode), {"bindings": {"vision": "deep"}})
    return worker.queue.add("analysis", config, worker.asset["id"])


@pytest.mark.asyncio
async def test_core_pipeline_publishes_empty_caption_and_preserves_default_pending(scene_worker, monkeypatch):  # noqa: F811
    from scenerecall import media
    worker = scene_worker(frame_count=12, with_subtitles=True)
    original_scan = media.scan_keyframes
    monkeypatch.setattr("scenerecall.keyframes.scan_keyframes", lambda *a, **kw: original_scan(
        *a, **{k: v for k, v in kw.items() if k != "sample_collector"}))
    monkeypatch.setattr("scenerecall.keyframes.extract_keyframes", media.extract_keyframes)
    encoder = Encoder()
    worker.queue.search.visual_encoder = encoder
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done
    assert done["retrieval"]["status"] == "core_ready"
    assert done["retrieval"]["default_completed_at"] is None
    assert done["retrieval"]["first_searchable_at"]
    assert done["retrieval"]["capabilities"]["local_records"] == "not_requested"
    assert worker.provider.calls == []
    records = [r for r in worker.library.records() if r.get("evidence_type") == "retrieval_window"]
    assert records and all(r["text"] == "" for r in records)
    hits = await worker.queue.search.search("纯背景画面")
    assert hits["results"][0]["id"] in {r["id"] for r in records}
    hit = hits["results"][0]
    assert hit["seek_ms"] == next(f["at_ms"] for f in hit["ordered_frames"] if f["id"] == hit["hit_frame_id"])
    assert hit["seek_ms"] % 1000 == 250
    first_images = len(encoder.images)
    worker.queue.update(job["id"], status="failed")
    worker.queue.control(job["id"], "retry")
    await worker.queue.execute(job["id"])
    assert worker.queue.get(job["id"])["status"] == "completed"
    assert len(encoder.images) == first_images
    RetrievalStore(worker.library).validate_all()


def test_new_strategy_requires_encoder_but_no_vlm_for_core(scene_worker):  # noqa: F811
    worker = scene_worker()
    request = AnalysisInput(asset_id=worker.asset["id"], analysis_strategy="retrieval-first-v1", retrieval_mode="core")
    with pytest.raises(ValueError, match="编码器"):
        worker.queue.validate(request, {"bindings": {}})
    worker.queue.search.visual_encoder = Encoder()
    _, config = worker.queue.validate(request, {"bindings": {}})
    assert config["profile_snapshots"] == {}
    assert config["speech_activity"] == "off"
    assert "scene_dependencies" not in config and "prior_memory" not in config


@pytest.mark.asyncio
async def test_failed_visual_stage_does_not_claim_default_completed(scene_worker, monkeypatch):  # noqa: F811
    from scenerecall import media
    worker = scene_worker(frame_count=6)
    original_scan = media.scan_keyframes
    monkeypatch.setattr("scenerecall.keyframes.scan_keyframes", lambda *a, **kw: original_scan(
        *a, **{k: v for k, v in kw.items() if k != "sample_collector"}))
    monkeypatch.setattr("scenerecall.keyframes.extract_keyframes", media.extract_keyframes)
    encoder = Encoder()
    async def fail(_items):
        raise ValueError("显式模型失败")
    encoder.encode_images = fail
    worker.queue.search.visual_encoder = encoder
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    failed = worker.queue.get(job["id"])
    assert failed["status"] == "failed"
    assert failed["retrieval"]["default_completed_at"] is None
    assert failed["retrieval"]["capabilities"]["visual_index"] == "pending"
