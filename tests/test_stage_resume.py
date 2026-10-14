"""Committed legacy stages and index builds resume after later-stage failures."""
import pytest

from scenerecall.jobs import JobQueue
from scenerecall.models import AnalysisInput
from scenerecall.providers import ProviderError
from scenerecall.search import SearchEngine
from test_jobs_review import worker as worker
from test_search import record, vector_provider


def enqueue(queue, asset, stages):
    _, config = queue.validate(AnalysisInput(asset_id=asset["id"], stages=stages,
        window_ms=1000, max_requests=100),
        {"bindings": {"vision": "test", "subtitle": "test", "embedding": "test"}})
    config["sampling_strategy"] = "uniform-v1"
    return queue.add("analysis", config, asset["id"])


@pytest.mark.asyncio
async def test_completed_ocr_skips_batch_and_track_reconstruction_after_index_failure(worker, monkeypatch):
    queue, asset, _ = worker
    queue.providers.fail_embedding = True
    job = enqueue(queue, asset, ["subtitle"])
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "failed"
    assert queue.get(job["id"])["legacy_stage_state"]["subtitle"]
    original_cues = queue.library.subtitles(asset["id"])

    def repeated(*args, **kwargs):
        raise AssertionError("completed OCR stage was replayed")

    monkeypatch.setattr("scenerecall.subtitles.candidate_frames", repeated)
    monkeypatch.setattr("scenerecall.subtitles.merge_ocr_frames", repeated)
    queue.providers.fail_embedding = False
    queue.control(job["id"], "retry")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "completed"
    assert queue.library.subtitles(asset["id"]) == original_cues
    assert queue.providers.calls == ["subtitle", "embedding", "embedding"]


@pytest.mark.asyncio
async def test_completed_vision_skips_each_window_context_after_index_failure(worker, monkeypatch):
    queue, asset, _ = worker
    queue.providers.fail_embedding = True
    job = enqueue(queue, asset, ["vision"])
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "failed"
    assert len(queue.get(job["id"])["legacy_stage_state"]["vision"]["units"]) == 4
    context = queue.character_context
    checks = []

    def counted(*args, **kwargs):
        checks.append(True)
        return context(*args, **kwargs)

    monkeypatch.setattr(queue, "character_context", counted)
    queue.providers.fail_embedding = False
    queue.control(job["id"], "retry")
    checks.clear()  # Control itself validates the requested cast revision.
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "completed"
    assert len(checks) == 2  # One entry check and one publication/index boundary.
    assert queue.providers.calls == [("vision", at) for at in (0, 1000, 2000, 3000)] + ["embedding", "embedding"]


@pytest.mark.asyncio
async def test_search_completion_checkpoint_survives_restart_and_invalidates_on_edit(tmp_path, monkeypatch):
    providers, calls = vector_provider(tmp_path)
    records = [record("v1", "拿杯子", kind="visual")]
    engine = SearchEngine(tmp_path / "indexes", providers)
    await engine.rebuild(records, "embed")
    table = engine.status()["embedding_namespaces"][0]["table_name"]
    restarted = SearchEngine(tmp_path / "indexes", providers)

    def repeated(*args, **kwargs):
        raise AssertionError("completed lexical publication was replayed")

    with monkeypatch.context() as guard:
        guard.setattr(restarted, "_write_record", repeated)
        cached = await restarted.rebuild(records, "embed")
    assert cached["resumed"] and len(calls) == 1
    assert restarted.status()["embedding_namespaces"][0]["table_name"] == table
    edited = [record("v1", "树上的鸟", kind="visual")]
    restarted.sync_asset(edited, "film-one")
    await restarted.rebuild(edited, "embed")
    assert len(calls) == 2
    assert restarted.status()["embedding_namespaces"][0]["table_name"] != table


@pytest.mark.asyncio
async def test_index_retry_skips_completed_lexical_stage_and_embedding_batches(worker, monkeypatch):
    queue, _, _ = worker
    records = [record(f"v{index}", f"第 {index} 个独立说明", kind="visual") for index in range(65)]
    monkeypatch.setattr(queue.library, "records", lambda: records)
    original_embed = queue.providers.embed
    batches = []

    async def fail_second(profile_id, texts):
        batches.append(len(texts))
        if len(batches) == 2:
            raise ProviderError("second embedding batch failed", request_count=1, cost=0)
        return await original_embed(profile_id, texts)

    monkeypatch.setattr(queue.providers, "embed", fail_second)
    job = queue.add("index", {"embedding_profile_id": "test", "max_requests": 100})
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "failed"
    assert batches == [64, 1]
    restarted = SearchEngine(queue.library.root / "indexes", queue.providers)
    queue = JobQueue(queue.library, queue.providers, restarted)
    monkeypatch.setattr(restarted, "_write_record", lambda *args: pytest.fail("lexical rebuild repeated"))
    queue.control(job["id"], "retry")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "completed"
    assert batches == [64, 1, 1]
    assert queue.get(job["id"])["request_count"] == 3


@pytest.mark.asyncio
async def test_embedding_receipt_recovers_response_and_accounting_before_cache_commit(worker, monkeypatch):
    queue, _, _ = worker
    monkeypatch.setattr(queue.library, "records", lambda: [record("v1", "杯子", kind="visual")])
    job = queue.add("index", {"embedding_profile_id": "test", "max_requests": 100})

    class ProcessExit(BaseException):
        pass

    with monkeypatch.context() as fault:
        fault.setattr(queue, "_account_scene_receipt", lambda *args: (_ for _ in ()).throw(ProcessExit()))
        with pytest.raises(ProcessExit):
            await queue.execute(job["id"])
    restarted = JobQueue(queue.library, queue.providers, queue.search)
    assert restarted.get(job["id"])["request_count"] == 1
    await restarted.execute(job["id"])
    assert restarted.get(job["id"])["status"] == "completed"
    assert restarted.get(job["id"])["request_count"] == 1
    assert queue.providers.calls == ["embedding"]
