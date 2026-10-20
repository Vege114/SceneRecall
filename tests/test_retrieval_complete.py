"""Complete retrieval contracts with synthetic media and a single transport double.

The actual provider entrypoint, schema validator, identity projection, stores,
budget reservations and queue execute. No real model or real video is accessed.
"""
import asyncio
import json
from copy import deepcopy

import pytest

from scenerecall.library import atomic_json, read_json
from scenerecall.models import AnalysisInput
from scenerecall.providers import ProviderManager
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.visual_identity import VisualIdentityPipeline
from test_retrieval_jobs import Encoder
from test_scene_jobs import SyntheticSceneProvider, scene_worker  # noqa: F401
from test_visual_identity import FakeCCIP


class RetrievalTransport(SyntheticSceneProvider):
    """Only the remote transport is replaced, not production prompt/validation."""

    analyze_retrieval = ProviderManager.analyze_retrieval
    _images = staticmethod(ProviderManager._images)

    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = None
        self.schemas = []

    async def _chat(self, profile, capability, instruction, content, validator, **options):
        payload = json.loads(content[0]["text"])
        self.calls.append({"stage": "local_records", "payload": deepcopy(payload)})
        self.schemas.append(deepcopy(options["output_schema"]))
        assert capability == "vision"
        assert "no episode story" in instruction
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        observations = []
        for unit in payload["units"]:
            ids = unit["local_frame_ids"]
            regions = [row for row in payload["identity_regions"] if row["frame_id"] in ids]
            observations.append({
                "unit_id": unit["id"], "summary": "红衣人物转身，窗外是蓝色天空",
                "evidence_frame_ids": ids,
                "entities": [{
                    "id": "P1", "kind": "person", "description": "穿红色衣服的人物",
                    "observed_traits": ["红衣"], "evidence_frame_ids": ids,
                    "identity_region_id": regions[0]["id"] if regions else None,
                    "candidate_character_ids": [payload["characters"][0]["id"]] if payload["characters"] else [],
                }],
                "events": [{
                    "action": "转身", "actor_ids": ["P1"], "temporal": len(ids) > 1,
                    "start_ms": unit["start_ms"], "end_ms": unit["end_ms"],
                    "evidence_frame_ids": ids,
                }],
            })
        return self._result(validator({"observations": observations}))


@pytest.fixture
def complete_worker(scene_worker, monkeypatch):  # noqa: F811
    from scenerecall import media

    original_scan = media.scan_keyframes
    monkeypatch.setattr("scenerecall.keyframes.scan_keyframes", lambda *a, **kw: original_scan(
        *a, **{key: value for key, value in kw.items() if key != "sample_collector"}))
    monkeypatch.setattr("scenerecall.keyframes.extract_keyframes", media.extract_keyframes)

    def create(frame_count=12, *, identity=False):
        worker = scene_worker(frame_count=frame_count, with_subtitles=True)
        worker.provider = RetrievalTransport()
        worker.queue.providers = worker.provider
        worker.queue.search.providers = worker.provider
        worker.queue.search.visual_encoder = Encoder()
        if identity:
            monkeypatch.setattr("scenerecall.visual_identity._backend", lambda _model: FakeCCIP())
            VisualIdentityPipeline(worker.library).configure(worker.asset["id"], {"enabled": True})
        return worker

    return create


def enqueue_complete(worker, **overrides):
    request = AnalysisInput(asset_id=worker.asset["id"], analysis_strategy="retrieval-first-v1",
        retrieval_mode="complete", execution={"max_concurrency": 1}, **overrides)
    _, config = worker.queue.validate(request, {"bindings": {"vision": "deep"}})
    return worker.queue.add("analysis", config, worker.asset["id"])


def raw_observations(worker, job):
    directory = worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"]
    return {path: path.read_bytes() for path in (directory / "retrieval-observations").glob("*.json")}


@pytest.mark.asyncio
async def test_complete_provider_schema_identity_store_search_chain(complete_worker):
    worker = complete_worker(identity=True)
    job = enqueue_complete(worker)
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done
    state = done["retrieval"]
    assert state["status"] == "completed"
    assert state["first_searchable_at"] <= state["default_completed_at"]
    assert set(state["capabilities"].values()) == {"completed"}
    assert state["coverage"]["processed_units"] == state["coverage"]["total_units"]
    assert len(worker.provider.calls) == 1 and done["request_count"] == 1
    assert worker.provider.schemas[0]["additionalProperties"] is False
    for definition in worker.provider.schemas[0]["$defs"].values():
        assert definition["required"] == list(definition["properties"])
    records = RetrievalStore(worker.library).records(worker.asset["id"])
    assert records and all(row["entities"][0]["binding_status"] == "unknown" for row in records)
    assert all(row["events"][0]["temporal"] for row in records)
    assert all(row["provenance"]["evidence_kind"] == "current_visual" for row in records)
    hits = await worker.queue.search.search("红衣人物转身")
    assert hits["results"]
    assert any(row.get("events") for row in hits["results"])
    RetrievalStore(worker.library).validate_all()


@pytest.mark.asyncio
async def test_complete_reports_explicitly_disabled_local_identity_instead_of_claiming_recognition(complete_worker):
    worker = complete_worker()
    job = enqueue_complete(worker)
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done
    state = done["retrieval"]
    assert state["contract_version"] == 2 and state["capabilities"]["identity"] == "disabled"
    assert state["identity"]["reason"] == "视觉身份识别未启用"
    assert state["default_completed_at"]
    assert all(value == "completed" for key, value in state["capabilities"].items() if key != "identity")


@pytest.mark.asyncio
async def test_human_identity_and_rename_reproject_without_rewriting_raw_observations(complete_worker):
    worker = complete_worker(identity=True)
    identity = worker.queue.characters.create(worker.asset["id"], {"name": "初始实名"})["id"]
    job = enqueue_complete(worker)
    await worker.queue.execute(job["id"])
    assert worker.queue.get(job["id"])["status"] == "completed", worker.queue.get(job["id"])
    originals = raw_observations(worker, job)
    store = RetrievalStore(worker.library)
    records = store.records(worker.asset["id"])
    region = records[0]["entities"][0]["identity_region_id"]
    assert region and records[0]["entities"][0]["character_id"] is None
    pipeline = VisualIdentityPipeline(worker.library)
    pipeline.review(worker.asset["id"], region, status="confirmed", character_id=identity, reason="合成测试人工核实")
    worker.queue.characters.update(worker.asset["id"], identity, {"name": "新实名", "aliases": ["旧译名"]})
    projected = store.records(worker.asset["id"])[0]
    assert projected["entities"][0]["binding_status"] == "human_confirmed"
    assert projected["entities"][0]["character_name"] == "新实名"
    assert projected["events"][0]["actor_names"] == ["新实名"]
    worker.queue.search.sync_asset(worker.library.records(worker.asset["id"]), worker.asset["id"])
    hits = await worker.queue.search.search("旧译名转身")
    assert any(row.get("entities", [{}])[0].get("character_id") == identity for row in hits["results"])
    pipeline.review(worker.asset["id"], region, status="unknown", reason="合成测试撤销")
    assert store.records(worker.asset["id"])[0]["entities"][0]["character_id"] is None
    assert raw_observations(worker, job) == originals
    assert len(worker.provider.calls) == 1


@pytest.mark.asyncio
async def test_success_receipt_replays_after_publication_failure_without_second_charge(complete_worker, monkeypatch):
    worker = complete_worker()
    job = enqueue_complete(worker)
    publish = RetrievalStore.publish_observation
    fail = True

    def fail_once(self, *args):
        nonlocal fail
        if fail:
            fail = False
            raise ValueError("Synthetic failure after durable successful receipt")
        return publish(self, *args)

    monkeypatch.setattr(RetrievalStore, "publish_observation", fail_once)
    await worker.queue.execute(job["id"])
    failed = worker.queue.get(job["id"])
    assert failed["status"] == "failed"
    assert failed["request_count"] == 1
    assert failed["retrieval"]["default_completed_at"] is None
    assert not raw_observations(worker, job)
    image_calls = len(worker.queue.search.visual_encoder.images)
    worker.queue.control(job["id"], "retry")
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done
    assert done["request_count"] == 1 and len(worker.provider.calls) == 1
    assert len(worker.queue.search.visual_encoder.images) == image_calls
    assert len(raw_observations(worker, job)) == done["retrieval"]["coverage"]["total_units"]


@pytest.mark.asyncio
async def test_budget_pause_keeps_core_results_and_resume_finishes_required_work(complete_worker):
    worker = complete_worker(frame_count=18)
    job = enqueue_complete(worker, max_requests=3)
    await worker.queue.execute(job["id"])
    paused = worker.queue.get(job["id"])
    assert paused["status"] == "paused", paused
    assert paused["request_count"] == 1
    assert paused["retrieval"]["default_completed_at"] is None
    assert paused["retrieval"]["capabilities"]["local_records"] == "pending"
    assert paused["retrieval"]["coverage"]["processed_units"] == 2
    assert (await worker.queue.search.search("蓝色天空"))["results"]
    originals = raw_observations(worker, job)
    worker.queue.control(job["id"], "resume", {"max_requests": 6})
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done
    assert done["request_count"] == 2 and len(worker.provider.calls) == 2
    assert done["retrieval"]["coverage"]["processed_units"] == 3
    assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.asyncio
async def test_cancel_drains_owned_request_and_never_marks_default_complete(complete_worker):
    worker = complete_worker(frame_count=18)
    worker.provider.release = asyncio.Event()
    job = enqueue_complete(worker)
    execution = asyncio.create_task(worker.queue.execute(job["id"]))
    try:
        await asyncio.wait_for(worker.provider.entered.wait(), timeout=10)
        worker.queue.control(job["id"], "cancel")
        worker.provider.release.set()
        await asyncio.wait_for(execution, timeout=10)
    finally:
        worker.provider.release.set()
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
    cancelled = worker.queue.get(job["id"])
    assert cancelled["status"] == "cancelled", cancelled
    assert cancelled["retrieval"]["default_completed_at"] is None
    assert cancelled["inflight_calls"] == {} and cancelled["request_count"] == 1
    assert len(worker.provider.calls) == 1
    worker.queue.control(job["id"], "retry")
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done
    assert done["request_count"] == 2 and len(worker.provider.calls) == 2


@pytest.mark.asyncio
async def test_resume_cannot_reauthenticate_modified_original_observation(complete_worker):
    worker = complete_worker()
    job = enqueue_complete(worker)
    await worker.queue.execute(job["id"])
    assert worker.queue.get(job["id"])["status"] == "completed"
    original_files = raw_observations(worker, job)
    path = next(iter(original_files))
    value = read_json(path)
    value["events"][0]["evidence_frame_ids"] = ["invented_nonexistent_frame"]
    atomic_json(path, value)
    active_path = RetrievalStore(worker.library).directory(worker.asset["id"]) / "active.json"
    original_active = read_json(active_path)
    worker.queue.update(job["id"], status="failed")
    worker.queue.control(job["id"], "retry")
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "failed", done
    assert read_json(active_path)["units"] == original_active["units"]
    assert len(worker.provider.calls) == 1
