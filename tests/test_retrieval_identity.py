"""Synthetic local-only identity integration; not neural accuracy calibration."""
from copy import deepcopy

import pytest

import scenerecall.keyframes  # noqa: F401 - native aliases precede patched synthetic media
from scenerecall.identity_bindings import POLICY_VERSION
from scenerecall.library import atomic_json, digest, read_json
from scenerecall.models import AnalysisInput
from scenerecall.retrieval_identity import merge, observation
from scenerecall.retrieval_store import STRATEGY, RetrievalStore
from scenerecall.visual_identity import VisualIdentityPipeline
from test_identity_bindings import calibrated  # noqa: F401
from test_retrieval_jobs import Encoder, enqueue
from test_scene_jobs import scene_worker  # noqa: F401
from test_visual_identity import FakeCCIP, visual  # noqa: F401


def synthetic_scan(monkeypatch):
    from scenerecall import media
    original = media.scan_keyframes
    monkeypatch.setattr("scenerecall.keyframes.scan_keyframes", lambda *a, **kw: original(
        *a, **{key: value for key, value in kw.items() if key != "sample_collector"}))
    monkeypatch.setattr("scenerecall.keyframes.extract_keyframes", media.extract_keyframes)


@pytest.mark.asyncio
async def test_core_runs_local_person_matching_without_generative_calls_and_incremental_reuses_visual_index(
        scene_worker, monkeypatch):  # noqa: F811
    synthetic_scan(monkeypatch)
    worker = scene_worker(frame_count=6)
    backend = FakeCCIP()
    monkeypatch.setattr("scenerecall.visual_identity._backend", lambda *_: backend)
    pipeline = VisualIdentityPipeline(worker.library)
    pipeline.configure(worker.asset["id"], {"enabled": True})
    encoder = Encoder()
    worker.queue.search.visual_encoder = encoder
    job = enqueue(worker)
    assert job["config"]["retrieval_contract_version"] == 2
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done
    assert done["retrieval"]["capabilities"]["identity"] == "completed"
    assert done["retrieval"]["capabilities"]["local_records"] == "not_requested"
    assert done["retrieval"]["identity"]["region_count"] > 0
    assert done["retrieval"]["coverage"]["processed_units"] == done["retrieval"]["coverage"]["total_units"]
    windows_before = RetrievalStore(worker.library).windows(worker.asset["id"])
    records = RetrievalStore(worker.library).records(worker.asset["id"])
    assert all(record["entities"] for record in records)
    assert all(entity["binding_status"] == "unknown" for record in records for entity in record["entities"])
    assert worker.provider.calls == []
    images, scans, detections = len(encoder.images), len(worker.scans), backend.detection_calls
    # The incremental entry does not require the generic image encoder at all.
    worker.queue.search.visual_encoder = None
    _, config = worker.queue.validate_retrieval_identity(worker.asset["id"], {"bindings": {}})
    extra = worker.queue.add("analysis", config, worker.asset["id"])
    await worker.queue.execute(extra["id"])
    result = worker.queue.get(extra["id"])
    assert result["status"] == "completed", result
    assert result["retrieval"]["identity_completed_at"]
    assert result["retrieval"]["capabilities"]["visual_index"] == "not_requested"
    assert (len(encoder.images), len(worker.scans), backend.detection_calls) == (images, scans, detections)
    assert RetrievalStore(worker.library).windows(worker.asset["id"]) == windows_before
    assert worker.queue.get(job["id"]) == done
    # Same-run retry replays immutable local observations, not the local worker.
    monkeypatch.setattr(VisualIdentityPipeline, "run", lambda *_a, **_kw: pytest.fail("identity receipt not reused"))
    worker.queue.update(extra["id"], status="failed")
    worker.queue.control(extra["id"], "retry")
    await worker.queue.execute(extra["id"])
    assert worker.queue.get(extra["id"])["status"] == "completed"


@pytest.mark.asyncio
async def test_legacy_core_keeps_no_identity_semantics_and_disabled_v2_is_explicit(scene_worker, monkeypatch):  # noqa: F811
    synthetic_scan(monkeypatch)
    worker = scene_worker(frame_count=3)
    worker.queue.search.visual_encoder = Encoder()
    monkeypatch.setattr("scenerecall.visual_identity._backend", lambda *_: pytest.fail("disabled model loaded"))
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["retrieval"]["capabilities"]["identity"] == "disabled"
    assert done["retrieval"]["identity"]["reason"] == "视觉身份识别未启用"
    with pytest.raises(ValueError, match="未启用"):
        worker.queue.validate_retrieval_identity(worker.asset["id"], {})
    _, config = worker.queue.validate(AnalysisInput(asset_id=worker.asset["id"], analysis_strategy=STRATEGY,
        retrieval_mode="core"), {"bindings": {}})
    config["retrieval_contract_version"] = 1
    config.pop("visual_identity_settings")
    config.pop("visual_identity_gallery")
    legacy = worker.queue.add("analysis", config, worker.asset["id"])
    await worker.queue.execute(legacy["id"])
    old = worker.queue.get(legacy["id"])
    assert old["status"] == "completed", old
    assert old["retrieval"]["capabilities"]["identity"] == "not_requested"
    assert not (worker.library.asset_dir(worker.asset["id"]) / "analyses" / legacy["run_id"] /
                "retrieval-local-identity.json").exists()


def current_window(library, raw_observation):
    return {"id": "rw_local", "asset_id": "asset_one", "source_fingerprint": "synthetic-video",
        "strategy": STRATEGY, "start_ms": 0, "end_ms": 2000, "kind": "visual", "summary": "", "text": "",
        "entities": [], "events": [], "evidence_type": "retrieval_window",
        "ordered_frames": deepcopy(raw_observation["ordered_frames"]),
        "evidence_frame_ids": list(raw_observation["evidence_frame_ids"]),
        "local_frame_ids": list(raw_observation["evidence_frame_ids"]), "held_frame_spans": []}


def test_local_regions_remain_distinct_candidates_and_human_decisions_project_without_vlm(calibrated):  # noqa: F811
    library, characters, _gallery, pipeline, backend, identity, _other, _refs, raw_observation = calibrated
    store = RetrievalStore(library)
    window = store.publish("asset_one", "run_before", current_window(library, raw_observation))
    original = deepcopy(window)
    local = observation(window, pipeline.reviews("asset_one"), run_id="run_local", signature="fixture")
    store.publish_identity("asset_one", "run_local", local)
    entities = store.records("asset_one")[0]["entities"]
    assert len(entities) == 2
    assert all(entity["binding_status"] == "candidate" and entity["character_id"] is None for entity in entities)
    assert all(entity["candidate_character_name"] == "Aoi" for entity in entities)
    assert entities[0]["identity_region_id"] != entities[1]["identity_region_id"]
    count = backend.feature_calls
    pipeline.review("asset_one", entities[0]["identity_region_id"], status="confirmed",
                    character_id=identity, reason="核对当前原片人物")
    characters.update("asset_one", identity, {"name": "新译名", "aliases": ["Aoi"]})
    renamed = store.records("asset_one")[0]["entities"]
    assert renamed[0]["binding_status"] == "human_confirmed" and renamed[0]["character_name"] == "新译名"
    assert renamed[1]["candidate_character_name"] == "新译名" and renamed[1]["candidate_character_aliases"] == ["Aoi"]
    assert backend.feature_calls == count
    assert store.windows("asset_one") == [original]
    store.validate_all()
    pipeline.review("asset_one", entities[1]["identity_region_id"], status="unknown", reason="排除错误候选")
    unknown = store.records("asset_one")[0]["entities"][1]
    assert unknown["candidate_character_name"] is None and unknown["candidate_character_aliases"] == []


def test_identity_sidecar_cannot_rewrite_window_or_invent_event_ownership(calibrated):  # noqa: F811
    library, _characters, _gallery, pipeline, _backend, _identity, _other, _refs, raw_observation = calibrated
    store = RetrievalStore(library)
    window = store.publish("asset_one", "run_before", current_window(library, raw_observation))
    local = observation(window, pipeline.reviews("asset_one"), run_id="run_local", signature="fixture")
    entity = raw_observation["entities"][0]
    merged = merge({**window, "entities": [entity], "events": raw_observation["events"]}, local)
    assert len(merged["entities"]) == 1  # the VLM explicitly owned both regions; no duplicate claims
    assert merged["events"] == raw_observation["events"]
    foreign = deepcopy(local)
    foreign["entities"][0]["evidence_frame_ids"] = ["outside_frame"]
    with pytest.raises(ValueError, match="局部证据"):
        store.publish_identity("asset_one", "run_local", foreign)
    store.publish_identity("asset_one", "run_local", local)
    changed = deepcopy(local)
    changed["entities"][0]["description"] = "不能覆盖原观察"
    with pytest.raises(ValueError, match="不可覆盖"):
        store.publish_identity("asset_one", "run_local", changed)
    path = library.asset_dir("asset_one") / "analyses/run_local/retrieval-identities/rw_local.json"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="激活引用无效"):
        store.records("asset_one")


@pytest.mark.asyncio
async def test_unavailable_local_worker_preserves_visual_index_but_fails_required_identity(scene_worker, monkeypatch):  # noqa: F811
    synthetic_scan(monkeypatch)
    worker = scene_worker(frame_count=3)
    worker.queue.search.visual_encoder = Encoder()
    VisualIdentityPipeline(worker.library).configure(worker.asset["id"], {"enabled": True})
    monkeypatch.setattr(VisualIdentityPipeline, "run", lambda *_a, **_kw: {
        "status": "unavailable", "degraded_reason": "显式测试本地模型不可用"})
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    failed = worker.queue.get(job["id"])
    assert failed["status"] == "failed" and "本地模型不可用" in failed["error"]
    assert failed["retrieval"]["capabilities"]["visual_index"] == "completed"
    assert failed["retrieval"]["capabilities"]["identity"] == "pending"
    assert RetrievalStore(worker.library).records(worker.asset["id"])
    assert worker.provider.calls == []


def test_identity_input_signature_changes_on_local_config_or_reference_revision(calibrated):  # noqa: F811
    from scenerecall.retrieval_identity import gallery_identity
    library, _characters, gallery, pipeline, _backend, identity, _other, refs, _raw_observation = calibrated
    before = gallery_identity(pipeline, "asset_one")
    gallery.update("asset_one", identity, refs[0]["id"], status="revoked", reason="撤销错误图")
    assert digest(gallery_identity(pipeline, "asset_one")) != digest(before)
    assert read_json(library.asset_dir("asset_one") / "identity/visual/settings.json")["calibration"]["binding_policy"] == POLICY_VERSION


@pytest.mark.asyncio
async def test_identity_pause_drains_and_resumes_without_sampling_or_visual_reencoding(scene_worker, monkeypatch):  # noqa: F811
    synthetic_scan(monkeypatch)
    worker = scene_worker(frame_count=3)
    encoder = Encoder()
    worker.queue.search.visual_encoder = encoder
    core = enqueue(worker)
    await worker.queue.execute(core["id"])
    VisualIdentityPipeline(worker.library).configure(worker.asset["id"], {"enabled": True})
    backend = FakeCCIP()
    monkeypatch.setattr("scenerecall.visual_identity._backend", lambda *_: backend)
    _, config = worker.queue.validate_retrieval_identity(worker.asset["id"], {})
    extra = worker.queue.add("analysis", config, worker.asset["id"])
    original_run = VisualIdentityPipeline.run

    def pause_once(self, *args, **kwargs):
        worker.queue.control(extra["id"], "pause")
        kwargs["checkpoint"]()
        pytest.fail("pause checkpoint must stop current local work")

    monkeypatch.setattr(VisualIdentityPipeline, "run", pause_once)
    await worker.queue.execute(extra["id"])
    assert worker.queue.get(extra["id"])["status"] == "paused"
    assert worker.queue.get(extra["id"])["retrieval"]["capabilities"]["identity"] == "pending"
    images, scans = len(encoder.images), len(worker.scans)
    monkeypatch.setattr(VisualIdentityPipeline, "run", original_run)
    worker.queue.control(extra["id"], "resume")
    await worker.queue.execute(extra["id"])
    result = worker.queue.get(extra["id"])
    assert result["status"] == "completed", result
    assert (len(encoder.images), len(worker.scans)) == (images, scans)
    # A config edit after enqueue cannot silently change the frozen matching run.
    _, changed = worker.queue.validate_retrieval_identity(worker.asset["id"], {})
    changed_job = worker.queue.add("analysis", changed, worker.asset["id"])
    VisualIdentityPipeline(worker.library).configure(worker.asset["id"], {"enabled": False})
    await worker.queue.execute(changed_job["id"])
    failure = worker.queue.get(changed_job["id"])
    assert failure["status"] == "failed" and "配置或角色参考已变化" in failure["error"]
