"""Evidence integration contracts; fixtures are not animation-accuracy benchmarks."""
from copy import deepcopy

import pytest

from scenerecall.identity_scene import bind_identity_evidence, identity_context, project_actions, project_audio_evidence
from scenerecall.scene_jobs import namespace_entities, split_visual_args
from scenerecall.scene_preparation import validate_visual_preparation
from scenerecall.scene_resume import SceneResumeMixin
from test_scene_jobs import enqueue, scene_worker  # noqa: F401
from test_scene_providers import chunk


def region(identity="r1", frame="f1", character="c19", status="candidate"):
    return {"id": identity, "entity_id": identity, "local_id": "P1", "frame_id": frame, "at_ms": 0,
            "box": [0, 0, .5, 1], "quality": {"accepted": True},
            "match": {"status": status, "character_id": character if status == "confirmed" else None,
                      "candidate_character_id": character if status == "candidate" else None,
                      "source": "ccip", "distance": .05, "margin": .1,
                      "candidates": [{"character_id": character, "distance": .05, "reference_ids": ["sample"]}]}}


def entity(**extra):
    return {"id": "P1", "kind": "person", "description": "举手", "evidence_frame_ids": ["f1"],
            "is_character": True, "character_id": "c19", "name": "不应将候选写成确认姓名",
            "match_confidence": .99, "identity_region_id": "r1", **extra}


def test_full_roster_before_topk_and_frame_local_identifiers():
    context = {"profiles": [{"id": f"c{i}", "name": f"角色{i}", "updated_at": str(99-i)} for i in range(24)]}
    result = {"status": "completed", "regions": [region(), region("r2", "f2", "c23")],
              "references": [{"character_id": "c19", "reference_id": "external", "path": "/reference.png"}]}
    selected = identity_context(context, result, [{"id": "f1"}], top_k=1)
    assert [p["id"] for p in selected["profiles"]] == ["c19"]
    assert [r["id"] for r in selected["identity_regions"]] == ["r1"]
    assert len(context["profiles"]) == 24
    value = bind_identity_evidence([entity()], selected)[0]
    assert value["character_id"] is None and value["name"] is None
    assert value["candidate_character_id"] == "c19"
    assert value["match_confidence"] is None and value["identity_match"]["distance"] == .05


def test_confirmed_regions_require_current_visual_evidence():
    context = {"identity_regions": [region(status="confirmed")]}
    confirmed = bind_identity_evidence([entity()], context)[0]
    assert confirmed["character_id"] == "c19" and confirmed["association_status"] == "confirmed"
    with pytest.raises(ValueError, match="当前画面"):
        bind_identity_evidence([entity(evidence_frame_ids=["f2"])], context)
    unknown = bind_identity_evidence([entity(identity_region_id="other")], context)[0]
    assert unknown["character_id"] is None


def test_identity_only_review_does_not_deepen_and_event_ids_follow_namespace():
    row = chunk(["f1"], needs_deep=True, needs_action_deep=False, needs_identity_review=True,
                entities=[entity(character_id=None)], events=[{"actor_ids": ["P1"], "target_ids": [],
                "action": "举手", "start_ms": 0, "end_ms": 500, "evidence_frame_ids": ["f1"]}])
    validated = validate_visual_preparation({"chunks": [row], "unresolved": []}, [{"id": "f1", "at_ms": 0}],
                                           set(), has_context=False, start_ms=0, end_ms=1000)["chunks"][0]
    assert not validated["needs_deep"]
    assert SceneResumeMixin._understanding_kind(validated, {}, False) == "initial"
    namespace_entities(validated, "batch_")
    assert validated["events"][0]["actor_ids"] == ["batch_P1"]
    projected = project_actions(validated)
    assert projected["events"][0]["actor_names"] == ["batch_P1"]
    assert row["events"][0]["actor_ids"] == ["P1"]


def test_fallback_keeps_audio_and_region_evidence_within_child_range():
    frames = [{"id": "f1", "at_ms": 0}, {"id": "f2", "at_ms": 1000}]
    args = [frames, [], {"cues": [], "start_ms": 0, "end_ms": 2000,
                        "speaker_dialogue": [{"start_ms": 900, "end_ms": 1100}]}, {},
            {"identity_regions": [region(), region("r2", "f2")], "region_images": []}]
    left, right = split_visual_args(args, 1)
    assert [r["id"] for r in left[4]["identity_regions"]] == ["r1"]
    assert [r["id"] for r in right[4]["identity_regions"]] == ["r2"]
    assert len(left[2]["speaker_dialogue"]) == len(right[2]["speaker_dialogue"]) == 1


def test_audio_projection_revocation_preserves_timing_and_raw_evidence():
    evidence = [{"id": "voice1", "character_id": "old", "association_status": "confirmed",
                 "start_ms": 500, "end_ms": 1500, "text": "原台词", "audio_ref": "original.wav"}]
    projected = project_audio_evidence(evidence, [{"id": "voice1", "character_id": "corrected",
        "character_name": "角色乙", "association_status": "confirmed", "association_source": "human"}])
    assert projected[0]["character_id"] == "corrected"
    assert projected[0]["audio_ref"] == "original.wav" and projected[0]["start_ms"] == 500
    assert evidence[0]["character_id"] == "old"
    revoked = project_audio_evidence(evidence, [])
    assert revoked[0]["association_status"] == "unknown" and revoked[0]["character_id"] is None


def test_resumed_prompts_refresh_identity_without_recomputing(scene_worker, monkeypatch):  # noqa: F811
    from scenerecall.audio_identity import AudioIdentityService
    from scenerecall.visual_identity import VisualIdentityPipeline
    from scenerecall.library import atomic_json
    worker = scene_worker(1)
    job = enqueue(worker)
    saved = {"visual": {"status": "completed", "regions": [region(status="confirmed")]},
        "audio": {"status": "completed", "records": [{"id": "voice1", "start_ms": 0, "end_ms": 1000,
            "character_id": "old", "association_status": "confirmed", "text": "翻译台词",
            "dialogue_attribution": "ambiguous", "phonetic_alignment": False}]}}
    atomic_json(worker.queue._identity_path(job), saved)
    seen = []
    def project(_self, asset_id, original):
        seen.append(asset_id)
        return {**original, "regions": [region(status="unknown", character=None)]}
    monkeypatch.setattr(VisualIdentityPipeline, "project_result", project)
    monkeypatch.setattr(AudioIdentityService, "records", lambda *a: [{"id": "voice1", "character_id": None,
        "association_status": "unknown", "association_source": "human"}])
    visual = worker.queue.scene_identity_context(job["id"], {}, [{"id": "f1"}])
    audio = worker.queue.scene_audio_context(job["id"], {}, 0, 1000)
    assert seen == [worker.asset["id"]]
    assert visual["identity_regions"][0]["match"]["character_id"] is None
    assert audio["speaker_dialogue"][0]["character_id"] is None
    assert audio["speaker_dialogue"][0]["dialogue_attribution"] == "ambiguous"
    assert audio["speaker_dialogue"][0]["phonetic_alignment"] is False


def test_identity_invalidation_is_limited_to_referenced_inferences(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(1)
    rows = [{"id": "affected", "entities": [{"identity_region_id": "r1"}], "inferences": [{"text": "推断"}]},
            {"id": "mapping_only", "entities": [{"identity_region_id": "r1"}]},
            {"id": "unrelated", "entities": [{"identity_region_id": "r2"}], "inferences": [{"text": "推断"}]},
            {"id": "heard", "audio_evidence": [{"id": "a1"}], "inferences": [{"text": "推断"}]}]
    monkeypatch.setattr(worker.store, "_raw", lambda _: rows)
    worker.store.invalidate_identity_inferences(worker.asset["id"], region_ids=["r1"], audio_ids=["a1"])
    marked = worker.queue.library.active(worker.asset["id"])["scene_staleness"]
    assert set(marked) == {"affected", "heard"}


@pytest.mark.asyncio
async def test_failed_frozen_request_refreshes_mapping_on_retry(scene_worker, monkeypatch):  # noqa: F811
    from scenerecall.providers import ProviderError
    worker = scene_worker(1)
    revision, seen = ["before_correction"], []
    original_context = worker.queue.scene_identity_context
    def context(job_id, value, frames):
        return {**original_context(job_id, value, frames), "test_identity_revision": revision[0]}
    monkeypatch.setattr(worker.queue, "scene_identity_context", context)
    original = worker.provider.prepare_visual
    async def prepare(*args, **kwargs):
        seen.append(args[5]["test_identity_revision"])
        if len(seen) == 1:
            raise ProviderError("合成可恢复故障", request_count=1, cost=.001)
        return await original(*args, **kwargs)
    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    assert worker.queue.get(job["id"])["status"] == "failed"
    revision[0] = "human_correction"
    worker.queue.control(job["id"], "retry")
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done.get("error")
    assert seen == ["before_correction", "human_correction"]
    history = worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"] / "scene-input-history"
    assert list(history.glob("*.json"))


@pytest.mark.asyncio
async def test_identity_prelude_before_scene_and_resume_without_recomputation(scene_worker, monkeypatch):  # noqa: F811
    from scenerecall.visual_identity import VisualIdentityPipeline
    from scenerecall.audio_identity import AudioIdentityService
    worker = scene_worker(2, with_subtitles=True)
    calls = []
    def visual(_self, asset, frames, **kwargs):
        calls.append("visual")
        kwargs["checkpoint"]()
        return {"status": "completed", "regions": [], "references": [], "selected_profiles": []}
    def audio(_self, asset, *args, **kwargs):
        calls.append("audio")
        kwargs["checkpoint"]()
        return {"status": "completed", "records": [{"id": "voice1", "speaker_id": asset + ":track0:speaker0",
            "start_ms": 0, "end_ms": 1500, "text": "字幕时间关联", "character_id": None,
            "candidate_character_id": None, "association_status": "unknown", "overlap": True,
            "subtitle_ids": [kwargs["subtitles"][0]["id"]]}]}
    monkeypatch.setattr(VisualIdentityPipeline, "run", visual)
    monkeypatch.setattr(AudioIdentityService, "run", audio)
    original = worker.provider.prepare_visual
    async def prepare(*args, **kwargs):
        assert set(calls) == {"visual", "audio"}
        assert args[3]["speaker_dialogue"][0]["overlap"]
        assert args[5]["identity_regions"] == []
        return await original(*args, **kwargs)
    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done.get("error")
    scenes = worker.store.list(worker.asset["id"])
    assert scenes[0]["audio_evidence"][0]["speaker_id"].startswith(worker.asset["id"])
    assert not scenes[0].get("entities")  # Offscreen audio did not make a person visible.
    plan = worker.queue.validated_plan(worker.asset["id"], 0, 2000, done["config"]["keyframe_plan_id"])
    await worker.queue.prepare_scene_identity(job["id"], plan["frames"], [])
    assert sorted(calls) == ["audio", "visual"]


@pytest.mark.asyncio
async def test_pure_character_rename_updates_projection_without_staling_new_scene(scene_worker):  # noqa: F811
    worker = scene_worker(1)
    created = worker.queue.characters.create(worker.asset["id"], {"name": "小葵"})
    identity = created["id"]
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    assert worker.queue.get(job["id"])["status"] == "completed"
    original = deepcopy(worker.store._raw(worker.asset["id"]))
    worker.queue.characters.update(worker.asset["id"], identity, {"name": "葵"})
    assert not worker.store.list(worker.asset["id"])[0]["stale"]
    assert worker.store._raw(worker.asset["id"]) == original
    worker.queue.characters.update(worker.asset["id"], identity, {"appearance": ["人工修正发色"]})
    assert worker.store.list(worker.asset["id"])[0]["stale"]
