"""Quality contracts against synthetic frames/providers; no real-media claims."""
import asyncio
import json
from copy import deepcopy

import pytest

from scenerecall.providers import ProviderError
from scenerecall.scene_jobs import LEGACY_PROMPT_VERSIONS, bounded_memory, prompt_versions
from scenerecall.scene_resume import SceneResumeMixin
from test_scene_jobs import counts, enqueue, execute, scene_worker  # noqa: F401
from test_scene_api import scene_app  # noqa: F401
from test_scene_providers import chunk, images, manager, reply


def visible_character(frame_id, **changes):
    return {"id": "visible", "kind": "character", "description": "红衣短发人物站在门前", "position": "center",
            "name": None, "is_character": True, "character_id": "character_known", "match_confidence": .95,
            "match_reason": "短发和面部特征与已有参考一致", "observed_traits": ["短发", "红衣"],
            "evidence_frame_ids": [frame_id], "portrait_frame_id": frame_id, "portrait_box": [0, 0, 1, 1], **changes}


def known_context(reference=None):
    return {"scope_id": "cast", "revision": 0, "profiles": [{"id": "character_known", "name": "已知角色"}],
            "reference_frames": [reference] if reference else [],
            "cast_policy": {"mode": "bangumi", "allow_new_profiles": False}}


@pytest.mark.asyncio
async def test_lightweight_preparation_links_a_visible_dossier_and_supplies_reference_image(tmp_path):
    frames = images(tmp_path)
    reference = {**frames[0], "character_id": "character_known"}
    value = {"chunks": [chunk(["frame-1", "frame-2"], content_kind="narrative",
                              entities=[visible_character("frame-1")])], "unresolved": []}
    requests = []
    providers = manager(tmp_path, lambda request: requests.append(request) or reply(value))
    result = await providers.prepare_visual("scene", frames[1:], [], None, None, known_context(reference))
    entity = result.data["chunks"][0]["entities"][0]
    assert entity["character_id"] == "character_known"
    assert not result.data["chunks"][0]["needs_deep"]
    assert result.request_count == len(requests) == 1
    payload = json.loads(requests[0].content)
    assert sum(part.get("type") == "image_url" for part in payload["messages"][1]["content"]) == 2
    assert "Do not OCR or list production names" in payload["messages"][0]["content"]


@pytest.mark.parametrize("changes", [
    {"character_id": "outside_roster"}, {"match_confidence": .4}, {"match_reason": ""},
    {"evidence_frame_ids": ["old-reference"], "portrait_frame_id": None},
    {"portrait_frame_id": "frame-1"}, {"name": "已知角色"},
])
@pytest.mark.asyncio
async def test_lightweight_identity_retains_full_evidence_and_match_guards(tmp_path, changes):
    frames = images(tmp_path, 2)
    value = {"chunks": [chunk(["frame-0", "frame-1"], entities=[visible_character("frame-0", **changes)])],
             "unresolved": []}
    providers = manager(tmp_path, lambda request: reply(value))
    with pytest.raises(ProviderError):
        await providers.prepare_visual("scene", frames, [], None, None, known_context())


@pytest.mark.asyncio
async def test_pure_credits_keep_frame_coverage_but_no_narrative_facts_or_extra_deep_requests(tmp_path):
    frames = images(tmp_path, 2)
    value = {"chunks": [chunk(["frame-0", "frame-1"], content_kind="credits", summary="导演：某人",
        visual_facts=[{"text": "制作名单", "frame_ids": ["frame-0"]}], entities=[visible_character("frame-0")],
        needs_deep=True, reason="识别小字")], "unresolved": []}
    providers = manager(tmp_path, lambda request: reply(value))
    result = (await providers.prepare_visual("scene", frames, [], None, None, known_context())).data["chunks"][0]
    assert result["summary"] == "片头／片尾演职员信息"
    assert result["frame_ids"] == ["frame-0", "frame-1"]
    assert not result["needs_deep"]
    assert all(result[field] == [] for field in ("visual_facts", "dialogue_facts", "inferences", "entities"))
    assert SceneResumeMixin._understanding_kind(result, {"character_reanalysis": True}, True) == "initial"
    assert bounded_memory({}, [{**result, "id": "credit", "asset_id": "asset", "start_ms": 0, "end_ms": 2000}])["recent"] == []


@pytest.mark.asyncio
async def test_request_local_versions_preserve_legacy_instructions_without_concurrent_leakage(tmp_path):
    requests = []
    frames = images(tmp_path, 1)
    value = {"chunks": [chunk(["frame-0"])], "unresolved": []}
    providers = manager(tmp_path, lambda request: requests.append(json.loads(request.content)) or reply(value))

    async def legacy():
        with providers.scene_prompt_version(LEGACY_PROMPT_VERSIONS["preparation"]):
            await providers.prepare_visual("scene", frames, [], None, None)

    await asyncio.gather(legacy(), providers.prepare_visual("scene", frames, [], None, None))
    prompts = [payload["messages"][0]["content"] for payload in requests]
    old = next(prompt for prompt in prompts if "scene-visual-preparation-v3" in prompt)
    new = next(prompt for prompt in prompts if "scene-visual-preparation-v4" in prompt)
    assert "content_kind=credits" not in old
    assert "content_kind=credits" in new
    assert prompt_versions({}) == LEGACY_PROMPT_VERSIONS
    assert prompt_versions({"scene_dependencies": {"prompt_versions": LEGACY_PROMPT_VERSIONS}}) == LEGACY_PROMPT_VERSIONS
    assert prompt_versions({"scene_prompt_versions": prompt_versions()}) == prompt_versions()


@pytest.mark.asyncio
async def test_prepared_character_is_published_without_deep_analysis_and_survives_resume(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(2)
    original = worker.provider.prepare_visual

    async def prepare(*args):
        result = await original(*args)
        first = result.data["chunks"][0]
        first["entities"] = [visible_character(first["frame_ids"][0], character_id=None, name="合成葵",
                                             match_reason="当前画面胸牌写着合成葵")]
        return result

    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    sync = worker.queue._sync_scene_index
    interrupted = False

    def fail_after_publication(job_id, asset_id):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise OSError("Synthetic interruption after scene publication")
        return sync(job_id, asset_id)

    monkeypatch.setattr(worker.queue, "_sync_scene_index", fail_after_publication)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "failed", job
    scene = worker.store.list(worker.asset["id"])[0]
    character_id = scene["entities"][0]["character_id"]
    assert character_id
    catalog = worker.queue.characters.catalog(worker.asset["id"])
    assert catalog["profiles"][0]["id"] == character_id
    assert catalog["profiles"][0]["appearance_count"] == 1
    assert counts(worker) == {"preparation": 1}
    worker.queue.control(job["id"], "retry")
    resumed = await execute(worker, job)
    assert resumed["status"] == "completed", resumed
    assert counts(worker) == {"preparation": 1}
    assert worker.store.list(worker.asset["id"])[0]["entities"][0]["character_id"] == character_id


@pytest.mark.asyncio
async def test_older_task_restores_legacy_stage_receipts_after_quality_prompt_upgrade(scene_worker):  # noqa: F811
    worker = scene_worker(30, with_subtitles=True)
    job = enqueue(worker)
    config = deepcopy(job["config"])
    config.pop("scene_prompt_versions")
    config.pop("transcript_context", None)
    worker.queue.update(job["id"], config=config)
    worker.provider.deep_times = {250}
    worker.provider.fail_deep_at = 24250
    failed = await execute(worker, job)
    assert failed["status"] == "failed", failed
    assert failed["config"]["scene_dependencies"]["prompt_versions"] == LEGACY_PROMPT_VERSIONS
    before = counts(worker)
    worker.queue.control(job["id"], "retry")
    done = await execute(worker, failed)
    assert done["status"] == "completed", done
    assert counts(worker)["transcript"] == before["transcript"]
    assert counts(worker)["preparation"] == before["preparation"]
    assert counts(worker)["deep_analysis"] == before["deep_analysis"] + 1


@pytest.mark.asyncio
async def test_context_refresh_preserves_character_ids_allocated_during_initial_publication(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(4)
    worker.provider.chunk_size = 2
    worker.provider.deep_times = {250}
    original = worker.provider.prepare_visual

    async def prepared(*args):
        result = await original(*args)
        second = result.data["chunks"][1]
        second["entities"] = [visible_character(second["frame_ids"][0], character_id=None, name="合成葵",
                                               match_reason="当前画面胸牌写着合成葵")]
        return result

    monkeypatch.setattr(worker.provider, "prepare_visual", prepared)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    current = worker.store._raw(worker.asset["id"])[1]
    assert current["run_id"].endswith("_context")
    identity = current["entities"][0]["character_id"]
    assert identity
    profile = worker.queue.characters.catalog(worker.asset["id"])["profiles"][0]
    assert profile["id"] == identity and profile["appearance_count"] == 1


@pytest.mark.asyncio
async def test_codex_preparation_accepts_character_and_mixed_credit_partitions(tmp_path, monkeypatch):
    from scenerecall import codex_cli
    from scenerecall.providers import _vision_output_schema
    frames = images(tmp_path, 2)
    value = {"chunks": [chunk(["frame-0"], content_kind="narrative", entities=[visible_character("frame-0")]),
                        chunk(["frame-1"], content_kind="credits", entities=[])], "unresolved": []}
    calls = []

    async def complete(instruction, content, model, timeout_s, **options):
        calls.append((instruction, options))
        return codex_cli.CodexCLIResult(json.dumps(value), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    monkeypatch.setattr(codex_cli, "complete", complete)
    providers = manager(tmp_path, lambda request: pytest.fail("Codex must not use HTTP"), provider_type="codex_cli")
    result = await providers.prepare_visual("scene", frames, [], None, None, known_context())
    assert result.request_count == len(calls) == 1
    assert result.data["chunks"][0]["entities"][0]["character_id"] == "character_known"
    assert result.data["chunks"][1]["summary"] == "片头／片尾演职员信息"
    schema = _vision_output_schema({"frame-0"}, {"character_known"}, 0, 1000, named_identities=False)
    properties = schema["$defs"]["Entity"]["properties"]
    assert properties["name"] == {"type": "null"}
    assert properties["character_id"]["anyOf"][0]["enum"] == ["character_known"]


@pytest.mark.asyncio
async def test_credit_subtitle_segments_keep_original_cues_out_of_narrative_context(tmp_path):
    from scenerecall.scene_jobs import subtitle_context
    cues = [{"id": "credit", "start_ms": 0, "end_ms": 1000, "text": "导演：某人"}]
    value = {"segments": [{"content_kind": "credits", "cue_ids": ["credit"], "summary": "导演某人"}], "unresolved": []}
    providers = manager(tmp_path, lambda request: reply(value))
    result = (await providers.prepare_subtitles("scene", cues)).data
    assert result["segments"][0] == {"content_kind": "credits", "cue_ids": ["credit"], "summary": "片头／片尾演职员信息"}
    context = subtitle_context(cues, result["segments"], 0, 1000)
    assert context["cues"] == cues
    assert context["segments"] == []


@pytest.mark.parametrize("suffix", ["/estimate", ""])
def test_credit_deepening_api_rejects_before_enqueue_or_model_work(scene_app, suffix):  # noqa: F811
    app, client, _, scene = scene_app
    credits = {**scene, "content_kind": "credits", "summary": "片头／片尾演职员信息", "needs_deep": False,
               "visual_facts": [], "dialogue_facts": [], "inferences": [], "entities": []}
    from scenerecall.scenes import SceneStore
    SceneStore(app.state.library).publish("scene_asset", "credit_run", [credits], 0, 10000)
    response = client.post(f"/api/assets/scene_asset/chunks/{scene['id']}/deepen{suffix}", json={})
    assert response.status_code == 400, response.text
    assert "Bangumi" in response.json()["detail"]
    assert app.state.jobs.list() == []
    with pytest.raises(ValueError, match="Bangumi"):
        app.state.jobs.estimate_deepen({}, credits)


@pytest.mark.asyncio
async def test_already_queued_credit_deepening_stops_before_any_model_call(scene_worker):  # noqa: F811
    from scenerecall.models import DeepenInput
    worker = scene_worker(2)
    first = await execute(worker, enqueue(worker))
    assert first["status"] == "completed", first
    scene = worker.store.list(worker.asset["id"])[0]
    config, _ = worker.queue.deepen_config(worker.asset["id"], scene["id"], DeepenInput(),
                                         {"bindings": {"vision": "deep"}})
    # Emulate a request admitted by an older version before content preflight.
    scene.update(content_kind="credits", summary="片头／片尾演职员信息", needs_deep=False,
                 visual_facts=[], dialogue_facts=[], inferences=[], entities=[])
    scene = worker.store.publish(worker.asset["id"], "credit_run", [scene], 0, 2000)[0]
    config["deepen_revision"] = worker.store.revision_digest(scene)
    queued = worker.queue.add("analysis", config, worker.asset["id"])
    before = counts(worker)
    failed = await execute(worker, queued)
    assert failed["status"] == "failed" and "Bangumi" in failed["error"], failed
    assert failed["request_count"] == 0
    assert counts(worker) == before
