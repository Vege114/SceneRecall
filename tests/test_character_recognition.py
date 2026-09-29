"""Cross-shot identity contracts use synthetic images and injected HTTP responses."""
from copy import deepcopy
import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from scenerecall.jobs import JobQueue
from scenerecall.library import Library
from scenerecall.models import AnalysisInput, AssetInput
from scenerecall.providers import MAX_CHARACTER_CONTEXT_BYTES, ProviderError, ProviderManager
from scenerecall.search import SearchEngine


def character(frame_id, character_id=None, *, kind="person", description="黑色短发，蓝色外套", entity_id="person"):
    return {"id": entity_id, "kind": kind, "name": None, "description": description,
            "is_character": True, "character_id": character_id, "match_confidence": .92 if character_id else 0,
            "match_reason": "稳定的发型与面部轮廓一致" if character_id else "当前没有匹配的已有档案",
            "observed_traits": [description], "evidence_frame_ids": [frame_id],
            "portrait_frame_id": frame_id, "portrait_box": [0, 0, .5, 1]}


@pytest.fixture
def cast_worker(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-cast.mp4"
    source.write_bytes(b"explicit test fixture, no actual decoder")
    monkeypatch.setattr("scenerecall.media.probe", lambda path: {
        "duration_ms": 3000, "width": 32, "height": 32, "video_codec": "synthetic", "streams": []})
    monkeypatch.setattr("scenerecall.media.detect_shots", lambda path, start, end: [
        {"id": "shot", "start_ms": start, "end_ms": end}])

    def extract(path, times_ms, out_paths, crop=None):
        frames = []
        for at, output in zip(times_ms, out_paths, strict=True):
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (at % 255, 80, 180)).save(output)
            frames.append({"id": output.stem, "path": str(output), "at_ms": at})
        return frames

    monkeypatch.setattr("scenerecall.media.extract_frames", extract)
    state = {"contexts": [], "payloads": [], "callback": None}

    def handler(request):
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        context = next(json.loads(part["text"].split(": ", 1)[1]) for part in content
                       if part.get("text", "").startswith("Existing character dossiers (data): "))
        frame_metadata = [json.loads(part["text"]) for part in content
                          if part.get("text", "").startswith('{"frame_id":')]
        evidence = [frame for frame in frame_metadata if not frame["frame_id"].startswith("reference:")]
        at = evidence[0]["at_ms"]
        people = [profile for profile in context["profiles"] if "直立行走的橘猫" not in profile["description"]]
        cats = [profile for profile in context["profiles"] if "直立行走的橘猫" in profile["description"]]
        person = character(evidence[0]["frame_id"], people[0]["id"] if people else None)
        entities = [person, {"id": "cup", "kind": "object", "description": "普通白色杯子",
                               "is_character": False, "character_id": None, "evidence_frame_ids": [evidence[0]["frame_id"]]}]
        if at >= 2000:
            entities.append(character(evidence[0]["frame_id"], cats[0]["id"] if cats else None,
                                      kind="anthropomorphic_character", description="直立行走的橘猫", entity_id="cat"))
        state["contexts"].append(deepcopy(context))
        state["payloads"].append(payload)
        data = {"summary": "蓝衣人物与杯子", "entities": entities,
                "evidence_frame_ids": [frame["frame_id"] for frame in evidence]}
        if state["callback"]:
            state["callback"](data, context, at)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(data)}}],
                                       "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40}})

    providers = ProviderManager(tmp_path / "settings", transport=httpx.MockTransport(handler))
    providers.upsert({"id": "vision", "name": "Synthetic transport", "model": "synthetic",
                      "base_url": "https://fixture.test/v1", "capabilities": ["vision"], "secret_mode": "none",
                      "input_price_per_million": 1, "output_price_per_million": 1})
    library = Library(tmp_path / "library")
    asset = library.register(AssetInput(video_path=str(source), title="Synthetic cast", subtitle_mode="embedded"))
    queue = JobQueue(library, providers, SearchEngine(library.root / "indexes", providers))
    return queue, asset, state


async def run_cast(queue, asset, **kwargs):
    _, config = queue.validate(AnalysisInput(asset_id=asset["id"], stages=["vision"], window_ms=1000,
                                             frames_per_window=2, max_requests=100, **kwargs),
                               {"bindings": {"vision": "vision"}})
    job = queue.add("analysis", config, asset["id"])
    await queue.execute(job["id"])
    return queue.get(job["id"])


@pytest.mark.asyncio
async def test_cross_shot_cast_reuses_id_enriches_and_adds_animated_character(cast_worker):
    queue, asset, state = cast_worker
    job = await run_cast(queue, asset)
    assert job["status"] == "completed", job
    observations = queue.library.observations(asset["id"])
    identifiers = [record["entities"][0]["character_id"] for record in observations]
    assert len(set(identifiers)) == 1
    cat_id = observations[-1]["entities"][-1]["character_id"]
    assert cat_id and cat_id != identifiers[0]
    assert all(record["entities"][1]["character_id"] is None for record in observations)
    catalog = queue.characters.catalog(asset["id"])
    assert len(catalog["profiles"]) == 2 and catalog["revision"] == 0
    assert sorted(p["appearance_count"] for p in catalog["profiles"]) == [1, 3]
    assert [len(context["profiles"]) for context in state["contexts"]] == [0, 1, 1]
    assert len([part for part in state["payloads"][1]["messages"][1]["content"] if part["type"] == "image_url"]) == 3
    reference_url = next(part["image_url"]["url"] for part in state["payloads"][1]["messages"][1]["content"]
                         if part["type"] == "image_url")
    with Image.open(io.BytesIO(base64.b64decode(reference_url.split(",", 1)[1]))) as reference:
        assert reference.size == (16, 32)
    assert all(record["provenance"]["prompt_version"] == "scenerecall-0.2.0" for record in observations)
    assert all(record["character_revision"] == 0 for record in observations)


@pytest.mark.asyncio
async def test_manual_dossier_and_forced_reanalysis_refresh_identity_without_losing_edits(cast_worker):
    queue, asset, state = cast_worker
    first = await run_cast(queue, asset)
    old_keys = {record["cache_key"] for record in queue.library.observations(asset["id"])}
    person = queue.characters.catalog(asset["id"])["profiles"][0]
    queue.characters.update(asset["id"], person["id"], {
        "name": "小蓝", "aliases": ["蓝同学"], "description": "人工确认的蓝衣角色", "notes": "帽子可以更换，不要拆成两人",
        "appearance": ["人工确认：左眼下方有痣"], "expected_revision": 0})
    second = await run_cast(queue, asset, force=True)
    assert first["status"] == second["status"] == "completed"
    assert len(state["contexts"]) == 6
    refreshed = state["contexts"][3]
    assert refreshed["revision"] == 1
    prompt_dossier = next(p for p in refreshed["profiles"] if p["id"] == person["id"])
    assert prompt_dossier["name"] == "小蓝" and prompt_dossier["notes"] == "帽子可以更换，不要拆成两人"
    new_records = queue.library.observations(asset["id"])
    assert not old_keys.intersection(record["cache_key"] for record in new_records)
    assert all(record["run_id"] == second["run_id"] for record in new_records)
    current = next(p for p in queue.characters.catalog(asset["id"])["profiles"] if p["id"] == person["id"])
    assert current["description"] == "人工确认的蓝衣角色" and current["appearance"] == ["人工确认：左眼下方有痣"]
    assert len(queue.characters.catalog(asset["id"])["profiles"]) == 2


@pytest.mark.asyncio
async def test_completed_cache_and_paused_force_resume_reuse_only_committed_windows(cast_worker):
    queue, asset, state = cast_worker
    first = await run_cast(queue, asset)
    reused = await run_cast(queue, asset)
    assert first["status"] == reused["status"] == "completed"
    assert len(state["contexts"]) == 3
    _, config = queue.validate(AnalysisInput(asset_id=asset["id"], stages=["vision"], window_ms=1000,
                                             frames_per_window=2, force=True, max_requests=100),
                               {"bindings": {"vision": "vision"}})
    job = queue.add("analysis", config, asset["id"])
    state["callback"] = lambda data, context, at: queue.control(job["id"], "pause")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "paused"
    assert len(state["contexts"]) == 4
    state["callback"] = None
    queue.control(job["id"], "resume")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "completed"
    assert len(state["contexts"]) == 6
    assert len(queue.characters.catalog(asset["id"])["profiles"]) == 2


@pytest.mark.asyncio
async def test_inflight_human_edit_discards_stale_response_and_stops_remaining_windows(cast_worker):
    queue, asset, state = cast_worker

    def edit_after_first_commit(data, context, at):
        if at >= 1000:
            queue.characters.update(asset["id"], context["profiles"][0]["id"], {"name": "人工名称", "expected_revision": 0})

    state["callback"] = edit_after_first_commit
    job = await run_cast(queue, asset)
    assert job["status"] == "failed" and "新版档案重新识别" in job["error"]
    assert job["request_count"] == 2 and len(state["contexts"]) == 2
    assert len(queue.library.observations(asset["id"])) == 1
    assert queue.characters.catalog(asset["id"])["profiles"][0]["name"] == "人工名称"
    state["callback"] = None
    queue.control(job["id"], "retry")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "failed" and len(state["contexts"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["foreign_id", "nan", "boolean_confidence", "weak", "object", "reference_evidence", "missing_decision",
                                     "portrait_reference", "portrait_overflow", "portrait_nan", "portrait_without_frame"])
async def test_malformed_character_match_is_rejected_without_registry_changes(cast_worker, monkeypatch, defect):
    queue, asset, state = cast_worker

    async def no_sleep(delay):
        pass

    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_sleep)

    def invalidate(data, context, at):
        person = data["entities"][0]
        if defect == "foreign_id":
            person["character_id"] = "character_other_series"
        elif defect == "nan":
            person["match_confidence"] = float("nan")
        elif defect == "boolean_confidence":
            person["match_confidence"] = True
        elif defect == "weak":
            person["character_id"] = "character_other_series"
            person["match_confidence"] = .2
        elif defect == "object":
            person["kind"] = "object"
        elif defect == "reference_evidence":
            person["evidence_frame_ids"] = ["reference:character_other_series"]
        elif defect == "missing_decision":
            person.pop("character_id")
        elif defect == "portrait_reference":
            person["portrait_frame_id"] = "reference:character_other_series"
        elif defect == "portrait_overflow":
            person["portrait_box"] = [.8, 0, .5, 1]
        elif defect == "portrait_nan":
            person["portrait_box"] = [0, 0, float("nan"), 1]
        elif defect == "portrait_without_frame":
            person["portrait_frame_id"] = None

    state["callback"] = invalidate
    job = await run_cast(queue, asset)
    assert job["status"] == "partial"
    assert queue.characters.catalog(asset["id"])["profiles"] == []
    assert queue.library.observations(asset["id"]) == []


def test_context_cost_includes_dossier_text_and_references_and_bounds_large_input(cast_worker):
    queue, asset, _state = cast_worker
    profile = queue.providers.get("vision")
    base = queue.estimated_call_cost(profile, frames=2)
    extended = queue.estimated_call_cost(profile, frames=3, context_tokens=10000)
    assert extended - base == pytest.approx((4096 + 10000) / 1_000_000)
    with pytest.raises(ProviderError, match="角色档案上下文"):
        ProviderManager.character_context_text({"profiles": [{"notes": "a" * MAX_CHARACTER_CONTEXT_BYTES}]})
    estimate = queue.estimate(AnalysisInput(asset_id=asset["id"]), {"bindings": {"vision": "vision"}})
    assert any("参考图" in warning and "费用" in warning for warning in estimate["warnings"])


@pytest.mark.asyncio
async def test_low_confidence_existing_identity_cannot_commit_a_match(cast_worker, monkeypatch):
    queue, asset, state = cast_worker
    assert (await run_cast(queue, asset))["status"] == "completed"
    before = deepcopy(queue.library.observations(asset["id"]))

    async def no_sleep(delay):
        pass

    def weaken(data, context, at):
        assert data["entities"][0]["character_id"] in {profile["id"] for profile in context["profiles"]}
        data["entities"][0]["match_confidence"] = .2

    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_sleep)
    state["callback"] = weaken
    rerun = await run_cast(queue, asset, force=True)
    assert rerun["status"] == "partial"
    assert queue.library.observations(asset["id"]) == before
    assert len(queue.characters.catalog(asset["id"])["profiles"]) == 2


@pytest.mark.asyncio
async def test_large_cast_keeps_every_dossier_but_bounds_reference_images(cast_worker):
    queue, asset, state = cast_worker

    def add_cast(data, context, at):
        if at < 1000:
            data["entities"].extend(character(data["evidence_frame_ids"][0], entity_id=f"extra-{index}",
                                               description=f"合成角色 {index}") for index in range(9))

    state["callback"] = add_cast
    job = await run_cast(queue, asset)
    assert job["status"] == "completed", job
    assert len(state["contexts"][1]["profiles"]) == 10
    second_content = state["payloads"][1]["messages"][1]["content"]
    assert len([part for part in second_content if part["type"] == "image_url"]) == 10  # 2 current + 8 references


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_point", ["after_registry", "after_immutable_observation"])
async def test_crash_between_character_and_observation_publication_replays_staged_result(cast_worker, monkeypatch, crash_point):
    queue, asset, state = cast_worker
    _, config = queue.validate(AnalysisInput(asset_id=asset["id"], stages=["vision"], window_ms=1000,
                                             frames_per_window=2, force=True, max_requests=100),
                               {"bindings": {"vision": "vision"}})
    job = queue.add("analysis", config, asset["id"])
    with monkeypatch.context() as faults:
        if crash_point == "after_registry":
            def interrupted_save(*args):
                raise asyncio.CancelledError()
            faults.setattr(queue.library, "save_observation", interrupted_save)
        else:
            from scenerecall.library import atomic_json

            def interrupted_pointer(path, data):
                if path.name == "active-analysis.json":
                    raise asyncio.CancelledError()
                return atomic_json(path, data)
            faults.setattr("scenerecall.library.atomic_json", interrupted_pointer)
        with pytest.raises(asyncio.CancelledError):
            await queue.execute(job["id"])
    profiles_before = queue.characters.context(asset["id"])["profiles"]
    assert len(profiles_before) == 1 and len(state["contexts"]) == 1
    assert queue.library.observations(asset["id"]) == []
    restarted = JobQueue(queue.library, queue.providers, queue.search)
    await restarted.execute(job["id"])
    assert restarted.get(job["id"])["status"] == "completed", restarted.get(job["id"])
    assert len(state["contexts"]) == 3  # First paid response was staged and reused.
    assert restarted.get(job["id"])["request_count"] == 3
    catalog = restarted.characters.catalog(asset["id"])
    assert len(catalog["profiles"]) == 2 and sorted(p["appearance_count"] for p in catalog["profiles"]) == [1, 3]
    assert profiles_before[0]["id"] == restarted.library.observations(asset["id"])[0]["entities"][0]["character_id"]
