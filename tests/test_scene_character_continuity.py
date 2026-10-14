"""Long scene identity continuity uses fabricated frames and an explicit adapter."""
from copy import deepcopy

import pytest
from PIL import Image

from scenerecall.jobs import CharacterContextChanged, JobQueue
from scenerecall.library import Library, atomic_json, source_snapshot
from scenerecall.models import AnalysisInput
from scenerecall.providers import AIResult, VisionObservation
from scenerecall.scene_jobs import STRATEGY, annotation_digest, prompt_versions
from scenerecall.scenes import SceneStore
from scenerecall.search import SearchEngine


class IdentityProvider:
    def __init__(self):
        self.calls = []
        self.callback = None
        self.profile = {"id": "vision", "name": "Synthetic adapter", "model": "synthetic",
                        "provider_type": "openai_compatible", "base_url": "https://synthetic.invalid/v1",
                        "capabilities": ["vision"], "input_price_per_million": 1, "output_price_per_million": 1}

    def get(self, identity):
        assert identity == "vision"
        return deepcopy(self.profile)

    async def analyze(self, identity, frames, start, end, context, subtitles, memory):
        assert len(frames) + len(context["reference_frames"]) <= 32
        assert len(context["reference_frames"]) <= 8
        self.calls.append({"start_ms": start, "context": deepcopy(context)})
        candidates = [profile for profile in context["profiles"] if "红衣" in profile["description"]]
        matched = candidates[0]["id"] if candidates else None
        data = {"summary": "红衣人物站立", "entities": [{"id": "person", "kind": "person", "name": None if matched else "合成葵",
                "is_character": True, "character_id": matched, "description": "红衣短发人物",
                "match_confidence": .95,
                "match_reason": "与参考图外观一致" if matched else "当前画面胸牌显示合成葵",
                "observed_traits": ["红衣", "短发"], "evidence_frame_ids": [frames[0]["id"]],
                "portrait_frame_id": frames[0]["id"], "portrait_box": [0, 0, 1, 1]}],
                "evidence_frame_ids": [frame["id"] for frame in frames], "events": [],
                "spatial_observations": [], "uncertainties": []}
        data = VisionObservation.model_validate(data).model_dump()
        if self.callback:
            self.callback(start, context)
        return AIResult(data, {"input_tokens": 20, "output_tokens": 20, "total_tokens": 40}, 1, .01)


@pytest.fixture
def worker(tmp_path):
    library = Library(tmp_path / "library")
    asset_id = "synthetic_asset"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"Synthetic metadata-only input, never decoded.")
    atomic_json(library.asset_dir(asset_id) / "manifest.json", {
        "schema_version": "1.0", "id": asset_id, "work_id": "synthetic_work", "title": "连续人物合成测试",
        "kind": "animation", "series": "测试动画", "season": 1, "episode": 1, "version": "test",
        "duration_ms": 27000, "fingerprint": "synthetic-source", "subtitle_mode": "embedded",
        "created_at": "2026-10-13T00:00:00Z"})
    atomic_json(library.root / "private/media-locations.json", {asset_id: source_snapshot(source)})
    frames = []
    for index in range(27):
        path = library.asset_dir(asset_id) / "frames" / f"frame_{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), "red").save(path)
        frame = {"id": f"frame_{index}", "path": str(path), "at_ms": index * 1000}
        library.register_frame(asset_id, frame)
        frames.append(frame)
    provider = IdentityProvider()
    queue = JobQueue(library, provider, SearchEngine(library.root / "indexes", provider))
    _, config = queue.validate(AnalysisInput(asset_id=asset_id, stages=["vision"], max_requests=100),
                               {"bindings": {"vision": "vision"}})
    store = SceneStore(library)
    prior = store.prior_memory(asset_id, 0)
    config["scene_dependencies"] = {**store.dependencies(asset_id), "prior_memory_digest": prior["digest"],
                                    "annotation_digest": annotation_digest(store, asset_id), "strategy": STRATEGY,
                                    "prompt_versions": prompt_versions()}
    job = queue.add("analysis", config, asset_id)
    identity = store.chunk_id(asset_id, 0, 27000)
    chunk = {"id": identity, "chunk_id": identity, "asset_id": asset_id, "run_id": job["run_id"],
             "start_ms": 0, "end_ms": 27000, "summary": "红衣人物", "frame_ids": [frame["id"] for frame in frames],
             "evidence_frame_ids": [frame["id"] for frame in frames], "subtitle_ids": [],
             "visual_facts": [], "dialogue_facts": [], "inferences": [], "dependencies": config["scene_dependencies"]}
    return queue, provider, job, chunk, frames


@pytest.mark.asyncio
async def test_new_character_reused_across_long_scene_parts_without_intermediate_publication(worker):
    queue, provider, job, chunk, frames = worker
    result = await queue._deepen_scene(job["id"], chunk, frames, [], {})
    assert len(provider.calls) == 2
    assert provider.calls[0]["context"]["profiles"] == []
    identifiers = {entity["character_id"] for entity in result["entities"]}
    assert len(identifiers) == 1
    second = provider.calls[1]["context"]
    assert {profile["id"] for profile in second["profiles"]} == identifiers
    assert second["reference_frames"][0]["path"] == frames[0]["path"]
    assert len(queue.characters.catalog(chunk["asset_id"])["profiles"]) == 1
    assert queue.library.observations(chunk["asset_id"]) == []
    assert SceneStore(queue.library).list(chunk["asset_id"]) == []
    assert {entity["id"] for entity in result["entities"]} == {"p0_person", "p1_person"}
    assert (await queue._deepen_scene(job["id"], chunk, frames, [], {})) == result
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_crash_after_registering_identity_replays_receipt_without_duplicate_or_charge(worker, monkeypatch):
    queue, provider, job, chunk, frames = worker
    import scenerecall.scene_jobs as module
    write = module.atomic_json
    failed = False

    def crash(path, value):
        nonlocal failed
        if "parts" in path.parts and path.name == "0.json" and not failed:
            failed = True
            raise OSError("Synthetic crash after registry commit")
        return write(path, value)

    monkeypatch.setattr(module, "atomic_json", crash)
    with pytest.raises(OSError, match="Synthetic crash"):
        await queue._deepen_scene(job["id"], chunk, frames, [], {})
    identity = queue.characters.catalog(chunk["asset_id"])["profiles"][0]["id"]
    assert len(provider.calls) == 1 and queue.get(job["id"])["request_count"] == 1
    result = await queue._deepen_scene(job["id"], chunk, frames, [], {})
    assert {entity["character_id"] for entity in result["entities"]} == {identity}
    assert len(provider.calls) == 2 and queue.get(job["id"])["request_count"] == 2
    assert len(queue.characters.catalog(chunk["asset_id"])["profiles"]) == 1
    assert queue.get(job["id"])["cost"] == pytest.approx(.02)


@pytest.mark.asyncio
async def test_saved_part_identity_reused_after_later_part_interruption(worker):
    queue, provider, job, chunk, frames = worker
    interrupted = False

    def stop(start, context):
        nonlocal interrupted
        if start == 24000 and not interrupted:
            interrupted = True
            raise RuntimeError("Synthetic interrupted response")

    provider.callback = stop
    with pytest.raises(RuntimeError):
        await queue._deepen_scene(job["id"], chunk, frames, [], {})
    result = await queue._deepen_scene(job["id"], chunk, frames, [], {})
    assert [call["start_ms"] for call in provider.calls] == [0, 24000, 24000]
    assert len({entity["character_id"] for entity in result["entities"]}) == 1
    assert len(queue.characters.catalog(chunk["asset_id"])["profiles"]) == 1


@pytest.mark.asyncio
async def test_legacy_anonymous_part_dossier_does_not_reenter_model_context(worker, monkeypatch):
    queue, provider, job, chunk, frames = worker
    from scenerecall.library import atomic_json, read_json
    apply = queue.characters.apply_observation
    legacy_id = None

    def anonymous_after_first_part(asset_id, record, expected_revision=None):
        nonlocal legacy_id
        linked = apply(asset_id, record, expected_revision)
        if legacy_id is None:
            legacy_id = linked["entities"][0]["character_id"]
            registry = queue.characters._read(linked["character_scope_id"])
            profile = registry["profiles"][legacy_id]
            profile.update(name="红衣学生", description="红衣学生\n黑色短发", edited_fields=[], user_edited=False)
            atomic_json(queue.characters._path(registry["scope_id"]), registry)
        return linked

    monkeypatch.setattr(queue.characters, "apply_observation", anonymous_after_first_part)
    result = await queue._deepen_scene(job["id"], chunk, frames, [], {})
    assert len(provider.calls) == 2
    second = provider.calls[1]["context"]
    assert legacy_id not in {profile["id"] for profile in second["profiles"]}
    assert legacy_id not in {reference["character_id"] for reference in second["reference_frames"]}
    part_path = (queue.library.asset_dir(chunk["asset_id"]) / "analyses" / job["run_id"] /
                 "scene-deep/parts" / chunk["id"] / "0.json")
    part = read_json(part_path)
    assert part["entities"][0]["character_id"] == legacy_id
    assert part["entities"][0]["evidence_frame_ids"] == [frames[0]["id"]]
    assert part["entities"][0]["description"] == "红衣短发人物"
    assert all(entity.get("character_id") != legacy_id for entity in result["entities"])


@pytest.mark.asyncio
async def test_human_edit_during_next_part_prevents_stale_identity_commit(worker):
    queue, provider, job, chunk, frames = worker

    def edit(start, context):
        if start == 24000:
            identity = context["profiles"][0]["id"]
            queue.characters.update(chunk["asset_id"], identity, {"name": "人工修订姓名"})

    provider.callback = edit
    with pytest.raises(CharacterContextChanged):
        await queue._deepen_scene(job["id"], chunk, frames, [], {})
    assert queue.characters.catalog(chunk["asset_id"])["profiles"][0]["name"] == "人工修订姓名"
    assert not (queue.library.asset_dir(chunk["asset_id"]) / "analyses" / job["run_id"] /
                "scene-deep/parts" / chunk["id"] / "1.json").exists()
    assert SceneStore(queue.library).list(chunk["asset_id"]) == []
