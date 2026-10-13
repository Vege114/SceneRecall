"""Exercise selected deep inputs through real orchestration and durable storage."""
import copy

import pytest

from scenerecall.library import Library
from scenerecall.models import DeepenInput
from scenerecall.scene_preparation import validate_visual_preparation
from scenerecall.scenes import SceneStore
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker
from test_scene_storage import chunk, seed


def select_frames(worker, monkeypatch, selector):
    prepare = worker.provider.prepare_visual
    selections = []

    async def prepare_selected(identity, frames, context_frames, subtitle_context, story_context, character_context=None):
        result = await prepare(identity, frames, context_frames, subtitle_context, story_context, character_context)
        for row in result.data["chunks"]:
            row["deep_frame_ids"] = selector(row["frame_ids"])
            selections.extend(row["deep_frame_ids"])
        result.data = validate_visual_preparation(result.data, frames, set(), has_context=bool(context_frames),
            cues=[], start_ms=subtitle_context["start_ms"], end_ms=subtitle_context["end_ms"])
        return result

    monkeypatch.setattr(worker.provider, "prepare_visual", prepare_selected)
    return selections


def sent_deep_frames(worker):
    return [frame["id"] for call in worker.provider.calls if call["stage"] == "deep_analysis" for frame in call["frames"]]


@pytest.mark.asyncio
async def test_automatic_selection_retains_complete_scene_index_and_auditable_inputs(scene_worker, monkeypatch):
    worker = scene_worker(13)
    worker.provider.deep_times = {250}
    selected = select_frames(worker, monkeypatch, lambda frames: [frames[0], frames[5], frames[-1]])
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    scene = worker.store.list(worker.asset["id"])[0]
    assert sent_deep_frames(worker) == selected == scene["deep_frame_ids"]
    assert scene["provenance"]["deep_analyzed_frame_ids"] == selected
    assert len(scene["frame_ids"]) == len(scene["evidence_frame_ids"]) == 13
    assert set(scene["visual_facts"][0]["frame_ids"]).issubset(selected)
    assert counts(worker) == {"preparation": 1, "deep_analysis": 1}
    assert job["deep_request_count"] == 1
    assert all(worker.library.frame_path(worker.asset["id"], frame_id).is_file() for frame_id in scene["frame_ids"])
    calls = copy.deepcopy(worker.provider.calls)
    await execute(worker, job)
    assert worker.provider.calls == calls
    assert worker.store.get(worker.asset["id"], scene["id"])["provenance"] == scene["provenance"]


@pytest.mark.asyncio
async def test_manual_deepen_uses_all_frames_despite_automatic_selection(scene_worker, monkeypatch):
    worker = scene_worker(13)
    selected = select_frames(worker, monkeypatch, lambda frames: [frames[0], frames[5], frames[-1]])
    initial = await execute(worker, enqueue(worker))
    assert initial["status"] == "completed", initial
    scene = worker.store.list(worker.asset["id"])[0]
    config, _ = worker.queue.deepen_config(worker.asset["id"], scene["id"], DeepenInput(max_requests=10),
                                         {"bindings": {"vision": "deep"}})
    job = await execute(worker, worker.queue.add("analysis", config, worker.asset["id"]))
    assert job["status"] == "completed", job
    upgraded = worker.store.get(worker.asset["id"], scene["id"])
    assert sent_deep_frames(worker) == scene["frame_ids"]
    assert upgraded["deep_frame_ids"] == selected
    assert upgraded["provenance"]["deep_analyzed_frame_ids"] == scene["frame_ids"]
    assert upgraded["id"] == scene["id"] and upgraded["frame_ids"] == scene["frame_ids"]
    assert counts(worker) == {"preparation": 1, "deep_analysis": 1}


@pytest.mark.parametrize("select_all", [False, True])
@pytest.mark.asyncio
async def test_continued_batches_union_important_frames_without_dropping_any(scene_worker, monkeypatch, select_all):
    worker = scene_worker(60)
    worker.provider.continue_batches = True
    worker.provider.deep_times = {250, 56250}
    selected = select_frames(worker, monkeypatch,
        lambda frames: list(frames) if select_all else [frames[0], frames[1], frames[-1]])
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    scenes = worker.store.list(worker.asset["id"])
    assert len(scenes) == 1 and len(scenes[0]["frame_ids"]) == 60
    assert sent_deep_frames(worker) == selected == scenes[0]["deep_frame_ids"]
    assert scenes[0]["provenance"]["deep_analyzed_frame_ids"] == selected
    assert set(scenes[0]["representative_frame_ids"]).issubset(selected)
    assert counts(worker) == {"preparation": 2, "deep_analysis": 3 if select_all else 1}


@pytest.mark.parametrize("provider_type,part_lengths", [("codex_cli", [12, 5]), ("openai_compatible", [17]),
                                                       ("openai_compatible", [24, 1])])
@pytest.mark.asyncio
async def test_provider_specific_batch_sizes_preserve_all_selected_important_frames(scene_worker, monkeypatch,
                                                                                   provider_type, part_lengths):
    worker = scene_worker(40)
    worker.provider.deep_times = {250}
    get = worker.provider.get
    monkeypatch.setattr(worker.provider, "get", lambda identity: {**get(identity), "provider_type": provider_type})
    selected = select_frames(worker, monkeypatch, lambda frames: frames[:sum(part_lengths) - 1] + [frames[-1]])
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    calls = [call for call in worker.provider.calls if call["stage"] == "deep_analysis"]
    assert [len(call["frames"]) for call in calls] == part_lengths
    assert sent_deep_frames(worker) == selected
    scene = worker.store.list(worker.asset["id"])[0]
    assert scene["provenance"]["deep_analyzed_frame_ids"] == selected
    assert scene["provenance"]["deep_batch_size"] == (12 if provider_type == "codex_cli" else 24)
    assert job["deep_request_count"] == len(part_lengths)
    assert calls[0]["start_ms"] == 0 and calls[-1]["end_ms"] == 40000
    assert all(before["end_ms"] == after["start_ms"] for before, after in zip(calls, calls[1:]))


@pytest.mark.parametrize("selected", [["frame_2000", "frame_0"], ["frame_2000"], ["missing_frame"],
                                     ["frame_0", "frame_0"], "frame_0", [None]])
def test_scene_store_rejects_invalid_deep_selection(tmp_path, selected):
    library = Library(tmp_path / "library")
    seed(library)
    with pytest.raises(ValueError):
        SceneStore(library).publish("asset_one", "run_bad", [chunk(deep_frame_ids=selected)], 0, 10000)


def test_deep_upgrade_retains_selection_and_is_idempotent(tmp_path):
    library = Library(tmp_path / "library")
    seed(library)
    store = SceneStore(library)
    selection = ["frame_0", "frame_4000"]
    original = store.publish("asset_one", "run_one", [chunk(deep_frame_ids=selection)], 0, 10000)[0]
    updated = {key: value for key, value in original.items() if key != "deep_frame_ids"}
    updated["provenance"]["deep_analyzed_frame_ids"] = selection
    deep = store.upgrade("asset_one", "run_one", updated)
    assert deep["deep_frame_ids"] == selection
    assert store.upgrade("asset_one", "run_one", updated) == deep
    store.validate_all("asset_one")


@pytest.mark.parametrize("analyzed", [[], ["missing_frame"], ["frame_2000", "frame_0"], ["frame_0", "frame_0"], None])
def test_deep_actual_input_audit_rejects_invalid_frame_references(tmp_path, analyzed):
    library = Library(tmp_path / "library")
    seed(library)
    with pytest.raises(ValueError):
        SceneStore(library).publish("asset_one", "run_bad", [chunk(provenance={"deep_analyzed_frame_ids": analyzed})], 0, 10000)
