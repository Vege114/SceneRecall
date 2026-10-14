"""Repair synthetic paid observation caches without issuing replacement requests."""
from copy import deepcopy

import pytest

from scenerecall.jobs import JobQueue
from scenerecall.library import atomic_json, read_json
from scenerecall.models import Observation
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


def character_observations(worker, monkeypatch):
    analyze = worker.provider.analyze

    async def observe(identity, frames, start, end, *contexts):
        result = await analyze(identity, frames, start, end, *contexts)
        result.data.update(
            entities=[{"id": "actor", "kind": "person", "description": "合成蓝衣人物",
                       "is_character": True, "name": "小蓝", "character_id": None, "match_confidence": .9,
                       "match_reason": "合成台词明确称呼：小蓝", "observed_traits": ["合成蓝衣"],
                       "evidence_frame_ids": [frames[1]["id"]],
                       "portrait_frame_id": frames[1]["id"], "portrait_box": [0, 0, 1, 1]}],
            events=[{"actor_ids": ["actor"], "target_ids": [], "action": "合成转身动作",
                     "start_ms": frames[1]["at_ms"], "end_ms": frames[1]["at_ms"],
                     "evidence_frame_ids": [frames[1]["id"]]}],
            spatial_observations=[{"subject_id": "actor", "object_id": None,
                                   "relation": "位于合成画面左侧", "evidence_frame_ids": [frames[2]["id"]]}],
        )
        return result

    monkeypatch.setattr(worker.provider, "analyze", observe)


def forbid_model_requests(worker, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic recovery must not issue a model request")

    for method in ("prepare_subtitles", "prepare_visual", "analyze", "reconcile_scene"):
        monkeypatch.setattr(worker.provider, method, forbidden)


async def interrupted_after_character_registration(worker, monkeypatch):
    """Leave the genuine paid receipt and registry, but no completed part file."""
    character_observations(worker, monkeypatch)
    worker.provider.deep_times = {250}
    job = enqueue(worker)
    apply = worker.queue.characters.apply_observation

    def interrupted(*args, **kwargs):
        apply(*args, **kwargs)
        raise RuntimeError("Explicit synthetic interruption after registry persistence")

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue.characters, "apply_observation", interrupted)
        failed = await execute(worker, job)
    assert failed["status"] == "failed", failed
    directory = worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"]
    deep_receipts = [path for path in (directory / "scene-receipts").glob("*.json")
                     if read_json(path)["stage"] == "deep_analysis"]
    assert len(deep_receipts) == 1
    assert not list((directory / "scene-deep" / "parts").glob("*/*.json"))
    scope = failed["config"]["character_snapshot"]["scope_id"]
    characters = set(worker.queue.characters._read(scope)["profiles"])
    assert len(characters) == 1
    return failed, directory, deep_receipts[0], characters


@pytest.mark.asyncio
async def test_old_paid_receipt_repairs_sparse_top_level_evidence_without_repaying_any_stage(
        scene_worker, monkeypatch):
    worker = scene_worker(8, with_subtitles=True)
    failed, directory, path, characters = await interrupted_after_character_registration(worker, monkeypatch)
    receipt = read_json(path)
    receipt["data"]["evidence_frame_ids"] = [receipt["data"]["evidence_frame_ids"][0]]
    atomic_json(path, receipt)
    original_bytes = path.read_bytes()
    original_calls = deepcopy(worker.provider.calls)
    original_accounting = {key: failed[key] for key in ("request_count", "usage", "known_cost", "scene_receipts")}
    assert counts(worker) == {"transcript": 1, "preparation": 1, "deep_analysis": 1}

    # Recreate the queue so recovery cannot rely on in-memory normalized data.
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    forbid_model_requests(worker, monkeypatch)
    worker.queue.control(failed["id"], "retry")
    completed = await execute(worker, failed)

    assert completed["status"] == "completed", completed
    assert {key: completed[key] for key in original_accounting} == original_accounting
    assert worker.provider.calls == original_calls and path.read_bytes() == original_bytes
    scene = worker.store.list(worker.asset["id"])[0]
    assert {entity["character_id"] for entity in scene["entities"]} == characters
    assert {entry["id"] for entry in worker.queue.characters.catalog(worker.asset["id"])["profiles"]} == characters
    part_path = next((directory / "scene-deep" / "parts").glob("*/*.json"))
    part = read_json(part_path)
    Observation.model_validate(part)
    current_ids = part["input_frame_ids"]
    assert part["evidence_frame_ids"] == current_ids[:3]
    assert part["entities"][0]["evidence_frame_ids"] == [current_ids[1]]
    assert part["events"][0]["evidence_frame_ids"] == [current_ids[1]]
    assert part["spatial_observations"][0]["evidence_frame_ids"] == [current_ids[2]]


@pytest.mark.parametrize("section", ["entities", "events", "spatial_observations"])
@pytest.mark.asyncio
async def test_cached_evidence_from_another_sub_batch_is_rejected_even_within_same_scene(
        scene_worker, monkeypatch, section):
    worker = scene_worker(30, with_subtitles=True)
    failed, directory, path, characters = await interrupted_after_character_registration(worker, monkeypatch)
    scene = worker.store.list(worker.asset["id"])[0]
    assert len(scene["frame_ids"]) == 30
    outside_current_batch = scene["frame_ids"][24]
    worker.library.frame_path(worker.asset["id"], outside_current_batch)
    receipt = read_json(path)
    assert outside_current_batch not in receipt["data"]["evidence_frame_ids"]
    receipt["data"][section][0]["evidence_frame_ids"] = [outside_current_batch]
    atomic_json(path, receipt)
    original_bytes, original_calls = path.read_bytes(), deepcopy(worker.provider.calls)
    original_count, original_usage = failed["request_count"], failed["usage"]
    forbid_model_requests(worker, monkeypatch)
    worker.queue.control(failed["id"], "retry")
    rejected = await execute(worker, failed)

    assert rejected["status"] == "failed", rejected
    assert rejected["request_count"] == original_count and rejected["usage"] == original_usage
    assert path.read_bytes() == original_bytes and worker.provider.calls == original_calls
    assert not list((directory / "scene-deep" / "parts").glob("*/*.json"))
    scope = rejected["config"]["character_snapshot"]["scope_id"]
    assert set(worker.queue.characters._read(scope)["profiles"]) == characters


@pytest.mark.asyncio
async def test_sparse_saved_part_replays_with_stable_characters_and_unchanged_paid_receipts(
        scene_worker, monkeypatch):
    worker = scene_worker(8, with_subtitles=True)
    character_observations(worker, monkeypatch)
    worker.provider.deep_times = {250}
    done = await execute(worker, enqueue(worker))
    assert done["status"] == "completed", done
    directory = worker.library.asset_dir(worker.asset["id"]) / "analyses" / done["run_id"]
    original_receipts = {path: path.read_bytes() for path in (directory / "scene-receipts").glob("*.json")}
    scene = worker.store.list(worker.asset["id"])[0]
    characters = {entity["character_id"] for entity in scene["entities"]}
    part_path = next((directory / "scene-deep" / "parts").glob("*/*.json"))
    part = read_json(part_path)
    part["evidence_frame_ids"] = [part["input_frame_ids"][0]]
    atomic_json(part_path, part)
    (directory / "scene-deep" / f"{scene['id']}.json").unlink()
    original_calls = deepcopy(worker.provider.calls)
    forbid_model_requests(worker, monkeypatch)
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    # Simulate interruption after the part was durable, before the final target.
    worker.queue.update(done["id"], status="failed")
    worker.queue.control(done["id"], "retry")
    replayed = await execute(worker, done)

    assert replayed["status"] == "completed", replayed
    assert replayed["request_count"] == done["request_count"] and replayed["usage"] == done["usage"]
    assert worker.provider.calls == original_calls
    assert all(path.read_bytes() == content for path, content in original_receipts.items())
    current = worker.store.list(worker.asset["id"])[0]
    Observation.model_validate(current)
    assert {entity["character_id"] for entity in current["entities"]} == characters
    assert {entry["id"] for entry in worker.queue.characters.catalog(worker.asset["id"])["profiles"]} == characters
