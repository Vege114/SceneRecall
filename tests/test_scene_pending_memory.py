"""Unfinished deep analysis cannot masquerade as completed story memory."""
from copy import deepcopy

import pytest

from scenerecall.library import atomic_json, read_json
from scenerecall.models import DeepenInput
from test_scene_jobs import enqueue, execute
from test_scene_jobs import scene_worker as scene_worker
from test_scene_storage import chunk, seed


@pytest.mark.asyncio
async def test_failed_deep_episode_has_pending_gap_and_is_not_completed_prior_memory(scene_worker):
    worker = scene_worker(4)
    next_episode = seed(worker.library, "asset_next", episode=2, series="合成连续剧")
    manifest_path = worker.library.asset_dir(worker.asset["id"]) / "manifest.json"
    atomic_json(manifest_path, {**read_json(manifest_path), "title": "合成连续剧", "season": 1, "episode": 1, "series": "合成连续剧"})
    worker.library.organization.assign([worker.asset["id"]], next_episode["project_id"])
    worker.provider.deep_times = {250}
    worker.provider.fail_deep_at = 0
    failed = await execute(worker, enqueue(worker))
    assert failed["status"] == "failed", failed
    initial = worker.store.list(worker.asset["id"])
    assert initial[0]["needs_deep"] and not initial[0]["context_stale"]
    assert (initial[0]["start_ms"], initial[0]["end_ms"]) == (0, 4000)
    memory = worker.store.prior_memory("asset_next", 0)
    assert memory["summaries"] == memory["provenance"] == []
    assert {"asset_id": worker.asset["id"], "start_ms": 0, "end_ms": 4000,
            "reason": "pending_deep_analysis"} in memory["gaps"]
    worker.queue.control(failed["id"], "retry")
    completed = await execute(worker, failed)
    assert completed["status"] == "completed", completed
    usable = worker.store.prior_memory("asset_next", 0)
    assert [row["asset_id"] for row in usable["summaries"]] == [worker.asset["id"]]
    assert not any(gap.get("reason") == "pending_deep_analysis" for gap in usable["gaps"])


@pytest.mark.parametrize("run_ocr", [False, True])
@pytest.mark.asyncio
async def test_later_range_refuses_unfinished_prior_deep_before_any_model_or_ocr_call(scene_worker, run_ocr):
    worker = scene_worker(8)
    worker.provider.deep_times = {250}
    worker.provider.fail_deep_at = 0
    first = await execute(worker, enqueue(worker, start_ms=0, end_ms=4000))
    assert first["status"] == "failed", first
    memory = worker.store.prior_memory(worker.asset["id"], 4000)
    assert memory["summaries"] == []
    assert memory["gaps"][0]["reason"] == "pending_deep_analysis"
    calls = deepcopy(worker.provider.calls)
    scans = deepcopy(worker.scans)
    later = await execute(worker, enqueue(worker, start_ms=4000, end_ms=8000,
                          stages=["subtitle", "vision"] if run_ocr else ["vision"]))
    assert later["status"] == "failed" and later["request_count"] == 0, later
    assert "请先完成该片段的深入分析" in later["error"] and "0–4" in later["error"]
    assert worker.provider.calls == calls and worker.scans == scans


@pytest.mark.asyncio
async def test_manual_deepen_can_finish_current_pending_chunk_then_later_range_runs(scene_worker):
    worker = scene_worker(8)
    worker.provider.deep_times = {250}
    worker.provider.fail_deep_at = 0
    first = await execute(worker, enqueue(worker, start_ms=0, end_ms=4000))
    assert first["status"] == "failed", first
    pending = worker.store.list(worker.asset["id"])[0]
    config, _ = worker.queue.deepen_config(worker.asset["id"], pending["id"], DeepenInput(max_requests=10),
                                         {"bindings": {"vision": "deep"}})
    deep = await execute(worker, worker.queue.add("analysis", config, worker.asset["id"]))
    assert deep["status"] == "completed" and deep["request_count"] == 1, deep
    assert not worker.store.get(worker.asset["id"], pending["id"])["needs_deep"]
    later = await execute(worker, enqueue(worker, start_ms=4000, end_ms=8000))
    assert later["status"] == "completed", later
    last_prep = next(call for call in reversed(worker.provider.calls) if call["stage"] == "preparation")
    assert last_prep["story_context"]["prior"]["summaries"][0]["chunk_id"] == pending["id"]


def test_earlier_episode_with_any_pending_chunk_excludes_all_its_summaries(tmp_path):
    from scenerecall.library import Library
    from scenerecall.scenes import SceneStore

    library = Library(tmp_path / "library")
    seed(library)
    seed(library, "asset_next", episode=2)
    store = SceneStore(library)
    store.publish("asset_one", "run_partial_deep", [chunk(0, 5000), chunk(5000, 10000, needs_deep=True)], 0, 10000)
    memory = store.prior_memory("asset_next", 0)
    assert memory["summaries"] == memory["provenance"] == []
    assert memory["gaps"] == [{"asset_id": "asset_one", "start_ms": 5000, "end_ms": 10000,
                               "reason": "pending_deep_analysis"}]
