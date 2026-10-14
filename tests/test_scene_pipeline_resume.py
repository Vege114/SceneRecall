"""Published scene work resumes without replaying completed understanding units.

All media, model calls and storage are synthetic and confined to pytest's tmp_path.
The spies deliberately reject local receipt replay and repeated publication, not
only duplicate provider calls: avoiding paid calls alone is not a fast resume.
"""
import asyncio
import copy

import pytest

from scenerecall.library import atomic_json, read_json
from scenerecall.models import DeepenInput
from scenerecall.scenes import SceneStore
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker


def state_path(worker, job):
    return worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"] / "scene-state.json"


def make_story(scene_worker, frames=20):
    worker = scene_worker(frames, with_subtitles=True)
    worker.provider.chunk_size = 4
    worker.provider.deep_times = {250}
    return worker


def published_understanding(worker, job):
    return {chunk["id"] for chunk in worker.store.list(worker.asset["id"])
            if chunk["run_id"] == job["run_id"] + "_context"
            or (chunk["run_id"] == job["run_id"] and chunk["understanding_level"] == "deep")}


def forbid_completed_work(worker, monkeypatch, completed_ids, *, forbid_all=False):
    """Reject completed units even when their paid receipt would be a cache hit."""
    deepen = worker.queue._deepen_scene
    scene_call = worker.queue._scene_call
    upgrade = SceneStore.upgrade
    publish = SceneStore.publish
    visits = []

    async def checked_deepen(job_id, chunk, *args, **kwargs):
        assert not forbid_all and chunk["id"] not in completed_ids, "completed deep analysis replayed"
        visits.append(("deepen", chunk["id"]))
        return await deepen(job_id, chunk, *args, **kwargs)

    async def checked_call(job_id, stage, unit, *args, **kwargs):
        assert stage not in {"transcript", "preparation"}, "completed preparation replayed"
        assert not forbid_all and unit not in completed_ids, "completed understanding receipt replayed"
        visits.append((stage, unit))
        return await scene_call(job_id, stage, unit, *args, **kwargs)

    def checked_upgrade(store, asset_id, run_id, chunk, **kwargs):
        assert not forbid_all and chunk["id"] not in completed_ids, "completed deep revision republished"
        visits.append(("upgrade", chunk["id"]))
        return upgrade(store, asset_id, run_id, chunk, **kwargs)

    def checked_publish(store, asset_id, run_id, chunks, *args, **kwargs):
        assert not forbid_all and not completed_ids.intersection(chunk["id"] for chunk in chunks), \
            "completed understanding revision republished"
        visits.extend(("publish", chunk["id"]) for chunk in chunks)
        return publish(store, asset_id, run_id, chunks, *args, **kwargs)

    monkeypatch.setattr(worker.queue, "_deepen_scene", checked_deepen)
    monkeypatch.setattr(worker.queue, "_scene_call", checked_call)
    monkeypatch.setattr(SceneStore, "upgrade", checked_upgrade)
    monkeypatch.setattr(SceneStore, "publish", checked_publish)
    return visits


@pytest.mark.parametrize("action", ["pause", "cancel"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.asyncio
async def test_resume_refresh_skips_published_deep_and_earlier_refresh(scene_worker, monkeypatch, action, legacy):
    worker = make_story(scene_worker)
    job = enqueue(worker)
    started, release = asyncio.Event(), asyncio.Event()
    reconcile = worker.provider.reconcile_scene
    refreshes = 0

    async def block_second_refresh(*args, **kwargs):
        nonlocal refreshes
        refreshes += 1
        if refreshes == 2:
            started.set()
            await release.wait()
        return await reconcile(*args, **kwargs)

    monkeypatch.setattr(worker.provider, "reconcile_scene", block_second_refresh)
    running = asyncio.create_task(worker.queue.execute(job["id"]))
    await asyncio.wait_for(started.wait(), 5)
    worker.queue.control(job["id"], action)
    release.set()
    await running
    stopped = worker.queue.get(job["id"])
    assert stopped["status"] == {"pause": "paused", "cancel": "cancelled"}[action], stopped
    completed_ids = published_understanding(worker, job)
    assert len(completed_ids) >= 2
    saved = read_json(state_path(worker, job))
    if legacy:
        saved.pop("understanding", None)
        atomic_json(state_path(worker, job), saved)
    before_pointers = copy.deepcopy(worker.library.active(worker.asset["id"])["scenes"])
    pending_index = saved.get("index_dirty", False) or legacy
    sync = worker.queue._sync_scene_index
    sync_states = []

    def checked_sync(*args, **kwargs):
        pointers = copy.deepcopy(worker.library.active(worker.asset["id"])["scenes"])
        sync_states.append(pointers)
        if pointers == before_pointers:
            assert pending_index and sync_states == [pointers], "completed units rebuilt their index"
        return sync(*args, **kwargs)

    visits = forbid_completed_work(worker, monkeypatch, completed_ids)
    monkeypatch.setattr(worker.queue, "_sync_scene_index", checked_sync)
    worker.queue.control(job["id"], "resume")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert counts(worker) == {"transcript": 1, "preparation": 1, "deep_analysis": 1, "context_refresh": 4}
    assert all(stage != "deepen" for stage, _ in visits)
    assert completed["stage_progress"]["deep_analysis"] == {"completed": 1, "total": 1, "status": "completed"}
    assert completed["stage_progress"]["context_refresh"] == {"completed": 4, "total": 4, "status": "completed"}
    assert completed["stage_metrics"]["deep_analysis"] == stopped["stage_metrics"]["deep_analysis"]
    assert completed["stage_metrics"]["preparation"] == stopped["stage_metrics"]["preparation"]
    scenes = worker.store.list(worker.asset["id"])
    assert all(not scene["stale"] for scene in scenes)
    refreshed = [call for call in worker.provider.calls if call["stage"] == "context_refresh"]
    assert all("深入核实" in str(call["story_context"]) for call in refreshed)
    assert any("合成场景结合前序深入结果更新理解" in str(call["story_context"]) for call in refreshed[1:])


@pytest.mark.parametrize("failure", ["after_upgrade", "before_publish", "after_publish", "index"])
@pytest.mark.asyncio
async def test_publication_or_index_failure_resumes_only_unfinished_work(scene_worker, monkeypatch, failure):
    worker = make_story(scene_worker, 16)
    job = enqueue(worker)
    publish = SceneStore.publish
    upgrade = SceneStore.upgrade
    sync = worker.queue._sync_scene_index
    fired = False

    def broken_upgrade(*args, **kwargs):
        nonlocal fired
        value = upgrade(*args, **kwargs)
        if failure == "after_upgrade" and not fired:
            fired = True
            raise ValueError("合成深入版本提交后中断")
        return value

    def broken_publish(store, asset_id, run_id, chunks, *args, **kwargs):
        nonlocal fired
        target = run_id == job["run_id"] + "_context" and not fired
        if target and failure == "before_publish":
            fired = True
            raise ValueError("合成发布前中断")
        value = publish(store, asset_id, run_id, chunks, *args, **kwargs)
        if target and failure == "after_publish":
            fired = True
            raise ValueError("合成活动版本提交后中断")
        return value

    def broken_sync(*args, **kwargs):
        nonlocal fired
        if failure == "index" and not fired and any(
                chunk["run_id"] == job["run_id"] + "_context" for chunk in worker.store.list(worker.asset["id"])):
            fired = True
            raise ValueError("合成索引更新中断")
        return sync(*args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(SceneStore, "publish", broken_publish)
        fault.setattr(SceneStore, "upgrade", broken_upgrade)
        fault.setattr(worker.queue, "_sync_scene_index", broken_sync)
        failed = await execute(worker, job)
    assert fired and failed["status"] == "failed", failed
    completed_ids = published_understanding(worker, job)
    assert len(completed_ids) == (1 if failure in {"after_upgrade", "before_publish"} else 2)
    visits = forbid_completed_work(worker, monkeypatch, completed_ids)
    before_calls = copy.deepcopy(worker.provider.calls)
    worker.queue.control(job["id"], "retry")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert counts(worker) == {"transcript": 1, "preparation": 1, "deep_analysis": 1, "context_refresh": 3}
    assert all(stage != "deepen" for stage, _ in visits)
    assert completed["stage_metrics"]["deep_analysis"] == failed["stage_metrics"]["deep_analysis"]
    assert worker.provider.calls[:len(before_calls)] == before_calls
    saved = read_json(state_path(worker, job))
    assert saved["understanding"]["completed"] == 4 and not saved.get("index_dirty")
    assert all(not chunk["stale"] for chunk in worker.store.list(worker.asset["id"]))


@pytest.mark.asyncio
async def test_embedding_failure_restores_whole_completed_understanding(scene_worker, monkeypatch):
    worker = make_story(scene_worker, 16)
    job = enqueue(worker)
    config = copy.deepcopy(job["config"])
    config["bindings"]["embedding"] = "synthetic-embedding"
    worker.queue.update(job["id"], config=config)

    async def interrupted(*args, **kwargs):
        raise ValueError("合成向量索引中断")

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "reindex", interrupted)
        failed = await execute(worker, job)
    assert failed["status"] == "failed", failed
    saved = read_json(state_path(worker, job))
    assert saved["understanding"]["completed"] == 4
    completed_ids = published_understanding(worker, job)
    assert len(completed_ids) == 4
    forbid_completed_work(worker, monkeypatch, completed_ids, forbid_all=True)
    before_calls = copy.deepcopy(worker.provider.calls)
    before_metrics = copy.deepcopy(failed["stage_metrics"])
    indexes = []

    def reject_sync(*args, **kwargs):
        raise AssertionError("completed understanding rebuilt its clean local index")

    async def complete_index(job_id, embedding_profile_id=None):
        indexes.append((job_id, embedding_profile_id))

    monkeypatch.setattr(worker.queue, "_sync_scene_index", reject_sync)
    monkeypatch.setattr(worker.queue, "reindex", complete_index)
    worker.queue.control(job["id"], "retry")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert indexes == [(job["id"], "synthetic-embedding")]
    assert worker.provider.calls == before_calls
    assert completed["stage_metrics"] == before_metrics
    assert completed["stage_progress"]["deep_analysis"]["status"] == "completed"
    assert completed["stage_progress"]["context_refresh"]["status"] == "completed"


@pytest.mark.parametrize("failure", ["after_upgrade", "index", "embedding"])
@pytest.mark.asyncio
async def test_manual_deepen_publication_resumes_without_replaying_finished_analysis(scene_worker, monkeypatch, failure):
    worker = scene_worker(8)
    initial = await execute(worker, enqueue(worker))
    assert initial["status"] == "completed", initial
    original = worker.store.list(worker.asset["id"])[0]
    config, _ = worker.queue.deepen_config(worker.asset["id"], original["id"],
        DeepenInput(max_requests=10), {"bindings": {"vision": "deep"}})
    if failure == "embedding":
        config["bindings"]["embedding"] = "synthetic-embedding"
    job = worker.queue.add("analysis", config, worker.asset["id"])
    upgrade = SceneStore.upgrade

    def interrupt_upgrade(*args, **kwargs):
        upgrade(*args, **kwargs)
        raise ValueError("合成手动深入活动版本提交后中断")

    def interrupt_sync(*args, **kwargs):
        raise ValueError("合成手动深入索引中断")

    async def interrupt_embedding(*args, **kwargs):
        raise ValueError("合成手动深入向量索引中断")

    with monkeypatch.context() as fault:
        if failure == "after_upgrade":
            fault.setattr(SceneStore, "upgrade", interrupt_upgrade)
        elif failure == "index":
            fault.setattr(worker.queue, "_sync_scene_index", interrupt_sync)
        else:
            fault.setattr(worker.queue, "reindex", interrupt_embedding)
        failed = await execute(worker, job)
    assert failed["status"] == "failed", failed
    active = worker.store.list(worker.asset["id"])[0]
    assert active["understanding_level"] == "deep" and active["run_id"] == job["run_id"]
    forbid_completed_work(worker, monkeypatch, {original["id"]}, forbid_all=True)
    before_calls = copy.deepcopy(worker.provider.calls)
    sync = worker.queue._sync_scene_index
    repairs, embeddings = [], []

    def repair_sync(*args, **kwargs):
        repairs.append(True)
        assert failure != "embedding" and len(repairs) == 1
        return sync(*args, **kwargs)

    async def complete_embedding(job_id, embedding_profile_id=None):
        embeddings.append((job_id, embedding_profile_id))

    monkeypatch.setattr(worker.queue, "_sync_scene_index", repair_sync)
    monkeypatch.setattr(worker.queue, "reindex", complete_embedding)
    worker.queue.control(job["id"], "retry")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert repairs == ([] if failure == "embedding" else [True])
    assert embeddings == ([(job["id"], "synthetic-embedding")] if failure == "embedding" else [])
    assert worker.provider.calls == before_calls
    assert completed["stage_metrics"]["deep_analysis"] == failed["stage_metrics"]["deep_analysis"]
    assert worker.store.list(worker.asset["id"])[0] == active
    assert not read_json(state_path(worker, job)).get("index_dirty")


@pytest.mark.parametrize("corruption", ["plan_digest", "completed", "revision_digest", "relative", "active_revision"])
@pytest.mark.asyncio
async def test_corrupt_resume_pointer_cannot_skip_or_spend(scene_worker, monkeypatch, corruption):
    worker = make_story(scene_worker, 16)
    job = enqueue(worker)
    scene_call = worker.queue._scene_call
    refreshes = 0

    async def fail_second_refresh(job_id, stage, *args, **kwargs):
        nonlocal refreshes
        if stage == "context_refresh":
            refreshes += 1
            if refreshes == 2:
                raise ValueError("合成后续更新中断")
        return await scene_call(job_id, stage, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "_scene_call", fail_second_refresh)
        failed = await execute(worker, job)
    assert failed["status"] == "failed", failed
    path = state_path(worker, job)
    saved = read_json(path)
    cursor = saved["understanding"]
    assert cursor["completed"] == 2
    if corruption == "plan_digest":
        cursor["plan_digest"] = "invalid synthetic plan digest"
    elif corruption == "completed":
        cursor["completed"] = 1000000
    elif corruption == "revision_digest":
        cursor["revisions"][0]["revision_digest"] = "invalid synthetic revision digest"
    elif corruption == "relative":
        cursor["revisions"][0]["relative"] = "../outside-synthetic-run.json"
    else:
        active = worker.store.list(worker.asset["id"])[0]
        worker.store.publish(worker.asset["id"], "synthetic_newer_revision",
            [{**active, "summary": "独立的新版本不属于原任务"}], active["start_ms"], active["end_ms"])
    atomic_json(path, saved)
    before_calls = copy.deepcopy(worker.provider.calls)
    before_pointers = copy.deepcopy(worker.library.active(worker.asset["id"])["scenes"])
    worker.queue.control(job["id"], "retry")
    refused = await execute(worker, job)
    assert refused["status"] == "failed", refused
    assert worker.provider.calls == before_calls
    assert worker.library.active(worker.asset["id"])["scenes"] == before_pointers


def preparation_signature(worker):
    """Compare evidence and text across independent synthetic asset identities."""
    registry = read_json(worker.library.asset_dir(worker.asset["id"]) / "frames" / "registry.json")
    cues = {cue["id"]: (cue["start_ms"], cue["end_ms"], cue["text"])
            for cue in worker.library.subtitles(worker.asset["id"])}

    def evidence(fact):
        return {**fact,
                "frame_ids": [registry[identity]["at_ms"] for identity in fact.get("frame_ids", [])],
                "cue_ids": [cues[identity] for identity in fact.get("cue_ids", [])]}

    signatures = []
    for scene in worker.store.list(worker.asset["id"]):
        signatures.append({
            **{key: scene.get(key) for key in ("start_ms", "end_ms", "summary", "needs_deep", "understanding_level")},
            **{key: [registry[identity]["at_ms"] for identity in scene.get(key, [])]
               for key in ("frame_ids", "evidence_frame_ids", "representative_frame_ids", "deep_frame_ids")},
            **{key: [evidence(fact) for fact in scene.get(key, [])]
               for key in ("visual_facts", "dialogue_facts", "inferences")},
        })
    return signatures


@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("action", ["pause", "cancel", "fail"])
@pytest.mark.asyncio
async def test_partial_visual_assembly_resumes_after_saved_batch_and_preserves_merging(
        scene_worker, monkeypatch, mode, action):
    execution = {"transcript": mode, "preparation": mode, "subtitle": mode, "max_concurrency": 2}
    reference = scene_worker(170, with_subtitles=True)
    reference.provider.continue_batches = True
    baseline = await execute(reference, enqueue(reference, execution=execution))
    assert baseline["status"] == "completed", baseline
    expected = preparation_signature(reference)
    assert len(expected) == 1

    worker = scene_worker(170, with_subtitles=True)
    worker.provider.continue_batches = True
    job = enqueue(worker, execution=execution)
    save = worker.queue._save_partial_scene_preparation
    interrupted_at = []

    def stop_after_save(*args, **kwargs):
        value = save(*args, **kwargs)
        completed = kwargs.get("completed", args[-1])
        interrupted_at.append(completed)
        if completed == 1:
            if action == "fail":
                raise ValueError("合成视觉组装完成一批后中断")
            worker.queue.control(job["id"], action)
        return value

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "_save_partial_scene_preparation", stop_after_save)
        stopped = await execute(worker, job)
    assert stopped["status"] == {"pause": "paused", "cancel": "cancelled", "fail": "failed"}[action], stopped
    assert interrupted_at == [1]
    partial_path = state_path(worker, job).with_name("scene-preparation-parts.json")
    saved = read_json(partial_path)
    assert saved["completed"] == 1
    assert len(saved["chunks"]) == 1 and len(saved["chunks"][0]["frame_ids"]) == 56
    assert worker.store.list(worker.asset["id"]) == []
    before_metrics = copy.deepcopy(stopped["stage_metrics"]["transcript"])
    scene_call = worker.queue._scene_call
    resumed_calls, assembled = [], []

    async def checked_call(job_id, stage, unit, *args, **kwargs):
        assert stage != "transcript", "completed transcript was replayed during partial visual resume"
        if stage == "preparation":
            assert int(unit) >= saved["completed"], "completed visual batch was replayed"
            resumed_calls.append(int(unit))
        return await scene_call(job_id, stage, unit, *args, **kwargs)

    def checked_save(*args, **kwargs):
        completed = kwargs.get("completed", args[-1])
        assert completed > saved["completed"], "earlier visual batch was assembled again"
        assembled.append(completed)
        return save(*args, **kwargs)

    monkeypatch.setattr(worker.queue, "_scene_call", checked_call)
    monkeypatch.setattr(worker.queue, "_save_partial_scene_preparation", checked_save)
    worker.queue.control(job["id"], "retry" if action == "fail" else "resume")
    completed = await execute(worker, job)
    assert completed["status"] == "completed", completed
    assert resumed_calls == ([1, 2, 3] if mode == "serial" else [])
    assert assembled == [2, 3, 4]
    assert counts(worker) == {"transcript": 1, "preparation": 4}
    assert completed["request_count"] == 5
    assert completed["stage_progress"]["preparation"] == {"completed": 4, "total": 4, "status": "completed"}
    assert completed["stage_metrics"]["transcript"] == before_metrics
    assert preparation_signature(worker) == expected
    actual = worker.store.list(worker.asset["id"])
    assert len(actual) == 1 and not actual[0]["stale"]
    assert len(actual[0]["frame_ids"]) == len(set(actual[0]["frame_ids"])) == 170


@pytest.mark.parametrize("corruption", ["completed", "coverage"])
@pytest.mark.asyncio
async def test_corrupted_partial_visual_assembly_is_refused_before_more_calls(scene_worker, monkeypatch, corruption):
    worker = scene_worker(170, with_subtitles=True)
    worker.provider.continue_batches = True
    job = enqueue(worker)
    save = worker.queue._save_partial_scene_preparation

    def interrupt_after_save(*args, **kwargs):
        save(*args, **kwargs)
        raise ValueError("合成视觉组装存盘后中断")

    with monkeypatch.context() as fault:
        fault.setattr(worker.queue, "_save_partial_scene_preparation", interrupt_after_save)
        failed = await execute(worker, job)
    assert failed["status"] == "failed", failed
    path = state_path(worker, job).with_name("scene-preparation-parts.json")
    saved = read_json(path)
    if corruption == "completed":
        saved["completed"] = 1000000
    else:
        saved["chunks"][0]["frame_ids"].pop()
    atomic_json(path, saved)
    before_calls = copy.deepcopy(worker.provider.calls)
    worker.queue.control(job["id"], "retry")
    refused = await execute(worker, job)
    assert refused["status"] == "failed", refused
    assert worker.provider.calls == before_calls
    assert worker.store.list(worker.asset["id"]) == []
