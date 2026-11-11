"""Large allowed rosters reach local selection without bypassing review or budgets."""
from copy import deepcopy

import pytest

from scenerecall.jobs import CharacterContextChanged
from scenerecall.models import AnalysisInput
from scenerecall.providers import ProviderError, ProviderManager
from scenerecall.scene_jobs import encoded_size
from scenerecall.visual_identity import VisualIdentityPipeline
from test_identity_scenes import region
from test_scene_jobs import enqueue, execute, scene_worker  # noqa: F401


def large_roster(worker, monkeypatch, *, enabled=True):
    context = worker.queue.characters.context(worker.asset["id"])
    context["profiles"] = [{"id": f"character_{index:03d}", "name": f"角色{index}",
        "aliases": [f"{index}_" + "完整别名" * 40], "description": "参考背景" * 1000,
        "reference_only": True, "identity_status": "unverified_prior", "presence_verified": False,
        "reference_policy": "Prior reference only; no current observation is established."}
        for index in range(80)]
    monkeypatch.setattr(worker.queue.characters, "context", lambda _: deepcopy(context))
    VisualIdentityPipeline(worker.library).configure(worker.asset["id"], {
        "enabled": enabled, "model": "ccip-caformer-5_fp32", "top_k": 5, "calibration": None})
    with pytest.raises(ProviderError):
        ProviderManager.character_context_text(context)
    with pytest.raises(ValueError, match="角色未被静默截断"):
        worker.queue._bounded_scene_profiles(context)
    return context


def fake_local_matches(worker, monkeypatch, context, *, match=True):
    def run(_self, asset_id, frames, **kwargs):
        kwargs["checkpoint"]()
        return {"status": "completed", "selected_profiles": deepcopy(context["profiles"]), "references": [],
                "regions": [region(f"r:{frame['id']}:{identity}", frame["id"], identity)
                            for frame in frames for identity in ("character_078", "character_079")] if match else []}
    monkeypatch.setattr(VisualIdentityPipeline, "run", run)
    monkeypatch.setattr(VisualIdentityPipeline, "project_result", lambda _self, _asset, result: deepcopy(result))


def test_local_roster_keeps_full_snapshot_and_human_gate_before_model_calls(scene_worker, monkeypatch):  # noqa: F811
    from scenerecall.characters import scope_id
    worker = scene_worker(2)
    context = large_roster(worker, monkeypatch)
    original = deepcopy(context)
    service = worker.queue._confirmation_service()
    review = {"scope_id": scope_id(worker.asset), "revision": context["revision"],
              "summary": {"total": 80, "pending": 80, "confirmed": 0, "skipped": 0}, "sampling": None}
    monkeypatch.setattr(service, "queue", lambda _: deepcopy(review))
    job = enqueue(worker, task_pipeline="unified-v1")
    assert worker.queue._scene_context(worker.asset["id"], job["config"])["profiles"] == context["profiles"]

    import asyncio
    asyncio.run(worker.queue.execute(job["id"]))
    waiting = worker.queue.get(job["id"])
    assert waiting["status"] == "waiting_input" and worker.provider.calls == []
    with pytest.raises(ValueError, match="确认或明确跳过"):
        worker.queue.continue_task_workflow(job["id"])
    review["summary"].update(pending=0, skipped=80)
    review["sampling"] = {"status": "running"}
    with pytest.raises(ValueError, match="采样尚未完成"):
        worker.queue.continue_task_workflow(job["id"])
    review["sampling"] = {"status": "completed"}
    context["revision"] += 1
    review["revision"] = context["revision"]
    continued = worker.queue.continue_task_workflow(job["id"])
    assert continued["config"]["character_snapshot"]["revision"] == context["revision"]
    assert worker.provider.calls == []
    worker.queue.control(job["id"], "pause")
    assert worker.queue.control(job["id"], "resume")["status"] == "queued"
    assert context["profiles"] == original["profiles"]
    context["revision"] += 1
    with pytest.raises(CharacterContextChanged):
        worker.queue._scene_context(worker.asset["id"], continued["config"])


def test_disabled_local_matching_rejects_oversized_full_roster_without_inference(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(1)
    large_roster(worker, monkeypatch, enabled=False)
    with pytest.raises(ValueError, match="角色未被静默截断"):
        enqueue(worker)
    assert worker.queue.list() == [] and worker.provider.calls == []


@pytest.mark.asyncio
async def test_large_cast_can_reach_review_to_select_smaller_roster(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(1)
    context = large_roster(worker, monkeypatch, enabled=False)
    summary = {"total": 80, "pending": 80, "confirmed": 0, "skipped": 0}
    monkeypatch.setattr(worker.queue._confirmation_service(), "queue", lambda _: {
        "scope_id": context["scope_id"], "revision": context["revision"], "summary": summary})
    job = enqueue(worker, task_pipeline="unified-v1")
    await worker.queue.execute(job["id"])
    assert worker.queue.get(job["id"])["status"] == "waiting_input"

    summary.update(pending=0, skipped=80)
    with pytest.raises(ValueError, match="角色未被静默截断"):
        worker.queue.continue_task_workflow(job["id"])
    context["profiles"] = context["profiles"][-1:]
    context["revision"] += 1
    summary.update(total=1, skipped=1)
    continued = worker.queue.continue_task_workflow(job["id"])
    assert continued["config"]["character_snapshot"]["revision"] == context["revision"]
    assert continued["status"] == "queued" and worker.provider.calls == []


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.asyncio
async def test_selected_late_roster_roles_are_bounded_in_preparation_and_deep_requests(
        scene_worker, monkeypatch, parallel):  # noqa: F811
    worker = scene_worker(4)
    context = large_roster(worker, monkeypatch)
    original = deepcopy(context)
    fake_local_matches(worker, monkeypatch, context)
    seen = []
    prepare, analyze = worker.provider.prepare_visual, worker.provider.analyze

    def check(value):
        assert {row["id"] for row in value["profiles"]} == {"character_078", "character_079"}
        assert all(row["aliases"] == original["profiles"][int(row["id"].split("_")[1])]["aliases"]
                   for row in value["profiles"])
        assert all(row["match"]["status"] == "candidate" and row["match"]["character_id"] is None
                   for row in value["identity_regions"])
        dossiers = {key: val for key, val in value.items()
                    if key not in {"identity_regions", "region_images", "reference_frames"}}
        assert encoded_size(dossiers) <= 18000
        ProviderManager.character_context_text(value)
        seen.append(deepcopy(value))

    async def prepare_checked(*args, **kwargs):
        check(args[5])
        return await prepare(*args, **kwargs)

    async def analyze_checked(*args, **kwargs):
        check(args[4])
        return await analyze(*args, **kwargs)

    monkeypatch.setattr(worker.provider, "prepare_visual", prepare_checked)
    monkeypatch.setattr(worker.provider, "analyze", analyze_checked)
    worker.provider.chunk_size = 1
    worker.provider.deep_times = {250}
    _, config = worker.queue.validate(AnalysisInput(asset_id=worker.asset["id"], analysis_strategy="scene-chunks-v1",
        stages=["vision"], execution={"preparation": "parallel" if parallel else "serial", "max_concurrency": 2}),
        {"bindings": {"preparation": "fast", "vision": "deep"}})
    config["visual_batch_size"] = 2
    job = worker.queue.add("analysis", config, worker.asset["id"])
    done = await execute(worker, job)
    assert done["status"] == "completed", done.get("error")
    assert len(seen) == 3  # Two preparation batches and one deep-analysis request.
    assert context == original


@pytest.mark.asyncio
async def test_no_local_candidate_keeps_full_roster_and_fails_explicitly_before_request(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(1)
    context = large_roster(worker, monkeypatch)
    original = deepcopy(context)
    fake_local_matches(worker, monkeypatch, context, match=False)
    done = await execute(worker, enqueue(worker))
    assert done["status"] == "failed" and "角色未被静默截断" in done["error"]
    assert done["request_count"] == 0 and worker.provider.calls == []
    assert context == original


def test_local_dossier_budget_preserves_large_region_evidence(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(1)
    large_roster(worker, monkeypatch)
    job = enqueue(worker)
    evidence = {"id": "r1", "frame_id": "f1", "match": {"reason": "x" * 200_000}}
    args = [[], [], {}, {}, {"profiles": [{"id": "last", "name": "候选"}], "identity_regions": [evidence],
                              "region_images": [], "reference_frames": []}]
    bounded = worker.queue._scene_request_args(job["config"], "preparation", args)
    assert bounded[4]["identity_regions"] == [evidence]
    with pytest.raises(ProviderError) as error:
        ProviderManager.character_context_text(bounded[4])
    assert error.value.code == "character_context_too_large" and error.value.request_count == 0


@pytest.mark.asyncio
async def test_enabled_local_selection_keeps_provider_evidence_split_contract(scene_worker, monkeypatch, tmp_path):  # noqa: F811
    from test_scene_context_split import assert_complete_coverage, oversized_context
    worker = scene_worker(6)
    VisualIdentityPipeline(worker.library).configure(worker.asset["id"], {"enabled": True})
    monkeypatch.setattr(VisualIdentityPipeline, "run", lambda *_args, **_kwargs: {
        "status": "completed", "regions": [], "selected_profiles": [], "references": []})
    contexts, transports, overflows = oversized_context(monkeypatch, worker, tmp_path)
    done = await execute(worker, enqueue(worker))
    assert done["status"] == "completed", done.get("error")
    assert done["request_count"] == len(transports) == 6
    assert all(code == "character_context_too_large" and count == 0 for _, code, count in overflows)
    assert_complete_coverage(contexts, transports)
