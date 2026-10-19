"""Deep preflight fallback preserves paid work, evidence and local identity refs."""
import copy
import json
from collections import Counter

import pytest

from scenerecall.jobs import JobQueue
from scenerecall.library import read_json
from scenerecall.providers import ProviderError
from scenerecall.scene_jobs import join_deep_results, split_deep_args, visual_split_index
from test_identity_scenes import region
from test_scene_jobs import enqueue, execute
from test_scene_jobs import scene_worker as scene_worker  # noqa: F401
from test_scene_providers import manager, reply


def deep_provider(worker, monkeypatch, tmp_path, *, pause=False, size=40_000, paid_error=False):
    worker.provider.deep_times = {250}
    contexts, transports, overflows = [], [], []

    def context(job_id, value, frames):
        rows = []
        for frame in frames:
            for index in range(2):
                row = region(f"r:{frame['id']}:{index}", frame["id"])
                row["match"]["reason"] = "x" * size
                rows.append(row)
        return {**value, "profiles": [{"id": "c19", "name": "Complete dossier prior", "description": "Unverified prior"}],
                "identity_regions": rows}

    def transport(request):
        content = json.loads(request.content)["messages"][1]["content"]
        dossiers = json.loads(next(part["text"].removeprefix("Existing character dossiers (data): ")
            for part in content if part.get("text", "").startswith("Existing character dossiers (data): ")))
        frames = [json.loads(part["text"]) for part in content if part.get("text", "").startswith('{"frame_id":')]
        transports.append({"frames": frames, "context": dossiers})
        if pause and len(transports) == 1:
            job = next(job for job in worker.queue.list() if job["status"] == "running")
            worker.queue.control(job["id"], "pause")
        entities, events, spatial = [], [], []
        for number, frame in enumerate(frames):
            fid, identity = frame["frame_id"], f"e{number}"
            entities.append({"id": identity, "kind": "person", "description": "举手", "is_character": True,
                "character_id": None, "name": None, "match_confidence": None, "match_reason": "当前人物身份尚不确定",
                "observed_traits": [],
                "evidence_frame_ids": [fid], "identity_region_id": f"r:{fid}:0",
                "portrait_frame_id": fid, "portrait_box": [0, 0, .5, 1]})
            events.append({"actor_ids": [identity], "target_ids": [], "action": "举手",
                "start_ms": frame["at_ms"], "end_ms": frame["at_ms"], "evidence_frame_ids": [fid]})
            spatial.append({"subject_id": identity, "object_id": None, "relation": "左侧", "evidence_frame_ids": [fid]})
        return reply({"summary": "当前人物举手", "entities": entities, "events": events,
            "spatial_observations": spatial, "uncertainties": ["候选身份未确认"],
            "evidence_frame_ids": [frame["frame_id"] for frame in frames]})

    real = manager(tmp_path, transport)

    async def analyze(identity, frames, first, last, characters, subtitles, memory):
        contexts.append(copy.deepcopy(characters))
        if paid_error:
            raise ProviderError("Synthetic paid error", code="character_context_too_large", request_count=1)
        try:
            return await real.analyze("scene", frames, first, last, characters, subtitles, memory)
        except ProviderError as exc:
            overflows.append((len(frames), exc.code, exc.request_count))
            raise

    monkeypatch.setattr(worker.queue, "scene_identity_context", context)
    monkeypatch.setattr(worker.provider, "analyze", analyze)
    return context, contexts, transports, overflows


def assert_coverage(contexts, transports):
    expected = contexts[0]["identity_regions"]
    frames = [row["frame_id"] for call in transports for row in call["frames"]]
    assert frames == list(dict.fromkeys(row["frame_id"] for row in expected))
    assert len(frames) == len(set(frames))
    actual = [row for call in transports for row in call["context"]["identity_regions"]]
    assert Counter(row["id"] for row in actual) == Counter(row["id"] for row in expected)
    for call in transports:
        assert call["context"]["profiles"] == contexts[0]["profiles"]
        assert all(row["frame_id"] in {frame["frame_id"] for frame in call["frames"]}
                   and row["match"]["status"] == "candidate" for row in call["context"]["identity_regions"])


@pytest.mark.asyncio
async def test_deep_real_provider_recurses_before_transport_and_keeps_original_part_progress(
        scene_worker, monkeypatch, tmp_path):
    worker = scene_worker(6)
    _, contexts, transports, overflows = deep_provider(worker, monkeypatch, tmp_path)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job.get("error")
    assert len(transports) == 6 and job["request_count"] == 7  # preparation plus paid leaves
    assert Counter(size for size, _, _ in overflows) == {6: 1, 3: 2, 2: 2}
    assert all(code == "character_context_too_large" and count == 0 for _, code, count in overflows)
    assert_coverage(contexts, transports)
    assert job["stage_progress"]["deep_analysis"] == {"completed": 1, "total": 1, "status": "completed"}
    assert job["stage_metrics"]["deep_analysis"]["request_count"] == 6
    assert job["inflight_calls"] == {} and job["inflight_attempts"] == 0
    run = worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"]
    parts = list((run / "scene-deep/parts").glob("*/*.json"))
    assert len(parts) == 1 and parts[0].stem == "0"
    part = read_json(parts[0])
    assert len(part["input_frame_ids"]) == 6 and len(part["entities"]) == 6
    ids = {row["id"] for row in part["entities"]}
    assert len(ids) == 6 and all(row["actor_ids"][0] in ids for row in part["events"])
    assert all(row["subject_id"] in ids for row in part["spatial_observations"])
    assert all(row["character_id"] is None and row["candidate_character_id"] == "c19"
               and row["association_status"] == "candidate" and row["identity_region_id"].startswith("r:")
               for row in part["entities"])


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_pause", [False, True])
async def test_deep_children_resume_after_pause_or_budget_without_repaying(scene_worker, monkeypatch, tmp_path, budget_pause):
    worker = scene_worker(6)
    context, contexts, transports, overflows = deep_provider(worker, monkeypatch, tmp_path, pause=not budget_pause)
    job = enqueue(worker, max_requests=4 if budget_pause else 500)
    paused = await execute(worker, job)
    assert paused["status"] == "paused" and paused["request_count"] == 2, paused.get("error")
    assert len(transports) == 1 and paused["inflight_calls"] == {}
    before = len(overflows)
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    monkeypatch.setattr(worker.queue, "scene_identity_context", context)
    worker.queue.control(job["id"], "resume", {"max_requests": 100})
    done = await execute(worker, job)
    assert done["status"] == "completed" and done["request_count"] == 7, done.get("error")
    assert_coverage(contexts, transports)
    assert len(overflows) == 5 and before >= 2
    # A new run reuses completed preparation and the original parent aggregate.
    again = await execute(worker, enqueue(worker))
    assert again["status"] == "completed" and again["request_count"] == 0, again.get("error")
    assert len(transports) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("paid", [False, True])
async def test_single_frame_or_paid_context_error_is_terminal(scene_worker, monkeypatch, tmp_path, paid):
    worker = scene_worker(6 if paid else 1)
    _, _, transports, _ = deep_provider(worker, monkeypatch, tmp_path, size=80_000, paid_error=paid)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "failed" and job["request_count"] == (2 if paid else 1)
    assert transports == [] and job["inflight_calls"] == {}
    run = worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"]
    assert not list((run / "scene-splits").glob("*.json"))


def test_deep_split_keeps_all_heard_priors_source_bounds_and_reference_evidence():
    frames = [{"id": f"f{i}", "at_ms": i * 100} for i in range(4)]
    context = {"profiles": [{"id": "prior", "description": "Complete biography"}],
        "reference_frames": [{"id": "reference:prior", "character_id": "prior"}],
        "identity_regions": [region(f"r{i}", frame["id"]) for i, frame in enumerate(frames)],
        "region_images": [{"id": f"crop{i}", "frame_id": frame["id"]} for i, frame in enumerate(frames)]}
    crossing = {"start_ms": 150, "end_ms": 250, "id": "crossing", "text": "Uncut heard text"}
    review = {**crossing, "checkpoint_sha256": "sha", "review_status": "candidate", "frame_ids": ["f1", "f2"],
              "tracks": [{"association_status": "candidate", "candidates": [{"character_id": "prior"}]}]}
    subtitles = {"start_ms": 0, "end_ms": 400, "cues": [{**crossing, "id": "cue"}],
        "segments": [{"cue_ids": ["cue"], "summary": "Heard summary"}],
        "speaker_dialogue": [crossing], "active_speaker_reviews": [review], "active_speaker_review_policy": "candidate only"}
    args = [frames, 0, 400, context, subtitles, {"summary": "Unchanged story memory"}]
    original = copy.deepcopy(args)
    children = split_deep_args(args, 2)
    assert args == original and [child[1:3] for child in children] == [[0, 200], [200, 400]]
    for child in children:
        assert child[3]["profiles"] == context["profiles"] and child[3]["reference_frames"] == context["reference_frames"]
        assert child[4]["cues"] == subtitles["cues"] and child[4]["speaker_dialogue"] == [crossing]
        assert child[4]["active_speaker_reviews"] == [review] and child[5] == args[5]
        assert all(row["frame_id"] in {frame["id"] for frame in child[0]} for row in child[3]["identity_regions"])
    assert visual_split_index([{**frame, "at_ms": 0} for frame in frames], min_frames=1) is None
    # A review outside a child must not acquire that child's coverage.
    subtitles["active_speaker_reviews"] = [{**review, "start_ms": 0, "end_ms": 100}]
    assert split_deep_args(args, 2)[1][4]["active_speaker_reviews"] == []


def test_deep_aggregate_rejects_foreign_frame_and_invalid_event_time():
    args = [[{"id": "f", "at_ms": 10}], 0, 20, {}, {}, {}]
    value = {"summary": "Visible", "entities": [], "events": [], "evidence_frame_ids": ["foreign"]}
    with pytest.raises(ValueError, match="本批输入之外"):
        join_deep_results([value], args)
    value.update(evidence_frame_ids=["f"], events=[{"actor_ids": [], "target_ids": [], "action": "x",
        "start_ms": 0, "end_ms": 21, "evidence_frame_ids": ["f"]}])
    with pytest.raises(ValueError, match="时间无效"):
        join_deep_results([value], args)
