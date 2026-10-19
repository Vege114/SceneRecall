"""Oversized visual contexts split before transport, keeping every evidence row."""
import json
from collections import Counter

import pytest

from scenerecall.jobs import JobQueue
from scenerecall.providers import MAX_CHARACTER_CONTEXT_BYTES, ProviderError, ProviderManager
from scenerecall.scene_jobs import visual_split_index
from test_identity_scenes import region
from test_scene_jobs import enqueue, execute
from test_scene_jobs import scene_worker as scene_worker  # noqa: F401
from test_scene_providers import chunk, manager, reply


def oversized_context(monkeypatch, worker, tmp_path, *, pause_after_first=False, bytes_per_region=20_000):
    original = worker.queue.scene_identity_context
    contexts, transports, overflows = [], [], []

    def context(job_id, value, frames):
        value = original(job_id, value, frames)
        rows = []
        for frame in frames:
            for index in range(2):
                row = region(f"r:{frame['id']}:{index}", frame["id"])
                row["match"]["reason"] = "x" * bytes_per_region
                rows.append(row)
        return {**value, "profiles": [{"id": "c19", "name": "Candidate prior"}], "identity_regions": rows}

    def transport(request):
        parts = json.loads(request.content)["messages"][1]["content"]
        data = json.loads(next(part["text"].removeprefix("Context data: ") for part in parts
                               if part.get("text", "").startswith("Context data: ")))
        mapping = json.loads(next(part["text"].removeprefix("Original frame map: ") for part in parts
                                  if part.get("text", "").startswith("Original frame map: ")))
        core = [row["frame_id"] for row in mapping if not row["reference_only"]]
        transports.append({"frames": core, "regions": data["character_context"]["identity_regions"]})
        if pause_after_first and len(transports) == 1:
            running = next(job for job in worker.queue.list() if job["status"] == "running")
            worker.queue.control(running["id"], "pause")
        return reply({"chunks": [chunk(core)], "unresolved": []})

    real = manager(tmp_path, transport)

    async def prepare(identity, frames, context_frames, subtitles, story, characters):
        contexts.append({"frames": [row["id"] for row in frames], "regions": characters["identity_regions"]})
        try:
            return await real.prepare_visual("scene", frames, context_frames, subtitles, story, characters)
        except ProviderError as exc:
            overflows.append((len(frames), exc.code, exc.request_count))
            raise

    monkeypatch.setattr(worker.queue, "scene_identity_context", context)
    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    return contexts, transports, overflows


def assert_complete_coverage(contexts, transports):
    expected = contexts[0]
    frames = [identity for call in transports for identity in call["frames"]]
    regions = [row["id"] for call in transports for row in call["regions"]]
    assert frames == expected["frames"] and len(frames) == len(set(frames))
    assert Counter(regions) == Counter(row["id"] for row in expected["regions"])
    assert all(count == 1 for count in Counter(regions).values())
    assert all(row["frame_id"] in call["frames"] for call in transports for row in call["regions"])
    assert all(row["match"]["status"] == "candidate" for call in transports for row in call["regions"])


@pytest.mark.asyncio
async def test_real_provider_context_over_128kb_splits_unpaid_parents_to_single_frames(
        scene_worker, monkeypatch, tmp_path):
    worker = scene_worker(6)
    contexts, transports, overflows = oversized_context(monkeypatch, worker, tmp_path)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job.get("error")
    with pytest.raises(ProviderError) as error:
        ProviderManager.character_context_text({"identity_regions": contexts[0]["regions"]})
    assert error.value.code == "character_context_too_large" and error.value.request_count == 0
    assert len(json.dumps(contexts[0]["regions"]).encode()) > MAX_CHARACTER_CONTEXT_BYTES
    assert Counter(size for size, _, _ in overflows) == {6: 1, 3: 2, 2: 2}
    assert all(code == "character_context_too_large" and count == 0 for _, code, count in overflows)
    assert len(transports) == 6 and job["request_count"] == 6
    assert job["known_cost"] == pytest.approx(6 * .00004)
    assert job["inflight_calls"] == {} and job["inflight_attempts"] == 0
    assert_complete_coverage(contexts, transports)
    scenes = worker.store.list(worker.asset["id"])
    assert [identity for scene in scenes for identity in scene["frame_ids"]] == contexts[0]["frames"]
    run = worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"]
    splits = [json.loads(path.read_text()) for path in (run / "scene-splits").glob("*.json")]
    assert len(splits) == 5 and {row["code"] for row in splits} == {"character_context_too_large"}


@pytest.mark.asyncio
async def test_context_split_resume_reuses_successful_child_and_never_retries_unpaid_parent(
        scene_worker, monkeypatch, tmp_path):
    worker = scene_worker(6)
    contexts, transports, overflows = oversized_context(monkeypatch, worker, tmp_path, pause_after_first=True)
    job = enqueue(worker)
    paused = await execute(worker, job)
    assert paused["status"] == "paused" and paused["request_count"] == 1, paused
    assert len(transports) == 1
    worker.queue = JobQueue(worker.library, worker.provider, worker.queue.search)
    # The normal projection method needs the same deterministic frozen fixture
    # after constructing a fresh queue, just as the real prelude is retained.
    context = contexts[0]

    def restore(job_id, value, frames):
        allowed = {frame["id"] for frame in frames}
        return {**value, "profiles": [{"id": "c19", "name": "Candidate prior"}],
                "identity_regions": [row for row in context["regions"] if row["frame_id"] in allowed]}

    monkeypatch.setattr(worker.queue, "scene_identity_context", restore)
    worker.queue.control(job["id"], "resume")
    done = await execute(worker, job)
    assert done["status"] == "completed" and done["request_count"] == 6, done
    assert_complete_coverage(contexts, transports)
    assert Counter(size for size, _, _ in overflows) == {6: 1, 3: 2, 2: 2}
    # Replaying the complete parent aggregate performs no transport or preflight.
    count = len(contexts)
    again = await execute(worker, enqueue(worker))
    assert again["status"] == "completed" and again["request_count"] == 0, again
    assert len(contexts) == count and len(transports) == 6


@pytest.mark.asyncio
async def test_single_frame_context_overflow_is_terminal_without_paid_transport(scene_worker, monkeypatch, tmp_path):
    worker = scene_worker(1)
    _, transports, overflows = oversized_context(monkeypatch, worker, tmp_path, bytes_per_region=80_000)
    done = await execute(worker, enqueue(worker))
    assert done["status"] == "failed" and done["request_count"] == 0, done
    assert overflows == [(1, "character_context_too_large", 0)] and transports == []
    run = worker.library.asset_dir(worker.asset["id"]) / "analyses" / done["run_id"]
    assert not list((run / "scene-splits").glob("*.json"))


@pytest.mark.asyncio
async def test_context_split_still_checks_child_budget_and_keeps_paid_leaf_for_resume(
        scene_worker, monkeypatch, tmp_path):
    worker = scene_worker(6)
    contexts, transports, _ = oversized_context(monkeypatch, worker, tmp_path)
    job = enqueue(worker, max_requests=3)
    paused = await execute(worker, job)
    assert paused["status"] == "paused" and paused["request_count"] == 1, paused
    assert len(transports) == 1 and paused["inflight_calls"] == {}
    worker.queue.control(job["id"], "resume", {"max_requests": 20})
    done = await execute(worker, job)
    assert done["status"] == "completed" and done["request_count"] == 6, done
    assert_complete_coverage(contexts, transports)


@pytest.mark.asyncio
async def test_context_error_after_paid_attempt_does_not_expand_work(scene_worker, monkeypatch):
    worker = scene_worker(6)
    calls = []

    async def prepare(*args):
        calls.append(True)
        raise ProviderError("Synthetic paid error", code="character_context_too_large", request_count=1)

    monkeypatch.setattr(worker.provider, "prepare_visual", prepare)
    done = await execute(worker, enqueue(worker))
    assert done["status"] == "failed" and done["request_count"] == 1 and calls == [True], done
    run = worker.library.asset_dir(worker.asset["id"]) / "analyses" / done["run_id"]
    assert not list((run / "scene-splits").glob("*.json"))


def test_context_overflow_minimum_does_not_change_paid_fallback_or_timestamp_boundaries():
    frames = [{"id": str(index), "at_ms": index // 2} for index in range(6)]
    assert visual_split_index(frames) is None
    assert visual_split_index(frames, min_frames=1) == 2
    assert visual_split_index([{**row, "at_ms": 0} for row in frames], min_frames=1) is None


@pytest.mark.asyncio
async def test_visual_context_nonfinite_json_error_is_not_reclassified(tmp_path):
    calls = []
    real = manager(tmp_path, lambda request: calls.append(request))
    from test_scene_providers import images
    with pytest.raises(ValueError) as error:
        await real.prepare_visual("scene", images(tmp_path, 1), [], None, {"bad": float("nan")})
    assert not isinstance(error.value, ProviderError) and not calls
