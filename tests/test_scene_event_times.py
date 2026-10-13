"""Provider and durable observation contracts agree on instantaneous events."""
import pytest
from pydantic import ValidationError

from scenerecall.library import read_json
from scenerecall.models import Observation
from scenerecall.providers import ProviderError
from test_scene_jobs import enqueue, execute
from test_scene_jobs import scene_worker as scene_worker
from test_scene_providers import images, manager, reply


def observation_data(frame_id, start_ms, end_ms):
    return {"summary": "画面中一处指示灯亮起", "entities": [],
            "events": [{"action": "指示灯亮起", "start_ms": start_ms, "end_ms": end_ms,
                        "actor_ids": [], "target_ids": [], "evidence_frame_ids": [frame_id]}],
            "spatial_observations": [], "uncertainties": [], "evidence_frame_ids": [frame_id]}


def durable_observation(data, start_ms=0, end_ms=1000):
    return {**data, "id": "observation", "asset_id": "asset", "run_id": "run", "window_id": "window",
            "shot_id": "shot", "start_ms": start_ms, "end_ms": end_ms}


@pytest.fixture(autouse=True)
def immediate_retries(monkeypatch):
    async def no_wait(_delay):
        pass
    monkeypatch.setattr("scenerecall.providers.asyncio.sleep", no_wait)


@pytest.mark.parametrize("event_range", [(250, 250), (0, 0), (250, 1000)])
@pytest.mark.asyncio
async def test_point_and_interval_events_pass_provider_and_storage_without_fabricating_duration(tmp_path, event_range):
    data = observation_data("frame-0", *event_range)
    provider = manager(tmp_path, lambda request: reply(data))
    result = await provider.analyze("scene", images(tmp_path, 1), 0, 1000, story_context={})
    stored = Observation.model_validate(durable_observation(result.data))
    event = stored.events[0]
    assert (event["start_ms"], event["end_ms"]) == event_range
    assert result.request_count == 1 and event["evidence_frame_ids"] == ["frame-0"]


@pytest.mark.parametrize("event_range", [(1000, 1000), (600, 500), (-1, 250), (500, 1001), (1001, 1001)])
@pytest.mark.asyncio
async def test_provider_and_storage_reject_window_end_point_reversed_and_outside_events(tmp_path, event_range):
    data = observation_data("frame-0", *event_range)
    provider = manager(tmp_path, lambda request: reply(data))
    with pytest.raises(ProviderError) as caught:
        await provider.analyze("scene", images(tmp_path, 1), 0, 1000, story_context={})
    assert caught.value.request_count == 3
    with pytest.raises(ValidationError, match="事件超出观察时间范围"):
        Observation.model_validate(durable_observation(data))


@pytest.mark.asyncio
async def test_scene_deep_job_publishes_single_frame_point_event_with_exact_timestamp(scene_worker, tmp_path, monkeypatch):
    worker = scene_worker(1)
    worker.provider.deep_times = {250}
    output = {}
    actual_provider = manager(tmp_path, lambda request: reply(output["data"]))

    async def analyze(identity, frames, start_ms, end_ms, character_context=None,
                      subtitle_context=None, story_context=None):
        assert len(frames) == 1
        when = frames[0]["at_ms"]
        output["data"] = observation_data(frames[0]["id"], when, when)
        return await actual_provider.analyze("scene", frames, start_ms, end_ms,
                                            character_context, subtitle_context, story_context)

    monkeypatch.setattr(worker.provider, "analyze", analyze)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    assert job["request_count"] == 2
    scene = worker.store.list(worker.asset["id"])[0]
    assert scene["understanding_level"] == "deep"
    event = scene["events"][0]
    assert event["start_ms"] == event["end_ms"] == 250
    assert event["evidence_frame_ids"] == scene["frame_ids"]
    parts = list((worker.library.asset_dir(worker.asset["id"]) / "analyses" / job["run_id"] /
                  "scene-deep/parts").glob("*/*.json"))
    assert len(parts) == 1
    staged = Observation.model_validate(read_json(parts[0]))
    assert staged.events[0]["start_ms"] == staged.events[0]["end_ms"] == 250
