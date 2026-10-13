"""Visual-similarity navigation stays separate from scene evidence and time."""
import copy
import json

import pytest

from scenerecall.library import Library
from scenerecall.scene_jobs import bounded_memory, link_similar_scenes
from scenerecall.scene_preparation import PreparedChunk, validate_visual_preparation
from scenerecall.scenes import SceneStore
from test_scene_jobs import counts, enqueue, execute
from test_scene_jobs import scene_worker as scene_worker
from test_scene_providers import chunk as prepared_chunk, images, manager, reply
from test_scene_storage import chunk, seed


@pytest.mark.asyncio
async def test_preparation_returns_visual_similarity_hint_in_existing_call(tmp_path):
    frames = images(tmp_path, 2)
    data = {"chunks": [prepared_chunk([frame["id"] for frame in frames], scene_key="  ＲＥＤ   Room  ")],
            "unresolved": []}
    requests = []
    provider = manager(tmp_path, lambda request: requests.append(request) or reply(data))
    result = await provider.prepare_visual("scene", frames, [], None, None)
    assert result.data["chunks"][0]["scene_key"] == "red room"
    assert result.request_count == len(requests) == 1
    prompt = json.loads(requests[0].content)["messages"][0]["content"]
    assert "Do not merge non-adjacent chunks" in prompt
    assert "never proof of the same location" in prompt
    assert PreparedChunk.model_validate(prepared_chunk(["legacy_frame"])).scene_key == ""


@pytest.mark.asyncio
async def test_nonadjacent_similar_scenes_stay_separate_and_deep_links_cost_no_extra_calls(scene_worker, monkeypatch):
    worker = scene_worker(12)
    worker.provider.chunk_size = 3
    worker.provider.deep_times = {9250}
    prepare = worker.provider.prepare_visual

    async def with_keys(identity, frames, context_frames, subtitle_context, story_context, character_context=None):
        result = await prepare(identity, frames, context_frames, subtitle_context, story_context, character_context)
        for row, key in zip(result.data["chunks"], ["红色 房间", "红色 房间", "蓝色走廊", " 红色　房间 "], strict=True):
            row["scene_key"] = key
        result.data = validate_visual_preparation(result.data, frames, set(), has_context=bool(context_frames),
            cues=[], start_ms=subtitle_context["start_ms"], end_ms=subtitle_context["end_ms"])
        return result

    monkeypatch.setattr(worker.provider, "prepare_visual", with_keys)
    job = await execute(worker, enqueue(worker))
    assert job["status"] == "completed", job
    scenes = worker.store.list(worker.asset["id"])
    assert len(scenes) == 4
    assert [(row["start_ms"], row["end_ms"]) for row in scenes] == [
        (0, 3250), (3250, 6250), (6250, 9250), (9250, 12000)]
    assert [row["related_chunk_ids"] for row in scenes] == [
        [scenes[3]["id"]], [scenes[3]["id"]], [], [scenes[0]["id"], scenes[1]["id"]]]
    assert scenes[3]["understanding_level"] == "deep" and scenes[3]["scene_key"] == "红色 房间"
    assert all(not row["context_stale"] for row in scenes)
    assert counts(worker) == {"preparation": 1, "deep_analysis": 1}
    assert job["request_count"] == 2
    assert len({frame for scene in scenes for frame in scene["frame_ids"]}) == 12
    before = copy.deepcopy(worker.provider.calls)
    await execute(worker, job)
    assert worker.provider.calls == before


def test_scene_key_memory_and_navigation_digest_boundaries(tmp_path):
    library = Library(tmp_path / "library")
    seed(library)
    store = SceneStore(library)
    scenes = [{**chunk(start, end), "id": store.chunk_id("asset_one", start, end), "asset_id": "asset_one",
               "scene_key": "门前"} for start, end in [(0, 3000), (3000, 5000), (5000, 10000)]]
    link_similar_scenes(scenes)
    published = store.publish("asset_one", "run_one", scenes, 0, 10000)
    memory = bounded_memory({}, published)
    assert all(row["scene_key"] == "门前" and "related_chunk_ids" not in row for row in memory["recent"])
    prior = store.prior_memory("asset_one", 5000)
    assert all(row["scene_key"] == "门前" and "related_chunk_ids" not in row for row in prior["summaries"])
    first = published[0]
    assert store.revision_digest(first) == store.revision_digest({**first, "related_chunk_ids": []})
    assert store.revision_digest(first) != store.revision_digest({**first, "scene_key": "另一处门前"})
    assert store.revision_digest({}) == store.revision_digest({"scene_key": "", "related_chunk_ids": []})
    without_navigation = {key: value for key, value in first.items() if key not in {"scene_key", "related_chunk_ids"}}
    upgraded = store.upgrade("asset_one", "run_deep", without_navigation)
    assert upgraded["scene_key"] == first["scene_key"]
    assert upgraded["related_chunk_ids"] == first["related_chunk_ids"]


@pytest.mark.parametrize("metadata", [
    {"scene_key": 12}, {"scene_key": "x" * 121}, {"scene_key": "door\nroom"},
    {"related_chunk_ids": "scene_other"}, {"related_chunk_ids": ["../scene_other"]},
    {"related_chunk_ids": ["scene_other", "scene_other"]}, {"related_chunk_ids": [None]},
])
def test_scene_similarity_metadata_is_validated(tmp_path, metadata):
    library = Library(tmp_path / "library")
    seed(library)
    with pytest.raises(ValueError):
        SceneStore(library).publish("asset_one", "run_bad", [chunk(**metadata)], 0, 10000)
    assert not SceneStore(library).list("asset_one")


def test_scene_similarity_links_cannot_reference_self_or_cross_assets(tmp_path):
    library = Library(tmp_path / "library")
    seed(library)
    store = SceneStore(library)
    with pytest.raises(ValueError):
        store.publish("asset_one", "run_bad", [chunk(related_chunk_ids=[store.chunk_id("asset_one", 0, 10000)])], 0, 10000)
    rows = [{"id": "scene_one", "asset_id": "asset_one", "start_ms": 0, "end_ms": 100, "scene_key": "门前"},
            {"id": "scene_two", "asset_id": "asset_two", "start_ms": 200, "end_ms": 300, "scene_key": "门前"}]
    link_similar_scenes(rows)
    assert all(row["related_chunk_ids"] == [] for row in rows)


def test_thousand_same_key_scenes_have_bounded_nearest_links_without_merging_or_quadratic_scanning():
    class CountedScene(dict):
        time_reads = 0

        def __getitem__(self, key):
            if key in {"start_ms", "end_ms"}:
                type(self).time_reads += 1
            return super().__getitem__(key)

    rows = [CountedScene(id=f"scene_{number:04d}", asset_id="asset_one", start_ms=number * 1000,
                         end_ms=(number + 1) * 1000, scene_key="  ＲＥＤ   Room  " if number % 2 else "red room",
                         frame_ids=[f"frame_{number}"],
                         visual_facts=[{"text": "可见场景", "frame_ids": [f"frame_{number}"]}])
            for number in range(1000)]
    original = copy.deepcopy(rows)
    # Unordered caller input must not change nearest-in-time selection or output.
    link_similar_scenes(list(reversed(rows)))
    assert CountedScene.time_reads < 150000
    assert sum(len(row["related_chunk_ids"]) for row in rows) <= 16000
    assert len(rows) == len(original) == 1000
    for index, (row, before) in enumerate(zip(rows, original, strict=True)):
        assert row["scene_key"] == "red room"
        assert {key: value for key, value in row.items() if key not in {"related_chunk_ids", "scene_key"}} == {
            key: value for key, value in before.items() if key != "scene_key"}
        related = [int(identity.removeprefix("scene_")) for identity in row["related_chunk_ids"]]
        assert len(related) == 16 and related == sorted(set(related))
        assert all(abs(other - index) > 1 for other in related)
    for index in [0, 7, 500, 992, 999]:
        expected = sorted(sorted((other for other in range(1000) if abs(other - index) > 1),
                                 key=lambda other: (abs(other - index), other))[:16])
        assert rows[index]["related_chunk_ids"] == [f"scene_{other:04d}" for other in expected]


def test_nearest_similarity_links_use_time_distance_not_number_of_chunks():
    bounds = [(0, 10), (10, 20), (20, 30), (100, 110), (111, 112), (113, 114), (115, 116),
              *[(start, start + 1) for start in range(117, 140, 2)]]
    rows = [{"id": f"scene_{number:02d}", "asset_id": "asset_one", "start_ms": start, "end_ms": end,
             "scene_key": "room"} for number, (start, end) in enumerate(bounds)]
    link_similar_scenes(rows)
    target = rows[3]
    expected = [row for row in rows if row["end_ms"] < target["start_ms"] or row["start_ms"] > target["end_ms"]]
    expected.sort(key=lambda row: (max(target["start_ms"] - row["end_ms"], row["start_ms"] - target["end_ms"]),
                                   row["start_ms"]))
    expected = sorted(expected[:16], key=lambda row: row["start_ms"])
    assert target["related_chunk_ids"] == [row["id"] for row in expected]
