"""Synthetic scene persistence, evidence, memory and incremental index regressions."""
import copy
import json
import zipfile
from unittest.mock import patch

import pytest

from scenerecall.characters import CharacterStore, scope_id
from scenerecall.library import Library, atomic_json, atomic_jsonl
from scenerecall.providers import ProviderManager
from scenerecall.scenes import SceneStore
from scenerecall.search import SearchEngine


def seed(library, identity="asset_one", *, episode=1, season=1, series="测试动画"):
    asset = {"schema_version": "1.0", "id": identity, "work_id": "work_" + identity,
             "title": identity, "kind": "animation", "series": series, "season": season, "episode": episode,
             "version": "test", "duration_ms": 10000, "fingerprint": identity, "subtitle_mode": "external",
             "created_at": "2026-10-13T00:00:00Z"}
    atomic_json(library.asset_dir(identity) / "manifest.json", asset)
    cues = []
    for at in range(0, 10000, 2000):
        frame_id = "frame_" + str(at)
        path = library.asset_dir(identity) / "frames" / (frame_id + ".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic evidence, no image decoding")
        library.register_frame(identity, {"id": frame_id, "path": str(path), "at_ms": at})
        cues.append({"id": "cue_" + str(at), "start_ms": at + 100, "end_ms": at + 900,
                     "text": "寻找宝藏" if at == 0 else "我们出发", "language": "zh", "source": "external"})
    atomic_jsonl(library.asset_dir(identity) / "subtitles" / "external.jsonl", cues)
    return library.get_asset(identity)


@pytest.fixture
def store(tmp_path):
    library = Library(tmp_path / "library")
    seed(library)
    return SceneStore(library)


def chunk(start=0, end=10000, *, summary="人物讨论寻找宝藏", **extra):
    frames = ["frame_" + str(at) for at in range(0, 10000, 2000) if start <= at < end]
    cues = ["cue_" + str(at) for at in range(0, 10000, 2000) if start < at + 900 and at + 100 < end]
    return {"start_ms": start, "end_ms": end, "summary": summary, "frame_ids": frames,
            "evidence_frame_ids": frames, "subtitle_ids": cues, "shot_ids": ["shot_one"],
            "representative_frame_ids": frames[:1], "visual_facts": [{"text": "人物站立", "frame_ids": frames[:1]}],
            "dialogue_facts": [{"text": "谈论宝藏", "cue_ids": cues[:1]}],
            "inferences": [{"text": "可能准备寻宝", "frame_ids": frames[:1], "cue_ids": cues[:1]}],
            "memory_after": {"summary": "准备寻宝"}, **extra}


def test_atomic_resegmentation_preserves_old_revision_and_historical_favorite(store):
    library = store.library
    old = store.publish("asset_one", "run_old", [chunk()], 0, 10000)[0]
    library.annotate("asset_one", {"record_id": old["id"], "favorite": True, "note": "保留证据"})
    original = (library.asset_dir("asset_one") / "analyses/run_old/scenes" / (old["id"] + ".json")).read_bytes()
    before = library.active("asset_one")
    with pytest.raises(ValueError, match="完整|连续"):
        store.publish("asset_one", "run_bad", [chunk(0, 4000), chunk(5000, 10000)], 0, 10000)
    assert library.active("asset_one") == before
    store.publish("asset_one", "run_new", [chunk(0, 5000), chunk(5000, 10000)], 0, 10000)
    assert len(store.list("asset_one")) == 2
    assert (library.asset_dir("asset_one") / "analyses/run_old/scenes" / (old["id"] + ".json")).read_bytes() == original
    historical = library.collections()[0]
    assert historical["is_historical"] and historical["kind"] == "scene" and historical["note"] == "保留证据"


@pytest.mark.parametrize("change", [
    {"frame_ids": ["frame_8000"], "evidence_frame_ids": ["frame_8000"]},
    {"visual_facts": [{"text": "旧帧事件", "frame_ids": ["frame_missing"]}]},
    {"visual_facts": [{"text": "台词不是画面事实", "cue_ids": ["cue_0"]}]},
    {"dialogue_facts": [{"text": "画面不是台词", "frame_ids": ["frame_0"]}]},
    {"dialogue_facts": [{"text": "其他块台词", "cue_ids": ["cue_8000"]}]},
])
def test_context_and_wrong_modal_references_cannot_be_current_evidence(store, change):
    with pytest.raises((ValueError, FileNotFoundError)):
        store.publish("asset_one", "run_bad", [chunk(0, 5000, **change)], 0, 5000)
    assert store.list("asset_one") == []


def test_upgrade_keeps_stable_id_and_marks_later_context_stale(store):
    first = store.publish("asset_one", "run_first", [chunk(0, 5000)], 0, 5000)[0]
    memory = store.prior_memory("asset_one", 5000)
    second = store.publish("asset_one", "run_next", [chunk(5000, 10000, memory_before=memory)], 5000, 10000)[0]
    upgraded = store.upgrade("asset_one", "run_deep", {**first, "summary": "人物拿出地图准备寻宝"})
    assert upgraded["id"] == first["id"] and upgraded["understanding_level"] == "deep"
    assert store.get("asset_one", second["id"])["context_stale"]
    assert [row["id"] for row in store.list("asset_one", include_stale=False)] == [first["id"]]
    with pytest.raises(ValueError, match="不可覆盖"):
        store.upgrade("asset_one", "run_deep", {**upgraded, "summary": "不能覆盖"})


def test_same_run_deep_revision_and_initial_retry_are_idempotent(store):
    original = store.publish("asset_one", "run_one", [chunk()], 0, 10000)[0]
    deep = store.upgrade("asset_one", "run_one", {**original, "summary": "深入理解：讨论寻宝而未实际发现宝藏"})
    assert store.upgrade("asset_one", "run_one", deep) == deep
    retried = store.publish("asset_one", "run_one", [original], 0, 10000)[0]
    assert retried == deep
    assert len(list((store.library.asset_dir("asset_one") / "analyses/run_one/scenes").glob("*.json"))) == 2
    assert store.get("asset_one", deep["id"])["understanding_level"] == "deep"
    assert store.revision_digest(deep) == store.revision_digest({**deep, "run_id": "another_run", "provenance": {"time": "changed"}})


def test_memory_is_project_and_episode_scoped_and_reports_gaps(store):
    library = store.library
    seed(library, "asset_two", episode=2)
    seed(library, "asset_other", episode=0, series="另一作品")
    seed(library, "asset_unknown", episode=None)
    for identity in ("asset_one", "asset_other", "asset_unknown"):
        store.publish(identity, "run_first", [chunk(0, 5000)], 0, 5000)
    memory = store.prior_memory("asset_two", 0)
    assert memory["summaries"] == []  # incomplete earlier episodes are not narrative context
    assert memory["gaps"] == [{"asset_id": "asset_one", "start_ms": 5000, "end_ms": 10000}]
    store.publish("asset_one", "run_second", [chunk(5000, 10000)], 5000, 10000)
    memory = store.prior_memory("asset_two", 0)
    assert {item["asset_id"] for item in memory["summaries"]} == {"asset_one"}
    assert store.prior_memory("asset_unknown", 0)["summaries"] == []
    assert len(json.dumps(memory).encode()) < 28000
    target = library.get_asset("asset_other")["project_id"]
    library.organization.assign(["asset_one"], target)
    assert store.prior_memory("asset_two", 0)["summaries"] == []


def test_memory_reports_missing_episode_and_uses_corrections_not_notes(store):
    library = store.library
    seed(library, "asset_three", episode=3)
    row = store.publish("asset_one", "run_one", [chunk()], 0, 10000)[0]
    initial = store.prior_memory("asset_three", 0)
    assert {"kind": "missing_episode", "season": 1, "start_episode": 2, "end_episode": 2} in initial["gaps"]
    library.annotate("asset_one", {"record_id": row["id"], "note": "个人笔记", "favorite": True})
    assert store.prior_memory("asset_three", 0)["digest"] == initial["digest"]
    library.annotate("asset_one", {"record_id": row["id"], "summary": "人工确认：他们只是在回忆旧旅行"})
    corrected = store.prior_memory("asset_three", 0)
    assert corrected["summaries"][0]["summary"] == "人工确认：他们只是在回忆旧旅行"
    assert corrected["digest"] != initial["digest"]
    assert "memory_after" not in corrected["summaries"][0]


def test_subtitle_correction_changes_dependency_but_note_does_not(store):
    library = store.library
    row = store.publish("asset_one", "run_one", [chunk()], 0, 10000)[0]
    original = store.subtitle_digest("asset_one")
    library.annotate("asset_one", {"record_id": "cue_0", "note": "拼写待检查", "favorite": True})
    assert store.subtitle_digest("asset_one") == original
    assert not store.get("asset_one", row["id"])["context_stale"]
    library.annotate("asset_one", {"record_id": "cue_0", "summary": "人工校正台词"})
    assert store.subtitle_digest("asset_one") != original
    assert store.get("asset_one", row["id"])["context_stale"]


def test_dynamic_subtitle_human_and_membership_staleness_without_mutating_results(store):
    library = store.library
    row = store.publish("asset_one", "run_first", [chunk()], 0, 10000)[0]
    path = library.asset_dir("asset_one") / "analyses/run_first/scenes" / (row["id"] + ".json")
    before = path.read_bytes()
    registry = CharacterStore(library)._read(scope_id(library.get_asset("asset_one")))
    registry["revision"] += 1
    atomic_json(CharacterStore(library)._path(registry["scope_id"]), registry)
    assert "角色人工档案已变化" in store.get("asset_one", row["id"])["stale_reasons"]
    cues = library.subtitles("asset_one")
    cues[0]["text"] = "更新台词"
    atomic_jsonl(library.asset_dir("asset_one") / "subtitles/external.jsonl", cues)
    assert "字幕来源或时间轴已变化" in store.get("asset_one", row["id"])["stale_reasons"]
    assert path.read_bytes() == before


def test_staleness_propagates_through_earlier_episode_context(store):
    library = store.library
    seed(library, "asset_two", episode=2)
    seed(library, "asset_three", episode=3)
    store.publish("asset_one", "run_one", [chunk()], 0, 10000)
    store.publish("asset_two", "run_two", [chunk(memory_before=store.prior_memory("asset_two", 0))], 0, 10000)
    memory = store.prior_memory("asset_three", 0)
    memory["provenance"] = [entry for entry in memory["provenance"] if entry["asset_id"] == "asset_two"]
    third = store.publish("asset_three", "run_three", [chunk(memory_before=memory)], 0, 10000)[0]
    cues = library.subtitles("asset_one")
    cues[0]["text"] = "修订了第一集的字幕"
    atomic_jsonl(library.asset_dir("asset_one") / "subtitles/external.jsonl", cues)
    assert store.get("asset_three", third["id"])["context_stale"]
    assert store.prior_memory("asset_three", 0)["summaries"] == []


def test_machine_character_enrichment_keeps_human_revision_and_scene_fresh(store):
    library = store.library
    scene = store.publish("asset_one", "run_scene", [chunk()], 0, 10000)[0]
    characters = CharacterStore(library)
    observation = {"id": "obs_character", "asset_id": "asset_one", "run_id": "run_character", "window_id": "window",
                   "shot_id": "shot_one", "start_ms": 0, "end_ms": 10000, "summary": "一个红衣人物",
                   "evidence_frame_ids": ["frame_0"], "entities": [{"id": "person", "kind": "person",
                   "description": "红色外套", "observed_traits": ["红色外套"], "evidence_frame_ids": ["frame_0"]}]}
    linked = characters.apply_observation("asset_one", observation)
    library.save_observation("asset_one", linked)
    assert characters.context("asset_one")["revision"] == scene["dependencies"]["character_revision"]
    assert not store.get("asset_one", scene["id"])["context_stale"]


def test_scene_projection_preserves_subtitle_hits_and_separate_modal_evidence(store):
    library = store.library
    library.save_observation("asset_one", {"id": "obs_old", "asset_id": "asset_one", "run_id": "run_old",
        "window_id": "window", "shot_id": "shot", "start_ms": 0, "end_ms": 10000,
        "summary": "旧视觉描述", "evidence_frame_ids": ["frame_0"]})
    store.publish("asset_one", "run_scene", [chunk()], 0, 10000)
    records = library.records("asset_one")
    assert {row["kind"] for row in records} == {"subtitle", "scene"}
    scene = next(row for row in records if row["kind"] == "scene")
    assert "宝藏" in scene["text"] and scene["visual_facts"][0]["text"] == "人物站立"
    assert library.observations("asset_one")[0]["id"] == "obs_old"


def test_scene_backup_roundtrip_and_deleted_asset_restore(store, tmp_path):
    library = store.library
    row = store.publish("asset_one", "run_scene", [chunk()], 0, 10000)[0]
    library.annotate("asset_one", {"record_id": row["id"], "favorite": True})
    backup = library.backup()
    path = library.root / "runtime/backups" / (backup["id"] + ".zip")
    restored = Library(tmp_path / "restored")
    restored.restore(path)
    assert SceneStore(restored).get("asset_one", row["id"])["summary"] == row["summary"]
    assert restored.collections()[0]["kind"] == "scene"
    deletion = restored.deletions.delete(kind="assets", asset_ids=["asset_one"])
    assert restored.collections() == []
    restored.deletions.restore(deletion["trash"]["id"])
    assert restored.collections()[0]["record_id"] == row["id"]


def test_backup_invalid_scene_reference_rejected_before_commit(store, tmp_path):
    library = store.library
    row = store.publish("asset_one", "run_scene", [chunk()], 0, 10000)[0]
    backup = library.backup()
    original = library.root / "runtime/backups" / (backup["id"] + ".zip")
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(tampered, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename.endswith("/scenes/" + row["id"] + ".json"):
                value = json.loads(data)
                value["visual_facts"][0]["frame_ids"] = ["missing_frame"]
                data = json.dumps(value).encode()
            target.writestr(item, data)
    restored = Library(tmp_path / "restored")
    with pytest.raises(ValueError):
        restored.restore(tampered)
    assert restored.list_assets() == []


@pytest.mark.asyncio
async def test_incremental_sync_updates_changed_scenes_only_and_removes_retired_ids(store, tmp_path):
    library = store.library
    engine = SearchEngine(tmp_path / "indexes", ProviderManager(tmp_path / "settings"))
    store.publish("asset_one", "run_old", [chunk()], 0, 10000)
    records = library.records("asset_one")
    engine.sync_asset(records)
    generation = engine._generation
    with patch.object(engine, "_write_record", wraps=engine._write_record) as write:
        assert engine.sync_asset(records) == {"upserted": 0, "removed": 0, "contexts_changed": 0}
        assert write.call_count == 0 and engine._generation == generation
        updated = copy.deepcopy(records)
        next(row for row in updated if row["kind"] == "scene")["text"] = "人物发现藏宝图"
        assert engine.sync_asset(updated) == {"upserted": 1, "removed": 0, "contexts_changed": 0}
        assert write.call_count == 1
    assert (await engine.search("藏宝图"))["results"][0]["kind"] == "scene"
    store.publish("asset_one", "run_new", [chunk(0, 5000), chunk(5000, 10000)], 0, 10000)
    result = engine.sync_asset(library.records("asset_one"))
    assert result == {"upserted": 2, "removed": 1, "contexts_changed": 0}
    engine.sync_asset([], asset_id="asset_one")
    assert engine.status()["records"] == engine.status()["contexts"] == 0
