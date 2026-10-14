"""Scene revisions remain portable across cast, organization and trash operations."""
import copy

import pytest

from scenerecall.characters import CharacterStore
from scenerecall.library import Library, atomic_json, read_json
from scenerecall.organization import OrganizationConflict
from scenerecall.scenes import SceneStore
from test_scene_storage import chunk, seed


def deep_scene(library, asset_id="asset_one", *, start=0, end=10000, run_id="run_scene", character_id=None, count=1):
    scenes = SceneStore(library)
    initial = scenes.publish(asset_id, run_id, [chunk(start, end)], start, end)[0]
    frame_id = initial["frame_ids"][0]
    names = ['阿红', '阿青', '阿蓝']
    entities = [{"id": f"person_{index}", "kind": "person", "is_character": True,
                 "name": None if character_id and index == 0 else names[index],
                 "match_reason": f"合成字幕明确称呼：{names[index]}",
                 "character_id": character_id if index == 0 else None, "match_confidence": .9,
                 "description": f"红衣人物 {index}", "observed_traits": ["红衣"],
                 "evidence_frame_ids": [frame_id], "portrait_frame_id": frame_id, "portrait_box": [0, 0, .5, 1]}
                for index in range(count)]
    record = {**initial, "entities": entities, "window_id": initial["id"], "shot_id": "scene"}
    linked = CharacterStore(library).apply_observation(asset_id, record)
    return scenes.upgrade(asset_id, run_id, linked)


@pytest.fixture
def library(tmp_path):
    result = Library(tmp_path / "library")
    seed(result)
    return result


def test_scene_character_project_move_preserves_merges_tombstones_and_immutable_revisions(library):
    characters = CharacterStore(library)
    original = deep_scene(library, count=3)
    first, merged, deleted = [entity["character_id"] for entity in original["entities"]]
    characters.update("asset_one", first, {"name": "阿红", "aliases": ["红同学"], "notes": "人工确认"})
    characters.merge("asset_one", merged, first)
    characters.delete("asset_one", deleted)
    library.annotate("asset_one", {"record_id": original["id"], "favorite": True})
    files = {path: path.read_bytes() for path in library.asset_dir("asset_one").glob("analyses/*/scenes/*.json")}
    target = library.organization.save_project({"name": "目标作品"})
    library.organization.assign(["asset_one"], target["id"])
    projected = next(row for row in library.records("asset_one") if row["kind"] == "scene")
    ids = [entity["character_id"] for entity in projected["entities"]]
    assert ids[0] == ids[1] and ids[0] != first and ids[2] is None
    assert projected["entities"][0]["character_name"] == "阿红"
    assert "红同学" in projected["entities"][0]["aliases"]
    assert projected["context_stale"]
    assert library.collections()[0]["record_id"] == original["id"]
    assert all(path.read_bytes() == content for path, content in files.items())
    characters.validate_all()


def test_historical_deep_favorite_survives_resegmentation_move_and_backup(library, tmp_path):
    characters = CharacterStore(library)
    original = deep_scene(library)
    characters.update("asset_one", original["entities"][0]["character_id"], {"name": "阿红"})
    library.annotate("asset_one", {"record_id": original["id"], "favorite": True, "note": "历史引用"})
    SceneStore(library).publish("asset_one", "run_new", [chunk(0, 5000), chunk(5000, 10000)], 0, 10000)
    target = library.organization.save_project({"name": "目标作品"})
    library.organization.assign(["asset_one"], target["id"])
    historical = library.collections()[0]
    assert historical["is_historical"] and historical["understanding_level"] == "deep"
    assert historical["entities"][0]["character_name"] == "阿红"
    backup = library.backup()
    archive = library.root / "runtime/backups" / (backup["id"] + ".zip")
    restored = Library(tmp_path / "restored")
    restored.restore(archive)
    result = restored.collections()[0]
    assert result["understanding_level"] == "deep" and result["note"] == "历史引用"
    assert result["entities"][0]["character_id"] == historical["entities"][0]["character_id"]
    assert result["entities"][0]["portrait_box"] == [0, 0, .5, 1]
    assert len(list(restored.asset_dir("asset_one").glob("analyses/run_scene/scenes/*.json"))) == 2
    CharacterStore(restored).validate_all()


def test_trash_restore_detects_changed_historical_deep_evidence(library):
    original = deep_scene(library)
    project_id = library.get_asset("asset_one")["project_id"]
    deletion = library.deletions.delete(kind="project", id=project_id, delete_contents=False)
    path = library.asset_dir("asset_one") / "analyses/run_scene/scenes" / (original["id"] + ".deep.json")
    altered = read_json(path)
    altered["summary"] = "删除后意外修改的证据"
    atomic_json(path, altered)
    with pytest.raises(OrganizationConflict, match="分析资料已更新"):
        library.deletions.restore(deletion["trash"]["id"])
    assert library.get_asset("asset_one")["project_id"] is None


def test_archived_deep_backup_restores_identity_and_marks_dependent_episode_stale(library, tmp_path):
    first = deep_scene(library)
    seed(library, "asset_two", episode=2)
    scenes = SceneStore(library)
    second = scenes.publish("asset_two", "run_second", [chunk(memory_before=scenes.prior_memory("asset_two", 0))], 0, 10000)[0]
    assert not scenes.get("asset_two", second["id"])["context_stale"]
    deletion = library.deletions.delete(kind="assets", asset_ids=["asset_one"])
    assert scenes.get("asset_two", second["id"])["context_stale"]
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime/backups" / (backup["id"] + ".zip"))
    restored.deletions.restore(deletion["trash"]["id"])
    assert SceneStore(restored).get("asset_one", first["id"])["understanding_level"] == "deep"
    assert SceneStore(restored).get("asset_two", second["id"])["context_stale"]
    assert len(CharacterStore(restored).catalog("asset_one")["profiles"]) == 1
    CharacterStore(restored).validate_all()


def test_moving_memory_source_to_another_project_invalidates_later_episode(library):
    deep_scene(library)
    seed(library, "asset_two", episode=2)
    scenes = SceneStore(library)
    memory = scenes.prior_memory("asset_two", 0)
    second = scenes.publish("asset_two", "run_second", [chunk(memory_before=memory)], 0, 10000)[0]
    assert memory["provenance"] and not scenes.get("asset_two", second["id"])["context_stale"]
    target = library.organization.save_project({"name": "另一作品"})
    library.organization.assign(["asset_one"], target["id"])
    assert "前序场景上下文已变化" in scenes.get("asset_two", second["id"])["stale_reasons"]
    assert scenes.prior_memory("asset_two", 0)["summaries"] == []


def test_backup_rejects_missing_character_in_inactive_deep_revision_before_commit(library, tmp_path):
    original = deep_scene(library)
    SceneStore(library).publish("asset_one", "run_new", [chunk(0, 5000), chunk(5000, 10000)], 0, 10000)
    path = library.asset_dir("asset_one") / "analyses/run_scene/scenes" / (original["id"] + ".deep.json")
    altered = read_json(path)
    altered["entities"][0]["character_id"] = "character_missing"
    atomic_json(path, altered)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    with pytest.raises(ValueError):
        restored.restore(library.root / "runtime/backups" / (backup["id"] + ".zip"))
    assert restored.list_assets() == []


def test_character_catalog_replaces_old_observation_covered_by_multiple_deep_chunks(library):
    characters = CharacterStore(library)
    legacy = {"id": "legacy", "asset_id": "asset_one", "run_id": "run_old", "window_id": "window", "shot_id": "shot",
              "start_ms": 0, "end_ms": 10000, "summary": "旧版人物观察", "evidence_frame_ids": ["frame_0"],
              "entities": [{"id": "person", "kind": "person", "description": "红衣", "name": "阿红",
                            "match_confidence": .9, "match_reason": "合成字幕明确称呼：阿红",
                            "evidence_frame_ids": ["frame_0"]}]}
    linked = characters.apply_observation("asset_one", legacy)
    library.save_observation("asset_one", linked)
    identity = linked["entities"][0]["character_id"]
    deep_scene(library, start=0, end=5000, run_id="run_first", character_id=identity)
    deep_scene(library, start=5000, end=10000, run_id="run_second", character_id=identity)
    appearances = characters.catalog("asset_one")["profiles"][0]["appearances"]
    assert len(appearances) == 2
    assert all(row["record_id"] != "legacy" for row in appearances)


def test_character_representative_uses_latest_time_across_legacy_and_scene_records(library):
    characters = CharacterStore(library)
    scene = deep_scene(library, start=0, end=5000)
    identity = scene["entities"][0]["character_id"]
    legacy = {"id": "later_legacy", "asset_id": "asset_one", "run_id": "run_old", "window_id": "window", "shot_id": "shot",
              "start_ms": 8000, "end_ms": 10000, "summary": "较晚人物观察", "evidence_frame_ids": ["frame_8000"],
              "entities": [{"id": "person", "kind": "person", "description": "红衣", "character_id": identity,
                            "match_confidence": .9, "evidence_frame_ids": ["frame_8000"]}]}
    library.save_observation("asset_one", characters.apply_observation("asset_one", copy.deepcopy(legacy)))
    assert characters.catalog("asset_one")["profiles"][0]["representative"]["frame_id"] == "frame_8000"
