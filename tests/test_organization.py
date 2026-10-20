"""Project moves preserve immutable evidence and project-local human decisions."""
import copy
import zipfile

import pytest

from scenerecall.characters import CharacterStore
from scenerecall.anime_profiles import AnimeProfileStore
from scenerecall.library import Library, atomic_json, read_json
from scenerecall.organization import OrganizationConflict


def seed(library, asset_id="asset_one", series="旧动画", episode=1, title=None):
    asset = {"schema_version": "1.0", "id": asset_id, "work_id": "work_" + asset_id,
             "title": title or series or asset_id, "kind": "animation", "series": series, "season": 1, "episode": episode,
             "version": "test", "duration_ms": 10000, "fingerprint": asset_id,
             "created_at": "2026-10-01T00:00:00Z", "subtitle_mode": "embedded"}
    atomic_json(library.asset_dir(asset_id) / "manifest.json", asset)
    frame = library.asset_dir(asset_id) / "frames" / "frame_test.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"synthetic storage-only frame")
    library.register_frame(asset_id, {"id": "frame_test", "path": str(frame), "at_ms": 0})
    return library.get_asset(asset_id)


def appearance(library, store, asset_id="asset_one", number=1, character_id=None):
    record = {"id": f"obs_{asset_id}_{number}", "asset_id": asset_id, "run_id": "run_test",
              "window_id": f"window_{number}", "shot_id": "shot_test", "start_ms": (number - 1) * 1000,
              "end_ms": number * 1000, "summary": "人物拿着杯子", "evidence_frame_ids": ["frame_test"],
              "entities": [{"id": "person", "kind": "person", "description": "红衣人物",
                            "name": f"合成葵{number}", "match_confidence": .95,
                            "match_reason": f"当前画面胸牌显示合成葵{number}",
                            "character_id": character_id, "evidence_frame_ids": ["frame_test"]}]}
    linked = store.apply_observation(asset_id, record)
    library.save_observation(asset_id, linked)
    return linked["entities"][0]["character_id"]


@pytest.fixture
def library(tmp_path):
    result = Library(tmp_path / "library")
    seed(result)
    return result


def test_legacy_migration_rename_nested_folders_and_delete_preserve_membership(library):
    organization = library.organization
    seed(library, "asset_two", "  旧动画  ", episode=2)
    assert len(organization.projects()) == 1
    project = organization.projects()[0]
    original = library.get_asset("asset_one")
    parent = organization.save_folder({"name": "动画"})
    child = organization.save_folder({"name": "秋季", "parent_id": parent["id"]})
    organization.save_project({"name": "新片名", "folder_id": child["id"], "tags": ["收藏", " 收藏 ", "动画"]}, project["id"])
    asset = library.get_asset("asset_one")
    assert asset["project_name"] == "新片名" and asset["tags"] == ["收藏", "动画"]
    assert asset["folder_ids"] == [child["id"], parent["id"]]
    assert asset["character_scope_id"] == original["character_scope_id"]
    before = organization.path.read_bytes()
    with pytest.raises(ValueError, match="循环"):
        organization.save_folder({"parent_id": child["id"]}, parent["id"])
    assert organization.path.read_bytes() == before
    organization.delete_folder(parent["id"])
    assert organization.folders()[0]["parent_id"] is None
    assert library.get_asset("asset_one")["folder_ids"] == [child["id"]]
    reopened = Library(library.root)
    assert reopened.organization.projects() == organization.projects()


def test_move_preserves_human_edits_redirects_tombstones_and_observation_bytes(library):
    store = CharacterStore(library)
    first = appearance(library, store)
    second = appearance(library, store, number=2)
    deleted = appearance(library, store, number=3)
    store.update("asset_one", first, {"name": "小葵", "notes": "人工笔记", "appearance": ["左眼伤疤"]})
    store.merge("asset_one", second, first)
    store.delete("asset_one", deleted)
    originals = {path: path.read_bytes() for path in library.asset_dir("asset_one").glob("analyses/*/observations/*.json")}
    target = library.organization.save_project({"name": "目标动画"})
    library.organization.assign(["asset_one"], target["id"])
    catalog = store.catalog("asset_one")
    assert len(catalog["profiles"]) == 1
    profile = catalog["profiles"][0]
    assert profile["name"] == "小葵" and "人工笔记" in profile["notes"]
    assert profile["appearance"] == ["左眼伤疤"] and profile["user_edited"]
    projected = store.decorate_observations("asset_one", library.observations("asset_one"))
    assert [row["entities"][0]["character_id"] for row in projected] == [profile["id"], profile["id"], None]
    assert all(path.read_bytes() == original for path, original in originals.items())
    assert library.get_asset("asset_one")["membership_revision"] == 1
    store.validate_all()


def test_same_provenance_reuses_profile_but_same_name_never_merges(library):
    store = CharacterStore(library)
    seed(library, "asset_two", episode=2)
    first = appearance(library, store)
    appearance(library, store, "asset_two", character_id=first)
    store.update("asset_one", first, {"name": "同名角色"})
    seed(library, "asset_other", "另一作品")
    other = appearance(library, store, "asset_other")
    store.update("asset_other", other, {"name": "同名角色"})
    target = library.get_asset("asset_other")["project_id"]
    library.organization.assign(["asset_one"], target)
    copied = next(profile for profile in store.catalog("asset_one")["profiles"] if profile["id"] != other)
    store.update("asset_one", copied["id"], {"notes": "目标项目已修订"})
    library.organization.assign(["asset_two"], target)
    assert len(store.catalog("asset_one")["profiles"]) == 2
    reused = next(profile for profile in store.catalog("asset_two")["profiles"] if profile["id"] == copied["id"])
    assert reused["notes"] == "目标项目已修订" and reused["appearance_count"] == 2
    store.validate_all()


def test_deleted_destination_import_is_not_resurrected(library):
    store = CharacterStore(library)
    seed(library, "asset_two", episode=2)
    first = appearance(library, store)
    appearance(library, store, "asset_two", character_id=first)
    target = library.organization.save_project({"name": "目标"})
    library.organization.assign(["asset_one"], target["id"])
    copied = store.catalog("asset_one")["profiles"][0]["id"]
    store.delete("asset_one", copied)
    library.organization.assign(["asset_two"], target["id"])
    assert store.catalog("asset_two")["profiles"] == []
    assert store.decorate_observations("asset_two", library.observations("asset_two"))[0]["entities"][0]["character_id"] is None
    store.validate_all()


@pytest.mark.parametrize("target_edit", ["edit", "merge", "delete"])
def test_returning_episode_reuses_original_shared_identity_and_target_decisions(library, target_edit):
    store = CharacterStore(library)
    seed(library, "asset_two", episode=2)
    original = appearance(library, store)
    appearance(library, store, "asset_two", character_id=original)
    original_project = library.get_asset("asset_one")["project_id"]
    intermediate = library.organization.save_project({"name": "临时项目"})
    library.organization.assign(["asset_one"], intermediate["id"])
    imported = store.catalog("asset_one")["profiles"][0]["id"]
    store.update("asset_one", imported, {"notes": "临时项目修订"})
    expected = original
    if target_edit == "edit":
        store.update("asset_two", original, {"name": "目标项目姓名", "notes": "目标项目修订"})
    elif target_edit == "merge":
        expected = appearance(library, store, "asset_two", number=2)
        store.update("asset_two", expected, {"name": "合并后姓名", "notes": "目标项目修订"})
        store.merge("asset_two", original, expected)
    else:
        store.delete("asset_two", original)
        expected = None
    library.organization.assign(["asset_one"], original_project)
    catalog = store.catalog("asset_one")
    projected = store.decorate_observations("asset_one", library.observations("asset_one"))
    assert projected[0]["entities"][0]["character_id"] == expected
    if expected:
        assert len(catalog["profiles"]) == 1
        assert catalog["profiles"][0]["id"] == expected
        assert "目标项目修订" in catalog["profiles"][0]["notes"]
        assert catalog["profiles"][0]["appearance_count"] >= 2
    else:
        assert catalog["profiles"] == []
    store.validate_all()


def test_repeat_moves_and_project_delete_keep_archives_backup_and_favorites(library, tmp_path):
    store = CharacterStore(library)
    first = appearance(library, store)
    store.update("asset_one", first, {"name": "人工姓名", "notes": "不会丢失"})
    library.annotate("asset_one", {"record_id": "obs_asset_one_1", "favorite": True, "note": "收藏笔记"})
    original = library.get_asset("asset_one")["project_id"]
    target = library.organization.save_project({"name": "中间项目"})
    library.organization.assign(["asset_one"], target["id"])
    library.organization.assign(["asset_one"], original)
    library.organization.delete_project(original)
    assert library.get_asset("asset_one")["project_id"] is None
    assert library.get_asset("asset_one")["membership_revision"] == 3
    assert library.collections()[0]["note"] == "收藏笔记"
    store.validate_all()
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert restored.records() == library.records()
    assert restored.organization.projects() == library.organization.projects()
    assert restored.collections() == library.collections()
    CharacterStore(restored).validate_all()


def test_move_conflicts_with_running_vision_and_pauses_queued_scope(library):
    class Queue:
        def __init__(self):
            self.jobs = [{"id": "job_test", "type": "analysis", "asset_id": "asset_one",
                          "status": "running", "config": {"stages": ["vision"]}}]

        def list(self):
            return copy.deepcopy(self.jobs)

        def update(self, job_id, **fields):
            self.jobs[0].update(fields)

    queue = Queue()
    target = library.organization.save_project({"name": "目标"})
    before = library.organization.path.read_bytes()
    with pytest.raises(OrganizationConflict):
        library.organization.assign(["asset_one"], target["id"], queue)
    assert library.organization.path.read_bytes() == before
    queue.jobs[0]["status"] = "queued"
    library.organization.assign(["asset_one"], target["id"], queue)
    assert queue.jobs[0]["status"] == "paused" and queue.jobs[0]["stale_organization"]


def test_prepared_move_rolls_forward_after_catalog_write_failure(library, monkeypatch):
    import scenerecall.organization as module
    store = CharacterStore(library)
    appearance(library, store)
    target = library.organization.save_project({"name": "目标"})
    original_write = module.atomic_json

    def interrupted(path, value):
        if path == library.organization.path:
            raise OSError("simulated power failure")
        original_write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_json", interrupted)
        with pytest.raises(OSError):
            library.organization.assign(["asset_one"], target["id"])
    assert library.organization.journal.exists()
    recovered = Library(library.root)
    assert recovered.get_asset("asset_one")["project_id"] == target["id"]
    assert recovered.get_asset("asset_one")["membership_revision"] == 1
    assert not recovered.organization.journal.exists()
    CharacterStore(recovered).validate_all()
    assert Library(library.root).get_asset("asset_one")["membership_revision"] == 1


@pytest.mark.parametrize("failure_point", ["prepare", "publish_asset", "publish_catalog"])
def test_restore_transaction_recovers_before_migrating_new_project_assets(library, tmp_path, monkeypatch, failure_point):
    import scenerecall.organization as module
    store = CharacterStore(library)
    appearance(library, store)
    project = library.organization.save_project({"name": "备份项目"})
    library.organization.assign(["asset_one"], project["id"])
    manifest_path = library.asset_dir("asset_one") / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["project_id"] = project["id"]
    atomic_json(manifest_path, manifest)
    backup = library.backup()
    archive = library.root / "runtime" / "backups" / f"{backup['id']}.zip"
    restored = Library(tmp_path / "destination")
    seed(restored, "asset_existing", "已经存在的作品")
    original_bytes = (restored.asset_dir("asset_existing") / "manifest.json").read_bytes()
    original_write, original_replace = module.atomic_json, module.atomic_replace

    def interrupted_write(path, value):
        if failure_point == "prepare" and path == restored.organization.journal and "files" in value:
            raise OSError("simulated failure before durable preparation")
        if failure_point == "publish_catalog" and path == restored.organization.path:
            raise OSError("simulated failure before catalog publication")
        original_write(path, value)

    def interrupted_replace(source, target):
        original_replace(source, target)
        if failure_point == "publish_asset" and target == restored.asset_dir("asset_one") / "manifest.json":
            raise OSError("simulated failure after publishing one asset")

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_json", interrupted_write)
        patch.setattr(module, "atomic_replace", interrupted_replace)
        with pytest.raises(OSError):
            restored.restore(archive)
    assert (restored.asset_dir("asset_existing") / "manifest.json").read_bytes() == original_bytes
    if failure_point == "prepare":
        assert not restored.organization.journal.exists()
        assert not (restored.asset_dir("asset_one") / "manifest.json").exists()
        assert len(Library(restored.root).list_assets()) == 1
    else:
        pending = read_json(restored.organization.journal)
        assert (restored.root / pending["staging"]).is_dir()
        recovered = Library(restored.root)
        assert recovered.get_asset("asset_one")["project_id"] == project["id"]
        assert len(recovered.list_assets()) == 2
        assert recovered.records("asset_one") == library.records("asset_one")
        assert not recovered.organization.journal.exists()
        assert not (recovered.root / pending["staging"]).exists()
        CharacterStore(recovered).validate_all()


def test_old_backup_without_organization_migrates_and_malformed_mapping_is_rejected(library, tmp_path):
    store = CharacterStore(library)
    appearance(library, store)
    backup = library.backup()
    old_archive = tmp_path / "old.zip"
    with zipfile.ZipFile(library.root / "runtime" / "backups" / f"{backup['id']}.zip") as source:
        with zipfile.ZipFile(old_archive, "w") as target:
            for info in source.infolist():
                if not info.filename.startswith("organization/"):
                    target.writestr(info, source.read(info))
    restored = Library(tmp_path / "old_restored")
    restored.restore(old_archive)
    assert restored.records() == library.records()
    catalog = read_json(restored.organization.path)
    catalog["assets"]["asset_one"]["project_id"] = "missing_project"
    atomic_json(restored.organization.path, catalog)
    with pytest.raises(ValueError, match="项目不存在"):
        Library(restored.root)


def synthetic_plan():
    return {"id": "plan_test", "plan_id": "plan_test", "asset_id": "asset_one", "fingerprint": "asset_one",
            "start_ms": 0, "end_ms": 10000, "algorithm": "content-keyframes-v1", "status": "completed",
            "shots": [{"id": "shot_test", "start_ms": 0, "end_ms": 10000}],
            "frames": [{"id": "frame_test", "shot_id": "shot_test", "pts": 0, "time_base": [1, 30],
                        "source_pts": 0, "source_time_base": [1, 30], "at_ms": 0}],
            "batches": [{"id": "batch_test", "shot_id": "shot_test", "start_ms": 0, "end_ms": 10000,
                         "frame_ids": ["frame_test"]}]}


@pytest.mark.parametrize("corruption", ["fingerprint", "pts", "missing_frame", "foreign_shot", "duplicate", "time_base"])
def test_keyframe_backup_rejects_corrupt_graph_before_merging(library, tmp_path, corruption):
    plan = synthetic_plan()
    if corruption == "fingerprint":
        plan["fingerprint"] = "another video"
    elif corruption == "pts":
        plan["frames"][0]["source_pts"] = 30
    elif corruption == "missing_frame":
        plan["frames"][0]["id"] = "missing"
    elif corruption == "foreign_shot":
        plan["batches"][0]["shot_id"] = "foreign"
    elif corruption == "duplicate":
        plan["batches"][0]["frame_ids"].append("frame_test")
    else:
        plan["frames"][0]["time_base"] = [1, 0]
    atomic_json(library.asset_dir("asset_one") / "keyframe-plans" / "plan_test.json", plan)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    with pytest.raises((ValueError, FileNotFoundError)):
        restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert restored.list_assets() == []


def test_completed_and_pending_keyframe_plans_restore_without_original_media(library, tmp_path):
    plan = synthetic_plan()
    atomic_json(library.asset_dir("asset_one") / "keyframe-plans" / "plan_test.json", plan)
    pending = {key: value for key, value in plan.items() if key not in {"frames", "batches", "shots"}}
    pending.update(id="plan_pending", plan_id="plan_pending", status="pending", job_id="old_runtime_job")
    atomic_json(library.asset_dir("asset_one") / "keyframe-plans" / "plan_pending.json", pending)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert restored.validate_keyframe_plan("asset_one", plan) == plan
    assert restored.validate_keyframe_plan("asset_one", pending) == pending


def test_keyframe_plan_preserves_precise_origin_for_legacy_rounded_manifest(library, tmp_path):
    manifest_path = library.asset_dir("asset_one") / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["start_time_ms"] = 0
    atomic_json(manifest_path, manifest)
    original_manifest = manifest_path.read_bytes()
    plan = synthetic_plan()
    plan["source_start_time_seconds"] = "1/30000"
    plan["frames"][0].update(source_pts=1, source_time_base=[1, 30000])
    assert library.validate_keyframe_plan("asset_one", plan) == plan
    atomic_json(library.asset_dir("asset_one") / "keyframe-plans" / "plan_test.json", plan)
    backup = library.backup()
    restored = Library(tmp_path / "precise_restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert restored.validate_keyframe_plan("asset_one", plan) == plan
    assert manifest_path.read_bytes() == original_manifest
    manifest["start_time_seconds"] = "0"
    atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="精确起始时间"):
        library.validate_keyframe_plan("asset_one", plan)


def bangumi_archive(subject_id=100):
    from scenerecall.bangumi import normalize_dossier_names
    return normalize_dossier_names({"source": "bangumi", "fetched_at": "2026-10-14T00:00:00Z",
            "subject": {"id": subject_id, "name": f"Synthetic anime {subject_id}", "name_cn": "测试动画",
                        "url": f"https://bgm.tv/subject/{subject_id}", "summary": "作品的先验资料"},
            "characters": [{"id": 200 + index, "name": f"Role {index}", "name_cn": f"条目角色 {index}",
                            "aliases": [f"原作别名 {index}"], "description": f"角色 {index} 的条目资料",
                            "url": f"https://bgm.tv/character/{200 + index}", "image_url": None}
                           for index in range(2)]})


def test_standalone_bangumi_move_inherits_archive_and_zero_appearance_cast(library):
    asset = seed(library, "asset_standalone", series="")
    assert asset["project_id"] is None
    archive = bangumi_archive()
    profiles = AnimeProfileStore(library)
    profiles.apply_asset(asset["id"], {"mode": "bangumi", "subject_id": 100}, archive)
    store = CharacterStore(library)
    unseen_role = store.catalog(asset["id"])["profiles"][0]
    store.update(asset["id"], unseen_role["id"], {"name": "出场前的人工姓名", "notes": "未识别前的人工作品设定"})
    original = store.catalog(asset["id"])
    destination = library.organization.save_project({"name": "新动画项目", "kind": "animation"})
    library.organization.assign([asset["id"]], destination["id"])
    assert library.get_asset(asset["id"])["anime_profile"] == {"mode": "bangumi", "dossier": archive}
    assert library.organization.get_project(destination["id"])["anime_profile"]["dossier"] == archive
    moved = store.catalog(asset["id"])
    assert {profile["name"] for profile in moved["profiles"]} == {profile["name"] for profile in original["profiles"]}
    assert len(moved["profiles"]) == 2 and all(profile["appearance_count"] == 0 for profile in moved["profiles"])
    corrected = next(profile for profile in moved["profiles"] if profile["name"] == "出场前的人工姓名")
    assert corrected["notes"] == "未识别前的人工作品设定" and corrected["user_edited"]
    assert all(profile["representative"] is None for profile in moved["profiles"])
    assert store.context(asset["id"])["work_summary"]["id"] == 100
    store.validate_all()


def test_move_into_bangumi_project_does_not_copy_a_nonroster_local_dossier(library):
    source = seed(library, "asset_local", series="", title="目标动画")
    target = seed(library, "asset_target", series="目标动画")
    store = CharacterStore(library)
    local_id = appearance(library, store, source["id"])
    original_path = next(library.asset_dir(source["id"]).glob("analyses/*/observations/*.json"))
    original = original_path.read_bytes()
    profiles = AnimeProfileStore(library)
    profiles.apply_project(target["project_id"], {"mode": "bangumi", "subject_id": 100}, bangumi_archive())
    library.organization.assign([source["id"]], target["project_id"])
    moved = library.get_asset(source["id"])
    registry = store._read(moved["character_scope_id"])
    assert len(registry["profiles"]) == 2
    assert all(profile.get("external_source") for profile in registry["profiles"].values())
    assert registry["imports"][source["character_scope_id"]][local_id] is None
    assert store.decorate_observations(source["id"], library.observations(source["id"]))[0]["entities"][0]["character_id"] is None
    assert original_path.read_bytes() == original
    library.organization.assign([source["id"]], None)
    assert len(store.catalog(source["id"])["profiles"]) == 2
    assert all(profile.get("external_source") for profile in store.catalog(source["id"])["profiles"])
    store.validate_all()


def test_same_bangumi_identity_move_reuses_destination_decisions_without_duplicate_cast(library):
    source_asset = seed(library, "asset_standalone", series="")
    target_asset = seed(library, "asset_target", series="目标动画")
    profiles = AnimeProfileStore(library)
    archive = bangumi_archive()
    selection = {"mode": "bangumi", "subject_id": 100}
    profiles.apply_asset(source_asset["id"], selection, archive)
    source_asset = library.get_asset(source_asset["id"])
    profiles.apply_project(target_asset["project_id"], selection, archive)
    store = CharacterStore(library)
    source_role = next(profile for profile in store.catalog(source_asset["id"])["profiles"]
                       if profile["external_source"]["character_id"] == 200)
    target_role = next(profile for profile in store.catalog(target_asset["id"])["profiles"]
                       if profile["external_source"]["character_id"] == 200)
    assert source_role["id"] != target_role["id"]
    appearance(library, store, source_asset["id"], character_id=source_role["id"])
    store.update(source_asset["id"], source_role["id"], {"name": "来源人工姓名", "notes": "来源笔记"})
    store.update(target_asset["id"], target_role["id"], {"name": "目标人工姓名", "notes": "目标笔记",
                                                         "description": "目标人工描述", "appearance": ["目标人工外观"]})
    originals = {path: path.read_bytes() for path in library.asset_dir(source_asset["id"]).glob("analyses/*/observations/*.json")}
    library.organization.assign([source_asset["id"]], target_asset["project_id"])
    moved = store.catalog(source_asset["id"])
    assert len(moved["profiles"]) == 2
    current = next(profile for profile in moved["profiles"] if profile["id"] == target_role["id"])
    assert current["name"] == "目标人工姓名" and current["notes"] == "目标笔记"
    assert current["description"] == "目标人工描述" and current["appearance"] == ["目标人工外观"]
    assert current["appearance_count"] == 1
    assert store._read(source_asset["character_scope_id"])["profiles"][source_role["id"]]["name"] == "来源人工姓名"
    projected = store.decorate_observations(source_asset["id"], library.observations(source_asset["id"]))
    assert projected[0]["entities"][0]["character_id"] == target_role["id"]
    assert all(path.read_bytes() == data for path, data in originals.items())
    store.validate_all()


def test_bangumi_move_to_series_with_different_subject_keeps_its_concrete_archive(library):
    source_asset = seed(library, "asset_standalone", series="")
    destination = library.organization.save_project({"name": "另一个条目", "kind": "animation"})
    profiles = AnimeProfileStore(library)
    profiles.apply_asset(source_asset["id"], {"mode": "bangumi", "subject_id": 100}, bangumi_archive())
    profiles.apply_project(destination["id"], {"mode": "bangumi", "subject_id": 101}, bangumi_archive(101))
    library.organization.assign([source_asset["id"]], destination["id"])
    moved = library.get_asset(source_asset["id"])
    assert moved["project_id"] == destination["id"]
    assert moved["anime_profile"]["dossier"]["subject"]["id"] == 100
    assert library.organization.get_project(destination["id"])["anime_profile"]["dossier"]["subject"]["id"] == 101
    assert CharacterStore(library).context(source_asset["id"])["work_summary"]["id"] == 100


def test_bound_project_move_to_unassigned_preserves_dossier_cast_and_backup(library, tmp_path):
    source_asset = library.get_asset("asset_one")
    archive = bangumi_archive()
    profiles = AnimeProfileStore(library)
    profiles.apply_project(source_asset["project_id"], {"mode": "bangumi", "subject_id": 100}, archive)
    store = CharacterStore(library)
    source_role = store.catalog(source_asset["id"])["profiles"][0]
    appearance(library, store, character_id=source_role["id"])
    originals = {path: path.read_bytes() for path in library.asset_dir(source_asset["id"]).glob("analyses/*/observations/*.json")}
    library.organization.assign([source_asset["id"]], None)
    current_asset = library.get_asset(source_asset["id"])
    current_cast = store.catalog(source_asset["id"])
    assert current_asset["project_id"] is None
    assert current_asset["anime_profile"] == {"mode": "bangumi", "dossier": archive}
    assert len(current_cast["profiles"]) == 2
    assert sorted(profile["appearance_count"] for profile in current_cast["profiles"]) == [0, 1]
    assert all(path.read_bytes() == data for path, data in originals.items())
    store.validate_all()
    backup = library.backup()
    restored = Library(tmp_path / "restored_bangumi_move")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert restored.get_asset(source_asset["id"])["anime_profile"] == current_asset["anime_profile"]
    assert CharacterStore(restored).catalog(source_asset["id"]) == current_cast
    assert restored.records() == library.records()
    assert restored.organization.projects() == library.organization.projects()


@pytest.mark.parametrize("decision", ["merge", "delete"])
def test_bangumi_move_honors_destination_identity_merge_or_deletion(library, decision):
    source_asset = seed(library, "asset_standalone", series="")
    target_asset = seed(library, "asset_target", series="目标动画")
    profiles = AnimeProfileStore(library)
    selection = {"mode": "bangumi", "subject_id": 100}
    archive = bangumi_archive()
    profiles.apply_asset(source_asset["id"], selection, archive)
    profiles.apply_project(target_asset["project_id"], selection, archive)
    store = CharacterStore(library)
    source_role = next(profile for profile in store.catalog(source_asset["id"])["profiles"]
                       if profile["external_source"]["character_id"] == 200)
    target_roles = {profile["external_source"]["character_id"]: profile["id"]
                    for profile in store.catalog(target_asset["id"])["profiles"]}
    appearance(library, store, source_asset["id"], character_id=source_role["id"])
    expected_id = None
    if decision == "merge":
        expected_id = target_roles[201]
        store.update(target_asset["id"], expected_id, {"name": "目标合并姓名"})
        store.merge(target_asset["id"], target_roles[200], expected_id)
    else:
        store.delete(target_asset["id"], target_roles[200])
    library.organization.assign([source_asset["id"]], target_asset["project_id"])
    assert len(store.catalog(source_asset["id"])["profiles"]) == 1
    projected = store.decorate_observations(source_asset["id"], library.observations(source_asset["id"]))
    assert projected[0]["entities"][0]["character_id"] == expected_id
    if expected_id:
        assert store.catalog(source_asset["id"])["profiles"][0]["name"] == "目标合并姓名"
    store.validate_all()


def test_bangumi_move_to_new_project_keeps_source_deletion_effective(library):
    asset = seed(library, "asset_standalone", series="")
    profiles = AnimeProfileStore(library)
    profiles.apply_asset(asset["id"], {"mode": "bangumi", "subject_id": 100}, bangumi_archive())
    store = CharacterStore(library)
    deleted_role = next(profile for profile in store.catalog(asset["id"])["profiles"]
                        if profile["external_source"]["character_id"] == 200)
    store.delete(asset["id"], deleted_role["id"])
    destination = library.organization.save_project({"name": "新动画项目", "kind": "animation"})
    library.organization.assign([asset["id"]], destination["id"])
    assert [profile["external_source"]["character_id"] for profile in store.catalog(asset["id"])["profiles"]] == [201]
    store.validate_all()
