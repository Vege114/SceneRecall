"""Cast identity, human correction and immutable-evidence storage regressions."""
import json
import zipfile

import pytest

from scenerecall.characters import CharacterStore, StaleCharacterRevision, scope_id
from scenerecall.library import Library, atomic_json, read_json
from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine


def seed(library, asset_id="asset_one", *, series="合成剧集", kind="animation", season=1, episode=1):
    asset = {"schema_version": "1.0", "id": asset_id, "work_id": "work_" + asset_id,
             "title": asset_id, "series": series, "kind": kind, "season": season, "episode": episode,
             "created_at": "2026-09-29T00:00:00Z", "duration_ms": 10000,
             "version": "test", "fingerprint": "synthetic" + asset_id}
    atomic_json(library.asset_dir(asset_id) / "manifest.json", asset)
    frame = library.asset_dir(asset_id) / "frames" / "frame_one.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"synthetic frame; storage test does not decode images")
    library.register_frame(asset_id, {"id": "frame_one", "path": str(frame), "at_ms": 0})
    return asset


@pytest.fixture
def cast(tmp_path):
    library = Library(tmp_path / "library")
    seed(library)
    return library, CharacterStore(library)


def observation(asset_id="asset_one", *, number=1, character_id=None, kind="person", **entity):
    return {"id": f"obs_{asset_id}_{number}", "asset_id": asset_id, "run_id": "run_test",
            "window_id": f"window_{number}", "shot_id": "shot_test", "start_ms": (number - 1) * 1000,
            "end_ms": number * 1000, "summary": "人物拿起杯子", "evidence_frame_ids": ["frame_one"],
            "entities": [{"id": "entity_one", "kind": kind, "description": "穿红衣服的人",
                          "character_id": character_id, "evidence_frame_ids": ["frame_one"], **entity}]}


def save(library, store, record, revision=None):
    with library.lock:
        linked = store.apply_observation(record["asset_id"], record, expected_revision=revision)
        library.save_observation(record["asset_id"], linked)
    return linked["entities"][0]["character_id"]


def test_series_scope_normalizes_names_and_spans_seasons_without_crossing_unrelated_works(cast):
    library, store = cast
    original = seed(library, series="  Ｍｙ   SERIES  ")
    sequel = seed(library, "asset_two", series="my series", season=2, episode=3)
    unrelated = seed(library, "asset_other", series="Different show")
    adaptation = seed(library, "asset_live", series="my series", kind="series")
    assert scope_id(original) == scope_id(sequel)
    assert scope_id(original) != scope_id(unrelated)
    assert scope_id(original) != scope_id(adaptation)
    character_id = save(library, store, observation())
    assert store.catalog("asset_two")["profiles"][0]["id"] == character_id
    assert store.catalog("asset_other")["profiles"] == []
    assert {asset["id"] for asset in store.catalog("asset_one")["assets"]} == {"asset_one", "asset_two"}
    seed(library, "asset_single", series="  ")
    seed(library, "asset_another_single", series="")
    assert store.context("asset_single")["scope_id"] == "work_asset_single"
    assert store.context("asset_single")["scope_id"] != store.context("asset_another_single")["scope_id"]


def test_match_reuses_profile_across_episodes_and_enriches_it(cast):
    library, store = cast
    seed(library, "asset_two", season=2, episode=1)
    character_id = save(library, store, observation(observed_traits=["红色外套"]))
    linked = save(library, store, observation("asset_two", character_id=character_id,
                                          description="戴黑色眼镜", observed_traits=["黑色眼镜"]))
    catalog = store.catalog("asset_one")
    assert linked == character_id and len(catalog["profiles"]) == 1
    profile = catalog["profiles"][0]
    assert profile["appearance_count"] == 2
    assert profile["appearance"] == ["红色外套", "黑色眼镜"]
    assert "戴黑色眼镜" in profile["description"]
    assert profile["representative"] == {"asset_id": "asset_two", "frame_id": "frame_one"}
    assert catalog["revision"] == 0
    assert store.context("asset_one")["profiles"][0]["updated_at"]


def test_human_fields_are_authoritative_while_other_fields_enrich(cast):
    library, store = cast
    character_id = save(library, store, observation())
    store.update("asset_one", character_id, {"name": "小葵", "aliases": ["阿葵"],
                                          "description": "人工确认：左眼有疤", "notes": "不是她的姐姐",
                                          "expected_revision": 0})
    save(library, store, observation(number=2, character_id=character_id, description="模型猜测的描述",
                                     observed_traits=["雨衣"]), revision=1)
    profile = store.catalog("asset_one")["profiles"][0]
    assert profile["name"] == "小葵" and profile["aliases"] == ["阿葵"]
    assert profile["description"] == "人工确认：左眼有疤" and profile["notes"] == "不是她的姐姐"
    assert profile["appearance"] == ["雨衣"] and profile["user_edited"]
    store.update("asset_one", character_id, {"appearance": [], "expected_revision": 1})
    save(library, store, observation(number=3, character_id=character_id, observed_traits=["红衣"]), revision=2)
    assert store.catalog("asset_one")["profiles"][0]["appearance"] == []


def test_human_edit_rejects_stale_inflight_result_without_creating_profile(cast):
    library, store = cast
    character_id = save(library, store, observation())
    store.update("asset_one", character_id, {"name": "新版姓名"})
    before = store.catalog("asset_one")
    with pytest.raises(StaleCharacterRevision):
        store.apply_observation("asset_one", observation(number=2), expected_revision=0)
    with pytest.raises(StaleCharacterRevision):
        store.update("asset_one", character_id, {"name": "旧浏览器修改", "expected_revision": 0})
    with pytest.raises(StaleCharacterRevision):
        store.delete("asset_one", character_id, expected_revision=0)
    assert store.catalog("asset_one") == before


def test_merges_and_deletion_resolve_historical_links_without_rewriting_observations(cast):
    library, store = cast
    first = save(library, store, observation())
    second = save(library, store, observation(number=2, description="侧脸角色"))
    files = list(library.asset_dir("asset_one").glob("analyses/*/observations/*.json"))
    originals = {path: path.read_bytes() for path in files}
    store.update("asset_one", first, {"name": "小葵", "aliases": ["阿葵"]})
    store.update("asset_one", second, {"name": "葵同学"})
    merged = store.merge("asset_one", second, first, expected_revision=2)
    assert merged["id"] == first and merged["appearance_count"] == 2
    assert merged["aliases"] == ["阿葵", "葵同学"]
    assert {entity["character_id"] for row in library.records() for entity in row["entities"]} == {first}
    assert all(row["entities"][0]["character_name"] == "小葵" for row in library.records())
    store.delete("asset_one", first, expected_revision=3)
    assert store.catalog("asset_one")["profiles"] == []
    assert all(row["entities"][0]["character_id"] is None for row in library.records())
    assert all("小葵" not in row["character"] for row in library.records())
    assert all(path.read_bytes() == content for path, content in originals.items())
    # A delayed result referencing a deleted/merged identity cannot resurrect it.
    assert store.apply_observation("asset_one", observation(number=3, character_id=second), 4)["entities"][0]["character_id"] is None


def test_merge_keeps_target_corrections_and_retains_source_information_in_notes(cast):
    library, store = cast
    target = save(library, store, observation())
    source = save(library, store, observation(number=2))
    store.update("asset_one", target, {"name": "小葵", "description": "人工确认主描述", "notes": "目标备注",
                                       "appearance": ["左眼有疤"]})
    store.update("asset_one", source, {"name": "另一角度", "description": "后背有图案", "notes": "来源备注",
                                       "appearance": ["右手戴手套"]})
    merged = store.merge("asset_one", source, target)
    assert merged["description"] == "人工确认主描述" and merged["appearance"] == ["左眼有疤"]
    for information in ("目标备注", "来源备注", "后背有图案", "右手戴手套"):
        assert information in merged["notes"]
    assert "notes" in merged["edited_fields"]


def test_current_portrait_uses_latest_episode_then_shot_and_matches_ai_context(cast):
    library, store = cast
    seed(library, "asset_two", season=2, episode=1)
    first = save(library, store, observation(portrait_frame_id="frame_one", portrait_box=[0, 0, .4, .5]))
    save(library, store, observation("asset_two", character_id=first,
                                    portrait_frame_id="frame_one", portrait_box=[.2, .1, .6, .8]))
    expected = {"asset_id": "asset_two", "frame_id": "frame_one", "box": [.2, .1, .6, .8]}
    assert store.portrait_reference("asset_one", first) == expected
    assert store.context("asset_one")["profiles"][0]["representative"] == expected
    # Processing an older episode later must not replace the latest appearance.
    save(library, store, observation(number=2, character_id=first, portrait_frame_id="frame_one", portrait_box=[0, 0, 1, 1]))
    assert store.portrait_reference("asset_one", first) == expected
    save(library, store, observation("asset_two", number=2, character_id=first,
                                    portrait_frame_id="frame_one", portrait_box=[.4, .4, .3, .3]))
    assert store.portrait_reference("asset_one", first)["box"] == [.4, .4, .3, .3]
    # A new interpretation detaches all newer-episode appearances. Return the
    # latest still-linked earlier scene, not the stale persisted portrait.
    replacement = observation("asset_two", kind="object", is_character=False)
    replacement.update(run_id="run_new", end_ms=3000)
    save(library, store, replacement)
    assert store.portrait_reference("asset_one", first) == {
        "asset_id": "asset_one", "frame_id": "frame_one", "box": [0, 0, 1, 1]}


@pytest.mark.parametrize("fields", [
    {"portrait_box": [0, 0, 1, 1]},
    {"portrait_frame_id": "unknown", "portrait_box": [0, 0, 1, 1]},
    {"portrait_frame_id": "frame_one", "portrait_box": [-.1, 0, 1, 1]},
    {"portrait_frame_id": "frame_one", "portrait_box": [0, 0, 0, 1]},
    {"portrait_frame_id": "frame_one", "portrait_box": [.9, 0, .2, 1]},
    {"portrait_frame_id": "frame_one", "portrait_box": [0, 0, float("nan"), 1]},
    {"portrait_frame_id": "frame_one", "portrait_box": [0, 0, True, 1]},
])
def test_portrait_invalid_box_or_frame_cannot_mutate_registry(cast, fields):
    _library, store = cast
    with pytest.raises(ValueError, match="角色图片"):
        store.apply_observation("asset_one", observation(**fields))
    assert store.catalog("asset_one")["profiles"] == []


def test_merge_notes_over_old_limit_remain_editable_and_larger_merge_rejects_without_data_loss(cast):
    library, store = cast
    first = save(library, store, observation())
    second = save(library, store, observation(number=2))
    store.update("asset_one", first, {"notes": "甲" * 6000})
    store.update("asset_one", second, {"notes": "乙" * 6000})
    merged = store.merge("asset_one", second, first)
    assert len(merged["notes"]) > 10000
    assert store.update("asset_one", first, {"notes": merged["notes"]})["notes"] == merged["notes"]
    third = save(library, store, observation(number=3))
    store.update("asset_one", third, {"notes": "丙" * 90000})
    before = store.catalog("asset_one")
    with pytest.raises(ValueError, match="100000"):
        store.merge("asset_one", third, first)
    assert store.catalog("asset_one") == before


@pytest.mark.parametrize("kind,is_character,expected", [
    ("person", True, True), ("character", True, True), ("anthropomorphic_character", True, True),
    ("animal", True, True), ("other", True, True), ("animal", False, False),
    ("object", True, False), ("person", False, False),
])
def test_only_character_entities_get_profiles(cast, kind, is_character, expected):
    library, store = cast
    result = store.apply_observation("asset_one", observation(kind=kind, is_character=is_character))
    assert bool(result["entities"][0]["character_id"]) is expected
    assert bool(store.catalog("asset_one")["profiles"]) is expected


def test_cross_scope_merge_and_model_references_are_rejected(cast):
    library, store = cast
    seed(library, "asset_two", series="不相关的剧")
    first = save(library, store, observation())
    other = save(library, store, observation("asset_two"))
    with pytest.raises(ValueError, match="不存在或不属于"):
        store.merge("asset_one", first, other)
    with pytest.raises(ValueError, match="不存在或不属于"):
        store.apply_observation("asset_one", observation(number=2, character_id=other))
    with pytest.raises(ValueError):
        store.apply_observation("asset_one", observation(number=2, character_id="../../invalid"))
    with pytest.raises(ValueError, match="可信度"):
        store.apply_observation("asset_one", observation(number=2, character_id=first, match_confidence=0.4))
    assert len(store.catalog("asset_one")["profiles"]) == 1


@pytest.mark.asyncio
async def test_profile_names_and_aliases_search_across_episodes_and_preserve_legacy_annotations(cast):
    library, store = cast
    seed(library, "asset_two", episode=2)
    character_id = save(library, store, observation())
    save(library, store, observation("asset_two", character_id=character_id))
    store.update("asset_one", character_id, {"name": "小葵", "aliases": ["太阳花"]})
    engine = SearchEngine(library.root / "indexes", ProviderManager(library.root / "private"))
    await engine.rebuild(library.records())
    assert len((await engine.search("太阳花"))["results"]) == 2
    assert len((await engine.search("杯子", {"character": "小葵"}))["results"]) == 2
    library.annotate("asset_one", {"record_id": "obs_asset_one_1", "entity_id": "entity_one",
                                   "character_name": "旧人工姓名", "aliases": ["旧别名"], "favorite": True})
    row = library.records("asset_one")[0]
    assert row["entities"][0]["name"] == "旧人工姓名"
    assert row["entities"][0]["character_name"] == "旧人工姓名"
    assert row["entities"][0]["aliases"] == ["旧别名"] and row["favorite"]
    assert row["review_status"] == "user_confirmed"


def test_backup_roundtrip_preserves_edited_profiles_redirects_and_evidence(cast, tmp_path):
    library, store = cast
    first = save(library, store, observation())
    second = save(library, store, observation(number=2))
    store.update("asset_one", first, {"name": "小葵", "notes": "人工作品设定"})
    store.merge("asset_one", second, first)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert CharacterStore(restored).catalog("asset_one") == store.catalog("asset_one")
    assert restored.records() == library.records()


@pytest.mark.parametrize("corruption", ["scope", "cycle", "foreign_evidence"])
def test_restore_rejects_invalid_registry_before_importing_any_assets(cast, tmp_path, corruption):
    library, store = cast
    first = save(library, store, observation())
    seed(library, "asset_other", series="其他作品")
    scope = store.context("asset_one")["scope_id"]
    path = library.root / "works" / scope / "characters.json"
    registry = read_json(path)
    if corruption == "scope":
        registry["scope_id"] = "wrong_scope"
    elif corruption == "cycle":
        registry["redirects"] = {"old_a": "old_b", "old_b": "old_a"}
    else:
        registry["profiles"][first]["representative"]["asset_id"] = "asset_other"
    backup = library.backup()
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(library.root / "runtime" / "backups" / f"{backup['id']}.zip") as source:
        with zipfile.ZipFile(damaged, "w") as destination:
            for info in source.infolist():
                data = json.dumps(registry).encode() if info.filename == str(path.relative_to(library.root)) else source.read(info)
                destination.writestr(info, data)
    restored = Library(tmp_path / "restored")
    with pytest.raises(ValueError):
        restored.restore(damaged)
    assert restored.list_assets() == []
