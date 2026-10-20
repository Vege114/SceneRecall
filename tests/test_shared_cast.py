"""Explicit work identity shares references without rewriting source evidence."""
import copy
from unittest.mock import patch

import pytest

from scenerecall.anime_profiles import AnimeProfileStore
from scenerecall.characters import CharacterStore
from scenerecall.library import Library, atomic_json, read_json
from scenerecall.shared_cast import shared_scope
from scenerecall.visual_identity import ReferenceGallery, VisualIdentityPipeline, validate_visual_artifacts
from test_organization import bangumi_archive
from test_visual_identity import picture


def episode(library, identity, number, *, subject=100, season=1, title="同名动画"):
    manifest = {"schema_version": "1.0", "id": identity, "work_id": "work_" + identity,
                "title": title, "series": "", "project_id": None, "kind": "animation",
                "season": season, "episode": number, "version": "test", "duration_ms": 10000,
                "fingerprint": identity, "created_at": "2026-10-01T00:00:00Z", "subtitle_mode": "none"}
    atomic_json(library.asset_dir(identity) / "manifest.json", manifest)
    # Reproduce archives written before sharing existed, including independently
    # downloaded references and different IDs for the same external character.
    with patch.object(library.organization, "share_anime_cast", return_value={}):
        asset = library.get_asset(identity)
    catalog = copy.deepcopy(library.organization._read())
    member = catalog["assets"][identity]
    member["anime_profile"] = {"mode": "bangumi", "dossier": bangumi_archive(subject)}
    store = CharacterStore(library)
    registry = store.seed_registry(store._read(asset["character_scope_id"]), member["anime_profile"]["dossier"])
    library.organization._commit(catalog, {asset["character_scope_id"]: registry})
    return asset, next(iter(registry["profiles"]))


def test_shares_bound_work_ids_refs_edits_and_preserves_old_evidence(tmp_path):
    library = Library(tmp_path / "library")
    first, old_one = episode(library, "asset_one", 1)
    second, old_two = episode(library, "asset_two", 2)
    assert old_one != old_two
    store, gallery = CharacterStore(library), ReferenceGallery(library)
    store.update("asset_one", old_one, {"name": "人工修正姓名", "notes": "人工笔记"})
    ref_one = gallery.add("asset_one", old_one, picture(), confirmed=True, single_character=True)
    ref_two = gallery.add("asset_two", old_two, picture(140), confirmed=False)
    directory = library.asset_dir("asset_two") / "identity" / "visual"
    raw_region = {"id": "region_test", "frame_id": "frame_test", "at_ms": 500,
                  "match": {"status": "candidate", "character_id": None, "candidate_character_id": old_two,
                            "candidates": [{"character_id": old_two, "reference_ids": [ref_two["id"]]}]}}
    atomic_json(directory / "regions" / "region_test.json", raw_region)
    atomic_json(directory / "decisions.json", {"decisions": {"region_test": {
        "status": "confirmed", "character_id": old_two, "source": "human", "reason": "原有人工决定"}}})
    before = {path: path.read_bytes() for path in (library.root / "assets").rglob("*.json")}
    old_registries = {scope: store._path(scope).read_bytes() for scope in
                      (first["character_scope_id"], second["character_scope_id"])}
    result = library.organization.share_anime_cast()
    assert result["migrated_count"] == 2
    cast_one, cast_two = store.catalog("asset_one"), store.catalog("asset_two")
    assert cast_one["scope_id"] == cast_two["scope_id"]
    role = cast_one["profiles"][0]
    assert role["id"] == cast_two["profiles"][0]["id"]
    assert role["name"] == "人工修正姓名" and role["notes"] == "人工笔记"
    references = gallery.list("asset_two")["references"]
    assert {row["id"] for row in references} == {ref_one["id"], ref_two["id"]}
    assert next(row for row in references if row["id"] == ref_one["id"])["status"] == "confirmed"
    projected = VisualIdentityPipeline(library)._project("asset_two", [raw_region])[0]
    assert projected["match"]["character_id"] == role["id"]
    assert projected["match"]["candidates"][0]["character_id"] == role["id"]
    assert all(path.read_bytes() == value for path, value in before.items())
    assert all(store._path(scope).read_bytes() == value for scope, value in old_registries.items())
    member = library.organization._read()["assets"]["asset_two"]
    assert member["bindings"][second["character_scope_id"]][old_two] == role["id"]
    assert library.organization.share_anime_cast()["migrated_count"] == 0
    gallery.update("asset_two", role["id"], ref_one["id"], status="revoked", reason="跨集撤销")
    assert gallery.list("asset_one", role["id"])["references"][0]["status"] == "revoked"
    store.update("asset_two", role["id"], {"name": "跨集改名"})
    assert store.catalog("asset_one")["profiles"][0]["name"] == "跨集改名"


@pytest.mark.parametrize("other", [{"subject": 101}, {"season": 2}])
def test_same_title_different_subject_or_season_never_shares(tmp_path, other):
    library = Library(tmp_path / "library")
    episode(library, "asset_one", 1)
    episode(library, "asset_two", 2, **other)
    library.organization.share_anime_cast()
    assert library.get_asset("asset_one")["character_scope_id"] != library.get_asset("asset_two")["character_scope_id"]


def test_new_episode_and_profile_refresh_join_existing_shared_cast_offline(tmp_path):
    library = Library(tmp_path / "library")
    episode(library, "asset_one", 1)
    library.organization.share_anime_cast()
    role = CharacterStore(library).catalog("asset_one")["profiles"][0]
    CharacterStore(library).update("asset_one", role["id"], {"name": "共享姓名"})
    episode(library, "asset_two", 2)
    AnimeProfileStore(library).apply_asset("asset_two", {"mode": "bangumi", "subject_id": 100}, bangumi_archive())
    assert CharacterStore(library).catalog("asset_two")["profiles"][0]["id"] == role["id"]
    assert CharacterStore(library).catalog("asset_two")["profiles"][0]["name"] == "共享姓名"
    restored = Library(library.root)
    assert restored.organization.share_anime_cast()["migrated_count"] == 0
    CharacterStore(restored).validate_all()
    validate_visual_artifacts(restored)


def test_explicit_deletion_is_not_resurrected_when_casts_join(tmp_path):
    library = Library(tmp_path / "library")
    _, role = episode(library, "asset_one", 1)
    episode(library, "asset_two", 2)
    external_id = CharacterStore(library).catalog("asset_one")["profiles"][0]["external_source"]["character_id"]
    CharacterStore(library).delete("asset_one", role)
    library.organization.share_anime_cast()
    for identity in ("asset_one", "asset_two"):
        profiles = CharacterStore(library).catalog(identity)["profiles"]
        assert len(profiles) == 1
        assert all(row["external_source"]["character_id"] != external_id for row in profiles)


def test_project_cast_compatibility_is_unchanged():
    member = {"project_id": "project_existing", "season": 1, "episode": 1,
              "anime_profile": {"mode": "bangumi", "dossier": bangumi_archive()}}
    assert shared_scope(member) is None


def test_shared_references_survive_moves_and_backup_restore(tmp_path):
    library = Library(tmp_path / "library")
    _, role = episode(library, "asset_one", 1)
    ref = ReferenceGallery(library).add("asset_one", role, picture(), confirmed=True, single_character=True)
    episode(library, "asset_two", 2)
    library.organization.share_anime_cast()
    scope = library.get_asset("asset_one")["character_scope_id"]
    project = library.organization.save_project({"name": "明确移动目标"})
    library.organization.assign(["asset_one"], project["id"])
    assert ReferenceGallery(library).list("asset_one")["references"][0]["id"] == ref["id"]
    library.organization.assign(["asset_one"], None)
    assert library.get_asset("asset_one")["character_scope_id"] == scope
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert {row["id"] for row in ReferenceGallery(restored).list("asset_two")["references"]} == {ref["id"]}
    assert CharacterStore(restored).catalog("asset_one")["profiles"] == CharacterStore(restored).catalog("asset_two")["profiles"]
    CharacterStore(restored).validate_all()
    validate_visual_artifacts(restored)


def test_gallery_journal_recovers_before_membership_publish(tmp_path, monkeypatch):
    library = Library(tmp_path / "library")
    _, role = episode(library, "asset_one", 1)
    ref = ReferenceGallery(library).add("asset_one", role, picture())
    episode(library, "asset_two", 2)
    organization = library.organization
    original = organization.recover
    calls = 0

    def interrupted():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic crash after journal prepare")
        return original()

    monkeypatch.setattr(organization, "recover", interrupted)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        organization.share_anime_cast()
    monkeypatch.setattr(organization, "recover", original)
    organization.recover()
    assert not organization.journal.exists()
    assert ReferenceGallery(library).list("asset_two")["references"][0]["id"] == ref["id"]
    assert read_json(organization.path)["assets"]["asset_one"]["character_scope_id"] == library.get_asset("asset_two")["character_scope_id"]
