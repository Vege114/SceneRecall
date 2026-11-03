"""Bangumi enrichment preserves IDs, source ranges and human review decisions."""
import copy

import pytest

from scenerecall.bangumi_knowledge import sync_bangumi_knowledge
from scenerecall.characters import CharacterStore, scope_id
from scenerecall.knowledge_store import KnowledgeCardInput, KnowledgeStore
from scenerecall.library import Library, atomic_json, read_json
from scenerecall.work_entities import WorkEntityStore


def dossier(subject_id=42):
    return {"source": "bangumi", "fetched_at": "2026-10-28T00:00:00Z", "subject": {
        "id": subject_id, "name": "合成动画", "name_cn": "合成动画", "summary": "只用于测试的学校故事。",
        "infobox": [{"key": "原作", "value": "合成制作委员会"}]}, "characters": [
        {"id": 7, "name": "テスト", "name_cn": "测试角色", "aliases": ["Rei"],
         "description": "📖她就读于星光学校，与另一角色共同练习。", "role": "主角", "infobox": [
             {"key": "别名", "value": [{"k": "英文名", "v": "Rei"}]},
             {"key": "学校", "value": "星光学校"}, {"key": "声优", "value": "测试演员"}]},
        {"id": 8, "name": "另一角色", "aliases": [], "description": "另一角色负责维护装置。", "infobox": []}]}


@pytest.fixture
def prepared(tmp_path):
    library = Library(tmp_path / "library")
    for identity, episode, subject in (("episode-one", 1, 42), ("episode-two", 2, 42), ("other-work", 1, 43)):
        atomic_json(library.asset_dir(identity) / "manifest.json", {
            "schema_version": "1.0", "id": identity, "work_id": "legacy-" + identity,
            "title": "合成动画", "series": "测试系列", "kind": "animation", "episode": episode,
            "season": 1, "created_at": "2026-10-28T00:00:00Z", "duration_ms": 1000,
            "version": "test", "fingerprint": "source-" + identity,
            "anime_profile": {"mode": "bangumi", "dossier": dossier(subject)}})
    for asset in library.list_assets():
        CharacterStore(library).seed_bangumi(scope_id(asset), asset["anime_profile"]["dossier"])
    return library, WorkEntityStore(library)


def refresh(library, asset_id, change):
    catalog = library.organization._read()
    change(catalog["assets"][asset_id]["anime_profile"]["dossier"])
    library.organization._commit(catalog)


def test_import_creates_sources_and_candidate_cards_using_existing_identity(prepared):
    library, entities = prepared
    result = sync_bangumi_knowledge(library, "episode-one")
    scope = result["work_scope_id"]
    knowledge = KnowledgeStore(entities)
    assert result["created_sources"] == 3 and result["created_cards"] == 3
    assert entities.scope("episode-two")["work_scope_id"] == scope
    assert entities.scope("other-work") is None
    assert entities.scope("episode-one")["entities"] == {}
    assert entities.scope("episode-one")["bindings"] == {}
    assert knowledge.projection(scope)["cards"] == []
    registry = CharacterStore(library)._read(scope_id(library.get_asset("episode-one")))
    allowed_ids = set(registry["profiles"])
    for card in knowledge.cards(scope):
        assert card["review_status"] == "candidate"
        assert card["target_ref"]["character_id"] in allowed_ids
        assert card["target_ref"]["asset_id"] == "episode-one"
        assert card["sources"][0]["quote"] == card["text"]
        assert card["sources"][0]["url"].startswith("https://bgm.tv/character/")
        assert card["text"] != "测试演员" and card["text"] != "Rei"
    subject = next(source for source in knowledge.sources(scope) if source["origin"]["record_kind"] == "subject")
    content = knowledge.source_content(scope, subject["source_id"])
    assert "合成制作委员会" in content["text"]
    assert subject["origin"]["fetched_at"]


def test_replay_and_other_episode_do_not_duplicate_or_reset_review(prepared):
    library, entities = prepared
    result = sync_bangumi_knowledge(library, "episode-one")
    scope = result["work_scope_id"]
    knowledge = KnowledgeStore(entities)
    cards = knowledge.cards(scope)
    payload = {key: cards[0][key] for key in KnowledgeCardInput.model_fields if key in cards[0]}
    knowledge.save_card(scope, {**payload, "review_status": "reviewed"}, cards[0]["card_id"])
    knowledge.revoke_card(scope, cards[1]["card_id"], "资料解读有误")
    before = entities._path(scope).read_bytes()
    repeated = sync_bangumi_knowledge(library, "episode-two")
    assert repeated["created_sources"] == repeated["created_cards"] == 0
    assert repeated["sources_reused"] == 3
    assert entities._path(scope).read_bytes() == before
    assert {row["review_status"] for row in knowledge.cards(scope)} == {"reviewed", "revoked", "candidate"}
    assert len(knowledge.projection(scope)["cards"]) == 1


def source_only_import(library, monkeypatch):
    original = KnowledgeStore._target
    registry = CharacterStore(library)._read(scope_id(library.get_asset("episode-one")))
    character_id = next(profile["id"] for profile in registry["profiles"].values()
                        if profile["external_source"]["character_id"] == 7)
    def unresolved(store, saved, reference):
        if reference.get("character_id") == character_id:
            raise ValueError("temporarily unresolved exact identity")
        return original(store, saved, reference)
    monkeypatch.setattr(KnowledgeStore, "_target", unresolved)
    result = sync_bangumi_knowledge(library, "episode-one")
    monkeypatch.setattr(KnowledgeStore, "_target", original)
    return result, character_id


def test_same_source_retry_fills_missing_candidates_after_exact_identity_is_available(prepared, monkeypatch):
    library, entities = prepared
    first, character_id = source_only_import(library, monkeypatch)
    assert first["created_sources"] == 3 and first["created_cards"] == 1 and first["warnings"]
    scope = first["work_scope_id"]
    knowledge = KnowledgeStore(entities)
    source_ids = {row["source_id"] for row in knowledge.sources(scope)}
    existing = knowledge.cards(scope)[0]
    payload = {key: existing[key] for key in KnowledgeCardInput.model_fields if key in existing}
    knowledge.save_card(scope, {**payload, "review_status": "reviewed"}, existing["card_id"])
    retry = sync_bangumi_knowledge(library, "episode-one")
    assert retry["created_sources"] == 0 and retry["created_cards"] == 2
    assert source_ids == {row["source_id"] for row in knowledge.sources(scope)}
    cards = knowledge.cards(scope)
    assert next(row for row in cards if row["card_id"] == existing["card_id"])["review_status"] == "reviewed"
    assert len([row for row in cards if row["target_ref"]["character_id"] == character_id
                and row["review_status"] == "candidate"]) == 2
    assert sync_bangumi_knowledge(library, "episode-one")["created_cards"] == 0


def test_same_source_retry_respects_manual_source_withdrawal(prepared, monkeypatch):
    library, entities = prepared
    first, _ = source_only_import(library, monkeypatch)
    scope = first["work_scope_id"]
    knowledge = KnowledgeStore(entities)
    source = next(row for row in knowledge.sources(scope) if row["origin"]["record_kind"] == "character"
                  and row["origin"]["record_id"] == 7)
    knowledge.revoke_source(scope, source["source_id"], "不要使用这个来源")
    retry = sync_bangumi_knowledge(library, "episode-one")
    assert retry["created_sources"] == retry["created_cards"] == 0
    assert len(knowledge.cards(scope)) == 1


def test_same_source_retry_preserves_manual_card_for_exact_quote(prepared, monkeypatch):
    library, entities = prepared
    first, character_id = source_only_import(library, monkeypatch)
    scope = first["work_scope_id"]
    knowledge = KnowledgeStore(entities)
    source = next(row for row in knowledge.sources(scope) if row["origin"]["record_kind"] == "character"
                  and row["origin"]["record_id"] == 7)
    text = knowledge.source_content(scope, source["source_id"])["text"]
    quote = dossier()["characters"][0]["description"]
    start = text.index(quote)
    card = knowledge.save_card(scope, {"text": "我已手工整理的说明", "target_ref": {
        "kind": "character", "asset_id": "episode-one", "character_scope_id": scope_id(library.get_asset("episode-one")),
        "character_id": character_id}, "source_refs": [{"source_id": source["source_id"],
            "snapshot_id": source["snapshot_id"], "start": start, "end": start + len(quote)}],
        "review_status": "reviewed"})
    retry = sync_bangumi_knowledge(library, "episode-one")
    assert retry["created_sources"] == 0 and retry["created_cards"] == 1  # Only the missing school field.
    current = next(row for row in knowledge.cards(scope) if row["card_id"] == card["card_id"])
    assert current["text"] == "我已手工整理的说明" and current["review_status"] == "reviewed"
    assert not any(row["text"] == quote for row in knowledge.cards(scope))


def test_revoked_source_is_not_resurrected_on_same_content(prepared):
    library, entities = prepared
    scope = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    knowledge = KnowledgeStore(entities)
    source = next(row for row in knowledge.sources(scope) if row["origin"]["record_kind"] == "character")
    knowledge.revoke_source(scope, source["source_id"], "来源存在错误")
    sync_bangumi_knowledge(library, "episode-one")
    assert len(knowledge.sources(scope)) == 3
    assert next(row for row in knowledge.sources(scope) if row["source_id"] == source["source_id"])["status"] == "revoked"


def test_refresh_invalidates_old_source_without_overwriting_human_text(prepared):
    library, entities = prepared
    scope = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    knowledge = KnowledgeStore(entities)
    card = next(row for row in knowledge.cards(scope) if row["text"].startswith("📖"))
    payload = {key: card[key] for key in KnowledgeCardInput.model_fields if key in card}
    knowledge.save_card(scope, {**payload, "text": "人工核对后的说明", "review_status": "reviewed"}, card["card_id"])
    before = knowledge.projection(scope)["signature"]
    refresh(library, "episode-one", lambda row: row["characters"][0].update(description="她已从星光学校毕业。"))
    result = sync_bangumi_knowledge(library, "episode-one")
    assert result["created_sources"] == 1 and result["created_cards"] == 2
    old = next(row for row in knowledge.cards(scope) if row["card_id"] == card["card_id"])
    assert old["text"] == "人工核对后的说明" and old["review_status"] == "reviewed"
    assert old["effective_status"] == "unavailable"
    assert knowledge.projection(scope)["cards"] == []
    assert knowledge.projection(scope)["signature"] != before
    assert any(row["text"] == "她已从星光学校毕业。" and row["review_status"] == "candidate"
               for row in knowledge.cards(scope))


def test_rejected_unchanged_field_does_not_return_as_new_candidate(prepared):
    library, entities = prepared
    scope = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    knowledge = KnowledgeStore(entities)
    school = next(row for row in knowledge.cards(scope) if row["text"] == "星光学校")
    knowledge.revoke_card(scope, school["card_id"], "此字段不适合用于知识检索")
    refresh(library, "episode-one", lambda row: row["characters"][0].update(description="新的角色简介"))
    result = sync_bangumi_knowledge(library, "episode-one")
    assert result["created_cards"] == 1
    assert len([row for row in knowledge.cards(scope) if row["text"] == "星光学校"]) == 1


def test_cross_work_same_names_and_shared_cast_do_not_share_knowledge(prepared):
    library, entities = prepared
    one = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    other = sync_bangumi_knowledge(library, "other-work")["work_scope_id"]
    assert one != other
    assert entities.scope("other-work")["bangumi_knowledge"]["subject_id"] == 43
    with pytest.raises(ValueError, match="指定作品"):
        sync_bangumi_knowledge(library, "episode-one", scope_id=other)
    assert all(row["origin"]["subject_id"] == 42 for row in KnowledgeStore(entities).sources(one))


def test_project_move_keeps_scope_and_cards_without_treating_it_as_version_change(prepared):
    library, entities = prepared
    scope = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    before = entities._path(scope).read_bytes()
    old_anime_id = library.get_asset("episode-one")["anime_id"]
    project = library.organization.save_project({"name": "重新整理的项目", "kind": "animation"})
    library.organization.assign(["episode-one"], project["id"])
    assert library.get_asset("episode-one")["anime_id"] != old_anime_id
    result = sync_bangumi_knowledge(library, "episode-one")
    assert result["work_scope_id"] == scope
    assert result["created_sources"] == result["created_cards"] == 0
    assert entities._path(scope).read_bytes() == before
    assert any("请编辑资料卡并重新选择内部对象" in warning for warning in result["warnings"])


@pytest.mark.parametrize("changes", [{"season": 2}, {"episode": None}])
def test_auto_scope_still_rejects_actual_season_or_media_version_change(prepared, changes):
    library, entities = prepared
    scope = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    path = library.asset_dir("episode-one") / "manifest.json"
    manifest = read_json(path)
    manifest.update(changes)
    atomic_json(path, manifest)
    catalog = library.organization._read()
    catalog["assets"]["episode-one"].update(changes)
    library.organization._commit(catalog)
    before = entities._path(scope).read_bytes()
    with pytest.raises(ValueError, match="另一动画版本"):
        sync_bangumi_knowledge(library, "episode-one")
    assert entities._path(scope).read_bytes() == before


def test_explicit_scope_refuses_another_subject(prepared):
    library, entities = prepared
    explicit = entities.create_scope("人工版本", "TV", ["episode-one", "other-work"])["work_scope_id"]
    sync_bangumi_knowledge(library, "episode-one", scope_id=explicit)
    with pytest.raises(ValueError, match="另一 Bangumi"):
        sync_bangumi_knowledge(library, "other-work", scope_id=explicit)


def test_latest_snapshot_used_and_partial_bad_character_does_not_abort(prepared):
    library, entities = prepared
    def change(row):
        latest = copy.deepcopy(row)
        latest["characters"][0]["description"] = "最新收到的简介"
        latest["characters"][1]["description"] = "\x00损坏内容"
        row["knowledge_snapshot"] = latest
    refresh(library, "episode-one", change)
    result = sync_bangumi_knowledge(library, "episode-one")
    cards = KnowledgeStore(entities).cards(result["work_scope_id"])
    assert result["status"] == "partial" and result["warnings"]
    assert any(row["text"] == "最新收到的简介" for row in cards)
    assert not any(row["text"].startswith("📖") for row in cards)


def test_failed_refresh_keeps_previous_good_source_and_manual_cards(prepared):
    library, entities = prepared
    scope = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    knowledge = KnowledgeStore(entities)
    old_card = next(row for row in knowledge.cards(scope) if row["text"].startswith("📖"))
    manual_source = knowledge.import_source(scope, "人工来源", "", "人工补充的资料".encode(), "manual.txt")
    manual_card = knowledge.save_card(scope, {"text": "人工补充的资料", "target_ref": old_card["target_ref"],
        "source_refs": [{"source_id": manual_source["source_id"], "snapshot_id": manual_source["snapshot_id"],
                         "start": 0, "end": len("人工补充的资料")}], "review_status": "reviewed"})
    refresh(library, "episode-one", lambda row: row["characters"][0].update(description="\x00损坏的新内容"))
    result = sync_bangumi_knowledge(library, "episode-one")
    assert result["warnings"]
    old_source_id = old_card["source_refs"][0]["source_id"]
    assert next(row for row in knowledge.sources(scope) if row["source_id"] == old_source_id)["status"] == "active"
    assert next(row for row in knowledge.cards(scope) if row["card_id"] == manual_card["card_id"])["effective_status"] == "reviewed"


def test_publication_failure_can_retry_without_duplicate_cards(prepared, monkeypatch):
    library, entities = prepared
    commit = WorkEntityStore._commit
    def fail_sync(store, saved, action, *args, **kwargs):
        if action == "bangumi_knowledge_sync":
            raise OSError("simulated publication interruption")
        return commit(store, saved, action, *args, **kwargs)
    monkeypatch.setattr(WorkEntityStore, "_commit", fail_sync)
    with pytest.raises(OSError, match="interruption"):
        sync_bangumi_knowledge(library, "episode-one")
    scope = entities.scope("episode-one")["work_scope_id"]
    assert KnowledgeStore(entities).cards(scope) == []
    monkeypatch.setattr(WorkEntityStore, "_commit", commit)
    result = sync_bangumi_knowledge(library, "episode-one")
    assert result["created_sources"] == result["source_count"] == 3
    assert result["created_cards"] == result["card_count"] == 3


def test_removed_roster_entry_loses_active_sources(prepared):
    library, entities = prepared
    scope = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    refresh(library, "episode-one", lambda row: row["characters"].pop())
    sync_bangumi_knowledge(library, "episode-one")
    source = next(row for row in KnowledgeStore(entities).sources(scope) if row["origin"]["record_id"] == 8)
    assert source["status"] == "revoked" and source["revoked_by_bangumi_refresh"]


def test_readded_roster_role_gets_new_candidate_without_restoring_old_review(prepared):
    library, entities = prepared
    scope = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    old_sources = {row["source_id"] for row in KnowledgeStore(entities).sources(scope)}
    refresh(library, "episode-one", lambda row: row["characters"].pop())
    sync_bangumi_knowledge(library, "episode-one")
    refresh(library, "episode-one", lambda row: row["characters"].append(dossier()["characters"][1]))
    result = sync_bangumi_knowledge(library, "episode-one")
    assert result["created_sources"] == result["created_cards"] == 1
    source = next(row for row in KnowledgeStore(entities).sources(scope) if row["source_id"] not in old_sources)
    assert source["status"] == "active"
    assert KnowledgeStore(entities).projection(scope)["cards"] == []


@pytest.mark.parametrize("old_time", ["2026-10-01T00:00:00Z", "invalid-date", ""])
def test_old_episode_snapshot_cannot_roll_back_new_sources_or_roster(prepared, old_time):
    library, entities = prepared
    refresh(library, "episode-one", lambda row: row.update(fetched_at="2026-10-20T00:00:00Z"))
    scope = sync_bangumi_knowledge(library, "episode-one")["work_scope_id"]
    def newer(row):
        row["fetched_at"] = "2026-10-25T00:00:00Z"
        row["characters"][0]["description"] = "较新的角色资料"
        row["characters"].pop()
    refresh(library, "episode-one", newer)
    sync_bangumi_knowledge(library, "episode-one")
    refresh(library, "episode-two", lambda row: row.update(fetched_at=old_time))
    before = entities._path(scope).read_bytes()
    result = sync_bangumi_knowledge(library, "episode-two")
    assert result["created_cards"] == result["created_sources"] == 0
    assert result["warnings"] and entities._path(scope).read_bytes() == before
    source = next(row for row in KnowledgeStore(entities).sources(scope) if row["origin"]["record_id"] == 8)
    assert source["status"] == "revoked"
    assert any(row["text"] == "较新的角色资料" for row in KnowledgeStore(entities).cards(scope))
