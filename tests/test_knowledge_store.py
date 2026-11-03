# ruff: noqa: F811
"""Source identity, review gates and revocation are authoritative over RAG output."""
import copy
import hashlib

import pytest

from scenerecall.characters import CharacterStore, scope_id
from scenerecall.knowledge_store import KnowledgeStore
from scenerecall.library import atomic_json, read_json
from test_work_entities import binding, image_bytes, observation, prop, work  # noqa: F401


@pytest.fixture
def knowledge(work):
    library, entities = work
    store = KnowledgeStore(entities)
    scope = entities.scope("episode-one")["work_scope_id"]
    concept = entities.create("episode-one", prop())
    text = "📖秒表能暂停时间；只有持有者能够使用。代价为疲劳。"
    source = store.import_source(scope, "官方资料", "https://example.org/manual", text.encode(), "manual.md")
    card = {"title": "时间暂停", "text": "持有者可以暂停时间", "target_ref": {
        "kind": "concept", "concept_id": concept["concept_id"]}, "source_refs": [{
            "source_id": source["source_id"], "snapshot_id": source["snapshot_id"], "start": 1, "end": len(text)}],
        "review_status": "reviewed", "conditions": ["必须持有秒表"], "costs": ["使用后疲劳"],
        "exceptions": ["不影响持有者"], "applicability": "TV 第 1 集"}
    return library, entities, store, scope, concept, source, card


def test_utf8_snapshot_is_exact_scoped_and_inert(knowledge, monkeypatch):
    _, entities, store, scope, _, source, _ = knowledge
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: pytest.fail("never fetch source URL"))
    result = store.source_content(scope, source["source_id"])
    assert result["text"].startswith("📖") and "path" not in result
    assert hashlib.sha256(result["text"].encode()).hexdigest() == source["sha256"]
    assert store.import_source(scope, source["title"], source["url"], result["text"].encode(), "other.txt") == source
    other = entities.scope("other-work")["work_scope_id"]
    with pytest.raises(FileNotFoundError):
        store.source_content(other, source["source_id"])
    for content, filename in ((b"\xff", "bad.txt"), (b"hello", "bad.pdf"), (b"\x00", "bad.md")):
        with pytest.raises(ValueError):
            store.import_source(scope, "bad", "https://example.org", content, filename)


def test_only_reviewed_cards_project_and_lexical_uses_identical_text(knowledge):
    _, _, store, scope, _, _, payload = knowledge
    candidate = store.save_card(scope, {**payload, "review_status": "candidate"})
    assert store.projection(scope)["cards"] == []
    reviewed = store.save_card(scope, payload, candidate["card_id"])
    projected = store.projection(scope)["cards"][0]
    assert projected["target_ref"] == payload["target_ref"]
    assert projected["statement_text"] == payload["text"]
    assert all(value in projected["text"] for value in ("秒表", "必须持有", "疲劳", "不影响", "TV 第 1 集"))
    assert reviewed["sources"][0]["quote"].startswith("秒表")
    hits = store.lexical_hits(scope, "疲劳")
    assert hits[0]["text"] == projected["text"] and hits[0]["has_film_evidence"] is False
    store.revoke_card(scope, candidate["card_id"], "原文解读有误")
    assert store.projection(scope)["cards"] == [] and store.lexical_hits(scope, "疲劳") == []


def test_revoked_or_changed_source_cannot_reappear_from_reviewed_card(knowledge):
    _, entities, store, scope, _, source, payload = knowledge
    store.save_card(scope, payload)
    before = store.projection(scope)["signature"]
    store.revoke_source(scope, source["source_id"], "出处撤回")
    assert store.projection(scope)["signature"] != before
    assert store.projection(scope)["cards"] == []
    assert store.cards(scope)[0]["effective_status"] == "unavailable"
    assert store.source_content(scope, source["source_id"])["text"]  # retained for audit only
    with pytest.raises(ValueError, match="撤销"):
        store.save_card(scope, payload)
    saved = read_json(entities._path(scope))
    path = entities._path(scope).parent / saved["source_documents"][source["source_id"]]["path"]
    path.write_text("changed outside application", encoding="utf-8")
    assert store.projection(scope)["cards"] == []
    with pytest.raises(ValueError, match="哈希"):
        store.source_content(scope, source["source_id"])


def test_spans_and_foreign_targets_are_not_name_matched(knowledge):
    _, entities, store, scope, _, _, payload = knowledge
    foreign = entities.create("other-work", prop())
    with pytest.raises(ValueError, match="当前作品"):
        store.save_card(scope, {**payload, "target_ref": {"kind": "concept", "concept_id": foreign["concept_id"]}})
    for changes in ({"start": 0, "end": 1000}, {"start": 2, "end": 1}, {"start": True},
                    {"snapshot_id": "wrong-snapshot"}):
        changed = copy.deepcopy(payload)
        changed["source_refs"][0].update(changes)
        with pytest.raises(ValueError):
            store.save_card(scope, changed)


def test_signature_changes_for_identity_not_film_bindings_or_references(knowledge):
    library, entities, store, scope, concept, _, payload = knowledge
    store.save_card(scope, payload)
    before = store.projection(scope)["signature"]
    observation(library)
    entities.bind("episode-one", binding(concept))
    entities.add_reference("episode-one", concept["concept_id"], image_bytes(), confirmed=True)
    assert store.projection(scope)["signature"] == before
    entities.update("episode-one", concept["concept_id"], {"name": "时停怀表", "aliases": ["新译名"]}, reason="译名核对")
    assert store.projection(scope)["signature"] != before
    assert "新译名" in store.projection(scope)["cards"][0]["text"]
    entities.revoke("episode-one", concept["concept_id"], "错误概念")
    assert store.projection(scope)["cards"] == []


def test_character_refs_are_real_current_scoped_ids_and_rename_invalidates(knowledge):
    library, _, store, scope, _, _, payload = knowledge
    characters = CharacterStore(library)
    profile = characters.create("episode-one", {"name": "爱城华恋", "aliases": ["Karen"]})
    cast_scope = scope_id(library.get_asset("episode-one"))
    target = {"kind": "character", "asset_id": "episode-one", "character_scope_id": cast_scope,
              "character_id": profile["id"]}
    store.save_card(scope, {**payload, "target_ref": target})
    before = store.projection(scope)["signature"]
    assert "Karen" in store.projection(scope)["cards"][0]["text"]
    characters.update("episode-one", profile["id"], {"name": "愛城華恋"})
    assert store.projection(scope)["signature"] != before
    with pytest.raises(ValueError, match="当前作品"):
        store.save_card(scope, {**payload, "target_ref": {**target, "asset_id": "other-work"}})
    with pytest.raises(ValueError):
        store.save_card(scope, {**payload, "target_ref": {**target, "character_id": "invented"}})


def test_relations_have_direction_and_graph_alignment_needs_reviewed_support(knowledge):
    _, entities, store, scope, concept, _, payload = knowledge
    other = entities.create("episode-one", {"type": "rule", "name": "时间暂停规则", "definition": "影响时间"})
    obj = {"kind": "concept", "concept_id": other["concept_id"]}
    with pytest.raises(ValueError, match="方向"):
        store.save_card(scope, {**payload, "kind": "relation"})
    relation = store.save_card(scope, {**payload, "kind": "relation", "predicate": "遵守", "object_ref": obj})
    assert "时间道具 遵守 时间暂停规则" in store.projection(scope)["cards"][0]["text"]
    a = store.save_alignment(scope, "TIME DEVICE", [payload["target_ref"]], [relation["card_id"]], "reviewed", "核对原文")
    assert a["target_refs"][0]["concept_id"] == concept["concept_id"]
    with pytest.raises(ValueError, match="唯一"):
        store.save_alignment(scope, "TIME DEVICE", [payload["target_ref"], obj], [relation["card_id"]], "reviewed", "同名")
    store.revoke_card(scope, relation["card_id"], "原文不支持关系")
    with pytest.raises(ValueError, match="失效"):
        store.save_alignment(scope, "TIME DEVICE", [payload["target_ref"]], [relation["card_id"]], "reviewed", "核对原文")


def test_legacy_work_version_and_relation_ids_are_stable_without_read_write(knowledge):
    _, entities, store, scope, _, _, _ = knowledge
    path = entities._path(scope)
    saved = read_json(path)
    saved.pop("work_version_id")
    first = next(iter(saved["entities"].values()))
    first.pop("definition")
    first["relations"] = [{"predicate": "引用", "target_concept_id": first["concept_id"], "source_ids": ["manual"]}]
    atomic_json(path, saved)
    before = path.read_bytes()
    a = store.scope_catalog(scope)
    b = store.scope_catalog(scope)
    assert a["work_version_id"] == b["work_version_id"]
    assert next(iter(a["entities"].values()))["relations"][0]["id"] == next(iter(b["entities"].values()))["relations"][0]["id"]
    assert path.read_bytes() == before
    entities.update("episode-one", first["concept_id"], {"definition": "新定义"}, reason="补全")
    assert read_json(path)["work_version_id"] == a["work_version_id"]


def test_abstract_concept_cannot_confirm_appearance(knowledge):
    library, entities, _, _, _, _, _ = knowledge
    observation(library)
    abstract = entities.create("episode-one", {"type": "ability", "name": "时间暂停", "definition": "操纵时间"})
    with pytest.raises(ValueError, match="抽象概念"):
        entities.bind("episode-one", binding(abstract))
    assert entities.bind("episode-one", binding(abstract, "candidate"))["status"] == "candidate"
