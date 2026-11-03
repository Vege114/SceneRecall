# ruff: noqa: F811
"""Risk-focused knowledge retrieval integration without remote model calls."""
import copy

import pytest

from scenerecall.knowledge_runtime import KnowledgeUnavailable
from scenerecall.knowledge_service import KnowledgeService, confirmed_character_evidence
from scenerecall.knowledge_store import KnowledgeStore
from scenerecall.providers import ProviderManager
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.search import SearchEngine, SearchSnapshotChanged
from test_work_entities import binding, observation, prop, work  # noqa: F401


class Runtime:
    def __init__(self):
        self.failure = None
        self.before_result = None
        self.calls = 0
        self.busy = False

    def status(self, projection):
        return {"enabled": True, "configured": True, "status": "ready", "generation": "test-generation",
                "signature": projection["signature"], "busy": self.busy, "error": None}

    async def query(self, projection, query):
        self.calls += 1
        if self.before_result:
            self.before_result()
        if self.failure:
            raise self.failure
        return {"generation": "test-generation", "data": {"status": "success", "data": {
            "entities": [{"entity_name": "HALLUCINATED DISPLAY NAME", "source_id": "chunk-a"}],
            "chunks": [{"chunk_id": "chunk-a", "content": "not an identity", "reference_id": "1"}]}},
            "mappings": {"chunks": {"chunk-a": [{"card_ids": [card["card_id"] for card in projection["cards"]]}]}},
            "diagnostics": {"calls": []}}


@pytest.fixture
def service(work, tmp_path):
    library, entities = work
    local, _ = observation(library)
    foreign, _ = observation(library, "other-work")
    foreign = {**foreign, "id": "other-window"}
    RetrievalStore(library).publish("other-work", "foreign-run", foreign)
    engine = SearchEngine(tmp_path / "search", ProviderManager(tmp_path / "private"))
    engine.upsert([local, foreign])
    local_concept = entities.create("episode-one", prop())
    foreign_concept = entities.create("other-work", prop())
    entities.bind("episode-one", binding(local_concept))
    entities.bind("other-work", {**binding(foreign_concept), "record_id": "other-window"})
    knowledge = KnowledgeStore(entities)
    scope = entities.scope("episode-one")["work_scope_id"]
    source = knowledge.import_source(scope, "时钟手册", "https://example.org/clock", "暂停时间并付出疲劳代价".encode(), "clock.md")
    card = knowledge.save_card(scope, {"text": "暂停时间并付出疲劳代价", "target_ref": {
        "kind": "concept", "concept_id": local_concept["concept_id"]}, "source_refs": [{
            "source_id": source["source_id"], "snapshot_id": source["snapshot_id"], "start": 0,
            "end": len("暂停时间并付出疲劳代价")}],
        "review_status": "reviewed"})
    runtime = Runtime()
    svc = KnowledgeService(knowledge, runtime)
    return svc, runtime, engine, scope, source, card, local_concept


@pytest.mark.asyncio
async def test_explicit_scope_applies_to_legacy_and_card_film_evidence(service):
    svc, _, engine, scope, _, _, concept = service
    result = await svc.asearch("暂停时间", {"_knowledge_scope": scope}, engine)
    assert [row["id"] for row in result["record_candidates"]] == ["window-one"]
    assert result["record_candidates"][0]["matched_concept_ids"] == [concept["concept_id"]]
    assert result["record_candidates"][0]["event_verified"] is False
    assert result["knowledge_hits"][0]["has_film_evidence"] is True
    assert svc.scopes({"_knowledge_scope": scope, "asset_id": "other-work"}, engine) == []


@pytest.mark.asyncio
async def test_semantic_chunk_maps_reviewed_card_not_display_name(service):
    svc, runtime, engine, scope, _, _, concept = service
    result = await svc.asearch("quasar teleport", {"_knowledge_scope": scope, "_knowledge_mode": "lightrag"}, engine)
    assert runtime.calls == 1
    assert len(result["knowledge_hits"]) == 1
    assert result["record_candidates"][0]["matched_concept_ids"] == [concept["concept_id"]]
    assert result["record_candidates"][0]["match_type"] == "partial"
    assert result["record_candidates"][0]["event_verified"] is False


@pytest.mark.asyncio
async def test_unavailable_runtime_retains_lexical_and_source_without_fake_film(service):
    svc, runtime, engine, scope, _, _, concept = service
    runtime.failure = KnowledgeUnavailable("not built", "stale")
    revoked = {**binding(concept, "revoked")}
    svc.store.bind("episode-one", revoked)
    result = await svc.asearch("疲劳", {"_knowledge_scope": scope, "_knowledge_mode": "lightrag"}, engine)
    assert result["record_candidates"] == []
    assert len(result["knowledge_hits"]) == 1
    assert result["knowledge_hits"][0]["has_film_evidence"] is False
    assert "record_id" not in result["knowledge_hits"][0]
    assert result["knowledge_hits"][0]["sources"][0]["url"] == "https://example.org/clock"
    assert result["diagnostics"]["degraded"]


@pytest.mark.asyncio
async def test_source_revoke_during_runtime_cannot_return_stale_mapping(service):
    svc, runtime, engine, scope, source, _, _ = service
    runtime.before_result = lambda: svc.knowledge.revoke_source(scope, source["source_id"], "撤销出处")
    with pytest.raises(SearchSnapshotChanged):
        await svc.asearch("quasar teleport", {"_knowledge_scope": scope, "_knowledge_mode": "lightrag"}, engine)
    runtime.before_result = None
    result = await svc.asearch("quasar teleport", {"_knowledge_scope": scope, "_knowledge_mode": "lightrag"}, engine)
    assert result["knowledge_hits"] == [] and result["record_candidates"] == []


@pytest.mark.asyncio
async def test_explicit_target_navigation_does_not_require_query_name_overlap(service):
    svc, _, engine, scope, _, card, concept = service
    result = await svc.asearch("xyz-unrelated", {"_knowledge_scope": scope, "_knowledge_target": card["target_ref"]}, engine)
    assert result["record_candidates"][0]["matched_concept_ids"] == [concept["concept_id"]]


def test_ambiguous_blocks_and_unmapped_graph_names_never_establish_identity(service):
    svc, _, _, scope, _, card, _ = service
    projection = svc.knowledge.projection(scope)
    other = copy.deepcopy(projection["cards"][0])
    other.update(card_id="other-card", target_ref={"kind": "concept", "concept_id": "foreign-concept"})
    response = {"data": {"data": {"entities": [{"entity_name": "时间道具"}], "chunks": [{"chunk_id": "shared"}]}},
                "mappings": {"chunks": {"shared": {"card_ids": [card["card_id"], "other-card"]}}}}
    hits, unmapped = svc._retrieved_cards(response, [*projection["cards"], other])
    assert len(hits) == 2 and all(hit["mapping_status"] == "ambiguous" for hit in hits)
    assert unmapped == 0
    hits, _ = svc._retrieved_cards({"data": {"entities": [{"entity_name": "时间道具"}]}}, projection["cards"])
    assert hits == []


def test_identity_revision_ignores_busy_but_tracks_decisions(service):
    svc, runtime, engine, scope, _, _, _ = service
    filters = {"_knowledge_scope": scope}
    before = svc.revision(filters, engine)
    runtime.busy = True
    assert svc.revision(filters, engine) == before
    path = svc.library.asset_dir("episode-one") / "identity" / "visual" / "decisions.json"
    from scenerecall.library import atomic_json
    atomic_json(path, {"decisions": {"x": {"status": "unknown"}}})
    assert svc.revision(filters, engine) != before


def test_character_appearance_requires_current_v2_binding_and_original_frames():
    binding_value = {"id": "identity-a", "character_id": "character-a", "status": "human_confirmed",
                     "policy_version": "identity-binding-v2", "character_scope_id": "cast-a"}
    record = {"evidence_frame_ids": ["frame-a"], "entities": [{"character_id": "character-a",
        "evidence_frame_ids": ["frame-a"], "identity_binding": binding_value}]}
    assert confirmed_character_evidence(record, {"character-a"}, "cast-a") == (["frame-a"], ["identity-a"], ["character-a"])
    assert confirmed_character_evidence(record, {"character-a"}, "cast-other") == ([], [], [])
    for change in ({"status": "candidate"}, {"policy_version": "legacy"}, {"id": ""}):
        invalid = copy.deepcopy(record)
        invalid["entities"][0]["identity_binding"].update(change)
        assert confirmed_character_evidence(invalid, {"character-a"}) == ([], [], [])
    record["entities"][0]["evidence_frame_ids"] = ["invented-frame"]
    assert confirmed_character_evidence(record, {"character-a"}) == ([], [], [])


def graph_only_response(card, *, key="Clock", kind="entities"):
    returned = {"entity_name": key} if kind == "entities" else {"src_id": "Clock", "tgt_id": "Pilot"}
    tracked = {"id": key} if kind == "entities" else {"source": "Clock", "target": "Pilot"}
    return {"data": {"data": {kind: [returned]}},
            "mappings": {"chunks": {"full-tracking-chunk": {"card_ids": [card["card_id"]]}}},
            "graph_tracking": {kind: [{**tracked, "supporting_chunk_ids": ["full-tracking-chunk"]}]}}


def reviewed_alignment(card, *, key="Clock"):
    return {"upstream_key": key, "status": "reviewed", "target_refs": [card["target_ref"]],
            "supporting_card_ids": [card["card_id"]],
            "supporting_card_versions": {card["card_id"]: card["version"]}}


def test_supporting_statement_edit_requires_graph_identity_to_be_reviewed_again(service):
    svc, _, _, scope, _, card, _ = service
    alignment = svc.knowledge.save_alignment(scope, "Clock", [card["target_ref"]], [card["card_id"]],
                                              "reviewed", "核对实体身份")
    from scenerecall.knowledge_store import KnowledgeCardInput
    raw = svc.knowledge.scope_catalog(scope)["knowledge_cards"][card["card_id"]]
    editable = {key: value for key, value in raw.items() if key in KnowledgeCardInput.model_fields}
    svc.knowledge.save_card(scope, {**editable, "text": "新版资料进一步区分时钟的使用条件"}, card["card_id"])
    assert svc.knowledge.alignments(scope)[0]["status"] == "unresolved"
    current = svc.knowledge.projection(scope)["cards"]
    response = graph_only_response(current[0])
    assert svc._retrieved_cards(response, current, [alignment])[0] == []
    refreshed = svc.knowledge.save_alignment(scope, "Clock", [card["target_ref"]], [card["card_id"]],
                                              "reviewed", "重新核对新版声明")
    assert svc.knowledge.alignments(scope)[0]["status"] == "reviewed"
    assert svc._retrieved_cards(response, current, [refreshed])[0][0]["mapping_basis"] == "reviewed_graph_alignment"


def test_reviewed_graph_alignment_resolves_full_tracking_without_returned_source_ids(service):
    svc, _, _, scope, _, _, _ = service
    cards = svc.knowledge.projection(scope)["cards"]
    response = graph_only_response(cards[0])
    hits, _ = svc._retrieved_cards(response, cards, [reviewed_alignment(cards[0])])
    assert len(hits) == 1
    assert hits[0]["target_ref"] == cards[0]["target_ref"]
    assert hits[0]["mapping_status"] == "reviewed"
    assert hits[0]["mapping_basis"] == "reviewed_graph_alignment"
    assert hits[0]["aligned_upstream_keys"] == ["Clock"]
    # Merely matching the displayed name or a temporary reference is insufficient.
    response["data"]["data"]["entities"][0]["entity_name"] = cards[0]["target_name"]
    response["data"]["data"]["entities"][0]["reference_id"] = "1"
    assert svc._retrieved_cards(response, cards)[0] == []


@pytest.mark.parametrize("change", ["unresolved", "ambiguous", "revoked", "multiple_targets", "missing_support",
                                         "untracked_support", "missing_chunk", "duplicate_node", "duplicate_alignment"])
def test_graph_identity_requires_unique_current_source_anchored_alignment(service, change):
    svc, _, _, scope, _, _, _ = service
    cards = svc.knowledge.projection(scope)["cards"]
    response = graph_only_response(cards[0])
    alignment = reviewed_alignment(cards[0])
    alignments = [alignment]
    if change in {"unresolved", "ambiguous", "revoked"}:
        alignment["status"] = change
    elif change == "multiple_targets":
        alignment["target_refs"].append({"kind": "concept", "concept_id": "same-name-other-concept"})
    elif change == "missing_support":
        alignment["supporting_card_ids"] = ["revoked-card"]
    elif change == "untracked_support":
        response["mappings"]["chunks"]["full-tracking-chunk"]["card_ids"] = ["unrelated-card"]
    elif change == "missing_chunk":
        response["graph_tracking"]["entities"][0]["supporting_chunk_ids"] = ["missing-chunk"]
    elif change == "duplicate_node":
        response["graph_tracking"]["entities"].append(copy.deepcopy(response["graph_tracking"]["entities"][0]))
    else:
        alignments.append({**alignment, "target_refs": [{"kind": "concept", "concept_id": "other-concept"}]})
    assert svc._retrieved_cards(response, cards, alignments)[0] == []


def test_revoked_card_cannot_be_revived_by_a_reviewed_graph_alignment(service):
    svc, _, _, scope, _, _, _ = service
    cards = svc.knowledge.projection(scope)["cards"]
    response, alignment = graph_only_response(cards[0]), reviewed_alignment(cards[0])
    revoked = [{**cards[0], "review_status": "revoked"}]
    assert svc._retrieved_cards(response, revoked, [alignment])[0] == []
    assert svc._retrieved_cards(response, [], [alignment])[0] == []


def test_relationship_graph_alignment_uses_directed_key_and_formal_relation_card(service):
    svc, _, _, scope, _, _, _ = service
    card = {**svc.knowledge.projection(scope)["cards"][0], "kind": "relation", "predicate": "由其操作"}
    response = graph_only_response(card, kind="relationships")
    alignment = reviewed_alignment(card, key="Clock->Pilot")
    hits, _ = svc._retrieved_cards(response, [card], [alignment])
    assert hits[0]["mapping_basis"] == "reviewed_graph_alignment"
    assert hits[0]["aligned_upstream_keys"] == ["Clock->Pilot"]
    # Reversing a relation requires its own reviewed direction.
    alignment["upstream_key"] = "Pilot->Clock"
    assert svc._retrieved_cards(response, [card], [alignment])[0] == []
    alignment["upstream_key"] = "Clock->Pilot"
    assert svc._retrieved_cards(response, [{**card, "kind": "fact"}], [alignment])[0] == []


@pytest.mark.asyncio
async def test_graph_only_alignment_is_used_by_async_recall_and_revocation_removes_it(service, monkeypatch):
    svc, runtime, engine, scope, _, card, concept = service
    svc.knowledge.save_alignment(scope, "Clock", [card["target_ref"]], [card["card_id"]], "reviewed", "核对原文主目标")

    async def query(projection, *_):
        return graph_only_response(projection["cards"][0])

    monkeypatch.setattr(runtime, "query", query)
    filters = {"_knowledge_scope": scope, "_knowledge_mode": "lightrag"}
    result = await svc.asearch("quasar teleport", filters, engine)
    assert result["knowledge_hits"][0]["mapping_basis"] == "reviewed_graph_alignment"
    assert result["record_candidates"][0]["matched_concept_ids"] == [concept["concept_id"]]
    assert result["record_candidates"][0]["event_verified"] is False
    svc.knowledge.save_alignment(scope, "Clock", [card["target_ref"]], [card["card_id"]], "revoked", "映射有歧义")
    result = await svc.asearch("quasar teleport", filters, engine)
    assert result["knowledge_hits"] == [] and result["record_candidates"] == []


@pytest.mark.asyncio
async def test_corrupt_derived_graph_mapping_falls_back_without_losing_current_lexical_evidence(service, monkeypatch):
    svc, runtime, engine, scope, _, _, concept = service

    async def corrupt(*_):
        return {"data": {"entities": [{"entity_name": "Clock"}], "chunks": [{"chunk_id": "broken"}]},
                "mappings": {"chunks": {"broken": ["invalid persisted map row"]}}}

    monkeypatch.setattr(runtime, "query", corrupt)
    result = await svc.asearch("疲劳", {"_knowledge_scope": scope, "_knowledge_mode": "lightrag"}, engine)
    assert result["diagnostics"]["degraded"]
    assert result["knowledge_hits"][0]["sources"]
    assert result["record_candidates"][0]["matched_concept_ids"] == [concept["concept_id"]]


@pytest.mark.parametrize("publication", ["decisions", "calibration", "characters", "gallery_trust", "reference_pixels",
                                             "frame_registry", "active_receipt", "source", "region", "match"])
def test_identity_publications_invalidate_knowledge_query_revision(service, publication):
    from scenerecall.characters import scope_id
    from scenerecall.library import atomic_json, read_json

    svc, _, engine, scope, _, _, _ = service
    base = svc.library.asset_dir("episode-one")
    identity = base / "identity" / "visual"
    character_root = svc.library.root / "works" / scope_id(svc.library.get_asset("episode-one"))
    gallery = character_root / "reference-images"
    # Prepare directories before taking the initial snapshot, so the assertion
    # covers publication/replacement rather than incidental parent creation.
    for directory in (identity / "regions", identity / "matches", gallery):
        directory.mkdir(parents=True, exist_ok=True)
    reference = gallery / "reference-current.png"
    reference.write_bytes(b"current trusted pixels")
    paths = {"decisions": identity / "decisions.json", "calibration": identity / "settings.json",
             "characters": character_root / "characters.json", "gallery_trust": gallery / "gallery.json",
             "frame_registry": base / "frames" / "registry.json", "active_receipt": base / "retrieval" / "active.json",
             "source": base / "manifest.json", "region": identity / "regions" / "region-current.json",
             "match": identity / "matches" / "match-current.json"}
    filters = {"_knowledge_scope": scope}
    before = svc.revision(filters, engine)
    if publication == "reference_pixels":
        reference.write_bytes(b"changed trusted reference pixels")
    else:
        path = paths[publication]
        saved = read_json(path, {})
        # Each authoritative input has a dedicated publication file (or an
        # atomic region/match replacement); no model or recursive scan is needed.
        atomic_json(path, {**saved, "test_publication": publication})
    assert svc.revision(filters, engine) != before


def test_identity_revision_ignores_unrelated_feature_crop_and_history_payloads(service, monkeypatch):
    from pathlib import Path
    from scenerecall.characters import scope_id

    svc, _, engine, scope, _, _, _ = service
    base = svc.library.asset_dir("episode-one")
    identity = base / "identity" / "visual"
    character_root = svc.library.root / "works" / scope_id(svc.library.get_asset("episode-one"))
    unrelated_roots = [identity / "features", identity / "crops", identity / "history",
                       base / "analyses" / "old-unpublished-run",
                       character_root / "reference-images" / "history",
                       svc.library.root / "cache" / "identity-features"]
    unrelated = []
    for directory in unrelated_roots:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "unrelated-payload.bin"
        path.write_bytes(b"old cached artifact")
        unrelated.append(path)
    original_frame = svc.library.frame_path("episode-one", "registered-frame")
    forbidden = [*unrelated_roots, original_frame]
    real_stat = Path.stat

    def bounded_stat(path, *args, **kwargs):
        if any(path == root or path.is_relative_to(root) for root in forbidden):
            pytest.fail("knowledge revision scanned unrelated feature/history or original frame pixels: " + str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", bounded_stat)
    before = svc.revision({"_knowledge_scope": scope}, engine)
    for path in unrelated:
        path.write_bytes(b"new larger cached artifact that must not change knowledge identity")
    assert svc.revision({"_knowledge_scope": scope}, engine) == before
