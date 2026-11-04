# ruff: noqa: F811
"""Uncapped local recall retains scope, review and current-evidence fences."""
import copy

import pytest

from scenerecall.library import atomic_json, read_json
from scenerecall.work_entities import observation_digest
from test_knowledge_service import service  # noqa: F401
from test_work_entities import work  # noqa: F401


CARD_QUERY = "仅在资料卡中的星海传送标记"


@pytest.fixture
def many_candidates(service):
    svc, runtime, engine, scope, source, card, concept = service
    saved = read_json(svc.store._path(scope))
    template_concept = saved["entities"][concept["concept_id"]]
    template_binding = next(iter(saved["bindings"].values()))
    template_card = saved["knowledge_cards"][card["card_id"]]
    template_record = engine.filtered_records({"asset_id": "episode-one"})[0]
    saved["entities"], saved["bindings"], saved["knowledge_cards"] = {}, {}, {}
    records = []
    for group in range(7):
        concept_id = f"concept-{group:02}"
        saved["entities"][concept_id] = {**copy.deepcopy(template_concept), "concept_id": concept_id}
        for offset in range(10):
            index = group * 10 + offset
            record = {**copy.deepcopy(template_record), "id": f"window-{index:03}",
                      "start_ms": index * 1000, "end_ms": (index + 1) * 1000}
            records.append(record)
            binding_id = f"binding-{index:03}"
            saved["bindings"][binding_id] = {**copy.deepcopy(template_binding), "id": binding_id,
                "record_id": record["id"], "concept_id": concept_id,
                "observation_digest": observation_digest(record)}
            card_id = f"card-{index:03}"
            saved["knowledge_cards"][card_id] = {**copy.deepcopy(template_card), "card_id": card_id,
                "text": CARD_QUERY, "target_ref": {"kind": "concept", "concept_id": concept_id}}
    # Uncapped candidate collection must still exclude revoked and stale evidence.
    for identity, status, stamp in (("revoked", "revoked", None), ("stale", "confirmed", "stale-digest")):
        record = {**copy.deepcopy(template_record), "id": identity}
        records.append(record)
        saved["bindings"][identity] = {**copy.deepcopy(template_binding), "id": identity,
            "record_id": identity, "concept_id": "concept-06", "status": status,
            "observation_digest": stamp or observation_digest(record)}
    atomic_json(svc.store._path(scope), saved)
    engine.remove_assets(["episode-one"])
    engine.upsert(records)
    return svc, runtime, engine, scope, source, records


def test_indexed_and_traversal_recall_remove_both_concept_and_record_caps(many_candidates):
    svc, _, engine, scope, _, records = many_candidates
    filters = {"asset_id": "episode-one"}
    uncapped = {**filters, "_all_candidates": True}
    legacy = svc.baseline.search("时间道具", filters, engine)
    all_rows = svc.baseline.search("时间道具", uncapped, engine)
    assert len(legacy) == 50
    assert len(all_rows) == 70
    assert all_rows[:50] == legacy
    assert all_rows == svc.store.search_occurrences("时间道具", uncapped, eligible_records=records)
    assert legacy == svc.store.search_occurrences("时间道具", filters, eligible_records=records)
    assert {value for row in all_rows for value in row["matched_concept_ids"]} == {
        f"concept-{index:02}" for index in range(7)}
    assert all(row["asset_id"] == "episode-one" and row["event_verified"] is False for row in all_rows)
    assert not {"stale", "revoked"}.intersection(row["id"] for row in all_rows)
    snapshot = svc.baseline._query_snapshot(filters, engine)
    assert len(svc.baseline.candidate_refs("时间道具", snapshot)[0][scope]) == 5
    assert len(svc.baseline.candidate_refs("时间道具", snapshot, limit=None)[0][scope]) == 7


@pytest.mark.asyncio
async def test_card_only_recall_keeps_all_cards_targets_and_film_records(many_candidates):
    svc, _, engine, scope, _, _ = many_candidates
    filters = {"asset_id": "episode-one"}
    snapshot = svc.baseline._query_snapshot(filters, engine)
    assert svc.baseline.candidate_refs(CARD_QUERY, snapshot, limit=None)[1] == []
    assert len(svc.knowledge.lexical_hits(scope, CARD_QUERY)) == 50
    assert len(svc.knowledge.lexical_hits(scope, CARD_QUERY, limit=None)) == 70
    legacy = await svc.asearch(CARD_QUERY, filters, engine)
    uncapped = await svc.asearch(CARD_QUERY, {**filters, "_all_candidates": True}, engine)
    assert len(legacy["knowledge_hits"]) == len(legacy["record_candidates"]) == 50
    assert len(uncapped["knowledge_hits"]) == len(uncapped["record_candidates"]) == 70
    assert {row["target_ref"]["concept_id"] for row in uncapped["knowledge_hits"]} == {
        f"concept-{index:02}" for index in range(7)}
    assert all(row["event_verified"] is False for row in uncapped["record_candidates"])
    assert uncapped["diagnostics"]["degraded"] == []


@pytest.mark.asyncio
async def test_uncapped_card_recall_cannot_revive_revoked_sources(many_candidates):
    svc, _, engine, scope, source, _ = many_candidates
    svc.knowledge.revoke_source(scope, source["source_id"], "撤销测试资料")
    result = await svc.asearch(CARD_QUERY, {"asset_id": "episode-one", "_all_candidates": True}, engine)
    assert result["knowledge_hits"] == result["record_candidates"] == []


def test_uncapped_character_appearances_still_require_current_confirmed_identity(service, monkeypatch):
    svc, _, engine, scope, _, _, _ = service
    target = {"kind": "character", "asset_id": "episode-one", "character_id": "character-a",
              "character_scope_id": "cast-a"}
    rows = [{"id": f"appearance-{index:03}", "asset_id": "episode-one", "kind": "visual",
             "start_ms": index * 1000, "end_ms": (index + 1) * 1000,
             "evidence_frame_ids": ["frame-a"], "entities": [{"character_id": "character-a",
                 "evidence_frame_ids": ["frame-a"], "identity_binding": {
                     "id": f"identity-{index}", "character_id": "character-a", "character_scope_id": "cast-a",
                     "status": "human_confirmed", "policy_version": "identity-binding-v2"}}]}
            for index in range(60)]
    invalid = copy.deepcopy(rows[0])
    invalid["id"] = "unconfirmed"
    invalid["entities"][0]["identity_binding"]["status"] = "candidate"
    rows.append(invalid)
    monkeypatch.setattr(engine, "filtered_records", lambda *_a, **_kw: rows)
    monkeypatch.setattr(svc.library, "records", lambda *_a, **_kw: rows)
    monkeypatch.setattr(svc, "scopes", lambda *_a, **_kw: [scope])
    hits = [{"target_ref": target, "target_names": ["爱城华恋"]}]
    legacy = svc._character_records(hits, "爱城华恋", {"asset_id": "episode-one"}, engine)
    uncapped = svc._character_records(hits, "爱城华恋", {"asset_id": "episode-one", "_all_candidates": True}, engine)
    assert len(legacy) == 50
    assert len(uncapped) == 60
    assert uncapped[:50] == legacy
    assert "unconfirmed" not in {row["id"] for row in uncapped}


@pytest.mark.asyncio
async def test_lightrag_reports_its_upstream_limits_in_uncapped_mode(service):
    svc, runtime, engine, scope, _, _, _ = service
    filters = {"asset_id": "episode-one", "_knowledge_scope": scope, "_knowledge_mode": "lightrag"}
    legacy = await svc.asearch("暂停时间", filters, engine)
    uncapped = await svc.asearch("暂停时间", {**filters, "_all_candidates": True}, engine)
    assert runtime.calls == 2
    assert legacy["diagnostics"]["degraded"] == []
    assert any("top_k=20" in text and "不能保证穷尽" in text for text in uncapped["diagnostics"]["degraded"])
    assert uncapped["record_candidates"] == legacy["record_candidates"]
