# ruff: noqa: F811
"""The occurrence index must preserve traversal semantics and current evidence."""
import copy

import pytest

from scenerecall.knowledge_recall import WorkEntityRecall
from scenerecall.library import atomic_json, read_json
from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine
from test_work_entities import binding, observation, prop, work  # noqa: F401


@pytest.fixture
def indexed(work, tmp_path):
    library, store = work
    record, _ = observation(library)
    concept = store.create("episode-one", prop())
    search = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "settings"))
    # Intentionally index before binding, as a delayed refresh must be supported.
    search.upsert([record])
    recall = WorkEntityRecall(store)
    return library, store, search, recall, record, concept


def expected(store, search, query="停止时间", filters=None):
    return store.search_occurrences(query, filters, eligible_records=search.filtered_records(filters))


def test_index_reads_latest_binding_before_search_projection_refresh(indexed):
    _, store, search, recall, record, concept = indexed
    before = recall.revision({}, search)
    assert recall.search("停止时间", {}, search) == []
    store.bind("episode-one", binding(concept))
    assert recall.revision({}, search) != before
    results = recall.search("停止时间", {}, search)
    assert results == expected(store, search)
    assert results[0]["id"] == record["id"]
    assert results[0]["match_type"] == "partial" and results[0]["event_verified"] is False
    assert search.filtered_records()[0].get("concept_bindings") is None


def test_warm_index_only_reads_candidate_records_and_preserves_return_isolation(indexed, monkeypatch):
    library, store, search, recall, record, concept = indexed
    store.bind("episode-one", binding(concept))
    search.upsert([{**record, "id": f"unbound-{index}", "text": "无关画面"} for index in range(300)])
    wanted = expected(store, search)
    seen = []
    read = search.filtered_records

    def selected(filters=None, record_ids=None):
        assert record_ids is not None, "indexed knowledge recall cannot scan every record"
        seen.append(list(record_ids))
        return read(filters, record_ids=record_ids)

    monkeypatch.setattr(search, "filtered_records", selected)
    monkeypatch.setattr(library, "records", lambda *args: pytest.fail("recursive/full library projection"))
    assert recall.search("停止时间", {}, search) == wanted
    builds = recall.stats["catalog_builds"]
    result = recall.search("停止时间", {}, search)
    result[0]["concept_bindings"][0]["concept_name"] = "caller modification"
    assert recall.search("停止时间", {}, search) == wanted
    assert seen == [[record["id"]]] * 3
    assert recall.stats["catalog_builds"] == builds
    assert recall.stats["candidate_records"] == 1


@pytest.mark.parametrize("change", ["rename", "revoke", "binding_revoke", "candidate", "fingerprint", "observation", "delete"])
def test_mutations_invalidate_current_occurrences(indexed, change):
    library, store, search, recall, record, concept = indexed
    store.bind("episode-one", binding(concept))
    assert recall.search("停止时间", {}, search)
    token = recall.revision({}, search)
    if change == "rename":
        store.update("episode-one", concept["concept_id"], {"name": "时停表", "aliases": ["新译名"]}, reason="译名更新")
    elif change == "revoke":
        store.revoke("episode-one", concept["concept_id"], "撤销")
    elif change == "binding_revoke":
        store.bind("episode-one", binding(concept, "revoked"))
    elif change == "candidate":
        store.bind("episode-one", binding(concept, "candidate"))
    elif change == "fingerprint":
        path = library.asset_dir("episode-one") / "manifest.json"
        saved = read_json(path)
        saved["fingerprint"] = "changed-source"
        atomic_json(path, saved)
    elif change == "observation":
        search.upsert([{**record, "end_ms": record["end_ms"] + 1}])
    else:
        search.remove_assets(["episode-one"])
    assert recall.search("停止时间", {}, search) == expected(store, search)
    if change not in {"observation"}:
        assert recall.revision({}, search) != token


def test_filters_and_cross_work_names_match_original_scope(indexed):
    _, store, search, recall, record, concept = indexed
    store.bind("episode-one", binding(concept))
    store.create("other-work", prop())
    projected = store.project_record("episode-one", record)
    projected["entities"][0].update(character_id="character-one", character_name="主角")
    projected.update(tags=["tag-one"], folder_ids=["folder-one"])
    search.upsert([projected])
    for filters in ({}, {"asset_id": "episode-one"}, {"asset_id": "other-work"},
                    {"character_id": "character-one"}, {"character_id": "other-character"},
                    {"concept_id": concept["concept_id"]}, {"concept_id": "foreign"},
                    {"tag": "tag-one"}, {"folder_id": "folder-one"}, {"kind": "subtitle"}):
        assert recall.search("停止时间", filters, search) == expected(store, search, filters=filters)


def test_catalog_replacement_and_membership_changes_are_detected(indexed):
    library, store, search, recall, record, concept = indexed
    store.bind("episode-one", binding(concept))
    assert recall.search("停止时间", {}, search)
    token = recall.revision({}, search)
    scope = store.scope("episode-one")["work_scope_id"]
    saved = read_json(store._path(scope))
    # Detect file replacement even without a well-behaved revision increment.
    saved["entities"][concept["concept_id"]]["name"] = "改过的名字"
    atomic_json(store._path(scope), saved)
    assert recall.revision({}, search) != token
    assert recall.search("停止时间", {}, search) == expected(store, search)
    token = recall.revision({}, search)
    memberships = read_json(store.membership_path)
    memberships["assets"].pop("episode-one")
    atomic_json(store.membership_path, memberships)
    assert recall.revision({}, search) != token
    assert recall.search("停止时间", {}, search) == []


def test_budget_disabled_and_evicted_indexes_are_result_equivalent(indexed):
    _, store, search, recall, _, concept = indexed
    store.bind("episode-one", binding(concept))
    wanted = expected(store, search)
    assert recall.search("停止时间", {}, search) == wanted
    recall.max_bytes = 1
    assert recall.search("停止时间", {}, search) == wanted
    assert recall.stats["bytes"] <= 1
    recall.max_bytes = 0
    assert recall.search("停止时间", {}, search) == wanted
    assert recall.stats["bytes"] == 0
    recall.clear()
    assert recall.stats["bytes"] == 0


def test_multiple_bindings_keep_order_and_validate_before_scoring(indexed):
    _, store, search, recall, record, first = indexed
    store.bind("episode-one", binding(first))
    second = store.create("episode-one", prop("另一枚时间道具"))
    second_binding = binding(second)
    second_binding["entity_id"] = None
    store.bind("episode-one", second_binding)
    wanted = expected(store, search)
    assert len(wanted[0]["matched_concept_ids"]) == 2
    assert recall.search("停止时间", {}, search) == wanted
    scope = store.scope("episode-one")["work_scope_id"]
    saved = read_json(store._path(scope))
    for value in saved["bindings"].values():
        if value["concept_id"] == first["concept_id"]:
            value["observation_digest"] = "stale"
    atomic_json(store._path(scope), saved)
    wanted = expected(store, search)
    assert wanted[0]["matched_concept_ids"] == [second["concept_id"]]
    assert recall.search("停止时间", {}, search) == wanted


def test_top_five_concepts_and_top_fifty_records_match_traversal(indexed):
    _, store, search, recall, record, first = indexed
    store.bind("episode-one", binding(first))
    scope = store.scope("episode-one")["work_scope_id"]
    for index in range(7):
        store.create("episode-one", prop(f"时间道具{index}"))
    saved = read_json(store._path(scope))
    template = next(iter(saved["bindings"].values()))
    records = []
    from scenerecall.work_entities import observation_digest
    for index in range(70):
        row = {**copy.deepcopy(record), "id": f"window-{index:03}", "start_ms": index * 1000,
               "end_ms": (index + 1) * 1000}
        records.append(row)
        for concept_id in saved["entities"]:
            value = {**template, "id": f"binding-{index}-{concept_id}", "record_id": row["id"],
                     "concept_id": concept_id, "observation_digest": observation_digest(row)}
            saved["bindings"][value["id"]] = value
    atomic_json(store._path(scope), saved)
    search.upsert(records)
    wanted = expected(store, search, "时间道具")
    assert len(wanted) == 50
    assert recall.search("时间道具", {}, search) == wanted


@pytest.mark.asyncio
async def test_provider_integration_matches_legacy_branch_without_promoting_facts(indexed):
    _, store, search, recall, _, concept = indexed
    store.bind("episode-one", binding(concept))
    search.concept_lookup = lambda query, filters: store.search_occurrences(
        query, filters, eligible_records=search.filtered_records(filters))
    baseline = await search.search("正在停止时间")
    search.concept_lookup = None
    search.knowledge_provider = recall
    assert await search.search("正在停止时间") == baseline
    assert baseline["results"][0]["match_type"] == "partial"
    assert baseline["results"][0]["event_verified"] is False


@pytest.mark.asyncio
async def test_query_retries_after_binding_revoked_during_provider_search(indexed, monkeypatch):
    _, store, search, recall, _, concept = indexed
    store.bind("episode-one", binding(concept))
    search.knowledge_provider = recall
    original = recall.search
    calls = []

    def revoke_after_recall(query, filters, engine):
        rows = original(query, filters, engine)
        calls.append(rows)
        if len(calls) == 1:
            store.bind("episode-one", binding(concept, "revoked"))
        return rows

    monkeypatch.setattr(recall, "search", revoke_after_recall)
    result = await search.search("停止时间")
    assert len(calls) == 2 and calls[0] and not calls[1]
    assert result["results"] == []


def test_empty_memberships_skip_sql_and_detect_first_scope(indexed, monkeypatch):
    _, store, search, recall, _, _ = indexed
    saved = read_json(store.membership_path)
    saved["assets"] = {}
    atomic_json(store.membership_path, saved)
    read_assets = search.filtered_asset_ids
    monkeypatch.setattr(search, "filtered_asset_ids", lambda *_: pytest.fail("empty knowledge must not scan SQL"))
    empty_token = recall.snapshot([]).token
    for _ in range(3):
        assert recall.revision({}, search) == empty_token
        assert recall.search("停止时间", {}, search) == []
    store.create_scope("首次接入知识", asset_ids=["episode-one"])
    concept = store.create("episode-one", prop())
    store.bind("episode-one", binding(concept))
    monkeypatch.setattr(search, "filtered_asset_ids", read_assets)
    assert recall.revision({}, search) != empty_token
    assert recall.search("停止时间", {}, search) == expected(store, search)
    assert recall.search("停止时间", {}, search)
