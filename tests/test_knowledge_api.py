# ruff: noqa: F811
"""HTTP knowledge contracts with real stores, queue and evidence projection."""
import asyncio

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scenerecall.jobs import JobQueue
from scenerecall.knowledge_api import register_knowledge_routes
from scenerecall.knowledge_jobs import discovery_view
from scenerecall.library import atomic_json
from scenerecall.providers import ProviderManager
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.search import SearchEngine
from test_visual_retrieval import Encoder
from test_work_entities import binding, image_bytes, observation, prop, work  # noqa: F401


@pytest.fixture
def knowledge_http(work, tmp_path):
    library, entities = work
    local, _ = observation(library)
    foreign, _ = observation(library, "other-work")
    foreign = {**foreign, "id": "foreign-window"}
    RetrievalStore(library).publish("other-work", "foreign-run", foreign)
    providers = ProviderManager(tmp_path / "private")
    search = SearchEngine(tmp_path / "search", providers)
    search.upsert([local, foreign])
    queue = JobQueue(library, providers, search)
    app = FastAPI()

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(_: Request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValidationError)
    async def validation(_: Request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    service = register_knowledge_routes(app, library, providers, search, queue, entities)
    scope = entities.scope("episode-one")["work_scope_id"]
    with TestClient(app) as client:
        yield client, service, search, queue, scope


def source_card(client, service, scope, *, reviewed=True, kind="rule"):
    concept = service.store.create("episode-one", {"type": kind, "name": "量子代偿规则", "definition": "规则可以独立于可见实体存在"})
    base = f"/api/work-entity-scopes/{scope}/knowledge"
    text = "量子代偿规则要求支付记忆；限于第一阶段。"
    imported = client.post(base + "/sources", data={"title": "有出处的规则资料", "url": "https://example.org/rules"},
                           files={"file": ("rules.md", text.encode(), "text/markdown")})
    assert imported.status_code == 200, imported.text
    source = imported.json()
    response = client.post(base + "/cards", json={"title": "量子代偿规则", "text": text,
        "target_ref": {"kind": "concept", "concept_id": concept["concept_id"]},
        "source_refs": [{"source_id": source["source_id"], "snapshot_id": source["snapshot_id"], "start": 0, "end": len(text)}],
        "review_status": "reviewed" if reviewed else "candidate", "costs": ["记忆"], "applicability": "第一阶段"})
    assert response.status_code == 200, response.text
    return base, source, response.json(), concept


def test_source_only_query_preserves_citation_without_inventing_film(knowledge_http):
    client, service, _, _, scope = knowledge_http
    base, source, card, _ = source_card(client, service, scope)
    raw = client.get(base + "/sources/" + source["source_id"])
    assert raw.json()["text"] == card["text"]
    response = client.post(base + "/query", json={"query": "量子代偿", "mode": "baseline"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["film_hits"] == []
    assert len(data["knowledge_hits"]) == 1
    hit = data["knowledge_hits"][0]
    assert hit["card_id"] == card["card_id"] and hit["has_film_evidence"] is False
    assert "record_id" not in hit
    assert hit["sources"][0]["url"] == "https://example.org/rules"
    assert hit["sources"][0]["quote"] == card["text"]


def test_card_patch_reviews_without_resending_content_and_revoke_removes_warm_hit(knowledge_http):
    client, service, _, _, scope = knowledge_http
    base, _, card, _ = source_card(client, service, scope, reviewed=False)
    assert client.post(base + "/query", json={"query": "量子代偿"}).json()["knowledge_hits"] == []
    response = client.patch(base + "/cards/" + card["card_id"], json={"review_status": "reviewed", "reason": "已核对原文与概念"})
    assert response.status_code == 200, response.text
    assert response.json()["text"] == card["text"]
    assert response.json()["source_refs"] == card["source_refs"]
    assert client.post(base + "/query", json={"query": "量子代偿"}).json()["knowledge_hits"]
    revoked = client.request("DELETE", base + "/cards/" + card["card_id"], json={"reason": "来源解释有误"})
    assert revoked.status_code == 200
    assert client.post(base + "/query", json={"query": "量子代偿"}).json()["knowledge_hits"] == []


def test_explicit_work_scope_fences_every_search_branch_and_conflicting_asset(knowledge_http):
    client, service, _, _, scope = knowledge_http
    base = f"/api/work-entity-scopes/{scope}/knowledge"
    response = client.post(base + "/query", json={"query": "圆形金色物体"})
    assert response.status_code == 200, response.text
    assert {row["asset_id"] for row in response.json()["film_hits"]} <= {"episode-one", "episode-two"}
    assert all(row["id"] != "foreign-window" for row in response.json()["film_hits"])
    conflict = client.post(base + "/query", json={"query": "圆形金色物体", "filters": {"asset_id": "other-work"}})
    assert conflict.status_code == 200, conflict.text
    assert conflict.json()["film_hits"] == [] and conflict.json()["knowledge_hits"] == []


def test_disabled_optional_runtime_keeps_lexical_knowledge_and_current_bindings(knowledge_http):
    client, service, _, _, scope = knowledge_http
    base, _, card, concept = source_card(client, service, scope, kind="prop")
    service.store.bind("episode-one", binding(concept))
    assert client.get("/api/knowledge/settings").json()["enabled"] is False
    result = client.post(base + "/query", json={"query": "量子代偿", "mode": "lightrag"})
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["knowledge_hits"][0]["card_id"] == card["card_id"]
    assert data["degraded"] and "回退" in " ".join(data["degraded"])
    assert [row["id"] for row in data["film_hits"]] == ["window-one"]
    assert data["film_hits"][0]["event_verified"] is False
    assert data["film_hits"][0]["match_type"] == "partial"


def test_build_is_explicit_deduplicated_and_source_revocation_blocks_old_card(knowledge_http):
    client, service, _, queue, scope = knowledge_http
    base, source, _, _ = source_card(client, service, scope)
    assert queue.list() == []
    assert client.post(base + "/query", json={"query": "量子代偿", "mode": "lightrag"}).status_code == 200
    assert queue.list() == []
    one = client.post(base + "/build").json()
    two = client.post(base + "/build").json()
    assert one["id"] == two["id"] and one["type"] == "knowledge_build"
    revoked = client.request("DELETE", base + "/sources/" + source["source_id"], json={"reason": "撤销不可靠来源"})
    assert revoked.status_code == 200
    assert client.post(base + "/query", json={"query": "量子代偿"}).json()["knowledge_hits"] == []
    assert client.post(base + "/build").status_code == 400


def test_graph_candidate_endpoint_reads_actual_generation_file_without_model_call(knowledge_http):
    client, service, _, _, scope = knowledge_http
    base, _, _, _ = source_card(client, service, scope)
    projection = service.knowledge.projection(scope)
    root = service.runtime._scope_root(projection)
    atomic_json(root / "active.json", {"generation": "generation-one", "status": "ready"})
    atomic_json(root / "generation-one" / "graph-candidates.json", {"entities": [{"entity_name": "抽取名称", "description": "候选"}], "relationships": []})
    response = client.get(base + "/graph-candidates")
    assert response.status_code == 200, response.text
    assert response.json()["entities"][0]["entity_name"] == "抽取名称"
    assert response.json()["status"]["enabled"] is False


def test_reference_candidate_http_confirmation_updates_current_binding_and_search(knowledge_http):
    client, service, search, queue, scope = knowledge_http
    base, _, _, concept = source_card(client, service, scope, kind="prop")
    ref = service.store.add_reference("episode-one", concept["concept_id"], image_bytes(), confirmed=True)
    search.visual_encoder = Encoder()
    frame = {"id": "registered-frame", "at_ms": 200, "pts": 200, "time_base": [1, 1000],
             "path": str(service.library.frame_path("episode-one", "registered-frame"))}

    async def prepare():
        await search.index_visual(service.library.records("episode-one"), [frame])
        job = queue.add("reference_discovery", {"work_scope_id": scope})
        queue.update(job["id"], status="running")
        await queue.discover_references(job["id"])

    asyncio.run(prepare())
    candidate = discovery_view(service.knowledge, search, scope)["candidates"][0]
    assert candidate["reference_ids"] == [ref["id"]]
    assert service.store.scope("episode-one")["bindings"] == {}
    route = base + "/reference-candidates/" + candidate["candidate_id"] + "/confirm"
    assert client.post(route, json={"reason": ""}).status_code in {400, 422}
    confirmed = client.post(route, json={"reason": "核对原帧金色表盘"})
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["decision_source"] == "human"
    response = client.post(base + "/query", json={"query": "量子代偿"})
    assert response.status_code == 200, response.text
    row = next(row for row in response.json()["film_hits"] if row["id"] == candidate["record_id"])
    assert row["event_verified"] is False and row["match_type"] == "partial"
    assert concept["concept_id"] in row["matched_concept_ids"]


@pytest.mark.parametrize("requested_scope", ["cast-current", "cast-foreign"])
def test_async_provider_keeps_character_scope_and_refuses_same_id_from_other_scope(knowledge_http, requested_scope):
    _, service, search, _, _ = knowledge_http
    record = service.library.records("episode-one")[0]
    record["entities"] = [{"id": "person-one", "character_id": "character-one",
        "evidence_frame_ids": ["registered-frame"], "identity_binding": {"id": "binding-one",
            "character_id": "character-one", "character_scope_id": "cast-current", "status": "human_confirmed",
            "policy_version": "identity-binding-v2"}}]
    search.upsert([record])
    candidate = {**record, "match_basis": "knowledge_character_appearance", "matched_character_ids": ["character-one"],
                 "matched_character_scope_id": requested_scope}

    class Provider:
        async def asearch(self, *_):
            return {"record_candidates": [candidate], "knowledge_hits": [], "diagnostics": {}}

    search.knowledge_provider = Provider()
    result = asyncio.run(search._concept("角色出场", {"asset_id": "episode-one"}))
    if requested_scope == "cast-foreign":
        assert result == []
    else:
        assert len(result) == 1 and result[0]["matched_character_scope_id"] == "cast-current"


def test_knowledge_http_final_projection_rechecks_character_scope(knowledge_http, monkeypatch):
    client, service, search, _, scope = knowledge_http
    current = service.library.records("episode-one")[0]
    current["entities"] = [{"id": "person-one", "character_id": "character-one",
        "evidence_frame_ids": ["registered-frame"], "identity_binding": {"id": "binding-one",
            "character_id": "character-one", "character_scope_id": "cast-current", "status": "human_confirmed",
            "policy_version": "identity-binding-v2"}}]
    candidate = {**current, "match_basis": "knowledge_character_appearance", "matched_character_ids": ["character-one"],
                 "matched_character_scope_id": "cast-foreign"}

    async def recalled(*_, **__):
        return {"results": [candidate], "knowledge_hits": [], "knowledge_diagnostics": {}}

    monkeypatch.setattr(search, "search", recalled)
    monkeypatch.setattr(service.library, "records", lambda *_, **_kw: [current])
    response = client.post(f"/api/work-entity-scopes/{scope}/knowledge/query", json={"query": "角色出场"})
    assert response.status_code == 200, response.text
    assert response.json()["film_hits"] == []


@pytest.mark.parametrize("target", [{}, {"kind": "unknown"}, {"kind": "concept"},
                                     {"kind": "concept", "concept_id": "../../other"},
                                     {"kind": "character", "character_id": "missing-scope"}])
def test_knowledge_query_rejects_malformed_explicit_target_without_server_error(knowledge_http, target):
    client, _, _, _, scope = knowledge_http
    response = client.post(f"/api/work-entity-scopes/{scope}/knowledge/query", json={"query": "出场", "target_ref": target})
    assert response.status_code in {400, 422}, response.text


def test_knowledge_panel_does_not_label_visual_neighbours_as_knowledge_linked_films(knowledge_http, monkeypatch):
    from scenerecall.main import create_app

    client, service, _, _, scope = knowledge_http
    base, _, card, _ = source_card(client, service, scope, kind="rule")
    app = create_app(service.library.root, start_worker=False, token="knowledge-http-test")
    search = app.state.search
    record = app.state.library.records("episode-one")[0]
    search.upsert([record])

    class LocalEncoder(Encoder):
        async def close(self):
            pass

    search.visual_encoder = LocalEncoder()

    async def visual_neighbour(*_):
        return [{**record, "visual_similarity": .8, "hit_frame_id": "registered-frame", "seek_ms": 200,
                 "retrieval_branches": ["visual"]}]

    monkeypatch.setattr(search, "_visual", visual_neighbour)
    # Use both real authenticated HTTP routes with the same engine/provider.
    # A visual neighbour is useful in full search but supplies no appearance
    # binding for this independently stored abstract rule.
    with TestClient(app, headers={"Authorization": "Bearer knowledge-http-test"}) as full_client:
        app.state.preparation.states["index"]["status"] = "ready"
        scoped = {"query": "量子代偿", "filters": {"asset_id": "episode-one"}}
        panel = full_client.post(base + "/query", json=scoped)
        assert panel.status_code == 200, panel.text
        assert panel.json()["knowledge_hits"][0]["card_id"] == card["card_id"]
        assert panel.json()["knowledge_hits"][0]["has_film_evidence"] is False
        assert panel.json()["film_hits"] == []
        full = full_client.post("/api/search", json={**scoped, "knowledge_scope_id": scope})
        assert full.status_code == 200, full.text
        hit = next(row for row in full.json()["results"] if row["id"] == record["id"])
        assert hit["retrieval_branches"] == ["visual"]
        assert hit["match_type"] == "partial"
