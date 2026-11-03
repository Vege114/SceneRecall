"""Authenticated knowledge controls share the existing library and task queue."""
from __future__ import annotations

from fastapi import Body, File, Form, UploadFile

from .execution import run_in_thread
from .knowledge_runtime import KnowledgeRuntime
from .knowledge_service import KnowledgeService
from .knowledge_store import KnowledgeStore
from .models import Contract
from pydantic import Field, field_validator
from typing import Literal


class KnowledgeQuery(Contract):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["baseline", "lightrag"] = "baseline"
    filters: dict = Field(default_factory=dict)
    target_ref: dict | None = None

    @field_validator("target_ref")
    @classmethod
    def valid_target(cls, value):
        if value is not None:
            from .knowledge_store import KnowledgeTarget
            return KnowledgeTarget.model_validate(value).model_dump(exclude_none=True)
        return None


class BangumiKnowledgeImport(Contract):
    asset_id: str = Field(min_length=1, max_length=180)


def register_knowledge_routes(app, library, providers, search, queue, entities):
    knowledge = KnowledgeStore(entities)
    runtime = KnowledgeRuntime(library, providers)
    service = KnowledgeService(knowledge, runtime)
    app.state.knowledge, app.state.knowledge_runtime = knowledge, runtime
    queue.knowledge, queue.knowledge_runtime = knowledge, runtime
    search.knowledge_provider = service
    prefix = "/api/work-entity-scopes/{scope}/knowledge"

    @app.post("/api/assets/{asset_id}/knowledge/bangumi")
    async def prepare_bangumi_knowledge(asset_id: str):
        # Use the server-owned archive. This endpoint never fetches a supplied
        # URL, accepts a client dossier or invokes a model.
        from .bangumi_knowledge import sync_bangumi_knowledge
        return await run_in_thread(sync_bangumi_knowledge, library, asset_id)

    @app.post(prefix + "/bangumi")
    async def import_bangumi_knowledge(scope: str, payload: BangumiKnowledgeImport):
        from .bangumi_knowledge import sync_bangumi_knowledge
        return await run_in_thread(sync_bangumi_knowledge, library, payload.asset_id, scope_id=scope)

    @app.get("/api/knowledge/settings")
    def settings():
        return runtime.settings()

    @app.put("/api/knowledge/settings")
    async def configure(payload: dict = Body(...)):
        await runtime.close()
        return runtime.configure(payload)

    @app.post("/api/knowledge/check")
    async def preflight():
        return await runtime.preflight()

    @app.post("/api/knowledge/close")
    async def close():
        await runtime.close()
        return {"closed": True}

    @app.get(prefix + "/status")
    async def status(scope: str):
        projection = await run_in_thread(knowledge.projection, scope)
        return runtime.status(projection)

    @app.get(prefix + "/sources")
    def sources(scope: str):
        return {"sources": knowledge.sources(scope)}

    @app.post(prefix + "/sources")
    async def add_source(scope: str, file: UploadFile = File(...), title: str = Form(...), url: str = Form("")):
        content = await file.read(2 * 1024 * 1024 + 1)
        return await run_in_thread(knowledge.import_source, scope, title, url, content, file.filename or "source.txt")

    @app.get(prefix + "/sources/{source_id}")
    def source(scope: str, source_id: str):
        return knowledge.source_content(scope, source_id)

    @app.delete(prefix + "/sources/{source_id}")
    def revoke_source(scope: str, source_id: str, payload: dict = Body(...)):
        return knowledge.revoke_source(scope, source_id, payload.get("reason", ""))

    @app.get(prefix + "/cards")
    def cards(scope: str):
        return {"cards": knowledge.cards(scope)}

    @app.post(prefix + "/cards")
    def add_card(scope: str, payload: dict = Body(...)):
        return knowledge.save_card(scope, payload)

    @app.patch(prefix + "/cards/{card_id}")
    def edit_card(scope: str, card_id: str, payload: dict = Body(...)):
        from .knowledge_store import KnowledgeCardInput
        from .library import safe_id
        with library.lock:
            previous = knowledge.scope_catalog(scope)["knowledge_cards"].get(safe_id(card_id))
            if previous is None:
                raise FileNotFoundError("资料卡不存在")
            editable = {key: value for key, value in previous.items() if key in KnowledgeCardInput.model_fields}
            return knowledge.save_card(scope, {**editable, **payload}, card_id=card_id)

    @app.delete(prefix + "/cards/{card_id}")
    def revoke_card(scope: str, card_id: str, payload: dict = Body(...)):
        return knowledge.revoke_card(scope, card_id, payload.get("reason", ""))

    @app.post(prefix + "/build")
    def build(scope: str):
        projection = knowledge.projection(scope)
        if not projection["cards"]:
            raise ValueError("请先添加有有效来源的已核对资料卡")
        for job in queue.list():
            if (job["type"] == "knowledge_build" and job["config"].get("work_scope_id") == scope
                    and job["status"] in {"queued", "running", "pausing", "cancelling"}):
                return job
        from .library import digest
        return queue.add("knowledge_build", {"work_scope_id": scope, "signature": projection["signature"],
                                             "runtime_config_revision": digest(runtime.settings())})

    @app.get(prefix + "/alignments")
    def alignments(scope: str):
        return {"alignments": knowledge.alignments(scope)}

    @app.post(prefix + "/alignments")
    def align(scope: str, payload: dict = Body(...)):
        return knowledge.save_alignment(scope, payload.get("upstream_key"), payload.get("target_refs", []),
            payload.get("supporting_card_ids", []), payload.get("status"), payload.get("reason", ""))

    @app.get(prefix + "/graph-candidates")
    def graph_candidates(scope: str):
        projection = knowledge.projection(scope)
        from .library import read_json
        manifest = read_json(runtime._scope_root(projection) / "active.json", {})
        candidates = {}
        if manifest.get("generation"):
            from .library import safe_id
            candidates = read_json(runtime._scope_root(projection) / safe_id(manifest["generation"]) / "graph-candidates.json", {})
        return {"generation": manifest.get("generation"), "status": runtime.status(projection),
                **candidates}

    @app.post(prefix + "/query")
    async def query(scope: str, payload: KnowledgeQuery):
        knowledge.scope_catalog(scope)
        result = await search.search(payload.query, filters=payload.filters, knowledge_scope_id=scope,
            knowledge_mode=payload.mode, knowledge_target=payload.target_ref, limit=50)
        # The engine rechecks SQL evidence. This endpoint also fences candidates
        # against the authoritative current library projection, just like /search.
        current = {}
        for asset_id in search.result_asset_ids(result["results"]):
            current.update((row["id"], row) for row in await run_in_thread(library.records, asset_id))
        film = []
        from .visual_index import evidence_digest
        for row in result["results"]:
            # This panel promises knowledge-linked evidence. Approximate visual
            # neighbours remain available from the full /api/search interface.
            if "concept" not in row.get("retrieval_branches", []):
                continue
            actual = current.get(row["id"])
            if actual is None or evidence_digest(actual) != evidence_digest(row):
                continue
            if row.get("match_basis") == "setting_concept_appearance":
                confirmed = {binding["concept_id"] for binding in actual.get("concept_bindings", []) if binding["status"] == "confirmed"}
                if not set(row.get("matched_concept_ids", [])) & confirmed:
                    continue
            elif row.get("match_basis") == "knowledge_character_appearance":
                from .knowledge_service import confirmed_character_evidence
                if not row.get("matched_character_scope_id") or not confirmed_character_evidence(
                        actual, set(row.get("matched_character_ids", [])), row["matched_character_scope_id"])[0]:
                    continue
            film.append(row)
        await search.check_completed_query()
        return {"knowledge_hits": result.get("knowledge_hits", []), "film_hits": film,
                "diagnostics": result.get("knowledge_diagnostics", {}), "degraded": result.get("degraded", [])}

    @app.post(prefix + "/reference-discovery")
    def discover(scope: str):
        knowledge.scope_catalog(scope)
        for job in queue.list():
            if (job["type"] == "reference_discovery" and job["config"].get("work_scope_id") == scope
                    and job["status"] in {"queued", "running", "pausing", "cancelling"}):
                return job
        return queue.add("reference_discovery", {"work_scope_id": scope})

    @app.get(prefix + "/reference-candidates")
    def reference_candidates(scope: str):
        from .knowledge_jobs import discovery_view
        return discovery_view(knowledge, search, scope)

    @app.post(prefix + "/reference-candidates/{candidate_id}/confirm")
    async def confirm_reference(scope: str, candidate_id: str, payload: dict = Body(...)):
        from .knowledge_jobs import confirm_discovery
        result = await run_in_thread(confirm_discovery, knowledge, search, scope, candidate_id, payload.get("reason", ""))
        await run_in_thread(search.sync_asset, library.records(result["asset_id"]), result["asset_id"])
        return result

    return service
