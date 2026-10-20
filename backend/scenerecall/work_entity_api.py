"""Local work-entity CRUD. Registered on the application's authenticated router."""
from __future__ import annotations

import asyncio
import inspect
from typing import Literal

from fastapi import File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import Field

from .models import Contract
from .work_entities import MAX_IMAGE_BYTES, MAX_PACKAGE_BYTES, BindingInput, EntityInput, WorkEntityStore


class ScopeCreate(Contract):
    title: str = Field(min_length=1, max_length=500)
    version: str = Field(default="", max_length=200)
    asset_ids: list[str] = Field(default_factory=list, max_length=200)


class ScopeAttach(Contract):
    work_scope_id: str


class EntityEdit(Contract):
    changes: dict
    reason: str = Field(min_length=1, max_length=2000)


class Reason(Contract):
    reason: str = Field(min_length=1, max_length=2000)


class ReferenceFromFrame(Contract):
    frame_id: str
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    confirmed: bool = False
    view: Literal["whole_scene", "landmark", "object_region"] = "whole_scene"


class ReferenceStatus(Reason):
    status: Literal["confirmed", "candidate", "revoked"]


class EntityContext(Contract):
    query: str = Field(default="", max_length=2000)
    concept_ids: list[str] = Field(default_factory=list, max_length=5)


def register_work_entity_routes(app, library, refresh=None):
    store = WorkEntityStore(library)
    app.state.work_entities = store

    async def changed(asset_id):
        if refresh:
            scope = store.scope(asset_id)
            for candidate in store.scope_view(scope["work_scope_id"])["asset_ids"] if scope else [asset_id]:
                result = refresh(candidate)
                if inspect.isawaitable(result):
                    await result

    @app.post("/api/work-entity-scopes")
    def create_scope(payload: ScopeCreate):
        return store.create_scope(**payload.model_dump())

    @app.get("/api/assets/{asset_id}/work-entity-scope")
    def get_scope(asset_id: str):
        scope = store.scope(asset_id)
        return store.scope_view(scope["work_scope_id"]) if scope else {"work_scope_id": None}

    @app.put("/api/assets/{asset_id}/work-entity-scope")
    async def attach_scope(asset_id: str, payload: ScopeAttach):
        value = store.attach(asset_id, payload.work_scope_id)
        await changed(asset_id)
        return value

    @app.get("/api/assets/{asset_id}/work-entities")
    def list_entities(asset_id: str):
        return store.list(asset_id)

    @app.post("/api/assets/{asset_id}/work-entities")
    async def create_entity(asset_id: str, payload: EntityInput):
        value = store.create(asset_id, payload.model_dump())
        await changed(asset_id)
        return value

    @app.post("/api/assets/{asset_id}/work-entities/context")
    def context(asset_id: str, payload: EntityContext):
        return store.context(asset_id, **payload.model_dump())

    @app.post("/api/assets/{asset_id}/work-entities/import")
    async def import_entities(asset_id: str, file: UploadFile = File(...)):
        content = await file.read(MAX_PACKAGE_BYTES + 1)
        value = await asyncio.to_thread(store.import_package, asset_id, content)
        await changed(asset_id)
        return {"entities": value}

    @app.get("/api/assets/{asset_id}/work-entities/{concept_id}")
    def get_entity(asset_id: str, concept_id: str):
        return store.get(asset_id, concept_id)

    @app.patch("/api/assets/{asset_id}/work-entities/{concept_id}")
    async def edit_entity(asset_id: str, concept_id: str, payload: EntityEdit):
        value = store.update(asset_id, concept_id, payload.changes, reason=payload.reason)
        await changed(asset_id)
        return value

    @app.delete("/api/assets/{asset_id}/work-entities/{concept_id}")
    async def revoke_entity(asset_id: str, concept_id: str, payload: Reason):
        value = store.revoke(asset_id, concept_id, payload.reason)
        await changed(asset_id)
        return value

    @app.get("/api/assets/{asset_id}/work-entities/{concept_id}/occurrences")
    def occurrences(asset_id: str, concept_id: str):
        return {"occurrences": store.occurrences(asset_id, concept_id), "facts_are_not_current_events": True}

    @app.post("/api/assets/{asset_id}/work-entities/{concept_id}/references")
    async def add_reference(asset_id: str, concept_id: str, file: UploadFile = File(...),
                            source_url: str = Form(""), confirmed: bool = Form(False),
                            view: Literal["whole_scene", "landmark", "object_region"] = Form("whole_scene")):
        content = await file.read(MAX_IMAGE_BYTES + 1)
        value = await asyncio.to_thread(store.add_reference, asset_id, concept_id, content,
                                       source_url=source_url, confirmed=confirmed, view=view)
        await changed(asset_id)
        return value

    @app.post("/api/assets/{asset_id}/work-entities/{concept_id}/references/from-frame")
    async def reference_from_frame(asset_id: str, concept_id: str, payload: ReferenceFromFrame):
        value = await asyncio.to_thread(store.reference_from_frame, asset_id, concept_id, **payload.model_dump())
        await changed(asset_id)
        return value

    @app.patch("/api/assets/{asset_id}/work-entities/{concept_id}/references/{reference_id}")
    async def update_reference(asset_id: str, concept_id: str, reference_id: str, payload: ReferenceStatus):
        value = store.update_reference(asset_id, concept_id, reference_id, **payload.model_dump())
        await changed(asset_id)
        return value

    @app.get("/api/work-entity-scopes/{work_scope_id}/entities/{concept_id}/references/{reference_id}/image")
    def reference_image(work_scope_id: str, concept_id: str, reference_id: str):
        return FileResponse(store.reference_path(work_scope_id, concept_id, reference_id), media_type="image/png")

    @app.post("/api/assets/{asset_id}/work-entity-bindings")
    async def bind_entity(asset_id: str, payload: BindingInput):
        value = store.bind(asset_id, payload.model_dump())
        await changed(asset_id)
        return value

    return store
