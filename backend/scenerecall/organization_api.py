"""Local-session-protected organization routes, registered by the application."""
from typing import Literal

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .organization import OrganizationConflict


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectCreate(Input):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["movie", "animation", "series"] = "animation"
    folder_id: str | None = None
    tags: list[str] = Field(default_factory=list, max_length=100)


class ProjectEdit(Input):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    kind: Literal["movie", "animation", "series"] | None = None
    folder_id: str | None = None
    tags: list[str] | None = Field(default=None, max_length=100)


class FolderCreate(Input):
    name: str = Field(min_length=1, max_length=200)
    parent_id: str | None = None


class FolderEdit(Input):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    parent_id: str | None = None


class Assignment(Input):
    asset_ids: list[str] = Field(min_length=1, max_length=10000)
    project_id: str | None


def register_organization_routes(app, library, queue, search):
    organization = library.organization

    @app.exception_handler(OrganizationConflict)
    async def organization_conflict(request: Request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    async def refresh():
        await search.rebuild(library.records())

    @app.get("/api/projects")
    async def projects():
        return organization.projects()

    @app.post("/api/projects")
    async def create_project(payload: ProjectCreate):
        return organization.save_project(payload.model_dump())

    @app.post("/api/projects/assign")
    async def assign(payload: Assignment):
        result = organization.assign(payload.asset_ids, payload.project_id, queue)
        await refresh()
        return result

    @app.patch("/api/projects/{project_id}")
    async def edit_project(project_id: str, payload: ProjectEdit):
        result = organization.save_project(payload.model_dump(exclude_unset=True), project_id)
        await refresh()
        return result

    @app.delete("/api/projects/{project_id}")
    async def delete_project(project_id: str):
        result = organization.delete_project(project_id, queue)
        await refresh()
        return {"deleted_id": project_id, **result}

    @app.get("/api/folders")
    async def folders():
        return organization.folders()

    @app.post("/api/folders")
    async def create_folder(payload: FolderCreate):
        return organization.save_folder(payload.model_dump())

    @app.patch("/api/folders/{folder_id}")
    async def edit_folder(folder_id: str, payload: FolderEdit):
        result = organization.save_folder(payload.model_dump(exclude_unset=True), folder_id)
        await refresh()
        return result

    @app.delete("/api/folders/{folder_id}")
    async def delete_folder(folder_id: str):
        organization.delete_folder(folder_id)
        await refresh()
        return {"deleted_id": folder_id}
