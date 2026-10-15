"""Authenticated local reference management and review routes."""
from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import Field

from .characters import CharacterStore
from .library import read_json, safe_id
from .models import Contract
from .visual_identity import MAX_IMAGE_BYTES, ReferenceGallery, VisualIdentityPipeline


class ReferenceCrop(Contract):
    frame_id: str
    box: list[float] = Field(min_length=4, max_length=4)
    confirmed: bool = False
    single_character: bool = False


class ReferenceEdit(Contract):
    status: Literal["confirmed", "candidate", "revoked"]
    reason: str = Field(min_length=1, max_length=4000)
    single_character: bool | None = None


class IdentityReview(Contract):
    status: Literal["confirmed", "candidate", "unknown"]
    character_id: str | None = None
    reason: str = Field(min_length=1, max_length=4000)


class CharacterCreate(Contract):
    name: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=100)
    description: str = Field(default="", max_length=20000)


class VisualRun(Contract):
    frame_ids: list[str] = Field(min_length=1, max_length=256)


def _public_reference(row):
    return {key: value for key, value in row.items() if key != "path"}


def _public_reviews(result):
    result = {key: value for key, value in result.items() if key not in {"selected_profiles", "references"}}
    result["regions"] = [{key: value for key, value in row.items() if key != "crop_path"}
                         for row in result["regions"]]
    return result


def register_visual_identity_routes(app, library, search):
    gallery = ReferenceGallery(library)
    pipeline = VisualIdentityPipeline(library)
    app.state.visual_identity = pipeline
    app.state.reference_gallery = gallery

    @app.post("/api/assets/{asset_id}/characters")
    def create_character(asset_id: str, payload: CharacterCreate):
        return CharacterStore(library).create(asset_id, payload.model_dump())

    @app.get("/api/assets/{asset_id}/characters/{character_id}/references")
    def list_references(asset_id: str, character_id: str):
        result = gallery.list(asset_id, character_id)
        result["references"] = [_public_reference(row) for row in result["references"]]
        return result

    @app.post("/api/assets/{asset_id}/characters/{character_id}/references")
    async def add_reference(asset_id: str, character_id: str, file: UploadFile = File(...),
                            source_kind: Literal["upload", "external_cached"] = Form("upload"),
                            source_url: str = Form(""), confirmed: bool = Form(False),
                            single_character: bool = Form(False)):
        # Uploaded/cached image bytes stay on this host. A URL is attribution,
        # never a fetch instruction (including redirects/private network URLs).
        content = await file.read(MAX_IMAGE_BYTES + 1)
        return _public_reference(await asyncio.to_thread(gallery.add, asset_id, character_id, content,
            source_kind=source_kind, source_url=source_url, confirmed=confirmed, single_character=single_character))

    @app.post("/api/assets/{asset_id}/characters/{character_id}/references/crop")
    def crop_reference(asset_id: str, character_id: str, payload: ReferenceCrop):
        return _public_reference(gallery.crop(asset_id, character_id, **payload.model_dump()))

    @app.patch("/api/assets/{asset_id}/characters/{character_id}/references/{reference_id}")
    def edit_reference(asset_id: str, character_id: str, reference_id: str, payload: ReferenceEdit):
        assets = CharacterStore(library).catalog(asset_id)["assets"]
        before = {asset["id"]: {row["id"]: row["match"].get("character_id")
                               for row in pipeline.reviews(asset["id"])["regions"]} for asset in assets}
        result = gallery.update(asset_id, character_id, reference_id, **payload.model_dump())
        from .scenes import SceneStore
        for asset in assets:
            changed = [row["id"] for row in pipeline.reviews(asset["id"])["regions"]
                       if before[asset["id"]].get(row["id"]) and before[asset["id"]][row["id"]] != row["match"].get("character_id")]
            SceneStore(library).invalidate_identity_inferences(asset["id"], region_ids=changed)
            if search is not None:
                search.upsert(library.records(asset["id"]))
        return _public_reference(result)

    @app.get("/api/assets/{asset_id}/characters/{character_id}/references/{reference_id}/image")
    def reference_image(asset_id: str, character_id: str, reference_id: str):
        row = next((row for row in gallery.list(asset_id, character_id)["references"]
                    if row["id"] == reference_id), None)
        if row is None:
            raise FileNotFoundError("参考图不存在")
        return FileResponse(row["path"], media_type="image/png")

    @app.get("/api/assets/{asset_id}/identity/settings")
    def identity_settings(asset_id: str):
        return pipeline.settings(asset_id)

    @app.put("/api/assets/{asset_id}/identity/settings")
    def configure_identity(asset_id: str, payload: dict):
        return pipeline.configure(asset_id, payload)

    @app.post("/api/assets/{asset_id}/identity/run")
    async def run_identity(asset_id: str, payload: VisualRun):
        registry = read_json(library.asset_dir(asset_id) / "frames" / "registry.json", {})
        frames = []
        for identifier in payload.frame_ids:
            library.frame_path(asset_id, identifier)
            frames.append({"id": identifier, "at_ms": registry[identifier]["at_ms"]})
        return _public_reviews(await asyncio.to_thread(pipeline.run, asset_id, frames, pipeline.settings(asset_id)))

    @app.get("/api/assets/{asset_id}/identity/reviews")
    def identity_reviews(asset_id: str):
        return _public_reviews(pipeline.reviews(asset_id))

    @app.post("/api/assets/{asset_id}/identity/reviews/{region_id}")
    def review_identity(asset_id: str, region_id: str, payload: IdentityReview):
        previous = next((row for row in pipeline.reviews(asset_id)["regions"] if row["id"] == region_id), None)
        result = pipeline.review(asset_id, region_id, **payload.model_dump())
        # Only mapping/display/search changes. Immutable prose and extracted
        # features remain intact; no episode/story provider run is scheduled.
        if previous and previous["match"].get("character_id") != result["match"].get("character_id"):
            from .scenes import SceneStore
            SceneStore(library).invalidate_identity_inferences(asset_id, region_ids=[region_id])
        if search is not None:
            search.upsert(library.records(asset_id))
        return {key: value for key, value in result.items() if key != "crop_path"}

    @app.get("/api/assets/{asset_id}/identity/regions/{region_id}/image")
    def identity_crop(asset_id: str, region_id: str):
        library.get_asset(asset_id)
        location = pipeline._directory(asset_id)
        if read_json(location / "regions" / f"{safe_id(region_id)}.json") is None:
            raise FileNotFoundError("人物区域不存在")
        return FileResponse(location / "crops" / f"{safe_id(region_id)}.png", media_type="image/png")
