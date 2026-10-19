"""Single-asset, loopback-only manual ASD annotation application."""
from __future__ import annotations

import argparse
import asyncio
import hmac
import os
from pathlib import Path
import secrets
import sys
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from annotation_store import AnnotationConflict, AsdAnnotationService, ReadOnlyLibrary
from scenerecall.library import atomic_json, read_json

SERVICE = "scenerecall-asd-annotator"
COOKIE = "asd_annotator_session"


class AnnotationRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_ms: int = Field(ge=0, strict=True)
    end_ms: int = Field(gt=0, strict=True)


class SessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=1, max_length=512, strict=True)


class HumanTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=7, max_length=180, strict=True)
    character_id: str | None
    boxes: list[Any] = Field(max_length=200)
    labels: list[Any] = Field(max_length=200)
    reviewed: bool = Field(strict=True)
    excluded: bool = Field(strict=True)
    note: str = Field(default="", max_length=4000, strict=True)


class AnnotationSave(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0, strict=True)
    note: str = Field(default="", max_length=8000, strict=True)
    tracks: list[HumanTrack] = Field(max_length=32)


def create_app(library_path, asset_id, data_dir, *, port=8766, token=None, service=None):
    library = service.library if service is not None else ReadOnlyLibrary(library_path, asset_id)
    service = service or AsdAnnotationService(library, data_dir=data_dir)
    token = token or secrets.token_urlsafe(32)
    app = FastAPI(title="ASD 人工标注", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.annotations, app.state.token = service, token

    @app.middleware("http")
    async def local_session(request: Request, call_next):
        if request.url.hostname not in {"localhost", "127.0.0.1", "::1", "[::1]", "testserver"}:
            return JSONResponse({"detail": "仅允许本机访问"}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            try:
                parsed = urlparse(origin)
                valid = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"} and parsed.port == port
            except ValueError:
                valid = False
            if not valid:
                return JSONResponse({"detail": "请求来源不受信任"}, status_code=403)
        if request.url.path == "/" and request.query_params.get("token") is not None:
            if not hmac.compare_digest(request.query_params["token"].encode(), token.encode()):
                return JSONResponse({"detail": "会话令牌无效"}, status_code=401)
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(COOKIE, token, httponly=True, samesite="strict", path="/")
        else:
            supplied = request.cookies.get(COOKIE, "")
            authorization = request.headers.get("authorization", "")
            if authorization.startswith("Bearer "):
                supplied = authorization[7:]
            public_static = request.method == "GET" and request.url.path in {"/", "/app.js", "/app.mjs", "/editor.js", "/styles.css"}
            if not public_static and request.url.path not in {"/api/health", "/api/session"} and not hmac.compare_digest(supplied.encode(), token.encode()):
                return JSONResponse({"detail": "请从启动器提供的本机会话链接进入"}, status_code=401)
            response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
                                 "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY"})
        return response

    @app.exception_handler(AnnotationConflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)[:800]}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)[:800]}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse({"detail": "本机标注片段、素材或参考图不存在"}, status_code=404)

    @app.get("/api/health")
    async def health():
        return {"service": SERVICE, "asset_id": asset_id, "pid": os.getpid(), "ready": True}

    @app.post("/api/session")
    async def session(payload: SessionInput):
        if not hmac.compare_digest(payload.token.encode(), token.encode()):
            raise HTTPException(401, "会话令牌无效")
        response = JSONResponse({"authenticated": True})
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", path="/")
        return response

    @app.get("/api/config")
    async def config():
        asset = library.get_asset(asset_id)
        return {"asset_id": asset_id, "title": asset["title"], "asset_title": asset["title"],
                "duration_ms": asset["duration_ms"], "source_fingerprint": asset["fingerprint"],
                "storage_root_info": str(service.data_dir)}

    @app.get("/api/characters")
    async def characters():
        return await asyncio.to_thread(service.characters.catalog, asset_id)

    @app.get("/api/characters/{character_id}/references/{reference_id}/image")
    async def reference(character_id: str, reference_id: str):
        path = await asyncio.to_thread(service.characters.reference_path, asset_id, character_id, reference_id)
        return FileResponse(path, media_type="image/png")

    @app.get("/api/asd-annotations")
    async def annotations():
        return await asyncio.to_thread(service.list, asset_id)

    @app.post("/api/asd-annotations/clips")
    async def prepare(payload: AnnotationRange):
        return await asyncio.to_thread(service.prepare, asset_id, payload.start_ms, payload.end_ms)

    @app.get("/api/asd-annotations/clips/{clip_id}")
    async def clip(clip_id: str):
        return await asyncio.to_thread(service.get, asset_id, clip_id)

    @app.put("/api/asd-annotations/clips/{clip_id}")
    async def save(clip_id: str, payload: AnnotationSave):
        return await asyncio.to_thread(service.save, asset_id, clip_id, payload.model_dump())

    @app.get("/api/asd-annotations/clips/{clip_id}/media")
    async def clip_media(clip_id: str):
        return FileResponse(await asyncio.to_thread(service.media_path, asset_id, clip_id), media_type="video/mp4")

    @app.get("/api/asd-annotations/export")
    async def export():
        return JSONResponse(await asyncio.to_thread(service.export, asset_id), headers={
            "Content-Disposition": 'attachment; filename="kaguya-asd-annotations.json"'})

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).with_name("index.html"), media_type="text/html")

    @app.get("/{filename}")
    async def static(filename: str):
        if filename not in {"app.mjs", "app.js", "editor.js", "styles.css"}:
            raise HTTPException(404, "文件不存在")
        return FileResponse(Path(__file__).with_name(filename), media_type="text/css" if filename.endswith(".css") else "text/javascript")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port 必须位于 1024–65535")
    private = args.data_dir / "server.private.json"
    previous = read_json(private, {})
    prior_token = previous.get("token")
    valid_prior = (previous.get("service") == SERVICE and previous.get("asset_id") == args.asset_id
                   and previous.get("port") == args.port and previous.get("host") == "127.0.0.1"
                   and isinstance(prior_token, str) and 32 <= len(prior_token) <= 128
                   and all(char.isalnum() or char in "-_" for char in prior_token))
    app = create_app(args.library, args.asset_id, args.data_dir, port=args.port, token=prior_token if valid_prior else None)
    token = app.state.token
    args.data_dir.mkdir(parents=True, exist_ok=True)
    state = {"service": SERVICE, "asset_id": args.asset_id, "pid": os.getpid(), "host": "127.0.0.1", "port": args.port,
             "token": token, "url": f"http://127.0.0.1:{args.port}/?token={token}"}
    atomic_json(private, state)
    private.chmod(0o600)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
