from __future__ import annotations

import asyncio
import hmac
import io
import mimetypes
import os
import secrets
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import codex_cli
from .characters import CharacterStore, StaleCharacterRevision
from .jobs import JobQueue
from .library import Library, atomic_json, read_json, safe_id, subtitle_track_info, uid
from .models import (
    AnalysisInput, AnnotationInput, AssetInput, CharacterEditInput,
    CharacterMergeInput, CharacterReanalysisInput, SearchInput, SubtitleExtractInput, SubtitleProbeInput,
)
from .providers import ProviderManager
from .search import SearchEngine

DEFAULTS = {"bindings": {k: None for k in ("vision", "subtitle", "embedding", "query", "decision", "answer")},
            "defaults": {"window_ms": 6000, "frames_per_window": 4, "subtitle_fps": 2,
                         "subtitle_crop": [0, .65, 1, .35], "max_requests": 1000, "max_cost": None}}


def create_app(data_dir: Path | None = None, start_worker=True, providers=None, token=None, allowed_ports=None):
    root = data_dir or Path(os.environ.get("SCENERECALL_DATA", str(Path.home() / "SceneRecallLibrary")))
    library = Library(Path(root))
    characters = CharacterStore(library)
    providers = providers or ProviderManager(library.root / "private")
    search = SearchEngine(library.root / "indexes", providers)
    queue = JobQueue(library, providers, search)
    session_token = token or secrets.token_urlsafe(32)
    allowed_ports = allowed_ports or {8765, 5173}

    @asynccontextmanager
    async def lifespan(app):
        # Reconcile derived lexical records with source files on every launch.
        await search.rebuild(library.records())
        if start_worker:
            await queue.start()
        yield
        await queue.stop()

    app = FastAPI(title="SceneRecall", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.library, app.state.providers, app.state.search, app.state.jobs = library, providers, search, queue
    app.state.session_token = session_token
    app.state.characters = characters

    @app.middleware("http")
    async def local_session(request: Request, call_next):
        from urllib.parse import urlparse
        host = request.url.hostname
        if host not in {"localhost", "127.0.0.1", "[::1]", "::1", "testserver"}:
            return JSONResponse({"detail": "只允许本机访问"}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            try:
                parsed = urlparse(origin)
                valid_origin = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"} and parsed.port in allowed_ports
            except ValueError:
                valid_origin = False
            if not valid_origin:
                return JSONResponse({"detail": "请求来源不受信任"}, status_code=403)
        if request.url.path.startswith("/api/") and request.url.path not in {"/api/session", "/api/health"}:
            supplied = request.cookies.get("scenerecall_session", "")
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                supplied = auth[7:]
            if not hmac.compare_digest(supplied.encode("utf-8"), session_token.encode("utf-8")):
                return JSONResponse({"detail": "请使用启动时的本机会话链接登录"}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)[:800]}, status_code=400)

    @app.exception_handler(StaleCharacterRevision)
    async def stale_characters(request, exc):
        return JSONResponse({"detail": str(exc)[:800]}, status_code=409)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse({"detail": "资料或文件不存在，请检查输入路径或重新定位原视频"}, status_code=404)

    def settings():
        value = read_json(library.root / "private" / "settings.json", {})
        return {"bindings": {**DEFAULTS["bindings"], **value.get("bindings", {})},
                "defaults": {**DEFAULTS["defaults"], **value.get("defaults", {})}}

    def enrich(record):
        record = dict(record)
        asset_id = record.get("asset_id", record.get("id"))
        thumbnail = record.get("thumbnail") or next(iter(record.get("evidence_frame_ids", [])), None)
        if thumbnail:
            record["thumbnail_url"] = f"/api/assets/{asset_id}/frames/{thumbnail}"
        record["media_url"] = f"/api/assets/{asset_id}/media"
        return record

    def public_job(job):
        return {**{key: value for key, value in job.items() if key not in {"config", "checkpoints"}},
                "budget": {"max_requests": job["config"].get("max_requests", 1000), "max_cost": job["config"].get("max_cost")}}

    def asset_view(asset):
        asset = dict(asset)
        observations = library.observations(asset["id"])
        if observations and observations[0].get("evidence_frame_ids"):
            asset["thumbnail_url"] = f"/api/assets/{asset['id']}/frames/{observations[0]['evidence_frame_ids'][0]}"
        asset["observation_count"] = len(observations)
        asset["subtitle_count"] = len(library.subtitles(asset["id"]))
        return asset

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.post("/api/session")
    async def session(payload: dict = Body(...)):
        supplied = payload.get("token", "")
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied.encode("utf-8"), session_token.encode("utf-8")):
            raise HTTPException(401, "本机会话令牌无效")
        response = JSONResponse({"ok": True})
        response.set_cookie("scenerecall_session", session_token, httponly=True, samesite="strict", path="/")
        return response

    @app.get("/api/bootstrap")
    async def bootstrap():
        assets = library.list_assets()
        return {"library_path": str(library.root), "settings": settings(), "provider_profiles": providers.list(),
                "stats": {"assets": len(assets), "records": len(library.records()),
                          "favorites": len(library.collections())},
                "tools": {"ffmpeg": bool(shutil.which("ffmpeg")), "ffprobe": bool(shutil.which("ffprobe"))},
                "version": __version__}

    @app.get("/api/settings")
    async def get_settings():
        return settings()

    @app.put("/api/settings")
    async def put_settings(payload: dict = Body(...)):
        current = settings()
        bindings = {**current["bindings"], **payload.get("bindings", {})}
        if set(bindings) - set(DEFAULTS["bindings"]):
            raise ValueError("未知模型能力")
        for capability, profile_id in bindings.items():
            if profile_id and capability not in providers.get(profile_id).get("capabilities", []):
                raise ValueError(f"所选模型未声明 {capability} 能力")
        defaults = {**current["defaults"], **payload.get("defaults", {})}
        AnalysisInput(asset_id="validation", **defaults)
        result = {"bindings": bindings, "defaults": defaults}
        atomic_json(library.root / "private" / "settings.json", result)
        return result

    @app.get("/api/profiles")
    async def profiles():
        return providers.list()

    @app.get("/api/codex/status")
    async def codex_status():
        return await codex_cli.status()

    @app.post("/api/profiles")
    async def profile_save(payload: dict = Body(...)):
        if payload.get("id") and isinstance(payload.get("capabilities"), list):
            removed = [capability for capability, profile_id in settings()["bindings"].items()
                       if profile_id == payload["id"] and capability not in payload["capabilities"]]
            if removed:
                raise ValueError("请先在能力分配中解除以下绑定并保存，再修改连接：" + "、".join(removed))
        return await asyncio.to_thread(providers.upsert, payload)

    @app.delete("/api/profiles/{profile_id}")
    async def profile_delete(profile_id: str):
        if profile_id in settings()["bindings"].values():
            raise ValueError("请先解除该模型的能力绑定")
        providers.delete(profile_id)
        return {"ok": True}

    @app.post("/api/profiles/{profile_id}/test")
    async def profile_test(profile_id: str, payload: dict = Body(...)):
        result = await providers.test(profile_id, payload.get("capability", "query"))
        return {"ok": True, "message": "连接与所选能力测试通过", "usage": result.usage, "request_count": result.request_count}

    @app.get("/api/assets")
    async def assets():
        return [asset_view(a) for a in library.list_assets()]

    @app.post("/api/media/subtitle-tracks")
    async def media_subtitle_tracks(payload: SubtitleProbeInput):
        return await asyncio.to_thread(subtitle_track_info, Path(payload.video_path))

    @app.post("/api/assets")
    async def asset_register(payload: AssetInput):
        if payload.subtitle_mode == "embedded" and not settings()["bindings"].get("subtitle"):
            raise ValueError("画面内嵌字幕模式请先配置并绑定字幕视觉模型")
        asset = await asyncio.to_thread(library.register, payload)
        await search.rebuild(library.records())
        return asset_view(asset)

    @app.get("/api/assets/{asset_id}/subtitle-tracks")
    async def asset_subtitle_tracks(asset_id: str):
        source = library.require_source(asset_id)
        return await asyncio.to_thread(subtitle_track_info, source)

    @app.post("/api/assets/{asset_id}/subtitles/extract")
    async def asset_subtitle_extract(asset_id: str, payload: SubtitleExtractInput):
        asset = await asyncio.to_thread(library.extract_subtitles, asset_id,
                                       payload.subtitle_stream_index, payload.subtitle_offset_ms)
        await search.rebuild(library.records())
        return asset_view(asset)

    @app.get("/api/assets/{asset_id}")
    async def asset_detail(asset_id: str):
        asset = asset_view(library.get_asset(asset_id))
        projection = {r["id"]: r for r in library.records(asset_id)}
        observations = [{**r, "model_summary": r["summary"],
                         **{k: projection[r["id"]][k] for k in ("summary", "entities", "review_status", "note", "favorite") if r["id"] in projection},
                         "asset_id": asset_id} for r in library.observations(asset_id)]
        return {"asset": asset, "observations": [enrich(r) for r in observations], "subtitles": library.subtitles(asset_id),
                "annotations": library.annotations(asset_id), "runs": library.runs(asset_id)}

    @app.post("/api/assets/{asset_id}/relocate")
    async def relocate(asset_id: str, payload: dict = Body(...)):
        return await asyncio.to_thread(library.relocate, asset_id, payload["video_path"])

    def refresh_character_search(asset_id: str):
        # Dossiers are shared across episodes, so all affected projections change.
        for related in characters.catalog(asset_id)["assets"]:
            search.upsert(library.records(related["id"]))

    @app.get("/api/assets/{asset_id}/characters")
    async def character_catalog(asset_id: str):
        return characters.catalog(asset_id)

    @app.get("/api/assets/{asset_id}/characters/{character_id}/portrait")
    async def character_portrait(asset_id: str, character_id: str):
        from PIL import Image, ImageOps

        with library.lock:
            reference = characters.portrait_reference(asset_id, character_id)
            path = library.frame_path(reference["asset_id"], reference["frame_id"])
        with Image.open(path) as source:
            picture = ImageOps.exif_transpose(source).convert("RGB")
            if reference.get("box"):
                x, y, width, height = reference["box"]
                left, top = int(x * picture.width), int(y * picture.height)
                right = max(left + 1, min(picture.width, round((x + width) * picture.width)))
                bottom = max(top + 1, min(picture.height, round((y + height) * picture.height)))
                picture = picture.crop((left, top, right, bottom))
            picture.thumbnail((768, 768))
            output = io.BytesIO()
            picture.save(output, format="JPEG", quality=90)
        return Response(output.getvalue(), media_type="image/jpeg")

    @app.patch("/api/assets/{asset_id}/characters/{character_id}")
    async def character_edit(asset_id: str, character_id: str, payload: CharacterEditInput):
        with library.lock:
            profile = characters.update(asset_id, character_id, payload.model_dump(exclude_unset=True))
            refresh_character_search(asset_id)
        return profile

    @app.delete("/api/assets/{asset_id}/characters/{character_id}")
    async def character_delete(asset_id: str, character_id: str,
                               expected_revision: int | None = Query(default=None, ge=0)):
        with library.lock:
            characters.delete(asset_id, character_id, expected_revision=expected_revision)
            refresh_character_search(asset_id)
        return {"ok": True}

    @app.post("/api/assets/{asset_id}/characters/{character_id}/merge")
    async def character_merge(asset_id: str, character_id: str, payload: CharacterMergeInput):
        with library.lock:
            profile = characters.merge(asset_id, character_id, payload.target_id,
                                       expected_revision=payload.expected_revision)
            refresh_character_search(asset_id)
        return profile

    @app.post("/api/assets/{asset_id}/characters/reanalyze")
    async def character_reanalyze(asset_id: str, payload: CharacterReanalysisInput):
        with library.lock:
            catalog = characters.catalog(asset_id)
            if payload.expected_revision is not None and payload.expected_revision != catalog["revision"]:
                raise StaleCharacterRevision("角色档案已更新，请刷新档案后重新提交识别")
            selected = catalog["assets"] if payload.scope == "series" else [library.get_asset(asset_id)]
            selected = sorted(selected, key=lambda item: (
                item.get("season") if item.get("season") is not None else -1,
                item.get("episode") if item.get("episode") is not None else -1,
                item["created_at"], item["id"],
            ))
            current = settings()
            # Validate every episode before enqueueing any, avoiding half-created batches.
            configs = []
            for asset in selected:
                if not asset["source_available"]:
                    raise ValueError(f"「{asset['title']}」原视频离线或已变化，请先定位原文件再重新识别")
                request = AnalysisInput(**{**current["defaults"], "asset_id": asset["id"],
                    "stages": ["vision"], "start_ms": 0, "end_ms": asset["duration_ms"], "force": True,
                    "max_requests": payload.max_requests, "max_cost": payload.max_cost})
                _, config = queue.validate(request, current)
                config["character_reanalysis"] = True
                configs.append((asset["id"], config))
            jobs = [public_job(queue.add("analysis", config, selected_id)) for selected_id, config in configs]
        return {"jobs": jobs, "revision": catalog["revision"], "scope_id": catalog["scope_id"]}

    @app.get("/api/assets/{asset_id}/media")
    async def media(asset_id: str):
        library.get_asset(asset_id)
        proxy = library.asset_dir(asset_id) / "proxy" / "playback.mp4"
        path = proxy if proxy.exists() else library.require_source(asset_id)
        if not path or not path.is_file():
            raise FileNotFoundError("原视频离线")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4")

    @app.get("/api/assets/{asset_id}/frames/{frame_id}")
    async def frame(asset_id: str, frame_id: str):
        return FileResponse(library.frame_path(asset_id, frame_id), media_type="image/jpeg")

    @app.post("/api/assets/{asset_id}/proxy")
    async def proxy(asset_id: str):
        library.get_asset(asset_id)
        return public_job(queue.add("proxy", {}, asset_id))

    @app.post("/api/jobs/estimate")
    async def estimate(payload: AnalysisInput):
        return queue.estimate(payload, settings())

    @app.post("/api/jobs")
    async def job_start(payload: AnalysisInput):
        asset, config = queue.validate(payload, settings())
        return public_job(queue.add("analysis", config, asset["id"]))

    @app.get("/api/jobs")
    async def jobs():
        return [public_job(j) for j in queue.list()]

    @app.get("/api/jobs/{job_id}")
    async def job_get(job_id: str):
        return public_job(queue.get(job_id))

    @app.post("/api/jobs/{job_id}/{action}")
    async def job_control(job_id: str, action: str, payload: dict | None = Body(default=None)):
        return public_job(queue.control(job_id, action, payload))

    @app.post("/api/search")
    async def query(payload: SearchInput):
        bindings = settings()["bindings"]
        result = await search.search(payload.query, filters=payload.filters, limit=payload.limit,
                                    **{k + "_profile_id": bindings.get(k) for k in ("embedding", "query", "decision", "answer")})
        actual_records = {r["id"]: r for r in library.records()}
        results = []
        changed = False
        for record in result.get("results", []):
            # Never let generated or stale result IDs turn into source citations.
            if record["id"] in actual_records:
                canonical = actual_records[record["id"]]
                changed |= canonical["text"] != record.get("text")
                ranking = {k: record[k] for k in ("score", "match_type", "match_reason", "evidence_ids") if k in record}
                results.append(enrich({**canonical, **ranking}))
            elif record.get("source_ids") and all(rid in actual_records for rid in record["source_ids"]):
                sources = [actual_records[rid] for rid in record["source_ids"]]
                if len({r["asset_id"] for r in sources}) != 1:
                    continue
                text = "\n".join(r["text"] for r in sources)
                changed |= text != record.get("text")
                results.append(enrich({**record, "text": text, "summary": text, "subtitle_text": text,
                                       "start_ms": min(r["start_ms"] for r in sources), "end_ms": max(r["end_ms"] for r in sources)}))
        if changed:
            result["explanation"] = None
            result.setdefault("degraded", []).append("资料已有新修正，已显示最新原文；请重建索引更新排序")
        return {**result, "results": results}

    @app.get("/api/index/status")
    async def index_status():
        return search.status()

    @app.post("/api/index/rebuild")
    async def index_rebuild(payload: dict = Body(default={})):
        profile_id = settings()["bindings"].get("embedding") if payload.get("embeddings") else None
        if payload.get("embeddings") and not profile_id:
            raise ValueError("请先绑定 embedding 模型")
        return public_job(queue.add("index", {"embedding_profile_id": profile_id, "max_requests": 1000}))

    @app.post("/api/assets/{asset_id}/annotations")
    async def annotation(asset_id: str, payload: AnnotationInput):
        value = library.annotate(asset_id, payload.model_dump(exclude_unset=True))
        # Refresh lexical projection immediately; vector namespace reports staleness until rebuild.
        search.upsert(library.records(asset_id))
        return value

    @app.get("/api/collections")
    async def collections():
        return [enrich(r) for r in library.collections()]

    @app.get("/api/export/citations")
    async def citations(format: str = "markdown"):
        records = library.collections()
        if format == "json":
            return JSONResponse(records, headers={"Content-Disposition": 'attachment; filename="scenerecall-citations.json"'})
        if format != "markdown":
            raise ValueError("导出格式仅支持 markdown 或 json")

        def stamp(ms):
            return f"{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d}.{ms%1000:03d}"

        lines = ["# SceneRecall 影评素材引用", ""]
        for r in records:
            episode = f" S{r['season']}E{r['episode']}" if r.get("episode") is not None else ""
            lines += [f"## {r['title']}{episode} · {r['version']}", "",
                      f"{stamp(r['start_ms'])}–{stamp(r['end_ms'])} · {r['source']} · {r['review_status']}", "",
                      r.get("summary", r["text"]), "", f"来源：{r['asset_id']} / {r['record_id']}", ""]
            if r.get("note"):
                lines += ["笔记：" + r["note"], ""]
        return PlainTextResponse("\n".join(lines), headers={"Content-Disposition": 'attachment; filename="scenerecall-citations.md"'})

    @app.post("/api/backups")
    async def backup(payload: dict = Body(default={})):
        return await asyncio.to_thread(library.backup, bool(payload.get("include_media")))

    @app.get("/api/backups/{backup_id}")
    async def backup_download(backup_id: str):
        path = library.root / "runtime" / "backups" / f"{safe_id(backup_id)}.zip"
        if not path.is_file():
            raise FileNotFoundError("备份不存在")
        return FileResponse(path, filename=path.name, media_type="application/zip")

    @app.post("/api/restore")
    async def restore(file: UploadFile = File(...)):
        path = library.root / "runtime" / f"{uid('upload')}.zip"
        try:
            with path.open("wb") as out:
                while chunk := await file.read(1024*1024):
                    out.write(chunk)
            result = await asyncio.to_thread(library.restore, path)
            await search.rebuild(library.records())
            return result
        finally:
            path.unlink(missing_ok=True)
            await file.close()

    frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        @app.get("/")
        async def missing_frontend():
            return PlainTextResponse("SceneRecall API 已启动。请在 frontend 中运行 npm run build，或使用 Vite 开发服务器。")
    return app


def run():
    import sys
    from .service import ServiceError, parser, stop_service

    cli = parser()
    args = cli.parse_args()
    if not 1024 <= args.port <= 65535:
        cli.error("端口必须在 1024–65535 之间")
    if args.command == "stop":
        try:
            if not args.dry_run:
                print(f"正在核实 SceneRecall 服务；正常退出最多等待 {args.timeout or 30:g} 秒……", flush=True)
            print(stop_service(args.data_dir, args.port, args.timeout or 30.0, args.dry_run))
        except ServiceError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from None
        return
    if args.dry_run or args.timeout is not None:
        cli.error("--dry-run 和 --timeout 仅适用于 stop")
    # Session links are displayed once, and HTTP access logs never retain the token query.
    app = create_app(args.data_dir, allowed_ports={args.port, 5173})
    print(f"\nSceneRecall · {args.data_dir.expanduser().resolve()}")
    print(f"打开本机会话：http://127.0.0.1:{args.port}/?token={app.state.session_token}\n", flush=True)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    run()
