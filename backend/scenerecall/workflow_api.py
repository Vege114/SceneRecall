"""Authenticated local file selection and durable workflow HTTP adapters."""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from fastapi import Body

from .models import (BatchAnalysisInput, BatchEstimateInput, ImportInput, ImportRetryInput,
                     KeyframePlanInput, LocalFilesInput)
from .workflows import RequestConflict

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm", ".mpg", ".mpeg",
                    ".ts", ".m2ts", ".mts", ".wmv", ".flv", ".ogv", ".vob"}


def list_local_files(payload: LocalFilesInput):
    raw = payload.path or str(Path.home())
    if "\x00" in raw or "://" in raw:
        raise ValueError("请选择本机目录")
    requested = Path(raw).expanduser()
    if not requested.is_absolute():
        raise ValueError("目录必须使用本机绝对路径")
    try:
        directory = requested.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("所选路径不是目录")
        extensions = VIDEO_EXTENSIONS if payload.kind == "video" else {".srt", ".ass", ".vtt"}
        entries = []
        with os.scandir(directory) as children:
            for child in children:
                if child.name.startswith("."):
                    continue
                try:
                    path = Path(child.path).resolve(strict=True)
                    is_dir = path.is_dir()
                    if not os.access(path, os.R_OK) or (is_dir and not os.access(path, os.X_OK)):
                        continue
                    if not is_dir and (not path.is_file() or path.suffix.lower() not in extensions):
                        continue
                    entries.append({"name": child.name, "path": str(path), "type": "directory" if is_dir else "file"})
                except (OSError, RuntimeError):
                    continue
    except PermissionError as exc:
        raise ValueError("无权读取这个目录，请选择其他目录") from exc
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        raise ValueError("目录不存在或无法访问") from exc
    entries.sort(key=lambda entry: (entry["type"] != "directory",
                 [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", entry["name"])]))
    end = payload.offset + payload.limit
    return {"directory": str(directory), "parent": str(directory.parent) if directory.parent != directory else None,
            "entries": entries[payload.offset:end], "next_offset": end if end < len(entries) else None,
            "total": len(entries)}


def register_workflow_routes(app, library, queue, settings, public_job):
    from fastapi.responses import JSONResponse

    @app.exception_handler(RequestConflict)
    async def request_conflict(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.post("/api/local-files/list")
    async def local_files(payload: LocalFilesInput):
        return await asyncio.to_thread(list_local_files, payload)

    @app.post("/api/imports")
    async def create_import(payload: ImportInput):
        return public_job(queue.add_import(payload))

    @app.get("/api/imports/{job_id}")
    async def get_import(job_id: str):
        job = queue.get(job_id)
        if job["type"] != "import":
            raise ValueError("这不是导入任务")
        projects = {item.get("project_id") for item in job["config"]["items"]}
        return {"job": public_job(job), "items": job["config"]["items"],
                "project_id": next(iter(projects)) if len(projects) == 1 else None}

    @app.post("/api/imports/{job_id}/retry")
    async def retry_import(job_id: str, payload: ImportRetryInput):
        return public_job(queue.retry_import(job_id, payload))

    @app.post("/api/keyframe-plans")
    async def create_plan(payload: KeyframePlanInput):
        plan = queue.ensure_plan(payload.asset_id, payload.start_ms, payload.end_ms)
        return queue.plan_view(plan["id"])

    @app.get("/api/keyframe-plans/{plan_id}")
    async def get_plan(plan_id: str):
        return queue.plan_view(plan_id)

    @app.post("/api/jobs/batch/estimate")
    async def estimate_batch(payload: BatchEstimateInput):
        return queue.estimate_batch(payload.items, settings())

    @app.post("/api/jobs/batch")
    async def create_batch(payload: BatchAnalysisInput):
        result = queue.add_batch(payload, settings())
        return {**result, "jobs": [public_job(job) for job in result["jobs"]]}

    @app.post("/api/jobs/batches/{batch_id}/{action}")
    async def control_batch(batch_id: str, action: str, payload: dict | None = Body(default=None)):
        result = queue.control_batch(batch_id, action, payload)
        return {**result, "jobs": [public_job(job) for job in result["jobs"]]}
