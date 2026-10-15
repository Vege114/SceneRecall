"""Authenticated local HTTP adapters for speaker evidence and voice exemplars."""
from __future__ import annotations

import asyncio
import inspect
import threading

from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .audio_identity import AudioIdentityService
from .library import read_json


class AudioRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_ms: int = Field(default=0, ge=0, strict=True)
    end_ms: int | None = Field(default=None, gt=0, strict=True)


class AudioConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    character_id: str | None
    note: str = Field(default="", max_length=2000)


class VoiceSampleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str | None = None
    audio_base64: str | None = Field(default=None, max_length=14 * 1024 * 1024)
    filename: str = Field(default="sample.wav", max_length=300)
    language: str = Field(min_length=2, max_length=32)
    dub_version: str = Field(min_length=1, max_length=200)
    source_kind: str = "original_clip"
    rights_note: str = Field(default="", max_length=2000)
    confirmed: bool = False


class VoiceSampleStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    note: str = Field(default="", max_length=2000)


def register_audio_routes(app, library, characters=None, refresh=None) -> AudioIdentityService:
    service = AudioIdentityService(library, characters)
    cancellations = {}

    async def refreshed(asset_id):
        if refresh:
            catalog = service.characters.catalog(asset_id)
            for asset in catalog.get("assets", [{"id": asset_id}]):
                value = refresh(asset["id"])
                if inspect.isawaitable(value):
                    await value

    @app.get("/api/assets/{asset_id}/audio/settings")
    async def audio_settings(asset_id: str):
        return {"settings": service.settings(asset_id), "availability": service.availability(asset_id)}

    @app.put("/api/assets/{asset_id}/audio/settings")
    async def save_audio_settings(asset_id: str, payload: dict):
        settings = await asyncio.to_thread(service.configure, asset_id, payload)
        return {"settings": settings, "availability": service.availability(asset_id)}

    @app.get("/api/assets/{asset_id}/audio/tracks")
    async def tracks(asset_id: str):
        return await asyncio.to_thread(service.audio_tracks, asset_id)

    @app.get("/api/assets/{asset_id}/audio/records")
    async def records(asset_id: str):
        status = read_json(service._directory(asset_id) / "status.json", {"status": "not_started", "degraded": [], "timings": {}})
        return {**status, "records": await asyncio.to_thread(service.records, asset_id)}

    @app.post("/api/assets/{asset_id}/audio/analyze")
    async def analyze(asset_id: str, payload: AudioRange):
        if asset_id in cancellations:
            from fastapi import HTTPException
            raise HTTPException(409, "该素材的独立音频任务已在运行")
        event = threading.Event()
        cancellations[asset_id] = event

        def checkpoint():
            library.get_asset(asset_id)  # Deletion must stop before another model call.
            if event.is_set():
                raise InterruptedError("音频任务已取消；已完成的特征可在续跑时复用")

        operation = asyncio.create_task(asyncio.to_thread(service.run, asset_id, payload.start_ms,
                                                          payload.end_ms, checkpoint=checkpoint))
        try:
            result = await asyncio.shield(operation)
            await refreshed(asset_id)
            return result
        except InterruptedError as exc:
            return {"status": "cancelled", "records": service.records(asset_id), "degraded": [str(exc)], "timings": {}}
        except asyncio.CancelledError:
            event.set()
            # Own the worker until its checkpoint stops it. Otherwise a second
            # request could reuse the same slot while the cancelled thread writes.
            try:
                await operation
            except Exception:
                pass
            raise
        finally:
            cancellations.pop(asset_id, None)

    @app.post("/api/assets/{asset_id}/audio/cancel")
    async def cancel_audio(asset_id: str):
        library.get_asset(asset_id)
        event = cancellations.get(asset_id)
        if event:
            event.set()
        return {"status": "cancelling" if event else "not_running",
                "reason": "当前本机模型调用完成后在检查点停止；已完成的 PCM 与特征保留"}

    @app.post("/api/assets/{asset_id}/audio/records/{record_id}/confirm")
    async def confirm(asset_id: str, record_id: str, payload: AudioConfirmation):
        row = service.confirm(asset_id, record_id, payload.character_id, payload.note)
        await refreshed(asset_id)
        return row

    @app.get("/api/assets/{asset_id}/audio/records/{record_id}/clip")
    async def clip(asset_id: str, record_id: str):
        return FileResponse(service.clip_path(asset_id, record_id), media_type="audio/wav")

    @app.get("/api/assets/{asset_id}/characters/{character_id}/voice-samples")
    async def samples(asset_id: str, character_id: str):
        return {"samples": service.samples(asset_id, character_id)}

    @app.post("/api/assets/{asset_id}/characters/{character_id}/voice-samples")
    async def add_sample(asset_id: str, character_id: str, payload: VoiceSampleInput):
        row = await asyncio.to_thread(service.add_sample, asset_id, character_id, payload.model_dump(exclude_none=True))
        await refreshed(asset_id)
        return row

    @app.patch("/api/assets/{asset_id}/characters/{character_id}/voice-samples/{sample_id}")
    async def update_sample(asset_id: str, character_id: str, sample_id: str, payload: VoiceSampleStatus):
        row = service.update_sample(asset_id, character_id, sample_id, payload.status, payload.note)
        await refreshed(asset_id)
        return row

    @app.delete("/api/assets/{asset_id}/characters/{character_id}/voice-samples/{sample_id}")
    async def revoke_sample(asset_id: str, character_id: str, sample_id: str):
        row = service.update_sample(asset_id, character_id, sample_id, "revoked", "用户删除样本；保留历史审计")
        await refreshed(asset_id)
        return row

    @app.get("/api/assets/{asset_id}/characters/{character_id}/voice-samples/{sample_id}/audio")
    async def sample_audio(asset_id: str, character_id: str, sample_id: str):
        sample = next((row for row in service.samples(asset_id, character_id) if row["id"] == sample_id), None)
        if sample is None:
            raise ValueError("声音样本不存在")
        return FileResponse(service._sample_file(asset_id, sample), media_type="audio/wav")

    @app.post("/api/assets/{asset_id}/audio/records/{record_id}/sync-review")
    async def sync_review(asset_id: str, record_id: str, payload: dict):
        from .av_sync import review_segment
        return await asyncio.to_thread(review_segment, service, asset_id, record_id, payload.get("tracks", []))

    return service
