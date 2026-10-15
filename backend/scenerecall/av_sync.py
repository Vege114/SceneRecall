"""Opt-in Animated_AD LWTNet temporal-adapter review on continuous source media.

The third-party code is deliberately not vendored: its repository currently has
no license grant. The user supplies an authorized local checkout and weights.
Scores remain review evidence, never automatic visible-action assertions.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import media
from .audio_adapters import model_identity
from .audio_media import extract_audio
from .library import atomic_json, digest, read_json
from .local_resources import LOCAL_MODEL_GATE

SYNC_PREPROCESS = "animated-ad-rgb255-25fps-480x270-pcm16-16k-v1"


def sync_availability(config: dict) -> dict:
    if not config.get("sync_enabled"):
        return {"available": False, "reason": "音画同步复核未启用"}
    root, checkpoint = Path(config.get("sync_repository_path") or "/nonexistent"), Path(config.get("sync_checkpoint_path") or "/nonexistent")
    if not (root / "character_recognition/audio_recognition/model/lwtnet.py").is_file() or not checkpoint.is_file():
        return {"available": False, "reason": "缺少 Animated_AD 本机代码或动画 temporal-adapter 权重；官方仓库未发现公开的动画权重下载"}
    if not config.get("sync_license_acknowledged"):
        return {"available": False, "reason": "Animated_AD 未提供明确代码许可证；需先确认本机使用授权"}
    if not config.get("sync_animation_verified"):
        return {"available": False, "reason": "未验证动画权重及 RGB 0..255 预处理；真人基础模型不能冒充动画复核"}
    return {"available": True, "reason": "已配置本机动画同步模型；仅输出待人工核对的短片段同步分数"}


def validate_track(track: dict, start_ms: int, end_ms: int) -> list[dict]:
    """Require actual dense source-timed boxes, not sparse keyframe collages."""
    boxes = track.get("boxes", [])
    if track.get("association_status") != "confirmed" or not track.get("character_id"):
        raise ValueError("同步候选必须具有已确认视觉身份；声音身份不能证明可见身份")
    if len(boxes) < 15:
        raise ValueError("同步复核需要至少 15 张真实连续帧的时间与人物框")
    last = None
    for row in boxes:
        at = row.get("at_ms")
        box = row.get("box", [])
        if not isinstance(at, (int, float)) or len(box) != 4:
            raise ValueError("连续人物轨迹数据无效")
        if last is not None and not 0 < at - last <= 80:
            raise ValueError("同步人物轨迹不连续；不可使用不等间隔关键帧拼图")
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError("同步人物框越界")
        last = at
    if boxes[0]["at_ms"] > start_ms + 40 or boxes[-1]["at_ms"] < end_ms - 80:
        raise ValueError("连续轨迹未覆盖待复核的原片时间范围")
    return boxes


def review_segment(service, asset_id: str, record_id: str, tracks: list[dict], *, checkpoint=None) -> dict:
    config = service.settings(asset_id)
    available = sync_availability(config)
    if not available["available"]:
        return {"status": "disabled", "reason": available["reason"], "results": []}
    record = service.record(asset_id, record_id)
    if record["association_status"] == "confirmed" and not record.get("identity_conflict"):
        return {"status": "not_needed", "reason": "仅对声音身份歧义或证据冲突进行同步复核", "results": []}
    start, end = record["start_ms"], min(record["end_ms"], record["start_ms"] + 8000)
    if end - start < 640:
        return {"status": "too_short", "reason": "片段不足同步模型的连续 15 帧窗口", "results": []}
    if len(tracks) > 8:
        raise ValueError("单次音画复核最多 8 条候选轨迹")
    allowed = {row["id"] for row in service.characters.catalog(asset_id)["profiles"]}
    for track in tracks:
        validate_track(track, start, end)
        if track["character_id"] not in allowed:
            raise ValueError("同步轨迹角色不在当前作品范围")
    root = service._directory(asset_id) / "sync"
    version = model_identity(config["sync_checkpoint_path"], "animated-ad-lwtnet-temporal")
    key = digest({"record": record_id, "tracks": tracks, "model": version, "preprocessing": SYNC_PREPROCESS})
    directory = root / key
    saved = read_json(directory / "result.json")
    if saved:
        return {**saved, "cache_hit": True}
    directory.mkdir(parents=True, exist_ok=True)
    source = service.library.require_source(asset_id)
    metadata = media.probe(source)
    extract_audio(source, directory / "audio.wav", record["audio_stream_index"], start, end, metadata,
                  checkpoint=checkpoint)
    origin = float(metadata.get("start_time_seconds", metadata.get("start_time_ms", 0) / 1000))
    # Decode continuous source PTS to a regular 25 fps grid. This is not a contact
    # sheet and preserves the exact same start/end origin as extracted audio.
    filters = (f"setpts=PTS-({origin:.9f})/TB,trim=start={start / 1000:.6f}:end={end / 1000:.6f},"
               f"setpts=PTS-({start / 1000:.6f})/TB,fps=fps=25:start_time=0,scale=480:270")
    media._run([media._tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-threads", "1", "-copyts", "-i", str(source),
                "-map", f"0:{metadata['video_stream_index']}", "-an", "-vf", filters, "-pix_fmt", "rgb24",
                "-f", "rawvideo", str(directory / "frames.rgb")], timeout=120)
    request = {"repository": str(Path(config["sync_repository_path"]).expanduser().resolve()),
               "checkpoint": str(Path(config["sync_checkpoint_path"]).expanduser().resolve()),
               "device": config["device"], "start_ms": start, "end_ms": end, "tracks": tracks,
               "model_version": version, "preprocessing": SYNC_PREPROCESS}
    atomic_json(directory / "request.json", request)
    if checkpoint:
        checkpoint()
    with LOCAL_MODEL_GATE:
        process = subprocess.run([sys.executable, "-m", "scenerecall.av_sync_worker", str(directory)],
                                 capture_output=True, timeout=180, check=False)
    if process.returncode:
        return {"status": "unavailable", "reason": "同步模型加载或调用失败；检查完整 adapter 权重及独立运行环境",
                "results": [], "model_version": version}
    value = json.loads(process.stdout)
    if value.get("status") != "review_required":
        raise ValueError("同步模型返回无效结果")
    atomic_json(directory / "result.json", value)
    return value
