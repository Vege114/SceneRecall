"""Source-bound human ASD drafts, separate from model predictions and identities."""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
import threading
import wave
from pathlib import Path

from scenerecall import media
from scenerecall.audio_media import audio_tracks, crop_pcm, PREPROCESS_VERSION
from scenerecall.library import SCHEMA, atomic_json, confined_path, digest, now, read_json, safe_id
from scenerecall.lrasd_preview import LrasdPreviewService, file_sha256

ANNOTATION_VERSION = "scenerecall-human-asd-v1"
GRID_VERSION = "presentation-rgb480x270-bilinear25-pcm16-mono16k-v1"
MEDIA_FILES = {"frames_rgb": "frames.rgb", "audio_wav": "audio.wav", "preview_mp4": "preview.mp4"}


class AnnotationConflict(ValueError):
    pass


def checked_range(start, end, duration):
    if (type(start) is not int or type(end) is not int or start < 0 or start % 40 or end % 40
            or not 640 <= end - start <= 8000 or end > duration):
        raise ValueError("标注片段须位于原片范围内，长 0.64–8 秒，起止均按 40ms 对齐")
    return (end - start) // 40


def checked_box(box):
    if box is None:
        return None
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
        raise ValueError("人物框必须是四个有限数值或 null")
    x, y, width, height = map(float, box)
    if min(x, y) < 0 or min(width, height) <= 0 or x >= 1 or y >= 1 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
        raise ValueError("归一化人物框越界或为空")
    return [x, y, min(width, 1 - x), min(height, 1 - y)]


def checked_tracks(tracks, frames, allowed):
    if not isinstance(tracks, list) or len(tracks) > 32:
        raise ValueError("每片段最多 32 条人工人物轨迹")
    result, seen = [], set()
    permitted = {"id", "character_id", "boxes", "labels", "reviewed", "excluded", "note"}
    for row in tracks:
        if not isinstance(row, dict) or set(row) - permitted or permitted - row.keys():
            raise ValueError("人工轨迹字段不完整或包含服务器专用字段")
        identity = safe_id(row["id"])
        if not identity.startswith("track_") or identity in seen:
            raise ValueError("人物轨迹 ID 必须唯一并以 track_ 开头")
        seen.add(identity)
        character = row["character_id"]
        if character is not None and (not isinstance(character, str) or character not in allowed):
            raise ValueError("角色不在当前作品允许的名单中")
        if type(row["reviewed"]) is not bool or type(row["excluded"]) is not bool:
            raise ValueError("reviewed/excluded 必须是显式布尔值")
        if not isinstance(row["note"], str) or len(row["note"]) > 4000:
            raise ValueError("轨迹备注过长或不是文本")
        if not isinstance(row["labels"], list) or len(row["labels"]) != frames:
            raise ValueError("人工标签必须逐帧覆盖片段")
        if any(type(value) is not int or value not in (-1, 0, 1) for value in row["labels"]):
            raise ValueError("人工标签仅允许 -1、0、1，不能使用布尔值")
        if not isinstance(row["boxes"], list) or len(row["boxes"]) != frames:
            raise ValueError("人物框必须逐帧覆盖片段，未出现帧使用 null")
        boxes = [checked_box(box) for box in row["boxes"]]
        if any(label != -1 and box is None for label, box in zip(row["labels"], boxes)):
            raise ValueError("已标为说话或未说话的帧必须有对应人物框")
        result.append({**copy.deepcopy(row), "boxes": boxes})
    return result


class ReadOnlyLibrary:
    """Read a single existing asset without Library's migrations or startup writes."""
    def __init__(self, root, asset_id):
        self.root = Path(root).expanduser().resolve()
        self.asset_id, self.lock = safe_id(asset_id), threading.RLock()
        self.get_asset(asset_id)

    def asset_dir(self, asset_id):
        if asset_id != self.asset_id:
            raise FileNotFoundError("独立标注程序仅绑定启动时选定的素材")
        return confined_path(self.root / "assets", safe_id(asset_id))

    def get_asset(self, asset_id):
        asset = read_json(self.asset_dir(asset_id) / "manifest.json")
        if not asset or asset.get("id") != asset_id:
            raise FileNotFoundError("绑定素材不存在")
        catalog = read_json(self.root / "organization" / "catalog.json", {})
        membership = catalog.get("assets", {}).get(asset_id, {})
        if membership.get("trash_id"):
            raise FileNotFoundError("绑定素材已在回收站")
        asset.update({key: membership[key] for key in ("character_scope_id", "anime_profile", "project_id") if key in membership})
        location = read_json(self.root / "private" / "media-locations.json", {}).get(asset_id)
        path = Path(location["path"] if isinstance(location, dict) else location) if location else None
        exists = bool(path and path.is_file())
        current = path.stat() if exists else None
        asset["source_available"] = bool(exists and isinstance(location, dict)
            and current.st_size == location.get("size") and current.st_mtime_ns == location.get("mtime_ns"))
        return asset

    def require_source(self, asset_id):
        if not self.get_asset(asset_id)["source_available"]:
            raise ValueError("原片离线或文件状态已改变，请先核对素材库中的源定位")
        item = read_json(self.root / "private" / "media-locations.json")[asset_id]
        return Path(item["path"])


class ReadOnlyAudio:
    def __init__(self, library):
        self.library = library

    def _directory(self, asset_id):
        return self.library.asset_dir(asset_id) / "audio"

    def settings(self, asset_id):
        return {"stream_index": None, "language": "ja", "sync_detector_path": "",
                **read_json(self._directory(asset_id) / "settings.json", {})}


class ReadOnlyCharacters:
    def __init__(self, library):
        self.library = library

    def catalog(self, asset_id):
        from scenerecall.characters import CharacterStore, scope_id
        asset = self.library.get_asset(asset_id)
        scope = scope_id(asset)
        registry = CharacterStore(self.library)._read(scope)  # Validation and read only.
        policy = CharacterStore.cast_policy(registry, asset.get("anime_profile"))
        profiles = [copy.deepcopy(profile) for profile in registry["profiles"].values()
                    if CharacterStore.profile_allowed(profile, policy)]
        gallery_dir = confined_path(self.library.root / "works", safe_id(scope)) / "reference-images"
        gallery = read_json(gallery_dir / "gallery.json", {"references": []})
        for profile in profiles:
            profile["references"] = []
            for row in gallery["references"]:
                if row.get("status") == "revoked" or CharacterStore._resolve(registry, row.get("character_id")) != profile["id"]:
                    continue
                identity = safe_id(row["id"])
                path = confined_path(gallery_dir, identity + ".png")
                if not path.is_file() or file_sha256(path) != row.get("sha256"):
                    continue
                if row.get("source", {}).get("kind") == "frame_crop" and row["source"].get("asset_id") != asset_id:
                    continue
                profile["references"].append({"id": identity, "status": row.get("status"), "source": row.get("source"),
                    "image_url": f"/api/characters/{profile['id']}/references/{identity}/image", "reference_only": True})
        return {"scope_id": scope, "profiles": profiles}

    def reference_path(self, asset_id, character_id, reference_id):
        catalog = self.catalog(asset_id)
        profile = next((row for row in catalog["profiles"] if row["id"] == character_id), None)
        if profile is None or not any(row["id"] == reference_id for row in profile["references"]):
            raise FileNotFoundError("该作品的本机角色参考图不存在")
        return confined_path(self.library.root / "works" / safe_id(catalog["scope_id"]) / "reference-images", safe_id(reference_id) + ".png")


class AsdAnnotationService:
    def __init__(self, library, characters=None, audio_service=None, data_dir=None):
        self.library = library
        if data_dir is None:
            raise ValueError("独立人工标注必须指定专用数据目录")
        self.data_dir = Path(data_dir).expanduser().resolve()
        if self.data_dir.is_relative_to(library.root.resolve()):
            raise ValueError("独立人工标注数据目录不能位于只读素材库内")
        if characters is None:
            characters = ReadOnlyCharacters(library)
        self.characters = characters
        if audio_service is None:
            audio_service = ReadOnlyAudio(library)
        self.audio = audio_service
        self.previews = getattr(audio_service, "lrasd_previews", None) or LrasdPreviewService(audio_service)
        self._guard, self._running = threading.RLock(), {}

    def _root(self, asset_id):
        self.library.get_asset(asset_id)
        root = (self.data_dir / "clips").resolve()
        if not root.is_relative_to(self.data_dir):
            raise ValueError("人工标注目录不能通过链接写到专用数据目录以外")
        return root

    def _source(self, asset_id):
        asset = self.library.get_asset(asset_id)
        source = Path(self.library.require_source(asset_id))
        stat = source.stat()
        return asset, source, {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _bound(self, asset_id, clip, snapshot=None):
        asset, source, stat = snapshot or self._source(asset_id)
        if (clip.get("schema_version") != SCHEMA or clip.get("annotation_version") != ANNOTATION_VERSION
                or clip.get("asset_id") != asset_id or clip.get("source_fingerprint") != asset["fingerprint"]
                or clip.get("source_stat") != stat):
            raise ValueError("原片指纹或文件状态已变化，不能继续修改该标注片段")
        return asset, source, stat

    def _read(self, asset_id, clip_id, snapshot=None):
        directory = confined_path(self._root(asset_id), safe_id(clip_id))
        clip = read_json(directory / "clip.json")
        if not clip or clip.get("id") != clip_id:
            raise FileNotFoundError("标注片段不存在")
        self._bound(asset_id, clip, snapshot)
        return directory, clip

    def get(self, asset_id, clip_id, *, _snapshot=None):
        with self._guard, self.library.lock:
            directory, clip = self._read(asset_id, clip_id, _snapshot)
            task = self._running.get((asset_id, clip_id))
            if clip["status"] == "preparing" and not (task and task.is_alive()):
                clip.update(status="failed", reason="上次准备随应用进程中断，可重新准备该片段", updated_at=now())
                atomic_json(directory / "clip.json", clip)
            return copy.deepcopy(clip)

    def list(self, asset_id):
        snapshot = self._source(asset_id)
        asset, _, stat = snapshot
        root = self._root(asset_id)
        clips = [self.get(asset_id, path.parent.name, _snapshot=snapshot) for path in sorted(root.glob("*/clip.json"))]
        final_asset, _, final_stat = self._source(asset_id)
        if final_asset["fingerprint"] != asset["fingerprint"] or final_stat != stat:
            raise ValueError("读取任务列表期间原片已变化，请重新加载")
        hidden = read_json(confined_path(self.data_dir, "hidden-proposals.json"), {})
        if hidden:
            if hidden.get("asset_id") != asset_id or hidden.get("source_fingerprint") != asset["fingerprint"]:
                raise ValueError("候选缓存清单与原片不匹配")
            hidden_ids = {safe_id(identity) for identity in hidden.get("clip_ids", [])}
            clips = [clip for clip in clips if clip["id"] not in hidden_ids or clip["revision"] != 0
                     or any(track.get("reviewed") or track.get("character_id") is not None
                            or any(label != -1 for label in track.get("labels", [])) for track in clip["tracks"])]
        clips.sort(key=lambda row: (row["start_ms"], row["id"]))
        tracks = [track for clip in clips if clip["status"] == "ready" for track in clip["tracks"]]
        labels = [label for track in tracks if track["reviewed"] and not track["excluded"] for label in track["labels"]]
        duration = asset["duration_ms"]
        default_start = 1204000 if duration >= 1208000 else 0
        default_end = min(duration // 40 * 40, default_start + 4000)
        return {"schema_version": SCHEMA, "annotation_version": ANNOTATION_VERSION, "asset_id": asset_id,
                "source_fingerprint": asset["fingerprint"], "asset_title": asset.get("title", asset_id),
                "duration_ms": asset["duration_ms"], "storage_root_info": str(self.data_dir), "clips": clips,
                "stats": {"clips": len(clips), "reviewed_tracks": sum(row["reviewed"] and not row["excluded"] for row in tracks),
                          "excluded_tracks": sum(row["excluded"] for row in tracks),
                          "labeled_frames": sum(value >= 0 for value in labels),
                          "positive_frames": labels.count(1), "negative_frames": labels.count(0)},
                "default_start_ms": default_start, "default_end_ms": default_end,
                "annotation_plan": self.annotation_plan(asset_id, clips)}

    def annotation_plan(self, asset_id, clips):
        path = confined_path(self.data_dir, "annotation-plan.json")
        if not path.exists():
            return None
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("标注任务计划过大")
        plan = json.loads(path.read_text(encoding="utf-8-sig"))
        asset = self.library.get_asset(asset_id)
        if (not isinstance(plan, dict) or plan.get("schema_version") != "scenerecall-asd-scene-plan-v1"
                or plan.get("asset_id") != asset_id or plan.get("source_fingerprint") != asset["fingerprint"]
                or not isinstance(plan.get("items"), list) or len(plan["items"]) > 200):
            raise ValueError("标注任务计划与当前原片不匹配")
        seen, scenes, result = set(), {}, copy.deepcopy(plan)
        indexed = {clip["id"]: clip for clip in clips}
        for item in result["items"]:
            if not isinstance(item, dict):
                raise ValueError("标注任务格式无效")
            identity, clip_id, scene = safe_id(item.get("id")), safe_id(item.get("clip_id")), safe_id(item.get("scene_id"))
            split = item.get("split")
            if identity in seen or split not in {"train", "val", "test"} or (scene in scenes and scenes[scene] != split):
                raise ValueError("标注任务重复或同场景跨训练验证集")
            seen.add(identity)
            scenes[scene] = split
            checked_range(item.get("start_ms"), item.get("end_ms"), asset["duration_ms"])
            clip = indexed.get(clip_id)
            if clip and (clip["start_ms"] != item["start_ms"] or clip["end_ms"] != item["end_ms"]):
                raise ValueError("标注任务时间范围与片段不一致")
            item["status"] = clip["status"] if clip else "missing"
        return result

    def _decode_inputs(self, source, directory, clip, config, metadata):
        """Reuse only a fingerprint-bound full presentation PCM; never seek-reset its origin."""
        cache = confined_path(self.data_dir, "pcm-cache")
        identity = read_json(confined_path(cache, "identity.json"), {})
        expected = {"source_fingerprint": clip["source_fingerprint"], "source_stat": clip["source_stat"],
                    "stream_index": config["stream_index"], "duration_ms": metadata["duration_ms"],
                    "origin_ms": metadata.get("start_time_ms", 0), "preprocessing": PREPROCESS_VERSION,
                    "relative_path": "japanese-presentation.wav"}
        if not identity or any(identity.get(key) != value for key, value in expected.items()):
            return self.previews._decode(source, directory, clip["start_ms"], clip["end_ms"], config)
        pcm = confined_path(cache, "japanese-presentation.wav")
        if (not pcm.is_file() or pcm.stat().st_size != identity.get("bytes") or file_sha256(pcm) != identity.get("sha256")):
            raise ValueError("原片音频缓存内容校验失败，请重新准备缓存")
        with wave.open(str(pcm)) as audio:
            if ((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes())
                    != (1, 2, 16000, metadata["duration_ms"] * 16)):
                raise ValueError("音频缓存不在原片完整呈现时间轴上")
        start, end = clip["start_ms"], clip["end_ms"]
        crop_pcm(pcm, directory / "audio.wav", start, end)
        origin = float(metadata.get("start_time_seconds", metadata.get("start_time_ms", 0) / 1000))
        filters = (f"setpts=PTS-({origin:.9f})/TB,trim=start={start / 1000:.6f}:end={end / 1000:.6f},"
                   f"setpts=PTS-({start / 1000:.6f})/TB,fps=fps=25:start_time=0,scale=480:270:flags=bilinear")
        seek = max(0., start / 1000 - 1.) + origin
        media._run([media._tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-threads", "1", "-copyts",
                    "-seek_timestamp", "1", "-ss", f"{seek:.9f}", "-i", source,
                    "-map", f"0:{metadata['video_stream_index']}", "-an", "-sn", "-dn", "-vf", filters,
                    "-pix_fmt", "rgb24", "-frames:v", str(clip["frame_count"]), "-f", "rawvideo",
                    str(directory / "frames.rgb")], timeout=120)
        if (directory / "frames.rgb").stat().st_size != clip["frame_count"] * 480 * 270 * 3:
            raise ValueError("原片未提供请求时间网格的完整连续画面")
        return config["stream_index"]

    def prepare(self, asset_id, start_ms, end_ms):
        asset, source, stat = self._source(asset_id)
        count = checked_range(start_ms, end_ms, asset["duration_ms"])
        config = self.audio.settings(asset_id)
        key = "asd_" + digest([ANNOTATION_VERSION, asset["fingerprint"], stat, start_ms, end_ms, config.get("stream_index")])
        with self._guard, self.library.lock:
            for (running_asset, running_id), task in self._running.items():
                if running_asset == asset_id and task.is_alive() and running_id != key:
                    raise AnnotationConflict("该素材已有标注片段正在准备，请等待完成")
            directory = confined_path(self._root(asset_id), key)
            saved = read_json(directory / "clip.json")
            if saved:
                self._bound(asset_id, saved)
                task = self._running.get((asset_id, key))
                if saved["status"] == "ready" or (saved["status"] == "preparing" and task and task.is_alive()):
                    return copy.deepcopy(saved)
            timestamp = now()
            clip = {"schema_version": SCHEMA, "annotation_version": ANNOTATION_VERSION, "asset_id": asset_id,
                    "id": key, "revision": 0, "status": "preparing", "start_ms": start_ms, "end_ms": end_ms,
                    "fps": 25, "frame_count": count, "clip_url": None, "tracks": [], "note": "",
                    "reason": "正在准备本机短片及待人工修订的人物框；尚无人工标签",
                    "source_fingerprint": asset["fingerprint"], "source_stat": stat,
                    "asset_title": asset.get("title", asset_id), "movie": asset.get("title", asset_id),
                    "audio_stream_index": None, "reviewed_at": None, "reviewer": None,
                    "created_at": timestamp, "updated_at": timestamp,
                    "server": {"preprocessing": GRID_VERSION, "configured_audio_stream_index": config.get("stream_index"),
                               "source_fingerprint": asset["fingerprint"], "source_stat": stat,
                               "source_sha256": asset["fingerprint"], "source_sha256_basis": "library_full_file_sha256",
                               "model_scores_copied": False, "initial_labels": "all_unknown"}}
            atomic_json(directory / "clip.json", clip)
            thread = threading.Thread(target=self._prepare_background, args=(asset_id, key, str(source), config),
                                      name=f"asd-prepare-{asset_id}", daemon=True)
            self._running[(asset_id, key)] = thread
            thread.start()
            return copy.deepcopy(clip)

    def _media_valid(self, directory, frames):
        if confined_path(directory, "frames.rgb").stat().st_size != frames * 480 * 270 * 3:
            raise ValueError("标注画面与连续 25fps 网格不一致")
        with wave.open(str(confined_path(directory, "audio.wav")), "rb") as audio:
            if ((audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getcomptype(), audio.getnframes())
                    != (1, 16000, 2, "NONE", frames * 640)):
                raise ValueError("标注音频必须为同时间范围的精确 16kHz 单声道 PCM16")
        path = confined_path(directory, "preview.mp4")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("标注短片不存在")

    def _reuse_preview(self, asset_id, clip, stream):
        root = self.library.asset_dir(asset_id) / "audio" / "lrasd-previews"
        for directory in sorted(root.glob("*")):
            if not directory.is_dir() or directory.is_symlink():
                continue
            try:
                identity = read_json(confined_path(directory, "identity.json"), {})
                result = read_json(confined_path(directory, "result.json"), {})
                if (identity.get("preview_version") != "scenerecall-lrasd-preview-v3"
                        or result.get("status") not in {"review_required", "no_tracks"}
                        or any(identity.get(key) != clip[key] for key in ("source_fingerprint", "start_ms", "end_ms"))
                        or identity.get("source_stat") != [clip["source_stat"]["size"], clip["source_stat"]["mtime_ns"]]
                        or result.get("audio_stream_index") != stream
                        or identity.get("audio_stream_index") not in (None, stream)
                        or identity.get("preprocessing") != "lrasd-gray112-pcm16-mfcc13-25fps-fullbody-v1"
                        or any(result.get(key) != clip[key] for key in ("source_fingerprint", "start_ms", "end_ms"))):
                    continue
                self._media_valid(directory, clip["frame_count"])
                request = read_json(confined_path(directory, "request.json"), {})
                if request:
                    if (request.get("start_ms") != clip["start_ms"] or request.get("end_ms") != clip["end_ms"]
                            or request.get("frame_width", 480) != 480 or request.get("frame_height", 270) != 270):
                        continue
                    tracks = request.get("tracks", [])
                elif result["status"] == "no_tracks":
                    tracks = []
                else:
                    continue
                return directory, tracks, {"kind": "reused_preview_raw_detections", "preview_job_id": directory.name,
                                            "preview_identity_sha256": file_sha256(directory / "identity.json"),
                                            "raw_request_sha256": file_sha256(directory / "request.json") if request else None,
                                            "detector_sha256": identity.get("detector_sha256")}
            except (ValueError, OSError, wave.Error):
                continue
        return None

    def _initial_tracks(self, raw, clip):
        tracks, seen = [], set()
        for row in raw:
            identity = safe_id(row.get("id"))
            if not identity.startswith("track_") or identity in seen:
                raise ValueError("原始检测轨迹 ID 无效或重复")
            seen.add(identity)
            boxes = [None] * clip["frame_count"]
            for observed in row.get("boxes", []):
                at = observed.get("at_ms")
                if (type(at) is not int or at < clip["start_ms"] or at >= clip["end_ms"]
                        or (at - clip["start_ms"]) % 40):
                    raise ValueError("原始人物框不在标注网格内")
                index = (at - clip["start_ms"]) // 40
                if boxes[index] is not None:
                    raise ValueError("原始人物框时间重复")
                boxes[index] = checked_box(observed.get("box"))
            tracks.append({"id": identity, "character_id": None, "boxes": boxes,
                           "labels": [-1] * clip["frame_count"], "reviewed": False, "excluded": False, "note": "",
                           "reviewed_at": None, "reviewer": None,
                           "provenance": {"box_source": "detector_proposal", "boxes_reviewed": False,
                                          "labels_source": "unknown", "model_scores_copied": False}})
        return tracks

    def _reuse_annotation_inputs(self, asset_id, directory, clip, stream):
        """Crop immutable, still-unreviewed proposal inputs; never inherit human edits."""
        for path in sorted(self._root(asset_id).glob("*/clip.json")):
            if path.parent == directory:
                continue
            parent = read_json(path, {})
            if (parent.get("status") != "ready" or parent.get("revision") != 0
                    or parent.get("source_fingerprint") != clip["source_fingerprint"]
                    or parent.get("source_stat") != clip["source_stat"] or parent.get("audio_stream_index") != stream
                    or parent.get("server", {}).get("preprocessing") != GRID_VERSION
                    or not (parent["start_ms"] <= clip["start_ms"] < clip["end_ms"] <= parent["end_ms"])):
                continue
            tracks = parent.get("tracks", [])
            if any(track.get("reviewed") or track.get("excluded") or track.get("character_id") is not None
                   or any(label != -1 for label in track.get("labels", []))
                   or track.get("provenance", {}).get("box_source") != "detector_proposal" for track in tracks):
                continue
            snapshot = file_sha256(path)
            try:
                self._media_valid(path.parent, parent["frame_count"])
                for key, name in MEDIA_FILES.items():
                    source = confined_path(path.parent, name)
                    saved = parent["server"]["files"][key]
                    if source.stat().st_size != saved.get("bytes") or file_sha256(source) != saved.get("sha256"):
                        raise ValueError("未复核候选缓存校验失败")
            except (OSError, ValueError, KeyError, wave.Error):
                continue
            first = (clip["start_ms"] - parent["start_ms"]) // 40
            last = first + clip["frame_count"]
            with confined_path(path.parent, "frames.rgb").open("rb") as source, (directory / "frames.rgb").open("wb") as target:
                source.seek(first * 480 * 270 * 3)
                remaining = clip["frame_count"] * 480 * 270 * 3
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("候选原始画面不完整")
                    target.write(chunk)
                    remaining -= len(chunk)
            crop_pcm(confined_path(path.parent, "audio.wav"), directory / "audio.wav", first * 40, last * 40)
            raw = []
            for track in tracks:
                boxes = [{"at_ms": parent["start_ms"] + frame * 40, "box": box}
                         for frame, box in enumerate(track["boxes"]) if first <= frame < last and box is not None]
                if boxes:
                    raw.append({"id": track["id"], "boxes": boxes})
            if file_sha256(path) != snapshot:
                raise AnnotationConflict("候选片段在裁剪时被编辑，请重新准备")
            return raw, {"kind": "cropped_unreviewed_detector_proposals", "parent_clip_id": parent["id"],
                         "parent_annotation_sha256": snapshot, "parent_media": parent["server"]["files"],
                         "parent_first_frame": first, "parent_last_frame_exclusive": last,
                         "model_scores_copied": False, "human_labels_copied": False}
        return None

    def _choose_stream(self, metadata, config):
        tracks = audio_tracks(metadata)
        stream = config.get("stream_index")
        if stream is not None:
            if type(stream) is not int or stream not in {row["index"] for row in tracks}:
                raise ValueError("标注所选音轨已不存在，请重新选择音轨")
            return stream
        japanese = next((row["index"] for row in tracks if row["language"].lower() in {"ja", "jpn", "japanese"}), None)
        stream = japanese if japanese is not None else next((row["index"] for row in tracks if row["default"]),
                                                            tracks[0]["index"] if tracks else None)
        if stream is None:
            raise ValueError("原片没有可用于人工 ASD 标注的音轨")
        return stream

    def _revision(self, directory, clip):
        name = f"{clip['revision']:08d}-{digest(clip)}.json"
        location = directory / "revisions" / name
        if not location.exists():
            atomic_json(location, clip)

    def _prepare_background(self, asset_id, clip_id, source, config):
        directory = confined_path(self._root(asset_id), clip_id)
        try:
            clip = read_json(directory / "clip.json")
            self._bound(asset_id, clip)
            metadata = media.probe(Path(source))
            checked_range(clip["start_ms"], clip["end_ms"], metadata["duration_ms"])
            stream = self._choose_stream(metadata, config)
            reused = self._reuse_preview(asset_id, clip, stream)
            cropped = self._reuse_annotation_inputs(asset_id, directory, clip, stream) if not reused else None
            reason = "媒体与检测框已准备；所有说话标签均未知，请逐帧人工复核"
            if reused:
                cached, raw, provenance = reused
                for name in MEDIA_FILES.values():
                    shutil.copyfile(confined_path(cached, name), confined_path(directory, name))
            elif cropped:
                raw, provenance = cropped
                self.previews._make_clip(directory, clip["start_ms"], clip["end_ms"])
            else:
                decoded = self._decode_inputs(source, directory, clip,
                                              {**config, "stream_index": stream, "_asset_id": asset_id}, metadata)
                if decoded != stream:
                    raise ValueError("解码音轨与标注音轨不一致")
                self.previews._make_clip(directory, clip["start_ms"], clip["end_ms"])
                raw, provenance = [], {"kind": "local_decode_manual_boxes", "model_scores_copied": False}
                detector = Path(config.get("sync_detector_path") or "/nonexistent").expanduser()
                python = shutil.which(os.environ.get("SCENERECALL_VISUAL_PYTHON", sys.executable))
                if detector.is_file() and python:
                    try:
                        atomic_json(directory / "preview-request.json", {"start_ms": clip["start_ms"],
                                    "end_ms": clip["end_ms"], "detector_path": str(detector)})
                        detection = self.previews._worker(python, "lrasd_preview_worker.py", directory, timeout=180)
                        raw = detection["tracks"]
                        provenance = {"kind": "local_decode_detector_proposals", "detector_sha256": file_sha256(detector),
                                      "tracking": detection.get("tracking"), "model_scores_copied": False}
                    except Exception as exc:
                        reason = f"短片已准备；人物检测不可用（{type(exc).__name__}），请手动画框并标注"
                else:
                    reason = "短片已准备；未配置可用人物检测器，可手动画框并标注"
            self._media_valid(directory, clip["frame_count"])
            # Invalid proposals must never block an otherwise usable manual clip.
            try:
                tracks = self._initial_tracks(raw, clip)
            except (ValueError, TypeError):
                tracks, reason = [], "短片已准备；原始检测框无效，请手动画框并标注"
            with self._guard, self.library.lock:
                self._bound(asset_id, clip)
                clip.update(status="ready", tracks=tracks, audio_stream_index=stream, reason=reason,
                            clip_url=f"/api/asd-annotations/clips/{clip_id}/media", updated_at=now())
                clip["server"].update(audio_stream_index=stream, preparation=provenance,
                                      files={key: {"relative_path": name, "sha256": file_sha256(directory / name),
                                                   "bytes": (directory / name).stat().st_size}
                                             for key, name in MEDIA_FILES.items()})
                self._revision(directory, clip)
                atomic_json(directory / "clip.json", clip)
        except Exception as exc:
            with self._guard, self.library.lock:
                try:
                    self.library.get_asset(asset_id)
                    clip = read_json(directory / "clip.json")
                    clip.update(status="failed", reason=str(exc)[:800], updated_at=now())
                    atomic_json(directory / "clip.json", clip)
                except (FileNotFoundError, ValueError):
                    pass
        finally:
            with self._guard:
                self._running.pop((asset_id, clip_id), None)

    def save(self, asset_id, clip_id, payload):
        if not isinstance(payload, dict) or set(payload) != {"revision", "note", "tracks"}:
            raise ValueError("保存标注仅接受 revision、note、tracks")
        if type(payload["revision"]) is not int or payload["revision"] < 0:
            raise ValueError("标注版本必须是非负整数")
        if not isinstance(payload["note"], str) or len(payload["note"]) > 8000:
            raise ValueError("片段备注过长或不是文本")
        with self._guard, self.library.lock:
            directory, clip = self._read(asset_id, clip_id)
            if clip["status"] != "ready":
                raise AnnotationConflict("片段尚未准备完成，不能保存人工标注")
            if clip["revision"] != payload["revision"]:
                raise AnnotationConflict("标注已被另一页面更新，请重新加载后保存")
            allowed = {row["id"] for row in self.characters.catalog(asset_id)["profiles"]}
            tracks = checked_tracks(payload["tracks"], clip["frame_count"], allowed)
            timestamp, prior = now(), {row["id"]: row for row in clip["tracks"]}
            for track in tracks:
                old = prior.get(track["id"])
                changed = old is None or old["boxes"] != track["boxes"]
                track.update(reviewed_at=timestamp if track["reviewed"] else None,
                             reviewer={"kind": "human"} if track["reviewed"] else None,
                             provenance={"box_source": "human_edited" if changed else old.get("provenance", {}).get("box_source", "human_edited"),
                                         "boxes_reviewed": track["reviewed"], "labels_source": "human_input",
                                         "model_scores_copied": False})
            clip.update(revision=clip["revision"] + 1, note=payload["note"], tracks=tracks, updated_at=timestamp,
                        reviewed_at=timestamp if any(row["reviewed"] for row in tracks) else None,
                        reviewer={"kind": "human"} if any(row["reviewed"] for row in tracks) else None)
            clip["server"]["last_save"] = {"kind": "human", "at": timestamp, "previous_revision": payload["revision"]}
            self._revision(directory, clip)
            atomic_json(directory / "clip.json", clip)
            return copy.deepcopy(clip)

    def media_path(self, asset_id, clip_id):
        directory, clip = self._read(asset_id, clip_id)
        if clip["status"] != "ready":
            raise FileNotFoundError("标注短片尚未准备完成")
        path = confined_path(directory, "preview.mp4")
        saved = clip.get("server", {}).get("files", {}).get("preview_mp4", {})
        if not path.is_file() or file_sha256(path) != saved.get("sha256"):
            raise ValueError("标注短片内容校验失败")
        return path

    def export(self, asset_id):
        result = self.list(asset_id)
        asset = self.library.get_asset(asset_id)
        result.update(asset_title=asset.get("title", asset_id), movie=asset.get("title", asset_id),
                      exported_at=now(), export_kind="human_asd_annotation_json")
        for clip in result["clips"]:
            clip["media"] = {key: f"clips/{clip['id']}/{name}" for key, name in MEDIA_FILES.items()}
        return result
