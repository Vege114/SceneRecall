"""Bounded, anonymous LR-ASD previews from local source media.

Detection and ASD are isolated in their configured local environments. Results
are review evidence; this module never writes identities or voice exemplars.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

from . import media
from .audio_media import audio_tracks, extract_audio
from .av_sync import sync_availability
from .av_sync_worker import LRASD_PREPROCESS, LRASD_SCORE_DOMAIN, validate_lrasd_bundle
from .library import SCHEMA, atomic_json, confined_path, digest, read_json, safe_id
from .local_resources import LOCAL_MODEL_GATE
from .lrasd_preview_worker import DETECTION_VERSION, TRACKING

PREVIEW_VERSION = "scenerecall-lrasd-preview-v3"


class PreviewBusy(RuntimeError):
    pass


def validate_range(start_ms, end_ms, duration_ms):
    if (type(start_ms) is not int or type(end_ms) is not int or start_ms < 0
            or not 640 <= end_ms - start_ms <= 8000 or end_ms > duration_ms):
        raise ValueError("LR-ASD 预览须为原片范围内 0.64–8 秒的片段")
    return start_ms, start_ms + (end_ms - start_ms) // 40 * 40


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def candidate_matches(pipeline, asset_id, tracks, directory):
    """Use the existing learned CCIP distance and immutable feature cache.

    External cached/candidate references are permitted as proposals. Their
    status/source stays attached and cannot produce a confirmed association.
    """
    from .visual_identity import IdentityUnavailable, _backend, match_candidates
    started = time.monotonic()
    metrics = {"feature_cache_hits": 0, "feature_ms": 0, "feature_extractions": 0}
    output = [{**row, "candidates": [], "character_id": None, "association_status": "candidate",
               "candidate_status": "unmatched"} for row in tracks]
    references = [row for row in pipeline.gallery.list(asset_id)["references"] if row.get("status") != "revoked"]
    if not references or not output:
        return output, {"matching_ms": round((time.monotonic() - started) * 1000)}, "尚无本机角色参考图，保留匿名人物轨迹"
    try:
        settings = pipeline.settings(asset_id)
        backend = pipeline.backend or _backend(settings["model"])
        version = backend.model_version
        usable, features = [], []
        for row in references:
            path = Path(row["path"])
            if not path.is_file() or file_sha256(path) != row.get("sha256"):
                continue
            detection_key = digest([row["sha256"], version, "preview-reference-single-person-v1"])
            location = pipeline.library.root / "cache" / "identity-reference-detections" / f"{detection_key}.json"
            detection = read_json(location)
            if detection is None:
                detection = {"detections": backend.detect(path)}
                atomic_json(location, detection)
            if len(detection["detections"]) != 1:
                continue
            usable.append(row)
            features.append(pipeline._feature(path, version, backend, metrics))
        by_id = {row["id"]: row for row in usable}
        for track in output:
            path = confined_path(directory, track["thumbnail_file"])
            # Recheck the crop's single-person constraint with the identity
            # model's own detector before computing an identity feature.
            if len(backend.detect(path)) != 1:
                track["candidate_status"] = "crop_not_single_person"
                continue
            if not features:
                continue
            query = pipeline._feature(path, version, backend, metrics)
            ranked = match_candidates(backend.distances(query, features), usable,
                                      top_k=settings.get("top_k", 5), model_version=version)
            track["candidates"] = [{**candidate, "association_status": "candidate", "source_verified": False,
                                     "reference_sources": [{"id": identity,
                                                            "kind": by_id[identity].get("source", {}).get("kind"),
                                                            "status": by_id[identity].get("status"),
                                                            "human_confirmed": by_id[identity].get("status") == "confirmed",
                                                            "source_verified": False}
                                                           for identity in candidate["reference_ids"]]}
                                    for candidate in ranked["candidates"]]
            track["candidate_status"] = "unverified_reference_match" if track["candidates"] else "unmatched"
            track["candidate_model_version"] = version
        reason = "CCIP 仅按本机参考图提出角色候选；外部资料图片的身份关联未经人工确认"
    except (IdentityUnavailable, OSError, ValueError, KeyError, RuntimeError) as exc:
        reason = f"可选视觉身份匹配不可用（{type(exc).__name__}），LR-ASD 人物概率仍可独立复核"
    return output, {"matching_ms": round((time.monotonic() - started) * 1000), **metrics}, reason


class LrasdPreviewService:
    def __init__(self, audio_service):
        self.audio = audio_service
        self.library = audio_service.library
        self._guard = threading.RLock()
        self._running = {}

    def _root(self, asset_id):
        return self.audio._directory(asset_id) / "lrasd-previews"

    def latest(self, asset_id):
        with self._guard:
            path = self._root(asset_id) / "latest.json"
            value = read_json(path, {"status": "not_started", "tracks": [], "timings": {},
                                     "reason": "尚未运行短片段 LR-ASD 预览", "cache_hit": False})
            current = self._running.get(asset_id)
            if value.get("status") == "running" and not (current and current[1].is_alive()):
                value.update(status="failed", reason="上次预览随应用进程中断，可重新运行该时间段", tracks=[])
                atomic_json(path, value)
            return value

    def thumbnail(self, asset_id, key, track_id):
        directory = self._root(asset_id) / safe_id(key)
        result = read_json(directory / "result.json", {})
        if not any(row["id"] == track_id for row in result.get("tracks", [])):
            raise FileNotFoundError("预览人物缩略图不存在")
        return confined_path(directory, f"{safe_id(track_id)}.png")

    def clip(self, asset_id, key):
        directory = self._root(asset_id) / safe_id(key)
        result = read_json(directory / "result.json", {})
        path = confined_path(directory, "preview.mp4")
        if (result.get("status") not in {"review_required", "no_tracks"}
                or result.get("preview_version") != PREVIEW_VERSION or not result.get("clip_url") or not path.is_file()):
            raise FileNotFoundError("已完成的 LR-ASD 预览短片不存在")
        return path

    def start(self, asset_id, start_ms, end_ms):
        asset = self.library.get_asset(asset_id)
        start, end = validate_range(start_ms, end_ms, asset["duration_ms"])
        config = self.audio.settings(asset_id)
        if not config.get("enabled") or not config.get("sync_enabled") or config.get("sync_model_type") != "lrasd":
            raise ValueError("请先保存已启用的 LR-ASD 音频及同步设置")
        available = sync_availability(config)
        if not available["available"]:
            raise ValueError(available["reason"])
        detector = Path(config.get("sync_detector_path") or "/nonexistent").expanduser()
        if not detector.is_file() or detector.suffix.lower() != ".onnx":
            raise ValueError("请配置本机 anime person ONNX 检测权重 sync_detector_path")
        visual_python = shutil.which(os.environ.get("SCENERECALL_VISUAL_PYTHON", sys.executable))
        if not visual_python or not Path(visual_python).is_file():
            raise ValueError("SCENERECALL_VISUAL_PYTHON 不是本机可执行 Python")
        source = self.library.require_source(asset_id)
        if not Path(source).is_file():
            raise ValueError("LR-ASD 预览仅接受已关联的本机原视频")
        bundle, _ = validate_lrasd_bundle(Path(config["sync_checkpoint_path"]).expanduser())
        from .visual_identity import VisualIdentityPipeline
        pipeline = VisualIdentityPipeline(self.library)
        references = pipeline.gallery.list(asset_id)
        reference_identity = [(row["id"], row.get("version"), row["character_id"], row.get("sha256"), row.get("status"))
                              for row in references["references"]]
        identity = {"schema_version": SCHEMA, "preview_version": PREVIEW_VERSION,
                    "source_fingerprint": asset["fingerprint"],
                    "source_stat": [Path(source).stat().st_size, Path(source).stat().st_mtime_ns],
                    "checkpoint_sha256": bundle["checkpoint_sha256"], "detector_sha256": file_sha256(detector),
                    "start_ms": start, "end_ms": end, "audio_stream_index": config.get("stream_index"),
                    "preprocessing": LRASD_PREPROCESS, "tracking_version": DETECTION_VERSION, "tracking": TRACKING,
                    "reference_gallery": reference_identity, "visual_settings": pipeline.settings(asset_id),
                    "device": config["device"]}
        key = digest(identity)
        with self._guard:
            current = self._running.get(asset_id)
            if current and current[1].is_alive():
                if current[0] == key:
                    return self.latest(asset_id)
                raise PreviewBusy("该素材已有 LR-ASD 预览运行中，请等待完成")
            root, directory = self._root(asset_id), self._root(asset_id) / key
            cached = read_json(directory / "result.json")
            if cached and cached.get("status") in {"review_required", "no_tracks"}:
                value = {**cached, "cache_hit": True}
                atomic_json(root / "latest.json", value)
                return value
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "identity.json", identity)
            value = {"schema_version": SCHEMA, "preview_version": PREVIEW_VERSION,
                     "job_id": key, "asset_id": asset_id, "status": "running",
                     "source_fingerprint": asset["fingerprint"],
                     "start_ms": start, "end_ms": end, "requested_end_ms": end_ms, "tracks": [], "timings": {},
                     "reason": "正在解码原片、逐帧检测人物并计算 LR-ASD 概率", "cache_hit": False}
            atomic_json(root / "latest.json", value)
            thread = threading.Thread(target=self._background, args=(asset_id, directory, value, identity, config,
                                                                     str(source), str(detector.resolve()), visual_python),
                                      daemon=True, name=f"lrasd-preview-{asset_id}")
            self._running[asset_id] = (key, thread)
            thread.start()
            return value

    def _worker(self, python, module, directory, *, timeout):
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                           OMP_NUM_THREADS="2")
        with LOCAL_MODEL_GATE:
            process = subprocess.run([python, str(Path(__file__).with_name(module)), str(directory)],
                                     env=environment, capture_output=True, timeout=timeout, check=False)
        if process.returncode:
            (directory / (module + ".stderr.log")).write_bytes(process.stderr[-32000:])
            raise RuntimeError(f"本机 {module} 加载或推理失败；检查独立环境与本机权重")
        value = json.loads(process.stdout)
        if not isinstance(value, dict):
            raise ValueError("本机模型未返回对象")
        return value

    def _decode(self, source, directory, start, end, config):
        metadata = media.probe(Path(source))
        tracks = audio_tracks(metadata)
        stream = config.get("stream_index")
        if stream is None:
            stream = next((row["index"] for row in tracks if row["default"]), tracks[0]["index"] if tracks else None)
        if stream is None:
            raise ValueError("原片没有可用音轨")
        deadline = time.monotonic() + 120
        def checkpoint():
            self.library.require_source(config["_asset_id"])
            if time.monotonic() > deadline:
                raise TimeoutError("短片段音频解码超时，请重试或缩短范围")
        extract_audio(Path(source), directory / "audio.wav", stream, start, end, metadata,
                      checkpoint=checkpoint)
        origin = float(metadata.get("start_time_seconds", metadata.get("start_time_ms", 0) / 1000))
        filters = (f"setpts=PTS-({origin:.9f})/TB,trim=start={start / 1000:.6f}:end={end / 1000:.6f},"
                   f"setpts=PTS-({start / 1000:.6f})/TB,fps=fps=25:start_time=0,scale=480:270:flags=bilinear")
        # Input seek is only a decode optimization. copyts and explicit absolute
        # timestamp seeking retain the same origin as extract_audio, including
        # delayed audio streams and nonzero container starts.
        seek = max(0., start / 1000 - 1.) + origin
        media._run([media._tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-threads", "1", "-copyts",
                    "-seek_timestamp", "1", "-ss", f"{seek:.9f}", "-i", source,
                    "-map", f"0:{metadata['video_stream_index']}", "-an", "-sn", "-dn", "-vf", filters,
                    "-pix_fmt", "rgb24", "-frames:v", str((end - start) // 40), "-f", "rawvideo",
                    str(directory / "frames.rgb")], timeout=120)
        if (directory / "frames.rgb").stat().st_size != (end - start) // 40 * 480 * 270 * 3:
            raise ValueError("原片未提供请求时间网格的完整连续画面")
        return stream

    def _make_clip(self, directory, start_ms, end_ms):
        """Mux the exact inference inputs into a small browser-playable clip."""
        frames = (end_ms - start_ms) // 40
        if not 16 <= frames <= 200 or (end_ms - start_ms) % 40:
            raise ValueError("预览短片时长不在连续 25fps 的允许范围内")
        if (directory / "frames.rgb").stat().st_size != frames * 480 * 270 * 3:
            raise ValueError("预览短片画面与推理时间网格不一致")
        with wave.open(str(directory / "audio.wav"), "rb") as source:
            if (source.getnchannels(), source.getframerate(), source.getsampwidth(), source.getnframes()) != (1, 16000, 2, frames * 640):
                raise ValueError("预览短片音频与推理时间网格不一致")
        temporary = directory / "preview.partial.mp4"
        try:
            media._run([media._tool("ffmpeg"), "-nostdin", "-v", "error", "-y",
                        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "480x270", "-framerate", "25",
                        "-i", str(directory / "frames.rgb"), "-i", str(directory / "audio.wav"),
                        "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(frames),
                        "-t", f"{(end_ms - start_ms) / 1000:.6f}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-threads:v", "1", "-c:a", "aac", "-b:a", "96k",
                        "-ac", "1", "-ar", "16000", "-movflags", "+faststart", str(temporary)], timeout=30)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ValueError("编码器没有生成有效短片文件")
            temporary.replace(directory / "preview.mp4")
        except (OSError, ValueError) as exc:
            raise RuntimeError("LR-ASD 预览短片编码失败；未发布完成结果") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def _background(self, asset_id, directory, initial, identity, config, source, detector, visual_python):
        started = time.monotonic()
        value = dict(initial)
        try:
            self.library.require_source(asset_id)
            current_bundle, _ = validate_lrasd_bundle(Path(config["sync_checkpoint_path"]).expanduser())
            if (current_bundle["checkpoint_sha256"] != identity["checkpoint_sha256"]
                    or file_sha256(detector) != identity["detector_sha256"]):
                raise ValueError("预览启动后本机模型文件已变化，请重新运行")
            decode = time.monotonic()
            stream = self._decode(source, directory, initial["start_ms"], initial["end_ms"], {**config, "_asset_id": asset_id})
            timings = {"decode_ms": round((time.monotonic() - decode) * 1000)}
            encode = time.monotonic()
            self._make_clip(directory, initial["start_ms"], initial["end_ms"])
            timings["clip_encode_ms"] = round((time.monotonic() - encode) * 1000)
            value["clip_url"] = f"/api/assets/{asset_id}/audio/lrasd-previews/{initial['job_id']}/clip"
            atomic_json(directory / "preview-request.json", {"start_ms": initial["start_ms"], "end_ms": initial["end_ms"],
                                                              "detector_path": detector})
            detected = self._worker(visual_python, "lrasd_preview_worker.py", directory, timeout=180)
            timings.update(detected["timings"])
            tracks = detected["tracks"]
            value.update(status="no_tracks", tracks=[], reason="片段中没有满足逐帧检测、连续性及镜头边界要求的至少 16 帧人物轨迹")
            if tracks:
                self.library.require_source(asset_id)
                model_start = time.monotonic()
                request = {"repository": str(Path(config["sync_repository_path"]).expanduser().resolve()),
                           "checkpoint": str(Path(config["sync_checkpoint_path"]).expanduser().resolve()),
                           "device": config["device"], "start_ms": initial["start_ms"], "end_ms": initial["end_ms"],
                           "tracks": tracks, "preview_only": True, "model_type": "lrasd",
                           "model_version": "lrasd:" + identity["checkpoint_sha256"], "preprocessing": LRASD_PREPROCESS}
                atomic_json(directory / "request.json", request)
                scored = self._worker(str(Path(config["sync_python_path"]).expanduser()), "av_sync_worker.py", directory, timeout=180)
                if scored.get("status") != "review_required" or len(scored.get("results", [])) != len(tracks):
                    raise ValueError("LR-ASD 预览轨迹数量或状态无效")
                timings["asd_ms"] = round((time.monotonic() - model_start) * 1000)
                by_id = {row["track_id"]: row for row in scored["results"]}
                for row in tracks:
                    row.update(by_id[row["id"]])
                    row["thumbnail_url"] = (f"/api/assets/{asset_id}/audio/lrasd-previews/{initial['job_id']}/"
                                             f"tracks/{row['id']}/thumbnail")
                    row["character_id"] = None
                from .visual_identity import VisualIdentityPipeline
                tracks, match_timings, match_reason = candidate_matches(VisualIdentityPipeline(self.library), asset_id, tracks, directory)
                timings.update(match_timings)
                # Boxes remain in private request/detection caches. The UI gets
                # source-timed probabilities, a crop and reversible candidates.
                public_tracks = [{key: item for key, item in row.items() if key not in {"boxes", "thumbnail_file"}} for row in tracks]
                value.update(status="review_required", tracks=public_tracks, reason=match_reason,
                             model_version=scored["model_version"], domain_validation=scored.get("domain_validation"),
                             calibration=scored.get("calibration"))
            value.update(timings={**timings, "total_ms": round((time.monotonic() - started) * 1000)},
                         checkpoint_sha256=identity["checkpoint_sha256"], detector_sha256=identity["detector_sha256"],
                         preprocessing=LRASD_PREPROCESS, score_domain=LRASD_SCORE_DOMAIN,
                         audio_stream_index=stream, tracking=detected["tracking"], cut_count=detected["cut_count"])
            self.library.require_source(asset_id)
            current_bundle, _ = validate_lrasd_bundle(Path(config["sync_checkpoint_path"]).expanduser())
            if (current_bundle["checkpoint_sha256"] != identity["checkpoint_sha256"]
                    or file_sha256(detector) != identity["detector_sha256"]):
                raise ValueError("推理期间本机模型文件已变化，结果未发布")
            atomic_json(directory / "result.json", value)
            atomic_json(self._root(asset_id) / "latest.json", value)
        except Exception as exc:
            value.update(status="unavailable" if isinstance(exc, (OSError, subprocess.TimeoutExpired, RuntimeError)) else "failed",
                         tracks=[], reason=str(exc)[:500], timings={"total_ms": round((time.monotonic() - started) * 1000)})
            # Deletion during execution must never recreate a deleted asset.
            try:
                self.library.get_asset(asset_id)
                atomic_json(directory / "failure.json", value)
                atomic_json(self._root(asset_id) / "latest.json", value)
            except (FileNotFoundError, ValueError):
                pass
        finally:
            with self._guard:
                current = self._running.get(asset_id)
                if current and current[0] == initial["job_id"]:
                    self._running.pop(asset_id, None)
