"""Auditable local speaker recognition, separate from visible-character evidence.

Expensive PCM, diarization and embeddings are immutable caches. Voice-bank and
human identity mappings are projections, so renames/corrections never re-encode.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import math
import shutil
import threading
import time
import wave
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from . import media
from .audio_adapters import (AudioUnavailable, ECAPAEmbedder, FasterWhisperASR, PyannoteDiarizer,
                             availability, cosine_distance, normalized)
from .audio_media import PREPROCESS_VERSION, audio_tracks, crop_pcm, extract_audio, quality
from .library import atomic_json, atomic_jsonl, confined_path, digest, now, read_json, read_jsonl, safe_id, uid

_RUN_LOCKS: dict[str, threading.Lock] = {}
_RUN_LOCKS_GUARD = threading.Lock()


class AudioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    stream_index: int | None = Field(default=None, ge=0, strict=True)
    language: str = Field(default="und", min_length=2, max_length=32)
    dub_version: str = Field(default="original", min_length=1, max_length=200)
    diarization_model_path: str = ""
    embedding_model_path: str = ""
    asr_enabled: bool = False
    asr_model_path: str = ""
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    calibration_path: str = ""
    top_k: int = Field(default=5, ge=1, le=50)
    max_duration_ms: int = Field(default=60000, ge=1000, le=14400000, strict=True)
    sync_enabled: bool = False
    sync_repository_path: str = ""
    sync_checkpoint_path: str = ""
    sync_animation_verified: bool = False
    sync_license_acknowledged: bool = False


def _overlap(left: dict, right: dict) -> bool:
    return left["start_ms"] < right["end_ms"] and left["end_ms"] > right["start_ms"]


def validate_turns(turns: list[dict], duration_ms: int) -> list[dict]:
    result = []
    for index, turn in enumerate(turns):
        start, end = turn["start_ms"], turn["end_ms"]
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
                or not isinstance(end, int) or end <= start or start < 0 or end > duration_ms + 1):
            raise ValueError("说话人模型返回越界或无效时间段")
        result.append({"start_ms": start, "end_ms": min(duration_ms, end),
                       "speaker": str(turn["speaker"]) if turn.get("speaker") is not None else f"unknown_{index}"})
    result.sort(key=lambda row: (row["start_ms"], row["end_ms"], row["speaker"]))
    for row in result:
        row["overlap"] = any(other["speaker"] != row["speaker"] and _overlap(row, other) for other in result)
    return result


def match_voice(vector: list[float], samples: list[dict], *, top_k: int = 5,
                calibration: dict | None = None, blocked: bool = False) -> dict:
    """Scores are distances, not confidence. No uncalibrated auto-confirmation."""
    groups = {}
    for sample in samples:
        groups.setdefault(sample["character_id"], []).append(
            (cosine_distance(vector, sample["embedding"]), sample["id"]))
    ranked = []
    for character_id, matches in groups.items():
        best = sorted(matches)[:3]
        ranked.append({"character_id": character_id, "distance": sum(row[0] for row in best) / len(best),
                       "sample_ids": [row[1] for row in best]})
    ranked.sort(key=lambda row: (row["distance"], row["character_id"]))
    margin = ranked[1]["distance"] - ranked[0]["distance"] if len(ranked) > 1 else None
    status = "candidate" if ranked else "unknown"
    if calibration and ranked:
        if ranked[0]["distance"] > calibration["candidate_max_distance"]:
            status = "unknown"
        elif (not blocked and ranked[0]["distance"] <= calibration["confirmed_max_distance"]
              and margin is not None and margin >= calibration["min_margin"]):
            status = "confirmed"
    return {"association_status": status, "character_id": ranked[0]["character_id"] if status == "confirmed" else None,
            "candidate_character_id": ranked[0]["character_id"] if status == "candidate" else None,
            "association_source": "voice_bank" if ranked else "unmatched", "candidates": ranked[:top_k],
            "distance": ranked[0]["distance"] if ranked else None, "candidate_margin": margin,
            "calibration_id": calibration.get("id") if calibration else None,
            "needs_identity_review": status != "confirmed"}


class AudioIdentityService:
    def __init__(self, library, characters=None, *, diarizer=None, embedder=None, asr=None):
        self.library = library
        if characters is None:
            from .characters import CharacterStore
            characters = CharacterStore(library)
        self.characters = characters
        self._diarizer, self._embedder, self._asr = diarizer, embedder, asr

    def _directory(self, asset_id: str) -> Path:
        self.library.get_asset(asset_id)
        return self.library.asset_dir(asset_id) / "audio"

    def settings(self, asset_id: str) -> dict:
        return AudioSettings.model_validate(read_json(self._directory(asset_id) / "settings.json", {})).model_dump()

    def configure(self, asset_id: str, payload: dict) -> dict:
        with self.library.lock:
            settings = AudioSettings.model_validate({**self.settings(asset_id), **payload}).model_dump()
            if settings["stream_index"] is not None and self.library.get_asset(asset_id).get("source_available", True):
                tracks = self.audio_tracks(asset_id)["tracks"]
                if settings["stream_index"] not in {track["index"] for track in tracks}:
                    raise ValueError("音轨选择无效")
            atomic_json(self._directory(asset_id) / "settings.json", settings)
        return settings

    def availability(self, asset_id: str) -> dict:
        config = self.settings(asset_id)
        from .av_sync import sync_availability
        items = {"diarization": availability("pyannote.audio", config["diarization_model_path"]),
                 "embedding": availability("speechbrain", config["embedding_model_path"]),
                 "asr": availability("faster_whisper", config["asr_model_path"]), "sync": sync_availability(config)}
        items["reasons"] = [item["reason"] for item in items.values() if not item["available"]]
        return items

    def audio_tracks(self, asset_id: str) -> dict:
        metadata = media.probe(self.library.require_source(asset_id))
        tracks = audio_tracks(metadata)
        recommended = next((track["index"] for track in tracks if track["default"]), tracks[0]["index"] if tracks else None)
        return {"tracks": tracks, "recommended_stream_index": recommended, "duration_ms": metadata["duration_ms"]}

    def _bank(self, asset_id: str) -> tuple[Path, dict]:
        catalog = self._catalog(asset_id)
        return self.library.root / "works" / safe_id(catalog["scope_id"]) / "voice-samples.jsonl", catalog

    def _catalog(self, asset_id: str) -> dict:
        """Identity-only snapshot: never recurse through scene/appearance projection."""
        if not hasattr(self.characters, "_read"):
            return self.characters.catalog(asset_id)  # Small injected test stores.
        from .characters import scope_id
        with self.library.lock:
            asset = self.library.get_asset(asset_id)
            scope = scope_id(asset)
            registry = self.characters._read(scope)
            policy = self.characters.cast_policy(registry, asset.get("anime_profile"))
            profiles = [copy.deepcopy(profile) for profile in registry["profiles"].values()
                        if self.characters.profile_allowed(profile, policy)]
        return {"scope_id": scope, "profiles": profiles, "registry": registry}

    def _resolve(self, catalog: dict, identity: str | None) -> str | None:
        if catalog.get("registry") is not None:
            return self.characters._resolve(catalog["registry"], identity)
        return identity

    def samples(self, asset_id: str, character_id: str | None = None) -> list[dict]:
        path, catalog = self._bank(asset_id)
        allowed = {profile["id"] for profile in catalog["profiles"]}
        requested_character = character_id is not None
        character_id = self._resolve(catalog, character_id)
        if requested_character and character_id not in allowed:
            raise ValueError("角色不在当前作品允许的档案中")
        rows = []
        for original in read_jsonl(path):
            resolved = self._resolve(catalog, original["character_id"])
            if resolved not in allowed or character_id is not None and resolved != character_id:
                continue
            if original["source_kind"] == "original_clip":
                from .characters import scope_id
                try:
                    source = self.library.get_asset(original["provenance"]["asset_id"])
                    if hasattr(self.characters, "_read") and scope_id(source) != catalog["scope_id"]:
                        continue
                except (FileNotFoundError, ValueError):
                    continue
            rows.append({**original, "character_id": resolved})
        return rows

    def _sample_file(self, asset_id: str, sample: dict) -> Path:
        path, _ = self._bank(asset_id)
        return confined_path(path.parent, sample["audio_ref"])

    def add_sample(self, asset_id: str, character_id: str, payload: dict) -> dict:
        """Original proposals never become trusted merely because someone is visible."""
        path, catalog = self._bank(asset_id)
        if character_id not in {profile["id"] for profile in catalog["profiles"]}:
            raise ValueError("角色不在当前作品允许的档案中")
        config = self.settings(asset_id)
        language, dub = payload.get("language", config["language"]), payload.get("dub_version", config["dub_version"])
        if not language or not dub:
            raise ValueError("声音样本须注明原声语言及配音版本")
        identity = uid("voice")
        destination = path.parent / "voice-audio" / f"{identity}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_kind = payload.get("source_kind", "original_clip")
        if source_kind == "original_clip":
            record = self.record(asset_id, payload.get("record_id", ""))
            if record["overlap"]:
                raise ValueError("重叠发言不能直接作为单角色可信声音样本")
            if (language, dub) != (record["language"], record["dub_version"]):
                raise ValueError("原片样本语言与配音版本必须与音频记录相同")
            shutil.copyfile(self.clip_path(asset_id, record["id"]), destination)
            provenance = {"asset_id": asset_id, "record_id": record["id"], "start_ms": record["start_ms"],
                          "end_ms": record["end_ms"], "audio_stream_index": record["audio_stream_index"],
                          "frame_ids": record.get("frame_ids", []), "proposal_only": not payload.get("confirmed", False)}
        elif source_kind == "external_licensed":
            if not str(payload.get("rights_note", "")).strip():
                raise ValueError("外部声音样本须注明使用权利及出处")
            try:
                raw = base64.b64decode(payload.get("audio_base64", ""), validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("声音文件编码无效") from exc
            if not raw or len(raw) > 10 * 1024 * 1024:
                raise ValueError("声音样本不能为空且不能超过 10 MB")
            # Uploaded files are WAV only; no executable codecs or remote URLs.
            destination.write_bytes(raw)
            provenance = {"filename": Path(payload.get("filename", "sample.wav")).name,
                          "rights_note": str(payload["rights_note"])[:2000], "external": True}
        else:
            raise ValueError("声音样本来源无效")
        try:
            info = quality(destination)
        except (ValueError, wave.Error, EOFError) as exc:
            destination.unlink(missing_ok=True)
            raise ValueError("请提供 16kHz 单声道 PCM16 WAV 声音样本") from exc
        if not info["usable"]:
            destination.unlink(missing_ok=True)
            raise ValueError("声音样本质量不足: " + ", ".join(info["reasons"]))
        row = {"id": identity, "character_id": character_id, "scope_id": catalog["scope_id"],
               "language": language, "dub_version": dub, "source_kind": source_kind,
               "status": "confirmed" if payload.get("confirmed") is True else "proposed",
               "confirmation_source": "human" if payload.get("confirmed") is True else None,
               "quality": info, "provenance": provenance, "version": 1, "created_at": now(),
               "audio_ref": destination.relative_to(path.parent).as_posix(),
               "content_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
        with self.library.lock:
            rows = read_jsonl(path)
            rows.append(row)
            atomic_jsonl(path, rows)
            self._audit(path.parent / "voice-audit.jsonl", {"action": "add", "sample_id": identity, "after": row})
        return row

    def update_sample(self, asset_id: str, character_id: str, sample_id: str, status: str, note: str = "") -> dict:
        if status not in {"confirmed", "proposed", "revoked"}:
            raise ValueError("声音样本状态无效")
        available = self.samples(asset_id, character_id)
        if sample_id not in {row["id"] for row in available}:
            raise ValueError("声音样本不存在")
        before_records = self._scope_record_identities(asset_id)
        path, catalog = self._bank(asset_id)
        with self.library.lock:
            rows = read_jsonl(path)
            row = next((row for row in rows if row["id"] == sample_id
                        and self._resolve(catalog, row["character_id"]) == self._resolve(catalog, character_id)), None)
            if row is None:
                raise ValueError("声音样本不存在")
            before = copy.deepcopy(row)
            row.update(status=status, version=row["version"] + 1, confirmation_source="human", updated_at=now())
            atomic_jsonl(path, rows)
            self._audit(path.parent / "voice-audit.jsonl", {"action": "status", "sample_id": sample_id,
                                                         "before": before, "after": row, "note": note[:2000]})
            # Projection recomputes against the current trusted bank; acoustic caches survive.
        self._invalidate_changed(before_records)
        return row

    def _scope_record_identities(self, asset_id: str) -> dict:
        # Called only for explicit trust edits, not on the read hot path.
        assets = self.characters.catalog(asset_id).get("assets", [{"id": asset_id}])
        return {asset["id"]: {row["id"]: (row["character_id"], row["association_status"])
                              for row in self.records(asset["id"])} for asset in assets}

    def _invalidate_changed(self, before: dict) -> None:
        from .scenes import SceneStore
        for asset_id, identities in before.items():
            current_rows = self.records(asset_id)
            current = {row["id"]: (row["character_id"], row["association_status"]) for row in current_rows}
            changed = [key for key, value in identities.items() if current.get(key) != value
                       and (value[0] is not None or (current.get(key) or (None,))[0] is not None)]
            if changed and hasattr(SceneStore, "invalidate_identity_inferences"):
                changed += [row["source_record_id"] for row in current_rows
                            if row["id"] in changed and row.get("source_record_id")]
                SceneStore(self.library).invalidate_identity_inferences(asset_id, audio_ids=changed)

    @staticmethod
    def _audit(path: Path, event: dict) -> None:
        rows = read_jsonl(path)
        atomic_jsonl(path, [*rows, {"id": uid("audit"), "at": now(), **event}])

    def _feature(self, path: Path, embedder, cache: Path, *, checkpoint=None) -> tuple[list[float], bool]:
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        key = digest({"sha256": content_hash, "model": embedder.version, "preprocess": PREPROCESS_VERSION})
        target = cache / f"{key}.json"
        saved = read_json(target)
        if saved:
            return saved["embedding"], True
        if checkpoint:
            checkpoint()
        vector = normalized(embedder.embed(path))
        atomic_json(target, {"embedding": vector, "model_version": embedder.version, "content_sha256": content_hash,
                             "preprocessing": PREPROCESS_VERSION})
        if checkpoint:
            checkpoint()
        return vector, False

    def _calibration(self, config: dict, version: str, scope: str) -> dict | None:
        if not config["calibration_path"]:
            return None
        value = read_json(Path(config["calibration_path"]).expanduser())
        if not isinstance(value, dict) or not value.get("id") or not value.get("evaluation_set_id"):
            raise ValueError("声音阈值必须来自有标注评测集的校准文件")
        if (value.get("model_version"), value.get("scope_id"), value.get("language"), value.get("dub_version")) != (
                version, scope, config["language"], config["dub_version"]):
            raise ValueError("声音阈值校准与当前模型、作品、语言或配音版本不一致")
        for field in ("candidate_max_distance", "confirmed_max_distance", "min_margin"):
            if not isinstance(value.get(field), (int, float)) or not math.isfinite(value[field]) or not 0 <= value[field] <= 2:
                raise ValueError("声音阈值校准数值无效")
        if value["confirmed_max_distance"] > value["candidate_max_distance"]:
            raise ValueError("声音确认阈值须严于候选阈值")
        return value

    def run(self, asset_id: str, start_ms: int = 0, end_ms: int | None = None, *, checkpoint=None,
            subtitles: list[dict] | None = None, frames: list[dict] | None = None,
            config_override: dict | None = None) -> dict:
        key = str(self._directory(asset_id).resolve())
        with _RUN_LOCKS_GUARD:
            lock = _RUN_LOCKS.setdefault(key, threading.Lock())
        while not lock.acquire(timeout=.1):
            if checkpoint:
                checkpoint()
        try:
            return self._run(asset_id, start_ms, end_ms, checkpoint=checkpoint, subtitles=subtitles,
                             frames=frames, config_override=config_override)
        finally:
            lock.release()

    def _run(self, asset_id: str, start_ms: int = 0, end_ms: int | None = None, *, checkpoint=None,
             subtitles: list[dict] | None = None, frames: list[dict] | None = None,
             config_override: dict | None = None) -> dict:
        started = time.perf_counter()
        config = AudioSettings.model_validate(config_override or self.settings(asset_id)).model_dump()
        root = self._directory(asset_id)
        result = {"status": "disabled", "records": [], "timings": {}, "degraded": [], "cache_hits": 0}
        if not config["enabled"]:
            result["degraded"] = ["音频角色识别未启用"]
            result["timings"]["total_s"] = time.perf_counter() - started
            return result
        if checkpoint:
            checkpoint()
        source = self.library.require_source(asset_id)
        metadata = media.probe(source)
        tracks = audio_tracks(metadata)
        if not tracks:
            result.update(status="no_audio", degraded=["素材没有音轨"])
            result["timings"]["total_s"] = time.perf_counter() - started
            atomic_json(root / "status.json", result)
            return result
        stream = config["stream_index"]
        if stream is None:
            stream = next((row["index"] for row in tracks if row["default"]), tracks[0]["index"])
        if stream not in {row["index"] for row in tracks}:
            raise ValueError("所选音轨不存在")
        end_ms = metadata["duration_ms"] if end_ms is None else end_ms
        if start_ms < 0 or end_ms <= start_ms or end_ms > metadata["duration_ms"]:
            raise ValueError("音频处理范围无效")
        if end_ms - start_ms > config["max_duration_ms"]:
            result.update(status="budget_exceeded", degraded=["音频范围超过本地处理时长预算；请缩短范围或明确提高音频预算"])
            result["timings"]["total_s"] = time.perf_counter() - started
            atomic_json(root / "status.json", result)
            return result
        diarizer = self._diarizer or PyannoteDiarizer(config["diarization_model_path"], config["device"])
        embedder = self._embedder or ECAPAEmbedder(config["embedding_model_path"], config["device"])
        calibration = self._calibration(config, embedder.version, self._catalog(asset_id)["scope_id"])
        input_cues = self.library.subtitles(asset_id) if subtitles is None else subtitles
        stat = source.stat()
        # The imported asset fingerprint plus current stat prevents reuse after replacement.
        source_key = {"asset_id": asset_id, "fingerprint": self.library.get_asset(asset_id).get("fingerprint"),
                      "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                      "stream_index": stream, "start_ms": start_ms, "end_ms": end_ms,
                      "origin_ms": metadata.get("start_time_ms", 0), "preprocessing": PREPROCESS_VERSION}
        extraction_key = digest(source_key)
        directory = root / "cache" / extraction_key
        directory.mkdir(parents=True, exist_ok=True)
        extracted = directory / "audio.wav"
        stamp = time.perf_counter()
        if not extracted.is_file():
            extract_audio(source, extracted, stream, start_ms, end_ms, metadata, checkpoint=checkpoint)
            atomic_json(directory / "extraction.json", source_key)
        else:
            result["cache_hits"] += 1
        result["timings"]["extraction_s"] = time.perf_counter() - stamp
        run_key = digest({**source_key, "diarizer": diarizer.version, "embedding": embedder.version,
                          "config": config, "subtitles": input_cues,
                          "calibration": calibration,
                          "frame_refs": [{"id": frame["id"], "at_ms": frame["at_ms"]} for frame in (frames or [])]})
        run_dir = root / "runs" / run_key
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / "calibration.json", calibration)
        result["run_id"] = run_key
        turns_path = directory / f"turns-{digest(diarizer.version)}.json"
        stamp = time.perf_counter()
        try:
            turns = read_json(turns_path)
            if turns is None:
                turns = validate_turns(diarizer.diarize(extracted), end_ms - start_ms)
                atomic_json(turns_path, turns)
            else:
                turns = validate_turns(turns, end_ms - start_ms)
                result["cache_hits"] += 1
        except (AudioUnavailable, ImportError, OSError, RuntimeError) as exc:
            result.update(status="unavailable", degraded=[str(exc) if isinstance(exc, AudioUnavailable)
                                                          else "自动说话人模型加载失败，请检查本机依赖与完整权重"])
            result["timings"]["diarization_s"] = time.perf_counter() - stamp
            result["timings"]["total_s"] = time.perf_counter() - started
            atomic_json(root / "status.json", result)
            return result
        result["timings"]["diarization_s"] = time.perf_counter() - stamp
        if checkpoint:
            checkpoint()
        cues = input_cues
        cues = [row for row in cues if row.get("text", "").strip() and not row.get("uncertain")
                and row.get("review_status") not in {"unreadable", "rejected", "uncertain"}
                and _overlap(row, {"start_ms": start_ms, "end_ms": end_ms})]
        stamp = time.perf_counter()
        if not cues and config["asr_enabled"]:
            asr = self._asr or FasterWhisperASR(config["asr_model_path"], config["device"])
            asr_path = directory / f"asr-{digest([asr.version, config['language']])}.json"
            try:
                cues = read_json(asr_path)
                if cues is None:
                    cues = [{**cue, "start_ms": cue["start_ms"] + start_ms, "end_ms": cue["end_ms"] + start_ms,
                             "id": f"asr_{digest([run_key, index])}"}
                            for index, cue in enumerate(asr.transcribe(extracted, config["language"]))]
                    atomic_json(asr_path, cues)
                else:
                    result["cache_hits"] += 1
            except (AudioUnavailable, ImportError, OSError, RuntimeError):
                cues = []
                result["degraded"].append("无可靠字幕且本机 ASR 不可用；保留声音时间证据，不生成台词")
        elif not cues:
            result["degraded"].append("无可靠字幕且 ASR 未启用；声音记录没有转写文本")
        result["timings"]["transcription_s"] = time.perf_counter() - stamp
        raw_records = []
        stamp = time.perf_counter()
        embedding_unavailable = False
        for index, turn in enumerate(turns):
            if checkpoint:
                checkpoint()
            # Human identity attaches to acoustic evidence, not execution knobs,
            # translated subtitle revisions, feature models or sync settings.
            identity = f"audio_{digest([extraction_key, diarizer.version, index, turn,
                                       config['language'], config['dub_version']])}"
            clip = directory / "clips" / f"{identity}.wav"
            if not clip.is_file():
                crop_pcm(extracted, clip, turn["start_ms"], turn["end_ms"])
            info = quality(clip)
            vector = None
            if info["usable"] and not embedding_unavailable:
                try:
                    vector, hit = self._feature(clip, embedder, root / "features", checkpoint=checkpoint)
                    result["cache_hits"] += int(hit)
                except (AudioUnavailable, ImportError, OSError, RuntimeError):
                    embedding_unavailable = True
                    result["degraded"].append("ECAPA 声音特征不可用；保留自动分段，角色身份未知")
            record = {"id": identity, "record_id": identity, "kind": "audio", "asset_id": asset_id,
                      "run_id": run_key, "start_ms": start_ms + turn["start_ms"], "end_ms": start_ms + turn["end_ms"],
                      "speaker_id": f"{asset_id}:{extraction_key}:{digest(diarizer.version)}:{turn['speaker']}",
                      "audio_stream_index": stream, "language": config["language"], "dub_version": config["dub_version"],
                      "audio_ref": clip.relative_to(self.library.asset_dir(asset_id)).as_posix(), "overlap": turn["overlap"],
                      "source": "local_audio", "review_status": "machine", "visual_evidence": False,
                      "evidence_frame_ids": [], "model_versions": {"diarization": diarizer.version,
                       "embedding": embedder.version, "preprocessing": PREPROCESS_VERSION}, "quality": info,
                      "embedding": vector, "created_at": now()}
            related = [cue for cue in cues if _overlap(cue, record)]
            record.update(subtitle_ids=[cue["id"] for cue in related],
                          text=" / ".join(cue["text"] for cue in related),
                          subtitle_association="time_overlap_only", phonetic_alignment=False,
                          frame_ids=[frame["id"] for frame in (frames or [])
                                     if record["start_ms"] <= frame["at_ms"] < record["end_ms"]])
            # One subtitle may overlap several speakers. Its complete text is contextual,
            # not a claim that this speaker uttered every word of a translated cue.
            record["dialogue_attribution"] = "ambiguous" if turn["overlap"] or any(
                sum(_overlap({"start_ms": start_ms + other["start_ms"], "end_ms": start_ms + other["end_ms"]}, cue)
                    for other in turns) > 1 for cue in related) else "time_associated"
            raw_records.append(record)
            segment_path = run_dir / "segments" / f"{identity}.json"
            saved_record = read_json(segment_path)
            if saved_record is not None and (saved_record.get("embedding") is not None or vector is None):
                raw_records[-1] = saved_record
            else:
                if saved_record is not None:
                    atomic_json(run_dir / "history" / f"{identity}-{digest(saved_record)}.json", saved_record)
                atomic_json(segment_path, record)
        result["timings"]["embedding_s"] = time.perf_counter() - stamp
        if checkpoint:
            checkpoint()
        previous_records = read_jsonl(run_dir / "records.jsonl")
        if previous_records and previous_records != raw_records:
            atomic_jsonl(run_dir / "history" / f"records-{digest(previous_records)}.jsonl", previous_records)
        atomic_jsonl(run_dir / "records.jsonl", raw_records)
        atomic_json(run_dir / "config.json", config)
        atomic_json(run_dir / "source.json", source_key)
        with self.library.lock:
            active = read_json(root / "active.json", {"runs": []})
            # New analysis replaces only overlapping active ranges; old raw runs survive.
            retained = []
            for row in active["runs"]:
                if not _overlap(row, {"start_ms": start_ms, "end_ms": end_ms}):
                    retained.append(row)
                else:
                    if row["start_ms"] < start_ms:
                        retained.append({**row, "end_ms": start_ms})
                    if row["end_ms"] > end_ms:
                        retained.append({**row, "start_ms": end_ms})
            active["runs"] = retained
            active["runs"].append({"run_id": run_key, "start_ms": start_ms, "end_ms": end_ms})
            atomic_json(root / "active.json", active)
        stamp = time.perf_counter()
        result["records"] = self.records(asset_id, start_ms, end_ms, embedder=embedder, checkpoint=checkpoint)
        # Historical matching receipts remain independently inspectable after a
        # sample revocation, correction or threshold change alters current projection.
        associations = [{key: row.get(key) for key in ("id", "character_id", "candidate_character_id",
                          "association_status", "association_source", "candidates", "distance", "candidate_margin",
                          "calibration_id", "identity_revision")} for row in result["records"]]
        atomic_jsonl(run_dir / "associations" / f"{digest(associations)}.jsonl", associations)
        result["timings"]["matching_s"] = time.perf_counter() - stamp
        result["status"] = "partial" if result["degraded"] else "completed" if turns else "no_speech"
        result["timings"]["total_s"] = time.perf_counter() - started
        if config["sync_enabled"]:
            from .av_sync import sync_availability
            sync = sync_availability(config)
            result["degraded"].append(sync["reason"] if not sync["available"] else "音画复核已配置；需连续人物轨迹后单独执行短片段复核")
            result["status"] = "partial"
        atomic_json(root / "status.json", {key: value for key, value in result.items() if key != "records"})
        return result

    def _raw_records(self, asset_id: str) -> list[dict]:
        root = self._directory(asset_id)
        active = read_json(root / "active.json", {"runs": []})
        result = []
        for run in active["runs"]:
            for row in read_jsonl(root / "runs" / safe_id(run["run_id"]) / "records.jsonl"):
                if not _overlap(row, run):
                    continue
                start, end = max(row["start_ms"], run["start_ms"]), min(row["end_ms"], run["end_ms"])
                if (start, end) != (row["start_ms"], row["end_ms"]):
                    # Preserve the unprocessed parts of an old utterance crossing
                    # a new partial run. Its original audio/identity stays auditable.
                    row = {**row, "id": row["id"] + "_part_" + digest([start, end]),
                           "source_record_id": row["id"], "source_start_ms": row["start_ms"],
                           "source_end_ms": row["end_ms"], "start_ms": start, "end_ms": end,
                           "projection_reason": "partial_reanalysis_preserved_remainder"}
                    row["record_id"] = row["id"]
                result.append(row)
        return result

    def records(self, asset_id: str, start_ms: int | None = None, end_ms: int | None = None, *,
                embedder=None, checkpoint=None) -> list[dict]:
        root = self._directory(asset_id)
        raw = self._raw_records(asset_id)
        if not raw:
            return []
        bank_path, catalog = self._bank(asset_id)
        profiles = {profile["id"]: profile for profile in catalog["profiles"]}
        overrides = read_json(root / "identity-overrides.json", {})
        bank = self.samples(asset_id)
        result, prepared = [], {}
        for record in raw:
            if start_ms is not None and record["end_ms"] <= start_ms or end_ms is not None and record["start_ms"] >= end_ms:
                continue
            row = {key: value for key, value in record.items() if key != "embedding"}
            config = read_json(root / "runs" / safe_id(record["run_id"]) / "config.json")
            model_version = record["model_versions"]["embedding"]
            key = (model_version, record["language"], record["dub_version"])
            if key not in prepared:
                samples = []
                current_embedder = embedder
                for sample in bank:
                    if (sample["status"] != "confirmed" or not sample["quality"]["usable"]
                            or (sample["language"], sample["dub_version"]) != key[1:]):
                        continue
                    cache = bank_path.parent / "voice-features"
                    # Reading projections never starts a missing heavyweight model. The
                    # explicit run/confirmation path materializes bank embeddings.
                    feature_key = digest({"sha256": sample["content_sha256"], "model": model_version,
                                          "preprocess": PREPROCESS_VERSION})
                    saved = read_json(cache / f"{feature_key}.json")
                    if saved:
                        vector = saved["embedding"]
                    elif embedder is not None and current_embedder.version == model_version:
                        try:
                            vector, _ = self._feature(self._sample_file(asset_id, sample), current_embedder,
                                                       cache, checkpoint=checkpoint)
                        except (AudioUnavailable, ImportError, OSError, RuntimeError):
                            continue
                    else:
                        continue
                    samples.append({**sample, "embedding": vector})
                prepared[key] = samples
            samples = prepared[key]
            calibration = read_json(root / "runs" / safe_id(record["run_id"]) / "calibration.json")
            row.update(match_voice(record["embedding"], samples, top_k=config["top_k"], calibration=calibration,
                                   blocked=record["overlap"]) if record.get("embedding") else
                       match_voice([1.0], [], top_k=config["top_k"]))
            override = overrides.get(record["id"]) or overrides.get(record.get("source_record_id"))
            if override:
                resolved = self._resolve(catalog, override["character_id"])
                character_id = resolved if resolved in profiles else None
                row.update(character_id=character_id, candidate_character_id=None, association_source="human",
                           association_status="confirmed" if character_id else "unknown", review_status="human",
                           needs_identity_review=False, identity_revision=override["version"])
            profile = profiles.get(row.get("character_id"))
            row["character_name"] = profile["name"] if profile else None
            row["character_aliases"] = profile.get("aliases", []) if profile else []
            row["audio_url"] = f"/api/assets/{asset_id}/audio/records/{row['id']}/clip"
            row["hearing_fact"] = "听到说话声音；不证明画面人物出现、动作或人物关系"
            result.append(row)
        return sorted(result, key=lambda row: (row["start_ms"], row["end_ms"], row["speaker_id"]))

    def record(self, asset_id: str, record_id: str) -> dict:
        safe_id(record_id)
        record = next((row for row in self.records(asset_id) if row["id"] == record_id), None)
        if not record:
            raise ValueError("音频记录不存在或已非当前版本")
        return record

    def clip_path(self, asset_id: str, record_id: str) -> Path:
        row = self.record(asset_id, record_id)
        source = confined_path(self.library.asset_dir(asset_id), row["audio_ref"])
        if not row.get("source_record_id"):
            return source
        destination = self._directory(asset_id) / "projection-clips" / f"{safe_id(record_id)}.wav"
        with self.library.lock:
            if not destination.is_file():
                crop_pcm(source, destination, row["start_ms"] - row["source_start_ms"],
                         row["end_ms"] - row["source_start_ms"])
        return destination

    def confirm(self, asset_id: str, record_id: str, character_id: str | None, note: str = "") -> dict:
        before_record = self.record(asset_id, record_id)
        if character_id and character_id not in {row["id"] for row in self.characters.catalog(asset_id)["profiles"]}:
            raise ValueError("角色不在当前作品允许的档案中")
        root = self._directory(asset_id)
        with self.library.lock:
            overrides = read_json(root / "identity-overrides.json", {})
            before = overrides.get(record_id)
            after = {"character_id": character_id, "source": "human", "at": now(),
                     "version": (before or {}).get("version", 0) + 1, "note": note[:2000]}
            overrides[record_id] = after
            atomic_json(root / "identity-overrides.json", overrides)
            self._audit(root / "identity-audit.jsonl", {"record_id": record_id, "before": before, "after": after})
        self._invalidate_changed({asset_id: {record_id: (before_record["character_id"], before_record["association_status"])}})
        return self.record(asset_id, record_id)

    def dialogue_context(self, asset_id: str, start_ms: int, end_ms: int) -> list[dict]:
        fields = ("id", "start_ms", "end_ms", "speaker_id", "character_id", "character_name", "candidate_character_id",
                  "association_status", "association_source", "text", "subtitle_ids", "overlap", "dialogue_attribution",
                  "subtitle_association", "hearing_fact")
        return [{key: row.get(key) for key in fields} for row in self.records(asset_id, start_ms, end_ms)]


def validate_audio_artifacts(library) -> None:
    """Validate imported audio's entire file graph before restore commits anything.

    Never open model/calibration paths supplied by an archive. These are runtime
    configuration, not media evidence references. Historical run calibration is
    a local JSON snapshot, so normal record reads never follow that external path.
    """
    from .characters import CharacterStore
    assets = {asset["id"]: asset for asset in library.list_assets(include_deleted=True)}
    for bank_path in (library.root / "works").glob("*/voice-samples.jsonl"):
        scope = safe_id(bank_path.parent.name)
        registry = read_json(bank_path.parent / "characters.json", {"profiles": {}, "redirects": {}})
        seen = set()
        for sample in read_jsonl(bank_path):
            identity = safe_id(sample["id"])
            if identity in seen or sample.get("scope_id") != scope:
                raise ValueError("备份声音样本身份或作品引用无效")
            CharacterStore._resolve(registry, safe_id(sample.get("character_id")), strict=True)
            seen.add(identity)
            if sample.get("audio_ref") != f"voice-audio/{identity}.wav":
                raise ValueError("备份声音样本文件引用无效")
            path = confined_path(bank_path.parent, sample["audio_ref"])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sample.get("content_sha256"):
                raise ValueError("备份声音样本文件缺失或内容不匹配")
            try:
                quality(path)
            except (wave.Error, EOFError) as exc:
                raise ValueError("备份声音样本格式无效") from exc
            if sample.get("status") not in {"confirmed", "proposed", "revoked"}:
                raise ValueError("备份声音样本状态无效")
            if sample.get("source_kind") == "original_clip":
                origin = sample.get("provenance", {})
                safe_id(origin.get("asset_id"))
                duration = assets.get(origin["asset_id"], {}).get("duration_ms", float("inf"))
                # A copied exemplar may retain historical provenance after its
                # source asset was permanently deleted; runtime trust excludes it.
                if not 0 <= origin.get("start_ms", -1) < origin.get("end_ms", -1) <= duration:
                    raise ValueError("备份声音样本时间范围无效")
            elif sample.get("source_kind") != "external_licensed":
                raise ValueError("备份声音样本来源无效")
    for asset_id, asset in assets.items():
        root = library.asset_dir(asset_id) / "audio"
        AudioSettings.model_validate(read_json(root / "settings.json", {}))
        run_records = {}
        for records_path in (root / "runs").glob("*/records.jsonl"):
            run_id = safe_id(records_path.parent.name)
            AudioSettings.model_validate(read_json(records_path.parent / "config.json", {}))
            records = read_jsonl(records_path)
            run_records[run_id] = records
            seen = set()
            for record in records:
                identity = safe_id(record["id"])
                if identity in seen or record.get("asset_id") != asset_id or record.get("run_id") != run_id:
                    raise ValueError("备份音频记录身份无效")
                seen.add(identity)
                if not 0 <= record["start_ms"] < record["end_ms"] <= asset["duration_ms"]:
                    raise ValueError("备份音频记录时间范围无效")
                if (not str(record.get("speaker_id", "")).startswith(asset_id + ":")
                        or record.get("visual_evidence") is not False or record.get("evidence_frame_ids") != []):
                    raise ValueError("备份音频说话人作用域或证据类型无效")
                stream_index = record.get("audio_stream_index")
                if isinstance(stream_index, bool) or not isinstance(stream_index, int) or stream_index < 0:
                    raise ValueError("备份音轨索引无效")
                relative = record.get("audio_ref", "")
                parts = Path(relative).parts
                if len(parts) != 5 or parts[:2] != ("audio", "cache") or parts[3:] != ("clips", f"{identity}.wav"):
                    raise ValueError("备份音频片段文件引用无效")
                safe_id(parts[2])
                path = confined_path(library.asset_dir(asset_id), relative)
                if not path.is_file():
                    raise ValueError("备份音频片段文件缺失")
                try:
                    with wave.open(str(path)) as wav:
                        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (
                                16000, 1, 2, (record["end_ms"] - record["start_ms"]) * 16):
                            raise ValueError("备份音频片段与时间轴不一致")
                except (wave.Error, EOFError) as exc:
                    raise ValueError("备份音频片段格式无效") from exc
                for frame_id in record.get("frame_ids", []):
                    library.frame_path(asset_id, frame_id)
        for run in read_json(root / "active.json", {"runs": []})["runs"]:
            if (run.get("run_id") not in run_records
                    or not 0 <= run["start_ms"] < run["end_ms"] <= asset["duration_ms"]):
                raise ValueError("备份音频活动版本引用无效")
