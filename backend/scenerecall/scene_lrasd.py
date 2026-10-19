"""Bounded continuous-media ASD samples owned by a formal scene-analysis run.

These receipts link model proposals to scenes, subtitles and existing audio
records. They never publish a character identity or add a voice exemplar.
"""
from __future__ import annotations

import copy

from .library import atomic_json, digest, now, read_json, safe_id

VERSION = "scene-lrasd-samples-v1"
MAX_WINDOWS = 3
WINDOW_MS = 2000


def enabled(settings):
    return bool(settings.get("enabled") and settings.get("sync_enabled")
                and settings.get("sync_model_type") == "lrasd")


def select_windows(chunks, cues, frames, shots, start_ms, end_ms):
    """Choose up to three spread-out subtitle windows inside one detected shot.

    Sparse keyframes are references only. The preview runner independently
    decodes the original continuous 25 fps picture/audio and checks cuts.
    """
    reliable = [cue for cue in cues if str(cue.get("text", "")).strip() and not cue.get("uncertain")
                and cue.get("review_status") not in {"unreadable", "rejected", "uncertain"}]
    pool = {}
    for scene in chunks:
        if scene.get("content_kind") == "credits":
            continue
        first, last = max(start_ms, scene["start_ms"]), min(end_ms, scene["end_ms"])
        bounds = shots or [{"start_ms": first, "end_ms": last}]
        for cue in reliable:
            if cue["start_ms"] >= last or cue["end_ms"] <= first:
                continue
            center = (max(first, cue["start_ms"]) + min(last, cue["end_ms"])) // 2
            for shot in bounds:
                left = (max(first, shot["start_ms"]) + 39) // 40 * 40
                right = min(last, shot["end_ms"]) // 40 * 40
                if right - left < WINDOW_MS:
                    continue
                at = max(left, min(right - WINDOW_MS, (center - WINDOW_MS // 2) // 40 * 40))
                stop = at + WINDOW_MS
                if cue["start_ms"] >= stop or cue["end_ms"] <= at:
                    continue
                key = (scene["id"], at, stop)
                pool[key] = {"scene_id": scene["id"], "scene_start_ms": scene["start_ms"],
                    "scene_end_ms": scene["end_ms"], "start_ms": at, "end_ms": stop,
                    "subtitle_ids": [item["id"] for item in reliable
                                     if item["start_ms"] < stop and item["end_ms"] > at],
                    "frame_ids": [frame["id"] for frame in frames if at <= frame["at_ms"] < stop
                                  and frame["id"] in scene.get("frame_ids", [])],
                    "scene_frame_ids": list(scene.get("frame_ids", [])),
                    "selection": "subtitle_overlap_within_detected_shot" if shots else "subtitle_overlap_cut_checked_by_worker"}
    selected = []
    for index in range(MAX_WINDOWS):
        target = start_ms + (end_ms - start_ms) * (index + .5) / MAX_WINDOWS
        available = [item for item in pool.values() if not any(
            item["start_ms"] < other["end_ms"] and item["end_ms"] > other["start_ms"] for other in selected)]
        if not available:
            break
        # Prefer separate scenes when possible, then the nearest third of the run.
        used = {item["scene_id"] for item in selected}
        chosen = min(available, key=lambda item: (item["scene_id"] in used,
            abs((item["start_ms"] + item["end_ms"]) / 2 - target), item["start_ms"], item["scene_id"]))
        selected.append(chosen)
    return sorted(selected, key=lambda item: (item["start_ms"], item["scene_id"]))


def receipt_path(library, asset_id, run_id):
    return library.asset_dir(asset_id) / "analyses" / safe_id(run_id) / "lrasd-scene-reviews.json"


def run_reviews(library, job, chunks, cues, frames, shots, previews, *, audio_records=(), checkpoint=None):
    """Persist each completed sample so pause/retry never repeats successful work."""
    settings = (job["config"].get("identity_settings") or {}).get("audio", {})
    if not enabled(settings):
        return None
    checkpoint = checkpoint or (lambda: None)
    checkpoint()
    asset_id, run_id = job["asset_id"], job["run_id"]
    config = job["config"]
    windows = select_windows(chunks, cues, frames, shots, config["start_ms"], config["end_ms"])
    asset = library.get_asset(asset_id)
    header = digest([VERSION, asset["fingerprint"], settings, windows])
    path = receipt_path(library, asset_id, run_id)
    saved = read_json(path, {})
    if saved and saved.get("header") != header:
        raise ValueError("正式任务 LR-ASD 抽样计划或配置已变化，请新建任务")
    value = {"version": VERSION, "source": "full_scene_job", "header": header,
             "asset_id": asset_id, "run_id": run_id, "job_id": job["id"],
             "created_at": saved.get("created_at", now()), "status": "running",
             "source_fingerprint": asset["fingerprint"], "settings_sha256": digest(settings),
             "coverage": {"mode": "bounded_scene_sample", "max_windows": MAX_WINDOWS,
                          "selected_windows": len(windows),
                          "selected_duration_ms": sum(item["end_ms"] - item["start_ms"] for item in windows),
                          "analysis_duration_ms": config["end_ms"] - config["start_ms"],
                          "full_episode_scored": False},
             "reviews": [], "degraded": []}
    previous = {(item["scene_id"], item["start_ms"], item["end_ms"]): item for item in saved.get("reviews", [])}
    value["reviews"] = [copy.deepcopy(previous[(item["scene_id"], item["start_ms"], item["end_ms"])])
                        for item in windows if (item["scene_id"], item["start_ms"], item["end_ms"]) in previous]
    atomic_json(path, value)
    for window in windows:
        checkpoint()
        key = (window["scene_id"], window["start_ms"], window["end_ms"])
        old = previous.get(key)
        if old and old.get("status") in {"review_required", "no_tracks"}:
            review = {**old, "cache_hit": True}
        else:
            try:
                # This entry point accepts times, never the sparse frame images.
                result = previews.run_for_job(asset_id, window["start_ms"], window["end_ms"],
                                              expected_settings=settings, checkpoint=checkpoint)
            except (OSError, ValueError, RuntimeError) as exc:
                result = {"status": "unavailable", "tracks": [], "timings": {}, "reason": str(exc)[:500]}
            review = {**copy.deepcopy(result), **window, "review_status": "candidate",
                      "audio_record_ids": [record["id"] for record in audio_records
                                           if record["start_ms"] < window["end_ms"]
                                           and record["end_ms"] > window["start_ms"]]}
            # Even a future matcher change cannot promote an ASD receipt to identity.
            for track in review.get("tracks", []):
                track.update(character_id=None, association_status="candidate")
                for candidate in track.get("candidates", []):
                    candidate.update(association_status="candidate", source_verified=False)
        value["reviews"] = [item for item in value["reviews"] if
                            (item["scene_id"], item["start_ms"], item["end_ms"]) != key] + [review]
        value["reviews"].sort(key=lambda item: (item["start_ms"], item["scene_id"]))
        atomic_json(path, value)
    if not windows:
        value["status"] = "unavailable"
        value["degraded"] = ["没有可供连续复核的带字幕、非演职员场景内两秒稳定时窗"]
    else:
        failures = [item for item in value["reviews"] if item.get("status") != "review_required"]
        value["status"] = "partial" if failures else "completed"
        value["degraded"] = [item.get("reason") or item.get("status", "unknown") for item in failures]
    value["updated_at"] = now()
    value["receipt_sha256"] = digest({key: item for key, item in value.items() if key != "receipt_sha256"})
    atomic_json(path, value)
    return value


def list_reviews(library, asset_id):
    library.get_asset(asset_id)
    directory = library.asset_dir(asset_id) / "analyses"
    paths = sorted(directory.glob("*/lrasd-scene-reviews.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)[:10]
    runs = [read_json(path) for path in paths]
    runs = [row for row in runs if isinstance(row, dict) and row.get("version") == VERSION]
    status = ("running" if any(row.get("status") == "running" for row in runs) else
              runs[0].get("status", "not_started") if runs else "not_started")
    return {"status": status, "runs": runs}


def context_reviews(library, asset_id, run_id, start, end):
    saved = read_json(receipt_path(library, asset_id, run_id), {})
    result = []
    for review in saved.get("reviews", []):
        if review.get("status") != "review_required" or review["start_ms"] >= end or review["end_ms"] <= start:
            continue
        result.append({key: copy.deepcopy(review.get(key)) for key in (
            "scene_id", "start_ms", "end_ms", "subtitle_ids", "frame_ids", "audio_record_ids", "review_status",
            "model_version", "checkpoint_sha256")})
        result[-1]["tracks"] = [{key: copy.deepcopy(track.get(key)) for key in (
            "id", "start_ms", "end_ms", "active_speaker_probability", "association_status", "candidates")}
            for track in review.get("tracks", [])]
    return result
