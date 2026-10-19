"""Build a self-contained offline Chinese performance report from collected JSON.

No application, library, API, network or model is accessed. Refresh evidence with
performance_collect.py separately; this builder only reads local report inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGES = (
    ("keyframe_scan", "关键帧预扫描"), ("transcript", "台词预理解"),
    ("audio_identity", "音频身份预处理"), ("visual_identity", "视觉身份预处理"),
    ("preparation", "视觉分块与简要索引"), ("lrasd", "LR-ASD 抽样复核"),
    ("deep_analysis", "深入视觉理解"), ("context_refresh", "连续上下文刷新"), ("index", "场景索引发布"),
)
STATUS = {"queued": "排队", "running": "运行中", "paused": "已暂停", "pausing": "正在暂停",
          "completed": "已完成", "partial": "部分完成", "failed": "失败", "unavailable": "不可用",
    "disabled": "未启用", "pending": "待开始", "not_started": "待开始", "not_persisted": "尚无保存结果",
    "cancelled": "已取消", "canceled": "已取消", "cancelling": "正在取消"}


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default


def stamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp() * 1000
    except (TypeError, ValueError):
        return None


def latest_run(episode):
    return max(episode.get("analysis_runs", []), key=lambda row: row["job"].get("created_at", ""), default={})


def interval(start, end, label, kind, detail):
    start, end = stamp(start), stamp(end)
    if start is None or end is None or end < start:
        return None
    return {"start": start, "end": end, "label": label, "kind": kind, "detail": detail}


def model(data, annotations, data_path, annotation_path):
    if data.get("schema_version") != "scenerecall-performance-v1":
        raise ValueError("Unsupported performance evidence schema")
    completion_scope = annotations.get("completion_scope", data.get("completion_scope", {
        "target_episodes": [1], "excluded_episodes": [2, 3]}))
    targets = set(completion_scope["target_episodes"])
    episodes = []
    for episode in data["episodes"]:
        run = latest_run(episode)
        job = run.get("job", {})
        progress = job.get("stage_progress", {})
        timings = run.get("stage_timings", {})
        scan = max(episode.get("scan_jobs", []), key=lambda row: row.get("created_at", ""), default={})
        rows, lanes = [], []
        known_stages = dict(STAGES)
        stage_order = list(STAGES) + [(key, key) for key in timings if key not in known_stages]
        for key, title in stage_order:
            values = timings.get(key, {})
            step = progress.get(key, {})
            state = step.get("status", "pending")
            wall = values.get("latest_recorded_wall", {}).get("value")
            wall_kind = "最近一次运行墙钟" if wall is not None else "未单独记录"
            cumulative = values.get("accumulated_elapsed", {}).get("value")
            segments = []
            detail = {"episode": episode["episode"], "stage": key, "measurement": values,
                      "source": run.get("job_source"), "progress": step}
            if key == "keyframe_scan":
                state = scan.get("status", "pending")
                wall = scan.get("created_to_last_update_seconds")
                wall = wall * 1000 if wall is not None else None
                wall_kind = "任务生命周期，含排队与暂停" if wall is not None else "未记录"
                detail["scan_job"] = scan
                value = interval(scan.get("created_at"), scan.get("updated_at"), "预扫描任务跨度", "envelope", detail)
                if value:
                    segments.append(value)
            elif key == "visual_identity":
                visual = run.get("visual_identity", {})
                state = visual.get("status", "not_persisted")
                if state == "not_persisted" and key in progress:
                    state = "retained_progress"
                if wall is None and visual.get("metrics", {}).get("total_ms") is not None:
                    wall = visual["metrics"]["total_ms"]
                    wall_kind = "已保存视觉结果的运行计时"
                detail["visual_metrics"] = visual
                for name in ("detections", "regions"):
                    cache = visual.get("cache", {}).get(name, {})
                    value = interval(cache.get("first_current_file_write_utc"), cache.get("last_current_file_write_utc"),
                                     "视觉缓存写入范围", "envelope", cache)
                    if value:
                        segments.append(value)
                        break
            elif key == "audio_identity":
                state = run.get("audio_identity", {}).get("status", "not_persisted")
                detail["audio"] = run.get("audio_identity", {})
            elif key == "lrasd":
                reviews = run.get("asd_reviews", {})
                state = reviews.get("status", "not_started")
                value = interval(reviews.get("created_at"), reviews.get("updated_at"), "ASD 回执写入跨度", "envelope", reviews)
                if value:
                    segments.append(value)
                detail["sampled_reviews"] = reviews
            else:
                receipts = [entry for entry in run.get("receipts", {}).get("entries", [])
                            if entry.get("stage") == key and entry.get("provider_success_receipt")]
                for entry in receipts:
                    estimate = entry.get("interval_estimate") or {}
                    value = interval(estimate.get("start_utc"), estimate.get("end_utc"), title + "成功调用", "call", entry)
                    if value:
                        segments.append(value)
                if key == "index" and values:
                    state = "recorded"
            if job.get("status") != "running" and state == "running":
                state = "retained_progress"
            recorded_text = "已完成任务的累计计时" if job.get("status") == "completed" else "已有计时，仍可能追加"
            state_text = {"retained_progress": "保留旧进度，当前未执行", "recorded": recorded_text}.get(state, STATUS.get(state, state))
            rows.append({"key": key, "title": title, "state": state, "state_text": state_text,
                         "progress_semantics": "original_planned_parent_units; children may add provider requests"
                         if key == "deep_analysis" else "stage-defined units",
                         "completed": step.get("completed"), "total": step.get("total"),
                         "wall_ms": wall, "wall_kind": wall_kind, "accumulated_ms": cumulative,
                         "requests": values.get("request_attempts"), "cache_hits": values.get("cache_hits"),
                         "fallback": values.get("fallback_count", 0), "failures": values.get("failure_counts", {}),
                         "detail": detail})
            if segments:
                lanes.append({"stage": key, "title": title, "segments": segments})
        lifecycle = run.get("lifecycle_envelope") or {}
        if not lifecycle and job.get("created_at"):
            end = job.get("updated_at") if job.get("status") == "completed" else data.get("capture_finished_utc")
            lifecycle = {"start_utc": job["created_at"], "end_utc": end,
                         "seconds": (stamp(end) - stamp(job["created_at"])) / 1000 if stamp(end) else None}
        episodes.append({"episode": episode["episode"], "asset_id": episode["asset_id"],
                         "in_scope": episode["episode"] in targets,
                         "scope_label": "本次完成目标" if episode["episode"] in targets else "范围外历史，按用户要求停止",
                         "title": episode.get("title"), "status": job.get("status", "not_started"),
                         "status_text": STATUS.get(job.get("status"), "尚无任务"), "job_id": job.get("id"),
                         "duration_ms": episode.get("media_duration_ms"), "lifecycle": lifecycle,
                         "rows": rows, "lanes": lanes, "progress": progress,
                         "requests": job.get("request_count", 0), "asd": run.get("asd_reviews", {}),
                         "status_notes": run.get("status_consistency_notes", [])})
    sources = {}

    def collect_sources(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str):
                key = value["path"] + "#" + str(value.get("selector", ""))
                sources.setdefault(key, {"id": hashlib.sha256(key.encode()).hexdigest()[:12], **value})
            for item in value.values():
                collect_sources(item)
        elif isinstance(value, list):
            for item in value:
                collect_sources(item)

    collect_sources(data)
    collect_sources(annotations)
    collect_sources({"path": str(data_path), "kind": "json"})
    collect_sources({"path": str(annotation_path), "kind": "json"})
    target_episodes = [ep for ep in episodes if ep["in_scope"]]
    all_complete = bool(targets) and len(target_episodes) == len(targets) and all(
        ep["status"] == "completed" for ep in target_episodes)
    scored = sum(ep["asd"].get("status_counts", {}).get("review_required", 0) for ep in target_episodes)
    no_tracks = sum(ep["asd"].get("status_counts", {}).get("no_tracks", 0) for ep in target_episodes)
    unavailable = sum(ep["asd"].get("status_counts", {}).get("unavailable", 0) for ep in target_episodes)
    dates = [stamp(scan.get("created_at")) for ep in data["episodes"] if ep["episode"] in targets
             for scan in ep.get("scan_jobs", [])]
    dates += [value for ep in target_episodes for lane in ep["lanes"] for seg in lane["segments"]
              for value in (seg["start"], seg["end"])]
    dates += [stamp(ep["lifecycle"].get("end_utc")) for ep in target_episodes]
    dates = [value for value in dates if value is not None]
    cutoff = stamp(data.get("capture_finished_utc"))
    return {"schema_version": "scenerecall-performance-report-v1", "as_of": data.get("capture_finished_utc"),
            "built_at": datetime.now(timezone.utc).isoformat(), "all_complete": all_complete,
            "completion_scope": completion_scope, "target_count": len(targets),
            "complete_count": sum(ep["status"] == "completed" for ep in target_episodes),
            "requests": sum(ep["requests"] for ep in target_episodes),
            "excluded_requests": sum(ep["requests"] for ep in episodes if not ep["in_scope"]),
            "asd_counts": {"scored": scored, "no_tracks": no_tracks, "unavailable": unavailable},
            "timeline_start": min(dates) if dates else cutoff,
            "timeline_end": max(dates) if all_complete and dates else max([cutoff] + dates) if cutoff else max(dates),
            "episodes": episodes, "annotations": annotations, "sources": list(sources.values()),
            "timeline_policy": data.get("timeline_policy", {}),
            "snapshot_changed": data.get("job_snapshot_changed_during_capture", []),
            "data_source": str(data_path), "schema_source": str(HERE / "performance_schema.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "performance_data.json")
    parser.add_argument("--annotations", type=Path, default=HERE / "performance_annotations.json")
    parser.add_argument("--output", type=Path, default=HERE / "performance_report.html")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(HERE)  # Report writes remain in this task's report directory.
    payload = model(read(args.data), read(args.annotations, {}), args.data.resolve(), args.annotations.resolve())
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = (HERE / "performance_report_template.html").read_text(encoding="utf-8")
    if template.count("@@REPORT_DATA@@") != 1:
        raise ValueError("Report template must contain one data placeholder")
    output.write_text(template.replace("@@REPORT_DATA@@", serialized), encoding="utf-8")
    print(json.dumps({"output": str(output), "all_complete": payload["all_complete"],
                      "source_count": len(payload["sources"]), "bytes": output.stat().st_size,
                      "as_of": payload["as_of"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
