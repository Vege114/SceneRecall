#!/usr/bin/env python3
"""Explicit new-content retrieval experiments in isolated libraries only.

No remote model calls, historical report updates, source-library writes or whole
episode runs are permitted by this entrypoint. Model installation is separate.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from scenerecall.jobs import JobQueue
from scenerecall.library import (Library, atomic_json, atomic_jsonl, confined_path, digest,
                                read_json, read_jsonl, safe_id, source_snapshot)
from scenerecall.models import AnalysisInput
from scenerecall.providers import ProviderManager
from scenerecall.visual_encoder import VisualEncoder
from benchmark_scenes import MeasuredSearch


class LocalOnlyProviders(ProviderManager):
    async def _request(self, *args, **kwargs):
        raise RuntimeError("检索基础实验禁止远程模型调用")

    async def _chat(self, *args, **kwargs):
        raise RuntimeError("检索基础实验禁止视觉语言模型调用")


def seed(clip, destination):
    source = Path(clip["source_library"]).resolve(strict=True)
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("实验资料库必须与原资料库独立")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("首次处理实验必须使用空目录")
    asset_id = safe_id(clip["asset_id"])
    root = source / "assets" / asset_id
    asset = read_json(root / "manifest.json")
    if asset is None:
        raise ValueError("实验资产不存在")
    a, b = clip["start_ms"], clip["end_ms"]
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= asset["duration_ms"] or b - a > 90000:
        raise ValueError("基础实验每段须在原片范围内且不超过九十秒")
    location = read_json(source / "private" / "media-locations.json", {})[asset_id]
    if isinstance(location, str) or source_snapshot(Path(location["path"])) != location:
        raise ValueError("原片位置或内容快照已变化")
    active = read_json(root / "active-analysis.json", {})
    pointer = active.get("container_subtitles")
    if pointer:
        cues = read_jsonl(confined_path(root, pointer["path"]))
    else:
        cues = read_jsonl(root / "subtitles" / "external.jsonl")
        for value in active.get("subtitle_runs", {}).values():
            cues.extend(read_jsonl(confined_path(root, value["path"])))
    cues = [c for c in cues if c["start_ms"] < b and c["end_ms"] > a]
    value = copy.deepcopy(asset)
    value.update(project_id=None, subtitle_mode="external")
    atomic_json(destination / "assets" / asset_id / "manifest.json", value)
    atomic_jsonl(destination / "assets" / asset_id / "subtitles" / "external.jsonl", cues)
    atomic_json(destination / "private" / "media-locations.json", {asset_id: location})
    return Library(destination)


def validate_paths(args, manifest):
    """Reject unsafe destinations before mkdir, configuration or report writes."""
    if not isinstance(manifest, dict) or not manifest.get("clips"):
        raise ValueError("需要包含短片段的实验 manifest")
    work = args.work_dir.resolve()
    output = args.output.resolve()
    config = (args.encoder_config or args.manifest.parent / "encoder.json").resolve()
    if output in {args.manifest.resolve(), config}:
        raise ValueError("实验报告不能覆盖输入清单或编码器配置")
    for clip in manifest["clips"]:
        source = Path(clip["source_library"]).resolve(strict=True)
        if (source == work or source in work.parents or work in source.parents
                or output == source or source in output.parents):
            raise ValueError("实验目录和报告必须与原资料库独立")
        safe_id(clip["id"])
        safe_id(clip["asset_id"])
        asset = read_json(source / "assets" / clip["asset_id"] / "manifest.json")
        a, b = clip["start_ms"], clip["end_ms"]
        if (not asset or type(a) is not int or type(b) is not int
                or not 0 <= a < b <= asset["duration_ms"] or b - a > 90000):
            raise ValueError("基础实验每段须在原片范围内且不超过九十秒")
    if args.output.exists():
        raise ValueError("报告已存在；请使用新的报告路径，保留已有实验记录")


async def execute(args):
    manifest = read_json(args.manifest)
    validate_paths(args, manifest)
    if not args.allow_local_inference:
        return {"status": "not_measured", "reason": "需要 --allow-local-inference 才运行真实本地模型"}
    if args.run_mode == "warm-process-new-content" and len(manifest["clips"]) < 2:
        raise ValueError("暖进程新内容需要至少两段不同范围，第一段只负责暖机并单列")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.encoder_config or args.manifest.parent / "encoder.json"
    encoder_config = read_json(config_path)
    if encoder_config["encoder_id"] != args.encoder:
        raise ValueError("编码器参数与独立配置不一致")
    runtime = args.work_dir / "encoder-runtime"
    atomic_json(runtime / "retrieval-encoder.json", encoder_config)
    shared = VisualEncoder(runtime)
    owned = [shared]
    report = {"experiment_version": 1, "manifest_digest": digest(manifest), "run_mode": args.run_mode,
              "remote_calls_allowed": False, "rows": [], "quality_acceptance": "not_evaluated"}
    try:
        seen = set()
        for index, clip in enumerate(manifest["clips"]):
            content = (clip["asset_id"], clip["start_ms"], clip["end_ms"])
            if args.run_mode == "warm-process-new-content" and content in seen:
                raise ValueError("暖进程试验不能重复使用同一内容范围")
            seen.add(content)
            destination = args.work_dir / safe_id(clip["id"]) / "library"
            started = time.perf_counter()
            started_at = datetime.now(timezone.utc)
            if args.run_mode == "receipt-replay":
                if not (destination / "library.json").is_file():
                    raise ValueError("回放模式需要该实验目录中已有成功任务")
                library = Library(destination)
            else:
                library = seed(clip, destination)
            providers = LocalOnlyProviders(library.root / "private")
            search = MeasuredSearch(library.root / "indexes", providers)
            encoder = VisualEncoder(runtime) if args.run_mode == "cold-content" else shared
            if encoder is not shared:
                owned.append(encoder)
            search.visual_encoder = encoder
            queue = JobQueue(library, providers, search)
            if args.run_mode == "receipt-replay":
                job = next(j for j in queue.list() if j["config"].get("analysis_strategy") == "retrieval-first-v1")
                frozen = job["config"]
                if (job["status"] != "completed" or frozen.get("retrieval_mode") != "core"
                        or (job["asset_id"], frozen["start_ms"], frozen["end_ms"]) != content
                        or frozen.get("encoder_identity") != encoder.identity()):
                    raise ValueError("回放只接受相同原片范围、编码器和已成功的核心任务")
                queue.update(job["id"], status="failed")
                queue.control(job["id"], "retry")
            else:
                request = AnalysisInput(asset_id=clip["asset_id"], analysis_strategy="retrieval-first-v1",
                    retrieval_mode="core", start_ms=clip["start_ms"], end_ms=clip["end_ms"])
                _, frozen = queue.validate(request, {"bindings": {}})
                job = queue.add("analysis", frozen, clip["asset_id"])
            previous_metrics = copy.deepcopy(queue.get(job["id"]).get("stage_metrics", {}))
            await queue.execute(job["id"])
            elapsed = round((time.perf_counter() - started) * 1000)
            done = queue.get(job["id"])
            attempt_metrics = copy.deepcopy(done.get("stage_metrics", {}))
            for stage, values in attempt_metrics.items():
                previous = previous_metrics.get(stage, {})
                for key in ("elapsed_ms", "request_count", "cache_hits"):
                    if key in values:
                        values[key] -= previous.get(key, 0)
            milestones = done.get("retrieval", {})
            first = milestones.get("first_searchable_at")
            first_ms = max(0, round((datetime.fromisoformat(first) - started_at).total_seconds() * 1000)) if first else None
            queries = []
            for query in clip.get("queries", []):
                question = query["query"] if isinstance(query, dict) else query
                query_started = time.perf_counter()
                result = await search.search(question, filters={"asset_id": clip["asset_id"]}, limit=5)
                queries.append({"query": question, "elapsed_ms": round((time.perf_counter() - query_started) * 1000),
                    "results": [{key: r.get(key) for key in ("id", "start_ms", "end_ms", "seek_ms", "hit_frame_id", "match_type")}
                                for r in result["results"]], "degraded": result["degraded"]})
            report["rows"].append({"clip_id": clip["id"], "status": done["status"], "error": done.get("error"),
                "duration_ms": clip["end_ms"] - clip["start_ms"], "total_wall_ms": elapsed,
                "first_visual_searchable_ms": first_ms, "default_completed_ms": None,
                "source_mode": ("receipt_replay" if args.run_mode == "receipt-replay"
                                else "previously_imported_asset_new_visual_content"),
                "warmup_only": args.run_mode == "warm-process-new-content" and index == 0,
                "default_complete": False, "retrieval": done.get("retrieval"), "stage_metrics": attempt_metrics,
                "request_count": done["request_count"], "encoder_identity": encoder.identity(),
                "encoder_runtime": encoder.metadata, "queries": queries})
            atomic_json(args.output, report)
            if args.run_mode == "cold-content":
                await encoder.close()
            if done["status"] != "completed":
                report["status"] = "failed"
                break
        report.setdefault("status", "measured_core_only")
        return report
    finally:
        for encoder in owned:
            await encoder.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--work-dir", type=Path, required=True)
    result.add_argument("--encoder-config", type=Path)
    result.add_argument("--encoder", choices=("siglip2-base-p16-256", "openclip-multilingual-b32"), required=True)
    result.add_argument("--run-mode", choices=("cold-content", "warm-process-new-content", "receipt-replay"), required=True)
    result.add_argument("--allow-local-inference", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    outcome = asyncio.run(execute(options))
    atomic_json(options.output, outcome)
    print(json.dumps({"status": outcome["status"], "rows": len(outcome.get("rows", []))}))
