"""Explainable wall-time estimates using stage waves and shared connections."""
from __future__ import annotations

import copy
import heapq
import math

from .execution import execution_width

TIMING_BASIS = ("按阶段有效并发和共享连接上限估计处理耗时；有同模型历史时使用每次调用实测时长的 0.7–1.5 倍，"
                "否则暂按每次 15–90 秒。多集先并行整理台词，再按集理解；不含本地扫描、重试、排队和向量索引。")


def _profile_stage(stage):
    return {"transcript": "preparation", "context_refresh": "preparation",
            "deep_analysis": "vision"}.get(stage, stage)


def _latency(history, config, stage):
    from .jobs import model_identity
    profile_stage = _profile_stage(stage)
    profile = config["profile_snapshots"].get(profile_stage)
    samples, elapsed = 0, 0
    for job in history:
        previous = job["config"].get("profile_snapshots", {}).get(profile_stage)
        if profile is None or previous is None or model_identity(previous) != model_identity(profile):
            continue
        metric = job.get("stage_metrics", {}).get("deep_analysis" if stage == "vision" else stage, {})
        count, duration = metric.get("request_count", 0), metric.get("elapsed_ms", 0)
        if count > 0 and duration > 0:
            samples += count
            elapsed += duration
    if samples:
        mean = elapsed / samples
        return max(1, round(mean * .7)), max(1, round(mean * 1.5)), "history"
    return 15000, 90000, "assumption"


def estimate_timing(queue, config, estimate):
    result = copy.deepcopy(estimate)
    stages = result.setdefault("stage_estimates", {})
    history = queue.list()[:40]
    low, high = 0, 0
    for stage, values in stages.items():
        profile = config["profile_snapshots"].get(_profile_stage(stage))
        live = queue.providers.get(profile["id"]) if profile else {}
        width = execution_width(config, stage, live) if stage in {"subtitle", "transcript", "preparation"} else 1
        minimum, maximum = values["requests_min"], values["requests_max"]
        a, b, source = _latency(history, config, stage)
        waves_min, waves_max = math.ceil(minimum / width), math.ceil(maximum / width)
        values.update(concurrency=width, waves_min=waves_min, waves_max=waves_max,
                      estimated_wall_ms_min=waves_min * a, estimated_wall_ms_max=waves_max * b,
                      request_wall_ms_min=a, request_wall_ms_max=b, latency_source=source,
                      profile_id=(profile or {}).get("id"),
                      connection_concurrency=live.get("max_concurrency", width))
        low += values["estimated_wall_ms_min"]
        high += values["estimated_wall_ms_max"]
    result.update(estimated_wall_ms_min=low, estimated_wall_ms_max=high, timing_basis=TIMING_BASIS,
                  parallel_episode_preparation=queue.independent_transcript(config))
    if result["parallel_episode_preparation"]:
        result["episode_preparation_concurrency"] = queue._episode_preparation_width([{"config": config}])
    return result


def _group_width(group):
    available = sum(sorted(group["file_slots"].values(), reverse=True)[:group["file_limit"]])
    count = sum(number for number, _latency in group["units"])
    return max(1, min(group["width"], available, count))


def _service_time(group):
    count = sum(number for number, _latency in group["units"])
    width = _group_width(group)
    units = [(number, latency) for number, latency in group["units"] if number]
    if not units:
        return 0
    if len({latency for _number, latency in units}) == 1:
        return math.ceil(count / width) * units[0][1]
    # A small deterministic list schedule retains whole request waves even
    # when different models/stages have different historical request durations.
    lanes = [0] * width
    for number, latency in sorted(units, key=lambda unit: unit[1], reverse=True):
        for _index in range(number):
            heapq.heapreplace(lanes, lanes[0] + latency)
    return max(lanes)


def batch_timing(estimates):
    """Respect the dialogue barrier, connection sharing and serial episode plot.

    The preparation estimate uses the longest file chain and total service time
    per connection. It is a planning estimate, not a scheduling guarantee.
    """
    totals = {}
    for estimate in estimates:
        for stage, counts in estimate.get("stage_estimates", {}).items():
            target = totals.setdefault(stage, {"requests_min": 0, "requests_max": 0,
                "estimated_wall_ms_min": 0, "estimated_wall_ms_max": 0, "concurrency": 1,
                "waves_min": 0, "waves_max": 0})
            for key in ("requests_min", "requests_max"):
                target[key] += counts[key]
            target["concurrency"] = max(target["concurrency"], counts.get("concurrency", 1))
    bounds = {}
    file_limit = max((estimate.get("episode_preparation_concurrency", len(estimates))
                      for estimate in estimates if estimate.get("parallel_episode_preparation")), default=1)
    for suffix in ("min", "max"):
        connections, longest, sequential = {}, 0, 0
        file_chains = {"units": [], "width": file_limit, "file_limit": file_limit, "file_slots": {}}
        for file_index, estimate in enumerate(estimates):
            prelude = 0
            for stage, counts in estimate.get("stage_estimates", {}).items():
                duration = counts.get(f"estimated_wall_ms_{suffix}", 0)
                if estimate.get("parallel_episode_preparation") and stage in {"subtitle", "transcript"}:
                    prelude += duration
                    key = counts.get("profile_id") or stage
                    group = connections.setdefault(key, {"units": [], "width": 1,
                        "file_slots": {}, "file_limit": file_limit})
                    latency = counts.get(f"request_wall_ms_{suffix}", 0)
                    group["units"].append((counts[f"requests_{suffix}"], latency))
                    group["width"] = max(group["width"], counts.get("connection_concurrency", 1))
                    group["file_slots"][file_index] = max(group["file_slots"].get(file_index, 0),
                        min(counts[f"requests_{suffix}"], counts.get("concurrency", 1)))
                else:
                    sequential += duration
                totals[stage][f"estimated_wall_ms_{suffix}"] += duration
                totals[stage][f"waves_{suffix}"] += counts.get(f"waves_{suffix}", counts[f"requests_{suffix}"])
            longest = max(longest, prelude)
            if prelude:
                file_chains["units"].append((1, prelude))
                file_chains["file_slots"][file_index] = 1
        service = max((_service_time(group) for group in connections.values()), default=0)
        bounds[f"estimated_wall_ms_{suffix}"] = max(longest, service, _service_time(file_chains)) + sequential
        for stage in ("subtitle", "transcript"):
            if stage not in totals:
                continue
            groups, serial_duration, serial_waves = {}, 0, 0
            for file_index, estimate in enumerate(estimates):
                counts = estimate.get("stage_estimates", {}).get(stage)
                if counts is None:
                    continue
                if not estimate.get("parallel_episode_preparation"):
                    serial_duration += counts.get(f"estimated_wall_ms_{suffix}", 0)
                    serial_waves += counts.get(f"waves_{suffix}", counts[f"requests_{suffix}"])
                    continue
                group = groups.setdefault(counts.get("profile_id") or stage,
                    {"units": [], "width": 1, "file_slots": {}, "file_limit": file_limit})
                group["units"].append((counts[f"requests_{suffix}"], counts[f"request_wall_ms_{suffix}"]))
                group["width"] = max(group["width"], counts.get("connection_concurrency", 1))
                group["file_slots"][file_index] = min(counts[f"requests_{suffix}"], counts.get("concurrency", 1))
            if groups:
                target = totals[stage]
                widths = [_group_width(group) for group in groups.values()]
                target["concurrency"] = max(widths)
                target[f"waves_{suffix}"] = serial_waves + max(
                    math.ceil(sum(number for number, _latency in group["units"]) / width)
                    for group, width in zip(groups.values(), widths, strict=True))
                target[f"estimated_wall_ms_{suffix}"] = serial_duration + max(
                    _service_time(group) for group in groups.values())
    return {**bounds, "stage_estimates": totals, "timing_basis": TIMING_BASIS}
