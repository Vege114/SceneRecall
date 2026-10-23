"""Evidence-bound scene preparation and selective deep analysis.

Provider receipts are durable before accounting/publication. A retry replays the
same bounded input memory, never a later automatically enriched character state.
"""
from __future__ import annotations

import asyncio
import copy
from contextlib import nullcontext
import math
import time

from .library import atomic_json, digest, read_json, uid
from .execution import bounded_map, execution_width, finish_on_cancel, yield_to_io
from .scene_checkpoints import SceneCheckpointMixin
from .scene_resume import SceneResumeMixin

STRATEGY = "scene-chunks-v1"
CORE_FRAMES = 56
CODEX_CORE_FRAMES = 24
MIN_VISUAL_FRAMES = 7
CONTEXT_FRAMES = 8
DEEP_FRAMES = 24
TRANSCRIPT_CUES = 100
TRANSCRIPT_BYTES = 24000
MEMORY_BYTES = 18000


def visual_batch_size(config):
    # Old tasks retain their original input units and paid receipt keys.
    return config.get("visual_batch_size", CORE_FRAMES)


def visual_split_index(frames, *, min_frames=MIN_VISUAL_FRAMES):
    if len(frames) < 2 * min_frames:
        return None
    candidates = [index for index in range(min_frames, len(frames) - min_frames + 1)
                  if frames[index - 1]["at_ms"] < frames[index]["at_ms"]]
    return min(candidates, key=lambda index: abs(index - len(frames) / 2)) if candidates else None


def visual_fallback_calls(frames):
    """Count the bounded binary fallback tree, excluding transport retries."""
    index = visual_split_index(frames)
    if index is None:
        return 1
    return 1 + visual_fallback_calls(frames[:index]) + visual_fallback_calls(frames[index:])


def split_visual_args(args, index):
    frames, context_frames, subtitles, *_ = args
    bounds = subtitles if isinstance(subtitles, dict) else {}
    cues = bounds.get("cues", []) if isinstance(subtitles, dict) else subtitles or []
    start = bounds.get("start_ms", frames[0]["at_ms"])
    end = bounds.get("end_ms", max([frames[-1]["at_ms"] + 1, *(cue["end_ms"] for cue in cues)]))
    boundary = frames[index]["at_ms"]
    result = []
    for core, reference, a, b in ((frames[:index], context_frames, start, boundary),
                                 (frames[index:], (context_frames + frames[:index])[-CONTEXT_FRAMES:], boundary, end)):
        local = [cue for cue in cues if cue["start_ms"] < b and cue["end_ms"] > a]
        ids = {cue["id"] for cue in local}
        data = {**bounds, "cues": local, "start_ms": a, "end_ms": b}
        if "segments" in data:
            data["segments"] = [{**segment, "cue_ids": [value for value in segment["cue_ids"] if value in ids]}
                                for segment in data["segments"] if set(segment["cue_ids"]) & ids]
        if "speaker_dialogue" in data:
            data["speaker_dialogue"] = [row for row in data["speaker_dialogue"]
                                        if row["start_ms"] < b and row["end_ms"] > a]
        data.pop("omitted_cues", None)
        data["notice"] = "细分批次沿用原批次的字幕采样；仅引用与当前子范围相交的字幕。"
        child = copy.deepcopy([core, reference, data, *args[3:]])
        if len(child) > 4 and child[4]:
            allowed = {frame["id"] for frame in core}
            for key in ("identity_regions", "region_images"):
                if key in child[4]:
                    child[4][key] = [row for row in child[4][key] if row["frame_id"] in allowed]
        result.append(child)
    return result


def split_deep_args(args, index):
    """Partition current evidence, retaining complete dossier and heard priors."""
    frames, start, end, *_ = args
    boundary = frames[index]["at_ms"]
    result = []
    for core, first, last in ((frames[:index], start, boundary), (frames[index:], boundary, end)):
        child = copy.deepcopy(args)
        child[:3] = [core, first, last]
        current = {frame["id"] for frame in core}
        context = child[3]
        if context:
            for key in ("identity_regions", "region_images"):
                if key in context:
                    context[key] = [row for row in context[key] if row["frame_id"] in current]
        subtitles = child[4]
        def overlaps(row):
            return row["start_ms"] < last and row["end_ms"] > first
        if isinstance(subtitles, list):
            child[4] = [row for row in subtitles if overlaps(row)]
        elif isinstance(subtitles, dict):
            if "cues" in subtitles:
                subtitles["cues"] = [row for row in subtitles["cues"] if overlaps(row)]
                cue_ids = {row["id"] for row in subtitles["cues"]}
                if "segments" in subtitles:
                    subtitles["segments"] = [{**row, "cue_ids": [value for value in row["cue_ids"]
                                                               if value in cue_ids]}
                        for row in subtitles["segments"] if set(row["cue_ids"]) & cue_ids]
            for key in ("speaker_dialogue", "active_speaker_reviews"):
                if key in subtitles:
                    # Source bounds, IDs, candidate status and model provenance
                    # stay intact even when heard evidence spans both children.
                    subtitles[key] = [row for row in subtitles[key] if overlaps(row)]
            for key, value in (("start_ms", first), ("end_ms", last)):
                if key in subtitles:
                    subtitles[key] = value
        result.append(child)
    return result


def join_deep_results(results, args):
    """Join validated child observations without conflating local identities."""
    from .models import normalize_observation_evidence
    from .providers import VisionObservation
    data = {"summary": "；".join(result["summary"] for result in results),
            "entities": [], "events": [], "spatial_observations": [], "uncertainties": [],
            "evidence_frame_ids": []}
    for number, result in enumerate(results):
        value = copy.deepcopy(result)
        mapping = {entity["id"]: f"c{number}_" + entity["id"] for entity in value.get("entities", [])}
        namespace_entities(value, f"c{number}_")
        for position in value.get("events", []) + value.get("spatial_observations", []):
            for key in ("subject_id", "object_id", "recipient_id", "entity_id"):
                if position.get(key):
                    position[key] = mapping[position[key]]
        for field in ("entities", "events", "spatial_observations", "uncertainties", "evidence_frame_ids"):
            data[field].extend(value.get(field, []))
    data = normalize_observation_evidence(data, [frame["id"] for frame in args[0]])
    # Validate shape without dumping: application-bound identity metadata on
    # entities must survive the aggregate exactly as on each paid child.
    item = VisionObservation.model_validate(data)
    ids = {entity.id for entity in item.entities}
    if len(ids) != len(item.entities):
        raise ValueError("深入理解细分结果包含重复实体")
    for event in item.events:
        if (not set(event.actor_ids + event.target_ids).issubset(ids)
                or not args[1] <= event.start_ms <= event.end_ms <= args[2]
                or event.start_ms >= args[2]):
            raise ValueError("深入理解细分结果的事件引用或时间无效")
    for position in item.spatial_observations:
        if position.subject_id not in ids or position.object_id and position.object_id not in ids:
            raise ValueError("深入理解细分结果的空间引用无效")
    return data


def join_visual_results(results, args):
    """Join continuation across fallback boundaries and recheck all evidence."""
    from pydantic import ValidationError
    from .scene_preparation import PreparedChunk, subtitle_context_data, validate_visual_preparation
    chunks, unresolved = [], []
    for child_index, result in enumerate(results):
        unresolved.extend(result.get("unresolved", []))
        for number, raw in enumerate(result["chunks"]):
            chunk = copy.deepcopy(raw)
            namespace_entities(chunk, f"c{child_index}_")
            if (number == 0 and chunks and chunk.get("continues_previous")
                    and chunk.get("content_kind") != "credits" and chunks[-1].get("content_kind") != "credits"):
                previous = chunks[-1]
                separate = copy.deepcopy(chunk)
                separate["continues_previous"] = False
                needed = set(deep_frame_ids(previous) + deep_frame_ids(chunk))
                chunk["frame_ids"] = previous["frame_ids"] + chunk["frame_ids"]
                chunk["deep_frame_ids"] = [value for value in chunk["frame_ids"] if value in needed]
                chunk["summary"] = previous["summary"] + "；" + chunk["summary"]
                chunk["reason"] = (previous.get("reason", "") + "；" + chunk.get("reason", "")).strip("；")
                chunk["scene_key"] = chunk.get("scene_key") or previous.get("scene_key", "")
                for field in ("visual_facts", "dialogue_facts", "inferences", "entities", "events"):
                    chunk[field] = previous.get(field, []) + chunk.get(field, [])
                chunk["needs_deep"] = previous["needs_deep"] or chunk["needs_deep"]
                merge_review_flags(chunk, previous)
                chunk["continues_previous"] = previous.get("continues_previous", False)
                representatives = list(dict.fromkeys(previous["representative_frame_ids"] + chunk["representative_frame_ids"]))
                chunk["representative_frame_ids"] = (representatives if len(representatives) <= 8
                                                       else representatives[:7] + [representatives[-1]])
                try:
                    PreparedChunk.model_validate(chunk)
                except ValidationError:
                    # Keep a valid boundary when merging would exceed schema
                    # limits; neither child's facts or text may be discarded.
                    chunk = separate
                else:
                    chunks.pop()
            chunks.append(chunk)
    frames, context_frames, subtitles, *_ = args
    data, cue_ids = subtitle_context_data(subtitles)
    cues = data.get("cues", []) if isinstance(data, dict) else data or []
    bounds = {key: data[key] for key in ("start_ms", "end_ms") if key in data} if isinstance(data, dict) else {}
    unresolved = list(dict.fromkeys(unresolved))
    if len(unresolved) > 64:
        unresolved = unresolved[:63] + ["\n".join(unresolved[63:])]
    return validate_visual_preparation({"chunks": chunks, "unresolved": unresolved},
        frames, cue_ids, has_context=bool(context_frames), cues=cues, **bounds)


def namespace_entities(record, prefix):
    """Scope local entities and every event reference together, never by appearance."""
    mapping = {entity["id"]: prefix + entity["id"] for entity in record.get("entities", [])}
    for entity in record.get("entities", []):
        entity["id"] = mapping[entity["id"]]
    for event in record.get("events", []):
        for key in ("actor_ids", "target_ids"):
            event[key] = [mapping[identity] for identity in event.get(key, [])]


def merge_review_flags(chunk, previous):
    if "needs_action_deep" in chunk or "needs_action_deep" in previous:
        chunk["needs_action_deep"] = chunk["needs_deep"]
    if "needs_identity_review" in chunk or "needs_identity_review" in previous:
        chunk["needs_identity_review"] = bool(chunk.get("needs_identity_review") or previous.get("needs_identity_review"))


def deep_batch_size(profile):
    # The same configured timeout applies to both paths. Long Codex image turns
    # can exceed it, so use a smaller original-image batch for compact scene
    # observations while retaining a few changes beyond the eight representatives.
    return 12 if profile.get("provider_type") == "codex_cli" else DEEP_FRAMES


def deep_frame_ids(chunk, *, all_frames=False):
    return chunk["frame_ids"] if all_frames else chunk.get("deep_frame_ids") or chunk["frame_ids"]


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(val) for key, val in value.items() if key not in {"path", "updated_at"}}
    if isinstance(value, list):
        return [canonical(val) for val in value]
    return value


def encoded_size(value):
    import json
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def annotation_digest(store, asset_id):
    fields = ("record_id", "entity_id", "summary", "character_name", "aliases")
    values = [{key: item[key] for key in fields if key in item} for item in store._annotations(asset_id)
              if any(item.get(key) is not None for key in ("summary", "character_name", "aliases"))]
    return digest(sorted(values, key=lambda value: str(value)))


LEGACY_PROMPT_VERSIONS = {
    "transcript": "scene-subtitle-preparation-v1", "preparation": "scene-visual-preparation-v3",
    "deep_analysis": "scene-contextual-vision-v5", "context_refresh": "scene-reconciliation-v2",
}

PRE_IDENTITY_PROMPT_VERSIONS = {
    "transcript": "scene-subtitle-preparation-v2", "preparation": "scene-visual-preparation-v4",
    "deep_analysis": "scene-contextual-vision-v6", "context_refresh": "scene-reconciliation-v3",
}


def prompt_versions(config=None):
    from .scene_preparation import (CONTEXTUAL_VISION_PROMPT_VERSION, RECONCILIATION_PROMPT_VERSION,
                                    SUBTITLE_PREPARATION_PROMPT_VERSION, VISUAL_PREPARATION_PROMPT_VERSION)
    current = {"transcript": SUBTITLE_PREPARATION_PROMPT_VERSION, "preparation": VISUAL_PREPARATION_PROMPT_VERSION,
               "deep_analysis": CONTEXTUAL_VISION_PROMPT_VERSION, "context_refresh": RECONCILIATION_PROMPT_VERSION}
    if config is None:
        return current
    frozen = (config.get("scene_prompt_versions") or (config.get("scene_dependencies") or {}).get("prompt_versions")
              or LEGACY_PROMPT_VERSIONS)
    if frozen not in (current, LEGACY_PROMPT_VERSIONS, PRE_IDENTITY_PROMPT_VERSIONS):
        raise ValueError("任务使用了不支持的场景提示词版本；已保存结果仍然保留")
    return copy.deepcopy(frozen)


def cue_batches(cues):
    batch = []
    for cue in cues:
        value = {key: cue[key] for key in ("id", "start_ms", "end_ms", "text", "language") if key in cue}
        if encoded_size(value) > TRANSCRIPT_BYTES:
            raise ValueError("单条字幕过长，请先修正字幕再分析")
        if batch and (len(batch) >= TRANSCRIPT_CUES or encoded_size(batch + [value]) > TRANSCRIPT_BYTES):
            yield batch
            batch = []
        batch.append(value)
    if batch:
        yield batch


def bounded_memory(prior, chunks):
    # Evidence types stay separate even in the compressed rolling state.
    entries = []
    for chunk in chunks:
        if chunk.get("content_kind") == "credits":
            continue
        entries.append({"chunk_id": chunk["id"], "asset_id": chunk["asset_id"],
                        "start_ms": chunk["start_ms"], "end_ms": chunk["end_ms"],
                        "summary": chunk["summary"][:1600],
                        "scene_key": chunk.get("scene_key", ""),
                        "visual_facts": [{**f, "text": f["text"][:400]} for f in chunk.get("visual_facts", [])[:3]],
                        "dialogue_facts": [{**f, "text": f["text"][:400]} for f in chunk.get("dialogue_facts", [])[-3:]],
                        "inferences": [{**f, "text": f["text"][:400]} for f in chunk.get("inferences", [])[:2]],
                        "unresolved": chunk.get("uncertainties", [])[:8]})
    from .scenes import SceneStore
    provenance = prior.get("provenance", []) + [
        {"asset_id": c["asset_id"], "chunk_id": c["id"], "revision_digest": SceneStore.revision_digest(c)} for c in chunks[-8:]]
    result = {"prior": prior, "recent": entries[-8:], "provenance": provenance[-20:]}
    while encoded_size(result) > MEMORY_BYTES and result["recent"]:
        result["recent"].pop(0)
    if encoded_size(result) > MEMORY_BYTES:
        result["prior"] = {"digest": prior.get("digest"), "gaps": prior.get("gaps", []),
                           "summaries": prior.get("summaries", [])[-1:]}
    if encoded_size(result) > MEMORY_BYTES:
        result["prior"] = {"digest": prior.get("digest"), "notice": "较早历史已压缩，仅使用当前证据"}
    return result


def link_similar_scenes(chunks):
    """Link at most 16 nearby candidates; never merge ranges or create evidence."""
    from .scene_preparation import normalize_scene_key
    grouped = {}
    for chunk in chunks:
        key = normalize_scene_key(chunk.get("scene_key", ""))
        chunk["scene_key"] = key
        chunk["related_chunk_ids"] = []
        if key:
            grouped.setdefault((chunk["asset_id"], key), []).append(chunk)
    for group in grouped.values():
        ordered = sorted(group, key=lambda chunk: (chunk["start_ms"], chunk["end_ms"], chunk["id"]))
        for index, chunk in enumerate(ordered):
            # Published chunks form disjoint chronological ranges. Walking out
            # from this position skips touching neighbors and visits at most 16
            # matches, rather than scanning/sorting every candidate per chunk.
            start, end = chunk["start_ms"], chunk["end_ms"]
            left, right, nearest = index - 1, index + 1, []
            while len(nearest) < 16 and (left >= 0 or right < len(ordered)):
                if left >= 0 and ordered[left]["end_ms"] >= start:
                    left -= 1
                    continue
                if right < len(ordered) and ordered[right]["start_ms"] <= end:
                    right += 1
                    continue
                if left >= 0 and (right == len(ordered)
                        or start - ordered[left]["end_ms"] <= ordered[right]["start_ms"] - end):
                    nearest.append(ordered[left])
                    left -= 1
                else:
                    nearest.append(ordered[right])
                    right += 1
            chunk["related_chunk_ids"] = [other["id"] for other in
                sorted(nearest, key=lambda other: (other["start_ms"], other["end_ms"], other["id"]))]


def subtitle_context(cues, segments, start, end):
    """Bound visual context while retaining complete original dialogue in storage.

    Dense/long static scenes use an evenly distributed cue sample. Transcript
    preparation still covers every cue; its summaries remain searchable facts.
    """
    local = [c for c in cues if c["end_ms"] > start and c["start_ms"] < end]
    selected = local
    if len(selected) > 100:
        selected = [local[round(i * (len(local) - 1) / 99)] for i in range(100)]
    selected = [{key: c[key] for key in ("id", "start_ms", "end_ms", "text")} for c in selected]
    while selected and encoded_size(selected) > 16000:
        if len(selected) == 1:
            selected = []
        else:
            selected = selected[::2]
    ids = {c["id"] for c in selected}
    summaries = [{"cue_ids": [i for i in s["cue_ids"] if i in ids], "summary": s["summary"][:600]}
                 for s in segments if s.get("content_kind") != "credits" and set(s["cue_ids"]) & ids]
    while summaries and encoded_size(summaries) > 5000:
        summaries.pop()
    return {"cues": selected, "segments": summaries, "start_ms": start, "end_ms": end,
            "omitted_cues": len(local) - len(selected),
            "notice": "原始字幕完整保留；未展示的台词不能补写为画面事实"}


class SceneJobMixin(SceneCheckpointMixin, SceneResumeMixin):
    async def _scene_local_context(self, job_id, method, *args):
        """Drain local projection work before honoring pause/cancel/shutdown."""
        self.checkpoint(job_id)
        result = await finish_on_cancel(asyncio.to_thread(method, *args))
        self.checkpoint(job_id)
        return result

    def _scene_progress(self, job_id, stage, **changes):
        """Current execution progress, independent of durable paid-call accounting."""
        progress = copy.deepcopy(self.get(job_id).get("stage_progress"))
        if progress is None:
            return
        item = progress.setdefault(stage, {"completed": 0, "total": None, "status": "pending"})
        item.update(changes)
        if item["total"] is not None and item["completed"] == item["total"]:
            item["status"] = "completed"
        self.update(job_id, stage_progress=progress)

    def _check_scene_range(self, asset_id, config, store=None):
        """Queued work must still cover whole active chunks before any paid call."""
        if config.get("deepen_chunk_id"):
            return
        if store is None:
            from .scenes import SceneStore
            store = SceneStore(self.library)
        start, end = config["start_ms"], config["end_ms"]
        overlap = [row for row in store._raw(asset_id)
                   if row["start_ms"] < end and row["end_ms"] > start
                   and not start <= row["start_ms"] < row["end_ms"] <= end]
        if overlap:
            a = min(start, *(row["start_ms"] for row in overlap))
            b = max(end, *(row["end_ms"] for row in overlap))
            raise ValueError(f"重分块范围需包含相交片段的完整边界，请选择 {a / 1000:g}–{b / 1000:g} 秒；单片段可使用深入分析")

    def _sync_scene_index(self, job_id, asset_id):
        started = time.monotonic()
        self.search.sync_asset(self.library.records(asset_id), asset_id=asset_id)
        self._scene_metric(job_id, "index", elapsed=time.monotonic() - started)

    def _scene_metric(self, job_id, stage, *, elapsed=0, requests=0, cache=0, usage=None, wall_elapsed=None,
                      failure_code=None, fallback=0, degraded_reason=None):
        metrics = copy.deepcopy(self.get(job_id).get("stage_metrics", {}))
        item = metrics.setdefault(stage, {"request_count": 0, "elapsed_ms": 0, "cache_hits": 0, "usage": {}})
        item["elapsed_ms"] += round(elapsed * 1000)
        if wall_elapsed is not None:
            item["wall_elapsed_ms"] = round(wall_elapsed * 1000)
        item["request_count"] += requests
        item["cache_hits"] += cache
        # Only stable application codes are diagnostic data. Never persist an
        # arbitrary exception attribute containing command output or secrets.
        if isinstance(failure_code, str) and failure_code in {
                "timeout", "invalid_output", "preflight", "preflight_version",
                "preflight_exec_help", "preflight_features", "preflight_login_status"}:
            item["last_failure_code"] = failure_code
            failures = item.setdefault("failure_counts", {})
            failures[failure_code] = failures.get(failure_code, 0) + 1
        if isinstance(failure_code, str) and failure_code in {"timeout", "invalid_output"}:
            key = failure_code + "_count"
            item[key] = item.get(key, 0) + 1
        if fallback:
            item["fallback_count"] = item.get("fallback_count", 0) + fallback
        if degraded_reason:
            item["degraded_reason"] = degraded_reason
        for key, value in (usage or {}).items():
            if isinstance(value, (int, float)):
                item["usage"][key] = item["usage"].get(key, 0) + value
        self.update(job_id, stage_metrics=metrics)

    def _account_scene_receipt(self, job_id, receipt):
        with self.library.lock:
            job = self.get(job_id)
            if receipt.get("job_id") != job_id or receipt["receipt_id"] in job.get("scene_receipts", []):
                return
            usage = dict(job.get("usage", {}))
            for key, value in receipt["usage"].items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
            count, cost = receipt["request_count"], receipt["cost"]
            incomplete = job.get("cost_incomplete", False) or (count > 0 and cost is None)
            known = job.get("known_cost", 0) + (cost or 0)
            metrics = copy.deepcopy(job.get("stage_metrics", {}))
            item = metrics.setdefault(receipt["stage"], {"request_count": 0, "elapsed_ms": 0, "cache_hits": 0, "usage": {}})
            item["request_count"] += count
            item["elapsed_ms"] += receipt["elapsed_ms"]
            for key, value in receipt["usage"].items():
                if isinstance(value, (int, float)):
                    item["usage"][key] = item["usage"].get(key, 0) + value
            # An aggregate performed no provider call and owns no reservation.
            # Legacy paid receipts without call_id still use their old release path.
            released = {} if receipt.get("split") else self._release_reservation(job, receipt.get("call_id"))
            self.update(job_id, usage=usage, request_count=job["request_count"] + count,
                        known_cost=known, cost=None if incomplete else known, cost_incomplete=incomplete,
                        pending_scene_receipt=None, stage_metrics=metrics, **released,
                        scene_receipts=job.get("scene_receipts", []) + [receipt["receipt_id"]])
        self._wake_reservations(job_id)

    def recover_scene_receipt(self, job):
        from .library import confined_path
        # Each parallel call owns its receipt path. Recover durable successes
        # before conservatively accounting only genuinely unresolved calls.
        for reservation in list(job.get("inflight_calls", {}).values()):
            pending = reservation.get("receipt_path")
            if pending:
                receipt = read_json(confined_path(self.library.root, pending))
                if receipt and receipt.get("job_id") == job["id"]:
                    self._account_scene_receipt(job["id"], receipt)
        pending = job.get("pending_scene_receipt")
        if pending and job.get("asset_id"):
            receipt = read_json(confined_path(self.library.asset_dir(job["asset_id"]), pending))
            if receipt and receipt.get("job_id") == job["id"]:
                self._account_scene_receipt(job["id"], receipt)
        return self.get(job["id"])

    async def _scene_call(self, job_id, stage, unit, profile, method, args, *, images=0):
        # A cache hit has no provider await. Let HTTP/control requests run even
        # when an entire serial stage can be restored from local receipts.
        await yield_to_io()
        from .jobs import model_identity
        from .models import normalize_observation_evidence
        job = self.checkpoint(job_id)
        versions = prompt_versions(job["config"])
        from .scenes import SceneStore
        root = self.library.asset_dir(job["asset_id"])
        run = root / "analyses" / job["run_id"]
        inputs = run / "scene-inputs" / f"{digest([stage, versions[stage], unit])}.json"
        frozen = read_json(inputs)
        if frozen is None:
            if stage == "preparation" and versions[stage] in {prompt_versions()[stage], "scene-visual-preparation-v4"}:
                images += sum(len((args[4] or {}).get(key, [])) for key in ("reference_frames", "region_images")) if len(args) > 4 else 0
            frozen = {"args": args, "images": images}
            atomic_json(inputs, frozen)
        args, images = frozen["args"], frozen["images"]
        def observation_data(data):
            # Old paid receipts can have sparse root evidence even though every
            # nested citation belongs to the original request. Normalize a copy
            # against those frozen inputs without replacing or repaying it.
            return (normalize_observation_evidence(data, [frame["id"] for frame in args[0]])
                    if stage == "deep_analysis" else copy.deepcopy(data))
        independent = stage == "transcript" and self.independent_transcript(job["config"])
        dependencies = (job["config"]["transcript_dependencies"] if independent
                        else job["config"]["scene_dependencies"])
        key = digest([STRATEGY, stage, versions[stage], unit, dependencies,
                      model_identity(profile), canonical(args)])
        output = run / "scene-receipts" / f"{key}.json"
        receipt = read_json(output)
        if receipt is None and not job["config"].get("force"):
            matches = sorted(root.glob(f"analyses/*/scene-receipts/{key}.json"))
            if matches:
                receipt = read_json(matches[-1])
        if receipt is not None:
            self._account_scene_receipt(job_id, receipt)
            self._scene_metric(job_id, stage, cache=1)
            return observation_data(receipt["data"])
        # Preserve paid successes above. An unsuccessful frozen input must not
        # overrule a later human correction on the next actual model request.
        refreshed = await self._scene_local_context(job_id, self.refresh_scene_request_identity, job_id, stage, args)
        if canonical(refreshed) != canonical(args):
            previous_key = key
            atomic_json(run / "scene-input-history" / f"{key}.json", frozen)
            if stage == "preparation":
                images += sum(len((refreshed[4] or {}).get(field, [])) - len((args[4] or {}).get(field, []))
                              for field in ("reference_frames", "region_images"))
            elif stage == "deep_analysis":
                images += (len((refreshed[3] or {}).get("reference_frames", []))
                           - len((args[3] or {}).get("reference_frames", [])))
            args = refreshed
            frozen = {"args": args, "images": images}
            atomic_json(inputs, frozen)
            key = digest([STRATEGY, stage, versions[stage], unit, dependencies,
                          model_identity(profile), canonical(args)])
            output = run / "scene-receipts" / f"{key}.json"
            refreshed_receipt = read_json(output)
            if refreshed_receipt is None and not job["config"].get("force"):
                matches = sorted(root.glob(f"analyses/*/scene-receipts/{key}.json"))
                if matches:
                    refreshed_receipt = read_json(matches[-1])
            if refreshed_receipt is not None:
                self._account_scene_receipt(job_id, refreshed_receipt)
                self._scene_metric(job_id, stage, cache=1)
                return observation_data(refreshed_receipt["data"])
            old_split = (read_json(run / "scene-splits" / f"{previous_key}.json")
                         if stage in {"preparation", "deep_analysis"} else None)
            if old_split is not None:
                atomic_json(run / "scene-splits" / f"{key}.json", old_split)
        if independent:
            self.check_transcript_dependencies(job["asset_id"], job["config"])
        else:
            await self._scene_local_context(job_id, self._check_scene_dependencies,
                job["asset_id"], job["config"], SceneStore(self.library))
        split_path = run / "scene-splits" / f"{key}.json"
        split = read_json(split_path) if stage in {"preparation", "deep_analysis"} else None
        if split is not None:
            return await self._split_scene_call(job_id, stage, unit, profile, method, args,
                                                split["index"], key, output)
        self.update(job_id, stage=stage, pending_scene_receipt=output.relative_to(root).as_posix())
        call_id = uid("call")
        await self.reserve_call(job_id, profile, call_id=call_id, stage=stage, frames=images,
                                receipt_path=output.relative_to(self.library.root).as_posix(),
                                context_tokens=encoded_size(canonical(args)))
        async def perform():
            started = time.monotonic()
            try:
                scope = (self.providers.scene_prompt_version(versions[stage])
                         if hasattr(self.providers, "scene_prompt_version") else nullcontext())
                with scope:
                    result = await self.invoke_profile(profile, method, *args)
            except Exception as exc:
                self.account(job_id, exc, call_id)
                self._scene_metric(job_id, stage, elapsed=time.monotonic() - started,
                                   requests=getattr(exc, "request_count", 0), usage=getattr(exc, "usage", {}),
                                   failure_code=getattr(exc, "code", None))
                context_too_large = (getattr(exc, "request_count", 0) == 0
                                     and getattr(exc, "code", None) == "character_context_too_large")
                if ((stage in {"preparation", "deep_analysis"} and context_too_large) or
                        (stage == "preparation" and getattr(exc, "request_count", 0) > 0
                         and getattr(exc, "code", None) in {"timeout", "invalid_output"})):
                    index = visual_split_index(args[0], min_frames=1 if context_too_large else MIN_VISUAL_FRAMES)
                    if index is not None:
                        # Freeze the fallback decision before any child admission.
                        # Resuming this run never resubmits the exhausted parent.
                        atomic_json(split_path, {"index": index, "code": exc.code})
                        self._scene_metric(job_id, stage, fallback=1)
                raise
            receipt = {"receipt_id": key, "call_id": call_id, "job_id": job_id, "stage": stage, "data": result.data,
                       "usage": result.usage, "request_count": result.request_count, "cost": result.cost,
                       "elapsed_ms": round((time.monotonic() - started) * 1000)}
            atomic_json(output, receipt)
            self._account_scene_receipt(job_id, receipt)
            return observation_data(result.data)
        try:
            return await finish_on_cancel(perform())
        except Exception:
            split = read_json(split_path) if stage in {"preparation", "deep_analysis"} else None
            if split is None:
                raise
            return await self._split_scene_call(job_id, stage, unit, profile, method, args,
                                                split["index"], key, output)

    async def _split_scene_call(self, job_id, stage, unit, profile, method, args, index, key, output):
        self.checkpoint(job_id)
        deep = stage == "deep_analysis"
        self.update(job_id, message=(f"深入理解输入过大，细分为较小批次（原 {len(args[0])} 帧）" if deep else
                                    f"视觉分块请求过大或输出无效，细分为较小批次（原 {len(args[0])} 帧）"))
        results = []
        for number, child in enumerate(split_deep_args(args, index) if deep else split_visual_args(args, index)):
            # Child calls own normal reservations, frozen inputs and receipts.
            # Controls/budgets are checked before each sequential admission.
            self.checkpoint(job_id)
            results.append(await self._scene_call(job_id, stage, [unit, "split", number], profile, method, child,
                images=(len(child[0]) + len((child[3] or {}).get("reference_frames", [])) if deep
                        else math.ceil((len(child[0]) + len(child[1])) / 16))))
        data = join_deep_results(results, args) if deep else join_visual_results(results, args)
        receipt = {"receipt_id": key, "job_id": job_id, "stage": stage, "data": data,
                   "usage": {}, "request_count": 0, "cost": 0, "elapsed_ms": 0, "split": True}
        atomic_json(output, receipt)
        self._account_scene_receipt(job_id, receipt)
        return copy.deepcopy(data)

    def estimate_scenes(self, config, plan):
        from .scenes import SceneStore
        from .scene_preparation import MAX_PREPARATION_OUTPUT_TOKENS
        frames = len(plan["frames"])
        cues = [c for c in SceneStore(self.library).subtitle_records(config["asset_id"])
                if c["end_ms"] > config["start_ms"] and c["start_ms"] < config["end_ms"]]
        transcript = len(list(cue_batches(cues)))
        batch_size = visual_batch_size(config)
        prep = math.ceil(frames / batch_size)
        prep_max = sum(visual_fallback_calls(plan["frames"][offset:offset + batch_size])
                       for offset in range(0, frames, batch_size))
        ocr_frames = (math.ceil((config["end_ms"] - config["start_ms"]) / 1000 * config["subtitle_fps"])
                      if "subtitle" in config["stages"] else 0)
        ocr = math.ceil(ocr_frames / 8)
        # OCR may produce one cue per sample; estimate a range before it exists.
        transcript_max = max(transcript, math.ceil(ocr_frames / TRANSCRIPT_CUES))
        minimum, maximum = prep + transcript + ocr, prep_max + transcript_max + ocr + 2 * frames
        stages = {"transcript": {"requests_min": transcript, "requests_max": transcript_max},
                  "preparation": {"requests_min": prep, "requests_max": prep_max},
                  "vision": {"requests_min": 0, "requests_max": frames},
                  "context_refresh": {"requests_min": 0, "requests_max": frames},
                  "subtitle": {"requests_min": 0 if ocr else 0, "requests_max": ocr}}
        def price(stage, images, text):
            profile = config["profile_snapshots"][stage]
            value = self.estimated_call_cost(profile, frames=images, context_tokens=text)
            if value is not None and stage == "preparation":
                value += (MAX_PREPARATION_OUTPUT_TOKENS - 4096) * profile["output_price_per_million"] / 1_000_000
            return value

        ptext = price("preparation", 0, 32000)
        reference_images = 8 if prompt_versions(config)["preparation"] == prompt_versions()["preparation"] else 0
        pvisual = price("preparation", math.ceil((batch_size + CONTEXT_FRAMES) / 16) + reference_images, 50000)
        pdeep = price("vision", deep_batch_size(config["profile_snapshots"]["vision"]) + 8, 50000)
        pocr = price("subtitle", 8, 0) if ocr else 0
        low = high = None
        if all(value is not None for value in (ptext, pvisual, pdeep, pocr)):
            low = transcript * ptext + prep * pvisual + ocr * pocr
            high = transcript_max * ptext + prep_max * pvisual + ocr * pocr + frames * (pdeep + ptext)
        return {"duration_ms": config["end_ms"] - config["start_ms"], "estimated_frames": frames + ocr_frames,
                "estimated_requests": minimum, "estimated_requests_min": minimum, "estimated_requests_max": maximum,
                "cost_estimate": low, "cost_estimate_min": low, "cost_estimate_max": high,
                "currency": "USD", "stage_estimates": stages,
                "warnings": ["视觉分块上限包含超时或无效输出后的逐级细分；深入理解与上下文更新按每帧独立复杂片段计算，不含传输重试和向量索引。",
                             "先复用本机关键帧并建立简要索引，复杂片段继续深入分析；费用与耗时以实际调用为准。"]}

    @staticmethod
    def _check_deepen_content(chunk):
        if chunk.get("content_kind") == "credits":
            raise ValueError("纯演职员信息无需深入理解，请在动画档案中查看 Bangumi 演职员资料")

    def deepen_config(self, asset_id, chunk_id, payload, settings):
        from .models import AnalysisInput
        from .scenes import SceneStore
        store = SceneStore(self.library)
        chunk = store.get(asset_id, chunk_id)
        self._check_deepen_content(chunk)
        request = AnalysisInput(asset_id=asset_id, analysis_strategy=STRATEGY,
                                start_ms=chunk["start_ms"], end_ms=chunk["end_ms"],
                                max_requests=payload.max_requests, max_cost=payload.max_cost,
                                model_overrides=payload.model_overrides, model_bindings=payload.model_bindings)
        # Manual deepening needs only vision; no new preparation work is performed.
        _, config = self.validate(request, settings, scene_deepening=True)
        config.update(analysis_strategy=STRATEGY, sampling_strategy=STRATEGY,
                      deepen_chunk_id=chunk_id, deepen_revision=store.revision_digest(chunk), force=True)
        return config, chunk

    def estimate_deepen(self, config, chunk):
        self._check_deepen_content(chunk)
        profile = config["profile_snapshots"]["vision"]
        batch_size = deep_batch_size(profile)
        calls = math.ceil(len(chunk["frame_ids"]) / batch_size)
        costs = [self.estimated_call_cost(profile, frames=min(batch_size, len(chunk["frame_ids"]) - offset) + 8,
                                           context_tokens=50000) for offset in range(0, len(chunk["frame_ids"]), batch_size)]
        cost = sum(costs) if all(c is not None for c in costs) else None
        return self.with_timing(config, {"duration_ms": chunk["end_ms"] - chunk["start_ms"], "estimated_frames": len(chunk["frame_ids"]),
                "estimated_requests": calls, "estimated_requests_min": calls, "estimated_requests_max": calls,
                "cost_estimate": cost, "currency": "USD",
                "stage_estimates": {"vision": {"requests_min": calls, "requests_max": calls}},
                "warnings": ["逐批读取该片段保留的关键帧；最多三次重试，继续复用同一片段的收藏和引用。"]})

    @staticmethod
    def _bounded_scene_profiles(context):
        # Compact before the legacy provider's whole-cast limit is checked. All
        # identities, aliases and reference policies survive every budget level.
        context = copy.deepcopy(context)
        context["profiles"] = sorted(context["profiles"], key=lambda p: p["id"])
        for profile in context["profiles"]:
            # Imported biographies initially occur twice. A pointer preserves
            # their reference-only meaning without spending the budget twice.
            if profile.get("reference_description") and profile["reference_description"] == profile.get("description"):
                profile.pop("reference_description")
                profile["reference_description_same_as"] = "description"
                context["dossier_detail_compacted"] = True

        def fits(value):
            return encoded_size(canonical(value)) <= 18000

        if fits(context):
            return context
        policies = {profile["reference_policy"] for profile in context["profiles"] if profile.get("reference_policy")}
        if len(policies) == 1:
            policy = next(iter(policies))
            if context.get("reference_policy", policy) == policy:
                context["reference_policy"] = policy
                for profile in context["profiles"]:
                    if profile.get("reference_policy") == policy:
                        profile["reference_policy"] = "context.reference_policy"
                        profile["reference_only"] = True
                context["dossier_detail_compacted"] = True
        # Name variants repeat aliases and the profile's source URL. Keep every
        # distinct name and source; compact only data already present verbatim.
        metadata_omitted = set()
        for profile in context["profiles"]:
            sources = [profile.get("external_source"), *profile.get("external_sources", [])]
            for source in sources:
                if source and source.get("source") == "bangumi":
                    for field in ("fetched_at", "image_url"):
                        if field in source:
                            source.pop(field)
                            metadata_omitted.add("external sources." + field)
            if profile.get("external_source") in profile.get("external_sources", []):
                profile["external_sources"] = [source for source in profile["external_sources"]
                                               if source != profile["external_source"]]
            names = {profile.get(field) for field in ("name", "name_cn", "name_ja", "name_en")}
            names.update(profile.get("aliases", []))
            variants = []
            for variant in profile.get("name_variants", []):
                repeated_source = any(source and variant.get("source") == source.get("source")
                                      and variant.get("source_url") == source.get("url") for source in sources)
                if repeated_source and variant.get("name") in names:
                    metadata_omitted.add("name_variants (names and sources retained in profile)")
                    continue
                if repeated_source:
                    variant.pop("source", None)
                    variant.pop("source_url", None)
                    metadata_omitted.add("name_variants.source/source_url (see external sources)")
                variants.append(variant)
            if profile.get("name_variants"):
                if variants:
                    profile["name_variants"] = variants
                else:
                    profile.pop("name_variants")
            for field in ("created_at", "updated_at", "representative"):
                if field in profile:
                    profile.pop(field)
                    metadata_omitted.add(field)
        if metadata_omitted:
            context["dossier_metadata_omitted"] = sorted(metadata_omitted)
        if fits(context):
            return context
        # Give every character a fair share before omitting prose altogether.
        # Rebuild each level from the same full text so markers never accumulate.
        complete = context
        for prose_limit, notes_limit, trait_count, trait_limit in (
                (1200, 600, 8, 100), (600, 300, 6, 80), (300, 160, 4, 60),
                (160, 80, 2, 40), (80, 40, 1, 32), (40, 20, 1, 24)):
            context = copy.deepcopy(complete)
            for profile in context["profiles"]:
                truncated = []
                for field in ("description", "reference_description", "notes"):
                    limit = notes_limit if field == "notes" else prose_limit
                    if len(profile.get(field, "")) > limit:
                        profile[field] = profile[field][:limit] + "（节选）"
                        truncated.append(field)
                for field in ("appearance", "observed_traits"):
                    if field not in profile:
                        continue
                    values = profile[field]
                    shortened = [value if len(value) <= trait_limit else value[:trait_limit] + "（节选）"
                                 for value in values[-trait_count:]]
                    if values != shortened:
                        profile[field] = shortened
                        truncated.append(field)
                if truncated:
                    profile["dossier_truncated_fields"] = truncated
            context["dossier_detail_truncated"] = True
            if fits(context):
                return context
        # Very large rosters can still require the explicit legacy fallback.
        # Keep all names/aliases, external sources and unverified-prior policies.
        optional = {"description", "reference_description", "notes", "appearance", "observed_traits",
                    "reference_description_same_as", "dossier_truncated_fields", "edited_fields"}
        context["profiles"] = [{key: value for key, value in profile.items() if key not in optional}
                               for profile in context["profiles"]]
        context["dossier_detail_omitted"] = True
        if fits(context):
            return context
        raise ValueError("完整角色名单超过上下文上限，请启用本地视觉匹配选择相关档案；角色未被静默截断")

    def _scene_context(self, asset_id, config):
        context = self.character_context(asset_id, config["character_snapshot"], scene=True)
        ids = {p["id"] for p in context["profiles"]}
        context["reference_frames"] = [f for f in context["reference_frames"] if f["character_id"] in ids]
        return context

    def _validate_scene_preparation(self, snapshot, asset_id, run_id, config, plan,
                                    transcript_total, preparation_total):
        """Check a completed stage once, without reloading the frame registry per chunk."""
        from .scenes import SceneStore

        if (not isinstance(snapshot, dict) or snapshot.get("asset_id") != asset_id
                or snapshot.get("run_id") != run_id
                or snapshot.get("dependencies") != config["scene_dependencies"]
                or type(snapshot.get("transcript_total")) is not int
                or type(snapshot.get("preparation_total")) is not int
                or snapshot["transcript_total"] != transcript_total
                or snapshot["preparation_total"] != preparation_total):
            raise ValueError("已保存的视觉分块阶段与当前任务不一致")
        chunks = snapshot.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("已保存的视觉分块阶段缺少完整片段")
        frame_lookup = {frame["id"]: frame for frame in plan["frames"]}
        cues = {cue["id"]: cue for cue in SceneStore(self.library).subtitle_records(asset_id)}
        boundary, observed = config["start_ms"], []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("已保存的视觉分块片段格式无效")
            start, end = chunk.get("start_ms"), chunk.get("end_ms")
            if (type(start) is not int or type(end) is not int or start != boundary
                    or not start < end <= config["end_ms"]
                    or chunk.get("asset_id") != asset_id or chunk.get("run_id") != run_id
                    or chunk.get("id") != SceneStore.chunk_id(asset_id, start, end)
                    or chunk.get("chunk_id") != chunk["id"]
                    or chunk.get("understanding_level") != "initial"
                    or chunk.get("dependencies") != config["scene_dependencies"]
                    or type(chunk.get("needs_deep")) is not bool
                    or not isinstance(chunk.get("summary"), str) or not chunk["summary"].strip()):
                raise ValueError("已保存的视觉分块片段边界或来源无效")
            for field in ("frame_ids", "evidence_frame_ids", "representative_frame_ids",
                          "deep_frame_ids", "subtitle_ids"):
                values = chunk.get(field, [])
                if (not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                        or len(set(values)) != len(values)):
                    raise ValueError("已保存的视觉分块证据引用无效")
            frame_ids = chunk.get("frame_ids", [])
            frames = set(frame_ids)
            if (not frame_ids or any(frame_id not in frame_lookup
                                    or not start <= frame_lookup[frame_id]["at_ms"] < end
                                    for frame_id in frame_ids)
                    or not set(chunk.get("evidence_frame_ids", [])).issubset(frames)
                    or not set(chunk.get("representative_frame_ids", [])).issubset(frames)):
                raise ValueError("已保存的视觉分块包含缺失或越界关键帧")
            deep_frames = set(chunk.get("deep_frame_ids", []))
            if deep_frames and (not deep_frames.issubset(frames)
                    or not set(chunk.get("representative_frame_ids", [])).issubset(deep_frames)
                    or chunk["deep_frame_ids"] != [frame_id for frame_id in frame_ids if frame_id in deep_frames]):
                raise ValueError("已保存的深入理解选帧与视觉分块不一致")
            entities = chunk.get("entities", [])
            if not isinstance(entities, list):
                raise ValueError("已保存的视觉分块角色实体无效")
            from .providers import Entity
            parsed_entities = [Entity.model_validate(entity) for entity in entities]
            if (len({entity.id for entity in parsed_entities}) != len(parsed_entities)
                    or any(not set(entity.evidence_frame_ids).issubset(frames) for entity in parsed_entities)):
                raise ValueError("已保存的视觉分块角色证据引用无效")
            subtitle_ids = set(chunk.get("subtitle_ids", []))
            if any(cue_id not in cues or cues[cue_id]["start_ms"] >= end
                   or cues[cue_id]["end_ms"] <= start for cue_id in subtitle_ids):
                raise ValueError("已保存的视觉分块字幕引用缺失或越界")
            for field, allowed in (("visual_facts", {"frame_ids"}), ("dialogue_facts", {"cue_ids"}),
                                   ("inferences", {"frame_ids", "cue_ids"})):
                facts = chunk.get(field)
                if not isinstance(facts, list):
                    raise ValueError("已保存的视觉分块事实格式无效")
                for fact in facts:
                    if (not isinstance(fact, dict) or not isinstance(fact.get("text"), str)
                            or not fact["text"].strip()):
                        raise ValueError("已保存的视觉分块事实文字无效")
                    refs = []
                    for key, evidence in (("frame_ids", set(chunk.get("evidence_frame_ids", []))),
                                          ("cue_ids", subtitle_ids)):
                        values = fact.get(key, [])
                        if (not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                                or not set(values).issubset(evidence) or (key not in allowed and values)):
                            raise ValueError("已保存的视觉分块事实引用无效")
                        refs.extend(values)
                    if not refs:
                        raise ValueError("已保存的视觉分块事实缺少证据")
            observed.extend(frame_ids)
            boundary = end
        if boundary != config["end_ms"] or observed != [frame["id"] for frame in plan["frames"]]:
            raise ValueError("已保存的视觉分块未完整覆盖任务范围和关键帧")
        return snapshot

    def _save_scene_preparation(self, asset_id, run_id, config, plan, chunks,
                                transcript_total, preparation_total):
        from .library import safe_id

        snapshot = {"asset_id": asset_id, "run_id": run_id,
                    "dependencies": copy.deepcopy(config["scene_dependencies"]),
                    "chunks": copy.deepcopy(chunks), "transcript_total": transcript_total,
                    "preparation_total": preparation_total}
        self._validate_scene_preparation(snapshot, asset_id, run_id, config, plan,
                                         transcript_total, preparation_total)
        target = self.library.asset_dir(asset_id) / "analyses" / safe_id(run_id) / "scene-prepared.json"
        atomic_json(target, snapshot)
        return snapshot

    def _load_scene_preparation(self, asset_id, run_id, config, plan, state,
                                transcript_total, preparation_total):
        from .library import safe_id

        directory = self.library.asset_dir(asset_id) / "analyses" / safe_id(run_id)
        snapshot = read_json(directory / "scene-prepared.json")
        if snapshot is not None:
            return self._validate_scene_preparation(snapshot, asset_id, run_id, config, plan,
                                                     transcript_total, preparation_total)
        if not state.get("published"):
            return None
        # Older runs already committed the complete preparation stage. Its own
        # initial revisions retain the original needs_deep flags and boundaries,
        # even after active pointers advance to deeper or refreshed revisions.
        paths = [path for path in (directory / "scenes").glob("*.json")
                 if not path.name.endswith(".deep.json")]
        if not paths:
            return None
        chunks = [read_json(path) for path in paths]
        if any(not isinstance(chunk, dict) for chunk in chunks):
            raise ValueError("已保存的视觉分块片段格式无效")
        chunks.sort(key=lambda chunk: (chunk.get("start_ms", -1), chunk.get("end_ms", -1)))
        return self._save_scene_preparation(asset_id, run_id, config, plan, chunks,
                                            transcript_total, preparation_total)

    async def analyze_scenes(self, job_id):
        from .scenes import SceneStore
        store = SceneStore(self.library)
        job = self.checkpoint(job_id)
        if self.independent_transcript(job["config"]) and not job.get("transcript_prefetched"):
            await self.prepare_episode_inputs(job_id)
            job = self.update(job_id, transcript_prefetched=True)
        config, asset_id, run_id = job["config"], job["asset_id"], job["run_id"]
        if (self.independent_transcript(config) and job.get("transcript_prefetched")
                and not config.get("scene_dependencies")):
            self.check_transcript_dependencies(asset_id, config)
        directory = self.library.asset_dir(asset_id) / "analyses" / run_id
        state_path = directory / "scene-state.json"
        state = read_json(state_path, {})
        # Stage checkpoints preserve completed counts. Uncommitted receipts or
        # parts are recovered locally without changing model inputs or versions.
        stage_progress = {stage: {"completed": 0, "total": None, "status": "pending"}
                          for stage in ("transcript", "preparation", "deep_analysis", "context_refresh")}
        if config.get("deepen_chunk_id"):
            for stage in ("transcript", "preparation", "context_refresh"):
                stage_progress[stage] = {"completed": 0, "total": 0, "status": "completed"}
        for stage in stage_progress:
            saved = job.get("stage_progress", {}).get(stage)
            # Independent dialogue has its own validated dependency snapshot,
            # completed before scene dependencies and the local visual prelude.
            # Other scene-stage counts need the scene dependency snapshot.
            if saved and (config.get("scene_dependencies") or
                          (stage == "transcript" and job.get("transcript_prefetched")
                           and self.independent_transcript(config))):
                stage_progress[stage] = copy.deepcopy(saved)
        if config.get("scene_dependencies") and job.get("stage_progress", {}).get("visual_identity"):
            stage_progress["visual_identity"] = copy.deepcopy(job["stage_progress"]["visual_identity"])
        self.update(job_id, stage_progress=stage_progress)
        self._check_scene_range(asset_id, config, store)
        # Inspect every preceding active block, since prompt-facing gap lists
        # are bounded and could omit an older unfinished deep analysis. The
        # current range itself is excluded so retry/manual deepening can finish it.
        pending = next((row for row in store._raw(asset_id)
                        if row["end_ms"] <= config["start_ms"] and row.get("needs_deep")), None)
        if pending is not None:
            raise ValueError(f"前序片段 {pending['start_ms'] / 1000:g}–{pending['end_ms'] / 1000:g} 秒尚待深入理解，"
                             "请先完成该片段的深入分析，再分析后续范围")
        self.library.require_source(asset_id)
        self._scene_context(asset_id, config)
        for profile in config["profile_snapshots"].values():
            if not self.profile_source_unchanged(config, profile):
                raise ValueError("模型配置已变化，请创建新任务")
        if "subtitle" in config["stages"] and not state.get("ocr_complete"):
            await self.analyze_legacy(job_id, {**config, "stages": ["subtitle"]}, skip_reindex=True)
            if self.get(job_id)["failures"]:
                raise ValueError("字幕识别未完成，连续理解已停止；重试将复用成功字幕批次")
            state["ocr_complete"] = True
            atomic_json(state_path, state)
        if not config.get("deepen_chunk_id") and not config.get("keyframe_plan_id"):
            started = time.monotonic()
            plan = await self.scan_plan(job_id, asset_id=asset_id, start_ms=config["start_ms"], end_ms=config["end_ms"])
            self._scene_metric(job_id, "keyframes", elapsed=time.monotonic() - started)
            config["keyframe_plan_id"] = plan["id"]
            self.update(job_id, config=config)
        prior = store.prior_memory(asset_id, config["start_ms"])
        dependencies = {"fingerprint": self.library.get_asset(asset_id)["fingerprint"],
                        "subtitle_digest": store.subtitle_digest(asset_id),
                        "character_revision": config["character_snapshot"]["revision"],
                        "membership_revision": config["character_snapshot"].get("membership_revision", 0),
                        "annotation_digest": annotation_digest(store, asset_id),
                        "prior_memory_digest": prior["digest"], "keyframe_plan_id": config.get("keyframe_plan_id"),
                        "strategy": STRATEGY, "prompt_versions": prompt_versions(config),
                        "deep_batch_size": deep_batch_size(config["profile_snapshots"]["vision"])}
        if visual_batch_size(config) != CORE_FRAMES:
            dependencies["visual_batch_size"] = visual_batch_size(config)
        if "identity_settings" in config:
            dependencies["identity_settings"] = digest(config["identity_settings"])
            dependencies["character_semantic_revision"] = store.dependencies(asset_id)["character_semantic_revision"]
        if config.get("scene_dependencies") and config["scene_dependencies"] != dependencies:
            raise ValueError("字幕或前序记忆已改变，请创建新任务；已保存的结果仍然保留")
        config["scene_dependencies"] = dependencies
        self.update(job_id, config=config)
        cues = [c for c in store.subtitle_records(asset_id)
                if c["end_ms"] > config["start_ms"] and c["start_ms"] < config["end_ms"]]
        def memory_for(chunks):
            return bounded_memory(prior, [store._project_semantics(asset_id, chunk) for chunk in chunks])
        if config.get("deepen_chunk_id"):
            original = state.get("original")
            if original is None:
                original = store.get(asset_id, config["deepen_chunk_id"])
                if store.revision_digest(original) != config["deepen_revision"]:
                    raise ValueError("片段已更新，请刷新后重新发起补查")
                state["original"] = original
                atomic_json(state_path, state)
            self._check_deepen_content(original)
            frames = self._scene_frames(asset_id, original["frame_ids"])
            await self.prepare_scene_identity(job_id, frames, cues)
            self._scene_progress(job_id, "deep_analysis",
                total=math.ceil(len(frames) / deep_batch_size(config["profile_snapshots"]["vision"])))
            chunk = {**original, "dependencies": dependencies, "run_id": run_id,
                     "subtitle_ids": [c["id"] for c in cues]}
            if original.get("dependencies", {}).get("subtitle_digest") != dependencies["subtitle_digest"]:
                chunk["dialogue_facts"] = [{"text": c["text"], "cue_ids": [c["id"]]} for c in cues if c["text"].strip()]
                chunk["inferences"] = []
            committed = self._load_manual_scene_commit(asset_id, run_id, config, original, state, state_path, store)
            if committed is None:
                chunk = await self._deepen_scene(job_id, chunk, frames, cues, bounded_memory(prior, []))
                self.checkpoint(job_id)
                self._check_scene_dependencies(asset_id, config, store)
                with self.library.lock:
                    chunk = store.upgrade(asset_id, run_id, chunk)
                state["deepen_committed"] = {"relative": store._relative(chunk),
                                             "revision_digest": store.revision_digest(chunk)}
                state["index_dirty"] = True
                atomic_json(state_path, state)
            else:
                self._scene_progress(job_id, "deep_analysis", completed=math.ceil(len(frames) /
                    deep_batch_size(config["profile_snapshots"]["vision"])), status="completed")
            self._repair_scene_index(job_id, asset_id, state, state_path)
        else:
            plan = (self.validated_plan(asset_id, config["start_ms"], config["end_ms"], config["keyframe_plan_id"])
                    if config.get("keyframe_plan_id") else await self.scan_plan(job_id, asset_id=asset_id,
                        start_ms=config["start_ms"], end_ms=config["end_ms"]))
            if not config.get("keyframe_plan_id"):
                config["keyframe_plan_id"] = plan["id"]
                self.update(job_id, config=config)
            paths = self.library.frame_paths(asset_id, [frame["id"] for frame in plan["frames"]])
            frames = [{**frame, "path": str(paths[frame["id"]])} for frame in plan["frames"]]
            await self.prepare_scene_identity(job_id, frames, cues)
            prep_profile = config["profile_snapshots"]["preparation"]
            batches = list(cue_batches(cues))
            batch_size = visual_batch_size(config)
            total = math.ceil(len(frames) / batch_size)
            self._scene_progress(job_id, "transcript", total=len(batches))
            self._scene_progress(job_id, "preparation", total=total)
            frame_lookup = {f["id"]: f for f in frames}
            restored = self._load_scene_preparation(asset_id, run_id, config, plan, state, len(batches), total)
            if restored is not None:
                chunks = restored["chunks"]
                self._scene_progress(job_id, "transcript", completed=len(batches), status="completed")
                self._scene_progress(job_id, "preparation", completed=total, status="completed")
                self.update(job_id, message="已恢复完整分块方案，继续深入理解", progress=max(.5, job.get("progress", 0)))
            else:
                segments, unresolved = await self.prepare_transcript_stage(
                    job_id, config, prior, cues, batches, prep_profile)
                chunks, assembled = self._load_partial_scene_preparation(
                    asset_id, run_id, config, plan, batch_size, len(batches))
                self._scene_progress(job_id, "preparation", total=total, completed=assembled)
                self.update(job_id, total=total, completed=assembled, progress=assembled / max(total, 1) * .5)
                preparation_started = time.monotonic()
                offsets = list(range(0, len(frames), batch_size))
                prepared = None
                parallel_preparation = (config.get("execution") or {}).get("preparation") == "parallel"
                if parallel_preparation:
                    frozen_memory = {**bounded_memory(prior, []), "execution_mode": "parallel",
                        "notice": "并行分块只共享已冻结的前序记忆；边界参考帧属于之前的时间范围，不能写成当前片段的事实。"}
                    result_path, result_state = self._scene_preparation_results(
                        asset_id, run_id, config, plan, prep_profile, frames, cues, segments, batch_size)
                    prepared = result_state["results"]
                    pending = [index for index in range(assembled, total) if prepared[index] is None]
                    frozen_characters = (await self._scene_local_context(job_id, self._scene_context, asset_id, config)
                                         if pending else None)
                    async def prepare_batch(position, index):
                        offset = offsets[index]
                        core = frames[offset:offset + batch_size]
                        context_frames = frames[max(0, offset - CONTEXT_FRAMES):offset]
                        a = config["start_ms"] if offset == 0 else core[0]["at_ms"]
                        b = frames[offset + batch_size]["at_ms"] if offset + batch_size < len(frames) else config["end_ms"]
                        self.update(job_id, message=f"并行视觉分块与简要索引：第 {index + 1}/{total} 组")
                        audio_context = await self._scene_local_context(job_id, self.scene_audio_context,
                            job_id, subtitle_context(cues, segments, a, b), a, b)
                        character_context = await self._scene_local_context(job_id, self.scene_identity_context,
                            job_id, frozen_characters, core)
                        result = await self._scene_call(job_id, "preparation", str(index), prep_profile,
                            self.providers.prepare_visual, [core, context_frames, audio_context,
                                copy.deepcopy(frozen_memory), character_context],
                            images=math.ceil((len(core) + len(context_frames)) / 16))
                        prepared[index] = result
                        atomic_json(result_path, result_state)
                        return result
                    def preparation_done(index, result):
                        finished = assembled + sum(value is not None for value in prepared[assembled:])
                        self._scene_progress(job_id, "preparation", completed=finished)
                        self.update(job_id, completed=finished, progress=finished / max(total, 1) * .5)
                    self._scene_progress(job_id, "preparation", status="running")
                    preparation_done(0, None)
                    await bounded_map(pending, prepare_batch,
                        execution_width(config, "preparation", self.providers.get(prep_profile["id"])),
                        lambda: self.checkpoint(job_id), preparation_done)

                for index in range(assembled, len(offsets)):
                    offset = offsets[index]
                    await yield_to_io()
                    self.checkpoint(job_id)
                    core, context_frames = frames[offset:offset + batch_size], frames[max(0, offset - CONTEXT_FRAMES):offset]
                    a = config["start_ms"] if offset == 0 else core[0]["at_ms"]
                    b = frames[offset + batch_size]["at_ms"] if offset + batch_size < len(frames) else config["end_ms"]
                    local_cues = [c for c in cues if c["end_ms"] > a and c["start_ms"] < b]
                    sub_context = await self._scene_local_context(job_id, self.scene_audio_context,
                        job_id, subtitle_context(cues, segments, a, b), a, b)
                    if prepared is not None:
                        self.update(job_id, message=f"整理已完成的视觉分块：{index + 1}/{total}")
                        result = prepared[index]
                    else:
                        self._scene_progress(job_id, "preparation", status="running")
                        self.update(job_id, message=f"视觉分块与简要索引：{index + 1}/{total}")
                        version = prompt_versions(config)["preparation"]
                        frozen = read_json(directory / "scene-inputs" / f"{digest(['preparation', version, str(index)])}.json")
                        if frozen is not None:
                            args = frozen["args"]
                        else:
                            character_context = await self._scene_local_context(
                                job_id, self._scene_context, asset_id, config)
                            character_context = await self._scene_local_context(
                                job_id, self.scene_identity_context, job_id, character_context, core)
                            args = [core, context_frames, sub_context, memory_for(chunks), character_context]
                        result = await self._scene_call(job_id, "preparation", str(index), prep_profile,
                            self.providers.prepare_visual, args,
                            images=math.ceil((len(core) + len(context_frames)) / 16))
                    for number, raw in enumerate(result["chunks"]):
                        raw = copy.deepcopy(raw)
                        namespace_entities(raw, f"v{index}_{number}_")
                        chosen = [frame_lookup[fid] for fid in raw["frame_ids"]]
                        start = a if number == 0 else chosen[0]["at_ms"]
                        next_raw = result["chunks"][number + 1] if number + 1 < len(result["chunks"]) else None
                        stop = frame_lookup[next_raw["frame_ids"][0]]["at_ms"] if next_raw else b
                        chunk_id = "scene_" + digest([asset_id, start, stop])
                        chunk = {**raw, "id": chunk_id, "chunk_id": chunk_id, "asset_id": asset_id, "run_id": run_id,
                                 "scene_key": raw.get("scene_key", ""),
                                 "start_ms": start, "end_ms": stop, "shot_ids": list(dict.fromkeys(f["shot_id"] for f in chosen)),
                                 "evidence_frame_ids": list(raw["frame_ids"]),
                                 "subtitle_ids": [c["id"] for c in local_cues if c["end_ms"] > start and c["start_ms"] < stop],
                                 "understanding_level": "initial", "dependencies": dependencies,
                                 "audio_evidence": await self._scene_local_context(
                                     job_id, self.scene_audio_evidence, job_id, start, stop),
                                 "provenance": {"strategy": STRATEGY, "keyframe_plan_id": plan["id"],
                                                "profile_id": prep_profile["id"], "model": prep_profile.get("model", "")},
                                 "uncertainties": [] if raw.get("content_kind") == "credits" else result.get("unresolved", []),
                                 "memory_before": copy.deepcopy(frozen_memory) if parallel_preparation else memory_for(chunks)}
                        # Keep transcript-only semantic coverage even when sparse frames
                        # required a bounded cue sample for the visual model.
                        own_cues = set(chunk["subtitle_ids"])
                        for segment in segments:
                            if (chunk.get("content_kind") != "credits" and segment.get("content_kind") != "credits"
                                    and set(segment["cue_ids"]).issubset(own_cues)):
                                fact = {"text": segment["summary"], "cue_ids": segment["cue_ids"]}
                                if fact not in chunk["dialogue_facts"]:
                                    chunk["dialogue_facts"].append(fact)
                        if (number == 0 and raw.get("continues_previous") and chunks
                                and chunk.get("content_kind") != "credits" and chunks[-1].get("content_kind") != "credits"):
                            previous = chunks.pop()
                            required_frames = list(dict.fromkeys(deep_frame_ids(previous) + deep_frame_ids(chunk)))
                            chunk["start_ms"] = previous["start_ms"]
                            chunk["summary"] = (previous["summary"] + "；" + chunk["summary"])[-6000:]
                            for key in ("frame_ids", "evidence_frame_ids", "subtitle_ids", "shot_ids"):
                                chunk[key] = list(dict.fromkeys(previous.get(key, []) + chunk.get(key, [])))
                            for key in ("visual_facts", "dialogue_facts", "inferences", "entities", "events"):
                                chunk[key] = previous.get(key, []) + chunk.get(key, [])
                            chunk["needs_deep"] = previous["needs_deep"] or chunk["needs_deep"]
                            merge_review_flags(chunk, previous)
                            chunk["audio_evidence"] = await self._scene_local_context(
                                job_id, self.scene_audio_evidence, job_id, chunk["start_ms"], stop)
                            chunk["deep_frame_ids"] = required_frames
                            chunk["scene_key"] = chunk["scene_key"] or previous.get("scene_key", "")
                            chunk["memory_before"] = previous["memory_before"]
                            chunk["id"] = chunk["chunk_id"] = "scene_" + digest([asset_id, chunk["start_ms"], stop])
                            reps = list(dict.fromkeys(previous["representative_frame_ids"] + chunk["representative_frame_ids"]))
                            chunk["representative_frame_ids"] = reps if len(reps) <= 8 else reps[:7] + [reps[-1]]
                        chunk["memory_after"] = memory_for(chunks + [chunk])
                        chunks.append(chunk)
                    self._save_partial_scene_preparation(asset_id, run_id, config, chunks, len(batches), index + 1)
                    if prepared is None:
                        self._scene_progress(job_id, "preparation", completed=index + 1)
                        self.update(job_id, completed=index + 1, progress=(index + 1) / max(total, 1) * .5)
                self._scene_metric(job_id, "preparation", wall_elapsed=time.monotonic() - preparation_started)
                if not chunks:
                    raise ValueError("此范围没有可分析的关键帧")
                link_similar_scenes(chunks)
                self._save_scene_preparation(asset_id, run_id, config, plan, chunks, len(batches), total)
            if not state.get("published"):
                self._check_scene_dependencies(asset_id, config, store)
                with self.library.lock:
                    self._scene_context(asset_id, config)
                    published = []
                    for chunk in chunks:
                        if chunk.get("entities"):
                            chunk = self.characters.apply_observation(asset_id,
                                {**chunk, "window_id": chunk["id"], "shot_id": "scene"},
                                config["character_snapshot"]["revision"])
                        published.append(chunk)
                    store.publish(asset_id, run_id, published, config["start_ms"], config["end_ms"])
                state["published"] = True
                state["index_dirty"] = True
                atomic_json(state_path, state)
            await self.prepare_scene_lrasd(job_id, chunks, cues, frames, plan.get("shots", []))
            self._repair_scene_index(job_id, asset_id, state, state_path)
            batch_size = deep_batch_size(config["profile_snapshots"]["vision"])
            deep_total = sum(math.ceil(len(deep_frame_ids(c, all_frames=config.get("character_reanalysis"))) / batch_size) for c in chunks
                             if self._understanding_kind(c, config, False) == "deep_analysis")
            deep_seen, refresh_total, refresh_completed = False, 0, 0
            for chunk in chunks:
                if self._understanding_kind(chunk, config, False) == "deep_analysis":
                    deep_seen = True
                elif deep_seen and chunk.get("content_kind") != "credits":
                    refresh_total += 1
            cursor, completed_chunks, active_chunks = self._load_scene_understanding(
                asset_id, run_id, config, chunks, state, state_path, store)
            self._repair_scene_index(job_id, asset_id, state, state_path)
            context_changed = cursor["context_changed"]
            refresh_completed = cursor["refresh_completed"]
            self._scene_progress(job_id, "deep_analysis", total=deep_total,
                                 completed=cursor["deep_completed"])
            self._scene_progress(job_id, "context_refresh", total=refresh_total,
                                 completed=refresh_completed)
            self.update(job_id, deep_request_count=deep_total, total=len(chunks),
                        completed=cursor["completed"], progress=.5 + .5 * cursor["completed"] / len(chunks))
            for index in range(cursor["completed"], len(chunks)):
                chunk = copy.deepcopy(active_chunks[chunks[index]["id"]])
                await yield_to_io()
                self.checkpoint(job_id)
                memory = bounded_memory(prior, completed_chunks)
                kind = self._understanding_kind(chunk, config, context_changed)
                if kind == "deep_analysis":
                    selected_ids = deep_frame_ids(chunk, all_frames=config.get("character_reanalysis"))
                    chunk = await self._deepen_scene(job_id, chunk, [frame_lookup[fid] for fid in selected_ids], cues, memory)
                    self._check_scene_dependencies(asset_id, config, store)
                    chunk = store.upgrade(asset_id, run_id, chunk, invalidate_dependents=False)
                    context_changed = True
                elif kind == "context_refresh":
                    self._scene_progress(job_id, "context_refresh", status="running")
                    prepared = {key: chunk[key] for key in ("chunk_id", "start_ms", "end_ms", "summary", "visual_facts", "dialogue_facts", "inferences")}
                    for field in ("visual_facts", "dialogue_facts", "inferences"):
                        prepared[field] = prepared[field][-16:]
                    result = await self._scene_call(job_id, "context_refresh", chunk["id"], prep_profile,
                        self.providers.reconcile_scene, [prepared, memory])
                    chunk = {**chunk, "summary": result["summary"], "inferences": result["inferences"],
                             "uncertainties": result.get("unresolved", []), "memory_before": memory}
                    chunk["memory_after"] = bounded_memory({}, [chunk])
                    self._check_scene_dependencies(asset_id, config, store)
                    chunk = store.publish(asset_id, run_id + "_context", [chunk], chunk["start_ms"], chunk["end_ms"])[0]
                    refresh_completed += 1
                    self._scene_progress(job_id, "context_refresh", completed=refresh_completed)
                self._commit_scene_understanding(cursor, chunk, kind,
                    self.get(job_id)["stage_progress"]["deep_analysis"]["completed"], state, state_path, store)
                self._repair_scene_index(job_id, asset_id, state, state_path)
                completed_chunks.append(store._project_semantics(asset_id, chunk))
                completed_chunks = completed_chunks[-8:]
                self.update(job_id, completed=index + 1, progress=.5 + .5 * (index + 1) / len(chunks))
        embedding = config.get("bindings", {}).get("embedding")
        if embedding:
            await self.reindex(job_id, embedding_profile_id=embedding)

    def _check_scene_dependencies(self, asset_id, config, store):
        self._check_scene_range(asset_id, config, store)
        self._scene_context(asset_id, config)
        if config["scene_dependencies"].get("prompt_versions") != prompt_versions(config):
            raise ValueError("理解提示词已升级，请创建新任务以保留清晰的版本与缓存边界")
        if store.subtitle_digest(asset_id) != config["scene_dependencies"]["subtitle_digest"]:
            raise ValueError("分析期间字幕已改变，旧结果未提交；请创建新任务")
        if store.prior_memory(asset_id, config["start_ms"])["digest"] != config["scene_dependencies"]["prior_memory_digest"]:
            raise ValueError("分析期间前序记忆已改变，旧结果未提交；请创建新任务")
        if annotation_digest(store, asset_id) != config["scene_dependencies"]["annotation_digest"]:
            raise ValueError("分析期间人工修订已改变，请创建新任务；已取得的结果保留")
        if config.get("deepen_chunk_id"):
            current = store.get(asset_id, config["deepen_chunk_id"])
            if store.revision_digest(current) != config["deepen_revision"]:
                jobs = [j for j in self.list() if j["config"] == config and j["run_id"] == current["run_id"]]
                if not jobs:
                    raise ValueError("此片段已有较新的分析结果，旧补查任务不会覆盖；请刷新后重新发起")

    def _scene_frames(self, asset_id, frame_ids):
        registry = read_json(self.library.asset_dir(asset_id) / "frames" / "registry.json", {})
        paths = self.library.frame_paths(asset_id, frame_ids, registry=registry)
        return [{**registry[fid], "id": fid, "path": str(paths[fid])} for fid in frame_ids]

    async def _deepen_scene(self, job_id, chunk, frames, cues, memory):
        from .models import Observation, normalize_observation_evidence
        job = self.checkpoint(job_id)
        config, asset_id = job["config"], job["asset_id"]
        profile = config["profile_snapshots"]["vision"]
        batch_size = deep_batch_size(profile)
        progress_base = self.get(job_id).get("stage_progress", {}).get("deep_analysis", {}).get("completed", 0)
        self._scene_progress(job_id, "deep_analysis", status="running")
        target = self.library.asset_dir(asset_id) / "analyses" / job["run_id"] / "scene-deep" / f"{chunk['id']}.json"
        completed = read_json(target)
        if completed:
            self._scene_progress(job_id, "deep_analysis", completed=progress_base + math.ceil(len(frames) / batch_size))
            return completed
        parts = []

        def context_with_parts():
            # Part observations are durable staging data, never published scenes.
            # Their registered identities must still be available to the next
            # model call before the complete chunk becomes active.
            with self.library.lock:
                context = self._scene_context(asset_id, config)
                registry = self.characters._read(context["scope_id"])
                recent = list(dict.fromkeys(entity["character_id"] for part in reversed(parts)
                    for entity in reversed(part.get("entities", [])) if entity.get("character_id")))
                existing = {entry["id"]: entry for entry in context["profiles"]}
                fields = ("id", "name", "aliases", "description", "notes", "appearance", "user_edited",
                          "edited_fields", "representative", "updated_at", "created_at")
                policy = context.get("cast_policy") or self.characters.cast_policy(registry)
                transient = {}
                for identity in recent:
                    stored = registry["profiles"].get(identity)
                    if stored is None:
                        raise ValueError("已完成分批的人物档案已变化，请创建新任务")
                    if not self.characters.profile_allowed(stored, policy):
                        continue
                    entry = {**copy.deepcopy(existing.get(identity, {})),
                             **{key: copy.deepcopy(stored[key]) for key in fields}}
                    for key in ("name_cn", "name_ja", "name_en", "name_variants", "external_source",
                                "external_sources", "reference_description"):
                        if key in stored:
                            entry[key] = copy.deepcopy(stored[key])
                    entry["observed_traits"] = list(stored["appearance"])
                    for field in ("appearance", "observed_traits"):
                        if field not in entry.get("edited_fields", []):
                            entry[field] = entry[field][-8:]
                    transient[identity] = entry
                recent = [identity for identity in recent if identity in transient]
                identities = list(dict.fromkeys([*recent, *existing]))
                context["profiles"] = [transient.get(identity, existing.get(identity)) for identity in identities]
                references = {}
                for identity in recent:
                    representative = transient[identity].get("representative")
                    if representative:
                        references[identity] = {"id": "reference:" + identity, "character_id": identity,
                            "path": str(self.library.frame_path(representative["asset_id"], representative["frame_id"])),
                            "at_ms": 0, "box": representative.get("box")}
                for reference in context["reference_frames"]:
                    references.setdefault(reference["character_id"], reference)
                allowed = {entry["id"] for entry in context["profiles"]}
                context["reference_frames"] = [reference for identity, reference in references.items() if identity in allowed][:8]
                context = self._bounded_scene_profiles(context)
                allowed = {entry["id"] for entry in context["profiles"]}
                context["reference_frames"] = [reference for reference in context["reference_frames"]
                                               if reference["character_id"] in allowed]
                return context

        for index, offset in enumerate(range(0, len(frames), batch_size)):
            await yield_to_io()
            self.checkpoint(job_id)
            current = frames[offset:offset + batch_size]
            start = chunk["start_ms"] if offset == 0 else current[0]["at_ms"]
            end = frames[offset + batch_size]["at_ms"] if offset + batch_size < len(frames) else chunk["end_ms"]
            part_path = target.parent / "parts" / chunk["id"] / f"{index}.json"
            part = read_json(part_path)
            if part is not None:
                part = normalize_observation_evidence(part, [frame["id"] for frame in current])
                Observation.model_validate(part)
                if (part.get("asset_id") != asset_id or part.get("run_id") != job["run_id"]
                        or part.get("id") != chunk["id"] or part.get("start_ms") != start or part.get("end_ms") != end
                        or part.get("input_frame_ids") != [frame["id"] for frame in current]):
                    raise ValueError("已保存的深入理解分批与当前任务不一致")
                parts.append(part)
                self._scene_progress(job_id, "deep_analysis", completed=progress_base + index + 1)
                continue
            local_subtitles = await self._scene_local_context(job_id, self.scene_audio_context,
                job_id, subtitle_context(cues, [], start, end), start, end)
            context = await self._scene_local_context(job_id, context_with_parts)
            context = await self._scene_local_context(job_id, self.scene_identity_context, job_id, context, current)
            self.update(job_id, message=f"深入理解 {start / 1000:.1f}–{end / 1000:.1f} 秒")
            part = await self._scene_call(job_id, "deep_analysis", [chunk["id"], index], profile,
                self.providers.analyze, [current, start, end, context, local_subtitles, memory],
                images=len(current) + len(context["reference_frames"]))
            # Namespace local entities when combining several model responses.
            ids = {entity["id"]: f"p{index}_{entity['id']}" for entity in part.get("entities", [])}
            for entity in part.get("entities", []):
                entity["id"] = ids[entity["id"]]
            for event in part.get("events", []):
                for key in ("actor_ids", "target_ids"):
                    event[key] = [ids[value] for value in event.get(key, [])]
            for position in part.get("spatial_observations", []):
                for key in ("subject_id", "object_id"):
                    if position.get(key):
                        position[key] = ids[position[key]]
            part["visual_facts"] = [{"text": part["summary"], "frame_ids": part["evidence_frame_ids"]}]
            part.update(id=chunk["id"], asset_id=asset_id, run_id=job["run_id"], window_id=chunk["id"], shot_id="scene",
                        start_ms=start, end_ms=end, input_frame_ids=[frame["id"] for frame in current],
                        new_character_entity_ids=[entity["id"] for entity in part.get("entities", [])
                                                  if entity.get("character_id") is None])
            def save_part():
                with self.library.lock:
                    self._scene_context(asset_id, config)
                    # Stable chunk + namespaced entity IDs make a crash after registry
                    # persistence but before the part file safe to replay from receipt.
                    saved = self.characters.apply_observation(asset_id, part, config["character_snapshot"]["revision"])
                    atomic_json(part_path, saved)
                    return saved
            # A successful paid receipt owns its local commit through shutdown.
            # Do not let a pause discard the part between accounting and saving.
            part = await finish_on_cancel(asyncio.to_thread(save_part))
            parts.append(part)
            self._scene_progress(job_id, "deep_analysis", completed=progress_base + index + 1)
            self.checkpoint(job_id)
        result = {**chunk, "run_id": job["run_id"], "summary": "；".join(p["summary"] for p in parts),
                  "understanding_level": "deep", "needs_deep": False, "needs_action_deep": False, "reason": "",
                  "memory_before": memory, "window_id": chunk["id"], "shot_id": "scene"}
        for key in ("entities", "events", "spatial_observations", "uncertainties", "visual_facts"):
            result[key] = [value for part in parts for value in part.get(key, [])]
        result["provenance"] = {**chunk.get("provenance", {}), "deep_profile_id": profile["id"],
                                "deep_model": profile.get("model", ""), "input_memory_digest": digest(memory),
                                "deep_analyzed_frame_ids": [frame["id"] for frame in frames],
                                "deep_batch_size": batch_size}
        # Keep a stable current-evidence set for character validation and portraits.
        result["evidence_frame_ids"] = list(chunk["frame_ids"])
        def save_result():
            with self.library.lock:
                self._scene_context(asset_id, config)
                # Replaying an allocation is different from claiming a confident
                # visual match: first appearances retain their original confidence.
                # The stable run/chunk/entity tuple reconstructs the same registry ID.
                created_here = {identity for part in parts for identity in part.get("new_character_entity_ids", [])}
                for entity in result["entities"]:
                    if entity["id"] in created_here:
                        entity["character_id"] = None
                saved = self.characters.apply_observation(asset_id, result, config["character_snapshot"]["revision"])
                saved["memory_after"] = bounded_memory({}, [saved])
                atomic_json(target, saved)
                return saved
        return await finish_on_cancel(asyncio.to_thread(save_result))
