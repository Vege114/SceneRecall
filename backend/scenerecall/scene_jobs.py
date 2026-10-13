"""Evidence-bound scene preparation and selective deep analysis.

Provider receipts are durable before accounting/publication. A retry replays the
same bounded input memory, never a later automatically enriched character state.
"""
from __future__ import annotations

import copy
import math
import time

from .library import atomic_json, digest, read_json, uid
from .execution import bounded_map, execution_width, finish_on_cancel

STRATEGY = "scene-chunks-v1"
CORE_FRAMES = 56
CONTEXT_FRAMES = 8
DEEP_FRAMES = 24
TRANSCRIPT_CUES = 100
TRANSCRIPT_BYTES = 24000
MEMORY_BYTES = 18000


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


def prompt_versions():
    from .scene_preparation import (CONTEXTUAL_VISION_PROMPT_VERSION, RECONCILIATION_PROMPT_VERSION,
                                    SUBTITLE_PREPARATION_PROMPT_VERSION, VISUAL_PREPARATION_PROMPT_VERSION)
    return {"transcript": SUBTITLE_PREPARATION_PROMPT_VERSION, "preparation": VISUAL_PREPARATION_PROMPT_VERSION,
            "deep_analysis": CONTEXTUAL_VISION_PROMPT_VERSION, "context_refresh": RECONCILIATION_PROMPT_VERSION}


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
                 for s in segments if set(s["cue_ids"]) & ids]
    while summaries and encoded_size(summaries) > 5000:
        summaries.pop()
    return {"cues": selected, "segments": summaries, "start_ms": start, "end_ms": end,
            "omitted_cues": len(local) - len(selected),
            "notice": "原始字幕完整保留；未展示的台词不能补写为画面事实"}


class SceneJobMixin:
    def _scene_progress(self, job_id, stage, **changes):
        """Current execution progress, independent of durable paid-call accounting."""
        progress = copy.deepcopy(self.get(job_id).get("stage_progress"))
        if progress is None:
            return
        item = progress[stage]
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

    def _scene_metric(self, job_id, stage, *, elapsed=0, requests=0, cache=0, usage=None, wall_elapsed=None):
        metrics = copy.deepcopy(self.get(job_id).get("stage_metrics", {}))
        item = metrics.setdefault(stage, {"request_count": 0, "elapsed_ms": 0, "cache_hits": 0, "usage": {}})
        item["elapsed_ms"] += round(elapsed * 1000)
        if wall_elapsed is not None:
            item["wall_elapsed_ms"] = round(wall_elapsed * 1000)
        item["request_count"] += requests
        item["cache_hits"] += cache
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
            released = self._release_reservation(job, receipt.get("call_id"))
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
        from .jobs import model_identity
        from .scene_preparation import (CONTEXTUAL_VISION_PROMPT_VERSION, SUBTITLE_PREPARATION_PROMPT_VERSION,
                                        VISUAL_PREPARATION_PROMPT_VERSION)
        versions = {"transcript": SUBTITLE_PREPARATION_PROMPT_VERSION,
                    "preparation": VISUAL_PREPARATION_PROMPT_VERSION, "deep_analysis": CONTEXTUAL_VISION_PROMPT_VERSION}
        if stage == "context_refresh":
            from .scene_preparation import RECONCILIATION_PROMPT_VERSION
            versions[stage] = RECONCILIATION_PROMPT_VERSION
        job = self.checkpoint(job_id)
        from .scenes import SceneStore
        self._check_scene_dependencies(job["asset_id"], job["config"], SceneStore(self.library))
        root = self.library.asset_dir(job["asset_id"])
        run = root / "analyses" / job["run_id"]
        inputs = run / "scene-inputs" / f"{digest([stage, versions[stage], unit])}.json"
        frozen = read_json(inputs)
        if frozen is None:
            frozen = {"args": args, "images": images}
            atomic_json(inputs, frozen)
        args, images = frozen["args"], frozen["images"]
        key = digest([STRATEGY, stage, versions[stage], unit, job["config"]["scene_dependencies"],
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
            return copy.deepcopy(receipt["data"])
        self.update(job_id, stage=stage, pending_scene_receipt=str(output.relative_to(root)))
        call_id = uid("call")
        await self.reserve_call(job_id, profile, call_id=call_id, stage=stage, frames=images,
                                receipt_path=str(output.relative_to(self.library.root)),
                                context_tokens=encoded_size(canonical(args)))
        async def perform():
            started = time.monotonic()
            try:
                result = await self.invoke_profile(profile, method, *args)
            except Exception as exc:
                self.account(job_id, exc, call_id)
                self._scene_metric(job_id, stage, elapsed=time.monotonic() - started,
                                   requests=getattr(exc, "request_count", 0), usage=getattr(exc, "usage", {}))
                raise
            receipt = {"receipt_id": key, "call_id": call_id, "job_id": job_id, "stage": stage, "data": result.data,
                       "usage": result.usage, "request_count": result.request_count, "cost": result.cost,
                       "elapsed_ms": round((time.monotonic() - started) * 1000)}
            atomic_json(output, receipt)
            self._account_scene_receipt(job_id, receipt)
            return copy.deepcopy(result.data)
        return await finish_on_cancel(perform())

    def estimate_scenes(self, config, plan):
        from .scenes import SceneStore
        from .scene_preparation import MAX_PREPARATION_OUTPUT_TOKENS
        frames = len(plan["frames"])
        cues = [c for c in SceneStore(self.library).subtitle_records(config["asset_id"])
                if c["end_ms"] > config["start_ms"] and c["start_ms"] < config["end_ms"]]
        transcript = len(list(cue_batches(cues)))
        prep = math.ceil(frames / CORE_FRAMES)
        ocr_frames = (math.ceil((config["end_ms"] - config["start_ms"]) / 1000 * config["subtitle_fps"])
                      if "subtitle" in config["stages"] else 0)
        ocr = math.ceil(ocr_frames / 8)
        # OCR may produce one cue per sample; estimate a range before it exists.
        transcript_max = max(transcript, math.ceil(ocr_frames / TRANSCRIPT_CUES))
        minimum, maximum = prep + transcript + ocr, prep + transcript_max + ocr + 2 * frames
        stages = {"transcript": {"requests_min": transcript, "requests_max": transcript_max},
                  "preparation": {"requests_min": prep, "requests_max": prep},
                  "vision": {"requests_min": 0, "requests_max": frames},
                  "context_refresh": {"requests_min": 0, "requests_max": frames},
                  "subtitle": {"requests_min": 0 if ocr else 0, "requests_max": ocr}}
        def price(stage, images, text):
            profile = config["profile_snapshots"][stage]
            value = self.estimated_call_cost(profile, frames=images, context_tokens=text)
            if value is not None and stage == "preparation":
                value += (MAX_PREPARATION_OUTPUT_TOKENS - 4096) * profile["output_price_per_million"] / 1_000_000
            return value

        ptext, pvisual = price("preparation", 0, 32000), price("preparation", 4, 50000)
        pdeep = price("vision", deep_batch_size(config["profile_snapshots"]["vision"]) + 8, 50000)
        pocr = price("subtitle", 8, 0) if ocr else 0
        low = high = None
        if all(value is not None for value in (ptext, pvisual, pdeep, pocr)):
            low = transcript * ptext + prep * pvisual + ocr * pocr
            high = transcript_max * ptext + prep * pvisual + ocr * pocr + frames * (pdeep + ptext)
        return {"duration_ms": config["end_ms"] - config["start_ms"], "estimated_frames": frames + ocr_frames,
                "estimated_requests": minimum, "estimated_requests_min": minimum, "estimated_requests_max": maximum,
                "cost_estimate": low, "cost_estimate_min": low, "cost_estimate_max": high,
                "currency": "USD", "stage_estimates": stages,
                "warnings": ["深入理解及后续文字上下文更新数量由分块结果决定；上限按每帧独立复杂片段保守计算，不含重试和向量索引。",
                             "先复用本机关键帧并建立简要索引，复杂片段继续深入分析；费用与耗时以实际调用为准。"]}

    def deepen_config(self, asset_id, chunk_id, payload, settings):
        from .models import AnalysisInput
        from .scenes import SceneStore
        store = SceneStore(self.library)
        chunk = store.get(asset_id, chunk_id)
        request = AnalysisInput(asset_id=asset_id, analysis_strategy=STRATEGY,
                                start_ms=chunk["start_ms"], end_ms=chunk["end_ms"],
                                max_requests=payload.max_requests, max_cost=payload.max_cost,
                                model_overrides=payload.model_overrides)
        # Manual deepening needs only vision; no new preparation work is performed.
        _, config = self.validate(request, settings, scene_deepening=True)
        config.update(analysis_strategy=STRATEGY, sampling_strategy=STRATEGY,
                      deepen_chunk_id=chunk_id, deepen_revision=store.revision_digest(chunk), force=True)
        return config, chunk

    def estimate_deepen(self, config, chunk):
        profile = config["profile_snapshots"]["vision"]
        batch_size = deep_batch_size(profile)
        calls = math.ceil(len(chunk["frame_ids"]) / batch_size)
        costs = [self.estimated_call_cost(profile, frames=min(batch_size, len(chunk["frame_ids"]) - offset) + 8,
                                           context_tokens=50000) for offset in range(0, len(chunk["frame_ids"]), batch_size)]
        cost = sum(costs) if all(c is not None for c in costs) else None
        return {"duration_ms": chunk["end_ms"] - chunk["start_ms"], "estimated_frames": len(chunk["frame_ids"]),
                "estimated_requests": calls, "estimated_requests_min": calls, "estimated_requests_max": calls,
                "cost_estimate": cost, "currency": "USD",
                "stage_estimates": {"vision": {"requests_min": calls, "requests_max": calls}},
                "warnings": ["逐批读取该片段保留的关键帧；最多三次重试，继续复用同一片段的收藏和引用。"]}

    @staticmethod
    def _bounded_scene_profiles(context):
        # Compact before the legacy provider's whole-cast limit is checked. The
        # complete human dossier remains stored locally; omissions are explicit.
        context = copy.deepcopy(context)
        context["profiles"] = sorted(context["profiles"], key=lambda p: p.get("updated_at", ""), reverse=True)[:16]
        for profile in context["profiles"]:
            for field in ("description", "notes"):
                if len(profile.get(field, "")) > 1200:
                    profile[field] = profile[field][:1200] + "（节选，后文未纳入本次上下文）"
            for field in ("appearance", "observed_traits", "aliases"):
                values = profile.get(field, [])
                selected = values[-8:]
                shortened = [value if len(value) <= 100 else value[:100] + "（节选）" for value in selected]
                if values != shortened:
                    profile["notes"] = profile.get("notes", "") + f"（{field} 仅展示部分，完整人工档案仍保留）"
                profile[field] = shortened
        while encoded_size(canonical(context)) > 18000 and len(context["profiles"]) > 1:
            context["profiles"].pop()
        if encoded_size(canonical(context)) > 18000:
            raise ValueError("单个人物档案摘要超过上下文上限，请缩短人物名称或档案内容")
        return context

    def _scene_context(self, asset_id, config):
        context = self.character_context(asset_id, config["character_snapshot"], scene=True)
        ids = {p["id"] for p in context["profiles"]}
        context["reference_frames"] = [f for f in context["reference_frames"] if f["character_id"] in ids]
        return context

    async def analyze_scenes(self, job_id):
        from .scenes import SceneStore
        store = SceneStore(self.library)
        job = self.checkpoint(job_id)
        config, asset_id, run_id = job["config"], job["asset_id"], job["run_id"]
        # Replayed receipts/parts rebuild these absolute per-execution counts.
        # Nothing here enters model inputs, receipt identities or scene versions.
        stage_progress = {stage: {"completed": 0, "total": None, "status": "pending"}
                          for stage in ("transcript", "preparation", "deep_analysis", "context_refresh")}
        if config.get("deepen_chunk_id"):
            for stage in ("transcript", "preparation", "context_refresh"):
                stage_progress[stage] = {"completed": 0, "total": 0, "status": "completed"}
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
        directory = self.library.asset_dir(asset_id) / "analyses" / run_id
        state_path = directory / "scene-state.json"
        state = read_json(state_path, {})
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
                        "strategy": STRATEGY, "prompt_versions": prompt_versions(),
                        "deep_batch_size": deep_batch_size(config["profile_snapshots"]["vision"])}
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
            frames = self._scene_frames(asset_id, original["frame_ids"])
            self._scene_progress(job_id, "deep_analysis",
                total=math.ceil(len(frames) / deep_batch_size(config["profile_snapshots"]["vision"])))
            chunk = {**original, "dependencies": dependencies, "run_id": run_id,
                     "subtitle_ids": [c["id"] for c in cues]}
            if original.get("dependencies", {}).get("subtitle_digest") != dependencies["subtitle_digest"]:
                chunk["dialogue_facts"] = [{"text": c["text"], "cue_ids": [c["id"]]} for c in cues if c["text"].strip()]
                chunk["inferences"] = []
            chunk = await self._deepen_scene(job_id, chunk, frames, cues, bounded_memory(prior, []))
            self.checkpoint(job_id)
            self._check_scene_dependencies(asset_id, config, store)
            with self.library.lock:
                store.upgrade(asset_id, run_id, chunk)
            self._sync_scene_index(job_id, asset_id)
        else:
            plan = (self.validated_plan(asset_id, config["start_ms"], config["end_ms"], config["keyframe_plan_id"])
                    if config.get("keyframe_plan_id") else await self.scan_plan(job_id, asset_id=asset_id,
                        start_ms=config["start_ms"], end_ms=config["end_ms"]))
            if not config.get("keyframe_plan_id"):
                config["keyframe_plan_id"] = plan["id"]
                self.update(job_id, config=config)
            paths = self.library.frame_paths(asset_id, [frame["id"] for frame in plan["frames"]])
            frames = [{**frame, "path": str(paths[frame["id"]])} for frame in plan["frames"]]
            prep_profile = config["profile_snapshots"]["preparation"]
            batches = list(cue_batches(cues))
            total = math.ceil(len(frames) / CORE_FRAMES)
            self._scene_progress(job_id, "transcript", total=len(batches))
            self._scene_progress(job_id, "preparation", total=total)
            transcript_started = time.monotonic()
            segments, unresolved = [], []
            transcript_width = execution_width(config, "transcript", self.providers.get(prep_profile["id"]))
            if (config.get("execution") or {}).get("transcript") == "parallel":
                frozen_prior = bounded_memory(prior, [])
                async def prepare_transcript(index, batch):
                    previous = batches[index - 1][-20:] if index else []
                    previous = copy.deepcopy(previous)
                    while previous and encoded_size(previous) > 6000:
                        previous.pop(0)
                    story = {"prior": copy.deepcopy(frozen_prior), "previous_cues": previous,
                             "execution_mode": "parallel", "previous_segments": [], "unresolved": [],
                             "notice": "并行批次只共享已冻结的前序记忆与原始相邻台词；本组之外的台词仅作上下文，不应写入本组证据。"}
                    self.update(job_id, message=f"并行台词预理解：第 {index + 1}/{len(batches)} 组")
                    return await self._scene_call(job_id, "transcript", str(index), prep_profile,
                        self.providers.prepare_subtitles, [batch, story])
                self._scene_progress(job_id, "transcript", status="running")
                finished = 0
                def transcript_done(index, result):
                    nonlocal finished
                    finished += 1
                    self._scene_progress(job_id, "transcript", completed=finished)
                results = await bounded_map(batches, prepare_transcript, transcript_width,
                    lambda: self.checkpoint(job_id), transcript_done)
                for result in results:
                    segments.extend(result["segments"])
                    unresolved.extend(result.get("unresolved", []))
            else:
                for index, batch in enumerate(batches):
                    self._scene_progress(job_id, "transcript", status="running")
                    self.update(job_id, message=f"台词预理解：第 {index + 1} 组")
                    story = {"prior": bounded_memory(prior, []), "previous_segments": segments[-6:], "unresolved": unresolved[-8:]}
                    result = await self._scene_call(job_id, "transcript", str(index), prep_profile,
                        self.providers.prepare_subtitles, [batch, story])
                    segments.extend(result["segments"])
                    unresolved = result.get("unresolved", [])
                    self._scene_progress(job_id, "transcript", completed=index + 1)
            self._scene_metric(job_id, "transcript", wall_elapsed=time.monotonic() - transcript_started)
            chunks = []
            self.update(job_id, total=total, completed=0)
            frame_lookup = {f["id"]: f for f in frames}
            preparation_started = time.monotonic()
            offsets = list(range(0, len(frames), CORE_FRAMES))
            prepared = None
            parallel_preparation = (config.get("execution") or {}).get("preparation") == "parallel"
            if parallel_preparation:
                frozen_memory = {**bounded_memory(prior, []), "execution_mode": "parallel",
                    "notice": "并行分块只共享已冻结的前序记忆；边界参考帧属于之前的时间范围，不能写成当前片段的事实。"}
                frozen_characters = self._scene_context(asset_id, config)
                async def prepare_batch(index, offset):
                    core = frames[offset:offset + CORE_FRAMES]
                    context_frames = frames[max(0, offset - CONTEXT_FRAMES):offset]
                    a = config["start_ms"] if offset == 0 else core[0]["at_ms"]
                    b = frames[offset + CORE_FRAMES]["at_ms"] if offset + CORE_FRAMES < len(frames) else config["end_ms"]
                    self.update(job_id, message=f"并行视觉分块与简要索引：第 {index + 1}/{total} 组")
                    return await self._scene_call(job_id, "preparation", str(index), prep_profile,
                        self.providers.prepare_visual, [core, context_frames, subtitle_context(cues, segments, a, b),
                            copy.deepcopy(frozen_memory), copy.deepcopy(frozen_characters)],
                        images=math.ceil((len(core) + len(context_frames)) / 16))
                finished = 0
                def preparation_done(index, result):
                    nonlocal finished
                    finished += 1
                    self._scene_progress(job_id, "preparation", completed=finished)
                    self.update(job_id, completed=finished, progress=finished / max(total, 1) * .5)
                self._scene_progress(job_id, "preparation", status="running")
                prepared = await bounded_map(offsets, prepare_batch,
                    execution_width(config, "preparation", self.providers.get(prep_profile["id"])),
                    lambda: self.checkpoint(job_id), preparation_done)

            for index, offset in enumerate(offsets):
                self.checkpoint(job_id)
                self._scene_context(asset_id, config)
                core, context_frames = frames[offset:offset + CORE_FRAMES], frames[max(0, offset - CONTEXT_FRAMES):offset]
                a = config["start_ms"] if offset == 0 else core[0]["at_ms"]
                b = frames[offset + CORE_FRAMES]["at_ms"] if offset + CORE_FRAMES < len(frames) else config["end_ms"]
                local_cues = [c for c in cues if c["end_ms"] > a and c["start_ms"] < b]
                sub_context = subtitle_context(cues, segments, a, b)
                if prepared is not None:
                    self.update(job_id, message=f"整理已完成的视觉分块：{index + 1}/{total}")
                    result = prepared[index]
                else:
                    self._scene_progress(job_id, "preparation", status="running")
                    self.update(job_id, message=f"视觉分块与简要索引：{index + 1}/{total}")
                    result = await self._scene_call(job_id, "preparation", str(index), prep_profile,
                        self.providers.prepare_visual, [core, context_frames, sub_context, memory_for(chunks),
                                                       self._scene_context(asset_id, config)],
                        images=math.ceil((len(core) + len(context_frames)) / 16))
                for number, raw in enumerate(result["chunks"]):
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
                             "provenance": {"strategy": STRATEGY, "keyframe_plan_id": plan["id"],
                                            "profile_id": prep_profile["id"], "model": prep_profile.get("model", "")},
                             "uncertainties": result.get("unresolved", []),
                             "memory_before": copy.deepcopy(frozen_memory) if parallel_preparation else memory_for(chunks)}
                    # Keep transcript-only semantic coverage even when sparse frames
                    # required a bounded cue sample for the visual model.
                    own_cues = set(chunk["subtitle_ids"])
                    for segment in segments:
                        if set(segment["cue_ids"]).issubset(own_cues):
                            fact = {"text": segment["summary"], "cue_ids": segment["cue_ids"]}
                            if fact not in chunk["dialogue_facts"]:
                                chunk["dialogue_facts"].append(fact)
                    if number == 0 and raw.get("continues_previous") and chunks:
                        previous = chunks.pop()
                        required_frames = list(dict.fromkeys(deep_frame_ids(previous) + deep_frame_ids(chunk)))
                        chunk["start_ms"] = previous["start_ms"]
                        chunk["summary"] = (previous["summary"] + "；" + chunk["summary"])[-6000:]
                        for key in ("frame_ids", "evidence_frame_ids", "subtitle_ids", "shot_ids"):
                            chunk[key] = list(dict.fromkeys(previous.get(key, []) + chunk.get(key, [])))
                        for key in ("visual_facts", "dialogue_facts", "inferences"):
                            chunk[key] = previous.get(key, []) + chunk.get(key, [])
                        chunk["needs_deep"] = previous["needs_deep"] or chunk["needs_deep"]
                        chunk["deep_frame_ids"] = required_frames
                        chunk["scene_key"] = chunk["scene_key"] or previous.get("scene_key", "")
                        chunk["memory_before"] = previous["memory_before"]
                        chunk["id"] = chunk["chunk_id"] = "scene_" + digest([asset_id, chunk["start_ms"], stop])
                        reps = list(dict.fromkeys(previous["representative_frame_ids"] + chunk["representative_frame_ids"]))
                        chunk["representative_frame_ids"] = reps if len(reps) <= 8 else reps[:7] + [reps[-1]]
                    chunk["memory_after"] = memory_for(chunks + [chunk])
                    chunks.append(chunk)
                if prepared is None:
                    self._scene_progress(job_id, "preparation", completed=index + 1)
                    self.update(job_id, completed=index + 1, progress=(index + 1) / max(total, 1) * .5)
            self._scene_metric(job_id, "preparation", wall_elapsed=time.monotonic() - preparation_started)
            if not chunks:
                raise ValueError("此范围没有可分析的关键帧")
            link_similar_scenes(chunks)
            if not state.get("published"):
                self._check_scene_dependencies(asset_id, config, store)
                store.publish(asset_id, run_id, chunks, config["start_ms"], config["end_ms"])
                self._sync_scene_index(job_id, asset_id)
                state["published"] = True
                atomic_json(state_path, state)
            completed_chunks = []
            context_changed = False
            batch_size = deep_batch_size(config["profile_snapshots"]["vision"])
            deep_total = sum(math.ceil(len(deep_frame_ids(c, all_frames=config.get("character_reanalysis"))) / batch_size) for c in chunks
                             if c["needs_deep"] or config.get("character_reanalysis"))
            deep_seen, refresh_total, refresh_completed = False, 0, 0
            for chunk in chunks:
                if chunk["needs_deep"] or config.get("character_reanalysis"):
                    deep_seen = True
                elif deep_seen:
                    refresh_total += 1
            self._scene_progress(job_id, "deep_analysis", total=deep_total)
            self._scene_progress(job_id, "context_refresh", total=refresh_total)
            self.update(job_id, deep_request_count=deep_total, total=len(chunks), completed=0)
            for index, chunk in enumerate(chunks):
                memory = bounded_memory(prior, completed_chunks)
                if chunk["needs_deep"] or config.get("character_reanalysis"):
                    selected_ids = deep_frame_ids(chunk, all_frames=config.get("character_reanalysis"))
                    chunk = await self._deepen_scene(job_id, chunk, [frame_lookup[fid] for fid in selected_ids], cues, memory)
                    self._check_scene_dependencies(asset_id, config, store)
                    chunk = store.upgrade(asset_id, run_id, chunk, invalidate_dependents=False)
                    context_changed = True
                    self._sync_scene_index(job_id, asset_id)
                elif context_changed:
                    self._scene_progress(job_id, "context_refresh", status="running")
                    # Earlier deep analysis changed the narrative. Reconcile only
                    # text-supported interpretation; no need to send these images again.
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
                    self._sync_scene_index(job_id, asset_id)
                    refresh_completed += 1
                    self._scene_progress(job_id, "context_refresh", completed=refresh_completed)
                completed_chunks.append(store.get(asset_id, chunk["id"]))
                self.update(job_id, completed=index + 1, progress=.5 + .5 * (index + 1) / len(chunks))
        embedding = config.get("bindings", {}).get("embedding")
        if embedding:
            await self.reindex(job_id, embedding_profile_id=embedding)

    def _check_scene_dependencies(self, asset_id, config, store):
        self._check_scene_range(asset_id, config, store)
        self._scene_context(asset_id, config)
        if config["scene_dependencies"].get("prompt_versions") != prompt_versions():
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
        from .models import Observation
        job = self.checkpoint(job_id)
        config, asset_id = job["config"], job["asset_id"]
        profile = config["profile_snapshots"]["vision"]
        batch_size = deep_batch_size(profile)
        self._scene_context(asset_id, config)
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
                    for entity in reversed(part.get("entities", [])) if entity.get("character_id")))[:16]
                existing = {entry["id"]: entry for entry in context["profiles"]}
                fields = ("id", "name", "aliases", "description", "notes", "appearance", "user_edited",
                          "edited_fields", "representative", "updated_at", "created_at")
                transient = {}
                for identity in recent:
                    stored = registry["profiles"].get(identity)
                    if stored is None:
                        raise ValueError("已完成分批的人物档案已变化，请创建新任务")
                    entry = {key: copy.deepcopy(stored[key]) for key in fields}
                    entry["observed_traits"] = list(stored["appearance"])
                    for field in ("appearance", "observed_traits"):
                        if field not in entry.get("edited_fields", []):
                            entry[field] = entry[field][-8:]
                    transient[identity] = entry
                identities = list(dict.fromkeys([*recent, *existing]))[:16]
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
                while encoded_size(canonical(context)) > 18000 and len(context["profiles"]) > 1:
                    context["profiles"].pop()
                allowed = {entry["id"] for entry in context["profiles"]}
                context["reference_frames"] = [reference for identity, reference in references.items() if identity in allowed][:8]
                context = self._bounded_scene_profiles(context)
                allowed = {entry["id"] for entry in context["profiles"]}
                context["reference_frames"] = [reference for reference in context["reference_frames"]
                                               if reference["character_id"] in allowed]
                return context

        for index, offset in enumerate(range(0, len(frames), batch_size)):
            self.checkpoint(job_id)
            self._scene_context(asset_id, config)
            current = frames[offset:offset + batch_size]
            start = chunk["start_ms"] if offset == 0 else current[0]["at_ms"]
            end = frames[offset + batch_size]["at_ms"] if offset + batch_size < len(frames) else chunk["end_ms"]
            part_path = target.parent / "parts" / chunk["id"] / f"{index}.json"
            part = read_json(part_path)
            if part is not None:
                Observation.model_validate(part)
                if (part.get("asset_id") != asset_id or part.get("run_id") != job["run_id"]
                        or part.get("id") != chunk["id"] or part.get("start_ms") != start or part.get("end_ms") != end
                        or part.get("input_frame_ids") != [frame["id"] for frame in current]):
                    raise ValueError("已保存的深入理解分批与当前任务不一致")
                parts.append(part)
                self._scene_progress(job_id, "deep_analysis", completed=progress_base + index + 1)
                continue
            local_subtitles = subtitle_context(cues, [], start, end)
            context = context_with_parts()
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
            with self.library.lock:
                self._scene_context(asset_id, config)
                # Stable chunk + namespaced entity IDs make a crash after registry
                # persistence but before the part file safe to replay from receipt.
                part = self.characters.apply_observation(asset_id, part, config["character_snapshot"]["revision"])
                atomic_json(part_path, part)
            parts.append(part)
            self._scene_progress(job_id, "deep_analysis", completed=progress_base + index + 1)
        result = {**chunk, "run_id": job["run_id"], "summary": "；".join(p["summary"] for p in parts),
                  "understanding_level": "deep", "needs_deep": False, "reason": "",
                  "memory_before": memory, "window_id": chunk["id"], "shot_id": "scene"}
        for key in ("entities", "events", "spatial_observations", "uncertainties", "visual_facts"):
            result[key] = [value for part in parts for value in part.get(key, [])]
        result["provenance"] = {**chunk.get("provenance", {}), "deep_profile_id": profile["id"],
                                "deep_model": profile.get("model", ""), "input_memory_digest": digest(memory),
                                "deep_analyzed_frame_ids": [frame["id"] for frame in frames],
                                "deep_batch_size": batch_size}
        # Keep a stable current-evidence set for character validation and portraits.
        result["evidence_frame_ids"] = list(chunk["frame_ids"])
        with self.library.lock:
            self._scene_context(asset_id, config)
            # Replaying an allocation is different from claiming a confident
            # visual match: first appearances retain their original confidence.
            # The stable run/chunk/entity tuple reconstructs the same registry ID.
            created_here = {identity for part in parts for identity in part.get("new_character_entity_ids", [])}
            for entity in result["entities"]:
                if entity["id"] in created_here:
                    entity["character_id"] = None
            result = self.characters.apply_observation(asset_id, result, config["character_snapshot"]["revision"])
            result["memory_after"] = bounded_memory({}, [result])
            atomic_json(target, result)
        return result
