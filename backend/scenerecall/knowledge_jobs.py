"""Optional knowledge builds and reference discovery use the ordinary task queue."""
from __future__ import annotations

import asyncio
from time import perf_counter

from .library import atomic_json, digest, read_json, safe_id
from .visual_index import evidence_digest


def discovery_path(knowledge, scope):
    return knowledge.store._path(scope).parent / "reference-candidates.json"


def discovery_inputs(knowledge, search, scope):
    store, library = knowledge.store, knowledge.store.library
    saved = knowledge.scope_catalog(scope)
    references, records, fingerprints = [], [], {}
    for asset_id in store.scope_view(scope)["asset_ids"]:
        fingerprints[asset_id] = library.get_asset(asset_id)["fingerprint"]
        records.extend(row for row in library.records(asset_id)
                       if row.get("source_fingerprint") == fingerprints[asset_id]
                       and (row.get("evidence_type") == "retrieval_window" or row.get("kind") in {"visual", "scene"}))
    for concept in saved["entities"].values():
        if concept["status"] != "active":
            continue
        for ref in concept.get("references", []):
            if ref["status"] != "confirmed":
                continue
            references.append({"id": ref["id"], "concept_id": concept["concept_id"], "work_scope_id": scope,
                "asset_id": scope,
                "source_fingerprint": ref["sha256"], "status": "confirmed", "version": ref.get("version"),
                "pts": 0, "time_base": [1, 1], "path": str(store.reference_path(scope, concept["concept_id"], ref["id"]))})
    signature = search.visual_index.namespace(search.visual_encoder)[0] if search.visual_encoder else None
    token = digest([scope, store.scope_view(scope)["asset_ids"], fingerprints, references, signature,
                    [(row["id"], evidence_digest(row)) for row in records]])
    return references, records, signature, token


def discovery_view(knowledge, search, scope):
    _, _, _, token = discovery_inputs(knowledge, search, scope)
    saved = read_json(discovery_path(knowledge, scope), {"candidates": []})
    return {**saved, "stale": saved.get("signature") != token}


def confirm_discovery(knowledge, search, scope, candidate_id, reason):
    store = knowledge.store
    with store.library.lock:
        saved = discovery_view(knowledge, search, scope)
        if saved["stale"]:
            raise ValueError("参考图、影片或编码器已变化，请重新发现候选")
        candidate = next((row for row in saved["candidates"] if row["candidate_id"] == safe_id(candidate_id)), None)
        if candidate is None:
            raise FileNotFoundError("参考图候选不存在")
        binding = store.bind(candidate["asset_id"], {"record_id": candidate["record_id"],
            "concept_id": candidate["concept_id"], "status": "confirmed", "entity_id": None,
            "evidence_frame_ids": [candidate["frame_id"]], "reason": reason})
        candidate.update(status="confirmed", binding_id=binding["id"])
        saved.pop("stale", None)
        atomic_json(discovery_path(knowledge, scope), saved)
        return binding


class KnowledgeJobMixin:
    async def _knowledge_operation(self, job_id, operation):
        task = asyncio.create_task(operation)
        try:
            while not task.done():
                self.checkpoint(job_id)
                await asyncio.wait({task}, timeout=.1)
            return await task
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def build_knowledge(self, job_id):
        started = perf_counter()
        scope = self.get(job_id)["config"]["work_scope_id"]
        projection = self.knowledge.projection(scope)
        frozen = self.get(job_id)["config"]["signature"]
        frozen_config = self.get(job_id)["config"].get("runtime_config_revision")
        if frozen_config and digest(self.knowledge_runtime.settings()) != frozen_config:
            raise ValueError("知识模型配置已变化，请创建新的建库任务")
        if projection["signature"] != frozen:
            raise ValueError("资料已变化，请创建新的知识建库任务")
        self.update(job_id, stage="knowledge_build", message="正在抽取并建立作品知识索引")

        def current():
            self.checkpoint(job_id)
            return self.knowledge.projection(scope)["signature"] == frozen

        previous_operation = self.knowledge_runtime._operation_id
        try:
            result = await self._knowledge_operation(job_id, self.knowledge_runtime.build(projection, is_current=current))
            self.update(job_id, knowledge_result=result, completed=len(projection["cards"]), total=len(projection["cards"]))
        finally:
            if self.knowledge_runtime._operation_id != previous_operation:
                diagnostics = list(self.knowledge_runtime._ledger)
                calls = [row for row in diagnostics if row.get("event") == "model_finished"]
                unknown = sum(row.get("request_count") is None for row in calls)
                usage = {}
                for row in calls:
                    for key, value in (row.get("usage") or {}).items():
                        if isinstance(value, (int, float)):
                            usage[key] = usage.get(key, 0) + value
                self.update(job_id, request_count=sum(row.get("request_count") or 0 for row in calls),
                    uncertain_request_count=unknown, usage=usage,
                    cost=sum(row["cost"] for row in calls) if calls and all(row.get("cost") is not None for row in calls) else None,
                    cost_incomplete=bool(unknown or any(row.get("cost") is None for row in calls)),
                    knowledge_diagnostics=diagnostics,
                    stage_metrics={"knowledge": {"wall_elapsed_ms": (perf_counter() - started) * 1000,
                        "elapsed_ms": sum(row.get("duration_ms") or 0 for row in calls),
                        "request_count": sum(row.get("request_count") or 0 for row in calls), "usage": usage}})

    async def discover_references(self, job_id):
        scope = self.get(job_id)["config"]["work_scope_id"]
        references, records, signature, token = discovery_inputs(self.knowledge, self.search, scope)
        if not references:
            raise ValueError("请先添加并确认概念参考图")
        self.update(job_id, stage="reference_discovery", message="正在比较参考图与已有影片帧特征")
        started = perf_counter()
        matches = await self._knowledge_operation(job_id,
            self.search.match_visual_references(records, references, detailed=True))
        by_record = {row["id"]: row for row in records}
        candidates = []
        for record_id, values in matches.items():
            record = by_record[record_id]
            for value in values:
                if not value.get("frame_id"):
                    continue
                candidates.append({**value, "candidate_id": "candidate_" + digest([job_id, record_id, value]),
                    "record_id": record_id, "asset_id": record["asset_id"], "work_scope_id": scope,
                    "start_ms": record["start_ms"], "end_ms": record["end_ms"], "status": "candidate",
                    "evidence_digest": evidence_digest(record), "encoder_signature": signature})
        # Bound the review queue by ranked candidates per concept, without
        # changing the matcher or confirming any result.
        candidates.sort(key=lambda row: (-row["similarity"], row["record_id"], row["concept_id"]))
        counts, selected = {}, []
        for row in candidates:
            if counts.get(row["concept_id"], 0) < 50:
                selected.append(row)
                counts[row["concept_id"]] = counts.get(row["concept_id"], 0) + 1
        with self.library.lock:
            self.checkpoint(job_id)
            if discovery_inputs(self.knowledge, self.search, scope)[3] != token:
                raise ValueError("参考图或影片证据在发现期间发生变化，未发布候选")
            atomic_json(discovery_path(self.knowledge, scope), {"signature": token, "job_id": job_id,
                "candidates": selected, "elapsed_ms": (perf_counter() - started) * 1000})
        self.update(job_id, completed=len(records), total=len(records), request_count=0,
                    reference_discovery={"candidates": len(selected), "elapsed_ms": (perf_counter() - started) * 1000})
