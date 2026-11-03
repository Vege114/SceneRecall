"""Optional knowledge builds use the ordinary task queue."""
from __future__ import annotations

import asyncio
from time import perf_counter

from .library import digest


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
