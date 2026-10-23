"""Local preparation and human review are durable phases of submitted tasks."""
from __future__ import annotations

import copy

from .library import now


class TaskLifecycleMixin:
    @staticmethod
    def unified_task(config):
        return (config.get("task_pipeline") == "unified-v1"
                and "vision" in config.get("stages", [])
                and not config.get("deepen_chunk_id")
                and not config.get("retrieval_operation"))

    def initialize_task_workflow(self, job):
        if job["type"] != "analysis" or not self.unified_task(job["config"]):
            return
        from .characters import scope_id
        asset = self.library.get_asset(job["asset_id"])
        job["workflow_progress"] = {
            name: {"status": "pending"} for name in ("keyframes", "character_confirmation", "analysis")}
        job["workflow_scope"] = {"scope_id": scope_id(asset),
                                 "membership_revision": asset.get("membership_revision", 0)}

    def workflow_step(self, job_id, name, status, **fields):
        with self.library.lock:
            progress = copy.deepcopy(self.get(job_id).get("workflow_progress", {}))
            progress[name] = {**progress.get(name, {}), **fields, "status": status}
            return self.update(job_id, workflow_progress=progress)

    def _confirmation_service(self):
        if not getattr(self, "character_confirmation", None):
            from .character_confirmation import CharacterConfirmation
            self.character_confirmation = CharacterConfirmation(self.library, search=self.search)
        return self.character_confirmation

    def _check_workflow_scope(self, job):
        from .characters import scope_id
        self._check_deletion(job)
        self.library.require_source(job["asset_id"])
        asset = self.library.get_asset(job["asset_id"])
        current = {"scope_id": scope_id(asset), "membership_revision": asset.get("membership_revision", 0)}
        if job.get("stale_organization") or current != job.get("workflow_scope"):
            raise ValueError("视频归属已经改变，请创建新的分析任务；已完成资料保留")

    def _finish_character_confirmation(self, job_id, review):
        """Snapshot approved references once, before any paid or inferred work."""
        job = self.get(job_id)
        self._check_workflow_scope(job)
        config = copy.deepcopy(job["config"])
        context = self.character_context(job["asset_id"], scene=config.get("analysis_strategy") == "scene-chunks-v1")
        config["character_snapshot"] = {key: context[key] for key in ("scope_id", "revision", "membership_revision")}
        if config.get("analysis_strategy") == "retrieval-first-v1":
            from .retrieval_identity import gallery_identity
            config["visual_identity_gallery"] = gallery_identity(self._confirmation_service().pipeline, job["asset_id"])
        self.update(job_id, config=config, input_required=None)
        self.workflow_step(job_id, "character_confirmation", "completed", completed=review["summary"]["total"],
                           total=review["summary"]["total"], revision=review["revision"], completed_at=now())

    def _review_requires_input(self, job, review):
        if review["summary"]["pending"] or (review.get("sampling") or {}).get("status") == "running":
            return True
        # A later episode must not bypass a work's explicit continuation just
        # because its last role decision was saved while the earlier episode
        # still displays the human-review gate.
        return any(other["id"] != job["id"] and other["status"] in {"waiting_input", "paused"}
                   and (other.get("input_required") or {}).get("scope_id") == review["scope_id"]
                   for other in self.list())

    async def prepare_task_workflow(self, job_id):
        """Return without occupying a queue worker while human decisions are pending."""
        from .jobs import StopJob
        job = self.checkpoint(job_id)
        if not self.unified_task(job["config"]):
            return
        self._check_workflow_scope(job)
        phases = job["workflow_progress"]
        config = copy.deepcopy(job["config"])
        if phases["keyframes"]["status"] != "completed":
            self.workflow_step(job_id, "keyframes", "running")
            if config.get("analysis_strategy") == "retrieval-first-v1":
                plan = await self._retrieval_plan(job_id)
            elif config.get("keyframe_plan_id"):
                plan = self.validated_plan(job["asset_id"], config["start_ms"], config["end_ms"], config["keyframe_plan_id"])
            else:
                plan = await self.scan_plan(job_id, asset_id=job["asset_id"],
                                            start_ms=config["start_ms"], end_ms=config["end_ms"])
                config["keyframe_plan_id"] = plan["id"]
                self.update(job_id, config=config)
            self.checkpoint(job_id)
            self.workflow_step(job_id, "keyframes", "completed", completed=len(plan["frames"]),
                               total=len(plan["frames"]), plan_id=plan["id"])
        if phases["character_confirmation"]["status"] != "completed":
            def review_and_freeze(check):
                # Projection can scan a large work. Keep it off the API event
                # loop and atomically freeze the same state that was reviewed.
                with self._confirmation_service()._guard, self.library.lock:
                    check()
                    self._check_workflow_scope(self.get(job_id))
                    review = self._confirmation_service().queue(job["asset_id"])
                    if self._review_requires_input(job, review):
                        done = review["summary"]["confirmed"] + review["summary"]["skipped"]
                        self.workflow_step(job_id, "character_confirmation", "waiting_input", completed=done,
                                           total=review["summary"]["total"])
                        self.update(job_id, status="waiting_input", stage="character_confirmation", progress=0,
                                    completed=done, total=review["summary"]["total"],
                                    input_required={"kind": "character_confirmation", "asset_id": job["asset_id"],
                                                    "scope_id": review["scope_id"]},
                                    message="等待角色核对：请确认或跳过待核对角色，然后继续任务")
                        raise StopJob()
                    self._finish_character_confirmation(job_id, review)
            await self._local_work(job_id, review_and_freeze)
        self.workflow_step(job_id, "analysis", "running")

    def continue_task_workflow(self, job_id):
        # Sampling start uses guard -> library; preserve that order when the
        # queue projection also reads its sampling status.
        with self._confirmation_service()._guard, self.library.lock:
            job = self.get(job_id)
            if job["status"] != "waiting_input" or not self.unified_task(job["config"]):
                raise ValueError("当前任务没有等待角色核对")
            self._check_workflow_scope(job)
            review = self._confirmation_service().queue(job["asset_id"])
            if review["summary"]["pending"]:
                raise ValueError("请先确认或明确跳过所有待核对角色，再继续任务")
            if (review.get("sampling") or {}).get("status") == "running":
                raise ValueError("本地角色采样尚未完成，请完成后继续任务")
            self._finish_character_confirmation(job_id, review)
            result = self.update(job_id, status="queued", message="角色核对完成，等待继续分析")
            # The decisions belong to a work. A single explicit continuation
            # also releases waiting episodes in this batch; each rechecks and
            # snapshots that shared roster when admitted.
            if job.get("batch_id"):
                for sibling in self.list():
                    if (sibling["id"] != job_id and sibling.get("batch_id") == job["batch_id"]
                            and sibling["status"] in {"waiting_input", "paused"}
                            and sibling.get("workflow_scope", {}).get("scope_id") == review["scope_id"]):
                        self.update(sibling["id"], status="paused" if sibling["status"] == "paused" else "queued",
                                    input_required=None, message="同作品角色核对已完成，等待继续分析")
            return result
