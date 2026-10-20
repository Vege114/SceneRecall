"""Bounded episode preparation ahead of chronological scene publication."""
from __future__ import annotations

import asyncio
import copy
import time

from .library import atomic_json, read_json


class EpisodePipelineMixin:
    @staticmethod
    def independent_transcript(config):
        return (config.get("transcript_context") == "independent-v1"
                and config.get("analysis_strategy") == "scene-chunks-v1"
                and "vision" in config.get("stages", []) and not config.get("deepen_chunk_id"))

    def transcript_dependencies(self, asset_id, config):
        from .scenes import SceneStore
        from .scene_jobs import prompt_versions
        return {"fingerprint": self.library.get_asset(asset_id)["fingerprint"],
                "subtitle_digest": SceneStore(self.library).subtitle_digest(asset_id),
                "start_ms": config["start_ms"], "end_ms": config["end_ms"],
                "context": "independent-v1", "prompt_version": prompt_versions(config)["transcript"]}

    def check_transcript_dependencies(self, asset_id, config):
        self.library.require_source(asset_id)
        if config.get("transcript_dependencies") != self.transcript_dependencies(asset_id, config):
            raise ValueError("台词或来源已改变，请创建新任务；已保存的台词整理结果仍保留")

    async def prepare_episode_inputs(self, job_id):
        """Only original dialogue is shared here; plot and cast publish later."""
        from .scene_jobs import cue_batches
        from .scenes import SceneStore

        job = self.checkpoint(job_id)
        config, asset_id = copy.deepcopy(job["config"]), job["asset_id"]
        if not self.independent_transcript(config):
            return
        self.library.require_source(asset_id)
        for profile in config["profile_snapshots"].values():
            if not self.profile_source_unchanged(config, profile):
                raise ValueError("模型配置已变化，请创建新任务")
        self._check_scene_range(asset_id, config)
        self._scene_context(asset_id, config)
        pending = next((row for row in SceneStore(self.library)._raw(asset_id)
                        if row["end_ms"] <= config["start_ms"] and row.get("needs_deep")), None)
        if pending is not None:
            raise ValueError(f"前序片段 {pending['start_ms'] / 1000:g}–{pending['end_ms'] / 1000:g} 秒尚待深入理解，"
                             "请先完成该片段的深入分析，再分析后续范围")
        directory = self.library.asset_dir(asset_id) / "analyses" / job["run_id"]
        state_path = directory / "scene-state.json"
        state = read_json(state_path, {})
        stages = copy.deepcopy(job.get("stage_progress") or {
            stage: {"completed": 0, "total": None, "status": "pending"}
            for stage in ("transcript", "preparation", "deep_analysis", "context_refresh")})
        self.update(job_id, stage_progress=stages)
        if "subtitle" in config["stages"] and not state.get("ocr_complete"):
            await self.analyze_legacy(job_id, {**config, "stages": ["subtitle"]}, skip_reindex=True)
            if self.get(job_id)["failures"]:
                raise ValueError("字幕识别未完成，连续理解已停止；重试将复用成功字幕批次")
            state["ocr_complete"] = True
            atomic_json(state_path, state)
        if not config.get("keyframe_plan_id"):
            started = time.monotonic()
            plan = await self.scan_plan(job_id, asset_id=asset_id,
                start_ms=config["start_ms"], end_ms=config["end_ms"])
            self._scene_metric(job_id, "keyframes", elapsed=time.monotonic() - started)
            config["keyframe_plan_id"] = plan["id"]
        else:
            self.validated_plan(asset_id, config["start_ms"], config["end_ms"], config["keyframe_plan_id"])
        dependencies = self.transcript_dependencies(asset_id, config)
        if config.get("transcript_dependencies") and config["transcript_dependencies"] != dependencies:
            raise ValueError("台词或来源已改变，请创建新任务；已保存的台词整理结果仍保留")
        config["transcript_dependencies"] = dependencies
        self.update(job_id, config=config)
        cues = [cue for cue in SceneStore(self.library).subtitle_records(asset_id)
                if cue["start_ms"] < config["end_ms"] and cue["end_ms"] > config["start_ms"]]
        await self.prepare_transcript_stage(job_id, config, {}, cues, list(cue_batches(cues)),
                                            config["profile_snapshots"]["preparation"])

    def _episode_preparation_width(self, jobs):
        widths = []
        for job in jobs:
            config = job["config"]
            task_limit = max(1, (config.get("execution") or {}).get("max_concurrency", 3))
            profiles = [config["profile_snapshots"]["preparation"]]
            if "subtitle" in config.get("stages", []):
                profiles.append(config["profile_snapshots"]["subtitle"])
            limits = {}
            for profile in profiles:
                try:
                    limits[profile["id"]] = self.providers.get(profile["id"]).get("max_concurrency", task_limit)
                except (ValueError, FileNotFoundError):
                    # The execution wrapper reports a deleted/broken profile on
                    # that job. Scheduling must not terminate unrelated workers.
                    limits[profile["id"]] = profile.get("max_concurrency", task_limit)
            # A narrow OCR connection must not occupy the whole file pipeline
            # while another connection is preparing the previous file's text.
            # Each connection still enforces its own shared in-flight limit.
            widths.append(min(task_limit, sum(limits.values())))
        return max(widths, default=1)

    async def pipeline_loop(self):
        """Own every worker, drain on shutdown, keep inferred history ordered.

        Local scans have a separate CPU bound. OCR and raw transcript preparation
        overlap across files; a batch's visual work starts after its dialogue
        barrier. There is exactly one scene-publication worker at a time.
        """
        running = {}
        try:
            while not self.closed:
                for identity, (task, _kind) in list(running.items()):
                    if task.done():
                        task.result()
                        running.pop(identity)
                jobs = list(reversed(self.list()))
                queued = [job for job in jobs if job["status"] == "queued" and job["id"] not in running]
                busy_assets = {self.get(identity).get("asset_id") for identity in running}

                def launch(job, kind):
                    running[job["id"]] = (asyncio.create_task(
                        self.execute(job["id"], prepare_only=kind == "dialogue")), kind)
                    busy_assets.add(job.get("asset_id"))

                scans = sum(kind == "scan" for _task, kind in running.values())
                for job in queued:
                    if scans >= self.scan_concurrency:
                        break
                    if job["type"] == "keyframe_scan" and job.get("asset_id") not in busy_assets:
                        launch(job, "scan")
                        scans += 1
                candidates = [job for job in queued if job["type"] == "analysis"
                              and self.independent_transcript(job["config"])
                              and not job.get("transcript_prefetched")]
                active_dialogue = [self.get(identity) for identity, (_task, kind) in running.items()
                                   if kind == "dialogue"]
                width = self._episode_preparation_width(candidates + active_dialogue)
                slots = max(0, width - len(active_dialogue))
                for job in candidates:
                    if not slots:
                        break
                    if job.get("asset_id") not in busy_assets:
                        launch(job, "dialogue")
                        slots -= 1
                if not any(kind == "retrieval" for _task, kind in running.values()):
                    for job in queued:
                        if (job["type"] == "analysis"
                                and job["config"].get("analysis_strategy") == "retrieval-first-v1"
                                and job.get("asset_id") not in busy_assets):
                            launch(job, "retrieval")
                            break
                if not any(kind == "main" for _task, kind in running.values()):
                    ordered = [job for job in jobs if job["type"] != "keyframe_scan"
                               and job["config"].get("analysis_strategy") != "retrieval-first-v1"
                               and job["status"] in {"queued", "running", "pausing", "cancelling"}]
                    for job in ordered[:1]:
                        if job["id"] in running or job["status"] != "queued":
                            break
                        if job.get("asset_id") in busy_assets:
                            continue
                        if self.independent_transcript(job["config"]):
                            if not job.get("transcript_prefetched"):
                                break
                            group = job.get("batch_id") or job["id"]
                            unfinished = [item for item in jobs
                                if (item.get("batch_id") or item["id"]) == group
                                and item["status"] in {"queued", "running", "pausing", "cancelling"}
                                and not item.get("transcript_prefetched")]
                            if unfinished:
                                break
                        launch(job, "main")
                        break
                await asyncio.sleep(.02 if running else .2)
        finally:
            # Cancelling a queue cancels admissions, then joins owned local and
            # paid work before the application releases its service resources.
            for task, _kind in running.values():
                task.cancel()
            if running:
                await asyncio.gather(*(task for task, _kind in running.values()), return_exceptions=True)

    def with_timing(self, config, estimate):
        from .timing import estimate_timing
        return estimate_timing(self, config, estimate)
