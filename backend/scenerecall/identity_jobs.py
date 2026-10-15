"""Optional local identity prelude for the existing durable scene queue."""
from __future__ import annotations

import asyncio
import copy
import time

from .execution import finish_on_cancel
from .library import atomic_json, digest, read_json
from .identity_scene import identity_context, project_audio_evidence


class IdentityJobMixin:
    def identity_settings(self, asset_id):
        from .visual_identity import VisualIdentityPipeline
        from .audio_identity import AudioIdentityService
        return {"visual": VisualIdentityPipeline(self.library).settings(asset_id),
                "audio": AudioIdentityService(self.library, self.characters).settings(asset_id)}

    def _identity_path(self, job):
        return self.library.asset_dir(job["asset_id"]) / "analyses" / job["run_id"] / "identity-prelude.json"

    async def prepare_scene_identity(self, job_id, frames, cues):
        job = self.checkpoint(job_id)
        config, asset_id = job["config"], job["asset_id"]
        settings = config.get("identity_settings")
        if settings is None:  # Frozen pre-upgrade tasks retain their inputs.
            return
        if settings != self.identity_settings(asset_id):
            raise ValueError("视听身份配置已变化，请新建任务；已提取特征可复用")
        header = digest([self.library.get_asset(asset_id)["fingerprint"], config["start_ms"],
                         config["end_ms"], settings, [frame["id"] for frame in frames]])
        path = self._identity_path(job)
        saved = read_json(path)
        if saved is not None:
            if saved.get("header") != header:
                raise ValueError("视听身份断点与当前素材或配置不一致")
            return
        from .visual_identity import VisualIdentityPipeline
        from .audio_identity import AudioIdentityService

        async def measured(stage, call):
            started = time.monotonic()
            # Thread work is drained on cancellation; its checkpoints see the
            # persistent queue control state and its feature cache stays usable.
            value = await finish_on_cancel(asyncio.to_thread(call))
            self._scene_metric(job_id, stage, elapsed=time.monotonic() - started,
                               cache=int(value.get("cache_hits", 0)), degraded_reason=value.get("degraded_reason")
                               or "；".join(value.get("degraded", [])))
            return value

        outcomes = await asyncio.gather(
            measured("visual_identity", lambda: VisualIdentityPipeline(self.library).run(
                asset_id, frames, config=settings["visual"], checkpoint=lambda: self.checkpoint(job_id))),
            measured("audio_identity", lambda: AudioIdentityService(self.library, self.characters).run(
                asset_id, config["start_ms"], config["end_ms"], checkpoint=lambda: self.checkpoint(job_id),
                subtitles=cues, frames=frames, config_override=settings["audio"])), return_exceptions=True)
        # Drain both owned local workers before propagating a failure. A sibling
        # may already be persisting a reusable feature/checkpoint.
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        visual, audio = outcomes
        self.checkpoint(job_id)
        atomic_json(path, {"header": header, "visual": visual, "audio": audio})
        statuses = {key: {"status": value.get("status"), "degraded": value.get("degraded", []),
                          "degraded_reason": value.get("degraded_reason")}
                    for key, value in (("visual", visual), ("audio", audio))}
        self.update(job_id, identity_status=statuses)

    def scene_identity_context(self, job_id, context, frames):
        from .visual_identity import VisualIdentityPipeline
        job = self.get(job_id)
        saved = read_json(self._identity_path(job), {})
        visual = saved.get("visual")
        if visual:
            visual = VisualIdentityPipeline(self.library).project_result(job["asset_id"], visual)
        return identity_context(context, visual, frames,
                                top_k=job["config"].get("identity_settings", {}).get("visual", {}).get("top_k", 5))

    def scene_audio_context(self, job_id, context, start, end):
        from .audio_identity import AudioIdentityService
        job = self.get(job_id)
        saved = read_json(self._identity_path(job), {})
        audio = saved.get("audio", {})
        if audio.get("status") not in {"completed", "partial"}:
            return context
        context = copy.deepcopy(context)
        rows = [record for record in audio.get("records", [])
                if record["start_ms"] < end and record["end_ms"] > start]
        if rows:
            current = AudioIdentityService(self.library, self.characters).records(job["asset_id"], start, end)
            rows = project_audio_evidence(rows, current)
        context["speaker_dialogue"] = [{key: row.get(key) for key in (
            "id", "speaker_id", "start_ms", "end_ms", "character_id", "candidate_character_id",
            "character_name", "association_status", "association_source", "subtitle_ids", "overlap", "text",
            "dialogue_attribution", "subtitle_association", "phonetic_alignment", "audio_ref", "audio_stream_index",
            "frame_ids", "model_versions", "language", "dub_version", "visual_evidence")}
            for row in rows[:80]]
        for row in context["speaker_dialogue"]:
            row["text"] = (row.get("text") or "")[:500]
        context["audio_context_omitted"] = max(0, len(rows) - 80)
        context["audio_policy"] = "heard_only; no_visible_presence_or_action; subtitle_overlap_not_alignment"
        return context

    def scene_audio_evidence(self, job_id, start, end):
        return self.scene_audio_context(job_id, {}, start, end).get("speaker_dialogue", [])

    def refresh_scene_request_identity(self, job_id, stage, args):
        """Refresh an uncompleted frozen request, keeping its media/story inputs."""
        if "identity_settings" not in self.get(job_id)["config"]:
            return args
        result = copy.deepcopy(args)
        if stage == "preparation" and len(result) > 4 and isinstance(result[2], dict):
            start, end = result[2].get("start_ms"), result[2].get("end_ms")
            if start is not None and end is not None:
                result[2] = self.scene_audio_context(job_id, result[2], start, end)
            result[4] = self.scene_identity_context(job_id, result[4], result[0])
        elif stage == "deep_analysis" and len(result) > 4:
            result[3] = self.scene_identity_context(job_id, result[3], result[0])
            result[4] = self.scene_audio_context(job_id, result[4], result[1], result[2])
        return result
