"""Durable stage checkpoints, separate from paid-call identities and accounting."""
from __future__ import annotations

import copy
import time

from .execution import bounded_map, execution_width, yield_to_io
from .library import atomic_json, digest, read_json, safe_id
from .scene_preparation import validate_subtitle_preparation


class SceneCheckpointMixin:
    @staticmethod
    def _transcript_outputs(results, parallel):
        segments, unresolved = [], []
        for result in results:
            if result is None:
                break
            segments.extend(result["segments"])
            if parallel:
                unresolved.extend(result.get("unresolved", []))
            else:
                unresolved = result.get("unresolved", [])
        return segments, unresolved

    @staticmethod
    def _transcript_story(index, batches, frozen_prior, results, parallel):
        from .scene_jobs import encoded_size

        if parallel:
            previous = copy.deepcopy(batches[index - 1][-20:] if index else [])
            while previous and encoded_size(previous) > 6000:
                previous.pop(0)
            return {"prior": copy.deepcopy(frozen_prior), "previous_cues": previous,
                    "execution_mode": "parallel", "previous_segments": [], "unresolved": [],
                    "notice": "并行批次只共享已冻结的前序记忆与原始相邻台词；本组之外的台词仅作上下文，不应写入本组证据。"}
        segments, unresolved = SceneCheckpointMixin._transcript_outputs(results[:index], False)
        return {"prior": copy.deepcopy(frozen_prior), "previous_segments": copy.deepcopy(segments[-6:]),
                "unresolved": copy.deepcopy(unresolved[-8:])}

    def _transcript_receipt(self, job, config, prep_profile, batches, index, story, directory):
        """Import old durable successes without executing the call admission path."""
        from .jobs import model_identity
        from .scene_jobs import STRATEGY, canonical, prompt_versions

        unit, version = str(index), prompt_versions(config)["transcript"]
        frozen = read_json(directory / "scene-inputs" / f"{digest(['transcript', version, unit])}.json")
        if frozen is None:
            return None
        if (not isinstance(frozen, dict) or frozen.get("images") != 0
                or not isinstance(frozen.get("args"), list) or len(frozen["args"]) != 2
                or canonical(frozen["args"]) != canonical([batches[index], story])):
            raise ValueError("已保存的台词整理输入与当前批次不一致")
        dependencies = (config["transcript_dependencies"] if self.independent_transcript(config)
                        else config["scene_dependencies"])
        key = digest([STRATEGY, "transcript", version, unit, dependencies,
                      model_identity(prep_profile), canonical(frozen["args"])])
        output = directory / "scene-receipts" / f"{key}.json"
        receipt = read_json(output)
        own_receipt = receipt is not None
        if receipt is None and not config.get("force"):
            root = self.library.asset_dir(job["asset_id"])
            matches = sorted(root.glob(f"analyses/*/scene-receipts/{key}.json"))
            if matches:
                receipt = read_json(matches[-1])
        if receipt is None:
            return None
        if (not isinstance(receipt, dict) or receipt.get("receipt_id") != key
                or receipt.get("stage") != "transcript"
                or (own_receipt and receipt.get("job_id") != job["id"])):
            raise ValueError("已保存的台词整理回执与当前任务不一致")
        result = validate_subtitle_preparation(receipt.get("data"), batches[index])
        self._account_scene_receipt(job["id"], receipt)
        self._scene_metric(job["id"], "transcript", cache=1)
        return result

    async def prepare_transcript_stage(self, job_id, config, prior, cues, batches, prep_profile):
        """Restore completed batches first, then schedule only missing work."""
        from .jobs import model_identity
        from .scene_jobs import bounded_memory, canonical, prompt_versions

        job = self.checkpoint(job_id)
        independent = self.independent_transcript(config)
        if independent:
            self.check_transcript_dependencies(job["asset_id"], config)
            prior = {}
        directory = self.library.asset_dir(job["asset_id"]) / "analyses" / safe_id(job["run_id"])
        target = directory / "scene-transcript.json"
        batches = list(batches)
        if [cue["id"] for batch in batches for cue in batch] != [cue["id"] for cue in cues]:
            raise ValueError("台词整理批次未完整覆盖当前字幕")
        parallel = (config.get("execution") or {}).get("transcript") == "parallel"
        header = {"version": 1, "asset_id": job["asset_id"], "run_id": job["run_id"],
                  "dependencies": copy.deepcopy(config["transcript_dependencies"] if independent
                                                  else config["scene_dependencies"]),
                  "prompt_version": prompt_versions(config)["transcript"],
                  "profile": model_identity(prep_profile), "parallel": parallel,
                  "prior_digest": digest(canonical(prior)),
                  "batch_digests": [digest(canonical(batch)) for batch in batches]}
        saved = read_json(target)
        if saved is None:
            saved = {"header": header, "results": [None] * len(batches), "complete": not batches}
        elif (not isinstance(saved, dict) or saved.get("header") != header
              or not isinstance(saved.get("results"), list) or len(saved["results"]) != len(batches)
              or type(saved.get("complete")) is not bool
              or saved["complete"] != all(result is not None for result in saved["results"])):
            raise ValueError("已保存的台词整理阶段与当前任务不一致")
        results = saved["results"]
        for index, result in enumerate(results):
            if result is not None:
                results[index] = validate_subtitle_preparation(result, batches[index])
        if not parallel:
            gap_seen = False
            for result in results:
                if result is None:
                    gap_seen = True
                elif gap_seen:
                    raise ValueError("已保存的串行台词整理批次存在断点缺口")

        def persist():
            saved["complete"] = all(result is not None for result in results)
            atomic_json(target, saved)

        frozen_prior = bounded_memory(prior, [])
        # Calls completed during pause, shutdown or a process exit may have a
        # receipt but no stage checkpoint yet. Import every valid success before
        # exposing the restored count or admitting any missing provider calls.
        imported = False
        for index, result in enumerate(results):
            if result is not None:
                continue
            if not parallel and any(previous is None for previous in results[:index]):
                break
            await yield_to_io()
            self.checkpoint(job_id)
            story = self._transcript_story(index, batches, frozen_prior, results, parallel)
            result = self._transcript_receipt(job, config, prep_profile, batches, index, story, directory)
            if result is not None:
                results[index] = result
                imported = True
        if imported or not target.exists():
            persist()
        completed = sum(result is not None for result in results)
        self._scene_progress(job_id, "transcript", total=len(batches), completed=completed,
                             status="completed" if completed == len(batches) else "running")
        if completed == len(batches):
            return self._transcript_outputs(results, parallel)
        started = time.monotonic()
        pending = [index for index, result in enumerate(results) if result is None]

        async def prepare(_position, index):
            story = self._transcript_story(index, batches, frozen_prior, results, parallel)
            prefix = "并行台词预理解" if parallel else "台词预理解"
            self.update(job_id, message=f"{prefix}：第 {index + 1}/{len(batches)} 组")
            result = await self._scene_call(job_id, "transcript", str(index), prep_profile,
                self.providers.prepare_subtitles, [batches[index], story])
            results[index] = validate_subtitle_preparation(result, batches[index])
            # Persist inside the owned operation: bounded_map drains sibling
            # successes after an error even when their on_result callbacks stop.
            persist()
            return results[index]

        def done(_position, _result):
            self._scene_progress(job_id, "transcript", completed=sum(result is not None for result in results))

        if parallel:
            await bounded_map(pending, prepare, execution_width(config, "transcript", self.providers.get(prep_profile["id"])),
                              lambda: self.checkpoint(job_id), done)
        else:
            for position, index in enumerate(pending):
                await yield_to_io()
                self.checkpoint(job_id)
                result = await prepare(position, index)
                done(position, result)
        self._scene_metric(job_id, "transcript", wall_elapsed=time.monotonic() - started)
        return self._transcript_outputs(results, parallel)
