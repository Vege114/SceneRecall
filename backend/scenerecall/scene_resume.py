"""Durable chronological publication cursors for scene understanding."""
from __future__ import annotations

from .library import atomic_json, digest, read_json


class SceneResumeMixin:
    def _scene_preparation_results(self, asset_id, run_id, config, plan, profile, frames, cues, segments, batch_size):
        from .jobs import model_identity
        from .scene_jobs import CONTEXT_FRAMES, prompt_versions, subtitle_context
        from .scene_preparation import validate_visual_preparation
        target = self.library.asset_dir(asset_id) / "analyses" / run_id / "scene-preparation-results.json"
        total = (len(frames) + batch_size - 1) // batch_size
        header = {"asset_id": asset_id, "run_id": run_id, "dependencies": config["scene_dependencies"],
                  "profile": model_identity(profile), "plan_id": plan["id"], "batch_size": batch_size,
                  "prompt_version": prompt_versions(config)["preparation"]}
        saved = read_json(target)
        if saved is None:
            saved = {"header": header, "results": [None] * total}
        elif (not isinstance(saved, dict) or saved.get("header") != header
                or not isinstance(saved.get("results"), list) or len(saved["results"]) != total):
            raise ValueError("已保存的并行视觉分块批次与任务不一致")
        for index, result in enumerate(saved["results"]):
            if result is None:
                continue
            offset = index * batch_size
            core = frames[offset:offset + batch_size]
            a = config["start_ms"] if offset == 0 else core[0]["at_ms"]
            b = frames[offset + batch_size]["at_ms"] if offset + batch_size < len(frames) else config["end_ms"]
            context = subtitle_context(cues, segments, a, b)
            saved["results"][index] = validate_visual_preparation(result, core,
                {cue["id"] for cue in context["cues"]}, has_context=bool(offset and CONTEXT_FRAMES),
                cues=context["cues"], start_ms=a, end_ms=b)
        return target, saved

    def _load_partial_scene_preparation(self, asset_id, run_id, config, plan, batch_size, transcript_total):
        target = self.library.asset_dir(asset_id) / "analyses" / run_id / "scene-preparation-parts.json"
        saved = read_json(target)
        if saved is None:
            return [], 0
        count = saved.get("completed") if isinstance(saved, dict) else None
        total = (len(plan["frames"]) + batch_size - 1) // batch_size
        if type(count) is not int or not 1 <= count <= total:
            raise ValueError("已保存的视觉分块批次断点无效")
        stop = min(count * batch_size, len(plan["frames"]))
        end = plan["frames"][stop]["at_ms"] if stop < len(plan["frames"]) else config["end_ms"]
        self._validate_scene_preparation(saved, asset_id, run_id, {**config, "end_ms": end},
            {**plan, "frames": plan["frames"][:stop]}, transcript_total, count)
        return saved["chunks"], count

    def _save_partial_scene_preparation(self, asset_id, run_id, config, chunks, transcript_total, completed):
        target = self.library.asset_dir(asset_id) / "analyses" / run_id / "scene-preparation-parts.json"
        atomic_json(target, {"asset_id": asset_id, "run_id": run_id,
            "dependencies": config["scene_dependencies"], "chunks": chunks,
            "transcript_total": transcript_total, "preparation_total": completed, "completed": completed})

    @staticmethod
    def _understanding_kind(chunk, config, context_changed):
        if chunk.get("content_kind") == "credits":
            return "initial"
        if chunk["needs_deep"] or config.get("character_reanalysis"):
            return "deep_analysis"
        return "context_refresh" if context_changed else "initial"

    def _understanding_identity(self, chunks, config):
        from .scenes import SceneStore
        return digest([config["scene_dependencies"], bool(config.get("character_reanalysis")),
                       [SceneStore.revision_digest(chunk) for chunk in chunks]])

    def _load_scene_understanding(self, asset_id, run_id, config, chunks, state, state_path, store):
        """Read committed revisions once; never replay their models or publication."""
        from .scene_jobs import deep_batch_size, deep_frame_ids
        identity = self._understanding_identity(chunks, config)
        saved = state.get("understanding")
        if saved is not None and (not isinstance(saved, dict) or saved.get("plan_digest") != identity
                or type(saved.get("completed")) is not int or not 0 <= saved["completed"] <= len(chunks)
                or not isinstance(saved.get("revisions"), list)
                or len(saved["revisions"]) != saved["completed"]):
            raise ValueError("已保存的连续理解断点与任务不一致")
        raw = {row["id"]: row for row in store._raw(asset_id)}
        marked = self.library.active(asset_id).get("scene_staleness", {})
        cursor = {"plan_digest": identity, "completed": 0, "deep_completed": 0,
                  "refresh_completed": 0, "context_changed": False, "revisions": []}
        batch_size = deep_batch_size(config["profile_snapshots"]["vision"])
        recent = []
        for index, original in enumerate(chunks):
            kind = self._understanding_kind(original, config, cursor["context_changed"])
            current = raw.get(original["id"])
            if current is None or marked.get(original["id"]):
                raise ValueError("任务中的场景已改变或不可用，不能继续旧断点")
            if (current["dependencies"] != config["scene_dependencies"]
                    or current["frame_ids"] != original["frame_ids"]):
                raise ValueError("任务中的场景来源或关键帧已改变")
            expected_run = run_id + "_context" if kind == "context_refresh" else run_id
            done = current["run_id"] == expected_run and (
                current["understanding_level"] == "deep" if kind == "deep_analysis"
                else current["understanding_level"] == "initial")
            reference = {"id": current["id"], "relative": store._relative(current),
                         "revision_digest": store.revision_digest(current)}
            if saved is not None and index < saved["completed"]:
                if not done or reference != saved["revisions"][index]:
                    raise ValueError("已完成的场景版本已改变，不能继续旧断点")
            elif not done:
                # This is the first unfinished unit. A different task's result
                # must never be overwritten by replaying this older task.
                if current["run_id"] != run_id or current["understanding_level"] != "initial":
                    raise ValueError("此片段已有其他任务的结果，请创建新任务")
                break
            if kind == "deep_analysis":
                cursor["context_changed"] = True
                cursor["deep_completed"] += (len(deep_frame_ids(original,
                    all_frames=config.get("character_reanalysis"))) + batch_size - 1) // batch_size
            elif kind == "context_refresh":
                cursor["refresh_completed"] += 1
            cursor["completed"] = index + 1
            cursor["revisions"].append(reference)
            recent.append(current)
            recent = recent[-8:]
        if saved is not None:
            # Check derived counts at the saved prefix rather than trusting UI
            # counters; a later committed pointer can legitimately advance it.
            count, refreshed, changed = 0, 0, False
            for chunk in chunks[:saved["completed"]]:
                kind = self._understanding_kind(chunk, config, changed)
                if kind == "deep_analysis":
                    changed = True
                    count += (len(deep_frame_ids(chunk, all_frames=config.get("character_reanalysis")))
                              + batch_size - 1) // batch_size
                elif kind == "context_refresh":
                    refreshed += 1
            if (saved.get("deep_completed") != count or saved.get("refresh_completed") != refreshed
                    or saved.get("context_changed") is not changed):
                raise ValueError("已保存的连续理解进度无效")
        if cursor != saved:
            # Recover a crash between scene publication and cursor persistence.
            if ((saved is not None and cursor["completed"] > saved["completed"])
                    or (saved is None and cursor["context_changed"])):
                state["index_dirty"] = True
            state["understanding"] = cursor
            atomic_json(state_path, state)
        annotations = store._annotations(asset_id)
        # Initial publication may have allocated persistent character IDs.
        # Keep the immutable preparation plan for cursor identity, but use the
        # already-read active revisions for all pending semantic updates.
        return cursor, [store._project_semantics(asset_id, row, annotations) for row in recent], raw

    def _commit_scene_understanding(self, cursor, chunk, kind, deep_completed, state, state_path, store):
        cursor["completed"] += 1
        cursor["deep_completed"] = deep_completed
        if kind == "deep_analysis":
            cursor["context_changed"] = True
        elif kind == "context_refresh":
            cursor["refresh_completed"] += 1
        cursor["revisions"].append({"id": chunk["id"], "relative": store._relative(chunk),
                                    "revision_digest": store.revision_digest(chunk)})
        state["understanding"] = cursor
        if kind != "initial":
            state["index_dirty"] = True
        atomic_json(state_path, state)

    def _repair_scene_index(self, job_id, asset_id, state, state_path):
        if state.get("index_dirty"):
            self._sync_scene_index(job_id, asset_id)
            state["index_dirty"] = False
            atomic_json(state_path, state)

    def _load_manual_scene_commit(self, asset_id, run_id, config, original, state, state_path, store):
        saved = state.get("deepen_committed")
        active = self.library.active(asset_id).get("scenes", {}).get(original["id"])
        expected = f"analyses/{run_id}/scenes/{original['id']}.deep.json"
        if saved is not None and (not isinstance(saved, dict) or saved.get("relative") != expected):
            raise ValueError("已保存的深入理解断点无效")
        if active != expected:
            if saved is not None:
                raise ValueError("已完成的深入理解版本已改变")
            return None
        chunk = read_json(self.library.asset_dir(asset_id) / expected)
        if (not chunk or chunk.get("id") != original["id"] or chunk.get("run_id") != run_id
                or chunk.get("asset_id") != asset_id or chunk.get("frame_ids") != original["frame_ids"]
                or chunk.get("dependencies") != config["scene_dependencies"]
                or chunk.get("understanding_level") != "deep"):
            raise ValueError("已保存的深入理解版本与任务不一致")
        reference = {"relative": expected, "revision_digest": store.revision_digest(chunk)}
        if saved is not None and saved != reference:
            raise ValueError("已完成的深入理解内容已改变")
        if saved is None:
            state["deepen_committed"] = reference
            state["index_dirty"] = True
            atomic_json(state_path, state)
        return chunk
