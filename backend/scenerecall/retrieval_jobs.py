"""Retrieval-first execution; no chronological story or scene receipts are inputs."""
from __future__ import annotations

import copy
import time

from .library import atomic_json, digest, now, read_json
from .retrieval_sampling import PARAMETERS, VERSION, RetrievalSampler, build_windows
from .retrieval_store import STRATEGY, RetrievalStore

CAPABILITIES = ("sampling", "visual_index", "subtitle_index", "local_records", "identity", "final_publish")


class RetrievalJobMixin:
    def validate_retrieval(self, request, settings, asset, end):
        required = {"vision"} if request.retrieval_mode == "complete" else set()
        if set(request.model_bindings) - required:
            raise ValueError("只能为本次任务启用的阶段选择模型连接")
        if set(request.model_overrides) - required:
            raise ValueError("只能为本次任务启用的阶段选择模型和思考等级")
        if request.stages != ["vision"]:
            raise ValueError("画面检索任务仅接受 vision；字幕使用已有文字轨，新增 OCR 请单独运行")
        encoder = getattr(self.search, "visual_encoder", None)
        if encoder is None:
            raise ValueError("请先配置独立的本地图文编码器")
        try:
            identity = encoder.identity()
        except RuntimeError as exc:
            raise ValueError(str(exc)) from None
        config = {**request.model_dump(), "end_ms": end, "analysis_strategy": STRATEGY,
                  "sampling_strategy": VERSION, "retrieval_contract_version": 2,
                  "sampling_parameters": dict(PARAMETERS), "encoder_identity": identity,
                  "source_fingerprint": asset["fingerprint"], "speech_activity": "off",
                  "identity_policy": "identity-binding-v2", "profile_snapshots": {},
                  "source_profile_snapshots": {}, "bindings": dict(settings.get("bindings", {}))}
        config["subtitle_digest"] = digest(self.library.subtitles(asset["id"]))
        from .visual_identity import VisualIdentityPipeline
        from .retrieval_identity import gallery_identity
        visual_pipeline = VisualIdentityPipeline(self.library)
        config["visual_identity_settings"] = visual_pipeline.settings(asset["id"])
        config["visual_identity_gallery"] = gallery_identity(visual_pipeline, asset["id"])
        from .work_entities import WorkEntityStore
        work_scope = WorkEntityStore(self.library).scope(asset["id"])
        config["work_scope_id"] = work_scope["work_scope_id"] if work_scope else None
        if "execution" not in request.model_fields_set:
            config["execution"]["max_concurrency"] = 4
        if request.keyframe_plan_id:
            self.validated_plan(asset["id"], request.start_ms, end, request.keyframe_plan_id)
        if request.retrieval_mode == "complete":
            profile_id = settings.get("bindings", {}).get("vision")
            if not profile_id:
                raise ValueError("完整检索流程需要轻量局部记录的 vision 模型；基础试验请选择 core")
            profile = copy.deepcopy(self.providers.get(profile_id))
            if "vision" not in profile.get("capabilities", []):
                raise ValueError("局部记录模型未声明 vision 能力")
            config["source_profile_snapshots"]["vision"] = copy.deepcopy(profile)
            if profile.get("provider_type") == "codex_cli":
                if request.task_pipeline != "unified-v1":
                    profile.update(model="gpt-6.1-sol", reasoning_effort="low")
                if request.max_cost is not None:
                    raise ValueError("Codex CLI 无法使用金额预算，请保留请求数预算")
            selected = request.model_overrides.get("vision")
            if selected is not None:
                if selected.profile_id != profile_id or selected.provider_type != profile.get("provider_type"):
                    raise ValueError("局部记录模型连接已变化")
                profile.update(selected.model_dump(exclude_unset=True, exclude={"profile_id", "provider_type"}))
            config["profile_snapshots"]["vision"] = profile
            config["retrieval_prompt_version"] = "retrieval-local-observation-v1"
        elif request.model_overrides:
            raise ValueError("核心检索试验不调用视觉语言模型")
        return asset, config

    def validate_retrieval_identity(self, asset_id, settings):
        """Create a separate, resumable local-only job for existing published windows."""
        from .models import AnalysisInput
        from .retrieval_identity import gallery_identity
        from .visual_identity import VisualIdentityPipeline
        asset = self.library.get_asset(asset_id)
        windows = RetrievalStore(self.library).windows(asset_id)
        if not windows:
            raise ValueError("请先建立画面检索索引，再补充本地人物匹配")
        pipeline = VisualIdentityPipeline(self.library)
        identity_settings = pipeline.settings(asset_id)
        if not identity_settings.get("enabled"):
            raise ValueError("本地视觉身份识别未启用，请先在角色参考中启用")
        request = AnalysisInput(asset_id=asset_id, analysis_strategy=STRATEGY, retrieval_mode="core")
        config = {**request.model_dump(), "retrieval_operation": "identity_only", "retrieval_contract_version": 2,
                  "start_ms": min(row["start_ms"] for row in windows),
                  "end_ms": max(row["end_ms"] for row in windows), "source_fingerprint": asset["fingerprint"],
                  "speech_activity": "off", "identity_policy": "identity-binding-v2",
                  "profile_snapshots": {}, "source_profile_snapshots": {}, "bindings": dict(settings.get("bindings", {})),
                  "visual_identity_settings": identity_settings,
                  "visual_identity_gallery": gallery_identity(pipeline, asset_id),
                  "retrieval_identity_units": [{"id": row["id"], "digest": digest(row)} for row in windows]}
        return asset, config

    async def _retrieval_plan(self, job_id):
        from .keyframes import extract_keyframes, scan_keyframes
        from .media import probe_for_asset
        job = self.checkpoint(job_id)
        config, asset_id = job["config"], job["asset_id"]
        root = self.library.asset_dir(asset_id)
        identity = "rp_" + digest([asset_id, config["source_fingerprint"], config["start_ms"],
                                    config["end_ms"], VERSION, config["sampling_parameters"]])
        path = root / "retrieval" / "plans" / f"{identity}.json"
        saved = read_json(path)
        if saved is not None:
            if (saved.get("source_fingerprint") != config["source_fingerprint"]
                    or saved.get("sampling_version") != VERSION):
                raise ValueError("检索采样计划身份不匹配")
            self.library.frame_paths(asset_id, [f["id"] for f in saved["frames"]])
            return saved
        self.update(job_id, stage="retrieval_sampling", message="正在扫描画面检索证据")
        source = self.library.require_source(asset_id)
        asset = self.library.get_asset(asset_id)

        def work(check):
            check()
            metadata = probe_for_asset(source, asset)
            collector = RetrievalSampler()
            data = scan_keyframes(source, config["start_ms"], config["end_ms"], checkpoint=check,
                                  sample_collector=collector, metadata=metadata)
            frames = {str((f["pts"], f["time_base"])): f for f in data["frames"] + collector.frames}
            if config.get("keyframe_plan_id"):
                old = self.validated_plan(asset_id, config["start_ms"], config["end_ms"], config["keyframe_plan_id"])
                for f in old["frames"]:
                    frames[str((f["pts"], f["time_base"]))] = f
            frames = sorted(frames.values(), key=lambda f: f["at_ms"])
            for frame in frames:
                frame["id"] = "kf_" + digest([asset_id, config["source_fingerprint"], frame["pts"], frame["time_base"]])
            registry = read_json(root / "frames" / "registry.json", {})
            missing = [f for f in frames if f["id"] not in registry
                       or not (root / registry[f["id"]]["path"]).is_file()]
            for offset in range(0, len(missing), 128):
                check()
                current = missing[offset:offset + 128]
                outputs = [root / "frames" / f"{f['id']}.jpg" for f in current]
                self.library.register_frames(asset_id,
                    extract_keyframes(source, current, outputs, checkpoint=check, metadata=metadata))
            check()
            self.library.require_source(asset_id)
            return {"id": identity, "sampling_version": VERSION, "parameters": config["sampling_parameters"],
                    "source_fingerprint": config["source_fingerprint"], "frames": frames,
                    "shots": data["shots"], "base_plan_id": config.get("keyframe_plan_id"),
                    "start_ms": config["start_ms"], "end_ms": config["end_ms"]}

        async with self._scan_slots:
            plan = await self._local_work(job_id, work)
        atomic_json(path, plan)
        return plan

    def _retrieval_metric(self, job_id, stage, started):
        metrics = copy.deepcopy(self.get(job_id).get("stage_metrics", {}))
        metrics.setdefault(stage, {"elapsed_ms": 0, "request_count": 0, "cache_hits": 0})["elapsed_ms"] += round(
            (time.perf_counter() - started) * 1000)
        self.update(job_id, stage_metrics=metrics)

    async def analyze_retrieval(self, job_id):
        job = self.checkpoint(job_id)
        config, asset_id, run_id = job["config"], job["asset_id"], job["run_id"]
        asset = self.library.get_asset(asset_id)
        if asset["fingerprint"] != config["source_fingerprint"]:
            raise ValueError("原片已变化，请创建新的检索任务")
        if config.get("retrieval_operation") == "identity_only":
            await self.analyze_retrieval_identity(job_id)
            return
        from .work_entities import WorkEntityStore
        work_scope = WorkEntityStore(self.library).scope(asset_id)
        if (work_scope["work_scope_id"] if work_scope else None) != config.get("work_scope_id"):
            raise ValueError("作品实体作用域已变化，请创建新的检索任务")
        self.library.require_source(asset_id)
        encoder = getattr(self.search, "visual_encoder", None)
        if encoder is None or encoder.identity() != config["encoder_identity"]:
            raise ValueError("图文编码器已变化，请创建新的检索任务")
        state = copy.deepcopy(job.get("retrieval") or {"status": "processing",
            "capabilities": {key: "pending" for key in CAPABILITIES}, "coverage": {},
            "started_at": now(), "first_searchable_at": None, "default_completed_at": None})
        state["contract_version"] = config.get("retrieval_contract_version", 1)
        state["status"] = "processing"
        self.update(job_id, retrieval=state)

        def capability(name):
            state["capabilities"][name] = "completed"
            self.update(job_id, retrieval=state)

        started = time.perf_counter()
        plan = await self._retrieval_plan(job_id)
        self._retrieval_metric(job_id, "retrieval_sampling", started)
        capability("sampling")
        cues = self.library.subtitles(asset_id)
        if digest(cues) != config.get("subtitle_digest"):
            raise ValueError("字幕资料已变化，请创建新的检索任务")
        units = build_windows(asset_id, asset["fingerprint"], config["start_ms"], config["end_ms"],
                              plan["frames"], plan["shots"], cues)
        state["coverage"] = {"start_ms": config["start_ms"], "end_ms": config["end_ms"],
                             "total_units": len(units), "published_units": 0, "processed_units": 0}
        store = RetrievalStore(self.library)
        for unit in units:
            self.checkpoint(job_id)
            store.publish(asset_id, run_id, {**unit, "strategy": STRATEGY, "kind": "visual",
                "evidence_type": "retrieval_window", "summary": "", "text": "", "entities": [], "events": [],
                "review_status": "unreviewed", "source": "retrieval"})
        started = time.perf_counter()
        self.update(job_id, stage="subtitle_index", message="正在发布已有字幕和画面证据")
        await self._local_work(job_id, lambda check: (check(), self.search.sync_asset(self.library.records(asset_id), asset_id)))
        state.setdefault("subtitle_searchable_at", now() if cues else None)
        state["subtitles_available"] = bool(cues)
        self._retrieval_metric(job_id, "subtitle_index", started)
        capability("subtitle_index")
        self.update(job_id, stage="visual_index", message="正在建立直接视觉索引")
        records = [r for r in self.library.records(asset_id) if r["id"] in {u["id"] for u in units}]
        paths = self.library.frame_paths(asset_id, [f["id"] for f in plan["frames"]])
        frames = [{**f, "path": str(paths[f["id"]]), "asset_id": asset_id,
                   "source_fingerprint": asset["fingerprint"]} for f in plan["frames"]]

        async def progress(value):
            self.checkpoint(job_id)
            published = value.get("published_records", 0)
            state["coverage"]["published_units"] = published
            if published and state.get("first_searchable_at") is None:
                state["first_searchable_at"] = now()
            self.update(job_id, retrieval=state, progress=value.get("completed", 0) / max(1, value.get("total", 1)),
                        message=f"视觉特征 {value.get('completed', 0)} / {value.get('total', 0)}")

        started = time.perf_counter()
        result = await self.search.index_visual(records, frames, encoder=encoder, progress=progress)
        self._retrieval_metric(job_id, "visual_index", started)
        state["visual_metrics"] = result
        state["encoder_runtime"] = copy.deepcopy(getattr(encoder, "metadata", {}))
        state["coverage"]["published_units"] = len(units)
        state["first_searchable_at"] = state.get("first_searchable_at") or now()
        capability("visual_index")
        visual = None
        if config.get("retrieval_contract_version", 1) >= 2:
            windows = store.windows(asset_id)
            current_ids = {unit["id"] for unit in units}
            visual = await self.retrieval_identity(job_id, [row for row in windows if row["id"] in current_ids],
                                                   frames, state)
        if config.get("retrieval_mode") == "complete":
            await self.retrieval_local_records(job_id, units, frames, cues, state, visual=visual)
        else:
            for name in (("local_records", "identity") if config.get("retrieval_contract_version", 1) < 2 else ("local_records",)):
                state["capabilities"][name] = "not_requested"
            if config.get("retrieval_contract_version", 1) >= 2:
                state["coverage"]["processed_units"] = len(units)
        self.checkpoint(job_id)
        self.update(job_id, stage="final_publish", message="正在完成检索索引发布")
        store.finalize(asset_id, run_id, config["start_ms"], config["end_ms"], [u["id"] for u in units])
        await self._local_work(job_id, lambda check: (check(), self.search.sync_asset(self.library.records(asset_id), asset_id)))
        capability("final_publish")
        complete = config.get("retrieval_mode") == "complete"
        if complete and any(value not in {"completed", "disabled"} for value in state["capabilities"].values()):
            raise ValueError("默认检索能力尚未全部完成")
        state.update(status="completed" if complete else "core_ready", default_completed_at=now() if complete else None)
        self.update(job_id, retrieval=state, message="默认检索流程完成" if complete else "核心检索完成；未请求生成式局部记录")

    async def retrieval_identity(self, job_id, windows, frames, state):
        """Run/cache local person evidence independently from a generative provider."""
        from .retrieval_identity import VERSION as identity_version, gallery_identity, observation
        from .visual_identity import VisualIdentityPipeline
        job = self.checkpoint(job_id)
        config, asset_id = job["config"], job["asset_id"]
        pipeline = VisualIdentityPipeline(self.library)
        if (pipeline.settings(asset_id) != config["visual_identity_settings"]
                or gallery_identity(pipeline, asset_id) != config["visual_identity_gallery"]):
            raise ValueError("本地人物配置或角色参考已变化，请新建人物匹配任务")
        wanted = {fid for unit in windows for fid in unit["local_frame_ids"]}
        selected = [frame for frame in frames if frame["id"] in wanted]
        signature = digest([identity_version, config["source_fingerprint"], config["visual_identity_settings"],
                            config["visual_identity_gallery"], sorted(wanted)])
        run = self.library.asset_dir(asset_id) / "analyses" / job["run_id"]
        path = run / "retrieval-local-identity.json"
        saved = read_json(path)
        if saved is not None and saved.get("input_signature") != signature:
            raise ValueError("本地人物匹配输入与已保存回执不一致")
        started = time.perf_counter()
        self.update(job_id, stage="identity", message="正在执行本地人物检测与角色参考匹配")
        if saved is None:
            def work(check):
                return pipeline.run(asset_id, selected,
                    config={**config["visual_identity_settings"], "identity_policy": "identity-binding-v2"},
                    checkpoint=check, progress=lambda done, total: self.update(job_id,
                        message=f"本地人物匹配 {done} / {total}", progress=done / max(1, total)))
            visual = await self._local_work(job_id, work)
            if visual.get("status") not in {"completed", "disabled"}:
                raise ValueError(visual.get("degraded_reason") or "本地人物匹配不可用；已发布画面索引保留")
            saved = {"input_signature": signature, "visual": visual}
            atomic_json(path, saved)
        visual = saved["visual"]
        if visual.get("status") not in {"completed", "disabled"}:
            raise ValueError("本地人物成功回执状态无效")
        store = RetrievalStore(self.library)
        for window in windows:
            self.checkpoint(job_id)
            store.publish_identity(asset_id, job["run_id"], observation(window, visual,
                run_id=job["run_id"], signature=signature))
        state["capabilities"]["identity"] = "completed" if visual["status"] == "completed" else "disabled"
        state["identity"] = {"status": visual["status"], "reason": visual.get("degraded_reason"),
                             "metrics": copy.deepcopy(visual.get("metrics", {})),
                             "region_count": len(visual.get("regions", [])), "input_signature": signature}
        self._retrieval_metric(job_id, "identity", started)
        self.update(job_id, retrieval=state)
        return visual

    async def analyze_retrieval_identity(self, job_id):
        """Only existing frames and an independent identity/index projection are touched."""
        job = self.checkpoint(job_id)
        config, asset_id = job["config"], job["asset_id"]
        store = RetrievalStore(self.library)
        windows = store.windows(asset_id)
        expected = config["retrieval_identity_units"]
        if [{"id": row["id"], "digest": digest(row)} for row in windows] != expected:
            raise ValueError("已发布画面窗口已变化，请新建人物匹配任务")
        state = copy.deepcopy(job.get("retrieval") or {"status": "processing", "started_at": now(),
            "capabilities": {key: "not_requested" for key in CAPABILITIES}, "default_completed_at": None})
        state["contract_version"] = config.get("retrieval_contract_version", 1)
        state["status"] = "processing"
        state["capabilities"].update(identity="pending", final_publish="pending")
        state["coverage"] = {"start_ms": config["start_ms"], "end_ms": config["end_ms"],
            "total_units": len(windows), "published_units": len(windows), "processed_units": 0}
        self.update(job_id, retrieval=state)
        frames = {frame["id"]: frame for window in windows for frame in window["ordered_frames"]}
        await self.retrieval_identity(job_id, windows, list(frames.values()), state)
        self.checkpoint(job_id)
        self.update(job_id, stage="final_publish", message="正在发布人物身份检索投影")
        await self._local_work(job_id, lambda check: (check(), self.search.sync_asset(self.library.records(asset_id), asset_id)))
        state["capabilities"]["final_publish"] = "completed"
        state["coverage"]["processed_units"] = len(windows)
        state.update(status="core_ready", identity_completed_at=now())
        self.update(job_id, retrieval=state, message="本地人物匹配和检索投影已完成")

    async def retrieval_local_records(self, job_id, units, frames, cues, state, *, visual=None):
        from .execution import bounded_map
        from .identity_scene import identity_context
        from .retrieval_protocol import VERSION as prompt_version, validate_batch
        from .visual_identity import VisualIdentityPipeline
        from .jobs import model_identity

        job = self.checkpoint(job_id)
        asset_id, config = job["asset_id"], job["config"]
        profile = config["profile_snapshots"]["vision"]
        if not self.profile_source_unchanged(config, profile):
            raise ValueError("局部记录模型连接已变化，请创建新任务")
        root = self.library.asset_dir(asset_id)
        run = root / "analyses" / job["run_id"]
        store = RetrievalStore(self.library)
        wanted = {fid for unit in units for fid in unit["local_frame_ids"]}
        selected = [f for f in frames if f["id"] in wanted]
        visual_pipeline = VisualIdentityPipeline(self.library)
        if config["visual_identity_settings"] != visual_pipeline.settings(asset_id):
            raise ValueError("视觉身份配置已变化，请新建任务")
        started = time.perf_counter()
        self.update(job_id, stage="identity", message="正在准备局部视觉身份候选")
        identity_path = run / "retrieval-identity.json"
        visual = visual if visual is not None else read_json(identity_path)
        if visual is None:
            visual = await self._local_work(job_id, lambda check: visual_pipeline.run(
                asset_id, selected, config={**config["visual_identity_settings"], "identity_policy": "identity-binding-v2"},
                checkpoint=check))
            if visual.get("status") not in {"completed", "disabled"}:
                raise ValueError("必需的视觉身份准备失败；已发布核心检索结果保留")
            atomic_json(identity_path, visual)
        self._retrieval_metric(job_id, "identity", started)
        started = time.perf_counter()
        self.update(job_id, stage="local_records", message="正在生成有序局部观察和事件")
        from .work_entities import WorkEntityStore
        work_store = WorkEntityStore(self.library)
        work_scope = work_store.scope(asset_id)
        visual_concepts = {}
        if work_scope:
            references = []
            for concept in work_store.list(asset_id)["entities"]:
                for reference in concept.get("references", []):
                    if reference.get("status") != "confirmed":
                        continue
                    references.append({"id": reference["id"], "concept_id": concept["concept_id"],
                        "asset_id": work_scope["work_scope_id"], "work_scope_id": work_scope["work_scope_id"],
                        "source_fingerprint": reference["sha256"], "sha256": reference["sha256"],
                        "pts": 0, "time_base": [1, 1], "at_ms": 0, "status": "confirmed",
                        "path": str(work_store.reference_path(work_scope["work_scope_id"], concept["concept_id"], reference["id"]))})
            if references:
                window_ids = {u["id"] for u in units}
                records = [r for r in self.library.records(asset_id) if r["id"] in window_ids]
                visual_concepts = await self.search.match_visual_references(records, references)
        by_frame = {f["id"]: f for f in frames}
        batches, current, count = [], [], 0
        for unit in units:
            n = len(unit["local_frame_ids"])
            if current and (len(current) == 2 or count + n > 12):
                batches.append(current)
                current, count = [], 0
            current.append(unit)
            count += n
        if current:
            batches.append(current)

        async def process(_index, batch):
            self.checkpoint(job_id)
            saved_observations = [run / "retrieval-observations" / f"{unit['id']}.json" for unit in batch]
            if all(path.is_file() for path in saved_observations):
                for path in saved_observations:
                    saved = read_json(path)
                    key = saved.get("provenance", {}).get("receipt_key")
                    if not isinstance(key, str) or len(key) != 24 or any(c not in "0123456789abcdef" for c in key):
                        raise ValueError("已保存局部观察缺少有效回执来源")
                    receipt = read_json(run / "retrieval-receipts" / f"{key}.json")
                    original = next((o for o in (receipt or {}).get("data", {}).get("observations", [])
                                     if o["unit_id"] == saved["unit_id"]), None)
                    if not original or {k: saved.get(k) for k in original} != original:
                        raise ValueError("已保存局部观察与成功回执不一致")
                    store.publish_observation(asset_id, job["run_id"], saved)
                state["coverage"]["processed_units"] += len(batch)
                self.update(job_id, retrieval=state)
                return
            local = [by_frame[fid] for u in batch for fid in u["local_frame_ids"]]
            context = await self._local_work(job_id, lambda check: (check(), self.characters.context(asset_id))[1])
            context = identity_context(context, visual, local, top_k=3)
            # Candidate retrieval runs on the full allowed gallery before bounding
            # the language-model context; names and old prose do not establish presence.
            context["profiles"] = context.get("profiles", [])[:5]
            permitted = {p["id"] for p in context["profiles"]}
            context["reference_frames"] = [r for r in context.get("reference_frames", [])
                                             if r.get("character_id") in permitted][:8]
            context["region_images"] = []
            concept_context = {"entities": [], "reference_frames": []}
            local_cues = [{k: cue[k] for k in ("id", "start_ms", "end_ms", "text")}
                          for cue in cues if any(cue["start_ms"] < u["end_ms"] and cue["end_ms"] > u["start_ms"] for u in batch)]
            if work_scope:
                ranked_concepts = {}
                for unit in batch:
                    for match in visual_concepts.get(unit["id"], []):
                        identity = match["concept_id"]
                        ranked_concepts[identity] = max(ranked_concepts.get(identity, -1), match["similarity"])
                concept_ids = sorted(ranked_concepts, key=lambda key: (-ranked_concepts[key], key))[:5]
                concepts = work_store.context(asset_id, query=" ".join(c["text"] for c in local_cues), concept_ids=concept_ids)
                concept_context["entities"] = concepts.get("profiles", [])
                concept_context["revision"] = concepts["revision"]
                concept_context["work_scope_id"] = concepts["work_scope_id"]
                for concept in concept_context["entities"]:
                    for reference in concept.get("references", []):
                        concept_context["reference_frames"].append({"id": "concept_reference_" + reference["id"],
                            "at_ms": 0, "path": str(work_store.reference_path(concepts["work_scope_id"],
                                concept["concept_id"], reference["id"]))})
                left = max(0, 8 - len(context["reference_frames"]))
                concept_context["reference_frames"] = concept_context["reference_frames"][:left]
            args = [batch, local, context, concept_context, local_cues]
            inputs = run / "retrieval-inputs" / f"{digest([u['id'] for u in batch])}.json"
            frozen = read_json(inputs)
            if frozen and (run / "retrieval-receipts" / f"{frozen['key']}.json").is_file():
                key, args = frozen["key"], frozen["args"]
                batch, local, context, concept_context, local_cues = args
                permitted = {p["id"] for p in context["profiles"]}
            else:
                key = digest([STRATEGY, prompt_version, config["source_fingerprint"], config["subtitle_digest"],
                              model_identity(profile), args])
                atomic_json(inputs, {"key": key, "prompt_version": prompt_version, "args": args})
            receipt = run / "retrieval-receipts" / f"{key}.json"
            data = await self.call(job_id, profile, self.providers.analyze_retrieval, *args,
                frames=len(local) + len(context["reference_frames"]) + len(concept_context["reference_frames"]),
                context_tokens=len(str(args)),
                receipt_path=receipt, receipt_key=key)
            data = validate_batch(data, batch, local, permitted,
                                  {row["concept_id"] for row in concept_context["entities"]},
                                  {r["id"] for r in context.get("identity_regions", [])})
            for observation in data["observations"]:
                observation.update(identity_policy="identity-binding-v2", provenance={"prompt_version": prompt_version,
                    "receipt_key": key, "model": model_identity(profile), "evidence_kind": "current_visual"})
                store.publish_observation(asset_id, job["run_id"], observation)
            state["coverage"]["processed_units"] += len(batch)
            self.update(job_id, retrieval=state)
            await self._local_work(job_id, lambda check: (check(), self.search.sync_asset(self.library.records(asset_id), asset_id)))

        live = self.providers.get(profile["id"])
        width = max(1, min(4, live.get("max_concurrency", 1), config.get("execution", {}).get("max_concurrency", 4)))
        await bounded_map(batches, process, width=width, checkpoint=lambda: self.checkpoint(job_id))
        self._retrieval_metric(job_id, "local_records", started)
        state["capabilities"]["local_records"] = "completed"
        if config.get("retrieval_contract_version", 1) < 2:
            state["capabilities"]["identity"] = "completed"
        self.update(job_id, retrieval=state)
