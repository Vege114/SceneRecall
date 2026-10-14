from __future__ import annotations

import asyncio
import copy
import math
import sqlite3
import time
from contextlib import nullcontext

from pydantic import ValidationError

from . import codex_cli
from .library import Library, atomic_json, digest, now, read_json, uid
from .models import AnalysisInput
from .providers import (AIResult, CHARACTER_INSTRUCTION_TOKENS, MAX_CHARACTER_REFERENCES, MAX_OUTPUT_TOKENS,
                        PROMPT_VERSION, SUBTITLE_PROMPT_VERSION, ProviderManager)
from .workflows import WorkflowQueueMixin
from .scene_jobs import CODEX_CORE_FRAMES, CORE_FRAMES, SceneJobMixin
from .execution import ReservationBusy, bounded_map, execution_width, finish_on_cancel

CODEX_COST_BUDGET_MESSAGE = "Codex CLI 使用账户订阅额度，无法按 API 单价计算金额预算；请移除费用预算并保留请求数预算"


class StopJob(Exception):
    pass


class CharacterContextChanged(ValueError):
    pass


def model_identity(profile: dict) -> dict:
    """Only settings affecting model output belong in recognition cache keys."""
    # Preserve the exact legacy API identity so upgrades reuse paid recognition.
    identity = {key: profile.get(key) for key in ("protocol", "base_url", "model")}
    provider_type = profile.get("provider_type") or "openai_compatible"
    if provider_type != "openai_compatible":
        identity["provider_type"] = provider_type
    if profile.get("reasoning_effort") is not None:
        identity["reasoning_effort"] = profile["reasoning_effort"]
    return identity


def error_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        details = []
        for error in exc.errors(include_input=False, include_url=False)[:3]:
            location = ".".join(str(value) for value in error["loc"])
            reason = error["msg"].removeprefix("Value error, ")
            details.append(f"{location}：{reason}" if location else reason)
        return ("分析结果校验失败：" + "；".join(details))[:300]
    return str(exc)[:300] if isinstance(exc, (ValueError, FileNotFoundError)) else "处理失败，请检查模型连接、素材与依赖后重试"


def canonical_ocr_frames(frames: list[dict], start_ms: int, end_ms: int, interval: int) -> list[dict]:
    """Consolidate duplicate VFR samples while retaining actual evidence times."""
    import json
    grouped = {}
    for frame in sorted(frames, key=lambda item: item["at_ms"]):
        at = frame["at_ms"]
        if not start_ms <= at < end_ms:
            continue
        if at not in grouped:
            grouped[at] = {**frame, "lines": [dict(line) for line in frame["lines"]],
                           "duplicate_frame_ids": []}
            continue
        current = grouped[at]
        current["duplicate_frame_ids"].append(frame["frame_id"])
        if json.dumps(current["lines"], sort_keys=True) != json.dumps(frame["lines"], sort_keys=True):
            # Conflicting readings of the exact same source frame are uncertainty,
            # never two successive subtitle events or fabricated timing.
            lines = {(line["text"], line.get("language", "und")): line
                     for line in current["lines"] + frame["lines"]}
            current["lines"] = [{**line, "uncertain": True} for line in lines.values()]
    result = list(grouped.values())
    for index, frame in enumerate(result):
        next_at = result[index + 1]["at_ms"] if index + 1 < len(result) else end_ms
        frame["end_ms"] = min(end_ms, frame["at_ms"] + interval, next_at)
    return result


class JobQueue(SceneJobMixin, WorkflowQueueMixin):
    def __init__(self, library: Library, providers, search):
        from .characters import CharacterStore

        self.library, self.providers, self.search = library, providers, search
        self.characters = CharacterStore(library)
        self.path = library.root / "runtime" / "jobs.sqlite"
        self.task = None
        self.closed = False
        self._reservation_events = {}
        self._executing_jobs = set()
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self.init_workflows()
        for job in self.list():
            if job["status"] in ("running", "pausing", "cancelling"):
                job = self.recover_scene_receipt(job)
                state = {"running": "queued", "pausing": "paused", "cancelling": "cancelled"}[job["status"]]
                reservation = sum(call["attempts"] for call in job.get("inflight_calls", {}).values()) if "inflight_calls" in job else job.get("inflight_attempts", 0)
                updates = {}
                if reservation:
                    updates = {"request_count": job["request_count"] + reservation,
                               "uncertain_request_count": job.get("uncertain_request_count", 0) + reservation,
                               "cost": None, "cost_incomplete": True, "inflight_attempts": 0, "inflight_calls": {}}
                self.update(job["id"], status=state, **updates,
                            message="已恢复中断的任务；在途请求按最多三次保守计入，实际费用待核对" if reservation else "已恢复中断的任务；复用已完成产物")

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=20)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def list(self) -> list[dict]:
        import json
        with self.connect() as db:
            return sorted([json.loads(r[0]) for r in db.execute("SELECT data FROM jobs")], key=lambda j: j["created_at"], reverse=True)

    def get(self, job_id: str) -> dict:
        import json
        with self.connect() as db:
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise FileNotFoundError("任务不存在")
        return json.loads(row[0])

    def update(self, job_id: str, **fields) -> dict:
        import json
        with self.library.lock, self.connect() as db:
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise FileNotFoundError("任务不存在")
            job = {**json.loads(row[0]), **fields, "updated_at": now()}
            db.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(job, ensure_ascii=False), job_id))
        return job

    def _new_job(self, job_type: str, config: dict, asset_id=None) -> dict:
        job = {"id": uid("job"), "type": job_type, "asset_id": asset_id, "status": "queued", "stage": "queued",
               "progress": 0, "completed": 0, "total": 0, "message": "等待处理", "error": None,
               "request_count": 0, "cost": None, "known_cost": 0.0, "cost_incomplete": False,
               "usage": {}, "created_at": now(), "updated_at": now(),
               "config": config, "run_id": uid("run"), "failures": [], "checkpoints": []}
        catalog = self.library.organization._read()
        job["deletion_revision"] = (catalog["assets"].get(asset_id, {}).get("deletion_revision", 0)
                                    if asset_id else catalog.get("deletion_revision", 0))
        if job_type == "analysis" and "vision" in config.get("stages", []):
            job["sampling_strategy"] = config.get("sampling_strategy", "uniform-v1")
        if config.get("warnings"):
            job["warnings"] = list(config["warnings"])
        return job

    def add(self, job_type: str, config: dict, asset_id=None) -> dict:
        import json
        job = self._new_job(job_type, config, asset_id)
        with self.library.lock, self.connect() as db:
            db.execute("INSERT INTO jobs VALUES (?,?)", (job["id"], json.dumps(job, ensure_ascii=False)))
        return job

    def validate(self, request: AnalysisInput, settings: dict, *, scene_deepening: bool = False) -> tuple[dict, dict]:
        asset = self.library.get_asset(request.asset_id)
        if not asset["source_available"]:
            raise ValueError("原视频离线，请先重新定位")
        end = request.end_ms or asset["duration_ms"]
        if not request.start_ms < end <= asset["duration_ms"]:
            raise ValueError("分析时间超出视频范围")
        bindings = settings.get("bindings", {})
        strategy = request.analysis_strategy or ("scene-chunks-v1" if bindings.get("preparation") else "content-keyframes-v1")
        if "vision" in request.stages and strategy == "scene-chunks-v1":
            from .scenes import SceneStore
            overlap = [row for row in SceneStore(self.library).list(request.asset_id)
                       if row["start_ms"] < end and row["end_ms"] > request.start_ms
                       and not request.start_ms <= row["start_ms"] < row["end_ms"] <= end]
            if overlap:
                a = min(request.start_ms, *(row["start_ms"] for row in overlap))
                b = max(end, *(row["end_ms"] for row in overlap))
                raise ValueError(f"重分块范围需包含相交片段的完整边界，请选择 {a / 1000:g}–{b / 1000:g} 秒；单片段可使用深入分析")
        snapshots, source_snapshots, sources = {}, {}, {}
        required = list(request.stages)
        if "vision" in required and strategy == "scene-chunks-v1" and not scene_deepening:
            required.append("preparation")
        if set(request.model_overrides) - set(required):
            raise ValueError("只能为本次任务启用的阶段选择模型和思考等级")
        for stage in required:
            if stage == "subtitle" and asset["subtitle_mode"] != "embedded":
                raise ValueError("容器或外挂文字字幕已在本地解析，无需调用字幕视觉模型")
            profile_id = bindings.get(stage)
            if not profile_id:
                raise ValueError(f"请先在设置中为 {stage} 绑定模型")
            if profile_id not in sources:
                sources[profile_id] = self.providers.get(profile_id)
            profile = copy.deepcopy(sources[profile_id])
            source_snapshots[stage] = copy.deepcopy(profile)
            if stage not in profile.get("capabilities", []):
                raise ValueError(f"模型未声明 {stage} 能力，请检查连接配置")
            selected = request.model_overrides.get(stage)
            if selected is not None:
                if selected.profile_id != profile_id or selected.provider_type != profile.get("provider_type"):
                    raise ValueError("模型来源或绑定连接已变化，请刷新后重新选择任务模型")
                profile = copy.deepcopy(profile)
                if "model" in selected.model_fields_set:
                    profile["model"] = selected.model
                if "reasoning_effort" in selected.model_fields_set:
                    profile["reasoning_effort"] = selected.reasoning_effort
                codex_cli.validate_model(profile.get("model", ""))
                codex_cli.validate_reasoning_effort(profile.get("model", ""), profile.get("reasoning_effort"))
            if request.max_cost is not None and profile.get("provider_type") == "codex_cli":
                raise ValueError(CODEX_COST_BUDGET_MESSAGE)
            if request.max_cost is not None and (profile.get("input_price_per_million") is None or profile.get("output_price_per_million") is None):
                raise ValueError("设置费用预算前，请填写所用模型的输入与输出单价")
            snapshots[stage] = profile
        config = {**request.model_dump(), "analysis_strategy": strategy, "end_ms": end,
                  "bindings": dict(bindings), "profile_snapshots": snapshots,
                  "source_profile_snapshots": source_snapshots,
                  "model_overrides": {stage: selected.model_dump(exclude_unset=True)
                                      for stage, selected in request.model_overrides.items()}}
        if strategy == "scene-chunks-v1" and "preparation" in snapshots:
            config["visual_batch_size"] = (CODEX_CORE_FRAMES
                if snapshots["preparation"].get("provider_type") == "codex_cli" else CORE_FRAMES)
        if "vision" in request.stages:
            context = self.character_context(request.asset_id, scene=strategy == "scene-chunks-v1")
            config["character_snapshot"] = {key: context[key] for key in ("scope_id", "revision", "membership_revision")}
            config["sampling_strategy"] = strategy
            if request.model_fields_set & {"window_ms", "frames_per_window"}:
                config["warnings"] = ["window_ms / frames_per_window 已弃用；新画面任务统一使用内容关键帧。"]
            if request.keyframe_plan_id:
                self.validated_plan(request.asset_id, request.start_ms, end, request.keyframe_plan_id)
        return asset, config

    def estimate(self, request: AnalysisInput, settings: dict) -> dict:
        _, config = self.validate(request, settings)
        duration = config["end_ms"] - request.start_ms
        plan = None
        if "vision" in request.stages:
            if not request.keyframe_plan_id:
                raise ValueError("请先扫描内容关键帧，再按实际选帧数量估算用量")
            plan = self.validated_plan(request.asset_id, request.start_ms, config["end_ms"], request.keyframe_plan_id)
        if config.get("analysis_strategy") == "scene-chunks-v1" and plan:
            return self.estimate_scenes(config, plan)
        vision = len(plan["batches"]) if plan else 0
        subtitle_frames = math.ceil(duration / 1000 * request.subtitle_fps) if "subtitle" in request.stages else 0
        subtitle_requests = math.ceil(subtitle_frames / 8)
        estimate_cost = 0.0
        unknown = False
        character_context = self.character_context(request.asset_id, config.get("character_snapshot")) if vision else None
        context_tokens = (len(ProviderManager.character_context_text(character_context).encode("utf-8"))
                          + CHARACTER_INSTRUCTION_TOKENS) if vision else 0
        reference_count = len((character_context or {}).get("reference_frames", []))
        calls = [("vision", len(batch["frame_ids"])) for batch in (plan or {}).get("batches", [])]
        calls += [("subtitle", min(8, subtitle_frames - i * 8)) for i in range(subtitle_requests)]
        for stage, frame_count in calls:
            profile = config["profile_snapshots"][stage]
            unit = self.estimated_call_cost(profile, frame_count + (reference_count if stage == "vision" else 0),
                                            context_tokens=context_tokens if stage == "vision" else 0)
            unknown |= unit is None
            estimate_cost += unit or 0
        warnings = ["画面请求数来自已完成的内容关键帧计划；字幕去重可能减少 OCR 请求。",
                    "估算不含自动向量索引；重试最多三次。费用以供应商实际用量为准。",
                    "请求数不足或预算不足时任务暂停，不会自动扩大预算。"]
        if vision:
            warnings.append("人物识别会逐镜头携带同一作品的现有档案及最多 8 张参考图；费用估算已含当前档案，后续档案增长会增加费用，每次请求前重新检查预算。")
        if any(profile.get("provider_type") == "codex_cli" for profile in config["profile_snapshots"].values()):
            warnings.append("Codex CLI 由已登录账户的订阅额度管理，金额费用未知，并不表示免费；请求数预算仍生效。")
        if request.model_fields_set & {"window_ms", "frames_per_window"}:
            warnings.append("window_ms / frames_per_window 已弃用；新画面任务统一使用内容关键帧。")
        return {"duration_ms": duration, "estimated_frames": len((plan or {}).get("frames", [])) + subtitle_frames,
                "estimated_requests": vision + subtitle_requests, "cost_estimate": None if unknown else round(estimate_cost, 6),
                "currency": "USD", "warnings": warnings}

    @staticmethod
    def estimated_call_cost(profile: dict, frames: int = 0, text_count: int = 1, context_tokens: int = 0):
        if profile.get("provider_type") == "codex_cli":
            return None
        a, b = profile.get("input_price_per_million"), profile.get("output_price_per_million")
        if a is None or b is None:
            return None
        return ((2048 * text_count + 4096 * frames + context_tokens) * a + MAX_OUTPUT_TOKENS * b) / 1_000_000

    def character_context(self, asset_id: str, expected: dict | None = None, *, scene: bool = False) -> dict:
        context = self.characters.context(asset_id)
        context["membership_revision"] = self.library.get_asset(asset_id).get("membership_revision", 0)
        keys = ("scope_id", "revision", "membership_revision") if expected and "membership_revision" in expected else ("scope_id", "revision")
        if expected is not None and any(context[key] != expected.get(key) for key in keys):
            raise CharacterContextChanged("角色档案已修改，此任务使用的是旧版档案。已完成结果保留；请在角色档案中按新版档案重新识别")
        if scene:
            context = self._bounded_scene_profiles(context)
        ProviderManager.character_context_text(context)
        references = []
        # All dossiers remain available as text. Bound expensive image context and
        # prefer recently enriched profiles when the cast has more than eight roles.
        for profile in sorted(context["profiles"], key=lambda p: p.get("updated_at", ""), reverse=True):
            representative = profile.get("representative")
            if not representative:
                continue
            try:
                path = self.library.frame_path(representative["asset_id"], representative["frame_id"])
            except (FileNotFoundError, ValueError):
                continue
            references.append({"id": "reference:" + profile["id"], "character_id": profile["id"],
                               "path": str(path), "at_ms": 0, "box": representative.get("box")})
            if len(references) == MAX_CHARACTER_REFERENCES:
                break
        return {**context, "reference_frames": references}

    def _check_deletion(self, job):
        catalog = self.library.organization._read()
        membership = catalog["assets"].get(job.get("asset_id"), {})
        revision = membership.get("deletion_revision", 0) if job.get("asset_id") else catalog.get("deletion_revision", 0)
        if (job.get("stale_deletion") or membership.get("trash_id")
                or ((job.get("asset_id") or job["type"] == "index") and revision != job.get("deletion_revision", 0))):
            raise ValueError("相关资料已删除或恢复，旧任务不可继续，请创建新任务")

    def control(self, job_id: str, action: str, overrides=None) -> dict:
        job = self.get(job_id)
        state = job["status"]
        if action == "pause" and state in ("queued", "running"):
            return self.update(job_id, status="paused" if state == "queued" else "pausing", message="将在当前步骤结束后暂停")
        if action == "cancel" and state not in ("completed", "cancelled"):
            return self.update(job_id, status="cancelling" if state in ("running", "pausing", "cancelling") or job_id in self._executing_jobs else "cancelled", message="正在取消；已提交的结果保留")
        if action in ("resume", "retry") and state in ("paused", "failed", "partial", "cancelled"):
            self._check_deletion(job)
            config = job["config"]
            if job.get("stale_organization"):
                raise ValueError("视频归属已经改变，请创建新的分析任务；已完成资料保留")
            if job["type"] == "analysis" and "vision" in config.get("stages", []):
                self.character_context(job["asset_id"], config.get("character_snapshot"))
            if (job["type"] == "analysis" and "subtitle" in config.get("stages", [])
                    and self.library.get_asset(job["asset_id"])["subtitle_mode"] != "embedded"):
                raise ValueError("字幕来源已切换为文字字幕，请创建只包含画面理解的新任务")
            if overrides:
                if "model_overrides" in overrides:
                    raise ValueError("恢复或重试会沿用任务已保存的模型选择；如需切换模型，请创建新任务")
                for key in ("max_requests", "max_cost"):
                    if key in overrides:
                        if key == "max_cost" and overrides[key] is None:
                            config[key] = None
                            continue
                        value = overrides[key]
                        if (isinstance(value, bool) or not isinstance(value, (int, float))
                                or not math.isfinite(value) or value <= 0):
                            raise ValueError("预算必须为正数")
                        if key == "max_requests" and (not isinstance(value, int) or value > 100000):
                            raise ValueError("请求预算必须是 1–100000 的整数")
                        config[key] = value
            if config.get("max_cost") is not None and any(
                    profile.get("provider_type") == "codex_cli"
                    for profile in config.get("profile_snapshots", {}).values()):
                raise ValueError(CODEX_COST_BUDGET_MESSAGE)
            return self.update(job_id, status="queued", config=config, error=None, failures=[], message="等待恢复，复用成功结果")
        raise ValueError("当前任务状态不支持此操作")

    def _pause_job(self, job_id, message):
        draining = job_id in self._executing_jobs
        self.update(job_id, status="pausing" if draining else "paused", message=message)

    def checkpoint(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job["status"] in ("pausing", "paused"):
            if job_id not in self._executing_jobs:
                self.update(job_id, status="paused", message="已暂停，可继续处理")
            raise StopJob()
        if job["status"] in ("cancelling", "cancelled"):
            if job_id not in self._executing_jobs:
                self.update(job_id, status="cancelled", message="已取消，已完成资料保留")
            raise StopJob()
        if (job["type"] == "analysis" and "subtitle" in job["config"].get("stages", [])
                and self.library.get_asset(job["asset_id"])["subtitle_mode"] != "embedded"):
            self._pause_job(job_id, "字幕来源已切换为文字字幕，旧 OCR 任务已暂停；请创建只包含画面理解的新任务")
            raise StopJob()
        return job

    def profile_source_unchanged(self, config: dict, profile: dict) -> bool:
        original = next((snapshot for snapshot in config.get("source_profile_snapshots", {}).values()
                         if snapshot["id"] == profile["id"]), profile)
        return model_identity(self.providers.get(profile["id"])) == model_identity(original)

    def before_call(self, job_id: str, profile: dict, frames=0, text_count=1, context_tokens=0,
                    *, call_id=None, receipt_path=None, stage=None, wait_for_budget=False):
        # Reserve retries and cost together with sibling calls in one transaction.
        with self.library.lock:
            job = self.checkpoint(job_id)
            current_profile = self.providers.get(profile["id"])
            if not self.profile_source_unchanged(job["config"], profile):
                self._pause_job(job_id, message="任务所用连接类型、模型地址或名称已变化，请还原配置后继续或创建新任务")
                raise StopJob()
            active = dict(job.get("inflight_calls", {}))
            legacy = job.get("inflight_attempts", 0) if "inflight_calls" not in job else 0
            reserved_attempts = sum(call["attempts"] for call in active.values()) + legacy
            count_budget = job["config"].get("max_requests", 1000)
            if job["request_count"] + reserved_attempts + 3 > count_budget:
                if wait_for_budget and reserved_attempts and job["request_count"] + 3 <= count_budget:
                    raise ReservationBusy()
                self._pause_job(job_id, message="剩余请求预算不足以预留最多三次尝试；请提高预算或创建新任务复用结果")
                raise StopJob()
            max_cost = job["config"].get("max_cost")
            reserve = 0
            if max_cost:
                if current_profile.get("provider_type") == "codex_cli":
                    self._pause_job(job_id, message=CODEX_COST_BUDGET_MESSAGE)
                    raise StopJob()
                if job.get("cost_incomplete"):
                    self._pause_job(job_id, message="供应商未返回完整费用用量；当前费用未知，请核对账单或移除费用预算后继续")
                    raise StopJob()
                estimate = self.estimated_call_cost(current_profile, frames, text_count, context_tokens)
                if estimate is not None and (stage or job.get("stage")) in {"transcript", "preparation", "context_refresh"}:
                    from .scene_preparation import MAX_PREPARATION_OUTPUT_TOKENS
                    estimate += ((MAX_PREPARATION_OUTPUT_TOKENS - MAX_OUTPUT_TOKENS)
                                 * current_profile["output_price_per_million"] / 1_000_000)
                reserve = (estimate or 0) * 3
                spent = job.get("known_cost", job["cost"] or 0)
                reserved_cost = sum(call.get("cost", 0) for call in active.values())
                if estimate is None or spent + reserved_cost + reserve > max_cost:
                    if wait_for_budget and active and estimate is not None and spent + reserve <= max_cost:
                        raise ReservationBusy()
                    self._pause_job(job_id, message="费用预算不足以预留下一批请求；请提高预算后继续")
                    raise StopJob()
            call_id = call_id or uid("call")
            active[call_id] = {"attempts": 3, "cost": reserve, "receipt_path": receipt_path,
                               "stage": stage or job.get("stage")}
            self.update(job_id, inflight_calls=active, inflight_attempts=reserved_attempts + 3)
            return call_id

    async def reserve_call(self, job_id, profile, **options):
        event = self._reservation_events.setdefault(job_id, asyncio.Event())
        while True:
            if self.closed:
                raise StopJob()
            event.clear()
            try:
                return self.before_call(job_id, profile, wait_for_budget=True, **options)
            except ReservationBusy:
                await event.wait()

    def _release_reservation(self, job, call_id=None):
        active = dict(job.get("inflight_calls", {}))
        if call_id is None and active:
            call_id = next(iter(active))
        active.pop(call_id, None)
        return {"inflight_calls": active,
                "inflight_attempts": sum(call["attempts"] for call in active.values())}

    def _wake_reservations(self, job_id):
        event = self._reservation_events.get(job_id)
        if event:
            event.set()

    def account(self, job_id: str, result, call_id=None):
        with self.library.lock:
            job = self.get(job_id)
            usage = dict(job["usage"])
            for key, value in getattr(result, "usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
            cost = getattr(result, "cost", None)
            attempts = getattr(result, "request_count", 0)
            incomplete = job.get("cost_incomplete", job["request_count"] > 0 and job["cost"] is None)
            incomplete |= attempts > 0 and cost is None
            known = job.get("known_cost", job["cost"] or 0) + (cost or 0)
            total_attempts = job["request_count"] + attempts
            self.update(job_id, usage=usage, request_count=total_attempts,
                        cost=known if total_attempts and not incomplete else None,
                        known_cost=known, cost_incomplete=incomplete,
                        **self._release_reservation(job, call_id))
        self._wake_reservations(job_id)

    async def invoke_profile(self, profile: dict, method, *args):
        scope = getattr(self.providers, "use_profile", None)
        with scope(profile) if scope else nullcontext():
            return await method(profile["id"], *args)

    async def call(self, job_id: str, profile: dict, method, *args, frames=0, text_count=1, context_tokens=0,
                   receipt_path=None, receipt_key=None):
        job = self.checkpoint(job_id)
        call_id = uid("call")
        receipt_key = receipt_key or call_id
        receipt_path = receipt_path or (self.library.root / "runtime" / "call-receipts" / job_id / f"{receipt_key}.json")
        cached = read_json(receipt_path)
        if cached is not None:
            self._account_scene_receipt(job_id, cached)
            return copy.deepcopy(cached["data"])
        await self.reserve_call(job_id, profile, call_id=call_id,
            receipt_path=str(receipt_path.relative_to(self.library.root)), frames=frames,
            text_count=text_count, context_tokens=context_tokens)
        async def perform():
            started = time.monotonic()
            try:
                result = await self.invoke_profile(profile, method, *args)
            except Exception as exc:
                self.account(job_id, exc, call_id)
                raise
            receipt = {"receipt_id": receipt_key, "call_id": call_id, "job_id": job_id,
                       "stage": job.get("stage", "unknown"), "data": result.data,
                       "usage": result.usage, "request_count": result.request_count, "cost": result.cost,
                       "elapsed_ms": round((time.monotonic() - started) * 1000)}
            atomic_json(receipt_path, receipt)
            self._account_scene_receipt(job_id, receipt)
            return copy.deepcopy(result.data)
        return await finish_on_cancel(perform())

    async def start(self):
        self.closed = False
        self.task = asyncio.create_task(self.loop())

    async def stop(self):
        self.closed = True
        for event in self._reservation_events.values():
            event.set()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def loop(self):
        while not self.closed:
            queued = [job for job in reversed(self.list()) if job["status"] == "queued"]
            if not queued:
                await asyncio.sleep(.5)
                continue
            await self.execute(queued[0]["id"])

    async def execute(self, job_id: str):
        with self.library.lock:
            current = self.get(job_id)
            if current["status"] != "queued":
                return
            try:
                self._check_deletion(current)
            except ValueError as exc:
                self.update(job_id, status="cancelled", stale_deletion=True, message=str(exc))
                return
            job = self.update(job_id, status="running", error=None)
            self._executing_jobs.add(job_id)
        try:
            if job["type"] == "analysis":
                await self.analyze(job_id)
            elif job["type"] == "index":
                await self.reindex(job_id)
            elif job["type"] == "import":
                await self.import_files(job_id)
            elif job["type"] == "keyframe_scan":
                await self.scan_plan(job_id)
            elif job["type"] == "proxy":
                from .media import create_proxy
                path = self.library.require_source(job["asset_id"])
                self.update(job_id, stage="proxy", message="正在生成浏览器播放代理")
                await asyncio.to_thread(create_proxy, path, self.library.asset_dir(job["asset_id"]) / "proxy" / "playback.mp4")
            self.checkpoint(job_id)
            failures = self.get(job_id)["failures"]
            self.update(job_id, status="partial" if failures else "completed", progress=1,
                        message=f"完成，{len(failures)} 个单元失败，可重试" if failures else "处理完成")
        except StopJob:
            stopped = self.get(job_id)
            if stopped["status"] in {"pausing", "paused"}:
                # All owned units have drained before terminal status permits
                # resume, move or deletion through the public API.
                self.update(job_id, status="paused")
            elif stopped["status"] in {"cancelling", "cancelled"}:
                self.update(job_id, status="cancelled", message="已取消，已完成资料保留")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Provider exceptions are already sanitized. Do not persist arbitrary HTTP bodies.
            message = error_message(exc)
            self.update(job_id, status="failed", error=message, message=message)
        finally:
            self._executing_jobs.discard(job_id)
            job = self.get(job_id)
            if job.get("asset_id") and job["type"] == "analysis":
                self.library.save_run(job["asset_id"], job["run_id"], {k: job[k] for k in
                    ("config", "status", "usage", "request_count", "cost", "known_cost", "cost_incomplete",
                     "created_at", "updated_at", "failures", "stage_metrics") if k in job})

    async def frame(self, asset_id, source, at_ms, crop=None):
        return (await self.frames(asset_id, source, [at_ms], crop))[0]

    async def frames(self, asset_id, source, times_ms, crop=None):
        from .media import extract_frames
        self.library.require_source(asset_id)
        frames, missing_times, missing_paths = {}, [], []
        registry = read_json(self.library.asset_dir(asset_id) / "frames" / "registry.json", {})
        for at in times_ms:
            frame_id = "f_" + digest([asset_id, at, crop])
            path = self.library.asset_dir(asset_id) / "frames" / f"{frame_id}.jpg"
            if path.is_file() and frame_id in registry:
                frames[at] = {"id": frame_id, "path": str(path), "at_ms": registry[frame_id]["at_ms"],
                              "requested_at_ms": at}
            else:
                missing_times.append(at)
                missing_paths.append(path)
        if missing_times:
            extracted = await asyncio.to_thread(extract_frames, source, missing_times, missing_paths, crop)
            for at, value in zip(missing_times, extracted, strict=True):
                value["requested_at_ms"] = at
                self.library.register_frame(asset_id, value)
                frames[at] = value
        return [frames[at] for at in times_ms]

    async def analyze(self, job_id: str):
        config = self.get(job_id)["config"]
        if config.get("analysis_strategy") == "scene-chunks-v1" and "vision" in config["stages"]:
            await self.analyze_scenes(job_id)
        else:
            await self.analyze_legacy(job_id)

    async def analyze_legacy(self, job_id: str, config_override=None, skip_reindex=False):
        from .characters import StaleCharacterRevision
        from .media import detect_shots, make_windows, sample_times
        from .subtitles import candidate_frames, merge_ocr_frames

        job = self.checkpoint(job_id)
        config, asset_id, run_id = job["config"], job["asset_id"], job["run_id"]
        if config_override is not None:
            config = config_override
        source = self.library.require_source(asset_id)
        directory = self.library.asset_dir(asset_id)
        for snapshot in config["profile_snapshots"].values():
            if not self.profile_source_unchanged(config, snapshot):
                raise ValueError("任务所用模型配置已变化，请创建新任务；已有识别结果仍可复用")
        start, end = config["start_ms"], config["end_ms"]
        windows = []
        plan = None
        if "vision" in config["stages"]:
            # Upgrade a pending pre-0.2 job once; every resumed/new job thereafter
            # remains bound to the human revision that was actually requested.
            context = self.character_context(asset_id, config.get("character_snapshot"))
            if "character_snapshot" not in config:
                config = {**config, "character_snapshot": {key: context[key] for key in ("scope_id", "revision")}}
                self.update(job_id, config=config)
            if config.get("sampling_strategy") == "content-keyframes-v1":
                if config.get("keyframe_plan_id"):
                    plan = self.validated_plan(asset_id, start, end, config["keyframe_plan_id"])
                else:
                    plan = await self.scan_plan(job_id, asset_id=asset_id, start_ms=start, end_ms=end)
                    config = {**config, "keyframe_plan_id": plan["id"]}
                    self.update(job_id, config=config)
                windows = plan["batches"]
            else:
                # Persisted pre-keyframe jobs retain their original paid checkpoints.
                self.update(job_id, stage="segmentation", message="旧版任务：正在检测镜头边界")
                segmentation_id = digest([start, end, "content-27-v1"])
                segmentation_file = directory / "segmentation" / segmentation_id / "shots.json"
                shots = read_json(segmentation_file)
                if shots is None:
                    shots = await asyncio.to_thread(detect_shots, source, start, end)
                    atomic_json(segmentation_file, shots)
                windows = make_windows(shots, config["window_ms"])
        interval = max(1, round(1000 / config["subtitle_fps"]))
        subtitle_times = list(range(start, end, interval)) if "subtitle" in config["stages"] else []
        total = len(windows) + math.ceil(len(subtitle_times) / 8)
        stage_state = copy.deepcopy(self.get(job_id).get("legacy_stage_state", {}))
        fingerprint = self.library.get_asset(asset_id)["fingerprint"]
        subtitle_signature = None
        subtitle_ready = False
        if subtitle_times:
            subtitle_signature = digest([fingerprint, start, end, interval, config["subtitle_crop"],
                model_identity(config["profile_snapshots"]["subtitle"]), SUBTITLE_PROMPT_VERSION])
            saved = stage_state.get("subtitle", {})
            subtitle_ready = (saved.get("signature") == subtitle_signature
                and saved.get("active") == self.library.active(asset_id).get("subtitle_runs")
                and saved.get("subtitle_digest") == digest(self.library.subtitles(asset_id)))
        completed = math.ceil(len(subtitle_times) / 8) if subtitle_ready else 0
        self.update(job_id, total=total, completed=completed)
        # Validate committed window references in one pass. Resuming the index
        # or OCR stage never reconstructs character context for finished vision.
        observations = {record.get("cache_key"): record for record in self.library.observations(asset_id)} if windows else {}
        vision_signature = (digest([fingerprint, windows, config.get("keyframe_plan_id"),
            config["frames_per_window"], config["character_snapshot"],
            model_identity(config["profile_snapshots"]["vision"]), PROMPT_VERSION]) if windows else None)
        saved_vision = stage_state.get("vision", {})
        finished_windows = (dict(saved_vision.get("units", {}))
            if saved_vision.get("signature") == vision_signature else {})

        def save_stage(stage, value):
            state = copy.deepcopy(self.get(job_id).get("legacy_stage_state", {}))
            state[stage] = value
            self.update(job_id, legacy_stage_state=state)

        pending_windows = []
        for window in windows:
            saved_key = finished_windows.get(window["id"])
            if isinstance(saved_key, str) and saved_key in observations:
                record = observations[saved_key]
                if (record.get("id") == "obs_" + digest([asset_id, window["start_ms"], window["end_ms"]])
                        and record.get("start_ms") == window["start_ms"]
                        and record.get("end_ms") == window["end_ms"]):
                    completed += 1
                    continue
            pending_windows.append(window)
        if len(pending_windows) != len(windows):
            self.update(job_id, completed=completed, progress=completed / max(total, 1),
                        message="已恢复完成的画面识别，继续后续阶段")
        for window in pending_windows:
            self.checkpoint(job_id)
            self.character_context(asset_id, config["character_snapshot"])
            profile = config["profile_snapshots"]["vision"]
            sampling = ([plan["id"], plan["algorithm"],
                         [[frame["pts"], frame["time_base"]] for frame in plan["frames"] if frame["id"] in window["frame_ids"]]]
                        if plan else config["frames_per_window"])
            cache_key = digest([self.library.get_asset(asset_id)["fingerprint"], window, model_identity(profile),
                                sampling, PROMPT_VERSION, "vision", config["character_snapshot"]])
            checkpoint_key = "vision:" + cache_key
            record_id = "obs_" + digest([asset_id, window["start_ms"], window["end_ms"]])
            stage_path = directory / "analyses" / run_id / "vision-batches" / f"{cache_key}.json"
            existing = observations.get(cache_key)
            try:
                if existing and (not config["force"] or existing.get("run_id") == run_id
                                 or checkpoint_key in self.get(job_id)["checkpoints"]):
                    finished_windows[window["id"]] = cache_key
                    save_stage("vision", {"signature": vision_signature, "units": finished_windows})
                    completed += 1
                    self.update(job_id, completed=completed, progress=completed/max(total, 1), message="复用已完成的画面识别")
                    continue
                self.update(job_id, stage="vision", message=f"画面识别 {window['start_ms']/1000:.1f}–{window['end_ms']/1000:.1f} 秒")
                data = read_json(stage_path)
                if data is None:
                    if plan:
                        wanted = set(window["frame_ids"])
                        frames = [{**frame, "path": str(self.library.frame_path(asset_id, frame["id"]))}
                                  for frame in plan["frames"] if frame["id"] in wanted]
                    else:
                        frames = await self.frames(asset_id, source, sample_times(window["start_ms"], window["end_ms"], config["frames_per_window"]))
                    frames = [frame for frame in frames if window["start_ms"] <= frame["at_ms"] < window["end_ms"]]
                    if not frames:
                        raise ValueError("此窗口没有可用的实际视频帧，未跨镜头取用证据")
                    context = self.character_context(asset_id, config["character_snapshot"])
                    context_text = ProviderManager.character_context_text(context)
                    # UTF-8 bytes bound dossier tokenization; reference images
                    # also consume the normal image reserve.
                    data = await self.call(job_id, profile, self.providers.analyze, frames, window["start_ms"], window["end_ms"],
                                           context, frames=len(frames) + len(context["reference_frames"]),
                                           context_tokens=len(context_text.encode("utf-8")) + CHARACTER_INSTRUCTION_TOKENS,
                                           receipt_path=directory / "analyses" / run_id / "vision-receipts" / f"{cache_key}.json",
                                           receipt_key="vision:" + cache_key)
                    allowed = {f["id"] for f in frames}
                    if not set(data.get("evidence_frame_ids", [])).issubset(allowed):
                        raise ValueError("模型引用了不存在的证据帧")
                    data = {**data, "id": record_id, "asset_id": asset_id, "run_id": run_id, "window_id": window["id"],
                            "shot_id": window["shot_id"], "start_ms": window["start_ms"], "end_ms": window["end_ms"],
                            "cache_key": cache_key, "evidence_frame_ids": data.get("evidence_frame_ids") or list(allowed),
                            "provenance": {"profile_id": profile["id"], "model": profile["model"], "prompt_version": PROMPT_VERSION,
                                           "character_scope_id": context["scope_id"], "character_revision": context["revision"],
                                           "character_context_digest": digest(context_text),
                                           "character_profile_ids": [p["id"] for p in context["profiles"]]}}
                    if plan:
                        data["provenance"].update(keyframe_plan_id=plan["id"], sampling_strategy=plan["algorithm"],
                            selected_frames=[{key: frame[key] for key in ("id", "pts", "time_base", "at_ms", "reason")} for frame in frames])
                    # Persist the validated provider result before either registry
                    # or observation publication. Replays preserve local entity IDs
                    # and cannot create orphan identities or repay a finished call.
                    atomic_json(stage_path, data)
                if any(data.get(key) != expected for key, expected in {
                    "id": record_id, "asset_id": asset_id, "run_id": run_id, "cache_key": cache_key,
                    "start_ms": window["start_ms"], "end_ms": window["end_ms"],
                }.items()):
                    raise ValueError("人物识别暂存结果不属于当前运行，请创建新任务")
                with self.library.lock:
                    self.character_context(asset_id, config["character_snapshot"])
                    data = self.characters.apply_observation(asset_id, data, expected_revision=config["character_snapshot"]["revision"])
                    self.library.save_observation(asset_id, data)
                self.search.upsert(self.library.records(asset_id))
                self.update(job_id, checkpoints=self.get(job_id)["checkpoints"] + [checkpoint_key])
                observations[cache_key] = data
                finished_windows[window["id"]] = cache_key
                save_stage("vision", {"signature": vision_signature, "units": finished_windows})
            except StopJob:
                raise
            except (CharacterContextChanged, StaleCharacterRevision):
                raise
            except Exception as exc:
                self.update(job_id, failures=self.get(job_id)["failures"] + [{"unit": window["id"], "message": error_message(exc)}])
            completed += 1
            self.update(job_id, completed=completed, progress=completed/max(total, 1))
        ocr_started = time.monotonic()
        async def ocr_batch(index, times):
            self.checkpoint(job_id)

            profile = config["profile_snapshots"]["subtitle"]
            cache_key = digest([self.library.get_asset(asset_id)["fingerprint"], times, min(end, times[-1] + interval),
                                config["subtitle_crop"], model_identity(profile), SUBTITLE_PROMPT_VERSION, "subtitle"])
            checkpoint_key = "subtitle:" + cache_key
            cache_path = directory / "analyses" / run_id / "subtitle-batches" / f"{cache_key}.json"
            possible = sorted((directory / "analyses").glob(f"*/subtitle-batches/{cache_key}.json"),
                              key=lambda path: path.stat().st_mtime_ns, reverse=True)
            cached = read_json(cache_path) if cache_path.exists() else (read_json(possible[0]) if possible and not config["force"] else None)
            try:
                self.update(job_id, stage="subtitle", message=f"字幕识别 {times[0]/1000:.1f} 秒")
                if cached is None:
                    sampled = await self.frames(asset_id, source, times, config["subtitle_crop"])
                    by_time = {}
                    for frame in sampled:
                        at = frame["at_ms"]
                        if not start <= at < end:
                            continue
                        if at in by_time:
                            by_time[at]["requested_times_ms"].append(frame["requested_at_ms"])
                        else:
                            by_time[at] = {**frame, "frame_id": frame["id"], "end_ms": min(end, at + interval),
                                           "requested_times_ms": [frame["requested_at_ms"]]}
                    frames = list(by_time.values())
                    if not frames:
                        raise ValueError("字幕批次没有落在分析范围内的实际视频帧")
                    frame_metadata = {frame["id"]: frame for frame in frames}
                    candidates = candidate_frames(frames)
                    data = await self.call(job_id, profile, self.providers.recognize_subtitles, candidates,
                                           frames=len(candidates),
                                           receipt_path=directory / "analyses" / run_id / "subtitle-receipts" / f"{cache_key}.json",
                                           receipt_key="subtitle:" + cache_key)
                    by_id = {r["frame_id"]: r for r in data}
                    if set(by_id) != {f["id"] for f in candidates}:
                        raise ValueError("字幕模型未逐帧返回结果；空字幕也须返回空行列表")
                    cached = []
                    for candidate in candidates:
                        result = by_id[candidate["id"]]
                        for original in candidate.get("source_frames", [candidate]):
                            frame_id = original.get("frame_id", candidate["id"])
                            cached.append({"frame_id": frame_id, "at_ms": original["at_ms"],
                                           "requested_times_ms": frame_metadata[frame_id]["requested_times_ms"],
                                           "end_ms": original.get("end_ms", min(end, original["at_ms"] + interval)),
                                           "lines": result["lines"]})
                    atomic_json(cache_path, cached)
                checkpoints = self.get(job_id)["checkpoints"]
                if checkpoint_key not in checkpoints:
                    self.update(job_id, checkpoints=checkpoints + [checkpoint_key])
            except StopJob:
                raise
            except Exception as exc:
                self.update(job_id, failures=self.get(job_id)["failures"] + [{"unit": checkpoint_key, "message": error_message(exc)}])
                return None
            return cached
        batches = ([] if subtitle_ready else
            [subtitle_times[offset:offset + 8] for offset in range(0, len(subtitle_times), 8)])
        def ocr_done(index, result):
            nonlocal completed
            completed += 1
            self.update(job_id, completed=completed, progress=completed / max(total, 1))
        results = await bounded_map(batches, ocr_batch,
            execution_width(config, "subtitle", self.providers.get(config["profile_snapshots"]["subtitle"]["id"])) if batches else 1,
            lambda: self.checkpoint(job_id), ocr_done)
        ocr_frames = [frame for result in results if result is not None for frame in result]
        all_ocr_ok = all(result is not None for result in results)
        if batches:
            self._scene_metric(job_id, "subtitle", wall_elapsed=time.monotonic() - ocr_started)
        if subtitle_ready:
            self.update(job_id, message="已恢复完成的字幕轨道，继续后续阶段")
        elif subtitle_times and all_ocr_ok:
            self.checkpoint(job_id)
            cues = merge_ocr_frames(canonical_ocr_frames(ocr_frames, start, end, interval), interval)
            self.library.save_subtitles(asset_id, run_id, cues, start, end)
            save_stage("subtitle", {"signature": subtitle_signature,
                "active": self.library.active(asset_id).get("subtitle_runs"),
                "subtitle_digest": digest(self.library.subtitles(asset_id))})
        elif subtitle_times and not all_ocr_ok:
            self.update(job_id, message="字幕批次有失败，完整轨道未切换；重试可复用已识别批次")
        self.checkpoint(job_id)
        if "vision" in config["stages"]:
            self.character_context(asset_id, config["character_snapshot"])
        if not skip_reindex:
            await self.reindex(job_id, embedding_profile_id=config.get("bindings", {}).get("embedding"))

    async def reindex(self, job_id: str, embedding_profile_id=None):
        job = self.checkpoint(job_id)
        if job["type"] == "index":
            embedding_profile_id = job["config"].get("embedding_profile_id")
        self.update(job_id, stage="index", message="正在更新本地搜索索引")
        records = self.library.records()
        # Publishing committed evidence to lexical search must not depend on a
        # paid embedding service or its remaining request budget.
        await self.search.rebuild(records)
        profile = self.providers.get(embedding_profile_id) if embedding_profile_id else None

        async def embed_batch(texts):
            # The normal durable-call protocol recovers accounting and response
            # data if the process stops before SearchEngine commits its cache.
            key = "embedding:" + digest([model_identity(profile), texts])
            data = await self.call(job_id, profile, self.providers.embed, texts,
                                   text_count=len(texts), receipt_key=key)
            return AIResult(data, {}, 0, 0.0)

        if profile:
            await self.search.rebuild(records, embedding_profile_id=embedding_profile_id,
                                      before_batch=lambda _: self.checkpoint(job_id), embed_batch=embed_batch)
