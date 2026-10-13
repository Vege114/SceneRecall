"""Durable local preparation/imports and grouped, independently budgeted analysis."""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from .library import atomic_json, digest, now, read_json, safe_id, uid
from .models import AssetInput

KEYFRAME_ALGORITHM = "content-keyframes-v1"


class RequestConflict(ValueError):
    pass


class WorkflowQueueMixin:
    def init_workflows(self):
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (kind TEXT, id TEXT, payload_hash TEXT NOT NULL, "
                       "result TEXT NOT NULL, PRIMARY KEY(kind,id))")

    def _existing_request(self, db, kind, request_id, payload):
        row = db.execute("SELECT payload_hash,result FROM requests WHERE kind=? AND id=?",
                         (kind, safe_id(request_id))).fetchone()
        if row and row[0] != digest(payload):
            raise RequestConflict("该提交标识已用于不同内容，请刷新后重新提交")
        return json.loads(row[1]) if row else None

    def _save_request(self, db, kind, request_id, payload, result):
        db.execute("INSERT INTO requests VALUES (?,?,?,?)",
                   (kind, request_id, digest(payload), json.dumps(result)))

    def add_import(self, request):
        payload = request.model_dump(exclude_unset=True)
        ids = [safe_id(item.item_id) for item in request.items]
        if len(set(ids)) != len(ids):
            raise ValueError("批次中的文件标识不能重复")
        with self.library.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = self._existing_request(db, "import", request.request_id, payload)
            if existing:
                return self.get(existing["job_id"])
            items = []
            for item in request.items:
                value = item.model_dump(exclude_unset=True)
                if "project_id" in request.model_fields_set:
                    value["project_id"] = request.project_id
                # Validate the destination before a persistent task is published.
                if value.get("project_id"):
                    self.library.organization.get_project(value["project_id"])
                items.append(value)
            job = self._new_job("import", {"items": items, "request_id": request.request_id})
            job["items"] = [{"item_id": item["item_id"], "name": Path(item["video_path"]).name,
                             "status": "pending"} for item in items]
            job["total"] = len(items)
            db.execute("INSERT INTO jobs VALUES (?,?)", (job["id"], json.dumps(job)))
            self._save_request(db, "import", request.request_id, payload, {"job_id": job["id"]})
        return job

    def retry_import(self, job_id, request):
        with self.library.lock:
            job = self.get(job_id)
            if job["type"] != "import" or job["status"] not in {"failed", "partial", "cancelled", "paused"}:
                raise ValueError("只能修正已停止导入任务中的失败项")
            supplied = {item.item_id: item for item in request.items}
            if len(supplied) != len(request.items):
                raise ValueError("重试文件标识不能重复")
            statuses = {item["item_id"]: item for item in job["items"]}
            if any(key not in statuses or statuses[key]["status"] in {"completed", "existing"} for key in supplied):
                raise ValueError("只能重试本批次未成功的文件")
            config = dict(job["config"])
            configs = {item["item_id"]: item for item in config["items"]}
            committed_keys = {asset.get("import_item_key") for asset in self.library.list_assets()}
            for item_id, item in supplied.items():
                value = item.model_dump(exclude_unset=True)
                if "project_id" not in value and "project_id" in configs[item_id]:
                    value["project_id"] = configs[item_id]["project_id"]
                if digest([job_id, item_id]) in committed_keys:
                    def normalized(raw):
                        asset_input = AssetInput.model_validate({key: val for key, val in raw.items() if key != "item_id"})
                        return asset_input.model_dump(), "project_id" in raw

                    if normalized(value) != normalized(configs[item_id]):
                        raise ValueError("该文件已经导入，不能修改其导入配置；请按原配置重试或新建导入条目")
                if value.get("project_id"):
                    self.library.organization.get_project(value["project_id"])
                configs[item_id] = value
                statuses[item_id] = {"item_id": item_id, "name": Path(item.video_path).name, "status": "pending"}
            config["items"] = [configs[item["item_id"]] for item in config["items"]]
            return self.update(job_id, config=config, items=list(statuses.values()), status="queued",
                               failures=[], error=None, message="等待重试未完成的文件")

    async def import_files(self, job_id):
        job = self.checkpoint(job_id)
        items = {item["item_id"]: dict(item) for item in job["items"]}
        for config in job["config"]["items"]:
            self.checkpoint(job_id)
            item_id = config["item_id"]
            if items[item_id]["status"] in {"completed", "existing"}:
                continue
            item_key = digest([job_id, item_id])
            items[item_id].update(status="running", error=None)
            self.update(job_id, stage="import", items=list(items.values()), message=f"正在导入 {items[item_id]['name']}")
            try:
                existing = next((asset for asset in self.library.list_assets()
                                 if asset.get("import_item_key") == item_key), None)
                if existing is None:
                    payload = AssetInput.model_validate({key: value for key, value in config.items() if key != "item_id"})
                    existing = await self._local_work(job_id, lambda _check: self.library.register(payload, import_item_key=item_key))
                items[item_id].update(status="completed", asset_id=existing["id"])
                self.search.upsert(self.library.records(existing["id"]))
            except Exception as exc:
                from .jobs import StopJob
                if isinstance(exc, StopJob):
                    raise
                if getattr(exc, "asset_id", None):
                    items[item_id].update(status="existing", asset_id=exc.asset_id, error=None)
                else:
                    # Source paths are input-only, not public task error strings.
                    from .jobs import error_message
                    detail = error_message(exc)
                    for key in ("video_path", "subtitle_path"):
                        if config.get(key):
                            detail = detail.replace(config[key], Path(config[key]).name)
                    items[item_id].update(status="failed", error=detail)
            done = sum(item["status"] in {"completed", "existing", "failed"} for item in items.values())
            failures = [{"unit": item["item_id"], "message": item["error"]}
                        for item in items.values() if item["status"] == "failed"]
            self.update(job_id, items=list(items.values()), completed=done, total=len(items),
                        progress=done / len(items), failures=failures)

    async def _local_work(self, job_id, operation):
        """A cancelled coroutine must join its worker and let it reap FFmpeg."""
        stopped = threading.Event()

        def check():
            if stopped.is_set() or self.closed:
                from .jobs import StopJob
                raise StopJob()
            self.checkpoint(job_id)

        task = asyncio.create_task(asyncio.to_thread(operation, check))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            stopped.set()
            try:
                await task
            except Exception:
                pass
            raise

    def _plan_path(self, asset_id, plan_id):
        return self.library.asset_dir(asset_id) / "keyframe-plans" / f"{safe_id(plan_id)}.json"

    def load_plan(self, plan_id):
        safe_id(plan_id)
        paths = list((self.library.root / "assets").glob(f"*/keyframe-plans/{plan_id}.json"))
        if len(paths) != 1:
            raise FileNotFoundError("关键帧计划不存在")
        plan = read_json(paths[0])
        if plan.get("id") != plan_id or plan.get("asset_id") != paths[0].parents[1].name:
            raise ValueError("关键帧计划身份无效")
        return plan

    def ensure_plan(self, asset_id, start_ms=0, end_ms=None):
        with self.library.lock:
            asset = self.library.get_asset(asset_id)
            self.library.require_source(asset_id)
            end = asset["duration_ms"] if end_ms is None else end_ms
            if not 0 <= start_ms < end <= asset["duration_ms"]:
                raise ValueError("关键帧扫描范围超出视频")
            plan_id = "plan_" + digest([asset_id, asset["fingerprint"], start_ms, end, KEYFRAME_ALGORITHM])
            path = self._plan_path(asset_id, plan_id)
            plan = read_json(path)
            if plan and plan.get("status") == "completed":
                try:
                    for frame in plan.get("frames", []):
                        self.library.frame_path(asset_id, frame["id"])
                    return plan
                except FileNotFoundError:
                    # Missing generated previews can be repaired from the source.
                    pass
            job = next((job for job in self.list() if job["type"] == "keyframe_scan"
                        and job["config"].get("plan_id") == plan_id), None)
            if job is None or job["status"] == "completed":
                job = self.add("keyframe_scan", {"plan_id": plan_id, "start_ms": start_ms, "end_ms": end}, asset_id)
            elif job["status"] in {"failed", "partial", "cancelled", "paused"}:
                job = self.control(job["id"], "resume" if job["status"] == "paused" else "retry")
            plan = {"id": plan_id, "plan_id": plan_id, "asset_id": asset_id, "fingerprint": asset["fingerprint"],
                    "start_ms": start_ms, "end_ms": end, "algorithm": KEYFRAME_ALGORITHM,
                    "status": "pending", "job_id": job["id"], "created_at": now()}
            atomic_json(path, plan)
            return {**plan, "status": job["status"]}

    def validated_plan(self, asset_id, start_ms, end_ms, plan_id):
        plan = self.load_plan(plan_id)
        asset = self.library.get_asset(asset_id)
        if (plan.get("asset_id") != asset_id or plan.get("fingerprint") != asset["fingerprint"]
                or plan.get("start_ms") != start_ms or plan.get("end_ms") != end_ms
                or plan.get("algorithm") != KEYFRAME_ALGORITHM):
            raise ValueError("关键帧计划与当前视频或分析范围不一致，请重新扫描")
        if plan.get("status") != "completed":
            raise ValueError("关键帧尚未扫描完成，请等待本地扫描后估算或分析")
        self.library.validate_keyframe_plan(asset_id, plan, expected_id=plan_id)
        return plan

    def plan_view(self, plan_id):
        plan = self.load_plan(plan_id)
        result = {key: value for key, value in plan.items() if key != "fingerprint"}
        if plan.get("status") != "completed" and plan.get("job_id"):
            try:
                job = self.get(plan["job_id"])
                result.update(status=job["status"], progress=job["progress"], error=job.get("error"), message=job["message"])
            except FileNotFoundError:
                result.update(status="pending", progress=0, message="扫描任务未包含在备份中，请重新开始本地扫描")
        result["frames"] = [{**frame, "thumbnail_url": f"/api/assets/{plan['asset_id']}/frames/{frame['id']}"}
                            for frame in plan.get("frames", [])]
        result["frame_count"] = len(result["frames"])
        result["batch_count"] = len(plan.get("batches", []))
        return result

    async def scan_plan(self, job_id, *, asset_id=None, start_ms=None, end_ms=None):
        from .media import scan_keyframes, extract_keyframes
        job = self.checkpoint(job_id)
        asset_id = asset_id or job["asset_id"]
        start_ms = job["config"]["start_ms"] if start_ms is None else start_ms
        end_ms = job["config"]["end_ms"] if end_ms is None else end_ms
        source = self.library.require_source(asset_id)
        asset = self.library.get_asset(asset_id)
        plan_id = "plan_" + digest([asset_id, asset["fingerprint"], start_ms, end_ms, KEYFRAME_ALGORITHM])
        path = self._plan_path(asset_id, plan_id)
        existing = read_json(path)
        if existing and existing.get("status") == "completed":
            return self.validated_plan(asset_id, start_ms, end_ms, plan_id)
        self.update(job_id, stage="keyframes", message="正在本地扫描内容关键帧")

        def progress(done, total):
            self.update(job_id, completed=done, total=total, progress=done / max(1, total),
                        message=f"本地关键帧扫描 {done / 1000:.1f} / {total / 1000:.1f} 秒")

        def work(check):
            data = scan_keyframes(source, start_ms, end_ms, checkpoint=check, progress=progress)
            rename = {}
            for frame in data["frames"]:
                frame_id = "kf_" + digest([asset_id, asset["fingerprint"], frame["pts"], frame["time_base"]])
                rename[frame["id"]] = frame_id
                frame["id"] = frame_id
            for batch in data["batches"]:
                batch["frame_ids"] = [rename[value] for value in batch["frame_ids"]]
            outputs = [self.library.asset_dir(asset_id) / "frames" / f"{frame['id']}.jpg" for frame in data["frames"]]
            self.update(job_id, stage="keyframes", message=f"正在保存 {len(outputs)} 张关键帧预览")
            for offset in range(0, len(outputs), 128):
                check()
                self.library.require_source(asset_id)
                extracted = extract_keyframes(source, data["frames"][offset:offset + 128],
                                              outputs[offset:offset + 128], checkpoint=check)
                for frame in extracted:
                    self.library.register_frame(asset_id, frame)
            check()
            self.library.require_source(asset_id)
            return data

        data = await self._local_work(job_id, work)
        plan = {**data, "id": plan_id, "plan_id": plan_id, "asset_id": asset_id,
                "fingerprint": asset["fingerprint"], "start_ms": start_ms, "end_ms": end_ms,
                "status": "completed", "job_id": job_id, "created_at": now()}
        with self.library.lock:
            self.checkpoint(job_id)
            atomic_json(path, plan)
        return plan

    def estimate_batch(self, items, settings):
        if len({item.asset_id for item in items}) != len(items):
            raise ValueError("同一批分析不能重复选择同一视频")
        estimates = [{"asset_id": item.asset_id, **self.estimate(item, settings)} for item in items]
        known = all(item["cost_estimate"] is not None for item in estimates)
        return {"items": estimates, "estimated_frames": sum(item["estimated_frames"] for item in estimates),
                "estimated_requests": sum(item["estimated_requests"] for item in estimates),
                "cost_estimate": round(sum(item["cost_estimate"] for item in estimates), 6) if known else None,
                "currency": "USD", "warnings": list(dict.fromkeys(w for item in estimates for w in item["warnings"]))}

    def add_batch(self, request, settings):
        payload = request.model_dump()
        if len({item.asset_id for item in request.items}) != len(request.items):
            raise ValueError("同一批分析不能重复选择同一视频")
        with self.library.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self._existing_request(db, "analysis", request.request_id, payload)
            if previous:
                return {"batch_id": previous["batch_id"], "jobs": [self.get(id) for id in previous["job_ids"]]}
            validated = [self.validate(item, settings) for item in request.items]
            validated.sort(key=lambda pair: (pair[0].get("project_id") or "", pair[0].get("season") or 0,
                                             pair[0].get("episode") or 0, pair[0]["title"], pair[0]["id"]))
            batch_id = uid("batch")
            jobs = []
            for asset, config in validated:
                job = self._new_job("analysis", config, asset["id"])
                job["batch_id"] = batch_id
                db.execute("INSERT INTO jobs VALUES (?,?)", (job["id"], json.dumps(job)))
                jobs.append(job)
            self._save_request(db, "analysis", request.request_id, payload,
                               {"batch_id": batch_id, "job_ids": [job["id"] for job in jobs]})
        return {"batch_id": batch_id, "jobs": jobs}

    def control_batch(self, batch_id, action, overrides=None):
        safe_id(batch_id)
        allowed = {"pause": {"queued", "running"}, "cancel": {"queued", "running", "pausing", "paused", "partial", "failed"},
                   "resume": {"paused"}, "retry": {"failed", "partial", "cancelled"}}
        if action not in allowed:
            raise ValueError("不支持的批次操作")
        with self.library.lock:
            jobs = [job for job in reversed(self.list()) if job.get("batch_id") == batch_id]
            if not jobs:
                raise FileNotFoundError("分析批次不存在")
            result, failures = [], []
            for job in jobs:
                if job["status"] not in allowed[action]:
                    result.append(job)
                    continue
                try:
                    result.append(self.control(job["id"], action, overrides))
                except ValueError as exc:
                    result.append(job)
                    failures.append({"job_id": job["id"], "message": str(exc)})
            return {"jobs": result, "failures": failures}
