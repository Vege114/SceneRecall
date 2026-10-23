"""Durable local preparation/imports and grouped, independently budgeted analysis."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from .keyframes import ALGORITHM_PARAMETERS
from .library import atomic_json, digest, now, read_json, safe_id, uid
from .models import AssetInput

KEYFRAME_ALGORITHM = "content-keyframes-v1"
LEGACY_KEYFRAME_DECODER_PROFILE = "ffmpeg-v1"
DEFAULT_KEYFRAME_DECODER_PROFILE = "pyav-libavfilter-v1"
KEYFRAME_DECODER_PROFILES = {"ffmpeg-v1": "ffmpeg", "pyav-libavfilter-v1": "pyav"}


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

    def existing_import(self, request):
        with self.connect() as db:
            result = self._existing_request(db, "import", request.request_id,
                                            request.model_dump(exclude_unset=True))
        return self.get(result["job_id"]) if result else None

    def add_import(self, request, *, anime_dossiers=None):
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
                if (value.get("anime_profile") or {}).get("mode") == "bangumi":
                    dossier = (anime_dossiers or {}).get(item.item_id)
                    if not dossier or dossier["subject"]["id"] != value["anime_profile"]["subject_id"]:
                        raise ValueError("请先读取所选 Bangumi 档案")
                    value["anime_dossier"] = dossier
                items.append(value)
            job = self._new_job("import", {"items": items, "request_id": request.request_id})
            job["items"] = [{"item_id": item["item_id"], "name": Path(item["video_path"]).name,
                             "status": "pending"} for item in items]
            job["total"] = len(items)
            db.execute("INSERT INTO jobs VALUES (?,?)", (job["id"], json.dumps(job)))
            self._save_request(db, "import", request.request_id, payload, {"job_id": job["id"]})
        return job

    def retry_import(self, job_id, request, *, anime_dossiers=None):
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
            committed_keys = {asset.get("import_item_key") for asset in self.library.list_assets(include_deleted=True)}
            for item_id, item in supplied.items():
                value = item.model_dump(exclude_unset=True)
                if "project_id" not in value and "project_id" in configs[item_id]:
                    value["project_id"] = configs[item_id]["project_id"]
                if "anime_profile" not in value and "anime_profile" in configs[item_id]:
                    value["anime_profile"] = configs[item_id]["anime_profile"]
                if configs[item_id].get("anime_profile") and value.get("anime_profile") != configs[item_id]["anime_profile"]:
                    raise ValueError("重试请沿用本批已保存的动画档案；更换作品请重新导入")
                if "anime_dossier" in configs[item_id]:
                    value["anime_dossier"] = configs[item_id]["anime_dossier"]
                elif (value.get("anime_profile") or {}).get("mode") == "bangumi":
                    dossier = (anime_dossiers or {}).get(item_id)
                    if not dossier or dossier["subject"]["id"] != value["anime_profile"]["subject_id"]:
                        raise ValueError("请先读取所选 Bangumi 档案")
                    value["anime_dossier"] = dossier
                if digest([job_id, item_id]) in committed_keys:
                    def normalized(raw):
                        asset_input = AssetInput.model_validate({key: val for key, val in raw.items()
                                                               if key not in {"item_id", "anime_profile", "anime_dossier"}})
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
            items[item_id].update(status="running", error=None, phase="import")
            self.update(job_id, stage="import", items=list(items.values()),
                        message=f"正在校验视频与提取字幕：{items[item_id]['name']}")
            try:
                def register(check):
                    check()
                    existing = next((asset for asset in self.library.list_assets(include_deleted=True)
                                     if asset.get("import_item_key") == item_key), None)
                    if existing and existing.get("deleted"):
                        from .library import DeletedAssetError
                        raise DeletedAssetError()
                    if existing is None:
                        payload = AssetInput.model_validate({key: value for key, value in config.items()
                                                            if key not in {"item_id", "anime_profile", "anime_dossier"}})
                        existing = self.library.register(payload, import_item_key=item_key)
                    return existing

                started = time.perf_counter()
                existing = await self._local_work(job_id, register)
                timings = {"import": round((time.perf_counter() - started) * 1000)}
                index_asset_ids = {existing["id"]}
                items[item_id].update(asset_id=existing["id"],
                                      warnings=existing.get("subtitle_import_warnings", []), timings_ms=timings)
                self.update(job_id, items=list(items.values()))
                self.checkpoint(job_id)
                if config.get("anime_profile"):
                    items[item_id]["phase"] = "import_profile"
                    self.update(job_id, stage="import_profile", items=list(items.values()),
                                message=f"正在同步作品档案：{items[item_id]['name']}")

                    def bind_profile(check):
                        from .anime_profiles import AnimeProfileStore
                        check()
                        old_anime_id = existing.get("anime_id")
                        AnimeProfileStore(self.library).apply_asset(existing["id"], config["anime_profile"],
                                                                   config.get("anime_dossier"), for_import=True)
                        current = self.library.get_asset(existing["id"])
                        # Earlier episodes can acquire a new animation identity or
                        # cast names. Compare their projections, preserving vectors
                        # and skipping unchanged rows instead of rewriting them.
                        anime_ids = {old_anime_id, current.get("anime_id")} - {None}
                        return {asset["id"] for asset in self.library.list_assets()
                                if asset.get("anime_id") in anime_ids
                                or asset.get("character_scope_id") == current.get("character_scope_id")}

                    started = time.perf_counter()
                    index_asset_ids.update(await self._local_work(job_id, bind_profile))
                    timings["import_profile"] = round((time.perf_counter() - started) * 1000)
                    self.checkpoint(job_id)
                items[item_id]["phase"] = "import_index"
                self.update(job_id, stage="import_index", items=list(items.values()),
                            message=f"正在建立搜索索引：{items[item_id]['name']}")

                def publish_index(check):
                    for asset_id in sorted(index_asset_ids):
                        # Serialize the source snapshot and publication with
                        # annotation, subtitle-switch and deletion mutations.
                        with self.library.lock:
                            check()
                            self.search.sync_asset(self.library.records(asset_id), asset_id)

                started = time.perf_counter()
                await self._local_work(job_id, publish_index)
                timings["import_index"] = round((time.perf_counter() - started) * 1000)
                self.checkpoint(job_id)
                items[item_id].update(status="completed", timings_ms=timings)
            except Exception as exc:
                from .jobs import StopJob
                if isinstance(exc, StopJob):
                    raise
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

    @staticmethod
    def _keyframe_decoder_profile(value):
        if not isinstance(value, str) or value not in KEYFRAME_DECODER_PROFILES:
            raise ValueError("不支持的关键帧解码配置")
        return value

    def _plan_decoder_profile(self, plan):
        return self._keyframe_decoder_profile(plan.get("decoder_profile", LEGACY_KEYFRAME_DECODER_PROFILE))

    def _keyframe_plan_id(self, asset, start_ms, end_ms, decoder_profile):
        profile = self._keyframe_decoder_profile(decoder_profile)
        identity = [asset["id"], asset["fingerprint"], start_ms, end_ms, KEYFRAME_ALGORITHM]
        if profile != LEGACY_KEYFRAME_DECODER_PROFILE:
            identity.append(profile)
        return "plan_" + digest(identity)

    def freeze_keyframe_decoder_profile(self, config):
        """New jobs snapshot their decoder; persisted jobs never call this on resume."""
        config = dict(config)
        if config.get("analysis_strategy") == "retrieval-first-v1":
            profile = LEGACY_KEYFRAME_DECODER_PROFILE
        elif config.get("keyframe_plan_id"):
            profile = self._plan_decoder_profile(self.load_plan(config["keyframe_plan_id"]))
            if "keyframe_decoder_profile" in config and config["keyframe_decoder_profile"] != profile:
                raise ValueError("任务的关键帧计划与冻结的解码配置不一致")
        elif "keyframe_decoder_profile" in config:
            profile = config["keyframe_decoder_profile"]
        elif config.get("plan_id"):
            plan = self.load_plan(config["plan_id"])
            profile = self._plan_decoder_profile(plan)
        else:
            profile = DEFAULT_KEYFRAME_DECODER_PROFILE
        config["keyframe_decoder_profile"] = self._keyframe_decoder_profile(profile)
        return config

    def _job_keyframe_decoder_profile(self, config, asset, start_ms, end_ms):
        profile = config.get("keyframe_decoder_profile")
        if profile is not None:
            self._keyframe_decoder_profile(profile)
        supplied_plan = config.get("keyframe_plan_id") or config.get("plan_id")
        if supplied_plan:
            matching = [candidate for candidate in KEYFRAME_DECODER_PROFILES
                        if supplied_plan == self._keyframe_plan_id(asset, start_ms, end_ms, candidate)]
            if not matching or profile is not None and profile != matching[0]:
                raise ValueError("任务的关键帧计划与冻结的解码配置不一致")
            profile = matching[0]
        return profile or LEGACY_KEYFRAME_DECODER_PROFILE

    def _keyframe_selection_cache(self, asset, plan_id, start_ms, end_ms,
                                 decoder_profile=LEGACY_KEYFRAME_DECODER_PROFILE):
        profile = self._keyframe_decoder_profile(decoder_profile)
        path = self.library.asset_dir(asset["id"]) / "keyframe-work" / safe_id(plan_id) / "selection.json"
        identity = digest([asset["id"], asset["fingerprint"], start_ms, end_ms,
                           KEYFRAME_ALGORITHM, ALGORITHM_PARAMETERS, 1, profile])
        return path, identity

    def _save_keyframe_selection(self, asset, plan_id, start_ms, end_ms, data):
        path, identity = self._keyframe_selection_cache(asset, plan_id, start_ms, end_ms,
                                                       self._plan_decoder_profile(data))
        atomic_json(path, {"identity": identity, "selection_digest": digest(data), "selection": data})

    @staticmethod
    def _keyframe_selection_data(plan):
        return {key: plan[key] for key in ("algorithm", "algorithm_parameters", "decoder_profile",
                "source_start_time_seconds", "shots", "frames", "batches") if key in plan}

    def load_plan(self, plan_id):
        safe_id(plan_id)
        paths = list((self.library.root / "assets").glob(f"*/keyframe-plans/{plan_id}.json"))
        if len(paths) != 1:
            raise FileNotFoundError("关键帧计划不存在")
        plan = read_json(paths[0])
        if plan.get("id") != plan_id or plan.get("asset_id") != paths[0].parents[1].name:
            raise ValueError("关键帧计划身份无效")
        return plan

    def ensure_plan(self, asset_id, start_ms=0, end_ms=None, *, decoder_profile=None):
        with self.library.lock:
            asset = self.library.get_asset(asset_id)
            self.library.require_source(asset_id)
            end = asset["duration_ms"] if end_ms is None else end_ms
            if not 0 <= start_ms < end <= asset["duration_ms"]:
                raise ValueError("关键帧扫描范围超出视频")
            if decoder_profile is None:
                # Existing completed plans remain useful when the new-job default changes.
                preferred = dict.fromkeys([LEGACY_KEYFRAME_DECODER_PROFILE, DEFAULT_KEYFRAME_DECODER_PROFILE,
                                          *KEYFRAME_DECODER_PROFILES])
                repair_profile = None
                for candidate in preferred:
                    candidate_id = self._keyframe_plan_id(asset, start_ms, end, candidate)
                    try:
                        return self.validated_plan(asset_id, start_ms, end, candidate_id)
                    except FileNotFoundError:
                        # A decoder-default change must not throw away an older
                        # completed selection merely because its previews are lost.
                        try:
                            damaged = self.load_plan(candidate_id)
                            self._validate_keyframe_plan_header(asset, start_ms, end, candidate_id, damaged)
                            if damaged.get("status") == "completed":
                                self.library.validate_keyframe_selection(asset_id, damaged, candidate_id)
                                repair_profile = repair_profile or candidate
                        except (FileNotFoundError, ValueError):
                            pass
                    except ValueError:
                        continue
                decoder_profile = repair_profile or DEFAULT_KEYFRAME_DECODER_PROFILE
            decoder_profile = self._keyframe_decoder_profile(decoder_profile)
            plan_id = self._keyframe_plan_id(asset, start_ms, end, decoder_profile)
            path = self._plan_path(asset_id, plan_id)
            plan = read_json(path)
            if plan and plan.get("status") == "completed":
                try:
                    return self.validated_plan(asset_id, start_ms, end, plan_id)
                except FileNotFoundError:
                    # Missing generated previews can be repaired from the source.
                    # Preserve pre-upgrade plans before replacing their public state
                    # with pending; those plans may not have a selection cache yet.
                    self.library.validate_keyframe_selection(asset_id, plan, plan_id)
                    if (plan.get("algorithm") != KEYFRAME_ALGORITHM
                            or self._plan_decoder_profile(plan) != decoder_profile):
                        raise ValueError("关键帧计划算法不匹配")
                    self._save_keyframe_selection(asset, plan_id, start_ms, end,
                                                  self._keyframe_selection_data(plan))
            job = next((job for job in self.list() if job["type"] == "keyframe_scan"
                        and job["config"].get("plan_id") == plan_id), None)
            if job is None or job["status"] == "completed":
                job = self.add("keyframe_scan", {"plan_id": plan_id, "start_ms": start_ms, "end_ms": end,
                                                "keyframe_decoder_profile": decoder_profile}, asset_id)
            elif job["status"] in {"failed", "partial", "cancelled", "paused"}:
                job = self.control(job["id"], "resume" if job["status"] == "paused" else "retry")
            plan = {"id": plan_id, "plan_id": plan_id, "asset_id": asset_id, "fingerprint": asset["fingerprint"],
                    "start_ms": start_ms, "end_ms": end, "algorithm": KEYFRAME_ALGORITHM,
                    "decoder_profile": decoder_profile,
                    "status": "pending", "job_id": job["id"], "created_at": now()}
            atomic_json(path, plan)
            return {**plan, "status": job["status"]}

    def _validate_keyframe_plan_header(self, asset, start_ms, end_ms, plan_id, plan):
        profile = self._plan_decoder_profile(plan)
        if (plan.get("asset_id") != asset["id"] or plan.get("fingerprint") != asset["fingerprint"]
                or plan.get("start_ms") != start_ms or plan.get("end_ms") != end_ms
                or plan.get("algorithm") != KEYFRAME_ALGORITHM
                or plan_id != self._keyframe_plan_id(asset, start_ms, end_ms, profile)):
            raise ValueError("关键帧计划与当前视频或分析范围不一致，请重新扫描")

    def validated_plan(self, asset_id, start_ms, end_ms, plan_id):
        plan = self.load_plan(plan_id)
        asset = self.library.get_asset(asset_id)
        self._validate_keyframe_plan_header(asset, start_ms, end_ms, plan_id, plan)
        if plan.get("status") != "completed":
            raise ValueError("关键帧尚未扫描完成，请等待本地扫描后估算或分析")
        self.library.validate_keyframe_plan(asset_id, plan, expected_id=plan_id)
        return plan

    def plan_view(self, plan_id):
        plan = self.load_plan(plan_id)
        self.library.get_asset(plan["asset_id"])
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
        from .media import scan_keyframes, extract_keyframes, probe
        job = self.checkpoint(job_id)
        asset_id = asset_id or job["asset_id"]
        start_ms = job["config"]["start_ms"] if start_ms is None else start_ms
        end_ms = job["config"]["end_ms"] if end_ms is None else end_ms
        source = self.library.require_source(asset_id)
        asset = self.library.get_asset(asset_id)
        decoder_profile = self._job_keyframe_decoder_profile(job["config"], asset, start_ms, end_ms)
        plan_id = self._keyframe_plan_id(asset, start_ms, end_ms, decoder_profile)
        path = self._plan_path(asset_id, plan_id)
        existing = read_json(path)
        if existing and existing.get("status") == "completed":
            try:
                return self.validated_plan(asset_id, start_ms, end_ms, plan_id)
            except FileNotFoundError:
                # A valid selection survives lost previews. Repair only the missing pictures.
                self.library.validate_keyframe_selection(asset_id, existing, plan_id)
        directory = self.library.asset_dir(asset_id)
        cache_path, cache_identity = self._keyframe_selection_cache(asset, plan_id, start_ms, end_ms, decoder_profile)
        self.update(job_id, stage="keyframes", message="正在本地扫描内容关键帧", progress=0)

        def progress(done, total):
            self.update(job_id, completed=done, total=total, progress=.75 * done / max(1, total),
                        message=f"本地关键帧扫描 {done / 1000:.1f} / {total / 1000:.1f} 秒")

        def selection_plan(data):
            return {**data, "id": plan_id, "plan_id": plan_id, "asset_id": asset_id,
                    "fingerprint": asset["fingerprint"], "start_ms": start_ms, "end_ms": end_ms,
                    "decoder_profile": decoder_profile,
                    "status": "completed", "job_id": job_id, "created_at": now()}

        def validate_selection(data):
            if (data.get("algorithm") != KEYFRAME_ALGORITHM
                    or self._plan_decoder_profile(data) != decoder_profile):
                raise ValueError("关键帧选择缓存算法不匹配")
            self.library.validate_keyframe_selection(asset_id, selection_plan(data), plan_id)
            for frame in data["frames"]:
                expected = "kf_" + digest([asset_id, asset["fingerprint"], frame["pts"], frame["time_base"]])
                if frame["id"] != expected:
                    raise ValueError("关键帧选择缓存身份不匹配")

        def work(check):
            attempt_started = time.perf_counter()
            metrics = {"selection_ms": 0, "extraction_ms": 0, "registration_ms": 0,
                       "validation_ms": 0, "elapsed_ms": 0,
                       "selected_frames": 0, "reused_frames": 0, "extracted_frames": 0,
                       "selection_cache_hit": False, "decoder_profile": decoder_profile}

            def report_metrics():
                metrics["elapsed_ms"] = round((time.perf_counter() - attempt_started) * 1000)
                self.update(job_id, keyframe_metrics=dict(metrics))

            try:
                check()
                metadata = probe(source)
                data = None
                if existing and existing.get("status") == "completed":
                    data = self._keyframe_selection_data(existing)
                    validate_selection(data)
                else:
                    try:
                        cached = read_json(cache_path)
                        if cached and cached.get("identity") == cache_identity:
                            candidate = cached["selection"]
                            if cached.get("selection_digest") == digest(candidate):
                                validate_selection(candidate)
                                data = candidate
                    except (ValueError, KeyError, TypeError, AttributeError):
                        # Disposable metadata never bypasses the source/PTS validation contract.
                        data = None
                metrics["selection_cache_hit"] = data is not None
                if data is None:
                    started = time.perf_counter()
                    try:
                        data = scan_keyframes(source, start_ms, end_ms, checkpoint=check, progress=progress,
                                              metadata=metadata, decoder=KEYFRAME_DECODER_PROFILES[decoder_profile])
                    finally:
                        metrics["selection_ms"] = round((time.perf_counter() - started) * 1000)
                    rename = {}
                    data["decoder_profile"] = decoder_profile
                    for frame in data["frames"]:
                        frame_id = "kf_" + digest([asset_id, asset["fingerprint"], frame["pts"], frame["time_base"]])
                        rename[frame["id"]] = frame_id
                        frame["id"] = frame_id
                    for batch in data["batches"]:
                        batch["frame_ids"] = [rename[value] for value in batch["frame_ids"]]
                    validate_selection(data)
                    self._save_keyframe_selection(asset, plan_id, start_ms, end_ms, data)
                frames = data["frames"]
                metrics["selected_frames"] = len(frames)
                registry = read_json(directory / "frames" / "registry.json", {})
                missing = []
                for frame in frames:
                    output = directory / "frames" / f"{frame['id']}.jpg"
                    entry = registry.get(frame["id"])
                    if (entry != {"path": output.relative_to(directory).as_posix(), "at_ms": frame["at_ms"]}
                            or not output.is_file()):
                        missing.append((frame, output))
                metrics["reused_frames"] = len(frames) - len(missing)

                def extraction_progress():
                    done = metrics["reused_frames"] + metrics["extracted_frames"]
                    self.update(job_id, completed=done, total=len(frames),
                                progress=.75 + .24 * done / max(1, len(frames)),
                                message=f"正在保存关键帧预览 {done} / {len(frames)} 张")
                    report_metrics()

                extraction_progress()
                for offset in range(0, len(missing), 128):
                    check()
                    self.library.require_source(asset_id)
                    current = missing[offset:offset + 128]
                    started = time.perf_counter()
                    try:
                        extracted = extract_keyframes(source, [item[0] for item in current],
                                                      [item[1] for item in current], checkpoint=check,
                                                      metadata=metadata)
                    finally:
                        metrics["extraction_ms"] += round((time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    try:
                        self.library.register_frames(asset_id, extracted)
                    finally:
                        metrics["registration_ms"] += round((time.perf_counter() - started) * 1000)
                    metrics["extracted_frames"] += len(extracted)
                    extraction_progress()
                check()
                self.library.require_source(asset_id)
                self.update(job_id, progress=.99, message="正在验证并发布关键帧计划")
                plan = selection_plan(data)
                with self.library.lock:
                    self.checkpoint(job_id)
                    started = time.perf_counter()
                    try:
                        self.library.validate_keyframe_plan(asset_id, plan, expected_id=plan_id)
                        atomic_json(path, plan)
                    finally:
                        metrics["validation_ms"] = round((time.perf_counter() - started) * 1000)
                self.update(job_id, completed=len(frames), total=len(frames), progress=1,
                            message=f"关键帧计划已保存，共 {len(frames)} 张")
                return plan
            finally:
                report_metrics()

        async with self._scan_slots:
            self.checkpoint(job_id)
            return await self._local_work(job_id, work)

    def estimate_batch(self, items, settings):
        from .timing import batch_timing
        if len({item.asset_id for item in items}) != len(items):
            raise ValueError("同一批分析不能重复选择同一视频")
        estimates = [{"asset_id": item.asset_id, **self.estimate(item, settings)} for item in items]
        known = all(item["cost_estimate"] is not None for item in estimates)
        return {"items": estimates, "estimated_frames": sum(item["estimated_frames"] for item in estimates),
                "estimated_requests": sum(item["estimated_requests"] for item in estimates),
                "estimated_requests_min": sum(item.get("estimated_requests_min", item["estimated_requests"]) for item in estimates),
                "estimated_requests_max": sum(item.get("estimated_requests_max", item["estimated_requests"]) for item in estimates),
                **batch_timing(estimates),
                "cost_estimate_min": (sum(item.get("cost_estimate_min", item["cost_estimate"]) for item in estimates)
                                      if known else None),
                "cost_estimate_max": (sum(item.get("cost_estimate_max", item["cost_estimate"]) for item in estimates)
                                      if known else None),
                "cost_estimate": round(sum(item["cost_estimate"] for item in estimates), 6) if known else None,
                "currency": "USD", "warnings": list(dict.fromkeys(w for item in estimates for w in item["warnings"]))}

    def add_batch(self, request, settings):
        payload = request.model_dump()
        for item, request_item in zip(payload["items"], request.items, strict=True):
            if item.get("analysis_strategy") != "retrieval-first-v1":
                item.pop("retrieval_mode", None)
            # The default serial schedule adds no new behavior to older requests.
            # Keep their durable idempotency fingerprints valid after upgrading.
            from .models import ExecutionOptions
            if item.get("execution") == ExecutionOptions().model_dump():
                item.pop("execution", None)
            # Empty selections keep pre-selection request fingerprints reusable.
            if not item.get("model_overrides"):
                item.pop("model_overrides", None)
            else:
                item["model_overrides"] = {stage: selection.model_dump(exclude_unset=True)
                                           for stage, selection in request_item.model_overrides.items()}
            if not item.get("model_bindings"):
                item.pop("model_bindings", None)
            if item.get("task_pipeline") is None:
                item.pop("task_pipeline", None)
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
        allowed = {"pause": {"queued", "running", "waiting_input"}, "cancel": {"queued", "running", "waiting_input", "pausing", "paused", "partial", "failed"},
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
