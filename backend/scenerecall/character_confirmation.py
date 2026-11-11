"""Work-scoped, human-reviewed original-frame exemplars before large runs.

Reading the queue never invokes a model. Optional sampling is bounded, local,
resumable and candidate-only; a confirmation affects exactly one region.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import stat
import threading
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .characters import CharacterStore, checked_box, scope_id
from .crop_artifacts import rebuild_crop_bytes
from .library import atomic_json, confined_path, digest, now, read_json, safe_id
from .models import Contract
from .visual_identity import PREPROCESS_VERSION, VisualIdentityPipeline, _image, image_quality


class ConfirmationDecision(Contract):
    action: Literal["confirm", "reject", "unclear", "skip", "resume"]
    source_asset_id: str | None = None
    region_id: str | None = None
    single_character: bool = False
    use_as_reference: bool = False
    reason: str = Field(default="", max_length=4000)


class ConfirmationSample(Contract):
    scope: Literal["asset", "work"] = "asset"
    max_frames: int = Field(default=12, ge=1, le=120, strict=True)
    max_assets: int = Field(default=24, ge=1, le=24, strict=True)
    frames_per_asset: int = Field(default=12, ge=1, le=24, strict=True)

    @model_validator(mode="after")
    def bounded_asset_request(self):
        if self.scope == "asset" and self.max_frames > 24:
            raise ValueError("单集候选采样预算为 1–24 帧")
        return self


class RecognitionSelection(Contract):
    character_ids: list[str]
    expected_revision: int = Field(ge=0, strict=True)
    scope_id: str


class ConfirmationCandidate(Contract):
    source_asset_id: str = Field(min_length=1, max_length=180)
    region_id: str = Field(min_length=1, max_length=180)

    @field_validator("source_asset_id", "region_id")
    @classmethod
    def checked_identifier(cls, value):
        return safe_id(value)


class ConfirmationBatch(Contract):
    candidates: list[ConfirmationCandidate] = Field(min_length=1, max_length=3)
    single_character: bool = Field(strict=True)
    use_as_reference: bool = Field(strict=True)
    reason: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def explicit_unique_selection(self):
        if not self.single_character or not self.use_as_reference:
            raise ValueError("请确认所选裁剪均为该角色的清晰单人画面，并同意作为后续识别参考")
        keys = [(row.source_asset_id, row.region_id) for row in self.candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("所选人物裁剪不能重复")
        return self


class CharacterConfirmation:
    def __init__(self, library, pipeline=None, search=None):
        self.library = library
        self.pipeline = pipeline or VisualIdentityPipeline(library)
        self.gallery = self.pipeline.gallery
        self.search = search
        self._guard = threading.RLock()
        self._running = {}
        self._cancellations = {}

    def _root(self, scope):
        return self.library.root / "works" / safe_id(scope) / "confirmation"

    def _read(self, scope):
        return read_json(self._root(scope) / "state.json", {
            "schema_version": "1.0", "scope_id": scope, "revision": 0, "decisions": {}, "audit": []})

    @staticmethod
    def _key(character_id, asset_id=None, region_id=None):
        return digest(["candidate", character_id, asset_id, region_id] if region_id
                      else ["role", character_id])

    def _save(self, scope, state, key, decision):
        previous = state["decisions"].get(key)
        state["decisions"][key] = copy.deepcopy(decision)
        state["revision"] += 1
        state["audit"].append({"key": key, "before": previous, "after": copy.deepcopy(decision), "at": now()})
        atomic_json(self._root(scope) / "state.json", state)

    def sampling(self, asset_id):
        scope = scope_id(self.library.get_asset(asset_id))
        with self._guard:
            path = self._root(scope) / "sampling.json"
            value = read_json(path)
            thread = self._running.get(scope)
            if value and value["status"] in {"running", "cancelling"} and not (thread and thread.is_alive()):
                value.update(status="interrupted", error="上次少量采样已中断，可继续并复用已完成帧", updated_at=now())
                atomic_json(path, value)
            return value

    def _shots(self, asset):
        """Use stored cuts only; never rescan a video merely to show the queue."""
        frames, intervals = {}, []
        root = self.library.asset_dir(asset["id"])
        plans = [*(root / "keyframe-plans").glob("*.json"), *(root / "retrieval" / "plans").glob("*.json")]
        for path in sorted(plans):
            plan = read_json(path, {})
            if plan.get("fingerprint", plan.get("source_fingerprint")) != asset["fingerprint"]:
                continue
            for row in plan.get("frames", []):
                if row.get("shot_id"):
                    frames[row["id"]] = row["shot_id"]
            intervals.extend((row["start_ms"], row["end_ms"], row["id"])
                             for row in plan.get("shots", []))
        return frames, sorted(set(intervals))

    def _role_allowed(self, catalog, asset, character_id):
        registry = CharacterStore(self.library)._read(catalog["scope_id"])
        profile = registry["profiles"].get(character_id)
        policy = CharacterStore.cast_policy(registry, asset.get("anime_profile"))
        return bool(profile and CharacterStore.profile_allowed(profile, policy))

    def _crop_inventory(self, asset_id):
        """One directory read, retaining full confinement checks for links.

        A normal file under a confined directory plus a checked filename has
        no traversable components. Symlinks and Windows reparse points retain
        the complete resolver, as do the few candidates actually displayed.
        """
        directory = self.pipeline._directory(asset_id)
        crops = confined_path(directory, "crops")
        try:
            with os.scandir(crops) as stream:
                entries = {entry.name: entry for entry in stream}
        except FileNotFoundError:
            entries = {}

        def exists(region_id):
            name = f"{safe_id(region_id)}.png"
            entry = entries.get(name)
            if entry is None:
                return False
            try:
                metadata = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                    return confined_path(directory, f"crops/{name}").is_file()
                return stat.S_ISREG(metadata.st_mode)
            except FileNotFoundError:
                return False
        return exists

    def _reference_diagnostics(self, asset_id, references, active_references):
        """Project persisted detector evidence without asking a backend to load.

        A missing/stale cache is untested, never a detection failure. Legacy
        receipts use their exact model version in the cache key; the real CCIP
        version string also identifies its configured model. New receipts name
        that input explicitly, including failed runs.
        """
        model = self.pipeline.settings(asset_id)["model"]
        try:
            latest = read_json(self.pipeline._directory(asset_id) / "latest.json", {})
        except (OSError, ValueError):
            latest = {}
        if not isinstance(latest, dict):
            latest = {}
        version = latest.get("model_version")
        same_model = (latest.get("model") == model or not latest.get("model")
                      and isinstance(version, str) and version.startswith("imgutils-") and f":{model}:" in version)
        version = version if same_model and isinstance(version, str) else None
        unavailable = same_model and latest.get("status") == "unavailable"
        grouped = {}
        for reference in references:
            if reference["id"] not in active_references:
                continue
            status, person_count = "not_checked", None
            if version:
                key = digest([reference["sha256"], version, PREPROCESS_VERSION])
                try:
                    cached = read_json(self.library.root / "cache" / "identity-reference-detections" / f"{key}.json")
                except (OSError, ValueError):
                    cached = None
                provenance = {"reference_sha256": reference["sha256"], "model": model,
                              "model_version": version, "preprocessing": PREPROCESS_VERSION}
                if (isinstance(cached, dict) and isinstance(cached.get("detections"), list)
                        and all(key not in cached or cached[key] == expected for key, expected in provenance.items())):
                    person_count = len(cached["detections"])
                    status = "usable" if person_count == 1 else "no_person" if person_count == 0 else "multiple_people"
            if status == "not_checked" and unavailable:
                status = "model_unavailable"
            reasons = {
                "not_checked": "该参考图尚无当前模型的有效检测记录；尚不能判断是否可用于自动匹配。",
                "no_person": "该参考图未检出人物，未参与本次自动匹配。",
                "multiple_people": f"该参考图检出 {person_count} 个人物，未参与单人匹配。",
                "model_unavailable": "上次检测记录显示本地视觉模型不可用，本参考尚未完成检测。"}
            actions = {
                "not_checked": "生成候选以检测已有参考；也可在参考图管理中替换清晰的单人图。",
                "no_person": "补充或替换清晰的单人参考图后重新生成；仍可对待判定裁剪人工选角核对。",
                "multiple_people": "在参考图管理中裁剪或替换为单人参考图后重新生成。",
                "model_unavailable": "检查本地模型环境后重试生成；无需撤销参考图或人工标注。"}
            summary = grouped.setdefault(reference["character_id"], {"usable_matching_reference_count": 0,
                "reference_issues": [], "reference_model": model, "reference_model_version": version})
            if status == "usable":
                summary["usable_matching_reference_count"] += 1
            else:
                summary["reference_issues"].append({"reference_id": reference["id"], "status": status,
                    "person_count": person_count, "reason": reasons[status], "action": actions[status], "model_version": version})
        for summary in grouped.values():
            issues = {row["status"] for row in summary["reference_issues"]}
            summary["reference_detection_status"] = ("usable" if summary["usable_matching_reference_count"] else
                "not_checked" if "not_checked" in issues else "model_unavailable" if "model_unavailable" in issues else "failed")
        return grouped, model, version

    @staticmethod
    def _candidate_assignment(region, proposals, config):
        """Choose only the closest identity; uncertainty never promotes Top-2.

        This is a candidate presentation, not an identity review or enrollment.
        Stored Top-K remains attached verbatim for audit.
        """
        match = region.get("match", {})
        if match.get("source") == "human" and match.get("candidate_character_id"):
            return match["candidate_character_id"], "human"
        if not proposals:
            return None, "no_reference"
        best = proposals[0]
        calibration = config.get("calibration")
        calibrated = (isinstance(calibration, dict)
                      and calibration.get("model_version") == match.get("model_version")
                      and calibration.get("aggregation") == "nearest-two-mean-v1"
                      and calibration.get("dataset_id") and calibration.get("evaluation_id")
                      and match.get("source") == "ccip_calibrated"
                      and match.get("calibration_id") == calibration.get("evaluation_id"))
        if not calibrated:
            return None, "uncalibrated"
        maximum, minimum = calibration.get("max_distance"), calibration.get("min_margin")
        if any(type(value) not in {int, float} or not math.isfinite(value) or value < 0
               for value in (maximum, minimum)):
            return None, "uncalibrated"
        if best["distance"] > maximum:
            return None, "low_confidence"
        # A missing runner-up cannot establish a separation margin.
        if len(proposals) < 2 or proposals[1]["distance"] <= best["distance"] or proposals[1]["distance"] - best["distance"] < minimum:
            return None, "ambiguous"
        # The matcher additionally checks trusted reference provenance.
        if match.get("status") != "confirmed" or match.get("character_id") != best["character_id"]:
            return None, "low_confidence"
        return best["character_id"], "calibrated"

    def queue(self, asset_id, *, review_pool_cursor=None, review_pool_limit=20):
        with self.library.lock:
            catalog = CharacterStore(self.library).roster(asset_id)
            scope = catalog["scope_id"]
            registry = CharacterStore(self.library)._read(scope)
            state = self._read(scope)
            profiles = {row["id"]: row for row in catalog["profiles"]}
            enabled = {key for key, row in profiles.items() if row.get("recognition_enabled", True)}
            decisions = {}
            for row in state["decisions"].values():
                identity = CharacterStore._resolve(registry, row["character_id"])
                if identity in profiles:
                    decisions[self._key(identity, row.get("source_asset_id"), row.get("region_id"))] = row
            references = self.gallery.list(asset_id, _catalog=catalog)["references"]
            trusted = [row for row in references if row["status"] == "confirmed" and row["quality"]["accepted"]]
            active_references = self.gallery.active_reference_ids(asset_id, include_candidates=True, _catalog=catalog)
            reference_diagnostics, reference_model, reference_model_version = self._reference_diagnostics(
                asset_id, references, active_references)
            reference_counts = {}
            for reference in references:
                if reference["id"] in active_references:
                    identity = reference["character_id"]
                    reference_counts[identity] = reference_counts.get(identity, 0) + 1
            pending_regions = {}
            for decision in decisions.values():
                if decision["action"] == "confirming":
                    identity = CharacterStore._resolve(registry, decision["character_id"])
                    pending_regions.setdefault((decision["source_asset_id"], decision["region_id"]), set()).add(identity)
            grouped, review_pool = {identity: [] for identity in profiles}, []
            for asset in catalog["assets"] if enabled else []:
                if asset.get("source_changed"):
                    continue
                directory = self.pipeline._directory(asset["id"])
                raw = [read_json(path) for path in sorted((directory / "regions").glob("*.json"))]
                if not raw:
                    continue
                frame_registry = read_json(self.library.asset_dir(asset["id"]) / "frames" / "registry.json", {})
                raw = [row for row in raw if row.get("quality", {}).get("accepted")
                       and row.get("source_fingerprint", asset["fingerprint"]) == asset["fingerprint"]
                       and row["at_ms"] == frame_registry.get(row["frame_id"], {}).get("at_ms")]
                raw_matches = {row["id"]: copy.deepcopy(row["match"].get("candidates", [])) for row in raw}
                regions = self.pipeline._project(asset["id"], raw, _catalog=catalog, _crop_paths=False)
                crop_exists = self._crop_inventory(asset["id"])
                frame_shots, shots = self._shots(asset)
                config = self.pipeline.settings(asset["id"])
                policy = CharacterStore.cast_policy(registry, asset.get("anime_profile"))
                allowed = {key for key in enabled if CharacterStore.profile_allowed(profiles[key], policy)}
                seen_pixels = set()
                def pixel_key(frame_id, box):
                    return (frame_registry.get(frame_id, {}).get("alias_of", frame_id), tuple(box or []))
                locked_pixels = {pixel_key(row["frame_id"], row["box"]) for row in regions
                                 if row["match"].get("source") == "human"
                                 and row["match"].get("status") in {"confirmed", "unknown"}}
                excluded_pixels = {}
                for decision in decisions.values():
                    if decision.get("source_asset_id") == asset["id"] and decision["action"] in {"confirm", "reject", "unclear"}:
                        excluded_pixels.setdefault(pixel_key(decision.get("frame_id"), decision.get("box")), set()).add(
                            CharacterStore._resolve(registry, decision["character_id"]))
                # Recovery and human decisions have precedence over an alias of the same pixels.
                regions.sort(key=lambda row: (not bool(pending_regions.get((asset["id"], row["id"]))),
                                               row["match"].get("source") != "human", row["id"]))
                for region in regions:
                    pending = pending_regions.get((asset["id"], region["id"]), set()) & allowed
                    match = region["match"]
                    if (not pending and match.get("source") == "human" and match.get("status") in {"confirmed", "unknown"}
                            or not crop_exists(region["id"]) and not self.pipeline.crop_available(
                                asset["id"], region, registry=frame_registry, asset=asset)):
                        continue
                    pixels = pixel_key(region["frame_id"], region["box"])
                    if pixels in locked_pixels and not pending:
                        continue
                    eligible = sorted(key for key in allowed if key not in excluded_pixels.get(pixels, set()) and decisions.get(
                        self._key(key, asset["id"], region["id"]), {}).get("action") not in {"confirm", "reject", "unclear"})
                    if not eligible:
                        continue
                    if pixels in seen_pixels:
                        continue
                    seen_pixels.add(pixels)
                    shot = frame_shots.get(region["frame_id"])
                    if not shot:
                        shot = next((key for start, end, key in shots if start <= region["at_ms"] < end), None)
                    candidates = match.get("candidates", [])
                    proposals = [{**row, "candidate_rank": rank, "candidate_total": len(candidates)}
                                 for rank, row in enumerate(candidates, 1)
                                 if row.get("reference_ids") and set(row["reference_ids"]).issubset(active_references)
                                 and type(row.get("distance")) in {int, float} and math.isfinite(row["distance"])
                                 and row["distance"] >= 0]
                    proposals.sort(key=lambda row: (row["distance"], row["character_id"]))
                    owner, confidence = self._candidate_assignment(region, proposals, config)
                    if pending:
                        owner, confidence = (next(iter(pending)), "human") if len(pending) == 1 else (None, "ambiguous")
                    # Never promote the next closest role when the winner is excluded or rejected.
                    if owner not in eligible:
                        owner = None
                    best = proposals[0] if proposals else {}
                    displayed_proposal = next((row for row in proposals if row["character_id"] == owner), {}) if owner else best
                    candidate = {
                        "region_id": region["id"], "asset_id": asset["id"], "asset_title": asset["title"],
                        "source_available": asset.get("source_available", False), "frame_id": region["frame_id"],
                        "recovery_required": bool(pending), "at_ms": region["at_ms"], "box": region["box"],
                        "crop_url": region["crop_url"], "shot_id": shot, "distance": displayed_proposal.get("distance"),
                        "quality": region["quality"], "candidate_rank": displayed_proposal.get("candidate_rank"),
                        "candidate_total": len(candidates), "raw_candidates": raw_matches[region["id"]],
                        "assignment_state": "assigned" if owner else "pending", "confidence_state": confidence,
                        "assigned_character_id": owner, "suggested_character_id": best.get("character_id"),
                        "eligible_character_ids": eligible}
                    if owner:
                        grouped[owner].append(candidate)
                    else:
                        review_pool.append(candidate)
            roles = []
            for identity, profile in profiles.items():
                choices = sorted(grouped[identity], key=lambda row: (
                    not row["recovery_required"], row["distance"] if row["distance"] is not None else float("inf"),
                    -row["quality"].get("sharpness", 0), row["asset_id"], row["at_ms"], row["region_id"]))
                selected = []
                for row in choices:
                    if any(row["asset_id"] == other["asset_id"] and (
                            row["shot_id"] and row["shot_id"] == other["shot_id"]
                            or abs(row["at_ms"] - other["at_ms"]) < 5000) for other in selected):
                        continue
                    selected.append(row)
                    if len(selected) == 3:
                        break
                count = sum(row["character_id"] == identity and row["source"]["kind"] == "frame_crop" for row in trusted)
                role_decision = decisions.get(self._key(identity), {})
                recovering = any(row["recovery_required"] for row in choices)
                status = "pending" if recovering else "confirmed" if count else "skipped" if role_decision.get("action") == "skip" else "pending"
                roles.append({"character_id": identity, "profile": profile, "status": status,
                              "recognition_enabled": identity in enabled,
                              "matching_reference_count": reference_counts.get(identity, 0),
                              **reference_diagnostics.get(identity, {"usable_matching_reference_count": 0,
                                  "reference_issues": [], "reference_detection_status": "missing",
                                  "reference_model": reference_model, "reference_model_version": reference_model_version}),
                              "trusted_reference_count": count, "candidates": selected, "available_count": len(choices),
                              "reference_preview_url": next((row["image_url"] for row in references
                                  if row["id"] in active_references and row["character_id"] == identity), None)})
            review_pool.sort(key=lambda row: (row["asset_id"], row["at_ms"], row["region_id"]))
            if type(review_pool_limit) is not int or not 1 <= review_pool_limit <= 120:
                raise ValueError("待判定页大小应为 1–120")
            # Cursor represents an offset only; membership changes never grant
            # access to another scope, nor do they authorize a confirmation.
            cursors = {digest([scope, "review-pool", offset]): offset for offset in range(len(review_pool) + 1)}
            if review_pool_cursor is not None and review_pool_cursor not in cursors:
                raise ValueError("待判定分页已变化，请返回首页刷新")
            pool_offset = cursors.get(review_pool_cursor, 0)
            displayed = review_pool[pool_offset:pool_offset + review_pool_limit]
            next_offset = pool_offset + len(displayed)
            next_cursor = digest([scope, "review-pool", next_offset]) if next_offset < len(review_pool) else None
            previous_cursor = digest([scope, "review-pool", max(0, pool_offset - review_pool_limit)]) if pool_offset else None
            for candidate in [*displayed, *(candidate for role in roles for candidate in role["candidates"])]:
                confined_path(self.pipeline._directory(candidate["asset_id"]), f"crops/{safe_id(candidate['region_id'])}.png")
            online = any(row.get("source_available") and not row.get("source_changed") for row in catalog["assets"])
        # Keep worker-guard and library-lock acquisition independent.
        sampling = self.sampling(asset_id)
        current_run = sampling if sampling and sampling.get("selected_character_ids") == sorted(enabled) else {}
        for role in roles:
            pool_count = sum(role["character_id"] in row["eligible_character_ids"] for row in review_pool)
            frames = current_run.get("completed", 0)
            coverage = ("disabled" if not role["recognition_enabled"] else
                        "confirmed" if role["status"] == "confirmed" else
                        "skipped" if role["status"] == "skipped" else
                        "missing_reference" if not role["matching_reference_count"] else
                        "reference_invalid" if role["reference_detection_status"] == "failed" else
                        "model_unavailable" if role["reference_detection_status"] == "model_unavailable" else
                        "reference_unchecked" if role["reference_detection_status"] == "not_checked" else
                        "ready" if role["available_count"] else
                        "pending_review" if pool_count else
                        "detection_invalid" if frames and not current_run.get("accepted_regions", 0) else
                        "budget_exhausted" if current_run.get("status") in {"completed", "partial"} else "needs_sampling")
            reasons = {"disabled": "未加入识别名单", "confirmed": "已有人工确认的原片参考",
                       "skipped": "本轮已跳过", "missing_reference": "缺少可用参考图；仍可在待判定池人工选角",
                       "reference_invalid": "已有参考图未通过单人检测，请补充或替换清晰单人参考；待判定池仍可人工选角核对。",
                       "reference_unchecked": "参考图尚未完成当前模型检测，请生成候选以检测；待判定数量不表示角色已被找到。",
                       "model_unavailable": "本地视觉模型不可用，参考检测尚未完成；请检查模型环境后重试生成。",
                       "ready": "已有独占候选，等待人工核对", "pending_review": "候选身份未确定，请在待判定池选角核对；该数量不表示角色已被找到。",
                       "detection_invalid": "本次已采样帧没有有效人物裁剪；不能据此判断角色不存在",
                       "budget_exhausted": "本次有界预算已用完，尚未得到可靠候选，可继续扩采",
                       "needs_sampling": "尚需跨集补充采样，不能保证角色出现在预算范围内"}
            role["coverage"] = {"status": coverage, "reason": reasons[coverage], "candidate_count": role["available_count"],
                                "review_pool_count": pool_count, "sampled_frames": frames,
                                "sampled_assets": sum(row.get("completed", 0) > 0 for row in current_run.get("assets", []))}
        active = [row for row in roles if row["recognition_enabled"]]
        reason = "" if online else "原片均离线或已变化，请重新定位原片；已有缓存仍可核对"
        if not enabled:
            reason = "请先选择需要识别的角色"
        return {"asset_id": asset_id, "scope_id": scope, "roles": roles, "revision": state["revision"],
                "roster_revision": catalog["revision"], "review_pool": displayed, "review_pool_count": len(review_pool),
                "review_pool_next_cursor": next_cursor, "review_pool_previous_cursor": previous_cursor,
                "review_pool_offset": pool_offset, "review_pool_limit": review_pool_limit,
                "summary": {"total": len(active), **{key: sum(row["status"] == key for row in active)
                            for key in ("confirmed", "skipped", "pending")},
                            "candidates": sum(row["available_count"] for row in active), "review_pool": len(review_pool)},
                "sampling": sampling, "generation": sampling, "sampling_available": not reason,
                "sampling_unavailable_reason": reason,
                "guidance": "仅确认当前裁剪并经明确授权入库。距离越小越相似；无校准或身份不确定的裁剪仅进入待判定池，不能证明角色出现或不存在。"}

    def decide(self, asset_id, character_id, *, action, source_asset_id=None, region_id=None,
               single_character=False, use_as_reference=False, reason="", _catalog=None, _defer_projection=False,
               _before_confirmation_write=None):
        if action not in {"confirm", "reject", "unclear", "skip", "resume"}:
            raise ValueError("角色确认操作无效")
        with self.library.lock:
            catalog = _catalog if _catalog is not None else CharacterStore(self.library).roster(asset_id)
            scope = catalog["scope_id"]
            registry = CharacterStore(self.library)._read(scope)
            character_id = CharacterStore._resolve(registry, character_id, strict=True)
            if character_id not in {row["id"] for row in catalog["profiles"]}:
                raise ValueError("角色不在当前作品名单中")
            if not registry["profiles"][character_id].get("recognition_enabled", True):
                raise ValueError("请先将该角色加入识别名单")
            state = self._read(scope)
            if action in {"skip", "resume"}:
                self._save(scope, state, self._key(character_id), {
                    "action": action, "character_id": character_id, "updated_at": now()})
                return
            if not source_asset_id or not region_id:
                raise ValueError("请选择要确认的人物裁剪")
            source = self.library.get_asset(source_asset_id)
            if scope_id(source) != scope or not self._role_allowed(catalog, source, character_id):
                raise ValueError("人物裁剪与角色不属于同一作品名单")
            if source.get("source_changed"):
                raise ValueError("原片已变化，请先重新定位并校验原片")
            raw = read_json(self.pipeline._directory(source_asset_id) / "regions" / f"{safe_id(region_id)}.json")
            if raw is None:
                raise FileNotFoundError("人物裁剪不存在")
            if raw.get("source_fingerprint", source["fingerprint"]) != source["fingerprint"]:
                raise ValueError("人物裁剪的原片指纹已变化")
            region = self.pipeline._project(source_asset_id, [raw], _catalog=catalog)[0]
            key = self._key(character_id, source_asset_id, region_id)
            previous = state["decisions"].get(key, {})
            if action == "confirm" and (not single_character or not use_as_reference):
                raise ValueError("请确认画面是清晰单人，并同意作为后续识别参考")
            if region["match"].get("source") == "human" and region["match"].get("status") == "confirmed" and region["match"].get("character_id") != character_id:
                raise ValueError("该区域已确认为另一角色，请先在人物复核中修正")
            if action != "confirm" and region["match"].get("source") == "human" and region["match"].get("status") == "confirmed":
                raise ValueError("已确认区域请在人物复核或参考图库中修改")
            registry_frames = read_json(self.library.asset_dir(source_asset_id) / "frames" / "registry.json", {})
            if region["at_ms"] != registry_frames.get(region["frame_id"], {}).get("at_ms"):
                raise ValueError("人物裁剪时间与注册帧不一致")
            canonical = registry_frames[region["frame_id"]].get("alias_of", region["frame_id"])
            reviews = read_json(self.pipeline._directory(source_asset_id) / "decisions.json", {"decisions": {}})["decisions"]
            for reviewed_id, review in reviews.items():
                if reviewed_id == region_id or review.get("source") != "human" or review.get("status") not in {"confirmed", "unknown"}:
                    continue
                reviewed = read_json(self.pipeline._directory(source_asset_id) / "regions" / f"{safe_id(reviewed_id)}.json", {})
                reviewed_frame = registry_frames.get(reviewed.get("frame_id"), {})
                if (reviewed_frame.get("alias_of", reviewed.get("frame_id")) == canonical and reviewed.get("box") == region["box"]):
                    raise ValueError("同一原片裁剪已有人工作出的复核决定，请先在人物复核中处理")
            self.library.frame_paths(source_asset_id, [region["frame_id"]], registry=registry_frames)
            box = checked_box(region["box"])
            if box is None:
                raise ValueError("人物裁剪缺少区域")
            decision = {"action": action, "character_id": character_id, "source_asset_id": source_asset_id,
                        "region_id": region_id, "source_fingerprint": source["fingerprint"],
                        "frame_id": region["frame_id"], "box": box, "updated_at": now(), "reason": reason}
            if action == "confirm":
                desired = (region["match"].get("source") == "human" and region["match"].get("status") == "confirmed"
                           and region["match"].get("character_id") == character_id)
                if (previous.get("action") in {"confirm", "confirming"} and not desired
                        and region["match"].get("source") == "human"
                        and region["match"].get("version", 0) > previous.get("prior_review_version", 0)):
                    raise ValueError("该区域在确认后已被人工修改，请在人物复核中处理，不能重试覆盖")
                decision["prior_review_version"] = previous.get("prior_review_version", region["match"].get("version", 0))
                try:
                    # Reuse the detector's verified exact pixel bounds. A .5
                    # coordinate can round differently after normalization.
                    encoded = rebuild_crop_bytes(self.library, source_asset_id, raw, registry=registry_frames)
                except ValueError as exc:
                    raise ValueError("原片裁剪已变化或未通过单人、尺寸、清晰度检查") from exc
                cropped = _image(encoded)
                # The immutable crop digest survives discarded PNGs and keeps
                # confirmation bound to the pixels originally recognized.
                expected_crop = raw.get("crop_sha256") or hashlib.sha256(
                    self.pipeline.crop_bytes(source_asset_id, raw)).hexdigest()
                if (not region.get("quality", {}).get("accepted")
                        or not image_quality(cropped, single_character=True)["accepted"]
                        or hashlib.sha256(encoded).hexdigest() != expected_crop):
                    raise ValueError("原片裁剪已变化或未通过单人、尺寸、清晰度检查")
                existing = next((row for row in self.gallery.list(source_asset_id, character_id, _catalog=catalog)["references"]
                                 if row["source"].get("confirmation_id") == key), None)
                if existing and existing["status"] != "confirmed":
                    raise ValueError("此参考图已被撤销或降级，请在参考图库中复核")
                if _before_confirmation_write is not None:
                    _before_confirmation_write()
                if previous.get("action") != "confirm":
                    self._save(scope, state, key, {**decision, "action": "confirming"})
                if existing is None:
                    existing = self.gallery.add(source_asset_id, character_id, encoded, source_kind="frame_crop",
                        confirmed=True, single_character=True, source={"asset_id": source_asset_id,
                        "frame_id": region["frame_id"], "box": box, "region_id": region_id,
                        "source_fingerprint": source["fingerprint"], "confirmation_id": key}, _catalog=catalog)
                decision["reference_id"] = existing["id"]
                # Recover a crash after reference creation without duplicate
                # references or repeated human-review revisions.
                if not desired:
                    self.pipeline.review(source_asset_id, region_id, status="confirmed", character_id=character_id,
                                         reason=reason.strip() or "角色确认：确认当前人物并授权正片裁剪作为后续识别参考",
                                         _catalog=catalog)
                if _defer_projection:
                    # The batch owns the same library lock through search
                    # publication. Leave each durable journal in confirming
                    # until its source asset's exact projection is published.
                    return {"scope_id": scope, "key": key, "decision": decision}
                from .scenes import SceneStore
                SceneStore(self.library).invalidate_identity_inferences(source_asset_id, region_ids=[region_id])
                if self.search is not None:
                    self.search.sync_asset(self.library.records(source_asset_id), source_asset_id)
            if previous.get("action") != action or action == "confirm" and previous.get("reference_id") != decision.get("reference_id"):
                self._save(scope, state, key, decision)

    def confirm_batch(self, asset_id, character_id, *, candidates, single_character, use_as_reference, reason=""):
        payload = ConfirmationBatch.model_validate({"candidates": candidates, "single_character": single_character,
                                                   "use_as_reference": use_as_reference, "reason": reason})
        with self.library.lock:
            catalog = CharacterStore(self.library).roster(asset_id)
            scope = catalog["scope_id"]
            registry = CharacterStore(self.library)._read(scope)
            character_id = CharacterStore._resolve(registry, character_id, strict=True)
            if character_id not in {row["id"] for row in catalog["profiles"]}:
                raise ValueError("角色不在当前作品名单中")
            if not registry["profiles"][character_id].get("recognition_enabled", True):
                raise ValueError("请先将该角色加入识别名单")
            results, tickets, sources, started = [], {}, {}, set()

            def failed(index, exc):
                item = results[index]
                key = self._key(character_id, item["source_asset_id"], item["region_id"])
                item.update(status="failed", error=str(exc))
                item["recovery_required"] = self._read(scope)["decisions"].get(key, {}).get("action") == "confirming"

            for candidate in payload.candidates:
                index = len(results)
                item = candidate.model_dump()
                results.append({**item, "status": "failed"})
                try:
                    tickets[index] = self.decide(asset_id, character_id, action="confirm", **item,
                        single_character=True, use_as_reference=True, reason=payload.reason,
                        _catalog=catalog, _defer_projection=True,
                        _before_confirmation_write=lambda: started.add(index))
                    sources.setdefault(candidate.source_asset_id, []).append(index)
                except Exception as exc:
                    failed(index, exc)
                    # A write may have succeeded before a subsequent review
                    # failed. Publish any durable changes once for this source,
                    # but leave the failed item's journal for an explicit retry.
                    if index in started and results[index]["recovery_required"]:
                        sources.setdefault(candidate.source_asset_id, []).append(index)

            from .scenes import SceneStore
            for source_id, indices in sources.items():
                try:
                    SceneStore(self.library).invalidate_identity_inferences(
                        source_id, region_ids=[results[index]["region_id"] for index in indices])
                    if self.search is not None:
                        self.search.sync_asset(self.library.records(source_id), source_id)
                except Exception as exc:
                    for index in indices:
                        if index in tickets:
                            failed(index, exc)
                    continue
                for index in indices:
                    if index not in tickets:
                        continue
                    ticket = tickets[index]
                    try:
                        # Read again after every item: a preceding finalize can
                        # succeed while a later one fails, without losing either
                        # the success or the unfinished confirmation journal.
                        state = self._read(scope)
                        previous = state["decisions"].get(ticket["key"], {})
                        decision = ticket["decision"]
                        if previous.get("action") != "confirm" or previous.get("reference_id") != decision["reference_id"]:
                            self._save(scope, state, ticket["key"], decision)
                        results[index].update(status="confirmed", recovery_required=False)
                    except Exception as exc:
                        failed(index, exc)
            confirmed = sum(row["status"] == "confirmed" for row in results)
            return {"results": results, "summary": {"requested": len(results), "confirmed": confirmed,
                                                    "failed": len(results) - confirmed}}

    @staticmethod
    def _selection(catalog):
        return sorted(row["id"] for row in catalog["profiles"] if row.get("recognition_enabled", True))

    def cancel_sampling(self, asset_id):
        scope = scope_id(self.library.get_asset(asset_id))
        with self._guard:
            thread = self._running.get(scope)
            value = read_json(self._root(scope) / "sampling.json")
            if value and thread and thread.is_alive():
                self._cancellations[scope].set()
                value.update(status="cancelling", updated_at=now())
                atomic_json(self._root(scope) / "sampling.json", value)
            elif value and value.get("status") in {"running", "cancelling"}:
                value.update(status="cancelled", updated_at=now())
                atomic_json(self._root(scope) / "sampling.json", value)
            return value

    def _sampling_signature(self, catalog, assets, budget, request_scope):
        return digest(["role-confirmation-v2", catalog["scope_id"], self._selection(catalog), request_scope, budget,
                       [(row["id"], row["fingerprint"], self.pipeline.settings(row["id"])["model"]) for row in assets],
                       [(asset["id"], [(row["id"], row["version"], row["character_id"], row["sha256"], row["status"])
                        for row in self.gallery.matching(asset["id"], _catalog=catalog)]) for asset in assets]])

    def _sampling_inputs_current(self, value):
        """Recheck every input in the persisted work signature, without inference."""
        catalog = CharacterStore(self.library).roster(value["asset_id"])
        if catalog["scope_id"] != value["scope_id"] or self._selection(catalog) != value["selected_character_ids"]:
            return False
        assets = sorted(catalog["assets"], key=lambda row: (row.get("season") or 0, row.get("episode") or 0, row["id"]))
        if value["request_scope"] == "asset":
            assets = [row for row in assets if row["id"] == value["asset_id"]]
        return bool(assets) and self._sampling_signature(catalog, assets, value["budget"], value["request_scope"]) == value["base_id"]

    def start_sampling(self, asset_id, max_frames=12, *, scope="asset", max_assets=24, frames_per_asset=12):
        payload = ConfirmationSample.model_validate({"scope": scope, "max_frames": max_frames,
            "max_assets": max_assets, "frames_per_asset": frames_per_asset})
        # Never hold the worker guard while obtaining the library lock.
        catalog = CharacterStore(self.library).roster(asset_id)
        work = catalog["scope_id"]
        selected = self._selection(catalog)
        if not selected:
            raise ValueError("请先选择需要识别的角色")
        assets = sorted(catalog["assets"], key=lambda row: (row.get("season") or 0, row.get("episode") or 0, row["id"]))
        if scope == "asset":
            assets = [row for row in assets if row["id"] == asset_id]
        if not assets:
            raise ValueError("作品没有可采样的视频")
        budget = {"max_frames": payload.max_frames, "max_assets": payload.max_assets if scope == "work" else 1,
                  "frames_per_asset": payload.frames_per_asset if scope == "work" else payload.max_frames}
        base_id = self._sampling_signature(catalog, assets, budget, scope)
        path = self._root(work) / "sampling.json"
        prior = self.sampling(asset_id) or {}
        evidence_id = self._sampling_signature(catalog, assets, {}, "evidence")
        history = read_json(self._root(work) / "sampled-history.json", {"frames": []})
        used = {(row["asset_id"], row["source_fingerprint"], row["at_ms"]) for row in history.get("frames", [])
                if row.get("evidence_id", row.get("base_id")) == evidence_id}
        resume = (prior.get("sampling_version") == "2.0" and prior.get("base_id") == base_id
                  and prior.get("status") in {"failed", "unavailable", "interrupted", "cancelled", "partial"})
        sample_round = prior.get("round", 0) if resume else prior.get("round", -1) + 1 if prior.get("base_id") == base_id else 0
        if resume:
            # A persisted plan is data, never authority to read another work.
            from .character_confirmation_validation import validate_sampling_v2
            validate_sampling_v2(self.library, work, prior)
            value = copy.deepcopy(prior)
            value.setdefault("attempt_history", []).append({"status": prior["status"], "completed": prior["completed"],
                "failed_frames": prior["failed_frames"], "error": prior.get("error"), "updated_at": prior["updated_at"],
                "failures": [{"asset_id": row["asset_id"], "frame_id": row["frame_id"], "error": row.get("error")}
                             for row in prior["plan"] if row["status"] == "failed"]})
            for row in value["plan"]:
                if row.get("status") == "failed":
                    row.update(status="pending")
                    row.pop("error", None)
            for child in value["assets"]:
                if child["total"]:
                    child["status"] = "completed" if child["completed"] == child["total"] else "pending"
                    child.pop("error", None)
            value.update(status="running", failed_frames=0, attempted=value["completed"], updated_at=now())
            value.pop("error", None)
        else:
            # Rotate episode windows on explicit later rounds; one global frame
            # budget is shared fairly across all selected episodes.
            offset = (sample_round * budget["max_assets"]) % len(assets)
            chosen = (assets[offset:] + assets[:offset])[:budget["max_assets"]]
            buckets, children = [], []
            for asset in chosen:
                identifier, fingerprint = asset["id"], asset["fingerprint"]
                child = {"asset_id": identifier, "source_fingerprint": fingerprint, "total": 0,
                         "completed": 0, "status": "pending"}
                children.append(child)
                if asset.get("source_changed"):
                    child.update(status="unavailable", error="原片指纹变化，请重新定位")
                    buckets.append([])
                    continue
                registry = read_json(self.library.asset_dir(identifier) / "frames" / "registry.json", {})
                stored = sorted({row["at_ms"] for row in registry.values() if type(row.get("at_ms")) is int
                                 and 0 <= row["at_ms"] < asset["duration_ms"]
                                 and (identifier, fingerprint, row["at_ms"]) not in used})
                limit = budget["frames_per_asset"]
                # Spread stored keyframes across the episode rather than taking
                # the beginning of a large registry. Pixels/features stay local.
                saved = [stored[min(len(stored) - 1, index * len(stored) // min(len(stored), limit))]
                         for index in range(min(len(stored), limit))]
                times = list(saved)
                for index in range(limit * 8):
                    requested = min(asset["duration_ms"] - 1, round(asset["duration_ms"] * (
                        ((index + 1) / (limit + 1) + sample_round * .61803398875) % 1)))
                    if (any(abs(requested - existing) <= 500 for existing in times)
                            or any(a == identifier and f == fingerprint and abs(requested - at) <= 500 for a, f, at in used)):
                        continue
                    times.append(requested)
                    if len(times) >= limit:
                        break
                buckets.append([{"asset_id": identifier, "source_fingerprint": fingerprint, "at_ms": at,
                                 "frame_id": "frame_" + digest([base_id, sample_round, identifier, at]), "status": "pending"}
                                for at in times[:limit]])
            plan = []
            for index in range(budget["frames_per_asset"]):
                for bucket, child in zip(buckets, children):
                    if index < len(bucket) and len(plan) < budget["max_frames"]:
                        plan.append(bucket[index])
                        child["total"] += 1
            plan_identity = [{key: row[key] for key in ("asset_id", "source_fingerprint", "at_ms", "frame_id")} for row in plan]
            value = {"schema_version": "1.0", "sampling_version": "2.0", "id": digest([base_id, sample_round, plan_identity]),
                     "base_id": base_id, "evidence_id": evidence_id, "round": sample_round, "asset_id": asset_id, "scope_id": work,
                     "request_scope": scope, "selected_character_ids": selected, "budget": budget,
                     "status": "running", "plan": plan, "assets": children, "total": len(plan),
                     "completed": 0, "attempted": 0, "failed_frames": 0, "frame_ids": [], "completed_frames": [],
                     "requested_times": [row["at_ms"] for row in plan], "reused_frames": 0,
                     "detected_regions": 0, "accepted_regions": 0, "updated_at": now()}
        with self._guard:
            thread = self._running.get(work)
            if thread and thread.is_alive():
                return copy.deepcopy(read_json(path))
            self._cancellations[work] = threading.Event()
            atomic_json(path, value)
            thread = threading.Thread(target=self._sample, args=(catalog, value), daemon=True,
                                      name=f"character-confirmation-{asset_id}")
            self._running[work] = thread
            thread.start()
            return copy.deepcopy(value)

    def _sample(self, catalog, value):
        from .media import extract_frame
        from .identity_bindings import POLICY_VERSION
        work = value["scope_id"]
        path = self._root(work) / "sampling.json"
        children = {row["asset_id"]: row for row in value["assets"]}
        stop = self._cancellations[work]
        history_path = self._root(work) / "sampled-history.json"
        try:
            for item in value["plan"]:
                if item["status"] == "completed":
                    continue
                if stop.is_set():
                    value.update(status="cancelled")
                    break
                with self.library.lock:
                    inputs_current = self._sampling_inputs_current(value)
                if not inputs_current:
                    value.update(status="invalidated", error="作品、名单、参考图或模型已变化，请重新生成")
                    break
                identifier, requested, asset_id = item["frame_id"], item["at_ms"], item["asset_id"]
                current = self.library.get_asset(asset_id)
                if scope_id(current) != work or current["fingerprint"] != item["source_fingerprint"] or current.get("source_changed"):
                    value.update(status="invalidated", error="采样期间来源或作品作用域已变化，请重新生成")
                    break
                child = children[asset_id]
                try:
                    registry = read_json(self.library.asset_dir(asset_id) / "frames" / "registry.json", {})
                    reused = identifier in registry
                    if identifier not in registry:
                        nearby = sorted((abs(row["at_ms"] - requested), key) for key, row in registry.items()
                                        if abs(row["at_ms"] - requested) <= 500)
                        for _, existing_id in nearby:
                            try:
                                self.library.frame_path(asset_id, existing_id)
                                self.library.register_sample_alias(asset_id, identifier, existing_id)
                                registry = read_json(self.library.asset_dir(asset_id) / "frames" / "registry.json", {})
                                reused = True
                                break
                            except FileNotFoundError:
                                continue
                    if identifier in registry:
                        self.library.frame_path(asset_id, identifier)
                        frame = {"id": identifier, "at_ms": registry[identifier]["at_ms"]}
                    else:
                        source = self.library.require_source(asset_id)
                        frame = extract_frame(source, requested, self.library.asset_dir(asset_id) / "frames" / f"{identifier}.png")
                        self.library.register_frame(asset_id, frame)
                    config = {**self.pipeline.settings(asset_id), "enabled": True, "calibration": None, "top_k": 20,
                              "identity_policy": POLICY_VERSION}
                    # Candidate-only matching cannot overwrite a canonical
                    # calibrated or manually reviewed region: use sample aliases.
                    result = self.pipeline.run(asset_id, [frame], config)
                    if result["status"] != "completed":
                        raise RuntimeError(result.get("degraded_reason") or "本地视觉模型不可用")
                    with self.library.lock:
                        if not self._sampling_inputs_current(value):
                            value.update(status="invalidated", error="生成期间作品、名单、参考图或模型已变化，请重新生成")
                            break
                        if stop.is_set():
                            value["status"] = "cancelled"
                        item.update(status="completed", actual_at_ms=frame["at_ms"])
                        value["completed_frames"].append({"asset_id": asset_id, "frame_id": frame["id"],
                            "at_ms": frame["at_ms"], "source_fingerprint": item["source_fingerprint"]})
                        value["frame_ids"].append(frame["id"])
                        value["completed"] += 1
                        value["reused_frames"] += int(reused)
                        value["detected_regions"] += len(result.get("regions", []))
                        value["accepted_regions"] += sum(bool(row.get("quality", {}).get("accepted")) for row in result.get("regions", []))
                        child["completed"] += 1
                        child.update(status="completed" if child["completed"] == child["total"] else "running")
                        record = {"asset_id": asset_id, "source_fingerprint": item["source_fingerprint"],
                                  "at_ms": frame["at_ms"], "base_id": value["base_id"], "evidence_id": value.get("evidence_id")}
                        try:
                            history = read_json(history_path, {"frames": []})
                            if record not in history["frames"]:
                                history["frames"].append(record)
                                atomic_json(history_path, history)
                        except Exception as exc:
                            value.setdefault("warnings", []).append("采样去重历史保存失败: " + str(exc))
                except Exception as exc:
                    item.update(status="failed", error=str(exc))
                    child.update(status="failed", error=str(exc))
                    value["failed_frames"] += 1
                    value["error"] = str(exc)
                    if value["request_scope"] == "asset":
                        value["status"] = "failed"
                value["attempted"] = value["completed"] + value["failed_frames"]
                value["updated_at"] = now()
                atomic_json(path, value)
                if value["status"] in {"failed", "cancelled"}:
                    break
            else:
                value["status"] = ("partial" if value["completed"] else "failed") if value["failed_frames"] else "completed"
            if value["total"] == 0:
                value.update(status="unavailable", error="没有尚未采样的有效帧，或作品原片均已变化")
            for child in children.values():
                if child["status"] in {"running", "pending"}:
                    child["status"] = "completed" if child["completed"] == child["total"] else value["status"]
        except Exception as exc:
            value.update(status="failed", error=str(exc))
        finally:
            value["updated_at"] = now()
            atomic_json(path, value)


def register_character_confirmation_routes(app, library, search, pipeline):
    service = CharacterConfirmation(library, pipeline, search)
    app.state.character_confirmation = service

    @app.get("/api/assets/{asset_id}/identity/confirmation")
    def confirmation_queue(asset_id: str, review_pool_cursor: str | None = None, review_pool_limit: int = 20):
        return service.queue(asset_id, review_pool_cursor=review_pool_cursor, review_pool_limit=review_pool_limit)

    @app.put("/api/assets/{asset_id}/identity/confirmation/selection")
    def confirmation_selection(asset_id: str, payload: RecognitionSelection):
        CharacterStore(library).set_recognition_selection(asset_id, payload.character_ids,
            expected_revision=payload.expected_revision, expected_scope=payload.scope_id)
        return service.queue(asset_id)

    @app.get("/api/assets/{asset_id}/identity/confirmation/sampling")
    def confirmation_sampling(asset_id: str):
        return {"sampling": service.sampling(asset_id)}

    @app.post("/api/assets/{asset_id}/identity/confirmation/characters/{character_id}/decision")
    def confirmation_decision(asset_id: str, character_id: str, payload: ConfirmationDecision):
        service.decide(asset_id, character_id, **payload.model_dump())
        return service.queue(asset_id)

    @app.post("/api/assets/{asset_id}/identity/confirmation/characters/{character_id}/confirm-batch")
    def confirmation_batch(asset_id: str, character_id: str, payload: ConfirmationBatch):
        result = service.confirm_batch(asset_id, character_id, **payload.model_dump())
        return {**result, "queue": service.queue(asset_id)}

    @app.post("/api/assets/{asset_id}/identity/confirmation/sample")
    def confirmation_sample(asset_id: str, payload: ConfirmationSample):
        service.start_sampling(asset_id, payload.max_frames, scope=payload.scope, max_assets=payload.max_assets,
                               frames_per_asset=payload.frames_per_asset)
        return service.queue(asset_id)

    @app.post("/api/assets/{asset_id}/identity/confirmation/sampling/cancel")
    def confirmation_cancel_sampling(asset_id: str):
        service.cancel_sampling(asset_id)
        return service.queue(asset_id)
