"""Work-scoped, human-reviewed original-frame exemplars before large runs.

Reading the queue never invokes a model. Optional sampling is bounded, local,
resumable and candidate-only; a confirmation affects exactly one region.
"""
from __future__ import annotations

import copy
import hashlib
import os
import stat
import threading
from pathlib import Path
from typing import Literal

from pydantic import Field

from .characters import CharacterStore, checked_box, scope_id
from .library import atomic_json, confined_path, digest, now, read_json, safe_id
from .models import Contract
from .visual_identity import VisualIdentityPipeline, _image, _png, image_quality


class ConfirmationDecision(Contract):
    action: Literal["confirm", "reject", "unclear", "skip", "resume"]
    source_asset_id: str | None = None
    region_id: str | None = None
    single_character: bool = False
    use_as_reference: bool = False
    reason: str = Field(default="", max_length=4000)


class ConfirmationSample(Contract):
    max_frames: int = Field(default=12, ge=1, le=24, strict=True)


class CharacterConfirmation:
    def __init__(self, library, pipeline=None, search=None):
        self.library = library
        self.pipeline = pipeline or VisualIdentityPipeline(library)
        self.gallery = self.pipeline.gallery
        self.search = search
        self._guard = threading.RLock()
        self._running = {}

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
            if value and value["status"] == "running" and not (thread and thread.is_alive()):
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

    def queue(self, asset_id):
        with self.library.lock:
            catalog = CharacterStore(self.library).roster(asset_id)
            scope = catalog["scope_id"]
            registry = CharacterStore(self.library)._read(scope)
            state = self._read(scope)
            profiles = {row["id"]: row for row in catalog["profiles"]}
            # Resolve merged identities when projecting historical exclusions.
            decisions = {}
            for row in state["decisions"].values():
                identity = CharacterStore._resolve(registry, row["character_id"])
                if identity in profiles:
                    key = self._key(identity, row.get("source_asset_id"), row.get("region_id"))
                    decisions[key] = row
            trusted = self.gallery.list(asset_id, trusted_only=True, _catalog=catalog)["references"]
            active_references = self.gallery.active_reference_ids(asset_id, include_candidates=True, _catalog=catalog)
            pending_regions = {}
            for decision in decisions.values():
                if decision["action"] == "confirming":
                    identity = CharacterStore._resolve(registry, decision["character_id"])
                    pending_regions.setdefault((decision["source_asset_id"], decision["region_id"]), set()).add(identity)
            grouped = {identity: [] for identity in profiles}
            for asset in catalog["assets"]:
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
                regions = self.pipeline._project(asset["id"], raw, _catalog=catalog, _crop_paths=False)
                crop_exists = self._crop_inventory(asset["id"])
                frame_shots, shots = self._shots(asset)
                policy = CharacterStore.cast_policy(registry, asset.get("anime_profile"))
                allowed = {identity for identity, profile in profiles.items()
                           if CharacterStore.profile_allowed(profile, policy)}
                for region in regions:
                    pending = pending_regions.get((asset["id"], region["id"]), set()) & allowed
                    if (not pending and region["match"].get("status") == "confirmed"
                            or not pending and region["match"].get("source") == "human" and region["match"].get("status") == "unknown"
                            or not crop_exists(region["id"])):
                        continue
                    shot = frame_shots.get(region["frame_id"])
                    if not shot:
                        shot = next((identity for start, end, identity in shots if start <= region["at_ms"] < end), None)
                    proposals = [row for row in region["match"].get("candidates", [])
                                 if row.get("reference_ids") and set(row["reference_ids"]).issubset(active_references)]
                    if region["match"].get("source") == "human":
                        candidate = region["match"].get("candidate_character_id")
                        proposals = [{"character_id": candidate, "distance": None}] if candidate else []
                    if pending:
                        proposals = [{"character_id": identity, "distance": None} for identity in sorted(pending)]
                    for proposed in proposals:
                        identity = proposed["character_id"]
                        if identity not in allowed:
                            continue
                        previous = decisions.get(self._key(identity, asset["id"], region["id"]))
                        if previous and previous["action"] in {"confirm", "reject", "unclear"}:
                            continue
                        grouped[identity].append({
                            "region_id": region["id"], "asset_id": asset["id"], "asset_title": asset["title"],
                            "source_available": asset.get("source_available", False), "frame_id": region["frame_id"],
                            "recovery_required": identity in pending,
                            "at_ms": region["at_ms"], "box": region["box"], "crop_url": region["crop_url"],
                            "shot_id": shot, "distance": proposed.get("distance"), "quality": region["quality"]})
            roles = []
            for identity, profile in profiles.items():
                choices = sorted(grouped[identity], key=lambda row: (
                    not row.get("recovery_required", False),
                    row["distance"] if row["distance"] is not None else float("inf"),
                    -row["quality"].get("sharpness", 0), row["asset_id"], row["at_ms"], row["region_id"]))
                selected = []
                for row in choices:
                    # An unplanned sample only promises temporal separation;
                    # a known shot is never represented twice in one group.
                    if any(row["asset_id"] == other["asset_id"] and (
                            row["shot_id"] and row["shot_id"] == other["shot_id"]
                            or abs(row["at_ms"] - other["at_ms"]) < 5000) for other in selected):
                        continue
                    selected.append(row)
                    if len(selected) == 3:
                        break
                count = sum(row["character_id"] == identity and row["source"]["kind"] == "frame_crop" for row in trusted)
                role_decision = decisions.get(self._key(identity), {})
                recovering = any(row.get("recovery_required") for row in choices)
                for candidate in selected:
                    # Resolve every displayed image using the same confinement
                    # boundary as other visual evidence endpoints.
                    confined_path(self.pipeline._directory(candidate["asset_id"]), f"crops/{safe_id(candidate['region_id'])}.png")
                status = "pending" if recovering else "confirmed" if count else "skipped" if role_decision.get("action") == "skip" else "pending"
                roles.append({"character_id": identity, "profile": profile, "status": status,
                              "trusted_reference_count": count, "candidates": selected, "available_count": len(choices)})
            asset = self.library.get_asset(asset_id)
        # Keep lock ordering independent from the sampling worker's guard.
        return {"asset_id": asset_id, "scope_id": scope, "roles": roles, "revision": state["revision"],
                "summary": {"total": len(roles), **{key: sum(row["status"] == key for row in roles)
                            for key in ("confirmed", "skipped", "pending")},
                            "candidates": sum(row["available_count"] for row in roles)},
                "sampling": self.sampling(asset_id), "sampling_available": asset.get("source_available", False),
                "sampling_unavailable_reason": "" if asset.get("source_available") else "原片离线或已变化，请先重新定位原片",
                "guidance": "确认仅标记当前区域，并将单人正片裁剪加入本作品可信图库。其他镜头仍需独立匹配与验证；未出场角色可跳过。"}

    def decide(self, asset_id, character_id, *, action, source_asset_id=None, region_id=None,
               single_character=False, use_as_reference=False, reason=""):
        if action not in {"confirm", "reject", "unclear", "skip", "resume"}:
            raise ValueError("角色确认操作无效")
        with self.library.lock:
            catalog = CharacterStore(self.library).roster(asset_id)
            scope = catalog["scope_id"]
            registry = CharacterStore(self.library)._read(scope)
            character_id = CharacterStore._resolve(registry, character_id, strict=True)
            if character_id not in {row["id"] for row in catalog["profiles"]}:
                raise ValueError("角色不在当前作品名单中")
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
            if region["match"].get("status") == "confirmed" and region["match"].get("character_id") != character_id:
                raise ValueError("该区域已确认为另一角色，请先在人物复核中修正")
            if action != "confirm" and region["match"].get("status") == "confirmed":
                raise ValueError("已确认区域请在人物复核或参考图库中修改")
            registry_frames = read_json(self.library.asset_dir(source_asset_id) / "frames" / "registry.json", {})
            if region["at_ms"] != registry_frames.get(region["frame_id"], {}).get("at_ms"):
                raise ValueError("人物裁剪时间与注册帧不一致")
            frame = self.library.frame_paths(source_asset_id, [region["frame_id"]], registry=registry_frames)[region["frame_id"]]
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
                picture = _image(frame.read_bytes())
                x, y, width, height = box
                cropped = picture.crop((round(x * picture.width), round(y * picture.height),
                                        round((x + width) * picture.width), round((y + height) * picture.height)))
                encoded = _png(cropped)
                if (not region.get("quality", {}).get("accepted")
                        or not image_quality(cropped, single_character=True)["accepted"]
                        or hashlib.sha256(encoded).digest() != hashlib.sha256(Path(region["crop_path"]).read_bytes()).digest()):
                    raise ValueError("原片裁剪已变化或未通过单人、尺寸、清晰度检查")
                existing = next((row for row in self.gallery.list(source_asset_id, character_id, _catalog=catalog)["references"]
                                 if row["source"].get("confirmation_id") == key), None)
                if existing and existing["status"] != "confirmed":
                    raise ValueError("此参考图已被撤销或降级，请在参考图库中复核")
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
                if region["match"].get("source") != "human" or region["match"].get("character_id") != character_id:
                    self.pipeline.review(source_asset_id, region_id, status="confirmed", character_id=character_id,
                                         reason=reason.strip() or "角色确认：确认当前人物并授权正片裁剪作为后续识别参考",
                                         _catalog=catalog)
                from .scenes import SceneStore
                SceneStore(self.library).invalidate_identity_inferences(source_asset_id, region_ids=[region_id])
                if self.search is not None:
                    self.search.sync_asset(self.library.records(source_asset_id), source_asset_id)
            if previous.get("action") != action or action == "confirm" and previous.get("reference_id") != decision.get("reference_id"):
                self._save(scope, state, key, decision)

    def start_sampling(self, asset_id, max_frames=12):
        if type(max_frames) is not int or not 1 <= max_frames <= 24:
            raise ValueError("少量采样必须限制为 1–24 帧")
        asset = self.library.get_asset(asset_id)
        source = self.library.require_source(asset_id)
        scope = scope_id(asset)
        if not self.gallery.matching(asset_id):
            raise ValueError("请先缓存 Bangumi 参考图或添加角色参考图，再开始少量采样")
        config = {**self.pipeline.settings(asset_id), "enabled": True, "calibration": None, "top_k": 20}
        from .identity_bindings import POLICY_VERSION
        config["identity_policy"] = POLICY_VERSION
        base_id = digest(["role-confirmation-v1", asset_id, asset["fingerprint"], scope, max_frames, config["model"]])
        with self._guard:
            thread = self._running.get(scope)
            if thread and thread.is_alive():
                return self.sampling(asset_id)
            path = self._root(scope) / "sampling.json"
            prior = read_json(path, {})
            resume = prior.get("base_id") == base_id and prior.get("status") != "completed"
            sample_round = prior.get("round", 0) if resume else prior.get("round", -1) + 1 if prior.get("base_id") == base_id else 0
            # Shift later explicitly requested rounds so dismissing a group
            # can discover other footage without escalating to an episode scan.
            times = sorted({min(asset["duration_ms"] - 1, round(asset["duration_ms"] * (
                ((index + 1) / (max_frames + 1) + sample_round * .61803398875) % 1)))
                for index in range(max_frames)})
            identity = digest([base_id, sample_round, times])
            value = {"schema_version": "1.0", "id": identity, "base_id": base_id, "round": sample_round,
                     "asset_id": asset_id, "scope_id": scope,
                     "status": "running", "source_fingerprint": asset["fingerprint"],
                     "requested_times": times, "total": len(times),
                     "completed": prior.get("completed", 0) if resume else 0,
                     "frame_ids": prior.get("frame_ids", []) if resume else [], "updated_at": now()}
            atomic_json(path, value)
            thread = threading.Thread(target=self._sample, args=(asset, source, config, value),
                                      daemon=True, name=f"character-confirmation-{asset_id}")
            self._running[scope] = thread
            thread.start()
            return copy.deepcopy(value)

    def _sample(self, asset, source, config, value):
        from .media import extract_frame
        asset_id, scope = asset["id"], value["scope_id"]
        path = self._root(scope) / "sampling.json"
        try:
            for requested in value["requested_times"][value["completed"]:]:
                current = self.library.get_asset(asset_id)
                if scope_id(current) != scope or current["fingerprint"] != asset["fingerprint"]:
                    raise ValueError("采样期间原片或作品归属已变化，请重新开始")
                self.library.require_source(asset_id)
                identifier = "frame_" + digest([value["id"], requested])
                registry = read_json(self.library.asset_dir(asset_id) / "frames" / "registry.json", {})
                if identifier not in registry:
                    nearby = sorted((abs(row["at_ms"] - requested), identity) for identity, row in registry.items()
                                    if identity not in value["frame_ids"] and abs(row["at_ms"] - requested) <= 500)
                    for _, existing_id in nearby:
                        try:
                            existing_path = self.library.frame_path(asset_id, existing_id)
                            # Reuse local pixels/detection/features with a
                            # sample-owned frame ID. Candidate-only sampling
                            # must not overwrite a canonical calibrated region.
                            self.library.register_frame(asset_id, {"id": identifier, "path": str(existing_path),
                                                                 "at_ms": registry[existing_id]["at_ms"]})
                            registry[identifier] = registry[existing_id]
                            break
                        except FileNotFoundError:
                            continue
                if identifier in registry:
                    self.library.frame_path(asset_id, identifier)
                    frame = {"id": identifier, "at_ms": registry[identifier]["at_ms"]}
                else:
                    frame = extract_frame(source, requested, self.library.asset_dir(asset_id) / "frames" / f"{identifier}.png")
                    self.library.register_frame(asset_id, frame)
                result = self.pipeline.run(asset_id, [frame], config)
                if result["status"] != "completed":
                    value.update(status="unavailable", error=result.get("degraded_reason", "本机视觉模型不可用"))
                    break
                value["frame_ids"].append(frame["id"])
                value.update(completed=value["completed"] + 1, updated_at=now())
                atomic_json(path, value)
            else:
                value.update(status="completed")
        except Exception as exc:
            value.update(status="failed", error=str(exc))
        finally:
            value["updated_at"] = now()
            atomic_json(path, value)


def register_character_confirmation_routes(app, library, search, pipeline):
    service = CharacterConfirmation(library, pipeline, search)
    app.state.character_confirmation = service

    @app.get("/api/assets/{asset_id}/identity/confirmation")
    def confirmation_queue(asset_id: str):
        return service.queue(asset_id)

    @app.get("/api/assets/{asset_id}/identity/confirmation/sampling")
    def confirmation_sampling(asset_id: str):
        return {"sampling": service.sampling(asset_id)}

    @app.post("/api/assets/{asset_id}/identity/confirmation/characters/{character_id}/decision")
    def confirmation_decision(asset_id: str, character_id: str, payload: ConfirmationDecision):
        service.decide(asset_id, character_id, **payload.model_dump())
        return service.queue(asset_id)

    @app.post("/api/assets/{asset_id}/identity/confirmation/sample")
    def confirmation_sample(asset_id: str, payload: ConfirmationSample):
        service.start_sampling(asset_id, payload.max_frames)
        return service.queue(asset_id)
