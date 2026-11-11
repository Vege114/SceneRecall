"""Validate character confirmation provenance before publishing a restored backup."""
from __future__ import annotations

from .characters import CharacterStore, checked_box, scope_id
from .library import digest, read_json, safe_id


def _identity(registry, value):
    safe_id(value)
    return CharacterStore._resolve(registry, value, strict=True)


def _count(value, *, maximum=None):
    return type(value) is int and value >= 0 and (maximum is None or value <= maximum)


def validate_sampling_v2(library, scope, sample):
    """Validate a bounded work plan without trusting its stored paths or IDs."""
    registry = CharacterStore(library)._read(scope)
    invalid = "角色确认跨集采样备份格式无效"
    statuses = {"running", "cancelling", "cancelled", "completed", "partial", "failed",
                "unavailable", "interrupted", "invalidated"}
    if (not isinstance(sample, dict) or sample.get("schema_version") != "1.0"
            or sample.get("sampling_version") != "2.0" or sample.get("scope_id") != scope
            or not isinstance(sample.get("status"), str) or sample["status"] not in statuses
            or not isinstance(sample.get("request_scope"), str) or sample["request_scope"] not in {"asset", "work"}
            or not isinstance(sample.get("updated_at"), str)
            or not isinstance(sample.get("plan"), list)
            or not isinstance(sample.get("assets"), list)
            or not isinstance(sample.get("completed_frames"), list)
            or not isinstance(sample.get("selected_character_ids"), list)
            or not isinstance(sample.get("budget"), dict)):
        raise ValueError(invalid)
    budget = sample["budget"]
    maximum = 24 if sample["request_scope"] == "asset" else 120
    if any(not _count(budget.get(key), maximum=limit) or budget[key] < 1 for key, limit in (
            ("max_frames", maximum), ("max_assets", 24), ("frames_per_asset", 24))):
        raise ValueError(invalid)
    total, completed, attempted = (sample.get(key) for key in ("total", "completed", "attempted"))
    if (not _count(total, maximum=budget["max_frames"]) or not _count(completed, maximum=total)
            or not _count(attempted, maximum=total) or not _count(sample.get("failed_frames"), maximum=total)
            or attempted != completed + sample["failed_frames"]
            or len(sample["plan"]) != total or len(sample["completed_frames"]) != completed
            or not _count(sample.get("round"))):
        raise ValueError(invalid)
    safe_id(sample.get("base_id"))
    selected = sample["selected_character_ids"]
    for character_id in selected:
        _identity(registry, character_id)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(invalid)
    entry_id = safe_id(sample.get("asset_id"))
    entry = library.get_asset(entry_id, include_deleted=True)
    if scope_id(entry) != scope:
        raise ValueError("角色确认跨集采样入口来源不一致")
    sources, planned, totals, failures = {}, {}, {}, 0
    def source(asset_id):
        if asset_id not in sources:
            sources[asset_id] = library.get_asset(asset_id, include_deleted=True)
        return sources[asset_id]

    for item in sample["plan"]:
        if not isinstance(item, dict):
            raise ValueError(invalid)
        asset_id, frame_id = safe_id(item.get("asset_id")), safe_id(item.get("frame_id"))
        asset = source(asset_id)
        if (scope_id(asset) != scope or item.get("source_fingerprint") != asset["fingerprint"]
                or type(item.get("at_ms")) is not int or not 0 <= item["at_ms"] < asset["duration_ms"]
                or (asset_id, frame_id) in planned
                or frame_id != "frame_" + digest([sample["base_id"], sample["round"], asset_id, item["at_ms"]])
                or not isinstance(item.get("status"), str) or item["status"] not in {"pending", "completed", "failed"}
                or "error" in item and not isinstance(item["error"], str)):
            raise ValueError("角色确认跨集采样计划来源或时间范围无效")
        planned[(asset_id, frame_id)] = item
        totals[asset_id] = totals.get(asset_id, 0) + 1
        failures += item["status"] == "failed"
    plan_identity = [{key: item[key] for key in ("asset_id", "source_fingerprint", "at_ms", "frame_id")}
                     for item in sample["plan"]]
    if (sample.get("id") != digest([sample["base_id"], sample["round"], plan_identity])
            or not isinstance(sample.get("requested_times"), list)
            or any(type(value) is not int for value in sample["requested_times"])
            or sample.get("requested_times") != [item["at_ms"] for item in sample["plan"]]
            or failures != sample["failed_frames"]):
        raise ValueError("角色确认跨集采样轮次或进度无效")
    if (len(sources) > budget["max_assets"]
            or any(value > budget["frames_per_asset"] for value in totals.values())
            or sample["request_scope"] == "asset" and set(sources) - {entry_id}):
        raise ValueError("角色确认跨集采样计划超过预算")
    completed_keys, per_asset, registries = set(), {}, {}
    for item in sample["completed_frames"]:
        if not isinstance(item, dict):
            raise ValueError(invalid)
        asset_id, frame_id = safe_id(item.get("asset_id")), safe_id(item.get("frame_id"))
        key = asset_id, frame_id
        if key not in planned or key in completed_keys or planned[key]["status"] != "completed":
            raise ValueError("角色确认跨集采样完成帧不属于计划或重复")
        if asset_id not in registries:
            registries[asset_id] = read_json(library.asset_dir(asset_id) / "frames" / "registry.json", {})
        frames = registries[asset_id]
        if (item.get("source_fingerprint") != planned[key]["source_fingerprint"]
                or type(item.get("at_ms")) is not int
                or item["at_ms"] != frames.get(frame_id, {}).get("at_ms")
                or type(planned[key].get("actual_at_ms")) is not int
                or item["at_ms"] != planned[key]["actual_at_ms"]):
            raise ValueError("角色确认跨集采样完成帧来源不一致")
        library.check_frame_entries(asset_id, [frame_id], registry=frames)
        completed_keys.add(key)
        per_asset[asset_id] = per_asset.get(asset_id, 0) + 1
    if {key for key, item in planned.items() if item["status"] == "completed"} != completed_keys:
        raise ValueError("角色确认跨集采样完成帧清单不完整")
    children = {}
    for child in sample["assets"]:
        if not isinstance(child, dict):
            raise ValueError(invalid)
        asset_id = safe_id(child.get("asset_id"))
        asset = source(asset_id)
        if (asset_id in children or scope_id(asset) != scope
                or child.get("source_fingerprint") != asset["fingerprint"]
                or not _count(child.get("total")) or child["total"] != totals.get(asset_id, 0)
                or not _count(child.get("completed")) or child["completed"] != per_asset.get(asset_id, 0)
                or not isinstance(child.get("status"), str) or child["status"] not in statuses | {"pending"}
                or "error" in child and not isinstance(child["error"], str)):
            raise ValueError("角色确认跨集采样分集进度无效")
        children[asset_id] = child
    if (set(children) != set(sources) or len(children) > budget["max_assets"]
            or sample["request_scope"] == "asset" and set(children) - {entry_id}):
        raise ValueError("角色确认跨集采样分集进度不完整")
    if sample.get("frame_ids") != [item["frame_id"] for item in sample["completed_frames"]]:
        raise ValueError("角色确认跨集采样兼容帧清单无效")
    if (sample["status"] == "completed" and (attempted != total or failures)
            or sample["status"] == "partial" and (attempted != total or not completed or not failures)):
        raise ValueError("角色确认跨集采样结束状态与进度不一致")
    for key in ("reused_frames", "detected_regions", "accepted_regions"):
        if key in sample and not _count(sample[key]):
            raise ValueError(invalid)
    if sample.get("reused_frames", 0) > completed or sample.get("accepted_regions", 0) > sample.get("detected_regions", 0):
        raise ValueError(invalid)
    if "error" in sample and not isinstance(sample["error"], str):
        raise ValueError(invalid)
    if (not isinstance(sample.get("warnings", []), list)
            or any(not isinstance(value, str) for value in sample.get("warnings", []))
            or not isinstance(sample.get("attempt_history", []), list)):
        raise ValueError(invalid)
    for attempt in sample.get("attempt_history", []):
        if (not isinstance(attempt, dict) or not isinstance(attempt.get("status"), str)
                or attempt["status"] not in statuses or not isinstance(attempt.get("updated_at"), str)
                or not _count(attempt.get("completed"), maximum=total)
                or not _count(attempt.get("failed_frames"), maximum=total)
                or attempt["completed"] + attempt["failed_frames"] > total
                or attempt.get("error") is not None and not isinstance(attempt["error"], str)
                or not isinstance(attempt.get("failures"), list)
                or len(attempt["failures"]) != attempt["failed_frames"]):
            raise ValueError("角色确认跨集采样重试审计无效")
        failures = set()
        for failure in attempt["failures"]:
            if not isinstance(failure, dict):
                raise ValueError("角色确认跨集采样重试审计无效")
            key = safe_id(failure.get("asset_id")), safe_id(failure.get("frame_id"))
            if (key not in planned or key in failures
                    or failure.get("error") is not None and not isinstance(failure["error"], str)):
                raise ValueError("角色确认跨集采样重试审计无效")
            failures.add(key)


def _candidate(library, directory, registry, key, row):
    asset_id, region_id = safe_id(row.get("source_asset_id")), safe_id(row.get("region_id"))
    if key != digest(["candidate", row["character_id"], asset_id, region_id]):
        raise ValueError("角色确认候选键与来源不一致")
    asset = library.get_asset(asset_id, include_deleted=True)
    if row.get("source_fingerprint") != asset["fingerprint"]:
        raise ValueError("角色确认候选的原片指纹不一致")
    frame_id = safe_id(row.get("frame_id"))
    library.check_frame_entries(asset_id, [frame_id])
    box = checked_box(row.get("box"))
    region = read_json(library.asset_dir(asset_id) / "identity" / "visual" / "regions" / f"{region_id}.json")
    if (box is None or not isinstance(region, dict) or region.get("id") != region_id
            or region.get("frame_id") != frame_id or region.get("box") != box):
        raise ValueError("角色确认候选缺少对应的原始人物区域")
    reference_id = row.get("reference_id")
    if row["action"] == "confirm" and not reference_id:
        raise ValueError("角色确认记录缺少参考图")
    if reference_id is None:
        return
    safe_id(reference_id)
    gallery = read_json(directory.parent / "reference-images" / "gallery.json", {"references": []})
    reference = next((item for item in gallery["references"] if item["id"] == reference_id), None)
    if reference is None:
        raise ValueError("角色确认引用的参考图不存在")
    source = reference.get("source", {})
    # A later merge or deletion changes the current projection, never the saved
    # source identities. Allow those histories while keeping evidence exact.
    character = _identity(registry, row["character_id"])
    reference_character = _identity(registry, reference.get("character_id"))
    same_character = (reference["character_id"] == row["character_id"]
                      or character is not None and reference_character == character)
    if (not same_character or source.get("kind") != "frame_crop"
            or source.get("asset_id") != asset_id or source.get("frame_id") != frame_id
            or source.get("box") != box or source.get("region_id") != region_id
            or source.get("source_fingerprint") != row["source_fingerprint"]
            or source.get("confirmation_id") != key):
        raise ValueError("角色确认参考图与原片人物来源不一致")


def validate_character_confirmation_artifacts(library):
    """Historical decisions may outlive current membership, never their evidence."""
    for directory in (library.root / "works").glob("*/confirmation"):
        scope = safe_id(directory.parent.name)
        registry = CharacterStore(library)._read(scope)
        state = read_json(directory / "state.json")
        if state is not None:
            if (not isinstance(state, dict) or state.get("scope_id") != scope
                    or type(state.get("revision")) is not int or state["revision"] < 0
                    or not isinstance(state.get("decisions"), dict) or not isinstance(state.get("audit"), list)
                    or any(not isinstance(item, dict) for item in state["audit"])):
                raise ValueError("角色确认备份格式无效")
            for key, row in state["decisions"].items():
                safe_id(key)
                if (not isinstance(row, dict) or not isinstance(row.get("updated_at"), str)
                        or not isinstance(row.get("action"), str)
                        or row.get("action") not in {"skip", "resume", "confirming", "confirm", "reject", "unclear"}):
                    raise ValueError("角色确认决定格式无效")
                if ("prior_review_version" in row and (
                        type(row["prior_review_version"]) is not int or row["prior_review_version"] < 0)
                        or "reason" in row and not isinstance(row["reason"], str)):
                    raise ValueError("角色确认修订信息无效")
                _identity(registry, row.get("character_id"))
                if row["action"] in {"skip", "resume"}:
                    if key != digest(["role", row["character_id"]]):
                        raise ValueError("角色确认状态键与角色不一致")
                else:
                    _candidate(library, directory, registry, key, row)
        sample = read_json(directory / "sampling.json")
        if sample is None:
            continue
        if isinstance(sample, dict) and sample.get("sampling_version") == "2.0":
            validate_sampling_v2(library, scope, sample)
            continue
        if (not isinstance(sample, dict) or sample.get("scope_id") != scope
                or sample.get("schema_version", "1.0") != "1.0"
                or sample.get("sampling_version", "1.0") != "1.0"
                or not isinstance(sample.get("status"), str)
                or sample.get("status") not in {"running", "completed", "failed", "unavailable", "interrupted"}
                or not isinstance(sample.get("updated_at"), str)
                or type(sample.get("total")) is not int or sample["total"] < 0
                or type(sample.get("completed")) is not int or not 0 <= sample["completed"] <= sample["total"]
                or not isinstance(sample.get("requested_times"), list)
                or not isinstance(sample.get("frame_ids"), list)
                or any(not isinstance(value, str) for value in sample["frame_ids"])):
            raise ValueError("角色确认采样备份格式无效")
        safe_id(sample.get("id"))
        if "base_id" in sample:
            safe_id(sample["base_id"])
            if (type(sample.get("round")) is not int or sample["round"] < 0
                    or sample["id"] != digest([sample["base_id"], sample["round"], sample["requested_times"]])):
                raise ValueError("角色确认采样轮次无效")
        asset_id = safe_id(sample.get("asset_id"))
        asset = library.get_asset(asset_id, include_deleted=True)
        if (sample.get("source_fingerprint") != asset["fingerprint"]
                or len(sample["requested_times"]) != sample["total"] or sample["total"] > 24
                or any(type(value) is not int or not 0 <= value < asset["duration_ms"]
                       for value in sample["requested_times"])
                or len(sample["frame_ids"]) != sample["completed"]
                or len(set(sample["frame_ids"])) != len(sample["frame_ids"])):
            raise ValueError("角色确认采样来源或时间范围无效")
        for frame_id in sample["frame_ids"]:
            # Reconstruct paths from owned IDs; saved paths never authorize reads.
            safe_id(frame_id)
            library.check_frame_entries(asset_id, [frame_id])
        if "error" in sample and not isinstance(sample["error"], str):
            raise ValueError("角色确认采样错误记录无效")
