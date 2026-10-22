"""Validate character confirmation provenance before publishing a restored backup."""
from __future__ import annotations

from .characters import CharacterStore, checked_box
from .library import digest, read_json, safe_id


def _identity(registry, value):
    safe_id(value)
    return CharacterStore._resolve(registry, value, strict=True)


def _candidate(library, directory, registry, key, row):
    asset_id, region_id = safe_id(row.get("source_asset_id")), safe_id(row.get("region_id"))
    if key != digest(["candidate", row["character_id"], asset_id, region_id]):
        raise ValueError("角色确认候选键与来源不一致")
    asset = library.get_asset(asset_id, include_deleted=True)
    if row.get("source_fingerprint") != asset["fingerprint"]:
        raise ValueError("角色确认候选的原片指纹不一致")
    frame_id = safe_id(row.get("frame_id"))
    library.frame_path(asset_id, frame_id)
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
        if (not isinstance(sample, dict) or sample.get("scope_id") != scope
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
            library.frame_path(asset_id, frame_id)
        if "error" in sample and not isinstance(sample["error"], str):
            raise ValueError("角色确认采样错误记录无效")
