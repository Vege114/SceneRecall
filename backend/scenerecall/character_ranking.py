"""Frame-local character support for ranking; absence of detections stays unknown."""
from __future__ import annotations

BRANCHES = ("lexical", "semantic", "visual", "concept")
CANDIDATE_WEIGHT = .5


def _target_keys(targets):
    return {(row.get("scope_id"), row.get("character_id")) for row in targets
            if row.get("scope_id") and row.get("character_id")}


def character_frame_support(record, targets):
    """Map each real frame to distinct target identities and their binding strength.

    Region evidence is projected by the authoritative binding resolver. Older
    local detector rows have a safe one-region/one-frame fallback; broader VLM
    evidence frame lists never establish an identity on every listed frame.
    """
    wanted = _target_keys(targets)
    allowed_frames = set(record.get("evidence_frame_ids") or [])
    if record.get("ordered_frames"):
        allowed_frames.intersection_update(row.get("id") for row in record["ordered_frames"])
    regions = {}
    for entity in record.get("entities") or []:
        if entity.get("kind", "person") not in {"person", "character", "anthropomorphic_character"} or entity.get("is_character") is False:
            continue
        binding = entity.get("identity_binding") or {}
        if binding.get("policy_version") != "identity-binding-v2" or not binding.get("id"):
            continue
        scope = binding.get("character_scope_id")
        status = binding.get("status")
        if status in {"human_confirmed", "auto_bound"}:
            identity = binding.get("character_id")
            if not identity or entity.get("character_id") != identity:
                continue
            strength = 1.
        elif status == "candidate":
            identity = binding.get("candidate_character_id")
            if not identity or entity.get("candidate_character_id") != identity or entity.get("character_id"):
                continue
            strength = CANDIDATE_WEIGHT
        else:
            continue
        target = (scope, identity)
        declared = {key for key in [entity.get("identity_region_id"), *(entity.get("identity_support_region_ids") or [])] if key}
        evidence = entity.get("identity_region_evidence")
        if evidence is None and entity.get("source") == "retrieval-local-identity-v1":
            frames = entity.get("evidence_frame_ids") or []
            if len(frames) == 1 and entity.get("identity_region_id") and not entity.get("identity_support_region_ids"):
                evidence = [{"region_id": entity["identity_region_id"], "frame_id": frames[0]}]
        for row in evidence or []:
            frame, region = row.get("frame_id"), row.get("region_id")
            if frame not in allowed_frames or frame not in (entity.get("evidence_frame_ids") or []) or region not in declared:
                continue
            # Include out-of-scope identities in conflict detection: the same
            # region cannot count twice, nor become trusted by dropping a rival.
            claims = regions.setdefault((frame, region), {})
            claims[target] = max(claims.get(target, 0), strength)
    result = {}
    for (frame, _region), claims in regions.items():
        if len(claims) != 1:
            continue
        target, strength = next(iter(claims.items()))
        if target in wanted:
            support = result.setdefault(frame, {})
            support[target] = max(support.get(target, 0), strength)
    return result


def frame_support_priority(support):
    """Prefer supported target identities, then the number of distinct targets."""
    return sum(support.values()), len(support)


def character_evidence_frame(record, targets):
    """Locate an actual identity-evidence frame when no visual hit exists."""
    support = character_frame_support(record, targets)
    if not support:
        return None
    times = {row.get("id"): row.get("at_ms", record.get("start_ms", 0))
             for row in record.get("ordered_frames") or []}
    return min(support, key=lambda frame: (-sum(support[frame].values()), -len(support[frame]),
                                          times.get(frame, record.get("start_ms", 0)), frame))


def summarize_character_match(record, targets, frame_id):
    required = len(_target_keys(targets))
    support = character_frame_support(record, targets).get(frame_id, {})
    confirmed = sum(value == 1 for value in support.values())
    candidate = sum(value == CANDIDATE_WEIGHT for value in support.values())
    if required and confirmed == required:
        status, reason = "confirmed", "命中帧有全部指定人物的有效身份依据；动作与位置仍需核对"
    elif required and confirmed + candidate == required:
        status, reason = "candidate", "命中帧覆盖全部指定人物候选；候选身份尚未确认"
    elif confirmed + candidate:
        status, reason = "partial", "命中帧仅支持部分指定人物；其余人物可能漏检或尚无身份依据"
    else:
        status, reason = "unverified", "命中帧的指定人物尚未证实（可能漏检）"
    return {"status": status, "required_count": required, "confirmed_count": confirmed,
            "candidate_count": candidate, **({"frame_id": frame_id} if frame_id else {}), "reason": reason}


def grounded_rrf_score(record, targets, contributions, frame_id):
    raw = sum(float(contributions.get(branch, 0)) for branch in BRANCHES)
    if not targets or record.get("kind") not in {"visual", "scene"}:
        return raw
    match = summarize_character_match(record, targets, frame_id)
    count = match["required_count"]
    coverage = ((match["confirmed_count"] + CANDIDATE_WEIGHT * match["candidate_count"]) / count) if count else 0.
    return sum(float(contributions.get(branch, 0)) *
               (1. if branch == "visual" else coverage)
               for branch in BRANCHES)
