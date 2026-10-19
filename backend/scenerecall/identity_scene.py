"""Join local identity evidence to scene inputs without promoting candidates.

No tracking is inferred here: a region belongs to exactly one original frame.
The raw descriptions remain immutable; display/search resolve participant IDs.
"""
from __future__ import annotations

import copy


def project_audio_evidence(evidence, current_records):
    """Resolve only association fields; keep original timing and evidence intact."""
    current = {row["id"]: row for row in current_records}
    fields = ("character_id", "character_name", "character_aliases", "candidate_character_id",
              "association_status", "association_source", "candidates", "identity_revision")
    result = copy.deepcopy(evidence)
    for row in result:
        latest = current.get(row["id"])
        if latest is None:
            row.update(character_id=None, character_name=None, character_aliases=[],
                       candidate_character_id=None, candidates=[], association_status="unknown",
                       association_source="evidence_unavailable")
        else:
            row.update({key: copy.deepcopy(latest.get(key)) for key in fields})
    return result


def identity_context(context, result, frames, *, top_k=5):
    context = copy.deepcopy(context)
    if not result or result.get("status") != "completed":
        return context
    current = {frame["id"] for frame in frames}
    regions = [row for row in result.get("regions", []) if row["frame_id"] in current]
    ranked = {}
    for region in regions:
        match = region.get("match", {})
        if match.get("status") not in {"candidate", "confirmed"}:
            continue
        for item in match.get("candidates", [])[:top_k]:
            identity = item["character_id"]
            ranked[identity] = min(ranked.get(identity, float("inf")), item["distance"])
        if match.get("character_id"):
            ranked[match["character_id"]] = -1
    # Top-K is per observed person; keep their union for multi-person batches.
    selected = sorted(ranked, key=lambda key: (ranked[key], key))
    complete = {p["id"]: p for p in [*result.get("selected_profiles", []), *context.get("profiles", [])]}
    selected = [key for key in selected if key in complete]
    if selected:
        context["profiles"] = [copy.deepcopy(complete[key]) for key in selected]
        context["retrieval"] = {"method": "complete_allowed_gallery_then_per_region_top_k", "top_k": top_k}
    else:
        # Failure to retrieve a visual match is not evidence that an allowed
        # Bangumi dossier is irrelevant. Keep the supplied roster/reference
        # context as priors while unknown regions still prevent identity claims.
        context["retrieval"] = {"method": "allowed_roster_without_visual_match", "top_k": top_k,
                                "identity_status": "unverified_prior", "reference_only": True,
                                "presence_verified": False}
    context["identity_regions"] = [{key: copy.deepcopy(row[key]) for key in
        ("id", "entity_id", "local_id", "frame_id", "at_ms", "box", "quality", "match") if key in row} for row in regions]
    # A bounded number of clear current crops supplements the partition sheets;
    # every region still has an original frame/box for audit.
    context["region_images"] = [{"id": row.get("id", row.get("entity_id")), "frame_id": row["frame_id"],
        "local_id": row["local_id"], "path": row["crop_path"], "at_ms": row["at_ms"]}
        for row in regions if row.get("crop_path")][:8]
    by_character = {}
    for reference in result.get("references", []):
        if reference["character_id"] in selected:
            by_character.setdefault(reference["character_id"], []).append(reference)
    # Round robin includes multiple references when space permits without letting
    # the first character consume the entire image budget.
    if not selected:
        return context
    context["reference_frames"] = []
    for index in range(8):
        for identity in selected:
            rows = by_character.get(identity, [])
            if index < len(rows) and len(context["reference_frames"]) < 8:
                context["reference_frames"].append({**rows[index], "at_ms": 0,
                    "id": "reference:" + rows[index].get("reference_id", identity)})
    return context


def bind_identity_evidence(entities, context):
    """Only a validated current region can carry its stored identity decision."""
    if "identity_regions" not in (context or {}):
        return entities
    regions = {row.get("id", row.get("entity_id")): row for row in context["identity_regions"]}
    result = copy.deepcopy(entities)
    for entity in result:
        if entity.get("kind") == "object" or entity.get("is_character") is False:
            continue
        region = regions.get(entity.get("identity_region_id"))
        if region is None:
            matches = [row for row in regions.values() if row["frame_id"] == entity.get("portrait_frame_id")
                       and row.get("box") == entity.get("portrait_box")]
            region = matches[0] if len(matches) == 1 else None
        if region and region["frame_id"] not in entity["evidence_frame_ids"]:
            raise ValueError("身份区域不属于该人物的当前画面证据")
        match = copy.deepcopy(region.get("match", {})) if region else {}
        status = match.get("status", "unknown")
        entity.update(identity_region_id=region.get("id", region.get("entity_id")) if region else None,
                      association_status=status, association_source=match.get("source", "unresolved_region"),
                      candidate_character_id=match.get("candidate_character_id"), identity_match=match,
                      character_id=match.get("character_id") if status == "confirmed" else None,
                      name=None)
        # Feature distance is never converted to a probability. Legacy confidence
        # belongs only to the language model and is cleared for local decisions.
        entity["match_confidence"] = None
        if region:
            entity.update(portrait_frame_id=region["frame_id"], portrait_box=region["box"],
                          local_id=region["local_id"])
    return result


def project_actions(record):
    """Resolve names by ID at read time; never replace words in source prose."""
    value = copy.deepcopy(record)
    entities = {entity["id"]: entity for entity in value.get("entities", [])}
    actions = []
    for event in value.get("events", []):
        for side in ("actor", "target"):
            participants = [entities[key] for key in event.get(side + "_ids", []) if key in entities]
            event[side + "_character_ids"] = [p["character_id"] for p in participants if p.get("character_id")]
            event[side + "_names"] = [(p.get("character_name") or p.get("name")) if p.get("character_id")
                                     else p.get("local_id", p["id"]) for p in participants]
        actions.append(" ".join([", ".join(event["actor_names"]), event.get("action", ""),
                                 ", ".join(event["target_names"])]).strip())
    value["resolved_actions"] = actions
    return value
