"""Local character evidence, independent of generative observations and image embeddings.

Each detected region is its own local entity. We deliberately do not infer a
cross-frame person track from nearest appearance or a matching candidate name.
The existing binding resolver alone decides whether evidence can acquire a name.
"""
from __future__ import annotations

import copy

from .identity_bindings import POLICY_VERSION
from .library import digest

VERSION = "retrieval-local-identity-v1"


def gallery_identity(pipeline, asset_id):
    gallery = pipeline.gallery.list(asset_id)
    return {"scope_id": gallery["scope_id"], "revision": gallery["revision"],
            "references": [{key: reference.get(key) for key in
                ("id", "character_id", "version", "sha256", "status", "quality")}
                for reference in gallery["references"]]}


def observation(window, visual, *, run_id, signature):
    """Attach exact current regions, retaining candidates without claiming identity."""
    local = set(window["local_frame_ids"])
    regions = [row for row in visual.get("regions", []) if row["frame_id"] in local]
    entities = [{"id": row["id"], "kind": "person", "is_character": True,
                 "description": "本地检测到的人物区域", "character_id": None,
                 "identity_region_id": row["id"], "identity_support_region_ids": [],
                 "evidence_frame_ids": [row["frame_id"]], "box": copy.deepcopy(row["box"]),
                 "source": VERSION} for row in regions]
    result = {"identity_observation_version": VERSION, "asset_id": window["asset_id"],
              "unit_id": window["id"], "source_fingerprint": window["source_fingerprint"],
              "window_digest": digest(window), "identity_policy": POLICY_VERSION,
              "identity_status": visual["status"], "entities": entities,
              "evidence_frame_ids": list(window["evidence_frame_ids"]),
              "provenance": {"run_id": run_id, "input_signature": signature,
                             "evidence_kind": "current_local_person_regions"}}
    result["id"] = "rio_" + digest(result)
    return result


def merge(record, identity):
    """Preserve ordered VLM events and explicit entity ownership when adding regions."""
    result = copy.deepcopy(record)
    entities = result.setdefault("entities", [])
    claimed = {region for entity in entities for region in
               [entity.get("identity_region_id"), *(entity.get("identity_support_region_ids") or [])] if region}
    entities.extend(copy.deepcopy(entity) for entity in identity["entities"]
                    if entity["identity_region_id"] not in claimed)
    result.update(identity_policy=POLICY_VERSION, local_identity={
        "id": identity["id"], "version": VERSION, "status": identity["identity_status"],
        "input_signature": identity["provenance"]["input_signature"]})
    return result
