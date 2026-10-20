"""Offline, idempotent cast sharing for explicitly bound animation episodes.

Only an exact Bangumi subject and season establish an unclassified work. Names
are never used to merge identities. Old registries, image files and observations
remain intact; membership bindings and redirects provide current projections.
"""
from __future__ import annotations

import copy
import hashlib

from .library import digest, now, safe_id


def shared_scope(member):
    """Project casts retain their established series-wide compatibility scope."""
    profile = member.get("anime_profile") or {}
    subject = profile.get("dossier", {}).get("subject", {}).get("id")
    if (member.get("project_id") is not None or member.get("episode") is None
            or profile.get("mode") != "bangumi" or type(subject) is not int or subject <= 0):
        return None
    return "cast_anime_" + digest(["bangumi", subject, member.get("season"), "episodes"])


def copy_gallery(library, source_scope, target_scope, mapping, target=None):
    """Stage immutable images first; publish the after-image through the journal.

    Keep reference IDs: existing region evidence names them. Already copied rows
    are authoritative, including subsequent human confirmation or revocation.
    """
    from .visual_identity import ReferenceGallery, _write_bytes

    gallery = ReferenceGallery(library)
    target = copy.deepcopy(target if target is not None else gallery._read(target_scope))
    previous = copy.deepcopy(target)
    source = gallery._read(source_scope)
    known = {row["id"]: row for row in target["references"]}
    for original in source["references"]:
        identity = mapping.get(original["character_id"])
        if identity is None:
            continue
        identifier = safe_id(original["id"])
        if identifier in known:
            if known[identifier]["sha256"] != original["sha256"]:
                raise ValueError("共享参考图 ID 冲突，原图库保持不变")
            continue
        content = (gallery._directory(source_scope) / f"{identifier}.png").read_bytes()
        if hashlib.sha256(content).hexdigest() != original["sha256"]:
            raise ValueError("共享参考图内容与已保存指纹不一致")
        path = gallery._directory(target_scope) / f"{identifier}.png"
        if path.exists() and path.read_bytes() != content:
            raise ValueError("共享参考图文件冲突，原图库保持不变")
        if not path.exists():
            _write_bytes(path, content)
        row = copy.deepcopy(original)
        row["character_id"] = identity
        row.setdefault("shared_from", {"scope_id": source_scope, "reference_id": identifier})
        target["references"].append(row)
        known[identifier] = row
    if target != previous:
        target["revision"] += 1
        target["audit"].append({"at": now(), "action": "share_cast", "source_scope_id": source_scope,
                                "source_revision": source["revision"], "source_audit": copy.deepcopy(source["audit"])})
    return target


def _merge_registry(store, target, source_scope, source):
    """External IDs and explicit provenance are authority; names are display."""
    from .characters import EDITABLE_FIELDS, external_sources, unique

    mapping = target.setdefault("imports", {}).setdefault(source_scope, {})
    external = target.setdefault("external_bindings", {}).setdefault("bangumi", {})
    audit = target.setdefault("sharing_audit", [])
    for source_id, profile in source["profiles"].items():
        if source_id in mapping:
            continue
        sources = external_sources(profile)
        keys = [str(row["character_id"]) for row in sources]
        known = next((key for key in keys if key in external), None)
        target_id = store._resolve(target, external[known], strict=True) if known is not None else None
        if known is None:
            target_id = "character_" + digest([target["scope_id"], source_scope, source_id])
            cloned = copy.deepcopy(profile)
            cloned.update(id=target_id, representative=None,
                          imported_from={"scope_id": source_scope, "character_id": source_id})
            target["profiles"][target_id] = cloned
        if target_id is not None:
            current = target["profiles"][target_id]
            old = copy.deepcopy(current)
            # Retain the latest explicit edits; archive conflicting before-images.
            current_time = current.get("shared_edit_at", current["updated_at"])
            if profile.get("user_edited") and (not current.get("user_edited") or profile["updated_at"] >= current_time):
                for field in EDITABLE_FIELDS.intersection(profile.get("edited_fields", [])):
                    current[field] = copy.deepcopy(profile[field])
                current["edited_fields"] = sorted(set(current["edited_fields"]) | set(profile["edited_fields"]))
                current["user_edited"] = True
                current["shared_edit_at"] = profile["updated_at"]
            current["aliases"] = unique([*current["aliases"], profile["name"], *profile["aliases"]])
            current["aliases"] = [value for value in current["aliases"] if value != current["name"]]
            origins = current.setdefault("identity_origins", [])
            for origin in [{"scope_id": source_scope, "character_id": source_id}, *profile.get("identity_origins", [])]:
                if origin not in origins:
                    origins.append(copy.deepcopy(origin))
            for key in keys:
                other_id = store._resolve(target, external.get(key))
                if other_id is not None and other_id != target_id:
                    other = target["profiles"].pop(other_id)
                    current["aliases"] = unique([*current["aliases"], other["name"], *other["aliases"]])
                    target["redirects"][other_id] = target_id
                external[key] = target_id
            if old != current:
                audit.append({"source_scope_id": source_scope, "source_character_id": source_id,
                              "target_character_id": target_id, "before": old, "source": copy.deepcopy(profile)})
        mapping[source_id] = target_id
    # Preserve explicit merges and deletions rather than reseeding deleted roles.
    for key, original_id in source.get("external_bindings", {}).get("bangumi", {}).items():
        if store._resolve(source, original_id, strict=True) is None:
            existing = store._resolve(target, external.get(key))
            if existing is not None:
                removed = target["profiles"].pop(existing)
                target["redirects"][existing] = None
                audit.append({"source_scope_id": source_scope, "action": "preserve_deletion", "before": removed})
            external[key] = None
    for source_id in source.get("redirects", {}):
        resolved = store._resolve(source, source_id, strict=True)
        mapping[source_id] = store._resolve(target, mapping.get(resolved))
    for source_id, target_id in list(mapping.items()):
        mapping[source_id] = store._resolve(target, target_id)
        if source_id not in target["profiles"]:
            target["redirects"][source_id] = mapping[source_id]
    for historical_scope, historical_map in source.get("imports", {}).items():
        destination = target["imports"].setdefault(historical_scope, {})
        for identity, source_id in historical_map.items():
            destination.setdefault(identity, mapping.get(store._resolve(source, source_id)))
    return mapping


def prepare_shared_cast(library, catalog):
    """Return journal writes, updating the supplied catalog without publishing it."""
    from .anime_profiles import merge_dossier
    from .characters import CharacterStore

    groups = {}
    for asset_id, member in catalog["assets"].items():
        scope = shared_scope(member)
        if scope and not member.get("trash_id"):
            groups.setdefault(scope, []).append((asset_id, member))
    store = CharacterStore(library)
    registries, writes, migrated = {}, {}, []
    for target_scope, members in groups.items():
        pending = [(identity, member) for identity, member in members if member["character_scope_id"] != target_scope]
        if not pending:
            continue
        dossier = members[0][1]["anime_profile"]["dossier"]
        for _, member in members[1:]:
            dossier = merge_dossier(dossier, member["anime_profile"]["dossier"])
        target = store.seed_registry(store._read(target_scope), dossier)
        galleries = None
        source_maps = {}
        for _, member in sorted(pending):
            source_scope = member["character_scope_id"]
            if source_scope not in source_maps:
                source_maps[source_scope] = _merge_registry(store, target, source_scope, store._read(source_scope))
                galleries = copy_gallery(library, source_scope, target_scope, source_maps[source_scope], galleries)
        target["revision"] += 1
        target["semantic_revision"] = target.get("semantic_revision", target["revision"] - 1) + 1
        store._validate_structure(target, target_scope)
        registries[target_scope] = target
        if galleries is not None:
            writes[f"works/{target_scope}/reference-images/gallery.json"] = galleries
        for asset_id, member in pending:
            old = member["character_scope_id"]
            mapping = source_maps[old]
            bindings = {scope: {identity: mapping.get(store._resolve(store._read(old), value))
                               for identity, value in values.items()}
                        for scope, values in member["bindings"].items()}
            bindings[old] = copy.deepcopy(mapping)
            member.update(character_scope_id=target_scope, bindings=bindings,
                          membership_revision=member["membership_revision"] + 1,
                          previous_scopes=list(dict.fromkeys([*member.get("previous_scopes", []), old])),
                          shared_cast_version=1)
            migrated.append(asset_id)
    return registries, writes, migrated
