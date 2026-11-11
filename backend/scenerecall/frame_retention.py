"""Locate explicit human evidence and frozen filename dependencies before eviction."""
from collections import defaultdict
from pathlib import PureWindowsPath

from .library import read_json, read_jsonl, safe_id


def protected_sample_frames(library, asset_id):
    """Map registered frame IDs to reasons for preserving independent evidence.

    Reads only persisted metadata. Callers synchronize publication with review
    and annotation writes, and handle multiple registrations of the same file.
    """
    asset = library.asset_dir(asset_id)
    registry = read_json(asset / "frames" / "registry.json", {})
    filenames = defaultdict(list)
    for frame_id, row in registry.items():
        filenames[PureWindowsPath(row["path"]).name.casefold()].append(frame_id)
    reasons = defaultdict(set)

    def collect(data, reason, *, paths_only=False):
        pending = [data]
        while pending:
            item = pending.pop()
            if isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, dict):
                pending.extend(item.values())
                if not paths_only:
                    pending.extend(item.keys())
            elif isinstance(item, str):
                if not paths_only and item in registry:
                    reasons[item].add(reason)
                if item.lower().endswith((".jpeg", ".jpg", ".png")):
                    for identifier in filenames.get(PureWindowsPath(item).name.casefold(), ()):
                        reasons[identifier].add(reason)

    for pattern, reason in (
        ("*/confirmation/*.json", "human_confirmation"),
        ("*/reference-images/gallery.json", "reference_gallery"),
        ("*/characters.json", "character_evidence"),
        ("entities/*/catalog.json", "work_entity_evidence"),
    ):
        for path in (library.root / "works").glob(pattern):
            collect(read_json(path), reason)
    for path in (library.root / "collections").glob("*.json"):
        collect(read_json(path), "collection")

    visual = asset / "identity" / "visual"
    for path in visual.glob("*decisions*.json"):
        value = read_json(path)
        collect(value, "human_identity_review")
        region_ids = set(value.get("decisions", {}))
        region_ids.update(row["region_id"] for row in value.get("audit", [])
                          if isinstance(row, dict) and row.get("region_id"))
        for region_id in region_ids:
            row = read_json(visual / "regions" / f"{safe_id(region_id)}.json", {})
            collect(row.get("frame_id"), "human_identity_review")

    runs = asset / "analyses"
    unresolved = {}
    for path in (asset / "annotations").glob("*.json"):
        annotation = read_json(path)
        bookmark_only = not annotation.get("note") and set(annotation) <= {
            "schema_version", "id", "record_id", "asset_id", "updated_at", "base_run_id",
            "favorite", "preview_frame_id", "preview_at_ms", "note"}
        cover = annotation.get("preview_frame_id")
        if bookmark_only:
            if not annotation.get("favorite"):
                continue
            if cover in registry:
                reasons[cover].add("favorite")
                continue
        identity = safe_id(annotation["record_id"])
        reason = "favorite" if annotation.get("favorite") else "annotation"
        collect(annotation, reason)
        found = False
        for folder in ("observations", "scenes", "retrieval-units", "retrieval-identities"):
            for name in (f"{identity}.json", f"{identity}.deep.json"):
                for path in runs.glob(f"*/{folder}/{name}"):
                    collect(read_json(path), reason)
                    found = True
        if not found:
            unresolved[identity] = reason
    if unresolved:
        # A timeline-only subtitle has no sampled evidence to pin. OCR cues
        # may carry frame IDs, including in historical runs. Look them up once
        # per file instead of expanding the whole episode for every favorite.
        matched_cues = set()
        for path in [asset / "subtitles" / "external.jsonl", *runs.glob("*/subtitles.jsonl")]:
            for cue in read_jsonl(path):
                identity = cue.get("id")
                if identity in unresolved:
                    matched_cues.add(identity)
                    collect(cue, unresolved[identity])
        if unresolved.keys() - matched_cues:
            for frame_id in registry:
                reasons[frame_id].add("unresolved_annotation")

    for folder in ("scene-inputs", "scene-input-history"):
        for path in (library.root / "assets").glob(f"*/analyses/*/{folder}/*.json"):
            collect(read_json(path), "frozen_path", paths_only=True)
    return dict(reasons)
