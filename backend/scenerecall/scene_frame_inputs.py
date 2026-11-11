"""Durable frame references for scene requests whose filenames are temporary leases."""
from collections import defaultdict

from .library import safe_id


def _map_paths(value, transform):
    if isinstance(value, (list, tuple)):
        return [_map_paths(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: transform(item) if key == "path" else _map_paths(item, transform)
                for key, item in value.items()}
    return value


def freeze_frame_paths(args):
    def freeze(path):
        reference = getattr(path, "frame_ref", None)
        if reference:
            asset_id, identity = reference
            return {"sample_asset_id": asset_id, "sample_frame_id": identity}
        return path
    return _map_paths(args, freeze)


def resolve_frame_paths(library, args):
    # Paid request keys already exclude the "path" field. Resolve after checking
    # receipts so replaying an existing result never decodes another picture.
    groups = defaultdict(set)

    def visit(path):
        if isinstance(path, dict) and set(path) == {"sample_asset_id", "sample_frame_id"}:
            groups[safe_id(path["sample_asset_id"])].add(safe_id(path["sample_frame_id"]))
        return path

    _map_paths(args, visit)
    if not groups:
        return args
    paths = {asset: library.frame_paths(asset, sorted(identities)) for asset, identities in groups.items()}

    def resolve(path):
        if isinstance(path, dict) and set(path) == {"sample_asset_id", "sample_frame_id"}:
            return str(paths[path["sample_asset_id"]][path["sample_frame_id"]])
        return path
    return _map_paths(args, resolve)
