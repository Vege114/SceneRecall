"""Rebuildable ordinary person crops, separate from permanent reference images.

No reconstructed image is cached on disk. Models which require a filename use
an operation-scoped temporary file; its lifetime ends even when inference fails.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import stat
import tempfile

from PIL import Image, ImageOps

from .characters import checked_box
from .library import confined_path, read_json, safe_id


MAX_CROP_SOURCE_BYTES = 20 * 1024 * 1024


def crop_retention(value="default"):
    """Resolve the app stage once per operation; an explicit asset policy wins."""
    if not isinstance(value, str) or value not in {"default", "retain", "discard"}:
        raise ValueError("crop_retention must be default, retain or discard")
    if value != "default":
        return value
    override = os.environ.get("SCENERECALL_CROP_RETENTION", "").strip().lower()
    if override:
        if override not in {"retain", "discard"}:
            raise ValueError("SCENERECALL_CROP_RETENTION must be retain or discard")
        return override
    stage = os.environ.get("SCENERECALL_STAGE", "development").strip().lower()
    if stage not in {"development", "test", "production"}:
        raise ValueError("SCENERECALL_STAGE must be development, test or production")
    return "discard" if stage == "production" else "retain"


def configured_crop_retention(config):
    """Use a task's frozen effective default across pause/restart boundaries."""
    value = config.get("crop_retention", "default")
    if value == "default" and "crop_retention_effective" in config:
        value = config["crop_retention_effective"]
        if not isinstance(value, str) or value not in {"retain", "discard"}:
            raise ValueError("Invalid frozen crop retention policy")
    return crop_retention(value)


def crop_path(library, asset_id, region_id):
    # Confinement alone permits links to a different artifact *inside* visual/.
    # Cleanup must address this exact ordinary filename, never a resolved alias.
    asset = library.asset_dir(asset_id).resolve()
    identity = asset / "identity"
    visual = identity / "visual"
    crops = visual / "crops"
    candidate = crops / f"{safe_id(region_id)}.png"
    for component in (identity, visual, crops, candidate):
        try:
            attributes = component.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(attributes.st_mode)
                or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("Person crop path contains a filesystem alias")
    resolved = confined_path(visual, f"crops/{candidate.name}")
    if resolved != candidate:
        raise ValueError("Person crop path is not its canonical ordinary filename")
    return candidate


def _source(library, asset_id, row, registry=None, asset=None, *, require_file=True):
    safe_id(row["id"])
    frame_id = safe_id(row["frame_id"])
    if checked_box(row.get("box")) is None:
        raise ValueError("Person crop has no valid source box")
    asset = library.get_asset(asset_id, include_deleted=True) if asset is None else asset
    if row.get("source_fingerprint", asset["fingerprint"]) != asset["fingerprint"]:
        raise ValueError("Person crop source fingerprint changed")
    if registry is None:
        registry = read_json(library.asset_dir(asset_id) / "frames" / "registry.json", {})
    entry = registry.get(frame_id)
    if not entry or entry.get("at_ms") != row.get("at_ms"):
        raise ValueError("Person crop source timestamp changed")
    path = confined_path(library.asset_dir(asset_id), entry["path"])
    if require_file and not path.is_file():
        raise FileNotFoundError("Person crop source frame is unavailable")
    return path


def crop_available(library, asset_id, row, registry=None, asset=None):
    """Metadata-only queue check; it never decodes or creates a crop."""
    # Resolve the owned crop path even when missing, rejecting escaped symlinks.
    retained = crop_path(library, asset_id, row["id"])
    try:
        _source(library, asset_id, row, registry, asset, require_file=not retained.is_file())
        return True
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return False


def rebuild_crop_bytes(library, asset_id, row, *, registry=None):
    """Encode the exact original detection bounds, checking durable provenance."""
    path = _source(library, asset_id, row, registry)
    if path.stat().st_size > MAX_CROP_SOURCE_BYTES:
        raise ValueError("Person crop source frame exceeds 20 MiB")
    return rebuild_crop_content(path.read_bytes(), row)


def rebuild_crop_content(content, row):
    """Crop bounded source bytes after the caller verifies their location.

    Both local preview/confirmation and provider transport use these exact
    pixels, including EXIF orientation and original detector rounding.
    """
    if len(content) > MAX_CROP_SOURCE_BYTES:
        raise ValueError("Person crop source frame exceeds 20 MiB")
    expected_frame = row.get("source_frame_sha256")
    if expected_frame and hashlib.sha256(content).hexdigest() != expected_frame:
        raise ValueError("Person crop source frame pixels changed")
    with Image.open(io.BytesIO(content)) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError("Person crop source frame is too large")
        picture = ImageOps.exif_transpose(source).convert("RGB")
    box = checked_box(row.get("box"))
    if box is None:
        raise ValueError("Person crop has no valid source box")
    x, y, width, height = box
    coordinates = [x * picture.width, y * picture.height,
                   (x + width) * picture.width, (y + height) * picture.height]
    bounds = row.get("crop_box_pixels", [round(value) for value in coordinates])
    if (not isinstance(bounds, list) or len(bounds) != 4
            or any(type(value) is not int for value in bounds)
            # Normalization can move an exact .5 by floating-point epsilon.
            # Preserve the original detector's rounded pixels in that case.
            or any(abs(pixel - value) > .5000001 for pixel, value in zip(bounds, coordinates))
            or not 0 <= bounds[0] < bounds[2] <= picture.width
            or not 0 <= bounds[1] < bounds[3] <= picture.height):
        raise ValueError("Person crop source bounds changed")
    output = io.BytesIO()
    picture.crop(tuple(bounds)).save(output, format="PNG")
    encoded = output.getvalue()
    if row.get("crop_sha256") and hashlib.sha256(encoded).hexdigest() != row["crop_sha256"]:
        raise ValueError("Reconstructed person crop pixels changed")
    return encoded


def crop_bytes(library, asset_id, row):
    """Read an owned crop or reconstruct it without changing persisted state."""
    retained = crop_path(library, asset_id, row["id"])
    _source(library, asset_id, row, require_file=not retained.is_file())
    if retained.is_file():
        try:
            content = retained.read_bytes()
        except FileNotFoundError:
            # A concurrent finished worker may have discarded the copy after
            # the existence check; the immutable source remains sufficient.
            return rebuild_crop_bytes(library, asset_id, row)
        if row.get("crop_sha256") and hashlib.sha256(content).hexdigest() != row["crop_sha256"]:
            raise ValueError("Stored person crop pixels changed")
        return content
    return rebuild_crop_bytes(library, asset_id, row)


@contextmanager
def temporary_crop(content):
    """Windows-compatible filename for a single worker call, always removed."""
    with tempfile.TemporaryDirectory(prefix="scenerecall-person-crop-") as directory:
        path = Path(directory) / "crop.png"
        path.write_bytes(content)
        yield path
