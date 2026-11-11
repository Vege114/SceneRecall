"""Retire registered sampled JPEGs after verifying exact reconstruction from their source."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall import sample_frames
from scenerecall.file_io import atomic_replace
from scenerecall.frame_retention import protected_sample_frames
from scenerecall.library import atomic_json, read_json, safe_id
from scenerecall.library_activity import library_activity
from compact_person_crops import check_legacy_services, plain_path


class Archive:
    """Use existing manifests without performing Library startup recovery."""
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        if not read_json(plain_path(self.root, self.root / "library.json")).get("id"):
            raise ValueError("资料库清单无效")
        self.sample_frames = sample_frames.SampleFrameStore(self)

    def asset_dir(self, asset_id):
        return plain_path(self.root, self.root / "assets" / safe_id(asset_id))


def jpeg_hash(path):
    with path.open("rb") as stream:
        if stream.read(3) != b"\xff\xd8\xff":
            raise ValueError(f"文件不是 JPEG：{path.name}")
        stream.seek(0)
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin_copy(asset, source, checksum):
    directory = asset / "preserved-frames"
    directory.mkdir(exist_ok=True)
    plain_path(asset, directory)
    target = directory / source.name
    if target.exists() or target.is_symlink():
        if jpeg_hash(plain_path(asset, target)) != checksum:
            raise ValueError("已留存的人工证据与待整理文件不同")
    else:
        temp = directory / f".sample-{uuid.uuid4().hex}.tmp"
        try:
            with source.open("rb") as origin, temp.open("xb") as dest:
                shutil.copyfileobj(origin, dest)
                dest.flush()
                os.fsync(dest.fileno())
            if jpeg_hash(temp) != checksum:
                raise ValueError("复制人工证据期间源文件发生了变化")
            atomic_replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    return target.relative_to(asset).as_posix()


def candidates(archive, asset_id, manifest, registry):
    asset = archive.asset_dir(asset_id)
    frames = plain_path(asset, asset / "frames")
    pictures = sorted(path for path in frames.iterdir() if path.suffix.casefold() in {".jpg", ".jpeg"}
                      and path.stem.startswith(("kf_", "f_")))
    if not pictures:
        return
    reasons = protected_sample_frames(archive, asset_id)
    paths = defaultdict(list)
    for identity, entry in registry.items():
        relative = entry["path"].replace("\\", "/")
        paths[relative.casefold()].append(identity)
    legacy = [p.stem for p in pictures if p.stem in registry and not registry[p.stem].get("sample")]
    recipes = sample_frames.legacy_recipes(archive, asset_id, legacy, registry, manifest["fingerprint"])
    for path in pictures:
        if not plain_path(asset, path).is_file():
            continue
        identity = safe_id(path.stem)
        entry = registry.get(identity)
        if not entry:
            yield path, [], None, "unregistered"
            continue
        direct = path.relative_to(asset).as_posix().casefold()
        registered = entry["path"].replace("\\", "/").casefold()
        if registered == f"preserved-frames/{path.name}".casefold() and entry.get("pinned_sha256"):
            # A previous verified run may have published the pin just before
            # interruption. Recheck both physical files before removing it.
            yield path, paths[registered], None, "pinned"
            continue
        if registered != direct:
            yield path, [], None, "other_path"
            continue
        ids = paths[direct]
        why = set().union(*(reasons.get(identifier, ()) for identifier in ids))
        if "frozen_path" in why:
            yield path, ids, None, "frozen_path"
        elif len(ids) > 1 or why or entry.get("pinned_sha256"):
            yield path, ids, None, "pinned"
        else:
            recipe = entry.get("sample") or recipes.get(identity)
            yield path, ids, recipe, "sample" if recipe is not None else "unknown_recipe"


def compact_asset(archive, asset_id, mode, batch_size, progress):
    asset = archive.asset_dir(asset_id)
    manifest = read_json(plain_path(asset, asset / "manifest.json"))
    if manifest.get("id") != asset_id:
        raise ValueError("素材与清单不匹配")
    totals = Counter(sample_count=0, sample_bytes=0, pinned_count=0, pinned_bytes=0,
                     verified_count=0, removed_count=0, removed_bytes=0)
    if not (asset / "frames").exists():
        return dict(totals)
    registry_path = asset / "frames" / "registry.json"
    if registry_path.exists() or registry_path.is_symlink():
        plain_path(asset, registry_path)
    registry = read_json(registry_path, {})
    pinned, samples = [], []
    for path, ids, recipe, kind in candidates(archive, asset_id, manifest, registry):
        totals[kind + "_count"] += 1
        totals[kind + "_bytes"] += path.stat().st_size
        if kind == "pinned":
            pinned.append((path, ids))
        elif kind == "sample":
            samples.append((path, ids[0], recipe))
    if mode == "estimate":
        return dict(totals)

    def publish(to_remove):
        if mode != "apply" or not to_remove:
            return
        atomic_json(registry_path, registry)
        for path, checksum in to_remove:
            # The marker and any preserved independent image are durable first.
            # Refuse to unlink a source that changed since verification.
            if jpeg_hash(plain_path(asset, path)) != checksum:
                raise ValueError("归档 JPEG 在核验期间发生了变化")
            size = path.stat().st_size
            path.unlink()
            totals["removed_bytes"] += size
            totals["removed_count"] += 1

    verified_pins = []
    for path, ids in pinned:
        checksum = jpeg_hash(path)
        if any(registry[i].get("pinned_sha256", checksum) != checksum for i in ids):
            raise ValueError("人工证据摘要与留存的原件不同")
        if mode == "apply":
            relative = pin_copy(asset, path, checksum)
            for identity in ids:
                updated = {**registry[identity], "path": relative, "pinned_sha256": checksum}
                updated.pop("sample", None)
                registry[identity] = updated
        verified_pins.append((path, checksum))
    publish(verified_pins)
    if not samples:
        return dict(totals)
    source, metadata = sample_frames.verify_source(archive, asset_id)
    samples.sort(key=lambda item: registry[item[1]]["at_ms"])
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        requests, originals = [], []
        for path, identity, recipe in batch:
            checksum = jpeg_hash(path)
            if recipe.get("jpeg_sha256") not in (None, checksum):
                raise ValueError("留存的采样 JPEG 与已登记的摘要不同")
            entry = {**registry[identity], "sample": {**recipe, "jpeg_sha256": checksum}}
            requests.append((identity, entry))
            originals.append((path, checksum))
        with tempfile.TemporaryDirectory(prefix="scenerecall-verify-jpeg-") as temporary:
            sample_frames.restore_batch(source, metadata, requests, Path(temporary))
        if mode == "apply":
            registry.update(requests)
        totals["verified_count"] += len(originals)
        publish(originals)
        if progress:
            progress({"asset_id": asset_id, "processed": start + len(batch), "total": len(samples)})
    return dict(totals)


def run(root, asset_ids=(), *, mode="estimate", batch_size=512, progress=None):
    if mode not in {"estimate", "verify", "apply"} or not 1 <= batch_size <= 4096:
        raise ValueError("整理模式或批量大小无效")
    archive = Archive(root)
    assets = plain_path(archive.root, archive.root / "assets")
    asset_ids = list(dict.fromkeys(asset_ids)) or sorted(p.name for p in assets.iterdir() if p.is_dir())
    if mode == "apply":
        runtime = plain_path(archive.root, archive.root / "runtime")
        lock = runtime / "library-activity.sqlite"
        if lock.exists() or lock.is_symlink():
            plain_path(archive.root, lock)
    with library_activity(archive.root, maintenance=True) if mode == "apply" else nullcontext():
        if mode == "apply":
            check_legacy_services(archive.root)
        results = {}
        for identity in asset_ids:
            results[identity] = compact_asset(archive, identity, mode, batch_size, progress)
            if progress:
                progress({"asset_id": identity, "completed": results[identity]})
    totals = Counter()
    for values in results.values():
        totals.update(values)
    return {"mode": mode, "assets": results, "totals": dict(totals)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=512)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    local_ffmpeg = Path(__file__).resolve().parents[1] / ".cache/ffmpeg/bin"
    if (local_ffmpeg / "ffmpeg.exe").is_file():
        os.environ["PATH"] = str(local_ffmpeg) + os.pathsep + os.environ.get("PATH", "")
    def report(value):
        print(json.dumps(value, ensure_ascii=False), file=sys.stderr, flush=True)
    result = run(args.library, args.asset, mode="apply" if args.apply else "verify" if args.verify else "estimate",
                 batch_size=args.batch_size, progress=report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
