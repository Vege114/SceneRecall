"""Source-verified reconstruction of registered sampled pictures."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import Future
from fractions import Fraction
import hashlib
from itertools import pairwise
import os
from pathlib import Path, PurePosixPath
import json
import re
import shutil
import tempfile
import threading
from zipfile import ZIP_STORED

from . import keyframes, media
from .library import confined_path, digest, read_json, safe_id, source_snapshot

SHA256 = re.compile(r"[0-9a-f]{64}")
DEFAULT_CACHE_BYTES = 64 * 1024**2


def source_identity(path):
    info = path.stat()
    return str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def is_sample_path(identity, relative):
    path = PurePosixPath(relative.replace("\\", "/"))
    return path.suffix.lower() in {".jpg", ".jpeg"} and path.parts == ("frames", identity + path.suffix)


def _time_base(value):
    if not isinstance(value, list) or len(value) != 2 or any(type(v) is not int or v <= 0 for v in value):
        raise ValueError("采样画面时间基无效")
    return value


def make_recipe(asset_id, fingerprint, frame, jpeg_sha256):
    """Bind an exact timestamp or an explicit uniform sampling request to the archived JPEG bytes."""
    if (not isinstance(fingerprint, str) or not SHA256.fullmatch(fingerprint)
            or jpeg_sha256 is not None and (not isinstance(jpeg_sha256, str) or not SHA256.fullmatch(jpeg_sha256))):
        raise ValueError("采样画面缺少有效指纹")
    if type(frame.get("at_ms")) is not int or frame["at_ms"] < 0:
        raise ValueError("采样画面位置无效")
    common = {"version": 1, "source_fingerprint": fingerprint, "jpeg_sha256": jpeg_sha256}
    if "sample_crop" in frame:
        requested, crop = frame.get("requested_at_ms"), frame["sample_crop"]
        if type(requested) is not int or requested < 0 or frame["id"] != "f_" + digest([asset_id, requested, crop]):
            raise ValueError("采样画面 ID 与请求时间或裁剪参数不匹配")
        media._crop_filter(crop)
        return {**common, "kind": "uniform", "requested_at_ms": requested, "crop": crop}
    if any(type(frame.get(k)) is not int for k in ("pts", "source_pts")):
        raise ValueError("采样画面缺少精确时间戳")
    time_base, source_base = _time_base(frame.get("time_base")), _time_base(frame.get("source_time_base"))
    if (frame["id"] != "kf_" + digest([asset_id, fingerprint, frame["pts"], time_base])
            or round(1000 * frame["pts"] * Fraction(*time_base)) != frame["at_ms"]):
        raise ValueError("采样画面 ID 或位置与计划不符")
    return {**common, "pts": frame["pts"], "time_base": time_base,
            "source_pts": frame["source_pts"], "source_time_base": source_base}


def validate_recipe(asset_id, fingerprint, identity, entry, *, legacy=False, manifest=None):
    recipe = entry["sample"]
    if (not isinstance(recipe, dict) or type(recipe.get("version")) is not int or recipe["version"] != 1
            or recipe.get("source_fingerprint") != fingerprint
            or not legacy and not isinstance(recipe.get("jpeg_sha256"), str)):
        raise ValueError("采样画面配方或原片指纹无效")
    frame = {**recipe, "id": identity, "at_ms": entry["at_ms"]}
    kind = recipe.get("kind", "exact")
    if kind == "uniform":
        if "crop" not in recipe:
            raise ValueError("采样画面配方缺少裁剪设置")
        frame["sample_crop"] = recipe["crop"]
    elif kind != "exact":
        raise ValueError("未知采样画面类型")
    make_recipe(asset_id, fingerprint, frame, recipe.get("jpeg_sha256"))
    if manifest is not None:
        if not 0 <= entry["at_ms"] < manifest["duration_ms"]:
            raise ValueError("采样画面不在原片范围内")
        if kind == "exact" and "start_time_seconds" in manifest:
            origin = frame["source_pts"] * Fraction(*frame["source_time_base"]) - frame["pts"] * Fraction(*frame["time_base"])
            if origin != Fraction(manifest["start_time_seconds"]):
                raise ValueError("采样画面与原片时间轴不一致")


def verify_source(library, asset_id):
    """Check the full imported SHA once per unchanged file, before using its frozen playback timeline."""
    manifest = read_json(library.asset_dir(asset_id) / "manifest.json", {})
    location = read_json(library.root / "private/media-locations.json", {}).get(asset_id)
    if (manifest.get("id") != asset_id or not isinstance(location, dict) or not location.get("path")
            or not SHA256.fullmatch(str(manifest.get("fingerprint", "")))):
        raise FileNotFoundError("原片离线或缺少完整指纹，请重新定位原片")
    source = Path(location["path"]).resolve()
    if not source.is_file() or source_snapshot(source) != location:
        raise FileNotFoundError("原片离线或已变化，请重新定位并校验原片")
    stamp = source_identity(source)
    fingerprint = manifest["fingerprint"]
    key = (asset_id, fingerprint, stamp, manifest.get("duration_ms"), manifest.get("start_time_seconds"))
    store = library.sample_frames
    with store.source_lock:
        metadata = store.sources.get(key)
        if metadata is None:
            if media.fingerprint(source) != fingerprint:
                raise ValueError("原视频内容与登记的 SHA-256 不一致")
            metadata = {**media.probe_for_asset(source, manifest), "_sample_asset": asset_id,
                        "_sample_fingerprint": fingerprint, "_sample_identity": stamp}
            if source_identity(source) != stamp:
                raise ValueError("原片在校验期间发生了变化")
            store.sources[key] = metadata
            while len(store.sources) > 32:
                store.sources.popitem(last=False)
        store.sources.move_to_end(key)
    return source, metadata


def restore_batch(source, metadata, requests, directory):
    """Decode private outputs in the caller's temporary directory and check each original JPEG hash."""
    if source_identity(source) != metadata["_sample_identity"]:
        raise ValueError("原片在画面重建前发生了变化")
    exact, outputs = defaultdict(list), {}
    for identity, entry in requests:
        safe_id(identity)
        sample = entry["sample"]
        validate_recipe(metadata["_sample_asset"], metadata["_sample_fingerprint"], identity, entry,
                        legacy=sample.get("jpeg_sha256") is None)
        output = directory / f"{identity}.jpg"
        outputs[identity] = output
        if sample.get("kind") == "uniform":
            [actual] = media.extract_frames(source, [sample["requested_at_ms"]], [output], sample["crop"])
            if actual["at_ms"] != entry["at_ms"]:
                raise ValueError("重建采样的真实时间与归档不一致")
        else:
            exact[tuple(sample["source_time_base"])].append(({**sample, "id": identity, "at_ms": entry["at_ms"]}, output))
    for time_base, group in exact.items():
        group.sort(key=lambda pair: pair[0]["source_pts"])
        # Distant hits should seek near each timestamp instead of decoding
        # all the intervening minutes. Dense ranges keep one continuous pass.
        breaks = [0]
        for index, (left, right) in enumerate(pairwise(group), 1):
            gap = right[0]["source_pts"] - left[0]["source_pts"]
            if gap * time_base[0] > 15 * time_base[1]:
                breaks.append(index)
        breaks.append(len(group))
        for start, end in pairwise(breaks):
            frames, paths = zip(*group[start:end])
            extract = keyframes.extract_keyframes if len(frames) <= 256 else keyframes.extract_keyframes_stream
            extract(source, list(frames), list(paths), metadata=metadata)
    if source_identity(source) != metadata["_sample_identity"]:
        raise ValueError("原片在画面重建时发生了变化")
    for identity, entry in requests:
        expected = entry["sample"].get("jpeg_sha256")
        if expected and hashlib.sha256(outputs[identity].read_bytes()).hexdigest() != expected:
            raise ValueError("重建的 JPEG 与归档画面的 SHA-256 不一致")
    return outputs


def legacy_recipes(library, asset_id, requested, registry, fingerprint):
    """Old registries only store milliseconds. Accept exact, asset-bound archived plans."""
    directory = library.asset_dir(asset_id)
    missing = {fid for fid in requested if fid.startswith("kf_")}
    recipes = {}
    if not missing:
        return recipes
    for folder, pattern in (("retrieval/plans", "rp_*.json"), ("keyframe-plans", "plan_*.json")):
        for candidate in sorted(confined_path(directory, folder).glob(pattern)):
            path = confined_path(directory, candidate.relative_to(directory).as_posix())
            plan = read_json(path, {})
            if (plan.get("source_fingerprint", plan.get("fingerprint")) != fingerprint
                    or plan.get("asset_id", asset_id) != asset_id or plan.get("id") != candidate.stem):
                continue
            for frame in plan.get("frames", []):
                fid = frame.get("id")
                if fid in missing and frame.get("at_ms") == registry[fid]["at_ms"]:
                    recipe = make_recipe(asset_id, fingerprint, frame, None)
                    if fid in recipes and recipe != recipes[fid]:
                        raise ValueError("已登记的关键帧计划时间戳冲突")
                    recipes[fid] = recipe
    return recipes


def archive_sample_frames(library, archive):
    """Fill missing sampled JPEG members in a portable ZIP, using its frozen registry."""
    added = 0
    for name in archive.namelist():
        parts = PurePosixPath(name).parts
        if len(parts) != 4 or parts[0] != "assets" or parts[2:] != ("frames", "registry.json"):
            continue
        asset_id = safe_id(parts[1])
        prefix = f"assets/{asset_id}/"
        registry = json.loads(archive.read(name))
        requested = []
        for identity, entry in registry.items():
            safe_id(identity)
            if entry.get("alias_of") or not identity.startswith(("kf_", "f_")):
                continue
            path = PurePosixPath(entry["path"].replace("\\", "/"))
            if path.suffix.lower() not in (".jpg", ".jpeg"):
                continue
            if entry.get("pinned_sha256"):
                pin = path if path.parts[:1] == ("preserved-frames",) else PurePosixPath("preserved-frames", identity + path.suffix)
                if prefix + pin.as_posix() not in archive.NameToInfo:
                    raise FileNotFoundError("备份缺少独立保存的人工画面")
                continue
            if not is_sample_path(identity, entry["path"]) or prefix + path.as_posix() in archive.NameToInfo:
                continue
            requested.append((identity, entry))
        if not requested:
            continue
        source, metadata = verify_source(library, asset_id)
        manifest = json.loads(archive.read(prefix + "manifest.json"))
        if metadata["_sample_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("生成备份时原片指纹已变化")
        legacy = [identity for identity, entry in requested if "sample" not in entry]
        recipes = legacy_recipes(library, asset_id, legacy, registry, manifest["fingerprint"]) if legacy else {}
        requested.sort(key=lambda item: item[1]["at_ms"])
        for offset in range(0, len(requested), 256):
            batch = []
            for identity, entry in requested[offset:offset + 256]:
                sample = entry.get("sample", recipes.get(identity))
                if not sample:
                    raise FileNotFoundError("已采样的 JPEG 缺失，且找不到对应的原片画面计划")
                batch.append((identity, {**entry, "sample": sample}))
            with tempfile.TemporaryDirectory(prefix="scenerecall-export-frames-") as temporary:
                outputs = restore_batch(source, metadata, batch, Path(temporary))
                for identity, entry in batch:
                    relative = entry["path"].replace("\\", "/")
                    archive.write(outputs[identity], prefix + relative, compress_type=ZIP_STORED)
                    added += 1
    return added


class _Filename(str):
    """Existing model adapters keep filenames as strings until their async request finishes."""
    def __deepcopy__(self, memo):
        return self


class _TemporaryFramePath(type(Path())):
    def __str__(self):
        name = _Filename(super().__str__())
        name.owner = getattr(self, "owner", None)
        name.frame_ref = getattr(self, "frame_ref", None)
        return name

    def __deepcopy__(self, memo):
        return self


class SampleFrameStore:
    def __init__(self, library):
        self.library = library
        self.source_lock = threading.Lock()
        self.sources = OrderedDict()
        self.registry_lock = threading.Lock()
        self.registries = OrderedDict()
        self.registry_source_bytes = 0
        self.lock = threading.Lock()
        self.inflight = {}
        self.decoders = threading.BoundedSemaphore(2)
        self.cache = OrderedDict()
        self.bytes = 0
        self.budget = DEFAULT_CACHE_BYTES

    def _registry(self, path, reader):
        """Internal read-only snapshots. External callers only receive copies of requested entries."""
        with self.registry_lock:
            for _ in range(2):
                try:
                    stamp = source_identity(path)
                except FileNotFoundError:
                    return reader(path, {})
                previous = self.registries.get(path)
                if previous and previous[0] == stamp:
                    self.registries.move_to_end(path)
                    return previous[1]
                records = reader(path, {})
                try:
                    stable = stamp == source_identity(path)
                except FileNotFoundError:
                    stable = False
                if not stable:
                    continue
                if previous:
                    self.registry_source_bytes -= previous[0][3]
                    del self.registries[path]
                # Bound both the retained source bytes and the number of decoded
                # registries. Very large registries still work without caching.
                if stamp[3] <= 16 * 1024**2:
                    self.registries[path] = (stamp, records)
                    self.registry_source_bytes += stamp[3]
                while self.registry_source_bytes > 24 * 1024**2 or len(self.registries) > 8:
                    _, (evicted, _) = self.registries.popitem(last=False)
                    self.registry_source_bytes -= evicted[3]
                return records
        raise ValueError("画面登记正在更新，请重试")

    def _remember(self, key, identity, source):
        owner = tempfile.TemporaryDirectory(prefix="scenerecall-frame-")
        path = _TemporaryFramePath(owner.name) / f"{identity}.jpg"
        path.owner = owner
        path.frame_ref = (key[0], identity)
        os.replace(source, path)
        size = path.stat().st_size
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.lock:
            if size <= self.budget:
                self.cache[key] = (path, size, content_hash)
                self.bytes += size
            while self.bytes > self.budget or len(self.cache) > 1024:
                _, (_, evicted, _) = self.cache.popitem(last=False)
                self.bytes -= evicted
        # Paths still held by a consumer keep their own temporary file alive
        # after cache eviction. The final reference removes its directory.
        return path

    def resolve(self, asset_id, entries):
        """Resolve sampled rows as a batch. Unrelated media/HTTP work can continue during decoding."""
        fingerprint = read_json(self.library.asset_dir(asset_id) / "manifest.json")["fingerprint"]
        for identity, entry, _ in entries:
            validate_recipe(asset_id, fingerprint, identity, entry, legacy=entry["sample"].get("jpeg_sha256") is None)
        checked_source, result, owned, waiting = None, {}, [], {}
        try:
            for identity, entry, archived in entries:
                try:
                    stamp = source_identity(archived)
                except FileNotFoundError:
                    checked_source = checked_source or verify_source(self.library, asset_id)
                    stamp = checked_source[1]["_sample_identity"]
                key = (asset_id, identity, digest(entry["sample"]), stamp)
                with self.lock:
                    cached = self.cache.get(key)
                    if cached is not None:
                        path, size, content_hash = cached
                        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == content_hash:
                            self.cache.move_to_end(key)
                            result[identity] = path
                            continue
                        del self.cache[key]
                        self.bytes -= size
                    future = self.inflight.get(key)
                    if future is None:
                        future = self.inflight[key] = Future()
                        owned.append((identity, entry, archived, key, future))
                    waiting[identity] = future
            if owned:
                self._build(asset_id, owned, checked_source)
        except BaseException as exc:
            for _, _, _, _, future in owned:
                if not future.done():
                    future.set_exception(exc)
            raise
        finally:
            with self.lock:
                for _, _, _, key, _ in owned:
                    self.inflight.pop(key, None)
        result.update((identity, future.result()) for identity, future in waiting.items())
        return result

    def _build(self, asset_id, owned, checked_source):
        with tempfile.TemporaryDirectory(prefix="scenerecall-frame-batch-") as staging:
            directory, pending = Path(staging), []
            for identity, entry, archived, key, future in owned:
                output = directory / f"{identity}.jpg"
                try:
                    shutil.copyfile(archived, output)
                except FileNotFoundError:
                    # Retention can release a registered file while it is opened.
                    pending.append((identity, entry, key, future))
                else:
                    expected = entry["sample"].get("jpeg_sha256")
                    if expected and hashlib.sha256(output.read_bytes()).hexdigest() != expected:
                        raise ValueError("已登记 JPEG 内容与采样配方不一致")
                    future.set_result(self._remember(key, identity, output))
            if pending:
                source, metadata = checked_source or verify_source(self.library, asset_id)
                with self.decoders:
                    outputs = restore_batch(source, metadata, [(fid, e) for fid, e, _, _ in pending], directory)
                for identity, _, key, future in pending:
                    future.set_result(self._remember(key, identity, outputs[identity]))
