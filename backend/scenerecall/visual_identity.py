"""Local character references, crop features and reversible identity decisions.

References never assert presence in a video. Raw detections/features are immutable
and independent of dossier names; review decisions are a separate projection.
No network/model download is performed by this module or its optional worker.
"""
from __future__ import annotations

import atexit
import copy
import hashlib
import io
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, ImageStat

from .characters import CharacterStore, checked_box, external_sources, scope_id
from .file_io import atomic_replace
from .library import atomic_json, confined_path, digest, now, read_json, safe_id, uid


PREPROCESS_VERSION = "single-person-rgb-ccip384-v1"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MODEL = "ccip-caformer-24-randaug-pruned"


class IdentityUnavailable(RuntimeError):
    pass


def _image(content: bytes) -> Image.Image:
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("参考图不能超过 20 MiB")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError("参考图像素过大")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("参考图不是有效的图片") from exc


def image_quality(picture: Image.Image, *, single_character: bool = False, person_count: int | None = None) -> dict:
    # Edge variance is a quality gate, not an identity score or a learned blur
    # classifier. Keep the raw value so datasets can calibrate this gate later.
    gray = picture.convert("L")
    inner = gray.filter(ImageFilter.FIND_EDGES).crop((1, 1, max(2, gray.width - 1), max(2, gray.height - 1)))
    sharpness = float(ImageStat.Stat(inner).var[0])
    reasons = []
    if min(picture.size) < 64:
        reasons.append("too_small")
    if sharpness < 8:
        reasons.append("blur_or_low_detail")
    if person_count is not None and person_count != 1:
        reasons.append("multiple_or_no_people")
    if not single_character:
        reasons.append("single_character_unverified")
    return {"width": picture.width, "height": picture.height, "sharpness": round(sharpness, 4),
            "single_character": single_character, "person_count": person_count,
            "accepted": not reasons, "reasons": reasons, "version": "edge-variance-v1"}


def _png(picture: Image.Image) -> bytes:
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


def _write_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uid('write')}.tmp")
    try:
        temporary.write_bytes(content)
        atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _allowed(library, asset_id, *, catalog=None):
    """Read the complete permitted roster without an appearance/recency limit."""
    characters = CharacterStore(library)
    # Catalog performs the existing local roster upgrade and enforces current
    # membership/animation restrictions before any reference is consulted.
    catalog = characters.catalog(asset_id) if catalog is None else catalog
    return catalog, {row["id"]: row for row in catalog["profiles"]}


def _bangumi_candidate(reference, profile):
    return (reference["status"] == "candidate" and reference["source"]["kind"] == "external_cached"
            and any(source.get("source") == "bangumi" and source.get("image_url")
                    and source["image_url"] == reference["source"].get("url")
                    for source in external_sources(profile)))


class ReferenceGallery:
    def __init__(self, library):
        self.library = library

    def _directory(self, scope):
        return self.library.root / "works" / safe_id(scope) / "reference-images"

    def _read(self, scope):
        return read_json(self._directory(scope) / "gallery.json", {"schema_version": "1.0", "revision": 0,
                                                                  "scope_id": scope, "references": [], "audit": []})

    def list(self, asset_id, character_id=None, *, trusted_only=False, _catalog=None):
        with self.library.lock:
            catalog, allowed = _allowed(self.library, asset_id, catalog=_catalog)
            scope = catalog["scope_id"]
            registry = CharacterStore(self.library)._read(scope)
            if character_id is not None:
                character_id = CharacterStore._resolve(registry, character_id, strict=True)
                if character_id not in allowed:
                    raise FileNotFoundError("角色不在当前作品的名单中")
            saved = self._read(scope)
            rows = []
            for original in saved["references"]:
                resolved = CharacterStore._resolve(registry, original["character_id"])
                if resolved not in allowed or character_id is not None and resolved != character_id:
                    continue
                row = copy.deepcopy(original)
                row["character_id"] = resolved
                # A cropped sample from a recycled or reassigned episode cannot
                # silently continue to serve as a model reference in this work.
                if row["source"]["kind"] == "frame_crop":
                    try:
                        source = self.library.get_asset(row["source"]["asset_id"])
                        if scope_id(source) != scope:
                            continue
                    except (FileNotFoundError, ValueError):
                        continue
                if trusted_only and (row["status"] != "confirmed" or not row["quality"]["accepted"]):
                    continue
                row["path"] = str(self._directory(scope) / f"{safe_id(row['id'])}.png")
                row["image_url"] = f"/api/assets/{asset_id}/characters/{resolved}/references/{row['id']}/image"
                row["reference_only"] = True
                rows.append(row)
            return {"scope_id": scope, "revision": saved["revision"], "references": rows}

    def trusted(self, asset_id):
        return self.list(asset_id, trusted_only=True)["references"]

    def matching(self, asset_id, *, _catalog=None):
        """Confirmed exemplars plus stored Bangumi images as candidates only."""
        active = self.active_reference_ids(asset_id, include_candidates=True)
        return [row for row in self.list(asset_id, _catalog=_catalog)["references"] if row["id"] in active]

    def active_reference_ids(self, asset_id, *, include_candidates=False):
        """Nonrecursive trust projection usable while catalog reads scenes."""
        asset = self.library.get_asset(asset_id)
        scope = scope_id(asset)
        registry = CharacterStore(self.library)._read(scope)
        policy = CharacterStore.cast_policy(registry, asset.get("anime_profile"))
        identities = {key for key, profile in registry["profiles"].items()
                      if CharacterStore.profile_allowed(profile, policy)}
        result = set()
        for row in self._read(scope)["references"]:
            resolved = CharacterStore._resolve(registry, row["character_id"])
            if (resolved not in identities
                    or not (row["status"] == "confirmed" and row["quality"]["accepted"]
                            or include_candidates and _bangumi_candidate(row, registry["profiles"][resolved]))):
                continue
            if row["source"]["kind"] == "frame_crop":
                try:
                    if scope_id(self.library.get_asset(row["source"]["asset_id"])) != scope:
                        continue
                except (FileNotFoundError, ValueError):
                    continue
            result.add(row["id"])
        return result

    def add(self, asset_id, character_id, content: bytes, *, source_kind="upload", source_url="",
            confirmed=False, single_character=False, source=None):
        if source_kind not in {"upload", "external_cached", "frame_crop"}:
            raise ValueError("参考图来源无效")
        if source_kind == "external_cached" and not source_url.strip():
            raise ValueError("已缓存外部参考图须保留来源 URL")
        if source_kind == "frame_crop" and not source:
            raise ValueError("原片裁剪必须通过注册证据帧生成")
        picture = _image(content)
        quality = image_quality(picture, single_character=bool(single_character))
        if confirmed and not quality["accepted"]:
            raise ValueError("参考图未通过单角色、尺寸或清晰度检查")
        encoded = _png(picture)
        with self.library.lock:
            catalog, allowed = _allowed(self.library, asset_id)
            character_id = CharacterStore._resolve(CharacterStore(self.library)._read(catalog["scope_id"]),
                                                   character_id, strict=True)
            if character_id not in allowed:
                raise FileNotFoundError("角色不在当前作品的名单中")
            scope = catalog["scope_id"]
            saved = self._read(scope)
            identifier = uid("reference")
            timestamp = now()
            row = {"id": identifier, "character_id": character_id, "source": {
                "kind": source_kind, **({"url": source_url.strip()} if source_url else {}), **(source or {})},
                "status": "confirmed" if confirmed else "candidate", "quality": quality, "version": 1,
                "sha256": hashlib.sha256(encoded).hexdigest(), "created_at": timestamp, "updated_at": timestamp,
                "confirmation_source": "human" if confirmed else None, "reference_only": True}
            _write_bytes(self._directory(scope) / f"{identifier}.png", encoded)
            saved["references"].append(row)
            saved["revision"] += 1
            saved["audit"].append({"at": timestamp, "id": identifier, "action": "create", "after": copy.deepcopy(row)})
            atomic_json(self._directory(scope) / "gallery.json", saved)
        return next(row for row in self.list(asset_id, character_id)["references"] if row["id"] == identifier)

    def crop(self, asset_id, character_id, frame_id, box, *, confirmed=False, single_character=False):
        box = checked_box(box)
        if box is None:
            raise ValueError("原片样本须指定单角色裁剪区域")
        path = self.library.frame_path(asset_id, frame_id)
        picture = _image(path.read_bytes())
        x, y, width, height = box
        crop = picture.crop((round(x * picture.width), round(y * picture.height),
                             round((x + width) * picture.width), round((y + height) * picture.height)))
        return self.add(asset_id, character_id, _png(crop), source_kind="frame_crop", confirmed=confirmed,
                        single_character=single_character, source={"asset_id": asset_id, "frame_id": frame_id, "box": box})

    def update(self, asset_id, character_id, reference_id, *, status, reason, single_character=None):
        if status not in {"confirmed", "candidate", "revoked"} or not str(reason).strip():
            raise ValueError("确认或撤销参考图须填写状态和修改原因")
        with self.library.lock:
            available = self.list(asset_id, character_id)
            current = next((row for row in available["references"] if row["id"] == reference_id), None)
            if current is None:
                raise FileNotFoundError("参考图不存在")
            saved = self._read(available["scope_id"])
            row = next(row for row in saved["references"] if row["id"] == reference_id)
            before = copy.deepcopy(row)
            quality = image_quality(_image(Path(current["path"]).read_bytes()), single_character=(
                bool(single_character) if single_character is not None else row["quality"]["single_character"]))
            if status == "confirmed" and not quality["accepted"]:
                raise ValueError("参考图未通过单角色、尺寸或清晰度检查")
            row.update(status=status, quality=quality, updated_at=now(), version=row["version"] + 1,
                       confirmation_source="human" if status == "confirmed" else None)
            saved["revision"] += 1
            saved["audit"].append({"at": now(), "id": reference_id, "action": status, "reason": str(reason).strip(),
                                   "before": before, "after": copy.deepcopy(row)})
            atomic_json(self._directory(available["scope_id"]) / "gallery.json", saved)
        return next(row for row in self.list(asset_id, character_id)["references"] if row["id"] == reference_id)


class ImgutilsBackend:
    """Resident, offline subprocess keeps optional ONNX imports out of the app.

    The upstream learned CCIP metric is used; cosine distance is not substituted.
    HF_HUB_OFFLINE is scoped to the worker, never the user's other providers.
    """
    def __init__(self, model=DEFAULT_MODEL, *, timeout_s=180):
        self.model = model
        self.timeout_s = timeout_s
        self._process = None
        self._reader = None
        self._lock = threading.RLock()
        self._metadata = None
        atexit.register(self.close)

    def close(self):
        with self._lock:
            if self._process is not None:
                process, self._process = self._process, None
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                if self._reader is not None:
                    self._reader.join(timeout=2)
                    self._reader = None
                for stream in (process.stdin, process.stdout):
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        # A failed stdin flush can fail again during close;
                        # preserve the original timeout/worker-unavailable error.
                        pass

    def _request(self, action, **arguments):
        from .local_resources import LOCAL_MODEL_GATE
        with self._lock, LOCAL_MODEL_GATE:
            if self._process is None or self._process.poll() is not None:
                self.close()
                executable = os.environ.get("SCENERECALL_VISUAL_PYTHON", sys.executable)
                resolved = shutil.which(executable)
                if not resolved or not Path(resolved).is_file():
                    raise IdentityUnavailable("SCENERECALL_VISUAL_PYTHON 不是可执行的本地 Python 路径")
                environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                                   HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
                # The dedicated ML environment need not install SceneRecall's
                # entire application dependency set; only this private worker
                # and optional visual dependencies are imported in that process.
                environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
                    str(Path(__file__).resolve().parent.parent), environment.get("PYTHONPATH")]))
                self._process = subprocess.Popen(
                    [resolved, "-u", "-m", "scenerecall.visual_identity_worker", self.model],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, encoding="utf-8", env=environment)
            try:
                self._process.stdin.write(json.dumps({"action": action, **arguments}) + "\n")
                self._process.stdin.flush()
                # Windows selectors support sockets, not subprocess pipes. Read
                # the entire JSON line in a thread so partial output is bounded
                # by the same timeout on every supported platform.
                response = queue.Queue(maxsize=1)
                stream = self._process.stdout

                def read_line():
                    try:
                        response.put(stream.readline())
                    except (OSError, ValueError) as exc:
                        response.put(exc)

                self._reader = threading.Thread(target=read_line, name="scenerecall-visual-output", daemon=True)
                self._reader.start()
                try:
                    output = response.get(timeout=self.timeout_s)
                except queue.Empty:
                    self.close()
                    raise IdentityUnavailable("imgutils 本地识别超时；已停止该 worker") from None
                self._reader.join()
                self._reader = None
                if isinstance(output, Exception):
                    raise output
                result = json.loads(output)
            except (BrokenPipeError, OSError, ValueError) as exc:
                self.close()
                raise IdentityUnavailable("imgutils worker 不可用，请检查可选依赖与本地权重") from exc
            if result.get("error"):
                raise IdentityUnavailable(result["error"])
            return result["result"]

    @property
    def model_version(self):
        if self._metadata is None:
            self._metadata = self._request("metadata")
        return self._metadata["model_version"]

    def detect(self, path):
        return self._request("detect", path=str(path))

    def feature(self, path):
        return self._request("feature", path=str(path))

    def distances(self, query, features):
        return self._request("distances", query=query, features=features)


_BACKENDS = {}
_BACKENDS_LOCK = threading.Lock()


def _backend(model):
    with _BACKENDS_LOCK:
        if model not in _BACKENDS:
            _BACKENDS[model] = ImgutilsBackend(model)
        return _BACKENDS[model]


def match_candidates(distances, references, *, top_k=5, calibration=None, model_version=None):
    """Search every permitted reference before choosing Top-K identities.

    Distance is the mean of the two closest exemplars (one when only one
    exists). Unconfirmed exemplars can suggest a candidate, never confirm it.
    Calibration must use this exact aggregation with confirmed exemplars.
    """
    grouped = {}
    if len(distances) != len(references):
        raise ValueError("CCIP 返回的距离数量不正确")
    for value, reference in zip(distances, references):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("CCIP 返回了无效距离")
        grouped.setdefault(reference["character_id"], []).append((value, reference["id"]))
    candidates = []
    for character_id, values in grouped.items():
        chosen = sorted(values)[:2]
        candidates.append({"character_id": character_id, "distance": sum(value for value, _ in chosen) / len(chosen),
                           "reference_ids": [reference_id for _, reference_id in chosen], "reference_count": len(values)})
    candidates.sort(key=lambda row: (row["distance"], row["character_id"]))
    best = candidates[0] if candidates else None
    by_reference = {row["id"]: row for row in references}
    trusted_best = bool(best) and all(by_reference[key].get("status", "confirmed") == "confirmed"
                                      for key in best["reference_ids"])
    margin = candidates[1]["distance"] - best["distance"] if len(candidates) > 1 else None
    status = "candidate" if best else "unknown"
    calibrated = (trusted_best and isinstance(calibration, dict) and calibration.get("model_version") == model_version
                  and calibration.get("aggregation") == "nearest-two-mean-v1"
                  and calibration.get("dataset_id") and calibration.get("evaluation_id"))
    if calibrated and best:
        maximum = calibration.get("max_distance")
        minimum_margin = calibration.get("min_margin")
        if (type(maximum) not in {int, float} or type(minimum_margin) not in {int, float}
                or not all(math.isfinite(value) and value >= 0 for value in (maximum, minimum_margin))):
            raise ValueError("视觉校准阈值无效")
        if best["distance"] > maximum:
            status = "unknown"
        elif margin is not None and margin >= minimum_margin:
            status = "confirmed"
    return {"status": status, "character_id": best["character_id"] if best and status == "confirmed" else None,
            "candidate_character_id": best["character_id"] if best and status == "candidate" else None,
            "distance": best["distance"] if best else None, "margin": margin,
            "source": ("ccip_unverified_reference" if best and not trusted_best else
                       "ccip_calibrated" if calibrated else "ccip_uncalibrated"), "model_version": model_version,
            "calibration_id": calibration.get("evaluation_id") if calibrated else None,
            "candidates": candidates[:max(1, min(20, top_k))], "aggregation": "nearest-two-mean-v1"}


class VisualIdentityPipeline:
    def __init__(self, library, backend=None):
        self.library = library
        self.backend = backend
        self.gallery = ReferenceGallery(library)

    def _directory(self, asset_id):
        return self.library.asset_dir(asset_id) / "identity" / "visual"

    def settings(self, asset_id):
        self.library.get_asset(asset_id)
        return {"enabled": False, "model": DEFAULT_MODEL, "top_k": 5, "calibration": None,
                **read_json(self._directory(asset_id) / "settings.json", {})}

    def configure(self, asset_id, payload):
        if set(payload) - {"enabled", "model", "top_k", "calibration"}:
            raise ValueError("视觉身份设置包含未知字段")
        with self.library.lock:
            settings = {**self.settings(asset_id), **payload}
            if (type(settings["enabled"]) is not bool or type(settings["top_k"]) is not int
                    or not 1 <= settings["top_k"] <= 20 or settings["model"] not in {
                        DEFAULT_MODEL, "ccip-caformer-6-randaug-pruned_fp32", "ccip-caformer-5_fp32"}):
                raise ValueError("视觉身份设置无效")
            if settings["calibration"] is not None:
                calibration = settings["calibration"]
                if (not isinstance(calibration, dict) or calibration.get("aggregation") != "nearest-two-mean-v1"
                        or any(not isinstance(calibration.get(key), str) or not calibration[key].strip()
                               for key in ("model_version", "dataset_id", "evaluation_id"))
                        or any(type(calibration.get(key)) not in {int, float}
                               or not math.isfinite(calibration[key]) or calibration[key] < 0
                               for key in ("max_distance", "min_margin"))):
                    raise ValueError("视觉阈值必须提供相同模型/聚合算法的标注集校准记录")
            atomic_json(self._directory(asset_id) / "settings.json", settings)
            return settings

    def _feature(self, path, model_version, backend, metrics):
        content = Path(path).read_bytes()
        key = digest([hashlib.sha256(content).hexdigest(), model_version, PREPROCESS_VERSION])
        location = self.library.root / "cache" / "identity-features" / f"{key}.json"
        cached = read_json(location)
        if cached is not None:
            metrics["feature_cache_hits"] += 1
            return cached["feature"]
        started = time.monotonic()
        feature = backend.feature(path)
        if not feature or any(type(value) not in {int, float} or not math.isfinite(value) for value in feature):
            raise ValueError("CCIP 返回了无效特征")
        metrics["feature_ms"] += round((time.monotonic() - started) * 1000)
        metrics["feature_extractions"] += 1
        atomic_json(location, {"key": key, "model_version": model_version, "preprocessing": PREPROCESS_VERSION,
                               "feature": feature})
        return feature

    def run(self, asset_id, frames, config=None, checkpoint=None, progress=None):
        config = config or {}
        checkpoint = checkpoint or (lambda: None)
        started = time.monotonic()
        asset = self.library.get_asset(asset_id)
        result = {"asset_id": asset_id, "status": "disabled", "regions": [], "selected_profiles": [], "references": [],
                  "metrics": {"model_load_ms": 0, "detection_ms": 0, "feature_ms": 0, "matching_ms": 0,
                              "feature_extractions": 0, "feature_cache_hits": 0, "detection_calls": 0,
                              "detection_cache_hits": 0, "rejected_references": 0, "total_ms": 0}}
        if not config.get("enabled", False):
            result["degraded_reason"] = "视觉身份识别未启用"
            return result
        metrics = result["metrics"]
        try:
            checkpoint()
            if progress is not None:
                progress(0, len(frames))
            load_start = time.monotonic()
            backend = self.backend or _backend(config.get("model", DEFAULT_MODEL))
            version = backend.model_version
            metrics["model_load_ms"] = round((time.monotonic() - load_start) * 1000)
            catalog, profiles = _allowed(self.library, asset_id)
            proposed_references = self.gallery.matching(asset_id)
            references = []
            # Read every permitted character's exemplars; there is no recency or
            # first-16 truncation before nearest-neighbour selection.
            features = []
            for reference in proposed_references:
                checkpoint()
                if reference["status"] != "confirmed":
                    # Downloading a Bangumi image does not confirm its identity.
                    # Recheck the local pixels before candidate-only matching;
                    # detector success never changes the gallery's trust state.
                    content = Path(reference["path"]).read_bytes()
                    if (hashlib.sha256(content).hexdigest() != reference["sha256"]
                            or not image_quality(_image(content), single_character=True)["accepted"]):
                        metrics["rejected_references"] += 1
                        continue
                # Human confirmation authorizes identity, not an exemption from
                # the single-person input constraint of the CCIP model.
                detection_key = digest([reference["sha256"], version, PREPROCESS_VERSION])
                detection_path = self.library.root / "cache" / "identity-reference-detections" / f"{detection_key}.json"
                detected = read_json(detection_path)
                if detected is None:
                    step_start = time.monotonic()
                    detected = {"detections": backend.detect(reference["path"])}
                    metrics["detection_calls"] += 1
                    metrics["detection_ms"] += round((time.monotonic() - step_start) * 1000)
                    atomic_json(detection_path, detected)
                else:
                    metrics["detection_cache_hits"] += 1
                if len(detected["detections"]) != 1:
                    metrics["rejected_references"] += 1
                    continue
                references.append(reference)
                features.append(self._feature(reference["path"], version, backend, metrics))
            frame_registry = read_json(self.library.asset_dir(asset_id) / "frames" / "registry.json", {})
            for frame_number, frame in enumerate(frames, 1):
                checkpoint()
                identifier = frame["id"]
                path = self.library.frame_path(asset_id, identifier)
                at_ms = frame_registry[identifier]["at_ms"]
                if frame.get("at_ms", at_ms) != at_ms:
                    raise ValueError("视觉身份帧时间与原片时间轴不一致")
                content = path.read_bytes()
                key = digest([asset_id, asset["fingerprint"], hashlib.sha256(content).hexdigest(), at_ms,
                              version, PREPROCESS_VERSION])
                detection_path = self._directory(asset_id) / "detections" / f"{key}.json"
                saved = read_json(detection_path)
                picture = _image(content)
                if saved is None:
                    step_start = time.monotonic()
                    detections = backend.detect(path)
                    metrics["detection_ms"] += round((time.monotonic() - step_start) * 1000)
                    metrics["detection_calls"] += 1
                    saved = {"key": key, "frame_id": identifier, "at_ms": at_ms, "detections": detections,
                             "model_version": version, "preprocessing": PREPROCESS_VERSION}
                    atomic_json(detection_path, saved)
                else:
                    metrics["detection_cache_hits"] += 1
                boxes = []
                for detection in saved["detections"]:
                    coordinates = detection["box"] if isinstance(detection, dict) else detection[0]
                    if len(coordinates) != 4 or any(not math.isfinite(float(v)) for v in coordinates):
                        raise ValueError("人物检测返回无效区域")
                    x0, y0, x1, y1 = map(float, coordinates)
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(picture.width, x1), min(picture.height, y1)
                    if x1 > x0 and y1 > y0:
                        boxes.append([x0, y0, x1, y1])
                boxes.sort(key=lambda box: (box[0], box[1], box[2], box[3]))
                for index, box in enumerate(boxes):
                    checkpoint()
                    x0, y0, x1, y1 = box
                    normalized_box = [x0 / picture.width, y0 / picture.height,
                                      (x1 - x0) / picture.width, (y1 - y0) / picture.height]
                    region_id = "region_" + digest([asset_id, identifier, key, normalized_box])
                    cropped = picture.crop(tuple(round(value) for value in box))
                    # Reject crops containing another detection's centre; a
                    # whole multi-person frame never enters single-person CCIP.
                    count = sum(x0 <= (other[0] + other[2]) / 2 <= x1 and y0 <= (other[1] + other[3]) / 2 <= y1
                                for other in boxes)
                    quality = image_quality(cropped, single_character=count == 1, person_count=count)
                    crop_path = self._directory(asset_id) / "crops" / f"{region_id}.png"
                    if not crop_path.exists():
                        _write_bytes(crop_path, _png(cropped))
                    match = match_candidates([], [], model_version=version)
                    if quality["accepted"] and references:
                        feature = self._feature(crop_path, version, backend, metrics)
                        match_key = digest(["reference-trust-v1", region_id, version,
                                            [(row["id"], row["version"], row["character_id"], row["sha256"], row["status"])
                                                               for row in references], config.get("calibration"),
                                            config.get("top_k", 5)])
                        match_path = self._directory(asset_id) / "matches" / f"{match_key}.json"
                        match = read_json(match_path)
                        if match is None:
                            step_start = time.monotonic()
                            distances = backend.distances(feature, features)
                            match = match_candidates(distances, references, top_k=config.get("top_k", 5),
                                                     calibration=config.get("calibration"), model_version=version)
                            metrics["matching_ms"] += round((time.monotonic() - step_start) * 1000)
                            atomic_json(match_path, match)
                    from .identity_bindings import POLICY_VERSION, stamp_match
                    match = stamp_match(match, references, config)
                    region = {"id": region_id, "entity_id": region_id, "local_id": f"P{index + 1}",
                              "frame_id": identifier, "at_ms": at_ms, "box": normalized_box, "quality": quality,
                              "crop_path": str(crop_path), "match": match, "model_version": version,
                              "preprocessing": PREPROCESS_VERSION}
                    if config.get("identity_policy") == POLICY_VERSION:
                        region.update(source_fingerprint=asset["fingerprint"], identity_policy=POLICY_VERSION)
                    # Per-region raw snapshots survive later review and partial
                    # recovery; the display manifest is only a projection.
                    history_path = self._directory(asset_id) / "region-history" / region_id / f"{digest(region)}.json"
                    if not history_path.exists():
                        atomic_json(history_path, region)
                    atomic_json(self._directory(asset_id) / "regions" / f"{region_id}.json", region)
                    result["regions"].append(region)
                if progress is not None:
                    progress(frame_number, len(frames))
            result.update(status="completed", model_version=version, scope_id=catalog["scope_id"],
                          gallery_revision=self.gallery.list(asset_id)["revision"])
            result["regions"] = self._project(asset_id, result["regions"])
            ids = list(dict.fromkeys(value for region in result["regions"]
                                    if region["match"].get("status") in {"candidate", "confirmed"}
                                    for value in [region["match"].get("character_id"),
                                                  *[row["character_id"] for row in region["match"].get("candidates", [])]]
                                    if value in profiles))
            result["selected_profiles"] = [profiles[value] for value in ids]
            result["references"] = [{"character_id": row["character_id"], "reference_id": row["id"],
                                      "path": row["path"], "reference_only": True,
                                      "reference_status": row["status"], "human_confirmed": row["status"] == "confirmed"}
                                     for value in ids for row in references if row["character_id"] == value]
            if not references:
                result["degraded_reason"] = "没有可用的本地角色参考图；检测区域保留为未知人物，角色资料仍仅作未验证先验"
        except IdentityUnavailable as exc:
            result.update(status="unavailable", degraded_reason=str(exc))
        metrics["total_ms"] = round((time.monotonic() - started) * 1000)
        atomic_json(self._directory(asset_id) / "latest.json", result)
        return result

    def _project(self, asset_id, regions, *, _catalog=None):
        decisions = read_json(self._directory(asset_id) / "decisions.json", {"decisions": {}})["decisions"]
        catalog, profiles = _allowed(self.library, asset_id, catalog=_catalog)
        registry = CharacterStore(self.library)._read(catalog["scope_id"])
        trusted = self.gallery.active_reference_ids(asset_id)
        matching = self.gallery.active_reference_ids(asset_id, include_candidates=True)
        result = copy.deepcopy(regions)
        for row in result:
            identifier = safe_id(row["id"])
            # Absolute paths persisted by an earlier machine/backup are never
            # authority to read a file. Reconstruct from the owning asset/id.
            row["crop_path"] = str(confined_path(self._directory(asset_id), f"crops/{identifier}.png"))
            decision = decisions.get(row["id"])
            if decision:
                row["match"].update({key: copy.deepcopy(value) for key, value in decision.items()
                                      if key in {"status", "character_id", "candidate_character_id", "source", "reason", "version"}})
            elif row["match"].get("candidates"):
                used = row["match"]["candidates"][0].get("reference_ids", [])
                if (row["match"].get("source") == "ccip_unverified_reference"
                        and used and set(used).issubset(matching)):
                    row["match"].update(status="candidate", character_id=None,
                                        candidate_character_id=row["match"]["candidates"][0]["character_id"])
                elif not used or not set(used).issubset(trusted):
                    row["match"].update(status="unknown", character_id=None, candidate_character_id=None,
                                        reason="参考图已撤销或不再可信；身份待重新核对")
            had_identity = bool(row["match"].get("character_id") or row["match"].get("candidate_character_id"))
            for key in ("character_id", "candidate_character_id"):
                if row["match"].get(key):
                    row["match"][key] = CharacterStore._resolve(registry, row["match"][key])
            for candidate in row["match"].get("candidates", []):
                candidate["character_id"] = CharacterStore._resolve(registry, candidate.get("character_id"))
            identity = row["match"].get("character_id")
            candidate = row["match"].get("candidate_character_id")
            if (identity and identity not in profiles or candidate and candidate not in profiles
                    or had_identity and not identity and not candidate):
                row["match"].update(status="unknown", character_id=None, candidate_character_id=None,
                                    reason="角色已删除或不再属于当前作品")
            if row["match"].get("character_id"):
                row["match"]["character_name"] = profiles[row["match"]["character_id"]]["name"]
            row["crop_url"] = f"/api/assets/{asset_id}/identity/regions/{row['id']}/image"
        return result

    def project_result(self, asset_id, result, *, frame_ids=None):
        """Refresh current mappings, optionally before copying unrelated regions."""
        if frame_ids is not None:
            current = set(frame_ids)
            result = {**result, "regions": [row for row in result.get("regions", []) if row["frame_id"] in current]}
        projected = copy.deepcopy(result)
        if projected.get("status") != "completed":
            return projected
        catalog, profiles = _allowed(self.library, asset_id)
        projected["regions"] = self._project(asset_id, projected.get("regions", []), _catalog=catalog)
        selected = list(dict.fromkeys(value for row in projected["regions"] for value in [
            row["match"].get("character_id"), row["match"].get("candidate_character_id"),
            *[candidate["character_id"] for candidate in row["match"].get("candidates", [])]]
            if row["match"].get("status") in {"candidate", "confirmed"} and value in profiles))
        projected["selected_profiles"] = [profiles[value] for value in selected]
        used_references = {identity for row in projected["regions"]
                           if row["match"].get("status") in {"candidate", "confirmed"}
                           for candidate in row["match"].get("candidates", [])
                           for identity in candidate.get("reference_ids", [])}
        # Confirmed exemplars remain available for every selected character.
        # Candidate images need support from these regions, never another shot.
        projected["references"] = [{"character_id": row["character_id"], "reference_id": row["id"],
                                    "path": row["path"], "reference_only": True,
                                    "reference_status": row["status"], "human_confirmed": row["status"] == "confirmed"}
                                   for row in self.gallery.matching(asset_id, _catalog=catalog) if row["character_id"] in selected
                                   and (row["status"] == "confirmed" or row["id"] in used_references)]
        return projected

    def reviews(self, asset_id):
        self.library.get_asset(asset_id)
        latest = read_json(self._directory(asset_id) / "latest.json", {"asset_id": asset_id, "status": "not_run",
                           "regions": [], "metrics": {}, "degraded_reason": "尚未运行视觉身份识别"})
        # Include successful partial batches even if a later operation stopped.
        regions = [read_json(path) for path in sorted((self._directory(asset_id) / "regions").glob("*.json"))]
        latest["regions"] = self._project(asset_id, regions)
        return latest

    def review(self, asset_id, region_id, *, status, character_id=None, reason):
        if status not in {"confirmed", "candidate", "unknown"} or not str(reason).strip():
            raise ValueError("人物身份修改须填写状态和原因")
        with self.library.lock:
            catalog, profiles = _allowed(self.library, asset_id)
            if character_id is not None:
                character_id = CharacterStore._resolve(CharacterStore(self.library)._read(catalog["scope_id"]),
                                                       character_id, strict=True)
            region = read_json(self._directory(asset_id) / "regions" / f"{safe_id(region_id)}.json")
            if region is None:
                raise FileNotFoundError("人物区域不存在")
            if status != "unknown" and character_id not in profiles:
                raise ValueError("请选择当前作品中的角色档案")
            saved = read_json(self._directory(asset_id) / "decisions.json", {"revision": 0, "decisions": {}, "audit": []})
            before = saved["decisions"].get(region_id)
            decision = {"status": status, "character_id": character_id if status == "confirmed" else None,
                        "candidate_character_id": character_id if status == "candidate" else None,
                        "source": "human", "reason": str(reason).strip(), "updated_at": now(),
                        "version": (before or {}).get("version", 0) + 1}
            saved["decisions"][region_id] = decision
            saved["revision"] += 1
            saved["audit"].append({"region_id": region_id, "before": before, "after": copy.deepcopy(decision)})
            atomic_json(self._directory(asset_id) / "decisions.json", saved)
            return self._project(asset_id, [region])[0]


def decorate_identity_records(library, asset_id, records, *, strict=False):
    """Apply explicit region bindings only; never merge entities by appearance.

    A model must retain identity_region_id or the pipeline's region entity_id.
    Nearby timestamps, similar descriptions, or being the only person cannot
    establish a binding. Event actor/target IDs remain local and unchanged.
    """
    from .identity_bindings import BindingResolver, uses_policy
    directory = library.asset_dir(asset_id) / "identity" / "visual"
    decisions = read_json(directory / "decisions.json", {"decisions": {}})["decisions"]
    if not decisions and not (directory / "regions").is_dir() and not any(uses_policy(row) for row in records):
        return records
    gallery = ReferenceGallery(library)
    trusted = gallery.active_reference_ids(asset_id)
    matching = gallery.active_reference_ids(asset_id, include_candidates=True)
    result = copy.deepcopy(records)
    resolver = None
    for index, record in enumerate(result):
        if uses_policy(record):
            resolver = resolver or BindingResolver(library, asset_id)
            result[index] = resolver.resolve(record)
            continue
        for entity in record.get("entities", []):
            identity = entity.get("identity_region_id", entity.get("id"))
            decision = decisions.get(identity)
            region = read_json(directory / "regions" / f"{safe_id(identity)}.json") if identity else None
            if region and "evidence_frame_ids" in record and region["frame_id"] not in (
                    entity.get("evidence_frame_ids") or record["evidence_frame_ids"]):
                if strict:
                    raise ValueError("身份区域不属于该人物的当前画面证据")
                continue
            if not decision and region:
                used = (region["match"].get("candidates") or [{}])[0].get("reference_ids", [])
                if (region["match"].get("source") == "ccip_unverified_reference"
                        and used and set(used).issubset(matching)):
                    decision = {"character_id": None, "status": "candidate", "source": "ccip_unverified_reference",
                                "candidate_character_id": region["match"]["candidates"][0]["character_id"],
                                "reason": "Bangumi 参考图只提供未验证角色候选，不能确认当前人物身份"}
                elif used and not set(used).issubset(trusted):
                    decision = {"character_id": None, "status": "unknown", "source": "reference_revoked",
                                "reason": "参考图已撤销或不再可信；身份待重新核对"}
            if decision:
                entity.update(character_id=decision["character_id"], identity_status=decision["status"],
                              association_status=decision["status"], association_source=decision.get("source", "human"),
                              candidate_character_id=decision.get("candidate_character_id"),
                              identity_source=decision.get("source", "human"), identity_reason=decision["reason"])
                if not decision["character_id"]:
                    entity.update(name=None, character_name=None, aliases=[])
    return result


def validate_visual_artifacts(library):
    """Validate restored references/regions before publishing an imported ZIP."""
    from .character_confirmation_validation import validate_character_confirmation_artifacts

    for location in (library.root / "works").glob("*/reference-images/gallery.json"):
        value = read_json(location)
        scope = safe_id(location.parent.parent.name)
        registry = CharacterStore(library)._read(scope)
        if (not isinstance(value, dict) or value.get("scope_id") != scope
                or type(value.get("revision")) is not int or value["revision"] < 0
                or not isinstance(value.get("references"), list) or not isinstance(value.get("audit"), list)):
            raise ValueError("参考图库备份格式无效")
        seen = set()
        for reference in value["references"]:
            identity = safe_id(reference["id"])
            if identity in seen:
                raise ValueError("参考图 ID 重复")
            seen.add(identity)
            CharacterStore._resolve(registry, reference["character_id"], strict=True)
            if (reference.get("status") not in {"confirmed", "candidate", "revoked"}
                    or type(reference.get("version")) is not int or reference["version"] < 1
                    or not isinstance(reference.get("quality"), dict)
                    or type(reference["quality"].get("accepted")) is not bool
                    or reference.get("source", {}).get("kind") not in {"upload", "external_cached", "frame_crop"}):
                raise ValueError("参考图确认或来源信息无效")
            path = confined_path(location.parent, f"{identity}.png")
            content = path.read_bytes()
            _image(content)
            if hashlib.sha256(content).hexdigest() != reference.get("sha256"):
                raise ValueError("参考图内容校验失败")
            source = reference["source"]
            if source["kind"] == "frame_crop":
                library.get_asset(source["asset_id"], include_deleted=True)
                library.frame_path(source["asset_id"], source["frame_id"])
                if checked_box(source.get("box")) is None:
                    raise ValueError("参考图裁剪来源缺少区域")
    for directory in (library.root / "assets").glob("*/identity/visual"):
        asset_id = safe_id(directory.parent.parent.name)
        library.get_asset(asset_id, include_deleted=True)
        frame_registry = read_json(library.asset_dir(asset_id) / "frames/registry.json", {})
        values = [read_json(path) for path in directory.glob("regions/*.json")]
        values.extend(read_json(path) for path in directory.glob("region-history/*/*.json"))
        latest = read_json(directory / "latest.json", {})
        values.extend(latest.get("regions", []))
        for path in library.asset_dir(asset_id).glob("analyses/*/identity-prelude.json"):
            values.extend(read_json(path, {}).get("visual", {}).get("regions", []))
        for row in values:
            identity = safe_id(row["id"])
            if row.get("entity_id") != identity or checked_box(row.get("box")) is None:
                raise ValueError("人物区域备份无效")
            frame_id = safe_id(row["frame_id"])
            if row["at_ms"] != frame_registry.get(frame_id, {}).get("at_ms"):
                raise ValueError("人物区域备份时间与证据帧不一致")
            library.frame_path(asset_id, frame_id)
            _image(confined_path(directory, f"crops/{identity}.png").read_bytes())
            if row.get("match", {}).get("status") not in {"confirmed", "candidate", "unknown"}:
                raise ValueError("人物身份关联状态无效")
        decisions = read_json(directory / "decisions.json", {"decisions": {}, "audit": []})
        if not isinstance(decisions.get("decisions"), dict) or not isinstance(decisions.get("audit"), list):
            raise ValueError("人物身份修订备份无效")
        for identity, decision in decisions["decisions"].items():
            if (not (directory / "regions" / f"{safe_id(identity)}.json").is_file()
                    or decision.get("status") not in {"confirmed", "candidate", "unknown"}):
                raise ValueError("人工身份修订缺少原始区域")
    validate_character_confirmation_artifacts(library)
