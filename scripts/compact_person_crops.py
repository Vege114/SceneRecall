"""Estimate, verify, or compact archived ordinary person crops or identity JSON, per asset."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import stat
import sys

from PIL import Image
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall.crop_artifacts import MAX_CROP_SOURCE_BYTES, rebuild_crop_content
from scenerecall.identity_storage import compact_identity_snapshot
from scenerecall.library import atomic_json, confined_path, read_json, safe_id
from scenerecall.library_activity import library_activity
from scenerecall.service import _launch_options


def alias(info):
    return (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def plain_path(root, path):
    """A crop, its source, and the metadata we update must have ordinary paths."""
    for part in (path, *path.parents):
        info = part.lstat()
        if alias(info):
            raise ValueError("路径是链接或重解析点")
        if part == root:
            return path
    raise ValueError("路径越出资料库")


def recipe(row):
    fields = ("id", "frame_id", "at_ms", "box", "source_fingerprint",
              "crop_box_pixels", "source_frame_sha256", "crop_sha256")
    return {key: row[key] for key in fields if key in row}


def saved_recipes(asset):
    """Old receipts and latest.json can still be used by resume or restore."""
    visual = asset / "identity" / "visual"
    sources = [visual / "latest.json"]
    for filename in ("identity-prelude.json", "retrieval-local-identity.json"):
        sources.extend((asset / "analyses").glob(f"*/{filename}"))
    rows = defaultdict(list)
    for path in sources:
        if not path.exists():
            continue
        data = read_json(plain_path(asset, path))
        visual_data = data if path == sources[0] else data.get("visual", {})
        for row in visual_data.get("regions", []):
            identity = safe_id(row["id"])
            variant = recipe(row)
            if variant not in rows[identity]:
                rows[identity].append(variant)
    return rows


def check_legacy_services(root):
    # Earlier builds do not participate in the maintenance lock. Also recognize
    # the published console-script and Windows launcher command lines.
    for process in psutil.process_iter(["cmdline"]):
        argv = process.info["cmdline"] or []
        if "-m" in argv and argv[argv.index("-m") + 1:][:1] == ["scenerecall.main"]:
            options = argv[argv.index("-m") + 2:]
        elif len(argv) > 1 and Path(argv[1]).name.lower() in {"scenerecall", "scenerecall.exe"}:
            options = argv[2:]
        elif argv and Path(argv[0]).name.lower() in {"scenerecall", "scenerecall.exe"}:
            options = argv[1:]
        else:
            continue
        try:
            env = process.environ()
            home = Path(env.get("USERPROFILE" if os.name == "nt" else "HOME") or Path.home())
            launch = _launch_options(options, Path(process.cwd()), home)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError("无法核实旧版 SceneRecall 进程，请先关闭它。") from exc
        if launch and launch[0] == root:
            raise RuntimeError("资料库正在使用。请先停止 SceneRecall 服务。")


def verified_region(asset, manifest, registry, variants, crop):
    region_id = safe_id(crop.stem)
    visual = asset / "identity" / "visual"
    region_path = plain_path(asset, visual / "regions" / f"{region_id}.json")
    raw = read_json(region_path)
    if raw["id"] != region_id or raw.get("entity_id") != region_id:
        raise ValueError("持久区域与文件名不一致")
    frame_id = safe_id(raw["frame_id"])
    frame = registry[frame_id]
    source = confined_path(asset, frame["path"])
    # Only independent, archived original frames can replace the ordinary PNG.
    # Do not rely on media paths, other crops, model caches, or a linked alias.
    raw_source = asset / frame["path"].replace("\\", "/")
    plain_path(asset, raw_source)
    if source.parent != asset / "frames" or source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("来源不是库内独立原帧")
    if not source.is_file() or source.stat().st_size > MAX_CROP_SOURCE_BYTES:
        raise ValueError("原帧不可用")
    frame_content = source.read_bytes()
    if crop.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("裁剪超过文件大小上限")
    stored = crop.read_bytes()
    if not stored:
        raise ValueError("裁剪为空")
    history = visual / "region-history" / region_id
    current_recipe = recipe(raw)
    recipes = [current_recipe, *(variant for variant in variants if variant != current_recipe)]
    if history.exists():
        for path in plain_path(asset, history).glob("*.json"):
            variant = recipe(read_json(plain_path(asset, path)))
            if variant not in recipes:
                recipes.append(variant)
    for row in recipes:
        if (row.get("id") != region_id or row.get("frame_id") != frame_id
                or type(row.get("at_ms")) is not int or row["at_ms"] != frame["at_ms"]
                or row.get("source_fingerprint", manifest["fingerprint"]) != manifest["fingerprint"]):
            raise ValueError("来源或历史区域不一致")
        if rebuild_crop_content(frame_content, row) != stored:
            raise ValueError("重建裁剪与留存文件的字节不同")
    raw.update(crop_retention="discard", source_fingerprint=manifest["fingerprint"],
               source_frame_sha256=hashlib.sha256(frame_content).hexdigest(),
               crop_sha256=hashlib.sha256(stored).hexdigest())
    return region_path, raw


def frozen_crop_names(root):
    """Keep exact filenames that an old frozen model request can reopen on retry."""
    names = set()
    for directory in ("scene-inputs", "scene-input-history"):
        for path in (root / "assets").glob(f"*/analyses/*/{directory}/*.json"):
            frozen = read_json(plain_path(root, path))
            if not isinstance(frozen, dict) or not isinstance(frozen.get("args"), list):
                raise ValueError(f"无法核实已归档的场景输入: {path.relative_to(root)}")
            pending = [frozen["args"]]
            while pending:
                item = pending.pop()
                if isinstance(item, list):
                    pending.extend(item)
                elif isinstance(item, dict):
                    value = item.get("path")
                    if isinstance(value, str):
                        name = PureWindowsPath(value).name.casefold()
                        if name.startswith("region_") and name.endswith(".png"):
                            names.add(name)
                    pending.extend(item.values())
    return names


def compact_asset(root, asset, mode, frozen_names):
    plain_path(root, asset)
    manifest = read_json(plain_path(asset, asset / "manifest.json"))
    if manifest["id"] != asset.name or not manifest.get("fingerprint"):
        raise ValueError("资产清单无效")
    result = {"asset_id": asset.name, "candidate_count": 0, "candidate_bytes": 0,
              "verified_count": 0, "verified_bytes": 0, "removed_count": 0, "removed_bytes": 0,
              "retained_count": 0, "retained_examples": []}
    crops = asset / "identity" / "visual" / "crops"
    if not crops.exists():
        return result
    candidates = sorted(plain_path(asset, crops).glob("region_*.png"))
    if candidates and mode != "estimate":
        registry = read_json(plain_path(asset, asset / "frames" / "registry.json"))
        registered = {confined_path(asset, frame["path"]) for frame in registry.values()}
        old_recipes = saved_recipes(asset)
    for crop in candidates:
        try:
            info = crop.lstat()
            if alias(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("裁剪不是独立普通文件")
            result["candidate_count"] += 1
            result["candidate_bytes"] += info.st_size
            if mode == "estimate":
                continue
            if crop.name.casefold() in frozen_names:
                raise ValueError("历史场景请求仍按文件名引用这份裁剪")
            if crop in registered:
                raise ValueError("该图片同时作为已登记原帧使用")
            region_path, raw = verified_region(asset, manifest, registry, old_recipes[crop.stem], crop)
        except (KeyError, ValueError, OSError, TypeError, Image.DecompressionBombError) as exc:
            result["retained_count"] += 1
            if len(result["retained_examples"]) < 5:
                result["retained_examples"].append({"crop": crop.name, "reason": str(exc)})
            continue
        result["verified_count"] += 1
        result["verified_bytes"] += info.st_size
        if mode == "apply":
            # Persist checksums before unlinking. After interruption a remaining
            # PNG is reverified on retry, and an absent one is previewable.
            atomic_json(region_path, raw)
            crop.unlink()
            result["removed_count"] += 1
            result["removed_bytes"] += info.st_size
    return result


def compact_json_asset(root, asset, mode):
    plain_path(root, asset)
    manifest = read_json(plain_path(asset, asset / "manifest.json"))
    if manifest.get("id") != asset.name:
        raise ValueError("素材清单与路径不匹配")
    latest = asset / "identity" / "visual" / "latest.json"
    paths = ([latest] if latest.is_file() else []) + sorted((asset / "analyses").glob("*/retrieval-local-identity.json"))
    result = {"asset_id": asset.name, "candidate_count": 0, "candidate_bytes": 0,
              "verified_savings_bytes": 0, "saved_bytes": 0, "retained_count": 0, "retained_examples": []}
    for path in paths:
        try:
            info = plain_path(asset, path).stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("快照不是独立普通文件")
            result["candidate_count"] += 1
            result["candidate_bytes"] += info.st_size
            if mode != "estimate":
                saved = compact_identity_snapshot(path, asset.name, apply=mode == "apply")
                result["verified_savings_bytes"] += saved
                if mode == "apply":
                    result["saved_bytes"] += saved
        except (ValueError, OSError) as exc:
            result["retained_count"] += 1
            if len(result["retained_examples"]) < 5:
                result["retained_examples"].append({"file": path.relative_to(asset).as_posix(), "reason": str(exc)})
    return result


def run(root, asset_ids=(), *, mode="estimate", identity_json_only=False):
    if mode not in {"estimate", "verify", "apply"}:
        raise ValueError("未知整理模式")
    root = Path(root).resolve(strict=True)
    if not read_json(plain_path(root, root / "library.json")).get("id"):
        raise ValueError("资料库清单无效")
    assets = plain_path(root, root / "assets")
    directories = ([assets / safe_id(identity) for identity in dict.fromkeys(asset_ids)] if asset_ids
                   else sorted(path for path in assets.iterdir() if path.is_dir()))
    if mode == "apply":
        runtime = plain_path(root, root / "runtime")
        lock = runtime / "library-activity.sqlite"
        if lock.exists() or lock.is_symlink():
            plain_path(root, lock)
    with library_activity(root, maintenance=True) if mode == "apply" else nullcontext():
        if mode == "apply":
            check_legacy_services(root)
        if identity_json_only:
            results = [compact_json_asset(root, asset, mode) for asset in directories]
        else:
            frozen = frozen_crop_names(root) if mode != "estimate" else set()
            results = [compact_asset(root, asset, mode, frozen) for asset in directories]
    totals = Counter()
    for result in results:
        totals.update({key: value for key, value in result.items() if isinstance(value, int)})
    return {"mode": mode, "assets": results, "totals": dict(totals)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--asset", action="append", default=[], help="Limit to this asset ID (repeatable)")
    parser.add_argument("--identity-json-only", action="store_true", help="Select only visual snapshots and retrieval identity JSON receipts")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", action="store_true", help="Check the selected files and report exact savings without writing")
    group.add_argument("--apply", action="store_true", help="Verify and compact the selected files. Stop library writers first.")
    args = parser.parse_args()
    mode = "apply" if args.apply else "verify" if args.verify else "estimate"
    print(json.dumps(run(args.library, args.asset, mode=mode, identity_json_only=args.identity_json_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
