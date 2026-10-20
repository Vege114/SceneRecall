"""Immutable retrieval evidence with an atomic, independently rebuildable projection."""
from __future__ import annotations

import copy

from .library import SCHEMA, atomic_json, confined_path, digest, now, read_json, safe_id

STRATEGY = "retrieval-first-v1"


class RetrievalStore:
    def __init__(self, library):
        self.library = library

    def directory(self, asset_id):
        return self.library.asset_dir(asset_id) / "retrieval"

    def validate(self, asset_id, record):
        asset = self.library.get_asset(asset_id, include_deleted=True)
        if (record.get("asset_id") != asset_id or record.get("source_fingerprint") != asset["fingerprint"]
                or record.get("strategy") != STRATEGY):
            raise ValueError("检索记录来源或策略不匹配")
        safe_id(record["id"])
        if (type(record.get("start_ms")) is not int or type(record.get("end_ms")) is not int
                or not 0 <= record["start_ms"] < record["end_ms"] <= asset["duration_ms"]):
            raise ValueError("检索记录范围无效")
        frames = record.get("ordered_frames", [])
        ids = [f["id"] for f in frames]
        if (not frames or len(set(ids)) != len(ids) or ids != record.get("evidence_frame_ids")
                or any(type(f.get("at_ms")) is not int or not 0 <= f["at_ms"] < asset["duration_ms"] for f in frames)
                or [f["at_ms"] for f in frames] != sorted(f["at_ms"] for f in frames)):
            raise ValueError("检索记录的有序源帧无效")
        self.library.frame_paths(asset_id, ids)
        registry = read_json(self.library.asset_dir(asset_id) / "frames" / "registry.json", {})
        if any(registry[f["id"]]["at_ms"] != f["at_ms"] for f in frames):
            raise ValueError("检索记录改变了原始证据帧时间")
        for frame in frames:
            if record["start_ms"] <= frame["at_ms"] < record["end_ms"]:
                continue
            held = record.get("held_frame_spans", [])
            if not (len(frames) == 1 and frame["at_ms"] < record["start_ms"]
                    and held == [{"frame_id": frame["id"], "start_ms": record["start_ms"], "end_ms": record["end_ms"]}]):
                raise ValueError("范围外源帧缺少明确的持帧覆盖证据")
        return record

    def publish(self, asset_id, run_id, record):
        record = self.validate(asset_id, {**copy.deepcopy(record), "run_id": safe_id(run_id)})
        root = self.library.asset_dir(asset_id)
        path = root / "analyses" / run_id / "retrieval-units" / f"{safe_id(record['id'])}.json"
        with self.library.lock:
            previous = read_json(path)
            if previous is not None and previous != record:
                raise ValueError("不可覆盖已保存的检索观察，请创建新运行")
            if previous is None:
                atomic_json(path, record)
            active_path = self.directory(asset_id) / "active.json"
            active = read_json(active_path, {"schema_version": SCHEMA, "retrieval_version": 1, "units": {}})
            prior_entry = active["units"].get(record["id"], {})
            active["units"][record["id"]] = {"path": path.relative_to(root).as_posix(), "digest": digest(record),
                                             "start_ms": record["start_ms"], "end_ms": record["end_ms"]}
            if prior_entry.get("digest") == digest(record):
                for key in ("observation", "identity"):
                    if prior_entry.get(key):
                        active["units"][record["id"]][key] = prior_entry[key]
            atomic_json(active_path, active)
        return record

    def records(self, asset_id):
        active = read_json(self.directory(asset_id) / "active.json", {"units": {}})
        result = []
        for identity, entry in active.get("units", {}).items():
            record = read_json(confined_path(self.library.asset_dir(asset_id), entry["path"]))
            if record is None or record.get("id") != identity or digest(record) != entry["digest"]:
                raise ValueError("检索记录激活引用不完整")
            if entry.get("observation"):
                observed = entry["observation"]
                observation = read_json(confined_path(self.library.asset_dir(asset_id), observed["path"]))
                if observation is None or digest(observation) != observed["digest"] or observation["unit_id"] != identity:
                    raise ValueError("局部观察激活引用无效")
                record = {**record, **{key: observation[key] for key in ("summary", "entities", "events",
                    "spatial_observations", "uncertainties", "identity_policy", "provenance") if key in observation}}
            if entry.get("identity"):
                from .retrieval_identity import merge
                reference = entry["identity"]
                identity_observation = read_json(confined_path(self.library.asset_dir(asset_id), reference["path"]))
                if (identity_observation is None or digest(identity_observation) != reference["digest"]
                        or identity_observation.get("unit_id") != identity
                        or identity_observation.get("window_digest") != entry["digest"]):
                    raise ValueError("本地人物观察激活引用无效")
                record = merge(record, identity_observation)
            result.append(copy.deepcopy(record))
        from .identity_bindings import resolve_observations
        return resolve_observations(self.library, asset_id, sorted(result, key=lambda row: (row["start_ms"], row["id"])))

    def windows(self, asset_id):
        """Read immutable source windows without applying any derived identity state."""
        active = read_json(self.directory(asset_id) / "active.json", {"units": {}})
        result = []
        for identity, entry in active["units"].items():
            window = read_json(confined_path(self.library.asset_dir(asset_id), entry["path"]))
            if not window or window.get("id") != identity or digest(window) != entry["digest"]:
                raise ValueError("检索记录激活引用不完整")
            result.append(self.validate(asset_id, window))
        return sorted(result, key=lambda row: (row["start_ms"], row["id"]))

    def publish_identity(self, asset_id, run_id, observation):
        """Publish a new sidecar; never replace the window or its generative receipt."""
        from .retrieval_identity import VERSION
        root = self.library.asset_dir(asset_id)
        unit_id = safe_id(observation["unit_id"])
        path = root / "analyses" / safe_id(run_id) / "retrieval-identities" / f"{unit_id}.json"
        with self.library.lock:
            active_path = self.directory(asset_id) / "active.json"
            active = read_json(active_path, {"units": {}})
            entry = active["units"].get(unit_id)
            if (not entry or observation.get("window_digest") != entry["digest"]
                    or observation.get("asset_id") != asset_id
                    or observation.get("identity_observation_version") != VERSION
                    or observation.get("identity_status") not in {"completed", "disabled"}):
                raise ValueError("本地人物观察与已发布画面窗口不匹配")
            window = read_json(confined_path(root, entry["path"]))
            if not window or digest(window) != entry["digest"]:
                raise ValueError("检索记录激活引用不完整")
            if (observation.get("source_fingerprint") != window["source_fingerprint"]
                    or observation.get("identity_policy") != "identity-binding-v2"
                    or observation.get("evidence_frame_ids") != window["evidence_frame_ids"]):
                raise ValueError("人物观察不能改变原片或画面证据")
            local = set(window["local_frame_ids"])
            if any(not entity.get("identity_region_id")
                   or not set(entity.get("evidence_frame_ids", [])).issubset(local)
                   or not entity.get("evidence_frame_ids") for entity in observation["entities"]):
                raise ValueError("人物区域必须属于当前窗口的局部证据帧")
            previous = read_json(path)
            published = entry.get("identity")
            if (published and published["path"] == path.relative_to(root).as_posix()
                    and (previous is None or digest(previous) != published["digest"])):
                raise ValueError("已保存的原始人物观察损坏，不能重新认证其哈希")
            if previous is not None and previous != observation:
                raise ValueError("不可覆盖原始本地人物观察")
            atomic_json(path, observation)
            entry["identity"] = {"path": path.relative_to(root).as_posix(), "digest": digest(observation)}
            atomic_json(active_path, active)

    def publish_observation(self, asset_id, run_id, observation):
        root = self.library.asset_dir(asset_id)
        identity = safe_id(observation["unit_id"])
        path = root / "analyses" / safe_id(run_id) / "retrieval-observations" / f"{identity}.json"
        with self.library.lock:
            active_path = self.directory(asset_id) / "active.json"
            active = read_json(active_path, {"units": {}})
            if identity not in active["units"]:
                raise ValueError("局部观察对应的画面窗口未发布")
            entry = active["units"][identity]
            window = read_json(confined_path(root, entry["path"]))
            if window.get("run_id") != run_id:
                raise ValueError("局部观察不能覆盖其他任务的已发布窗口")
            prior = read_json(path)
            observed = entry.get("observation")
            if (observed and observed["path"] == path.relative_to(root).as_posix()
                    and (prior is None or digest(prior) != observed["digest"])):
                raise ValueError("已保存的原始局部观察损坏，不能重新认证其哈希")
            if prior is not None and prior != observation:
                raise ValueError("不可覆盖原始局部观察")
            atomic_json(path, observation)
            entry["observation"] = {"path": path.relative_to(root).as_posix(), "digest": digest(observation)}
            atomic_json(active_path, active)

    def finalize(self, asset_id, run_id, start_ms, end_ms, unit_ids):
        with self.library.lock:
            path = self.directory(asset_id) / "active.json"
            active = read_json(path, {"schema_version": SCHEMA, "retrieval_version": 1, "units": {}})
            if not set(unit_ids).issubset(active["units"]):
                raise ValueError("检索范围尚未全部发布")
            active["units"] = {key: value for key, value in active["units"].items()
                               if key in unit_ids or not start_ms <= value["start_ms"] < value["end_ms"] <= end_ms}
            active["last_run"] = {"run_id": run_id, "start_ms": start_ms, "end_ms": end_ms, "published_at": now()}
            atomic_json(path, active)

    def validate_all(self):
        for asset in self.library.list_assets():
            for record in self.records(asset["id"]):
                self.validate(asset["id"], record)
