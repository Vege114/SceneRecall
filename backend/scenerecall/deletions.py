"""Recoverable application deletion, with explicit irreversible source deletion.

Catalog after-images are committed before touching a source. Private source
intents never enter portable backups. Interrupted unlink operations are inspected,
never replayed against a path which may now contain another file.
"""
from __future__ import annotations

import copy
import os
import stat
from pathlib import Path

from .library import atomic_json, digest, now, read_json, safe_id, uid
from .organization import OrganizationConflict


BUSY = {"running", "pausing", "cancelling"}


class DeletionStore:
    def __init__(self, library, queue=None, search=None):
        self.library, self.queue, self.search = library, queue, search
        self.organization = library.organization
        self.source_intents = library.root / "private" / "source-deletions.json"

    @staticmethod
    def _identity(path):
        value = path.lstat()
        if not stat.S_ISREG(value.st_mode) or path.resolve() != path:
            raise ValueError("源文件不是普通文件或包含符号链接")
        return {"dev": value.st_dev, "ino": value.st_ino, "size": value.st_size, "mtime_ns": value.st_mtime_ns}

    @staticmethod
    def _unlink_source(path):
        os.unlink(path)

    def _scope(self, catalog, kind, asset_ids=None, id=None, delete_contents=False):
        active = {key: {identity: item for identity, item in catalog[key].items() if not item.get("trash_id")}
                  for key in ("assets", "projects", "folders")}
        assets, projects, folders = [], [], []
        if kind == "assets":
            assets = list(dict.fromkeys(asset_ids or []))
            if not assets:
                raise ValueError("请选择至少一个视频")
            if any(identity not in active["assets"] for identity in assets):
                raise FileNotFoundError("视频不存在或已在回收站")
            title = "删除视频"
        elif kind == "project":
            if id not in active["projects"]:
                raise FileNotFoundError("项目不存在或已在回收站")
            projects = [id]
            title = active["projects"][id]["name"]
            if delete_contents:
                assets = [identity for identity, item in active["assets"].items() if item["project_id"] == id]
        elif kind == "folder":
            if id not in active["folders"]:
                raise FileNotFoundError("文件夹不存在或已在回收站")
            title = active["folders"][id]["name"]
            folders = [id]
            if delete_contents:
                for folder in folders:
                    folders.extend(identity for identity, item in active["folders"].items()
                                   if item["parent_id"] == folder and identity not in folders)
                projects = [identity for identity, item in active["projects"].items() if item["folder_id"] in folders]
                assets = [identity for identity, item in active["assets"].items() if item["project_id"] in projects]
        else:
            raise ValueError("删除类型无效")
        # Membership changes when deleting a project while retaining its videos.
        affected = assets or ([identity for identity, item in active["assets"].items()
                              if item["project_id"] in projects] if not delete_contents else [])
        return {"asset_ids": sorted(assets), "project_ids": sorted(projects), "folder_ids": sorted(folders),
                "affected_asset_ids": sorted(affected), "title": title}

    def _jobs(self, catalog, scope):
        scopes = {catalog["assets"][identity]["character_scope_id"] for identity in scope["affected_asset_ids"]}
        affected = []
        for job in self.queue.list() if self.queue else []:
            membership = catalog["assets"].get(job.get("asset_id"), {})
            if (job.get("asset_id") in scope["affected_asset_ids"] or job["type"] == "index"
                    or (job["type"] == "analysis" and "vision" in job.get("config", {}).get("stages", [])
                        and membership.get("character_scope_id") in scopes)
                    or (job["type"] == "import" and any(item.get("project_id") in scope["project_ids"]
                         for item in job.get("config", {}).get("items", [])))):
                affected.append(job)
        return affected

    def _sources(self, catalog, assets, delete_source, locations):
        from .media import fingerprint
        selected = set(assets)
        groups = {}
        for asset_id in assets:
            location = locations.get(asset_id)
            path = location.get("path") if isinstance(location, dict) else location
            groups.setdefault(path, []).append(asset_id)
        result = []
        for raw_path, asset_ids in groups.items():
            item = {"asset_ids": asset_ids, "path": raw_path or "", "status": "ready", "message": "原始视频文件"}
            result.append(item)
            if not raw_path:
                item.update(status="missing", message="原视频离线，无法删除源文件")
                continue
            path = Path(raw_path)
            try:
                identity = self._identity(path)
                item["identity"] = identity
                shared = []
                for other, location in locations.items():
                    if other in selected or other not in catalog["assets"]:
                        continue
                    other_path = Path(location.get("path") if isinstance(location, dict) else location)
                    try:
                        other_stat = other_path.stat()
                        same = (other_stat.st_dev, other_stat.st_ino) == (identity["dev"], identity["ino"])
                    except OSError:
                        same = False
                    if other_path.resolve() == path.resolve() or same:
                        shared.append(other)
                if shared:
                    item.update(status="shared", message="该源文件还被未选中的视频使用，请先处理共享引用")
                    continue
                checksums = {read_json(self.library.asset_dir(asset_id) / "manifest.json")["fingerprint"] for asset_id in asset_ids}
                snapshot_ok = all(isinstance(locations.get(asset_id), dict)
                                  and locations[asset_id].get("size") == identity["size"]
                                  and locations[asset_id].get("mtime_ns") == identity["mtime_ns"] for asset_id in asset_ids)
                checksum = fingerprint(path) if delete_source else next(iter(checksums))
                if not snapshot_ok or len(checksums) != 1 or checksum not in checksums or self._identity(path) != identity:
                    item.update(status="changed", message="原文件已变化，请先重新定位并校验")
                    continue
                item["fingerprint"] = checksum
            except FileNotFoundError:
                item.update(status="missing", message="原视频离线，无法删除源文件")
            except (OSError, ValueError):
                item.update(status="unsafe", message="源文件不是可安全删除的普通文件，或当前无法读取")
        return result

    def _prepare(self, kind, asset_ids=None, id=None, delete_contents=False, delete_source=False):
        with self.library.lock:
            self.organization.migrate()
            catalog = self.organization._read()
            scope = self._scope(catalog, kind, asset_ids, id, delete_contents)
            locations = read_json(self.library.root / "private" / "media-locations.json", {})
        if delete_source and not scope["asset_ids"]:
            raise ValueError("此操作保留视频，不能同时删除源文件")
        sources = self._sources(catalog, scope["asset_ids"], delete_source, locations)
        jobs = self._jobs(catalog, scope)
        assets = [{"id": identity, "title": read_json(self.library.asset_dir(identity) / "manifest.json")["title"]}
                  for identity in scope["asset_ids"]]
        projects = [{"id": identity, "name": catalog["projects"][identity]["name"]} for identity in scope["project_ids"]]
        folders = [{"id": identity, "name": catalog["folders"][identity]["name"]} for identity in scope["folder_ids"]]
        token = digest([catalog, locations, kind, scope, delete_contents, delete_source, sources])
        public = {"token": token, "kind": kind, "title": scope["title"], "asset_count": len(assets),
                  "project_count": len(projects), "folder_count": len(folders), "assets": assets,
                  "projects": projects, "folders": folders, "delete_contents": delete_contents,
                  "delete_source": delete_source, "source_files_preserved": not delete_source,
                  "source_file_count": len(sources), "source_files": [
                      {key: item[key] for key in ("asset_ids", "path", "status", "message")} for item in sources],
                  "blocked_jobs": [{"id": job["id"], "status": job["status"], "message": "请先暂停并等待任务停止"}
                                   for job in jobs if job["status"] in BUSY]}
        return public, catalog, locations, scope, sources

    def preview(self, **request):
        return self._prepare(**request)[0]

    def _evidence_digest(self, asset_ids):
        return digest({asset_id: {path.relative_to(self.library.asset_dir(asset_id)).as_posix(): read_json(path)
                                 for path in sorted([*self.library.asset_dir(asset_id).glob("analyses/*/observations/*.json"),
                                                     *self.library.asset_dir(asset_id).glob("analyses/*/scenes/*.json")])}
                       for asset_id in asset_ids})

    def delete(self, *, expected_token=None, **request):
        public, before, locations, scope, sources = self._prepare(**request)
        if (expected_token is not None and expected_token != public["token"]) or (public["delete_source"] and not expected_token):
            raise OrganizationConflict("删除范围或源文件已变化，请重新预览并确认")
        with self.library.lock:
            if before != self.organization._read() or locations != read_json(self.library.root / "private" / "media-locations.json", {}):
                raise OrganizationConflict("删除范围已变化，请重新预览")
            jobs = self._jobs(before, scope)
            if any(job["status"] in BUSY for job in jobs):
                raise OrganizationConflict("相关视频、角色或索引任务正在运行，请先暂停并等待当前步骤结束")
            if public["delete_source"]:
                if any(item["status"] != "ready" for item in sources):
                    raise OrganizationConflict("部分源文件离线、已变化或存在共享引用，请改为仅删除应用资料")
                try:
                    if any(self._identity(Path(item["path"])) != item["identity"] for item in sources):
                        raise ValueError("changed")
                except (OSError, ValueError) as exc:
                    raise OrganizationConflict("源文件已变化，请重新预览") from exc
            catalog = copy.deepcopy(before)
            registries = {}
            trash_id = uid("trash")
            if public["kind"] == "project" and not public["delete_contents"] and scope["affected_asset_ids"]:
                catalog, registries = self.organization.assign(scope["affected_asset_ids"], None, self.queue, _defer=True)
            if public["kind"] == "folder" and not public["delete_contents"]:
                folder_id = scope["folder_ids"][0]
                parent = catalog["folders"][folder_id]["parent_id"]
                for item in catalog["folders"].values():
                    if item["parent_id"] == folder_id and not item.get("trash_id"):
                        item["parent_id"] = parent
                for item in catalog["projects"].values():
                    if item["folder_id"] == folder_id and not item.get("trash_id"):
                        item["folder_id"] = parent
            for category in ("assets", "projects", "folders"):
                for identity in scope[category[:-1] + "_ids"]:
                    catalog[category][identity]["trash_id"] = trash_id
                    if category == "assets":
                        catalog[category][identity]["membership_revision"] += 1
            from .characters import CharacterStore
            store = CharacterStore(self.library)
            for identity in scope["affected_asset_ids"]:
                role_scope = before["assets"][identity]["character_scope_id"]
                if role_scope not in registries:
                    registries[role_scope] = store._read(role_scope)
                    registries[role_scope]["revision"] += 1
            catalog["deletion_revision"] = before.get("deletion_revision", 0) + 1
            for identity in scope["affected_asset_ids"]:
                catalog["assets"][identity]["deletion_revision"] = before["assets"][identity].get("deletion_revision", 0) + 1
            patches = {category: {identity: {"before": before[category][identity], "after": item}
                                  for identity, item in catalog[category].items() if item != before[category].get(identity)}
                       for category in ("assets", "projects", "folders")}
            entry = {key: public[key] for key in ("kind", "title", "asset_count", "project_count", "folder_count", "assets",
                     "projects", "folders", "delete_contents", "delete_source")}
            entry.update(id=trash_id, deleted_at=now(), patches=patches,
                         source_results=[{"asset_ids": item["asset_ids"], "status": "pending", "message": "等待删除源文件"}
                                         for item in sources] if public["delete_source"] else [],
                         registry_revisions={role_scope: registry["revision"] for role_scope, registry in registries.items()}
                         if public["kind"] == "project" and not public["delete_contents"] else {})
            if entry["registry_revisions"]:
                entry["registry_digests"] = {role_scope: digest(registry) for role_scope, registry in registries.items()}
                entry["evidence_digest"] = self._evidence_digest(scope["affected_asset_ids"])
            catalog.setdefault("trash", {})[trash_id] = entry
            if public["delete_source"]:
                intents = read_json(self.source_intents, {})
                intents[trash_id] = sources
                atomic_json(self.source_intents, intents)
            self.organization._commit(catalog, registries)
            if self.search:
                self.search.remove_assets(scope["asset_ids"])
            self._invalidate_jobs(jobs)
        if public["delete_source"]:
            self._delete_sources(trash_id, sources)
        return {"trash": self._public(self.organization._read()["trash"][trash_id]),
                "deleted_asset_ids": scope["asset_ids"], "deleted_project_ids": scope["project_ids"],
                "deleted_folder_ids": scope["folder_ids"]}

    def _invalidate_jobs(self, jobs):
        if self.queue:
            for job in jobs:
                if job["status"] not in {"completed", "running", "pausing", "cancelling"}:
                    self.queue.update(job["id"], status="cancelled", stale_deletion=True,
                                      message="相关资料已删除，历史任务不可继续；恢复资料后请创建新任务")

    def _set_source_result(self, trash_id, index, status, message):
        with self.library.lock:
            catalog = self.organization._read()
            result = catalog["trash"][trash_id]["source_results"][index]
            result.update(status=status, message=message)
            if status == "deleted":
                locations_path = self.library.root / "private" / "media-locations.json"
                locations = read_json(locations_path, {})
                for asset_id in result["asset_ids"]:
                    locations.pop(asset_id, None)
                atomic_json(locations_path, locations)
            self.organization._commit(catalog)

    def _delete_sources(self, trash_id, sources):
        from .media import fingerprint
        selected = {asset_id for item in sources for asset_id in item["asset_ids"]}
        for index, item in enumerate(sources):
            path = Path(item["path"])
            try:
                if (self._identity(path) != item["identity"] or fingerprint(path) != item["fingerprint"]
                        or self._identity(path) != item["identity"]):
                    raise ValueError("source changed")
                with self.library.lock:
                    # A relocate/import can add a reference during the long hash.
                    # Recheck live references and file identity while publication
                    # is excluded, then unlink only this explicitly previewed path.
                    locations = read_json(self.library.root / "private" / "media-locations.json", {})
                    catalog = self.organization._read()
                    for asset_id, location in locations.items():
                        if asset_id in selected or asset_id not in catalog["assets"]:
                            continue
                        other = Path(location["path"] if isinstance(location, dict) else location)
                        try:
                            info = other.stat()
                            same = (info.st_dev, info.st_ino) == (item["identity"]["dev"], item["identity"]["ino"])
                        except OSError:
                            same = False
                        if other.resolve() == path or same:
                            raise ValueError("source gained a shared reference")
                    if self._identity(path) != item["identity"]:
                        raise ValueError("source changed")
                    self._unlink_source(path)
            except (OSError, ValueError):
                self._set_source_result(trash_id, index, "error", "源文件未删除：文件变化、权限不足或删除失败；应用资料已移入回收站")
            else:
                self._set_source_result(trash_id, index, "deleted", "源文件已永久删除，无法恢复")

    def reconcile_sources(self):
        """After interruption, never replay unlink. Paths are private and untrusted."""
        with self.library.lock:
            catalog = self.organization._read()
            intents = read_json(self.source_intents, {})
            changed = False
            for identity, entry in catalog.get("trash", {}).items():
                for index, result in enumerate(entry.get("source_results", [])):
                    if result["status"] != "pending":
                        continue
                    items = intents.get(identity, [])
                    item = items[index] if index < len(items) else None
                    missing = bool(item and not os.path.lexists(item["path"]))
                    locations_path = self.library.root / "private" / "media-locations.json"
                    locations = read_json(locations_path, {})
                    for asset_id in result["asset_ids"]:
                        locations.pop(asset_id, None)
                    atomic_json(locations_path, locations)
                    result.update(status="error",
                                  message="删除曾中断，原位置已无文件，删除状态不确定；未自动重试" if missing else "删除曾中断，未自动重试源文件删除")
                    changed = True
            if changed:
                self.organization._commit(catalog)

    @staticmethod
    def _public(entry):
        return {key: copy.deepcopy(value) for key, value in entry.items()
                if key not in {"patches", "registry_revisions", "registry_digests", "evidence_digest", "restored_at"}}

    def list(self):
        return [self._public(entry) for entry in reversed(list(self.organization._read().get("trash", {}).values()))
                if not entry.get("restored_at")]

    def restore(self, trash_id):
        safe_id(trash_id)
        with self.library.lock:
            catalog = self.organization._read()
            entry = catalog.get("trash", {}).get(trash_id)
            if not entry or entry.get("restored_at"):
                raise FileNotFoundError("回收记录不存在")
            if any(item["status"] == "pending" for item in entry.get("source_results", [])):
                raise OrganizationConflict("源文件删除仍在进行，请稍后恢复资料")
            from .characters import CharacterStore
            store = CharacterStore(self.library)
            for scope, revision in entry.get("registry_revisions", {}).items():
                if store._read(scope)["revision"] != revision:
                    raise OrganizationConflict("删除后角色资料已更新，恢复会覆盖新关联，已保留回收记录")
            affected = list(entry["patches"]["assets"])
            if (any(digest(store._read(scope)) != checksum for scope, checksum in entry.get("registry_digests", {}).items())
                    or (entry.get("evidence_digest") is not None and self._evidence_digest(affected) != entry["evidence_digest"])):
                raise OrganizationConflict("删除后分析资料已更新，恢复会覆盖新关联，已保留回收记录")
            jobs = self._jobs(catalog, {"affected_asset_ids": affected, "project_ids": list(entry["patches"]["projects"])})
            if any(job["status"] in BUSY for job in jobs):
                raise OrganizationConflict("相关任务正在运行，请停止后恢复")
            for category, patches in entry["patches"].items():
                for identity, patch in patches.items():
                    if catalog[category].get(identity) != patch["after"]:
                        raise OrganizationConflict("删除后分类或视频归属已变化，恢复会覆盖新修改，已保留回收记录")
                    restored = copy.deepcopy(patch["before"])
                    if category == "assets":
                        restored["membership_revision"] = patch["after"]["membership_revision"] + 1
                        restored["deletion_revision"] = patch["after"].get("deletion_revision", 0) + 1
                        restored["previous_scopes"] = list(dict.fromkeys([*restored.get("previous_scopes", []),
                            *patch["after"].get("previous_scopes", []), patch["after"]["character_scope_id"]]))
                    catalog[category][identity] = restored
            for category, field, target in (("assets", "project_id", "projects"), ("projects", "folder_id", "folders"),
                                            ("folders", "parent_id", "folders")):
                for identity in entry["patches"][category]:
                    parent = catalog[category][identity].get(field)
                    if parent and catalog[target].get(parent, {}).get("trash_id"):
                        raise OrganizationConflict("原所属项目或文件夹仍在回收站，请先恢复上级分类")
            registries = {}
            for identity in affected:
                scope = catalog["assets"][identity]["character_scope_id"]
                if scope not in registries:
                    registries[scope] = store._read(scope)
                    registries[scope]["revision"] += 1
            catalog["trash"][trash_id]["restored_at"] = now()
            catalog["deletion_revision"] = catalog.get("deletion_revision", 0) + 1
            self.organization._commit(catalog, registries)
            self._invalidate_jobs(jobs)
            return {"restored": self._public(entry), "restored_asset_ids": [item["id"] for item in entry["assets"]],
                    "restored_project_ids": [item["id"] for item in entry["projects"]],
                    "restored_folder_ids": [item["id"] for item in entry["folders"]]}
