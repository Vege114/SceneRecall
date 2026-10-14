"""Logical projects and folders over immutable video evidence.

The catalog is authoritative for membership. Durable after-image transactions
publish character copies and membership together and roll forward after a crash.
"""
from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

from .library import SCHEMA, atomic_json, confined_path, digest, read_json, safe_id, same_contents, uid


class OrganizationConflict(ValueError):
    pass


def legacy_scope(asset: dict) -> str:
    from .characters import normalized
    series = normalized(asset.get("series") or "")
    return "cast_" + digest([asset.get("kind", "movie"), series]) if series else safe_id(asset["work_id"])


class OrganizationStore:
    def __init__(self, library):
        self.library = library
        self.path = library.root / "organization" / "catalog.json"
        self.journal = library.root / "runtime" / "organization-transaction.json"
        with library.lock:
            self.recover()
            if not self.path.exists():
                atomic_json(self.path, {"schema_version": SCHEMA, "organization_version": 1,
                                       "projects": {}, "folders": {}, "assets": {}})
            self.validate(check_assets=False)
            self.migrate()

    def recover(self):
        transaction = read_json(self.journal)
        if transaction is None:
            return
        if not isinstance(transaction, dict) or not isinstance(transaction.get("writes"), dict):
            raise ValueError("组织迁移事务格式无效")
        for relative in transaction["writes"]:
            parts = Path(relative).parts
            if relative not in {"organization/catalog.json", "private/media-locations.json"} and not (
                    len(parts) == 3 and parts[0] == "works" and parts[2] == "characters.json"):
                raise ValueError("组织迁移事务包含非法路径")
            confined_path(self.library.root, relative)
        staging = None
        files = transaction.get("files", [])
        if not isinstance(files, list):
            raise ValueError("资料恢复事务格式无效")
        if "staging" in transaction:
            parts = Path(transaction["staging"]).parts
            if len(parts) != 2 or parts[0] != "runtime" or not parts[1].startswith("restore_"):
                raise ValueError("资料恢复暂存目录无效")
            staging = confined_path(self.library.root, transaction["staging"])
        if files and staging is None:
            raise ValueError("资料恢复事务缺少暂存目录")
        for relative in files:
            if (not isinstance(relative, str) or not Path(relative).parts
                    or Path(relative).parts[0] not in {"assets", "works", "collections", "media", "organization"}):
                raise ValueError("资料恢复事务包含非法目标")
            target = confined_path(self.library.root, relative)
            source = confined_path(staging, relative)
            if source.is_file():
                if target.exists():
                    if not target.is_file() or not same_contents(target, source):
                        raise ValueError(f"恢复冲突，保留事务等待修复: {relative}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, target)
            elif not target.is_file():
                raise ValueError(f"资料恢复事务缺少文件: {relative}")
        # Catalog last: it never advertises a character copy before it exists.
        for relative, value in sorted(transaction["writes"].items(), key=lambda item: item[0] == "organization/catalog.json"):
            atomic_json(confined_path(self.library.root, relative), value)
        self.journal.unlink()
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    def _read(self):
        with self.library.lock:
            self.recover()
            return read_json(self.path)

    def _commit(self, catalog, registries=None):
        self.validate(catalog, check_assets=False)
        writes = {f"works/{safe_id(scope)}/characters.json": registry
                  for scope, registry in (registries or {}).items()}
        writes["organization/catalog.json"] = catalog
        atomic_json(self.journal, {"schema_version": SCHEMA, "id": uid("organization_tx"), "writes": writes})
        self.recover()

    def commit_restore(self, staging, paths, catalog, locations):
        """Prepare recovery before publishing even the first restored asset file."""
        self.validate(catalog, check_assets=False)
        transaction = {"schema_version": SCHEMA, "id": uid("restore_tx"),
                       "staging": str(staging.relative_to(self.library.root)),
                       "files": [str(path.relative_to(staging)) for path in paths],
                       "writes": {"organization/catalog.json": catalog,
                                  "private/media-locations.json": locations}}
        atomic_json(self.journal, transaction)
        self.recover()

    @staticmethod
    def _name(value):
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200:
            raise ValueError("名称须为 1–200 字")
        return value.strip()

    @staticmethod
    def _tags(values):
        from .characters import normalized
        if not isinstance(values, list) or len(values) > 100:
            raise ValueError("标签须为不超过 100 项的列表")
        result, seen = [], set()
        for value in values:
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > 80:
                raise ValueError("标签须为 1–80 字")
            value = value.strip()
            key = normalized(value)
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result

    def migrate(self):
        with self.library.lock:
            catalog = self._read()
            changed = False
            for path in (self.library.root / "assets").glob("*/manifest.json"):
                asset = read_json(path)
                if asset["id"] not in catalog["assets"]:
                    self._add_asset(catalog, asset)
                    changed = True
            if changed:
                self._commit(catalog)

    def _add_asset(self, catalog, asset):
        from .characters import normalized
        if not isinstance(asset, dict) or not {"id", "work_id", "title", "created_at", "duration_ms", "version", "fingerprint"}.issubset(asset):
            raise ValueError("资产清单不完整")
        old_scope = legacy_scope(asset)
        project_id = asset.get("project_id")
        if "project_id" not in asset and normalized(asset.get("series") or ""):
            project_id = "project_" + digest(["legacy", asset.get("kind", "movie"), normalized(asset["series"])])
            project_scope = old_scope
            while catalog["projects"].get(project_id, {}).get("trash_id"):
                project_id = "project_" + digest(["legacy-recreated", project_id, catalog["projects"][project_id]["trash_id"]])
                project_scope = "cast_" + project_id
            catalog["projects"].setdefault(project_id, {
                "id": project_id, "name": asset["series"].strip(), "kind": asset.get("kind", "movie"),
                "folder_id": None, "tags": [], "character_scope_id": project_scope,
            })
        if project_id is not None and (project_id not in catalog["projects"] or catalog["projects"][project_id].get("trash_id")):
            raise ValueError("视频归属的项目不存在")
        scope = (catalog["projects"][project_id]["character_scope_id"] if project_id else
                 asset.get("character_scope_id") or old_scope)
        catalog["assets"][asset["id"]] = {
            "project_id": project_id, "character_scope_id": scope,
            "legacy_character_scope_id": asset.get("character_scope_id") or old_scope,
            "membership_revision": asset.get("membership_revision", 0), "bindings": {}, "previous_scopes": [],
        }

    def project_asset(self, asset, *, include_deleted=False):
        with self.library.lock:
            catalog = self._read()
            if asset["id"] not in catalog["assets"]:
                self._add_asset(catalog, asset)
                self._commit(catalog)
            membership = catalog["assets"][asset["id"]]
            if membership.get("trash_id") and not include_deleted:
                raise FileNotFoundError("视频已在回收站，请先恢复")
            project = catalog["projects"].get(membership["project_id"])
            folder_id = project["folder_id"] if project else None
            ancestors = []
            current = folder_id
            while current:
                if current in ancestors:
                    raise ValueError("文件夹层级包含循环")
                ancestors.append(current)
                current = catalog["folders"][current]["parent_id"]
            return {**asset, **{key: membership[key] for key in (
                "project_id", "character_scope_id", "membership_revision")},
                "project_name": project["name"] if project else None,
                "series": project["name"] if project else asset.get("series", ""),
                "folder_id": folder_id, "folder_ids": ancestors, "tags": project["tags"] if project else [],
                "anime_profile": project.get("anime_profile") if project else membership.get("anime_profile"),
                "deleted": bool(membership.get("trash_id")), "trash_id": membership.get("trash_id")}

    def projects(self):
        self.migrate()
        with self.library.lock:
            catalog = self._read()
            return [{**project, "asset_count": sum(item["project_id"] == project["id"]
                    for item in catalog["assets"].values() if not item.get("trash_id"))}
                    for project in catalog["projects"].values() if not project.get("trash_id")]

    def folders(self):
        with self.library.lock:
            return [folder for folder in self._read()["folders"].values() if not folder.get("trash_id")]

    def get_project(self, project_id):
        safe_id(project_id)
        project = next((project for project in self.projects() if project["id"] == project_id), None)
        if project is None:
            raise FileNotFoundError("项目不存在")
        return project

    def save_project(self, payload, project_id=None):
        with self.library.lock:
            catalog = self._read()
            if project_id is not None and (project_id not in catalog["projects"] or catalog["projects"][project_id].get("trash_id")):
                raise FileNotFoundError("项目不存在")
            project_id = project_id or uid("project")
            previous = catalog["projects"].get(project_id, {
                "id": project_id, "kind": "animation", "folder_id": None, "tags": [],
                "character_scope_id": "cast_" + project_id})
            item = {**previous, **payload, "id": project_id}
            item["name"] = self._name(item.get("name"))
            item["tags"] = self._tags(item["tags"])
            if item.get("folder_id") and catalog["folders"].get(item["folder_id"], {}).get("trash_id"):
                raise ValueError("文件夹已在回收站")
            catalog["projects"][project_id] = item
            self._commit(catalog)
            return next(project for project in self.projects() if project["id"] == project_id)

    def save_folder(self, payload, folder_id=None):
        with self.library.lock:
            catalog = self._read()
            if folder_id is not None and (folder_id not in catalog["folders"] or catalog["folders"][folder_id].get("trash_id")):
                raise FileNotFoundError("文件夹不存在")
            folder_id = folder_id or uid("folder")
            item = {**catalog["folders"].get(folder_id, {"parent_id": None}), **payload, "id": folder_id}
            item["name"] = self._name(item.get("name"))
            if item.get("parent_id") and catalog["folders"].get(item["parent_id"], {}).get("trash_id"):
                raise ValueError("父文件夹已在回收站")
            catalog["folders"][folder_id] = item
            self._commit(catalog)
            return item

    def delete_folder(self, folder_id):
        from .deletions import DeletionStore
        return DeletionStore(self.library).delete(kind="folder", id=folder_id)

    def resolve_reference(self, asset_id, record, character_id, registry, *, strict=False, catalog=None):
        from .characters import CharacterStore
        if character_id is None:
            return None
        catalog = catalog or self._read()
        membership = catalog["assets"][asset_id]
        source_scope = record.get("character_scope_id") or membership["legacy_character_scope_id"]
        mapped = membership["bindings"].get(source_scope, {})
        if character_id in mapped:
            character_id = mapped[character_id]
        elif source_scope != membership["character_scope_id"]:
            if strict:
                raise ValueError("历史角色缺少当前项目的身份映射")
            return None
        return CharacterStore._resolve(registry, character_id, strict=strict)

    @staticmethod
    def _identity_lineage(store, source_scope, source_id, profile):
        """Only explicit copy provenance establishes cross-project equivalence."""
        result, seen = [], set()
        pending = [{"scope_id": source_scope, "character_id": source_id},
                   *profile.get("identity_origins", [])]
        if profile.get("imported_from"):
            pending.append(profile["imported_from"])
        while pending:
            origin = pending.pop(0)
            scope, identity = safe_id(origin["scope_id"]), safe_id(origin["character_id"])
            if (scope, identity) in seen:
                continue
            seen.add((scope, identity))
            result.append({"scope_id": scope, "character_id": identity})
            registry = store._read(scope)
            resolved = store._resolve(registry, identity, strict=True)
            if resolved is not None:
                ancestor = registry["profiles"][resolved]
                pending.extend(ancestor.get("identity_origins", []))
                if ancestor.get("imported_from"):
                    pending.append(ancestor["imported_from"])
        return result

    def assign(self, asset_ids, project_id, queue=None, *, delete_project_id=None, _defer=False):
        from .characters import CharacterStore, external_sources
        from .anime_profiles import merge_dossier, profile_key
        with self.library.lock:
            self.migrate()
            catalog = self._read()
            if project_id is not None and (project_id not in catalog["projects"] or catalog["projects"][project_id].get("trash_id")):
                raise FileNotFoundError("项目不存在")
            ids = list(dict.fromkeys(asset_ids))
            if not ids and delete_project_id is None:
                raise ValueError("请选择至少一个视频")
            if any(asset_id not in catalog["assets"] or catalog["assets"][asset_id].get("trash_id") for asset_id in ids):
                raise FileNotFoundError("视频不存在")
            ids = [asset_id for asset_id in ids if catalog["assets"][asset_id]["project_id"] != project_id]
            # Validate/inherit the archive before copying any cast or publishing
            # membership. All episodes in a project share one show identity.
            source_archives = {}
            for asset_id in ids:
                member = catalog["assets"][asset_id]
                source_project = catalog["projects"].get(member["project_id"])
                profile = source_project.get("anime_profile") if source_project else member.get("anime_profile")
                source_archives[asset_id] = copy.deepcopy(profile)
                if profile and project_id:
                    project = catalog["projects"][project_id]
                    if project["kind"] != "animation":
                        raise OrganizationConflict("带有动画档案的视频须移入动画项目，或移至未归类")
                    previous = project.get("anime_profile")
                    if (previous and previous["mode"] == profile["mode"] == "bangumi"
                            and profile_key(previous) != profile_key(profile)):
                        raise OrganizationConflict("源视频与目标项目使用不同 Bangumi 条目，请分别归入对应作品项目")
                    if not previous or (previous["mode"] == "scratch" and profile["mode"] == "bangumi"):
                        project["anime_profile"] = copy.deepcopy(profile)
                    elif previous["mode"] == profile["mode"] == "bangumi":
                        previous["dossier"] = merge_dossier(previous["dossier"], profile["dossier"])
            destination = catalog["projects"][project_id]["character_scope_id"] if project_id else None
            scopes = {catalog["assets"][asset_id]["character_scope_id"] for asset_id in ids}
            if destination:
                scopes.add(destination)
            jobs = queue.list() if queue else []
            affected_jobs = [job for job in jobs if job["type"] == "analysis"
                and "vision" in job.get("config", {}).get("stages", [])
                and catalog["assets"].get(job.get("asset_id"), {}).get("character_scope_id") in scopes]
            if any(job["status"] in {"running", "pausing", "cancelling"} for job in affected_jobs):
                raise OrganizationConflict("源项目或目标项目正在分析画面，请暂停并等待当前步骤结束后再移动视频")
            store = CharacterStore(self.library)
            registries = {scope: store._read(scope) for scope in scopes}
            old_registries = copy.deepcopy(registries)
            for asset_id in ids:
                old = copy.deepcopy(catalog["assets"][asset_id])
                source_scope = old["character_scope_id"]
                target_scope = destination or "cast_" + asset_id
                if target_scope not in registries:
                    registries[target_scope] = store._read(target_scope)
                target = registries[target_scope]
                archive = (catalog["projects"][project_id].get("anime_profile") if project_id
                           else source_archives[asset_id] or old.get("anime_profile"))
                source_policy = store.cast_policy(old_registries[source_scope], source_archives[asset_id])
                target_policy = store.cast_policy(target, archive)
                imports = target.setdefault("imports", {})
                bindings = {}
                evidence_dir = self.library.asset_dir(asset_id)
                for path in [*evidence_dir.glob("analyses/*/observations/*.json"),
                             *evidence_dir.glob("analyses/*/scenes/*.json")]:
                    record = read_json(path)
                    original_scope = record.get("character_scope_id") or old["legacy_character_scope_id"]
                    for entity in record.get("entities", []):
                        original_id = entity.get("character_id")
                        if not original_id:
                            continue
                        source_id = self.resolve_reference(asset_id, record, original_id,
                            old_registries[source_scope], strict=True, catalog=catalog)
                        if source_id is not None:
                            source_profile = old_registries[source_scope]["profiles"][source_id]
                            if (not store.profile_allowed(source_profile, source_policy)
                                    or not store.profile_allowed(source_profile, target_policy)):
                                imports.setdefault(source_scope, {})[source_id] = None
                                source_id = None
                        target_id = None
                        if source_id is not None:
                            imported = imports.setdefault(source_scope, {})
                            if source_id in imported:
                                target_id = store._resolve(target, imported[source_id], strict=True)
                            else:
                                profile = copy.deepcopy(old_registries[source_scope]["profiles"][source_id])
                                lineage = self._identity_lineage(store, source_scope, source_id, profile)
                                known = False
                                for external in external_sources(profile):
                                    source_key = str(external["character_id"])
                                    mapping = target.get("external_bindings", {}).get("bangumi", {})
                                    if source_key in mapping:
                                        target_id = store._resolve(target, mapping[source_key], strict=True)
                                        known = True
                                        break
                                # Returning a copied role to its original project
                                # must reuse that project's human decisions.
                                for origin in lineage:
                                    if known:
                                        break
                                    scope, identity = origin["scope_id"], origin["character_id"]
                                    if scope == target_scope:
                                        target_id = store._resolve(target, identity, strict=True)
                                        known = True
                                        break
                                if not known:
                                    for origin in lineage:
                                        previously_imported = imports.get(origin["scope_id"], {})
                                        if origin["character_id"] in previously_imported:
                                            target_id = store._resolve(target, previously_imported[origin["character_id"]], strict=True)
                                            known = True
                                            break
                                if not known:
                                    target_id = "character_" + digest([target_scope, source_scope, source_id])
                                    profile.update(id=target_id, representative=None, identity_origins=lineage,
                                                   imported_from={"scope_id": source_scope, "character_id": source_id})
                                    target["profiles"][target_id] = profile
                                    store._remember_external_identity(target, profile)
                                imported[source_id] = target_id
                        bindings.setdefault(original_scope, {})[original_id] = target_id
                # Initial external roles have no observations yet. Carry their
                # edited profiles and identity decisions as well as the dossier.
                source_registry = old_registries[source_scope]
                for source_id, source_profile in source_registry["profiles"].items():
                    if (not external_sources(source_profile)
                            or not store.profile_allowed(source_profile, source_policy)
                            or not store.profile_allowed(source_profile, target_policy)):
                        continue
                    imported = imports.setdefault(source_scope, {})
                    if source_id in imported:
                        continue
                    target_id, known = None, False
                    for external in external_sources(source_profile):
                        mapping = target.get("external_bindings", {}).get("bangumi", {})
                        identity = str(external["character_id"])
                        if identity in mapping:
                            target_id = store._resolve(target, mapping[identity], strict=True)
                            known = True
                            break
                    if not known:
                        target_id = "character_" + digest([target_scope, source_scope, source_id])
                        profile = copy.deepcopy(source_profile)
                        lineage = self._identity_lineage(store, source_scope, source_id, profile)
                        profile.update(id=target_id, representative=None, identity_origins=lineage,
                                       imported_from={"scope_id": source_scope, "character_id": source_id})
                        target["profiles"][target_id] = profile
                        store._remember_external_identity(target, profile)
                    imported[source_id] = target_id
                for identity, source_id in source_registry.get("external_bindings", {}).get("bangumi", {}).items():
                    if store._resolve(source_registry, source_id, strict=True) is None:
                        target.setdefault("external_bindings", {}).setdefault("bangumi", {}).setdefault(identity, None)
                catalog["assets"][asset_id] = {**old, "project_id": project_id,
                    "character_scope_id": target_scope, "membership_revision": old["membership_revision"] + 1,
                    "bindings": bindings,
                    "previous_scopes": list(dict.fromkeys([*old.get("previous_scopes", []), source_scope]))}
                if not project_id and source_archives[asset_id]:
                    catalog["assets"][asset_id]["anime_profile"] = source_archives[asset_id]
                archive = (catalog["projects"][project_id].get("anime_profile") if project_id
                           else catalog["assets"][asset_id].get("anime_profile"))
                if archive and archive["mode"] == "bangumi":
                    registries[target_scope] = store.seed_registry(target, archive["dossier"])
            if delete_project_id is not None:
                if delete_project_id not in catalog["projects"]:
                    raise FileNotFoundError("项目不存在")
                del catalog["projects"][delete_project_id]
            for registry in registries.values():
                registry["revision"] += 1
            if _defer:
                return catalog, registries
            if ids or delete_project_id is not None:
                self._commit(catalog, registries)
                for job in affected_jobs:
                    if job["status"] in {"queued", "paused", "failed", "partial"}:
                        queue.update(job["id"], status="paused", stale_organization=True,
                                     message="项目归属已变化，旧任务已暂停；请按当前项目重新创建分析任务")
            return {"assets": [self.library.get_asset(asset_id) for asset_id in asset_ids], "moved_count": len(ids)}

    def delete_project(self, project_id, queue=None):
        from .deletions import DeletionStore
        return DeletionStore(self.library, queue).delete(kind="project", id=project_id)

    def validate(self, catalog=None, *, check_assets=True):
        catalog = catalog or self._read()
        if not isinstance(catalog, dict) or catalog.get("organization_version") != 1 or any(
                not isinstance(catalog.get(key), dict) for key in ("projects", "folders", "assets")):
            raise ValueError("组织目录格式无效")
        if type(catalog.get("deletion_revision", 0)) is not int or catalog.get("deletion_revision", 0) < 0:
            raise ValueError("删除版本无效")
        for folder_id, folder in catalog["folders"].items():
            safe_id(folder_id)
            if folder.get("id") != folder_id:
                raise ValueError("文件夹 ID 不一致")
            self._name(folder.get("name"))
            visited = {folder_id}
            parent = folder.get("parent_id")
            while parent is not None:
                if parent not in catalog["folders"] or parent in visited:
                    raise ValueError("文件夹不存在或层级包含循环")
                visited.add(parent)
                parent = catalog["folders"][parent].get("parent_id")
        for project_id, project in catalog["projects"].items():
            from .anime_profiles import validate_profile
            validate_profile(project.get("anime_profile"))
            safe_id(project_id)
            if project.get("id") != project_id or project.get("kind") not in {"movie", "animation", "series"}:
                raise ValueError("项目 ID 或类型无效")
            self._name(project.get("name"))
            self._tags(project.get("tags"))
            safe_id(project["character_scope_id"])
            if project.get("folder_id") is not None and project["folder_id"] not in catalog["folders"]:
                raise ValueError("项目文件夹不存在")
        for asset_id, membership in catalog["assets"].items():
            from .anime_profiles import validate_profile
            validate_profile(membership.get("anime_profile"))
            safe_id(asset_id)
            if check_assets and not (self.library.asset_dir(asset_id) / "manifest.json").is_file():
                raise ValueError("项目引用的视频不存在")
            project = catalog["projects"].get(membership.get("project_id"))
            if membership.get("project_id") is not None and project is None:
                raise ValueError("视频项目不存在")
            safe_id(membership["character_scope_id"])
            safe_id(membership["legacy_character_scope_id"])
            if not isinstance(membership.get("previous_scopes", []), list):
                raise ValueError("历史项目来源无效")
            for scope in membership.get("previous_scopes", []):
                safe_id(scope)
            if project and membership["character_scope_id"] != project["character_scope_id"]:
                raise ValueError("视频角色档案与项目不一致")
            if type(membership.get("deletion_revision", 0)) is not int or membership.get("deletion_revision", 0) < 0:
                raise ValueError("删除版本无效")
            if type(membership.get("membership_revision")) is not int or membership["membership_revision"] < 0:
                raise ValueError("视频归属版本无效")
            if not isinstance(membership.get("bindings"), dict):
                raise ValueError("历史角色映射无效")
            for source, mapping in membership["bindings"].items():
                safe_id(source)
                if not isinstance(mapping, dict):
                    raise ValueError("历史角色映射无效")
                for original, target in mapping.items():
                    safe_id(original)
                    if target is not None:
                        safe_id(target)
        trash = catalog.get("trash", {})
        if not isinstance(trash, dict):
            raise ValueError("回收站格式无效")
        for trash_id, entry in trash.items():
            safe_id(trash_id)
            if (not isinstance(entry, dict) or entry.get("id") != trash_id
                    or entry.get("kind") not in {"assets", "project", "folder"}
                    or not isinstance(entry.get("patches"), dict)
                    or set(entry["patches"]) != {"assets", "projects", "folders"}
                    or not isinstance(entry.get("source_results", []), list)):
                raise ValueError("回收记录或恢复快照无效")
            for category in ("assets", "projects", "folders"):
                patches = entry["patches"][category]
                listed = entry.get(category)
                if not isinstance(patches, dict) or not isinstance(listed, list):
                    raise ValueError("回收记录引用无效")
                identities = [item.get("id") for item in listed if isinstance(item, dict)]
                if (len(identities) != len(listed) or len(set(identities)) != len(identities)
                        or entry.get(category[:-1] + "_count") != len(identities)):
                    raise ValueError("回收记录数量无效")
                archived = set()
                for identity, patch in patches.items():
                    safe_id(identity)
                    if identity not in catalog[category] or not isinstance(patch, dict) or set(patch) != {"before", "after"}:
                        raise ValueError("恢复快照引用不存在")
                    for phase in ("before", "after"):
                        item = patch[phase]
                        if not isinstance(item, dict) or (category != "assets" and item.get("id") != identity):
                            raise ValueError("恢复快照身份无效")
                        if item.get("trash_id") and (phase != "after" or item["trash_id"] != trash_id):
                            raise ValueError("恢复快照回收引用无效")
                        if category == "assets":
                            for key in ("character_scope_id", "legacy_character_scope_id"):
                                safe_id(item.get(key))
                            if (type(item.get("membership_revision")) is not int or item["membership_revision"] < 0
                                    or not isinstance(item.get("bindings"), dict)
                                    or not isinstance(item.get("previous_scopes", []), list)):
                                raise ValueError("恢复快照视频关联无效")
                            for role_scope in item.get("previous_scopes", []):
                                safe_id(role_scope)
                            for role_scope, mapping in item["bindings"].items():
                                safe_id(role_scope)
                                if not isinstance(mapping, dict):
                                    raise ValueError("恢复角色映射无效")
                                for original, target in mapping.items():
                                    safe_id(original)
                                    if target is not None:
                                        safe_id(target)
                            parent, target_category = item.get("project_id"), "projects"
                        else:
                            self._name(item.get("name"))
                            parent, target_category = item.get("parent_id" if category == "folders" else "folder_id"), "folders"
                            if category == "projects":
                                safe_id(item.get("character_scope_id"))
                                self._tags(item.get("tags"))
                                if item.get("kind") not in {"movie", "animation", "series"}:
                                    raise ValueError("恢复快照项目类型无效")
                        if parent is not None and parent not in catalog[target_category]:
                            raise ValueError("恢复快照缺少原分类")
                    if patch["after"].get("trash_id") == trash_id:
                        archived.add(identity)
                if archived != set(identities):
                    raise ValueError("回收记录与恢复快照不一致")
            for result in entry.get("source_results", []):
                if (not isinstance(result, dict) or result.get("status") not in {"pending", "deleted", "error"}
                        or not isinstance(result.get("asset_ids"), list)
                        or not set(result["asset_ids"]).issubset({item["id"] for item in entry["assets"]})):
                    raise ValueError("源文件删除状态无效")
            for field in ("registry_revisions", "registry_digests"):
                if not isinstance(entry.get(field, {}), dict):
                    raise ValueError("角色恢复快照无效")
                for role_scope, value in entry.get(field, {}).items():
                    safe_id(role_scope)
                    if (field == "registry_revisions" and (type(value) is not int or value < 0)) or (field == "registry_digests" and not isinstance(value, str)):
                        raise ValueError("角色恢复版本无效")
        for category in ("assets", "projects", "folders"):
            for identity, item in catalog[category].items():
                trash_id = item.get("trash_id")
                if trash_id:
                    entry = trash.get(trash_id)
                    if (not entry or entry.get("restored_at")
                            or entry["patches"][category].get(identity, {}).get("after", {}).get("trash_id") != trash_id):
                        raise ValueError("已删除资料缺少可恢复的回收记录")
                else:
                    field, target_category = (("project_id", "projects") if category == "assets" else
                                              ("folder_id", "folders") if category == "projects" else ("parent_id", "folders"))
                    parent = item.get(field)
                    if parent and catalog[target_category][parent].get("trash_id"):
                        raise ValueError("有效资料引用已删除分类")
        return catalog

    def merge_catalog(self, incoming):
        current = self._read()
        result = copy.deepcopy(current)
        for category in ("projects", "folders", "assets", "trash"):
            result.setdefault(category, {})
            for identity, item in incoming.get(category, {}).items():
                if identity in result[category] and result[category][identity] != item:
                    raise ValueError(f"恢复冲突，未覆盖现有组织记录: {identity}")
                result[category][identity] = copy.deepcopy(item)
        self.validate(result, check_assets=False)
        return result
