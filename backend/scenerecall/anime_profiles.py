"""Server-owned Bangumi snapshots and atomic source-defined cast initialization."""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone

from .bangumi import BangumiClient, BangumiError, normalize_dossier_names, parse_subject_reference
from .library import atomic_json, read_json


def validate_profile(profile):
    if profile is None:
        return
    if not isinstance(profile, dict) or profile.get("mode") not in {"scratch", "bangumi"}:
        raise ValueError("动画档案格式无效")
    if profile["mode"] == "scratch":
        if set(profile) != {"mode"}:
            raise ValueError("从零建档不能包含外部资料")
        return
    dossier = profile.get("dossier")
    if not isinstance(dossier, dict) or dossier.get("source") != "bangumi":
        raise ValueError("动画档案来源无效")
    subject = dossier.get("subject")
    if (not isinstance(subject, dict) or type(subject.get("id")) is not int or subject["id"] <= 0
            or not isinstance(subject.get("name"), str) or not isinstance(dossier.get("characters"), list)
            or not isinstance(dossier.get("fetched_at"), str)):
        raise ValueError("Bangumi 档案内容无效")


def selection_key(selection):
    return (selection["mode"], selection.get("subject_id"))


def profile_key(profile):
    return (profile["mode"], profile.get("dossier", {}).get("subject", {}).get("id"))


def check_binding(current, selection):
    # Adding a source to a scratch archive is safe. Switching an established
    # show's identity would contaminate its shared cast and past observations.
    if current and current["mode"] == "bangumi" and profile_key(current) != selection_key(selection):
        raise ValueError("此项目已使用另一个 Bangumi 档案，请使用独立项目导入不同作品")


def _name_priority(character, field, name):
    languages = {"name_cn": {"zh-Hans", "zh-Hant"}, "name_ja": {"ja"},
                 "name_en": {"en", "ja-Latn", "und"}}[field]
    priorities = [4 if entry.get("source") == "bangumi" and entry.get("language") == "en"
                  else 3 if entry.get("source") == "bangumi" else 1
                  for entry in character.get("name_variants", [])
                  if entry.get("name") == name and entry.get("language") in languages]
    return max(priorities, default=0)


def _dossier_timestamp(dossier):
    """Accept legacy naive ISO timestamps as UTC, and distrust invalid dates."""
    value = dossier.get("fetched_at")
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def merge_dossier(previous, incoming):
    """Use the newest dated roster, even when replaying older import snapshots."""
    previous = normalize_dossier_names(previous)
    incoming = normalize_dossier_names(incoming)
    previous_time, incoming_time = _dossier_timestamp(previous), _dossier_timestamp(incoming)
    # A moved episode or resumed import can carry an older dossier. Invalid
    # timestamps cannot establish that its roster supersedes the saved one.
    roster = (incoming if incoming_time is not None and (previous_time is None or incoming_time >= previous_time)
              else previous)
    result = copy.deepcopy(previous)
    for key, value in incoming.items():
        if key not in {"characters", "subject", "fetched_at", "warnings"} and not result.get(key):
            result[key] = copy.deepcopy(value)
    for key, value in incoming["subject"].items():
        if not result["subject"].get(key) and value:
            result["subject"][key] = copy.deepcopy(value)
    roles = {role["id"]: role for role in result["characters"]}
    incoming_roles = {role["id"]: role for role in incoming["characters"]}
    result["characters"] = []
    for roster_role in roster["characters"]:
        role = incoming_roles.get(roster_role["id"], roster_role)
        if role["id"] not in roles:
            result["characters"].append(copy.deepcopy(role))
        else:
            existing = copy.deepcopy(roles[role["id"]])
            for key, value in role.items():
                if key in {"aliases", "name_variants"}:
                    for entry in value:
                        if entry not in existing.setdefault(key, []):
                            existing[key].append(copy.deepcopy(entry))
                elif key in {"name_cn", "name_ja", "name_en"} and value:
                    if (not existing.get(key)
                            or _name_priority(role, key, value) > _name_priority(roles[role["id"]], key, existing[key])):
                        existing[key] = copy.deepcopy(value)
                elif not existing.get(key) and value:
                    existing[key] = copy.deepcopy(value)
            if not role.get("detail_error"):
                existing.pop("detail_error", None)
            result["characters"].append(existing)
    result["warnings"] = copy.deepcopy(roster.get("warnings", []))
    result["fetched_at"] = roster["fetched_at"]
    return result


class AnimeProfileStore:
    def __init__(self, library, client=None):
        self.library = library
        self.client = client or BangumiClient()
        self._locks = {}

    async def search(self, query):
        try:
            candidates = await self.client.search(query)
            return {"status": "matches" if candidates else "not_found", "candidates": candidates}
        except BangumiError as exc:
            return {"status": "error", "candidates": [], "error": str(exc)}

    async def dossier(self, reference):
        subject_id = parse_subject_reference(reference)
        lock = self._locks.setdefault(subject_id, asyncio.Lock())
        async with lock:
            path = self.library.root / "runtime" / "bangumi" / f"{subject_id}.json"
            cached = read_json(path)
            if cached:
                try:
                    validate_profile({"mode": "bangumi", "dossier": cached})
                    fetched = datetime.fromisoformat(cached["fetched_at"].replace("Z", "+00:00"))
                    age = (datetime.now(timezone.utc) - fetched).total_seconds()
                    if cached["subject"]["id"] == subject_id and 0 <= age < 86400:
                        upgraded = normalize_dossier_names(cached)
                        if upgraded != cached:
                            atomic_json(path, upgraded)
                        cached = upgraded
                        return copy.deepcopy(cached)
                except (ValueError, TypeError, KeyError):
                    pass
            dossier = normalize_dossier_names(await self.client.dossier(subject_id))
            validate_profile({"mode": "bangumi", "dossier": dossier})
            if dossier["subject"]["id"] != subject_id:
                raise BangumiError("Bangumi 返回的条目与所选作品不一致，请重试")
            atomic_json(path, dossier)
            return copy.deepcopy(dossier)

    async def prepare_import(self, request):
        """Fetch once per selected subject, before publishing a persistent job."""
        selected = {}
        subjects = set()
        for item in request.items:
            if item.anime_profile is None:
                continue
            project_id = request.project_id if "project_id" in request.model_fields_set else item.project_id
            project = self.library.organization.get_project(project_id) if project_id else None
            if item.kind != "animation" and (not project or project["kind"] != "animation"):
                raise ValueError("只有动画可以使用 Bangumi 初始档案")
            selection = item.anime_profile.model_dump(exclude_none=True)
            if project:
                check_binding(project.get("anime_profile"), selection)
                previous = selected.setdefault(project_id, selection_key(selection))
                if previous != selection_key(selection):
                    raise ValueError("同一动画项目的各集须使用同一个初始档案")
            if selection["mode"] == "bangumi":
                subjects.add(selection["subject_id"])
        # Sequential subjects, concurrent bounded character reads inside the CLI
        # adapter: a large batch must not create unbounded subprocess fan-out.
        dossiers = {subject_id: await self.dossier(subject_id) for subject_id in sorted(subjects)}
        return {item.item_id: dossiers[item.anime_profile.subject_id] for item in request.items
                if item.anime_profile and item.anime_profile.mode == "bangumi"}

    def apply_asset(self, asset_id, selection, dossier=None):
        with self.library.lock:
            asset = self.library.get_asset(asset_id)
            project_id = asset.get("project_id")
            return self._apply(selection, dossier, project_id=project_id,
                               asset_id=None if project_id else asset_id)

    def apply_project(self, project_id, selection, dossier=None):
        return self._apply(selection, dossier, project_id=project_id)

    def _apply(self, selection, dossier, *, project_id=None, asset_id=None):
        from .characters import CharacterStore
        with self.library.lock:
            organization = self.library.organization
            catalog = copy.deepcopy(organization._read())
            if project_id:
                target = catalog["projects"].get(project_id)
                if not target or target.get("trash_id"):
                    raise FileNotFoundError("项目不存在")
                if target["kind"] != "animation":
                    raise ValueError("只有动画项目可以使用 Bangumi 初始档案")
            else:
                asset = self.library.get_asset(asset_id)
                if asset["kind"] != "animation":
                    raise ValueError("只有动画可以使用 Bangumi 初始档案")
                target = catalog["assets"][asset_id]
            check_binding(target.get("anime_profile"), selection)
            profile = {"mode": selection["mode"]}
            if selection["mode"] == "bangumi":
                if not dossier or dossier["subject"]["id"] != selection["subject_id"]:
                    raise ValueError("导入任务缺少所选 Bangumi 档案，请重新提交")
                dossier = normalize_dossier_names(dossier)
                profile["dossier"] = copy.deepcopy(dossier)
            validate_profile(profile)
            previous = target.get("anime_profile")
            if previous and profile_key(previous) == profile_key(profile):
                if dossier:
                    profile["dossier"] = merge_dossier(previous["dossier"], dossier)
                else:
                    return copy.deepcopy(previous)
            store = CharacterStore(self.library)
            scope = target["character_scope_id"]
            registry = store._read(scope)
            seeded = store.seed_registry(registry, profile["dossier"]) if dossier else registry
            if previous == profile and seeded == registry:
                return copy.deepcopy(previous)
            if seeded["revision"] == registry["revision"]:
                seeded["revision"] += 1
            target["anime_profile"] = profile
            # The existing roll-forward journal publishes the cast before the
            # catalog, and recovers both after a crash.
            organization._commit(catalog, {scope: seeded})
            return copy.deepcopy(profile)
