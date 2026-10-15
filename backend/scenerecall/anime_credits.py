"""Source-backed credits, kept outside narrative and character evidence."""
from __future__ import annotations

import asyncio
import copy
import re
import unicodedata
from datetime import datetime, timezone

from .bangumi import BangumiError, _id, _text
from .library import atomic_json, read_json

MAX_STAFF = 2000
PAGE_SIZE = 100


def episode_match(value: str, episode: int | None) -> bool | None:
    """Unknown/free-form upstream scopes never prove episode participation."""
    if episode is None or not value.strip():
        return None
    value = unicodedata.normalize("NFKC", value).strip()
    if value.lower() in {"全", "全话", "全話", "全集", "all"}:
        return True
    value = re.sub(r"[、，;；/\s]+", ",", value).strip(",")
    value = re.sub(r"[～~−–—]", "-", value)
    parts = value.split(",")
    spans = []
    for part in parts:
        match = re.fullmatch(r"(?:第|EP\.?|ep\.?)?(\d+)(?:-(\d+))?(?:话|話|集)?", part)
        if not match:
            return None
        start, end = int(match[1]), int(match[2] or match[1])
        if start > end:
            return None
        spans.append((start, end))
    return any(start <= episode <= end for start, end in spans)


class AnimeCreditsStore:
    def __init__(self, library, client):
        self.library = library
        self.client = client
        self._locks: dict[int, asyncio.Lock] = {}

    def _identity(self, asset_id):
        asset = self.library.get_asset(asset_id)
        profile = asset.get("anime_profile") or {}
        subject = profile.get("dossier", {}).get("subject", {}) if profile.get("mode") == "bangumi" else {}
        return asset, subject.get("id")

    def _path(self, subject_id):
        # Organization metadata participates in ordinary metadata backups.
        return self.library.root / "organization" / "bangumi-credits" / f"{_id(subject_id)}.json"

    def get(self, asset_id):
        asset, subject_id = self._identity(asset_id)
        result = {"asset_id": asset_id, "episode": asset.get("episode"), "subject_id": subject_id,
                  "source": "bangumi", "status": "missing" if subject_id else "unavailable",
                  "rows": [], "warnings": []}
        if not subject_id:
            return result
        saved = read_json(self._path(subject_id))
        if not saved or saved.get("subject_id") != subject_id:
            return result
        result.update(status="completed", fetched_at=saved["fetched_at"],
                      url=f"https://bgm.tv/subject/{subject_id}/persons", warnings=saved.get("warnings", []))
        for row in saved["rows"]:
            match = episode_match(row["episodes"], asset.get("episode"))
            if match is False:
                continue
            result["rows"].append({**copy.deepcopy(row), "scope": "episode" if match else
                                   "unparsed" if row["episodes"].strip() else "subject"})
        return result

    async def fetch(self, asset_id):
        _asset, subject_id = self._identity(asset_id)
        if not subject_id:
            raise ValueError("请先为这部动画关联 Bangumi 档案")
        async with self._locks.setdefault(subject_id, asyncio.Lock()):
            rows, seen, warnings = [], set(), []
            try:
                async with asyncio.timeout(max(120, self.client.timeout_s * 3)):
                    for offset in range(0, MAX_STAFF, PAGE_SIZE):
                        response = await self.client._run("subject", "staff", str(subject_id),
                                                          "--limit", str(PAGE_SIZE), "--offset", str(offset))
                        page = self.client._items(response)
                        added = 0
                        for relation in page:
                            person = relation.get("staff", relation.get("person", relation))
                            if not isinstance(person, dict) or not isinstance(relation.get("positions", []), list):
                                raise BangumiError("Bangumi 演职员资料格式不正确", code="invalid_response")
                            identifier = _id(person.get("id"))
                            for position in relation.get("positions", []):
                                if not isinstance(position, dict) or not isinstance(position.get("type"), dict):
                                    raise BangumiError("Bangumi 演职员职位格式不正确", code="invalid_response")
                                role = position.get("type") or {}
                                label = _text(role.get("cn") or role.get("jp") or role.get("en"))
                                episodes = _text(position.get("appearEps", position.get("eps", "")))
                                key = (identifier, label, episodes)
                                if not label or key in seen:
                                    continue
                                seen.add(key)
                                added += 1
                                rows.append({"person_id": identifier, "name": _text(person.get("name")),
                                             "name_cn": _text(person.get("nameCN") or person.get("name_cn")),
                                             "position": label, "episodes": episodes,
                                             "url": f"https://bgm.tv/person/{identifier}"})
                        total = response.get("total") if isinstance(response, dict) else None
                        if len(page) < PAGE_SIZE or (isinstance(total, int) and offset + len(page) >= total):
                            break
                        if not added:
                            warnings.append("来源分页未返回新的演职员记录，已停止读取；名单可能不完整")
                            break
                    else:
                        warnings.append(f"演职员数量达到读取上限 {MAX_STAFF}，名单可能不完整")
            except TimeoutError:
                raise BangumiError("Bangumi 演职员资料读取超时，已有资料仍保留", code="timeout") from None
            # No partial snapshot replaces a successful previous table.
            atomic_json(self._path(subject_id), {"source": "bangumi", "subject_id": subject_id,
                        "fetched_at": datetime.now(timezone.utc).isoformat(), "rows": rows, "warnings": warnings})
        return self.get(asset_id)
