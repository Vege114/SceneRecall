"""Local, scope-aware character shorthand resolution for search queries.

Names normalize the query used by retrieval and evidence checks for every record
type. They never establish that a character or an action is visible, and do not
rewrite any source dialogue.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict

from .characters import CharacterStore, scope_id
from .library import digest
from .search_timing import timed


_HAN = re.compile(r"[\u3400-\u9fff]+\Z")
_LEFT = re.compile(r"(?:^|[\s，。！？：；、,!?;:（(\[\"“]|与|和|跟|及|找|看|是|有|让|把|被|的|关于|角色|人物|搜索|查找|寻找)$")
_RIGHT = re.compile(r"(?:$|[\s，。！？：；、,!?;:）)\]\"”]|与|和|跟|及|在|站|坐|走|跑|看|说|拿|穿|抱|打|唱|跳|笑|哭|向|从|到|的|一起|同时|正在|出现)")
_JOIN = re.compile(r"\s*(?:与|和|跟|及|、)\s*\Z")


def _text(value):
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _boundaries(query, start, end):
    return bool(_LEFT.search(query[:start]) and _RIGHT.match(query[end:]))


def _latin_boundary(query, start, end, name):
    if name[0].isascii() and name[0].isalnum() and start and query[start - 1].isascii() and query[start - 1].isalnum():
        return False
    if name[-1].isascii() and name[-1].isalnum() and end < len(query) and query[end].isascii() and query[end].isalnum():
        return False
    return True


def name_index(profiles):
    """Exact aliases outrank inferred pieces; equal-strength collisions abstain."""
    values = {}
    for profile in profiles:
        identity = (profile["scope_id"], profile["id"])
        canonical = profile["name"]
        names = [canonical, *(profile.get("aliases") or []),
                 *(profile.get(field, "") for field in ("name_cn", "name_ja", "name_en")),
                 *(row.get("name", "") for row in profile.get("name_variants", []))]
        for value in names:
            name = _text(value)
            if not name:
                continue
            values.setdefault(name, {}).setdefault(2, {})[identity] = canonical
            # Bounded Chinese pieces, rather than edit distance over all evidence.
            if _HAN.fullmatch(name) and 2 < len(name) <= 16:
                for size in range(2, len(name)):
                    for offset in range(len(name) - size + 1):
                        values.setdefault(name[offset:offset + size], {}).setdefault(1, {})[identity] = canonical
        # One-character inferred names require a coordinated, resolved person
        # mention at query time. This keeps 光线/阳光/灯光 out of name expansion.
        for value in (canonical, profile.get("name_cn", "")):
            name = _text(value)
            if _HAN.fullmatch(name) and 2 < len(name) <= 6:
                values.setdefault(name[-1], {}).setdefault(0, {})[identity] = canonical
    return values


def resolve_text(query, names):
    normalized = _text(query)
    candidates = []
    for name, ranks in names.items():
        rank = max(ranks)
        identities = ranks[rank]
        if len(identities) != 1:
            continue
        identity, canonical = next(iter(identities.items()))
        offset = 0
        while (start := normalized.find(name, offset)) >= 0:
            end = start + len(name)
            offset = end
            if not _latin_boundary(normalized, start, end, name):
                continue
            if (rank < 2 or len(name) == 1) and not _boundaries(normalized, start, end):
                continue
            candidates.append({"start": start, "end": end, "rank": rank,
                               "scope_id": identity[0], "character_id": identity[1],
                               "name": canonical, "matched_text": name})
    selected = []
    for candidate in sorted(candidates, key=lambda row: (-(row["end"] - row["start"]), -row["rank"], row["start"])):
        if candidate["rank"] == 0:
            continue
        if not any(candidate["start"] < row["end"] and candidate["end"] > row["start"] for row in selected):
            selected.append(candidate)
    for candidate in candidates:
        if candidate["rank"] != 0 or any(candidate["start"] < row["end"] and candidate["end"] > row["start"] for row in selected):
            continue
        coordinated = any(row["scope_id"] == candidate["scope_id"]
                          and row["character_id"] != candidate["character_id"]
                          and row["end"] - row["start"] >= 2
                          and (row["end"] <= candidate["start"] and _JOIN.fullmatch(normalized[row["end"]:candidate["start"]])
                               or candidate["end"] <= row["start"] and _JOIN.fullmatch(normalized[candidate["end"]:row["start"]]))
                          for row in selected)
        if coordinated:
            selected.append(candidate)
    selected.sort(key=lambda row: row["start"])
    rewritten = normalized
    for row in reversed(selected):
        rewritten = rewritten[:row["start"]] + row["name"] + rewritten[row["end"]:]
    # Once a character is resolved, normalize the surrounding text for both
    # full and abbreviated names, including case and full-width punctuation.
    text = rewritten if selected else query
    identities = {(row["scope_id"], row["character_id"]) for row in selected}
    # Derive the same lexical expansion from the resolved identities for every
    # spelling. Inferred substrings already occur in canonical-name tokens;
    # registered nicknames/translations may share no characters with them.
    aliases = sorted(name for name, ranks in names.items()
                     if len(ranks.get(2, {})) == 1 and identities.intersection(ranks[2]))
    return {"text": text, "matches": selected, "changed": text != query, "aliases": aliases}


class CharacterQueryResolver:
    def __init__(self, library):
        self.library = library
        self.characters = CharacterStore(library)
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def _snapshot(self, filters, engine):
        # The index supplies the actual search scope. Character filters are
        # excluded because their own spelling may need resolution first.
        scoped = {key: value for key, value in filters.items() if key not in {"character", "character_id"}}
        with self.library.lock, self._lock:
            organization = self.library.organization._read()
            policies = {}
            for asset_id in engine.filtered_asset_ids(scoped, visual_only=False):
                try:
                    asset = self.library.get_asset(asset_id, _catalog=organization)
                except FileNotFoundError:
                    # Source deletion can precede disposable-index publication.
                    continue
                scope = scope_id(asset)
                policies.setdefault(scope, []).append(asset.get("anime_profile"))
            profiles, stamps = [], []
            for scope, archives in sorted(policies.items()):
                stamp = self.characters._file_signature(self.characters._path(scope))
                key = (scope, stamp, digest(archives))
                saved = self._cache.get(key)
                if saved is None:
                    registry = self.characters._read(scope)
                    cast_policies = [self.characters.cast_policy(registry, archive) for archive in archives]
                    saved = [{**profile, "scope_id": scope} for profile in registry["profiles"].values()
                             if any(self.characters.profile_allowed(profile, policy) for policy in cast_policies)]
                    self._cache[key] = saved
                    while len(self._cache) > 64:
                        self._cache.popitem(last=False)
                self._cache.move_to_end(key)
                profiles.extend(saved)
                stamps.append(key)
            signature = digest(stamps)
            names_key = ("names", signature)
            names = self._cache.get(names_key)
            if names is None:
                names = name_index(profiles)
                self._cache[names_key] = names
                while len(self._cache) > 64:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(names_key)
            return signature, names

    @timed("query.character_names.revision")
    def revision(self, filters, engine):
        return self._snapshot(filters, engine)[0]

    def resolve(self, query, filters, engine):
        return resolve_text(query, self._snapshot(filters, engine)[1])
