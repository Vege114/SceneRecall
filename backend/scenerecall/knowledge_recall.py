"""Revisioned knowledge recall over authoritative dossiers and indexed evidence.

Providers propose record references, then project those candidates against current
evidence. Knowledge facts select appearances; they never establish current events.
"""
from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

from .library import digest


class KnowledgeRecallProvider(Protocol):
    def revision(self, filters: dict, engine) -> str:
        """Identity of knowledge and source state relevant to this query scope."""
        ...

    def search(self, query: str, filters: dict, engine) -> list[dict]:
        """Return projected candidates; callers must recheck current indexed evidence."""
        ...


def _size(value, seen=None):
    """Account shared index references only once, including Python containers."""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    result = sys.getsizeof(value)
    if isinstance(value, dict):
        result += sum(_size(k, seen) + _size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set)):
        result += sum(_size(item, seen) for item in value)
    return result


def _stamp(path):
    try:
        value = path.stat()
    except FileNotFoundError:
        return None
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True)
class KnowledgeSnapshot:
    token: str
    catalogs: dict
    asset_scopes: dict
    fingerprints: dict


class WorkEntityRecall:
    """Bounded, disposable occurrence indexes over the latest work catalogs.

The search index's projected concept bindings are deliberately not authoritative:
new bindings must be searchable before a delayed search projection is refreshed.
"""

    def __init__(self, store, max_bytes=64 * 1024 * 1024):
        self.store = store
        self.max_bytes = max(0, int(max_bytes))
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "bytes": 0,
                      "candidate_records": 0, "catalog_builds": 0}

    def clear(self):
        with self._lock:
            self._cache.clear()
            self.stats["bytes"] = 0

    def _load(self, path, loader, transform=lambda value: value):
        key = str(path)
        with self._lock:
            while self._cache and self.stats["bytes"] > self.max_bytes:
                _, removed = self._cache.popitem(last=False)
                self.stats["bytes"] -= removed[3]
                self.stats["evictions"] += 1
            stamp = _stamp(path)
            cached = self._cache.get(key)
            if cached is not None and cached[0] == stamp and self.max_bytes > 0:
                self.stats["hits"] += 1
                self._cache.move_to_end(key)
                return cached[1], cached[2]
            if cached is not None:
                self._cache.pop(key)
                self.stats["bytes"] -= cached[3]
            self.stats["misses"] += 1
            for _ in range(2):
                value = loader()
                after = _stamp(path)
                if after == stamp:
                    break
                stamp = after
            else:
                raise ValueError("作品知识资料正在更新，请重试")
            identity = digest([key, stamp, value.get("revision"), digest(value)])
            value = transform(value)
            size = _size((key, stamp, value, identity))
            while self._cache and self.stats["bytes"] + size > self.max_bytes:
                _, removed = self._cache.popitem(last=False)
                self.stats["bytes"] -= removed[3]
                self.stats["evictions"] += 1
            if size <= self.max_bytes:
                self._cache[key] = (stamp, value, identity, size)
                self.stats["bytes"] += size
            return value, identity

    def _catalog(self, saved):
        by_concept, by_record = {}, {}
        for binding in saved["bindings"].values():
            concept = saved["entities"].get(binding.get("concept_id"))
            if binding["status"] != "confirmed" or not concept or concept["status"] != "active":
                continue
            by_concept.setdefault(binding["concept_id"], []).append(binding)
            by_record.setdefault((binding["asset_id"], binding["record_id"]), []).append(binding)
        self.stats["catalog_builds"] += 1
        return {"work_scope_id": saved["work_scope_id"], "revision": saved["revision"],
                "entities": saved["entities"], "by_concept": by_concept, "by_record": by_record}

    def snapshot(self, asset_ids):
        store = self.store
        catalogs, identities, scopes, fingerprints = {}, {}, {}, {}
        # The application's writes use this lock. Atomic file identity checks in
        # _load additionally catch replacement by another store/process.
        with store.library.lock:
            memberships, membership_identity = self._load(store.membership_path, store._memberships)
            for asset_id in sorted(set(asset_ids)):
                scope = memberships["assets"].get(asset_id)
                if scope is None:
                    continue
                try:
                    fingerprint = store.library.get_asset(asset_id)["fingerprint"]
                except (FileNotFoundError, ValueError):
                    continue
                if scope not in catalogs:
                    catalogs[scope], identities[scope] = self._load(
                        store._path(scope), lambda scope=scope: store._read(scope), self._catalog)
                scopes[asset_id], fingerprints[asset_id] = scope, fingerprint
        token = digest([membership_identity, identities, scopes, fingerprints])
        return KnowledgeSnapshot(token, catalogs, scopes, fingerprints)

    def _query_snapshot(self, filters, engine):
        # Most libraries can use visual search before importing any dossiers.
        # Keep their knowledge branch independent of the number of indexed rows.
        # File identity is still checked, so adding the first scope invalidates
        # this fast path without restarting the provider.
        with self.store.library.lock:
            memberships, identity = self._load(self.store.membership_path, self.store._memberships)
            if not memberships["assets"]:
                return KnowledgeSnapshot(digest([identity, {}, {}, {}]), {}, {}, {})
        return self.snapshot(engine.filtered_asset_ids(filters))

    def revision(self, filters, engine):
        return self._query_snapshot(filters, engine).token

    def candidate_refs(self, query, snapshot):
        selected = {scope: {row["concept_id"]: score for score, row in self.store._rank_concepts(saved, query)}
                    for scope, saved in snapshot.catalogs.items()}
        ids = set()
        for scope, concepts in selected.items():
            for concept_id in concepts:
                for binding in snapshot.catalogs[scope]["by_concept"].get(concept_id, ()):
                    if snapshot.asset_scopes.get(binding["asset_id"]) == scope:
                        ids.add(binding["record_id"])
        return selected, sorted(ids)

    def project(self, records, selected, snapshot, filters):
        results = []
        for record in records:
            asset_id = record.get("asset_id")
            scope = snapshot.asset_scopes.get(asset_id)
            if scope is None or not selected.get(scope):
                continue
            catalog = snapshot.catalogs[scope]
            projected = self.store._project(record, catalog, snapshot.fingerprints[asset_id],
                catalog["by_record"].get((asset_id, record["id"]), ()))
            if not self.store._matches_filters(projected, filters):
                continue
            match = self.store._occurrence_match(projected, selected[scope])
            if match is not None:
                results.append(match)
        results.sort(key=lambda pair: (-pair[0], pair[1]["asset_id"], pair[1]["start_ms"], pair[1]["id"]))
        return [row for _, row in results[:50]]

    def search(self, query, filters, engine):
        if not isinstance(query, str) or not query.strip():
            return []
        snapshot = self._query_snapshot(filters, engine)
        selected, ids = self.candidate_refs(query, snapshot)
        if not ids:
            self.stats["candidate_records"] = 0
            return []
        records = engine.filtered_records(filters, record_ids=ids)
        self.stats["candidate_records"] = len(records)
        return self.project(records, selected, snapshot, filters)
