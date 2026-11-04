"""Disposable query snapshots, fenced by committed SQLite data revisions."""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from contextvars import ContextVar


_matrix_build_details = ContextVar("matrix_build_details", default=None)


def note_matrix_build(**values):
    """Attach known accounting to this build only, never keys or source content."""
    details = _matrix_build_details.get()
    if details is not None:
        details.update(values)


def initialize_revision(db):
    """Track writes from every connection, without changing job generations.

    FTS5's ordinary content table reflects lexical inserts/deletes; SQLite does
    not allow triggers on the virtual FTS table itself. Query-vector cache writes
    deliberately do not invalidate evidence snapshots.
    """
    db.execute("""CREATE TABLE IF NOT EXISTS query_revision (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
        identity TEXT NOT NULL, revision INTEGER NOT NULL)""")
    db.execute("INSERT OR IGNORE INTO query_revision VALUES (1,?,0)", (uuid.uuid4().hex,))
    tables = ("records", "namespaces", "visual_namespaces", "visual_cache",
              "visual_records", "visual_links", "evidence_fts_content")
    for table in tables:
        for action in ("INSERT", "UPDATE", "DELETE"):
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS query_revision_{table}_{action.lower()}
                AFTER {action} ON {table} BEGIN
                UPDATE query_revision SET revision=revision+1 WHERE singleton=1;
                END""")


def query_revision(db):
    row = db.execute("SELECT identity,revision FROM query_revision WHERE singleton=1").fetchone()
    return row[0], row[1]


class MatrixCache:
    """Byte-bounded LRU; one builder at a time bounds transient matrix storage.

    Builders return (snapshot, retained_bytes), or None to request exact streaming
    fallback. A failed allocation is also a fallback, never a query failure.
    Cached blocks are immutable and never contain result dictionaries.
    """
    def __init__(self, max_bytes=1024 * 1024 * 1024):
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._entries = OrderedDict()
        self._oversized = OrderedDict()
        self._max_bytes = max(0, int(max_bytes))
        self._bytes = 0
        self._epoch = 0
        self._counts = dict(hits=0, misses=0, builds=0, fallbacks=0, evictions=0)

    @property
    def max_bytes(self):
        with self._lock:
            return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value):
        with self._lock:
            self._max_bytes = max(0, int(value))
            self.clear()

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._oversized.clear()
            self._bytes = 0
            self._epoch += 1

    def stats(self):
        with self._lock:
            return {**self._counts, "bytes": self._bytes, "entries": len(self._entries),
                    "max_bytes": self._max_bytes}

    def _get(self, key):
        if key in self._entries:
            value = self._entries.pop(key)
            self._entries[key] = value
            self._counts["hits"] += 1
            return value[0]
        return None

    def _fallback_details(self, key, details):
        details.update(cache_status="fallback", cache_budget_bytes=self._max_bytes)
        if not self._max_bytes:
            details["fallback_reason"] = "cache_disabled"
        else:
            details.update(self._oversized.get(key) or {"fallback_reason": "unknown_memoized"})
            details["memoized_fallback"] = True

    def get_or_build(self, key, builder, *, diagnostics=None):
        details = diagnostics if diagnostics is not None else {}
        details.clear()
        details.update(cache_status="miss", fallback_reason=None, memoized_fallback=False,
                       build_attempted=False, attempted_retained_bytes=None, retained_bytes_phase=None)
        with self._lock:
            details["cache_budget_bytes"] = self._max_bytes
            value = self._get(key)
            if value is not None:
                details.update(cache_status="hit", cached_retained_bytes=self._entries[key][1])
                return value
            if not self._max_bytes or key in self._oversized:
                self._counts["fallbacks"] += 1
                self._fallback_details(key, details)
                return None
            self._counts["misses"] += 1
        # Double check after joining a concurrent builder. Only one construction
        # buffer is live even when different scopes miss simultaneously.
        with self._build_lock:
            with self._lock:
                value = self._get(key)
                if value is not None:
                    details.update(cache_status="hit", cached_retained_bytes=self._entries[key][1])
                    return value
                if not self._max_bytes or key in self._oversized:
                    self._counts["fallbacks"] += 1
                    self._fallback_details(key, details)
                    return None
                budget, epoch = self._max_bytes, self._epoch
                details.update(build_budget_bytes=budget, build_attempted=True)
            token = _matrix_build_details.set(details)
            try:
                built = builder(budget)
            except MemoryError:
                details["fallback_reason"] = "allocation_error"
                built = None
            finally:
                _matrix_build_details.reset(token)
            with self._lock:
                if built is None:
                    self._counts["fallbacks"] += 1
                    details.update(cache_status="fallback",
                                   fallback_reason=details["fallback_reason"] or "builder_declined")
                    if epoch == self._epoch:
                        self._oversized[key] = {field: details[field] for field in (
                            "fallback_reason", "attempted_retained_bytes", "retained_bytes_phase")}
                        while len(self._oversized) > 128:
                            self._oversized.popitem(last=False)
                    return None
                value, size = built
                details.update(cache_status="built", attempted_retained_bytes=size, retained_bytes_phase="complete")
                self._counts["builds"] += 1
                if epoch != self._epoch or size > self._max_bytes:
                    details.update(cache_status="built_uncached_epoch" if epoch != self._epoch else "built_uncached_budget",
                                   cache_budget_bytes=self._max_bytes)
                    return value
                while self._entries and self._bytes + size > self._max_bytes:
                    _, (_, old_size) = self._entries.popitem(last=False)
                    self._bytes -= old_size
                    self._counts["evictions"] += 1
                self._entries[key] = value, size
                self._bytes += size
                return value
