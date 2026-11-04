"""Request-scoped search wall intervals, with no query text or process-global trace.

Offsets share one monotonic clock, including executor threads. Children can overlap;
their elapsed times must never be added to their parent's wall time.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import math
import threading
import time
import uuid
from contextvars import ContextVar


_current = ContextVar("search_timing", default=None)
_parent = ContextVar("search_timing_parent", default=None)
_IDENTITY_LIMIT = 32768


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _public_operations(values):
    return {name: {key: round(value, 3) if isinstance(value, float) else value
                   for key, value in stats.items()} for name, stats in values.items()}


class SearchTrace:
    def __init__(self):
        self.request_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        self.spans = []
        self.attributes = {}
        self.operations = {}
        self._operation_identities = {}
        self._span_operations = {}
        self._span_identities = {}
        self._span_counters = {}

    @staticmethod
    def _accumulate(values, identities, name, elapsed_ms, count, identity, attributes, failed):
        stats = values.setdefault(name, {"count": 0, "elapsed_ms": 0.0})
        stats["count"] += count
        stats["elapsed_ms"] += elapsed_ms
        if failed:
            stats["error_count"] = stats.get("error_count", 0) + 1
        for key, value in attributes.items():
            if _numeric(value) and key not in {"count", "elapsed_ms", "error_count", "identity_calls",
                                              "unique_count", "repeated_count", "identity_limit_reached"}:
                stats[key] = stats.get(key, 0) + value
        if identity is not None:
            # Identities stay request-local. Only counts ever enter the response.
            seen = identities.setdefault(name, set())
            stats["identity_calls"] = stats.get("identity_calls", 0) + 1
            stats.setdefault("unique_count", 0)
            stats.setdefault("repeated_count", 0)
            if identity in seen:
                stats["repeated_count"] += 1
            elif len(seen) < _IDENTITY_LIMIT:
                seen.add(identity)
                stats["unique_count"] += 1
            else:
                stats["identity_limit_reached"] = True

    def record_operation(self, parent, name, elapsed_ms, count, identity, attributes, failed):
        with self._lock:
            self._accumulate(self.operations, self._operation_identities, name,
                             elapsed_ms, count, identity, attributes, failed)
            if parent is not None:
                self._accumulate(self._span_operations.setdefault(parent, {}),
                                 self._span_identities.setdefault(parent, {}), name,
                                 elapsed_ms, count, identity, attributes, failed)

    def count(self, parent, name, value):
        if not _numeric(value):
            return
        with self._lock:
            target = self.attributes if parent is None else self._span_counters.setdefault(parent, {})
            target[name] = target.get(name, 0) + value

    def snapshot(self):
        with self._lock:
            spans = []
            for original in self.spans:
                row = dict(original)
                attributes = dict(row.get("attributes", {}))
                attributes.update(self._span_counters.get(row["id"], {}))
                for name, stats in _public_operations(self._span_operations.get(row["id"], {})).items():
                    attributes.update({f"{name}_{'ms' if key == 'elapsed_ms' else key}": value
                                       for key, value in stats.items()})
                row["attributes"] = attributes
                spans.append(row)
            return {"schema": "search-timing-v1", "request_id": self.request_id,
                    "total_ms": round((time.perf_counter() - self.started) * 1000, 3),
                    "boundary": "application middleware entry to response headers ready; excludes socket ingress and body transfer",
                    "attributes": dict(self.attributes), "spans": spans,
                    "operations": _public_operations(self.operations)}


def current_trace():
    return _current.get()


def timing_count(name, value=1):
    """Add a numeric count to the nearest span, without process-global deltas."""
    trace = current_trace()
    if trace is not None:
        trace.count(_parent.get(), name, value)


class aggregate_operation:
    """Aggregate small operations without allocating a timeline span per file.

    Local metrics belong only to the nearest span; request totals count each
    invocation once. Nested operation durations may overlap and are not additive.
    """
    def __init__(self, name, *, identity=None, count=1, **attributes):
        self.trace = current_trace()
        self.parent = _parent.get()
        self.name, self.identity, self.count = name, identity, count
        self.attributes = attributes

    def __enter__(self):
        if self.trace is not None:
            self.started = time.perf_counter()
        return self

    def set(self, **attributes):
        self.attributes.update(attributes)

    def __exit__(self, kind, error, tb):
        if self.trace is not None:
            self.trace.record_operation(self.parent, self.name, (time.perf_counter() - self.started) * 1000,
                                        self.count, self.identity, self.attributes, kind is not None)


@contextlib.contextmanager
def timed_lock(lock, name="lock_wait"):
    """Measure acquisition only, preserving the lock's ordinary lifetime."""
    with aggregate_operation(name):
        lock.acquire()
    try:
        yield
    finally:
        lock.release()


@contextlib.contextmanager
def trace_request():
    trace = SearchTrace()
    token, parent = _current.set(trace), _parent.set(None)
    try:
        yield trace
    finally:
        _parent.reset(parent)
        _current.reset(token)


class trace_span:
    def __init__(self, name, *, started_at=None, **attributes):
        self.name, self.attributes, self.started_at = name, attributes, started_at
        self.trace = current_trace()
        self.closed = False

    def __enter__(self):
        if self.trace is not None:
            self.started = self.started_at if self.started_at is not None else time.perf_counter()
            with self.trace._lock:
                self.row = {"id": len(self.trace.spans), "parent_id": _parent.get(), "name": self.name,
                            "start_ms": round((self.started - self.trace.started) * 1000, 3)}
                self.trace.spans.append(self.row)
            self.token = _parent.set(self.row["id"])
        return self

    def set(self, **attributes):
        self.attributes.update(attributes)

    def __exit__(self, kind, error, tb):
        if self.closed:
            return
        self.closed = True
        if self.trace is not None:
            ended = time.perf_counter()
            with self.trace._lock:
                self.row.update(end_ms=round((ended - self.trace.started) * 1000, 3),
                                elapsed_ms=round((ended - self.started) * 1000, 3),
                                attributes=dict(self.attributes), status="error" if kind else "ok")
            _parent.reset(self.token)


def record_interval(name, started_at, **attributes):
    with trace_span(name, started_at=started_at, **attributes):
        pass


def timed(name):
    """Keep signatures available to FastAPI and support executor callbacks."""
    def decorate(callback):
        if inspect.iscoroutinefunction(callback):
            @functools.wraps(callback)
            async def asynchronous(*args, **kwargs):
                with trace_span(name):
                    return await callback(*args, **kwargs)
            return asynchronous

        @functools.wraps(callback)
        def synchronous(*args, **kwargs):
            with trace_span(name):
                return callback(*args, **kwargs)
        return synchronous
    return decorate
