"""Request-scoped search wall intervals, with no query text or process-global trace.

Offsets share one monotonic clock, including executor threads. Children can overlap;
their elapsed times must never be added to their parent's wall time.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import threading
import time
import uuid
from contextvars import ContextVar


_current = ContextVar("search_timing", default=None)
_parent = ContextVar("search_timing_parent", default=None)


class SearchTrace:
    def __init__(self):
        self.request_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        self.spans = []
        self.attributes = {}

    def snapshot(self):
        with self._lock:
            return {"schema": "search-timing-v1", "request_id": self.request_id,
                    "total_ms": round((time.perf_counter() - self.started) * 1000, 3),
                    "boundary": "application middleware entry to response headers ready; excludes socket ingress and body transfer",
                    "attributes": dict(self.attributes), "spans": [dict(row) for row in self.spans]}


def current_trace():
    return _current.get()


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
