"""Owned, serial background preparation; HTTP readiness does not imply search readiness."""
from __future__ import annotations

import asyncio
import logging
import time

from .execution import run_in_thread
from .search import SearchGenerationConflict
from .subtitle_projection import compact_subtitle_records

logger = logging.getLogger(__name__)


class StartupPreparation:
    statistics_interval = 60.0

    def __init__(self, library, search, queue, *, start_worker=True):
        self.library, self.search, self.queue = library, search, queue
        self.start_worker = start_worker
        self.started = self.closed = self.worker_started = False
        self.task = None
        self.revision = self.active_mutations = 0
        self.idle = asyncio.Event()
        self.idle.set()
        self.states = {name: {"status": "pending"} for name in ("index", "statistics", "encoder")}
        self.stats = {}
        self.encoder = {"available": False, "offline": True, "reason": "正在检查本地检索模型"}
        self.statistics_checked = float("-inf")
        self.encoder_checked = float("-inf")

    def snapshot(self):
        return {name: dict(value) for name, value in self.states.items()}

    def start(self):
        self.started, self.closed = True, False
        self.request_refresh()

    def request_refresh(self):
        if not self.started or self.closed or self.task and not self.task.done():
            return
        if (self.states["index"]["status"] == "pending"
                or self.states["encoder"]["status"] == "pending"
                or self.states["encoder"]["status"] == "ready"
                and time.monotonic() - self.encoder_checked >= self.statistics_interval
                or self.states["statistics"]["status"] != "error"
                and time.monotonic() - self.statistics_checked >= self.statistics_interval):
            self.task = asyncio.create_task(self._run(), name="scenerecall-startup")

    def retry(self):
        for name, state in self.states.items():
            if state["status"] == "error":
                self.states[name] = {"status": "pending"}
                if name == "statistics":
                    self.statistics_checked = float("-inf")
        if not self.encoder.get("available"):
            self.states["encoder"] = {"status": "pending"}
        self.request_refresh()

    def begin_mutation(self):
        self.active_mutations += 1
        self.revision += 1
        self.idle.clear()

    def end_mutation(self):
        self.active_mutations -= 1
        self.revision += 1
        self.statistics_checked = float("-inf")
        if self.states["statistics"]["status"] == "ready":
            self.states["statistics"] = {"status": "pending"}
        if not self.active_mutations:
            self.idle.set()

    def _guard(self, revision):
        if self.closed or self.active_mutations or self.revision != revision:
            raise SearchGenerationConflict("资料正在更新，后台索引将重新核对")

    async def _index(self):
        self.states["index"] = {"status": "running", "message": "正在准备搜索索引，可以先浏览资料库"}
        while not self.closed:
            await self.idle.wait()
            revision = self.revision
            generation = self.search._generation
            raw = None

            def source():
                nonlocal raw
                raw = self.library.records(compact_subtitles=False)
                return compact_subtitle_records(raw)

            try:
                await self.search.rebuild_from_source(source, publication_guard=lambda: self._guard(revision))
                self._guard(revision)
                self.states["index"] = {"status": "ready"}
                return raw, revision, self.search._generation
            except Exception as exc:
                if (not isinstance(exc, SearchGenerationConflict) and revision == self.revision
                        and generation == self.search._generation and not self.active_mutations):
                    raise
                # A user edit or a completed job wins over the startup snapshot.
                # Release the old projection before collecting its replacement.
                raw = None
                await asyncio.sleep(0.25)

    async def _statistics(self, raw=None):
        await self.idle.wait()
        revision, generation = self.revision, self.search._generation
        if raw is not None:
            rows, snapshot_revision, snapshot_generation = raw
            raw = rows if (snapshot_revision, snapshot_generation) == (revision, generation) else None
        self.states["statistics"] = {"status": "running", "message": "正在更新资料库统计"}
        self.statistics_checked = time.monotonic()
        try:
            value = await run_in_thread(self.library.statistics, records=raw)
            if (revision != self.revision or generation != self.search._generation or self.active_mutations):
                self.states["statistics"] = {"status": "pending", "message": "资料已更新，统计稍后刷新"}
                return
            self.stats = value
            self.states["statistics"] = {"status": "ready"}
        except Exception:
            if revision != self.revision or generation != self.search._generation or self.active_mutations:
                self.states["statistics"] = {"status": "pending", "message": "资料已更新，统计稍后刷新"}
                return
            logger.exception("Library statistics preparation failed")
            self.states["statistics"] = {"status": "error", "message": "资料库统计更新失败，可重试"}

    async def _run(self):
        raw = None
        if self.states["index"]["status"] == "pending":
            try:
                raw = await self._index()
            except Exception:
                logger.exception("Search startup preparation failed")
                self.states["index"] = {"status": "error", "message": "搜索索引准备失败，请重试；详情见本机日志"}
        if self.states["index"]["status"] == "ready":
            if (self.states["statistics"]["status"] != "error"
                    and time.monotonic() - self.statistics_checked >= self.statistics_interval):
                await self._statistics(raw)
            raw = None
            if self.start_worker and not self.worker_started and not self.closed:
                await self.queue.start()
                self.worker_started = True
        elif self.states["statistics"]["status"] == "pending":
            self.states["statistics"] = {"status": "error", "message": "索引准备完成后可更新资料库统计"}
        if (self.states["encoder"]["status"] == "pending" or self.states["encoder"]["status"] == "ready"
                and time.monotonic() - self.encoder_checked >= self.statistics_interval):
            self.states["encoder"] = {"status": "running", "message": "正在检查本地检索模型"}
            try:
                self.encoder = await run_in_thread(self.search.visual_encoder.settings)
                self.encoder_checked = time.monotonic()
                self.states["encoder"] = {"status": "ready"}
            except Exception:
                logger.exception("Visual encoder preparation failed")
                self.states["encoder"] = {"status": "error", "message": "本地检索模型检查失败，可重试"}

    async def stop(self):
        self.closed = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
