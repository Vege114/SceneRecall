"""Every startup thread is drained before resources close, even on repeated stop."""
import asyncio
import threading

import pytest

from scenerecall.main import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["statistics", "encoder"])
async def test_shutdown_repeated_cancellation_drains_startup_threads(tmp_path, monkeypatch, phase):
    app = create_app(tmp_path, start_worker=False, token="startup-cancel-test")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    closed = []

    def work(*args, **kwargs):
        started.set()
        assert release.wait(5), "worker release timed out"
        finished.set()
        return {}

    async def close():
        assert finished.is_set(), "encoder closed while a preparation thread was running"
        closed.append(True)

    monkeypatch.setattr(app.state.library, "statistics", work if phase == "statistics" else lambda **_: {})
    monkeypatch.setattr(app.state.search.visual_encoder, "settings", work if phase == "encoder" else lambda: {})
    monkeypatch.setattr(app.state.search.visual_encoder, "close", close)
    context = app.router.lifespan_context(app)
    await context.__aenter__()
    shutdown = None
    try:
        assert await asyncio.to_thread(started.wait, 2), "preparation did not reach the selected phase"
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        await asyncio.sleep(.01)
        app.state.preparation.task.cancel()
        await asyncio.sleep(.01)
        assert not shutdown.done() and not closed
    finally:
        release.set()
        if shutdown is None:
            await context.__aexit__(None, None, None)
        else:
            await asyncio.wait_for(shutdown, 2)
    assert closed and app.state.preparation.task.done()
