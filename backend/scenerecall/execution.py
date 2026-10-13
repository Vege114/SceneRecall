"""Bounded unit scheduling with deterministic output and owned task lifetimes."""
from __future__ import annotations

import asyncio


class ReservationBusy(Exception):
    """A call can fit the budget after currently reserved calls finish."""


async def finish_on_cancel(operation):
    """Keep a submitted call and its receipt together during queue shutdown."""
    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            pass  # The operation has already accounted the provider failure.
        raise


def execution_width(config, stage, profile=None):
    options = config.get("execution") or {}
    if options.get(stage, "serial") != "parallel":
        return 1
    width = max(1, min(8, options.get("max_concurrency", 3)))
    return min(width, max(1, (profile or {}).get("max_concurrency", width)))


async def bounded_map(items, operation, width, checkpoint, on_result=None):
    """Stop admitting units on errors/control changes, and drain admitted work.

    Paid calls may finish while pausing or shutting down. Their operation owns
    its durable receipt/accounting; awaiting every owned task prevents orphaned
    calls and leaves those successes available to the next execution.
    """
    values = list(items)
    results = [None] * len(values)
    pending = {}
    cursor = 0
    try:
        while cursor < len(values) or pending:
            while cursor < len(values) and len(pending) < width:
                checkpoint()
                task = asyncio.create_task(operation(cursor, values[cursor]))
                pending[task] = cursor
                cursor += 1
            if not pending:
                break
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # Stable tie handling keeps diagnostics independent of set order.
            for task in sorted(done, key=pending.get):
                index = pending.pop(task)
                results[index] = task.result()
                if on_result:
                    on_result(index, results[index])
        checkpoint()
        return results
    except BaseException:
        if pending:
            # Shield the drain from cancellation of the queue loop itself.
            drain = asyncio.gather(*pending, return_exceptions=True)
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                await drain
        raise
