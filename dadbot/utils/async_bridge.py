from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Coroutine
from contextvars import copy_context
from typing import Any, TypeVar

from dadbot.core.runtime_errors import TransientExecutionError

logger = logging.getLogger(__name__)
T = TypeVar("T")


def run_coro_in_thread(coro: Coroutine[Any, Any, T], *, timeout_seconds: float = 180.0) -> T:
    context = copy_context()
    cancelled = threading.Event()
    running: concurrent.futures.Future[tuple[asyncio.AbstractEventLoop, asyncio.Task[Any]]] = (
        concurrent.futures.Future()
    )

    async def run() -> T:
        task = asyncio.current_task()
        assert task is not None
        running.set_result((asyncio.get_running_loop(), task))
        if cancelled.is_set():
            coro.close()
            raise asyncio.CancelledError
        return await coro

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    worker = pool.submit(context.run, asyncio.run, run())
    try:
        return worker.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        if worker.done():
            return worker.result()
        cancelled.set()
        if running.done():
            loop, task = running.result()
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                if not loop.is_closed():
                    raise
        logger.error("Coroutine thread bridge exceeded %ss; cancellation requested", timeout_seconds)
        raise TransientExecutionError(
            f"Coroutine thread bridge exceeded {timeout_seconds}s",
        ) from exc
    finally:
        # Waiting here would negate result(timeout=...) for an unresponsive worker.
        pool.shutdown(wait=False)
