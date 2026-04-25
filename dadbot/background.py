from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, RLock
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BackgroundTask:
    task_id: str
    kind: str
    status: str = "queued"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class BackgroundTaskManager:
    """Thread-pool backed background runner for sync and async callables."""

    def __init__(self, max_workers: int = 8, thread_name_prefix: str = "dadbot-bg") -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._lock = RLock()
        self._shutdown = False
        self._shutdown_event = Event()
        self._tasks: dict[str, BackgroundTask] = {}
        self._running_tasks: dict[str, Future[Any]] = {}

    def submit(
        self,
        func,
        *args,
        task_kind: str = "background",
        metadata: dict[str, Any] | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> Future[Any]:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("BackgroundTaskManager is already shut down")

            resolved_task_id = str(task_id or uuid.uuid4().hex)
            task = BackgroundTask(
                task_id=resolved_task_id,
                kind=str(task_kind or "background"),
                metadata=dict(metadata or {}),
            )
            self._tasks[resolved_task_id] = task

        future = self.executor.submit(self._run_task, task.task_id, func, args, kwargs)
        future.dadbot_task_id = task.task_id
        future.dadbot_task_kind = task.kind
        with self._lock:
            self._running_tasks[task.task_id] = future
        future.add_done_callback(self._cleanup_task)
        return future

    def _run_task(self, task_id: str, func, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self._update_task_status(task_id, "running")
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
        except Exception as exc:
            self._update_task_status(task_id, "failed", error=str(exc))
            logger.exception("Background task %s failed", task_id)
            raise

        self._update_task_status(task_id, "completed")
        return result

    def _update_task_status(self, task_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = status
            if status == "running":
                task.started_at = datetime.now()
            elif status in {"completed", "failed"}:
                task.completed_at = datetime.now()
            if error:
                task.error = error

    def _cleanup_task(self, future: Future[Any]) -> None:
        task_id = getattr(future, "dadbot_task_id", None)
        if not task_id:
            return
        with self._lock:
            self._running_tasks.pop(task_id, None)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._shutdown_event.set()
        self.executor.shutdown(wait=wait, cancel_futures=False)

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        return self._shutdown_event.wait(timeout)

    def get_task(self, task_id: str) -> BackgroundTask | None:
        with self._lock:
            return self._tasks.get(str(task_id or ""))

    def list_tasks(self, limit: int = 20) -> list[BackgroundTask]:
        with self._lock:
            tasks = list(self._tasks.values())
        return sorted(tasks, key=lambda task: task.created_at, reverse=True)[: max(0, int(limit or 0))]


__all__ = ["BackgroundTask", "BackgroundTaskManager"]
