from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dadbot.core.control_plane_projection import ExecutionProjection


@dataclass(slots=True)
class SchedulerLoop:
    """Minimal projection-driven scheduler loop.

    The loop itself is side-effect free except for the injected claim and
    dispatch callables. Runnable candidates come only from the projection.
    """

    projection: ExecutionProjection
    max_attempts: int = 3

    def runnable_candidates(
        self,
        *,
        execution_ids: Iterable[str],
        now: datetime | None = None,
    ) -> list[str]:
        return self.projection.get_runnable(
            now=now or datetime.now(),
            execution_ids=list(execution_ids),
        )

    def dispatch_once(
        self,
        *,
        execution_ids: Iterable[str],
        try_claim: Callable[[str], bool],
        dispatch_to_worker: Callable[[str], Any],
        now: datetime | None = None,
    ) -> list[str]:
        dispatched: list[str] = []
        for execution_id in self.runnable_candidates(execution_ids=execution_ids, now=now):
            state = self.projection.get(execution_id)
            if state is None:
                continue
            if int(state.attempt_count) >= int(self.max_attempts):
                continue
            if not bool(try_claim(execution_id)):
                continue
            dispatch_to_worker(execution_id)
            dispatched.append(execution_id)
        return dispatched
