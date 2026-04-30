"""dadbot.core.trace_binder — Execution trace lifecycle authority.

Binds, validates, and releases the execution trace context for a single job
execution.  No graph imports, no policy logic — pure trace lifecycle management.
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable, Coroutine, TypeVar

from dadbot.core.execution_trace_context import ExecutionTraceRecorder, bind_execution_trace

_T = TypeVar("_T")


class TraceBinder:
    """Manages the :func:`bind_execution_trace` lifecycle for a single job.

    Authority contract
    ------------------
    - Validates ``trace_id`` is non-empty before binding.
    - Wraps an async callable inside ``bind_execution_trace``.
    - Zero graph imports, zero policy logic.
    """

    async def run(
        self,
        trace_id: str,
        prompt: str,
        metadata: dict[str, Any],
        fn: Callable[[], Coroutine[Any, Any, _T]],
    ) -> _T:
        """Bind trace context, execute *fn*, then release.

        Parameters
        ----------
        trace_id:
            Non-empty correlation identifier for the trace.
        prompt:
            Raw user prompt (stored in the recorder for downstream observers).
        metadata:
            Supplementary metadata attached to the recorder.
        fn:
            Zero-argument async callable to invoke inside the bound context.

        Raises
        ------
        RuntimeError
            If ``trace_id`` is empty.
        """
        if not trace_id:
            raise RuntimeError("TraceBinder.run requires a non-empty trace_id")

        recorder = ExecutionTraceRecorder(
            trace_id=trace_id,
            prompt=prompt,
            metadata=metadata,
        )
        with bind_execution_trace(recorder, required=False):
            return await fn()
