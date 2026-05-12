from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dadbot.core.contracts.lifecycle_events import LifecycleEvent
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_writer import LedgerWriter


class LedgerWriterAdapter:
    """Single gateway for writing execution lifecycle events to the ledger."""

    def __init__(
        self,
        ledger: ExecutionLedger,
        *,
        scope_validator: Callable[[str], None] | None = None,
    ) -> None:
        self._writer = LedgerWriter(ledger)
        self._scope_validator = scope_validator

    def _guard(self, op: str) -> None:
        if callable(self._scope_validator):
            self._scope_validator(op)

    def append_job_submitted(self, job: Any) -> dict[str, Any]:
        self._guard("ledger.append_job_submitted")
        return self._writer.append_job_submitted(job)

    def append_job_queued(self, job: Any) -> dict[str, Any]:
        self._guard("ledger.append_job_queued")
        return self._writer.append_job_queued(job)

    def append_job_started(self, job: Any) -> dict[str, Any]:
        self._guard("ledger.append_job_started")
        return self._writer.append_job_started(job)

    def append_job_completed(self, job: Any, result: Any) -> dict[str, Any]:
        self._guard("ledger.append_job_completed")
        return self._writer.append_job_completed(job, result)

    def append_job_failed(
        self,
        job: Any,
        error: Any,
        *,
        failure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._guard("ledger.append_job_failed")
        return self._writer.append_job_failed(job, error, failure=failure)

    def append_session_bound(
        self,
        session_id: str,
        job_id: str = "",
        *,
        trace_token: str = "",
        step_key: str = "control_plane.bind_session",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        self._guard("ledger.append_session_bound")
        return self._writer.append_session_bound(
            session_id,
            job_id,
            trace_token=trace_token or str(legacy_trace or ""),
            step_key=step_key or str(legacy_step or ""),
            **legacy_kwargs,
        )

    def append_runtime_witness(
        self,
        component: str,
        trace_token: str = "",
        session_id: str = "",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        self._guard("ledger.append_runtime_witness")
        return self._writer.append_runtime_witness(
            component,
            trace_token=trace_token or str(legacy_trace or ""),
            session_id=session_id,
            **legacy_kwargs,
        )

    def write_event(self, **kwargs: Any) -> dict[str, Any]:
        self._guard("ledger.write_event")
        return self._writer.write_event(**kwargs)

    def append_execution_lifecycle(
        self,
        event: LifecycleEvent,
        *,
        session_id: str,
        trace_token: str = "",
        step_key: str = "",
        committed: bool = False,
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        self._guard("ledger.append_execution_lifecycle")
        return self._writer.append_execution_lifecycle(
            event,
            session_id=session_id,
            trace_token=trace_token or str(legacy_trace or ""),
            step_key=step_key or str(legacy_step or ""),
            committed=committed,
            **legacy_kwargs,
        )

    def append_effect_begin(
        self,
        *,
        session_id: str,
        trace_token: str = "",
        effect_id: str,
        request_id: str = "",
        step_key: str = "scheduler.execute.effect.begin",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        self._guard("ledger.append_effect_begin")
        return self._writer.append_effect_begin(
            session_id=session_id,
            trace_token=trace_token or str(legacy_trace or ""),
            effect_id=effect_id,
            request_id=request_id,
            step_key=step_key or str(legacy_step or ""),
            **legacy_kwargs,
        )

    def append_effect_commit(
        self,
        *,
        session_id: str,
        trace_token: str = "",
        effect_id: str,
        request_id: str = "",
        step_key: str = "scheduler.execute.effect.commit",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        self._guard("ledger.append_effect_commit")
        return self._writer.append_effect_commit(
            session_id=session_id,
            trace_token=trace_token or str(legacy_trace or ""),
            effect_id=effect_id,
            request_id=request_id,
            step_key=step_key or str(legacy_step or ""),
            **legacy_kwargs,
        )


__all__ = ["LedgerWriterAdapter"]
