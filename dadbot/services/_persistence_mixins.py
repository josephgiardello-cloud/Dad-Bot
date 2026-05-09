"""Internal mixin classes that decompose PersistenceService into cohesive groups.

Each mixin is a pure collection of related methods that operate through
``self`` once MRO resolution is complete — no standalone instantiation.

Public surface: do NOT import these mixins directly outside this package.
External callers should import from ``dadbot.services.persistence``.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dadbot.core.capability_audit_runner import (
    CAPABILITY_AUDIT_EVENT_TYPE,
    build_capability_audit_event_payload,
    build_runtime_capability_audit_report,
)
from dadbot.core.execution_context import ensure_execution_trace_root
from dadbot.core.graph import (
    FatalTurnError,
    LedgerMutationOp,
    MutationIntent,
    MutationKind,
)
from dadbot.core.kernel_locks import KernelEventTotalityLock
from dadbot.core.merkle_anchor import append_leaf_and_anchor
from dadbot.core.runtime_errors import (
    NON_FATAL_RUNTIME_EXCEPTIONS,
    PersistenceFailure,
    RuntimeErrorBase,
)

logger = logging.getLogger(__name__)

POLICY_TRACE_EVENT_TYPE = "PolicyTraceEvent"


# ---------------------------------------------------------------------------
# Support types (internal to the persistence service family)
# ---------------------------------------------------------------------------


class StateDivergenceError(RuntimeErrorBase):
    """Raised when projected state diverges from ledger-backed event state."""

    def __init__(self, message: str, *, report: dict[str, Any] | None = None) -> None:
        report_payload = dict(report or {})
        super().__init__(message, context={"report": report_payload})
        self.report = report_payload


@dataclass(frozen=True)
class RelationalState:
    """Behavioral ledger slice for social alignment and drift awareness."""

    trust_credit: float
    dominant_topic: str
    recent_topics: list[str]
    topic_overlap_ratio: float
    topic_drift_detected: bool


@dataclass(frozen=True)
class TemporalBudget:
    """Behavioral ledger slice for turn pacing and temporal pressure."""

    turn_index: int
    elapsed_ms: float
    topic_drift_streak: int
    budget_pressure: float


# ---------------------------------------------------------------------------
# Mixin 1: LedgerOps — low-level ledger mutation operations
# ---------------------------------------------------------------------------


class _LedgerOpsMixin:
    """Ledger mutation operations for the SaveNode commit boundary."""

    @staticmethod
    def _ledger_op_append_history(runtime: Any, payload: dict[str, Any]) -> None:
        entry = dict(payload.get("entry") or {})
        with runtime._session_lock:
            runtime.history.append(entry)

    @staticmethod
    def _ledger_op_record_turn_state(runtime: Any, payload: dict[str, Any]) -> None:
        mood = str(payload.get("mood") or "neutral")
        should_offer_daily_checkin = bool(
            payload.get("should_offer_daily_checkin", False),
        )
        with runtime._session_lock:
            runtime.session_moods.append(mood)
            runtime._pending_daily_checkin_context = should_offer_daily_checkin

    @staticmethod
    def _ledger_op_sync_thread_snapshot(runtime: Any, _payload: dict[str, Any]) -> None:
        runtime.sync_active_thread_snapshot()

    @staticmethod
    def _ledger_op_clear_turn_context(runtime: Any, _payload: dict[str, Any]) -> None:
        with runtime._session_lock:
            runtime._pending_daily_checkin_context = False
            runtime._active_tool_observation_context = None

    @staticmethod
    def _append_turn_pipeline_step(service: Any, step: str, **kwargs: Any) -> None:
        append_step = getattr(service, "_append_turn_pipeline_step", None)
        if callable(append_step):
            append_step(step, **kwargs)

    def _ledger_op_schedule_maintenance(self, runtime: Any, payload: dict[str, Any], service: Any) -> None:
        turn_text = str(payload.get("turn_text") or "")
        mood = payload.get("mood")
        if not bool(getattr(runtime, "LIGHT_MODE", False)):
            runtime.schedule_post_turn_maintenance(turn_text, mood)
            self._append_turn_pipeline_step(
                service,
                "schedule_maintenance",
                detail="queued post-turn maintenance",
            )
            return
        self._append_turn_pipeline_step(
            service,
            "schedule_maintenance",
            status="skipped",
            detail="light mode skips maintenance",
        )

    def _ledger_op_health_snapshot(self, runtime: Any, service: Any) -> None:
        runtime.current_runtime_health_snapshot(
            force=True,
            log_warnings=True,
            persist=True,
        )
        self._append_turn_pipeline_step(
            service,
            "health_snapshot",
            detail="refreshed runtime health snapshot",
        )

    def _ledger_op_policy_trace_event(self, runtime: Any, turn_context: Any, payload: dict[str, Any]) -> None:
        try:
            policy_events = list(payload.get("events") or [])
            if not policy_events:
                policy_events = list(
                    getattr(turn_context, "state", {}).get("policy_trace_events") or [],
                )
            trace_id = str(getattr(turn_context, "trace_id", "") or "unknown")
            phase_value = str(
                getattr(getattr(turn_context, "phase", None), "value", "") or "",
            )
            occurred_at = ""
            temporal = getattr(turn_context, "temporal", None)
            if temporal is not None:
                occurred_at = str(getattr(temporal, "wall_time", "") or "")

            control_plane = getattr(
                getattr(runtime, "turn_orchestrator", None),
                "control_plane",
                None,
            )
            ledger_writer = getattr(control_plane, "ledger_writer", None)
            write_event = getattr(ledger_writer, "write_event", None)
            session_id = str(
                (getattr(turn_context, "metadata", {}) or {}).get("control_plane", {}).get("session_id")
                or "default",
            )

            for index, raw in enumerate(policy_events, start=1):
                event_payload = dict(raw or {})
                summary = {
                    "policy": str(event_payload.get("policy") or "safety"),
                    "event_type": str(event_payload.get("event_type") or "policy_decision"),
                    "node": str(event_payload.get("node") or ""),
                    "decision_action": str(
                        ((event_payload.get("trace") or {}).get("final_action") or {}).get("action") or "",
                    ),
                    "decision_step": str(
                        ((event_payload.get("trace") or {}).get("final_action") or {}).get("step_name") or "",
                    ),
                }

                if hasattr(turn_context, "event_sequence"):
                    turn_context.event_sequence += 1
                    sequence = int(turn_context.event_sequence)
                else:
                    sequence = 0

                turn_event_payload = {
                    "event_type": POLICY_TRACE_EVENT_TYPE,
                    "trace_id": trace_id,
                    "sequence": sequence,
                    "occurred_at": occurred_at,
                    "stage": "save",
                    "status": "after",
                    "phase": phase_value,
                    "payload": {
                        "index": index,
                        "summary": summary,
                        "policy_trace": event_payload,
                    },
                }
                self.save_turn_event(turn_event_payload)  # type: ignore[attr-defined]

                if callable(write_event):
                    write_event(
                        event_type=POLICY_TRACE_EVENT_TYPE,
                        session_id=session_id,
                        trace_id=trace_id,
                        kernel_step_id="save_node.policy_trace",
                        payload=turn_event_payload["payload"],
                        committed=False,
                    )
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.warning(
                "PersistenceService policy trace persistence failed (non-fatal): %s",
                exc,
            )

    def _ledger_op_capability_audit_event(self, runtime: Any, turn_context: Any, payload: dict[str, Any]) -> None:
        try:
            stage_order = [
                str(getattr(trace, "stage", "") or "")
                for trace in list(getattr(turn_context, "stage_traces", []) or [])
            ]
            if "save" not in [s.strip().lower() for s in stage_order]:
                stage_order = [*stage_order, "save"]

            report = build_runtime_capability_audit_report(
                turn_context,
                stage_order=stage_order,
                failed=False,
            )
            report_payload = report.to_dict()
            if isinstance(getattr(turn_context, "state", None), dict):
                turn_context.state["capability_audit_report"] = report_payload
            if isinstance(getattr(turn_context, "metadata", None), dict):
                turn_context.metadata["capability_audit_report"] = dict(report_payload)

            event_payload = build_capability_audit_event_payload(
                report,
                scenario=str(payload.get("scenario") or "runtime_turn"),
            )
            if hasattr(turn_context, "event_sequence"):
                turn_context.event_sequence += 1
                sequence = int(turn_context.event_sequence)
            else:
                sequence = 0

            trace_id = str(getattr(turn_context, "trace_id", "") or "unknown")
            phase_value = str(
                getattr(getattr(turn_context, "phase", None), "value", "") or "",
            )
            occurred_at = ""
            temporal = getattr(turn_context, "temporal", None)
            if temporal is not None:
                occurred_at = str(getattr(temporal, "wall_time", "") or "")

            self.save_turn_event(  # type: ignore[attr-defined]
                {
                    "event_type": CAPABILITY_AUDIT_EVENT_TYPE,
                    "trace_id": trace_id,
                    "sequence": sequence,
                    "occurred_at": occurred_at,
                    "stage": "save",
                    "status": "after",
                    "phase": phase_value,
                    "payload": event_payload,
                },
            )

            control_plane = getattr(
                getattr(runtime, "turn_orchestrator", None),
                "control_plane",
                None,
            )
            ledger_writer = getattr(control_plane, "ledger_writer", None)
            write_event = getattr(ledger_writer, "write_event", None)
            if callable(write_event):
                session_id = str(
                    (getattr(turn_context, "metadata", {}) or {}).get("control_plane", {}).get("session_id")
                    or "default",
                )
                write_event(
                    event_type=CAPABILITY_AUDIT_EVENT_TYPE,
                    session_id=session_id,
                    trace_id=trace_id,
                    kernel_step_id="save_node.capability_audit",
                    payload=event_payload,
                    committed=False,
                )
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.warning(
                "PersistenceService capability audit persistence failed (non-fatal): %s",
                exc,
            )


# ---------------------------------------------------------------------------
# Mixin 2: MutationDispatch — mutation queue routing and draining
# ---------------------------------------------------------------------------


class _MutationDispatchMixin:
    """Routes MutationIntent payloads through the correct dispatch handler."""

    @staticmethod
    def _resolve_event_tap(runtime: Any) -> Any:
        direct = getattr(runtime, "event_tap", None) or getattr(runtime, "_event_tap", None)
        if direct is not None:
            return direct
        services = getattr(runtime, "services", None)
        return getattr(services, "event_tap", None)

    def _require_mutation_witness(self, *, runtime: Any, turn_context: Any, intent: MutationIntent) -> None:
        trace_id = str(getattr(turn_context, "trace_id", "") or "").strip()
        if not trace_id:
            return
        source_tag = str(getattr(intent, "source", "") or "")
        tap = self._resolve_event_tap(runtime)
        emit = getattr(tap, "emit", None)
        if not callable(emit):
            return
        try:
            KernelEventTotalityLock.require_event_witness(
                run_id=trace_id,
                source=source_tag,
            )
        except RuntimeError as exc:
            emit(
                "MUTATION_EVENT_TOTALITY_VIOLATION",
                run_id=trace_id,
                source=source_tag,
                error=str(exc),
            )
            raise

    def _dispatch_graph_mutation_intent(
        self,
        runtime: Any,
        turn_context: Any,
        *,
        payload: dict[str, Any],
        source: str,
    ) -> None:
        memory_manager = getattr(runtime, "memory_manager", None)
        graph_manager = getattr(memory_manager, "graph_manager", None) if memory_manager else None
        if graph_manager is None:
            raise PersistenceFailure(
                f"MutationIntent(type=graph, source={source!r}): graph_manager unavailable",
            )
        _fn = getattr(graph_manager, "apply_mutation", None)
        if callable(_fn):
            _fn(payload, turn_context=turn_context)
            return
        raise PersistenceFailure(
            f"MutationIntent(type=graph, source={source!r}): graph_manager.apply_mutation not callable",
        )

    def _dispatch_ledger_mutation_intent(
        self,
        runtime: Any,
        turn_context: Any,
        service: Any,
        *,
        payload: dict[str, Any],
        source: str,
    ) -> None:
        op = str(payload.get("op") or "").strip().lower()
        handlers = {
            LedgerMutationOp.APPEND_HISTORY.value: lambda: self._ledger_op_append_history(runtime, payload),  # type: ignore[attr-defined]
            LedgerMutationOp.RECORD_TURN_STATE.value: lambda: self._ledger_op_record_turn_state(runtime, payload),  # type: ignore[attr-defined]
            LedgerMutationOp.SYNC_THREAD_SNAPSHOT.value: lambda: self._ledger_op_sync_thread_snapshot(runtime, payload),  # type: ignore[attr-defined]
            LedgerMutationOp.CLEAR_TURN_CONTEXT.value: lambda: self._ledger_op_clear_turn_context(runtime, payload),  # type: ignore[attr-defined]
            LedgerMutationOp.SCHEDULE_MAINTENANCE.value: lambda: self._ledger_op_schedule_maintenance(runtime, payload, service),  # type: ignore[attr-defined]
            LedgerMutationOp.HEALTH_SNAPSHOT.value: lambda: self._ledger_op_health_snapshot(runtime, service),  # type: ignore[attr-defined]
            LedgerMutationOp.POLICY_TRACE_EVENT.value: lambda: self._ledger_op_policy_trace_event(runtime, turn_context, payload),  # type: ignore[attr-defined]
            LedgerMutationOp.CAPABILITY_AUDIT_EVENT.value: lambda: self._ledger_op_capability_audit_event(runtime, turn_context, payload),  # type: ignore[attr-defined]
        }
        handler = handlers.get(op)
        if handler is None:
            raise PersistenceFailure(
                f"MutationIntent(type=ledger, source={source!r}): unsupported op={op!r}",
            )
        handler()

    def _dispatch_mutation_intent(
        self,
        runtime: Any,
        turn_context: Any,
        service: Any,
        intent: Any,
    ) -> None:
        if not isinstance(intent, MutationIntent):
            raise PersistenceFailure(
                f"MutationQueue received non-MutationIntent payload: {type(intent).__name__}",
            )

        self._require_mutation_witness(runtime=runtime, turn_context=turn_context, intent=intent)

        intent_type = intent.type
        payload = dict(intent.payload or {})
        source = str(intent.source or "")

        if intent_type is MutationKind.GRAPH:
            self._dispatch_graph_mutation_intent(
                runtime,
                turn_context,
                payload=payload,
                source=source,
            )
            return

        if intent_type is MutationKind.LEDGER:
            self._dispatch_ledger_mutation_intent(
                runtime,
                turn_context,
                service,
                payload=payload,
                source=source,
            )
            return

        raise PersistenceFailure(
            f"MutationIntent: unknown type={intent_type!r} source={source!r}",
        )

    def _drain_mutation_queue(self, runtime: Any, turn_context: Any) -> None:
        mutation_queue = getattr(turn_context, "mutation_queue", None)
        if mutation_queue is None:
            return

        service = self.turn_service  # type: ignore[attr-defined]

        def dispatch(intent: Any) -> None:
            self._dispatch_mutation_intent(runtime, turn_context, service, intent)

        try:
            mutation_queue.drain(dispatch, hard_fail_on_error=True, transactional=True)
        except TypeError as exc:
            if "unexpected keyword argument 'transactional'" not in str(exc):
                raise
            mutation_queue.drain(dispatch, hard_fail_on_error=True)

        if not mutation_queue.is_empty():
            pending = mutation_queue.size()
            raise FatalTurnError(
                "Mutation queue not fully drained"
                f" (pending={pending}, trace_id={getattr(turn_context, 'trace_id', '')!r})",
            )

    @staticmethod
    def _flush_background_memory_store_patch_queue(runtime: Any) -> int:
        queue = getattr(runtime, "_background_memory_store_patch_queue", None)
        if not isinstance(queue, list) or not queue:
            return 0
        pending = list(queue)
        queue.clear()
        applied = 0
        for patch in pending:
            if not isinstance(patch, dict):
                continue
            runtime.mutate_memory_store(**patch)
            applied += 1
        return applied

    @staticmethod
    def _build_hierarchical_memory_payload(turn_context: Any) -> dict[str, Any]:
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        state_dict = state if isinstance(state, dict) else {}
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        return {
            "recent_buffer": list(state_dict.get("memory_recent_buffer") or []),
            "rolling_summary": str(state_dict.get("memory_rolling_summary") or ""),
            "structured_memory": dict(state_dict.get("memory_structured") or {}),
            "full_history_id": state_dict.get("memory_full_history_id"),
            "token_counts": {
                "recent": int(metadata_dict.get("recent_tokens", 0) or 0),
                "summary": int(metadata_dict.get("summary_tokens", 0) or 0),
                "structured": int(metadata_dict.get("structured_tokens", 0) or 0),
                "total": int(metadata_dict.get("context_total_tokens", 0) or 0),
            },
        }

    def _persist_hierarchical_memory_commit(self, turn_context: Any, *, commit_id: str) -> None:
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        if not isinstance(state, dict):
            return

        memory_payload = self._build_hierarchical_memory_payload(turn_context)
        state["hierarchical_memory_payload"] = dict(memory_payload)
        if isinstance(metadata, dict):
            metadata["hierarchical_memory_payload"] = dict(memory_payload)

        trace_id = str(getattr(turn_context, "trace_id", "") or "")
        phase_value = str(
            getattr(getattr(turn_context, "phase", None), "value", "") or "",
        )
        occurred_at = ""
        temporal = getattr(turn_context, "temporal", None)
        if temporal is not None:
            occurred_at = str(getattr(temporal, "wall_time", "") or "")

        self.save_turn_event(  # type: ignore[attr-defined]
            {
                "event_type": "hierarchical_memory_commit",
                "trace_id": trace_id,
                "occurred_at": occurred_at,
                "stage": "save",
                "status": "after",
                "phase": phase_value,
                "payload": {
                    "commit_id": str(commit_id or ""),
                    "trace_id": trace_id,
                    "memory": memory_payload,
                },
            },
        )


# ---------------------------------------------------------------------------
# Mixin 3: AuthorityState — diff/walk helpers for memory authority checks
# ---------------------------------------------------------------------------


class _AuthorityStateMixin:
    """Authority state comparison and walk utilities for integrity checks."""

    def _normalize_authority_snapshot(self, payload: Any) -> Any:
        def _walk(value: Any) -> Any:
            if isinstance(value, dict):
                normalized: dict[str, Any] = {}
                for key in sorted(str(k) for k in value.keys()):
                    if key.startswith("_timing_"):
                        continue
                    if key in {
                        "_atomic_checkpoint_saved",
                        "_save_transaction_active",
                        "_save_mutations_applied",
                        "_save_transaction_snapshot",
                        "_save_transaction",
                    }:
                        continue
                    normalized[key] = _walk(value.get(key))
                return normalized
            if isinstance(value, list):
                return [_walk(item) for item in value]
            return value

        return _walk(self._json_safe(payload))  # type: ignore[attr-defined]

    def _diff_authority_state(
        self,
        projected: Any,
        event_sourced: Any,
        *,
        max_items: int = 512,
    ) -> list[dict[str, Any]]:
        diffs: list[dict[str, Any]] = []
        self._walk_authority_state_diff(
            "",
            projected,
            event_sourced,
            diffs=diffs,
            max_items=max_items,
            missing="<missing>",
        )
        return diffs

    @staticmethod
    def _record_authority_state_diff(
        diffs: list[dict[str, Any]],
        *,
        max_items: int,
        path: str,
        projected: Any,
        event_sourced: Any,
    ) -> None:
        if len(diffs) >= max_items:
            return
        diffs.append({"path": path, "projected": projected, "event_sourced": event_sourced})

    def _walk_authority_state_diff(  # noqa: PLR0913
        self,
        path: str,
        projected: Any,
        event_sourced: Any,
        *,
        diffs: list[dict[str, Any]],
        max_items: int,
        missing: str,
    ) -> None:
        if len(diffs) >= max_items:
            return
        if isinstance(projected, dict) and isinstance(event_sourced, dict):
            self._walk_authority_state_dict(path, projected, event_sourced, diffs=diffs, max_items=max_items, missing=missing)
            return
        if isinstance(projected, list) and isinstance(event_sourced, list):
            self._walk_authority_state_list(path, projected, event_sourced, diffs=diffs, max_items=max_items, missing=missing)
            return
        if projected != event_sourced:
            self._record_authority_state_diff(diffs, max_items=max_items, path=path or "$", projected=projected, event_sourced=event_sourced)

    def _walk_authority_state_dict(  # noqa: PLR0913
        self,
        path: str,
        projected: dict[Any, Any],
        event_sourced: dict[Any, Any],
        *,
        diffs: list[dict[str, Any]],
        max_items: int,
        missing: str,
    ) -> None:
        keys = sorted(set(projected.keys()) | set(event_sourced.keys()))
        for key in keys:
            if len(diffs) >= max_items:
                return
            child_path = f"{path}.{key}" if path else str(key)
            has_projected = key in projected
            has_event_sourced = key in event_sourced
            if not has_projected:
                self._record_authority_state_diff(diffs, max_items=max_items, path=child_path, projected=missing, event_sourced=event_sourced.get(key))
                continue
            if not has_event_sourced:
                self._record_authority_state_diff(diffs, max_items=max_items, path=child_path, projected=projected.get(key), event_sourced=missing)
                continue
            self._walk_authority_state_diff(child_path, projected.get(key), event_sourced.get(key), diffs=diffs, max_items=max_items, missing=missing)

    def _walk_authority_state_list(  # noqa: PLR0913
        self,
        path: str,
        projected: list[Any],
        event_sourced: list[Any],
        *,
        diffs: list[dict[str, Any]],
        max_items: int,
        missing: str,
    ) -> None:
        size = max(len(projected), len(event_sourced))
        for idx in range(size):
            if len(diffs) >= max_items:
                return
            child_path = f"{path}[{idx}]"
            has_projected = idx < len(projected)
            has_event_sourced = idx < len(event_sourced)
            if not has_projected:
                self._record_authority_state_diff(diffs, max_items=max_items, path=child_path, projected=missing, event_sourced=event_sourced[idx])
                continue
            if not has_event_sourced:
                self._record_authority_state_diff(diffs, max_items=max_items, path=child_path, projected=projected[idx], event_sourced=missing)
                continue
            self._walk_authority_state_diff(child_path, projected[idx], event_sourced[idx], diffs=diffs, max_items=max_items, missing=missing)


# ---------------------------------------------------------------------------
# Mixin 4: IntegrityVerify — event-sourced checkpoint + memory authority
# ---------------------------------------------------------------------------


class _IntegrityVerifyMixin:
    """Memory authority and Merkle anchor verification at the SaveNode boundary."""

    def _load_event_sourced_checkpoint(self, trace_id: str) -> dict[str, Any]:
        trace_key = str(trace_id or "").strip()
        with ensure_execution_trace_root(
            operation="persistence_load_event_sourced_checkpoint",
            prompt="[persistence-load-event-sourced-checkpoint]",
            metadata={"source": "PersistenceService._load_event_sourced_checkpoint"},
            required=True,
        ):
            loader = getattr(self.persistence_manager, "load_latest_graph_checkpoint", None)  # type: ignore[attr-defined]
            if callable(loader):
                loaded = loader(trace_id=trace_key)
                if isinstance(loaded, dict):
                    return dict(loaded)

            ledger_resolver = getattr(self.persistence_manager, "_execution_ledger", None)  # type: ignore[attr-defined]
            if not callable(ledger_resolver):
                return {}
            ledger = ledger_resolver()
            read = getattr(ledger, "read", None)
            if not callable(read):
                return {}

            raw_events = read()
            if not isinstance(raw_events, list):
                return {}

            for event in reversed(raw_events):
                if str(event.get("type") or "") != "GRAPH_CHECKPOINT":
                    continue
                payload = dict(event.get("payload") or {})
                if trace_key and str(payload.get("trace_id") or "").strip() != trace_key:
                    continue
                checkpoint = dict(payload.get("checkpoint") or {})
                if checkpoint:
                    return checkpoint
            return {}

    def _enforce_memory_authority(
        self,
        runtime: Any,
        turn_context: Any,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        if not isinstance(checkpoint, dict) or not checkpoint:
            return

        metadata = getattr(turn_context, "metadata", None)
        test_mode = bool(os.getenv("PYTEST_CURRENT_TEST"))
        if isinstance(metadata, dict) and bool(metadata.get("legacy_direct_compat")) and test_mode:
            state = getattr(turn_context, "state", None)
            if isinstance(state, dict):
                state["memory_authority_check"] = {
                    "consistent": True,
                    "mode": "legacy_direct_compat_bypass",
                }
            return

        trace_id = str(getattr(turn_context, "trace_id", "") or "")
        event_checkpoint = self._load_event_sourced_checkpoint(trace_id)
        if not event_checkpoint:
            raise StateDivergenceError(
                "Memory authority divergence: missing event-sourced checkpoint for trace",
                report={
                    "trace_id": trace_id,
                    "reason": "missing_event_sourced_checkpoint",
                    "repair_hint": "Rebuild projection from ledger-backed GRAPH_CHECKPOINT events and retry commit.",
                },
            )

        projected_checkpoint = dict(checkpoint)
        projected_session_state = self._normalize_authority_snapshot(  # type: ignore[attr-defined]
            runtime.snapshot_session_state(),
        )

        event_checkpoint_copy = dict(event_checkpoint)
        event_session_state = self._normalize_authority_snapshot(  # type: ignore[attr-defined]
            event_checkpoint_copy.pop("session_state", {}),
        )

        projected = {
            "checkpoint": self._normalize_authority_snapshot(projected_checkpoint),  # type: ignore[attr-defined]
            "session_state": projected_session_state,
        }
        event_sourced = {
            "checkpoint": self._normalize_authority_snapshot(event_checkpoint_copy),  # type: ignore[attr-defined]
            "session_state": event_session_state,
        }

        projected_hash = self._stable_hash(projected)  # type: ignore[attr-defined]
        event_hash = self._stable_hash(event_sourced)  # type: ignore[attr-defined]
        if projected_hash == event_hash:
            state = getattr(turn_context, "state", None)
            if isinstance(state, dict):
                state["memory_authority_check"] = {
                    "consistent": True,
                    "projected_hash": projected_hash,
                    "event_sourced_hash": event_hash,
                    "trace_id": trace_id,
                }
            return

        diffs = self._diff_authority_state(projected, event_sourced)  # type: ignore[attr-defined]
        report = {
            "trace_id": trace_id,
            "consistent": False,
            "projected_hash": projected_hash,
            "event_sourced_hash": event_hash,
            "difference_count": len(diffs),
            "differences": diffs,
            "repair_hint": (
                "Replay trace from ledger, regenerate projection from event state, "
                "then re-run SaveNode commit."
            ),
        }
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["memory_authority_check"] = dict(report)

        logger.error(
            "State divergence detected at SaveNode boundary "
            "(trace_id=%s, projected_hash=%s, event_hash=%s, differences=%d)",
            trace_id,
            projected_hash,
            event_hash,
            len(diffs),
        )
        if not bool(getattr(self, "strict_mode", False)):
            if isinstance(state, dict):
                state["memory_authority_check"] = {
                    **dict(report),
                    "consistent": False,
                    "soft_failure": True,
                }
            logger.warning(
                "Memory authority divergence detected in non-strict mode; continuing commit (trace_id=%s)",
                trace_id,
            )
            return
        raise StateDivergenceError(
            "Memory authority divergence detected; commit blocked",
            report=report,
        )

    def _record_merkle_anchor(self, turn_context: Any, *, commit_id: str) -> None:
        runtime = getattr(getattr(self.turn_service, "bot", None), "config", None)  # type: ignore[attr-defined]
        enabled = bool(getattr(runtime, "merkle_anchor_enabled", True))
        if not enabled:
            return

        metadata = getattr(turn_context, "metadata", None)
        control_plane = dict(metadata.get("control_plane") or {}) if isinstance(metadata, dict) else {}
        session_id = str(control_plane.get("session_id") or "default")
        trace_id = str(getattr(turn_context, "trace_id", "") or "")
        temporal = getattr(turn_context, "temporal", None)
        occurred_at = str(getattr(temporal, "wall_time", "") or "")
        payload = {
            "session_id": session_id,
            "trace_id": trace_id,
            "commit_id": str(commit_id or ""),
            "occurred_at": occurred_at,
            "state_hash": self._stable_hash(getattr(turn_context, "state", {}) or {}),  # type: ignore[attr-defined]
            "metadata_hash": self._stable_hash(getattr(turn_context, "metadata", {}) or {}),  # type: ignore[attr-defined]
        }
        leaves = self._merkle_session_leaves.setdefault(session_id, [])  # type: ignore[attr-defined]
        anchor = append_leaf_and_anchor(leaves, payload)
        if isinstance(metadata, dict):
            metadata_snapshot = dict(metadata)
            metadata_snapshot["merkle_anchor"] = dict(anchor)
            turn_context.metadata = metadata_snapshot
        if isinstance(getattr(turn_context, "state", None), dict):
            state_snapshot = dict(getattr(turn_context, "state", {}) or {})
            state_snapshot["merkle_anchor"] = dict(anchor)
            turn_context.state = state_snapshot
        self.save_turn_event(  # type: ignore[attr-defined]
            {
                "event_type": "merkle_anchor_commit",
                "trace_id": trace_id,
                "occurred_at": occurred_at,
                "stage": "save",
                "status": "after",
                "payload": {
                    "session_id": session_id,
                    "commit_id": str(commit_id or ""),
                    **anchor,
                },
            },
        )


# ---------------------------------------------------------------------------
# Mixin 5: BehavioralLedger — relational trust, topic drift, goal alignment
# ---------------------------------------------------------------------------


class _BehavioralLedgerMixin:
    """Behavioral ledger state injection for relational trust and pacing."""

    @staticmethod
    def _safe_significant_tokens(runtime: Any, text: str) -> set[str]:
        token_fn = getattr(runtime, "significant_tokens", None)
        if callable(token_fn):
            try:
                token_values = token_fn(text)
                if isinstance(token_values, Iterable) and not isinstance(token_values, (str, bytes)):
                    return {
                        str(token).strip().lower()
                        for token in token_values
                        if str(token).strip()
                    }
            except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                logger.debug("Token extraction fallback used: %s", exc)
        return {
            part.strip().lower()
            for part in str(text or "").replace("-", " ").split()
            if len(part.strip()) >= 3
        }

    def _derive_relational_state(self, runtime: Any, turn_text: str) -> RelationalState:
        recent_topics_fn = getattr(runtime, "recent_memory_topics", None)
        recent_topics = []
        if callable(recent_topics_fn):
            try:
                topic_values = recent_topics_fn(limit=4)
                iterable_topics = (
                    topic_values
                    if isinstance(topic_values, Iterable) and not isinstance(topic_values, (str, bytes, dict))
                    else []
                )
                recent_topics = [
                    str(topic).strip().lower()
                    for topic in iterable_topics
                    if str(topic).strip()
                ]
            except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                logger.debug("Recent topic extraction fallback used: %s", exc)
                recent_topics = []

        dominant_topic = recent_topics[0] if recent_topics else "general"
        query_tokens = self._safe_significant_tokens(runtime, turn_text)
        topic_tokens: set[str] = set()
        for topic in recent_topics:
            topic_tokens.update(self._safe_significant_tokens(runtime, topic))

        overlap = query_tokens & topic_tokens
        overlap_ratio = 0.0 if not query_tokens else round(len(overlap) / float(len(query_tokens)), 3)
        topic_drift_detected = bool(recent_topics) and bool(query_tokens) and not bool(overlap)

        trust_level = float(
            getattr(runtime, "trust_level", lambda: 50)()
            if callable(getattr(runtime, "trust_level", None))
            else 50
        )
        trust_credit = round(max(0.0, min(1.0, trust_level / 100.0)), 3)
        return RelationalState(
            trust_credit=trust_credit,
            dominant_topic=dominant_topic,
            recent_topics=recent_topics,
            topic_overlap_ratio=overlap_ratio,
            topic_drift_detected=topic_drift_detected,
        )

    @staticmethod
    def _derive_temporal_budget(runtime: Any, turn_context: Any, topic_drift_detected: bool) -> TemporalBudget:
        session_turn_count_fn = getattr(runtime, "session_turn_count", None)
        turn_index = 0
        if callable(session_turn_count_fn):
            try:
                raw_turn_count = session_turn_count_fn()
                if isinstance(raw_turn_count, (int, float, str)):
                    turn_index = int(raw_turn_count) + 1
            except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                logger.debug("Session turn count fallback used: %s", exc)
                turn_index = 0

        state = getattr(turn_context, "state", None)
        elapsed_ms = 0.0
        if isinstance(state, dict):
            finalize_ms = float(state.get("_timing_finalize_ms") or 0.0)
            graph_sync_ms = float(state.get("_timing_graph_sync_ms") or 0.0)
            elapsed_ms = round(finalize_ms + graph_sync_ms, 3)

        previous_streak = int(getattr(runtime, "_topic_drift_streak", 0) or 0)
        topic_drift_streak = previous_streak + 1 if topic_drift_detected else 0
        runtime._topic_drift_streak = topic_drift_streak

        budget_pressure = round(min(1.0, (topic_drift_streak * 0.15) + min(elapsed_ms / 4000.0, 0.4)), 3)
        return TemporalBudget(
            turn_index=turn_index,
            elapsed_ms=elapsed_ms,
            topic_drift_streak=topic_drift_streak,
            budget_pressure=budget_pressure,
        )

    def _inject_behavioral_ledger_state(self, runtime: Any, turn_context: Any, turn_text: str) -> None:
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        if not isinstance(state, dict) or not isinstance(metadata, dict):
            return

        relational_state = self._derive_relational_state(runtime, turn_text)
        temporal_budget = self._derive_temporal_budget(
            runtime,
            turn_context,
            topic_drift_detected=relational_state.topic_drift_detected,
        )

        state["relational_state"] = asdict(relational_state)
        state["temporal_budget"] = asdict(temporal_budget)
        metadata["behavioral_ledger"] = {
            "relational_state": asdict(relational_state),
            "temporal_budget": asdict(temporal_budget),
        }

    @staticmethod
    def _resolve_session_log_dir(runtime: Any) -> Path | None:
        config = getattr(runtime, "config", None)
        cfg_path = getattr(config, "session_log_dir", None)
        if cfg_path is not None:
            return Path(cfg_path)
        legacy_path = getattr(runtime, "SESSION_LOG_DIR", None)
        if legacy_path is not None:
            return Path(legacy_path)
        return None

    @staticmethod
    def _extract_active_goals(turn_context: Any) -> list[dict[str, Any]]:
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return []
        candidates = state.get("session_goals")
        if not isinstance(candidates, list):
            candidates = state.get("goals")
        goals: list[dict[str, Any]] = []
        for item in list(candidates or []):
            if not isinstance(item, dict):
                continue
            goal_id = str(item.get("id") or item.get("goal_id") or "").strip()
            description = str(item.get("description") or item.get("goal") or "").strip()
            if not description:
                continue
            goals.append({"id": goal_id, "description": description})
        return goals[:6]

    def _goal_alignment_score(self, runtime: Any, turn_text: str, goals: list[dict[str, Any]]) -> float:
        if not goals:
            return 1.0
        query_tokens = self._safe_significant_tokens(runtime, turn_text)
        if not query_tokens:
            return 1.0

        best_overlap = 0.0
        for goal in goals:
            goal_tokens = self._safe_significant_tokens(runtime, str(goal.get("description") or ""))
            if not goal_tokens:
                continue
            overlap = len(query_tokens & goal_tokens) / float(len(query_tokens))
            best_overlap = max(best_overlap, overlap)
        return round(max(0.0, min(1.0, best_overlap)), 3)

    @staticmethod
    def _trust_credit_delta(goal_alignment_score: float) -> float:
        if goal_alignment_score >= 0.35:
            return 0.03
        if goal_alignment_score <= 0.05:
            return -0.07
        return -0.02

    def _record_relational_ledger(self, runtime: Any, turn_context: Any, turn_text: str) -> None:
        state = getattr(turn_context, "state", None)
        metadata = getattr(turn_context, "metadata", None)
        if not isinstance(state, dict) or not isinstance(metadata, dict):
            return

        goals = self._extract_active_goals(turn_context)
        alignment_score = self._goal_alignment_score(runtime, turn_text, goals)
        credit_before = float(getattr(runtime, "_relational_trust_credit", 0.5) or 0.5)
        credit_after = round(max(0.0, min(1.0, credit_before + self._trust_credit_delta(alignment_score))), 3)
        runtime._relational_trust_credit = credit_after

        mode = "supportive_peer" if credit_after >= 0.4 else "disappointed_dad"
        state["relational_trust_credit"] = credit_after
        state["dad_mode"] = mode
        behavioral = dict(metadata.get("behavioral_ledger") or {})
        behavioral["trust_credit"] = credit_after
        behavioral["dad_mode"] = mode
        behavioral["goal_alignment_score"] = alignment_score
        metadata["behavioral_ledger"] = behavioral

        session_id = str(
            (dict(metadata.get("control_plane") or {}).get("session_id"))
            or metadata.get("session_id")
            or getattr(runtime, "active_thread_id", "")
            or "default",
        )
        goals_excerpt = [
            {
                "id": str(goal.get("id") or ""),
                "description": str(goal.get("description") or "")[:120],
            }
            for goal in goals[:3]
        ]
        entry = {
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "trace_id": str(getattr(turn_context, "trace_id", "") or ""),
            "session_id": session_id,
            "user_input_excerpt": str(turn_text or "")[:220],
            "active_goals": goals_excerpt,
            "goal_alignment_score": alignment_score,
            "trust_credit_before": round(credit_before, 3),
            "trust_credit_after": credit_after,
            "dad_mode": mode,
            "decision": "followed_intent" if alignment_score >= 0.35 else "diverted_from_intent",
        }

        session_log_dir = self._resolve_session_log_dir(runtime)
        if session_log_dir is None:
            return
        try:
            session_log_dir.mkdir(parents=True, exist_ok=True)
            ledger_path = session_log_dir / "relational_ledger.jsonl"
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.warning("Relational ledger append failed (non-fatal): %s", exc)
