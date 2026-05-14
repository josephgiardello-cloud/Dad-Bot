from __future__ import annotations

import base64
import contextlib
import copy
import gzip
import hashlib
import json
import os
import pickle
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.core.execution_context import (
    RuntimeTraceViolation,
    active_execution_trace,
    ensure_execution_trace_root,
    record_execution_step,
)
from dadbot.core.contracts.lifecycle_events import (
    Claimed,
    LeaseExpired,
    LeaseRenewed,
    Redelivered,
    Released,
    Submitted,
)
from dadbot.core.control_plane_projection import ExecutionProjection
from dadbot.core.control_plane_reducer import ExecutionState, ExecutionStatus, lease_expired
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.execution_replay_engine import verify_terminal_state_replay_equivalence
from dadbot.core.kernel_locks import KernelReplaySequenceLock
from dadbot.core.persistence_schema_config import is_strict_persistence_schema_mode
from dadbot.core.persistence_record_schema import (
    PersistenceSchemaError,
    PERSISTENCE_SCHEMA_VERSION,
    assert_valid_checkpoint_record,
    assert_valid_replay_record,
    checkpoint_record_errors,
    normalize_checkpoint_record,
    normalize_replay_record,
    normalize_state_snapshot,
    normalize_trace_event,
    trace_event_errors,
)

POLICY_TRACE_EVENT_TYPE = "PolicyTraceEvent"
_WRITE_P95_SLO_MS = 15.0
_WRITE_P99_SLO_MS = 30.0
_COMPACTION_P95_SLO_MS = 40.0
_COMPACTION_P99_SLO_MS = 80.0
_PERSISTENCE_LATENCY_WINDOW = 256
_DEFAULT_COMPACTION_INTERVAL_EVENTS = 25
_ELEVATED_COMPACTION_INTERVAL_EVENTS = 50
_DEFAULT_RECOMMENDED_RETENTION_EVENTS = 1600
_HIGH_PRESSURE_RECOMMENDED_RETENTION_EVENTS = 1200
_COMPACTION_REPORT_SCHEMA_VERSION = "turn-compaction.v1"
_MAX_SNAPSHOT_BYTES = 256_000
_BANNED_SNAPSHOT_KEYS = {
    "state",
    "metadata",
    "session_state",
    "execution_trace",
    "execution_graph",
    "events",
    "memory_projection",
    "ui_projection",
    "replayed_state",
    "replayed_metadata",
}
_CANONICAL_CHECKPOINT_FIELDS = {
    "trace_id",
    "session_id",
    "stage",
    "status",
    "phase",
    "checkpoint_hash",
    "prev_checkpoint_hash",
    "event_sequence_id",
    "occurred_at",
    "execution_mode",
}


class ClaimConflict(RuntimeError):
    """Raised when an execution is already owned by another active worker."""


class NotOwner(RuntimeError):
    """Raised when a worker attempts to renew/release without ownership."""


class ConversationPersistenceManager:
    """Owns conversation persistence, snapshot rehydration, and session-log writes."""

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self._telemetry: dict[str, Any] = {
            "write_latencies_ms": [],
            "compaction_latencies_ms": [],
            "write_count": 0,
            "compaction_count": 0,
            "last_write_ms": 0.0,
            "last_compaction_ms": 0.0,
        }

    @staticmethod
    def _rolling_append(values: list[float], value: float, *, limit: int = _PERSISTENCE_LATENCY_WINDOW) -> None:
        values.append(float(value))
        if len(values) > int(limit):
            del values[: len(values) - int(limit)]

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(v) for v in values)
        idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * float(p)))))
        return float(ordered[idx])

    def persistence_telemetry_snapshot(self) -> dict[str, Any]:
        writes = list(self._telemetry.get("write_latencies_ms") or [])
        compactions = list(self._telemetry.get("compaction_latencies_ms") or [])
        write_p95 = self._percentile(writes, 0.95)
        write_p99 = self._percentile(writes, 0.99)
        compaction_p95 = self._percentile(compactions, 0.95)
        compaction_p99 = self._percentile(compactions, 0.99)

        ledger = None
        ledger_rebuilds = 0
        try:
            ledger = self._execution_ledger()
        except Exception:
            ledger = None
        if ledger is not None and callable(getattr(ledger, "telemetry_snapshot", None)):
            ledger_rebuilds = int((ledger.telemetry_snapshot() or {}).get("cache_rebuild_count") or 0)

        write_p95_ok = bool(write_p95 <= _WRITE_P95_SLO_MS)
        write_p99_ok = bool(write_p99 <= _WRITE_P99_SLO_MS)
        compaction_p95_ok = bool(compaction_p95 <= _COMPACTION_P95_SLO_MS)
        compaction_p99_ok = bool(compaction_p99 <= _COMPACTION_P99_SLO_MS)

        under_pressure = not (write_p95_ok and write_p99_ok and compaction_p95_ok and compaction_p99_ok)
        recommended_compaction_interval = (
            _ELEVATED_COMPACTION_INTERVAL_EVENTS
            if under_pressure
            else _DEFAULT_COMPACTION_INTERVAL_EVENTS
        )
        recommended_retention_events = (
            _HIGH_PRESSURE_RECOMMENDED_RETENTION_EVENTS
            if under_pressure
            else _DEFAULT_RECOMMENDED_RETENTION_EVENTS
        )

        return {
            "write_count": int(self._telemetry.get("write_count") or 0),
            "compaction_count": int(self._telemetry.get("compaction_count") or 0),
            "last_write_ms": float(self._telemetry.get("last_write_ms") or 0.0),
            "last_compaction_ms": float(self._telemetry.get("last_compaction_ms") or 0.0),
            "write_p95_ms": float(write_p95),
            "write_p99_ms": float(write_p99),
            "compaction_p95_ms": float(compaction_p95),
            "compaction_p99_ms": float(compaction_p99),
            "slo": {
                "write_p95_ms": float(_WRITE_P95_SLO_MS),
                "write_p99_ms": float(_WRITE_P99_SLO_MS),
                "compaction_p95_ms": float(_COMPACTION_P95_SLO_MS),
                "compaction_p99_ms": float(_COMPACTION_P99_SLO_MS),
            },
            "slo_ok": {
                "write_p95": bool(write_p95_ok),
                "write_p99": bool(write_p99_ok),
                "compaction_p95": bool(compaction_p95_ok),
                "compaction_p99": bool(compaction_p99_ok),
            },
            "policy": {
                "active_compaction_interval_events": int(_DEFAULT_COMPACTION_INTERVAL_EVENTS),
                "recommended_compaction_interval_events": int(recommended_compaction_interval),
                "recommended_retention_events": int(recommended_retention_events),
            },
            "cache_rebuild_count": int(ledger_rebuilds),
        }

    def _save_commit_active(self, turn_context: Any | None = None) -> bool:
        commit_active = bool(getattr(self.bot, "_graph_commit_active", False))
        if not commit_active:
            return False
        if turn_context is None:
            return True
        active_stage = str(getattr(turn_context, "state", {}).get("_active_graph_stage") or "").strip().lower()
        return active_stage in {"save", ""}

    @staticmethod
    def _payload_size_bytes(payload: dict[str, Any]) -> int:
        return len(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8"))

    def _assert_snapshot_invariants(self, payload: dict[str, Any], *, label: str) -> None:
        keys = set(str(k) for k in dict(payload or {}).keys())
        forbidden = sorted(keys & _BANNED_SNAPSHOT_KEYS)
        if forbidden:
            raise RuntimeError(f"{label} contains banned derived keys: {', '.join(forbidden)}")
        if self._payload_size_bytes(dict(payload or {})) > _MAX_SNAPSHOT_BYTES:
            raise RuntimeError(f"{label} exceeds max snapshot size ({_MAX_SNAPSHOT_BYTES} bytes)")

    def _thin_checkpoint_payload(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        source = dict(checkpoint or {})
        thin: dict[str, Any] = {}
        for key in _CANONICAL_CHECKPOINT_FIELDS:
            if key in source:
                thin[key] = copy.deepcopy(source.get(key))

        metadata = dict(source.get("metadata") or {})
        determinism = dict(metadata.get("determinism") or {})
        lock_hash = str(determinism.get("lock_hash") or "").strip()
        enforced = bool(determinism.get("enforced", False))
        if lock_hash or enforced:
            thin["determinism_lock"] = {
                "lock_hash": lock_hash,
                "enforced": enforced,
            }

        # Checkpoint is the canonical execution-state representation.
        # Persist deterministic execution state in-checkpoint, not as side payloads.
        execution_state = {
            "state": copy.deepcopy(dict(source.get("state") or {})),
            "metadata": copy.deepcopy(metadata),
            "phase_history": copy.deepcopy(list(source.get("phase_history") or [])),
            "stage_traces": copy.deepcopy(list(source.get("stage_traces") or [])),
            "event_sequence": int(source.get("event_sequence") or 0),
        }
        if any(execution_state.values()):
            thin["execution_state"] = execution_state
        return normalize_checkpoint_record(thin)

    @staticmethod
    def _stable_json_sha(payload: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8"),
        ).hexdigest()

    def _latest_checkpoint_for_trace(self, trace_token: str) -> dict[str, Any] | None:
        normalized = str(trace_token or "").strip()
        if not normalized:
            return None
        ledger = self._execution_ledger()
        for event in reversed(ledger.read()):
            if str(event.get("type") or "") != "GRAPH_CHECKPOINT":
                continue
            payload = dict(event.get("payload") or {})
            if str(payload.get("trace_id") or "").strip() != normalized:
                continue
            checkpoint = dict(payload.get("checkpoint") or {})
            if checkpoint:
                return normalize_checkpoint_record(checkpoint)
        return None

    def _build_checkpoint_continuity(self, checkpoint: dict[str, Any], *, trace_token: str) -> dict[str, Any]:
        replay = self.replay_turn_events(trace_token=trace_token)
        determinism = dict(replay.get("determinism") or {})
        continuity = {
            "execution_fingerprint": str(determinism.get("execution_fingerprint") or "").strip(),
            "strict_sequence_hash": str(replay.get("strict_sequence_hash") or "").strip(),
            "event_count": int(replay.get("event_count") or 0),
        }
        checkpoint_payload = dict(checkpoint or {})
        checkpoint_payload["continuity"] = continuity
        return normalize_checkpoint_record(checkpoint_payload)

    @staticmethod
    def _parse_iso_timestamp(value: str) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        with contextlib.suppress(ValueError):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        return None

    def _lease_expires_at(self, *, lease_seconds: int) -> str:
        anchor = self._parse_iso_timestamp(self._active_turn_wall_time()) or datetime.now()
        return (anchor + timedelta(seconds=max(int(lease_seconds or 0), 1))).isoformat(timespec="seconds")

    def _current_handoff_lease(self, trace_token: str) -> dict[str, Any]:
        normalized = str(trace_token or "").strip()
        if not normalized:
            return {
                "status": "unclaimed",
                "worker_id": "",
                "lease_expires_at": "",
                "lease_seconds": 0,
                "active": False,
            }

        _execution_id, state = self._lifecycle_state_for_trace(normalized)
        if state is None:
            return {
                "status": "unclaimed",
                "worker_id": "",
                "lease_expires_at": "",
                "lease_seconds": 0,
                "active": False,
            }

        now = datetime.now()
        expiry = state.lease_expiry
        expires_at = expiry.isoformat(timespec="seconds") if isinstance(expiry, datetime) else ""
        is_active = bool(state.owner and expiry and not lease_expired(state, now=now))
        status = "unclaimed"
        if state.status == ExecutionStatus.RELEASED:
            status = "released"
        elif state.status == ExecutionStatus.EXPIRED or bool(state.owner and expiry and lease_expired(state, now=now)):
            status = "expired"
        elif is_active:
            status = "claimed"

        lease_seconds = 0
        if expiry is not None:
            lease_seconds = max(0, int((expiry - now).total_seconds()))

        return {
            "status": status,
            "worker_id": str(state.owner or ""),
            "lease_expires_at": expires_at,
            "lease_seconds": int(lease_seconds),
            "active": bool(is_active),
        }

    def _lifecycle_projection(self) -> ExecutionProjection:
        projection = ExecutionProjection()
        projection.rebuild_from_ledger(self._execution_ledger().read())
        return projection

    def _resolve_execution_id_for_trace(self, trace_token: str) -> str:
        normalized_trace_id = str(trace_token or "").strip()
        if not normalized_trace_id:
            return ""
        ledger = self._execution_ledger()
        for event in reversed(ledger.read()):
            if str(event.get("type") or "") == "EXECUTION_LIFECYCLE":
                if str(event.get("trace_id") or "").strip() != normalized_trace_id:
                    continue
                payload = dict(event.get("payload") or {})
                execution_id = str(payload.get("execution_id") or "").strip()
                if execution_id:
                    return execution_id
            if str(event.get("type") or "") != "JOB_SUBMITTED":
                continue
            if str(event.get("trace_id") or "").strip() != normalized_trace_id:
                continue
            payload = dict(event.get("payload") or {})
            execution_id = str(payload.get("job_id") or "").strip()
            if execution_id:
                return execution_id
        return ""

    def _resolve_session_id_for_trace(self, trace_token: str) -> str:
        normalized_trace_id = str(trace_token or "").strip()
        if not normalized_trace_id:
            return "default"
        ledger = self._execution_ledger()
        for event in reversed(ledger.read()):
            if str(event.get("trace_id") or "").strip() != normalized_trace_id:
                continue
            session_id = str(event.get("session_id") or "").strip()
            if session_id:
                return session_id
        return "default"

    def _lifecycle_state_for_trace(self, trace_token: str) -> tuple[str, ExecutionState | None]:
        execution_id = self._resolve_execution_id_for_trace(trace_token)
        if not execution_id:
            return "", None
        state = self._lifecycle_projection().get(execution_id)
        return execution_id, state

    @staticmethod
    def _derived_execution_id(trace_token: str) -> str:
        normalized_trace = str(trace_token or "").strip()
        digest = hashlib.sha256(normalized_trace.encode("utf-8")).hexdigest()[:20]
        return f"job-{digest}"

    def _append_lifecycle_event_for_trace(
        self,
        *,
        trace_token: str,
        session_id: str,
        event: Any,
        step_key: str,
    ) -> None:
        self._append_ledger_event(
            event_type="EXECUTION_LIFECYCLE",
            trace_token=str(trace_token or "").strip(),
            payload=event.to_payload(),
            step_key=step_key,
            session_id=str(session_id or "default"),
        )

    def _assert_checkpoint_runtime_transition(self, checkpoint: dict[str, Any]) -> None:
        trace_id = str((checkpoint or {}).get("trace_id") or "").strip()
        if not trace_id:
            return
        prior = self._latest_checkpoint_for_trace(trace_id)
        if not isinstance(prior, dict):
            return
        current_prev = str((checkpoint or {}).get("prev_checkpoint_hash") or "").strip()
        prior_hash = str(prior.get("checkpoint_hash") or "").strip()
        if prior_hash and current_prev and current_prev != prior_hash:
            raise RuntimeError(
                "Invalid checkpoint transition: prev_checkpoint_hash does not match latest checkpoint_hash",
            )
        current_event_seq = int((checkpoint or {}).get("event_sequence_id") or 0)
        prior_event_seq = int(prior.get("event_sequence_id") or 0)
        if current_event_seq and prior_event_seq and current_event_seq < prior_event_seq:
            raise RuntimeError(
                "Invalid checkpoint transition: event_sequence_id regressed",
            )

    def _snapshot_mutation_queue(self) -> list[dict[str, Any]]:
        queue = getattr(self.bot, "_deferred_save_boundary_snapshots", None)
        if not isinstance(queue, list):
            queue = []
            self.bot._deferred_save_boundary_snapshots = queue
        return queue

    def _queue_snapshot_for_save_boundary(
        self,
        snapshot_payload: dict[str, Any],
    ) -> None:
        queue = self._snapshot_mutation_queue()
        queue.append(copy.deepcopy(dict(snapshot_payload or {})))
        if len(queue) > 32:
            del queue[:-32]

    def _active_turn_wall_time(self) -> str:
        temporal = getattr(self.bot, "_current_turn_time_base", None)
        wall_time = str(getattr(temporal, "wall_time", "") or "").strip()
        if wall_time:
            return wall_time
        return datetime.now().isoformat(timespec="seconds")

    def _active_turn_file_token(self) -> str:
        wall_time = self._active_turn_wall_time()
        compact = wall_time.replace(":", "").replace("T", "-")
        compact = compact.replace("+", "-").replace("Z", "")
        compact = compact.split(".")[0]
        return compact or datetime.now().strftime("%Y%m%d-%H%M%S")

    # DEPRECATED compat path — NO NEW CALLERS. This fallback allows callers that set
    # DADBOT_ALLOW_NULL_TRACE env or ALLOW_LEGACY_NULL_TRACE bot flag to bypass the
    # active trace requirement. Remove when all compat paths are fully migrated.
    # Expiry: 2026-Q3.
    def _is_legacy_direct_compat_mode(self) -> bool:
        """DEPRECATED: Check if we're in legacy direct-compat mode (marked explicitly only)."""
        # Only allow null-trace fallback if explicitly enabled via environment or flag
        env_flag = str(os.getenv("DADBOT_ALLOW_NULL_TRACE", "")).strip().lower()
        if env_flag in {"1", "true", "yes", "on"}:
            return True
        return bool(getattr(self.bot, "ALLOW_LEGACY_NULL_TRACE", False))

    def _require_active_trace(self, operation: str) -> None:
        # The null-trace fallback branch below is DEPRECATED \u2014 NO NEW CALLERS.
        # Production code must always run under an active execution trace context.
        # The fallback only applies to callers that explicitly set DADBOT_ALLOW_NULL_TRACE
        # or ALLOW_LEGACY_NULL_TRACE. Expiry: 2026-Q3.
        recorder = active_execution_trace()
        if recorder is None:
            if self._is_legacy_direct_compat_mode():
                # Preserve strict production behavior while unblocking legacy tests
                # that still call persistence surfaces outside the execution graph.
                # This fallback ONLY applies to explicitly marked compat paths.
                null_trace_events = getattr(self.bot, "_null_trace_events", None)
                if not isinstance(null_trace_events, list):
                    null_trace_events = []
                    self.bot._null_trace_events = null_trace_events
                null_trace_events.append(
                    {
                        "operation": str(operation or "").strip().lower(),
                        "layer": "persistence",
                        "fallback": "null_trace",
                    },
                )
                return
            raise RuntimeTraceViolation(
                f"ConversationPersistenceManager operation '{operation}' requires an active trace context",
            )
        record_execution_step(
            operation,
            payload={"layer": "persistence"},
            required=True,
        )

    def _execution_ledger(self) -> Any:
        orchestrator = getattr(self.bot, "turn_orchestrator", None)
        control_plane = getattr(orchestrator, "control_plane", None)
        ledger = getattr(control_plane, "ledger", None)
        if ledger is None:
            raise RuntimeTraceViolation("ExecutionLedger is required for conversation persistence authority")
        return ledger

    def _strict_schema_reject_enabled(self) -> bool:
        # Use centralized config resolver: checks env + bot attribute
        return is_strict_persistence_schema_mode(
            service_strict_mode=getattr(self.bot, "STRICT_SCHEMA_REJECT", None)
        )

    def _ledger_writer(self) -> LedgerWriter:
        return LedgerWriter(self._execution_ledger())

    def _append_ledger_event(
        self,
        *,
        event_type: str,
        trace_token: str,
        payload: dict[str, Any],
        step_key: str,
        session_id: str = "default",
    ) -> dict[str, Any]:
        return self._ledger_writer().write_event(
            str(event_type or "").strip() or "TURN_EVENT",
            session_id=str(session_id or "default"),
            trace_token=str(trace_token or "unknown").strip() or "unknown",
            step_key=str(step_key or "conversation_persistence"),
            payload=copy.deepcopy(dict(payload or {})),
        )

    def _derived_exports_enabled(self) -> bool:
        return bool(getattr(self.bot, "ENABLE_DERIVED_PERSISTENCE_EXPORTS", False))

    def _run_derived_async(self, fn: Any, *args: Any) -> None:
        if not self._derived_exports_enabled():
            return
        try:
            thread = threading.Thread(target=fn, args=args, daemon=True)
            thread.start()
        except Exception:
            pass

    def _apply_snapshot_mutations(
        self,
        snapshot_payload: dict[str, Any],
        *,
        turn_context: Any | None = None,
    ) -> None:
        chat_history = list(snapshot_payload.get("history", []))
        if chat_history and chat_history[0].get("role") == "system":
            chat_history = chat_history[1:]
        if not chat_history:
            return

        # Deterministic temporal anchors come from TurnContext; defer if absent.
        if turn_context is None:
            self._queue_snapshot_for_save_boundary(snapshot_payload)
            return

        previous_snapshot = self.bot.snapshot_session_state()
        self.bot.load_session_state_snapshot(snapshot_payload)
        try:
            self.bot.update_memory_store(chat_history, turn_context=turn_context)
            if not self.bot.LIGHT_MODE:
                self.bot.consolidate_memories(turn_context=turn_context)
            self.bot.archive_session_context(chat_history, turn_context=turn_context)
            if not self.bot.LIGHT_MODE:
                self.bot.refresh_relationship_timeline(
                    force=True,
                    turn_context=turn_context,
                )
            self.save_session_log(chat_history)
        finally:
            self.bot.load_session_state_snapshot(previous_snapshot)

    def flush_deferred_save_boundary_mutations(
        self,
        turn_context: Any | None = None,
    ) -> int:
        if not self._save_commit_active(turn_context):
            return 0
        queue = self._snapshot_mutation_queue()
        if not queue:
            return 0
        processed = 0
        pending = list(queue)
        queue.clear()
        for payload in pending:
            try:
                self._apply_snapshot_mutations(payload, turn_context=turn_context)
                processed += 1
            except Exception:
                # Re-queue failed payload for next SaveNode boundary.
                queue.append(payload)
        return processed

    def persist_conversation(self) -> None:
        with ensure_execution_trace_root(
            operation="persist_conversation",
            prompt="[conversation-persistence]",
            metadata={"source": "ConversationPersistenceManager.persist_conversation"},
            required=True,
        ):
            self._require_active_trace("persist_conversation")
            chat_history = list(self.bot.conversation_history() or [])
            if not chat_history:
                return
            # Strict mode: persist only (non-mutating). Memory/relationship mutation is SaveNode-owned.
            self.save_session_log(chat_history)

    def persist_conversation_snapshot(
        self,
        snapshot: dict,
        turn_context: Any | None = None,
    ) -> None:
        with ensure_execution_trace_root(
            operation="persist_conversation_snapshot",
            prompt="[conversation-persistence-snapshot]",
            metadata={
                "source": "ConversationPersistenceManager.persist_conversation_snapshot",
            },
            required=True,
        ):
            self._require_active_trace("persist_conversation_snapshot")
            snapshot_payload = copy.deepcopy(dict(snapshot or {}))
            chat_history = list(snapshot_payload.get("history", []))
            if chat_history and chat_history[0].get("role") == "system":
                chat_history = chat_history[1:]
            if not chat_history:
                return

            if not self._save_commit_active(turn_context):
                self._queue_snapshot_for_save_boundary(snapshot_payload)
                self.save_session_log(chat_history)
                return

            self._apply_snapshot_mutations(snapshot_payload, turn_context=turn_context)

    def save_session_log(self, history: list[dict]) -> None:
        self._require_active_trace("save_session_log")
        created_at = self._active_turn_wall_time()
        payload = {
            "created_at": created_at,
            "tenant_id": self.bot.config.tenant_id,
            "model": self.bot.config.active_model,
            "embedding_model": self.bot.config.active_embedding_model,
            "session_summary": self.bot.session_summary,
            "relationship_state": self.bot.relationship_state(),
            "history": history,
        }
        timestamp = self._active_turn_file_token()
        io_lock = getattr(self.bot, "_io_lock", None)
        if io_lock is None:
            if self.bot._tenant_document_store is not None:
                self.bot._tenant_document_store.save_session_state(
                    f"session-log:{timestamp}",
                    payload,
                )
                return
            self.bot.config.session_log_dir.mkdir(parents=True, exist_ok=True)
            session_path = self.bot.config.session_log_dir / f"session-{timestamp}.json"
            self.bot.write_json_atomically(session_path, payload, backup=False)
            return

        with io_lock:
            if self.bot._tenant_document_store is not None:
                self.bot._tenant_document_store.save_session_state(
                    f"session-log:{timestamp}",
                    payload,
                )
                return
            self.bot.SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
            session_path = self.bot.SESSION_LOG_DIR / f"session-{timestamp}.json"
            self.bot.write_json_atomically(session_path, payload, backup=False)

    def persist_graph_checkpoint(
        self,
        checkpoint: dict,
        _skip_turn_event: bool = False,
    ) -> None:
        self._require_active_trace("persist_graph_checkpoint")
        payload = copy.deepcopy(dict(checkpoint or {}))
        if not payload:
            return
        payload = self._thin_checkpoint_payload(payload)

        trace_id = str(payload.get("trace_id") or "unknown").strip() or "unknown"
        stage = str(payload.get("stage") or "stage").strip().replace(" ", "-") or "stage"
        status = str(payload.get("status") or "unknown").strip().replace(" ", "-") or "unknown"
        payload["trace_id"] = trace_id
        payload["stage"] = stage
        payload["status"] = status
        payload.setdefault("execution_mode", "live")
        if not str(payload.get("occurred_at") or "").strip():
            payload["occurred_at"] = self._active_turn_wall_time()
        if self._strict_schema_reject_enabled():
            payload = assert_valid_checkpoint_record(payload)
        else:
            payload = normalize_checkpoint_record(payload)
            errors = checkpoint_record_errors(payload)
            if errors:
                raise PersistenceSchemaError(
                    "Malformed checkpoint payload: " + "; ".join(errors),
                )
        self._assert_checkpoint_runtime_transition(payload)
        payload = self._build_checkpoint_continuity(payload, trace_token=trace_id)
        self._assert_snapshot_invariants(payload, label="graph_checkpoint")
        self._append_ledger_event(
            event_type="GRAPH_CHECKPOINT",
            trace_token=trace_id,
            payload={
                "trace_id": trace_id,
                "stage": stage,
                "status": status,
                "checkpoint": payload,
            },
            step_key="persist_graph_checkpoint",
            session_id=str(payload.get("session_id") or "default"),
        )

        # When called directly (not from TurnGraph), write a turn event so that
        # validate_replay_determinism can fold the lock_hash from checkpoints.
        # TurnGraph calls persist_graph_checkpoint with _skip_turn_event=True and
        # emits its own turn event via save_turn_event to avoid duplicate writes.
        if not _skip_turn_event:
            determinism_lock = dict(payload.get("determinism_lock") or {})
            session_state = {}
            with contextlib.suppress(Exception):
                maybe_snapshot = self.bot.snapshot_session_state()
                if isinstance(maybe_snapshot, dict):
                    session_state = maybe_snapshot
            self.persist_turn_event(
                {
                    "event_type": "graph_checkpoint",
                    "trace_id": trace_id,
                    "stage": stage,
                    "status": status,
                    "determinism_lock": determinism_lock,
                    "checkpoint": payload,
                    "session_state": session_state,
                },
            )

        # Optional derived export layer; never used as runtime authority.
        self._run_derived_async(
            self._export_checkpoint_projection,
            trace_id,
            stage,
            status,
            payload,
        )

    def _export_checkpoint_projection(
        self,
        trace_token: str,
        stage: str,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        timestamp = self._active_turn_file_token()
        binary_payload = gzip.compress(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        )
        wrapped_payload = {
            "format": "gzip+pickle",
            "trace_id": trace_token,
            "stage": stage,
            "status": status,
            "created_at": self._active_turn_wall_time(),
            "payload_b64": base64.b64encode(binary_payload).decode("ascii"),
        }
        if self.bot._tenant_document_store is not None:
            checkpoint_key = f"graph-checkpoint:{trace_token}:{timestamp}:{stage}:{status}"
            self.bot._tenant_document_store.save_session_state(checkpoint_key, wrapped_payload)
            self.bot._tenant_document_store.save_session_state(f"graph-checkpoint-latest:{trace_token}", wrapped_payload)
            self.bot._tenant_document_store.save_session_state("graph-checkpoint-latest", wrapped_payload)
            return
        checkpoint_dir = self.bot.SESSION_LOG_DIR / "graph_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{timestamp}-{trace_token[:12]}-{stage}-{status}.bin"
        latest_path = checkpoint_dir / f"latest-{trace_token[:12]}.bin"
        latest_global_path = checkpoint_dir / "latest.bin"
        checkpoint_path.write_bytes(binary_payload)
        latest_path.write_bytes(binary_payload)
        latest_global_path.write_bytes(binary_payload)

    def _turn_event_dir(self) -> Path:
        event_dir = self.bot.SESSION_LOG_DIR / "turn_events"
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir

    def _turn_event_path(self, trace_token: str) -> Path:
        normalized = str(trace_token or "unknown").strip() or "unknown"
        return self._turn_event_dir() / f"{normalized[:40]}.jsonl"

    def _snapshot_path(self, trace_token: str) -> Path:
        normalized = str(trace_token or "unknown").strip() or "unknown"
        return self._turn_event_dir() / f"snapshot-{normalized[:40]}.json"

    def _compaction_report_path(self, trace_token: str) -> Path:
        normalized = str(trace_token or "unknown").strip() or "unknown"
        return self._turn_event_dir() / f"compaction-{normalized[:40]}.json"

    def _current_recommended_retention_events(self) -> int:
        writes = list(self._telemetry.get("write_latencies_ms") or [])
        compactions = list(self._telemetry.get("compaction_latencies_ms") or [])
        write_p95 = self._percentile(writes, 0.95)
        write_p99 = self._percentile(writes, 0.99)
        compaction_p95 = self._percentile(compactions, 0.95)
        compaction_p99 = self._percentile(compactions, 0.99)
        under_pressure = not (
            bool(write_p95 <= _WRITE_P95_SLO_MS)
            and bool(write_p99 <= _WRITE_P99_SLO_MS)
            and bool(compaction_p95 <= _COMPACTION_P95_SLO_MS)
            and bool(compaction_p99 <= _COMPACTION_P99_SLO_MS)
        )
        return (
            _HIGH_PRESSURE_RECOMMENDED_RETENTION_EVENTS
            if under_pressure
            else _DEFAULT_RECOMMENDED_RETENTION_EVENTS
        )

    def _semantic_compaction_summary(self) -> str:
        memory_store = getattr(self.bot, "MEMORY_STORE", None)
        if hasattr(memory_store, "get"):
            summary = str(memory_store.get("last_memory_compaction_summary") or "").strip()
            if summary:
                return summary

        consolidated_getter = getattr(self.bot, "consolidated_memories", None)
        if callable(consolidated_getter):
            entries = list(consolidated_getter() or [])[-3:]
            snippets = [
                str(entry.get("summary") or "").strip()
                for entry in entries
                if isinstance(entry, dict) and str(entry.get("summary") or "").strip()
            ]
            if snippets:
                return " | ".join(snippets)

        return ""

    def _compute_next_sequence(
        self,
        trace_token: str,
        *,
        events_snapshot: list[dict[str, Any]] | None = None,
    ) -> int:
        _ = trace_token
        _ = events_snapshot
        ledger = self._execution_ledger()
        get_next_trace_sequence = getattr(ledger, "get_next_trace_sequence", None)
        if callable(get_next_trace_sequence):
            return int(get_next_trace_sequence(trace_token))
        get_next_sequence = getattr(ledger, "get_next_sequence", None)
        if callable(get_next_sequence):
            return int(get_next_sequence())
        events = self.list_turn_events(
            trace_token=trace_token,
            events_snapshot=events_snapshot,
        )
        if not events:
            return 1
        return int(events[-1].get("sequence") or len(events)) + 1

    @staticmethod
    def _use_full_compaction_snapshot(sequence: int) -> bool:
        # Most compaction checkpoints can run off the tail snapshot; periodically
        # force a full ledger view to keep strict hash/reporting anchored.
        return sequence % 250 == 0

    def persist_turn_event(self, event: dict[str, Any]) -> None:
        self._require_active_trace("persist_turn_event")
        write_started = time.perf_counter()
        payload = copy.deepcopy(dict(event or {}))
        if not payload:
            return
        trace_id = str(payload.get("trace_id") or "unknown").strip() or "unknown"
        payload["trace_id"] = trace_id
        sequence = self._compute_next_sequence(trace_id)
        payload.setdefault("sequence", sequence)
        event_id_seed = f"{trace_id}:{int(payload.get('sequence') or sequence)}"
        payload.setdefault(
            "event_id",
            hashlib.sha256(event_id_seed.encode()).hexdigest()[:16],
        )
        payload.setdefault("occurred_at", self._active_turn_wall_time())
        payload.setdefault("persistence_schema_version", PERSISTENCE_SCHEMA_VERSION)
        payload = normalize_trace_event(payload)
        self._append_ledger_event(
            event_type="TURN_EVENT",
            trace_token=trace_id,
            payload=payload,
            step_key="persist_turn_event",
            session_id=str(payload.get("session_id") or "default"),
        )

        # Optional derived export layer; never used as runtime authority.
        self._run_derived_async(self._export_turn_event_projection, trace_id, payload)

        # Snapshot compaction every 25 events: replay can start from latest snapshot.
        compact_sequence = int(payload.get("sequence") or 0)
        if compact_sequence > 0 and compact_sequence % _DEFAULT_COMPACTION_INTERVAL_EVENTS == 0:
            compaction_started = time.perf_counter()
            ledger = self._execution_ledger()
            snapshot_full = self._use_full_compaction_snapshot(compact_sequence)
            compaction_snapshot = ledger.read(full=snapshot_full)
            replay = self.replay_turn_events(
                trace_token=trace_id,
                events_snapshot=compaction_snapshot,
            )
            determinism = dict(replay.get("determinism") or {})
            snapshot_payload = {
                "trace_id": trace_id,
                "created_at": self._active_turn_wall_time(),
                "last_sequence": compact_sequence,
                "phase": replay.get("phase"),
                "persistence_schema_version": PERSISTENCE_SCHEMA_VERSION,
                "strict_sequence_hash": str(replay.get("strict_sequence_hash") or ""),
                "event_count": int(replay.get("event_count") or 0),
                "determinism_lock": {
                    "lock_hash": str(determinism.get("lock_hash") or ""),
                    "consistent": bool(determinism.get("consistent", True)),
                },
            }
            snapshot_payload = normalize_state_snapshot(snapshot_payload)
            self._assert_snapshot_invariants(snapshot_payload, label="turn_snapshot")
            if self.bot._tenant_document_store is not None:
                self.bot._tenant_document_store.save_session_state(
                    f"turn-snapshot-latest:{trace_id}",
                    snapshot_payload,
                )
            else:
                snapshot_path = self._snapshot_path(trace_id)
                snapshot_path.write_text(
                    json.dumps(snapshot_payload, sort_keys=True, ensure_ascii=True),
                    encoding="utf-8",
                )
            compaction_ms = (time.perf_counter() - compaction_started) * 1000.0
            self._telemetry["compaction_count"] = int(self._telemetry.get("compaction_count") or 0) + 1
            self._telemetry["last_compaction_ms"] = float(compaction_ms)
            self._rolling_append(self._telemetry["compaction_latencies_ms"], compaction_ms)

            compaction_report = {
                "schema_version": _COMPACTION_REPORT_SCHEMA_VERSION,
                "persistence_schema_version": PERSISTENCE_SCHEMA_VERSION,
                "trace_id": trace_id,
                "created_at": snapshot_payload["created_at"],
                "last_sequence": compact_sequence,
                "snapshot_mode": "full" if snapshot_full else "tail",
                "event_count": int(replay.get("event_count") or 0),
                "strict_sequence_hash": str(replay.get("strict_sequence_hash") or ""),
                "recommended_retention_events": self._current_recommended_retention_events(),
                "semantic_compaction_summary": self._semantic_compaction_summary(),
                "consolidated_memory_count": len(list(getattr(self.bot, "consolidated_memories", lambda: [])() or [])),
                "compaction_latency_ms": float(compaction_ms),
            }
            self._compaction_report_path(trace_id).write_text(
                json.dumps(compaction_report, sort_keys=True, ensure_ascii=True),
                encoding="utf-8",
            )

        write_ms = (time.perf_counter() - write_started) * 1000.0
        self._telemetry["write_count"] = int(self._telemetry.get("write_count") or 0) + 1
        self._telemetry["last_write_ms"] = float(write_ms)
        self._rolling_append(self._telemetry["write_latencies_ms"], write_ms)

    def _export_turn_event_projection(self, trace_token: str, payload: dict[str, Any]) -> None:
        if self.bot._tenant_document_store is not None:
            key = f"turn-events:{trace_token}"
            existing = self.bot._tenant_document_store.load_session_state(key)
            events = list(existing) if isinstance(existing, list) else []
            events.append(payload)
            self.bot._tenant_document_store.save_session_state(key, events)
            self.bot._tenant_document_store.save_session_state(f"turn-events-latest:{trace_token}", payload)
            return
        event_path = self._turn_event_path(trace_token)
        line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def list_turn_events(
        self,
        trace_token: str,
        limit: int = 0,
        *,
        events_snapshot: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        self._require_active_trace("list_turn_events")
        normalized = str(trace_token or "").strip()
        if not normalized:
            return []

        snapshot = events_snapshot
        if snapshot is None:
            ledger = self._execution_ledger()
            snapshot = ledger.read()
        events: list[dict[str, Any]] = []
        strict_reject = self._strict_schema_reject_enabled()
        for event in snapshot:
            if str(event.get("type") or "") != "TURN_EVENT":
                continue
            payload = dict(event.get("payload") or {})
            if str(payload.get("trace_id") or "").strip() != normalized:
                continue
            normalized_payload = normalize_trace_event(payload)
            errors = trace_event_errors(normalized_payload)
            if errors and strict_reject:
                raise PersistenceSchemaError(
                    "Malformed persisted turn event: " + "; ".join(errors),
                )
            events.append(normalized_payload)

        events.sort(key=lambda item: int(item.get("sequence") or 0))
        if limit and limit > 0:
            return events[-limit:]
        return events

    def list_policy_trace_events(
        self,
        *,
        trace_token: str = "",
        limit: int = 0,
        **legacy_kwargs: Any,
    ) -> list[dict[str, Any]]:
        self._require_active_trace("list_policy_trace_events")
        normalized = str(trace_token or legacy_kwargs.get("trace_id") or "").strip()

        ledger = self._execution_ledger()
        collected: list[dict[str, Any]] = []
        for event in ledger.read():
            event_type = str(event.get("type") or "")
            if event_type == "TURN_EVENT":
                payload = dict(event.get("payload") or {})
                payload_event_type = str(payload.get("event_type") or "").strip().lower()
                if payload_event_type not in {
                    str(POLICY_TRACE_EVENT_TYPE).strip().lower(),
                    "policytraceevent",
                }:
                    continue
                if normalized and str(payload.get("trace_id") or "").strip() != normalized:
                    continue
                payload["event_type"] = POLICY_TRACE_EVENT_TYPE
                collected.append(payload)
                continue
            if event_type != POLICY_TRACE_EVENT_TYPE:
                continue
            if normalized and str(event.get("trace_id") or "").strip() != normalized:
                continue
            collected.append(
                {
                    "event_type": POLICY_TRACE_EVENT_TYPE,
                    "trace_id": str(event.get("trace_id") or ""),
                    "sequence": int(event.get("sequence") or 0),
                    "occurred_at": str(event.get("timestamp") or ""),
                    "stage": "save",
                    "status": "after",
                    "payload": dict(event.get("payload") or {}),
                },
            )

        collected.sort(key=lambda item: int(item.get("sequence") or 0))
        if limit and limit > 0:
            return collected[-int(limit):]
        return collected

    def summarize_policy_trace_events(
        self,
        *,
        trace_token: str = "",
        limit: int = 0,
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        self._require_active_trace("summarize_policy_trace_events")
        resolved_trace = str(trace_token or legacy_kwargs.get("trace_id") or "").strip()
        events = self.list_policy_trace_events(trace_token=resolved_trace, limit=limit)

        action_counts: dict[str, int] = {}
        policies_seen: set[str] = set()
        latest_action = ""
        latest_step_name = ""
        latest_trace_id = ""

        for event in events:
            latest_trace_id = str(event.get("trace_id") or latest_trace_id)
            payload = dict(event.get("payload") or {})
            summary = dict(payload.get("summary") or {})
            policy = str(summary.get("policy") or payload.get("policy") or "").strip()
            if policy:
                policies_seen.add(policy)

            action = str(
                summary.get("decision_action")
                or summary.get("action")
                or payload.get("action")
                or "",
            ).strip()
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
                latest_action = action

            step_name = str(
                summary.get("step_name")
                or payload.get("step_name")
                or "",
            ).strip()
            if step_name:
                latest_step_name = step_name

        return {
            "event_type": POLICY_TRACE_EVENT_TYPE,
            "event_count": len(events),
            "policies": sorted(policies_seen),
            "action_counts": action_counts,
            "latest_action": latest_action,
            "latest_step_name": latest_step_name,
            "latest_trace_id": latest_trace_id,
        }

    @staticmethod
    def _fold_events(
        events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
        replayed_state: dict[str, Any] = {}
        replayed_metadata: dict[str, Any] = {}
        phase = "PLAN"
        seen_lock_hashes: set[str] = set()
        latest_execution_identity: dict[str, Any] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            determinism_lock = event.get("determinism_lock")
            if isinstance(determinism_lock, dict):
                lock_hash = str(determinism_lock.get("lock_hash") or "").strip()
                if lock_hash:
                    seen_lock_hashes.add(lock_hash)
            checkpoint = event.get("checkpoint")
            if isinstance(checkpoint, dict):
                replayed_determinism = checkpoint.get("determinism_lock")
                if isinstance(replayed_determinism, dict):
                    lock_hash = str(replayed_determinism.get("lock_hash") or "").strip()
                    if lock_hash:
                        seen_lock_hashes.add(lock_hash)
                checkpoint_phase = str(checkpoint.get("phase") or "").strip()
                if checkpoint_phase:
                    phase = checkpoint_phase
            if str(event.get("event_type") or "") == "execution_identity":
                identity_data = event.get("identity")
                if isinstance(identity_data, dict):
                    latest_execution_identity = dict(identity_data)
            event_phase = str(event.get("phase") or "").strip()
            if event_phase:
                phase = event_phase

        determinism_summary = {
            "lock_hash": next(iter(seen_lock_hashes)) if len(seen_lock_hashes) == 1 else "",
            "lock_hashes": sorted(seen_lock_hashes),
            "consistent": len(seen_lock_hashes) <= 1,
            "execution_identity": latest_execution_identity,
            "execution_fingerprint": str(
                latest_execution_identity.get("fingerprint") or "",
            ),
        }
        return replayed_state, replayed_metadata, phase, determinism_summary

    def replay_turn_events(
        self,
        trace_token: str,
        *,
        events_snapshot: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._require_active_trace("replay_turn_events")
        normalized = str(trace_token or "").strip()
        events = self.list_turn_events(
            trace_token=normalized,
            events_snapshot=events_snapshot,
        )
        strict_sequence_hash, strict_sequence = KernelReplaySequenceLock.strict_hash(
            trace_token=normalized,
            events=events,
        )
        replayed_state, replayed_metadata, phase, determinism = self._fold_events(
            events,
        )
        replay_record = normalize_replay_record(
            trace_token=normalized,
            events=events,
            strict_sequence_hash=strict_sequence_hash,
            strict_sequence=strict_sequence,
            phase=phase,
            replayed_state=replayed_state,
            replayed_metadata=replayed_metadata,
            determinism=determinism,
        )
        if self._strict_schema_reject_enabled():
            assert_valid_replay_record(replay_record)
        return replay_record

    def validate_replay_determinism(
        self,
        trace_token: str,
        expected_lock_hash: str = "",
        *,
        expected_terminal_state: dict[str, Any] | None = None,
        expected_execution_trace_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_active_trace("validate_replay_determinism")
        replay = self.replay_turn_events(trace_token=trace_token)
        determinism = dict(replay.get("determinism") or {})
        observed_hash = str(determinism.get("lock_hash") or "").strip()
        expected_hash = str(expected_lock_hash or "").strip()
        matches_expected = True
        if expected_hash:
            matches_expected = observed_hash == expected_hash

        terminal_equivalence: dict[str, Any] = {}
        repaired_trace_context: dict[str, Any] = {}
        effective_terminal_state = (
            expected_terminal_state
            if isinstance(expected_terminal_state, dict)
            else dict(
                (replay.get("replayed_metadata") or {}).get("terminal_state") or {},
            )
        )
        effective_trace_context = (
            expected_execution_trace_context
            if isinstance(expected_execution_trace_context, dict)
            else dict(
                (replay.get("replayed_metadata") or {}).get("execution_trace_context") or {},
            )
        )
        if (
            isinstance(effective_terminal_state, dict)
            and isinstance(effective_trace_context, dict)
            and effective_terminal_state
            and effective_trace_context
        ):
            # Phase 3: ledger replay uses canonical events; no repair needed.
            repaired_trace_context = effective_trace_context
            terminal_equivalence = verify_terminal_state_replay_equivalence(
                terminal_state_seed=effective_terminal_state,
                execution_trace_context=repaired_trace_context,
                enforce_dag_equivalence=True,
            )
        return {
            "trace_id": str(trace_token or "").strip(),
            "consistent": bool(determinism.get("consistent", True)),
            "observed_lock_hash": observed_hash,
            "expected_lock_hash": expected_hash,
            "matches_expected": matches_expected,
            "strict_sequence_hash": str(replay.get("strict_sequence_hash") or ""),
            "strict_sequence_count": len(list(replay.get("strict_sequence") or [])),
            "lock_hashes": list(determinism.get("lock_hashes") or []),
            "execution_identity": dict(determinism.get("execution_identity") or {}),
            "execution_fingerprint": str(
                determinism.get("execution_fingerprint") or "",
            ),
            "terminal_state_equivalence": terminal_equivalence,
            "trace_repair_applied": bool(
                isinstance(effective_trace_context, dict) and effective_trace_context != repaired_trace_context,
            ),
            "verification_path": "posthoc",
        }

    def load_latest_graph_checkpoint(self, trace_token: str = "", **legacy_kwargs: Any) -> dict | None:
        with ensure_execution_trace_root(
            operation="load_latest_graph_checkpoint",
            prompt="[conversation-persistence-load-checkpoint]",
            metadata={"source": "ConversationPersistenceManager.load_latest_graph_checkpoint"},
            required=True,
        ):
            self._require_active_trace("load_latest_graph_checkpoint")
            normalized_trace_id = str(trace_token or legacy_kwargs.get("trace_id") or "").strip()
            ledger = self._execution_ledger()
            for event in reversed(ledger.read()):
                if str(event.get("type") or "") != "GRAPH_CHECKPOINT":
                    continue
                payload = dict(event.get("payload") or {})
                event_trace_id = str(payload.get("trace_id") or "").strip()
                if normalized_trace_id and event_trace_id != normalized_trace_id:
                    continue
                checkpoint = dict(payload.get("checkpoint") or {})
                if checkpoint:
                    if self._strict_schema_reject_enabled():
                        return assert_valid_checkpoint_record(checkpoint)
                    checkpoint = normalize_checkpoint_record(checkpoint)
                    errors = checkpoint_record_errors(checkpoint)
                    if errors:
                        raise PersistenceSchemaError(
                            "Malformed persisted checkpoint: " + "; ".join(errors),
                        )
                    return checkpoint
            return None

    def resume_graph_checkpoint(self, trace_token: str = "") -> dict | None:
        with ensure_execution_trace_root(
            operation="resume_graph_checkpoint",
            prompt="[conversation-persistence-resume-checkpoint]",
            metadata={"source": "ConversationPersistenceManager.resume_graph_checkpoint"},
            required=True,
        ):
            self._require_active_trace("resume_graph_checkpoint")
            checkpoint = self.load_latest_graph_checkpoint(trace_token=trace_token)
            if not isinstance(checkpoint, dict):
                return None
            session_state = checkpoint.get("session_state")
            if not isinstance(session_state, dict):
                events = self.list_turn_events(trace_token=trace_token)
                for event in reversed(events):
                    if str(event.get("event_type") or "").strip().lower() != "graph_checkpoint":
                        continue
                    event_state = event.get("session_state")
                    if isinstance(event_state, dict):
                        session_state = event_state
                        break
            if isinstance(session_state, dict):
                self.bot.load_session_state_snapshot(session_state)
            resumed = dict(checkpoint)
            resumed["execution_mode"] = "recovery"
            return normalize_checkpoint_record(resumed)

    def export_execution_handoff(self, trace_token: str, *, worker_id: str = "") -> dict[str, Any]:
        with ensure_execution_trace_root(
            operation="export_execution_handoff",
            prompt="[conversation-persistence-export-handoff]",
            metadata={"source": "ConversationPersistenceManager.export_execution_handoff"},
            required=True,
        ):
            self._require_active_trace("export_execution_handoff")
            checkpoint = self.load_latest_graph_checkpoint(trace_token=trace_token)
            if not isinstance(checkpoint, dict):
                raise RuntimeError("No checkpoint available for handoff")
            canonical = normalize_checkpoint_record(checkpoint)
            continuity = dict(canonical.get("continuity") or {})
            lease = self._current_handoff_lease(str(canonical.get("trace_id") or ""))
            normalized_worker_id = str(worker_id or "").strip()
            if normalized_worker_id and bool(lease.get("active")) and str(lease.get("worker_id") or "") != normalized_worker_id:
                raise RuntimeError("Execution handoff claim is owned by another worker")
            payload = {
                "schema_version": "execution-handoff.v1",
                "trace_id": str(canonical.get("trace_id") or ""),
                "session_id": str(canonical.get("session_id") or "default"),
                "worker_id": normalized_worker_id,
                "checkpoint": canonical,
                "continuity": continuity,
                "lease": lease,
                "handoff_hash": self._stable_json_sha(
                    {
                        "trace_id": str(canonical.get("trace_id") or ""),
                        "checkpoint_hash": str(canonical.get("checkpoint_hash") or ""),
                        "strict_sequence_hash": str(continuity.get("strict_sequence_hash") or ""),
                    },
                ),
            }
            self.persist_turn_event(
                {
                    "event_type": "execution_handoff_exported",
                    "trace_id": str(canonical.get("trace_id") or ""),
                    "stage": "save",
                    "status": "after",
                    "continuity": continuity,
                },
            )
            return payload

    def claim_execution_handoff(
        self,
        trace_token: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> dict[str, Any]:
        with ensure_execution_trace_root(
            operation="claim_execution_handoff",
            prompt="[conversation-persistence-claim-handoff]",
            metadata={"source": "ConversationPersistenceManager.claim_execution_handoff"},
            required=True,
        ):
            self._require_active_trace("claim_execution_handoff")
            normalized_trace_id = str(trace_token or "").strip()
            normalized_worker_id = str(worker_id or "").strip()
            if not normalized_trace_id or not normalized_worker_id:
                raise RuntimeError("trace_id and worker_id are required to claim execution handoff")

            execution_id, state = self._lifecycle_state_for_trace(normalized_trace_id)
            session_id = self._resolve_session_id_for_trace(normalized_trace_id)
            if not execution_id:
                execution_id = self._derived_execution_id(normalized_trace_id)
                self._append_lifecycle_event_for_trace(
                    trace_token=normalized_trace_id,
                    session_id=session_id,
                    event=Submitted(
                        execution_id=execution_id,
                        occurred_at=datetime.now(),
                    ),
                    step_key="conversation_persistence.claim_execution_handoff.submitted",
                )
                state = self._lifecycle_projection().get(execution_id)
            if state is None:
                raise RuntimeError("No lifecycle state found for execution handoff claim")
            if state.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}:
                raise ClaimConflict("Execution handoff cannot be claimed from terminal state")

            now = datetime.now()
            if state.owner and not lease_expired(state, now=now) and state.owner != normalized_worker_id:
                raise ClaimConflict("Execution handoff already claimed by another worker")

            if state.owner and lease_expired(state, now=now):
                self._append_lifecycle_event_for_trace(
                    trace_token=normalized_trace_id,
                    session_id=session_id,
                    event=LeaseExpired(
                        execution_id=execution_id,
                        occurred_at=now,
                        worker_id=str(state.owner),
                    ),
                    step_key="conversation_persistence.claim_execution_handoff.expire",
                )
                self._append_lifecycle_event_for_trace(
                    trace_token=normalized_trace_id,
                    session_id=session_id,
                    event=Redelivered(
                        execution_id=execution_id,
                        occurred_at=now,
                        previous_worker_id=str(state.owner),
                        new_worker_id=normalized_worker_id,
                    ),
                    step_key="conversation_persistence.claim_execution_handoff.redeliver",
                )

            expiry = now + timedelta(seconds=max(int(lease_seconds or 0), 1))
            self._append_lifecycle_event_for_trace(
                trace_token=normalized_trace_id,
                session_id=session_id,
                event=Claimed(
                    execution_id=execution_id,
                    occurred_at=now,
                    worker_id=normalized_worker_id,
                    lease_expiry=expiry,
                ),
                step_key="conversation_persistence.claim_execution_handoff.claim",
            )
            lease_payload = self._current_handoff_lease(normalized_trace_id)
            self.persist_turn_event(
                {
                    "event_type": "execution_handoff_claimed",
                    "trace_id": normalized_trace_id,
                    "stage": "save",
                    "status": "after",
                    "worker_id": normalized_worker_id,
                    "lease_seconds": int(lease_payload.get("lease_seconds") or 0),
                    "lease_expires_at": str(lease_payload.get("lease_expires_at") or ""),
                },
            )
            return lease_payload

    def renew_execution_handoff_lease(
        self,
        trace_token: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> dict[str, Any]:
        with ensure_execution_trace_root(
            operation="renew_execution_handoff_lease",
            prompt="[conversation-persistence-renew-handoff-lease]",
            metadata={"source": "ConversationPersistenceManager.renew_execution_handoff_lease"},
            required=True,
        ):
            self._require_active_trace("renew_execution_handoff_lease")
            normalized_trace_id = str(trace_token or "").strip()
            normalized_worker_id = str(worker_id or "").strip()
            execution_id, state = self._lifecycle_state_for_trace(normalized_trace_id)
            if not execution_id or state is None:
                raise RuntimeError("No lifecycle state found for execution handoff lease renew")
            now = datetime.now()
            if state.owner != normalized_worker_id:
                raise NotOwner("Execution handoff lease is owned by another worker")
            if state.lease_expiry is not None and lease_expired(state, now=now):
                raise NotOwner("Execution handoff lease is expired")

            renewed_expiry = now + timedelta(seconds=max(int(lease_seconds or 0), 1))
            self._append_lifecycle_event_for_trace(
                trace_token=normalized_trace_id,
                session_id=self._resolve_session_id_for_trace(normalized_trace_id),
                event=LeaseRenewed(
                    execution_id=execution_id,
                    occurred_at=now,
                    worker_id=normalized_worker_id,
                    lease_expiry=renewed_expiry,
                ),
                step_key="conversation_persistence.renew_execution_handoff_lease",
            )

            renewed = self._current_handoff_lease(normalized_trace_id)
            renewed["lease_seconds"] = int(max(int(lease_seconds or 0), 1))
            self.persist_turn_event(
                {
                    "event_type": "execution_handoff_lease_renewed",
                    "trace_id": normalized_trace_id,
                    "stage": "save",
                    "status": "after",
                    "worker_id": normalized_worker_id,
                    "lease_seconds": int(renewed.get("lease_seconds") or 0),
                    "lease_expires_at": str(renewed.get("lease_expires_at") or ""),
                },
            )
            return renewed

    def release_execution_handoff_claim(self, trace_token: str, *, worker_id: str) -> dict[str, Any]:
        with ensure_execution_trace_root(
            operation="release_execution_handoff_claim",
            prompt="[conversation-persistence-release-handoff-claim]",
            metadata={"source": "ConversationPersistenceManager.release_execution_handoff_claim"},
            required=True,
        ):
            self._require_active_trace("release_execution_handoff_claim")
            normalized_trace_id = str(trace_token or "").strip()
            normalized_worker_id = str(worker_id or "").strip()
            execution_id, state = self._lifecycle_state_for_trace(normalized_trace_id)
            if not execution_id or state is None:
                raise RuntimeError("No lifecycle state found for execution handoff claim release")
            if state.owner != normalized_worker_id:
                raise NotOwner("Execution handoff claim is owned by another worker")

            self._append_lifecycle_event_for_trace(
                trace_token=normalized_trace_id,
                session_id=self._resolve_session_id_for_trace(normalized_trace_id),
                event=Released(
                    execution_id=execution_id,
                    occurred_at=datetime.now(),
                    worker_id=normalized_worker_id,
                ),
                step_key="conversation_persistence.release_execution_handoff_claim",
            )

            released = {
                "status": "released",
                "worker_id": normalized_worker_id,
                "lease_seconds": 0,
                "lease_expires_at": "",
                "active": False,
            }
            self.persist_turn_event(
                {
                    "event_type": "execution_handoff_claim_released",
                    "trace_id": normalized_trace_id,
                    "stage": "save",
                    "status": "after",
                    "worker_id": normalized_worker_id,
                },
            )
            return released

    def import_execution_handoff(self, handoff: dict[str, Any]) -> dict[str, Any]:
        with ensure_execution_trace_root(
            operation="import_execution_handoff",
            prompt="[conversation-persistence-import-handoff]",
            metadata={"source": "ConversationPersistenceManager.import_execution_handoff"},
            required=True,
        ):
            self._require_active_trace("import_execution_handoff")
            payload = dict(handoff or {})
            checkpoint = normalize_checkpoint_record(dict(payload.get("checkpoint") or {}))
            continuity = dict(payload.get("continuity") or {})
            lease = dict(payload.get("lease") or {})
            claimed_worker_id = str(lease.get("worker_id") or payload.get("worker_id") or "").strip()
            expected_hash = str(payload.get("handoff_hash") or "").strip()
            actual_hash = self._stable_json_sha(
                {
                    "trace_id": str(checkpoint.get("trace_id") or ""),
                    "checkpoint_hash": str(checkpoint.get("checkpoint_hash") or ""),
                    "strict_sequence_hash": str(continuity.get("strict_sequence_hash") or ""),
                },
            )
            if expected_hash and expected_hash != actual_hash:
                raise RuntimeError("Execution handoff integrity mismatch")

            current_lease = self._current_handoff_lease(str(checkpoint.get("trace_id") or ""))
            if claimed_worker_id:
                if not bool(current_lease.get("active")):
                    raise RuntimeError("Execution handoff import requires an active worker claim")
                if str(current_lease.get("worker_id") or "") != claimed_worker_id:
                    raise RuntimeError("Execution handoff claim is owned by another worker")

            imported = dict(checkpoint)
            imported["execution_mode"] = "recovery"
            self.persist_graph_checkpoint(imported, _skip_turn_event=True)
            self.persist_turn_event(
                {
                    "event_type": "execution_handoff_imported",
                    "trace_id": str(imported.get("trace_id") or ""),
                    "stage": "save",
                    "status": "after",
                    "worker_id": claimed_worker_id,
                    "continuity": continuity,
                },
            )
            return normalize_checkpoint_record(imported)


__all__ = ["ConversationPersistenceManager"]

