from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict, dataclass
from collections.abc import Iterable
from typing import Any

from dadbot.core.capability_audit_runner import (
    CAPABILITY_AUDIT_EVENT_TYPE,
    build_capability_audit_event_payload,
    build_runtime_capability_audit_report,
)
from dadbot.core.execution_context import (
    RuntimeTraceViolation,
    ensure_execution_trace_root,
)
from dadbot.core.kernel_locks import KernelEventTotalityLock
from dadbot.core.kernel_mutation_gate import apply_event, emit_event
from dadbot.core.graph import (
    FatalTurnError,
    LedgerMutationOp,
    MemoryMutationOp,
    MutationIntent,
    MutationKind,
)
from dadbot.core.merkle_anchor import append_leaf_and_anchor
from dadbot.core.persistence import AbstractCheckpointer
from dadbot.core.post_commit_events import PostCommitEvent
from dadbot.core.runtime_errors import (
    ExecutionStageError,
    InvariantViolation,
    NON_FATAL_RUNTIME_EXCEPTIONS,
    PersistenceFailure,
)
from dadbot.managers.conversation_persistence import ConversationPersistenceManager

logger = logging.getLogger(__name__)

POLICY_TRACE_EVENT_TYPE = "PolicyTraceEvent"


class StateDivergenceError(RuntimeError):
    """Raised when projected state diverges from ledger-backed event state."""

    def __init__(self, message: str, *, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = dict(report or {})


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


class PersistenceService:
    """Service wrapper for durable turn/session persistence.

    The ``finalize_turn`` method is the atomic commit point for the SaveNode.
    It delegates to ``TurnService.finalize_user_turn``, which appends
    conversation history, schedules background maintenance, runs internal
    reflection, takes a health snapshot, and persists the session — all in a
    single call so no partial-state is ever written to disk.
    """

    def __init__(
        self,
        persistence_manager: ConversationPersistenceManager,
        turn_service: Any = None,
    ):
        self.persistence_manager = persistence_manager
        # Wired by ServiceRegistry.boot() after wire_runtime_managers has run.
        self.turn_service = turn_service
        self.checkpointer: AbstractCheckpointer | None = None
        self.strict_mode: bool = False
        self._merkle_session_leaves: dict[str, list[str]] = {}

    @staticmethod
    def _stable_hash(payload: Any) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode(
                "utf-8",
            ),
        ).hexdigest()

    @staticmethod
    def _json_safe(payload: Any) -> Any:
        return json.loads(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str),
        )

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

        # Drift = strong prior topic context with no lexical overlap on this turn.
        topic_drift_detected = bool(recent_topics) and bool(query_tokens) and not bool(overlap)

        trust_level = float(getattr(runtime, "trust_level", lambda: 50)() if callable(getattr(runtime, "trust_level", None)) else 50)
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

        # Pressure rises with sustained drift and expensive turns.
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
            if overlap > best_overlap:
                best_overlap = overlap
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

        return _walk(self._json_safe(payload))

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
        diffs.append(
            {
                "path": path,
                "projected": projected,
                "event_sourced": event_sourced,
            },
        )

    def _walk_authority_state_diff(
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
            self._walk_authority_state_dict(
                path,
                projected,
                event_sourced,
                diffs=diffs,
                max_items=max_items,
                missing=missing,
            )
            return
        if isinstance(projected, list) and isinstance(event_sourced, list):
            self._walk_authority_state_list(
                path,
                projected,
                event_sourced,
                diffs=diffs,
                max_items=max_items,
                missing=missing,
            )
            return
        if projected != event_sourced:
            self._record_authority_state_diff(
                diffs,
                max_items=max_items,
                path=path or "$",
                projected=projected,
                event_sourced=event_sourced,
            )

    def _walk_authority_state_dict(
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
                self._record_authority_state_diff(
                    diffs,
                    max_items=max_items,
                    path=child_path,
                    projected=missing,
                    event_sourced=event_sourced.get(key),
                )
                continue
            if not has_event_sourced:
                self._record_authority_state_diff(
                    diffs,
                    max_items=max_items,
                    path=child_path,
                    projected=projected.get(key),
                    event_sourced=missing,
                )
                continue
            self._walk_authority_state_diff(
                child_path,
                projected.get(key),
                event_sourced.get(key),
                diffs=diffs,
                max_items=max_items,
                missing=missing,
            )

    def _walk_authority_state_list(
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
                self._record_authority_state_diff(
                    diffs,
                    max_items=max_items,
                    path=child_path,
                    projected=missing,
                    event_sourced=event_sourced[idx],
                )
                continue
            if not has_event_sourced:
                self._record_authority_state_diff(
                    diffs,
                    max_items=max_items,
                    path=child_path,
                    projected=projected[idx],
                    event_sourced=missing,
                )
                continue
            self._walk_authority_state_diff(
                child_path,
                projected[idx],
                event_sourced[idx],
                diffs=diffs,
                max_items=max_items,
                missing=missing,
            )

    def _load_event_sourced_checkpoint(self, trace_id: str) -> dict[str, Any]:
        trace_key = str(trace_id or "").strip()
        with ensure_execution_trace_root(
            operation="persistence_load_event_sourced_checkpoint",
            prompt="[persistence-load-event-sourced-checkpoint]",
            metadata={"source": "PersistenceService._load_event_sourced_checkpoint"},
            required=True,
        ):
            loader = getattr(self.persistence_manager, "load_latest_graph_checkpoint", None)
            if callable(loader):
                loaded = loader(trace_id=trace_key)
                if isinstance(loaded, dict):
                    return dict(loaded)

            ledger_resolver = getattr(self.persistence_manager, "_execution_ledger", None)
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
        # DEPRECATED compat bypass — NO NEW CALLERS. This path allows legacy direct-compat
        # turns (marked via metadata["legacy_direct_compat"]=True) to skip memory authority
        # divergence checks in test mode only. Remove when _finalize_direct_compat_turn is
        # deleted. Expiry: 2026-Q3.
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
        projected_session_state = self._normalize_authority_snapshot(
            runtime.snapshot_session_state(),
        )

        event_checkpoint_copy = dict(event_checkpoint)
        event_session_state = self._normalize_authority_snapshot(
            event_checkpoint_copy.pop("session_state", {}),
        )

        projected = {
            "checkpoint": self._normalize_authority_snapshot(projected_checkpoint),
            "session_state": projected_session_state,
        }
        event_sourced = {
            "checkpoint": self._normalize_authority_snapshot(event_checkpoint_copy),
            "session_state": event_session_state,
        }

        projected_hash = self._stable_hash(projected)
        event_hash = self._stable_hash(event_sourced)
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

        diffs = self._diff_authority_state(projected, event_sourced)
        report = {
            "trace_id": trace_id,
            "consistent": False,
            "projected_hash": projected_hash,
            "event_sourced_hash": event_hash,
            "difference_count": len(diffs),
            "differences": diffs,
            "repair_hint": "Replay trace from ledger, regenerate projection from event state, then re-run SaveNode commit.",
        }
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["memory_authority_check"] = dict(report)

        logger.error(
            "State divergence detected at SaveNode boundary (trace_id=%s, projected_hash=%s, event_hash=%s, differences=%d)",
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
        runtime = getattr(getattr(self.turn_service, "bot", None), "config", None)
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
            "state_hash": self._stable_hash(getattr(turn_context, "state", {}) or {}),
            "metadata_hash": self._stable_hash(
                getattr(turn_context, "metadata", {}) or {},
            ),
        }
        leaves = self._merkle_session_leaves.setdefault(session_id, [])
        anchor = append_leaf_and_anchor(leaves, payload)
        if isinstance(metadata, dict):
            metadata_snapshot = dict(metadata)
            metadata_snapshot["merkle_anchor"] = dict(anchor)
            turn_context.metadata = metadata_snapshot
        if isinstance(getattr(turn_context, "state", None), dict):
            state_snapshot = dict(getattr(turn_context, "state", {}) or {})
            state_snapshot["merkle_anchor"] = dict(anchor)
            turn_context.state = state_snapshot
        self.save_turn_event(
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

    def set_checkpointer(self, checkpointer: AbstractCheckpointer | None) -> None:
        self.checkpointer = checkpointer

    @staticmethod
    def _call_nonfatal(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
        if not callable(callable_obj):
            return None
        try:
            return callable_obj(*args, **kwargs)
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.warning(
                "PersistenceService post-finalize hook failed (non-fatal): %s",
                exc,
            )
            return None

    def _apply_memory_decay(self, memory_manager: Any, ctx: Any) -> Any:
        """Persistence boundary guard.

        Memory decay and lifecycle evolution are post-commit responsibilities
        and must run in post-commit worker / maintenance services only.
        """
        raise ExecutionStageError(
            "PersistenceService._apply_memory_decay is not allowed",
            context={"required_path": "post-commit worker or maintenance service"},
        )

    def _apply_pending_save_boundary_mutations(
        self,
        runtime: Any,
        turn_context: Any,
    ) -> None:
        state = getattr(turn_context, "state", None)
        if not isinstance(state, dict):
            return

        pending_moods = list(state.get("_pending_mood_updates") or [])
        pending_relationship = list(state.get("_pending_relationship_updates") or [])
        if not pending_moods and not pending_relationship:
            # Backward compatibility with older deferred key.
            legacy = list(state.get("_deferred_turn_state_updates") or [])
            if legacy:
                pending_moods = [{"mood": str(item.get("mood") or "neutral")} for item in legacy]
                pending_relationship = [
                    {
                        "op": "update",
                        "user_input": str(item.get("user_input") or ""),
                        "mood": str(item.get("mood") or "neutral"),
                    }
                    for item in legacy
                    if str(item.get("user_input") or "").strip()
                ]

        memory = getattr(runtime, "memory", None)
        if memory is None:
            raise PersistenceFailure("SaveNode strict mode requires runtime.memory")

        for item in pending_moods:
            mood = str(item.get("mood") or "neutral")
            memory.save_mood_state(mood)

        clear_event = emit_event(
            "MUTATION_EVENT",
            {
                "op": "dict_update",
                "updates": {
                    "_pending_mood_updates": [],
                    "_pending_relationship_updates": [],
                    "_deferred_turn_state_updates": [],
                },
            },
            source="PersistenceService._apply_pending_save_boundary_mutations",
        )
        turn_context.state = apply_event(
            clear_event,
            state,
            lambda current, evt: {**current, **dict(evt.payload.get("updates") or {})},
        )

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
            trace_id = str(
                getattr(turn_context, "trace_id", "") or "unknown",
            )
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
                        ((event_payload.get("trace") or {}).get("final_action") or {}).get("action")
                        or "",
                    ),
                    "decision_step": str(
                        ((event_payload.get("trace") or {}).get("final_action") or {}).get("step_name")
                        or "",
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
                self.save_turn_event(turn_event_payload)

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
        # Non-authoritative observability write: never block turn correctness.
        try:
            stage_order = [
                str(getattr(trace, "stage", "") or "")
                for trace in list(
                    getattr(turn_context, "stage_traces", []) or [],
                )
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
                turn_context.metadata["capability_audit_report"] = dict(
                    report_payload,
                )

            event_payload = build_capability_audit_event_payload(
                report,
                scenario=str(payload.get("scenario") or "runtime_turn"),
            )
            if hasattr(turn_context, "event_sequence"):
                turn_context.event_sequence += 1
                sequence = int(turn_context.event_sequence)
            else:
                sequence = 0

            trace_id = str(
                getattr(turn_context, "trace_id", "") or "unknown",
            )
            phase_value = str(
                getattr(getattr(turn_context, "phase", None), "value", "") or "",
            )
            occurred_at = ""
            temporal = getattr(turn_context, "temporal", None)
            if temporal is not None:
                occurred_at = str(getattr(temporal, "wall_time", "") or "")

            self.save_turn_event(
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

    def _dispatch_graph_mutation_intent(self, runtime: Any, turn_context: Any, *, payload: dict[str, Any], source: str) -> None:
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
            LedgerMutationOp.APPEND_HISTORY.value: lambda: self._ledger_op_append_history(runtime, payload),
            LedgerMutationOp.RECORD_TURN_STATE.value: lambda: self._ledger_op_record_turn_state(runtime, payload),
            LedgerMutationOp.SYNC_THREAD_SNAPSHOT.value: lambda: self._ledger_op_sync_thread_snapshot(runtime, payload),
            LedgerMutationOp.CLEAR_TURN_CONTEXT.value: lambda: self._ledger_op_clear_turn_context(runtime, payload),
            LedgerMutationOp.SCHEDULE_MAINTENANCE.value: lambda: self._ledger_op_schedule_maintenance(runtime, payload, service),
            LedgerMutationOp.HEALTH_SNAPSHOT.value: lambda: self._ledger_op_health_snapshot(runtime, service),
            LedgerMutationOp.POLICY_TRACE_EVENT.value: lambda: self._ledger_op_policy_trace_event(runtime, turn_context, payload),
            LedgerMutationOp.CAPABILITY_AUDIT_EVENT.value: lambda: self._ledger_op_capability_audit_event(runtime, turn_context, payload),
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

        service = self.turn_service
        dispatch = lambda intent: self._dispatch_mutation_intent(
            runtime,
            turn_context,
            service,
            intent,
        )

        try:
            mutation_queue.drain(
                dispatch,
                hard_fail_on_error=True,
                transactional=True,
            )
        except TypeError as exc:
            # Backward compatibility for tests/wrappers that monkeypatch
            # MutationQueue.drain with a legacy two-parameter signature.
            if "unexpected keyword argument 'transactional'" not in str(exc):
                raise
            mutation_queue.drain(
                dispatch,
                hard_fail_on_error=True,
            )
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

    def _persist_hierarchical_memory_commit(
        self,
        turn_context: Any,
        *,
        commit_id: str,
    ) -> None:
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

        self.save_turn_event(
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

    def _publish_post_commit_ready(self, runtime: Any, turn_context: Any) -> None:
        """Publish the post-commit readiness event and return immediately."""
        event_bus = getattr(runtime, "_runtime_event_bus", None)
        publish = getattr(event_bus, "publish", None)
        if not callable(publish):
            logger.warning(
                "Post-commit worker unavailable: runtime event bus missing publish(); skipping post-commit event"
            )
            return

        metadata = getattr(turn_context, "metadata", None)
        control_plane = dict(metadata.get("control_plane") or {}) if isinstance(metadata, dict) else {}
        publish(
            PostCommitEvent(
                session_id=str(control_plane.get("session_id") or "default"),
                trace_id=str(getattr(turn_context, "trace_id", "") or ""),
                tenant_id=str(
                    getattr(getattr(runtime, "config", None), "tenant_id", "default")
                    or "default"
                ),
                payload={"turn_context": turn_context},
            )
        )

    def _commit_post_finalize_side_effects(self, turn_context: Any) -> None:
        """Run strict SaveNode mutation sequence before final ledger commit."""
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if runtime is None:
            raise PersistenceFailure(
                "SaveNode strict mode requires an attached turn_service runtime",
            )

        # --- Drain MutationQueue FIRST at the canonical SaveNode commit boundary ---
        # Every mutation queued outside this boundary (deferred from earlier stages or
        # direct-path callers) must execute here. Any failure is a hard fail — nothing
        # is silently dropped.
        self._drain_mutation_queue(runtime, turn_context)
        # --------------------------------------------------------------------------

        # Apply any pending non-SaveNode mutation intents at the canonical commit boundary.
        self._apply_pending_save_boundary_mutations(runtime, turn_context)
        self._flush_background_memory_store_patch_queue(runtime)
        flush_deferred = getattr(
            self.persistence_manager,
            "flush_deferred_save_boundary_mutations",
            None,
        )
        if callable(flush_deferred):
            self._call_nonfatal(flush_deferred, turn_context)

        relationship_manager = getattr(runtime, "relationship_manager", None)
        materialize_projection = getattr(
            relationship_manager,
            "materialize_projection",
            None,
        )
        if not callable(materialize_projection):
            raise PersistenceFailure(
                "SaveNode strict mode requires relationship_projector.materialize_projection",
            )
        materialize_projection(turn_context=turn_context)

        memory_manager = getattr(runtime, "memory_manager", None)
        graph_manager = getattr(memory_manager, "graph_manager", None) if memory_manager is not None else None
        sync_graph_store = getattr(graph_manager, "sync_graph_store", None)
        if not callable(sync_graph_store):
            raise PersistenceFailure(
                "SaveNode strict mode requires memory_graph_manager.sync_graph_store",
            )
        graph_sync_started = time.perf_counter()
        sync_graph_store(turn_context=turn_context)
        graph_sync_ms = round((time.perf_counter() - graph_sync_started) * 1000, 3)
        if isinstance(getattr(turn_context, "state", None), dict):
            turn_context.state["_timing_graph_sync_ms"] = graph_sync_ms

        # Post-commit publish is intentionally performed by finalize_turn after
        # checkpoint/checkpointer writes complete to avoid worker-thread state
        # mutations racing with in-flight finalize snapshotting.

    @staticmethod
    def _capture_transaction_snapshot(
        runtime: Any,
        turn_context: Any,
    ) -> dict[str, Any]:
        graph_manager = getattr(
            getattr(runtime, "memory_manager", None),
            "graph_manager",
            None,
        )
        graph_snapshot = {"nodes": [], "edges": [], "updated_at": None}
        background_patch_queue = getattr(
            runtime,
            "_background_memory_store_patch_queue",
            None,
        )
        if graph_manager is not None:
            snapshot_builder = getattr(graph_manager, "graph_snapshot", None)
            if callable(snapshot_builder):
                graph_snapshot = snapshot_builder() or graph_snapshot

        return {
            "memory_store": deepcopy(dict(getattr(runtime, "MEMORY_STORE", {}) or {})),
            "graph_snapshot": deepcopy(graph_snapshot),
            "session_state": deepcopy(runtime.snapshot_session_state()),
            "last_turn_pipeline": deepcopy(
                dict(getattr(runtime, "_last_turn_pipeline", {}) or {}),
            ),
            "background_patch_queue": deepcopy(list(background_patch_queue))
            if isinstance(background_patch_queue, list)
            else None,
            "turn_state": deepcopy(dict(getattr(turn_context, "state", {}) or {})),
            "metadata": deepcopy(dict(getattr(turn_context, "metadata", {}) or {})),
        }

    @staticmethod
    def _restore_transaction_snapshot(
        runtime: Any,
        turn_context: Any,
        snapshot: dict[str, Any],
    ) -> None:
        session_state = dict(snapshot.get("session_state", {}) or {})
        if session_state:
            runtime.load_session_state_snapshot(deepcopy(session_state))
        else:
            runtime.MEMORY_STORE = deepcopy(
                dict(snapshot.get("memory_store", {}) or {}),
            )
        runtime._last_turn_pipeline = deepcopy(
            dict(snapshot.get("last_turn_pipeline", {}) or {}),
        )
        background_patch_queue = snapshot.get("background_patch_queue")
        if isinstance(background_patch_queue, list):
            runtime._background_memory_store_patch_queue = deepcopy(
                background_patch_queue,
            )

        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state.clear()
            state.update(deepcopy(dict(snapshot.get("turn_state", {}) or {})))

        metadata = getattr(turn_context, "metadata", None)
        if isinstance(metadata, dict):
            metadata.clear()
            metadata.update(deepcopy(dict(snapshot.get("metadata", {}) or {})))

        graph_snapshot = dict(snapshot.get("graph_snapshot", {}) or {})
        graph_manager = getattr(
            getattr(runtime, "memory_manager", None),
            "graph_manager",
            None,
        )
        backend = getattr(graph_manager, "_graph_store_backend", None) if graph_manager is not None else None
        replace_graph = getattr(backend, "replace_graph", None)
        if callable(replace_graph):
            replace_graph(
                deepcopy(list(graph_snapshot.get("nodes", []) or [])),
                deepcopy(list(graph_snapshot.get("edges", []) or [])),
            )

    def begin_transaction(self, turn_context: Any) -> None:
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise InvariantViolation("TemporalNode required — execution invalid")
        state = getattr(turn_context, "state", None)
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if isinstance(state, dict):
            if runtime is not None:
                state["_save_transaction_snapshot"] = self._capture_transaction_snapshot(runtime, turn_context)
            trace_id = str(getattr(turn_context, "trace_id", "") or "")
            if trace_id:
                commit_seed = f"{trace_id}:{str(getattr(turn_context, 'user_input', '') or '')}"
                commit_id = hashlib.sha256(commit_seed.encode("utf-8")).hexdigest()[:32]
            else:
                commit_id = uuid.uuid4().hex
            state["_save_transaction"] = {
                "commit_id": commit_id,
                "trace_id": trace_id,
                "started_at": str(getattr(temporal, "wall_time", "") or ""),
                "status": "active",
            }
            state["_save_transaction_active"] = True

    def apply_mutations(self, turn_context: Any) -> None:
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise InvariantViolation("TemporalNode required — execution invalid")
        service = self.turn_service
        runtime = None if service is None else getattr(service, "bot", None)
        if runtime is None:
            raise PersistenceFailure("SaveNode strict mode requires turn_service.bot")
        # SaveNode finalize_turn performs the canonical commit sequence in strict mode.
        # This hook exists to preserve transaction staging semantics in core.nodes.SaveNode.
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            state["_save_mutations_applied"] = False

    def commit_transaction(self, turn_context: Any) -> None:
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            transaction = dict(state.get("_save_transaction", {}) or {})
            if transaction:
                temporal = getattr(turn_context, "temporal", None)
                transaction["status"] = "committed"
                transaction["committed_at"] = str(
                    getattr(temporal, "wall_time", "") or "",
                )
                state["_save_transaction"] = transaction
                commit_id = str(transaction.get("commit_id") or "")
                state["last_commit_id"] = commit_id
                state["last_transaction_status"] = "committed"
                metadata = getattr(turn_context, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["last_commit_id"] = commit_id
                    metadata["last_transaction_status"] = "committed"
                self._persist_hierarchical_memory_commit(
                    turn_context,
                    commit_id=commit_id,
                )
                self._record_merkle_anchor(turn_context, commit_id=commit_id)
            state["_save_transaction_active"] = False
            state["_save_mutations_applied"] = False

    def rollback_transaction(self, turn_context: Any) -> None:
        state = getattr(turn_context, "state", None)
        if isinstance(state, dict):
            snapshot = dict(state.get("_save_transaction_snapshot", {}) or {})
            service = self.turn_service
            runtime = None if service is None else getattr(service, "bot", None)
            if snapshot and runtime is not None:
                self._restore_transaction_snapshot(runtime, turn_context, snapshot)
            transaction = dict(state.get("_save_transaction", {}) or {})
            if transaction:
                temporal = getattr(turn_context, "temporal", None)
                transaction["status"] = "rolled_back"
                transaction["rolled_back_at"] = str(
                    getattr(temporal, "wall_time", "") or "",
                )
                state["_save_transaction"] = transaction
            state["_save_transaction_active"] = False
            state["_save_mutations_applied"] = False
            state.pop("_save_transaction_snapshot", None)
        mutation_queue = getattr(turn_context, "mutation_queue", None)
        if mutation_queue is not None:
            # Roll back queue runtime bookkeeping so failed-turn contracts are deterministic.
            reset_fn = getattr(mutation_queue, "reset_for_rollback", None)
            if callable(reset_fn):
                reset_fn()

    def final_ledger_commit(
        self,
        turn_text: str,
        mood: str,
        reply: str,
        norm_attachments: Any,
        turn_context: Any,
    ) -> tuple:
        if self.turn_service is None:
            raise PersistenceFailure(
                "SaveNode strict mode requires graph turn_service wiring in Phase 4",
            )
        return self.turn_service.finalize_user_turn(
            turn_text,
            mood,
            reply,
            norm_attachments,
            turn_context=turn_context,
        )

    def finalize_turn(self, turn_context: Any, result: Any) -> tuple:
        """Atomically commit history, maintenance, reflection, health snapshot, and persistence."""
        with ensure_execution_trace_root(
            operation="persistence_finalize_turn",
            prompt="[persistence-finalize-turn]",
            metadata={"source": "PersistenceService.finalize_turn"},
            required=True,
        ):
            return self._finalize_turn_impl(turn_context, result)

    @staticmethod
    def _finalize_fast_path_if_already_finalized(turn_context: Any, result: Any) -> tuple | None:
        # Session exit was already handled inside prepare_user_turn_async; avoid double-commit.
        if not turn_context.state.get("already_finalized"):
            return None
        if isinstance(result, tuple) and len(result) >= 2:
            return result
        return (
            str(result or ""),
            bool(turn_context.state.get("should_end", False)),
        )

    @staticmethod
    def _finalize_inputs(turn_context: Any, result: Any) -> tuple[str, str, Any, str]:
        turn_text = turn_context.state.get("turn_text") or turn_context.user_input
        mood = turn_context.state.get("mood") or "neutral"
        norm_attachments = turn_context.state.get("norm_attachments") or turn_context.attachments
        reply = result[0] if isinstance(result, tuple) else str(result or "")
        return str(turn_text or ""), str(mood or "neutral"), norm_attachments, reply

    def _finalize_validate_mutation_set(self, turn_context: Any) -> tuple[Any, Any]:
        service = self.turn_service
        if service is None:
            raise PersistenceFailure(
                "Strict mode requires graph turn_service wiring in Phase 4",
            )

        runtime = getattr(service, "bot", None)
        if runtime is None:
            raise PersistenceFailure("SaveNode strict mode requires turn_service.bot")
        if getattr(turn_context, "temporal", None) is None:
            raise InvariantViolation("TemporalNode required — execution invalid")
        return service, runtime

    def _finalize_apply_final_graph_commit(
        self,
        runtime: Any,
        turn_context: Any,
        *,
        turn_text: str,
    ) -> None:
        self.begin_transaction(turn_context)
        self.apply_mutations(turn_context)
        # Persist behavioral ledger state before SaveNode checkpoint capture.
        self._inject_behavioral_ledger_state(runtime, turn_context, turn_text)
        self._record_relational_ledger(runtime, turn_context, turn_text)

    def _finalize_apply_ledger_commit(
        self,
        runtime: Any,
        turn_context: Any,
        *,
        turn_text: str,
        mood: str,
        reply: str,
        norm_attachments: Any,
    ) -> tuple:
        finalized = self.final_ledger_commit(
            turn_text,
            mood,
            reply,
            norm_attachments,
            turn_context,
        )
        self._commit_post_finalize_side_effects(turn_context)
        return finalized

    def _finalize_trace_envelope(
        self,
        runtime: Any,
        turn_context: Any,
    ) -> None:
        # Atomic checkpoint capture inside finalize boundary.
        checkpoint = None
        checkpoint_snapshot = getattr(turn_context, "checkpoint_snapshot", None)
        if callable(checkpoint_snapshot):
            checkpoint = checkpoint_snapshot(
                stage="save",
                status="atomic_finalize",
                error=None,
            )
            turn_context.state["_atomic_checkpoint_saved"] = True
            save_graph_checkpoint = getattr(self, "save_graph_checkpoint", None)
            if callable(save_graph_checkpoint):
                save_graph_checkpoint(checkpoint, _skip_turn_event=True)

        if self.checkpointer is not None and checkpoint is not None:
            _md = dict(getattr(turn_context, "metadata", {}) or {})
            control_plane = dict(_md.get("control_plane") or {})
            session_id = str(
                control_plane.get("session_id") or _md.get("session_id") or "default",
            )
            trace_id = str(getattr(turn_context, "trace_id", "") or "")
            manifest = dict(
                getattr(turn_context, "metadata", {}).get("determinism_manifest") or {},
            )
            try:
                self.checkpointer.save_checkpoint(
                    session_id=session_id,
                    trace_id=trace_id,
                    checkpoint=dict(checkpoint) if isinstance(checkpoint, dict) else {},
                    manifest=manifest,
                )
            except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                logger.error("PersistenceService.checkpointer save failed: %s", exc)
                if bool(self.strict_mode):
                    raise PersistenceFailure(
                        "PersistenceService.checkpointer save failed",
                        context={"error": str(exc)},
                    ) from exc

        self._enforce_memory_authority(
            runtime,
            turn_context,
            checkpoint=dict(checkpoint) if isinstance(checkpoint, dict) else None,
        )

    def _finalize_emit_completion_event(
        self,
        runtime: Any,
        turn_context: Any,
        service: Any,
    ) -> None:
        # Completion side effects are centralized after all commit validations.
        memory_ops_started = time.perf_counter()
        self._publish_post_commit_ready(runtime, turn_context)
        memory_ops_ms = round((time.perf_counter() - memory_ops_started) * 1000, 3)
        if isinstance(getattr(turn_context, "state", None), dict):
            turn_context.state["_timing_memory_ops_ms"] = memory_ops_ms

        complete_pipeline = getattr(service, "_complete_turn_pipeline", None)
        if callable(complete_pipeline):
            complete_pipeline(should_end=False)
        self.commit_transaction(turn_context)

    def _finalize_turn_impl(self, turn_context: Any, result: Any) -> tuple:
        """Internal finalize implementation executed under an active trace context."""
        fast_path = self._finalize_fast_path_if_already_finalized(turn_context, result)
        if fast_path is not None:
            return fast_path

        turn_text, mood, norm_attachments, reply = self._finalize_inputs(turn_context, result)
        service, runtime = self._finalize_validate_mutation_set(turn_context)

        previous_temporal = getattr(runtime, "_current_turn_time_base", None)
        previous_commit_active = bool(getattr(runtime, "_graph_commit_active", False))
        finalize_started = time.perf_counter()
        try:
            if previous_commit_active:
                logger.warning(
                    "PersistenceService.finalize_turn detected stale _graph_commit_active=True; "
                    "forcing fresh SaveNode boundary",
                )
            runtime._current_turn_time_base = getattr(turn_context, "temporal", None)
            runtime._graph_commit_active = True

            # 1) Validate final mutation set (preconditions + transaction gate)
            # 2) Apply final graph commit
            self._finalize_apply_final_graph_commit(
                runtime,
                turn_context,
                turn_text=turn_text,
            )

            # 3) Apply ledger commit
            finalized = self._finalize_apply_ledger_commit(
                runtime,
                turn_context,
                turn_text=turn_text,
                mood=mood,
                reply=reply,
                norm_attachments=norm_attachments,
            )

            # 4) Finalize trace envelope
            self._finalize_trace_envelope(runtime, turn_context)

            # 5) Emit completion event (single centralized post-commit side-effect stage)
            self._finalize_emit_completion_event(runtime, turn_context, service)
            return finalized
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            with contextlib.suppress(Exception):
                self.rollback_transaction(turn_context)
            logger.error(
                "PersistenceService.finalize_turn strict-mode failure: %s",
                exc,
            )
            raise PersistenceFailure(
                "PersistenceService.finalize_turn strict-mode failure",
                context={
                    "trace_id": str(getattr(turn_context, "trace_id", "") or ""),
                    "error": str(exc),
                },
            ) from exc
        finally:
            if isinstance(getattr(turn_context, "state", None), dict):
                turn_context.state["_timing_finalize_ms"] = round(
                    (time.perf_counter() - finalize_started) * 1000,
                    3,
                )
            runtime._graph_commit_active = False
            runtime._current_turn_time_base = previous_temporal

    def save_turn(self, turn_context: Any, result: Any) -> None:
        snapshot_builder = getattr(turn_context, "snapshot", None)
        if callable(snapshot_builder):
            _snap = snapshot_builder(result)
            self.persistence_manager.persist_conversation_snapshot(
                dict(_snap) if isinstance(_snap, dict) else {},
                turn_context=turn_context,
            )
            return
        self.persistence_manager.persist_conversation()

    def save_graph_checkpoint(
        self,
        checkpoint: dict[str, Any],
        _skip_turn_event: bool = False,
    ) -> None:
        try:
            with ensure_execution_trace_root(
                operation="persist_graph_checkpoint",
                prompt="[persistence-service-save-checkpoint]",
                metadata={"source": "PersistenceService.save_graph_checkpoint"},
                required=True,
            ):
                self.persistence_manager.persist_graph_checkpoint(
                    checkpoint,
                    _skip_turn_event=_skip_turn_event,
                )
        except RuntimeTraceViolation:
            trace_id = str((checkpoint or {}).get("trace_id") or "").strip()
            if not trace_id:
                logger.warning(
                    "PersistenceService.save_graph_checkpoint skipped due to null trace_id fallback",
                )
                return
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.save_graph_checkpoint failed: %s", exc)

    def save_turn_event(self, event: dict[str, Any]) -> None:
        try:
            self.persistence_manager.persist_turn_event(event)
        except RuntimeTraceViolation:
            trace_id = str((event or {}).get("trace_id") or "").strip()
            if not trace_id:
                logger.warning(
                    "PersistenceService.save_turn_event skipped due to null trace_id fallback",
                )
                return
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.save_turn_event failed: %s", exc)

    def list_turn_events(self, trace_id: str, limit: int = 0) -> list[dict[str, Any]]:
        try:
            return self.persistence_manager.list_turn_events(
                trace_id=trace_id,
                limit=limit,
            )
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return []
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.list_turn_events failed: %s", exc)
            return []

    def list_policy_trace_events(
        self,
        *,
        trace_id: str = "",
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        try:
            return self.persistence_manager.list_policy_trace_events(
                trace_id=trace_id,
                limit=limit,
            )
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return []
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.list_policy_trace_events failed: %s", exc)
            return []

    def summarize_policy_trace_events(
        self,
        *,
        trace_id: str = "",
        limit: int = 0,
    ) -> dict[str, Any]:
        try:
            return self.persistence_manager.summarize_policy_trace_events(
                trace_id=trace_id,
                limit=limit,
            )
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return {
                    "event_type": "PolicyTraceEvent",
                    "event_count": 0,
                    "policies": [],
                    "action_counts": {},
                    "latest_action": "",
                    "latest_step_name": "",
                    "latest_trace_id": "",
                }
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.summarize_policy_trace_events failed: %s", exc)
            return {
                "event_type": "PolicyTraceEvent",
                "event_count": 0,
                "policies": [],
                "action_counts": {},
                "latest_action": "",
                "latest_step_name": "",
                "latest_trace_id": "",
            }

    def replay_turn_events(self, trace_id: str) -> dict[str, Any]:
        try:
            return self.persistence_manager.replay_turn_events(trace_id=trace_id)
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return {"trace_id": "", "events": [], "replayed_state": {}}
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("PersistenceService.replay_turn_events failed: %s", exc)
            return {"trace_id": str(trace_id or ""), "events": [], "replayed_state": {}}

    def validate_replay_determinism(
        self,
        trace_id: str,
        expected_lock_hash: str = "",
    ) -> dict[str, Any]:
        try:
            return self.persistence_manager.validate_replay_determinism(
                trace_id=trace_id,
                expected_lock_hash=expected_lock_hash,
            )
        except RuntimeTraceViolation:
            if not str(trace_id or "").strip():
                return {
                    "trace_id": "",
                    "consistent": False,
                    "observed_lock_hash": "",
                    "expected_lock_hash": str(expected_lock_hash or ""),
                    "matches_expected": False,
                    "lock_hashes": [],
                }
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error(
                "PersistenceService.validate_replay_determinism failed: %s",
                exc,
            )
            return {
                "trace_id": str(trace_id or ""),
                "consistent": False,
                "observed_lock_hash": "",
                "expected_lock_hash": str(expected_lock_hash or ""),
                "matches_expected": False,
                "lock_hashes": [],
            }

    def persist_conversation(self) -> None:
        self.persistence_manager.persist_conversation()
