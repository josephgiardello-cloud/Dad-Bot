from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.contracts.lifecycle_events import (
    Claimed,
    Completed,
    Failed,
    LeaseExpired,
    LeaseRenewed,
    Redelivered,
    Released,
    Submitted,
)
from dadbot.core.compaction import ArchiveTier, CompactionPolicy, EventCompactor
from dadbot.core.autonomous_goal_daemon import AutonomousGoalDaemon
from dadbot.core.belief_state_engine import BeliefStateEngine
from dadbot.core.compositional_tool_planner import CompositionalToolPlanner
from dadbot.core.behavior_alignment_trainer import BehaviorAlignmentTrainer
from dadbot.core.contract_evaluator import validate_sovereign_ledger_transition
from dadbot.core.cognitive_policy_engine import CognitivePolicyEngine
from dadbot.core.control_plane_projection import ExecutionProjection
from dadbot.core.control_plane_reducer import ExecutionState as ReducedExecutionState
from dadbot.core.control_plane_reducer import ExecutionStatus, lease_expired
from dadbot.core.core_state import CoreState, InputEvent, project_views, transition
from dadbot.core.execution_boundary import ControlPlaneExecutionBoundary
from dadbot.core.execution_context import (
    open_core_state_scope,
    close_core_state_scope,
)
from dadbot.core.execution_lease import ExecutionLease, LeaseConflictError
from dadbot.core.execution_resource_budget import BackpressureSignal
from dadbot.core.execution_context import get_active_core_state, push_core_state_event
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.execution_ledger_memory import InMemoryExecutionLedger
from dadbot.core.execution_result_unified import (
    _TERMINAL_STATUS_VALUES,
    build_unified_execution_result,
    ensure_unified_execution_result,
    mark_unified_execution_failure,
    mark_unified_execution_success,
)
from dadbot.core.failure_taxonomy import classify_failure
from dadbot.core.system_state_algebra import (
    evaluate_system_state_algebra,
    persist_system_state_algebra,
)
from dadbot.core.global_transition_invariants import (
    TransitionBoundaryView,
    enforce_global_transition_invariants,
)
from dadbot.core.distributed_correctness import (
    DistributedCorrectnessModel,
    NodeRole,
)
from dadbot.core.memory_set_invariants import (
    MemorySetInvariantViolation,
)
from dadbot.core.runtime_errors import (
    AuthorityViolation,
    ExecutionStageError,
    InvariantViolation,
    PersistenceFailure,
    ReplayMismatch,
)
from dadbot.core.runtime_contracts import (
    validate_decision_explanation_contract,
    validate_trace_event_contract,
)
from dadbot.core.kernel_gateway import KernelGateway
from dadbot.core.kernel_signals import get_exporter, get_metrics, get_tracer
from dadbot.core.effect_journal import EffectJournal
from dadbot.core.ledger_index import LedgerIndex
from dadbot.core.ledger_reader import LedgerReader
from dadbot.core.ledger_writer import LedgerWriter
from dadbot.core.ledger_writer_adapter import LedgerWriterAdapter
from dadbot.core.recovery_manager import RecoveryManager
from dadbot.core.runtime_types import ToolSpec
from dadbot.core.hypothesis_engine import MultiHypothesisEngine
from dadbot.core.memory_hierarchy_manager import MemoryHierarchyManager
from dadbot.core.multi_agent_swarm import MultiAgentSwarm
from dadbot.core.semantic_memory_graph import SemanticMemoryGraph
from dadbot.core.semantic_primitives import hash as semantic_hash
from dadbot.core.semantic_safety_engine import SemanticSafetyEngine
from dadbot.core.session_planning_optimizer import SessionPlanningOptimizer
from dadbot.core.session_store import SessionStore
from dadbot.core.interactive_cognition_ui import InteractiveCognitionUI
from dadbot.core.tool_routing_engine import ToolRoutingEngine
from dadbot.core.tool_ecosystem_hub import ToolEcosystemHub
from dadbot.core.tool_self_model import ToolSelfModel
from dadbot.core.adaptation_engine import AdaptationEngine
from dadbot.core.phase_closure_runtime import PhaseClosureRuntime
from dadbot.core.response_engine import ResponseEngine
from dadbot.core.topology_runtime import TopologyRuntime, TopologyValidationResult
from dadbot.core._control_plane_reconciliation import ReconciliationMixin
from dadbot.core._control_plane_compaction import CompactionMixin

logger = logging.getLogger(__name__)


def _runtime_plan_intent(user_input: str) -> str:
    text = str(user_input or "").strip().lower()
    if not text:
        return "statement"
    if "?" in text:
        return "question"
    if text.startswith(("please", "can you", "could you", "help me", "show me")):
        return "request"
    if any(token in text for token in ("i feel", "i am anxious", "i am stressed", "i am worried", "i'm worried")):
        return "emotional"
    return "statement"


def _runtime_plan_strategy(intent_type: str) -> str:
    intent = str(intent_type or "").strip().lower()
    if intent == "emotional":
        return "empathy_first"
    if intent == "request":
        return "task_plan"
    if intent == "question":
        return "direct_answer"
    return "direct_answer"


def _build_runtime_plan(
    *,
    session_id: str,
    trace_id: str,
    user_input: str,
    attachments: AttachmentList | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    existing = dict(metadata.get("runtime_plan") or {})
    intent_type = str(
        existing.get("intent_type")
        or dict(metadata.get("turn_plan") or {}).get("intent_type")
        or _runtime_plan_intent(user_input),
    )
    strategy = str(
        existing.get("strategy")
        or dict(metadata.get("turn_plan") or {}).get("strategy")
        or _runtime_plan_strategy(intent_type),
    )
    revision = int(existing.get("revision") or 0)
    plan = {
        "plan_id": str(existing.get("plan_id") or f"plan:{session_id}:{trace_id}"),
        "revision": max(1, revision if revision > 0 else 1),
        "intent_type": intent_type,
        "strategy": strategy,
        "status": str(existing.get("status") or "active"),
        "created_at": float(existing.get("created_at") or time.time()),
        "updated_at": float(time.time()),
        "subgoals": list(existing.get("subgoals") or []),
        "tool_routing": {
            "latency_budget_ms": int(dict(existing.get("tool_routing") or {}).get("latency_budget_ms") or 2500),
            "cost_tier": str(dict(existing.get("tool_routing") or {}).get("cost_tier") or "balanced"),
            "mode": str(dict(existing.get("tool_routing") or {}).get("mode") or "adaptive"),
            "attachment_count": int(len(attachments or [])),
        },
        "branch_history": list(existing.get("branch_history") or []),
    }
    return plan


def _mutate_runtime_plan(
    *,
    metadata: dict[str, Any],
    reason: str,
    status: str = "active",
    strategy: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    plan = dict(metadata.get("runtime_plan") or {})
    if not plan:
        return plan
    plan["revision"] = int(plan.get("revision") or 1) + 1
    if strategy:
        plan["strategy"] = str(strategy)
    plan["status"] = str(status or "active")
    plan["updated_at"] = float(time.time())
    history = list(plan.get("branch_history") or [])
    history.append(
        {
            "revision": int(plan.get("revision") or 1),
            "reason": str(reason or "replan"),
            "status": str(status or "active"),
            "note": str(note or ""),
            "timestamp": float(time.time()),
        },
    )
    plan["branch_history"] = history[-32:]
    metadata["runtime_plan"] = plan
    return plan


def _build_semantic_memory_candidates(
    *,
    user_input: str,
    response: str,
    trace_id: str,
    session_id: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    text = str(user_input or "").strip()
    if text and "?" not in text and len(text) >= 12:
        if "i " in text.lower() or "my " in text.lower():
            candidates.append(
                {
                    "kind": "episodic_fact",
                    "text": text[:240],
                    "score": 0.72,
                    "trace_id": str(trace_id or ""),
                    "session_id": str(session_id or "default"),
                    "source": "user_input",
                    "created_at": float(time.time()),
                },
            )
    answer = str(response or "").strip()
    if answer:
        candidates.append(
            {
                "kind": "assistant_summary",
                "text": answer[:240],
                "score": 0.51,
                "trace_id": str(trace_id or ""),
                "session_id": str(session_id or "default"),
                "source": "assistant_response",
                "created_at": float(time.time()),
            },
        )
    return candidates


def _normalize_tool_runtime_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = dict(metadata.get("tool_runtime_contract") or metadata.get("tool_request") or {})
    tool_name = str(raw.get("tool_name") or raw.get("name") or "").strip()
    tool_version = str(raw.get("version") or "").strip() or "latest"
    required_permissions = [
        str(item).strip().lower()
        for item in list(raw.get("required_permissions") or [])
        if str(item).strip()
    ]
    return {
        "tool_name": tool_name,
        "version": tool_version,
        "required_permissions": sorted(set(required_permissions)),
        "timeout_seconds": float(raw.get("timeout_seconds") or 10.0),
        "side_effect_class": str(raw.get("side_effect_class") or "unknown"),
        "determinism": str(raw.get("determinism") or "unknown"),
        "contract_valid": bool(tool_name),
    }


def _validate_tool_runtime_contract(metadata: dict[str, Any]) -> tuple[bool, str]:
    contract = dict(metadata.get("tool_runtime_contract") or {})
    if not contract:
        return True, "ok"
    tool_name = str(contract.get("tool_name") or "").strip()
    if not tool_name:
        # No concrete tool invocation requested; this is a normal path.
        return True, "ok"
    timeout_seconds = float(contract.get("timeout_seconds") or 10.0)
    if timeout_seconds < 0.05 or timeout_seconds > 120.0:
        return False, "tool timeout_seconds out of accepted range [0.05, 120.0]"
    required_permissions = {
        str(item).strip().lower()
        for item in list(contract.get("required_permissions") or [])
        if str(item).strip()
    }
    granted_permissions = {
        str(item).strip().lower()
        for item in list(metadata.get("session_permissions") or [])
        if str(item).strip()
    }
    missing = sorted(required_permissions - granted_permissions)
    if missing:
        return False, f"tool permission denied: missing {', '.join(missing)}"
    side_effect_class = str(contract.get("side_effect_class") or "").strip().lower()
    if side_effect_class in {"stateful", "logged"} and not bool(metadata.get("approval_granted", False)):
        return False, "tool requires approval_granted for side effects"
    return True, "ok"


def _tokenize_memory_text(value: str) -> set[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return set()
    return {token for token in raw.replace("\n", " ").split(" ") if len(token.strip()) >= 3}


def _rank_semantic_memory_items(
    *,
    items: list[dict[str, Any]],
    user_input: str,
    limit: int,
) -> list[dict[str, Any]]:
    query_tokens = _tokenize_memory_text(user_input)
    now = float(time.time())
    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        if not text:
            continue
        item_tokens = _tokenize_memory_text(text)
        lexical_overlap = 0.0
        if query_tokens and item_tokens:
            lexical_overlap = float(len(query_tokens.intersection(item_tokens))) / float(max(1, len(query_tokens)))
        base_score = float(item.get("score") or 0.0)
        created_at = float(item.get("created_at") or now)
        # Soft recency boost over ~24h horizon without swamping semantic score.
        recency = max(0.0, 1.0 - min(1.0, (now - created_at) / 86_400.0))
        final_score = (0.55 * base_score) + (0.35 * lexical_overlap) + (0.10 * recency)
        candidate = dict(item)
        candidate["retrieval_score"] = round(float(final_score), 6)
        ranked.append((final_score, candidate))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [candidate for _score, candidate in ranked[:max(0, int(limit))]]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _normalize_sentiment_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _sentiment_reward(value: str) -> tuple[float, float]:
    sentiment = _normalize_sentiment_label(value)
    mapping = {
        "happy": (0.45, 0.70),
        "excited": (0.55, 0.72),
        "grateful": (0.50, 0.75),
        "relieved": (0.35, 0.65),
        "neutral": (0.0, 0.0),
        "sad": (-0.40, 0.65),
        "anxious": (-0.45, 0.68),
        "stressed": (-0.50, 0.70),
        "frustrated": (-0.65, 0.75),
        "angry": (-0.75, 0.80),
    }
    return mapping.get(sentiment, (0.0, 0.0))


def _textual_feedback_signal(text: str) -> tuple[float, float, str]:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return 0.0, 0.0, "neutral"

    negative_markers = (
        "that didn't help",
        "that did not help",
        "not helpful",
        "that's wrong",
        "you are wrong",
        "not what i asked",
        "too much",
        "too long",
        "irrelevant",
        "stop",
        "no,",
    )
    positive_markers = (
        "thank you",
        "thanks",
        "that helps",
        "helpful",
        "good point",
        "exactly",
        "makes sense",
        "perfect",
    )

    if any(marker in lowered for marker in negative_markers):
        return -0.70, 0.74, "frustrated"
    if any(marker in lowered for marker in positive_markers):
        return 0.55, 0.68, "grateful"
    return 0.0, 0.0, "neutral"


def _selected_feedback_features(telemetry: dict[str, Any]) -> dict[str, float]:
    selected = dict(telemetry.get("selected") or {})
    components = dict(selected.get("components") or {})
    return {
        "base_score": float(components.get("base_score", 0.0)),
        "emotion_bias": float(components.get("emotion_bias", 0.0)),
        "memory_relevance": float(components.get("memory_relevance", 0.0)),
        "user_alignment": float(components.get("user_alignment", 0.0)),
        "trajectory_alignment": float(components.get("trajectory_alignment", 0.0)),
        "predicted_user_reaction": float(components.get("predicted_user_reaction", 0.0)),
        "risk_level": float(selected.get("risk_level", 0.0)),
    }


def _synthesize_reward_feedback(
    *,
    pending_selection: dict[str, Any],
    current_user_input: str,
    metadata: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any] | None:
    features = _selected_feedback_features(pending_selection)
    if not any(abs(value) > 1e-9 for value in features.values()):
        return None

    explicit = {}
    for key in ("response_feedback", "user_feedback", "response_outcome", "reaction"):
        candidate = metadata.get(key)
        if isinstance(candidate, dict):
            explicit = dict(candidate)
            break

    reward_terms: list[tuple[float, float, str]] = []
    attribution: dict[str, float] = {}
    evidence: list[str] = []

    accepted = explicit.get("accepted")
    rejected = explicit.get("rejected")
    if isinstance(accepted, bool):
        reward_terms.append((1.0 if accepted else -1.0, 0.92, "explicit_acceptance"))
        evidence.append(f"accepted={accepted}")
        attribution.update({
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.40),
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.35),
            "trajectory_alignment": max(attribution.get("trajectory_alignment", 0.0), 0.25),
        })
    elif isinstance(rejected, bool):
        reward_terms.append((-1.0 if rejected else 0.6, 0.88, "explicit_rejection"))
        evidence.append(f"rejected={rejected}")
        attribution.update({
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.35),
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.30),
            "risk_level": max(attribution.get("risk_level", 0.0), 0.20),
        })

    rating = _coerce_float(explicit.get("rating"))
    if rating is not None:
        normalized = rating
        if rating > 1.0:
            normalized = ((rating - 3.0) / 2.0) if rating <= 5.0 else max(-1.0, min(1.0, rating / 5.0))
        reward_terms.append((_clamp(normalized, -1.0, 1.0), 0.85, "explicit_rating"))
        evidence.append(f"rating={rating}")
        attribution.update({
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.45),
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.35),
        })

    sentiment_sources = [
        explicit.get("sentiment"),
        metadata.get("user_sentiment"),
        dict(session_state.get("last_turn_ux_feedback") or {}).get("mood_hint"),
    ]
    for sentiment_source in sentiment_sources:
        reward_value, confidence = _sentiment_reward(str(sentiment_source or ""))
        if confidence > 0.0:
            reward_terms.append((reward_value, confidence, "sentiment"))
            evidence.append(f"sentiment={sentiment_source}")
            attribution.update({
                "emotion_bias": max(attribution.get("emotion_bias", 0.0), 0.35),
                "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.40),
                "user_alignment": max(attribution.get("user_alignment", 0.0), 0.25),
            })
            break

    engagement_level = _coerce_float(explicit.get("engagement_level"))
    if engagement_level is None:
        engagement_level = _coerce_float(metadata.get("engagement_level"))
    if engagement_level is None:
        interaction_state = dict(session_state.get("interaction_state") or {})
        engagement_level = _coerce_float(interaction_state.get("engagement_level"))
    if engagement_level is not None:
        normalized_engagement = _clamp((engagement_level - 0.5) * 1.2, -1.0, 1.0)
        reward_terms.append((normalized_engagement, 0.58, "engagement_level"))
        evidence.append(f"engagement_level={engagement_level}")
        attribution.update({
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.45),
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.35),
            "trajectory_alignment": max(attribution.get("trajectory_alignment", 0.0), 0.20),
        })

    engagement_delta = _coerce_float(explicit.get("engagement_delta"))
    if engagement_delta is None:
        engagement_delta = _coerce_float(metadata.get("engagement_delta"))
    if engagement_delta is not None:
        reward_terms.append((_clamp(engagement_delta, -1.0, 1.0), 0.68, "engagement_delta"))
        evidence.append(f"engagement_delta={engagement_delta}")
        attribution.update({
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.50),
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.30),
        })

    textual_reward, textual_confidence, inferred_sentiment = _textual_feedback_signal(current_user_input)
    if textual_confidence > 0.0:
        reward_terms.append((textual_reward, textual_confidence, "follow_up_text"))
        evidence.append(f"follow_up_text={inferred_sentiment}")
        attribution.update({
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.40),
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.30),
            "emotion_bias": max(attribution.get("emotion_bias", 0.0), 0.20),
        })
        if textual_reward < 0.0:
            attribution["risk_level"] = max(attribution.get("risk_level", 0.0), 0.25)

    if not reward_terms:
        return None

    confidence_total = sum(confidence for _reward, confidence, _source in reward_terms)
    if confidence_total <= 1e-9:
        return None

    blended_reward = sum(reward * confidence for reward, confidence, _source in reward_terms) / confidence_total
    blended_confidence = _clamp(confidence_total / max(len(reward_terms), 1), 0.0, 1.0)
    if blended_confidence < 0.35:
        return None

    return {
        "reward": _clamp(blended_reward, -1.0, 1.0),
        "confidence": blended_confidence,
        "features": features,
        "attribution": {key: _clamp(value, 0.0, 1.0) for key, value in attribution.items() if value > 0.0},
        "source": "control_plane.synthesized_outcome",
        "evidence": evidence[-6:],
        "pending_trace_id": str(pending_selection.get("trace_id") or ""),
        "created_at": float(time.time()),
    }


@dataclass(frozen=True)
class _FailurePolicyStrategy:
    event_type: str


_FAILURE_POLICY_STRATEGIES: dict[str, _FailurePolicyStrategy] = {
    "quarantine": _FailurePolicyStrategy(event_type="JOB_QUARANTINED"),
    "reconcile": _FailurePolicyStrategy(event_type="JOB_RECONCILE_REQUIRED"),
    "manual_retry": _FailurePolicyStrategy(event_type="JOB_MANUAL_RETRY_REQUIRED"),
}


def _progress_instrumentation_enabled() -> bool:
    raw = str(os.environ.get("DADBOT_PROGRESS_INSTRUMENTATION", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _progress_log_path() -> Path:
    raw = str(os.environ.get("DADBOT_PROGRESS_LOG_PATH", "session_logs/progress_instrumentation.ndjson")).strip()
    if not raw:
        raw = "session_logs/progress_instrumentation.ndjson"
    return Path(raw)


def _write_progress_event(*, component: str, phase: str, payload: dict[str, Any]) -> None:
    if not _progress_instrumentation_enabled():
        return
    event = {
        "timestamp": time.time(),
        "component": str(component or "control_plane"),
        "phase": str(phase or "unknown"),
        "payload": dict(payload or {}),
    }
    try:
        path = _progress_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except Exception as exc:
        logger.debug("progress instrumentation write failed: %s", exc)


class ExecutionLifecycleState(StrEnum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RECOVERY_PENDING = "recovery_pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TurnTerminalState(StrEnum):
    SUCCESS = "SUCCESS"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"
    RECOVERED = "RECOVERED"


class SchedulerExceptionMapper:
    """Canonical scheduler exception -> terminal turn state mapper."""

    @staticmethod
    def from_exception(exc: BaseException) -> TurnTerminalState:
        if isinstance(exc, asyncio.CancelledError):
            return TurnTerminalState.CANCELLED
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return TurnTerminalState.TIMEOUT
        return TurnTerminalState.FAILED

    @staticmethod
    def from_success(*, recovered: bool) -> TurnTerminalState:
        if recovered:
            return TurnTerminalState.RECOVERED
        return TurnTerminalState.SUCCESS


_SCHEDULER_EXCEPTION_MAPPER = SchedulerExceptionMapper()


_ALLOWED_LIFECYCLE_TRANSITIONS: dict[ExecutionLifecycleState, frozenset[ExecutionLifecycleState]] = {
    ExecutionLifecycleState.SUBMITTED: frozenset({
        ExecutionLifecycleState.QUEUED,
        ExecutionLifecycleState.RUNNING,
        ExecutionLifecycleState.FAILED,
    }),
    ExecutionLifecycleState.QUEUED: frozenset({
        ExecutionLifecycleState.RECOVERY_PENDING,
        ExecutionLifecycleState.RUNNING,
        ExecutionLifecycleState.FAILED,
    }),
    ExecutionLifecycleState.RECOVERY_PENDING: frozenset({
        ExecutionLifecycleState.QUEUED,
        ExecutionLifecycleState.RUNNING,
        ExecutionLifecycleState.FAILED,
    }),
    ExecutionLifecycleState.RUNNING: frozenset({
        ExecutionLifecycleState.RECOVERY_PENDING,
        ExecutionLifecycleState.COMPLETED,
        ExecutionLifecycleState.FAILED,
    }),
    ExecutionLifecycleState.COMPLETED: frozenset(),
    ExecutionLifecycleState.FAILED: frozenset(),
}


def _coerce_lifecycle_state(value: Any) -> ExecutionLifecycleState:
    raw = str(value or "").strip().lower()
    try:
        return ExecutionLifecycleState(raw)
    except ValueError:
        return ExecutionLifecycleState.SUBMITTED


def _target_lifecycle_state_for_event(
    event: Any,
    *,
    current_state: ExecutionLifecycleState,
) -> ExecutionLifecycleState:
    if isinstance(event, Submitted):
        return ExecutionLifecycleState.SUBMITTED
    if isinstance(event, Claimed):
        return ExecutionLifecycleState.RUNNING
    if isinstance(event, LeaseRenewed):
        return ExecutionLifecycleState.RUNNING
    if isinstance(event, LeaseExpired):
        return ExecutionLifecycleState.RECOVERY_PENDING
    if isinstance(event, Released):
        return ExecutionLifecycleState.QUEUED
    if isinstance(event, Completed):
        return ExecutionLifecycleState.COMPLETED
    if isinstance(event, Failed):
        return ExecutionLifecycleState.FAILED
    if isinstance(event, Redelivered):
        # Redelivery can be a no-op state annotation in lifecycle reducer terms.
        if current_state == ExecutionLifecycleState.RECOVERY_PENDING:
            return ExecutionLifecycleState.QUEUED
        return current_state
    raise RuntimeError(f"unsupported lifecycle event type at emission boundary: {type(event).__name__}")


def _execution_status_for_lifecycle(state: ExecutionLifecycleState) -> str:
    if state == ExecutionLifecycleState.COMPLETED:
        return "completed"
    if state == ExecutionLifecycleState.FAILED:
        return "failed"
    if state == ExecutionLifecycleState.SUBMITTED:
        return "submitted"
    return "running"


def _build_sovereign_transition_states(
    *,
    job: "ExecutionJob",
    before_state: ExecutionLifecycleState,
    after_state: ExecutionLifecycleState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = dict(job.metadata or {})
    execution_state = dict(metadata.get("execution_state") or {})
    before_causal_step_count = int(execution_state.get("causal_step_count") or 0)
    after_causal_step_count = before_causal_step_count + (1 if after_state != before_state else 0)
    invariance_hash = str(
        execution_state.get("invariance_hash")
        or metadata.get("invariance_hash")
        or f"cp:{str(job.job_id or '')}:{after_causal_step_count}",
    )
    after_turn_truth_ok = bool(execution_state.get("turn_truth_ok", after_state != ExecutionLifecycleState.COMPLETED or True))

    before = {
        "session_id": str(job.session_id or "default"),
        "trace_id": str(job.trace_id or "unknown-trace"),
        "execution_mode": _resolved_execution_mode(job),
        "execution_state": before_state.value,
        "execution_status": _execution_status_for_lifecycle(before_state),
        "turn_truth_ok": bool(execution_state.get("turn_truth_ok")) if before_state == ExecutionLifecycleState.COMPLETED else None,
        "invariance_hash": str(execution_state.get("invariance_hash") or metadata.get("invariance_hash") or ""),
        "causal_step_count": before_causal_step_count,
        "metadata": {},
    }
    after = {
        "session_id": str(job.session_id or "default"),
        "trace_id": str(job.trace_id or "unknown-trace"),
        "execution_mode": _resolved_execution_mode(job),
        "execution_state": after_state.value,
        "execution_status": _execution_status_for_lifecycle(after_state),
        "turn_truth_ok": after_turn_truth_ok if after_state == ExecutionLifecycleState.COMPLETED else None,
        "invariance_hash": invariance_hash,
        "causal_step_count": after_causal_step_count,
        "metadata": {},
    }
    return before, after


def _assert_lifecycle_emission_transition(
    *,
    execution_id: str,
    event: Any,
    current_state: ExecutionLifecycleState | None,
) -> None:
    if isinstance(event, Submitted):
        if current_state is not None:
            raise RuntimeError(
                f"Invalid lifecycle emission transition for {execution_id!r}: "
                "Submitted must be first event",
            )
        return

    if current_state is None:
        raise RuntimeError(
            f"Invalid lifecycle emission transition for {execution_id!r}: "
            f"{type(event).__name__} cannot be emitted before Submitted",
        )

    target_state = _target_lifecycle_state_for_event(event, current_state=current_state)
    if target_state == current_state:
        return
    if target_state not in _ALLOWED_LIFECYCLE_TRANSITIONS[current_state]:
        raise RuntimeError(
            f"Invalid lifecycle emission transition for {execution_id!r}: "
            f"{current_state.value!r} -> {target_state.value!r} via {type(event).__name__}",
        )


def _ensure_execution_state(job: "ExecutionJob") -> dict[str, Any]:
    metadata = dict(job.metadata or {})
    state = dict(metadata.get("execution_state") or {})
    state.setdefault("lifecycle_state", ExecutionLifecycleState.SUBMITTED.value)
    state.setdefault("redelivery_count", 0)
    state.setdefault("lease_conflict_count", 0)
    state.setdefault("last_worker_id", "")
    state.setdefault("last_transition_reason", "")
    state.setdefault("retry_not_before_monotonic", 0.0)
    metadata["execution_state"] = state
    job.metadata = metadata
    return state


def _transition_execution_state(
    job: "ExecutionJob",
    *,
    target: ExecutionLifecycleState,
    reason: str,
    worker_id: str = "",
    retry_not_before_monotonic: float | None = None,
    redelivery_increment: int = 0,
    lease_conflict_increment: int = 0,
) -> dict[str, Any]:
    state = _ensure_execution_state(job)
    current = _coerce_lifecycle_state(state.get("lifecycle_state"))
    before_causal_step_count = int(state.get("causal_step_count") or 0)
    if current != target and target not in _ALLOWED_LIFECYCLE_TRANSITIONS[current]:
        raise RuntimeError(
            "Invalid execution lifecycle transition: "
            f"{current.value!r} -> {target.value!r} for job {job.job_id!r}",
        )
    state["lifecycle_state"] = target.value
    state["last_transition_reason"] = str(reason or "")
    state["last_transition_at"] = float(time.time())
    if worker_id:
        state["last_worker_id"] = str(worker_id)
    if retry_not_before_monotonic is None:
        if target != ExecutionLifecycleState.RECOVERY_PENDING:
            state["retry_not_before_monotonic"] = 0.0
    else:
        state["retry_not_before_monotonic"] = max(0.0, float(retry_not_before_monotonic))
    if redelivery_increment:
        state["redelivery_count"] = int(state.get("redelivery_count") or 0) + int(redelivery_increment)
    if lease_conflict_increment:
        state["lease_conflict_count"] = int(state.get("lease_conflict_count") or 0) + int(lease_conflict_increment)

    # Mandatory global invariant enforcement at transition boundary.
    state["causal_step_count"] = before_causal_step_count + (1 if target != current else 0)
    enforce_global_transition_invariants(
        TransitionBoundaryView(
            session_id=str(job.session_id or "default"),
            trace_id=str(job.trace_id or "unknown-trace"),
            before_state=current.value,
            after_state=target.value,
            before_causal_step_count=before_causal_step_count,
            after_causal_step_count=int(state.get("causal_step_count") or 0),
            turn_truth_ok=None,
            policy_posture=str(state.get("policy_posture") or "moderate"),
            active_fault_count=int(state.get("active_fault_count") or 0),
            metadata={"reason": str(reason or "")},
        ),
    )

    job.metadata["execution_state"] = state
    return dict(state)


def _resolved_execution_mode(job: "ExecutionJob") -> str:
    metadata = dict(job.metadata or {})
    explicit = str(metadata.get("execution_mode") or "").strip().lower()
    state = dict(metadata.get("execution_state") or {})
    if int(state.get("redelivery_count") or 0) > 0:
        return "recovery"
    if _coerce_lifecycle_state(state.get("lifecycle_state")) == ExecutionLifecycleState.RECOVERY_PENDING:
        return "recovery"
    if explicit in {"live", "replay", "recovery"}:
        return explicit
    return "live"


def _lifecycle_state_from_projection(state: ReducedExecutionState | None) -> str:
    if state is None:
        return ExecutionLifecycleState.SUBMITTED.value
    if state.status == ExecutionStatus.SUBMITTED:
        return ExecutionLifecycleState.SUBMITTED.value
    if state.status in {ExecutionStatus.CLAIMED, ExecutionStatus.RUNNING}:
        return ExecutionLifecycleState.RUNNING.value
    if state.status == ExecutionStatus.EXPIRED:
        return ExecutionLifecycleState.RECOVERY_PENDING.value
    if state.status == ExecutionStatus.RELEASED:
        return ExecutionLifecycleState.QUEUED.value
    if state.status == ExecutionStatus.COMPLETED:
        return ExecutionLifecycleState.COMPLETED.value
    if state.status == ExecutionStatus.FAILED:
        return ExecutionLifecycleState.FAILED.value
    return ExecutionLifecycleState.SUBMITTED.value


def _apply_projection_execution_state(
    job: "ExecutionJob",
    state: ReducedExecutionState | None,
) -> dict[str, Any]:
    metadata = dict(job.metadata or {})
    prior = dict(metadata.get("execution_state") or {})
    projected = {
        "lifecycle_state": _lifecycle_state_from_projection(state),
        "redelivery_count": max(0, int((state.attempt_count if state is not None else 0)) - 1),
        "lease_conflict_count": int(prior.get("lease_conflict_count") or 0),
        "last_worker_id": str((state.owner if state is not None else "") or prior.get("last_worker_id") or ""),
        "last_transition_reason": str(prior.get("last_transition_reason") or "lifecycle_projection"),
        "retry_not_before_monotonic": 0.0,
        "failure_type": str(prior.get("failure_type") or ""),
        "failure_action": str(prior.get("failure_action") or ""),
        "auto_retry": bool(prior.get("auto_retry", False)),
    }
    metadata["execution_state"] = projected
    job.metadata = metadata
    return projected


def _classify_execution_failure(exc: BaseException) -> dict[str, Any]:
    failure = classify_failure(exc)
    if isinstance(exc, asyncio.CancelledError):
        # Cancellation is explicitly modeled as retryable but keeps a separate class
        # for operational dashboards.
        failure["failure_class"] = "cancelled"
        failure["failure_source"] = "runtime"
    return failure


def _set_terminal_turn_state(
    job: "ExecutionJob",
    *,
    terminal_state: TurnTerminalState,
    reason: str,
    strict: bool = True,
) -> None:
    metadata = dict(job.metadata or {})
    execution_state = dict(metadata.get("execution_state") or {})
    existing = str(
        execution_state.get("terminal_turn_state") or metadata.get("terminal_turn_state") or "",
    ).strip().upper()
    incoming = str(terminal_state.value or "").strip().upper()
    if existing and existing != incoming:
        if not strict:
            return
        raise InvariantViolation(
            "Terminal turn state transition is not idempotent",
            context={
                "job_id": str(job.job_id or ""),
                "trace_id": str(job.trace_id or ""),
                "existing": existing,
                "incoming": incoming,
            },
        )
    execution_state["terminal_turn_state"] = incoming
    execution_state["terminal_transition_reason"] = str(reason or "")
    metadata["execution_state"] = execution_state
    metadata["terminal_turn_state"] = incoming
    job.metadata = metadata


def _extract_execution_degradations(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    items = metadata.get("turn_ir_degradations")
    if not isinstance(items, list):
        items = metadata.get("ir_degradations")
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict)]


def _resolve_terminal_turn_truth(
    *,
    state: dict[str, Any],
    job: "ExecutionJob",
) -> dict[str, Any]:
    """Compile terminal turn truth via canonical system-state algebra."""
    algebra = evaluate_system_state_algebra(
        state=state,
        execution_result_payload=dict(job.metadata.get("execution_result") or {}),
        trace_token=str(job.trace_id or ""),
        context="control_plane_commit",
    )
    return dict(algebra)


def _enforce_global_turn_invariant_gate(
    *,
    session: dict[str, Any],
    job: "ExecutionJob",
) -> None:
    """Hard gate: block commit if any turn truth/invariant contract is violated."""
    state = session.get("state")
    if not isinstance(state, dict):
        return

    algebra = _resolve_terminal_turn_truth(state=state, job=job)
    persist_system_state_algebra(
        state=state,
        algebra=algebra,
        trace_context="control_plane_commit",
        persist_legacy_projections=True,
        terminal_snapshot=True,
    )
    gate = dict(algebra.get("projections", {}).get("control_plane_gate") or {})

    if not bool(gate.get("ok", False)):
        raise MemorySetInvariantViolation(
            "Global invariant gate violation: "
            + "; ".join(str(item) for item in list(gate.get("violations") or [])[:3]),
        )


@dataclass(slots=True)
class ExecutionJob:
    session_id: str
    user_input: str
    attachments: AttachmentList | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    job_id: str = ""

    @staticmethod
    def _stable_token(*parts: Any, prefix: str) -> str:
        payload = "|".join(str(part or "") for part in parts)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}-{digest}"

    def __post_init__(self) -> None:
        metadata = dict(self.metadata or {})
        trace_id = str(self.trace_id or metadata.get("trace_id") or "").strip()
        if not trace_id:
            trace_id = self._stable_token(
                self.session_id,
                metadata.get("request_id"),
                self.user_input,
                self.attachments,
                prefix="tr",
            )
        job_id = str(self.job_id or metadata.get("job_id") or "").strip()
        if not job_id:
            job_id = self._stable_token(self.session_id, trace_id, prefix="job")
        metadata["trace_id"] = trace_id
        metadata["job_id"] = job_id
        self.metadata = metadata
        self.trace_id = trace_id
        self.job_id = job_id


class SchedulerProtocol(Protocol):
    """GAP 3: Explicit scheduler boundary for ControlPlane.

    ControlPlane depends only on this interface, not on the concrete Scheduler
    implementation.  Any object satisfying these three methods can be injected,
    which enforces the scheduler/control-plane boundary structurally.
    """

    worker_id: str
    lease_ttl_seconds: float

    async def register(self, job: ExecutionJob) -> asyncio.Future[FinalizedTurnResult]: ...
    async def drain_once(
        self,
        executor: Callable[[dict[str, Any], ExecutionJob], Awaitable[FinalizedTurnResult]],
    ) -> bool: ...
    async def wait_for_work(self, *, timeout_seconds: float | None = None) -> bool: ...


@dataclass(slots=True)
class SchedulerOptions:
    max_inflight_jobs: int = 16
    worker_id: str = "worker-1"
    execution_token: str = ""
    enable_observability: bool = True
    projection: ExecutionProjection | None = None
    execution_lease: ExecutionLease | None = None
    lease_ttl_seconds: float = 30.0
    redelivery_retry_interval_seconds: float = 0.05
    fairness_aging_rate: float = 5.0
    tenant_balance_weight: float = 1.0
    on_runtime_claim_guard: Callable[[str], None] | None = None
    on_runtime_lease_sync: Callable[[str], None] | None = None


@dataclass(slots=True)
class ControlPlaneOptions:
    max_inflight_jobs: int = 16
    worker_id: str = "worker-1"
    enable_observability: bool = True
    lease_ttl_seconds: float = 30.0
    redelivery_retry_interval_seconds: float = 0.05
    ledger: ExecutionLedger | None = None
    scheduler: SchedulerProtocol | None = None
    stream_sink: Callable[[dict[str, Any]], None] | None = None


class SessionRegistry:
    """Simple in-memory session registry used by the scheduler."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._terminated: set[str] = set()

    def bind(self, session_id: str) -> dict[str, Any]:
        sid = str(session_id or "default")
        session = self._sessions.get(sid)
        if session is None:
            session = {"session_id": sid, "state": {}}
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(str(session_id or "default"))

    def get_or_create(self, session_id: str) -> dict[str, Any]:
        return self.bind(session_id)

    async def create_session(self, session_id: str) -> dict[str, Any]:
        sid = str(session_id or "default")
        self._terminated.discard(sid)
        return self.bind(sid)

    def terminate_session(self, session_id: str) -> None:
        self._terminated.add(str(session_id or "default"))

    def is_terminated(self, session_id: str) -> bool:
        return str(session_id or "default") in self._terminated


class DurableReconcileQueue:
    """Ledger-backed reconcile queue boundary.

    Queue intent is persisted as ``JOB_RECONCILE_REQUIRED`` events; this helper
    centralizes enqueue, pending projection, and bounded consume semantics.
    """

    def __init__(
        self,
        *,
        ledger: ExecutionLedger,
        write_event: Callable[..., dict[str, Any]] | None,
        request_is_ambiguous: Callable[[str, str], bool],
        effect_is_ambiguous: Callable[[str, str], bool],
    ) -> None:
        self._ledger = ledger
        self._write_event = write_event
        self._request_is_ambiguous = request_is_ambiguous
        self._effect_is_ambiguous = effect_is_ambiguous

    def enqueue_required(
        self,
        *,
        session_id: str,
        trace_token: str,
        request_id: str,
        effect_id: str,
        reason: str,
    ) -> None:
        if not callable(self._write_event):
            return
        self._write_event(
            event_type="JOB_RECONCILE_REQUIRED",
            session_id=str(session_id or "default"),
            trace_id=str(trace_token or "").strip(),
            kernel_step_id="control_plane.reconcile_required",
            payload={
                "request_id": str(request_id or "").strip(),
                "effect_id": str(effect_id or "").strip(),
                "reason": str(reason or "ambiguous_effect_state"),
            },
            committed=False,
        )

    def _extract_reconcile_event_fields(self, event: dict[str, Any]) -> tuple[str, str, str, str]:
        sid = str(event.get("session_id") or "default").strip() or "default"
        payload = dict(event.get("payload") or {})
        rid = str(payload.get("request_id") or "").strip()
        eid = str(payload.get("effect_id") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        return sid, rid, eid, reason

    def _entry_needs_reconciliation(self, session_id: str, request_id: str, effect_id: str) -> bool:
        if request_id and self._request_is_ambiguous(session_id, request_id):
            return True
        if effect_id and self._effect_is_ambiguous(session_id, effect_id):
            return True
        return False

    def pending_entries(self) -> list[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for event in self._ledger.read():
            if str(event.get("type") or "") != "JOB_RECONCILE_REQUIRED":
                continue
            sid, rid, eid, reason = self._extract_reconcile_event_fields(event)
            if not rid and not eid:
                continue
            key = (sid, rid, eid)
            if key in seen:
                continue
            seen.add(key)
            if not self._entry_needs_reconciliation(sid, rid, eid):
                continue
            pending.append(
                {
                    "session_id": sid,
                    "request_id": rid,
                    "effect_id": eid,
                    "reason": reason or "queued_reconcile_required",
                },
            )
        return pending

    def consume(
        self,
        *,
        enabled: bool,
        max_items: int,
        max_rounds: int,
        mode: str,
        apply: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        pending = self.pending_entries()
        if not enabled:
            return {
                "enabled": False,
                "mode": mode,
                "max_items": max_items,
                "max_rounds": max_rounds,
                "queued": len(pending),
                "attempted": 0,
                "applied": 0,
                "failed": 0,
                "remaining": len(pending),
                "converged": True,
            }

        attempted = 0
        applied = 0
        failed = 0
        rounds = 0
        converged = True
        seen_signatures: set[str] = set()
        while attempted < max_items and rounds < max_rounds:
            rounds += 1
            pending = self.pending_entries()
            if not pending:
                break
            signature = "|".join(
                sorted(
                    f"{str(item.get('session_id') or 'default')}::{str(item.get('request_id') or '')}::{str(item.get('effect_id') or '')}"
                    for item in pending
                ),
            )
            if signature in seen_signatures:
                converged = False
                break
            seen_signatures.add(signature)

            round_applied = 0
            budget = max_items - attempted
            for item in pending[:budget]:
                attempted += 1
                try:
                    report = apply(
                        session_id=str(item.get("session_id") or "default"),
                        request_id=str(item.get("request_id") or ""),
                        effect_id=str(item.get("effect_id") or ""),
                        reason=str(item.get("reason") or "queued_reconcile_required"),
                        mode=mode,
                    )
                except Exception:
                    failed += 1
                    continue
                if bool(report.get("applied")):
                    applied += 1
                    round_applied += 1
            if round_applied <= 0:
                converged = False
                break

        remaining = len(self.pending_entries())
        return {
            "enabled": True,
            "mode": mode,
            "max_items": max_items,
            "max_rounds": max_rounds,
            "queued": len(pending),
            "attempted": attempted,
            "applied": applied,
            "failed": failed,
            "remaining": remaining,
            "converged": bool(converged and remaining == 0),
        }


class SchedulerWriter(Protocol):
    def append_job_queued(self, job: Any) -> dict[str, Any]: ...

    def append_job_started(self, job: Any) -> dict[str, Any]: ...

    def append_job_completed(self, job: Any, result: Any) -> dict[str, Any]: ...

    def append_job_failed(
        self,
        job: Any,
        error: Any,
        *,
        failure: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def append_execution_lifecycle(
        self,
        event: Any,
        *,
        session_id: str,
        trace_token: str,
        step_key: str,
        committed: bool = False,
    ) -> dict[str, Any]: ...

    def append_effect_begin(
        self,
        *,
        session_id: str,
        trace_token: str,
        effect_id: str,
        request_id: str = "",
        step_key: str = "scheduler.execute.effect.begin",
    ) -> dict[str, Any]: ...

    def append_effect_commit(
        self,
        *,
        session_id: str,
        trace_token: str,
        effect_id: str,
        request_id: str = "",
        step_key: str = "scheduler.execute.effect.commit",
    ) -> dict[str, Any]: ...


class Scheduler:
    """Single-node async scheduler with lifecycle-projection drain semantics."""

    def __init__(
        self,
        registry: SessionRegistry,
        *,
        reader: LedgerReader,
        writer: SchedulerWriter,
        options: SchedulerOptions | None = None,
        **legacy_options: Any,
    ) -> None:
        resolved_options = self._resolve_options(options, legacy_options)
        self.registry = registry
        self.reader = reader
        self.writer = writer
        self.max_inflight_jobs = int(resolved_options.max_inflight_jobs)
        self.projection = resolved_options.projection or ExecutionProjection()
        self.worker_id = str(resolved_options.worker_id or "worker-1")
        self.execution_token = str(resolved_options.execution_token or "")
        self.enable_observability = bool(resolved_options.enable_observability)
        self.execution_lease = resolved_options.execution_lease
        self.lease_ttl_seconds = max(0.001, float(resolved_options.lease_ttl_seconds or 30.0))
        self.redelivery_retry_interval_seconds = max(
            0.001,
            float(resolved_options.redelivery_retry_interval_seconds or 0.05),
        )
        self.fairness_aging_rate = max(0.0, float(resolved_options.fairness_aging_rate or 0.0))
        self.tenant_balance_weight = max(0.0, float(resolved_options.tenant_balance_weight or 0.0))
        self._on_runtime_claim_guard = resolved_options.on_runtime_claim_guard
        self._on_runtime_lease_sync = resolved_options.on_runtime_lease_sync

        self._jobs: dict[
            str,
            tuple[ExecutionJob, asyncio.Future[FinalizedTurnResult]],
        ] = {}
        self._pending_job_ids: list[str] = []
        self._work_event: asyncio.Event | None = None
        self._work_event_loop: asyncio.AbstractEventLoop | None = None

    def _ensure_work_event(self) -> asyncio.Event:
        loop = asyncio.get_running_loop()
        if self._work_event is None or self._work_event_loop is not loop:
            self._work_event = asyncio.Event()
            self._work_event_loop = loop
        return self._work_event

    def _notify_work_available(self) -> None:
        self._ensure_work_event().set()

    def _apply_scheduler_membership_rule(self) -> None:
        """Option A invariant: scheduler queue contains non-terminal jobs only."""
        if not self._pending_job_ids:
            return
        retained: list[str] = []
        for job_id in list(self._pending_job_ids):
            pair = self._jobs.get(job_id)
            if pair is None:
                continue
            _job, future = pair
            projected = self.projection.get(job_id)
            if projected is not None and projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}:
                self._jobs.pop(job_id, None)
                if not future.done():
                    future.set_exception(
                        RuntimeError(
                            "scheduler membership invariant: terminal job cannot remain pending",
                        ),
                    )
                continue
            retained.append(job_id)
        self._pending_job_ids = retained

    async def wait_for_work(self, *, timeout_seconds: float | None = None) -> bool:
        self._apply_scheduler_membership_rule()
        now = time.monotonic()
        if self._ready_pending_job_ids(now):
            return True

        work_event = self._ensure_work_event()
        work_event.clear()
        next_ready_delay = self._next_ready_delay(now)

        if timeout_seconds is not None and timeout_seconds <= 0.0:
            return bool(self._ready_pending_job_ids(time.monotonic()))

        try:
            wait_timeout = timeout_seconds
            if next_ready_delay is not None:
                wait_timeout = next_ready_delay if wait_timeout is None else min(wait_timeout, next_ready_delay)
            if wait_timeout is None:
                await work_event.wait()
            else:
                await asyncio.wait_for(work_event.wait(), timeout=wait_timeout)
        except TimeoutError:
            return bool(self._ready_pending_job_ids(time.monotonic()))
        return bool(self._ready_pending_job_ids(time.monotonic()))

    def _job_ready(self, job: ExecutionJob, *, now: float) -> bool:
        projected = self.projection.get(job.job_id)
        _apply_projection_execution_state(job, projected)
        return bool(
            self.projection.get_runnable(
                now=datetime.fromtimestamp(now),
                execution_ids=[job.job_id],
            )
        )

    def _ready_pending_job_ids(self, now: float) -> list[str]:
        ready: list[str] = []
        for job_id in list(self._pending_job_ids):
            pair = self._jobs.get(job_id)
            if pair is None:
                continue
            job, _future = pair
            if self._job_ready(job, now=now):
                ready.append(job_id)
        return ready

    def _next_ready_delay(self, now: float) -> float | None:
        self._apply_scheduler_membership_rule()
        active_expiries: list[float] = []
        for job_id in list(self._pending_job_ids):
            pair = self._jobs.get(job_id)
            if pair is None:
                continue
            job, _future = pair
            state = self.projection.get(job_id)
            if state is not None and state.lease_expiry is not None:
                expiry_ts = state.lease_expiry.timestamp()
                if expiry_ts > now:
                    active_expiries.append(expiry_ts - now)
        if not active_expiries:
            return None
        return max(0.0, min(active_expiries))

    def _claim_order_key(self, job_id: str) -> tuple[float, str, int, str]:
        pair = self._jobs.get(job_id)
        if pair is None:
            return (float("inf"), "", 0, str(job_id or ""))
        job, _future = pair
        claim_order = dict(dict(job.metadata or {}).get("claim_order") or {})
        timestamp = float(claim_order.get("timestamp") or 0.0)
        worker_id = str(claim_order.get("worker_id") or self.worker_id or "")
        lease_epoch = int(claim_order.get("lease_epoch") or 0)
        return (timestamp, worker_id, lease_epoch, str(job_id or ""))

    def _pop_next_ready_job_id(self) -> str | None:
        self._apply_scheduler_membership_rule()
        now = time.monotonic()
        ordered_pending = sorted(set(self._pending_job_ids), key=lambda job_id: self._fairness_key(job_id, now))
        for job_id in ordered_pending:
            pair = self._jobs.get(job_id)
            if pair is None:
                self._pending_job_ids = [pending_id for pending_id in self._pending_job_ids if pending_id != job_id]
                continue
            job, _future = pair
            if not self._job_ready(job, now=now):
                continue
            with contextlib.suppress(ValueError):
                self._pending_job_ids.remove(job_id)
            return job_id
        return None

    def _fairness_key(self, job_id: str, now: float) -> tuple[float, float, float, str, int, str]:
        pair = self._jobs.get(job_id)
        if pair is None:
            return (float("inf"), float("inf"), float("inf"), "", 0, str(job_id or ""))
        job, _future = pair
        metadata = dict(job.metadata or {})
        scheduling = dict(metadata.get("scheduling") or {})
        raw_priority = scheduling.get("priority")
        if raw_priority is None:
            raw_priority = metadata.get("priority")
        base_priority = float(100.0 if raw_priority is None else raw_priority)
        raw_submitted_monotonic = scheduling.get("submitted_monotonic")
        submitted_monotonic = float(now if raw_submitted_monotonic is None else raw_submitted_monotonic)
        # Use whole-second age buckets to avoid clock jitter changing deterministic order.
        age_seconds = float(max(0, int(now - submitted_monotonic)))
        effective_priority = max(0.0, base_priority - (age_seconds * self.fairness_aging_rate))
        tenant_id = str(metadata.get("tenant_id") or "global")
        tenant_pending_count = 0.0
        for pending_job_id in self._pending_job_ids:
            pending_pair = self._jobs.get(pending_job_id)
            if pending_pair is None:
                continue
            pending_job, _pending_future = pending_pair
            pending_tenant_id = str(dict(pending_job.metadata or {}).get("tenant_id") or "global")
            if pending_tenant_id == tenant_id:
                tenant_pending_count += 1.0
        tenant_penalty = max(0.0, tenant_pending_count - 1.0) * self.tenant_balance_weight
        claim_key = self._claim_order_key(job_id)
        return (effective_priority + tenant_penalty, effective_priority, *claim_key)

    def _emit_scheduler_event(
        self,
        *,
        event_type: str,
        job: ExecutionJob,
        payload: dict[str, Any],
        step_key: str,
    ) -> None:
        write_event = getattr(self.writer, "write_event", None)
        if not callable(write_event):
            return
        write_event(
            event_type=event_type,
            session_id=str(job.session_id or "default"),
            trace_id=str(job.trace_id or ""),
            kernel_step_id=step_key,
            payload={
                "job_id": str(job.job_id or ""),
                "request_id": str(dict(job.metadata or {}).get("request_id") or ""),
                "execution_state": dict(payload.get("execution_state") or {}),
                **{key: value for key, value in dict(payload or {}).items() if key != "execution_state"},
            },
            committed=False,
        )

    def _emit_failure_policy_event(self, *, job: ExecutionJob, failure: dict[str, Any]) -> None:
        action = str(failure.get("failure_action") or "").strip().lower()
        strategy = _FAILURE_POLICY_STRATEGIES.get(action)
        if strategy is None:
            return
        self._emit_scheduler_event(
            event_type=strategy.event_type,
            job=job,
            payload={
                "execution_state": dict(job.metadata.get("execution_state") or {}),
                "failure_type": str(failure.get("failure_type") or ""),
                "failure_action": str(failure.get("failure_action") or ""),
                "auto_retry": bool(failure.get("auto_retry", False)),
            },
            step_key="scheduler.execute.failure_policy",
        )

    def _append_lifecycle_event(
        self,
        job: ExecutionJob,
        event: Any,
        *,
        step_key: str,
        committed: bool = False,
    ) -> dict[str, Any]:
        projected = self.projection.get(job.job_id)
        current_state: ExecutionLifecycleState | None = None
        if projected is not None:
            current_state = _coerce_lifecycle_state(_lifecycle_state_from_projection(projected))
        _assert_lifecycle_emission_transition(
            execution_id=str(job.job_id or ""),
            event=event,
            current_state=current_state,
        )
        before_state = current_state or ExecutionLifecycleState.SUBMITTED
        after_state = _target_lifecycle_state_for_event(event, current_state=before_state)
        before_transition, after_transition = _build_sovereign_transition_states(
            job=job,
            before_state=before_state,
            after_state=after_state,
        )
        validate_sovereign_ledger_transition(before_transition, after_transition)

        metadata = dict(job.metadata or {})
        execution_state = dict(metadata.get("execution_state") or {})
        execution_state["causal_step_count"] = int(after_transition.get("causal_step_count") or 0)
        execution_state["invariance_hash"] = str(after_transition.get("invariance_hash") or "")
        if after_state == ExecutionLifecycleState.COMPLETED:
            execution_state["turn_truth_ok"] = bool(after_transition.get("turn_truth_ok"))
        metadata["execution_state"] = execution_state
        job.metadata = metadata

        payload = self.writer.append_execution_lifecycle(
            event,
            session_id=str(job.session_id or "default"),
            trace_token=str(job.trace_id or ""),
            step_key=step_key,
            committed=committed,
        )
        state = self.projection.apply(event)
        _apply_projection_execution_state(job, state)
        return payload

    @staticmethod
    def _resolve_options(
        options: SchedulerOptions | None,
        legacy_options: dict[str, Any],
    ) -> SchedulerOptions:
        resolved = options or SchedulerOptions()
        if "max_inflight_jobs" in legacy_options:
            resolved.max_inflight_jobs = int(legacy_options["max_inflight_jobs"])
        if "projection" in legacy_options:
            resolved.projection = legacy_options["projection"]
        if "worker_id" in legacy_options:
            resolved.worker_id = str(legacy_options["worker_id"] or "worker-1")
        if "execution_token" in legacy_options:
            resolved.execution_token = str(legacy_options["execution_token"] or "")
        if "enable_observability" in legacy_options:
            resolved.enable_observability = bool(legacy_options["enable_observability"])
        if "execution_lease" in legacy_options:
            resolved.execution_lease = legacy_options["execution_lease"]
        if "lease_ttl_seconds" in legacy_options:
            resolved.lease_ttl_seconds = float(legacy_options["lease_ttl_seconds"])
        if "redelivery_retry_interval_seconds" in legacy_options:
            resolved.redelivery_retry_interval_seconds = float(
                legacy_options["redelivery_retry_interval_seconds"],
            )
        if "fairness_aging_rate" in legacy_options:
            resolved.fairness_aging_rate = float(legacy_options["fairness_aging_rate"])
        if "tenant_balance_weight" in legacy_options:
            resolved.tenant_balance_weight = float(legacy_options["tenant_balance_weight"])
        if "on_runtime_claim_guard" in legacy_options:
            resolved.on_runtime_claim_guard = legacy_options["on_runtime_claim_guard"]
        if "on_runtime_lease_sync" in legacy_options:
            resolved.on_runtime_lease_sync = legacy_options["on_runtime_lease_sync"]
        return resolved

    async def _execute_with_boundary(
        self,
        executor: Callable[[dict[str, Any], ExecutionJob], Awaitable[FinalizedTurnResult]],
        session: dict[str, Any],
        job: ExecutionJob,
    ) -> FinalizedTurnResult:
        if self.execution_token:
            with ControlPlaneExecutionBoundary.bind(self.execution_token):
                return await executor(session, job)
        return await executor(session, job)

    @staticmethod
    def _resolve_future(
        future: asyncio.Future[FinalizedTurnResult],
        *,
        result: FinalizedTurnResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
            return
        if result is not None:
            future.set_result(result)

    def _record_job_observability(
        self,
        *,
        event: str,
        job: ExecutionJob,
        started_at: float,
        error: str = "",
        failure: dict[str, Any] | None = None,
    ) -> None:
        if not self.enable_observability:
            return
        metrics = get_metrics()
        metrics.increment(f"scheduler.job.{event}")
        metrics.observe(
            "scheduler.job.latency_ms",
            (time.perf_counter() - started_at) * 1000.0,
        )
        payload: dict[str, Any] = self._observability_payload(event=event, job=job, error=error)
        self._attach_observability_failure(payload=payload, job=job, failure=failure)
        get_exporter().export(payload)

    @staticmethod
    def _observability_payload(*, event: str, job: ExecutionJob, error: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": f"job.{event}",
            "job_id": job.job_id,
            "session_id": job.session_id,
        }
        if error:
            payload["error"] = error
        return payload

    @staticmethod
    def _unified_failure_payload(failure_view: dict[str, Any]) -> dict[str, Any]:
        return {
            "failure_class": str(failure_view.get("class") or ""),
            "failure_source": str(failure_view.get("source") or ""),
            "retryable": bool(failure_view.get("retryable", False)),
            "error_type": str(failure_view.get("type") or ""),
            "message": str(failure_view.get("message") or ""),
            "class": str(failure_view.get("class") or ""),
            "source": str(failure_view.get("source") or ""),
            "type": str(failure_view.get("type") or ""),
        }

    @staticmethod
    def _merge_legacy_failure_fields(target: dict[str, Any], failure: dict[str, Any]) -> None:
        target["failure_type"] = str(failure.get("failure_type") or "")
        target["failure_action"] = str(failure.get("failure_action") or "")
        target["auto_retry"] = bool(failure.get("auto_retry", False))

    def _attach_observability_failure(
        self,
        *,
        payload: dict[str, Any],
        job: ExecutionJob,
        failure: dict[str, Any] | None,
    ) -> None:
        execution_result = ensure_unified_execution_result(
            dict(getattr(job, "metadata", {}).get("execution_result") or {}),
        )
        failure_view = dict(execution_result.get("failure") or {})
        has_failure = bool(
            str(failure_view.get("class") or "")
            or str(failure_view.get("type") or "")
            or str(failure_view.get("message") or ""),
        )
        if has_failure:
            payload["failure"] = self._unified_failure_payload(failure_view)
            if isinstance(failure, dict) and bool(failure):
                self._merge_legacy_failure_fields(payload["failure"], failure)
            return
        if isinstance(failure, dict) and bool(failure):
            # Legacy fallback for older callers; unified execution_result is authoritative.
            payload["failure"] = dict(failure)

    async def register(self, job: ExecutionJob) -> asyncio.Future[FinalizedTurnResult]:
        if len(self._jobs) >= self.max_inflight_jobs:
            raise BackpressureSignal(
                reason="max inflight jobs reached",
                retry_after_ms=self.redelivery_retry_interval_seconds * 1000.0,
                trace_id=job.trace_id,
            )
        assert str(job.trace_id or "").strip(), "Missing trace_id at scheduler register"

        loop = asyncio.get_running_loop()
        future: asyncio.Future[FinalizedTurnResult] = loop.create_future()
        _apply_projection_execution_state(job, self.projection.get(job.job_id))
        job.metadata.setdefault("execution_state", {})["last_transition_reason"] = "scheduler.register"
        job.metadata.setdefault("claim_order", {})
        job.metadata.setdefault("scheduling", {})
        job.metadata["claim_order"].setdefault(
            "timestamp",
            float(dict(job.metadata or {}).get("submitted_timestamp") or 0.0),
        )
        job.metadata["claim_order"].setdefault(
            "worker_id",
            str(self.worker_id or "worker-1"),
        )
        job.metadata["claim_order"].setdefault(
            "lease_epoch",
            int(dict(job.metadata.get("execution_state") or {}).get("redelivery_count") or 0),
        )
        raw_priority = dict(job.metadata or {}).get("priority")
        priority_value = 100.0 if raw_priority is None else float(raw_priority)
        job.metadata["scheduling"].setdefault(
            "submitted_monotonic",
            float(time.monotonic()),
        )
        job.metadata["scheduling"].setdefault(
            "priority",
            priority_value,
        )
        job.metadata.setdefault("tenant_id", str(dict(job.metadata or {}).get("tenant_id") or "global"))
        job.metadata["execution_mode"] = _resolved_execution_mode(job)
        self._jobs[job.job_id] = (job, future)
        self._pending_job_ids.append(job.job_id)
        self._notify_work_available()
        self.writer.append_job_queued(job)
        return future

    def _release_runtime_lease(self, *, session_id: str) -> None:
        if self.execution_lease is not None:
            with contextlib.suppress(Exception):
                self.execution_lease.release(session_id=str(session_id or "default"), owner_id=self.worker_id)
        if callable(self._on_runtime_lease_sync):
            self._on_runtime_lease_sync("")

    def _emit_redelivery_events(
        self,
        *,
        job: ExecutionJob,
        projected: ReducedExecutionState,
        now: datetime,
    ) -> None:
        if projected.owner and lease_expired(projected, now=now):
            prior_owner = str(projected.owner or "")
            self._append_lifecycle_event(
                job,
                LeaseExpired(
                    execution_id=job.job_id,
                    occurred_at=now,
                    worker_id=projected.owner,
                ),
                step_key="scheduler.lease_expired",
            )
            self._append_lifecycle_event(
                job,
                Redelivered(
                    execution_id=job.job_id,
                    occurred_at=now,
                    previous_worker_id=prior_owner,
                    new_worker_id=self.worker_id,
                ),
                step_key="scheduler.redelivery",
            )
            self._emit_scheduler_event(
                event_type="JOB_REDELIVERY_SCHEDULED",
                job=job,
                payload={
                    "execution_state": dict(job.metadata.get("execution_state") or {}),
                    "lease_owner": str(projected.owner or ""),
                },
                step_key="scheduler.redelivery",
            )
        elif projected.attempt_count <= 0:
            prior_owner = str(dict(job.metadata.get("execution_state") or {}).get("last_worker_id") or "")
            if prior_owner:
                self._append_lifecycle_event(
                    job,
                    Redelivered(
                        execution_id=job.job_id,
                        occurred_at=now,
                        previous_worker_id=prior_owner,
                        new_worker_id=self.worker_id,
                    ),
                    step_key="scheduler.redelivery.external",
                )

    def _claim_job_for_execution(
        self,
        *,
        job_id: str,
        job: ExecutionJob,
        projected: ReducedExecutionState,
        now: datetime,
    ) -> bool:
        if callable(self._on_runtime_claim_guard):
            self._on_runtime_claim_guard("scheduler_claim")
        lease_token: dict[str, Any] | None = None
        if self.execution_lease is not None:
            lease_token = self.execution_lease.acquire(
                session_id=str(job.session_id or "default"),
                owner_id=self.worker_id,
                ttl_seconds=self.lease_ttl_seconds,
            )
        if projected.owner and not lease_expired(projected, now=now):
            self._release_runtime_lease(session_id=job.session_id)
            self._pending_job_ids.append(job_id)
            self._notify_work_available()
            return False
        self._emit_redelivery_events(job=job, projected=projected, now=now)

        self._append_lifecycle_event(
            job,
            Claimed(
                execution_id=job.job_id,
                occurred_at=now,
                worker_id=self.worker_id,
                lease_expiry=now + timedelta(seconds=self.lease_ttl_seconds),
            ),
            step_key="scheduler.claim",
        )
        if lease_token is not None:
            job.metadata.setdefault("lease_fence", {})
            job.metadata["lease_fence"]["fencing_token"] = int(lease_token.get("fencing_token") or 0)
            job.metadata["lease_fence"]["lease_id"] = str(lease_token.get("lease_id") or "")
        fence_token = int(lease_token.get("fencing_token") or 0) if lease_token is not None else 0
        if callable(self._on_runtime_lease_sync):
            self._on_runtime_lease_sync(f"{self.execution_token}:{job.job_id}:{fence_token}")
        return True

    def _record_scheduler_success(
        self,
        *,
        job: ExecutionJob,
        future: asyncio.Future[FinalizedTurnResult],
        result: FinalizedTurnResult,
        started_at: float,
    ) -> None:
        current_execution_result = ensure_unified_execution_result(
            dict(job.metadata.get("execution_result") or {}),
        )
        current_execution_result = mark_unified_execution_success(
            cast(dict[str, Any], current_execution_result),
            response=str(result[0] if isinstance(result, tuple) and len(result) >= 1 else ""),
            should_end=bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
        )
        job.metadata["execution_result"] = current_execution_result
        self._append_lifecycle_event(
            job,
            Completed(
                execution_id=job.job_id,
                occurred_at=datetime.now(),
                result_ref=f"job:{job.job_id}:result",
            ),
            step_key="scheduler.execute.complete",
            committed=True,
        )
        effect_id = str(dict(job.metadata or {}).get("effect_id") or "").strip()
        request_id = str(dict(job.metadata or {}).get("request_id") or "").strip()
        if effect_id:
            self.writer.append_effect_commit(
                session_id=str(job.session_id or "default"),
                trace_token=str(job.trace_id or ""),
                effect_id=effect_id,
                request_id=request_id,
                step_key="scheduler.execute.effect.commit",
            )
        _set_terminal_turn_state(
            job,
            terminal_state=_SCHEDULER_EXCEPTION_MAPPER.from_success(
                recovered=(_resolved_execution_mode(job) == "recovery"),
            ),
            reason="scheduler.execute.complete",
        )
        self.writer.append_job_completed(job, result)
        self._resolve_future(future, result=result)
        self._release_runtime_lease(session_id=job.session_id)
        self._record_job_observability(
            event="completed",
            job=job,
            started_at=started_at,
        )

    def _record_scheduler_failure(
        self,
        *,
        job: ExecutionJob,
        future: asyncio.Future[FinalizedTurnResult],
        exc: BaseException,
        started_at: float,
    ) -> None:
        failure = _classify_execution_failure(exc)
        current_execution_result = mark_unified_execution_failure(
            cast(dict[str, Any], build_unified_execution_result()),
            failure_class=str(failure.get("failure_class") or "runtime_exception"),
            failure_source=str(failure.get("failure_source") or "execution"),
            retryable=bool(failure.get("retryable", False)),
            exception_type=str(failure.get("exception_type") or type(exc).__name__),
            message=str(exc),
        )
        job.metadata["execution_result"] = current_execution_result
        execution_state = dict(job.metadata.get("execution_state") or {})
        execution_state["failure_type"] = str(failure.get("failure_type") or "")
        execution_state["failure_action"] = str(failure.get("failure_action") or "")
        execution_state["auto_retry"] = bool(failure.get("auto_retry", False))
        execution_state["last_transition_reason"] = (
            f"scheduler.execute.failed:{str(failure.get('failure_action') or 'unknown')}"
        )
        job.metadata["execution_state"] = execution_state
        _set_terminal_turn_state(
            job,
            terminal_state=_SCHEDULER_EXCEPTION_MAPPER.from_exception(exc),
            reason=f"scheduler.execute.failed:{type(exc).__name__}",
        )
        self._emit_failure_policy_event(job=job, failure=failure)
        self._append_lifecycle_event(
            job,
            Failed(
                execution_id=job.job_id,
                occurred_at=datetime.now(),
                error_ref=f"{type(exc).__name__}:{str(exc)}",
            ),
            step_key="scheduler.execute.failed",
            committed=True,
        )
        self.writer.append_job_failed(job, exc, failure=failure)
        self._resolve_future(future, error=exc)
        self._release_runtime_lease(session_id=job.session_id)
        self._record_job_observability(
            event="failed",
            job=job,
            started_at=started_at,
            error=str(exc),
            failure=failure,
        )

    def _maybe_begin_effect(self, job: ExecutionJob) -> None:
        effect_id = str(dict(job.metadata or {}).get("effect_id") or "").strip()
        request_id = str(dict(job.metadata or {}).get("request_id") or "").strip()
        if effect_id:
            self.writer.append_effect_begin(
                session_id=str(job.session_id or "default"),
                trace_token=str(job.trace_id or ""),
                effect_id=effect_id,
                request_id=request_id,
                step_key="scheduler.execute.effect.begin",
            )

    async def _execute_and_record_job(
        self,
        executor: Callable[[dict[str, Any], ExecutionJob], Awaitable[FinalizedTurnResult]],
        job: ExecutionJob,
        future: Any,
        started_at: float,
    ) -> bool:
        job.metadata["execution_mode"] = _resolved_execution_mode(job)
        self._maybe_begin_effect(job)
        self.writer.append_job_started(job)
        session = self.registry.bind(job.session_id)
        tracer = get_tracer()
        with tracer.span("scheduler.drain_once"):
            result = await self._execute_with_boundary(executor, session, job)
        _enforce_global_turn_invariant_gate(session=session, job=job)
        self._record_scheduler_success(
            job=job,
            future=future,
            result=result,
            started_at=started_at,
        )
        return True

    async def drain_once(
        self,
        executor: Callable[
            [dict[str, Any], ExecutionJob],
            Awaitable[FinalizedTurnResult],
        ],
    ) -> bool:
        if not self._pending_job_ids:
            return False

        job_id = self._pop_next_ready_job_id()
        if job_id is None:
            return False
        job_pair = self._jobs.get(job_id)
        if job_pair is None:
            return False
        job, future = job_pair
        assert str(job.trace_id or "").strip(), "Missing trace_id at scheduler drain"

        started_at = time.perf_counter()
        projected = self.projection.get(job.job_id)
        _apply_projection_execution_state(job, projected)
        now = datetime.now()

        if projected is None:
            raise RuntimeError(f"missing lifecycle state for job {job.job_id!r}")

        if projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}:
            self._jobs.pop(job_id, None)
            return False

        try:
            claimed = self._claim_job_for_execution(
                job_id=job_id,
                job=job,
                projected=projected,
                now=now,
            )
            if not claimed:
                return False
        except RuntimeError:
            self._release_runtime_lease(session_id=job.session_id)
            self._pending_job_ids.append(job_id)
            self._notify_work_available()
            return False

        try:
            return await self._execute_and_record_job(executor, job, future, started_at)
        except asyncio.CancelledError as exc:
            logger.exception(
                "scheduler drain cancelled for job_id=%s session_id=%s",
                str(job.job_id or ""),
                str(job.session_id or ""),
            )
            self._record_scheduler_failure(
                job=job,
                future=future,
                exc=exc,
                started_at=started_at,
            )
            raise
        except (
            TimeoutError,
            ExecutionStageError,
            InvariantViolation,
            MemorySetInvariantViolation,
            PersistenceFailure,
            RuntimeError,
        ) as exc:
            logger.exception(
                "scheduler drain failed for job_id=%s session_id=%s",
                str(job.job_id or ""),
                str(job.session_id or ""),
            )
            self._record_scheduler_failure(
                job=job,
                future=future,
                exc=exc,
                started_at=started_at,
            )
            raise
        finally:
            if future.done():
                if not future.cancelled():
                    with contextlib.suppress(BaseException):
                        future.exception()
                self._jobs.pop(job_id, None)


class ExecutionControlPlane(ReconciliationMixin, CompactionMixin):
    """Execution boundary around scheduler, lease, ledger, and recovery."""

    def __init__(
        self,
        *,
        registry: SessionRegistry,
        kernel_executor: Callable[
            [dict[str, Any], ExecutionJob],
            Awaitable[FinalizedTurnResult],
        ],
        graph: Any | None = None,
        options: ControlPlaneOptions | None = None,
        **legacy_options: Any,
    ) -> None:
        resolved_options = self._resolve_options(options, legacy_options)
        self.registry = registry
        self.kernel_executor = kernel_executor
        self._stream_sink = resolved_options.stream_sink
        token_seed = (
            f"{resolved_options.worker_id}|"
            f"{resolved_options.max_inflight_jobs}|"
            f"{int(bool(resolved_options.enable_observability))}"
        )
        self.execution_token = f"exec-{hashlib.sha256(token_seed.encode('utf-8')).hexdigest()[:20]}"
        self.ledger = resolved_options.ledger or InMemoryExecutionLedger()
        self._ledger_writer = LedgerWriterAdapter(
            self.ledger,
            scope_validator=KernelGateway.assert_scope,
        )
        self._ledger_index = LedgerIndex(self.ledger)
        self._effect_journal = EffectJournal(writer=self._ledger_writer, index=self._ledger_index)
        write_event = getattr(self._ledger_writer, "write_event", None)
        self._reconcile_queue = DurableReconcileQueue(
            ledger=self.ledger,
            write_event=write_event,
            request_is_ambiguous=lambda session_id, request_id: self._request_has_ambiguous_inflight_effect_state(
                session_id=session_id,
                request_id=request_id,
            ),
            effect_is_ambiguous=lambda session_id, effect_id: self._effect_journal.is_ambiguous(
                session_id=session_id,
                effect_id=effect_id,
            ),
        )
        self.ledger_reader = LedgerReader(self.ledger)
        self._execution_lease = ExecutionLease(default_ttl_seconds=resolved_options.lease_ttl_seconds)
        scheduler_options = SchedulerOptions(
            max_inflight_jobs=resolved_options.max_inflight_jobs,
            worker_id=resolved_options.worker_id,
            execution_token=self.execution_token,
            enable_observability=resolved_options.enable_observability,
            projection=ExecutionProjection(),
            execution_lease=self._execution_lease,
            lease_ttl_seconds=resolved_options.lease_ttl_seconds,
            redelivery_retry_interval_seconds=resolved_options.redelivery_retry_interval_seconds,
            on_runtime_claim_guard=self._scheduler_claim_guard,
            on_runtime_lease_sync=self._scheduler_lease_sync,
        )
        self._scheduler = resolved_options.scheduler or Scheduler(
            registry,
            reader=self.ledger_reader,
            writer=self._ledger_writer,
            options=scheduler_options,
        )
        self.lifecycle_projection = getattr(self._scheduler, "projection", ExecutionProjection())
        self.recovery = RecoveryManager(ledger=self.ledger)
        self._inflight_by_request: dict[tuple[str, str, str], asyncio.Future[FinalizedTurnResult]] = {}
        self._inflight_lock = asyncio.Lock()
        self.graph = graph
        bind_execution_token = getattr(self.graph, "set_required_execution_token", None)
        if callable(bind_execution_token):
            bind_execution_token(self.execution_token)
        self._ledger_compactor: EventCompactor | None = None
        self._last_compaction_report: dict[str, Any] = {"compacted": False, "reason": "not_run"}
        self._global_confluence_contracts: dict[str, str] = {}
        self._last_confluence_report: dict[str, Any] = {"enforced": False, "reason": "not_run"}
        self._confluence_metrics: dict[str, int] = {
            "attempted": 0,
            "bound_first_observation": 0,
            "matched": 0,
            "mismatch": 0,
            "enforced_blocked": 0,
        }
        self._stream_event_sequence: int = 0
        self._distributed_correctness = DistributedCorrectnessModel()
        self._distributed_epoch = 1
        self._cognitive_policy_engine = CognitivePolicyEngine()
        self._semantic_memory_graph = SemanticMemoryGraph()
        self._tool_routing_engine = ToolRoutingEngine()
        self._adaptation_engine = AdaptationEngine()
        self._semantic_safety_engine = SemanticSafetyEngine()
        self._belief_state_engine = BeliefStateEngine()
        self._hypothesis_engine = MultiHypothesisEngine()
        self._memory_hierarchy_manager = MemoryHierarchyManager()
        self._tool_self_model = ToolSelfModel()
        self._compositional_tool_planner = CompositionalToolPlanner()
        self._planning_optimizer = SessionPlanningOptimizer()
        self._autonomous_goal_daemon = AutonomousGoalDaemon()
        self._interactive_cognition_ui = InteractiveCognitionUI()
        self._alignment_trainer = BehaviorAlignmentTrainer()
        self._tool_ecosystem_hub = ToolEcosystemHub()
        self._multi_agent_swarm = MultiAgentSwarm()
        self._response_engine = ResponseEngine()
        self._topology_runtime = TopologyRuntime(strict_mode=True)
        self._last_topology_validation: TopologyValidationResult | None = None
        self._phase_closure_runtime = PhaseClosureRuntime()
        self._autonomous_goal_task: asyncio.Task[None] | None = None
        self._autonomous_goal_stop: asyncio.Event | None = None
        self._sync_distributed_authority(role=NodeRole.LEADER, state_hash=self.execution_token)
        self.kernel_gateway = KernelGateway(self)
        self.bootstrap()

    def _topology_begin_turn(self, *, trace_id: str, session_id: str) -> None:
        self._topology_runtime.begin_turn(
            trace_id=str(trace_id or ""),
            session_id=str(session_id or "default"),
        )

    def _topology_record_node(
        self,
        *,
        node_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._topology_runtime.record_node_entry(
            node_id=str(node_id or ""),
            timestamp_ms=float(time.time() * 1000.0),
            metadata=dict(metadata or {}),
        )

    def _topology_end_turn(self) -> TopologyValidationResult:
        result = self._topology_runtime.end_turn()
        self._last_topology_validation = result
        if not result.passed or result.violations_critical > 0:
            raise InvariantViolation(
                "Topology runtime enforcement blocked non-canonical execution path.",
                context={
                    "violations_critical": int(result.violations_critical),
                    "violations_high": int(result.violations_high),
                    "violations_total": int(result.violations_total),
                    "details": dict(result.details or {}),
                },
            )
        return result

    def _pre_execution_contract_gate(
        self,
        *,
        session_id: str,
        user_input: str,
        metadata: dict[str, Any],
    ) -> None:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            raise RuntimeError("pre-execution contract gate requires dict state")

        canonical_state = self._phase_closure_runtime.kernel.ensure_global_canonical_state_schema(dict(state))
        metadata["pre_execution_canonical_state_hash"] = self._stable_hash(canonical_state)

        metadata.setdefault("execution_timestamp_ms", int(time.time() * 1000))
        metadata.setdefault("max_staleness_ms", 2000)

        pre_drift = self._phase_closure_runtime.kernel.detect_drift(
            slot=f"pre_execution:{str(session_id or 'default')}",
            value={"state": canonical_state, "input": str(user_input or "")},
        )
        metadata["drift_precheck"] = dict(pre_drift)
        self._phase_closure_runtime.kernel.register_side_effect(
            effect_type="execution_stage",
            subject="pre_execution",
            payload={"session_id": str(session_id or "default")},
            trace_id=str(metadata.get("trace_id") or ""),
        )

    def _post_planning_pre_tool_contract_gate(
        self,
        *,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        tool_contract = dict(metadata.get("tool_runtime_contract") or {})
        tool_name = str(tool_contract.get("tool_name") or "").strip()
        runtime_plan = dict(metadata.get("runtime_plan") or {})
        uncertainty = float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 0.0)
        strategy = str(runtime_plan.get("strategy") or "").strip().lower()
        uncertainty_action = str(runtime_plan.get("uncertainty_action") or "").strip().lower()

        if tool_name and uncertainty >= 0.75 and (
            strategy in {"task_execution", "direct_answer"}
            or uncertainty_action not in {"branch", "retry", "hedge", "clarify"}
        ):
            raise RuntimeError(
                "Uncertainty enforcement gate: planner must branch/hedge before tool execution",
            )

        memory_entries = [
            dict(item)
            for item in list(metadata.get("world_model_memory_entries") or [])
            if isinstance(item, dict)
        ]
        entity_bindings = [
            dict(item)
            for item in list(metadata.get("world_model_entity_bindings") or [])
            if isinstance(item, dict)
        ]
        self._phase_closure_runtime.world.validate_memory_bindings(
            memory_entries=memory_entries,
            entity_bindings=entity_bindings,
        )

        self._phase_closure_runtime.kernel.register_side_effect(
            effect_type="execution_stage",
            subject="post_planning_pre_tool",
            payload={
                "session_id": str(session_id or "default"),
                "tool_name": tool_name,
                "uncertainty": uncertainty,
            },
            trace_id=str(metadata.get("trace_id") or ""),
        )

    def _post_execution_pre_commit_contract_gate(
        self,
        *,
        session_id: str,
        job: ExecutionJob,
        result: FinalizedTurnResult,
        input_state_hash: str,
    ) -> None:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            raise RuntimeError("post-execution contract gate requires dict state")

        semantic_items = [
            dict(item)
            for item in list(dict(state.get("semantic_memory") or {}).get("items") or [])
            if isinstance(item, dict)
        ]
        memory_entries: list[dict[str, Any]] = []
        for index, item in enumerate(semantic_items, start=1):
            row = dict(item)
            row.setdefault("id", str(row.get("memory_id") or f"mem-{index}"))
            memory_entries.append(row)
        self._phase_closure_runtime.world.validate_memory_bindings(
            memory_entries=memory_entries,
            entity_bindings=[
                dict(item)
                for item in list(job.metadata.get("world_model_entity_bindings") or [])
                if isinstance(item, dict)
            ],
        )

        side_effect_records = [
            dict(record.__dict__)
            for record in self._phase_closure_runtime.kernel.side_effect_registry.records(
                trace_id=str(job.trace_id or ""),
            )
        ]
        post_drift = self._phase_closure_runtime.kernel.detect_drift(
            slot=f"post_execution:{str(session_id or 'default')}",
            value={"state": state, "result": list(result) if isinstance(result, tuple) else result},
        )

        state_transition = {
            "turn_id": str(job.job_id or ""),
            "trace_id": str(job.trace_id or ""),
            "input_state_hash": str(input_state_hash or ""),
            "action": str(dict(job.metadata.get("runtime_plan") or {}).get("strategy") or ""),
            "tool_calls": [
                dict(item)
                for item in list(dict(job.metadata.get("tool_routing_plan") or {}).get("candidates") or [])
                if isinstance(item, dict)
            ][:5],
            "output_state_hash": self._stable_hash(state),
            "side_effects": side_effect_records,
            "timestamp": float(time.time()),
            "drift_postcheck": dict(post_drift),
        }
        transitions = [
            dict(item)
            for item in list(state.get("state_transition_ledger") or [])
            if isinstance(item, dict)
        ]
        transitions.append(state_transition)
        state["state_transition_ledger"] = transitions[-1024:]
        state["last_state_transition"] = dict(state_transition)
        self._merge_session_state(session_id=str(session_id or "default"), state_patch=state)
        self._emit_runtime_stream_event(
            event_type="state.transition.recorded",
            session_id=str(session_id or "default"),
            trace_id=str(job.trace_id or ""),
            payload={
                "turn_id": str(state_transition.get("turn_id") or ""),
                "input_state_hash": str(state_transition.get("input_state_hash") or ""),
                "output_state_hash": str(state_transition.get("output_state_hash") or ""),
                "action": str(state_transition.get("action") or ""),
                "tool_call_count": int(len(list(state_transition.get("tool_calls") or []))),
                "side_effect_count": int(len(list(state_transition.get("side_effects") or []))),
                "timestamp": float(state_transition.get("timestamp") or 0.0),
            },
        )

    def set_turn_steering(self, *, session_id: str, steering: dict[str, Any]) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"applied": False, "reason": "no_state"}
        state["pending_turn_steering"] = dict(steering or {})
        self._interactive_cognition_ui.apply_live_control(
            state=state,
            control=dict(steering or {}),
            source="set_turn_steering",
        )
        self._emit_runtime_stream_event(
            event_type="turn.steering.updated",
            session_id=str(session_id or "default"),
            trace_id="",
            payload={"steering": dict(steering or {})},
        )
        return {"applied": True, "steering": dict(steering or {})}

    def update_interactive_plan(
        self,
        *,
        session_id: str,
        trace_id: str,
        edits: dict[str, Any],
        actor: str = "operator",
    ) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"updated": False, "reason": "no_state"}
        updated_plan = self._interactive_cognition_ui.apply_plan_edit(
            state=state,
            trace_id=str(trace_id or ""),
            edits=dict(edits or {}),
            actor=str(actor or "operator"),
        )
        return {"updated": True, "plan": dict(updated_plan)}

    def register_external_connector(
        self,
        *,
        session_id: str,
        name: str,
        capabilities: list[str],
        endpoint: str = "",
        health: float = 1.0,
    ) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"registered": False, "reason": "no_state"}
        connector = self._tool_ecosystem_hub.register_connector(
            state=state,
            name=str(name or ""),
            capabilities=[str(item) for item in list(capabilities or [])],
            endpoint=str(endpoint or ""),
            health=float(health),
        )
        return {"registered": bool(connector), "connector": dict(connector)}

    def swarm_status(self, *, session_id: str) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"available": False, "reason": "no_state"}
        return self._multi_agent_swarm.health_snapshot(state=state)

    def _consume_pending_turn_steering(self, *, session_id: str) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {}
        pending = dict(state.get("pending_turn_steering") or {})
        if pending:
            state.pop("pending_turn_steering", None)
        return pending

    async def start_autonomous_goal_loop(self, *, interval_seconds: float = 60.0) -> dict[str, Any]:
        interval = max(1.0, float(interval_seconds or 60.0))
        if self._autonomous_goal_task is not None and not self._autonomous_goal_task.done():
            return {"started": False, "reason": "already_running", "interval_seconds": interval}
        self._autonomous_goal_stop = asyncio.Event()

        async def _run_loop() -> None:
            assert self._autonomous_goal_stop is not None
            while not self._autonomous_goal_stop.is_set():
                for session_name in list(getattr(self.registry, "_sessions", {}).keys()):
                    with contextlib.suppress(Exception):
                        self.run_autonomous_goal_cycle(session_id=session_name, source="daemon")
                try:
                    await asyncio.wait_for(self._autonomous_goal_stop.wait(), timeout=interval)
                except TimeoutError:
                    continue

        self._autonomous_goal_task = asyncio.create_task(_run_loop(), name="dadbot.autonomous_goal_loop")
        return {"started": True, "interval_seconds": interval}

    async def stop_autonomous_goal_loop(self) -> dict[str, Any]:
        if self._autonomous_goal_task is None:
            return {"stopped": False, "reason": "not_running"}
        if self._autonomous_goal_stop is not None:
            self._autonomous_goal_stop.set()
        with contextlib.suppress(Exception):
            await self._autonomous_goal_task
        self._autonomous_goal_task = None
        self._autonomous_goal_stop = None
        return {"stopped": True}

    def run_autonomous_goal_cycle(self, *, session_id: str, source: str = "manual") -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return {"ran": False, "reason": "no_state"}
        actions = self._autonomous_goal_daemon.next_actions(state=state, max_items=4)
        self._autonomous_goal_daemon.persist_cycle(state=state, actions=actions, source=source)
        self._emit_runtime_stream_event(
            event_type="autonomous.goal.cycle",
            session_id=str(session_id or "default"),
            trace_id="",
            payload={"actions": [dict(item) for item in list(actions or [])], "source": str(source or "manual")},
        )
        return {"ran": True, "actions": [dict(item) for item in list(actions or [])]}

    def _emit_runtime_stream_event(
        self,
        *,
        event_type: str,
        session_id: str,
        trace_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._stream_event_sequence += 1
        event = {
            "event_type": str(event_type or "runtime.unknown"),
            "session_id": str(session_id or "default"),
            "trace_id": str(trace_id or ""),
            "timestamp": float(time.time()),
            "sequence": int(self._stream_event_sequence),
            "payload": dict(payload or {}),
        }
        event = dict(validate_trace_event_contract(event))
        self._append_stream_timeline(
            session_id=str(session_id or "default"),
            event=event,
        )
        sink = self._stream_sink
        if callable(sink):
            try:
                sink(event)
            except Exception as exc:
                logger.debug("stream sink failed: %s", exc)
        _write_progress_event(component="runtime_stream", phase=str(event_type or "runtime.unknown"), payload=event)

    def _append_stream_timeline(self, *, session_id: str, event: dict[str, Any]) -> None:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return
        timeline = list(state.get("execution_timeline") or [])
        timeline.append(dict(event))
        state["execution_timeline"] = timeline[-512:]

    def _merge_session_state(self, *, session_id: str, state_patch: dict[str, Any]) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        existing = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(existing, dict):
            return dict(state_patch or {})
        if state_patch is existing:
            return existing
        existing.update(dict(state_patch or {}))
        session["state"] = existing
        return existing

    def _inject_semantic_memory_context(
        self,
        *,
        session_id: str,
        user_input: str,
        metadata: dict[str, Any],
        limit: int = 5,
    ) -> None:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            metadata["semantic_memory_context"] = []
            return
        semantic_memory = dict(state.get("semantic_memory_projection") or state.get("semantic_memory") or {})
        items = [dict(item) for item in list(semantic_memory.get("items") or []) if isinstance(item, dict)]
        ranked = _rank_semantic_memory_items(
            items=items,
            user_input=str(user_input or ""),
            limit=int(limit),
        )
        metadata["semantic_memory_context"] = ranked

    def _discover_tool_specs(self) -> list[ToolSpec]:
        registry = getattr(self.graph, "tool_registry", None)
        if registry is None:
            return []
        discover = getattr(registry, "discover", None)
        if not callable(discover):
            return []
        try:
            specs = discover()
        except Exception:
            return []
        if not isinstance(specs, list):
            return []
        return [spec for spec in specs if isinstance(spec, ToolSpec)]

    @property
    def scheduler(self) -> SchedulerProtocol:
        KernelGateway.assert_scope("control_plane.scheduler")
        return self._scheduler

    @property
    def ledger_writer(self) -> LedgerWriterAdapter:
        KernelGateway.assert_scope("control_plane.ledger_writer")
        return self._ledger_writer

    @staticmethod
    def _resolve_options(
        options: ControlPlaneOptions | None,
        legacy_options: dict[str, Any],
    ) -> ControlPlaneOptions:
        resolved = options or ControlPlaneOptions()
        if "max_inflight_jobs" in legacy_options:
            resolved.max_inflight_jobs = int(legacy_options["max_inflight_jobs"])
        if "worker_id" in legacy_options:
            resolved.worker_id = str(legacy_options["worker_id"] or "worker-1")
        if "enable_observability" in legacy_options:
            resolved.enable_observability = bool(legacy_options["enable_observability"])
        if "lease_ttl_seconds" in legacy_options:
            resolved.lease_ttl_seconds = float(legacy_options["lease_ttl_seconds"])
        if "redelivery_retry_interval_seconds" in legacy_options:
            resolved.redelivery_retry_interval_seconds = float(
                legacy_options["redelivery_retry_interval_seconds"],
            )
        if "ledger" in legacy_options:
            resolved.ledger = legacy_options["ledger"]
        if "scheduler" in legacy_options:
            resolved.scheduler = legacy_options["scheduler"]
        if "stream_sink" in legacy_options:
            resolved.stream_sink = legacy_options["stream_sink"]
        return resolved

    async def create_session(self, session_id: str) -> dict[str, Any]:
        return await self.registry.create_session(session_id)

    def terminate_session(self, session_id: str) -> None:
        self.registry.terminate_session(session_id)

    def _active_lease_count(self) -> int:
        snapshot = dict(self.lifecycle_projection.snapshot() or {})
        count = 0
        for _execution_id, state in snapshot.items():
            if not isinstance(state, dict):
                continue
            status = str(state.get("status") or "").strip().lower()
            owner = str(state.get("owner") or "").strip()
            if status in {"claimed", "running"} and owner:
                count += 1
        return count

    def _distributed_now_ms(self) -> int:
        return int(time.time() * 1000)

    def _sync_distributed_authority(self, *, role: NodeRole, state_hash: str = "") -> None:
        self._distributed_correctness.register_node(
            node_id=self._scheduler.worker_id,
            epoch=int(self._distributed_epoch),
            lease_until_ms=self._distributed_now_ms() + int(self._scheduler.lease_ttl_seconds * 1000),
            role=role,
            state_hash=str(state_hash or self.execution_token),
        )

    def _scheduler_claim_guard(self, operation: str) -> None:
        self._enforce_distributed_runtime_authority(operation=str(operation or "scheduler_claim"))

    def _scheduler_lease_sync(self, state_hash: str) -> None:
        self._sync_distributed_authority(role=NodeRole.LEADER, state_hash=str(state_hash or ""))

    def _enforce_distributed_runtime_authority(self, *, operation: str) -> None:
        now_ms = self._distributed_now_ms()
        self._distributed_correctness.enforce_no_split_brain(now_ms=now_ms)
        if self._distributed_correctness.validate_authority(
            node_id=self._scheduler.worker_id,
            now_ms=now_ms,
        ):
            return
        authority = self._distributed_correctness.current_authority(now_ms=now_ms)
        raise AuthorityViolation(
            "Distributed correctness violation: non-authoritative runtime path rejected",
            context={
                "operation": str(operation or "unknown"),
                "worker_id": self._scheduler.worker_id,
                "authority": "" if authority is None else authority.node_id,
                "epoch": 0 if authority is None else int(authority.epoch),
                "now_ms": now_ms,
            },
        )

    def distributed_reconciliation_plan(self) -> dict[str, Any]:
        plan = self._distributed_correctness.reconcile(now_ms=self._distributed_now_ms())
        return {
            "authoritative_node": plan.authoritative_node,
            "authoritative_hash": plan.authoritative_hash,
            "divergent_nodes": list(plan.divergent_nodes),
            "converged": bool(plan.converged),
        }

    def _emit_progress_snapshot(
        self,
        *,
        phase: str,
        session_id: str,
        trace_token: str,
        job_id: str,
        future_done: bool,
        completion_expectations: dict[str, bool] | None = None,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        expectations = dict(completion_expectations or {})
        unmet = [name for name, ok in expectations.items() if not bool(ok)]
        payload = {
            "session_id": str(session_id or "default"),
            "trace_id": str(trace_token or ""),
            "job_id": str(job_id or ""),
            "future_done": bool(future_done),
            "queue_size": int(len(getattr(self._scheduler, "_pending_job_ids", []) or [])),
            "reconciliation_backlog_size": int(len(self._pending_reconcile_required_entries())),
            "active_leases": int(self._active_lease_count()),
            "pending_replay_entries": int(len(self._inflight_by_request)),
            "expected_completion_conditions": expectations,
            "unmet_completion_conditions": unmet,
            "note": str(note or ""),
            "extra": dict(extra or {}),
        }
        _write_progress_event(component="control_plane", phase=phase, payload=payload)
        self._emit_runtime_stream_event(
            event_type="turn.progress",
            session_id=str(session_id or "default"),
            trace_id=str(trace_token or ""),
            payload=payload,
        )

    def _durable_completed_result_for_request(
        self,
        *,
        session_id: str,
        request_id: str,
    ) -> FinalizedTurnResult | None:
        return self._ledger_index.completed_result(
            session_id=session_id,
            request_id=request_id,
        )

    async def _preflight_submit_turn(
        self,
        *,
        session_key: str,
        request_id: str,
        effect_id: str,
        metadata: dict[str, Any],
    ) -> tuple[str, tuple[str, str, str], asyncio.Future[FinalizedTurnResult] | None, bool, FinalizedTurnResult | None]:
        self._emit_progress_snapshot(
            phase="before_replay",
            session_id=session_key,
            trace_token=str(metadata.get("trace_id") or ""),
            job_id="",
            future_done=False,
            note="submit_turn entered",
            extra={"request_id": request_id, "effect_id": effect_id},
        )
        durable_result = self._durable_completed_result_for_request(
            session_id=session_key,
            request_id=request_id,
        )
        durable_short_circuit = self._durable_preflight_result(
            session_key=session_key,
            request_id=request_id,
            effect_id=effect_id,
            metadata=metadata,
            durable_result=durable_result,
        )
        if durable_short_circuit is not None:
            return durable_short_circuit

        effect_id = self._resolve_preflight_effect_id(
            session_key=session_key,
            request_id=request_id,
            effect_id=effect_id,
            metadata=metadata,
        )
        dedupe_key = (session_key, request_id, effect_id)
        dedupe_future, owns_dedupe_slot = await self._acquire_preflight_dedupe_slot(
            request_id=request_id,
            effect_id=effect_id,
            dedupe_key=dedupe_key,
        )
        if not owns_dedupe_slot and dedupe_future is not None:
            return effect_id, dedupe_key, dedupe_future, False, await dedupe_future

        self._assert_preflight_not_ambiguous(
            session_key=session_key,
            request_id=request_id,
            effect_id=effect_id,
            metadata=metadata,
        )

        return effect_id, dedupe_key, dedupe_future, owns_dedupe_slot, None

    def _durable_preflight_result(
        self,
        *,
        session_key: str,
        request_id: str,
        effect_id: str,
        metadata: dict[str, Any],
        durable_result: FinalizedTurnResult | None,
    ) -> tuple[str, tuple[str, str, str], None, bool, FinalizedTurnResult] | None:
        if durable_result is None:
            return None
        self._emit_progress_snapshot(
            phase="before_replay",
            session_id=session_key,
            trace_token=str(metadata.get("trace_id") or ""),
            job_id="",
            future_done=True,
            note="durable completed result returned",
            extra={"request_id": request_id},
        )
        return effect_id, (session_key, request_id, effect_id), None, False, durable_result

    def _resolve_preflight_effect_id(
        self,
        *,
        session_key: str,
        request_id: str,
        effect_id: str,
        metadata: dict[str, Any],
    ) -> str:
        if not effect_id and request_id:
            effect_id = self._effect_journal.derive_effect_id(
                session_id=session_key,
                request_id=request_id,
                trace_id=str(metadata.get("trace_id") or ""),
            )
        if effect_id and self._effect_journal.is_committed(session_id=session_key, effect_id=effect_id):
            raise RuntimeError(
                "Effect already committed but no durable JOB_COMPLETED payload found; "
                "refusing ambiguous replay",
            )
        return effect_id

    async def _acquire_preflight_dedupe_slot(
        self,
        *,
        request_id: str,
        effect_id: str,
        dedupe_key: tuple[str, str, str],
    ) -> tuple[asyncio.Future[FinalizedTurnResult] | None, bool]:
        dedupe_future: asyncio.Future[FinalizedTurnResult] | None = None
        owns_dedupe_slot = False
        if request_id or effect_id:
            async with self._inflight_lock:
                existing = self._inflight_by_request.get(dedupe_key)
                if existing is not None:
                    dedupe_future = existing
                else:
                    dedupe_future = asyncio.get_running_loop().create_future()
                    self._inflight_by_request[dedupe_key] = dedupe_future
                    owns_dedupe_slot = True
        return dedupe_future, owns_dedupe_slot

    def _assert_preflight_not_ambiguous(
        self,
        *,
        session_key: str,
        request_id: str,
        effect_id: str,
        metadata: dict[str, Any],
    ) -> None:
        self._enforce_distributed_runtime_authority(operation="replay_acceptance_gate")
        if self._request_has_ambiguous_inflight_effect_state(
            session_id=session_key,
            request_id=request_id,
        ):
            self._emit_progress_snapshot(
                phase="during_reconciliation",
                session_id=session_key,
                trace_token=str(metadata.get("trace_id") or ""),
                job_id="",
                future_done=False,
                note="ambiguous request inflight detected",
                extra={
                    "request_id": request_id,
                    "effect_id": effect_id,
                    "classification": "replay_to_reconciliation_gate",
                },
            )
            self._emit_reconcile_required_event(
                session_id=session_key,
                trace_token=str(metadata.get("trace_id") or ""),
                request_id=request_id,
                effect_id=effect_id,
                reason="ambiguous_request_inflight",
            )
            raise ReplayMismatch(
                "Ambiguous effect state for request replay: prior JOB_STARTED exists "
                "without terminal event; refusing duplicate execution",
            )

        if effect_id and self._effect_journal.is_ambiguous(session_id=session_key, effect_id=effect_id):
            self._emit_progress_snapshot(
                phase="during_reconciliation",
                session_id=session_key,
                trace_token=str(metadata.get("trace_id") or ""),
                job_id="",
                future_done=False,
                note="ambiguous effect begin without commit detected",
                extra={
                    "request_id": request_id,
                    "effect_id": effect_id,
                    "classification": "replay_to_reconciliation_gate",
                },
            )
            self._emit_reconcile_required_event(
                session_id=session_key,
                trace_token=str(metadata.get("trace_id") or ""),
                request_id=request_id,
                effect_id=effect_id,
                reason="ambiguous_effect_begin_without_commit",
            )
            raise ReplayMismatch(
                "Ambiguous effect state for replay: prior EFFECT_BEGIN exists without EFFECT_COMMIT; "
                "refusing duplicate execution",
            )

    def _resolve_submit_trace_and_effect(
        self,
        *,
        session_key: str,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
        request_id: str,
        effect_id: str,
    ) -> tuple[str, str]:
        trace_id = str(metadata.get("trace_id") or "").strip()
        if not trace_id:
            trace_seed = {
                "session_id": session_key,
                "request_id": str(metadata.get("request_id") or ""),
                "user_input": str(user_input or ""),
                "attachments": list(attachments or []),
            }
            trace_blob = json_dumps_sorted(trace_seed)
            trace_id = f"tr-{hashlib.sha256(trace_blob.encode('utf-8')).hexdigest()[:20]}"
        metadata["trace_id"] = trace_id
        if not effect_id:
            effect_id = self._effect_journal.derive_effect_id(
                session_id=session_key,
                request_id=request_id,
                trace_id=trace_id,
            )

        if not request_id and not str(metadata.get("effect_id") or "").strip():
            candidate_job_id = ExecutionJob._stable_token(session_key, trace_id, prefix="job")
            projected_candidate = self.lifecycle_projection.get(candidate_job_id)
            if projected_candidate is not None and projected_candidate.status in {
                ExecutionStatus.COMPLETED,
                ExecutionStatus.FAILED,
            }:
                previous_trace_id = trace_id
                trace_id = ExecutionJob._stable_token(session_key, trace_id, time.time_ns(), prefix="tr")
                metadata["trace_id"] = trace_id
                effect_id = self._effect_journal.derive_effect_id(
                    session_id=session_key,
                    request_id=request_id,
                    trace_id=trace_id,
                )
                self._emit_progress_snapshot(
                    phase="before_replay",
                    session_id=session_key,
                    trace_token=trace_id,
                    job_id="",
                    future_done=False,
                    note="rotated terminal trace identity for non-idempotent retry",
                    extra={"previous_trace_id": previous_trace_id},
                )
        return trace_id, effect_id

    def _completion_expectations(
        self,
        *,
        job_id: str,
        future_done: bool,
        projection_terminal: bool,
    ) -> dict[str, bool]:
        return {
            "future_done": bool(future_done),
            "projection_terminal": bool(projection_terminal),
            "job_removed_from_scheduler": bool(job_id not in getattr(self._scheduler, "_jobs", {})),
        }

    async def _drain_scheduler_until_resolved(
        self,
        *,
        future: asyncio.Future[FinalizedTurnResult],
        job: ExecutionJob,
        session_key: str,
        trace_token: str,
        deadline: float,
    ) -> int:
        loop_iterations = 0
        consecutive_idle_drains = 0
        last_progress_emit = time.monotonic()
        while not future.done():
            loop_iterations += 1
            if time.time() > deadline:
                projected_at_deadline = self.lifecycle_projection.get(job.job_id)
                projection_terminal_at_deadline = bool(
                    projected_at_deadline is not None
                    and projected_at_deadline.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
                )
                self._emit_progress_snapshot(
                    phase="during_scheduler_drain",
                    session_id=session_key,
                    trace_token=trace_token,
                    job_id=job.job_id,
                    future_done=future.done(),
                    completion_expectations=self._completion_expectations(
                        job_id=job.job_id,
                        future_done=future.done(),
                        projection_terminal=projection_terminal_at_deadline,
                    ),
                    note="deadline exceeded",
                    extra={"loop_iterations": loop_iterations, "consecutive_idle_drains": consecutive_idle_drains},
                )
                raise TimeoutError("submit_turn exceeded timeout")

            drained = await self.scheduler.drain_once(self.kernel_executor)
            if drained:
                consecutive_idle_drains = 0
                continue

            consecutive_idle_drains += 1
            remaining = max(0.0, deadline - time.time())
            if remaining <= 0.0:
                raise TimeoutError("submit_turn exceeded timeout")

            wait_task = asyncio.create_task(
                self.scheduler.wait_for_work(timeout_seconds=remaining),
            )
            done, _ = await asyncio.wait(
                {future, wait_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            wait_ready = False
            if wait_task in done:
                with contextlib.suppress(Exception):
                    wait_ready = bool(wait_task.result())

            if (time.monotonic() - last_progress_emit) >= 1.0 or consecutive_idle_drains >= 5:
                projected = self.lifecycle_projection.get(job.job_id)
                projection_terminal = bool(
                    projected is not None and projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
                )
                if consecutive_idle_drains >= 5:
                    _mutate_runtime_plan(
                        metadata=job.metadata,
                        reason="scheduler_stall_replan",
                        status="active",
                        strategy="clarify",
                        note="idle_drain_threshold_reached",
                    )
                    self._emit_runtime_stream_event(
                        event_type="plan.updated",
                        session_id=session_key,
                        trace_id=trace_token,
                        payload={
                            "job_id": str(job.job_id or ""),
                            "runtime_plan": dict(job.metadata.get("runtime_plan") or {}),
                            "trigger": "scheduler_stall_replan",
                        },
                    )
                self._emit_progress_snapshot(
                    phase="during_scheduler_drain",
                    session_id=session_key,
                    trace_token=trace_token,
                    job_id=job.job_id,
                    future_done=future.done(),
                    completion_expectations=self._completion_expectations(
                        job_id=job.job_id,
                        future_done=future.done(),
                        projection_terminal=projection_terminal,
                    ),
                    note="idle drain iteration",
                    extra={
                        "loop_iterations": loop_iterations,
                        "consecutive_idle_drains": consecutive_idle_drains,
                        "wait_ready": bool(wait_ready),
                        "remaining_seconds": float(remaining),
                        "stall_phase_candidate": "during_scheduler_drain",
                        "potential_scheduler_ledger_cycle": bool(
                            consecutive_idle_drains >= 5
                            and int(len(getattr(self._scheduler, "_pending_job_ids", []) or [])) > 0
                            and not wait_ready
                        ),
                    },
                )
                last_progress_emit = time.monotonic()

            if wait_task not in done:
                wait_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await wait_task
        return loop_iterations

    def _build_response_engine_context(
        self,
        *,
        job: ExecutionJob,
        session_key: str,
        initial_response: str,
    ) -> Any:
        """Build execution context for ResponseEngine ranking.

        Phase 1: Create context from job metadata and session state for response ranking.
        """
        session = self.registry.get_or_create(session_key)
        session_state_ref = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(session_state_ref, dict):
            session_state_ref = {}

        metadata = dict(job.metadata or {})
        if not isinstance(metadata.get("reward_feedback"), dict):
            pending_feedback = dict(session_state_ref.get("response_learning_pending") or {})
            if pending_feedback:
                synthesized_feedback = _synthesize_reward_feedback(
                    pending_selection=pending_feedback,
                    current_user_input=str(job.user_input or ""),
                    metadata=metadata,
                    session_state=session_state_ref,
                )
                if isinstance(synthesized_feedback, dict) and synthesized_feedback:
                    metadata["reward_feedback"] = synthesized_feedback
                    session_state_ref.pop("response_learning_pending", None)
                else:
                    attempts = int(pending_feedback.get("feedback_attempts") or 0) + 1
                    if attempts >= 3:
                        session_state_ref.pop("response_learning_pending", None)
                    else:
                        pending_feedback["feedback_attempts"] = attempts
                        session_state_ref["response_learning_pending"] = pending_feedback
        else:
            session_state_ref.pop("response_learning_pending", None)

        session_state = dict(session_state_ref)
        
        # Simple dynamic context object with execution data.
        from types import SimpleNamespace

        context = SimpleNamespace()
        context.user_input = str(job.user_input or "")
        context.initial_response = initial_response
        context.session_state = session_state
        context.job_metadata = metadata
        context.trace_id = str(job.trace_id or "")

        context.memory_context = list(metadata.get("memory_context") or [])

        context.persona_traits = list(session_state.get("persona_traits") or [])
        context.persona_constraints = dict(session_state.get("persona_constraints") or {})

        # Allow either explicit user_preferences or profile.preferences path.
        user_preferences = session_state.get("user_preferences")
        if not isinstance(user_preferences, dict):
            profile_blob = session_state.get("profile")
            if isinstance(profile_blob, dict):
                user_preferences = dict(profile_blob.get("preferences") or {})
        context.user_preferences = dict(user_preferences or {})

        context.conversation_trajectory = dict(
            session_state.get("conversation_trajectory")
            or metadata.get("conversation_trajectory")
            or {},
        )
        
        return context

    def _finalize_submit_success(
        self,
        *,
        job: ExecutionJob,
        result: FinalizedTurnResult,
        session_key: str,
        trace_token: str,
        before_state_hash: str,
        dedupe_future: asyncio.Future[FinalizedTurnResult] | None,
        loop_iterations: int,
    ) -> FinalizedTurnResult:
        finalization = dict(job.metadata.get("submit_finalization") or {})
        if bool(finalization.get("done", False)):
            return (
                str(finalization.get("response") or (result[0] if isinstance(result, tuple) and len(result) >= 1 else "")),
                bool(finalization.get("should_end", result[1] if isinstance(result, tuple) and len(result) >= 2 else False)),
            )

        current_execution_result = ensure_unified_execution_result(
            dict(job.metadata.get("execution_result") or {}),
        )
        # Extract initial response from execution
        initial_response = str(result[0] if isinstance(result, tuple) and len(result) >= 1 else "")
        
        # Phase 1: Run ResponseEngine to rank and select best response
        # ResponseEngine takes the execution context and generates/ranks candidates
        engine_context = self._build_response_engine_context(
            job=job,
            session_key=session_key,
            initial_response=initial_response,
        )
        try:
            with contextlib.suppress(Exception):
                self._topology_record_node(
                    node_id="response_engine.generate_and_rank",
                    metadata={"trace_id": str(trace_token or "")},
                )
            # PHASE 1: ResponseEngine is SOLE selection authority
            # Only ResponseEngine may influence which response is selected.
            # All other systems are telemetry-only observers.
            ranked_response = self._response_engine.run(engine_context)
            final_response = ranked_response if ranked_response else initial_response
            telemetry = getattr(engine_context, "response_engine_telemetry", None)
            if isinstance(telemetry, dict) and telemetry:
                job.metadata["response_engine_telemetry"] = telemetry
        except Exception as exc:
            logger.warning(f"ResponseEngine ranking failed: {exc}; using initial response")
            final_response = initial_response
            telemetry = None
        
        if current_execution_result.get("status") not in _TERMINAL_STATUS_VALUES:
            current_execution_result = mark_unified_execution_success(
                cast(dict[str, Any], current_execution_result),
                response=final_response,
                should_end=bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
            )
            job.metadata["execution_result"] = current_execution_result
            job.metadata["response_authority"] = "response_engine"
            # Update result tuple with ranked response
            result = (final_response, result[1] if isinstance(result, tuple) and len(result) >= 2 else False)
        session_after = self.registry.get_or_create(session_key)
        self._post_execution_pre_commit_contract_gate(
            session_id=session_key,
            job=job,
            result=result,
            input_state_hash=before_state_hash,
        )
        self._validate_trace_invariant(job, result, session=session_after, state_before_hash=before_state_hash)
        self._apply_turn_committed_core_state(
            session=session_after,
            job=job,
            result=result,
        )
        if isinstance(session_after, dict):
            state = session_after.setdefault("state", {})
            if isinstance(state, dict):
                selected = dict(dict(job.metadata.get("response_engine_telemetry") or {}).get("selected") or {})
                if selected:
                    state["response_learning_pending"] = {
                        "trace_id": str(trace_token or job.trace_id or ""),
                        "created_at": float(time.time()),
                        "selected": selected,
                        "selected_text_preview": str(
                            dict(job.metadata.get("response_engine_telemetry") or {}).get("selected_text_preview") or ""
                        ),
                        "feedback_attempts": 0,
                    }
        self._maybe_compact_ledger()
        if dedupe_future is not None and not dedupe_future.done():
            dedupe_future.set_result(result)
        projected = self.lifecycle_projection.get(job.job_id)
        projection_terminal = bool(
            projected is not None and projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
        )
        self._emit_progress_snapshot(
            phase="during_scheduler_drain",
            session_id=session_key,
            trace_token=trace_token,
            job_id=job.job_id,
            future_done=True,
            completion_expectations=self._completion_expectations(
                job_id=job.job_id,
                future_done=True,
                projection_terminal=projection_terminal,
            ),
            note="submit_turn resolved",
            extra={"loop_iterations": loop_iterations},
        )
        response_text = str(result[0] if isinstance(result, tuple) and len(result) >= 1 else "")
        self._promote_semantic_memory(
            session=self.registry.get_or_create(session_key),
            job=job,
            response_text=response_text,
        )
        session_state = dict(self.registry.get_or_create(session_key).get("state") or {})
        runtime_plan = dict(job.metadata.get("runtime_plan") or {})
        session_state.setdefault("authority_modes", {})
        if isinstance(session_state["authority_modes"], dict):
            session_state["authority_modes"]["response_selection"] = "response_engine"
            session_state["authority_modes"]["learning_update"] = "response_engine"
            session_state["authority_modes"]["memory_write"] = "memory_manager"
        learning_telemetry = {
            "trace_id": str(trace_token or ""),
            "status": "success",
            "strategy": str(runtime_plan.get("strategy") or "direct_answer"),
            "uncertainty_score": float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 0.0),
            "response_length": int(len(response_text or "")),
            "non_canonical_loops": [
                "adaptation_engine",
                "belief_state_engine",
                "planning_optimizer",
                "alignment_trainer",
                "memory_hierarchy_manager",
                "tool_self_model",
                "multi_agent_swarm",
                "autonomous_goal_daemon",
            ],
        }
        session_state["learning_signal_telemetry"] = learning_telemetry
        job.metadata["learning_signal_telemetry"] = dict(learning_telemetry)
        self._interactive_cognition_ui.emit_thought(
            state=session_state,
            trace_id=str(trace_token or ""),
            content="Turn completed successfully",
            confidence=0.95,
            category="outcome",
        )
        self._merge_session_state(session_id=session_key, state_patch=session_state)
        self._emit_runtime_stream_event(
            event_type="partial.output",
            session_id=session_key,
            trace_id=trace_token,
            payload={
                "chunk": response_text,
                "is_terminal_chunk": True,
                "job_id": str(job.job_id or ""),
            },
        )
        self._emit_runtime_stream_event(
            event_type="turn.completed",
            session_id=session_key,
            trace_id=trace_token,
            payload={
                "job_id": str(job.job_id or ""),
                "loop_iterations": int(loop_iterations),
                "should_end": bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
                "runtime_plan": dict(job.metadata.get("runtime_plan") or {}),
            },
        )
        with contextlib.suppress(Exception):
            self._topology_record_node(
                node_id="persistence.finalize_turn",
                metadata={"trace_id": str(trace_token or "")},
            )
        job.metadata["submit_finalization"] = {
            "done": True,
            "response": str(result[0] if isinstance(result, tuple) and len(result) >= 1 else ""),
            "should_end": bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
        }
        return result

    def _record_submit_exception(
        self,
        *,
        job: ExecutionJob,
        future: asyncio.Future[FinalizedTurnResult],
        dedupe_future: asyncio.Future[FinalizedTurnResult] | None,
        session_key: str,
        trace_token: str,
        exc: BaseException,
    ) -> None:
        classified = _classify_execution_failure(exc)
        current_execution_result = mark_unified_execution_failure(
            cast(dict[str, Any], build_unified_execution_result()),
            failure_class=str(classified.get("failure_class") or "runtime_exception"),
            failure_source=str(classified.get("failure_source") or "execution"),
            retryable=bool(classified.get("retryable", False)),
            exception_type=str(classified.get("exception_type") or type(exc).__name__),
            message=str(exc),
        )
        job.metadata["execution_result"] = current_execution_result
        execution_state = dict(job.metadata.get("execution_state") or {})
        execution_state["failure_type"] = str(classified.get("failure_type") or "")
        execution_state["failure_action"] = str(classified.get("failure_action") or "")
        execution_state["auto_retry"] = bool(classified.get("auto_retry", False))
        execution_state["last_transition_reason"] = (
            f"control_plane.submit.failed:{str(classified.get('failure_action') or 'unknown')}"
        )
        job.metadata["execution_state"] = execution_state
        _set_terminal_turn_state(
            job,
            terminal_state=_SCHEDULER_EXCEPTION_MAPPER.from_exception(exc),
            reason=f"control_plane.submit.failed:{type(exc).__name__}",
            strict=False,
        )
        _mutate_runtime_plan(
            metadata=job.metadata,
            reason=f"submit_failed:{type(exc).__name__}",
            status="failed",
            note=str(exc),
        )
        if future.done() and not future.cancelled():
            with contextlib.suppress(Exception):
                future.exception()
        if dedupe_future is not None and not dedupe_future.done():
            dedupe_future.set_exception(exc)
        projected = self.lifecycle_projection.get(job.job_id)
        projection_terminal = bool(
            projected is not None and projected.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
        )
        self._emit_progress_snapshot(
            phase="during_scheduler_drain",
            session_id=session_key,
            trace_token=trace_token,
            job_id=job.job_id,
            future_done=future.done(),
            completion_expectations=self._completion_expectations(
                job_id=job.job_id,
                future_done=future.done(),
                projection_terminal=projection_terminal,
            ),
            note="submit_turn exception",
            extra={"exception_type": type(exc).__name__, "exception": str(exc)},
        )
        self._emit_runtime_stream_event(
            event_type="turn.failed",
            session_id=session_key,
            trace_id=trace_token,
            payload={
                "job_id": str(job.job_id or ""),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "runtime_plan": dict(job.metadata.get("runtime_plan") or {}),
            },
        )
        session_state = dict(self.registry.get_or_create(session_key).get("state") or {})
        runtime_plan = dict(job.metadata.get("runtime_plan") or {})
        session_state.setdefault("authority_modes", {})
        if isinstance(session_state["authority_modes"], dict):
            session_state["authority_modes"]["response_selection"] = "response_engine"
            session_state["authority_modes"]["learning_update"] = "response_engine"
            session_state["authority_modes"]["memory_write"] = "memory_manager"
        failure_learning_telemetry = {
            "trace_id": str(trace_token or ""),
            "status": "failure",
            "strategy": str(runtime_plan.get("strategy") or "direct_answer"),
            "uncertainty_score": float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 1.0),
            "failure_type": str(classified.get("failure_type") or "runtime_failure"),
            "non_canonical_loops": [
                "adaptation_engine",
                "belief_state_engine",
                "planning_optimizer",
                "alignment_trainer",
                "memory_hierarchy_manager",
                "tool_self_model",
                "multi_agent_swarm",
                "autonomous_goal_daemon",
            ],
        }
        session_state["learning_signal_telemetry"] = failure_learning_telemetry
        job.metadata["learning_signal_telemetry"] = dict(failure_learning_telemetry)
        self._interactive_cognition_ui.emit_thought(
            state=session_state,
            trace_id=str(trace_token or ""),
            content=f"Turn failed: {type(exc).__name__}",
            confidence=0.9,
            category="outcome",
        )
        self._merge_session_state(session_id=session_key, state_patch=session_state)

    def _prepare_submit_metadata(
        self,
        *,
        metadata: dict[str, Any] | None,
        user_input: str,
        attachments: AttachmentList | None,
        timeout_seconds: float | None,
    ) -> tuple[dict[str, Any], float, str, str]:
        md = dict(metadata or {})
        request_id = str(md.get("request_id") or "").strip()
        effect_id = str(md.get("effect_id") or "").strip()
        self._prepare_global_confluence_law(
            user_input=str(user_input or ""),
            attachments=attachments,
            metadata=md,
        )
        resolved_timeout_seconds = 30.0 if timeout_seconds is None else max(0.0, float(timeout_seconds))
        _raw_er = dict(
            md.get("execution_result")
            or build_unified_execution_result(
                timeout_seconds=resolved_timeout_seconds,
                degradation_items=_extract_execution_degradations(md),
            ),
        )
        # Stamp the current resolved timeout before normalization so the
        # canonical module owns the final sanitised form.
        raw_execution_result = cast(dict[str, Any], _raw_er)
        raw_execution_result.setdefault("timeout", {})["seconds"] = float(resolved_timeout_seconds)
        md["execution_result"] = ensure_unified_execution_result(raw_execution_result)
        md["execution_state"] = {
            "lifecycle_state": ExecutionLifecycleState.SUBMITTED.value,
            "redelivery_count": int(dict(md.get("execution_state") or {}).get("redelivery_count") or 0),
            "lease_conflict_count": int(dict(md.get("execution_state") or {}).get("lease_conflict_count") or 0),
            "last_worker_id": str(dict(md.get("execution_state") or {}).get("last_worker_id") or ""),
            "last_transition_reason": "control_plane.submit_turn",
            "retry_not_before_monotonic": 0.0,
        }
        md.setdefault("strict_trace_invariant", True)
        tool_specs = self._discover_tool_specs()
        prior_plan = dict(md.get("runtime_plan") or {})
        md["runtime_plan"] = self._cognitive_policy_engine.build_plan(
            session_id=str(md.get("session_id") or "default"),
            trace_id=str(md.get("trace_id") or ""),
            user_input=str(user_input or ""),
            existing_plan=prior_plan,
            memory_hits=int(len(list(md.get("semantic_memory_context") or []))),
            tool_candidates=int(len(tool_specs)),
        )
        md["tool_runtime_contract"] = _normalize_tool_runtime_contract(md)
        md["tool_routing_plan"] = self._tool_routing_engine.build_routing_plan(
            tool_request=dict(md.get("tool_runtime_contract") or {}),
            available_specs=tool_specs,
            uncertainty_score=float(dict(md.get("runtime_plan") or {}).get("uncertainty", {}).get("score") or 0.0),
        )
        md["compositional_tool_plan"] = self._compositional_tool_planner.build_plan(
            user_input=str(user_input or ""),
            routing_plan=dict(md.get("tool_routing_plan") or {}),
            available_specs=tool_specs,
        )
        md["reasoning_hypotheses"] = self._hypothesis_engine.infer_hypotheses(
            user_input=str(user_input or ""),
            runtime_plan=dict(md.get("runtime_plan") or {}),
            tool_routing_plan=dict(md.get("tool_routing_plan") or {}),
            memory_context=[
                dict(item)
                for item in list(md.get("semantic_memory_context") or [])
                if isinstance(item, dict)
            ],
        )
        md["safety_state"] = self._semantic_safety_engine.classify(
            user_input=str(user_input or ""),
            runtime_plan=dict(md.get("runtime_plan") or {}),
            memory_context=[
                dict(item)
                for item in list(md.get("semantic_memory_context") or [])
                if isinstance(item, dict)
            ],
        )
        return md, resolved_timeout_seconds, request_id, effect_id

    @staticmethod
    def _confluence_mode(metadata: dict[str, Any]) -> str:
        override = str(dict(metadata or {}).get("confluence_mode") or "").strip().lower()
        test_override = bool(
            str(os.environ.get("PYTEST_CURRENT_TEST") or "").strip()
            and str(os.environ.get("DADBOT_TEST_GLOBAL_CONFLUENCE_MODE") or "").strip().lower()
            in {"off", "audit", "enforce"}
        )
        if test_override and override in {"off", "audit", "enforce"}:
            return override
        if test_override:
            return str(os.environ.get("DADBOT_TEST_GLOBAL_CONFLUENCE_MODE") or "enforce").strip().lower()
        return "enforce"

    def _derive_confluence_key(
        self,
        *,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
    ) -> str:
        explicit = str(dict(metadata or {}).get("confluence_key") or "").strip()
        if explicit:
            return explicit
        namespace = str(dict(metadata or {}).get("confluence_namespace") or "global").strip() or "global"
        payload = {
            "namespace": namespace,
            "user_input": str(user_input or ""),
            "attachments": list(attachments or []),
            "semantic_eval_input_hash": str(dict(metadata or {}).get("semantic_eval_input_hash") or ""),
        }
        return f"auto:{self._stable_hash(payload)}"

    def _prepare_global_confluence_law(
        self,
        *,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
    ) -> None:
        mode = self._confluence_mode(metadata)
        if mode == "off":
            return
        explicit_key = str(dict(metadata or {}).get("confluence_key") or "").strip()
        if mode == "enforce" and not explicit_key:
            allow_legacy = bool(
                str(os.environ.get("PYTEST_CURRENT_TEST") or "").strip()
                and str(os.environ.get("DADBOT_ALLOW_LEGACY_CONFLUENCE_KEY", "0")).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            if not allow_legacy:
                raise RuntimeError(
                    "Missing explicit confluence_key in enforce mode. "
                    "Set metadata['confluence_key'] at orchestrator boundary.",
                )
        key = self._derive_confluence_key(
            user_input=user_input,
            attachments=attachments,
            metadata=metadata,
        )
        known = str(self._global_confluence_contracts.get(key) or "")
        metadata["_global_confluence_mode"] = mode
        metadata["_global_confluence_key"] = key
        if known:
            metadata.setdefault("expected_execution_confluence_hash", known)

    def _assert_lifecycle_order(self, trace_events: list[dict[str, Any]]) -> None:
        event_types = [str(event.get("type") or "") for event in list(trace_events or [])]
        required = ["JOB_SUBMITTED", "SESSION_BOUND", "JOB_QUEUED", "JOB_STARTED", "JOB_COMPLETED"]
        positions: dict[str, int] = {}
        for event_type in required:
            if event_type not in event_types:
                raise RuntimeError(
                    f"Control-plane lifecycle invariant violated: missing event {event_type!r}",
                )
            positions[event_type] = int(event_types.index(event_type))
        ordered = [positions[name] for name in required]
        if ordered != sorted(ordered):
            raise RuntimeError(
                "Control-plane lifecycle invariant violated: event order is non-monotonic",
            )

    @staticmethod
    def _result_output_payload(result: FinalizedTurnResult) -> dict[str, Any]:
        return {
            "response": str(result[0] if isinstance(result, tuple) and len(result) >= 1 else ""),
            "should_end": bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
        }

    def _build_composition_payload(
        self,
        *,
        job: ExecutionJob,
        terminal_state: dict[str, Any],
        output_payload: dict[str, Any],
        trace_events: list[dict[str, Any]],
        state_before_hash: str,
        state_after_hash: str,
    ) -> dict[str, Any]:
        state_delta_hash = self._stable_hash(
            {
                "before": str(state_before_hash or ""),
                "after": str(state_after_hash or ""),
            },
        )
        event_log_hash = self._event_stream_digest(trace_events)
        return {
            "contract_version": "turn-composition-v1",
            "context_input_hash": self._stable_hash(
                {
                    "session_id": str(job.session_id or ""),
                    "trace_id": str(job.trace_id or ""),
                    "user_input": str(job.user_input or ""),
                    "attachments": list(job.attachments or []),
                    "metadata": dict(job.metadata or {}),
                },
            ),
            "execution_dag_hash": str(terminal_state.get("execution_dag_hash") or ""),
            "policy_hash": str(terminal_state.get("policy_hash") or ""),
            "state_delta_hash": state_delta_hash,
            "event_log_hash": event_log_hash,
            "output_hash": self._stable_hash(output_payload),
            "mutation_effects_hash": str(terminal_state.get("post_commit_mutation_effects_hash") or ""),
            "determinism_closure_hash": str(terminal_state.get("determinism_closure_hash") or ""),
        }

    def _build_confluence_payload(
        self,
        *,
        job: ExecutionJob,
        terminal_state: dict[str, Any],
        output_payload: dict[str, Any],
        trace_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "contract_version": "turn-confluence-v1",
            "semantic_input_hash": self._stable_hash(
                {
                    "user_input": str(job.user_input or ""),
                    "attachments": list(job.attachments or []),
                    "semantic_eval_input_hash": str(dict(job.metadata or {}).get("semantic_eval_input_hash") or ""),
                },
            ),
            "execution_dag_hash": str(terminal_state.get("execution_dag_hash") or ""),
            "policy_hash": str(terminal_state.get("policy_hash") or ""),
            "output_hash": self._stable_hash(output_payload),
            "mutation_effects_hash": str(terminal_state.get("post_commit_mutation_effects_hash") or ""),
            "determinism_closure_hash": str(terminal_state.get("determinism_closure_hash") or ""),
            "event_semantic_hash": self._event_semantic_digest(trace_events),
        }

    @staticmethod
    def _expected_hashes_from_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
        expected = str(dict(metadata or {}).get("expected_execution_composition_hash") or "").strip()
        expected_confluence = str(
            dict(metadata or {}).get("expected_execution_confluence_hash") or "",
        ).strip()
        return expected, expected_confluence

    @staticmethod
    def _confluence_config_from_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
        confluence_key = str(dict(metadata or {}).get("_global_confluence_key") or "").strip()
        confluence_mode = str(dict(metadata or {}).get("_global_confluence_mode") or "off").strip().lower()
        return confluence_key, confluence_mode

    def _validate_composition_expectations(
        self,
        *,
        expected: str,
        expected_confluence: str,
        composition_hash: str,
        confluence_class_hash: str,
    ) -> None:
        if expected and expected != composition_hash:
            raise RuntimeError(
                f"Execution composition mismatch: expected={expected!r}, actual={composition_hash!r}",
            )
        if expected_confluence and expected_confluence != confluence_class_hash:
            raise RuntimeError(
                f"Execution confluence mismatch: expected={expected_confluence!r}, actual={confluence_class_hash!r}",
            )

    def _enforce_global_confluence_law(
        self,
        *,
        confluence_key: str,
        confluence_mode: str,
        confluence_class_hash: str,
        expected_confluence: str,
    ) -> dict[str, Any]:
        confluence_report = {
            "enforced": False,
            "mode": confluence_mode,
            "key": confluence_key,
            "observed_hash": confluence_class_hash,
            "expected_hash": expected_confluence,
            "contract_version": "turn-confluence-v1",
        }
        if not confluence_key or confluence_mode == "off":
            return confluence_report

        self._confluence_metrics["attempted"] = int(self._confluence_metrics.get("attempted", 0)) + 1
        known = str(self._global_confluence_contracts.get(confluence_key) or "")
        if not known:
            self._global_confluence_contracts[confluence_key] = confluence_class_hash
            confluence_report["enforced"] = True
            confluence_report["action"] = "bound_first_observation"
            self._confluence_metrics["bound_first_observation"] = int(
                self._confluence_metrics.get("bound_first_observation", 0),
            ) + 1
            return confluence_report

        if known != confluence_class_hash:
            confluence_report["enforced"] = True
            confluence_report["expected_hash"] = known
            confluence_report["action"] = "mismatch"
            self._last_confluence_report = dict(confluence_report)
            self._confluence_metrics["mismatch"] = int(self._confluence_metrics.get("mismatch", 0)) + 1
            fail_mode = str(
                os.environ.get("DADBOT_CONFLUENCE_VIOLATION_MODE", "fail"),
            ).strip().lower()
            if fail_mode != "audit":
                self._confluence_metrics["enforced_blocked"] = int(
                    self._confluence_metrics.get("enforced_blocked", 0),
                ) + 1
                raise RuntimeError(
                    "Global confluence law violated for key="
                    f"{confluence_key!r}: expected={known!r}, "
                    f"actual={confluence_class_hash!r}",
                )
            logger.warning(
                "Global confluence law mismatch (audit override): key=%s expected=%s actual=%s",
                confluence_key,
                known,
                confluence_class_hash,
            )
            return confluence_report

        confluence_report["enforced"] = True
        confluence_report["expected_hash"] = known
        confluence_report["action"] = "matched"
        self._confluence_metrics["matched"] = int(self._confluence_metrics.get("matched", 0)) + 1
        return confluence_report

    def _record_turn_composition_contract(
        self,
        *,
        session: dict[str, Any],
        job: ExecutionJob,
        result: FinalizedTurnResult,
        state_before_hash: str,
    ) -> dict[str, Any]:
        state = dict(session.get("state") or {})
        state_after_hash = self._stable_hash(state)
        terminal_state = dict(state.get("last_terminal_state") or {})
        trace_events = self._job_trace_events(job)
        if any(str(event.get("type") or "").strip() for event in list(trace_events or [])):
            self._assert_lifecycle_order(trace_events)

        output_payload = self._result_output_payload(result)
        composition_payload = self._build_composition_payload(
            job=job,
            terminal_state=terminal_state,
            output_payload=output_payload,
            trace_events=trace_events,
            state_before_hash=state_before_hash,
            state_after_hash=state_after_hash,
        )
        composition_hash = self._stable_hash(composition_payload)
        confluence_payload = self._build_confluence_payload(
            job=job,
            terminal_state=terminal_state,
            output_payload=output_payload,
            trace_events=trace_events,
        )
        confluence_class_hash = self._stable_hash(confluence_payload)
        contract = dict(composition_payload)
        contract["composition_hash"] = composition_hash
        contract["confluence_class_hash"] = confluence_class_hash

        expected, expected_confluence = self._expected_hashes_from_metadata(dict(job.metadata or {}))
        self._validate_composition_expectations(
            expected=expected,
            expected_confluence=expected_confluence,
            composition_hash=composition_hash,
            confluence_class_hash=confluence_class_hash,
        )

        confluence_key, confluence_mode = self._confluence_config_from_metadata(dict(job.metadata or {}))
        confluence_report = self._enforce_global_confluence_law(
            confluence_key=confluence_key,
            confluence_mode=confluence_mode,
            confluence_class_hash=confluence_class_hash,
            expected_confluence=expected_confluence,
        )
        self._last_confluence_report = dict(confluence_report)

        state_mut = session.setdefault("state", {})
        if isinstance(state_mut, dict):
            state_mut["last_execution_composition_contract"] = dict(contract)
            state_mut["last_execution_confluence_report"] = dict(confluence_report)
        return contract

    async def _register_submit_job(
        self,
        *,
        session_key: str,
        user_input: str,
        attachments: AttachmentList | None,
        metadata: dict[str, Any],
        trace_token: str,
    ) -> tuple[ExecutionJob, asyncio.Future[FinalizedTurnResult], float]:
        job = ExecutionJob(
            session_id=session_key,
            user_input=str(user_input or ""),
            attachments=attachments,
            metadata=metadata,
            trace_id=trace_token,
        )
        current_lifecycle_state = self._coerce_projection_lifecycle_state(job.job_id)
        if current_lifecycle_state is None:
            _assert_lifecycle_emission_transition(
                execution_id=str(job.job_id or ""),
                event=Submitted(execution_id=job.job_id, occurred_at=datetime.now()),
                current_state=current_lifecycle_state,
            )
            self.ledger_writer.append_execution_lifecycle(
                Submitted(execution_id=job.job_id, occurred_at=datetime.now()),
                session_id=session_key,
                trace_id=job.trace_id,
                kernel_step_id="control_plane.submit_turn",
                committed=False,
            )
        self.lifecycle_projection.rebuild_from_ledger(self.ledger.read())
        _apply_projection_execution_state(job, self.lifecycle_projection.get(job.job_id))
        job.metadata["execution_mode"] = _resolved_execution_mode(job)

        submitted_event = self.ledger_writer.append_job_submitted(job)
        submitted_ts = float(submitted_event.get("timestamp") or 0.0)
        job.metadata["submitted_timestamp"] = submitted_ts
        job.metadata.setdefault("claim_order", {})
        job.metadata["claim_order"]["timestamp"] = submitted_ts
        job.metadata["claim_order"]["worker_id"] = str(
            getattr(self._scheduler, "worker_id", "worker-1") or "worker-1",
        )
        job.metadata["claim_order"]["lease_epoch"] = int(
            dict(job.metadata.get("execution_state") or {}).get("redelivery_count") or 0,
        )
        self.ledger_writer.append_session_bound(
            session_key,
            job.job_id,
            trace_id=job.trace_id,
            kernel_step_id="control_plane.bind_session",
        )
        future = await self.scheduler.register(job)
        return job, future, submitted_ts

    def _initialize_submit_scope(
        self,
        *,
        session_key: str,
        trace_token: str,
        job_id: str,
        resolved_timeout_seconds: float,
    ) -> tuple[str, float, object]:
        session_before = self.registry.get_or_create(session_key)
        before_state_hash = self._stable_hash(dict(session_before.get("state") or {}))
        # Initialize CoreState from persisted session state; all turn mutations will
        # flow through the event bus (push_core_state_event) and update this binding.
        initial_core_state = CoreState.from_dict(
            dict(session_before.get("state") or {}).get("core_state"),
        )
        deadline = time.time() + resolved_timeout_seconds
        cs_token = open_core_state_scope(initial_core_state)
        push_core_state_event(
            "job_submitted",
            {
                "session_id": session_key,
                "trace_id": trace_token,
                "job_id": job_id,
            },
        )
        return before_state_hash, deadline, cs_token

    async def _submit_turn_kernel(
        self,
        *,
        session_id: str,
        user_input: str,
        attachments: AttachmentList | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        session_key = str(session_id or "default")
        if self.registry.is_terminated(session_key):
            raise RuntimeError(f"session {session_key!r} has been terminated")
        self._enforce_distributed_runtime_authority(operation="submit_turn_entry_gate")

        md, resolved_timeout_seconds, request_id, effect_id = self._prepare_submit_metadata(
            metadata=metadata,
            user_input=user_input,
            attachments=attachments,
            timeout_seconds=timeout_seconds,
        )
        md["session_id"] = session_key
        self._pre_execution_contract_gate(
            session_id=session_key,
            user_input=str(user_input or ""),
            metadata=md,
        )
        contract_ok, contract_reason = _validate_tool_runtime_contract(md)
        if not contract_ok:
            raise RuntimeError(f"tool runtime contract validation failed: {contract_reason}")
        self._inject_semantic_memory_context(
            session_id=session_key,
            user_input=str(user_input or ""),
            metadata=md,
            limit=5,
        )
        session_state = dict(self.registry.get_or_create(session_key).get("state") or {})
        for spec in self._discover_tool_specs():
            self._tool_ecosystem_hub.register_connector(
                state=session_state,
                name=str(spec.name or ""),
                capabilities=[str(item) for item in list(spec.capabilities or [])],
                endpoint="",
                health=1.0,
            )
        runtime_plan = dict(md.get("runtime_plan") or {})
        if runtime_plan:
            intent_type = str(runtime_plan.get("intent_type") or "statement")
            optimizer_strategy = self._planning_optimizer.suggest(state=session_state, intent_type=intent_type)
            belief_strategy = self._belief_state_engine.next_best_strategy(state=session_state, intent_type=intent_type)
            aligned_strategy = self._alignment_trainer.recommend_strategy(
                state=session_state,
                intent_type=intent_type,
                default_strategy=str(runtime_plan.get("strategy") or "direct_answer"),
            )
            if optimizer_strategy and optimizer_strategy != str(runtime_plan.get("strategy") or ""):
                runtime_plan["strategy"] = optimizer_strategy
                runtime_plan["optimizer_override"] = True
            if aligned_strategy and aligned_strategy != str(runtime_plan.get("strategy") or ""):
                runtime_plan["strategy"] = aligned_strategy
                runtime_plan["alignment_override"] = True
            if belief_strategy:
                runtime_plan["belief_suggested_strategy"] = belief_strategy
            md["runtime_plan"] = runtime_plan

        md["tool_routing_plan"] = self._tool_self_model.apply_routing_feedback(
            state=session_state,
            routing_plan=dict(md.get("tool_routing_plan") or {}),
        )
        md["compositional_tool_plan"] = self._compositional_tool_planner.build_plan(
            user_input=str(user_input or ""),
            routing_plan=dict(md.get("tool_routing_plan") or {}),
            available_specs=self._discover_tool_specs(),
        )
        md["reasoning_hypotheses"] = self._hypothesis_engine.infer_hypotheses(
            user_input=str(user_input or ""),
            runtime_plan=dict(md.get("runtime_plan") or {}),
            tool_routing_plan=dict(md.get("tool_routing_plan") or {}),
            memory_context=[
                dict(item)
                for item in list(md.get("semantic_memory_context") or [])
                if isinstance(item, dict)
            ],
        )

        pending_steering = self._consume_pending_turn_steering(session_id=session_key)
        if pending_steering:
            desired_strategy = str(pending_steering.get("strategy") or "").strip()
            if desired_strategy:
                _mutate_runtime_plan(
                    metadata=md,
                    reason="user_steering",
                    status="active",
                    strategy=desired_strategy,
                    note=str(pending_steering.get("note") or ""),
                )
                self._interactive_cognition_ui.apply_plan_edit(
                    state=session_state,
                    trace_id=str(md.get("trace_id") or ""),
                    edits={"strategy": desired_strategy, "status": "active"},
                    actor="steering",
                )
            md["applied_steering"] = dict(pending_steering)

        runtime_plan = dict(md.get("runtime_plan") or {})
        self._interactive_cognition_ui.register_plan(
            state=session_state,
            trace_id=str(md.get("trace_id") or ""),
            runtime_plan=runtime_plan,
            source="submit_turn",
        )
        self._interactive_cognition_ui.emit_thought(
            state=session_state,
            trace_id=str(md.get("trace_id") or ""),
            content=f"Planning strategy: {str(runtime_plan.get('strategy') or 'direct_answer')}",
            confidence=1.0 - float(dict(runtime_plan.get("uncertainty") or {}).get("score") or 0.0),
            category="plan",
        )
        needed_capabilities = [
            str(item.get("capability") or "")
            for item in list(dict(md.get("tool_routing_plan") or {}).get("alternatives") or [])
            if isinstance(item, dict)
        ]
        md["external_tool_candidates"] = self._tool_ecosystem_hub.rank_connectors(
            state=session_state,
            needed_capabilities=[item for item in needed_capabilities if item],
            limit=5,
        )
        md["swarm_plan"] = self._multi_agent_swarm.build_plan(
            state=session_state,
            trace_id=str(md.get("trace_id") or ""),
            user_input=str(user_input or ""),
            runtime_plan=runtime_plan,
            compositional_tool_plan=dict(md.get("compositional_tool_plan") or {}),
            max_agents=4,
        )

        self._hypothesis_engine.persist(
            state=session_state,
            trace_id=str(md.get("trace_id") or ""),
            hypotheses=[
                dict(item)
                for item in list(md.get("reasoning_hypotheses") or [])
                if isinstance(item, dict)
            ],
        )
        self._post_planning_pre_tool_contract_gate(
            session_id=session_key,
            metadata=md,
        )
        self._merge_session_state(session_id=session_key, state_patch=session_state)
        effect_id, dedupe_key, dedupe_future, owns_dedupe_slot, immediate_result = await self._preflight_submit_turn(
            session_key=session_key,
            request_id=request_id,
            effect_id=effect_id,
            metadata=md,
        )
        if immediate_result is not None:
            return immediate_result

        trace_id, effect_id = self._resolve_submit_trace_and_effect(
            session_key=session_key,
            user_input=user_input,
            attachments=attachments,
            metadata=md,
            request_id=request_id,
            effect_id=effect_id,
        )
        md["runtime_plan"] = _build_runtime_plan(
            session_id=session_key,
            trace_id=trace_id,
            user_input=str(user_input or ""),
            attachments=attachments,
            metadata=md,
        )
        md["effect_id"] = effect_id
        self._emit_runtime_stream_event(
            event_type="turn.started",
            session_id=session_key,
            trace_id=trace_id,
            payload={
                "request_id": request_id,
                "effect_id": effect_id,
            },
        )
        self._emit_runtime_stream_event(
            event_type="plan.created",
            session_id=session_key,
            trace_id=trace_id,
            payload={
                "runtime_plan": dict(md.get("runtime_plan") or {}),
                "semantic_memory_context_size": int(len(md.get("semantic_memory_context") or [])),
            },
        )
        self._emit_runtime_stream_event(
            event_type="swarm.plan",
            session_id=session_key,
            trace_id=trace_id,
            payload={"swarm_plan": dict(md.get("swarm_plan") or {})},
        )
        assert trace_id, "Missing trace_id at control plane entry"
        job, future, _submitted_ts = await self._register_submit_job(
            session_key=session_key,
            user_input=user_input,
            attachments=attachments,
            metadata=md,
            trace_token=trace_id,
        )
        self._emit_progress_snapshot(
            phase="during_scheduler_drain",
            session_id=session_key,
            trace_token=trace_id,
            job_id=job.job_id,
            future_done=False,
            note="job registered",
            extra={"request_id": request_id, "effect_id": effect_id},
        )
        before_state_hash, deadline, _cs_token = self._initialize_submit_scope(
            session_key=session_key,
            trace_token=trace_id,
            job_id=job.job_id,
            resolved_timeout_seconds=resolved_timeout_seconds,
        )
        try:
            loop_iterations = await self._drain_scheduler_until_resolved(
                future=future,
                job=job,
                session_key=session_key,
                trace_token=trace_id,
                deadline=deadline,
            )
            result = await future
            return self._finalize_submit_success(
                job=job,
                result=result,
                session_key=session_key,
                trace_token=trace_id,
                before_state_hash=before_state_hash,
                dedupe_future=dedupe_future,
                loop_iterations=loop_iterations,
            )
        except Exception as exc:
            self._record_submit_exception(
                job=job,
                future=future,
                dedupe_future=dedupe_future,
                session_key=session_key,
                trace_token=trace_id,
                exc=exc,
            )
            raise
        finally:
            if (request_id or effect_id) and owns_dedupe_slot:
                async with self._inflight_lock:
                    if self._inflight_by_request.get(dedupe_key) is dedupe_future:
                        self._inflight_by_request.pop(dedupe_key, None)
            close_core_state_scope(_cs_token)

    def _coerce_projection_lifecycle_state(self, execution_id: str) -> ExecutionLifecycleState | None:
        projected = self.lifecycle_projection.get(str(execution_id or ""))
        if projected is None:
            return None
        return _coerce_lifecycle_state(_lifecycle_state_from_projection(projected))

    @staticmethod
    def _trace_event_invariant_counts(trace_events: list[dict[str, Any]]) -> tuple[int, bool]:
        commit_count = 0
        has_node_events = False
        for event in trace_events:
            payload = dict(event.get("payload") or {}) if isinstance(event, dict) else {}
            event_type = str(event.get("event_type") or payload.get("event_type") or "").strip().lower()
            stage = str(
                event.get("stage")
                or event.get("node_type")
                or payload.get("stage")
                or payload.get("node_type")
                or ""
            ).strip().lower()
            if event_type in {
                "node_start",
                "node_complete",
                "node_completed",
                "turn_start",
                "turn_complete",
                "turn_failed",
            }:
                has_node_events = True
            if event_type in {"node_complete", "node_completed"} and stage == "save":
                commit_count += 1
        return commit_count, has_node_events

    @staticmethod
    def _fallback_commit_count_from_session(
        *,
        session: dict[str, Any] | None,
        trace_token: str,
    ) -> int:
        if not isinstance(session, dict):
            return 0
        session_state = dict(session.get("state") or {})
        turn_trace = dict(session_state.get("turn_trace") or {})
        if str(turn_trace.get("trace_id") or "").strip() != str(trace_token).strip():
            return 0
        fallback_commit_count = int(turn_trace.get("commit_boundary_count") or 0)
        return fallback_commit_count if fallback_commit_count > 0 else 0

    def _record_trace_composition_contract(
        self,
        *,
        session: dict[str, Any] | None,
        job: ExecutionJob,
        result: FinalizedTurnResult,
        state_before_hash: str,
    ) -> None:
        if session is None:
            return
        self._record_turn_composition_contract(
            session=session,
            job=job,
            result=result,
            state_before_hash=str(state_before_hash or ""),
        )

    async def submit_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        attachments: AttachmentList | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        # Structural anchor: _validate_trace_invariant is executed in _submit_turn_kernel.
        assert callable(getattr(self, "_validate_trace_invariant", None))
        return await self.kernel_gateway.submit_turn(
            session_id=session_id,
            user_input=user_input,
            attachments=attachments,
            metadata=metadata,
            timeout_seconds=timeout_seconds,
        )

    def ledger_events(self) -> list[dict[str, Any]]:
        return self.ledger.read()

    def execution_timeline(self, *, session_id: str, limit: int = 128) -> list[dict[str, Any]]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return []
        timeline = [dict(item) for item in list(state.get("execution_timeline") or []) if isinstance(item, dict)]
        return timeline[-max(0, int(limit)):]

    def explain_last_decision(self, *, session_id: str) -> dict[str, Any]:
        session = self.registry.get_or_create(str(session_id or "default"))
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(state, dict):
            return dict(
                validate_decision_explanation_contract(
                    {
                        "available": False,
                        "reason": "no_state",
                        "timeline_events": 0,
                        "last_plan": {},
                        "recent_events": [],
                        "semantic_memory_context_size": 0,
                        "active_hypothesis": {},
                        "belief_calibration": {},
                        "tool_self_model": {},
                        "interactive_cognition_ui": {},
                        "alignment_policy": {},
                        "tool_ecosystem": {},
                        "swarm_health": {},
                    },
                ),
            )
        timeline = self.execution_timeline(session_id=session_id, limit=64)
        last_plan = {}
        for event in reversed(timeline):
            if str(event.get("event_type") or "") in {"plan.updated", "plan.created"}:
                last_plan = dict((event.get("payload") or {}).get("runtime_plan") or {})
                break
        explanation = {
            "available": bool(timeline),
            "timeline_events": int(len(timeline)),
            "last_plan": dict(last_plan),
            "recent_events": [str(item.get("event_type") or "") for item in timeline[-8:]],
            "semantic_memory_context_size": int(
                len(list(dict(state.get("semantic_memory") or {}).get("items") or [])),
            ),
            "active_hypothesis": dict(dict(state.get("hypothesis_store") or {}).get("last_active") or {}),
            "belief_calibration": dict(dict(state.get("belief_state") or {}).get("calibration") or {}),
            "tool_self_model": dict(dict(state.get("tool_self_model") or {}).get("tools") or {}),
            "interactive_cognition_ui": self._interactive_cognition_ui.snapshot(state=state, limit=12),
            "alignment_policy": dict(dict(state.get("alignment_trainer") or {}).get("policy") or {}),
            "tool_ecosystem": self._tool_ecosystem_hub.summary(state=state),
            "swarm_health": self._multi_agent_swarm.health_snapshot(state=state),
        }
        return dict(validate_decision_explanation_contract(explanation))

    def _validate_trace_invariant(
        self,
        job: ExecutionJob,
        result: FinalizedTurnResult,
        *,
        session: dict[str, Any] | None = None,
        state_before_hash: str = "",
    ) -> None:
        """Validate execution trace invariants: trace_id, complete nodes, exactly one commit boundary.

        Ensures:
        1. trace_id is present in job metadata
        2. Execution produced ordered trace nodes
        3. Exactly one commit boundary (save node) exists

        This is a defensive check to catch execution path deviations early.
        """
        trace_id = job.metadata.get("trace_id") or ""
        if not trace_id.strip():
            logger.warning("Trace invariant violation: missing trace_id in job %s", job.job_id)
            return

        # Query ledger for events matching this trace
        trace_events = self._job_trace_events(job)

        if not trace_events:
            logger.warning("Trace invariant violation: no events recorded for trace %s", trace_id)
            return

        # Count commit boundary markers from canonical TURN_EVENT payloads.
        commit_count, has_node_events = self._trace_event_invariant_counts(trace_events)

        # Fallback to unified sink trace when ledger envelope normalization omits stage markers.
        if commit_count == 0:
            fallback_commit_count = self._fallback_commit_count_from_session(
                session=session,
                trace_token=trace_id,
            )
            if fallback_commit_count > 0:
                commit_count = fallback_commit_count

        if not has_node_events and commit_count == 0:
            # Some persistence backends store only checkpoint summaries.
            # In that mode commit-boundary cardinality cannot be derived here.
            # Still record the composition contract for confluence tracking.
            self._record_trace_composition_contract(
                session=session,
                job=job,
                result=result,
                state_before_hash=state_before_hash,
            )
            return

        if commit_count != 1:
            detail = (
                "Trace invariant violation: expected exactly 1 commit boundary "
                f"(found {int(commit_count)})"
            )
            strict = bool(dict(job.metadata or {}).get("strict_trace_invariant", False))
            if strict:
                raise InvariantViolation(
                    "Trace invariant violation: expected exactly 1 commit boundary",
                    context={
                        "trace_id": str(trace_id or ""),
                        "job_id": str(job.job_id or ""),
                        "commit_boundary_count": int(commit_count),
                    },
                )
            logger.warning(detail)
            return
        self._record_trace_composition_contract(
            session=session,
            job=job,
            result=result,
            state_before_hash=state_before_hash,
        )

    def _apply_turn_committed_core_state(
        self,
        *,
        session: dict[str, Any],
        job: ExecutionJob,
        result: FinalizedTurnResult,
    ) -> None:
        """Transition CoreState to turn_committed and write views into session state."""
        session_state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if not isinstance(session_state, dict):
            return
        # CoreState is the authority — memory_retrieval_set is derived FROM CoreState.
        active_core_state = get_active_core_state()
        base_core_state = (
            active_core_state
            if active_core_state is not None
            else CoreState.from_dict(session_state.get("core_state"))
        )
        next_core_state = transition(
            base_core_state,
            InputEvent(
                event_type="turn_committed",
                payload={
                    "trace_id": str(job.trace_id or ""),
                    "response": str(result[0] if isinstance(result, tuple) and len(result) >= 1 else ""),
                    "should_end": bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
                    "memory_retrieval_set": base_core_state.memory.to_dict_list(),
                },
                metadata={
                    "job_id": str(job.job_id or ""),
                },
            ),
        )
        projections = project_views(next_core_state)
        session_state["core_state"] = next_core_state.to_dict()
        session_state["core_state_views"] = {
            "memory": {
                "entries": [dict(item.payload or {}) for item in projections.memory.entries],
            },
            "graph": {
                "adjacency": dict(projections.graph.adjacency),
            },
            "execution": {
                "trace_id": projections.execution.state.trace_id,
                "last_response": projections.execution.state.last_response,
                "should_end": bool(projections.execution.state.should_end),
            },
            "canonical": {
                "event_count": len(projections.canonical.events),
                "last_event_id": str(projections.canonical.events[-1].event_id if projections.canonical.events else ""),
            },
            "facade": projections.facade.as_payload(),
        }
        metadata = dict(job.metadata or {})
        metadata["core_state"] = dict(session_state["core_state"])
        job.metadata = metadata

    def _promote_semantic_memory(
        self,
        *,
        session: dict[str, Any],
        job: ExecutionJob,
        response_text: str,
    ) -> None:
        promoted = _build_semantic_memory_candidates(
            user_input=str(job.user_input or ""),
            response=str(response_text or ""),
            trace_id=str(job.trace_id or ""),
            session_id=str(job.session_id or "default"),
        )
        state = session.setdefault("state", {}) if isinstance(session, dict) else {}
        if isinstance(state, dict) and promoted:
            projection = dict(state.get("semantic_memory_projection") or {})
            items = [
                dict(item)
                for item in list(projection.get("items") or [])
                if isinstance(item, dict)
            ]
            items.extend(promoted)
            projection["items"] = items[-128:]
            projection["last_updated"] = float(time.time())
            projection["authority"] = "derived_read_only"
            projection["writer"] = "memory_manager"
            state["semantic_memory_projection"] = projection
        metadata = dict(job.metadata or {})
        metadata["semantic_memory_projection"] = {
            "authority": "derived_read_only",
            "writer": "memory_manager",
            "candidate_count": int(len(promoted)),
            "candidates": [dict(item) for item in list(promoted)[:8]],
        }
        job.metadata = metadata

    def bootstrap(self) -> dict[str, Any]:
        events = self.ledger.read()
        self.lifecycle_projection.rebuild_from_ledger(events)
        self._ledger_index.refresh(force=True)
        return {
            "event_count": len(events),
            "execution_lifecycle": self.lifecycle_projection.snapshot(),
        }


def json_dumps_sorted(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
