from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json as _json
import logging
import os
import platform
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from typing import Any, cast

from dadbot.contracts import (
    AttachmentList,
    FinalizedTurnResult,
    GenericSovereignPayload,
    LogicBranchPayload,
    PlannerDecisionPayload,
    PolicyVetoPayload,
    SovereignEvent,
    SovereignEventType,
)
from dadbot.core import shadow_mode as _shadow_mode
from dadbot.core.composite_friction import CompositeFrictionEngine, FrictionSignals
from dadbot.core.control_plane import (
    ExecutionControlPlane,
    ExecutionJob,
    SessionRegistry,
)
from dadbot.core.execution_binder import TraceBinder
from dadbot.core.execution_result_unified import ensure_unified_execution_result
from dadbot.core.execution_terminal_state import build_execution_terminal_state
from dadbot.core.goal_resynthesis import GoalRecalibrationEngine
from dadbot.core.graph import TurnGraph
from dadbot.core.graph_context import TurnContext
from dadbot.core.interfaces import (
    HealthService,
    InferenceService,
    validate_pipeline_services,
)
from dadbot.core.job_builder import JobBuilder
from dadbot.core.persistence.base import CheckpointNotFoundError
from dadbot.core.reflection_ir import DriftReflectionEngine
from dadbot.core.runtime_errors import (
    NON_FATAL_RUNTIME_EXCEPTIONS,
    ExecutionStageError,
    InvariantViolation,
)
from dadbot.registry import ServiceRegistry, boot_registry

logger = logging.getLogger(__name__)


class DeterminismViolationError(RuntimeError):
    """Retained for import compatibility with existing test files."""


DeterminismViolation = DeterminismViolationError


def _stable_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode(
            "utf-8",
        ),
    ).hexdigest()


def _normalize_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _dependency_versions_snapshot() -> dict[str, str]:
    deps: dict[str, str] = {}
    for name in ("pytest", "pydantic", "langchain", "openai"):
        try:
            deps[name] = str(importlib_metadata.version(name))
        except importlib_metadata.PackageNotFoundError:
            logger.debug("Dependency %s is not installed; skipping determinism manifest pin", name)
            continue
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.debug("Dependency version probe failed for %s: %s", name, exc)
            continue
    return deps


def _build_determinism_manifest() -> dict[str, Any]:
    env_keys = sorted(str(key) for key in os.environ.keys())
    env_hash = _stable_sha256({"env_keys": env_keys})
    manifest = {
        "python_version": str(platform.python_version()),
        "env_hash": env_hash,
        "dependency_versions": _dependency_versions_snapshot(),
        "timezone": str(os.environ.get("TZ") or "local"),
    }
    manifest["manifest_hash"] = _stable_sha256(manifest)
    return manifest


def _normalize_timeout_seconds(timeout_seconds: float | None) -> float:
    if timeout_seconds is None:
        return 30.0
    return max(0.0, float(timeout_seconds))


class DadBotOrchestrator:
    """Graph-governed turn orchestrator routed through ExecutionControlPlane."""

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
        *,
        strict: bool = False,
        enable_observability: bool = True,
        **kwargs: Any,
    ) -> None:
        config_path = str(kwargs.pop("config_path", "config.yaml"))
        bot = kwargs.pop("bot", None)
        checkpointer = kwargs.pop("checkpointer", None)
        self.bot = bot
        self.registry = registry or boot_registry(config_path=config_path, bot=bot)
        self._strict = bool(strict)
        self._enable_observability = bool(enable_observability)
        self._last_turn_context: TurnContext | None = None
        self._checkpointer = checkpointer

        storage = self.registry.get("storage", optional=True)
        if checkpointer is not None and storage is not None and callable(getattr(storage, "set_checkpointer", None)):
            storage.set_checkpointer(checkpointer)

        self._job_builder = JobBuilder()
        self._trace_binder = TraceBinder()
        self.graph = self._build_turn_graph()
        self.control_plane = ExecutionControlPlane(
            registry=SessionRegistry(),
            kernel_executor=self._execute_job,
            graph=self.graph,
            enable_observability=self._enable_observability,
        )
        self.session_registry = self.control_plane.registry
        self._reflection_engine = DriftReflectionEngine()  # Lazy initialization; ledger path set per-session
        self._friction_engine = CompositeFrictionEngine()
        self._goal_recalibration_engine = GoalRecalibrationEngine()

    def _goal_alignment_guard_enabled(self) -> bool:
        config = getattr(self.bot, "config", None)
        if config is not None:
            explicit = getattr(config, "goal_alignment_guard_enabled", None)
            if explicit is not None:
                return bool(explicit)
            runtime_config = getattr(config, "runtime_config", None)
            if runtime_config is not None:
                runtime_flag = getattr(runtime_config, "goal_alignment_guard_enabled", None)
                if runtime_flag is not None:
                    return bool(runtime_flag)
        return bool(
            str(os.environ.get("DADBOT_GOAL_ALIGNMENT_GUARD_ENABLED", "1")).strip().lower()
            not in {"0", "false", "no", "off"}
        )

    def _resolve_ledger_path(self, session_id: str) -> str | None:
        """
        Resolve the path to relational_ledger.jsonl for the given session.

        Returns:
            Full path to ledger, or None if session log directory cannot be resolved.
        """
        try:
            # Try to get session log dir from bot
            if self.bot is not None:
                session_log_dir = getattr(self.bot, "SESSION_LOG_DIR", None)
                if session_log_dir is not None:
                    import pathlib
                    ledger_path = pathlib.Path(session_log_dir) / "relational_ledger.jsonl"
                    return str(ledger_path)
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.debug("Failed to resolve ledger path: %s", exc)
        return None

    def _analyze_behavioral_reflection(self, session_id: str, context: TurnContext) -> None:
        """
        Analyze behavioral patterns from the relational ledger and attach reflection summary to context.

        This enriches context.state with reflection data for use in HUD rendering and adaptive interventions.
        """
        ledger_path = self._resolve_ledger_path(session_id)
        if ledger_path is None:
            # Ledger not available; reflection summary remains unavailable
            return

        try:
            self._reflection_engine.ledger_path = ledger_path
            reflection_summary = self._reflection_engine.analyze_ledger()
            evidence_graph = self._reflection_engine.get_evidence_graph_snapshot(max_edges=12)

            # Store reflection summary in context for downstream use
            context.state["reflection_summary"] = {
                "current_risk_level": reflection_summary.current_risk_level,
                "predicted_drift_probability": reflection_summary.predicted_drift_probability,
                "likely_trigger_category": reflection_summary.likely_trigger_category,
                "recommended_intervention": reflection_summary.recommended_intervention,
                "intervention_justification": reflection_summary.intervention_justification,
                "confidence_score": reflection_summary.confidence_score,
                "observable_signals": reflection_summary.observable_signals,
                "recent_episode_count": reflection_summary.recent_episode_count,
                "primary_pattern_name": (
                    reflection_summary.primary_pattern.pattern_name
                    if reflection_summary.primary_pattern
                    else None
                ),
                "primary_pattern_confidence": (
                    reflection_summary.primary_pattern.confidence
                    if reflection_summary.primary_pattern
                    else 0.0
                ),
                "evidence_graph": evidence_graph,
            }
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.debug("Behavioral reflection analysis failed (non-fatal): %s", exc)

    def _analyze_composite_friction(self, context: TurnContext) -> None:
        """Compute friction signals and attach goal re-synthesis guidance to context."""
        try:
            state = context.state if isinstance(context.state, dict) else {}
            reflection = dict(state.get("reflection_summary") or {})
            temporal_budget = dict(state.get("temporal_budget") or {})
            session_turn_count = int(temporal_budget.get("turn_index") or 0)

            halt_streak = int(state.get("goal_alignment_diversion_streak") or 0)
            if bool(state.get("goal_alignment_mandatory_halt", False)) and halt_streak < 3:
                halt_streak = 3

            risk_level = str(reflection.get("current_risk_level") or "moderate").strip().lower()
            recovery_success_defaults = {
                "critical": 0.20,
                "high": 0.40,
                "moderate": 0.65,
                "low": 0.85,
            }
            recovery_success_rate = float(recovery_success_defaults.get(risk_level, 0.65))

            topic_drift_frequency = float(reflection.get("predicted_drift_probability") or 0.0)
            recent_episodes = int(reflection.get("recent_episode_count") or 0)
            recurring_topic_patterns = max(0, min(3, recent_episodes // 2))
            if reflection.get("primary_pattern_name"):
                recurring_topic_patterns = max(recurring_topic_patterns, 1)

            goals_raw = state.get("session_goals")
            if not isinstance(goals_raw, list):
                goals_raw = state.get("goals")
            unresolved_objectives_count = 0
            for item in list(goals_raw or []):
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status") or "").strip().lower()
                if status not in {"done", "completed", "closed"}:
                    unresolved_objectives_count += 1

            checkpoint_stability = 1.0 if bool(getattr(context, "last_checkpoint_hash", "")) else 0.65
            signals = FrictionSignals(
                halt_streak=halt_streak,
                recovery_success_rate=recovery_success_rate,
                topic_drift_frequency=topic_drift_frequency,
                recurring_topic_patterns=recurring_topic_patterns,
                session_turn_count=session_turn_count,
                unresolved_objectives_count=unresolved_objectives_count,
                checkpoint_stability=checkpoint_stability,
            )
            friction = self._friction_engine.compute_friction(signals)
            friction_payload = {
                "composite_score": friction.composite_score,
                "risk_level": friction.risk_level,
                "should_trigger_re_synthesis": friction.should_trigger_re_synthesis,
                "individual_signals": dict(friction.individual_signals),
                "confidence": friction.confidence,
                "primary_friction_factor": friction.primary_friction_factor,
                "recommended_intervention": friction.recommended_intervention,
            }
            state["friction_analysis"] = friction_payload

            goals = [dict(item) for item in list(goals_raw or []) if isinstance(item, dict)]
            resynthesis = self._goal_recalibration_engine.synthesize(
                goals=goals,
                friction_analysis=friction_payload,
                reflection_summary=reflection or None,
            )
            state["goal_resynthesis"] = self._goal_recalibration_engine.to_context_payload(resynthesis)
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.debug("Composite friction analysis failed (non-fatal): %s", exc)

    def _build_turn_graph(self) -> TurnGraph:
        llm = self.registry.get("llm", optional=True)
        health = self.registry.get("health", optional=True)
        validate_pipeline_services(
            {
                "llm": (llm, InferenceService),
                "health": (health, HealthService),
            },
            raise_on_failure=self._strict,
        )
        return TurnGraph(
            registry=self.registry,
        )

    def _build_turn_context(
        self,
        *,
        user_input: str,
        attachments: AttachmentList | None,
        session_id: str,
        metadata: dict[str, Any] | None,
    ) -> TurnContext:
        """Backward-compat shim — delegates to JobBuilder.build."""
        return self._job_builder.build(
            user_input=user_input,
            attachments=attachments,
            session_id=session_id,
            metadata=metadata,
        )

    def _emit_sovereign_event(
        self,
        context: TurnContext,
        *,
        event_type: SovereignEventType,
        payload: dict[str, Any],
    ) -> SovereignEvent:
        previous_checksum = str(context.metadata.get("sovereign_event_checksum") or "")
        if event_type == SovereignEventType.PLANNER_DECISION:
            sovereign_payload = PlannerDecisionPayload(**payload)
        elif event_type == SovereignEventType.POLICY_VETO:
            sovereign_payload = PolicyVetoPayload(**payload)
        elif event_type == SovereignEventType.LOGIC_BRANCH:
            sovereign_payload = LogicBranchPayload(**payload)
        else:
            sovereign_payload = GenericSovereignPayload(data=dict(payload or {}))

        event = SovereignEvent(
            turn_id=str(context.trace_id or ""),
            event_type=event_type.value,
            payload=sovereign_payload,
            previous_checksum=previous_checksum,
        )
        stream = list(context.state.get("sovereign_events") or [])
        stream.append(event.to_ledger_event())
        context.state["sovereign_events"] = stream
        context.metadata["sovereign_event_checksum"] = event.checksum
        context.metadata["sovereign_event_count"] = len(stream)
        return event

    # ------------------------------------------------------------------
    # _execute_job helpers
    # ------------------------------------------------------------------

    def _check_manifest_drift(
        self,
        manifest: dict[str, Any],
        prior_manifest: dict[str, Any],
    ) -> None:
        """Check for runtime drift between current and prior manifest."""
        if not prior_manifest:
            return
        prior_py = str(prior_manifest.get("python_version") or "")
        prior_env = str(prior_manifest.get("env_hash") or "")
        current_py = str(manifest.get("python_version") or "")
        current_env = str(manifest.get("env_hash") or "")
        py_drift = prior_py and current_py and prior_py != current_py
        env_drift = prior_env and current_env and prior_env != current_env
        if py_drift or env_drift:
            message = (
                "Manifest drift detected before turn execution: "
                f"python_version {prior_py!r}->{current_py!r}, "
                f"env_hash {prior_env!r}->{current_env!r}"
            )
            if self._strict:
                raise DeterminismViolation(message)
            logger.warning(message)

    def _load_checkpoint_data(
        self,
        session_id: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Load checkpoint if checkpointer is configured; return ``None`` if absent."""
        if self._checkpointer is None:
            return None
        load_checkpoint = getattr(self._checkpointer, "load_checkpoint", None)
        if not callable(load_checkpoint):
            return None
        try:
            return cast(
                "dict[str, Any]",
                load_checkpoint(session_id, current_manifest=manifest, strict=bool(self._strict)),
            )
        except CheckpointNotFoundError:
            return None
        except DeterminismViolation:
            raise
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            if self._strict:
                raise ExecutionStageError(
                    "Checkpoint manifest verification failed",
                    context={"error": str(exc)},
                ) from exc
            return None

    def _stamp_determinism_metadata(
        self,
        context: TurnContext,
        job: ExecutionJob,
        manifest: dict[str, Any],
    ) -> None:
        """Compute and attach all determinism hashes to *context.metadata*."""
        determinism = dict(context.metadata.get("determinism") or {})
        determinism["manifest"] = dict(manifest)
        determinism["manifest_hash"] = str(manifest.get("manifest_hash") or "")
        determinism["determinism_manifest_hash"] = str(manifest.get("manifest_hash") or "")
        tool_trace_hash = _stable_sha256(
            {
                "user_input": str(context.user_input or ""),
                "attachments": list(getattr(job, "attachments", None) or []),
                "tool_plan": context.state.get("tool_ir") or context.state.get("tool_plan") or [],
            },
        )
        lock_hash = _stable_sha256(
            {
                "user_input": str(context.user_input or ""),
                "attachments": list(getattr(job, "attachments", None) or []),
                "manifest_hash": str(manifest.get("manifest_hash") or ""),
                "tool_trace_hash": tool_trace_hash,
            },
        )
        # Stable memory fingerprint — a hash over the current memory retrieval set.
        memory_fingerprint = _stable_sha256(
            {"retrieval_set": context.state.get("memory_retrieval_set") or []},
        )
        determinism["tool_trace_hash"] = tool_trace_hash
        determinism["lock_hash"] = lock_hash
        determinism["lock_version"] = max(2, int(determinism.get("lock_version") or 2))
        determinism["memory_fingerprint"] = memory_fingerprint
        determinism["env_hash"] = str(manifest.get("env_hash") or "")
        determinism["enforced"] = bool(self._strict)
        context.metadata["determinism"] = determinism
        context.metadata["determinism_manifest"] = dict(manifest)

    def _update_session_state_after_turn(  # noqa: C901
        self,
        session: dict[str, Any],
        context: TurnContext,
        result: FinalizedTurnResult,
        manifest: dict[str, Any],
    ) -> None:
        """Stamp last_result, terminal state, goals, and manifest into session state."""
        state = session.setdefault("state", {})
        if not isinstance(state, dict):
            return
        typed_state = cast("dict[str, Any]", state)
        typed_state["last_result"] = result
        typed_state["execution_result"] = ensure_unified_execution_result(
            dict(context.metadata.get("execution_result") or {}),
        )
        prior_goals = typed_state.get("goals")
        persisted_goals = list(cast("list[Any]", prior_goals)) if isinstance(prior_goals, list) else []
        goals_any_raw = context.state.get("goals")
        if isinstance(goals_any_raw, list):
            persisted_goals = list(cast("list[Any]", goals_any_raw))
        else:
            new_goals_raw = context.state.get("new_goals")
            if isinstance(new_goals_raw, list) and new_goals_raw:
                seen_ids = {
                    str(item.get("id") or "")
                    for item in persisted_goals
                    if isinstance(item, dict)
                }
                for raw in new_goals_raw:
                    if not isinstance(raw, dict):
                        continue
                    goal_id = str(raw.get("id") or "")
                    if goal_id and goal_id in seen_ids:
                        continue
                    persisted_goals.append(dict(raw))
                    if goal_id:
                        seen_ids.add(goal_id)
        typed_state["goals"] = persisted_goals
        terminal_state = build_execution_terminal_state(
            context,
            finalized_result=result,
        )
        typed_state["last_terminal_state"] = terminal_state.to_dict()
        typed_state["last_execution_trace_context"] = {
            "final_hash": str(terminal_state.final_trace_hash or context.trace_id or ""),
        }
        typed_state["last_determinism_manifest"] = dict(manifest)
        memory_snapshot = dict(context.state.get("memory_snapshot") or {})
        typed_state["last_memory_full_history_id"] = str(
            memory_snapshot.get("memory_full_history_id") or "",
        )
        typed_state["last_checkpoint_hash"] = str(getattr(context, "last_checkpoint_hash", "") or "")
        typed_state["prev_checkpoint_hash"] = str(getattr(context, "prev_checkpoint_hash", "") or "")
        typed_state["goal_alignment_guard_enabled"] = bool(context.state.get("goal_alignment_guard_enabled", False))
        typed_state["goal_alignment_diversion_streak"] = int(context.state.get("goal_alignment_diversion_streak") or 0)
        typed_state["goal_alignment_mandatory_halt"] = bool(context.state.get("goal_alignment_mandatory_halt", False))
        typed_state["cognition_stream"] = list(context.state.get("cognition_stream") or [])
        typed_state["subconscious_memory_fragments"] = list(context.state.get("subconscious_memory_fragments") or [])
        typed_state["memory_retrieval_set"] = list(context.state.get("memory_retrieval_set") or [])
        typed_state["last_integrity_status"] = {
            "merkle_check_passed": not bool(context.metadata.get("integrity_failure", False)),
            "reason": str(context.metadata.get("integrity_failure_reason") or ""),
            "diagnostics": dict(context.metadata.get("integrity_failure_diagnostics") or {}),
        }
        # Persist reflection summary if available
        if "reflection_summary" in context.state:
            typed_state["last_reflection_summary"] = dict(context.state["reflection_summary"])
        if "friction_analysis" in context.state:
            typed_state["last_friction_analysis"] = dict(context.state["friction_analysis"])
        if "goal_resynthesis" in context.state:
            typed_state["last_goal_resynthesis"] = dict(context.state["goal_resynthesis"])

    def _publish_health_evidence(self, context: TurnContext) -> None:
        """Populate bot._last_turn_health_evidence and bot._last_capability_audit_report from context."""
        stage_order = [
            str(getattr(t, "stage", "") or "").strip().lower()
            for t in list(getattr(context, "stage_traces", []) or [])
        ]
        mutation_queue_snapshot = {}
        try:
            mutation_queue_snapshot = dict(context.mutation_queue.snapshot())
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.debug("Failed to snapshot mutation queue for health evidence: %s", exc)
        evidence = {
            "trace_id": str(getattr(context, "trace_id", "") or ""),
            "stage_order": stage_order,
            "mutation_queue": mutation_queue_snapshot,
        }
        # Write to bot if accessible
        bot = getattr(self, "bot", None)
        if bot is not None:
            bot._last_turn_health_evidence = evidence
            audit_report = dict(context.state.get("capability_audit_report") or {})
            if audit_report:
                bot._last_capability_audit_report = audit_report

    def _subconscious_checksum(self, fragment: dict[str, Any]) -> str:
        payload = {
            "summary": str(fragment.get("summary") or "").strip(),
            "timestamp": str(fragment.get("timestamp") or "").strip(),
            "category": str(fragment.get("category") or "").strip(),
            "mood": str(fragment.get("mood") or "").strip(),
            "event": str(fragment.get("event") or "").strip(),
        }
        return _stable_sha256(payload)

    def _apply_integrity_failure(
        self,
        context: TurnContext,
        *,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        resolved_reason = str(reason or "integrity_failure").strip() or "integrity_failure"
        context.state["refusal_state"] = "integrity_failure"
        context.state["integrity_failure_reason"] = resolved_reason
        context.state["should_end"] = True
        context.metadata["integrity_failure"] = True
        context.metadata["integrity_failure_reason"] = resolved_reason
        if diagnostics:
            context.metadata["integrity_failure_diagnostics"] = dict(diagnostics)

    @staticmethod
    def _collect_sovereign_checksums(payload: Any) -> set[str]:
        checksums: set[str] = set()
        stack: list[Any] = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                checksum = str(item.get("checksum") or "").strip()
                previous_checksum = str(item.get("previous_checksum") or "").strip()
                event_type = str(item.get("event_type") or "").strip()
                if checksum and (previous_checksum or event_type):
                    checksums.add(checksum)
                for value in item.values():
                    if isinstance(value, (dict, list, tuple)):
                        stack.append(value)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        return checksums

    def _authoritative_sovereign_checksum_set(self, context: TurnContext) -> set[str]:
        checksums: set[str] = set()
        checksums |= self._collect_sovereign_checksums(context.state.get("sovereign_events") or [])

        ledger = getattr(getattr(self, "control_plane", None), "ledger", None)
        read_events = getattr(ledger, "read", None)
        if callable(read_events):
            try:
                events = read_events(full=False)
            except TypeError:
                events = read_events()
            except NON_FATAL_RUNTIME_EXCEPTIONS:
                events = []
            checksums |= self._collect_sovereign_checksums(events)

        checksum_head = str(context.metadata.get("sovereign_event_checksum") or "").strip()
        if checksum_head:
            checksums.add(checksum_head)
        return checksums

    @staticmethod
    def _fragment_sovereign_checksum(memory: dict[str, Any]) -> str:
        candidate = (
            memory.get("sovereign_event_checksum")
            or memory.get("signed_sovereign_checksum")
            or memory.get("ledger_checksum")
            or memory.get("event_checksum")
            or memory.get("source_event_checksum")
            or ""
        )
        return str(candidate or "").strip()

    def _run_subconscious_retrieval(self, context: TurnContext, user_input: str) -> None:
        bot = self.bot
        if bot is None:
            return
        memory_catalog = getattr(bot, "memory_catalog", None)
        semantic_matches = getattr(bot, "semantic_memory_matches", None)
        if not callable(memory_catalog) or not callable(semantic_matches):
            return

        raw_memories = memory_catalog()
        memories = list(raw_memories) if isinstance(raw_memories, list | tuple) else []
        if not memories:
            context.state["subconscious_memory_fragments"] = []
            return

        try:
            raw_matches = semantic_matches(user_input, memories, limit=9)
            matches = list(raw_matches) if isinstance(raw_matches, list | tuple) else []
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.debug("Subconscious retrieval unavailable (non-fatal): %s", exc)
            context.state["subconscious_memory_fragments"] = []
            return

        cutoff = datetime.now(timezone.utc) - timedelta(days=180)
        vetted: list[dict[str, Any]] = []
        dropped_integrity = 0
        handshake_checksums = self._authoritative_sovereign_checksum_set(context)
        if not handshake_checksums:
            self._apply_integrity_failure(
                context,
                reason="integrity_handshake_ledger_unavailable",
                diagnostics={"stage": "subconscious_retrieval"},
            )
            context.state["subconscious_memory_fragments"] = []
            context.state["memory_retrieval_set"] = []
            context.metadata["subconscious_retrieval"] = {
                "query": str(user_input or ""),
                "retrieved": 0,
                "dropped_integrity": 0,
                "max_age_days": 180,
                "similarity_threshold": 0.2,
                "aborted": True,
                "abort_reason": "integrity_handshake_ledger_unavailable",
            }
            return

        for raw in matches:
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                continue
            similarity_raw, memory = raw[0], raw[1]
            if not isinstance(memory, dict):
                continue
            try:
                similarity = float(similarity_raw)
            except (TypeError, ValueError):
                similarity = 0.0
            if similarity < 0.2:
                continue

            timestamp_raw = (
                memory.get("updated_at")
                or memory.get("created_at")
                or memory.get("timestamp")
                or ""
            )
            parsed_timestamp = _normalize_timestamp(timestamp_raw)
            if parsed_timestamp is None or parsed_timestamp < cutoff:
                continue

            fragment = {
                "memory_id": str(
                    memory.get("id")
                    or memory.get("memory_id")
                    or memory.get("slug")
                    or memory.get("key")
                    or ""
                ),
                "summary": str(memory.get("summary") or memory.get("text") or "").strip(),
                "category": str(memory.get("category") or "").strip(),
                "mood": str(memory.get("mood") or "").strip(),
                "event": str(memory.get("event") or "").strip(),
                "timestamp": parsed_timestamp.isoformat().replace("+00:00", "Z"),
                "similarity_score": round(similarity, 4),
                "source": "subconscious_reflex",
            }
            if not fragment["summary"]:
                continue

            computed_checksum = self._subconscious_checksum(fragment)
            known_checksum = str(
                memory.get("checksum")
                or memory.get("content_hash")
                or memory.get("hash")
                or ""
            ).strip()
            if known_checksum and known_checksum != computed_checksum:
                dropped_integrity += 1
                self._apply_integrity_failure(
                    context,
                    reason="integrity_handshake_fragment_hash_mismatch",
                    diagnostics={
                        "memory_id": fragment["memory_id"],
                        "known_checksum": known_checksum,
                        "computed_checksum": computed_checksum,
                    },
                )
                context.state["subconscious_memory_fragments"] = []
                context.state["memory_retrieval_set"] = []
                context.metadata["subconscious_retrieval"] = {
                    "query": str(user_input or ""),
                    "retrieved": 0,
                    "dropped_integrity": dropped_integrity,
                    "max_age_days": 180,
                    "similarity_threshold": 0.2,
                    "aborted": True,
                    "abort_reason": "integrity_handshake_fragment_hash_mismatch",
                }
                return

            sovereign_checksum = self._fragment_sovereign_checksum(memory)
            if not sovereign_checksum:
                dropped_integrity += 1
                self._apply_integrity_failure(
                    context,
                    reason="integrity_handshake_missing_sovereign_checksum",
                    diagnostics={"memory_id": fragment["memory_id"]},
                )
                context.state["subconscious_memory_fragments"] = []
                context.state["memory_retrieval_set"] = []
                context.metadata["subconscious_retrieval"] = {
                    "query": str(user_input or ""),
                    "retrieved": 0,
                    "dropped_integrity": dropped_integrity,
                    "max_age_days": 180,
                    "similarity_threshold": 0.2,
                    "aborted": True,
                    "abort_reason": "integrity_handshake_missing_sovereign_checksum",
                }
                return

            if sovereign_checksum not in handshake_checksums:
                dropped_integrity += 1
                self._apply_integrity_failure(
                    context,
                    reason="integrity_handshake_sovereign_checksum_mismatch",
                    diagnostics={
                        "memory_id": fragment["memory_id"],
                        "sovereign_checksum": sovereign_checksum,
                    },
                )
                context.state["subconscious_memory_fragments"] = []
                context.state["memory_retrieval_set"] = []
                context.metadata["subconscious_retrieval"] = {
                    "query": str(user_input or ""),
                    "retrieved": 0,
                    "dropped_integrity": dropped_integrity,
                    "max_age_days": 180,
                    "similarity_threshold": 0.2,
                    "aborted": True,
                    "abort_reason": "integrity_handshake_sovereign_checksum_mismatch",
                }
                return

            fragment["checksum"] = computed_checksum
            fragment["sovereign_event_checksum"] = sovereign_checksum
            vetted.append(fragment)
            if len(vetted) >= 5:
                break

        context.state["subconscious_memory_fragments"] = list(vetted)
        context.state["memory_retrieval_set"] = list(vetted)
        context.metadata["subconscious_retrieval"] = {
            "query": str(user_input or ""),
            "retrieved": len(vetted),
            "dropped_integrity": dropped_integrity,
            "max_age_days": 180,
            "similarity_threshold": 0.2,
        }

    async def _execute_job(
        self,
        session: dict[str, Any],
        job: ExecutionJob,
    ) -> FinalizedTurnResult:
        context = self._job_builder.build(
            user_input=str(job.user_input or ""),
            attachments=job.attachments,
            session_id=str(job.session_id or "default"),
            metadata=dict(job.metadata or {}),
        )
        session_state = session.get("state")
        if isinstance(session_state, dict):
            session_goals = session_state.get("goals")
            if isinstance(session_goals, list):
                context.state["session_goals"] = list(session_goals)
            context.state["goal_alignment_diversion_streak"] = int(
                session_state.get("goal_alignment_diversion_streak") or 0
            )
            context.state["goal_alignment_mandatory_halt"] = bool(
                session_state.get("goal_alignment_mandatory_halt", False)
            )
        context.state["goal_alignment_guard_enabled"] = self._goal_alignment_guard_enabled()
        if not context.trace_id:
            raise InvariantViolation("TurnContext.trace_id must be non-empty")

        # Gideon Reflex: read-only subconscious retrieval before standard graph inference.
        self._run_subconscious_retrieval(context, str(job.user_input or ""))

        manifest = _build_determinism_manifest()
        prior_manifest = dict(session.get("state", {}).get("last_determinism_manifest") or {})
        self._check_manifest_drift(manifest, prior_manifest)

        loaded_checkpoint = self._load_checkpoint_data(str(job.session_id or "default"), manifest)
        if isinstance(loaded_checkpoint, dict) and loaded_checkpoint:
            context.last_checkpoint_hash = str(loaded_checkpoint.get("checkpoint_hash") or "")
            context.prev_checkpoint_hash = str(loaded_checkpoint.get("prev_checkpoint_hash") or "")

        self._stamp_determinism_metadata(context, job, manifest)
        self._emit_sovereign_event(
            context,
            event_type=SovereignEventType.LOGIC_BRANCH,
            payload={
                "branch_name": "orchestrator._execute_job",
                "condition": "graph.execute",
                "outcome": "start",
                "metadata": {
                    "session_id": str(job.session_id or "default"),
                    "job_id": str(job.job_id or ""),
                },
            },
        )

        # Analyze behavioral reflection patterns from relational ledger
        self._analyze_behavioral_reflection(str(job.session_id or "default"), context)
        self._analyze_composite_friction(context)

        async def _run() -> FinalizedTurnResult:
            result = await self.graph.execute(context)
            return result

        if self.bot is not None:
            setattr(self.bot, "_active_turn_context", context)
        try:
            try:
                result = await self._trace_binder.run(
                    trace_id=context.trace_id,
                    prompt=str(job.user_input or ""),
                    metadata={"session_id": str(job.session_id or "default")},
                    fn=_run,
                )
            except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                self._emit_sovereign_event(
                    context,
                    event_type=SovereignEventType.POLICY_VETO,
                    payload={
                        "policy_rule": "execution.kernel_failure",
                        "reason": str(exc),
                        "severity": "critical",
                        "metadata": {
                            "error_type": type(exc).__name__,
                            "session_id": str(job.session_id or "default"),
                        },
                    },
                )
                raise
            self._emit_sovereign_event(
                context,
                event_type=SovereignEventType.PLANNER_DECISION,
                payload={
                    "planner_node": "turn_graph",
                    "selected_branch": "execute.completed",
                    "rationale": "graph execution reached terminal result",
                    "metadata": {
                        "should_end": bool(result[1] if isinstance(result, tuple) and len(result) > 1 else False),
                    },
                },
            )
            self._last_turn_context = context
            self._update_session_state_after_turn(session, context, result, manifest)
            self._publish_health_evidence(context)
        finally:
            if self.bot is not None and getattr(self.bot, "_active_turn_context", None) is context:
                setattr(self.bot, "_active_turn_context", None)

        # Shadow mode — non-blocking observation hook (observation phase only)
        if _shadow_mode.is_enabled():
            try:
                _stage_traces = [
                    {"stage": t.stage, "duration_ms": t.duration_ms, "error": t.error}
                    for t in (context.stage_traces or [])
                ]
                _latency_ms = float(
                    (context.metadata or {}).get("turn_latency_ms") or 0.0
                )
                _final_output = str(result[0] if isinstance(result, tuple) and result else "")
                _shadow_snapshot = {
                    "inputs": {"prompt": str(job.user_input or "")},
                    "outputs_per_step": _stage_traces,
                    "final_output": _final_output,
                }
                _shadow_mode.shadow_log(
                    snapshot=_shadow_snapshot,
                    trace_id=str(context.trace_id or ""),
                    session_id=str(job.session_id or "default"),
                    event_count=len(_stage_traces),
                    latency_ms=_latency_ms,
                )
            except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
                logger.debug("Shadow mode logging failed (non-fatal): %s", exc)

        return result

    async def _submit_turn_via_control_plane(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        session_id: str = "default",
        confluence_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        normalized_timeout = _normalize_timeout_seconds(timeout_seconds)
        confluence_mode = str(os.environ.get("DADBOT_GLOBAL_CONFLUENCE_MODE", "enforce")).strip().lower()
        if confluence_mode not in {"off", "audit", "enforce"}:
            confluence_mode = "enforce"
        explicit_key = str(confluence_key or "").strip()
        allow_legacy = str(os.environ.get("DADBOT_ALLOW_LEGACY_CONFLUENCE_KEY", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if confluence_mode == "enforce" and not explicit_key:
            if not allow_legacy:
                raise InvariantViolation(
                    "Missing explicit confluence key in enforce mode.",
                    context={"session_id": str(session_id or "default")},
                )
            legacy_payload = {
                "session_id": str(session_id or "default"),
                "user_input": str(user_input or ""),
                "attachments": list(attachments or []),
            }
            explicit_key = f"legacy:{_stable_sha256(legacy_payload)}"
            logger.warning("Using legacy confluence fallback key because DADBOT_ALLOW_LEGACY_CONFLUENCE_KEY is enabled")
        outbound_metadata: dict[str, Any] = {"confluence_mode": confluence_mode}
        if explicit_key:
            outbound_metadata["confluence_key"] = explicit_key
        if isinstance(metadata, dict) and metadata:
            outbound_metadata.update(dict(metadata))
        return await self.control_plane.submit_turn(
            session_id=session_id,
            user_input=user_input,
            attachments=attachments,
            metadata=outbound_metadata,
            timeout_seconds=normalized_timeout,
        )

    async def handle_turn(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        session_id: str = "default",
        confluence_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        """Canonical async turn entry-point: all paths converge through control plane.

        This is the single authoritative execution path. No shortcuts or delegation branches.
        Every request produces: (1) complete ordered trace, (2) exactly one commit boundary.
        """
        try:
            normalized_timeout = _normalize_timeout_seconds(timeout_seconds)
            return await self._submit_turn_via_control_plane(
                user_input,
                attachments=attachments,
                session_id=session_id,
                confluence_key=confluence_key,
                metadata=metadata,
                timeout_seconds=normalized_timeout,
            )
        except TimeoutError:
            logger.warning("Inference timed out after %ss", _normalize_timeout_seconds(timeout_seconds))
            return ("Sorry, I timed out while thinking. Please try again.", False)
        except NON_FATAL_RUNTIME_EXCEPTIONS as exc:
            logger.error("Inference failed: %s", exc)
            return ("Something went wrong. Please try again.", False)

    def run(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        confluence_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FinalizedTurnResult:
        """Synchronous turn entry-point: delegates to canonical async path."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            if "no running event loop" not in str(exc).lower():
                raise
            loop = None
        coro = self.handle_turn(
            user_input,
            attachments=attachments,
            confluence_key=confluence_key,
            metadata=metadata,
        )
        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]
        return asyncio.run(coro)

    async def run_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        confluence_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FinalizedTurnResult:
        """Async variant of run(): delegates to canonical handle_turn()."""
        return await self.handle_turn(
            user_input,
            attachments=attachments,
            confluence_key=confluence_key,
            metadata=metadata,
        )
