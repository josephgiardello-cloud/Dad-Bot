from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json as _json
import logging
import os
import platform
from importlib import metadata as importlib_metadata
from collections.abc import Awaitable
from typing import Any, cast

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core import shadow_mode as _shadow_mode
from dadbot.core.control_plane import (
    ExecutionControlPlane,
    ExecutionJob,
    SessionRegistry,
)
from dadbot.core.execution_contract import TurnDelivery, TurnResponse, live_turn_request
from dadbot.core.graph import TurnGraph
from dadbot.core.graph_context import TurnContext
from dadbot.core.interfaces import (
    HealthService,
    InferenceService,
    validate_pipeline_services,
)
from dadbot.core.job_builder import JobBuilder
from dadbot.core.persistence.base import CheckpointNotFoundError
from dadbot.core.execution_binder import TraceBinder
from dadbot.core.execution_terminal_state import build_execution_terminal_state
from dadbot.core.reflection_ir import DriftReflectionEngine
from dadbot.core.composite_friction import CompositeFrictionEngine, FrictionSignals
from dadbot.core.goal_resynthesis import GoalRecalibrationEngine
from dadbot.registry import ServiceRegistry, boot_registry

logger = logging.getLogger(__name__)


class DeterminismViolation(RuntimeError):
    """Retained for import compatibility with existing test files."""


def _stable_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode(
            "utf-8",
        ),
    ).hexdigest()


def _dependency_versions_snapshot() -> dict[str, str]:
    deps: dict[str, str] = {}
    for name in ("pytest", "pydantic", "langchain", "openai"):
        try:
            deps[name] = str(importlib_metadata.version(name))
        except Exception:  # noqa: BLE001 — importlib_metadata probe; skip missing packages
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
        return bool(str(os.environ.get("DADBOT_GOAL_ALIGNMENT_GUARD_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"})

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
        except Exception as exc:
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
                "primary_pattern_name": reflection_summary.primary_pattern.pattern_name if reflection_summary.primary_pattern else None,
                "primary_pattern_confidence": reflection_summary.primary_pattern.confidence if reflection_summary.primary_pattern else 0.0,
                "evidence_graph": evidence_graph,
            }
        except Exception as exc:
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
        except Exception as exc:
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
        except Exception as exc:
            if self._strict:
                raise DeterminismViolation(
                    f"checkpoint manifest verification failed: {exc}",
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

    def _update_session_state_after_turn(
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
        final_output = str(result[0] if isinstance(result, tuple) else result or "")
        terminal_state = build_execution_terminal_state(
            context,
            finalized_result=result,
        )
        typed_state["last_terminal_state"] = terminal_state.to_dict()
        typed_state["last_execution_trace_context"] = {
            "final_hash": str(terminal_state.final_trace_hash or context.trace_id or ""),
        }
        typed_state["last_determinism_manifest"] = dict(manifest)
        typed_state["last_memory_full_history_id"] = str(context.state.get("memory_full_history_id") or "")
        typed_state["last_checkpoint_hash"] = str(getattr(context, "last_checkpoint_hash", "") or "")
        typed_state["prev_checkpoint_hash"] = str(getattr(context, "prev_checkpoint_hash", "") or "")
        typed_state["goal_alignment_guard_enabled"] = bool(context.state.get("goal_alignment_guard_enabled", False))
        typed_state["goal_alignment_diversion_streak"] = int(context.state.get("goal_alignment_diversion_streak") or 0)
        typed_state["goal_alignment_mandatory_halt"] = bool(context.state.get("goal_alignment_mandatory_halt", False))
        # Persist reflection summary if available
        if "reflection_summary" in context.state:
            typed_state["last_reflection_summary"] = dict(context.state["reflection_summary"])
        if "friction_analysis" in context.state:
            typed_state["last_friction_analysis"] = dict(context.state["friction_analysis"])
        if "goal_resynthesis" in context.state:
            typed_state["last_goal_resynthesis"] = dict(context.state["goal_resynthesis"])

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
            context.state["goal_alignment_diversion_streak"] = int(session_state.get("goal_alignment_diversion_streak") or 0)
            context.state["goal_alignment_mandatory_halt"] = bool(session_state.get("goal_alignment_mandatory_halt", False))
        context.state["goal_alignment_guard_enabled"] = self._goal_alignment_guard_enabled()
        if not context.trace_id:
            raise RuntimeError("TurnContext.trace_id must be non-empty")

        manifest = _build_determinism_manifest()
        prior_manifest = dict(session.get("state", {}).get("last_determinism_manifest") or {})
        self._check_manifest_drift(manifest, prior_manifest)

        loaded_checkpoint = self._load_checkpoint_data(str(job.session_id or "default"), manifest)
        if isinstance(loaded_checkpoint, dict) and loaded_checkpoint:
            context.last_checkpoint_hash = str(loaded_checkpoint.get("checkpoint_hash") or "")
            context.prev_checkpoint_hash = str(loaded_checkpoint.get("prev_checkpoint_hash") or "")

        self._stamp_determinism_metadata(context, job, manifest)

        # Analyze behavioral reflection patterns from relational ledger
        self._analyze_behavioral_reflection(str(job.session_id or "default"), context)
        self._analyze_composite_friction(context)

        async def _run() -> FinalizedTurnResult:
            result = await self.graph.execute(context)
            return result

        result = await self._trace_binder.run(
            trace_id=context.trace_id,
            prompt=str(job.user_input or ""),
            metadata={"session_id": str(job.session_id or "default")},
            fn=_run,
        )
        self._last_turn_context = context
        self._update_session_state_after_turn(session, context, result, manifest)

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
            except Exception as exc:  # noqa: BLE001
                logger.debug("Shadow mode logging failed (non-fatal): %s", exc)

        return result

    async def _submit_turn_via_control_plane(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        session_id: str = "default",
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        return await self.control_plane.submit_turn(
            session_id=session_id,
            user_input=user_input,
            attachments=attachments,
            metadata={},
            timeout_seconds=timeout_seconds,
        )

    def _can_delegate_to_bot_execute_turn(self) -> bool:
        if self.bot is None:
            return False
        direct = getattr(self.bot, "_turn_orchestrator", None)
        via_service = getattr(getattr(self.bot, "services", None), "turn_orchestrator", None)
        return self is direct or self is via_service

    async def handle_turn(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        session_id: str = "default",
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        execute_turn = getattr(self.bot, "execute_turn", None)
        if callable(execute_turn) and self._can_delegate_to_bot_execute_turn():
            response = await cast(
                Awaitable[TurnResponse],
                execute_turn(
                    live_turn_request(
                        user_input,
                        attachments=list(attachments or []),
                        delivery=TurnDelivery.ASYNC,
                        session_id=session_id,
                        timeout_seconds=timeout_seconds,
                    ),
                ),
            )
            return response.as_result()
        try:
            return await self._submit_turn_via_control_plane(
                user_input,
                attachments=attachments,
                session_id=session_id,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            logger.warning("Inference timed out after %ss", timeout_seconds or 30)
            return ("Sorry, I timed out while thinking. Please try again.", False)
        except Exception as exc:  # noqa: BLE001 — outermost inference guard; must return user-facing fallback
            logger.error("Inference failed: %s", exc)
            return ("Something went wrong. Please try again.", False)

    def run(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        execute_turn = getattr(self.bot, "execute_turn", None)
        if callable(execute_turn) and self._can_delegate_to_bot_execute_turn():
            response = cast(
                TurnResponse,
                execute_turn(
                    live_turn_request(
                        user_input,
                        attachments=list(attachments or []),
                        delivery=TurnDelivery.SYNC,
                        session_id="default",
                    ),
                ),
            )
            return response.as_result()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        coro = self.handle_turn(user_input, attachments=attachments)
        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]
        return asyncio.run(coro)

    async def run_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        execute_turn = getattr(self.bot, "execute_turn", None)
        if callable(execute_turn) and self._can_delegate_to_bot_execute_turn():
            response = await cast(
                Awaitable[TurnResponse],
                execute_turn(
                    live_turn_request(
                        user_input,
                        attachments=list(attachments or []),
                        delivery=TurnDelivery.ASYNC,
                        session_id="default",
                    ),
                ),
            )
            return response.as_result()
        return await self.handle_turn(user_input, attachments=attachments)
