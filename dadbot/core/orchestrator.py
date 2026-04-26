from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
from dadbot.core.graph import TurnContext, TurnGraph, TurnTemporalAxis
from dadbot.core.kernel import TurnKernel, bayesian_policy_gate
from dadbot.core.interfaces import (
    HealthService,
    InferenceService,
    MemoryService,
    PersistenceService,
    SafetyPolicyService,
    validate_pipeline_services,
)
from dadbot.core.nodes import ContextBuilderNode, HealthNode, InferenceNode, ReflectionNode, SafetyNode, SaveNode, TemporalNode
from dadbot.core.observability import CorrelationContext, configure_exporter
from dadbot.registry import ServiceRegistry, boot_registry

logger = logging.getLogger(__name__)


def _stable_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


class DadBotOrchestrator:
    """Refactored orchestrator that replaces monolithic turn orchestration."""

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
        *,
        config_path: str = "config.yaml",
        bot=None,
        strict: bool = False,
        enable_observability: bool = True,
    ):
        self.bot = bot
        self._strict = strict
        self.registry = registry or boot_registry(config_path=config_path, bot=bot)
        if enable_observability:
            configure_exporter(enabled=True)
        self.graph = self._build_turn_graph()
        self.session_registry = SessionRegistry()
        self.control_plane = ExecutionControlPlane(
            registry=self.session_registry,
            kernel_executor=self._execute_job,
        )
        self.scheduler = self.control_plane.scheduler
        set_required_execution_token = getattr(self.graph, "set_required_execution_token", None)
        if callable(set_required_execution_token):
            set_required_execution_token(self.control_plane.execution_token)
        set_execution_witness_emitter = getattr(self.graph, "set_execution_witness_emitter", None)
        if callable(set_execution_witness_emitter):
            set_execution_witness_emitter(self._emit_graph_execution_witness)

    def _emit_graph_execution_witness(self, component: str, context: TurnContext) -> None:
        trace_metadata = dict(context.metadata.get("trace") or {})
        control_plane_metadata = dict(context.metadata.get("control_plane") or {})
        self.control_plane.ledger_writer.append_execution_witness(
            component=component,
            session_id=str(control_plane_metadata.get("session_id") or "default"),
            trace_id=str(context.trace_id or ""),
            correlation_id=str(trace_metadata.get("correlation_id") or ""),
            payload={
                "phase": str(getattr(context.phase, "value", context.phase) or ""),
            },
        )

    def _build_turn_graph(self) -> TurnGraph:
        """Wires pipeline: TemporalNode → (Health || Memory) → Inference → Safety → Reflection → Save.

        TemporalNode is enforced as a sequential first stage. All subsequent stages
        (including the parallel preflight group) depend on temporal context being populated.
        """
        health = self.registry.get("health")
        memory = self.registry.get("memory")
        llm = self.registry.get("llm")
        safety = self.registry.get("safety")
        storage = self.registry.get("storage")
        reflection = self.registry.get("reflection")

        # Validate service contracts; raises in strict mode, warns otherwise.
        validate_pipeline_services({
            "health": (health, HealthService),
            "memory": (memory, MemoryService),
            "llm": (llm, InferenceService),
            "safety": (safety, SafetyPolicyService),
            "storage": (storage, PersistenceService),
        }, raise_on_failure=self._strict)

        graph = TurnGraph(registry=self.registry)
        # Stage 1: TemporalNode runs ALONE, sequentially, before anything else.
        # This is the absolute first stage. No fallback to system time is permitted.
        graph.add_node("temporal", TemporalNode())
        # Stage 2: HealthNode and ContextBuilderNode run in parallel AFTER temporal is resolved.
        graph.add_node(
            "preflight",
            (
                HealthNode(health),
                ContextBuilderNode(memory),
            ),
        )
        graph.add_node("inference", InferenceNode(llm))
        graph.add_node("safety", SafetyNode(safety))
        graph.add_node("reflection", ReflectionNode(reflection))
        graph.add_node("save", SaveNode(storage))

        graph.set_edge("temporal", "preflight")
        graph.set_edge("preflight", "inference")
        graph.set_edge("inference", "safety")
        graph.set_edge("safety", "reflection")
        graph.set_edge("reflection", "save")

        runtime_bot = getattr(llm, 'bot', None)
        if runtime_bot is not None:
            kernel = TurnKernel(policy_gate=bayesian_policy_gate(runtime_bot))
        else:
            kernel = TurnKernel()
        graph.set_kernel(kernel)

        return graph

    def _build_turn_context(self, user_input: str, attachments: AttachmentList | None = None) -> TurnContext:
        context = TurnContext(user_input=user_input, attachments=attachments)
        context.metadata.setdefault("temporal", context.temporal_snapshot())
        previous_health = dict(getattr(self.bot, "_last_turn_health_state", {}) or {}) if self.bot is not None else {}
        previous_status = str(previous_health.get("status") or "").strip().lower()
        if previous_status:
            context.metadata["turn_health_previous_status"] = previous_status
        llm_service = self.registry.get("llm")
        runtime_bot = getattr(llm_service, "bot", None)
        llm_provider = str(getattr(runtime_bot, "LLM_PROVIDER", "ollama") or "ollama")
        llm_model = str(getattr(runtime_bot, "LLM_MODEL", "") or "")

        # Determinism-safe, per-turn blackboard seed and fingerprint.
        blackboard_seed: dict[str, str] = {}
        blackboard_seed_fingerprint = _stable_sha256(blackboard_seed)
        context.metadata.setdefault("agent_blackboard_seed", dict(blackboard_seed))
        context.state.setdefault("agent_blackboard", dict(blackboard_seed))

        lock_payload = {
            "user_input": str(user_input or ""),
            "attachments": list(attachments or []),
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "state_machine": "PLAN_ACT_OBSERVE_RESPOND",
            "agent_blackboard_seed": blackboard_seed,
            "agent_blackboard_seed_fingerprint": blackboard_seed_fingerprint,
        }
        lock_hash = _stable_sha256(lock_payload)
        if self._strict:
            determinism = {
                "state_machine": "PLAN_ACT_OBSERVE_RESPOND",
                "enforced": True,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "seed_policy": "fixed_seed",
                "temperature_policy": "0.0",
                "lock_version": 3,
                "lock_hash": lock_hash,
                "lock_id": f"det-{lock_hash[:16]}",
                "memory_layers_included": True,
                "memory_fingerprint": str(getattr(self.bot, "_last_memory_fingerprint", "") or ""),
                "agent_blackboard_seed_fingerprint": blackboard_seed_fingerprint,
            }
        else:
            determinism = {
                "state_machine": "PLAN_ACT_OBSERVE_RESPOND",
                "enforced": False,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "seed_policy": "unseeded",
                "temperature_policy": "runtime_default",
                "lock_version": 3,
                "lock_hash": lock_hash,
                "lock_id": f"det-{lock_hash[:16]}",
                "memory_layers_included": True,
                "agent_blackboard_seed_fingerprint": blackboard_seed_fingerprint,
            }
        context.metadata.setdefault("determinism", determinism)
        if self._strict:
            # Deterministic replay envelope: same lock hash -> same temporal axis.
            context.temporal = TurnTemporalAxis.from_lock_hash(lock_hash)
            context.metadata["temporal"] = context.temporal_snapshot()
            if runtime_bot is not None:
                setattr(runtime_bot, "LLM_TEMPERATURE", 0.0)
                setattr(runtime_bot, "LLM_SEED", 42)
                set_deterministic = getattr(runtime_bot, "set_deterministic", None)
                if callable(set_deterministic):
                    set_deterministic(True)
        return context

    def _update_bot_last_state_safely(self, context: TurnContext) -> None:
        """Thread-safe update of legacy bot state mirrored from TurnContext."""
        if self.bot is None:
            return
        lock = getattr(self.bot, "_state_lock", None) or getattr(self.bot, "_session_lock", None)
        if lock is None:
            lock = contextlib.nullcontext()
        with lock:
            self.bot._last_turn_health_state = dict(context.state.get("turn_health_state") or {})
            self.bot._last_turn_ux_feedback = dict(context.state.get("ux_feedback") or {})
            self.bot._last_turn_health_evidence = dict(context.state.get("turn_health_evidence") or {})
            self.bot._last_capability_audit_report = dict(context.state.get("capability_audit_report") or {})
            self.bot._last_commit_id = context.state.get("last_commit_id")

    async def _execute_job(self, session: dict, job) -> FinalizedTurnResult:
        correlation_id = str(
            getattr(job, "metadata", {}).get("correlation_id")
            or CorrelationContext.current()
            or CorrelationContext.ensure()
            or ""
        ).strip()
        context = self._build_turn_context(job.user_input, attachments=job.attachments)
        # Executor boundary: trace must exist and correlation must be propagated.
        job_trace_id = str(
            getattr(job, "metadata", {}).get("trace_id")
            or correlation_id
            or ""
        ).strip()
        if job_trace_id:
            context.trace_id = job_trace_id
        assert context.trace_id, "Tracing not initialized: TurnContext.trace_id is empty"
        context.metadata.setdefault("trace", {})["trace_id"] = context.trace_id
        context.metadata.setdefault("trace", {})["correlation_id"] = correlation_id
        context.metadata.setdefault(
            "control_plane",
            {
                "session_id": str(session.get("session_id") or job.session_id),
                "job_id": str(getattr(job, "job_id", "")),
                "status": str(session.get("status") or "active"),
                "enforced": True,
            },
        )
        with CorrelationContext.bind(correlation_id):
            try:
                context.metadata["audit_mode"] = bool(getattr(job, "metadata", {}).get("audit_mode", False))
                result = await self.graph.execute(context, audit_mode=bool(context.metadata.get("audit_mode")))
            except Exception as exc:
                logger.exception("Turn execution failed")
                context.state["error"] = str(exc)
                result = ("", False)
            finally:
                self._update_bot_last_state_safely(context)
                self._last_turn_context = context
        final_blackboard = dict(context.state.get("agent_blackboard") or {})
        final_blackboard_fingerprint = _stable_sha256(final_blackboard)
        determinism_meta = dict(context.metadata.get("determinism") or {})
        determinism_meta["agent_blackboard_final_fingerprint"] = final_blackboard_fingerprint
        context.metadata["determinism"] = determinism_meta

        session_state = session.setdefault("state", {})
        session_state["last_result"] = result
        session_state["last_trace_id"] = context.trace_id
        session_state["last_turn_health_state"] = dict(context.state.get("turn_health_state") or {})
        session_state["last_turn_ux_feedback"] = dict(context.state.get("ux_feedback") or {})
        session_state["last_memory_full_history_id"] = str(context.state.get("memory_full_history_id") or "")
        session_state["last_memory_structured"] = dict(context.state.get("memory_structured") or {})
        session_state["last_agent_blackboard"] = final_blackboard
        session_state["last_arbitration_log"] = list(context.state.get("delegation_arbitration_log") or [])
        session_state["last_determinism"] = dict(context.metadata.get("determinism") or {})
        return result

    async def handle_turn(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        session_id: str = "default",
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        """Single turn entry point routed through the unified execution control plane."""
        return await self.control_plane.submit_turn(
            session_id=session_id,
            user_input=user_input,
            attachments=attachments,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _run_coro_in_thread(coro):
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    async def _run_graph_turn_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        return await self.handle_turn(user_input, attachments=attachments)

    def _run_graph_turn_sync(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self._run_graph_turn_async(user_input, attachments=attachments)
        if loop is not None and loop.is_running():
            return self._run_coro_in_thread(coro)
        else:
            return asyncio.run(coro)

    def run(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        return self._run_graph_turn_sync(user_input, attachments=attachments)

    async def run_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        return await self._run_graph_turn_async(user_input, attachments=attachments)
