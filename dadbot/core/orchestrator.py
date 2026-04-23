from __future__ import annotations

import hashlib
import json
import logging

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.control_plane import ExecutionControlPlane, SessionRegistry
from dadbot.core.graph import TurnContext, TurnGraph
from dadbot.core.kernel import TurnKernel, bayesian_policy_gate
from dadbot.core.interfaces import (
    HealthService,
    InferenceService,
    MemoryService,
    PersistenceService,
    SafetyPolicyService,
    validate_pipeline_services,
)
from dadbot.core.nodes import HealthNode, InferenceNode, MemoryNode, SafetyNode, SaveNode
from dadbot.core.observability import CorrelationContext, configure_exporter
from dadbot.registry import ServiceRegistry, boot_registry

logger = logging.getLogger(__name__)


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
        """Wires pipeline stages: (Health || Memory) -> Inference -> Safety -> Save."""
        health = self.registry.get("health")
        memory = self.registry.get("memory")
        llm = self.registry.get("llm")
        safety = self.registry.get("safety")
        storage = self.registry.get("storage")

        # Validate service contracts; raises in strict mode, warns otherwise.
        validate_pipeline_services({
            "health": (health, HealthService),
            "memory": (memory, MemoryService),
            "llm": (llm, InferenceService),
            "safety": (safety, SafetyPolicyService),
            "storage": (storage, PersistenceService),
        }, raise_on_failure=self._strict)

        graph = TurnGraph(registry=self.registry)
        graph.add_node(
            "preflight",
            (
                HealthNode(health),
                MemoryNode(memory),
            ),
        )
        graph.add_node("inference", InferenceNode(llm))
        graph.add_node("safety", SafetyNode(safety))
        graph.add_node("save", SaveNode(storage))

        graph.set_edge("preflight", "inference")
        graph.set_edge("inference", "safety")
        graph.set_edge("safety", "save")

        runtime_bot = getattr(llm, 'bot', None)
        if runtime_bot is not None:
            kernel = TurnKernel(policy_gate=bayesian_policy_gate(runtime_bot))
        else:
            kernel = TurnKernel()
        graph.set_kernel(kernel)

        return graph

    def _build_turn_context(self, user_input: str, attachments: AttachmentList | None = None) -> TurnContext:
        context = TurnContext(user_input=user_input, attachments=attachments)
        llm_service = self.registry.get("llm")
        runtime_bot = getattr(llm_service, "bot", None)
        llm_provider = str(getattr(runtime_bot, "LLM_PROVIDER", "ollama") or "ollama")
        llm_model = str(getattr(runtime_bot, "LLM_MODEL", "") or "")
        lock_payload = {
            "user_input": str(user_input or ""),
            "attachments": list(attachments or []),
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "state_machine": "PLAN_ACT_OBSERVE_RESPOND",
        }
        lock_hash = hashlib.sha256(json.dumps(lock_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        context.metadata.setdefault(
            "determinism",
            {
                "state_machine": "PLAN_ACT_OBSERVE_RESPOND",
                "enforced": True,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "seed_policy": "unseeded",
                "temperature_policy": "runtime_default",
                "lock_version": 1,
                "lock_hash": lock_hash,
                "lock_id": f"det-{lock_hash[:16]}",
            },
        )
        return context

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
            result = await self.graph.execute(context)
        session.setdefault("state", {})["last_result"] = result
        session.setdefault("state", {})["last_trace_id"] = context.trace_id
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
