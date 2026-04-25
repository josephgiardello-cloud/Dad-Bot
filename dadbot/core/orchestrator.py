from __future__ import annotations

import asyncio
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
from dadbot.core.nodes import ContextBuilderNode, HealthNode, InferenceNode, SafetyNode, SaveNode
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
                ContextBuilderNode(memory),
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

    def graph_turns_enabled(self) -> bool:
        return bool(getattr(self.bot, "_turn_graph_enabled", False))

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
        return asyncio.run(coro)

    def _record_graph_fallback(self, exc: Exception, *, mode: str) -> None:
        logger.warning(
            "Graph orchestration degraded to legacy %s pipeline; preserving response continuity: %s",
            mode,
            exc,
        )
        record_runtime_issue = getattr(self.bot, "record_runtime_issue", None)
        if callable(record_runtime_issue):
            record_runtime_issue(
                "turn_graph",
                f"turn_service_{mode}_fallback",
                exc=exc,
                level=logging.WARNING,
                metadata={
                    "degraded_mode": f"legacy_{mode}",
                    "graph_enabled": True,
                    "path": "graph_to_legacy",
                },
            )

    def _append_signoff(self, text: str) -> str:
        finalization = getattr(self.bot, "reply_finalization", None)
        append_signoff = getattr(finalization, "append_signoff", None)
        if callable(append_signoff):
            return append_signoff(text)
        legacy = getattr(self.bot, "finalize_reply", None)
        if callable(legacy):
            return legacy(text)
        return str(text or "")

    def _ollama_retryable_errors(self):
        handler = getattr(self.bot, "ollama_retryable_errors", None)
        if callable(handler):
            return handler()
        return (ConnectionError, TimeoutError, OSError)

    def run(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        turn_service = getattr(self.bot, "turn_service")
        if self.graph_turns_enabled():
            try:
                return self._run_graph_turn_sync(user_input, attachments=attachments)
            except Exception as exc:
                self._record_graph_fallback(exc, mode="sync")
        try:
            return turn_service.process_user_message(user_input, attachments)
        except self._ollama_retryable_errors() as exc:
            logger.error("Ollama connection error during turn; returning friendly fallback: %s", exc)
            friendly = (
                "I'm having a little trouble thinking right now - my connection seems to be down. "
                "Give me a moment and try again, buddy."
            )
            return self._append_signoff(friendly), False
        except Exception as exc:
            logger.error("Unexpected error during turn processing; returning safe fallback: %s", exc)
            return self._append_signoff(
                "Something went sideways on my end. I'm okay - just try again in a second."
            ), False

    async def run_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        turn_service = getattr(self.bot, "turn_service")
        if self.graph_turns_enabled():
            try:
                return await self._run_graph_turn_async(user_input, attachments=attachments)
            except Exception as exc:
                self._record_graph_fallback(exc, mode="async")
        return await turn_service.process_user_message_async(user_input, attachments)
