from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, cast

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.control_plane import ExecutionControlPlane, ExecutionJob, SessionRegistry
from dadbot.core.graph import TurnContext, TurnGraph
from dadbot.core.execution_trace_context import ExecutionTraceRecorder, bind_execution_trace
from dadbot.core.interfaces import HealthService, InferenceService, validate_pipeline_services
from dadbot.registry import ServiceRegistry, boot_registry

logger = logging.getLogger(__name__)


class DeterminismViolation(RuntimeError):
    """Retained for import compatibility with existing test files."""


class DadBotOrchestrator:
    """Graph-governed turn orchestrator routed through ExecutionControlPlane."""

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
        *,
        config_path: str = "config.yaml",
        bot: Any | None = None,
        strict: bool = False,
        enable_observability: bool = True,
        checkpointer: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        self.bot = bot
        self.registry = registry or boot_registry(config_path=config_path, bot=bot)
        self._strict = bool(strict)
        self._enable_observability = bool(enable_observability)
        self._last_turn_context: TurnContext | None = None

        storage = self.registry.get("storage", optional=True)
        if checkpointer is not None and storage is not None and callable(getattr(storage, "set_checkpointer", None)):
            storage.set_checkpointer(checkpointer)

        self.graph = self._build_turn_graph()
        self.control_plane = ExecutionControlPlane(
            registry=SessionRegistry(),
            kernel_executor=self._execute_job,
            graph=self.graph,
            enable_observability=self._enable_observability,
        )
        self.session_registry = self.control_plane.registry

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
        return TurnGraph(registry=self.registry)

    def _build_turn_context(
        self,
        *,
        user_input: str,
        attachments: AttachmentList | None,
        session_id: str,
        metadata: dict[str, Any] | None,
    ) -> TurnContext:
        md = dict(metadata or {})
        trace_id = str(md.get("trace_id") or "")
        context_kwargs: dict[str, Any] = {
            "user_input": user_input,
            "attachments": attachments,
            "metadata": {"session_id": str(session_id or "default"), **md},
        }
        if trace_id:
            context_kwargs["trace_id"] = trace_id
        context = TurnContext(**context_kwargs)
        return context

    async def _execute_job(self, session: dict[str, Any], job: ExecutionJob) -> FinalizedTurnResult:
        context = self._build_turn_context(
            user_input=str(job.user_input or ""),
            attachments=job.attachments,
            session_id=str(job.session_id or "default"),
            metadata=dict(job.metadata or {}),
        )
        if not context.trace_id:
            raise RuntimeError("TurnContext.trace_id must be non-empty")

        recorder = ExecutionTraceRecorder(
            trace_id=context.trace_id,
            prompt=str(job.user_input or ""),
            metadata={"session_id": str(job.session_id or "default")},
        )

        with bind_execution_trace(recorder, required=False):
            result = await self.graph.execute(context)

        self._last_turn_context = context
        state = session.setdefault("state", {})
        if isinstance(state, dict):
            typed_state = cast(dict[str, Any], state)
            typed_state["last_result"] = result
            goals_any_raw = context.state.get("goals")
            goals_any: list[Any]
            if isinstance(goals_any_raw, list):
                goals_any = list(cast(list[Any], goals_any_raw))
            else:
                prior_goals = typed_state.get("goals")
                goals_any = list(cast(list[Any], prior_goals)) if isinstance(prior_goals, list) else []
            typed_state["goals"] = list(goals_any)
        return result

    async def handle_turn(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        session_id: str = "default",
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        try:
            return await self.control_plane.submit_turn(
                session_id=session_id,
                user_input=user_input,
                attachments=attachments,
                metadata={},
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            logger.warning("Inference timed out after %ss", timeout_seconds or 30)
            return ("Sorry, I timed out while thinking. Please try again.", False)
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return ("Something went wrong. Please try again.", False)

    def run(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
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
        return await self.handle_turn(user_input, attachments=attachments)
