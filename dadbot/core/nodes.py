from __future__ import annotations

import logging
from typing import Any

from dadbot.core.graph import TurnContext

logger = logging.getLogger(__name__)


class TemporalNode:
    """Publishes the canonical frozen turn time into state and metadata."""

    name = "temporal"

    async def run(self, context: TurnContext) -> TurnContext:
        if getattr(context, "temporal", None) is None:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        temporal_payload = context.temporal_snapshot()
        context.state["temporal"] = temporal_payload
        context.metadata.setdefault("temporal", temporal_payload)
        return context


class HealthNode:
    """Runs periodic maintenance and proactive engagement checks before the turn."""

    def __init__(self, health_manager: Any):
        self.mgr = health_manager

    async def run(self, context: TurnContext) -> TurnContext:
        tick = getattr(self.mgr, "tick", None)
        if callable(tick):
            try:
                context.state["health"] = tick(context)
            except Exception as exc:
                logger.warning("HealthNode.tick failed (non-fatal): %s", exc)
        return context


class ContextBuilderNode:
    """Builds rich contextual payload (profile/relationship/memory/cross-session)."""

    name = "context_builder"

    def __init__(self, memory_manager: Any):
        self.mgr = memory_manager

    async def run(self, context: TurnContext) -> TurnContext:
        if getattr(context, "temporal", None) is None:
            raise RuntimeError("TemporalNode missing — deterministic execution violated")
        if callable(getattr(self.mgr, "query", None)):
            try:
                context.state.setdefault("temporal", context.temporal_snapshot())
                context.state["memories"] = await self.mgr.query(context.user_input)
            except Exception as exc:
                logger.warning("MemoryNode.query failed (non-fatal): %s", exc)
            return context

        build_context = getattr(self.mgr, "build_context", None)
        if callable(build_context):
            try:
                rich_context = build_context(context)
                if isinstance(rich_context, dict):
                    rich_context.setdefault("temporal", context.temporal_snapshot())
                context.state["rich_context"] = rich_context
            except Exception as exc:
                logger.warning("MemoryNode.build_context failed (non-fatal): %s", exc)
        return context


class MemoryNode(ContextBuilderNode):
    """Backward-compatible alias for legacy pipeline wiring."""

    name = "memory"


class InferenceNode:
    """Pure cognition loop executor â€” runs AgentService.run_agent as the sole inference path.

    No fallback generation path exists.  If the bound manager does not expose
    ``run_agent``, the node raises ``RuntimeError`` at run-time rather than
    silently routing through a legacy path.
    """

    def __init__(self, llm_manager: Any):
        self.mgr = llm_manager

    @staticmethod
    def _fallback_candidate(message: str) -> tuple[str, bool]:
        return (str(message or "Unable to generate a reply right now."), False)

    async def run(self, context: TurnContext) -> TurnContext:
        run_agent = getattr(self.mgr, "run_agent", None)
        if not callable(run_agent):
            raise RuntimeError(
                f"InferenceNode: bound manager {type(self.mgr).__name__!r} does not expose "
                "'run_agent'; only AgentService is a valid inference provider."
            )
        rich_context = context.state.get("rich_context", {})
        try:
            context.state["candidate"] = await run_agent(context, rich_context)
        except Exception as exc:
            logger.error("InferenceNode.run_agent failed: %s", exc)
            context.state["candidate"] = self._fallback_candidate("Something went sideways. Try again in a moment.")
        return context


class SafetyNode:
    """Applies TONY score tone constraints and reply policy enforcement."""

    def __init__(self, safety_manager: Any):
        self.mgr = safety_manager

    async def run(self, context: TurnContext) -> TurnContext:
        # Session exit was already fully handled by InferenceNode -- skip.
        if context.state.get("already_finalized"):
            return context

        candidate = context.state.get("candidate")
        enforce = getattr(self.mgr, "enforce_policies", None)
        if callable(enforce):
            try:
                context.state["safe_result"] = enforce(context, candidate)
            except Exception as exc:
                logger.error("SafetyNode.enforce_policies failed: %s", exc)
                context.state["safe_result"] = candidate
            return context

        validate = getattr(self.mgr, "validate", None)
        if callable(validate):
            try:
                context.state["safe_result"] = validate(candidate)
            except Exception as exc:
                logger.error("SafetyNode.validate failed: %s", exc)
                context.state["safe_result"] = candidate
            return context

        context.state["safe_result"] = candidate
        return context


class SaveNode:
    """Atomically commits history, maintenance, health snapshot, and persistence."""

    def __init__(self, storage_manager: Any):
        self.mgr = storage_manager

    def _result_from_context(self, context: TurnContext) -> Any:
        return context.state.get("safe_result") or context.state.get("candidate")

    @staticmethod
    def _kernel_shadow_validate(context: TurnContext, *, stage: str, operation: str) -> None:
        kernel = context.state.get("_execution_kernel") if isinstance(context.state, dict) else None
        if kernel is None:
            return
        if bool(getattr(kernel, "strict", False)):
            kernel.validate(stage=stage, operation=operation, context=context)
            return
        try:
            result = kernel.validate(stage=stage, operation=operation, context=context)
            if not bool(getattr(result, "ok", True)):
                logger.warning("[KERNEL SHADOW VIOLATION] %s", getattr(result, "reason", "validation failed"))
        except Exception as exc:
            logger.warning("[KERNEL SHADOW VIOLATION] %s", exc)

    def _finalize_turn(self, context: TurnContext, result: Any) -> bool:
        finalize = getattr(self.mgr, "finalize_turn", None)
        if not callable(finalize):
            raise RuntimeError("SaveNode requires finalize_turn in Phase 4 strict mode")
        try:
            finalized = finalize(context, result)
            context.state["safe_result"] = finalized
            return True
        except Exception as exc:
            logger.error("SaveNode.finalize_turn failed in strict mode: %s", exc)
            raise

    async def run(self, context: TurnContext) -> TurnContext:
        self._kernel_shadow_validate(
            context,
            stage="save_node_pre",
            operation="core.nodes.SaveNode.run",
        )
        if getattr(context, "temporal", None) is None:
            raise RuntimeError("TemporalNode required — execution invalid")
        context.state.setdefault("temporal", context.temporal_snapshot())
        context.metadata.setdefault("temporal", context.temporal_snapshot())
        result = self._result_from_context(context)
        begin_transaction = getattr(self.mgr, "begin_transaction", None)
        apply_mutations = getattr(self.mgr, "apply_mutations", None)
        commit_transaction = getattr(self.mgr, "commit_transaction", None)
        if not callable(begin_transaction) or not callable(apply_mutations) or not callable(commit_transaction):
            raise RuntimeError("SaveNode requires begin/apply/commit transaction hooks in Phase 4 strict mode")

        begin_transaction(context)
        try:
            apply_mutations(context)
            self._finalize_turn(context, result)
            commit_transaction(context)
        except Exception:
            rollback_transaction = getattr(self.mgr, "rollback_transaction", None)
            if callable(rollback_transaction):
                rollback_transaction(context)
            raise
        self._kernel_shadow_validate(
            context,
            stage="save_node_post",
            operation="core.nodes.SaveNode.run",
        )
        return context


class ReflectionNode:
    """Runs optional post-save reflection hooks without impacting reply continuity."""

    name = "reflection"

    def __init__(self, reflection_manager: Any):
        self.mgr = reflection_manager

    async def run(self, context: TurnContext) -> TurnContext:
        if self.mgr is None:
            return context

        result = context.state.get("safe_result") or context.state.get("candidate")
        turn_text = context.state.get("turn_text") or context.user_input
        current_mood = context.state.get("mood") or "neutral"
        reply_text = result[0] if isinstance(result, tuple) else str(result or "")

        reflect_after_turn = getattr(self.mgr, "reflect_after_turn", None)
        if callable(reflect_after_turn):
            try:
                context.state["reflection"] = reflect_after_turn(turn_text, current_mood, reply_text)
            except TypeError:
                context.state["reflection"] = reflect_after_turn(context, result)
            except Exception as exc:
                logger.warning("ReflectionNode.reflect_after_turn failed (non-fatal): %s", exc)
            return context

        reflect = getattr(self.mgr, "reflect", None)
        if callable(reflect):
            try:
                context.state["reflection"] = reflect(context, result)
            except TypeError:
                try:
                    context.state["reflection"] = reflect(force=True)
                except TypeError:
                    context.state["reflection"] = reflect(context)
                except Exception as exc:
                    logger.warning("ReflectionNode.reflect failed (non-fatal): %s", exc)
            except Exception as exc:
                logger.warning("ReflectionNode.reflect failed (non-fatal): %s", exc)
        return context


