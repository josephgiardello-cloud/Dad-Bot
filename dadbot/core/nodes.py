from __future__ import annotations

import logging
from typing import Any

from dadbot.core.graph import TurnContext

logger = logging.getLogger(__name__)


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
        if callable(getattr(self.mgr, "query", None)):
            try:
                context.state["memories"] = await self.mgr.query(context.user_input)
            except Exception as exc:
                logger.warning("MemoryNode.query failed (non-fatal): %s", exc)
            return context

        build_context = getattr(self.mgr, "build_context", None)
        if callable(build_context):
            try:
                context.state["rich_context"] = build_context(context)
            except Exception as exc:
                logger.warning("MemoryNode.build_context failed (non-fatal): %s", exc)
        return context


class MemoryNode(ContextBuilderNode):
    """Backward-compatible alias for legacy pipeline wiring."""

    name = "memory"


class InferenceNode:
    """Drives mood detection, direct reply routing, agentic tools, and LLM inference."""

    def __init__(self, llm_manager: Any):
        self.mgr = llm_manager

    async def run(self, context: TurnContext) -> TurnContext:
        rich_context = context.state.get("rich_context", {})
        run_agent = getattr(self.mgr, "run_agent", None)
        if callable(run_agent):
            try:
                context.state["candidate"] = await run_agent(context, rich_context)
            except Exception as exc:
                logger.error("InferenceNode.run_agent failed: %s", exc)
                context.state["candidate"] = ("Something went sideways. Try again in a moment.", False)
            return context

        generate = getattr(self.mgr, "generate", None)
        if callable(generate):
            try:
                context.state["candidate"] = await generate(context.user_input, rich_context)
            except Exception as exc:
                logger.error("InferenceNode.generate failed: %s", exc)
                context.state["candidate"] = ("Unable to generate a reply right now.", False)
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

    async def run(self, context: TurnContext) -> TurnContext:
        result = context.state.get("safe_result") or context.state.get("candidate")

        # Prefer atomic finalize_turn: history + maintenance + reflection + health + persist.
        finalize = getattr(self.mgr, "finalize_turn", None)
        if callable(finalize):
            try:
                finalized = finalize(context, result)
                context.state["safe_result"] = finalized
            except Exception as exc:
                logger.error("SaveNode.finalize_turn failed; falling back to basic save: %s", exc)
                _basic_save(self.mgr, context, result)
            return context

        _basic_save(self.mgr, context, result)
        return context


def _basic_save(mgr: Any, context: TurnContext, result: Any) -> None:
    """Fallback persistence when finalize_turn is unavailable."""
    save_turn = getattr(mgr, "save_turn", None)
    if callable(save_turn):
        try:
            save_turn(context, result)
            return
        except Exception as exc:
            logger.error("SaveNode._basic_save save_turn failed: %s", exc)
    persist = getattr(mgr, "persist_conversation", None)
    if callable(persist):
        try:
            persist()
        except Exception as exc:
            logger.error("SaveNode._basic_save persist_conversation failed: %s", exc)
