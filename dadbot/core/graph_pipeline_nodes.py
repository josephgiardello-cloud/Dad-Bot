"""graph_pipeline_nodes — canonical pipeline node stubs for TurnGraph.

These are the minimal service-delegation node implementations used as defaults
when TurnGraph is constructed without custom nodes.  The full production
implementations live in dadbot.core.nodes.

Extracted from graph.py to reduce TurnGraph god-class surface area.
All names are re-exported from dadbot.core.graph for backward compatibility.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_types import NodeType

logger = logging.getLogger(__name__)


class GraphNode(Protocol):
    @property
    def name(self) -> str: ...

    def dependencies(self) -> tuple[str, ...]: ...

    async def run(self, registry: Any, ctx: TurnContext) -> None: ...

    async def execute(self, registry: Any, turn_context: TurnContext) -> None: ...


class _NodeContractMixin:
    def dependencies(self) -> tuple[str, ...]:
        return ()

    async def run(self, registry: Any, ctx: TurnContext) -> None:
        await self.execute(registry, ctx)


class HealthNode(_NodeContractMixin):
    name = "health"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("maintenance_service")
        turn_context.state["health"] = service.tick(turn_context)


class ContextBuilderNode(_NodeContractMixin):
    name = "context_builder"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("context_service")
        turn_context.state["rich_context"] = service.build_context(turn_context)


MemoryNode = ContextBuilderNode


class InferenceNode(_NodeContractMixin):
    name = "inference"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("agent_service")
        rich_context = turn_context.state.get("rich_context", {})
        turn_context.state["candidate"] = await service.run_agent(
            turn_context,
            rich_context,
        )


class SafetyNode(_NodeContractMixin):
    name = "safety"

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("safety_service")
        candidate = turn_context.state.get("candidate")
        turn_context.state["safe_result"] = service.enforce_policies(
            turn_context,
            candidate,
        )


class SaveNode(_NodeContractMixin):
    name = "save"
    node_type = NodeType.COMMIT

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        service = registry.get("persistence_service")
        result = turn_context.state.get("safe_result")
        finalize = getattr(service, "finalize_turn", None)
        if callable(finalize):
            try:
                turn_context.state["safe_result"] = finalize(turn_context, result)
                save_checkpoint = getattr(service, "save_graph_checkpoint", None)
                if callable(save_checkpoint):
                    checkpoint = turn_context.checkpoint_snapshot(
                        stage="save",
                        status="atomic_commit",
                        error=None,
                    )
                    save_checkpoint(checkpoint, _skip_turn_event=True)
                return
            except Exception as exc:  # noqa: BLE001 — SaveNode optimistic path; non-fatal
                logger.debug("SaveNode optimistic checkpoint skipped: %s", exc)
        service.save_turn(turn_context, result)


class TemporalNode(_NodeContractMixin):
    name = "temporal"
    node_type = NodeType.STANDARD

    async def execute(self, _registry: Any, turn_context: TurnContext) -> None:
        if getattr(turn_context, "temporal", None) is None:
            raise RuntimeError(
                "TemporalNode missing — deterministic execution violated",
            )
        temporal_payload = turn_context.temporal_snapshot()
        turn_context.state.setdefault("temporal", temporal_payload)
        turn_context.metadata.setdefault("temporal", temporal_payload)


class ReflectionNode(_NodeContractMixin):
    name = "reflection"
    node_type = NodeType.STANDARD

    async def execute(self, registry: Any, turn_context: TurnContext) -> None:
        try:
            service = registry.get("reflection")
        except (KeyError, AttributeError):
            return
        result = turn_context.state.get("safe_result") or turn_context.state.get(
            "candidate",
        )
        turn_text = turn_context.state.get("turn_text") or turn_context.user_input
        current_mood = turn_context.state.get("mood") or "neutral"
        reply_text = result[0] if isinstance(result, tuple) else str(result or "")

        reflect_after_turn = getattr(service, "reflect_after_turn", None)
        if callable(reflect_after_turn):
            turn_context.state["reflection"] = reflect_after_turn(
                turn_text,
                current_mood,
                reply_text,
            )
            return

        reflect = getattr(service, "reflect", None)
        if callable(reflect):
            turn_context.state["reflection"] = reflect(turn_context, result)
