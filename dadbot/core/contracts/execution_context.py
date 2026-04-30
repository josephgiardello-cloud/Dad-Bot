from __future__ import annotations

from typing import Protocol, runtime_checkable

from dadbot.core.graph_context import TurnContext


@runtime_checkable
class ExecutionContextCarrier(Protocol):
    """Contract: executable units receive a canonical TurnContext as `ctx`."""

    ctx: TurnContext


def require_turn_context(ctx: TurnContext | None) -> TurnContext:
    """Fail fast when call sites skip the canonical execution context."""

    if not isinstance(ctx, TurnContext):
        raise RuntimeError("Execution contract requires ctx: TurnContext")
    return ctx
