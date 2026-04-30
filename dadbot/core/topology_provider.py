from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, TypedDict


class TurnPipelineState(TypedDict):
    """State carrier for a single turn through the topology pipeline."""

    context: Any
    short_circuit: bool
    abort: bool
    error: str | None


NodeExecutorFn = Callable[
    [str, Any, TurnPipelineState],
    Awaitable[dict],
]


class TopologyGraph(Protocol):
    """Compiled topology graph runtime used by TurnGraph.execute."""

    async def ainvoke(self, state: TurnPipelineState) -> TurnPipelineState: ...


class TopologyProvider(Protocol):
    """Abstraction over topology engines (e.g. LangGraph)."""

    def build(self) -> TopologyGraph: ...


TopologyProviderFactory = Callable[
    [list[tuple[str, Any]], NodeExecutorFn],
    TopologyProvider,
]
