"""LangGraph topology declaration for the Dad Bot turn pipeline.

Layer 1 of the 3-layer architecture:

    LangGraph (topology)  ──▶  TurnGraph (execution kernel)  ──▶  ControlPlane (system)

This module declares ONLY:

  - Pipeline state schema  (TurnPipelineState)
  - Node registration
  - Edge structure and conditional routing

No policy, mutation, recovery, retry, or execution logic lives here.
All execution concerns are handled by Layer 2 (TurnGraph) via the
``node_executor`` callback.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from dadbot.core.topology_provider import NodeExecutorFn, TopologyProvider, TurnPipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing predicate  — Layer 1 only, no policy or retry logic here
# ---------------------------------------------------------------------------

_CONTINUE = "__continue__"


def _route_after_stage(state: TurnPipelineState) -> str:
    """Routing predicate applied after every non-terminal stage.

    Only topology-level concerns live here.  The ``abort`` and
    ``short_circuit`` flags are set by Layer 2 (TurnGraph execution kernel);
    this function merely reads them to decide the next hop.
    """
    if state.get("abort") or state.get("short_circuit"):
        return END
    return _CONTINUE


# ---------------------------------------------------------------------------
# Pipeline builder — declarative topology construction
# ---------------------------------------------------------------------------

class LangGraphTopologyProvider(TopologyProvider):
    """LangGraph-backed implementation of the TopologyProvider interface."""

    def __init__(
        self,
        pipeline_items: list[tuple[str, Any]],
        node_executor: NodeExecutorFn,
    ) -> None:
        self._pipeline_items = pipeline_items
        self._node_executor = node_executor

    def build(self) -> Any:
        return _build_turn_pipeline(self._pipeline_items, self._node_executor)


def create_topology_provider(
    pipeline_items: list[tuple[str, Any]],
    node_executor: NodeExecutorFn,
) -> TopologyProvider:
    """Factory for the LangGraph TopologyProvider implementation."""
    return LangGraphTopologyProvider(pipeline_items, node_executor)


def _build_turn_pipeline(
    pipeline_items: list[tuple[str, Any]],
    node_executor: NodeExecutorFn,
) -> Any:
    """Build a compiled LangGraph ``StateGraph`` for one turn execution.

    This function is **purely declarative**: it wires stage nodes and edges
    together using ``node_executor`` as the execution backend.  All
    execution, policy, mutation, and recovery logic lives inside
    ``node_executor`` (supplied by Layer 2 / TurnGraph).

    Parameters
    ----------
    pipeline_items:
        Ordered list of ``(stage_name, node_object)`` pairs that define the
        turn pipeline.  The graph topology is derived entirely from this
        ordered list; no routing rules live here.
    node_executor:
        Async callable provided by TurnGraph (Layer 2) that handles kernel
        validation, capability enforcement, mutation guards, execution trace
        recording, and crash-safe recovery for a single stage.

        Signature: ``async (stage_name, node, state: TurnPipelineState) -> dict``

        The returned dict is a **partial** ``TurnPipelineState`` update that
        LangGraph merges into the running state.

    Returns
    -------
    A compiled LangGraph runnable that supports ``.ainvoke(initial_state)``.
    """
    if not pipeline_items:
        raise ValueError("build_turn_pipeline: pipeline_items must be non-empty")

    stage_names = [name for name, _ in pipeline_items]
    builder: StateGraph = StateGraph(TurnPipelineState)

    # Register one LangGraph node per pipeline stage.
    # Explicit default-argument capture avoids the Python late-binding closure
    # bug: without ``sn=stage_name, node=stage_node`` defaults, every closure
    # would capture the loop variable by reference and resolve to the last
    # iteration's values.
    for stage_name, stage_node in pipeline_items:
        def _make_lg_node(sn: str = stage_name, node: Any = stage_node) -> Callable:
            async def _lg_node_fn(state: TurnPipelineState) -> dict:
                return await node_executor(sn, node, state)
            _lg_node_fn.__name__ = f"lg_stage_{sn}"
            return _lg_node_fn

        builder.add_node(stage_name, _make_lg_node())

    # Wire edges: every stage except the last gets a conditional edge so that
    # short-circuit and abort flags can terminate the pipeline early.
    for i, name in enumerate(stage_names[:-1]):
        next_name = stage_names[i + 1]
        builder.add_conditional_edges(
            name,
            _route_after_stage,
            {_CONTINUE: next_name, END: END},
        )

    # The final stage always terminates.
    builder.add_edge(stage_names[-1], END)

    builder.set_entry_point(stage_names[0])
    return builder.compile()


__all__ = ["LangGraphTopologyProvider", "create_topology_provider"]
