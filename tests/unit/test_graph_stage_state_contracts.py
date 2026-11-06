from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dadbot.core.graph import NodeContractViolationError, TurnContext, TurnGraph


@pytest.mark.parametrize(
    ("preceding", "state", "produces", "short_circuit", "error"),
    [
        (["context_builder"], {}, True, False, "missing required inputs"),
        (["context_builder"], {"rich_context": {}}, False, False, "missing required outputs"),
        (["context_builder"], {"rich_context": {}}, True, False, None),
        ([], {}, True, False, None),
        (["context_builder"], {"rich_context": {}}, False, True, None),
    ],
)
def test_stage_enforces_state_contracts_with_partial_and_terminal_graph_support(
    preceding, state, produces, short_circuit, error,
):
    async def execute(_node, context):
        if produces:
            context.state["candidate"] = "reply"
        if short_circuit:
            context.short_circuit = True
            context.short_circuit_result = ("terminal reply", False)
        return context

    graph = SimpleNamespace(
        _execute_node=AsyncMock(side_effect=execute),
        **{
            name: Mock()
            for name in (
                "_enforce_identity_tuple_immutable", "_record_stage_start", "_emit_checkpoint",
                "_enforce_kernel_invariants", "_capture_memory_before", "_record_stage_success",
                "_record_stage_failure", "_maybe_transition_phase", "_record_edge_transition",
                "_capture_memory_after",
            )
        },
    )
    context = TurnContext(user_input="hello")
    context.state.update(state)
    pipeline = [(name, object()) for name in [*preceding, "inference"]]
    operation = TurnGraph._run_stage(
        graph, idx=len(preceding), pipeline_items=pipeline, turn_context=context,
        emitted_phases=set(), persistence_service=None, persistence_is_lightweight=True,
    )
    if error:
        with pytest.raises(NodeContractViolationError, match=error):
            asyncio.run(operation)
        graph._record_stage_failure.assert_called_once()
        graph._record_stage_success.assert_not_called()
        if "inputs" in error:
            graph._execute_node.assert_not_awaited()
    else:
        result = asyncio.run(operation)
        assert result[0] is context
        assert result[1] is short_circuit
        graph._record_stage_success.assert_called_once()
        graph._record_stage_failure.assert_not_called()
