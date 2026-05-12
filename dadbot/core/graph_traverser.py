from __future__ import annotations

import asyncio
import inspect
from typing import Any, cast

from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_pipeline_nodes import _invoke_node_run_compat


class TurnGraphTraverser:
    """Executes graph nodes while topology is managed elsewhere."""

    def __init__(self, *, registry: Any) -> None:
        self.registry = registry

    async def run_parallel_subnodes(self, node: Any, turn_context: TurnContext) -> TurnContext:
        async def _run_subnode(subnode: Any) -> None:
            sub_run = getattr(subnode, "run", None)
            if callable(sub_run):
                result = await _invoke_node_run_compat(sub_run, self.registry, turn_context)
                if result is not None:
                    return
                return
            sub_execute = getattr(subnode, "execute", None)
            if callable(sub_execute):
                params = inspect.signature(sub_execute).parameters
                if len(params) >= 2:
                    await cast("Any", sub_execute)(self.registry, turn_context)
                else:
                    await cast("Any", sub_execute)(turn_context)
                return
            raise TypeError(f"Unsupported node type: {type(subnode).__name__}")

        await asyncio.gather(*(_run_subnode(subnode) for subnode in node))
        return turn_context

    async def call_node_direct(self, node: Any, turn_context: TurnContext) -> TurnContext:
        if isinstance(node, (tuple, list)):
            return await self.run_parallel_subnodes(node, turn_context)

        run_method = getattr(node, "run", None)
        if callable(run_method):
            result = await _invoke_node_run_compat(run_method, self.registry, turn_context)
            return cast("TurnContext", result or turn_context)

        execute_method = getattr(node, "execute", None)
        if callable(execute_method):
            execute_params = inspect.signature(execute_method).parameters
            if len(execute_params) >= 2:
                await cast("Any", execute_method)(self.registry, turn_context)
            else:
                await cast("Any", execute_method)(turn_context)
            return turn_context

        raise TypeError(f"Unsupported node type: {type(node).__name__}")
