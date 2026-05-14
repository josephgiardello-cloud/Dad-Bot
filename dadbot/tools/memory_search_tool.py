"""
MemorySearch Tool - Phase 5 Integration

Allows the orchestrator to "consciously" retrieve relevant historical context
from the SovereignMemory during turn execution, before the LLM sees the user prompt.

This tool is registered in the ToolRegistry and can be called by:
1. Safety Policy IR (to filter and validate retrieved context)
2. Orchestrator (subconscious search before LLM inference)
3. User (explicit "Remember when..." queries)

Signature:
    search(
        query: str,
        context_limit: int = 4,
        time_window_days: Optional[int] = None,
        event_type_filter: Optional[str] = None,
    ) -> list[dict]

Returns:
    List of indexed event payloads ranked by semantic similarity
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from dadbot.core.runtime_types import (
    CanonicalPayload,
    ToolDeterminismClass,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
    ToolSideEffectClass,
    ToolSpec,
)

logger = logging.getLogger(__name__)


# Tool specification
MEMORY_SEARCH_SPEC = ToolSpec(
    name="memory_search",
    version="1.0.0",
    description="Search sovereign memory for relevant historical context",
    capabilities=frozenset({"semantic_search", "context_retrieval", "history_lookup"}),
    determinism=ToolDeterminismClass.READ_ONLY,
    side_effect_class=ToolSideEffectClass.PURE,
)


def execute_memory_search(invocation: ToolInvocation) -> ToolResult:
    """
    Execute memory search tool.
    
    This is the executor function registered with ToolRegistry.
    It retrieves historical context from SovereignMemory based on the query.
    """
    import time
    from dadbot.services.vector_memory import get_global_memory

    start_time = time.perf_counter()

    try:
        # Parse input
        params = invocation.arguments or {}
        query = str(params.get("query") or "").strip()
        context_limit = int(params.get("context_limit", 4))
        time_window_days = params.get("time_window_days")
        event_type_filter = params.get("event_type_filter")

        if not query:
            return ToolResult(
                tool_name=invocation.tool_spec.name,
                invocation_id=invocation.invocation_id,
                status=ToolExecutionStatus.ERROR,
                payload=CanonicalPayload(
                    {
                        "status": "error",
                        "fragments_found": 0,
                        "context": [],
                        "message": "Query cannot be empty",
                    },
                    payload_type="memory_search_error",
                ),
                error="Empty query provided",
                latency_ms=0,
                replay_safe=True,
            )

        # Get global memory instance
        memory = get_global_memory()
        if memory is None:
            logger.warning("SovereignMemory not initialized")
            return ToolResult(
                tool_name=invocation.tool_spec.name,
                invocation_id=invocation.invocation_id,
                status=ToolExecutionStatus.ERROR,
                payload=CanonicalPayload(
                    {
                        "status": "error",
                        "fragments_found": 0,
                        "context": [],
                        "message": "Memory service not available",
                    },
                    payload_type="memory_search_error",
                ),
                error="SovereignMemory not initialized",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                replay_safe=True,
            )

        # Retrieve context
        fragments = memory.retrieve_context(
            query=query,
            limit=context_limit,
            time_window_days=time_window_days,
            event_type_filter=event_type_filter,
        )

        # Format output
        context_list = []
        for fragment in fragments:
            # Truncate content for output
            content_preview = fragment.content[:500] if fragment.content else ""
            if len(fragment.content or "") > 500:
                content_preview += "..."

            context_list.append(
                {
                    "event_id": fragment.event_id,
                    "event_type": fragment.event_type,
                    "timestamp": fragment.timestamp,
                    "similarity_score": round(fragment.similarity_score, 3),
                    "content_preview": content_preview,
                }
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        if not context_list:
            return ToolResult(
                tool_name=invocation.tool_spec.name,
                invocation_id=invocation.invocation_id,
                status=ToolExecutionStatus.OK,
                payload=CanonicalPayload(
                    {
                        "status": "empty",
                        "fragments_found": 0,
                        "context": [],
                        "message": f"No historical context found for query: {query}",
                    },
                    payload_type="memory_search_result",
                ),
                latency_ms=latency_ms,
                replay_safe=True,
            )

        return ToolResult(
            tool_name=invocation.tool_spec.name,
            invocation_id=invocation.invocation_id,
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload(
                {
                    "status": "success",
                    "fragments_found": len(context_list),
                    "context": context_list,
                    "message": f"Retrieved {len(context_list)} relevant memory fragments",
                },
                payload_type="memory_search_result",
            ),
            latency_ms=latency_ms,
            replay_safe=True,
        )

    except Exception as e:
        logger.error(f"Memory search execution failed: {e}")
        return ToolResult(
            tool_name=invocation.tool_spec.name,
            invocation_id=invocation.invocation_id,
            status=ToolExecutionStatus.ERROR,
            payload=CanonicalPayload(
                {
                    "status": "error",
                    "fragments_found": 0,
                    "context": [],
                    "message": f"Error during memory search: {str(e)}",
                },
                payload_type="memory_search_error",
            ),
            error=str(e),
            latency_ms=(time.perf_counter() - start_time) * 1000,
            replay_safe=True,
        )


def register_memory_search_tool(registry: Any) -> None:
    """
    Register the memory_search tool with a ToolRegistry.
    
    Usage:
        from dadbot.core.tool_registry import ToolRegistry
        from dadbot.tools.memory_search_tool import register_memory_search_tool
        
        registry = ToolRegistry()
        register_memory_search_tool(registry)
    """
    registry.register(MEMORY_SEARCH_SPEC, execute_memory_search)
    logger.info("MemorySearch tool registered successfully")
