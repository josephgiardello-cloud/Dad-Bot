"""Compatibility layer between tool_ir (legacy) and runtime_types (Phase A/B).

This module enables bidirectional conversion between:
- tool_ir.ToolStatus (SUCCESS, RETRY, CONTRACT_VIOLATION, FATAL)
- runtime_types.ToolExecutionStatus (OK, ERROR, TIMEOUT, DENIED, DEGRADED, SKIPPED)

And between:
- tool_ir.ToolContractResult (legacy tool dispatch result)
- runtime_types.ToolResult (new Phase A/B result)

Purpose:
- During Phase B.1/B.2: Run both old and new dispatch systems in parallel
- During Phase B.2: Adapter layer converts results without breaking old code
- During Phase B.3: Remove this layer after full cutover
"""

from __future__ import annotations

from typing import Any

from dadbot.core import tool_ir
from dadbot.core import runtime_types as rt
from dadbot.core.tool_registry import ToolExecutionContext, ToolRegistry


def tool_ir_status_to_execution_status(
    ir_status: tool_ir.ToolStatus | str,
) -> rt.ToolExecutionStatus:
    """Convert legacy tool_ir.ToolStatus to canonical ToolExecutionStatus.
    
    Mapping:
      SUCCESS → OK
      RETRY → DEGRADED (transient, can retry)
      CONTRACT_VIOLATION → DENIED (contract not met, caller error)
      FATAL → ERROR (non-recoverable)
    """
    if isinstance(ir_status, str):
        ir_status = tool_ir.ToolStatus(ir_status)
    
    status_map = {
        tool_ir.ToolStatus.SUCCESS: rt.ToolExecutionStatus.OK,
        tool_ir.ToolStatus.RETRY: rt.ToolExecutionStatus.DEGRADED,
        tool_ir.ToolStatus.CONTRACT_VIOLATION: rt.ToolExecutionStatus.DENIED,
        tool_ir.ToolStatus.FATAL: rt.ToolExecutionStatus.ERROR,
    }
    return status_map[ir_status]


def execution_status_to_tool_ir_status(
    ex_status: rt.ToolExecutionStatus | str,
) -> tool_ir.ToolStatus:
    """Convert canonical ToolExecutionStatus back to legacy tool_ir.ToolStatus.
    
    Reverse Mapping:
      OK → SUCCESS
      DEGRADED → RETRY
      DENIED → CONTRACT_VIOLATION
      ERROR → FATAL
      TIMEOUT → FATAL (treat as non-recoverable)
      SKIPPED → FATAL (not normal path for legacy)
    """
    if isinstance(ex_status, str):
        ex_status = rt.ToolExecutionStatus(ex_status)
    
    status_map = {
        rt.ToolExecutionStatus.OK: tool_ir.ToolStatus.SUCCESS,
        rt.ToolExecutionStatus.DEGRADED: tool_ir.ToolStatus.RETRY,
        rt.ToolExecutionStatus.DENIED: tool_ir.ToolStatus.CONTRACT_VIOLATION,
        rt.ToolExecutionStatus.ERROR: tool_ir.ToolStatus.FATAL,
        rt.ToolExecutionStatus.TIMEOUT: tool_ir.ToolStatus.FATAL,
        rt.ToolExecutionStatus.SKIPPED: tool_ir.ToolStatus.FATAL,
    }
    return status_map[ex_status]


def contract_result_to_tool_result(
    contract_result: tool_ir.ToolContractResult,
) -> rt.ToolResult:
    """Convert legacy ToolContractResult → new ToolResult.
    
    Preserves tool name, status, and data as payload.
    Marks as not replayable (safe default for legacy results).
    """
    payload = None
    if contract_result.data is not None:
        payload = rt.CanonicalPayload(
            content=contract_result.data,
            payload_type="legacy_tool_result",
        )
    
    error_msg = ""
    if contract_result.status in (
        tool_ir.ToolStatus.CONTRACT_VIOLATION,
        tool_ir.ToolStatus.FATAL,
    ):
        # Include error context and repair hint
        error_parts = []
        if contract_result.error_context:
            error_parts.append(f"context: {contract_result.error_context}")
        if contract_result.repair_hint:
            error_parts.append(f"hint: {contract_result.repair_hint}")
        error_msg = "; ".join(error_parts)
    
    return rt.ToolResult(
        tool_name=contract_result.tool_name,
        invocation_id="",  # Legacy doesn't have invocation_id
        status=tool_ir_status_to_execution_status(contract_result.status),
        payload=payload,
        error=error_msg,
        latency_ms=0.0,  # Unknown from legacy
        replay_safe=False,  # Conservative default
        metadata={
            "legacy_error_context": contract_result.error_context,
            "legacy_repair_hint": contract_result.repair_hint,
        },
    )


def tool_result_to_contract_result(
    tool_result: rt.ToolResult,
) -> tool_ir.ToolContractResult:
    """Convert new ToolResult → legacy ToolContractResult.
    
    Extracts payload as data, converts status back.
    Preserves error context in error_context field.
    """
    data = None
    if tool_result.payload is not None:
        data = tool_result.payload.content
    
    error_context = dict(tool_result.metadata.get("legacy_error_context", {}))
    if tool_result.error:
        error_context["execution_error"] = tool_result.error
    
    repair_hint = tool_result.metadata.get("legacy_repair_hint", "")
    if not repair_hint and tool_result.status == rt.ToolExecutionStatus.DENIED:
        repair_hint = "Tool invocation was denied by policy or contract."
    
    return tool_ir.ToolContractResult(
        tool_name=tool_result.tool_name,
        status=execution_status_to_tool_ir_status(tool_result.status),
        data=data,
        error_context=error_context,
        repair_hint=repair_hint,
    )


class LegacyToolAdapter:
    """Phase B.2 adapter: Bridges old dispatch to new registry.
    
    Old code calls: adapter.execute_tool(tool_name, args) → dict
    New code calls: executor(invocation) → ToolResult
    
    This adapter translates between them without breaking old callers.
    """

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        """Initialize adapter with optional registry.
        
        If registry is None, old dispatch is used (backward compat).
        """
        self.registry = registry
        self.context = (
            ToolExecutionContext(registry)
            if registry is not None
            else None
        )

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        invocation_id: str = "",
    ) -> dict[str, Any]:
        """Execute tool and return dict (old interface).
        
        Args:
            tool_name: Name of tool to execute
            arguments: Dict of tool arguments
            invocation_id: Optional trace ID (for new registry path)
        
        Returns:
            Dict with keys: status, output, error, latency_ms, tool_name
        """
        if self.context is None:
            raise RuntimeError(
                "Adapter has no registry; cannot execute tools. "
                "Initialize with ToolRegistry."
            )
        
        # Resolve tool from registry
        resolved = self.context.registry.resolve(tool_name)
        if not resolved:
            return {
                "tool_name": tool_name,
                "status": tool_ir.ToolStatus.FATAL.value,
                "output": None,
                "error": f"Tool {tool_name!r} not registered",
                "latency_ms": 0.0,
            }
        
        spec, _ = resolved
        
        # Create invocation
        invocation = rt.ToolInvocation(
            invocation_id=invocation_id or f"legacyadapter-{tool_name}",
            tool_spec=spec,
            arguments=arguments or {},
        )
        
        # Execute via new context
        result = self.context.execute(invocation)
        
        # Convert result to legacy dict format
        legacy_status = execution_status_to_tool_ir_status(result.status)
        
        return {
            "tool_name": result.tool_name,
            "status": legacy_status.value,
            "output": result.payload.content if result.payload else None,
            "error": result.error or None,
            "latency_ms": result.latency_ms,
        }


__all__ = [
    "tool_ir_status_to_execution_status",
    "execution_status_to_tool_ir_status",
    "contract_result_to_tool_result",
    "tool_result_to_contract_result",
    "LegacyToolAdapter",
]
