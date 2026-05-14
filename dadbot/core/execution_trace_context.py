from __future__ import annotations

"""Compatibility re-export for legacy execution trace imports.

Canonical implementation lives in dadbot.core.execution_context.
This module preserves the legacy import path without duplicating logic.
"""

from dadbot.core.execution_context import ExternalSystemCallGraph
from dadbot.core.execution_context import ExecutionTraceContext
from dadbot.core.execution_context import ExecutionTraceRecorder
from dadbot.core.execution_context import RuntimeTraceViolation
from dadbot.core.execution_context import ToolExecutionTraceNode
from dadbot.core.execution_context import active_execution_trace
from dadbot.core.execution_context import bind_execution_trace
from dadbot.core.execution_context import build_execution_trace_context
from dadbot.core.execution_context import build_external_system_call_graph
from dadbot.core.execution_context import build_tool_invocation_projection
from dadbot.core.execution_context import canonicalize_execution_trace_context
from dadbot.core.execution_context import derive_execution_trace_hash
from dadbot.core.execution_context import ensure_execution_trace_root
from dadbot.core.execution_context import record_execution_step
from dadbot.core.execution_context import record_external_system_call
from dadbot.core.execution_context import require_execution_trace

__all__ = [
    "ExternalSystemCallGraph",
    "ExecutionTraceContext",
    "ExecutionTraceRecorder",
    "RuntimeTraceViolation",
    "ToolExecutionTraceNode",
    "active_execution_trace",
    "bind_execution_trace",
    "build_execution_trace_context",
    "build_external_system_call_graph",
    "build_tool_invocation_projection",
    "canonicalize_execution_trace_context",
    "derive_execution_trace_hash",
    "ensure_execution_trace_root",
    "record_execution_step",
    "record_external_system_call",
    "require_execution_trace",
]
