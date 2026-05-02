"""Kernel-owned single execution spine for all agentic tool calls.

All tool execution in the service layer MUST go through ``execute_tool``.
The private sandbox implementation is unreachable outside the allowed core
execution-spine modules and enforced by CI RULE16_TOOL_SANDBOX_ISOLATION.

Why a function, not a class?
-----------------------------
The ToolSandbox is intentionally a per-call object — one sandbox per tool
invocation, scoped to a single agentic turn step.  Wrapping it in a singleton
or a long-lived object would break the isolation guarantee.  This module owns
the instantiation decision; callers see only the ``execute_tool`` interface.
"""

from __future__ import annotations

from typing import Any, Callable

from dadbot.core._tool_sandbox import ToolExecutionRecord, _ToolSandbox


def execute_tool(
    *,
    tool_name: str,
    parameters: dict[str, Any] | None = None,
    executor: Callable[[], Any],
    compensating_action: Callable[[], None] | None = None,
) -> ToolExecutionRecord:
    """Single kernel-owned execution spine for all agentic tool calls.

    Creates a scoped private tool sandbox, executes the tool with idempotency and
    failure isolation, and returns the execution record.  Never raises.

    Parameters
    ----------
    tool_name:
        Canonical name of the tool (e.g. ``"set_reminder"``).
    parameters:
        Tool call parameters dict.  Used for idempotency key derivation.
    executor:
        Zero-argument callable that performs the actual tool work.
    compensating_action:
        Optional zero-argument callable registered for LIFO rollback.
        Callers that need the execution result inside the compensating action
        should close over a result holder populated after this function returns.
    """
    sandbox = _ToolSandbox()
    return sandbox.execute(
        tool_name=tool_name,
        parameters=parameters,
        executor=executor,
        compensating_action=compensating_action,
    )
