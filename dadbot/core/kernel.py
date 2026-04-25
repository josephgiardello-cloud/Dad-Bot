"""Execution kernel â€” the single state-transition authority for turn processing.

Architecture contract
---------------------
``TurnKernel.execute_step()`` is the ONLY function permitted to drive turn state
transitions.  Every graph node execution passes through it.  Everything else is:

  - input preparation (building context, prompts, parameters)
  - policy suggestion (Bayesian scoring, mood detection)
  - event emission (checkpoints, audit logs, telemetry)

State mutations (``turn_context.state[key] = value``) are only legal inside a
call to ``execute_step()``.  This collapses three previous execution authorities
(graph, orchestrator, service layer) into one, which directly improves:

  * determinism â€” one path means one audit trail
  * replay correctness â€” kernel_audit in metadata records every state write
  * debugging clarity â€” policy decisions are recorded at the point they happen
  * failure reasoning â€” rejected steps are surfaced, not silently skipped

Policy gate
-----------
An optional ``policy_gate`` callable is checked before each step.  If it
returns ``PolicyDecision(allowed=False)`` the step is skipped and the rejection
is appended to ``turn_context.metadata["kernel_rejections"]``.  This is the
authoritative chokepoint for Bayesian governance.

``bayesian_policy_gate(bot)`` creates a gate backed by the bot's planner-debug
snapshot.  It propagates the Bayesian tool-bias into turn state so downstream
services (AgentService, TurnService) don't need to re-read planner_debug.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from dadbot.core.graph import TurnContext

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """Result of a single policy gate evaluation."""

    allowed: bool
    reason: str = ""
    # Suggested recovery action when allowed=False.
    # "reject" = drop the step result; "replan" = ask the pipeline to re-plan.
    action: str = "proceed"   # "proceed" | "reject" | "replan"


@dataclass
class KernelStepResult:
    """Execution record returned by ``TurnKernel.execute_step``."""

    step_name: str
    status: str                               # "ok" | "rejected" | "error"
    state_keys_written: list[str] = field(default_factory=list)
    policy: PolicyDecision | None = None
    error: str = ""


class KernelViolation(RuntimeError):
    """Raised when a disallowed action attempts to reach the execution kernel."""


# Step names that map to TurnPhase.ACT.  The policy gate only inspects these;
# all other steps are allowed unconditionally at the kernel level.
_ACT_STEP_NAMES: frozenset[str] = frozenset(
    {"inference", "agent", "tool", "act", "tool_execution"}
)

# Tokens that signal an explicit user request for tool-assisted responses.
_EXPLICIT_TOOL_KEYWORDS: frozenset[str] = frozenset({
    "remind", "reminder", "set alarm", "alarm",
    "search for", "search the web", "look up", "look it up",
    "find out", "find me", "check if", "google", "what is", "who is",
})


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class TurnKernel:
    """Single execution authority for all turn state transitions.

    Usage (inside TurnGraph._execute_node)::

        kernel_result = await self._kernel.execute_step(
            turn_context, node_name, _call_node
        )
        if kernel_result.status == "error":
            raise RuntimeError(kernel_result.error)
        # "rejected" â†’ step skipped; pipeline continues.

    The kernel records every step in ``turn_context.metadata["kernel_audit"]``
    so the full decision trail is available for replay and debugging.
    """

    def __init__(
        self,
        *,
        policy_gate: Callable[["TurnContext", str], PolicyDecision] | None = None,
    ) -> None:
        self._policy_gate = policy_gate

    async def execute_step(
        self,
        turn_context: "TurnContext",
        step_name: str,
        step_fn: Callable[[], Awaitable[Any]],
    ) -> KernelStepResult:
        """Execute *step_fn* under policy enforcement and write-key accounting.

        This is the ONLY entry point for state mutations.  Nodes still write
        directly to ``turn_context.state`` as a side effect; the kernel tracks
        which keys changed and records the outcome in the audit trail.

        Parameters
        ----------
        turn_context:
            The current turn's shared state carrier.
        step_name:
            Logical name of the step (used for policy lookup and audit trail).
        step_fn:
            Zero-arg async callable that performs the step.  May mutate
            ``turn_context.state`` as a side effect.
        """
        # â”€â”€ Phase 1: Policy gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        turn_context.metadata["kernel_lineage"] = {
            "kernel_step_id": str(step_name or ""),
            "trace_id": str(getattr(turn_context, "trace_id", "") or ""),
        }

        policy: PolicyDecision | None = None
        if self._policy_gate is not None:
            policy = self._policy_gate(turn_context, step_name)
            if not policy.allowed:
                rejection: dict[str, Any] = {
                    "step": step_name,
                    "reason": policy.reason,
                    "action": policy.action,
                }
                turn_context.metadata.setdefault("kernel_rejections", []).append(rejection)
                logger.info(
                    "TurnKernel: step %r rejected by policy gate: %s (action=%s)",
                    step_name,
                    policy.reason,
                    policy.action,
                )
                return KernelStepResult(
                    step_name=step_name,
                    status="rejected",
                    policy=policy,
                )

        # â”€â”€ Phase 2: Execute with write-key accounting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        keys_before: frozenset[str] = frozenset(turn_context.state)
        try:
            await step_fn()
        except Exception as exc:
            logger.error("TurnKernel: step %r raised: %s", step_name, exc)
            return KernelStepResult(
                step_name=step_name,
                status="error",
                policy=policy,
                error=str(exc),
            )

        keys_written: list[str] = sorted(frozenset(turn_context.state) - keys_before)

        # â”€â”€ Phase 3: Append audit record â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        audit_entry: dict[str, Any] = {
            "step": step_name,
            "status": "ok",
            "wrote": keys_written,
            "policy": policy.reason if policy else "unrestricted",
            "kernel_step_id": str(step_name or ""),
            "trace_id": str(getattr(turn_context, "trace_id", "") or ""),
        }
        turn_context.metadata.setdefault("kernel_audit", []).append(audit_entry)

        return KernelStepResult(
            step_name=step_name,
            status="ok",
            state_keys_written=keys_written,
            policy=policy,
        )


# ---------------------------------------------------------------------------
# Bayesian policy gate factory
# ---------------------------------------------------------------------------

def bayesian_policy_gate(bot: Any) -> Callable[["TurnContext", str], PolicyDecision]:
    """Return a policy gate backed by the bot's Bayesian planner state.

    Gate behaviour
    --------------
    * Non-ACT steps: always allowed (memory build, safety, save are never blocked).
    * ACT steps (inference / tool / agent):

      1. Reads current Bayesian tool-bias from ``bot.planner_debug_snapshot()``.
      2. Injects ``_bayesian_tool_bias_kernel`` into ``turn_context.state`` so
         AgentService and TurnService pick up the kernel-authorised bias without
         re-computing it.
      3. When bias is ``"defer_tools_unless_explicit"`` and the user input
         contains no explicit tool-request keywords, sets
         ``turn_context.state["_bayesian_tools_blocked"] = True`` â€” a hard signal
         for AgentService to skip tool-planning entirely.

    Note: the LLM inference call is always *allowed* so Tony always receives a
    response.  Per-tool gating remains with ``TurnService._bayesian_tool_gate``.
    The kernel gate here provides:

    * The single chokepoint where Bayesian policy is recorded in metadata.
    * Authoritative propagation of tool-bias into turn state.
    * The hook for future hard-block policies (maintenance mode, crisis override).
    """
    def _gate(turn_context: "TurnContext", step_name: str) -> PolicyDecision:
        if step_name not in _ACT_STEP_NAMES:
            return PolicyDecision(
                allowed=True,
                reason=f"non-act step: {step_name!r}",
            )

        planner_debug: dict[str, Any] = {}
        try:
            planner_debug = bot.planner_debug_snapshot() or {}
        except Exception:
            pass

        tool_bias = str(planner_debug.get("bayesian_tool_bias") or "planner_default")

        # Propagate kernel-authorised bias into turn state.
        turn_context.state["_bayesian_tool_bias_kernel"] = tool_bias

        # Hard-signal tool block when Bayesian policy says defer and user
        # has not explicitly requested a tool-assisted response.
        if tool_bias == "defer_tools_unless_explicit":
            user_lower = str(turn_context.user_input or "").lower()
            has_explicit = any(kw in user_lower for kw in _EXPLICIT_TOOL_KEYWORDS)
            if not has_explicit:
                turn_context.state["_bayesian_tools_blocked"] = True

        turn_context.metadata["kernel_policy"] = {
            "step": step_name,
            "tool_bias": tool_bias,
            "tools_blocked": turn_context.state.get("_bayesian_tools_blocked", False),
            "enforced": True,
        }

        return PolicyDecision(
            allowed=True,
            reason=f"policy={tool_bias!r}; kernel gate recorded",
        )

    return _gate
