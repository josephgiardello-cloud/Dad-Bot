"""Service-level Protocol contracts for the DadBot turn pipeline.

Each protocol corresponds to one graph node's service dependency.  Decorated
with ``@runtime_checkable`` so ``isinstance(service, HealthService)`` works at
boot for structural validation without requiring full inheritance.

Usage
-----
Validation is performed in ``DadBotOrchestrator._build_turn_graph``.  If a
registered service is missing a required method the orchestrator logs a warning
(not a hard failure) so degraded-mode recovery still works.

To add a new pipeline service:
1. Define a ``@runtime_checkable`` Protocol here.
2. Register the service in ``dadbot/registry.py``.
3. Add a validation entry in ``DadBotOrchestrator._build_turn_graph``.
"""
from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dadbot.contracts import FinalizedTurnResult

if TYPE_CHECKING:
    from dadbot.core.graph import TurnContext

logger = logging.getLogger(__name__)


@runtime_checkable
class HealthService(Protocol):
    """Runs pre-turn maintenance and proactive engagement checks."""

    def tick(self, context: "TurnContext") -> dict[str, Any]: ...


@runtime_checkable
class MemoryService(Protocol):
    """Builds contextual memory / profile payload for the current turn."""

    def build_context(self, context: "TurnContext") -> dict[str, Any]: ...


@runtime_checkable
class InferenceService(Protocol):
    """Drives LLM inference and agentic tool execution.

    ``run_agent`` *must* be a coroutine function â€” validated explicitly at
    boot because ``@runtime_checkable`` only checks name presence, not
    async/sync distinction.
    """

    async def run_agent(self, context: "TurnContext", rich_context: dict[str, Any]) -> Any: ...


@runtime_checkable
class SafetyPolicyService(Protocol):
    """Enforces reply safety and TONY-score tone policies."""

    def enforce_policies(self, context: "TurnContext", candidate: Any) -> FinalizedTurnResult: ...


@runtime_checkable
class PersistenceService(Protocol):
    """Persists the finalized turn result and flushes session state."""

    def save_turn(self, context: "TurnContext", result: FinalizedTurnResult) -> None: ...


@runtime_checkable
class PipelineNode(Protocol):
    """Interface every composable graph node must satisfy."""

    name: str

    async def run(self, context: "TurnContext") -> "TurnContext": ...


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _service_issues(service: Any, protocol: type, name: str) -> list[str]:
    """Return human-readable issues if *service* does not satisfy *protocol*."""
    issues: list[str] = []

    # Structural check: all public Protocol members must be present
    for member in [m for m in dir(protocol) if not m.startswith("_")]:
        if not hasattr(service, member):
            issues.append(f"{name}.{member}: missing")

    # Async check: InferenceService.run_agent must be a coroutine function
    if protocol is InferenceService:
        run_agent = getattr(service, "run_agent", None)
        if run_agent is not None and not inspect.iscoroutinefunction(run_agent):
            issues.append(f"{name}.run_agent: must be async (currently sync)")

    return issues


def validate_pipeline_services(
    services: dict[str, tuple[Any, type]],
    *,
    raise_on_failure: bool = False,
) -> list[str]:
    """Validate a mapping of ``{alias: (instance, Protocol)}`` pairs.

    Args:
        services: mapping of service alias â†’ (instance, Protocol class)
        raise_on_failure: if True, raises ``RuntimeError`` on any issue;
            otherwise logs warnings and returns the issue list.

    Returns:
        List of issue strings (empty if all services are conformant).
    """
    all_issues: list[str] = []
    for alias, (instance, protocol) in services.items():
        issues = _service_issues(instance, protocol, alias)
        all_issues.extend(issues)
        for issue in issues:
            logger.warning("Service contract violation: %s", issue)

    if raise_on_failure and all_issues:
        raise RuntimeError(
            "Pipeline service contract violations:\n" + "\n".join(f"  - {i}" for i in all_issues)
        )
    return all_issues
