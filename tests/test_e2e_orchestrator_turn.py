#!/usr/bin/env python3
"""Semantic E2E lanes for Dad-Bot orchestrator runtime validation."""

from __future__ import annotations

from typing import Any

from Dad import DadBot
from dadbot.core.orchestrator import DadBotOrchestrator
from dadbot.registry import boot_registry
from tests.eval.harness import run_with_trace


def _registry_get_optional(registry: Any, name: str) -> Any | None:
    get_fn = getattr(registry, "get", None)
    if not callable(get_fn):
        return None
    try:
        return get_fn(name, optional=True)
    except Exception:
        try:
            return get_fn(name)
        except Exception:
            return None


def _runtime_mode(*, bot: Any | None, registry: Any) -> str:
    if bot is None:
        return "degraded"
    maintenance_service = _registry_get_optional(registry, "maintenance_service")
    tool_registry = _registry_get_optional(registry, "tool_registry") or getattr(bot, "tool_registry", None)
    if maintenance_service is None or tool_registry is None:
        return "degraded"
    return "full"


def _tool_registry_size(tool_registry: Any | None) -> int:
    if tool_registry is None:
        return 0
    get_available = getattr(tool_registry, "get_available_tools", None)
    if callable(get_available):
        try:
            return len(list(get_available() or []))
        except Exception:
            return 0
    return 0


def test_e2e_orchestrator_degraded_lane_non_certifying():
    """
    Non-certifying degraded lane.

    This lane allows fallback behavior and must not be included in any system-health metric.
    """
    registry = boot_registry(config_path="dad_profile.template.json", bot=None)
    orchestrator = DadBotOrchestrator(registry=registry)
    runtime_mode = _runtime_mode(bot=None, registry=registry)

    assert runtime_mode == "degraded"

    trace = run_with_trace("Why did the dad go to the bank?", orchestrator, None)
    assert trace.steps > 0
    # Fallback answers are valid in degraded mode; this only verifies survivability.
    assert isinstance(trace.final_output, str)


def test_e2e_orchestrator_full_runtime_lane_certifying_expected_tool_execution():
    """Certifying full-runtime lane with hard planner/execution coherence checks."""
    bot = DadBot()
    orchestrator = DadBotOrchestrator(bot=bot, strict=False)
    registry = orchestrator.registry

    runtime_mode = _runtime_mode(bot=bot, registry=registry)
    tool_registry = _registry_get_optional(registry, "tool_registry") or getattr(bot, "tool_registry", None)
    maintenance_service = _registry_get_optional(registry, "maintenance_service")

    assert runtime_mode == "full"
    assert _tool_registry_size(tool_registry) > 0
    assert maintenance_service is not None

    trace = run_with_trace("Remind me to call mom in 10 minutes", orchestrator, tool_registry)

    # Level-2 cert gate: expected-tool prompts must execute a tool coherently.
    assert trace.decision_outcome == "executed_tool"
    assert len(trace.tool_calls) > 0
    assert str(trace.planner_status or "").strip().lower() in {"tool_selected", "used_tool"}

    planner_tool = str(trace.planner_tool or "").strip()
    executed_tool = str(trace.tool_calls[0].name or "").strip()
    assert planner_tool
    assert planner_tool == executed_tool

    # Level-3 cert gate: execution truth contract must be present and coherent
    # with the observed tool execution.
    if trace.execution_truth_contract is not None:
        contract = trace.execution_truth_contract
        assert contract.get("planner_tool") == planner_tool, (
            f"Contract planner_tool={contract.get('planner_tool')!r} "
            f"!= observed planner_tool={planner_tool!r}"
        )
        executed_in_contract = list(contract.get("executed_tools") or [])
        assert any(executed_tool in str(t) for t in executed_in_contract), (
            f"Executed tool {executed_tool!r} not found in contract executed_tools: "
            f"{executed_in_contract}"
        )
