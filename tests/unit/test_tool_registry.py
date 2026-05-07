"""Tests for Phase B: Tool Registry and Executor."""

from __future__ import annotations

import pytest

from dadbot.core.runtime_types import (
    CanonicalPayload,
    ExecutionIdentity,
    ToolDeterminismClass,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
    ToolSideEffectClass,
    ToolSpec,
)
from dadbot.core.tool_registry import (
    ToolContract,
    ToolExecutionContext,
    ToolRegistry,
)

pytestmark = pytest.mark.unit


def _make_simple_executor(status=ToolExecutionStatus.OK, payload_content="ok"):
    """Factory for test executors."""

    def executor(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            tool_name=invocation.tool_spec.name,
            invocation_id=invocation.invocation_id,
            status=status,
            payload=CanonicalPayload(payload_content, payload_type="test_output"),
            latency_ms=10.0,
            replay_safe=True,
        )

    return executor


def test_tool_registry_registers_and_resolves_tool():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="weather",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    executor = _make_simple_executor()

    registry.register(spec, executor)
    resolved = registry.resolve("weather", version="1.0.0")

    assert resolved is not None
    resolved_spec, resolved_executor = resolved
    assert resolved_spec.name == "weather"
    assert resolved_executor is executor


def test_tool_registry_case_insensitive_lookup():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="Weather",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    executor = _make_simple_executor()

    registry.register(spec, executor)
    resolved = registry.resolve("weather")

    assert resolved is not None
    assert resolved[0].name == "Weather"


def test_tool_registry_returns_latest_version_when_none_specified():
    registry = ToolRegistry()
    spec_v1 = ToolSpec(
        name="api",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    spec_v2 = ToolSpec(
        name="api",
        version="2.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )

    registry.register(spec_v1, _make_simple_executor())
    registry.register(spec_v2, _make_simple_executor())

    resolved = registry.resolve("api")  # No version specified
    assert resolved is not None
    assert resolved[0].version == "2.0.0"


def test_tool_registry_discover_by_capability():
    registry = ToolRegistry()
    spec_weather = ToolSpec(
        name="weather",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
        capabilities=frozenset({"lookup_temperature", "forecast"}),
    )
    spec_calendar = ToolSpec(
        name="calendar",
        version="1.0.0",
        determinism=ToolDeterminismClass.DETERMINISTIC,
        side_effect_class=ToolSideEffectClass.LOGGED,
        capabilities=frozenset({"schedule_event", "check_availability"}),
    )

    registry.register(spec_weather, _make_simple_executor())
    registry.register(spec_calendar, _make_simple_executor())

    # Discover by capability
    results = registry.discover(capability="lookup_temperature")
    assert len(results) == 1
    assert results[0].name == "weather"


def test_tool_registry_discover_by_determinism():
    registry = ToolRegistry()
    spec_readonly = ToolSpec(
        name="readonly_tool",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    spec_stateful = ToolSpec(
        name="stateful_tool",
        version="1.0.0",
        determinism=ToolDeterminismClass.DETERMINISTIC,
        side_effect_class=ToolSideEffectClass.STATEFUL,
    )

    registry.register(spec_readonly, _make_simple_executor())
    registry.register(spec_stateful, _make_simple_executor())

    results = registry.discover(determinism=ToolDeterminismClass.READ_ONLY)
    assert len(results) == 1
    assert results[0].name == "readonly_tool"


def test_tool_registry_discover_by_side_effects():
    registry = ToolRegistry()
    spec_pure = ToolSpec(
        name="pure_tool",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    spec_logged = ToolSpec(
        name="logged_tool",
        version="1.0.0",
        determinism=ToolDeterminismClass.DETERMINISTIC,
        side_effect_class=ToolSideEffectClass.LOGGED,
    )

    registry.register(spec_pure, _make_simple_executor())
    registry.register(spec_logged, _make_simple_executor())

    # Discover tools WITHOUT side effects
    pure_results = registry.discover(has_side_effects=False)
    assert len(pure_results) == 1
    assert pure_results[0].name == "pure_tool"

    # Discover tools WITH side effects
    effects_results = registry.discover(has_side_effects=True)
    assert len(effects_results) == 1
    assert effects_results[0].name == "logged_tool"


def test_tool_registry_list_registered():
    registry = ToolRegistry()
    spec_v1 = ToolSpec(
        name="tool",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    spec_v2 = ToolSpec(
        name="tool",
        version="2.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )

    registry.register(spec_v1, _make_simple_executor())
    registry.register(spec_v2, _make_simple_executor())

    listing = registry.list_registered()
    assert "tool" in listing
    assert set(listing["tool"]) == {"1.0.0", "2.0.0"}


def test_tool_registry_rejects_invalid_spec():
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="requires non-empty name"):
        registry.register(
            ToolSpec(
                name="",
                version="1.0.0",
                determinism=ToolDeterminismClass.READ_ONLY,
                side_effect_class=ToolSideEffectClass.PURE,
            ),
            _make_simple_executor(),
        )


def test_tool_registry_rejects_non_callable_executor():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="test",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )

    with pytest.raises(ValueError, match="must be callable"):
        registry.register(spec, "not_callable")


def test_tool_contract_validates_tool_name():
    spec = ToolSpec(
        name="weather",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    invocation = ToolInvocation(
        invocation_id="inv-1",
        tool_spec=ToolSpec(
            name="calendar",
            version="1.0.0",
            determinism=ToolDeterminismClass.READ_ONLY,
            side_effect_class=ToolSideEffectClass.PURE,
        ),
    )

    is_valid, error = ToolContract.validate(spec, invocation)
    assert is_valid is False
    assert "tool name mismatch" in error


def test_tool_execution_context_executes_registered_tool():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="greet",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    executor = _make_simple_executor(
        status=ToolExecutionStatus.OK,
        payload_content={"greeting": "hello world"},
    )
    registry.register(spec, executor)

    context = ToolExecutionContext(registry)
    invocation = ToolInvocation(
        invocation_id="inv-greet-1",
        tool_spec=spec,
    )

    result = context.execute(invocation)

    assert result.status == ToolExecutionStatus.OK
    assert result.succeeded() is True
    assert context.execution_count == 1
    assert context.last_result is result


def test_tool_execution_context_returns_error_for_unregistered_tool():
    registry = ToolRegistry()
    context = ToolExecutionContext(registry)

    spec = ToolSpec(
        name="nonexistent",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    invocation = ToolInvocation(
        invocation_id="inv-1",
        tool_spec=spec,
    )

    result = context.execute(invocation)

    assert result.status == ToolExecutionStatus.ERROR
    assert "not registered" in result.error
    assert result.failed() is True


def test_tool_execution_context_marks_replayable_based_on_spec():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="readonly",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )

    def executor_no_replay(invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            tool_name=invocation.tool_spec.name,
            invocation_id=invocation.invocation_id,
            status=ToolExecutionStatus.OK,
            payload=CanonicalPayload("data", payload_type="output"),
            replay_safe=False,  # Explicitly false
        )

    registry.register(spec, executor_no_replay)
    context = ToolExecutionContext(registry)
    invocation = ToolInvocation(
        invocation_id="inv-1",
        tool_spec=spec,
    )

    result = context.execute(invocation)

    # Should be marked replayable because spec is READ_ONLY, even though executor said false
    assert result.replay_safe is True
    assert result.is_replayable() is True
