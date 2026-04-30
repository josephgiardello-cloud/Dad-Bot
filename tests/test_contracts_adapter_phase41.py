"""Phase 4.1 — contracts_adapter explicit fallback contract tests.

Verifies:
- FallbackRegistry tracks declared vs undeclared fallbacks
- ContextSchemaAdapter.ensure_context_builder_methods() emits FallbackEvents
- Undeclared fallback in strict mode raises ContractViolationError
- safe_build_memory_context emits fallback event on error
- ContractAdapterRegistry.audit_fallback_usage() surfaces events
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dadbot.core.contracts_adapter import (
    ContextSchemaAdapter,
    ContractAdapterRegistry,
    ContractViolationError,
    FallbackEvent,
    FallbackRegistration,
    FallbackRegistry,
)

# ---------------------------------------------------------------------------
# FallbackRegistry unit tests
# ---------------------------------------------------------------------------


def test_fallback_registry_declared_returns_callable():
    reg = FallbackRegistry()
    reg.register(
        FallbackRegistration(
            name="my_method",
            version="1.0.0",
            fallback_callable=lambda: "sentinel",
            contract_description="test",
            substituted_signature="my_method(self) -> str",
        )
    )
    result = reg.use("my_method", source="test", reason="missing")
    assert callable(result)
    assert result() == "sentinel"


def test_fallback_registry_emits_event_on_use():
    reg = FallbackRegistry()
    reg.register(
        FallbackRegistration(
            name="fn",
            version="2.0.0",
            fallback_callable=lambda: None,
            contract_description="desc",
            substituted_signature="fn() -> None",
        )
    )
    reg.use("fn", source="src", reason="absent")
    events = reg.audit()
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, FallbackEvent)
    assert ev.fallback_source == "src"
    assert ev.fallback_reason == "absent"
    assert ev.contract_violation_trigger == "fn"
    assert ev.declared is True
    assert ev.version == "2.0.0"


def test_fallback_registry_undeclared_lenient_returns_noop():
    reg = FallbackRegistry()
    fn = reg.use("unknown_method", source="s", reason="r", strict=False)
    assert callable(fn)
    assert fn() is None  # no-op
    events = reg.audit()
    assert events[0].declared is False


def test_fallback_registry_undeclared_strict_raises():
    reg = FallbackRegistry()
    with pytest.raises(ContractViolationError, match="Undeclared fallback"):
        reg.use("unknown_method", source="s", reason="r", strict=True)


def test_fallback_registry_declared_names():
    reg = FallbackRegistry()
    reg.register(
        FallbackRegistration(
            name="a",
            version="1",
            fallback_callable=lambda: None,
            contract_description="",
            substituted_signature="",
        )
    )
    reg.register(
        FallbackRegistration(
            name="b",
            version="1",
            fallback_callable=lambda: None,
            contract_description="",
            substituted_signature="",
        )
    )
    assert set(reg.declared_names()) == {"a", "b"}


# ---------------------------------------------------------------------------
# ContextSchemaAdapter — ensure_context_builder_methods
# ---------------------------------------------------------------------------


def test_ensure_context_builder_methods_injects_missing():
    """Methods absent on the stub are injected via declared fallbacks."""
    adapter = ContextSchemaAdapter(bot=None)
    stub = SimpleNamespace()  # no methods at all
    events = adapter.ensure_context_builder_methods(stub)
    # All 7 required methods should now be present
    for name in [
        "build_core_persona_prompt",
        "build_dynamic_profile_context",
        "build_relationship_context",
        "build_session_summary_context",
        "build_memory_context",
        "build_relevant_context",
        "build_cross_session_context",
    ]:
        assert hasattr(stub, name), f"Expected '{name}' to be injected"
    assert len(events) == 7


def test_ensure_context_builder_methods_skips_present():
    """Methods already present are not replaced."""
    adapter = ContextSchemaAdapter(bot=None)
    sentinel = object()
    stub = SimpleNamespace(build_core_persona_prompt=sentinel)
    events = adapter.ensure_context_builder_methods(stub)
    assert stub.build_core_persona_prompt is sentinel
    assert len(events) == 6  # only the 6 missing ones


def test_ensure_context_builder_methods_none_is_noop():
    adapter = ContextSchemaAdapter(bot=None)
    events = adapter.ensure_context_builder_methods(None)
    assert events == []


def test_ensure_context_builder_methods_strict_undeclared_raises(monkeypatch):
    """Strict mode + undeclared method → ContractViolationError."""
    # Use a custom registry with no declarations
    empty_reg = FallbackRegistry()
    adapter = ContextSchemaAdapter(bot=None, fallback_registry=empty_reg)
    stub = SimpleNamespace()

    monkeypatch.setenv("PHASE4_STRICT", "1")
    try:
        with pytest.raises(ContractViolationError):
            adapter.ensure_context_builder_methods(stub)
    finally:
        monkeypatch.delenv("PHASE4_STRICT", raising=False)


def test_ensure_context_builder_methods_lenient_with_module_registry():
    """Default (lenient) mode completes without error; all 7 methods declared."""
    adapter = ContextSchemaAdapter(bot=None)
    stub = SimpleNamespace()
    events = adapter.ensure_context_builder_methods(stub)
    # All declared → no ContractViolationError; events are FallbackEvent instances
    assert all(isinstance(e, FallbackEvent) for e in events)
    assert all(e.declared for e in events)


# ---------------------------------------------------------------------------
# ContextSchemaAdapter — safe_build_memory_context
# ---------------------------------------------------------------------------


def test_safe_build_memory_context_returns_fallback_on_none():
    adapter = ContextSchemaAdapter(bot=None)
    result = adapter.safe_build_memory_context(None, "hi", fallback="FB")
    assert result == "FB"


def test_safe_build_memory_context_returns_result_on_success():
    adapter = ContextSchemaAdapter(bot=None)
    stub = SimpleNamespace(build_memory_context=lambda user_input: f"ctx:{user_input}")
    result = adapter.safe_build_memory_context(stub, "hello")
    assert result == "ctx:hello"


def test_safe_build_memory_context_emits_event_on_exception():
    adapter = ContextSchemaAdapter(bot=None)

    def _raise(user_input: str) -> str:
        raise ValueError("boom")

    stub = SimpleNamespace(build_memory_context=_raise)
    result = adapter.safe_build_memory_context(stub, "x", fallback="safe")
    assert result == "safe"
    events = adapter.fallback_audit()
    assert any("build_memory_context" in e.contract_violation_trigger and "boom" in e.fallback_reason for e in events)


# ---------------------------------------------------------------------------
# ContractAdapterRegistry
# ---------------------------------------------------------------------------


def test_contract_adapter_registry_audit_fallback_usage():
    registry = ContractAdapterRegistry(bot=None)
    # Use a fresh isolated registry to avoid cross-test event leakage from the module-level one.
    fresh_reg = FallbackRegistry()
    from dadbot.core.contracts_adapter import _CONTEXT_BUILDER_FALLBACK_REGISTRY as _module_reg

    for name in _module_reg.declared_names():
        decl = _module_reg._declarations[name]
        fresh_reg.register(decl)
    registry.context_schema._registry = fresh_reg
    stub = SimpleNamespace()
    registry.context_schema.ensure_context_builder_methods(stub)
    events = registry.audit_fallback_usage()
    assert len(events) == 7
    assert all(isinstance(e, FallbackEvent) for e in events)


def test_contract_adapter_registry_validate_contracts_pass():
    """validate_contracts should return all True for a minimal but valid bot stub."""
    bot = SimpleNamespace(
        memory=SimpleNamespace(
            relationship_history=lambda **kw: [],
            mutate_memory_store=lambda **kw: None,
        ),
        runtime_timestamp=lambda: "now",
    )
    registry = ContractAdapterRegistry(bot=bot)
    results = registry.validate_contracts(raise_on_failure=True)
    assert all(results.values()), results
