"""Phase B Integration Tests: Bootstrap from legacy registry and compatibility.

Tests validate:
1. Compatibility converters work bidirectionally
2. Legacy 3-tool registry can be loaded into new ToolRegistry
3. Old and new dispatch systems coexist without conflicts
4. Results convert correctly both directions
"""

from __future__ import annotations

import pytest

from dadbot.core import nodes, tool_ir
from dadbot.core.runtime_types import (
    CanonicalPayload,
    ToolDeterminismClass,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
    ToolSideEffectClass,
    ToolSpec,
)
from dadbot.core.runtime_types_compat import (
    LegacyToolAdapter,
    contract_result_to_tool_result,
    execution_status_to_tool_ir_status,
    tool_ir_status_to_execution_status,
    tool_result_to_contract_result,
)
from dadbot.core.tool_registry import ToolRegistry

pytestmark = pytest.mark.unit


class TestStatusConversion:
    """Test bidirectional status enum conversion."""

    def test_success_converts_to_ok(self):
        result = tool_ir_status_to_execution_status(tool_ir.ToolStatus.SUCCESS)
        assert result == ToolExecutionStatus.OK

    def test_retry_converts_to_degraded(self):
        result = tool_ir_status_to_execution_status(tool_ir.ToolStatus.RETRY)
        assert result == ToolExecutionStatus.DEGRADED

    def test_contract_violation_converts_to_denied(self):
        result = tool_ir_status_to_execution_status(
            tool_ir.ToolStatus.CONTRACT_VIOLATION
        )
        assert result == ToolExecutionStatus.DENIED

    def test_fatal_converts_to_error(self):
        result = tool_ir_status_to_execution_status(tool_ir.ToolStatus.FATAL)
        assert result == ToolExecutionStatus.ERROR

    def test_ok_converts_back_to_success(self):
        result = execution_status_to_tool_ir_status(ToolExecutionStatus.OK)
        assert result == tool_ir.ToolStatus.SUCCESS

    def test_degraded_converts_back_to_retry(self):
        result = execution_status_to_tool_ir_status(ToolExecutionStatus.DEGRADED)
        assert result == tool_ir.ToolStatus.RETRY

    def test_denied_converts_back_to_contract_violation(self):
        result = execution_status_to_tool_ir_status(ToolExecutionStatus.DENIED)
        assert result == tool_ir.ToolStatus.CONTRACT_VIOLATION

    def test_error_converts_back_to_fatal(self):
        result = execution_status_to_tool_ir_status(ToolExecutionStatus.ERROR)
        assert result == tool_ir.ToolStatus.FATAL

    def test_timeout_converts_to_fatal(self):
        result = execution_status_to_tool_ir_status(ToolExecutionStatus.TIMEOUT)
        assert result == tool_ir.ToolStatus.FATAL


class TestResultConversion:
    """Test bidirectional result type conversion."""

    def test_contract_result_to_tool_result_success(self):
        contract_result = tool_ir.ToolContractResult(
            tool_name="test_tool",
            status=tool_ir.ToolStatus.SUCCESS,
            data={"answer": 42},
            error_context={},
            repair_hint="",
        )

        result = contract_result_to_tool_result(contract_result)

        assert result.tool_name == "test_tool"
        assert result.status == ToolExecutionStatus.OK
        assert result.payload is not None
        assert result.payload.content == {"answer": 42}
        assert result.error == ""
        assert result.replay_safe is False

    def test_contract_result_to_tool_result_error(self):
        contract_result = tool_ir.ToolContractResult(
            tool_name="test_tool",
            status=tool_ir.ToolStatus.FATAL,
            data=None,
            error_context={"exception": "ValueError"},
            repair_hint="Check input format",
        )

        result = contract_result_to_tool_result(contract_result)

        assert result.tool_name == "test_tool"
        assert result.status == ToolExecutionStatus.ERROR
        assert result.payload is None
        assert "ValueError" in result.error
        assert "Check input format" in result.error

    def test_tool_result_to_contract_result_roundtrip(self):
        original = tool_ir.ToolContractResult(
            tool_name="weather",
            status=tool_ir.ToolStatus.SUCCESS,
            data={"temp": 72},
            error_context={},
            repair_hint="",
        )

        # Convert to new, then back
        as_new = contract_result_to_tool_result(original)
        back_to_legacy = tool_result_to_contract_result(as_new)

        assert back_to_legacy.tool_name == original.tool_name
        assert back_to_legacy.status == original.status
        assert back_to_legacy.data == original.data

    def test_tool_result_denied_converts_with_repair_hint(self):
        result = ToolResult(
            tool_name="restricted_tool",
            invocation_id="inv-1",
            status=ToolExecutionStatus.DENIED,
            error="Policy denied access",
            replay_safe=False,
        )

        contract = tool_result_to_contract_result(result)

        assert contract.status == tool_ir.ToolStatus.CONTRACT_VIOLATION
        assert contract.repair_hint == "Tool invocation was denied by policy or contract."


class TestLegacyAdapterWithRealTools:
    """Test adapter using actual tools from nodes.py registry."""

    def test_adapter_can_execute_memory_lookup_via_registry(self):
        """Test executing real memory_lookup tool through adapter.
        
        This validates Phase B.2: Old code uses adapter, new registry under hood.
        """
        # Create registry and bootstrap from legacy _TOOL_REGISTRY
        registry = ToolRegistry()
        
        # Register the real memory_lookup tool from nodes.py
        legacy_registration = nodes.get_registered_tool("memory_lookup")
        assert legacy_registration is not None
        
        spec = ToolSpec(
            name="memory_lookup",
            version="1.0.0",
            determinism=ToolDeterminismClass.READ_ONLY,
            side_effect_class=ToolSideEffectClass.PURE,
            required_permissions=frozenset(),
        )
        
        # Create a wrapper executor that calls the legacy handler
        def executor(invocation: ToolInvocation) -> ToolResult:
            # Call legacy handler
            legacy_result = legacy_registration.handler(
                invocation.arguments,
                context=None,  # In real test, would be TurnContext
            )
            
            # Convert if needed
            if isinstance(legacy_result, tool_ir.ToolContractResult):
                return contract_result_to_tool_result(legacy_result)
            
            # Assume success
            return ToolResult(
                tool_name=invocation.tool_spec.name,
                invocation_id=invocation.invocation_id,
                status=ToolExecutionStatus.OK,
                payload=CanonicalPayload(legacy_result, payload_type="legacy"),
                latency_ms=5.0,
                replay_safe=True,
            )
        
        registry.register(spec, executor)
        
        # Create adapter
        adapter = LegacyToolAdapter(registry)
        
        # Execute via old interface
        result_dict = adapter.execute_tool(
            "memory_lookup",
            arguments={"query": "test query"},
            invocation_id="test-inv-1",
        )
        
        # Validate old interface
        assert "status" in result_dict
        assert "output" in result_dict
        assert "tool_name" in result_dict
        assert result_dict["tool_name"] == "memory_lookup"

    def test_legacy_tool_registry_list(self):
        """Verify actual tools registered in nodes.py."""
        tools = nodes.get_registered_tool_names()
        
        # Should have at least the 3 basic tools
        assert "memory_lookup" in tools
        assert "echo" in tools
        assert "current_time" in tools

    def test_legacy_echo_tool_has_required_args(self):
        """Verify echo tool requires 'message' arg."""
        tool_info = nodes.get_tool_required_args()
        
        assert "echo" in tool_info
        assert "message" in tool_info["echo"]

    def test_legacy_current_time_has_no_required_args(self):
        """Verify current_time tool needs no args."""
        tool_info = nodes.get_tool_required_args()
        
        assert "current_time" in tool_info
        assert len(tool_info["current_time"]) == 0


class TestAdapterErrorHandling:
    """Test adapter handles errors gracefully."""

    def test_adapter_returns_error_for_unregistered_tool(self):
        registry = ToolRegistry()
        adapter = LegacyToolAdapter(registry)
        
        result_dict = adapter.execute_tool(
            "nonexistent_tool",
            arguments={},
        )
        
        assert result_dict["status"] == tool_ir.ToolStatus.FATAL.value
        assert "not registered" in result_dict["error"]
        assert result_dict["output"] is None

    def test_adapter_requires_registry(self):
        adapter = LegacyToolAdapter(registry=None)
        
        with pytest.raises(RuntimeError, match="no registry"):
            adapter.execute_tool("any_tool", arguments={})


class TestCompatibilityLayerMetadata:
    """Test metadata preservation in conversions."""

    def test_legacy_error_context_preserved(self):
        contract_result = tool_ir.ToolContractResult(
            tool_name="api_tool",
            status=tool_ir.ToolStatus.FATAL,
            data=None,
            error_context={"http_status": 500, "retry_after": 60},
            repair_hint="Server overloaded, retry later",
        )

        result = contract_result_to_tool_result(contract_result)
        
        assert result.metadata["legacy_error_context"] == {
            "http_status": 500,
            "retry_after": 60,
        }
        assert result.metadata["legacy_repair_hint"] == "Server overloaded, retry later"

    def test_roundtrip_preserves_error_context(self):
        original_context = {"code": "NETWORK_TIMEOUT", "retries": 3}
        
        contract = tool_ir.ToolContractResult(
            tool_name="api_call",
            status=tool_ir.ToolStatus.RETRY,
            data=None,
            error_context=original_context,
            repair_hint="Transient network failure",
        )

        # Convert: legacy → new → legacy
        as_new = contract_result_to_tool_result(contract)
        back_to_legacy = tool_result_to_contract_result(as_new)
        
        # Error context should be preserved (in new metadata)
        assert back_to_legacy.error_context.get("code") == "NETWORK_TIMEOUT"
