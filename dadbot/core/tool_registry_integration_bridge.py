"""Integration bridge from old tool dispatch pattern to new Tool Registry.

This document and accompanying examples show how the new ToolRegistry (Phase B)
replaces the monolithic _dispatch_builtin_tool() pattern with:

  1. Declarative tool specifications (ToolSpec with contracts)
  2. Separate executor functions (Protocol-based, testable, substitutable)
  3. Version negotiation and capability discovery
  4. Typed results (ToolResult) replacing untyped dicts

Target file for integration: dadbot/utils/external_tool_runtime.py

Migration Strategy:
  - Phase B.1: Parallel registration (new registry coexists with old dispatch)
  - Phase B.2: Adapter layer (old callers use registry via ToolExecutionContext)
  - Phase B.3: Cutover (remove old _dispatch_builtin_tool, use registry directly)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dadbot.core.runtime_types import (
    CanonicalPayload,
    ExecutionIdentity,
    ToolDeterminismClass,
    ToolInvocation,
    ToolResult,
    ToolSideEffectClass,
    ToolSpec,
    ToolStatus,
)
from dadbot.core.tool_registry import ToolExecutionContext, ToolRegistry


@dataclass
class MigrationExample:
    """Example showing old dispatch → new registry mapping."""

    old_pattern: str
    new_pattern: str
    rationale: str


EXAMPLES = [
    MigrationExample(
        old_pattern="""
# OLD: Monolithic if/elif tree
def _dispatch_builtin_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "weather":
        return handle_weather(arguments.get("location"))
    elif tool_name == "calendar":
        return handle_calendar(arguments.get("event_id"))
    else:
        return {"error": "Unknown tool"}
""",
        new_pattern="""
# NEW: Registry-based dispatch
registry = ToolRegistry()

spec_weather = ToolSpec(
    name="weather",
    version="1.0.0",
    determinism=ToolDeterminismClass.READ_ONLY,
    side_effect_class=ToolSideEffectClass.PURE,
    capabilities=frozenset({"temperature_lookup", "forecast"}),
)
registry.register(spec_weather, execute_weather_tool)

spec_calendar = ToolSpec(
    name="calendar",
    version="2.0.0",
    determinism=ToolDeterminismClass.DETERMINISTIC,
    side_effect_class=ToolSideEffectClass.LOGGED,
    capabilities=frozenset({"schedule", "availability_check"}),
)
registry.register(spec_calendar, execute_calendar_tool)

context = ToolExecutionContext(registry)
invocation = ToolInvocation(invocation_id="inv-1", tool_spec=spec_weather)
result = context.execute(invocation)  # ToolResult (typed, auditable)
""",
        rationale="""
Advantages:
  1. Declarative: Tool contracts are data, not control flow
  2. Version negotiation: Supports multiple versions, semver sorting
  3. Capability discovery: Tools can be filtered, not just dispatched
  4. Typed results: ToolResult has status, payload, effects (not generic dict)
  5. Testability: Executors are substitutable Protocol instances
  6. Policy integration: Results can be routed through policy compiler
""",
    ),
    MigrationExample(
        old_pattern="""
# OLD: Untyped executor
def handle_weather(location: str) -> dict[str, Any]:
    temp = fetch_temperature(location)
    return {
        "location": location,
        "temperature": temp,
        "unit": "F",
        # Implicit contract: caller expects these keys
    }
""",
        new_pattern="""
# NEW: Typed executor (Protocol)
from typing import Protocol

class ToolExecutor(Protocol):
    def __call__(self, invocation: ToolInvocation) -> ToolResult: ...

def execute_weather_tool(invocation: ToolInvocation) -> ToolResult:
    location = invocation.arguments.get("location", "")
    temp = fetch_temperature(location)
    
    payload = CanonicalPayload(
        content={"location": location, "temperature": temp, "unit": "F"},
        payload_type="weather_data",
    )
    return ToolResult(
        tool_name=invocation.tool_spec.name,
        invocation_id=invocation.invocation_id,
        status=ToolStatus.OK,
        payload=payload,
        latency_ms=15.5,
        replay_safe=True,  # READ_ONLY tools are always replayable
    )
""",
        rationale="""
Improvements:
  1. Explicit contract: Executor signature is Protocol, not implicit
  2. Deterministic payloads: CanonicalPayload is hashable JSON
  3. Status tracking: ToolStatus enum, not implicit success
  4. Audit trail: latency_ms, replay_safe flag for recovery
  5. Structured error handling: ToolStatus.ERROR with error field
""",
    ),
    MigrationExample(
        old_pattern="""
# OLD: Caller filters tools manually
def find_readonly_tools(tool_list: list[str]) -> list[str]:
    # Magic constant checks or external docs
    readonly = ["weather", "calendar_list", "static_data"]
    return [t for t in tool_list if t in readonly]
""",
        new_pattern="""
# NEW: Registry provides capability discovery
readonly_tools = registry.discover(
    determinism=ToolDeterminismClass.READ_ONLY
)
tool_names = [spec.name for spec in readonly_tools]
""",
        rationale="""
Registry-based discovery:
  1. Single source of truth: Tool properties declared once
  2. Extensible: Filter by determinism, side-effects, capability, or combination
  3. Type-safe: Returns list[ToolSpec], not generic list[str]
  4. Auditable: Policy can inspect specs before execution
""",
    ),
    MigrationExample(
        old_pattern="""
# OLD: Error handling is implicit
def call_tool(tool_name: str, args: dict) -> dict:
    try:
        result = _dispatch_builtin_tool(tool_name, args)
        if "error" in result:
            # Unclear: Is this a tool error or runtime error?
            log_error(result["error"])
        return result
    except Exception as e:
        # Unstructured error
        return {"error": str(e)}
""",
        new_pattern="""
# NEW: Error handling is structured via ToolStatus enum
def call_tool(tool_name: str, args: dict) -> ToolResult:
    invocation = ToolInvocation(
        invocation_id=generate_id(),
        tool_spec=spec,
        arguments=args,
    )
    result = context.execute(invocation)
    
    if result.failed():
        # Explicit: status is ToolStatus.ERROR or TIMEOUT or DENIED
        policy_decision = policy_compiler.evaluate(result)
        # Can apply recovery strategy based on status
        return policy_decision.final_output
    
    return result
""",
        rationale="""
Structured error handling:
  1. ToolStatus enum: ERROR, TIMEOUT, DENIED, DEGRADED, SKIPPED
  2. Explicit error field: result.error has context
  3. Replayability: result.replay_safe prevents broken recovery
  4. Policy integration: Errors can trigger policy rules
""",
    ),
]


def example_phase_b1_parallel_registration():
    """Example: Phase B.1 - New registry coexists with old dispatch."""
    print("=== Phase B.1: Parallel Registration ===")
    print("""
    # Initialize new registry alongside old dispatch
    registry = ToolRegistry()
    
    # Register new tools with specs
    spec_weather = ToolSpec(
        name="weather",
        version="1.0.0",
        determinism=ToolDeterminismClass.READ_ONLY,
        side_effect_class=ToolSideEffectClass.PURE,
    )
    registry.register(spec_weather, execute_weather_tool)
    
    # Old code still calls _dispatch_builtin_tool()
    # New code calls registry-based execution
    # Both coexist until cutover is validated
    """)


def example_phase_b2_adapter_layer():
    """Example: Phase B.2 - Adapter bridges old and new."""
    print("=== Phase B.2: Adapter Layer ===")
    print("""
    class LegacyToolAdapter:
        '''Adapter for old _dispatch_builtin_tool callers.'''
        
        def __init__(self, registry: ToolRegistry):
            self.context = ToolExecutionContext(registry)
        
        def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            '''Old interface: (tool_name, args) -> dict'''
            # Resolve tool from registry
            resolved = self.context.registry.resolve(tool_name)
            if not resolved:
                return {"error": f"Tool {tool_name} not registered"}
            
            spec, _ = resolved
            invocation = ToolInvocation(
                invocation_id=generate_id(),
                tool_spec=spec,
                arguments=arguments,
            )
            result = self.context.execute(invocation)
            
            # Convert ToolResult back to dict for old callers
            return {
                "status": result.status.value,
                "payload": result.payload.content,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }
    
    # Old code continues unchanged
    adapter = LegacyToolAdapter(registry)
    result_dict = adapter.call_tool("weather", {"location": "NYC"})
    """)


def example_phase_b3_cutover():
    """Example: Phase B.3 - Cutover removes old dispatch."""
    print("=== Phase B.3: Cutover ===")
    print("""
    # OLD CODE REMOVED:
    # def _dispatch_builtin_tool(...): ...
    
    # NEW CODE (direct registry usage):
    class ToolExecutor:
        def __init__(self, registry: ToolRegistry):
            self.context = ToolExecutionContext(registry)
        
        def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
            resolved = self.context.registry.resolve(tool_name)
            if not resolved:
                return ToolResult(
                    tool_name=tool_name,
                    invocation_id=kwargs.get("invocation_id", "unknown"),
                    status=ToolStatus.ERROR,
                    payload=CanonicalPayload({"error": "not registered"}, "error"),
                    latency_ms=0,
                    replay_safe=False,
                )
            
            spec, _ = resolved
            invocation = ToolInvocation(
                invocation_id=kwargs.get("invocation_id", generate_id()),
                tool_spec=spec,
                arguments=kwargs,
            )
            return self.context.execute(invocation)
    
    # Usage: Direct, typed results
    executor = ToolExecutor(registry)
    result = executor.execute_tool("weather", location="NYC", invocation_id="inv-1")
    assert isinstance(result, ToolResult)
    """)


if __name__ == "__main__":
    print("\n".join([f"\n## Example {i+1}\nOLD:\n{ex.old_pattern}\n\nNEW:\n{ex.new_pattern}\n\nRationale:\n{ex.rationale}\n" for i, ex in enumerate(EXAMPLES)]))

    print("\n\n=== MIGRATION PHASES ===\n")
    example_phase_b1_parallel_registration()
    print("\n")
    example_phase_b2_adapter_layer()
    print("\n")
    example_phase_b3_cutover()

    print("\n\n=== INTEGRATION CHECKLIST ===")
    print("""
    [] Phase B.1: Register all tools from _dispatch_builtin_tool in ToolRegistry
         - Identify all tool branches in external_tool_runtime.py
         - Create ToolSpec for each tool
         - Implement ToolExecutor Protocol for each
         - Validate tool version, determinism, side-effects
    
    [] Phase B.2: Create LegacyToolAdapter for backward compatibility
         - Implement call_tool(tool_name, arguments) -> dict
         - Validate old callers receive same dict structure
         - Run existing test suite with adapter
    
    [] Phase B.3: Cutover (after all callers migrated)
         - Remove _dispatch_builtin_tool()
         - Remove LegacyToolAdapter
         - Direct callers to ToolExecutor
         - Validate performance (registry lookup vs direct dispatch)
    
    [] Phase C: Policy Integration
         - Route ToolResult through policy_compiler
         - Emit PolicyEffect for tool execution
         - Build audit trail of decisions
    
    [] Phase D: Recovery Strategy Selection
         - Policy decisions trigger RecoveryAction
         - Retry logic uses Checkpoint for replay
         - Bounded retries for nondeterministic tools
    """)
