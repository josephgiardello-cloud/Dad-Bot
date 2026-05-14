# Phase B Refinements: Type Collisions & Integration Gaps

## Critical Issues Found

### 1. ToolStatus Enum Collision

**Current State:**
- `tool_ir.py` defines: `ToolStatus` = {SUCCESS, RETRY, CONTRACT_VIOLATION, FATAL}
- `runtime_types.py` defines: `ToolStatus` = {OK, ERROR, TIMEOUT, DENIED, DEGRADED, SKIPPED}
- `nodes.py` imports from `tool_ir.py`

**Problem:**
- Two incompatible ToolStatus enums with same name
- `nodes.py` imports tool_ir.ToolStatus, causing namespace collision with Phase A types
- dispatch_registered_tool() returns ToolContractResult (tool_ir), not ToolResult (runtime_types)

**Impact:**
- Cannot import both without aliasing
- Phase B ToolRegistry.execute() returns ToolResult(status=ToolStatus...) but dispatch_registered_tool() returns ToolContractResult(status=ToolStatus...)
- Breaks backward compatibility

**Solution:**
Rename Phase A enums to avoid collision:
- `ToolStatus` → `ToolExecutionStatus` (more specific, Phase A/B canonical)
- Keep `ToolContractResult` as bridge layer (old system outputs this)
- Create adapters: ToolContractResult ↔ ToolResult

### 2. ToolResult Type Collision

**Current State:**
- `tool_ir.py` defines: `ToolResult` = {tool_name, status, output, deterministic_id}
- `runtime_types.py` defines: `ToolResult` = {tool_name, status, payload, effects, latency_ms, replay_safe, error}

**Problem:**
- tool_ir.ToolResult is simple, untyped payload (Any)
- runtime_types.ToolResult has structured fields for audit/replay
- They're incompatible

**Solution:**
Treat as parallel systems:
- Keep tool_ir.ToolResult (legacy, old dispatch returns this)
- Keep runtime_types.ToolResult (new registry returns this)
- Create explicit conversion layer

### 3. Registry System Collision

**Current State:**
- `nodes.py` has: register_tool(), get_registered_tool(), _TOOL_REGISTRY dict
- `tool_registry.py` has: ToolRegistry class with register(), resolve(), discover()

**Problem:**
- Two incompatible registration systems
- dispatch_registered_tool() uses nodes._TOOL_REGISTRY
- ToolExecutionContext uses ToolRegistry

**Solution:**
- Phase B.1: Run both in parallel
- Phase B.2: ToolRegistry.bootstrap_from_legacy(nodes._TOOL_REGISTRY) for migration
- Phase B.3: Eventually replace (after tests pass)

### 4. Only 3 Tools in Current System

**Actual Registered Tools:**
- memory_lookup (query)
- echo (message)
- current_time ()

**Problem:**
- Small set makes Phase B integration testing difficult
- tool_registry_integration_bridge.py examples show weather, calendar, etc. (fictional)
- Need integration tests with real tools

**Solution:**
- Create integration test with actual 3 tools
- Validate registry can bootstrap from them
- Test bidirectional conversion

## Refinement Plan

### Step 1: Rename Phase A Types (Avoid Collision)
- `ToolStatus` → `ToolExecutionStatus`
- Update runtime_types.py and test_runtime_types.py
- Update tool_registry.py and test_tool_registry.py

### Step 2: Create Compatibility Layer
- File: `dadbot/core/runtime_types_compat.py`
- Functions:
  - `contract_result_to_tool_result(ToolContractResult) → ToolResult`
  - `tool_result_to_contract_result(ToolResult) → ToolContractResult`
  - Enum mapping: old ToolStatus ↔ ToolExecutionStatus

### Step 3: Bootstrap Integration Test
- File: `tests/unit/test_tool_registry_bootstrap.py`
- Test 1: Load 3 real tools from nodes._TOOL_REGISTRY into ToolRegistry
- Test 2: Call via old dispatch_registered_tool() - should work unchanged
- Test 3: Call via new ToolExecutionContext.execute() - should work
- Test 4: Results convert correctly both directions

### Step 4: Update Integration Bridge
- Reference actual dispatch_registered_tool() (not fictional _dispatch_builtin_tool)
- Document real tools (memory_lookup, echo, current_time)
- Show B.1/B.2/B.3 with concrete examples
- Update migration timeline

### Step 5: Write Bridge Layer Example
- Create: `dadbot/core/registry_bootstrap.py`
- `ToolRegistry.from_legacy_registry(nodes._TOOL_REGISTRY) → ToolRegistry`
- Validates all legacy tools can be registered
- Shows how Phase B.1 parallel registration works

## Files to Modify

| File | Change | Reason |
|------|--------|--------|
| dadbot/core/runtime_types.py | Rename ToolStatus → ToolExecutionStatus | Avoid collision with tool_ir.ToolStatus |
| tests/unit/test_runtime_types.py | Update imports/references | Follow renames |
| dadbot/core/tool_registry.py | Update ToolStatus → ToolExecutionStatus | Follow renames |
| tests/unit/test_tool_registry.py | Update imports/references | Follow renames |
| dadbot/core/runtime_types_compat.py | NEW - Compatibility layer | Bridge old/new |
| tests/unit/test_tool_registry_bootstrap.py | NEW - Integration tests | Validate real tools |
| dadbot/core/registry_bootstrap.py | NEW - Bootstrap helper | Phase B.1 support |
| dadbot/core/tool_registry_integration_bridge.py | Update examples | Use real tools |

## Success Criteria

- [ ] ToolStatus collision resolved (renamed to ToolExecutionStatus)
- [ ] tool_ir.ToolStatus still works (no breaking changes to old code)
- [ ] Compatibility converters tested (ToolContractResult ↔ ToolResult)
- [ ] Real 3-tool integration test passes
- [ ] Both dispatch_registered_tool() and ToolExecutionContext.execute() work correctly
- [ ] Bootstrap from legacy registry works without errors
- [ ] All existing tests still pass
- [ ] Phase C can proceed without type conflicts

## Estimated Effort

- Renames: 10 min
- Compatibility layer: 20 min
- Bootstrap helper: 15 min
- Integration tests: 30 min
- Bridge doc updates: 10 min
- **Total: ~1.5 hours for full refinement**

## Risk Assessment

**Low Risk:**
- Renames are mechanical
- Old code unaffected (tool_ir.py unchanged)
- New code isolated in compat layer

**Medium Risk:**
- Integration tests might reveal assumptions about tool structure
- Bootstrap might need field mapping tweaks

**Mitigation:**
- Keep tool_ir.py completely untouched initially
- Run full test suite after each step
- Document assumptions in bootstrap code
