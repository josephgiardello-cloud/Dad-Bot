# DadBot Core Architecture Simplification Roadmap

## Current State: 200+ files in `dadbot/core/`

This document identifies over-engineered layers and proposes a **60-70% reduction** in surface area while preserving core functionality.

---

## Tier 1: Essential Core (Keep As-Is)

These files are **non-negotiable** for the system to function:

### Relationship & Memory (10 files)
- `memory/` — Long-term memory store and queries
- `dadbot.py` — Facade (will be simplified)
- `orchestrator.py` — Turn execution orchestration
- `graph.py`, `nodes.py`, `graph_traverser.py` — Graph execution pipeline
- `turn_mixin.py`, `llm_mixin.py`, `health_mixin.py` — Core behaviors

### Determinism & Correctness (3 files)
- `execution_trace_context.py` — Trace context for replay
- `determinism_seal.py` — Determinism sealing
- `contracts.py` — Core contracts (not contract_evaluator, not contract_propagation)

### Services (5-6 files)
- Runtime services that are actually used by the facade
- Personality, mood, relationship managers (in services/)

---

## Tier 2: Redundant/Duplicated (DELETE or CONSOLIDATE)

These files have overlapping concerns and could be consolidated into 2-3 unified implementations:

### Execution Layer Explosion (70+ files → 5)
**Current:** `execution_*.py` (70+ variants)  
**Issues:** Multiple abstractions for the same concept:
  - `execution_context.py`, `execution_kernel.py`, `execution_kernel_spec.py`
  - `execution_result_unified.py`, `execution_result_schema.py`
  - `execution_semantics.py`, `execution_equivalence.py`, `execution_equivalence_oracle.py`

**Proposal:** Consolidate into:
- `execution.py` — Single unified execution model
- `execution_schema.py` — Unified result schema
- `determinism.py` — Determinism guarantees (already exists)

**Impact:** Delete 65 files, 1 unified module

### State Management Redundancy (8+ files → 1)
**Current:** `*_state*.py` files scattered across core
- `core_state.py`
- `system_state_model.py`
- `system_state_algebra.py`
- `belief_state_engine.py`
- `emotion_state.py`
- `implicit_state_guard.py`

**Proposal:** Single `RuntimeState` class that covers all state management  
**Impact:** Delete 6 files

### Control Plane Duplication (4+ files → 1)
**Current:**
- `control_plane.py`
- `control_plane_projection.py`
- `control_plane_reducer.py`

**Proposal:** Consolidate into single `ControlPlane` with internal helpers  
**Impact:** Delete 2-3 files

### Contract/Invariant Explosion (8+ files → 1)
**Current:** 
- `contracts.py`, `contract_evaluator.py`, `contract_propagation.py`, `contracts_adapter.py`
- `invariants/`, `invariant_engine.py`, `invariant_registry.py`, `invariant_gate.py`

**Proposal:** Single unified contract/invariant checker  
**Impact:** Delete 6-7 files

### IR Layers (5+ files → 1-2)
**Current:** Multiple IR representations for different concerns
- `policy_ir.py`
- `planner_ir.py`
- `reasoning_ir.py`
- `recovery_ir.py`
- `turn_ir.py`
- `tool_ir.py`

**Proposal:** Unified `ExecutionPlan` dataclass, not multiple IRs  
**Impact:** Delete 4-5 files

### Snapshot/Ledger Duplication (6+ files → 1-2)
**Current:**
- `snapshot_engine.py`, `decision_snapshot.py`, `system_snapshot.py`
- `ledger/` (backend, reader, writer, adapter, index)

**Proposal:** Simple `Snapshot` class + `EventLog` for ledger  
**Impact:** Delete 4-5 files, consolidate ledger into single module

### Topology/Graph Duplication (5+ files → 1)
**Current:**
- `topology_provider.py`, `topology_runtime.py`
- `graph_topology.py`, `graph_algebra.py`, `lg_topology.py`
- `execution_topology_graph.py`

**Proposal:** Single `Topology` class  
**Impact:** Delete 4 files

---

## Tier 3: Frameworks That Are Probably Dead Weight

These represent ambitious features that may not justify their complexity:

- `multi_agent_swarm.py` — Is multi-agent actually used?
- `autonomous_goal_daemon.py` — Is autonomous goal execution working?
- `behavior_alignment_trainer.py` — Is RLHF integration working?
- `hypothesis_engine.py`, `belief_state_engine.py` — Are these actually core?
- `distributed_correctness.py` — Is distributed mode actually used?
- `policy_compiler.py` — Is policy compilation working and necessary?

**Proposal:** Audit which of these are actually exercised in tests. Delete or move to `/archive/experimental/`.

---

## Tier 4: Organization of Remaining Essential Files

After consolidation, `dadbot/core/` would have:

```
dadbot/core/
├── dadbot.py                 # Facade (simplified)
├── orchestrator.py           # Turn orchestration
├── execution.py              # Unified execution model
├── contracts.py              # Unified contract/invariant checking
├── runtime_state.py          # Unified state management
├── control_plane.py          # Unified control plane
├── snapshot.py               # Unified snapshot/replay
├── event_log.py              # Unified event ledger
├── graph.py                  # Graph execution
├── nodes.py                  # Graph nodes
├── graph_traverser.py        # Graph traversal
├── turn_mixin.py, llm_mixin.py, health_mixin.py
├── boot_mixin.py, compat_mixin.py, convenience_mixin.py
├── execution_trace_context.py
├── determinism.py, determinism_seal.py
├── topology.py               # Unified topology
├── memory/                   # Unchanged
├── services/                 # Unchanged
└── testing/                  # Unchanged
```

**Result:** ~30-40 files instead of 200+

---

## Tier 5: Simplify the Facade (`dadbot.py`)

Current facade is **603 lines** of routing/delegation plumbing. Simplifications:

### 1. Remove Over-Specific Routing Maps
**Current:**
- `_CONFIG_ATTR_MAP` (36 entries)
- `_RUNTIME_STATE_ATTR_MAP` (11 entries)
- `_INTERNAL_RUNTIME_ATTR_MAP` (10 entries)
- `_MANAGER_DELEGATE_CHAIN` (36 entries)

**Proposal:** Replace with generic `__getattr__` that routes through:
1. Config (if attribute exists)
2. Services container (O(1) lookup)
3. Fail with AttributeError

**Impact:** Delete 150+ lines, gain clarity

### 2. Remove `_ManagerDescriptor` Class
**Current:** 20 lines of boilerplate to replace `@property` / `@setter` pairs  
**Proposal:** Use `__getattr__` + `__setattr__` instead  
**Impact:** Delete 20 lines, simplify

### 3. Consolidate Multiple `@property` Methods
Many redundant aliases (e.g., `context_builder → context_service`)  
**Proposal:** Keep only public API; rest route through `__getattr__`  
**Impact:** Delete 50-100 lines

### Result
Reduce `dadbot.py` from 603 → 250-300 lines while preserving public API

---

## Proposed Phasing

### Phase 1: Facade Simplification (Low Risk)
- Simplify `dadbot.py` by removing redundant routing
- Keep all behavior intact, just cleaner code
- **Time:** 2-3 hours
- **Tests:** Existing facade tests should pass

### Phase 2: Consolidate State Management (Medium Risk)
- Merge 8 state files into single `RuntimeState` class
- Update orchestrator to use unified state API
- **Time:** 4-6 hours
- **Tests:** Re-run determinism tests

### Phase 3: Consolidate Execution Layer (Medium-High Risk)
- Merge 70 execution files into 5
- Unified execution schema
- **Time:** 8-12 hours
- **Tests:** Full test suite

### Phase 4: Remove Dead Frameworks (Low Risk)
- Audit usage of `multi_agent_swarm`, `behavior_alignment_trainer`, etc.
- Archive or delete unused frameworks
- **Time:** 2-3 hours
- **Tests:** Existing tests should still pass

---

## Success Criteria

- ✅ `dadbot/core/` reduced from 200+ → 40-50 files
- ✅ Facade `dadbot.py` reduced from 603 → 250-300 lines
- ✅ **All existing tests pass** (no behavior change)
- ✅ No new abstractions; only deletions
- ✅ Remaining code is more obvious and maintainable

---

## Notes

- **This is NOT a refactor of core behavior.** The system's actual logic remains unchanged.
- **This IS a cleanup of scaffolding and redundancy** that has accumulated over iterations.
- **Priority is deletion**, not reorganization. If we're unsure if something is used, mark it for archival.
- **Conservative approach:** Do shallow consolidations first (Phase 1-2). Only tackle deep changes after confidence builds.

