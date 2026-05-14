# Cut #4: Execution-Mode & Delivery Path Audit

## Current Execution Mode Handling

### Execution Modes Defined
- **live**: Normal execution, first attempt
- **recovery**: Redelivery after failure, replicates from checkpoint
- **replay**: Historical replay (not actively used in branching)
- **degraded**: Graph degradation (tracked in runtime_issues, not formal mode)

### Execution Mode Resolution (Fragmented Across 2 Paths)

#### Path 1: control_plane.py::_resolved_execution_mode(job)
```python
def _resolved_execution_mode(job: "ExecutionJob") -> str:
    metadata = dict(job.metadata or {})
    explicit = str(metadata.get("execution_mode") or "").strip().lower()
    state = dict(metadata.get("execution_state") or {})
    if int(state.get("redelivery_count") or 0) > 0:
        return "recovery"
    if _coerce_lifecycle_state(state.get("lifecycle_state")) == ExecutionLifecycleState.RECOVERY_PENDING:
        return "recovery"
    if explicit in {"live", "replay", "recovery"}:
        return explicit
    return "live"
```
- Called at: job completion, lifecycle transitions, telemetry
- Location: control_plane.py:878

#### Path 2: orchestrator.py::_OrchestratorStateCoordinator.resolve_execution_mode(job, checkpoint)
```python
@staticmethod
def resolve_execution_mode(job, checkpoint):
    explicit = _coerce_execution_mode(dict(job.metadata or {}).get("execution_mode"))
    if explicit != "live":
        return explicit
    execution_state = dict(dict(job.metadata or {}).get("execution_state") or {})
    lifecycle_state = str(execution_state.get("lifecycle_state") or "").strip().lower()
    redelivery_count = int(execution_state.get("redelivery_count") or 0)
    if isinstance(checkpoint, dict) and checkpoint and (redelivery_count > 0 or lifecycle_state == "recovery_pending"):
        return "recovery"
    return "live"
```
- Called at: orchestrator.py:827-839 in `_prepare_execution_mode_from_checkpoint`
- Location: orchestrator.py:706

### Recovery Mode Behavior
- `orchestrator.py:839`: If execution_mode == "recovery", calls `_hydrate_context_from_checkpoint(context, loaded_checkpoint)`
- Checkpoint restoration includes:
  - State metadata (`_hydrate_checkpoint_state_metadata`)
  - Phase history (`_hydrate_phase_history`)
  - Stage traces (`_hydrate_stage_traces`)
  - Bookkeeping (`_hydrate_checkpoint_bookkeeping`)

### Degraded Mode Observation
- Not a formal execution mode, but recorded in runtime_issues
- Reflects in status reporting and UI warnings
- Triggered by graph exceptions (caught and logged, then pipeline continues)
- NO branching logic: system just continues with whatever output is available

### Current Delivery Paths (Multiple, Not Consolidated)

#### Delivery Path 1: ResponseEngine Selection (control_plane.py)
1. `execute_from_graph_context()` (line 3488)
2. `_build_response_engine_context(job, session_key, initial_response)`
3. ResponseEngine ranks responses based on:
   - Context
   - Session continuity
   - Response validity
4. Returns final response via `_finalize_submit_success()`

#### Delivery Path 2: Graph Pipeline (orchestrator + graph.py)
1. `orchestrator.py::_prepare_execution_mode_from_checkpoint()` resolves mode
2. If recovery mode: hydrates checkpoint state
3. `graph.py::execute(context)` runs pipeline
4. Pipeline stages:
   - Semantic understanding
   - Policy check (with bounded repair recovery from Cut #3)
   - Response generation/ranking
   - Safety gates
   - Delivery

#### Delivery Path 3: Legacy Fallback (when graph unavailable)
- Referenced in dad_streamlit.py as "degraded"
- Falls back to legacy turn processing
- No formal execution-mode awareness

### Identity/Authority Issues

#### Issue 1: Dual Mode Resolution
- Control plane and orchestrator both implement `resolve_execution_mode()`
- Different logic (control_plane checks explicit, redelivery_count, lifecycle; orchestrator adds checkpoint awareness)
- Both resolve to same value, but resolution logic is duplicated

#### Issue 2: Delivery Authority Fragmentation
- ResponseEngine is marked as "sole selection authority" (control_plane.py:3537 comment)
- But graph pipeline also selects responses
- Two independent response selection paths can diverge

#### Issue 3: Recovery Mode Loose Coupling
- Recovery mode resolved in orchestrator, but recovery state applied inconsistently
- Some checkpoints fully restored, others partially (depends on call site)
- No canonical recovery entry point

#### Issue 4: Degraded Mode Outside System
- Degraded transitions tracked in runtime_issues
- Not integrated into execution_mode enum
- Causes UI-layer branching instead of runtime-layer consolidation

## Cut #4 Implementation Plan

### Goal
Single canonical execution flow with mode-aware branching at well-defined decision points.

### Changes Needed

1. **Unified ExecutionModeResolver** (new file: `dadbot/core/execution_mode.py`)
   - Single `resolve_execution_mode(job, checkpoint) -> ExecutionMode`
   - Enum: `ExecutionMode.LIVE, RECOVERY, REPLAY, DEGRADED`
   - Called once at execution entry, stored in context
   - Remove duplicates from control_plane and orchestrator

2. **Canonical Recovery Entry Point** (orchestrator.py)
   - Move recovery restoration logic to single method
   - Call once at execution mode resolution
   - Never call elsewhere

3. **Unified Delivery Path** (control_plane.py + graph.py)
   - Remove parallel ResponseEngine/graph delivery paths
   - Define single canonical delivery: graph pipeline → ResponseEngine rank → final delivery
   - Add mode-aware branching only at recovery/replay decision points

4. **Degraded Mode Integration** (graph.py)
   - Catch graph exceptions
   - Set execution_mode to DEGRADED instead of just logging
   - Trigger fallback path explicitly

5. **Execution-Mode Guard Tests**
   - Test each mode transitions correctly
   - Test recovery restores full state
   - Test degraded triggers fallback
   - Test replay returns deterministic response

### Files to Modify
- NEW: `dadbot/core/execution_mode.py` (ExecutionModeResolver)
- `dadbot/core/control_plane.py` (remove _resolved_execution_mode, use resolver)
- `dadbot/core/orchestrator.py` (remove resolve_execution_mode, use resolver)
- `dadbot/core/graph.py` (integrate degraded mode, single delivery path)
- `tests/test_execution_mode_*.py` (new guard tests)
