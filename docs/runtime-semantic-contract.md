# Runtime Semantic Contract (One Page)

Status: Active runtime contract.

## Purpose

Define one unambiguous execution semantic for turn traces so runtime validation,
deterministic hashing, replay checks, and failure handling all use the same model.

## Primary Ordering Model

- Ordering model: lineage_graph
- Source of truth: stage lineage ancestry, not pure adjacent sequence transitions.
- Linear sequence is still required for contiguous event identity and deterministic
  replay indexing.

Code anchors:
- dadbot/core/invariant_gate.py
  - PRIMARY_EXECUTION_ORDERING_MODEL
  - validate_execution_semantics(...)

## Event Semantic Classes

- structural: stage_enter, stage_skip, turn_start
- transitional: kernel_ok, kernel_rejected, kernel_error, parallel_start, parallel_done
- terminal: stage_done, stage_error, turn_succeeded, turn_failed, turn_short_circuit

Code anchors:
- dadbot/core/invariant_gate.py
  - EXECUTION_EVENT_CLASS

## Validity Domains

- linear_sequences
  - sequence values must be contiguous and monotonic.
- lineage_graphs
  - stage-scoped events must preserve open-lineage ancestry constraints.
  - transitional events are lineage-validated (not strict FSM adjacency validated).
- hybrid_paths
  - terminal closure and no-post-terminal continuation enforcement.
  - closed-lineage required for successful completion.
  - open-lineage tolerated for terminal failure and short-circuit outcomes.

Code anchors:
- dadbot/core/invariant_gate.py
  - EVENT_VALIDITY_DOMAIN
  - validate_execution_semantics(...)

## Canonical Trace Reduction Rule

- Rule id: lineage-minimal-v1
- Function: reduce_execution_trace(...)
- Behavior:
  - keep all structural and terminal events;
  - compress transitional noise by keeping only the final transitional event in a
    stage segment before that stage reaches terminal closure;
  - preserve transitional events on kernel alias stages that are outside declared
    pipeline stage names.

Code anchors:
- dadbot/core/invariant_gate.py
  - CANONICAL_TRACE_REDUCTION_RULE_VERSION
  - reduce_execution_trace(...)

## Enforcement Points

- Transition-time checks:
  - every trace append is semantically validated on the current in-progress trace.
  - closure is not forced during in-progress execution.

- Finalization-time checks:
  - full trace is validated;
  - canonical reduced trace is computed and validated;
  - contract hash is computed from reduced canonical events only.

Code anchors:
- dadbot/core/graph.py
  - _record_execution_trace(...)
  - _finalize_execution_trace_contract(...)

## Contract Outputs

- execution_trace_contract fields:
  - version
  - event_count
  - reduced_event_count
  - ordering_model
  - reduction_rule
  - trace_hash
- canonical_execution_trace stored in turn state for downstream verification.

Code anchors:
- dadbot/core/graph.py
  - execution_trace_contract payload assembly
  - turn_context.state["canonical_execution_trace"]

## Determinism Boundary Conditions

- deterministic fallback ids derived from stable hash tokens.
- scheduler wait behavior bounded by max_scheduler_cycles, not wall-clock branching.

Code anchors:
- dadbot/core/control_plane.py
  - ExecutionJob._stable_token(...)
  - max_scheduler_cycles handling in submit loop
