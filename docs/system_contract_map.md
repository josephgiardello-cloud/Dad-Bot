# System Contract Map

Status: normative.
Version: 2026-05-07.v2.
Scope: graph, ledger, persistence, error model.

## 1) Contract Authority Order

When semantics conflict, apply this order:

1. Runtime behavior in implementation code.
2. Contract tests listed in this document.
3. This contract map.
4. Older narrative docs.

If a semantic change is intentional, update code, tests, and this map in the same change.

## 2) Graph Contract

### G-1 Node Binding Semantics

- Rule: duplicate node registration is latest-write-wins.
- Authority: dadbot/core/graph.py, TurnGraph.add_node.
- Behavior: calling add_node with the same name replaces the previous binding in _node_map.

Test anchors:

- tests/test_graph_unit_contracts.py::TestPipelineSemantics::test_duplicate_node_registration_keeps_latest_binding

### G-2 Pipeline Resolution Semantics

- Rule: if node-map mode is active and entry node is set, pipeline traversal follows configured edges from entry to tail.
- Authority: dadbot/core/graph.py, TurnGraph._pipeline_items.

Test anchors:

- tests/test_graph_unit_contracts.py::TestPipelineSemantics::test_pipeline_items_follow_registered_edges

### G-3 Execution Token Boundary

- Rule: when a required execution token is set, execution without a matching token boundary fails.
- Authority: dadbot/core/graph.py, set_required_execution_token and execute path checks.

Test anchors:

- tests/test_graph_unit_contracts.py::TestExecutionTokenBoundary::test_execute_raises_when_token_mismatch

## 3) Ledger Contract

### L-1 Sealed Events View

- Rule: sealed_events is an immutable tuple container over live authoritative event dict entries.
- Authority: dadbot/core/execution_ledger.py, ExecutionLedger.sealed_events.
- Consequence: tuple shape is immutable, but dict item mutation is reflected in ledger state.

Test anchors:

- tests/test_production_hardening_wave4.py::TestSealedEventsAndReplayFilters::test_sealed_events_returns_tuple
- tests/test_production_hardening_wave4.py::TestSealedEventsAndReplayFilters::test_sealed_events_exposes_live_authoritative_events

### L-2 Write Boundary and Causal Chain

- Rule: strict_writes mode requires WriteBoundaryGuard.
- Rule: per-session causal parent chain must match session head.
- Authority: dadbot/core/execution_ledger.py, _ensure_write_allowed and write.

Test anchors:

- tests/test_production_hardening_wave3.py::TestWriteBoundary::test_strict_mode_blocks_direct_write
- tests/test_production_hardening_wave3.py::TestWriteBoundary::test_write_boundary_guard_allows_write

### L-3 Sequence and Replay Semantics

- Rule: ledger maintains global sequence plus per-trace next-sequence counters.
- Rule: strict replay equality uses contiguous per-trace sequence with deterministic event id derivation.
- Authority:
  - dadbot/core/execution_ledger.py, get_next_sequence and get_next_trace_sequence.
  - dadbot/core/kernel_locks.py, KernelReplaySequenceLock.strict_hash and canonical_event.

Test anchors:

- tests/test_boundary_mutation_closure.py::test_replay_is_source_of_truth
- tests/test_boundary_mutation_closure.py::test_replay_matches_live_execution

### L-4 Replay Filter Semantics

- Rule: replay filters are applied during load-from-backend replay ingestion.
- Authority: dadbot/core/execution_ledger.py, add_replay_filter and load path.

Test anchors:

- tests/test_production_hardening_wave4.py::TestSealedEventsAndReplayFilters::test_replay_filter_applied_on_load

### L-5 Ledger Telemetry

- Rule: cache_rebuild_count is a first-class counter incremented on cache rebuild and exported in telemetry_snapshot.
- Authority: dadbot/core/execution_ledger.py, _rebuild_cache and telemetry_snapshot.

Test anchors:

- tests/test_contract_runtime_telemetry.py::test_execution_ledger_telemetry_snapshot_tracks_cache_rebuilds

## 4) Persistence Contract

### P-1 Turn Event Authority

- Rule: persist_turn_event requires active execution trace context.
- Rule: turn sequence assignment uses per-trace O(1) accessor when available.
- Authority: dadbot/managers/conversation_persistence.py, persist_turn_event and _compute_next_sequence.

Test anchors:

- tests/test_ledger_authority_invariants.py::test_persistence_events_are_ledger_authoritative

### P-2 Snapshot Thinness and Invariants

- Rule: checkpoint/snapshot payloads are thin and must exclude banned derived keys.
- Rule: payload size limit is enforced.
- Authority: dadbot/managers/conversation_persistence.py, _assert_snapshot_invariants and _thin_checkpoint_payload.

Test anchors:

- tests/test_ledger_authority_invariants.py::test_graph_checkpoint_snapshot_is_canonical_and_thin

### P-3 Compaction Policy Contract

- Rule: active compaction interval is every 25 turn events.
- Rule: periodic full snapshot reads still occur for strict anchoring.
- Rule: telemetry publishes recommended interval and retention under pressure.
- Authority: dadbot/managers/conversation_persistence.py.

Test anchors:

- tests/test_production_hardening_wave4.py::TestCompaction::test_compact_removes_pre_snapshot_events

### P-4 Post-Commit Emission Contract

- Rule: post-commit ready event is emitted once on successful finalize path.
- Rule: no post-commit ready event is emitted on failed finalize path.
- Authority: dadbot/services/persistence.py, _finalize_turn_impl and _finalize_emit_completion_event.

Test anchors:

- tests/test_post_commit_events.py::test_finalize_turn_emits_post_commit_event_once_on_success
- tests/test_post_commit_events.py::test_finalize_turn_does_not_emit_post_commit_event_on_failed_commit

### P-5 Memory Authority Divergence Contract

- Rule: strict mode blocks commit on divergence against event-sourced checkpoint authority.
- Rule: non-strict mode records soft failure and allows commit continuation.
- Authority: dadbot/services/persistence.py, _enforce_memory_authority.

Test anchors:

- tests/test_post_commit_events.py::test_finalize_turn_blocks_commit_on_memory_authority_divergence
- tests/test_post_commit_events.py::test_finalize_turn_allows_commit_when_memory_authority_matches

### P-6 Persistence Telemetry and SLO Contract

- Rule: persistence telemetry exposes write and compaction latency percentiles, counters, and SLO status.
- Rule: policy recommendation fields are always emitted.
- Authority: dadbot/managers/conversation_persistence.py, persistence_telemetry_snapshot.

Test anchors:

- tests/test_contract_runtime_telemetry.py::test_persistence_telemetry_snapshot_exposes_slo_and_policy

## 5) Error Model Contract

### E-1 Runtime Error Taxonomy

- Canonical runtime boundary errors:
  - RuntimeExecutionError
  - InvariantViolation
  - ProjectionMismatch
  - PersistenceFailure
  - ExecutionStageError
- Authority: dadbot/core/runtime_errors.py.

Test anchors:

- tests/test_post_commit_events.py::test_finalize_turn_does_not_emit_post_commit_event_on_failed_commit

### E-2 Finalize Wrapping Semantics

- Rule: finalize-path commit failures are wrapped as PersistenceFailure in strict finalize flow.
- Rule: underlying root failure is preserved in __cause__.
- Authority: dadbot/services/persistence.py, _finalize_turn_impl.

Test anchors:

- tests/test_post_commit_events.py::test_finalize_turn_does_not_emit_post_commit_event_on_failed_commit
- tests/test_post_commit_events.py::test_finalize_turn_blocks_commit_on_memory_authority_divergence

### E-3 Non-Fatal Runtime Catch Surface

- Rule: known runtime non-fatal exception classes are centrally enumerated and reused for guarded catch/rethrow boundaries.
- Authority: dadbot/core/runtime_errors.py, NON_FATAL_RUNTIME_EXCEPTIONS.

Test anchors:

- tests/test_post_commit_events.py::test_finalize_turn_does_not_emit_post_commit_event_on_failed_commit

## 6) Drift Control Rules

### D-1 Required Update Bundle

Any semantic change touching G-1..G-3, L-1..L-5, P-1..P-6, or E-1..E-3 must include:

1. code changes,
2. test updates/additions,
3. contract map version bump.

Test anchors:

- tests/test_contract_map_governance.py::test_contract_map_declares_required_update_bundle_rule

### D-2 CI Contract Gate

The following suites are the minimum semantic gate for this map:

- tests/test_graph_unit_contracts.py
- tests/test_production_hardening_wave4.py::TestSealedEventsAndReplayFilters
- tests/test_post_commit_events.py

Test anchors:

- tests/test_contract_map_governance.py::test_contract_map_declares_ci_contract_gate_rule
- tests/test_contract_map_governance.py::test_ci_gate_exposes_contract_gate_flag

Compiler workflow:

- Compile and validate anchors:
  - c:/Users/josep/OneDrive/Desktop/Dad-Bot/.venv/Scripts/python.exe tools/contract_test_compiler.py --validate-nodeids --fail-on-untested
- Execute compiled contract suite:
  - c:/Users/josep/OneDrive/Desktop/Dad-Bot/.venv/Scripts/python.exe tools/contract_test_compiler.py --run-tests --fail-on-untested
- CI drift check (manifest up-to-date):
  - c:/Users/josep/OneDrive/Desktop/Dad-Bot/.venv/Scripts/python.exe tools/contract_test_compiler.py --check --fail-on-untested

### D-3 Documentation Staleness Rule

If any older doc contradicts this map, this map is authoritative and the older doc must be updated or marked historical.

Test anchors:

- tests/test_contract_map_governance.py::test_contract_map_declares_documentation_staleness_rule

## 7) Revision Log

- 2026-05-07.v1
  - Formalized latest-write graph node binding.
  - Formalized sealed-events tuple-plus-live-entry semantics.
  - Formalized strict finalize wrapper behavior using PersistenceFailure with preserved cause.
  - Added telemetry/SLO and compaction policy contract statements.

- 2026-05-07.v2
  - Added contract-test anchors for all contract IDs.
  - Enabled strict compiler workflow with --fail-on-untested in CI guidance.