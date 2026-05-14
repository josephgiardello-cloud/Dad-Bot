# P3/P4 Closure Checklist

Purpose: close the remaining determinism/confluence/replay/observability gaps with one concrete code change and one concrete test per gap.

## Section Summary

| Section | Current Completeness | Target | Gap Count |
|---|---:|---:|---:|
| Facade Routing | 100% | 100% | 0 |
| Memory Mutation/Commit/Observability | 100% | 100% | 0 |
| Memory Canonical Confluence | 100% | 100% | 0 |
| Graph Projection Invalidation | 100% | 100% | 0 |
| Canonical Event Boundary | 100% | 100% | 0 |
| Determinism Boundary | 100% | 100% | 0 |
| External Nondeterminism Graph | 100% | 100% | 0 |
| Replay/State Lineage Completeness | 100% | 100% | 0 |

## Closure Items

### 1) Determinism docs vs behavior mismatch

- Gap: RECORD-mode doc states structural drift detection, but implementation does not compare newly produced values against previous sealed values for the same slot.
- Code change:
  - Update docstring to match actual behavior OR implement explicit structural drift check in RECORD mode.
  - Preferred target: implement optional strict drift check behind an explicit flag in `DeterminismBoundary`.
- Primary files:
  - `dadbot/core/determinism.py`
- Concrete test:
  - Add `test_determinism_boundary_record_mode_detects_structural_drift_when_enabled`.
  - Arrange: inject baseline slot, then capture same slot with divergent payload under strict option.
  - Assert: raises `DeterminismViolation` with `reason` containing `structural` or `drift`.
- Status: done

### 2) Trace validation coverage narrower than canonical strip policy

- Gap: `validate_trace` checks only `FORBIDDEN_TRACE_FIELDS`, while `NON_CANONICAL_PAYLOAD_FIELDS` is broader.
- Code change:
  - Expand `validate_trace` to enforce either:
    - all keys in `NON_CANONICAL_PAYLOAD_FIELDS`, or
    - a strict-mode parameter that enforces full list.
- Primary files:
  - `dadbot/core/canonical_event.py`
- Concrete test:
  - Add `test_validate_trace_rejects_request_and_correlation_fields`.
  - Arrange: trace payload contains `request_id` and `correlation_id`.
  - Assert: `AssertionError` identifies offending field and event index.
- Status: done

### 3) Memory ordering total-order tie-break completeness

- Gap: current key (`updated_at`, `created_at`, `summary`) can tie, leaving residual input-order dependence.
- Code change:
  - Extend `memory_sort_key` with stable fallback dimensions (e.g., category, mood, and deterministic hash of normalized entry).
- Primary files:
  - `dadbot/memory/normalizers.py`
- Concrete test:
  - Add `test_memory_confluence_total_order_on_sort_key_ties`.
  - Arrange: two entries with same timestamps/summary but different other fields and reversed insertion order.
  - Assert: normalized store order and `canonical_state_hash` are identical across permutations.
- Status: done

### 4) Graph invalidation regression over multi-turn sequence

- Gap: single-operation invalidation is covered; repeated save cycles under evolving generation need direct regression.
- Code change:
  - Preserve current behavior and add lightweight instrumentation assertion path in tests only.
- Primary files:
  - `dadbot/memory/graph_manager.py`
  - `dadbot/memory/storage.py`
- Concrete test:
  - Add `test_graph_projection_generation_advances_across_repeated_mutations`.
  - Arrange: perform N `mutate_memory_store` operations with graph-tracked keys.
  - Assert: `_memory_graph_generation` strictly increases and projection cache is empty before sync each cycle.
- Status: done

### 5) Canonical hash observability event payload assertions are shallow

- Gap: test only validates presence/length of `_last_memory_state_hash`, not emitted event payload fidelity.
- Code change:
  - No behavior change required; add stronger assertions on the emitted `memory_state_canonicalized` step payload.
- Primary files:
  - `dadbot/memory/storage.py`
  - `tests/unit/test_p3_p4_runtime_contracts.py`
- Concrete test:
  - Add `test_memory_state_canonicalized_event_payload_matches_before_after_hashes`.
  - Arrange: mutate known baseline state.
  - Assert: event includes exact `before_hash`, `after_hash`, and correct `changed` boolean.
- Status: done

### 6) External system call graph observability completeness

- Gap: current test validates hash/time-token stability, but not metadata completeness and deterministic_id propagation semantics.
- Code change:
  - Ensure payload consistently carries deterministic metadata (`deterministic_id`, normalized operation/system/status) for graph consumers.
- Primary files:
  - `dadbot/core/execution_context.py`
- Concrete test:
  - Add `test_external_system_call_payload_contains_normalized_deterministic_metadata`.
  - Arrange: record call with mixed-case operation/system and deterministic_id.
  - Assert: payload fields are normalized and deterministic_id retained.
- Status: done

### 7) Replay completeness across process boundary

- Gap: no explicit test proving sealed-slot replay after restart-like rehydration from persisted artifacts.
- Code change:
  - Add small helper path or fixture utility that injects sealed values from persisted snapshot then executes REPLAY mode.
- Primary files:
  - `dadbot/core/determinism.py`
  - integration test area for replay path
- Concrete test:
  - Add `test_replay_completeness_rehydrates_sealed_slots_after_restart`.
  - Arrange: capture in RECORD, persist snapshot, create fresh boundary, inject, switch REPLAY.
  - Assert: replay returns sealed values without calling producer functions.
- Status: done

### 8) Facade mapping parity matrix

- Gap: map-driven routing is strong, but no strict parity test to ensure all map entries resolve/get/set correctly against backing providers.
- Code change:
  - No runtime behavior change required; add a map-parity test matrix.
- Primary files:
  - `dadbot/core/dadbot.py`
- Concrete test:
  - Add `test_dadbot_attribute_map_roundtrip_parity`.
  - Arrange: iterate `_CONFIG_ATTR_MAP`, `_RUNTIME_STATE_ATTR_MAP`, `_INTERNAL_RUNTIME_ATTR_MAP` with stubs.
  - Assert: get/set roundtrip is correct or explicitly exempted with rationale.
- Status: done

## Execution Order (Recommended)

1. Canonical event validation scope (Item 2)
2. Determinism boundary doc/behavior alignment (Item 1)
3. Memory total-order tie-break (Item 3)
4. Observability payload-strengthening tests (Items 5 and 6)
5. Multi-turn graph invalidation regression (Item 4)
6. Replay rehydration completeness (Item 7)
7. Facade parity matrix (Item 8)

## Exit Criteria

- All eight tests implemented and passing in unit/integration lanes as appropriate.
- No regression in DEV and INTEGRATION lanes.
- For each item, expected invariant is expressed in a failing-first test before code change.
- Update this checklist statuses from `open` to `done` with commit hashes.
