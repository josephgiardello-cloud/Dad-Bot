# Contract Failure Map

Canonical categories:
- schema mismatch
- missing field
- outdated assertion
- ordering change
- deterministic replay drift

## Mapped failures from full-cert snapshot

- tests/unit/test_tool_execution_contract.py::TestContractValidationFunctions::test_validate_output_success
  - category: outdated assertion
  - note: expected message prefix drift ("Contract violation" capitalization/string format)

- tests/unit/test_tool_execution_contract.py::TestContractValidationFunctions::test_validate_input_success
  - category: outdated assertion
  - note: expected message prefix drift ("Contract violation" capitalization/string format)

- tests/unit/test_tool_execution_contract.py::TestDefaultGenericContract::test_generic_contract_accepts_any_output
  - category: schema mismatch
  - note: generic contract should allow additional fields but was enforcing closed output fields

- tests/unit/test_tool_execution_contract.py::TestDefaultGenericContract::test_generic_contract_accepts_any_input
  - category: schema mismatch
  - note: generic contract should allow additional fields but was enforcing closed input fields

- tests/test_phase4a.py::TestPhase4ALargeStateCheckpoint::*
  - category: missing field
  - note: persistence contract introduced required checkpoint/manifest fields; compatibility layer now normalizes missing fields

- tests/unit/test_priority_graph_trace_enrichment.py::*
  - category: ordering change
  - note: kernel post_execute strictness changed behavior/order by raising instead of shadow logging

- tests/unit/test_priority1_canonical_execution_path.py::TestTraceInvariants::test_validate_trace_invariant_called_after_execution
  - category: ordering change
  - note: submit path moved invariant call to kernel path; source-level assertion expected direct reference

- tests/unit/test_priority1_canonical_execution_path.py::TestTraceInvariants::test_validate_trace_invariant_no_commits
  - category: outdated assertion
  - note: expected warning path, but implementation raised hard invariant

- tests/unit/test_priority1_canonical_execution_path.py::TestTraceInvariants::test_validate_trace_invariant_multiple_commits
  - category: outdated assertion
  - note: expected warning path, but implementation raised hard invariant

- tests/unit/test_failure_policy_engine.py::TestRetryBackoff::test_retry_delay_with_jitter
  - category: deterministic replay drift
  - note: jitter was effectively constant; no variability across retries

- tests/test_turn_kernel.py::test_control_plane_serializes_turns_within_same_session
  - category: missing field
  - note: runtime payload shape drift likely removed expected counter field

- tests/test_e2e_orchestrator_turn.py::test_e2e_orchestrator_full_runtime_lane_certifying_expected_tool_execution
  - category: ordering change
  - note: tool decision policy path changed from expected-tool execution to no-tool-needed

## Layer-first remediation order used

1. kernel + execution_result schema
   - restored strict/shadow behavior compatibility for kernel validation
   - preserved execution_result contract validation boundary

2. control_plane output contracts
   - added canonical validation for decision explanation output
   - added canonical validation for runtime trace events

3. persistence + replay
   - added backward-compatible checkpoint/manifest normalization before strict contract checks

4. tests
   - no direct test edits applied in this pass
