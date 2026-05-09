# DAD-BOT OFFICIAL CAPABILITY RECORD

Generated: 2026-05-09T12:01:58+00:00

## Baseline Summary
- Baseline source: tests/phase4_baselines.json
- Previous baseline available: no

## Certification Matrix

| Section / Subsection | Claimed Feature | Test Coverage | Current Benchmark / Metric | Change from Baseline | Status | Notes |
|---|---|---|---|---|---|---|
| Startup / Cold Start | DadBot cold-start envelope | startup import timing probe | 4.466s total | n/a | Partial | Target <= 1.8s guard |
| Lanes / DEV | Lane stability and speed | pytest marker lane | 0 tests, 90.06s, fail | n/a | Gap | Baseline=n/as |
| Lanes / INTEGRATION | Lane stability and speed | pytest marker lane | 0 tests, 90.03s, fail | n/a | Gap | Baseline=n/as |
| Lanes / DURABILITY_P4 | Lane stability and speed | pytest marker lane | 112 tests, 131.05s, pass | n/a | Proven | Baseline=n/as |
| Lanes / SOAK | Lane stability and speed | pytest marker lane | 13 tests, 52.02s, pass | n/a | Proven | Baseline=n/as |
| Lanes / UI | Lane stability and speed | pytest marker lane | 5 tests, 35.01s, pass | n/a | Proven | Baseline=n/as |
| Lanes / FULL_CERT | Lane stability and speed | pytest marker lane | 2361 tests, 553.20s, fail | n/a | Gap | Baseline=n/as |
| Persistence / DB Footprint | Soak-related DB growth visibility | sqlite size snapshot | 147456 bytes | n/a | Proven | Aggregated from root SQLite files |

## Top Slow Tests (Global)

1. [DURABILITY_P4] 62.43s call     tests/system_validation/test_drift_monitor.py::TestBoundaryGateRuntimeInvariant::test_boundary_gate_is_reproducible
2. [DURABILITY_P4] 34.95s call     tests/system_validation/test_drift_monitor.py::TestBoundaryGateRuntimeInvariant::test_boundary_gate_exits_zero
3. [DURABILITY_P4] 8.79s call     tests/test_phase4a.py::TestPhase4ATrueProcessRestart::test_checkpoint_survives_real_process_boundary
4. [DURABILITY_P4] 1.71s call     tests/system_validation/test_adversarial_scale.py::TestLongChainReplay::test_invariants_hold_across_long_chain
5. [DURABILITY_P4] 1.30s call     tests/test_phase4a.py::TestPhase4ARealCheckpointing::test_orchestrator_restores_state_after_simulated_restart
6. [DURABILITY_P4] 1.25s call     tests/test_phase4a.py::TestPhase4ARealCheckpointing::test_manifest_drift_warning_in_lenient_mode
7. [DURABILITY_P4] 1.25s call     tests/test_phase4a.py::TestPhase4ARealCheckpointing::test_orchestrator_saves_checkpoint_after_real_turn
8. [DURABILITY_P4] 1.21s call     tests/test_phase4a.py::TestPhase4AToolDeterminismAcrossRestarts::test_tool_trace_hash_changes_with_different_inputs
9. [DURABILITY_P4] 1.16s call     tests/test_phase4a.py::TestPhase4AToolDeterminismAcrossRestarts::test_tool_trace_hash_stable_across_independent_sessions
10. [DURABILITY_P4] 1.16s call     tests/test_phase4a.py::TestPhase4ADeterminismVerification::test_lock_hash_stable_across_identical_inputs
