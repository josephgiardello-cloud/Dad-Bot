# DAD-BOT OFFICIAL CAPABILITY RECORD

Generated: 2026-05-03T14:11:59+00:00

## Baseline Summary
- Baseline source: tests/phase4_baselines.json
- Previous baseline available: yes

## Certification Matrix

| Section / Subsection | Claimed Feature | Test Coverage | Current Benchmark / Metric | Change from Baseline | Status | Notes |
|---|---|---|---|---|---|---|
| Startup / Cold Start | DadBot cold-start envelope | startup import timing probe | 3.452s total | -2.8% | Partial | Target <= 1.8s guard |
| Lanes / DEV | Lane stability and speed | pytest marker lane | 584 tests, 21.01s, pass | n/a | Proven | Baseline=n/as |
| Lanes / INTEGRATION | Lane stability and speed | pytest marker lane | 27 tests, 47.04s, pass | n/a | Proven | Baseline=n/as |
| Lanes / DURABILITY_P4 | Lane stability and speed | pytest marker lane | 52 tests, 131.04s, pass | +3.4% | Proven | Baseline=126.736s |
| Lanes / SOAK | Lane stability and speed | pytest marker lane | 57 tests, 71.03s, pass | n/a | Proven | Baseline=n/as |
| Lanes / UI | Lane stability and speed | pytest marker lane | 5 tests, 27.01s, pass | +5.1% | Proven | Baseline=25.705s |
| Lanes / FULL_CERT | Lane stability and speed | pytest marker lane | 1853 tests, 657.36s, pass | +2.1% | Proven | Baseline=643.867s |
| Persistence / DB Footprint | Soak-related DB growth visibility | sqlite size snapshot | 135168 bytes | +0.0% | Proven | Aggregated from root SQLite files |

## Top Slow Tests (Global)

1. [DEV] 3.54s call     tests/test_tool_sandbox_isolation.py::test_private_tool_sandbox_imports_are_repo_isolated
2. [DEV] 0.41s call     tests/test_cross_system_load_invariants.py::TestMixedSessionCompletion::test_100_turns_complete_without_exception
3. [INTEGRATION] 15.01s call     tests/test_service_api_integration.py::test_api_event_stream_websocket_replays_and_streams_tenant_events
4. [INTEGRATION] 8.73s call     tests/test_phase4a.py::TestPhase4ATrueProcessRestart::test_checkpoint_survives_real_process_boundary
5. [INTEGRATION] 1.63s call     tests/test_phase4a.py::TestPhase4ADeterminismVerification::test_lock_hash_stable_across_identical_inputs
6. [INTEGRATION] 1.59s call     tests/test_phase4a.py::TestPhase4AToolDeterminismAcrossRestarts::test_tool_trace_hash_changes_with_different_inputs
7. [INTEGRATION] 1.57s call     tests/test_phase4a.py::TestPhase4ARealCheckpointing::test_manifest_drift_warning_in_lenient_mode
8. [INTEGRATION] 1.57s call     tests/test_phase4a.py::TestPhase4AToolDeterminismAcrossRestarts::test_tool_trace_hash_stable_across_independent_sessions
9. [INTEGRATION] 1.52s call     tests/test_phase4a.py::TestPhase4ARealCheckpointing::test_orchestrator_restores_state_after_simulated_restart
10. [INTEGRATION] 1.35s call     tests/test_phase4a.py::TestPhase4ADeterminismVerification::test_persistence_metrics_are_observable_per_session
