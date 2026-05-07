# DAD-BOT OFFICIAL CAPABILITY RECORD

Generated: 2026-05-06T20:51:25+00:00

## Baseline Summary
- Baseline source: tests/phase4_baselines.json
- Previous baseline available: yes

## Certification Matrix

| Section / Subsection | Claimed Feature | Test Coverage | Current Benchmark / Metric | Change from Baseline | Status | Notes |
|---|---|---|---|---|---|---|
| Startup / Cold Start | DadBot cold-start envelope | startup import timing probe | 3.399s total | -1.5% | Partial | Target <= 1.8s guard |
| Lanes / DEV | Lane stability and speed | pytest marker lane | 602 tests, 17.01s, pass | -19.0% | Proven | Baseline=21.01s |
| Lanes / INTEGRATION | Lane stability and speed | pytest marker lane | 27 tests, 40.02s, pass | -14.9% | Proven | Baseline=47.044s |
| Lanes / DURABILITY_P4 | Lane stability and speed | pytest marker lane | 52 tests, 119.05s, pass | -9.2% | Proven | Baseline=131.043s |
| Lanes / SOAK | Lane stability and speed | pytest marker lane | 4 tests, 46.02s, pass | -35.2% | Proven | Baseline=71.026s |
| Lanes / UI | Lane stability and speed | pytest marker lane | 5 tests, 25.01s, pass | -7.4% | Proven | Baseline=27.012s |
| Lanes / FULL_CERT | Lane stability and speed | pytest marker lane | 1870 tests, 385.12s, pass | -41.4% | Proven | Baseline=657.358s |
| Persistence / DB Footprint | Soak-related DB growth visibility | sqlite size snapshot | 135168 bytes | +0.0% | Proven | Aggregated from root SQLite files |

## Top Slow Tests (Global)

1. [DEV] 2.76s call     tests/test_tool_sandbox_isolation.py::test_private_tool_sandbox_imports_are_repo_isolated
2. [DEV] 0.14s call     tests/test_cross_system_load_invariants.py::TestMixedSessionCompletion::test_100_turns_complete_without_exception
3. [INTEGRATION] 15.01s call     tests/test_service_api_integration.py::test_api_event_stream_websocket_replays_and_streams_tenant_events
4. [INTEGRATION] 7.57s call     tests/test_phase4a.py::TestPhase4ATrueProcessRestart::test_checkpoint_survives_real_process_boundary
5. [INTEGRATION] 1.35s call     tests/test_phase4a.py::TestPhase4ADeterminismVerification::test_persistence_metrics_are_observable_per_session
6. [INTEGRATION] 1.05s call     tests/test_phase4a.py::TestPhase4ARealCheckpointing::test_manifest_drift_warning_in_lenient_mode
7. [INTEGRATION] 1.04s call     tests/test_phase4a.py::TestPhase4ARealCheckpointing::test_orchestrator_restores_state_after_simulated_restart
8. [INTEGRATION] 1.03s call     tests/test_phase4a.py::TestPhase4ADeterminismVerification::test_checkpoint_chain_integrity_after_two_real_turns
9. [INTEGRATION] 1.01s call     tests/test_phase4a.py::TestPhase4AToolDeterminismAcrossRestarts::test_tool_trace_hash_changes_with_different_inputs
10. [INTEGRATION] 1.00s call     tests/test_phase4a.py::TestPhase4AToolDeterminismAcrossRestarts::test_tool_trace_hash_stable_across_independent_sessions
