# DAD-BOT OFFICIAL CAPABILITY RECORD

Generated: 2026-04-27T20:37:43+00:00

## Baseline Summary
- Baseline source: tests/phase4_baselines.json
- Previous baseline available: no

## Certification Matrix

| Section / Subsection | Claimed Feature | Test Coverage | Current Benchmark / Metric | Change from Baseline | Status | Notes |
|---|---|---|---|---|---|---|
| Startup / Cold Start | DadBot cold-start envelope | startup import timing probe | 2.788s total | n/a | Partial | Target <= 1.8s guard |
| Lanes / DEV | Lane stability and speed | pytest marker lane | 411 tests, 7.21s, pass | n/a | Proven | Baseline=n/as |
| Lanes / INTEGRATION | Lane stability and speed | pytest marker lane | 27 tests, 35.51s, pass | n/a | Proven | Baseline=n/as |
| Lanes / DURABILITY_P4 | Lane stability and speed | pytest marker lane | 22 tests, 22.17s, pass | n/a | Proven | Baseline=n/as |
| Lanes / SOAK | Lane stability and speed | pytest marker lane | 4 tests, 108.22s, pass | n/a | Proven | Baseline=n/as |
| Lanes / UI | Lane stability and speed | pytest marker lane | 5 tests, 21.21s, pass | n/a | Proven | Baseline=n/as |
| Lanes / FULL_CERT | Lane stability and speed | pytest marker lane | 1491 tests, 251.94s, pass | n/a | Proven | Baseline=n/as |
| Persistence / DB Footprint | Soak-related DB growth visibility | sqlite size snapshot | 118784 bytes | n/a | Proven | Aggregated from root SQLite files |

## Top Slow Tests (Global)

