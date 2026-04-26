# Changelog

## 2026-04-26 - Phase 4 Closure

- Hardened delegation safety: depth guard propagation, capped sub-task fan-out, and explicit arbitration log events.
- Added per-turn blackboard determinism guarantees: seed/final fingerprints included in determinism metadata and session state snapshots.
- Added delegation visibility in normal replies with concise summary text (for non-debug UX).
- Improved sub-task error propagation with user-friendly failure strings and failure counts in arbitration metadata.
- Expanded property verification suite:
  - parallel vs sequential timing proof
  - depth-guard flag assertion
  - final-reply delegation summary assertion
  - sub-task failure propagation assertion
  - blackboard fingerprint determinism assertion
- Added configuration knobs for property-suite replay/memory-layer turn counts.
- Validation completed:
  - Fast certification gate (50/50/30/10): PASS, 100/100
  - Property verification suite: 19 passed, 0 failed
