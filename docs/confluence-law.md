# Confluence Law in Dad-Bot

## Purpose

Confluence Law guarantees that turns in the same semantic equivalence class converge to the same execution-level contract hash.

## Invariants

- Equivalence class identity is declared by an explicit `confluence_key`.
- In `enforce` mode, missing `confluence_key` is a hard failure.
- First observation for a key binds the expected confluence hash.
- Later observations for the same key must match the bound hash.
- In `enforce` mode, mismatch is a hard failure.
- In `audit` mode, mismatch is recorded and logged.

## Failure Semantics

- Missing key in `enforce` mode raises a runtime invariant failure before execution is accepted.
- Divergent hash in `enforce` mode raises a runtime error and blocks turn completion.
- Legacy fallback key generation is disabled by default and only available with:
  - `DADBOT_ALLOW_LEGACY_CONFLUENCE_KEY=1`

## Operational Metrics

ExecutionControlPlane now publishes confluence counters via boot reconcile output:

- `attempted`
- `bound_first_observation`
- `matched`
- `mismatch`
- `enforced_blocked`

These are available under `execution_confluence_metrics` in `boot_reconcile()` status payloads.

## Configuration

- `DADBOT_GLOBAL_CONFLUENCE_MODE`: `off` | `audit` | `enforce`
- Per-turn override: metadata `confluence_mode`
- Per-turn class key: metadata `confluence_key`

Recommended production baseline:

- Set global mode to `enforce`.
- Require callers to provide explicit keys at orchestrator boundary.
- Keep legacy fallback disabled.
