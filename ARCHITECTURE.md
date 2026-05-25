# Dad-Bot Architecture

## Current State and Dual-Path Strategy
Dad-Bot is in an intentional dual-path migration:

- Authoritative execution path: control-plane-backed turn execution.
- Thin path (feature-flagged): thin turn handler adapter that forwards into the same authoritative control plane.

Goal of dual-path period:

- Keep behavior stable for users.
- Collect real runtime feedback on the thin path.
- Incrementally remove facade mixins and over-coupled surfaces.

## Source of Truth Decision
Single source of truth for execution ordering remains in:

- `dadbot/core/control_plane.py`

This module owns the canonical submit-turn phase contract and runtime enforcement.
No secondary scheduler/orchestrator may define independent phase semantics.

## Thin Spine Role
Thin spine lives in:

- `dadbot/core/turn_handler.py`

Role:

- Stateless deterministic adapter.
- Normalizes turn context and delegates to authoritative submit path.
- No state authority, no independent scheduling, no persistence ownership.

## Feature Flags Reference
- `DADBOT_USE_THIN_TURN_HANDLER`
  - **Deprecated (legacy compatibility only).**
  - Thin-spine is now canonical and effectively always enabled in the shipped runtime.
  - Older documentation and UI affordances may still refer to a “toggle”, but routing should be treated as converged.
  - If a legacy-path comparison is needed, use the explicit equivalence validation tooling rather than a runtime toggle.

## Migration Table
| Surface | Status | Target | Notes |
|---|---|---|---|
| `DadBotConvenienceMixin` | Migrated | Composed helper (`ConvenienceHelpers`) | Public methods preserved in `DadBot`; no API break expected |
| `DadBotCompatMixin` | Pending | Evaluate split into small compatibility services | Larger and more coupled than convenience; do in dedicated slice |
| `dad_streamlit.py` routing | In progress | Thin-path toggle + runtime metrics | Toggle and per-path latency/count metrics now visible in sidebar |
| `control_plane.py` sequencing | Stabilized | Keep as authority | Submit-turn phase order explicit and runtime-validated |

## Mental Model
1. UI/runtime submits user turn.
2. Turn path chosen by feature flag (legacy vs thin adapter).
3. Both paths converge into the same control-plane authority.
4. Control plane enforces deterministic phase ordering.
5. Plugins/services provide optional capabilities around the authoritative spine.

## Text Diagram

```text
[Streamlit / CLI / API]
        |
        v
[Path Toggle]
  |                \
  | thin enabled    \ legacy
  v                  v
[TurnHandler]     [Legacy entry adapter]
        \          /
         v        v
      [Execution Control Plane]  <-- single ordering authority
                 |
                 v
      [Schedulers / Ledgers / Persistence / Services]
```

## Decision Log
- Keep control plane as authority during migration to prevent distributed orchestration logic.
- Use thin adapter only as stateless ingress normalization.
- Remove one mixin cluster per slice with tests after each step.
