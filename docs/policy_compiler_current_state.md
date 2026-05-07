# Policy Compiler Current State

## What It Transforms Today

Input:
- Runtime safety service surface (whether `enforce_policies(turn_context, candidate)` and/or `validate(candidate)` exist)
- Current turn candidate payload
- Current turn context

Transform:
- Compiles the available service methods into an ordered `PolicyPlan` of `PolicyStep` entries
- Evaluates the plan deterministically in first-match order:
  1. `enforce_policies` (binary: context + candidate)
  2. `validate` (unary: candidate)
  3. fallback passthrough

Output:
- `PolicyDecision` with:
  - `action` (`handled` or `passthrough`)
  - `output` (safe result payload)
  - `step_name` (which step produced output)
  - `details` (policy/kind metadata)

## Honest Assessment

This is now a real execution component in the safety path (not dead code), but it is still closer to a structured validation/dispatch layer than a full rule engine.

What it has now:
- Explicit input -> plan -> decision pipeline
- Deterministic ordering and typed decision object
- Shared usage across both safety-node implementations

What it does not yet have:
- Rich rule language (conditions, priorities, predicates, combinators)
- Multi-rule accumulation/aggregation semantics
- First-class policy trace explaining which rules were considered/fired with full evidence
- Cross-domain policy composition beyond safety dispatch

## Next Step to Mature It

Add first-class decision tracing to `PolicyDecision.details`:
- normalized input summary
- rules considered in order
- selected rule and reason
- emitted policy events for downstream observability/audit
