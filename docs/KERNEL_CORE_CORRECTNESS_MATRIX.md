# Kernel/Core Correctness Matrix

## Critical correctness gaps (still real work)

### 1) Single source of truth is not fully absolute
- Status: mostly enforced, not absolute by invariant gate.
- Current risk: some entry paths can still reason from non-projection artifacts (checkpoint shortcuts) rather than projection rebuilt from events.
- Required closure:
  - Enforce a runtime invariant: lifecycle state reads for decisions must be projection-derived from ledger events.
  - Add a guard that rejects direct checkpoint/lifecycle shortcuts in decision paths.

### 2) Global execution-boundary idempotency not fully complete
- Status: materially hardened.
- Implemented now:
  - Durable request-level dedupe via `JOB_COMPLETED` lookup keyed by `(session_id, request_id)` in `ExecutionControlPlane`.
-  - Fail-closed replay guard: if a prior matching request has `JOB_STARTED` with no terminal event, replay is rejected to prevent duplicate side effects.
- Remaining risk:
  - Availability tradeoff under ambiguous crash windows (request is blocked until reconciled).
- Required closure:
  - Add operation-level durable effect registry with begin/commit phases and replay-safe skip behavior.

### 3) Lifecycle state machine not fully constrained at emission
- Status: closed for control-plane lifecycle emission.
- Implemented now:
  - Hard emission-time transition validation blocks illegal lifecycle events before ledger append.

### 4) Lease + scheduling race closure not fully formalized
- Status: hardened for scheduler ordering.
- Implemented now:
  - Deterministic pending job selection by stable ordered job ids, independent of pending-list insertion race.
- Remaining risk:
  - Cross-worker tie-break proof under distributed clocks still needs explicit formal tests.

### 5) Failure taxonomy still partially implicit
- Status: failure handling exists but classification is not fully exhaustive.
- Required closure:
  - Explicit buckets: retryable, terminal, poisoned execution, unknown state recovery, partial execution.
  - Map each bucket to lifecycle transitions and scheduler policy.

### 6) Restart determinism not yet fully provable for orchestration decisions
- Status: state replay is strong.
- Remaining risk: lease selection and redelivery ordering are not formally proven deterministic end-to-end.
- Required closure:
  - Deterministic scheduler/redelivery decision proof tests under replayed timelines.

### 7) Concurrency + distributed correctness remains partially implicit
- Status: leases, worker claims, and request dedupe are in place.
- Remaining gap:
  - formal concurrency model
  - strict race-condition modeling
  - deterministic scheduling semantics under reorderings
  - mathematically enforced lease correctness
- Result:
  - mid-to-upper tier correctness, not elite distributed-systems rigor yet.

### 8) Failure handling + edge-case modeling remains policy-light
- Status: recovery paths and replay/restart logic exist.
- Remaining gap:
  - poison messages
  - partial commits
  - stuck leases
  - ambiguous state resolution
  - explicit recovery policy per failure class
- Result:
  - above average, but not fully production-hardened.

### 9) Idempotency guarantees are stronger but still incomplete
- Status: request-level dedupe is strong.
- Remaining gap:
  - effect-layer atomicity
  - scheduler-layer idempotency
  - execution-layer idempotency
  - external side-effect atomicity under crash
- Result:
  - mid-to-strong, not production-safe yet.

### 10) Schema + contract discipline is strong but still evolving
- Status: schema system and validation layers are solid.
- Remaining gap:
  - fully versioned backward-compatibility story
  - strict API contracts across services
- Result:
  - above average, with room for hardening.

### 11) Operational readiness is the biggest gap
- Status: core runtime works.
- Remaining gap:
  - load/backpressure strategy
  - fault injection at scale
  - observability-first design
  - throttling/limits
  - scaling-behavior validation
- Result:
  - below production standard for real infra deployment.

### 12) Complexity is justified but not yet minimized
- Status: system is architecturally strong.
- Remaining gap:
  - too many interacting subsystems in the hot path
  - failure behavior would benefit from fewer moving parts
- Result:
  - strong design, but not yet operationally minimal.

## Non-hardening improvements (architecture quality)

### 1) Projection separation refinement
- Split read-model projection and decision-model projection for clearer boundaries.

### 2) Memory model unification polish
- Clarify memory type semantics and persistence rules in a single contract.

### 3) Scheduler fairness policy
- Add starvation prevention and explicit prioritization policy.

### 4) Observability structure
- Add first-class queryable execution trace model and runtime state inspection schema.

### 5) API/contract ergonomics
- Tighten public/internal boundary definitions and contract surface docs.

## Basically done already (avoid over-optimization)
- Event sourcing foundation.
- Persistence schema system.
- Deterministic execution model.
- Replay harness.
- Checkpointing system.
- Worker lease concept.
- Lifecycle event modeling.
- Reducer correctness core.
- Projection system.

## Latest delta (May 11, 2026)
- Completed: removed remaining external lease authority input path from control-plane options.
- Completed: durable request-level idempotency across control-plane restarts using shared ledger terminal records.
- Added test: restart dedupe regression for `(session_id, request_id)`.
- Completed: ambiguous crash-window replay guard blocks duplicate side effects when prior `JOB_STARTED` exists without terminal event.
- Completed: lifecycle emission-time transition gate rejects invalid transitions before append.
- Completed: deterministic scheduler pending ordering under concurrency.
- Added taxonomy: concurrency/distributed correctness, failure handling, idempotency, schema discipline, operational readiness, and complexity tradeoff.
- Added tests:
  - `test_control_plane_idempotency_blocks_ambiguous_effect_replay`
  - `test_scheduler_rejects_invalid_emission_transition_before_ledger_write`
  - `test_scheduler_pending_order_is_deterministic_under_concurrency`
