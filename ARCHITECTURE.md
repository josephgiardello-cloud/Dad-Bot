# Dad-Bot Architecture

## Current Execution Path
Dad-Bot uses one canonical turn path. `DadBotTurnMixin` creates a `TurnHandler`,
which normalizes the request and delegates to the control plane. There is no
runtime path toggle or supported legacy execution branch.

## Execution-Order Authority
`dadbot/core/control_plane.py` owns submit-turn ordering and enforces its phase
contract. No secondary scheduler or orchestrator defines phase semantics.

`dadbot/core/turn_handler.py` is the stateless ingress adapter: it normalizes
turn context and delegates to the control plane. It owns neither scheduling nor
persistence.

## Checkpoint and Replay Responsibilities
Checkpoint-like components have separate roles and are not interchangeable:

- `ExecutionLedger` records canonical execution events and replay history.
- `GRAPH_CHECKPOINT` events in `ExecutionLedger` are the authoritative turn
  checkpoints. `SQLiteCheckpointer`, when explicitly configured, is an
  auxiliary snapshot store that the orchestrator may use to hydrate turn
  context.
- The default execution ledger is in-memory. Set
  `DADBOT_DURABLE_EXECUTION_LEDGER=1` to use and reload a local WAL at
  `SESSION_LOG_DIR/execution_ledger.jsonl` (or `DADBOT_SESSION_LOG_DIR` when
  no bot session directory is available).
- Control-plane session state is persisted through `SESSION_STATE_UPDATED`
  events and rebuilt into the session registry during ledger bootstrap.
  Canonical system-state algebra regenerates its read-only legacy projections
  after replay; the registry is a projection, not the recovery authority.
- The WAL is append-only and has no automatic retention or compaction policy.
  Its disk usage grows with committed events, including session-state updates.
  Keep durable mode opt-in and do not prune WAL records until snapshot-based
  recovery and retention are validated end to end.
- Kernel invariant failures block graph execution by default. For local
  diagnostics only, `DADBOT_ALLOW_KERNEL_VALIDATION_SHADOW=1` restores
  warning-only behavior; startup rejects this override in production.
- `TurnResumeStore` stores per-turn resume markers as ledger events.
- Replay validation with an explicit expected lock hash fails overall when the
  observed lock hash is missing or differs; matching individual fields is not
  sufficient to report a valid replay.
- `DurableCheckpoint` is a process-local ledger-head verification helper. Its
  checkpoint history is not persisted and must not be treated as crash recovery.
- The unused standalone per-node `ExecutionCheckpointKernel` replay framework
  has been removed; tool I/O recording remains part of the active turn path.

Ledger replay and persisted snapshot restoration remain distinct mechanisms.
Consolidating them requires a migration that proves snapshot state can be
rebuilt or safely resumed from the ledger before removing snapshot storage.

## Routing Compatibility
The historical `DADBOT_USE_THIN_TURN_HANDLER` setting does not control routing.
Do not use it to select an execution path; comparisons belong in equivalence
validation tooling.

## Migration Table
| Surface | Status | Target | Notes |
|---|---|---|---|
| `DadBotConvenienceMixin` | Migrated | Composed helper (`ConvenienceHelpers`) | Public methods preserved in `DadBot`; no API break expected |
| `DadBotCompatMixin` | Pending | Evaluate split into small compatibility services | Larger and more coupled than convenience; do in dedicated slice |
| Turn routing | Converged | `DadBotTurnMixin` → `TurnHandler` → control plane | Missing control-plane submit entrypoint is a configuration error |
| `control_plane.py` sequencing | Stabilized | Keep as authority | Submit-turn phase order explicit and runtime-validated |
