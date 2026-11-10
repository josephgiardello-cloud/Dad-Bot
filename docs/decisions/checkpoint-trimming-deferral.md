# Checkpoint trimming deferral

> **DEFERRED — not Phase 4 current scope**

**Owner:** Durability/ledger maintainers (role; accountable team assignment
required before implementation).

## Decision

Do not trim the execution log in the current Phase 4 scope. Safe trimming
cannot be implemented with the current in-memory checkpoint and snapshot
state and a WAL verifier that expects the original event chain.

The current compactor refuses trimming with `durable_checkpoint_required`.
There is no durable registry of every consumer's acknowledged cursor, so a
trim cannot prove that a slow or offline consumer no longer needs an event.
Snapshots are not yet a durable, replay-complete replacement for the trimmed
prefix. Removing events also breaks verification of the chain from its
beginning unless the retained log has a verifiable segment boundary or anchor.
Finally, no integrated rebuild path exists when a consumer falls behind the
retained log. Without these guarantees, compaction can make replay incomplete
or silently corrupt recovery.

## Safe behavior and deferral cost

Trimming remains disabled; the compactor rejects unsafe requests. Deferral
preserves replay and integrity behavior, but the WAL continues to grow, uses
increasing disk space, and can make reads, verification, startup recovery, and
backups more expensive.

## Required substrate before B2

1. A durable, restart-safe cursor registry covering every log consumer, with
   explicit acknowledgement semantics and a trim watermark no later than the
   minimum acknowledged cursor.
2. A TTL ceiling and documented policy for consumers that stop acknowledging;
   expiry must not silently imply successful replay.
3. Durable replay-complete snapshots with validated coverage through a
   specific log cursor.
4. Chain-preserving segments or authenticated anchors that validate the
   retained suffix and snapshot boundary after prefix removal.
5. An integrated fall-off-log recovery path that detects a stale consumer and
   rebuilds it from a validated snapshot and retained events.
6. Tests proving that a lagging consumer rebuilds rather than corrupting
   replay, and that trim never removes an event needed by any non-expired
   consumer. Include restart/crash cases around cursor acknowledgement,
   snapshot publication, and trim.

No cursor registry, snapshot redesign, chain format, rebuild behavior, or
trimming code is introduced by this deferral.
