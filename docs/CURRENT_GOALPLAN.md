# Current Goalplan

Last updated: 2026-11-11

## Goal

The kernel goalpost is reached: accept a user message, plan and execute a tool
call, save state atomically, and recover after a crash.

## Active Priorities

1. **Agent Driver Loop:** Use hybrid memory for observation, structured
   reflection with bounded conflict and timestamp signals, and a pre-dispatch
   plan gate. Carry bounded goal-board updates in canonical turn metadata.
   Heartbeats and queued proactive events reuse one bounded loop turn; keep
   memory consolidation at the canonical post-commit boundary.
2. **Infrastructure wrappers:** Add distributed locking, secret management,
   and scalable storage adapters without changing persistence contracts.

## Kernel Scope

The approved [Agent and Memory Hardening Plan](AGENT_MEMORY_PLAN.md) defines
the next certification milestones. It distinguishes the implemented experimental
loop from production-hardening work; baseline certification is in progress.

Keep kernel internals stable. Kernel changes are limited to proven correctness
bugs and regressions in atomic commits, replay determinism, or restart recovery.
Structural rewrites and style-only refactors are out of scope.

## Completion

The loop is complete when it repeats Observation -> Reflection -> Action ->
Commit with rejected plans kept from dispatch and accepted goal-board snapshots
recorded in canonical turn metadata. Proactive events must retain tenant
context, process a bounded number per heartbeat, and remain queued after a
rejected or failed plan.

This file defines the current scope. Only the user can change the goalpost.
