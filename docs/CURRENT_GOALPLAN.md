# Current Goalplan (Locked)

Last updated: 2026-05-12
Owner override: user-controlled only

## Goalpost Definition

The Kernel is considered finished.

Kernel goalpost:
A code library that can ingest a user message, plan a tool call, execute it, save state atomically, and recover perfectly from a crash.

Status: REACHED.

## What This Means Right Now

Work is now split into:

1. Infrastructure wrappers (not kernel redesign)
2. Agent Driver Loop (Observation -> Reflection -> Action)

Any additional kernel changes are out of scope unless they fix a proven regression in:

- atomic commit safety
- replay determinism
- restart recovery

## Current Execution Priority

Priority 1: Agent Driver Loop

- Implement loop runner for Observation -> Reflection -> Action -> Commit -> Repeat.
- Add stop guards: max turns, failure budget, no-op detector, manual interrupt.
- Add loop-level telemetry: turn count, action type, tool latency, commit status.

Priority 2: Infrastructure wrappers

- Concurrency and Locking: add distributed lock provider path (for multi-worker/server runtime).
- Secrets and Security: add secret manager provider path for tool credentials.
- Storage Scaling: keep current persistence contract, add scalable backend adapter path.

## Freeze Rules (Do Not Drift)

Do not spend time redesigning kernel internals while this goalplan is active.

Allowed kernel edits:

- bug fixes that fail existing correctness tests
- crash/restart/replay safety regressions

Not allowed during this goalplan:

- structural kernel rewrites
- abstraction expansion that does not unlock Agent Driver Loop or wrappers
- speculative refactors for style-only reasons

## Completion Target For This Goalplan

The active goalplan is complete when the system can autonomously run a stable loop:

Observation -> Reflection -> Action -> Commit -> Repeat

with deterministic recovery behavior and policy-bounded failure handling.

## Change Control

This file is authoritative for current scope.
Only the user can move this goalpost.
