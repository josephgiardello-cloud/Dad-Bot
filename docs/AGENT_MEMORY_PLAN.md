# Agent and Memory Hardening Plan

Date: 2026-11-11  
Status: Milestone 0 verified for the default local suite; milestone 1 is in progress.

## Goal

Make the existing autonomous loop reliable across turns, tenants, and restarts.
Reuse the canonical execution ledger, memory query, consolidation worker, and
authority contracts. Do not build a second agent or memory engine.

The earlier A-D work delivered an experimental foundation, not production
certification. This plan combines the outstanding test failures with the deeper
memory, temporal, and autonomy work.

## Verified baseline

| Area | Current evidence | Remaining work |
| --- | --- | --- |
| Driver Loop | Bounded turns, structured reflection, plan validation, goal-board metadata, and canonical dispatch exist in [AgentDriverLoop](../dadbot/runtime/agent_driver_loop.py). | Verify actual outcome and commit status; recover committed goal state across runs; distinguish proposals from committed progress. |
| Memory-first observation | [AssistantRuntime](../dadbot/assistant_runtime.py) enables memory observation by default for `run_agent_loop`. Its provider calls `relevant_memories_for_input`. | Certify every supported loop ingress and measure retrieval quality. Use the richer graph/semantic/consolidated/archive bundle where appropriate rather than assuming the current provider covers all sources. |
| Online consolidation | [PersistenceService](../dadbot/services/persistence.py) publishes post-commit readiness; [PostCommitWorker](../dadbot/services/post_commit_worker.py) invokes consolidation. | Durable scheduling, authority propagation, idempotency, meaningful success reporting, and recovery after process loss. |
| Durable memory authority | [MemoryStorageBackend](../dadbot/memory/storage.py) authorizes memory access, fences local index writes, and records tenant-specific projection status in the ledger. | Certify completeness across bridge, worker, snapshot, and supported adapter paths; metadata is not absent, but coverage must be proven. |
| Temporal/conflict handling | Reflection receives bounded timestamp/conflict signals. [Bi-temporal edge tests](../tests/test_bitemporal_graph_edges.py) cover existing edge validity fields. | Consistent as-of retrieval, provenance-aware corrections, late-arrival handling, and conflict decisions surviving restart. |
| Proactivity | A heartbeat processes at most one queued event through the loop; its reflection plan has an empty tool allowlist. | Durable deduplication and worker ownership; verify downstream tool enforcement, not just reflection policy. |
| Goalplan | [CURRENT_GOALPLAN](CURRENT_GOALPLAN.md) was updated on November 11, 2026. | Replace broad capability claims with acceptance evidence after each hardening milestone. |

These are repository observations, not a measured comparison with Zep or another
product.

### Validation status

The unfiltered default suite at `d02ac36d` passed on November 12, 2026: **4,659 passed,
123 skipped, 10 expected failures, zero failures**. Test deadlines and production
authority checks are unchanged.

The original scale gates in [Phase 4A](../tests/test_phase4a.py) also passed:

| Gate | Time | Limit |
| --- | --- | --- |
| 110 canonical turns | 117.64 s | 120 s |
| Eight sessions, five turns each | 22.00 s | 60 s |
| 20-turn exact pruning count | 10.00 s | 30 s |

Baseline repairs cover tenant-qualified fixtures, materialized checkpoint views,
same-request tool replay, deterministic pipeline clocks, selective-copy
instrumentation, Streamlit/Windows process isolation, and import-cycle removal.
Canonical hashing and chain verification now share the existing
[canonical event module](../dadbot/core/canonical_event.py). UI and heartbeat
dispatch share the same [runtime lock](../dadbot/core/runtime_locks.py).

PostgreSQL, Redis, live-model integrations, optional dependencies, slow/stress
runs, and explicitly quarantined cases remain unverified where skipped.
Skips are not passes. This baseline does not certify every memory projection,
consolidation path, or production deployment.

### Local history format

Session updates and graph-checkpoint payloads can replace cumulative list or
dictionary prefixes with version-1 append references. Version 2 also supports a
retained list suffix plus new entries for rolling windows. Each reference
identifies its field path, full prior length, and SHA-256 base hash; suffix
offsets are range-checked. Replacements, deletions, version-1 references, and
legacy full snapshots retain their existing meaning. Earlier windows remain
in the ledger. Decoders reject missing, corrupt, overlapping, or cross-root
references and unsupported versions.

Only verified trusted-local native/file/CRC backends produce references.
Authenticated and custom producers retain full payloads. Session-state checkpoint
references remain checksum-protected separately from noncanonical checkpoint
contents. Checkpoint readers materialize complete payloads; those views are not
sealed events and must not be used as replay input.

## Non-negotiable contracts

1. The ledger remains authoritative. Indexes, prompt snippets, and loop snapshot
   files cannot grant authority or override committed state.
2. Authenticated work must retain tenant/session/request identity and revalidate
   live authority at execution and publication. Trusted-local execution remains
   explicitly supported.
3. A plan is not authorization. Proactive events and retrieved text cannot grant
   tool permissions.
4. Nondeterministic model output becomes durable only through validated,
   provenance-linked records. Replay consumes those records without calling the
   model or redispatching tools.
5. Failed, rejected, and uncertain outcomes must not become successful semantic
   facts or goal progress. Uncertain external effects require review, not resend.
6. Projection failures do not roll back a committed turn. Report degraded state
   and schedule bounded repair instead of silently claiming success.
7. Preserve the kernel, SaveNode, replay, and restart contracts. Kernel changes
   require a reproduced correctness failure, not an architectural preference.

## Sequence and dependencies

| Milestone | Depends on | Deliverable |
| --- | --- | --- |
| 0. Establish a clean baseline | None | Named, reproducible failures and trustworthy test gates |
| 1. Close authority and recovery gaps | 0 | Tenant-scoped projections, bridge reads, and recovery paths |
| 2. Certify memory-first observation | 1 | One default retrieval path with measured quality and explicit degradation |
| 3. Make loop progress truthful and durable | 1, 2 | Receipt-backed outcomes and recoverable goal state |
| 4. Harden online consolidation | 1, 3 | Recoverable, idempotent post-commit learning |
| 5. Deepen temporal/conflict semantics | 2, 4 | As-of recall and audited corrections with source lineage |
| 6. Harden proactive execution | 3, 4 | Restart-safe trigger ownership and outcome handling |
| 7. Certify infrastructure wrappers | 1, 4, 6 | Deployment-required locking, secrets, and storage integration |

Documentation cleanup and narrow evidence collection can run alongside these
milestones. Do not broaden production exposure before its prerequisites pass.

## 0. Establish a clean baseline

### Work

- Preserve and validate the pending kernel fixture fix. Publish it separately
  from feature work after approval.
- Run the broad suite in bounded groups, preserving test names and JUnit results.
  Start with the first failing group rather than rerunning the whole suite blindly.
- Reproduce each failure individually, then in its preceding test group to detect
  order-dependent state leakage.
- Separate product bugs, invalid fixtures, shared process/environment state,
  missing integration services, and excessive test duration using evidence.
- Fix product bugs at their producer. Fix fixtures only when they demonstrably
  violate the intended contract; never disable a production invariant to pass.
- Establish isolated temporary paths, explicit runtime environment, and required
  service prerequisites. Do not reuse shared production resources for validation.
- Run an unfiltered full suite after the identified failures are resolved.

### Gate

Every failure has a test name, reproducer, cause, and disposition. Required local
suites pass without undocumented exclusions. Service-dependent suites are either
run with their documented prerequisites or explicitly reported as unverified;
skips are not passes.

## 1. Close authority and recovery gaps

### Work

- Trace actual ingress into memory recall, index synchronization, consolidation,
  snapshot load/restore, and proactive workers. Build a supported-path matrix
  before adding metadata or adapters.
- Keep [MemoryIndexer](../dadbot/runtime/semantic_memory_bridge.py) bound to an
  explicit tenant. Filter canonical tenant/session identity before limiting
  events, even when the underlying reader returns several tenants.
- Audit the bridge's index synchronization against committed catalog fences.
  Observation must stay read-only; indexing/repair belongs behind committed-state
  authority, not arbitrary event materialization.
- Extend existing projection metadata only where missing: schema version,
  source tenant revision, applied revision, generation, dirty/degraded state,
  and rebuild provenance. Reuse `PROJECTION_STATUS_UPDATED`; do not add a parallel
  status store.
- Verify tenant-qualified index locations, caches, recovery status, and adapter
  filters under concurrent access.
- Reconcile [loop snapshots](../dadbot/runtime/loop_snapshot.py) with canonical
  authority. Tenant ownership and conservative driver-record accounting are
  implemented; canonical receipt reconciliation and durable goal state remain
  open. A snapshot must not serve as an alternative authority.
- Make supported recovery paths fail explicitly on corrupt or mismatched state.
  Preserve source memory and ledger history during migration.

### Gate

Two tenants using the same session name cannot read, index, restore, or mutate
each other's data. Revocation between scheduling, execution, and publication
blocks the affected operation. Stale/missing projections rebuild from the proper
tenant's committed state; corrupt authority never falls back to staged state.

Extend existing authority, tenant-memory, local-mutation, projection-status, and
loop-resume tests rather than duplicating their coverage.

### Supported-path status

| Path | Authority boundary | Status |
| --- | --- | --- |
| Loop observation | Tenant-bound provider, committed memory/insight catalogs, verified semantic read fence | Implemented; does not publish access updates or schedule index repair |
| Explicit semantic repair | Committed catalog and existing tenant revision/publication fences | Implemented; empty catalogs clear obsolete rows; failures propagate |
| Ledger recall fallback | Live authority and canonical tenant/session filtering before the event limit | Implemented |
| Projection recovery | Versioned ledger-backed status and tenant-qualified SQLite indexes | Local publication ordering covered; adapter and multi-process rebuild certification remain open |
| Loop snapshots | Tenant/session-hashed paths, exact payload ownership, live authority, atomic publication | Diagnostic snapshots hardened; canonical receipt reconciliation and production resume wiring remain open |
| Executive startup tasks | Tenant-qualified Markdown boards, owner headers, live authority, and fenced atomic updates | Implemented; unowned legacy tasks are preserved but excluded |

Observation providers pass `record_access=False` to hybrid retrieval; custom
hybrid adapters must accept that keyword. Ordinary retrieval retains access
recording by default. `semantic_sync_index=True` explicitly requests repair from
the committed catalog; the loop default is false. Reflection reads consolidated
facts through `consolidated_memory_catalog()`, not the mutation-capable
`consolidated_memories()` accessor.

Semantic lookup propagates authorization, unavailable-authority, and corrupt
ledger errors. Captured authenticated providers cannot downgrade to trusted-local
access when their authority context disappears, and retained contexts recheck
membership after revocation. These repairs do not complete milestone 1.

Recall repair (`7755dcb7`) targeted validation: **324 passed, 13 slow/stress tests skipped** across authority,
recall, recovery, isolation, and architecture checks. The unchanged scale gates
passed in 101.21 s / 120 s, 13.41 s / 60 s, and 5.51 s / 30 s. The full suite was
not rerun for this repair group. Changed-file syntax/import lint checks pass;
overall lint remains non-clean, including argument-count and complexity
diagnostics on the extended retrieval API.

Executive task boards use `executive_tasks.<sha256-tenant-id>.md` under the
workspace session-log directory and require an exact owner header. The pending
reader requires `tenant_id`; missing boards return no tasks without creating
files. Corrupt, duplicate-ID, or mismatched-owner boards fail explicitly.
Mutations hold the tenant authority and file locks through atomic publication.
Task fields must fit a single Markdown row; task IDs no longer depend on seconds.

The old unowned board is neither read into observations nor reassigned on startup.
Migration requires an operator to assign rows to verified owners and re-add them
through each tenant's executive tools, retaining the original board.

Executive-task repair validation: **230 passed, zero skips** across the task,
MCP, UI continuation, driver, tenant, authority, and architecture selection.
Coverage includes a fresh-process reload, cross-tenant read/completion attempts,
corrupt-source preservation, failed publication, concurrent updates, and captured
authority revocation. This is targeted evidence, not a new full-suite result.

Loop snapshots require an explicit tenant and versioned, owner-matched JSON.
Missing reads do not create files; corrupt, ambiguous, or mismatched snapshots
fail without replacement or deletion. Unowned legacy files remain untouched and
excluded. New managers refuse to overwrite an existing run; reconcile canonical
state before explicitly removing its snapshot.

The reflection wrapper records only prior driver dispatch records, not the
pending turn index. Failed, rejected, and skipped attempts do not advance the
completed count. Finalization checks result identity and counters against its
records. The wrapper can lag the last dispatch after interruption. These are
local diagnostics, not receipt-backed proof: they cannot authorize skipping or
resending actions. `AgentDriverLoop.run()` has no snapshot-resume API, and
AssistantRuntime does not wire this standalone manager into production recovery.

Snapshot repair validation: **151 passed, zero skips** across snapshot lifecycle,
driver, memory bridge, authority, tenant isolation, and architecture tests.
Coverage includes fresh-process reload, corruption preservation, publication
failure, session-name collisions, and revocation. No new changed-file lint or
Pylance error diagnostics; the full suite was not rerun.

Projection recovery retains `PROJECTION_STATUS_UPDATED` as its status authority.
Metadata version 1 records the attempted committed source revision, last
successfully applied revision, required revision, generation, dirty flag, and
rebuild reason. A failed attempt does not advance the applied revision.
Semantic recovery uses the committed catalog, not staged Observation state.
Authority failures propagate rather than being recorded as backend degradation;
status reads and publication revalidate live tenant authority.

Legacy successful statuses infer an applied revision from their source. Failed
statuses retain prior successful evidence, or leave the applied revision unknown
when none exists. Invalid or future-revision metadata fails explicitly without
rewriting ledger history. Metadata versioning does not certify every adapter's
physical index schema.

Projection repair (`8c129806`) validation: **202 passed, zero skips** across recovery,
process-termination, authority, isolation, and architecture tests. The unchanged
scale deadlines passed at 112.18 s / 120 s, 20.62 s / 60 s, and 8.96 s / 30 s.
No new lint diagnostics; existing lint/type debt remains. This is targeted
evidence, not a new full-suite result. Milestone 1 remains open.

New projection generations use the ledger publication sequence, not a count of
rebuild attempts. Allocation and append share the ledger lock; head-conflict
retries allocate again after reload. Replay follows durable event order, retains
known successful applied revisions on failure, and excludes reports older than
the known applied revision from the current status. Superseded attempts remain
in the audit history. Existing WAL/revision fences still guard index writes.

Concurrent-publication validation: **208 passed, zero skips**, including
overlapping rebuilds, independent ledger writers, late failures, and the existing
stale-index/crash-recovery cases. Scale gates passed at 102.80 s / 120 s,
20.80 s / 60 s, and 8.77 s / 30 s. No new lint diagnostics. Full-suite,
multi-process rebuild, and non-SQLite adapter certification remain separate.

## 2. Certify memory-first observation

### Work

- Enumerate supported autonomous entry points and deliberate opt-outs. The
  AssistantRuntime default already exists; raw loop construction and custom hooks
  need explicit contracts, not an unconditional global behavior change.
- Route production observation through the existing
  [MemoryQueryManager](../dadbot/managers/memory_query.py). Evaluate its
  `retrieve_context` bundle before adding another ranker.
- Preserve source IDs, authority revision, provenance, temporal validity, and
  conflict references through retrieval and prompt pruning.
- Resolve entities, rank lexical/semantic evidence, include relevant graph and
  consolidated facts, deduplicate, and honor a bounded context budget.
- Distinguish empty recall from unavailable/corrupt authority. Safe lexical or
  semantic degradation must be visible; an authorization failure cannot become
  an empty successful result.
- Remove retrieval-time writes where they undermine read-only observation or
  cause repeated full-index rebuilds.
- Add a small, versioned, synthetic retrieval dataset: exact facts, paraphrases,
  graph relationships, cross-session preferences, superseded facts, irrelevant
  queries, and tenant collisions.

### Proposed measurable gate

- Evidence recall@5 >= 0.85 and nDCG@5 >= 0.80 on the labeled dataset.
- No regression against the current path on exact-fact cases.
- Zero cross-tenant or invalid/superseded evidence in current-fact answers.
- Every supported default loop turn uses the certified provider; opted-out paths
  are separately documented and tested.
- Candidate p95 recall latency <= 1.20 times the recorded baseline on the same
  hardware/corpus. Freeze corpus size, model/backend, context budget, and absolute
  latency budget during milestone 0 before treating performance as certified.

These are proposed acceptance targets, not current measurements. Ranking quality
must be measured; merely checking that the provider was called is insufficient.

## 3. Make loop progress truthful and durable

### Work

- Keep structured reflection and the existing pre-dispatch plan gate.
- Obtain completion evidence from the canonical response/receipt contract.
  `_dispatch_plan` currently labels any normally returned response as committed;
  test error-shaped responses and publication failure before changing that logic.
- Separate proposed goal updates from committed goal progress. `run` currently
  updates its in-memory board before dispatch; failed work must not appear as
  completed progress.
- Persist and recover goal identity, accepted progress, unresolved blockers,
  next step, and revision through the existing canonical state/ledger path.
  Keep the schema bounded; do not create a second world-model database.
- Add outcome verification based on tool receipts and required postconditions.
  Use an optional critic only for ambiguity that deterministic checks cannot
  settle. Its assessment is not execution authority.
- Feed observed outcomes and consolidation status into the next reflection.
  Use bounded retry/escalation policies that distinguish definitely-undispatched
  work from uncertain external effects.

### Execution boundary

At the loop level:

`Observe -> Reflect -> Validate plan -> Canonical execute/commit -> Verify outcome
-> Update goal evidence -> Continue or stop`

Within canonical execution, retain existing tool checks and SaveNode ordering.
A loop-level critic after return cannot prevent a commit or undo an external
effect that already happened. If a postcondition must gate a commit, implement it
inside the existing canonical boundary and prove the change with a reproducer.

### Gate

Rejected plans never dispatch. Failure-shaped responses never count as committed
success. Failed/uncertain actions do not advance successful goal progress.
Restart restores the last committed board and remaining budgets. Recorded
outcomes replay without model calls or duplicate external execution.

## 4. Harden online consolidation

### Work

- Preserve the existing post-commit integration; do not call a second
  consolidation job after each loop turn.
- Replace reliance on ephemeral readiness alone with recoverable work linked to
  a committed tenant/session/trace receipt. Reuse the ledger and existing owned
  task mechanisms where their contracts fit.
- Make pending work reconstructible across the commit/enqueue crash window.
  Use a stable idempotency key based on tenant, committed source IDs/revisions,
  and consolidation policy version.
- Carry and revalidate authority when the worker runs and when it publishes.
  Thread context or a retained callable must not substitute for authorization.
- Separate extracted episodes from proposed semantic facts. Validate sources,
  provenance, schema, contradiction handling, and supersession before committing
  accepted semantic changes.
- Do not promote generated replies or arbitrary tool text to facts without
  supporting evidence. Filter failed, rejected, and uncertain source outcomes.
- Report `skipped`, `deferred`, `degraded`, `failed`, and `published` distinctly.
  An operation returning the existing view is not proof that new memory was saved.
- Add bounded queues, batch sizes, model budgets, backoff, backlog visibility,
  and shutdown behavior. Preserve intentional LIGHT_MODE/source-mode behavior.
- Reconcile the manual [MemoryConsolidationJob](../dadbot/runtime/semantic_memory_bridge.py)
  with the canonical path; retain a thin manual trigger rather than two policies.

### Gate

Crash tests at commit/enqueue, worker execution, publication, and acknowledgement
lose no eligible work and produce at most one committed semantic result per
idempotency key. A repeated delivery may recompute, but cannot publish duplicates.
Invalid output/revocation leaves authoritative memory unchanged and reports why.
New accepted facts are retrievable at their committed revision without rebuilding
the entire index on each observation.

## 5. Deepen temporal and conflict semantics

### Work

- Reuse graph `valid_from`/`valid_until`, event time, and ingestion time. Define
  UTC normalization and exact interval boundaries before extending records.
- Support both "what was true at T?" and "what did we know at T?" where the
  retained provenance supports the distinction. Never invent missing dates.
- Model corrections and supersession without destroying historical evidence.
  Late-arriving events must not make an older fact current merely because it
  was ingested later.
- Resolve entities and scopes before treating two statements as contradictions.
  Separate temporal changes, incompatible claims, and uncertain evidence.
- Make automatic decisions conservative and provenance-linked. Escalate
  unresolved conflicts to existing review surfaces; user resolution does not
  grant new tool permission.
- Feed bounded, source-linked conflict and validity information into reflection
  and consolidation, not an unbounded narrative history.

### Gate

A synthetic sequence with late arrivals, corrections, timezone offsets,
overlapping validity, and contradictory evidence yields the expected current and
historical answers before and after restart. Superseded facts do not win
current-fact retrieval. Resolution records preserve both source lineage and the
reason for the accepted state.

## 6. Harden proactive execution

### Work

- Move processed-trigger identity and terminal outcome into the canonical
  tenant-scoped ledger/state path. The current 256-entry in-memory cache is an
  optimization, not a restart guarantee.
- Reuse existing task ownership/claim semantics for multiple workers. Separate
  durable processing acknowledgement from the UI's queued-message lifecycle.
- Handle committed response, suppressed tick, reflection stop/no-op, rejected
  plan, definitely-undispatched failure, and uncertain dispatch separately.
- Verify tool denial end to end in canonical dispatch for default heartbeat
  turns. Do not rely solely on an empty reflection allowlist or event wording.
- Keep one-event/one-turn bounds, quiet hours, rate limits, shutdown, and a
  disable switch. Scheduled goals use explicit user-owned policy, not event text
  as authorization.

### Gate

Process restart and competing workers do not produce duplicate committed
processing for the same tenant trigger. UI delivery still works. Revoked or
suppressed work cannot dispatch. Default proactive turns cannot invoke tools,
and uncertain external work is not automatically resent.

## 7. Certify infrastructure wrappers

### Work

- Inventory existing locking, authority stores, secret configuration, and
  durable adapters. Add only gaps required by a named deployment profile.
- Certify lease expiry/fencing, worker ownership, retry bounds, and stale-writer
  rejection without replacing the ledger's integrity checks.
- Integrate secret loading and rotation with explicit failure and redacted
  diagnostics. Do not move secrets into repository configuration or plan files.
- Test tenant scoping, revision conflicts, and recovery on supported database
  and queue adapters using real service integration tests.
- Keep local-first startup simple. Distributed services remain optional until
  the deployment requires them.

### Gate

The chosen profile passes real cross-process/service tests for revocation,
concurrent ownership, restart, and stale writes. Report other profiles as
uncertified rather than claiming general distributed readiness.

## Human-maintainability gate

Apply this to every milestone, not as a separate rewrite:

- Prefer existing mechanisms; delete redundant wrappers when safely proven.
- Keep contracts and acceptance evidence in the main docs; avoid new status
  diaries and repeated architectural explanations.
- Use short comments for non-obvious invariants only.
- Require a short human review of non-trivial changes: clear happy path,
  necessary abstractions, credible failure behavior, and concise documentation.
- Update README/goalplan claims only when the corresponding gate passes.

## Release and rollback

1. Deliver independently reviewable commits with focused regressions.
2. Run the smallest relevant suite first, then interaction/authority tests, then
   the unfiltered broad suite at milestone release.
3. Start richer critic/consolidation policies in opt-in or shadow mode.
   Shadow observations cannot write memory or dispatch external work.
4. Roll back behavior by disabling the new policy. Do not erase ledger history
   or reconstruct authority from old indexes; durable corrections require
   explicit compensating records and compatible replay.
5. Publish exact test results, unresolved failures, integration prerequisites,
   quality metrics, and known limitations. Commit/push only after the agreed gate
   passes, or after an explicit exception with the caveat recorded.

## Completion

The plan is complete when a tenant-scoped goal can observe evidence, reflect,
act through canonical execution, verify outcomes, consolidate accepted learning,
and resume after restart without losing work, leaking another tenant's state,
inventing success, or repeating uncertain external effects.

Required evidence includes:

- A clean broad local suite and executed tests for the selected deployment profile.
- Retrieval targets measured on the fixed dataset.
- Crash/revocation/concurrency cases for progress, learning, and triggers.
- One end-to-end scenario: remember a sourced fact, complete a bounded goal,
  consolidate its verified outcome, restart, handle a later correction, and
  answer current/historical questions with the correct lineage.

## Scope and estimate

This is an estimated 4-8 week sequence, not a delivery promise. Re-estimate after
milestone 0 reveals the broad-suite failures and milestone 1 fixes the supported
backend matrix. Move beyond each gate only with passing evidence.

Not included: kernel rewrites, a new agent framework, wholesale memory-store
replacement, unbounded autonomy, policy self-modification, or claims of parity
with another product without a shared benchmark.

Current execution step: milestone 0. The validated fixture correction is
published; diagnosis of the remaining broad-suite failures is in progress.
