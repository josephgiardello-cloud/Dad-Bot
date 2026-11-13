# Agent and Memory Hardening Plan

Date: 2026-11-11  
Status: Milestone 0 has a passing default local baseline. Supported-profile
authority/recovery acceptance and opt-in, receipt-verified File-WAL loop resume
are implemented. PostgreSQL/Redis primitives have real-service evidence;
non-file loop resume, non-SQLite physical projections, goal postconditions, and
live-model quality remain uncertified. Milestone 2's synthetic File-WAL/SQLite
observation quality and ingress checks pass, and the final runtime passed its
unfiltered release gate. Skipped capabilities remain outside certification.

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
| Driver Loop | Bounded turns, structured reflection, plan validation, canonical request/run identities, and scoped completion verification exist in [AgentDriverLoop](../dadbot/runtime/agent_driver_loop.py). Successful post-turn processing persists a bounded goal board and hash-only tool outcome summaries. [AssistantRuntime](../dadbot/assistant_runtime.py) reconciles missing progress and offers opt-in, receipt-verified File-WAL resume with durable budgets and a pre-dispatch marker. | Verify goal postconditions; reconcile uncertain external effects and support receipt-key rotation. Non-file loop resume remains explicitly unavailable. |
| Memory-first observation | [AssistantRuntime](../dadbot/assistant_runtime.py) uses the read-only `retrieve_context` bundle by default. The versioned synthetic File-WAL/SQLite profile passes measured quality, budget, provenance, isolation, five actual canonical ingress checks, and the unfiltered release gate. | Live-model quality and non-SQLite observation profiles remain uncertified; paraphrase performance is weak with offline embeddings. |
| Online consolidation | [PersistenceService](../dadbot/services/persistence.py) publishes post-commit readiness; [PostCommitWorker](../dadbot/services/post_commit_worker.py) invokes consolidation. | Durable scheduling, authority propagation, idempotency, meaningful success reporting, and recovery after process loss. |
| Durable memory authority | [MemoryStorageBackend](../dadbot/memory/storage.py) authorizes memory access, fences local index writes, and records tenant-specific projection status. Bridge, worker, snapshot, File-WAL/SQLite recovery, and PostgreSQL/Redis primitive paths have executed acceptance coverage below. | Non-SQLite physical memory projection recovery and authenticated PostgreSQL execution are not certified; do not infer those capabilities from primitive service tests. |
| Temporal/conflict handling | Reflection receives bounded timestamp/conflict signals. [Bi-temporal edge tests](../tests/test_bitemporal_graph_edges.py) cover existing edge validity fields. | Consistent as-of retrieval, provenance-aware corrections, late-arrival handling, and conflict decisions surviving restart. |
| Proactivity | A heartbeat processes at most one queued event through the loop; its reflection plan has an empty tool allowlist. | Durable deduplication and worker ownership; verify downstream tool enforcement, not just reflection policy. |
| Goalplan | [CURRENT_GOALPLAN](CURRENT_GOALPLAN.md) was updated on November 11, 2026. | Replace broad capability claims with acceptance evidence after each hardening milestone. |

These are repository observations, not a measured comparison with Zep or another
product.

### Validation status

The final milestone-2 runtime passed the unfiltered default suite on
November 13, 2026: **4,812 passed, 124 skipped, 10 expected failures, zero
failures** in **3,632.18 seconds**. No acceptance deadlines, authentication
checks, or service skip conditions were relaxed. The milestone-2 section below
records the timing anomalies, final evidence-only fingerprint correction,
focused rerun, measured quality, and remaining certification boundaries.

The final authority/recovery checkout passed the unfiltered default suite on
November 12, 2026: **4,789 passed, 124 skipped, 10 expected failures, zero failures**
in **1,347.74 seconds (22m 27s)**. The tested checkout was `1d64625f` plus the
recovery changes described below and the fixture corrections in this section.
No acceptance deadlines or authority checks were relaxed.

The first full run of that recovery checkout produced **4,783 passed, 124 skipped,
10 expected failures, six failures** in **1,391.11 seconds**. All six reproduced
in the focused selection (**six failed, 13 passed**) and were fixture contract
mismatches:

- The [agentic loop test](../tests/test_agentic_methods.py) replaced canonical
  execution with a success-shaped response without a committed completion.
  It now records real canonical turns using offline model transport, checks
  committed goal-board progression, and asserts two canonical completions.
- Five [response-engine integration tests](../tests/unit/test_response_engine_control_plane_integration.py)
  set the tenant in job metadata but left `job.tenant_id` as an automatically
  created Mock. The fixtures now bind the explicit job owner to their authorized
  tenant scope. Tenant mismatch rejection remains unchanged.

Both owning modules and the directly related driver/progress regressions then
passed together: **43 passed, zero failures, zero skips**. Changed-fixture
syntax/import lint checks and `git diff --check` passed.

The successful unfiltered rerun used the command below, fresh isolated
`TEMP`/`TMP` and pytest storage, and no selectors or marker exclusions.
Its [Phase 4A](../tests/test_phase4a.py) gates passed at **115.072 s / 120 s**
(110 canonical turns), **21.030 s / 60 s** (eight sessions, five turns each),
and **8.561 s / 30 s** (exact pruning count), measured from JUnit.

The **124 skips** include PostgreSQL/PGVector (36), Redis (6), local live-model
checks (6), missing optional `litellm` collection (1), production Compose
certification (2), slow/stress lanes (71), and explicitly quarantined cases (2).
This local full-suite result does not certify those paths. The separate
**437-test** supported-profile run below did execute selected PostgreSQL/Redis
primitives against disposable services; it does not certify PGVector,
non-SQLite physical projections, authenticated PostgreSQL execution, or live-model
quality. The 10 expected failures are separate from the 124 skips.

The earlier milestone-0 unfiltered default suite passed on November 12, 2026: **4,743 passed,
123 skipped, 10 expected failures, zero failures** in **1,361.19 seconds
(22m 41s)**. The tested checkout was `827499a9` plus the fixture-only correction
described below, published together with this record. No production code,
acceptance deadlines, or authority checks changed.

The first run on unchanged `827499a9` produced **4,742 passed, 123 skipped,
10 expected failures, one failure** in 1,354.16 seconds. The failure in
`test_reminder_tool_authorization_and_restart_idempotency` in the
[regression module](../tests/test_dadbot_regression.py) reproduced alone:
the initial fixture bot and authenticated reminder request had different tenant
owners. The fixture now constructs its bot with an explicit canonical tenant;
the request and restarted bot use that configured owner. Authorization denial,
successful reminder creation, and restart idempotency remain asserted. The
complete regression module then passed: **153 passed, 10 expected failures**.

The final full rerun used no test selectors or marker exclusions:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp <fresh-isolated-directory> -o faulthandler_timeout=0 --tb=short -ra --junitxml <report.xml>
```

`TEMP` and `TMP` used an isolated short path. The faulthandler override disables
auxiliary stack dumps, not pytest-timeout or the scale acceptance limits.
The [Phase 4A](../tests/test_phase4a.py) scale gates passed in that full run
(JUnit elapsed case times):

| Gate | Time | Limit |
| --- | --- | --- |
| 110 canonical turns | 115.36 s | 120 s |
| Eight sessions, five turns each | 21.76 s | 60 s |
| 20-turn exact pruning count | 9.56 s | 30 s |

The earlier unfiltered baseline at `d02ac36d` was **4,659 passed, 123 skipped,
10 expected failures, zero failures**.

Baseline repairs cover tenant-qualified fixtures, materialized checkpoint views,
same-request tool replay, deterministic pipeline clocks, selective-copy
instrumentation, Streamlit/Windows process isolation, and import-cycle removal.
Canonical hashing and chain verification now share the existing
[canonical event module](../dadbot/core/canonical_event.py). UI and heartbeat
dispatch share the same [runtime lock](../dadbot/core/runtime_locks.py).

Service integrations, optional dependencies, slow/stress runs, and explicitly
quarantined cases remain unverified where skipped, except for the separate
executed supported-profile evidence below. Skips are not passes. The local
baseline does not certify every memory projection, consolidation path, or
production deployment.

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
  implemented. Production resume now uses a canonical cursor and reverified
  completion receipts, never snapshot counters. A snapshot must not serve as
  an alternative authority.
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
| Projection recovery | Versioned ledger-backed status and tenant-qualified SQLite indexes | File-WAL/SQLite concurrent graph and semantic rebuilds, process-loss repair, corruption preservation, and revocation exercised; non-SQLite physical projections remain uncertified |
| Loop snapshots and resume | Tenant/session-hashed diagnostic files; canonical cursor and exact completion/receipt verification for execution | File-WAL resume wired through the kernel gateway; missing or incompatible completion evidence blocks resend. Snapshot files remain untouched and non-authoritative |
| Executive startup tasks | Tenant-qualified Markdown boards, owner headers, live authority, and fenced atomic updates | Implemented; unowned legacy tasks are preserved but excluded |
| PostgreSQL primitives | Isolated execution-ledger streams; tenant-owned session/task paths and forced RLS with a non-bypass role | Real service tests cover restart, process loss, corruption, competing writers, and same-key tenant isolation. This is not authenticated PostgreSQL execution or physical memory-index certification |
| Redis and Redis/PostgreSQL composite | Tenant-owned session/task envelopes; PostgreSQL durable authority in the composite path | Real service tests cover fresh-process recovery, concurrent tenant writes, scoped termination, corrupt envelopes, and durable composite task recovery/publication. Redis alone is not an execution WAL |

The selected service tests use `DADBOT_TEST_POSTGRES_DSN` and
`DADBOT_TEST_REDIS_URL`, disposable loopback-only containers, unique table/key
namespaces, and a PostgreSQL `NOSUPERUSER NOBYPASSRLS` login. They do not contact
the existing application containers or live models. Primitive acceptance does
not enable non-file loop resume; an explicit regression verifies that refusal.

Final supported-path validation on November 12, 2026: **437 passed, zero failures,
zero skips** in **265.15 seconds**. The selection covered loop/progress/receipt
tests, semantic bridges, projection status and repair, delegated workers,
authenticated WAL boundaries, SQLite service publication, tenant memory restore
and mutation, PostgreSQL/Redis integration, checkpoint adapters, and ledger
authority invariants. It used isolated pytest storage, no marker exclusions,
and `-o faulthandler_timeout=0`; the normal test deadlines were unchanged.
Focused recovery helpers and added tests pass Ruff; Pylance reports no diagnostics
in the new recovery module, progress module, gateway, or driver. The facade still
has unrelated existing type/lint diagnostics, and the keyword-rich loop APIs
exceed the repository's argument-count lint threshold.

This is not an unfiltered full-suite result. Live-model quality, optional
dependency/service paths outside this selection, slow/stress lanes outside this
selection, receipt-key rotation, and non-SQLite physical memory projections
remain uncertified. The disposable service containers were removed after testing.

Production recovery uses `durable=True` to start and `resume=True` to continue
with the same tenant/session, initial-input hash, and policy. A cross-process
File-WAL claim serializes competing loop runs. The cursor persists before
dispatch and after each definite outcome; an interrupted dispatch requires its
exact `JOB_COMPLETED`, recorded result, and valid stable-key receipt chain.
Uncertain work remains blocked even when a caller attempts a new ordinary loop.
Ready checkpoints retain turn/failure/no-op budgets and system observations.
Completed checkpoints return without model calls or duplicate execution.
Standalone snapshots are not migrated, deleted, or used as resume input.

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

Validated goal updates are now applied to the in-memory goal blackboard only
after a dispatch returns committed. A failed dispatch cannot make its proposal
appear as progress to later reflections. That initial repair did not persist
goal state; the subsequent canonical progress and cursor paths described above do.

Snapshot repair validation: **151 passed, zero skips** across snapshot lifecycle,
driver, memory bridge, authority, tenant isolation, and architecture tests.
Coverage includes fresh-process reload, corruption preservation, publication
failure, session-name collisions, and revocation. No new changed-file lint or
Pylance error diagnostics; the full suite was not rerun.

Goal progress repair validation: **36 passed, zero failures** across the driver
loop and snapshot regression modules. This is targeted evidence, not a new
full-suite result.

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

Two independent processes now have a regression test for simultaneous stale
projection-status publication: both events must commit with distinct ledger
sequences and the replayed status must reflect the latest sequence. This
validates cross-process status append/retry, not concurrent physical index
rebuilds or non-SQLite adapters; those remain open.

Concurrent-publication validation: **208 passed, zero skips**, including
overlapping rebuilds, independent ledger writers, late failures, and the existing
stale-index/crash-recovery cases. Scale gates passed at 102.80 s / 120 s,
20.80 s / 60 s, and 8.77 s / 30 s. No new lint diagnostics. Full-suite,
multi-process rebuild, and non-SQLite adapter certification remain separate.

Cross-process status-publication validation: **1 passed** with two synchronized
processes starting from the same ledger head. The broader projection suite was
not rerun for this addition.

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

### Approved measurable gate

- Evidence recall@5 >= 0.85 and nDCG@5 >= 0.80 on the labeled dataset.
- No regression against the current path on exact-fact cases.
- Zero cross-tenant or invalid/superseded evidence in current-fact answers.
- Every supported default loop turn uses the certified provider; opted-out paths
  are separately documented and tested.
- Candidate p95 recall latency <= 1.20 times the recorded baseline on the same
  hardware/corpus. Freeze corpus size, model/backend, context budget, and absolute
  latency budget before treating performance as certified.

The user approved a synthetic File-WAL/SQLite profile with deterministic offline
embeddings and an absolute p95 budget of **200 ms**. The quality and ingress selection and final-runtime unfiltered release run pass
for this profile.

### Measured profile and results

The versioned corpus and regression gate are in
[memory_benchmark.py](../tests/eval/memory_benchmark.py) and
[test_memory_observation_benchmark.py](../tests/eval/test_memory_observation_benchmark.py).
The profile freezes:

- Dataset `memory-observation.v1`, corpus/query date **2026-11-12**.
- **55 records**: 50 raw facts, three active consolidated facts, one superseded
  consolidated fact, and one archive.
- **257 queries**: 200 exact forms, all 50 authored paraphrases, and seven
  graph/cross-session/supersession/archive/irrelevant cases; 255 positive queries.
- Corpus SHA256
  `4233c90f57b6cbc6437bdb655f124ab8603837e2e12563d2c5e3fd8d7f5094ac`,
  covering exactly the 55 normalized source records, not unrelated runtime
  mood/relationship timestamps.
- File-WAL/SQLite, `tests.conftest.fake_embed_texts` 12-dimensional offline
  vectors, top-k **5**, rendered context budget **12,000 characters**.
- Windows `Windows-11-10.0.26200-SP0`,
  `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`.

Both paths are warmed and measured on the same immutable catalog/query order.
The baseline is the previous read-only raw-memory hybrid provider; the candidate
adds the existing query manager's native graph/consolidated/archive evidence.
Binary nDCG credits each relevant summary once and excludes metadata-only
matches. Supplemental recall gates prevent the exact forms from hiding missing
source groups. The table reports the final fingerprint-guarded owning-module
run; the unfiltered release also passed the same quality gates, with candidate
p95 **75.191 ms** against baseline **79.754 ms**.

| Measure | Paired baseline | Candidate | Gate |
| --- | --- | --- | --- |
| Recall@5 | 0.866667 | **0.886275** | >= 0.85 |
| nDCG@5 | 0.858621 | **0.874182** | >= 0.80 |
| p95 retrieval | 76.539 ms | **90.779 ms** | <= 91.847 ms and <= 200 ms |
| Exact recall / nDCG | 1.0 / 1.0 | **1.0 / 1.0** | No exact-fact loss |
| Paraphrase recall / nDCG | 0.42 / 0.378969 | **0.42 / 0.365712** | Report separately |
| Graph recall / nDCG | 0.0 / 0.0 | **1.0 / 0.815465** | Recall 1.0 |
| Cross-session, supersession, archive recall / nDCG | 0.0 / 0.0 | **1.0 / 1.0** each | Recall 1.0 |

Neither irrelevant query receives evidence; superseded facts are excluded.
Tenant-collision tests use independent populated canonical catalogs. Ledger
equality checks verify that observation/degradation/concurrent reads do not
publish authority or access updates.

This aggregate is dominated by exact forms. The weak offline embeddings leave
paraphrase recall at **42%**, and paraphrase nDCG is slightly lower than baseline.
These results do not establish general semantic quality, live-model quality,
general historical/correction semantics, or parity with another memory product.

### Implemented observation contract

[MemoryQueryManager](../dadbot/managers/memory_query.py) preserves the ordinary
`retrieve_context` access-recording/model-selection behavior by default.
`record_access=False` uses committed tenant snapshots and the existing
semantic/BM25 ranker, qualifies native graph evidence against query entities,
deduplicates summaries, and checks catalog/graph revisions. Read-only ranking
keeps each query's scores and revision local rather than consuming shared
last-query diagnostics. Graph read/sync results expose the verified revision
without relaxing stale-runtime fences.

[The bridge](../dadbot/runtime/semantic_memory_bridge.py) carries content-derived
source IDs, tenant/revision, available raw provenance, timestamps, conflicts,
source metadata, and edge validity. It rejects malformed bundles, missing IDs,
and foreign tenant rows. Consolidated/archive schemas lack complete raw-fact
provenance; no lineage is fabricated. Embedding failure is visibly degraded,
and legacy adapter fallbacks are explicitly degraded. Revoked authority,
corrupt WALs, and changed revisions are errors, not empty successful recall.
After corruption, the ledger remains blocked until a full verified backend
reload; the corruption test restores its owned WAL and performs that reload.

[The pruner](../dadbot/runtime/context_pruner.py) enforces the complete rendered
budget, including identity/task/headings, drops entire evidence entries and
corresponding reflection signals, reports dropped snippets, and rejects pinned
inputs that cannot fit. Recall remains anchored to the initial goal while the
latest reply and system observation remain available in the rendered task.
Default executive startup notes enter the task before pruning; custom/disabled
observation hooks retain their prior composition contract. A regression test
first reproduced an oversized startup prefix escaping the budget, then verified
both bounded startup rendering and explicit pre-dispatch refusal when pinned
goal/startup text alone cannot fit.
The facade's default top-k intentionally changes **3 -> 5**.

| Entry point | Evidence |
| --- | --- |
| Default interactive `AssistantRuntime.run_agent_loop` | Two real canonical completions with native observation and kernel source/revision/provenance |
| Opt-in durable loop | Same populated-catalog checks with stable signed receipts |
| Canonical resume | Interrupt before the second dispatch; continue with preserved initial goal and native evidence |
| Queued proactive trigger | Actual canonical dispatch with native observation/kernel evidence |
| Heartbeat dispatch | Same evidence through heartbeat processing |
| Disabled memory observation / custom hook | Default observation provider bypassed before dispatch |
| Raw `AgentDriverLoop` | Remains explicitly hook-controlled; no global default injection |

Observation opt-outs do not bypass canonical kernel integrity checks. Actual
File-WAL agent-loop turns now use positive committed tenant-revision evidence in
subconscious retrieval, rather than requiring sovereign checksum fields that
the normalized memory schema does not retain. The path checks turn/evidence
ownership and publishes derived source IDs/revision/provenance; it never
fabricates sovereign checksums. Ordinary/non-file integrity behavior remains
unchanged.

### Validation and boundaries

The final owning-module and dependent authority/recovery selection passed:
**334 passed, 16 skipped, zero failures** in **211.37 seconds**. It includes
the benchmark, populated-catalog ingresses, corruption/revocation refusal,
overlapping read-only queries, tenant collisions, stale graph refusal, staged
memory isolation, process-exit index recovery, and canonical loop resume.
The 16 skips are 13 slow/stress memory cases and three live Ollama embedding cases.

The broader selection exposed and corrected a diagnostic summary leak, stale
test interception of the old retrieval call, graph read/sync result inconsistency,
and corruption-test teardown without a verified reload. Additional kernel
evidence assertions accommodate the existing canonical "shared that" summary
normalization while requiring the original provenance evidence.
Editor diagnostics report no errors in the changed observation modules.
New benchmark/test syntax/import lint checks pass; existing ranker/adapter
complexity and other lint debt are not a clean whole-module lint certification.

The final runtime's unfiltered default release passed on November 13, 2026:
**4,812 passed, 124 skipped, 10 expected failures, zero failures** in
**3,632.18 seconds**. The unchanged scale gates passed at **68.708 s / 120 s**
(110 turns), **13.681 s / 60 s** (eight concurrent sessions), and
**5.761 s / 30 s** (exact pruning count), from JUnit.

The first broad observation run produced **4,805 passed, five failures,
124 skipped, 10 expected failures** in **5,800.06 seconds**. All five failures
were WebSocket ticket validation cases returning 401 instead of their expected
403/422. Those fixtures mint one-hour tokens at collection, and the run recorded
a **4,810.73-second normalization-test teardown**, consistent with expiry before
ticket tests ran. The ticket module passed alone (**30 passed, one service skip**).
An unchanged unfiltered rerun passed **4,810 tests**, with **124 skipped** and
**10 expected failures**, in **984.73 seconds**. No authentication code or ticket
fixtures were changed. The startup budget regression was then fixed and verified
with **88 owning-module tests, zero skips**, followed by the final unfiltered
run above.

The final run also recorded an unexplained **2,656.70-second durable-ingress
case**; the focused case completed in about four seconds. These wall-time stalls
are recorded, not explained away or used to claim end-to-end loop latency
certification. The retrieval p95 measurements and scale limits above are
separate measured gates.

After that full run, only benchmark evidence generation changed: its hash now
covers the actual 55-record corpus and asserts its fixed fingerprint instead of
hashing unrelated runtime timestamps. Canonical WAL comparison verified identical
50 raw, four consolidated, and one archive records across the November 12/13
runs. The final test files then passed together: **88 passed, zero skips** in
**64.63 seconds**, including the fixed fingerprint, quality/latency thresholds,
all ingress cases, and startup-budget checks. No runtime code changed after the
successful final full run. Focused test lint, production syntax/import checks,
editor diagnostics, and `git diff --check` passed within the limits stated above.

The 124 skips retain the earlier breakdown: 36 PostgreSQL/PGVector, six Redis,
six live-model, one optional `litellm` collection, two production Compose,
71 slow/stress, and two quarantined cases. The 10 expected failures are separate.
PostgreSQL/Redis/PGVector, non-SQLite physical projections, authenticated
PostgreSQL execution, non-file loop resume, and live-model checks retain their
previous explicit certification limits.

## 3. Make loop progress truthful and durable

### Work

- Keep structured reflection and the existing pre-dispatch plan gate.
- Each dispatch now carries a unique request ID and loop-run ID. When the
  production control plane is available, count a normal response as committed
  only after the tenant/session/request-scoped `LedgerIndex` confirms
  `JOB_COMPLETED`; missing evidence and dispatch exceptions stop as uncertain.
- Goal proposals now update in-memory progress only after verified completion.
  The successful post-turn path persists a bounded, versioned board plus its
  tenant/session/trace/job/request/run/turn identity in canonical session state.
  `AssistantRuntime` restores only state matching the active tenant and session.
- The persisted goal schema covers the accepted board and last commit identity.
  Opt-in File-WAL runs additionally commit a bounded cursor, pending request
  marker, policy/input binding, and remaining budgets through the kernel gateway.
  This is not a second world-model database and never authorizes uncertain resend.
- Agent-loop completion metadata now carries at most 16 hash/status/tool identity
  summaries from checksum-validated tool events for the current trace. The
  canonical `JOB_COMPLETED` lookup feeds them into the next reflection as
  observations; raw tool inputs/outputs are excluded, and these summaries are
  explicitly not proof of goal postconditions.
- On startup, recover missing goal progress only from tenant/session-scoped
  canonical `JOB_COMPLETED` metadata after the persisted request marker. Each
  recovered completion must carry the same pre-dispatch goal-board snapshot as
  the current projection; stale or ambiguous tails fail explicitly. Recovery
  commits the reconstructed progress through `SESSION_STATE_UPDATED` and never
  redispatches the completed action.
- Add outcome verification based on tool receipts and required postconditions.
  Use an optional critic only for ambiguity that deterministic checks cannot
  settle. Its assessment is not execution authority.
- The canonical graph now emits receipts after successful, non-skipped stages
  and their after-checkpoints, verifies turn scope, order, HMAC signatures, and
  chain continuity, and stores a bounded envelope in `JOB_COMPLETED` metadata
  for agent-loop turns. This path is enabled only with a configured key of at
  least 32 bytes; the default process-random signer is explicitly unavailable
  for durable evidence.
- Receipts establish signed stage/checkpoint-chain evidence, not goal
  postcondition success. Partial File-WAL loops can resume after re-verifying
  canonical completion evidence; uncertain external-effect reconciliation,
  deterministic postconditions, and key rotation remain open.
- Use bounded retry/escalation policies that distinguish definitely-undispatched
  work from uncertain external effects.
- Feed observed outcomes and consolidation status into the next reflection;
  receipt-backed outcome verification remains open.
- Focused loop, progress-state, structured-output, facade, ledger-index,
  tool-recording, recovery, and receipt tests passed with isolated pytest
  temporary storage. Targeted Ruff checks and `git diff --check` passed; the
  broader lint selection still reports existing diagnostics in legacy files.
  The full suite was not rerun for this change. Deterministic postcondition
  checks, non-file loop recovery, and non-SQLite physical projection
  certification remain open; supported-profile acceptance evidence is above.

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

Current execution step: the supported File-WAL/SQLite authority/recovery paths
and selected PostgreSQL/Redis primitives now have executed acceptance evidence.
Opt-in File-WAL loop recovery is wired to canonical completion receipts.
Do not promote authenticated PostgreSQL, non-SQLite physical memory projections,
non-file loop resume, or live-model behavior to certified status.
Milestone 2's synthetic File-WAL/SQLite quality, bounded evidence, default ingress,
and final-runtime unfiltered release gates now pass. Next is milestone 3:
goal postconditions, uncertain-effect reconciliation, and key rotation.
Service-dependent and other skipped capabilities retain the explicit limits
recorded above; unexplained long wall-time stalls are not latency certification.
