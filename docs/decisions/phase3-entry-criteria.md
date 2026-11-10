# Phase 3 entry criteria — SQLite task and response ownership

> **Historical baseline.** The seven failures below were observed before
> tenant-owned task storage was wired. Current verification is recorded under
> “Current Stage 1 verification” and supersedes the historical failure status.

Historical pre-implementation rerun:
`.\.venv\Scripts\python.exe -m pytest --basetemp=<session-temp>\signoff-preflight-phase3 tests\unit\test_sqlite_service_state.py -q`
reported **7 failed** and the remaining tests in that file passed. Failure
descriptions are based on that run. The Phase 1.5 validator accepts exact
legacy canonical `tenant-a` values for compatibility, but does not confer
ownership; none of these failures is caused by rewriting an ID.

| Node ID | Current failure mode | Owning Phase 3 task | Phase 1.5 behavior contribution | Acceptance test when implemented |
|---|---|---|---|---|
| `tests/unit/test_sqlite_service_state.py::test_task_namespace_isolated_across_processes` | Tenant B's raw namespaced lookup returns tenant A's same-ID task. | **3.2** — wire `SQLiteOwnedTasks` and tenant/task composite-key access. | None; exact legacy ID remains unchanged. | Fresh process using B's tenant context returns no A-owned task for the same task ID. |
| `tests/unit/test_sqlite_service_state.py::test_identical_task_ids_do_not_overwrite_another_namespace` | Second tenant's write overwrites the first tenant's same-ID task. | **3.2** — tenant/task composite uniqueness and writes. | None; no normalization occurs. | Concurrent/sequential writes with identical task IDs preserve both owners' records and reads. |
| `tests/unit/test_sqlite_service_state.py::test_task_result_and_response_are_namespace_owned` | Second tenant's task result/response is returned to the first tenant. | **3.2** — scope response/result APIs with task ownership. | None; no normalization occurs. | Identical task IDs have isolated task, result, and response retrieval under both tenant contexts. |
| `tests/unit/test_sqlite_service_state.py::test_incomplete_task_listing_filters_before_limit` | Listing applies a global limit before ownership filtering and returns another tenant's task. | **3.3** — tenant-filtered recovery/listing before pagination/limit. | None; no normalization occurs. | Mixed-tenant rows with `limit=1` return the caller tenant's first eligible task, never another tenant's row. |
| `tests/unit/test_sqlite_service_state.py::test_legacy_task_is_quarantined_without_implicit_ownership` | Recovery does not quarantine an unqualified legacy task and does not raise. | **3.3** — tenant-aware recovery and explicit legacy-row quarantine. | **No direct behavior change:** Phase 1.5 preserves exact legacy IDs but explicitly does not treat syntax as provenance or ownership. | A legacy task without an authenticated tenant binding is quarantined and cannot be enqueued or executed. |
| `tests/unit/test_sqlite_service_state.py::test_raw_store_rejects_unqualified_task_lookup` | Raw store permits a task lookup without a tenant key. | **3.2** — remove/inaccessibly wrap raw task APIs; require owner-qualified methods. | None; this is a storage API boundary omission. | Every application-facing read/write path requires tenant context; direct unqualified task lookup fails closed. |
| `tests/unit/test_sqlite_service_state.py::test_fresh_process_recovers_same_task_id_for_both_owners` | Recovery finds only one of two same-ID tasks across owners. | **3.3** (depends on **3.2**) — tenant-scoped recovery over composite task identity. | None; exact legacy IDs are preserved. | Restart recovers both tasks independently under their matching tenant scopes; no global list is exposed. |

## Exit gate

All seven node IDs must pass unmarked after Phase 3 task/response scoping and
tenant-aware recovery are implemented. The raw store must not remain an
application-accessible fallback.

### Current Stage 1 verification — 2026-11-09

The seven node IDs above were rerun unmarked after wiring `SQLiteOwnedTasks`
through orchestration and tenant-scoping task, response, incomplete-listing,
and recovery access. **All seven passed; none failed or errored.** The same
run also passed the real-process concurrent same-session task/response test.
This closes the listed task/response/recovery gate only; it does not close
Phase 3.3–3.9 store work or the deferred memory matrix cases.

### Current Phase 3.4 progress — core sessions and SQLite checkpoints

The control-plane registry and termination state, job/dedupe/lease identity,
session-state replay, public ledger/timeline reads, worker/orchestrator
propagation, and scheduler/reconciliation scopes are tenant-qualified.
Unowned session events and queued jobs are excluded from recovery. Durable
SQLite checkpoints now use `(tenant_id, session_id, trace_id)`, check stored
metadata against row ownership, scope hash-chain and cleanup queries, and
quarantine pre-v2 unowned tables. Fresh-process same-ID tests cover both
control-plane session replay and checkpoint recovery; tests also cover
cross-tenant reads/deletes, concurrency, missing/mismatched owners, and
interrupted checkpoint writes.

The focused command covering persistence, tenant-session, ledger-bound,
ledger-reader, turn-kernel, lease, WAL recovery, reconciliation, service
architecture, and app-runtime tests passed. The previously excluded three
service-architecture and two app-runtime regressions are included and pass.
The stale-authority mirror regression was reproduced: when an authoritative
document-store read failed, the default `allow_tenant_mirror_fallback=True`
could silently import stale JSON and write it back. The default now derives
from `_local_ephemeral_document_store`; persistent stores fail closed, while
the explicit local-ephemeral fallback remains available. The regression test
asserts that a failed authoritative read neither loads the stale JSON nor
writes it back; it passed in the final run.

**Phase 3.4 remains open:** Redis and PostgreSQL tenant-owned session
repositories are now implemented and wired through service orchestration and
tenant-bound runtime/document persistence. Redis uses tenant-qualified keys
and atomic legacy-key guards; PostgreSQL uses owner-qualified tables, forced
RLS, tenant-scoped queries, and quarantines unowned legacy rows by rejecting
access pending explicit migration. Both live adapters were exercised against
disposable local services. PostgreSQL ran as a `NOSUPERUSER NOBYPASSRLS` role;
tests verified forced-RLS cross-tenant read/write denial and transaction-local
tenant-context reset. Redis and PostgreSQL tests covered same-ID isolation,
metadata mismatch, tenant-scoped listing, termination, concurrency, and
fresh-process recovery. The final live-backend selection completed with
**41 passed, 0 skipped, 0 failed**.

Authenticated session termination now revokes the affected tenant/session
delegations before deleting persisted session state and advances a durable
tenant/session revocation revision in the authority SQLite store. Authenticated
workers poll that state while idle and evict only the matching tenant/session
cache. A spawned-worker regression confirms that termination under one tenant
causes its bot to be recreated while the same session ID under another tenant
remains cached and authorized. Existing authority databases must be upgraded
by rerunning the idempotent `SQLiteDelegationStore.initialize`; workers and API
startup fail closed if the revocation schema is absent. No legacy checkpoint
or event was assigned an owner.

The final combined service/session/authority regression selection completed
with **119 passed, 0 skipped, 0 failed**. The previously excluded
service-architecture and app-runtime tests were included in that run.

Semantic/graph-dependent memory matrix cases remain **BLOCKED** pending an
explicit decision to schedule Phase 3.5/3.6.
