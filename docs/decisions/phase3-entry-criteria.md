# Phase 3 entry criteria — SQLite task and response ownership

> **Inventory only. None of these failures is fixed by this document or the
> current change. They remain Phase 3 exit-gate acceptance tests.**

Pre-flight rerun:
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
application-accessible fallback. This change intentionally does not alter
SQLite schemas, raw-store APIs, recovery, task listing, or any of these tests.
