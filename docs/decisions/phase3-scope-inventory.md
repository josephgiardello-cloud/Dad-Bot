# Phase 3 store-scoping inventory

> **Inventory and status snapshot.** The decisions in
> [SIGN-OFF-PACKET.md](SIGN-OFF-PACKET.md) are signed. This document does not
> itself implement store changes; current Stage 1 task-store implementation
> and the deferred memory operations are recorded in
> [TENANT_EXECUTION_PATH_AUDIT.md](../../TENANT_EXECUTION_PATH_AUDIT.md).
> Remaining store ownership below is the proposed work breakdown for
> Phase 3.3–3.9. Stage 1 task ownership is complete. Core session ownership,
> Redis/PostgreSQL adapters, and cross-process worker-cache invalidation are
> implemented; Phase 3.4 remains open pending other independent gates. Semantic and graph memory
> repair remain blocked until their phases are deliberately scheduled.
> Cache invalidation and the stale-authority JSON-mirror regression remain
> separate open Phase 3.4 gates; this follow-up does not close either item.

| Store / surface | Current key, constraint, and query pattern | Implementation / remaining work | Proposed Phase 3 owner | `default` decision interaction | Legacy-task migration dependency | Acceptance test after scope lands |
|---|---|---|---|---|---|---|
| Tasks | [`dadbot_system/owned_sqlite_tasks.py`](../../dadbot_system/owned_sqlite_tasks.py): SQLite `owned_tasks_v1` uses `PRIMARY KEY (tenant_id, task_id)`; task reads/writes require both values. | **Implemented in Stage 1** for the service task store: composite identity, tenant-required APIs, and legacy-row quarantine. | **3.2 — task, response, and incomplete-task ownership** | **Yes.** `default` is accepted only behind the signed non-production compatibility gate; it is not inferred from missing task ownership. | **Yes, first.** Unqualified legacy rows are rejected/quarantined, not adopted by a tenant. | Same task ID and session under two tenants preserve separate records across processes; the seven task-state gate nodes pass. |
| Responses / task results | [`dadbot_system/owned_sqlite_tasks.py`](../../dadbot_system/owned_sqlite_tasks.py): response is stored on the composite task row and read/written only by `(tenant_id, task_id)`. | **Implemented in Stage 1** through owner-qualified result and response APIs. | **3.2 — task, response, and incomplete-task ownership** | **Yes**, inherited from task ownership; no implicit `default` response namespace. | **Yes**, same task-row quarantine as above. | Identical task IDs under two tenants preserve distinct responses across process restart; the task-state gate passes. |
| Incomplete-task listing and recovery | [`dadbot_system/owned_sqlite_tasks.py`](../../dadbot_system/owned_sqlite_tasks.py): pending query filters `tenant_id` before ordering/limit; recovery enumerates owners only from owned rows. | **Implemented in Stage 1** for task-owned listing and recovery. | **3.2 — task, response, and incomplete-task ownership** | **Yes.** `default` is never an implicit recovery principal. | **Yes, first.** Legacy rows are quarantined and excluded from recovery. | Mixed-owner listing with a limit and fresh-process recovery return only owner-qualified tasks; the seven task-state gate nodes pass. |
| Session registry | [`dadbot/core/control_plane.py`](../../dadbot/core/control_plane.py): registry, termination state, jobs, leases, dedupe, and public timelines use tenant-qualified identity; [`dadbot/core/session_store.py`](../../dadbot/core/session_store.py) replays only owner-stamped events. Production service snapshots/events route through tenant-bound views to [`dadbot_system/owned_sessions.py`](../../dadbot_system/owned_sessions.py): SQLite/PostgreSQL use composite `(tenant_id, session_id)` rows and Redis uses tenant-qualified keys. PostgreSQL applies forced RLS; Redis operations atomically reject colliding legacy keys. | Core lifecycle and backend adapters are implemented and wired. Live Redis and PostgreSQL integration passed against disposable services using a non-superuser `NOBYPASSRLS` PostgreSQL role. Authenticated termination commits a tenant/session revocation revision to the authority SQLite store; worker processes poll while idle and evict only that owner/session cache. Existing authority databases must be upgraded by rerunning the idempotent `SQLiteDelegationStore.initialize`. | **3.4 — sessions and checkpoints (open)** | **Yes.** Authenticated paths carry explicit tenant identity; unowned ledger events and backend records are not replayed/adopted. | **Yes.** Legacy service rows and unowned ledger events are rejected pending explicit migration. | Same session ID across tenants remains isolated through creation, recovery, reads, listing, mutation, termination, and cross-process cache invalidation. Live backend parity is demonstrated for tested paths; checkpoint backend parity and other Phase 3.4 gates remain outstanding. |
| Checkpoints | SQLite [`dadbot/core/persistence/checkpointer.py`](../../dadbot/core/persistence/checkpointer.py) schema v2 keys rows and write evidence by `(tenant_id, session_id, trace_id)`, scopes chain verification and maintenance queries, and validates tenant/session/trace metadata. Both SQLite and graph-ledger recovery revalidate delegated authority in authenticated execution; graph checkpoints bind `session_id` into their hash and validate against the tenant/session/trace event envelope. Prior unowned SQLite tables are quarantined. | SQLite ownership, transactional writes, cross-tenant restore rejection, revocation-aware recovery, and fresh-process recovery are implemented. Graph-checkpoint writes remain production-wired through the tenant-bound execution ledger. Redis/PostgreSQL checkpoint adapters and real integration parity are not available and remain blocked. | **3.4 — sessions and checkpoints (open)** | **Yes.** Tenant comes from the active request scope or an explicit/configured owner; row metadata cannot override it. Graph restore/listing uses explicit or active tenant/session identity. | **Yes.** Existing unowned checkpoint rows remain quarantined; no implicit assignment to `"default"`. | Same session/trace under two tenants stores and loads distinct checkpoints; concurrent/interrupted SQLite writes, revocation-aware recovery, and fresh-process recovery preserve owner and hash-chain verification. Live Redis/PostgreSQL checkpoint evidence is still required. |
| Semantic index | [`dadbot_system/semantic_index.py`](../../dadbot_system/semantic_index.py#L115): SQLite `semantic_memories.summary_key TEXT PRIMARY KEY`; upsert/delete/fetch key by normalized summary. [`dadbot/memory/semantic_manager.py`](../../dadbot/memory/semantic_manager.py#L588) derives that key from normalized summary text; PostgreSQL vector backend is also selectable. | Use tenant-aware row keys/unique constraints and require tenant predicates for candidate, recent, keyed, delete, and count operations across SQLite and PGVector; partition caches consistently. Physical isolation remains undecided. | **3.5 — semantic index** | **Yes.** No implicit namespace for `default`; exact treatment follows packet Item 4. | **No direct task migration**, unless semantic records are restored/rebuilt from a legacy task; then rebuild from approved owner-bound source data. | Identical summaries in two tenants produce separately addressable vectors; query, delete, and rebuild in one tenant cannot observe or mutate the other's records. |
| Graph store | [`dadbot_system/graph_store.py`](../../dadbot_system/graph_store.py#L81): `graph_nodes.node_key` and `graph_edges.edge_key` are global primary keys; graph reads select all rows and replacement writes use those keys. SQLite and PostgreSQL variants follow the same key model. | Add tenant-qualified keys/unique constraints and tenant-filter every graph read, replace, delete, and edge traversal; resolve physical index/database isolation separately under deficit 27. | **3.6 — graph store** | **Yes.** Do not treat a shared graph or `default` as proof of organization sharing. | **No direct task migration**, except any graph reconstruction sourced from legacy task/session data must use only owner-verified source records. | Identical node/edge logical IDs coexist for two tenants; reads, traversal, replacement, and deletion are owner-isolated. |
| Streamlit threads | [`dadbot/ui/streamlit_context.py`](../../dadbot/ui/streamlit_context.py#L151): `chat_threads.thread_id TEXT PRIMARY KEY`; `load_threads()` selects every row and `save_thread()` upserts by thread ID. | Add owner-qualified composite identity; load and save only after a trusted browser/session principal is resolved; scope any thread-to-memory snapshot links consistently. | **3.7 — Streamlit thread persistence** | **Yes.** The current `"default"` thread fallback is not an entitlement or authenticated principal. | **No direct task migration.** If thread history is linked to recovered tasks, apply the task classification before exposing linked results. | Same thread ID under two tenants yields isolated histories; missing context cannot load or write any default/global thread. |
| External-tool idempotency registry | [`dadbot/core/kernel_locks.py`](../../dadbot/core/kernel_locks.py#L58): process-wide `_records: dict[str, Any]`, addressed by the supplied key; deterministic keys hash tool name and payload. [`dadbot/core/external_tool_runtime.py`](../../dadbot/core/external_tool_runtime.py#L807) accepts caller `_idempotency_key` verbatim or derives one with a fixed global scope. | Derive the effective key from trusted tenant context plus tool and idempotency key; persist/expire records according to the side-effect replay contract and scope reads/writes/clears. Do not trust caller-supplied tenant strings as authority. | **3.8 — external-tool idempotency** | **Yes.** An idempotency key under `default` cannot be shared across tenants implicitly. | **No direct task migration**, unless an idempotency record is reconstructed from a legacy task; then verify the task owner first. | Same caller idempotency key and tool invocation in two tenants do not replay one another's result; restart behavior matches the documented persistence guarantee. |

### Current Phase 3.4 progress — core sessions and SQLite checkpoints

The core control-plane session registry, job/dedupe/lease identities,
ledger-backed session projection, bootstrap replay, timeline/event readers,
and scheduler/reconciliation scopes now carry tenant ownership. Unowned legacy
events are not restored or scheduled. SQLite checkpoints use tenant-qualified
keys, owner-checked metadata and chain queries; old unowned checkpoint tables
are quarantined rather than adopted. Same-ID cross-tenant, concurrent, crash
replay, checkpoint interruption, and fresh-process tests pass.

**Phase 3.4 remains OPEN.** Redis and PostgreSQL have production-wired
`OwnedSessionStore` adapters with tenant-qualified storage and fail-closed
legacy handling. Live Redis and PostgreSQL integration tests passed against
disposable services; PostgreSQL used and verified a `NOSUPERUSER NOBYPASSRLS`
role. The live selection also covered fresh-process recovery, concurrency,
forced-RLS context reset, and the PostgreSQL execution-ledger integration
cases. Authenticated termination commits a tenant/session revocation revision;
idle workers poll the authority store and evict only the matching cache.
Do not close Phase 3.4 until checkpoint backend parity and the independent
remaining lifecycle gates are satisfied.

Memory matrix cases that require semantic or graph repair are **BLOCKED** until
Phase 3.5/3.6 is deliberately scheduled. This work does not start either
phase.

## Proposed sequencing and cross-store acceptance

1. **3.2** establishes the trusted tenant-qualified task/response APIs.
2. **3.3** scopes listing and recovery on those APIs and quarantines unowned
   legacy tasks.
3. **3.4–3.8** apply the approved owner identity to each listed store; do not
   infer a universal `"default"` policy or a shared-index policy.
4. **3.9 — end-to-end migration and validation:** run migration dry-runs,
   verify rollback/recovery and legacy quarantine, then execute a two-tenant
   same-ID acceptance matrix covering task, response, session, checkpoint,
   semantic, graph, thread, and idempotency reads/writes. Require zero
   cross-owner results and preserve auditable source-to-target mapping.

Before any migration, record its approved owner, mapping, compatibility
window, rollback strategy, and counts. This inventory is not authorization to
perform a migration or choose physical isolation.
