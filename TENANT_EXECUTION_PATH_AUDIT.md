# Tenant identity, ownership and execution-path audit

Status: **partial repository-wide investigation; tenant isolation is not
fully established**. The original findings below are an audit snapshot. The
later execution records supersede stale statuses where noted; they are not a
claim that every file, runtime combination or attack scenario has been
traced. All changes remain local to `tenant-identity-remediation`; no push or
merge has been performed.

## Evidence conventions and boundaries

- **Reproduced:** observed with a real process or a previously run test.
- **Code-observed:** follows directly from the indicated implementation; an
  end-to-end exploit or deployed configuration has not been demonstrated.
- **Unknown:** a dependency, configuration, call path, or enforcement has not
  been proven. Absence of a discovered test is not proof that none exists.
- Trust boundaries considered: local CLI/UI user to runtime; authenticated
  network client to API; API to control plane and worker; process to shared
  file/database/broker; local MCP client to tools; external service to tool;
  persisted records back into recovery.
- The production Python inventory contains hundreds of tracked files,
  including 279 under `dadbot/core`. This investigation samples their entry
  and storage boundaries, not all control paths within them. External
  integrations, alternate adapters and deployment combinations require
  further trace and executable coverage.

## Prioritized findings

1. **Historically reproduced; Phase 0 fix now covered by a strict acceptance
   test.** Two separate real DadBot
   processes with different tenant IDs sharing the same local paths cause
   tenant B to load tenant A's synthetic memory in the old implementation.
   The test is no longer xfailed and passes normally after Phase 0. See the
   updated evidence below; this does not prove all storage modes.
2. **Code-observed: shared indexes lack tenant ownership.** SQLite semantic
   and graph tables have global primary keys and unscoped deletion and
   replacement; pgvector semantic storage also uses a global summary key.
   This is independent of the import bug. A catalog hash check or tenant
   filter in a caller cannot scope all write and administrative operations.
3. **Historically code-observed; pulse route fixed and regression-tested in
   Phase 0.** The pulse route now resolves snapshots with both tenant and
   session identity; its same-session, two-tenant test asserts separate
   cognition and reflection payloads.
4. **Historically code-observed; the compatibility-named normalizer now
   validates without rewriting.** `tenant/a` is rejected rather than changed
   to `tenant-a`; see the Phase 1 call-site inventory and migration report.
   Exact legacy-format IDs remain accepted during migration, so issuance
   provenance is not yet enforced at every call site.
5. **Code-observed: the worker factory may drop identity.** A custom factory
   raising `TypeError` is retried without `tenant_id`
   ([`worker.py`](dadbot_system/worker.py#L109)). This also catches
   `TypeError` from inside the factory. The production/default factory passes
   tenant ID; whether a particular custom deployment uses the fallback is
   unknown.
6. **Closed for authenticated spatial and admin runtime effects.** Spatial
   WebSocket requests now submit tenant-bound delegated tasks using the
   authenticated principal. Admin automation requires a runtime bot bound to
   the principal's tenant and executes within a live authority scope; action
   tools fail closed without verified authority. Two-tenant, missing-context,
   and revocation regressions cover these paths. Trusted-local action execution
   remains supported.
7. **Code-observed: local MCP and generic tool helpers have different trust
   models.** Local MCP uses one provided or global bot and exposes memory,
   file and action tools; no per-call tenant principal is visible in the
   examined handlers. Its intended local transport and upstream authorization
   must be verified before classifying exposure. The generic external tool
   runtime allows execution if its optional permission enforcer is absent;
   determine which live call sites install one.

## Execution-path inventory

Each row gives: **entry**, identity and ownership source; enforcement and
failure behavior; downstream dependencies; existing relevant coverage; and
known bypasses or unknowns. Tests named are existing tests, not necessarily
executed for this report.

### Startup, identity and lifecycle

| Entry and identity / ownership | Enforcement and failure | Downstream | Tests | Bypass / unknown |
|---|---|---|---|---|
| CLI, Streamlit and service boot -> `DadBot` constructor -> [`_initialize_config`](dadbot/core/boot_mixin.py#L197): explicit tenant or environment, normalized by [`DadBotConfig`](dadbot/config.py#L344). Paths come from global runtime configuration. | Normalization replaces characters and defaults empty input. Path creation does not establish per-tenant ownership. Invalid environment modes fail separately. | Profile, memory, indexes, logs, ledger and background managers. | [`test_config_drift.py`](tests/test_config_drift.py), [`test_execution_ledger_boot_order.py`](tests/unit/test_execution_ledger_boot_order.py). | Normalization collision reproduced; tenant ownership of paths and existing-data migration are not enforced. Streamlit and CLI authorization are local trust assumptions, not network-API guarantees. |
| [`_initialize_document_store`](dadbot/core/boot_mixin.py#L233): configured store, namespace `tenant-doc:<tenant>` plus the explicit tenant identity. | Session/document snapshots route to the backend's `owned_sessions` repository; legacy namespaced rows are rejected by the owner-aware backend rather than adopted. | Profile and memory snapshot hydration. | [`test_dadbot_regression.py`](tests/test_dadbot_regression.py) document-store tenant test; [`test_app_runtime_startup.py`](tests/test_app_runtime_startup.py#L216); owned SQLite legacy-rejection test. | Generic state-store methods and shared filesystem fallbacks remain separate paths; authenticated Redis/PostgreSQL live-service behavior still needs integration evidence. |
| [`_hydrate_profile_and_memory`](dadbot/core/boot_mixin.py#L351): runtime config and namespaced store. | [`load_profile`](dadbot/managers/runtime_storage.py#L88) falls back to global profile file if no tenant document. | Profile facts, prompts, memory initialization. | Profile and document-store tests above. | Code-observed unowned profile fallback; no fresh-process cross-tenant profile test identified in this investigation. |
| [`shutdown`](dadbot/core/boot_mixin.py#L575): bot-bound tenant/config, no caller principal. | Waits for background manager and calls save in `finally`; errors propagate, completion flag set only after save. | Snapshot/ledger and per-bot caches. | Lifecycle and authority tests in [`test_local_authority_mutation_application.py`](tests/unit/test_local_authority_mutation_application.py). | Cross-process shutdown with shared paths and pending indexing/exports needs acceptance tests. |

### API, service identity and event boundaries

| Entry and identity / ownership | Enforcement and failure | Downstream | Tests | Bypass / unknown |
|---|---|---|---|---|
| [`_http_principal`](dadbot_system/api.py#L262): bearer/header token verified by `ServiceTokenManager`; health bypasses token. | Missing/invalid token -> 401 when auth enabled; optional auth-disabled mode gives anonymous default tenant read/write/admin. Individual handlers enforce scopes. | Chat, automation, events and task APIs. | [`test_service_api_integration.py`](tests/test_service_api_integration.py#L314). | Production policy permits configuration to disable authentication; whether deployment should prohibit it is a design decision. Health is intentionally unauthenticated. |
| [`submit_chat`, channel ingest and turn](dadbot_system/api.py#L872): principal tenant and requested tenant. | Write scope, process-local rate limit and exact requested/principal check; identity is inserted into request metadata. | Orchestrator, control plane, task store and worker. | [`test_api_rejects_tenant_mismatch`](tests/test_service_api_integration.py#L794). | Subsequent lossy normalization and shared backends can break isolation despite ingress check. Audit every alternate ingress and metadata consumer. |
| [`get_task_status`](dadbot_system/api.py#L1024): principal and stored task payload. | Read scope; load by task ID, then compare payload tenant; missing -> 404, mismatch -> 403. | Global task/response store. | API integration tests; cross-tenant task-ID coverage not confirmed. | [`NamespacedStateStore`](dadbot_system/state.py#L268) forwards task methods without prefix; correctness depends on API checks and payload integrity. Missing tenant field and internal callers need tests. |
| Events and replay ([`api.py`](dadbot_system/api.py#L1129), [`api.py`](dadbot_system/api.py#L1180)): principal or path-bound ticket and optional requested tenant. | Read scope; omitted tenant resolves to the principal, while an explicit mismatch is rejected; live event stream filters both session and tenant. | Namespaced event store and event bus. | [`test_service_api_integration.py`](tests/test_service_api_integration.py): omitted-tenant event/replay reads and cross-tenant rejection. | Ticket reuse and event recovery remain separate checks. |
| Pulse ([`api.py`](dadbot_system/api.py#L1216)): authenticated effective tenant. | Handshake validates tenant; `_session_state_snapshot` resolves the orchestrator session with both session ID and tenant ID. | Tenant-keyed orchestrator session container. | Pulse happy-path and same-session cross-tenant integration tests. | Alternate orchestrator implementations without tenant-aware `session()` return an empty snapshot rather than falling back to the shared registry. |
| Spatial and admin automation ([`api.py`](dadbot_system/api.py#L1035), [`api.py`](dadbot_system/api.py#L1248)): authenticated principal and effective tenant. | Spatial submits a delegated task; admin runtime operations require matching runtime tenant and active authority; missing authority blocks live action tools. | Worker, runtime maintenance, browser/MCP. | [`test_service_api_integration.py`](tests/test_service_api_integration.py): same-session tenant isolation, runtime mismatch, and membership revocation. | Other direct runtime integrations remain in the wider audit scope. |
| [`recover_pending_tasks`](dadbot_system/orchestration.py#L164), `apply_worker_result`: persisted task tenant, request and result. | Postgres recovery enumerates tenant IDs from its transactionally updated task catalog; Redis + Postgres exposes a tenant-qualified composite store with durable status precedence; request digest canonicalization survives serialization. | Broker, worker, state store and response/event persistence. | Live Postgres API-lifespan, tenant-catalog, RLS, and Redis + Postgres recovery/publication tests pass; persisted recovery was run in a fresh Python process. | Broader Phase 3.4 gates listed below remain open. |

### Ledger, memory, projections and replay

| Entry and identity / ownership | Enforcement and failure | Downstream | Tests | Bypass / unknown |
|---|---|---|---|---|
| [`establish_local_tenant`](dadbot/core/local_tenant_establishment.py#L39): bot-config tenant and per-tenant evidence alongside shared WAL. | Serialized WAL lock, initial revision and evidence validation; if another tenant already has committed memory, a fresh tenant starts from defaults without reading the shared mirror. Otherwise, an existing mirror without ownership evidence is rejected. WAL recovery repairs only the tenant document store, not the shared mirror. | [`_load_legacy_for_establishment`](dadbot/memory/storage.py#L719), tenant document store and WAL. | [`test_local_tenant_establishment_application.py`](tests/unit/test_local_tenant_establishment_application.py), passing two-tenant acceptance and unowned-mirror rejection. | Legacy mirror migration requires explicit ownership evidence; an existing unowned mirror blocks first-time establishment. Shared profile ownership remains a separate concern. |
| [`commit_local_tenant` / `_commit_local_projection`](dadbot/memory/storage.py#L278): bot-config tenant and expected revision. | WAL writer fence and revision checks; indexed changes are explicitly rejected; uncertain errors set recovery-required. | Ledger, memory projection, snapshot repair. | [`test_local_authority_mutation_application.py`](tests/unit/test_local_authority_mutation_application.py), [`test_tenant_memory_ledger.py`](tests/unit/test_tenant_memory_ledger.py). | Guard is local-file-authority-specific; do not infer authorization or backend-wide atomicity. |
| [`replay_tenant_memory`](dadbot/core/tenant_memory_ledger.py#L74): event payload tenant. | Validates per-tenant revision, operation IDs and committed state; corrupt tenant events fail replay. | Execution ledger backend and hydration. | [`test_tenant_memory_ledger.py`](tests/unit/test_tenant_memory_ledger.py#L28). | Integrity and labels are not access control; a caller with shared ledger access can request another tenant unless a boundary enforces it. |
| Memory query/mutation/clear/export ([`storage.py`](dadbot/memory/storage.py#L690), [`storage.py`](dadbot/memory/storage.py#L866)): current bot projection and configured tenant. | Local indexed mutation guard; export writes projection to caller-supplied path. | Legacy JSON and backup, document store, semantic and graph. | [`test_tenant_memory_restore_authority.py`](tests/unit/test_tenant_memory_restore_authority.py), memory tests. | Export ownership, OS permissions, direct manager callers and shared backup paths require end-to-end boundary tests. |
| Semantic read/rebuild/clear ([`semantic_manager.py`](dadbot/memory/semantic_manager.py#L82), [`semantic_index.py`](dadbot_system/semantic_index.py#L129), [`semantic_index.py`](dadbot_system/semantic_index.py#L470)): catalog and bot tenant; physical SQLite and pgvector rows have no tenant column. | Local file-ledger reads check catalog hashes under fence; local SQLite candidate rechecks revision and publishes transactionally. Clear and row deletion are global. | SQLite/pgvector, embedding cache, retrieval and startup recovery. | [`test_semantic_index_repair_failures.py`](tests/unit/test_semantic_index_repair_failures.py), [`test_local_authority_mutation_application.py`](tests/unit/test_local_authority_mutation_application.py). | Tenant-scoped index storage, generations and full embedding validation missing. Non-local path does not use this local hash verification. Shared SQLite clear may unlink the database on failure; consequences need reproduction. |
| Graph build/read/replace ([`graph_views.py`](dadbot/memory/graph_views.py#L169), [`graph_store.py`](dadbot_system/graph_store.py#L78)): current bot projection; global graph keys. | Local sync checks revision before replacement; local reads reconstruct and fence. SQLite replace deletes all rows transactionally. | Graph store and memory/context managers. | Local graph recovery and stale-writer tests in [`test_local_authority_mutation_application.py`](tests/unit/test_local_authority_mutation_application.py). | Physical shared store and non-local paths are not proven tenant-isolated. Direct backend operations bypass manager-level protections. |
| Session state/replay ([`control_plane.py`](dadbot/core/control_plane.py#L1897), [`control_plane.py`](dadbot/core/control_plane.py#L3937)): session ID, event stream. | State update written to ledger before registry update; bootstrap projects events keyed by session ID. | Session registry and durable execution ledger. | [`test_control_plane_ledger_bounds.py`](tests/unit/test_control_plane_ledger_bounds.py), replay suites. | Tenant is not part of the inspected registry key or session-state event payload; test equal session IDs across tenants and ledger streams. |
| Separate run/event store ([`event_durability.py`](dadbot_system/event_durability.py#L70)): run records include tenant; events are keyed by run ID. | Optional payload encryption; event operations in inspected path accept run ID. | Event tap, run history, checkpoints. | [`test_event_durability_security.py`](tests/test_event_durability_security.py). | Tenant checks in every event reader/writer and any API reachability were not traced; a tenant column alone is not enforcement. |

### Tools, integrations, background work, cache and exports

| Entry and identity / ownership | Enforcement and failure | Downstream | Tests | Bypass / unknown |
|---|---|---|---|---|
| [`DadBotTaskProcessor`](dadbot_system/worker.py#L109): `WorkerTask.tenant_id`, normalized bot cache key; request also carries tenant. | Default bot factory passes tenant; optional factory falls back without it on `TypeError`; task execution sets and restores request policy on bot. | Bot construction, snapshot load, turn and external tool pipeline. | Service integration/worker tests. | No full proof that task/request/bot tenant agree, nor that mutable request policy is isolated under concurrent access. |
| [`authorize_tool_execution`](dadbot/core/action_mixin.py#L27): configured tool names plus optional per-request allowed-tools policy. | Rejects names outside fixed allowlist; missing service policy permits allowed names. | Tool executor and integrations. | [`test_service_split.py`](tests/test_service_split.py) tool/memory authority tests. | Direct action methods, background calls and MCP handlers may not use this decision point. Trace each live integration before asserting a universal guard. |
| [`ExternalToolRuntime`](dadbot/core/external_tool_runtime.py#L636): optional permission enforcer and payload. | No enforcer -> allow in inspected helper; policy-sensitive idempotency context is assembled separately. | HTTP clients, registered handlers, optional memory sink. | [`external_tool_runtime_test.py`](tests/external_tool_runtime_test.py). | Which production factories always supply enforcers and whether tenant is present in all idempotency contexts remains unknown. |
| [`local_mcp_server.build_server`](dadbot_system/local_mcp_server.py#L310): provided bot or process-global bot; handlers expose memory, reminders, local filesystem and network research. | Workspace path resolution restricts examined file paths to project root. Per-request tenant/auth check is not visible in handlers. | Stdio MCP server, bot, workspace, browser and network. | MCP tests exist; exact transport and unauthorized-client coverage not reviewed. | Local access assumptions, client approval and alternate transports must be mapped. Workspace path containment is not tenant ownership. |
| Background boot tasks ([`boot_mixin.py`](dadbot/core/boot_mixin.py#L432)) and API-triggered learning: bot-bound state, sometimes API principal only at submission. | Background manager lifecycle and action-specific checks; exceptions have varying handling. | Graph refresh, calendar, heartbeat, learning and snapshots. | Boot and maintenance tests. | Tenant propagation, policy expiry, concurrency and shutdown across every scheduler and task type unverified. |
| Semantic embedding cache ([`semantic_manager.py`](dadbot/memory/semantic_manager.py#L126)): database path from global semantic config; key based on text/model. | Per-instance memory cache and SQLite operations. | Embeddings and semantic indexing. | Semantic determinism tests. | Shared persistent cache has no examined tenant namespace; confidentiality, inference, invalidation and deletion requirements need explicit assessment. |
| Source approval and answer cache ([`source_grounding.py`](dadbot/services/source_grounding.py#L58)): bot config tenant. | SQLite primary keys and queries include tenant; answer cache key includes tenant. | Memory/profile evidence and citation approval. | [`test_source_grounding.py`](tests/unit/test_source_grounding.py#L93). | Tenant-keyed approvals cannot compensate for unowned source data; full cross-process/shared-data tests needed. |
| Export, session log and handoff ([`storage.py`](dadbot/memory/storage.py#L866), [`conversation_persistence.py`](dadbot/managers/conversation_persistence.py#L643), [`conversation_persistence.py`](dadbot/managers/conversation_persistence.py#L1328)): bot tenant, configured paths, trace and worker claim. | Export writes requested path; session payload includes tenant but filename is timestamp-based; handoff has hash and worker lease checks. | File storage, backups, restore, CLI and UI. | Restore and handoff tests; tenant ownership coverage not verified. | File permissions, collisions, import ownership, trace-to-tenant binding and data retention need separate review. |

## Architectural causes and remediation sequence

### Task/result production boundary inventory (P1; incomplete)

This inventory documents the current code, not an implemented security
boundary. The seven production ownership acceptance cases remain failing.

- [`launch_api_service`](dadbot/runtime/launcher.py#L310) and the alternate
  service launcher in [`app_runtime.py`](dadbot/app_runtime.py#L975) use
  [`build_service_state_store`](dadbot/app_runtime.py#L843). Explicit SQLite
  configuration selects `SQLiteStateStore` only outside production. Otherwise
  the factory selects in-memory storage, or a `CompositeStateStore` combining
  Redis and PostgreSQL; production requires the PostgreSQL store. The factory
  currently catches configured Redis/PostgreSQL initialization failures and
  may fall back. The worker also constructs the service state store in
  production.
- [`StateStore`](dadbot_system/state.py#L84) defines unqualified task save,
  load, incomplete listing, result save and response save/load. SQLite
  [`tasks`](dadbot_system/state.py#L116) is keyed by `id` and stores response
  in the same row. PostgreSQL uses `task_id` as its sole primary key and a
  response column. Redis uses separate `task:{task_id}` and
  `response:{task_id}` keys. In-memory storage uses separate task and response
  dictionaries keyed by task ID. The composite delegates to its component
  stores, and the namespaced adapter forwards task IDs unchanged.
- [`submit_chat`](dadbot_system/orchestration.py#L101) persists a task before
  enqueueing it. [`_apply_worker_result`](dadbot_system/orchestration.py#L265)
  loads by task ID, verifies identity and, for authenticated execution,
  verifies the signed result before publishing. Status and response methods
  also load by task ID. [`record_response`](dadbot_system/state.py#L667)
  writes a response through the container store. The API veto path writes a
  task and response directly. The runtime adapter polls task status and
  response; the assistant and runtime-orchestration manager access background
  task records through the runtime state container.
- [`recover_pending_tasks`](dadbot_system/orchestration.py#L186) globally
  lists incomplete payloads, reconstructs the task, verifies delegation when
  authenticated and queues it. Its in-process deduplication set is keyed by
  task ID alone. API lifespan invokes recovery without a tenant or principal.
  Worker retries reverify the execute delegation before each attempt. The
  worker queue and result queue contain serialized task/result objects; the
  orchestrator consumes and publishes results.
- The examined state-store protocol has no task deletion or cancellation
  method. Background manager cleanup removes an in-process future, not a
  persisted task; worker retries and scheduler shutdown/cancellation are
  separate paths. Whether other dynamic integrations, checkpoint/replay
  adapters, or direct database consumers bypass this inventory remains open.
- [`SQLiteOwnedTasks`](dadbot_system/owned_sqlite_tasks.py#L18) is used only
  by its isolated tests; production does not instantiate it. Its separate
  table and caller-supplied tenant argument are neither the canonical store
  nor verified authority. Wiring this component alone would leave the
  original unqualified store and other backends accessible.

**Authority decision needed before production changes:** current HTTP
[`_resolve_principal`](dadbot_system/api.py#L211) verifies a bearer token but
does not check live membership for read operations; auth-disabled mode creates
an anonymous default-tenant principal. The only existing storage execution
context, [`require_tenant_authority`](dadbot_system/delegation.py#L39),
requires a task-bound execute delegation. It does not provide a read
principal, installation-bound local operator, or a startup service identity
that can enumerate tenants. Recovery currently reads payloads before verifying
their execute delegations. A tenant string or task metadata must not become
authority, and execute delegation must not become administrative migration
authority. Provisioning, permissions and verification for startup recovery
and explicit trusted-local operation therefore remain design inputs.

The selected direction is a separately provisioned service identity with
per-tenant recovery grants; trusted-local protected task operations are to
fail closed until an installation-bound local operator is provisioned. The
service credential source and provisioning/rotation mechanism are not yet
specified. A generic admin scope or task execute delegation is not a
substitute for a recovery grant or migration authority.

1. **Contain shared deployment risk now:** document that the present shared
   local root and projection tables are not a proven multi-tenant boundary.
   Do not delete, move or reassign existing data. The user explicitly requires
   preservation of availability for an existing single-user installation.
2. **Finish the inventory before architecture changes:** enumerate all
   ingress/egress, runtime factories, store implementations, direct backend
   operations, identity normalization, session registries, workers, retries,
   caches, backups, exports, replay and lifecycle consumers. Mark inactive,
   test-only and optional components with evidence. Include deployment-specific
   process, filesystem and database trust boundaries.
3. **Design a single identity and ownership contract:** canonical identity
   must not collapse distinct principals; propagate it to every resource and
   action. Define whether admin is tenant-local or global and how principal,
   bot, task and resource ownership are compared. Establish a crash-safe,
   explicit legacy ownership/migration protocol with a compatible single-user
   path; no implicit assignment to an arbitrary newly appearing tenant.
4. **Define acceptance gates first:** real-process two-tenant tests covering
   same session IDs and content, all read/write/delete/clear/export paths,
   SQLite and configured PostgreSQL, concurrent startup/recovery, shutdown,
   authentication/scope failures, worker retries, MCP and cache boundaries.
   Inspect persisted records from fresh processes. No xfail counts as a pass.
5. **Only then implement shared ownership and enforcement**, not separate
   filters at individual callers. Preserve existing mutation safeguards until
   authoritative commits and both projection recovery paths have independent,
   restart-safe metadata, publication fencing and tenant isolation proven.
6. Continue the user's remaining invariant order: authorization, durable
   ledger authority, atomic recovery/projections, lifecycle/concurrency,
   replay, retrieval, integrations, then observability.

## Validation and known baseline

No tests or application modifications were made during this audit turn. A
read-only Python normalization probe confirmed the collision above. Earlier
worktree evidence: 175 passed in a selected runtime/API/egress run; an
additional selection had 123 passed and 3 failed; the kernel boundary gate
reported 7 violations; 4,104 tests were collected. These are historical
selection results, not a full current suite or security certification.

## Remaining inventory needed before calling this complete

- Trace all 279 `dadbot/core` modules reachable from production graph,
  persistence, control plane, tool execution, recovery and replay; classify
  experimental/unreachable paths rather than assuming them inactive.
- Trace all UI/CLI/API aliases, authentication and ticket flows, gateway
  adapters, client token issuance and every direct tool entry point.
- Audit PostgreSQL state, ledger and graph paths with actual cross-tenant
  fixtures; inspect Redis queues, event buses, run stores and result recovery.
- Inventory all on-disk files, backups, profiles, email/calendar, embeddings,
  model caches, logs, traces and diagnostics for ownership and exposure.
- Verify background job identity and permission propagation, event/handoff
  integrity and tenant checks, race behavior, cancellation and shutdown.
- Build a test-to-path matrix and run the relevant negative, real-process and
  concurrency tests. Existing unit tests and the prior failures are not a
  substitute for this matrix.

## Phase 0 closure and Phase 1 status

### Phase 0 closure evidence

- The memory acceptance test is now strict: it has no xfail marker and asserts
  the synthetic tenant-A marker is absent for a fresh tenant B. Exact
  test location:
  [`test_fresh_tenant_must_not_import_another_tenants_local_memory`](tests/unit/test_tenant_isolation_acceptance.py#L10).
  Exact
  validation command: `.\.venv\Scripts\python.exe -m pytest
  --basetemp=C:\Users\josep\.copilot\session-state\61c71cc0-eb4b-4829-8d72-751a75e37727\phase1-preflight-memory
  tests\unit\test_tenant_isolation_acceptance.py -q` — **1 passed**.
- The pulse test creates two tenants with the same `shared-session` and asserts
  distinct `cognition_stream` and `reflection_summary` markers for each. The
  production auth-off test sets `DADBOT_ENVIRONMENT=production` and asserts
  startup raises; the development test sets `DADBOT_ENVIRONMENT=development`
  and asserts startup succeeds
  ([`test_service_api_integration.py:444`](tests/test_service_api_integration.py#L444),
  [`test_service_api_integration.py:481`](tests/test_service_api_integration.py#L481),
  [`test_service_api_integration.py:494`](tests/test_service_api_integration.py#L494)).
  Exact validation command: `.\.venv\Scripts\python.exe -m pytest
  --basetemp=C:\Users\josep\.copilot\session-state\61c71cc0-eb4b-4829-8d72-751a75e37727\phase1-preflight-pulse-auth
  tests\test_service_api_integration.py -k
  "pulse_websocket_does_not_read_another_tenants_session or
  refuses_auth_off_in_production or allows_auth_off_in_development" -q` —
  **3 passed**.
- Phase 0 assumes the shared memory mirror is a legacy single-owner artifact;
  distinct tenants have no implicit shared-mirror entitlement. If sign-off
  authorizes sharing, it must be an explicit ownership grant and the fix must
  be re-reviewed; no first-run/revision-zero fallback is restored.
- The earlier working-tree snapshot was 74 status entries / 31 untracked after
  ignoring pytest temp directories. At the Phase 1 migration-report run it was
  81 entries / 35 untracked, with **0 untracked `.pytest_tmp_*` entries**.
  The ignore rule is `.pytest_tmp_*/` in [`.gitignore`](.gitignore).

### Phase 1 implementation status

- **Decision record (Phase 1 snapshot):** proposed at that time; the later
  signed choices are recorded in
  [`docs/decisions/SIGN-OFF-PACKET.md`](docs/decisions/SIGN-OFF-PACKET.md)
  and [tenant-identity.md](docs/decisions/tenant-identity.md).
  It recommends organization/workspace identity for hosted multi-user
  deployments, but records alternatives and the unanswered owner questions.
- **Contract:** [`tenant_identity.py`](dadbot_system/tenant_identity.py#L33)
  mints versioned opaque IDs (`t1_` plus 26 lowercase base32 characters) and
  rejects non-ASCII, whitespace, path/separator, punctuation-only, uppercase,
  empty, and over-format inputs without rewriting. Exact legacy IDs (including
  `default`) are still syntactically accepted for transition; the validator
  does not prove that an accepted legacy ID has a provisioning record. This
  provenance/default gap remains open and must not be mistaken for completed
  issuance enforcement.
- **Provisioning seam:** [`TenantIdentityProvisioner`](dadbot_system/tenant_identity.py#L21) /
  [`issue_tenant_identity`](dadbot_system/tenant_identity.py#L73) support injected
  provisioning; [`SQLiteDelegationStore.provision_tenant_identity`](dadbot_system/delegation.py#L209)
  records an opaque identity, issuer/subject membership, scopes, and exact
  aliases in one transaction. Legacy membership grants remain available.
  Therefore this is
  not yet the only code path capable of producing a signed token for an
  accepted legacy ID.
- **Token compatibility:** new token claims pass the validator
  ([`ServiceTokenManager.issue`](dadbot_system/security.py#L141));
  verification deliberately preserves signed v1 claims so existing tokens
  remain verifiable until their signed expiration. No fixed cutoff or data
  migration has been authorized; verification alone does not establish a
  legacy claim's current ownership.
- **Dry-run migration report:**
  [`docs/decisions/tenant-identity-migration-report.md`](docs/decisions/tenant-identity-migration-report.md).
  Invocation: `.\.venv\Scripts\python.exe -m tools.tenant_identity_migration
  --dry-run --output docs\decisions\tenant-identity-migration-report.md`.
  It observed 220 tenant-bearing values in 22 legacy-normalized groups and
  found three collision groups: `default` (missing/empty versus literal),
  `family-a` (`family-a` versus `family/a`), and `tenant-a` (`tenant-a`
  versus `tenant/a`). No ID, alias, membership, token, or state was changed.
- **Current normalizer call-site inventory:** all entries below now invoke
  strict validation and preserve exact accepted values; they never lowercase,
  trim, replace separators, or default invalid values. Every source is
  treated as external/unverified unless a provisioning record is checked.
  External forms must be resolved through an exact registered alias before
  calling these paths.

| File:line | Current behavior | New behavior | Owner | Input trust |
|---|---|---|---|---|
| `dadbot_system/contracts.py:420,450,472` | Serialize `WorkerTask`, `WorkerResult`, and `EventEnvelope` tenant fields through the normalizer. | Validate exact ID; raise on invalid/noncanonical value. | Contracts/runtime | Internal object value; issuance provenance not checked. |
| `dadbot_system/client.py:53` | Normalize environment/default tenant while constructing client config. | Validate exact configured ID; no missing/invalid fallback at this call. | Service client | External configuration; resolve before use. |
| `dadbot_system/client.py:120,133,159,277,296,316,352,363,375,425,486` | Normalize method-supplied tenant or configured fallback for HTTP/URL/session methods. | Validate exact value; aliases are not rewritten. | Service client | Method argument/config, not proven issued. |
| `dadbot/app_runtime.py:919,992,1111,1166` | Normalize CLI/runtime tenant during startup and client setup. | Validate exact value; malformed input raises. | App runtime | CLI/environment input; resolve before use. |
| `dadbot/config.py:348` | Normalize tenant during config initialization. | Validate exact value; malformed input raises. | Runtime configuration | Config/environment input; provenance not checked. |
| `dadbot_system/orchestration.py:93,114` | Normalize request tenant in execution graph/metadata. | Validate exact value; no alias rewriting. | Orchestration | Request-derived; API authentication is not alias resolution. |
| `dadbot_system/state.py:633` | Normalize tenant when constructing the state container. | Validate exact value; no alias rewriting. | State/persistence | Caller-provided tenant; issuance provenance not checked. |
| `dadbot/runtime/launcher.py:379,424` | Normalize CLI tenant before environment/client setup. | Validate exact value; malformed input raises. | Runtime launcher | CLI/environment input; resolve before use. |

### Broken-test triage from Phase 1 work

| Test(s) | Triage | Evidence / action |
|---|---|---|
| `tests/unit/test_tenant_identity.py` invalid-value table | Intentional behavior change: invalid inputs must raise and must not be rewritten. | Added coverage for all requested separators, case, whitespace, punctuation-only, empty, Unicode, and `None`. |
| `tests/unit/test_service_identity_binding.py::test_request_deserialization_does_not_normalize_tenant_aliases` and alias-session checks | Intentional behavior change; they assert rejection or mismatch rather than expecting normalization. | These are negative contract tests; keep them. |
| `tests/test_config_drift.py::TestDadBotConfigInstantiation::test_tenant_id_rejects_noncanonical_input` | Intentional behavior change. | Existing lossy-looking `"My Tenant!"` input must raise. |
| First targeted Phase 1 run: three validator error-message mismatches and four `NameError: re is not defined` failures in service identity tests. | Real regression during implementation, fixed. | Restored the `re` import used by `normalize_channel_name`, harmonized the validator error wording, and reran the targeted selection successfully. |
| Follow-up length-bound tests initially found that long legacy IDs and malformed `t1_` IDs could pass the legacy pattern. | Real validator defect, fixed. | Added a 64-character upper bound and reserved the `t1_` prefix for the exact versioned format; both cases now reject. |
| Legacy fixture/config IDs not covered by the targeted selection | Migration-related, not classified as regressions without a failing test and owner mapping. | The dry-run report lists their values/collisions; do not rewrite fixtures or real data by guessing ownership. |

### Phase 1 validation results

- `.\.venv\Scripts\python.exe -m pytest --basetemp=<session-temp>\phase1-identity-final3 tests\unit\test_tenant_identity.py tests\test_config_drift.py tests\unit\test_service_identity_binding.py tests\test_api_models_contract.py -q` — **58 passed**. This includes canonical input/length tests, issuance, alias collision, token compatibility, and normalizer behavior.
- `.\.venv\Scripts\python.exe -m pytest --basetemp=<session-temp>\phase1-preflight-memory-final2 tests\unit\test_tenant_isolation_acceptance.py -q` — **1 passed**.
- `.\.venv\Scripts\python.exe -m pytest --basetemp=<session-temp>\phase1-preflight-pulse-auth-final tests\test_service_api_integration.py -k "pulse_websocket_does_not_read_another_tenants_session or refuses_auth_off_in_production or allows_auth_off_in_development" -q` — **3 passed**.
- Ruff `check` on the touched tenant-identity, call-site, migration, and test files with `--select F401,F821,I001` — **all checks passed**.
- `python -m compileall -q dadbot dadbot_system` plus the migration tool and affected tests — **passed**.
- `git diff --check` — **passed** (Git emitted only a line-ending notice for the pre-existing working-tree change in `tests/test_memory_manager.py`).
- Migration command `.\.venv\Scripts\python.exe -m tools.tenant_identity_migration --dry-run --output docs\decisions\tenant-identity-migration-report.md` — **completed**; report output is at the linked path above.
- No full test suite was run.

### Carried-forward Phase 4A test triage

These tracked test files predate the tenant remediation branch and are part
of the carried-forward WIP commit set:

| Test file | Current evidence / failure mode | Owning triage |
|---|---|---|
| [Phase 4A tests](tests/test_phase4a.py) | The historical legacy copy included an undefined `required_fields` reference, a mock trace assertion failure, a real-turn checkpoint failure, and fixture-scope errors. Its shared assertions have since been mapped to the canonical suite and its unique checks preserved there; the redundant copy is removed. No `NameError` from `conluence_key_or_turn` and no failure attributed to Phase 1.2/1.5 identity behavior appeared in the original maxfail-50 stage; those interactions remained unisolated. | Phase 4 for checkpoint/durability assertions; carried WIP for fixture/setup issues. Storage failures belong to Phase 3. |
| [test_phase4a.py](tests/test_phase4a.py) | The maxfail-50 suite stage reported 16 failures in checkpoint, determinism, scale/pruning, and capability-profile assertions, plus seven setup errors (including missing `self` fixture on module-level tests). No current failure was attributed to Phase 1.2 validator or Phase 1.5 normalizer behavior; that possibility was not separately isolated. | Phase 4 for durability/checkpoint behavior; carried WIP for malformed fixtures/tests. Storage failures belong to Phase 3 only when confirmed storage-related. |
| [test_phase4a_adversarial_collapse.py](tests/test_phase4a_adversarial_collapse.py) | Present in base history and modified in the carried WIP set. The maxfail-50 stage did not report a failure from this file; it does not isolate a possible Phase 1.2/1.5 interaction. | Phase 4 for durability; Phase 3 for confirmed storage failures; otherwise carried WIP. |

These files are classified as carried-forward WIP rather than remediation
tests because they predate this branch and contain unrelated Phase 4 work.
No Phase 4A test is fixed or deleted in this change.

### Remaining risks and items not done

- At the time this Phase 1 snapshot was written, deficits 9/18/19 were
  unmapped. The current crosswalk below assigns the requested names:
  **9 identity normalization, 18 tenant definition, and 19 legacy alias
  migration**. Remaining risks are that provisioning provenance is not
  enforced by every normalizer caller and existing `default`/legacy IDs
  remain unmigrated.
- **Not done in this change:** tenant-context middleware (Phase 2); tenant
  filters or store-boundary refactoring (Phase 3); PostgreSQL/RLS; wiring
  `SQLiteOwnedTasks` into orchestration; semantic/graph/session/checkpoint/
  idempotency scoping; durable projection recovery/retention; production
  environment or Compose changes; full-suite testing; signing the decision
  record; or mutating real state.

## Phase 1 close-out and non-blocking preparation

This section supersedes earlier worktree counts and validation snapshots
above. It records the current close-out; the entire working tree remains
uncommitted and contains unrelated pre-existing changes.

### Enforceable now vs. compatibility window

**Enforceable now**

- `normalize_tenant_id` is now a fail-closed validator: it preserves exact
  valid input and raises for malformed/noncanonical input. The canonical
  issued form is opaque, versioned, ASCII, and the parser itself never maps
  missing/invalid input to `default`. Existing model/config/runtime
  constructors still inject `default` explicitly; that behavior is in the
  compatibility window, not globally retired.
- Provisioning can mint a random opaque identifier and atomically record the
  issuer/subject membership and exact aliases through the injected
  `TenantIdentityProvisioner` seam.
- External aliases require exact issuer-scoped lookup; there is no lossy
  rewrite, and the Phase 0 shared-memory fallback remains fail-closed.
- The legacy acceptance logger records exact ID, issuer, subject, Unix
  timestamp, and acceptance path. Missing actor context is recorded as
  `unbound`, which is not ownership evidence.

**Still in the compatibility window**

- Exact legacy canonical IDs, including literal `default`, are still accepted
  by identity validation; syntactic acceptance does not prove a provisioning
  record or membership.
- Signed v1 tokens retain their signed tenant claim through token expiry.
  Token verification therefore preserves compatibility and is not the
  provenance enforcement boundary.
- A few old call paths have no bound actor and emit `unbound`; these events
  must be reconciled before cutover.

### Legacy-ID and `default` acceptance seams

| Seam | Current behavior and evidence |
|---|---|
| Validator/normalizer | [`tenant_identity.py:67-86`](dadbot_system/tenant_identity.py#L67) rejects malformed and noncanonical values, but accepts exact `default` and exact legacy IDs matching the restricted legacy grammar and logs each acceptance. [`contracts.py:44`](dadbot_system/contracts.py#L44) keeps the historical `normalize_tenant_id` name but delegates to this validator. |
| Token issuance and verification | [`security.py:158-164`](dadbot_system/security.py#L158) permits syntactically valid legacy IDs in new signed tokens and logs issuer/subject. [`security.py:207-218`](dadbot_system/security.py#L207) deliberately verifies signed v1 legacy claims until expiry and logs them; its `or "default"` field fallback remains a separate defaulting seam. |
| Membership and alias operations | [`delegation.py:195-200`](dadbot_system/delegation.py#L195), [`delegation.py:225-230`](dadbot_system/delegation.py#L225), and [`delegation.py:289-297`](dadbot_system/delegation.py#L289) accept/log syntactically valid legacy IDs for grant, provisioning, and revocation. [`delegation.py:280-287`](dadbot_system/delegation.py#L280) validates resolved alias targets; [`delegation.py:309-319`](dadbot_system/delegation.py#L309) accepts task IDs matching the signed principal during delegation issuance. Membership storage remains keyed by exact issuer/subject/tenant; no issuance-provenance check is universal. |
| Auth-off principal | [`api.py:215-219`](dadbot_system/api.py#L215) explicitly constructs anonymous `default` with privileged local scopes when auth is off; [`api.py:111`](dadbot_system/api.py#L111) blocks this in production. [`api.py:252-258`](dadbot_system/api.py#L252) validates the resolved tenant and logs this path with auth-off actor context. |
| Explicit `default` constructors and fallbacks | `default` is still supplied by [`contracts.py:14,192,230,407,437,463`](dadbot_system/contracts.py#L14), [`client.py:53-54,160`](dadbot_system/client.py#L53), [`config.py:347`](dadbot/config.py#L347), [`models.py:322,341`](dadbot/models.py#L322), [`state.py:627`](dadbot_system/state.py#L627), [`orchestration.py:65,390`](dadbot_system/orchestration.py#L65), [`worker.py:111,208,391,414`](dadbot_system/worker.py#L111), [`api_models.py:223`](dadbot/api_models.py#L223), and additional turn/persistence helpers in `dadbot/core` and `dadbot/services`. These are explicit legacy defaults, not parser normalization, and require migration or an approved compatibility policy. |
| Shared-memory mirror entitlement | [`local_tenant_establishment.py:82`](dadbot/core/local_tenant_establishment.py#L82) treats a shared JSON mirror as insufficient evidence to establish another tenant. A missing mirror produces empty state; no legacy ID or `default` value independently grants mirror access. |
| Tests that assume `default` | Existing references include `tests/test_kernel_freeze_phase1.py:23`, `tests/test_models.py:208`, `tests/test_service_architecture.py:29,364`, `tests/test_service_client.py:95,101,190`, `tests/test_service_websocket_tickets.py:137`, and `tests/unit/test_service_verified_principal.py:91`. Identity validator coverage is in `tests/unit/test_tenant_identity.py`. These must be updated or explicitly retained according to the signed option. |

Decision records remain unsigned: [tenant semantics](docs/decisions/tenant-identity.md),
[collision choices](docs/decisions/tenant-identity-collisions.md), and
[compatibility window](docs/decisions/tenant-identity-compatibility.md).
The migration inventory is a dry run only at
[tenant-identity-migration-report.md](docs/decisions/tenant-identity-migration-report.md);
220 observations, 22 normalized groups, three collision groups, and no
state changes.

### Signed tenant-identity decisions

Items 1–6 of the [sign-off packet](docs/decisions/SIGN-OFF-PACKET.md) were
signed on **2026-11-09**: split each collision group; retain `default` only
for compatibility outside production behind
`DADBOT_ALLOW_LEGACY_DEFAULT_TENANT`; assign the compatibility window to
Platform Lead through **2027-02-07**; and acknowledge the conditional Phase 0
mirror assumption. These policy choices do not establish which real-world
owner holds any scanned state. No membership, token, alias, or tenant data
has been migrated, and there is no implicit mirror entitlement.

For this implementation stage, recovery tenant enumeration is authorized
from tenant IDs on persisted `SQLiteOwnedTasks` rows only; it is not a
general tenant catalog. Legacy unqualified tasks remain quarantined. RLS
scope is limited to Phase 3.2 task/response/incomplete-task storage, verified
with a local `NOBYPASSRLS` role; it does not reopen authenticated PostgreSQL
boot or alter Phase 3.3–3.9 stores.

### D1-D3 evidence

- **D1, deficit 6 — import boundary gate (historical baseline before Phase
  2.3):** Import Linter config initially reported seven expected violations:
  kernel/observability reports
  `dadbot.core.otel_bridge -> dadbot.core.trace_export` at lines 86 and 106,
  and `dadbot_system.runtime_signals -> dadbot_system.telemetry` at lines 72
  and 78; gateway/observability reports
  `dadbot_system.api -> dadbot_system.telemetry` at line 351. The kernel DAG
  contract repeats the two `otel_bridge` paths, for seven reported
  violations in three broken contracts. All seven have since been fixed;
  current gate results are recorded below.
- **D2, deficit 23 — test F821:** the 49 undefined-name diagnostics were
  fixed in `tests/persistence_test.py`,
  `tests/test_determinism_capability_receipts.py`,
  `tests/test_durable_execution.py`,
  `tests/test_phase4a_adversarial_collapse.py`, and
  `tests/test_phase4a_fixed.py`. `ruff check tests --select F821` reports
  **All checks passed** (zero remaining). These were real test-code defects,
  not intentional behavior changes; no tests were deleted at that stage.
  Later Phase 4A consolidation removed the redundant copy after mapping its
  assertions and preserving its unique checks in the canonical suite.
- **D3, deficit 24 — contract anchors:** replaced the two stale anchors with
  current test IDs, then ran `tools/contract_test_compiler.py
  --validate-nodeids`: 20 contracts, 23 node IDs, zero untested contracts,
  exit 0. `--check --validate-nodeids` reports all node IDs collectable,
  exit 0, and a SHA-256 comparison confirmed the ignored manifest was not
  rewritten. A focused regression test covers no-rewrite behavior both with
  and without `--run-tests`.

### Test-failure triage

| Test(s) | Classification | Evidence / action |
|---|---|---|
| 49 F821 diagnostics in the five D2 files listed above | Real test-code defects; fixed. | Undefined test helpers/fixtures (including misspelled helper names and required-field variables) were restored; no tests were removed. The current F821 gate is zero. |
| `test_task_namespace_isolated_across_processes`; `test_identical_task_ids_do_not_overwrite_another_namespace`; `test_task_result_and_response_are_namespace_owned`; `test_incomplete_task_listing_filters_before_limit`; `test_legacy_task_is_quarantined_without_implicit_ownership`; `test_raw_store_rejects_unqualified_task_lookup`; `test_fresh_process_recovers_same_task_id_for_both_owners` in `tests/unit/test_sqlite_service_state.py` | Historical Phase 3 storage-boundary gaps, not regressions from identity logging. | These seven failures were exposed by the earlier focused authority selection. They now pass after Stage 1 task/response/incomplete-task scoping; see the current verification section below. |
| Focused identity/compiler/delegation/service-binding selection, excluding `test_sqlite_service_state.py` | Passed. | **58 passed** after the legacy logging and compiler no-rewrite changes. |

The source audit does not define or number deficits **9, 18, or 19**; their
status is therefore **unmapped / not safely attributable**, not closed.
While the compatibility window remains open, the material risks associated
with those references are still legacy-ID provenance, the ownership meaning
of `default`, and the unresolved tenant-semantic/collision decisions.

### Historical worktree baseline and branch/commit recommendation

The following status was captured before the later approved branch execution;
the follow-up execution record below is authoritative for current branch state.

Current `git status --short`: **109 entries, 47 untracked**; **0 untracked
`.pytest_tmp_*` entries**. `.gitignore` contains `.pytest_tmp_*/`. No branch
was created and nothing was staged or committed.

Given the mixed 109-entry worktree, recommendation (not executed): first
create a dedicated branch after reviewing ownership of all existing changes,
then split the ship set into (1) Phase 0 isolation/auth fixes and Phase 1
identity/provisioning/logging implementation, (2) focused acceptance and
identity regression tests, (3) proposed decision/audit documents, and
(4) import-linter, F821, and contract-compiler tooling. Do not combine the
unsigned migration decisions with an implicit data migration.

### Historical non-goals at the earlier Phase 1 / Phase 2.3 snapshot

- No human sign-off or adoption of organization/user/deployment semantics,
  collision mappings, `default` treatment, compatibility owner, or expiry.
- No tenant-context middleware (Phase 2.1/2.2), per-query tenant filters,
  RLS, or broader store refactor (Phase 3).
- No provenance enforcement or retirement of legacy IDs; no real-state
  migration, merge, split, or token rewrite.
- No production environment or Compose changes (Phase 5).
- No Phase 2.1/2.2 TenantContext middleware wiring; only the unwired interface
  contract is defined.
- No full-suite run; the recursive scanner remains excluded.
- No branch creation, staging, or commit.

## Sign-off, traceability, and Phase 3 readiness (current)

This section supersedes earlier status prose where it conflicts with the
signed packet and current execution evidence.

### Audit finding to numbered remediation mapping

| Audit finding | Numbered remediation deficit / item | Status | Evidence |
|---|---|---|---|
| Finding 1 — shared memory mirror could establish another tenant's state | Phase 0 item 1 | **Closed for the verified local mirror path**; conditional assumption remains. | [strict process-isolation acceptance test](tests/unit/test_tenant_isolation_acceptance.py#L10); conditional assumption below. |
| Finding 2 — shared semantic/graph indexes and stores lack tenant ownership | Phase 3 items 7–8; task/response scoping is Phase 3.2 | **Blocked / out of scope here.** | [Phase 3 entry criteria](docs/decisions/phase3-entry-criteria.md); seven failing state-store node IDs below. |
| Finding 3 — pulse session lookup was not tenant-scoped | Phase 0 item 2 | **Closed for the pulse route.** | [same-session two-tenant pulse test](tests/test_service_api_integration.py#L444). |
| Finding 4 — lossy tenant ID normalizer | **Deficit 9 — identity normalization** | **Partial:** strict validator and opaque issuance exist; split policy is signed, but owner-backed mappings and legacy provenance retirement remain open. | [tenant identity validator](dadbot_system/tenant_identity.py#L67); [collision inventory](docs/decisions/tenant-identity-migration-report.md). |
| Finding 5 — worker factory may drop tenant identity on `TypeError` | Phase 1 item 5 / verified execution-context boundary | **Partial:** HTTP/WebSocket tenant binding and task-store handoff are implemented; the custom worker-factory `TypeError` fallback remains a separate risk. | [worker identity binding tests](tests/unit/test_service_identity_binding.py); [TenantContext implementation](dadbot_system/tenant_context.py#L15). |
| Finding 6 — singleton runtime-bot resolution in spatial/automation paths | Phase 1 item 5; Phase 2.1/2.2 ingress-to-runtime binding | **Partially closed:** request/socket context is bound and scoped; broader runtime-bot operations beyond the Stage 1 route coverage remain in the execution-path inventory. | [API spatial/admin path inventory](#execution-path-inventory); Stage 1 validation below. |
| Finding 7 — local MCP and generic tool helpers have different trust models | Phase 3 item 8 for tenant-scoped tool/idempotency state; execution-context binding item 5 | **Open / incomplete discovery.** | [MCP/tool inventory](#execution-path-inventory); broader tool entry-point tracing remains incomplete. |
| Tenant meaning and ownership were not defined | **Deficit 18 — tenant definition** | **Policy signed; actual owner mapping remains unverified.** | [signed packet](docs/decisions/SIGN-OFF-PACKET.md); [tenant semantics](docs/decisions/tenant-identity.md). |
| Legacy aliases and collision migration lack owner-backed mapping | **Deficit 19 — legacy alias migration** | **Split policy signed; migration remains blocked on owner evidence.** | [collision decisions](docs/decisions/tenant-identity-collisions.md); [migration report](docs/decisions/tenant-identity-migration-report.md). |
| Kernel imported service telemetry and optional exporters | **Deficit 6 — kernel boundary gate** | **Closed for the seven reported import violations; gate now passes.** | [kernel boundary check](ci/kernel_boundary_check.py); configured Import Linter contracts in [pyproject.toml](pyproject.toml#L145). |
| Undefined names in tests | **Deficit 23 — test F821** | **Closed:** zero test F821 diagnostics. | `ruff check tests --select F821` — All checks passed. |
| Stale contract-test node IDs / check-mode manifest rewrite | **Deficit 24 — contract anchors** | **Closed:** IDs validate and check mode leaves the manifest unchanged. | `tools/contract_test_compiler.py --check --validate-nodeids`; manifest SHA-256 unchanged. |

The available architecture specification does not contain a separate
numbered 1–24 deficit catalog. This crosswalk uses its numbered remediation
items where present and explicitly records the requested numbered deficits
6, 9, 18, 19, 23, and 24; it does not invent labels for unspecified IDs.

### Conditional assumptions

Phase 0 mirror fix assumes the shared mirror is a legacy single-owner
artifact; if the tenant semantics decision selects organization-shared
mirrors, Phase 0.1 must be revisited. This exact assumption is sign-off
packet Item 6. Revisit does not mean restoring first-run fallback: any
sharing must use explicit ownership evidence and an approved grant.

### Phase 0 pre-flight evidence (current rerun)

- `.\.venv\Scripts\python.exe -m pytest --basetemp=<session-temp>\signoff-preflight-memory tests\unit\test_tenant_isolation_acceptance.py -q` — **1 passed**, without `--runxfail`.
- `.\.venv\Scripts\python.exe -m pytest --basetemp=<session-temp>\signoff-preflight-api tests\test_service_api_integration.py -k "pulse_websocket_does_not_read_another_tenants_session or refuses_auth_off_in_production or allows_auth_off_in_development" -q` — **3 passed**, without `--runxfail`.
- The pulse test creates two tenants with identical `shared-session` IDs and checks tenant-specific cognition and reflection fields. Production auth-off raises; development auth-off constructs the app.

### D1-D3 current status

- **D1 / deficit 6:** baseline had seven violations (API to telemetry; runtime
  signals to telemetry at two sites; kernel OTel bridge to trace exporter at
  two sites, duplicated in the dependency-DAG report). Phase 2.3 moved the
  optional `otel_bridge` and `trace_export` to `dadbot/telemetry`, routed
  service tracing through an injected runtime-signals adapter, and removed
  direct observability imports from kernel/API modules. Current
  `python ci/kernel_boundary_check.py` exits **0**; `lint-imports --no-cache`
  reports **3 kept, 0 broken**. The seven source violations are fixed.
- **D2 / deficit 23:** `ruff check tests --select F821` — **0 diagnostics**.
- **D3 / deficit 24:** `--validate-nodeids` succeeded (20 contracts, 23
  anchors, zero untested); `--check --validate-nodeids` succeeded and
  manifest SHA-256 remained unchanged.

### Phase 3 entry failures — historical baseline, current gate passes

The pre-Stage-1 `tests/unit/test_sqlite_service_state.py` run reproduced
these seven failures; their historical failure modes, ownership and
Phase 1.5 impact are detailed in
[phase3-entry-criteria.md](docs/decisions/phase3-entry-criteria.md):

| Node ID suffix | Owning work |
|---|---|
| `test_task_namespace_isolated_across_processes` | 3.2 — tenant/task ownership |
| `test_identical_task_ids_do_not_overwrite_another_namespace` | 3.2 — composite task key |
| `test_task_result_and_response_are_namespace_owned` | 3.2 — response/result scoping |
| `test_incomplete_task_listing_filters_before_limit` | 3.3 — tenant-filtered listing |
| `test_legacy_task_is_quarantined_without_implicit_ownership` | 3.3 — recovery quarantine; legacy syntax acceptance is not ownership |
| `test_raw_store_rejects_unqualified_task_lookup` | 3.2 — raw API removal / qualified lookup |
| `test_fresh_process_recovers_same_task_id_for_both_owners` | 3.3, dependent on 3.2 — tenant-aware recovery |

After the Stage 1 task/response/incomplete-task changes, all seven nodes pass
unmarked. The updated verification is in
[phase3-entry-criteria.md](docs/decisions/phase3-entry-criteria.md). This
closes the named task-state exit gate, not the remaining Phase 3.3–3.9 stores.

The historical exact node-ID rerun on 2026-11-09 used a session-local `--basetemp` after the
default Windows pytest temp location returned `PermissionError`; all seven
executed and failed, with no setup errors. No Phase 0/1 change was identified
in the tested storage/recovery paths; these remain the recorded Phase 3
baseline. Phase 0 covers local mirror ownership, pulse lookup and auth-off;
Phase 1 covers identity validation/provisioning, not task-store query scope.

| Exact node ID | Result | Observed failure |
|---|---|---|
| `tests/unit/test_sqlite_service_state.py::test_task_namespace_isolated_across_processes` | Historical **FAIL**; current **PASS** | Composite task ownership is preserved across processes. |
| `tests/unit/test_sqlite_service_state.py::test_identical_task_ids_do_not_overwrite_another_namespace` | Historical **FAIL**; current **PASS** | Identical task IDs coexist under tenant-qualified keys. |
| `tests/unit/test_sqlite_service_state.py::test_task_result_and_response_are_namespace_owned` | Historical **FAIL**; current **PASS** | Response/result reads and writes use the tenant-qualified task key. |
| `tests/unit/test_sqlite_service_state.py::test_incomplete_task_listing_filters_before_limit` | Historical **FAIL**; current **PASS** | Incomplete listing scopes before applying the limit. |
| `tests/unit/test_sqlite_service_state.py::test_legacy_task_is_quarantined_without_implicit_ownership` | Historical **FAIL**; current **PASS** | Unqualified legacy rows are rejected from recovery. |
| `tests/unit/test_sqlite_service_state.py::test_raw_store_rejects_unqualified_task_lookup` | Historical **FAIL**; current **PASS** | Unqualified application task lookup is rejected. |
| `tests/unit/test_sqlite_service_state.py::test_fresh_process_recovers_same_task_id_for_both_owners` | Historical **FAIL**; current **PASS** | Fresh-process recovery discovers both tenant-owned tasks. |

The current run also passes
`tests/unit/test_owned_sqlite_tasks.py::test_concurrent_tenant_writes_survive_fresh_construction`,
which writes the same task and session IDs from separate processes and verifies
tenant-specific task lists and responses after fresh construction. Memory
recall in matrix case 1 and memory clear/export/delete in case 3 are explicitly
deferred until the tenant-scoped semantic/graph work in Phase 3.5/3.6; see
the Stage 1 record below.

### Signed decisions and remaining implementation boundary

All six items in [SIGN-OFF-PACKET.md](docs/decisions/SIGN-OFF-PACKET.md) are
signed: the three collision groups are split; `default` is reserved for
compatibility-only use, prohibited in production, and gated in dev/local;
the compatibility owner is Platform Lead with expiry 2027-02-07; and the
Phase 0 conditional assumption is acknowledged. No owner mapping, membership,
token, alias, or mirror is created by the policy choice alone.

### Tool-action explanation endpoint

The authenticated `GET /v1/sessions/{session_id}/actions/explain` endpoint
returns a bounded explanation of tool actions in the latest turn. The worker
copies at most 50 action summaries into its session-state result before
signing it. Each summary is projected only after validating the event
checksum chain and SHA-256 input/output fingerprints. Explanations include
the tool, status, authorization decision and authority mode, trace/timestamp,
and event-chain references; raw arguments and results are excluded. The route
uses the authenticated tenant's session snapshot and read authorization.

Focused verification command:
`py -3 -m pytest --basetemp=.pytest_tmp_explain_final -q tests/test_action_explanation.py tests/test_execute_tool_recording.py tests/test_service_api_integration.py tests/test_service_architecture.py tests/test_e2e_orchestrator_turn.py::test_e2e_orchestrator_full_runtime_lane_certifying_expected_tool_execution`
— **59 passed, 0 failed, 0 skipped**. This includes the tenant-scoped
`test_explain_session_actions_is_tenant_scoped_and_uses_persisted_audit_records`,
the `test_worker_result_persists_bounded_tool_action_audit` handoff regression,
event checksum-projection tests, and the reminder tool end-to-end test.
`compileall`, focused Ruff `--select F`, and touched-file `git diff --check`
also passed. This does not provide a cross-turn history, reconstruct arbitrary
conversation context, or establish an independent proof beyond the checksum
and authority evidence captured in the event record.

The branch and commit split are recorded in
[branch-commit-plan.md](docs/decisions/branch-commit-plan.md); they were
executed locally after approval, with no push or merge.

### Current implementation validation

- Observability adapter tests: **16 passed** after moving the tracing bridge.
- TenantContext tests: **4 passed**, including fail-closed lookup and
  isolation across concurrent async tasks.
- Service identity/delegation tests: **32 passed**.
- Phase 0 strict memory acceptance: **1 passed**; pulse and auth-off
  selection: **3 passed**, all without `--runxfail`.
- `compileall` and `git diff --check` completed successfully.
- All-rule Ruff on the changed Python files reports **38 diagnostics**,
  predominantly complexity, line-length, and broad-exception findings in
  touched files. No sweeping style changes were made. The narrowed
  `--select E4,E7,E9,F,I` gate passes, as does `ruff check tests --select
  F821`.
- At the time this validation snapshot was written, no full test suite had
  been run. See the follow-up execution record below for the staged suite
  result and current changes.

## Follow-up execution record

### Part A — approved local commit split

The approved seven-set split was completed on branch
`tenant-identity-remediation`: the six tenant-remediation sets and one
separate carried-forward pre-existing WIP set are recorded with full commit
SHAs and file lists in
[branch-commit-plan.md](docs/decisions/branch-commit-plan.md). The later
`ef28cc66` and `6616ed01` commits also carry pre-existing durability/config
WIP with explicit provenance bodies; they are separate from tenant identity
remediation and Stage 1. No push or merge was performed.
- **Phase 7: closed** (per the current remediation status); the seven-set
  local commit record is in [branch-commit-plan.md](docs/decisions/branch-commit-plan.md).

### Part B — independent follow-up work

- **B1 / PostgreSQL authenticated boot:** completed. Authenticated boot with
  the PostgreSQL execution-ledger backend now fails with a direct error and
  a pointer to [postgres-authenticated-boot.md](docs/decisions/postgres-authenticated-boot.md).
  PostgreSQL state/ledger primitives remain documented as experimental
  trusted-local components; this does not select tenant or `default`
  semantics.
- **B2 / deficit 10 — safe log retention:** **deferred, not implemented.**
  `EventCompactor.compact()` currently refuses trimming with
  `durable_checkpoint_required` ([compaction.py](dadbot/core/compaction.py#L144)).
  `DurableCheckpoint` keeps its history only in process memory
  ([durable_checkpoint.py](dadbot/core/durable_checkpoint.py#L17)), while
  `SnapshotEngine` keeps snapshots in process and the ledger WAL verifies a
  chain from its beginning. There is no durable all-consumer cursor registry,
  durable replay-complete state snapshot, chain-preserving segment/anchor
  format, or integrated fall-off-log rebuild path. Trimming now would risk
  losing replay-critical events or invalidating chain verification. No
  unsafe trim, partial cursor mechanism, or test-only success path was added.
  The explicit Phase 4 deferral, owner role, substrate, and required tests are
  recorded in
  [checkpoint-trimming-deferral.md](docs/decisions/checkpoint-trimming-deferral.md).
- **B3 / deficit 11 — projection status:** mechanism and focused tests
  completed. Per-projection generation/status events are written to the
  durable WAL and reloaded on restart; graph and semantic synchronization
  remain independent. The focused restart/degradation and independent-sync
  tests passed. No physical semantic/graph index isolation was selected.
- **B4 / deficits 7 and 25 — production config:** completed for required
  config completeness. Both production env files contain
  `DADBOT_AUTHORITY_SQLITE_PATH`, `DADBOT_DURABLE_EXECUTION_LEDGER`, and
  `DADBOT_EXECUTION_LEDGER_BACKEND`; the local ignored file's values are not
  committed or disclosed. Compose now requires `POSTGRES_PASSWORD`, CI
  verifies Compose rejects an unset password and accepts a validation value,
  and `ci/production_env_check.py` checks the three required example keys.
  No auth-off or `default` policy was changed.

### Part C — Phase 3 inventory

The nine-store inventory is in
[phase3-scope-inventory.md](docs/decisions/phase3-scope-inventory.md). It
records current keys/constraints/query patterns, the required scoped change,
proposed Phase 3.2–3.9 ownership, `default` and legacy-task dependencies, and
acceptance tests. It is an inventory only; no store schema/query/filter was
changed.

### Part D — historical scanner-excluded suite baseline

The counts and first-30 disposition table below are the pre-Stage-1 suite
baseline. The current bounded suite rerun and complete Stage 1 disposition
are recorded at the end of this audit.

The recursive audit scanner excluded per the prior audit is
`tests/invariants/test_full_system_audit_matrix.py`; that file invokes
`run_full_system_audit`, whose import-graph check recursively scans the
repository. The self-invoking wrapper
`tests/test_phase4a_patched.py` is also excluded because importing it calls
`pytest.main(["tests/test_phase4a.py"])` at module scope and recursively
starts a nested test run; the canonical `tests/test_phase4a.py` is otherwise
still collected.

The 300-turn `test_soak_simulation_replay_loop` was previously observed to
consume about 1.78 GB and stall during ledger compaction. It is now explicitly
quarantined with a pytest skip marker; its assertions are unchanged. The
owner is the **durability/ledger maintainers** role. Re-enable it only after
the 300-turn replay case completes in under 120 seconds and under 1 GiB peak
memory in the designated soak environment, with ledger integrity assertions
still passing. The default suite records it as skipped rather than silently
omitting it; this does not assert the underlying performance issue is fixed.

The corrected collection command, with the two exclusions above, reported
**4,245 collected tests**. The resource-bounded run used
`--maxfail=500 --timeout=120`, reached natural completion (141 failures is
below the maxfail threshold), and advanced beyond the previous 44% soak-test
stall. JUnit recorded **4,246 result cases**: **3,972 passed, 141 failed,
11 errored, and 122 skipped**. The one-case difference between collection
output and JUnit records is unexplained and is retained here rather than
silently normalizing the counts. The suite completed but is not green; this
is the new baseline.

Collection command:
`.\.venv\Scripts\python.exe -m pytest --collect-only -q -p no:randomly --ignore=tests/invariants/test_full_system_audit_matrix.py --ignore=tests/test_phase4a_patched.py`.
Execution command:
`.\.venv\Scripts\python.exe -m pytest -q -p no:randomly -o faulthandler_timeout=0 --timeout=120 --tb=short --maxfail=500 --junitxml=<session-file> --ignore=tests/invariants/test_full_system_audit_matrix.py --ignore=tests/test_phase4a_patched.py --basetemp=<session-temp>`.
JUnit and captured console output were kept outside the repository. The seven
known `test_sqlite_service_state.py` failures all reproduced and are excluded
from the first-30 list below.

| # | Node ID | Triage |
|---|---|---|
| 1 | `tests/distribution_readiness/tier1_kernel/test_contract_enforcement.py::TestNodeStageContracts::test_execute_stamps_contract_version_in_determinism_manifest` | Not remediation-related; SaveNode stage invariant fails. |
| 2 | `tests/phase4/test_phase4_compliance.py::test_no_unseeded_random_imports_in_core` | Not remediation-related; scanner reports `import random` in carried-forward `dadbot/core/fault_injection.py`. |
| 3 | `tests/test_agentic_local_actions.py::test_tool_registry_parses_calendar_and_email_commands` | Not remediation-related; `DadBot.parse_tool_command` is unavailable. |
| 4 | `tests/test_agentic_methods.py::test_plan_and_act_basic` | Not remediation-related; `DadBot.plan_and_act` is unavailable. |
| 5 | `tests/test_agentic_methods.py::test_background_job_runs` | Not remediation-related; `DadBot.schedule_background_job` is unavailable. |
| 6 | `tests/test_agentic_methods.py::test_goal_and_progress` | Not remediation-related; `DadBot.set_goal` is unavailable. |
| 7 | `tests/test_agentic_methods.py::test_reflect_and_learn` | Not remediation-related; `DadBot.reflect_and_learn` is unavailable. |
| 8 | `tests/test_agentic_methods.py::test_notify_user` | Not remediation-related; `DadBot.notify_user` is unavailable. |
| 9 | `tests/test_app_runtime_startup.py::test_production_state_stores_require_postgres` | Not remediation-related; test `SimpleNamespace` fixture lacks `persistence.sqlite_path`. |
| 10 | `tests/test_app_runtime_startup.py::test_development_state_store_keeps_in_memory_fallback` | Not remediation-related; same incomplete `persistence.sqlite_path` fixture. |
| 11 | `tests/test_dadbot_regression.py::TestDadBotRegression::test_prepare_user_turn_sets_daily_checkin_context_once` | Phase 4; strict SaveNode commit fails because the determinism lock hash is missing. |
| 12 | `tests/test_durability_and_leases.py::TestLeaseSchedulerIntegration::test_single_worker_acquires_and_releases_lease_per_job` | Phase 4; lease turn fails the canonical response candidate gate. |
| 13 | `tests/test_durability_and_leases.py::TestLeaseSchedulerIntegration::test_lease_conflict_requeues_and_eventually_executes` | Phase 4; lease turn fails the canonical response candidate gate. |
| 14 | `tests/test_long_term_signals.py::test_should_evolve_persona_respects_session_gap` | Not remediation-related; persona evolution expectation mismatch. |
| 15 | `tests/test_long_term_signals.py::test_update_trait_impact_from_relationship_feedback_reinforces_latest_trait` | Not remediation-related; long-term signal behavior assertion. |
| 16 | `tests/test_long_term_signals.py::test_reject_persona_trait_removes_latest_entry` | Not remediation-related; long-term signal behavior assertion. |
| 17 | `tests/test_long_term_signals.py::test_generate_wisdom_insight_uses_mocked_ollama_and_deduplicates_recent_entries` | Not remediation-related; wisdom insight behavior assertion. |
| 18 | `tests/test_long_term_signals.py::test_detect_life_patterns_skips_existing_pattern_identity` | Not remediation-related; life-pattern behavior assertion. |
| 19 | `tests/test_long_term_signals.py::test_detect_life_patterns_queues_proactive_message_for_new_pattern` | Not remediation-related; life-pattern behavior assertion. |
| 20 | `tests/test_loyal_elephie_integration.py::TestHybridRetriever::test_rrf_fusion` | Not remediation-related; test expects a mapping but retriever returns a tuple. |
| 21 | `tests/test_memory_manager.py::test_forgetting_uses_distinct_backups_for_same_turn_and_preserves_retained_memories` | Phase 4; backup write fails because the backup temp file's parent path is absent. |
| 22 | `tests/test_memory_manager.py::test_forgotten_memory_is_restorable_after_store_reload` | Phase 4; backup write fails because the backup temp file's parent path is absent. |
| 23 | `tests/test_memory_manager.py::test_manual_forgetting_archives_and_restores_without_turn_context` | Phase 4; backup write fails because the backup temp file's parent path is absent. |
| 24 | `tests/test_memory_manager.py::test_archived_memory_can_be_deleted_without_changing_active_memory` | Phase 4; backup write fails because the backup temp file's parent path is absent. |
| 25 | `tests/test_memory_manager.py::test_forgetting_rejects_stale_catalog_without_losing_new_memory` | Phase 4; backup write fails because the backup temp file's parent path is absent. |
| 26 | `tests/test_memory_manager.py::test_archived_memory_restore_failure_preserves_archive` | Phase 4; backup write fails because the backup temp file's parent path is absent. |
| 27 | `tests/test_memory_manager.py::test_forgetting_preserves_distinct_kept_memories_with_same_dedup_key` | Phase 4; backup write fails because the backup temp file's parent path is absent. |
| 28 | `tests/test_phase1_response_authority.py::TestPhase1ResponseAuthority::test_contract_single_callable_output_authority` | Not remediation-related; the test's `SimpleNamespace` fixture lacks `LIGHT_MODE`. |
| 29 | `tests/test_phase4a.py::TestPhase4ACapabilityMeasurement::test_capability_profile_structure` | Phase 4; capability-profile test raises `UnboundLocalError` for `state`. |
| 30 | `tests/test_phase4a.py::TestPhase4ARealCheckpointing::test_orchestrator_restores_state_after_simulated_restart` | Infrastructure; the test attempts an unmocked live Ollama call. |

These are triage classifications only; no failure was fixed in that baseline
change. Categories reflect observed failure mechanisms; failures were not
reproduced separately for that classification. The earlier unbounded run stalled at 44% in the
300-turn soak test during deep-copy work in `execution_ledger.read`, reached
from ledger compaction. With the test explicitly quarantined, the bounded
default-suite run completed.

### Part E — current deficit status

| Deficit | Current status |
|---|---|
| 2, 3, 4, 14, 15, 27 | **Signed; partial implementation:** Phase 2.1/2.2 and Phase 3.1–3.3 task-state work are validated. Core control-plane session ownership, SQLite checkpoint ownership, production-wired Redis/PostgreSQL session adapters, and durable cross-process worker-cache invalidation are implemented. Phase 3.4 remains open pending checkpoint backend parity and other independent gates. Phase 3.5–3.9 remain open; physical semantic/graph index isolation (27) remains intentionally undecided. |
| 5 | **Status set:** authenticated PostgreSQL boot is explicitly excluded; see the B1 decision. |
| 6 | **Closed:** kernel boundary checker passes; Import Linter reports 4 kept / 0 broken, including the owned-store factory contract. |
| 7, 25 | **Closed for production configuration completeness**; the `default` compatibility policy is signed, but production auth-off behavior is unchanged by that policy. |
| 10 | **DEFERRED:** trimming remains disabled pending the substrate in [checkpoint-trimming-deferral.md](docs/decisions/checkpoint-trimming-deferral.md). |
| 11 | **Mechanism complete and focused tests pass:** projection generation/status survives restart with independent graph/semantic sync. |
| 20 | **Partial:** scanner-excluded suite baseline is complete; full disposition of the 141 failures and 11 errors is the current Stage 0 task. |
| 21 | **Closed:** carried-forward work is committed separately with explicit provenance; this is commit hygiene, not a claim that discovery is complete. |
| 22 | **Resource-bounded suite gate established:** soak case quarantined with owner/re-enable criterion; 4,245 collected, 4,246 JUnit result records, 3,972 passed, 141 failed, 11 errored, 122 skipped. Run completed non-green. |
| 23 | **Partial:** the test F821 gate was zero; the complete suite now ran but remains non-green, and no new full-suite-wide F821 scan was part of this change. |
| 24 | **Closed:** contract validation passed and manifest SHA remained unchanged. |

Phase 2.1/2.2, task-state recovery, core control-plane sessions, SQLite
checkpoints, tenant-owned Redis/PostgreSQL session adapters, and durable
cross-process worker-cache invalidation are implemented and wired into
production callers. Redis and PostgreSQL session integration ran against
disposable local services, including PostgreSQL under a
`NOSUPERUSER NOBYPASSRLS` role. Phase 3.4 remains open because checkpoint
backend parity and independent lifecycle gates are not fully demonstrated.
Phase 3.5–3.9 remain open. The semantic/graph-dependent memory portions of
matrix cases 1 and 3 are **BLOCKED** until Phase 3.5/3.6 is deliberately
scheduled.

### Current change validation

- Strict Phase 0 tests without `--runxfail`: **1 memory-isolation pass and
  3 pulse/auth-off passes**.
- `python ci/kernel_boundary_check.py`: **passed**; Import Linter:
  **4 kept, 0 broken**.
- The seven listed Phase 3 node IDs: **7 passed, 0 failed, 0 errored**.
- Focused Phase 3.4 validation (session lifecycle, ledger readers, SQLite
  checkpoints, recovery, lease/reconciliation, service architecture, and app
  runtime): passed; the previously excluded three service-architecture and
  two app-runtime regressions are included. Fresh-process same-ID session and
  checkpoint recovery pass.
- Redis/PostgreSQL tenant-owned session adapters are production-wired through
  `TenantOwnedStateStoreView`, `NamespacedStateStore`, `DadBotOrchestrator`,
  and `build_runtime_state_bundle`. The live-backend selection
  (`test_owned_redis_sessions.py`, `test_owned_postgres_sessions.py`,
  `test_postgres_execution_ledger.py`, `test_owned_sqlite_sessions.py`, and
  `test_control_plane_tenant_sessions.py`) completed with **41 passed, 0
  skipped, 0 failed** against Redis 8.0.5 and PostgreSQL 18. PostgreSQL used a
  disposable application role verified as `NOSUPERUSER NOBYPASSRLS`; tests
  confirmed cross-tenant RLS read/write denial and transaction-local tenant
  context reset after commit and rollback. Redis and PostgreSQL session tests
  exercised same-ID tenant isolation, mutation/listing/termination,
  mismatched ownership, unowned-record rejection, concurrency, and
  fresh-process recovery. The PostgreSQL session adapter's colliding
  truncated table-name bug and a legacy-row query placeholder mismatch were
  found by these live tests and corrected.
- Durable worker-cache invalidation is production-wired through the authority
  SQLite store. `revoke_session` commits a monotonically increasing revision
  and latest tenant/session revocation record in the same transaction as
  delegation revocation. Authenticated worker processes poll every 100 ms
  while idle and evict only the matching tenant/session cache; task-time
  authority checks remain in force. A spawned `worker_process_main` test
  demonstrated same-ID isolation for two tenants across termination and later
  work. Upgrading an existing authority database requires rerunning the
  idempotent `SQLiteDelegationStore.initialize`; startup fails closed without
  the revocation schema.
- Authority/worker/API regression selection (`test_durable_submission_authority.py`,
  `test_delegated_authority_acceptance.py`, `test_service_api_integration.py`,
  `test_service_websocket_tickets.py`, and `test_sqlite_service_state.py`):
  **110 passed, 1 skipped, 0 failed**. The skipped live Redis websocket-ticket
  test requires `DADBOT_TEST_REDIS_URL`; live Redis session-store evidence is
  reported separately above. The run exposed a websocket-ticket fixture that
  had not provisioned its browser principal's membership; the fixture now
  grants explicit read/write membership rather than weakening authorization.
- The previous **246 passed, 26 skipped, 0 failed, 0 errors** result was a
  DSN-less run and is superseded for backend parity by the live 41-test run.
- Authenticated `DadBotOrchestrator.terminate_session` revokes delegated
  authority scoped to `(tenant_id, session_id)` before deleting persisted
  session state. An in-flight worker revalidates authority and fails closed;
  idle workers consume durable revocation revisions and evict the matching
  cache within the polling interval. The test verifies another tenant's
  same-ID cache remains intact. No public API deletion route was found; the
  production termination caller currently available is the orchestrator
  method. SQLite termination rollback on an interrupted revocation write is
  covered; checkpoint Redis/PostgreSQL parity and their interrupted-write
  behavior remain unverified.
- The stale JSON-mirror path was reproduced. `load_memory_store` now derives
  its default fallback permission from `_local_ephemeral_document_store`;
  persistent-store outages fail closed instead of restoring stale JSON, while
  the intentional local-ephemeral fallback remains available. The regression
  test verifies no stale state is re-written after a failed authoritative
  read.
- The final combined service/session/authority selection passed **119 tests**
  with no skips or failures, including the previously excluded
  service-architecture and app-runtime regressions. The memory-manager and
  durable-submission-authority selection collected **106 tests** and
  completed with exit code 0; existing marked skips were unchanged. The
  stale-mirror regression passed in a separate focused run (**1 passed**). A
  long pytest temporary path caused Windows backup path errors on an initial
  run; rerunning with a short `%TEMP%` base passed.
- Ruff's full configured rules reported existing import-order, complexity,
  line-length, and other findings across the already-dirty files; no
  repository-wide lint-clean claim is made. Final focused Ruff F checks,
  `compileall` for the changed Python files, and `git diff --check` passed.
  The selected E501 check found one existing worker-readiness log line outside
  this change.
- Explicit selection of `test_soak_simulation_replay_loop`: **skipped** by
  its quarantine marker.
- Semantic/graph memory matrix cases remain blocked; this pass did not start
  Phase 3.5 or 3.6.

### Phase 3.4 checkpoint and revocation follow-up

- The graph checkpoint production path is `JobBuilder.build` ->
  `TurnContext.checkpoint_snapshot` -> `PersistenceService.save_graph_checkpoint`
  -> `ConversationPersistenceManager.persist_graph_checkpoint` ->
  tenant-bound `ExecutionLedger`. Checkpoint snapshots now carry `session_id`
  from the job metadata and include it in the checkpoint hash. Reads and turn
  event replay are scoped by active/explicit tenant and session; loading rejects
  checkpoint trace/session metadata that disagrees with its ledger event.
- `StreamlitRuntime.restore_checkpoint` now supplies the active thread and
  configured tenant to the resume path. Recent-checkpoint listing filters by
  the same identities and excludes malformed envelope/checkpoint pairs. No
  production callsite for `ReplayModeMixin._try_load_checkpoint` was found;
  that legacy helper remains a separate unverified surface.
- SQLite checkpoint validation now checks serialized `trace_id` as well as
  tenant/session identity. Existing SQLite tests cover same-ID tenant
  isolation, fresh-process recovery, concurrent operations, legacy-row
  quarantine, tampered metadata, and transactional interrupted writes.
- Authenticated worker execution fails closed when durable session-revocation
  authority cannot be read. An injected SQLite revocation-write failure now
  verifies that termination rolls back the revision/delegation update and
  does not evict or delete the in-memory session.
- Checkpoint restore now revalidates live delegation authority at both
  `SQLiteCheckpointer`'s tenant-resolution boundary and the graph-ledger
  recovery boundary. The core orchestrator requires authority when configured
  for authenticated storage; missing or revoked authority is rejected before
  checkpoint state is hydrated. A real SQLite authority + authenticated file
  WAL regression verifies revocation blocks both checkpoint restore paths.
- Focused validation passed: `.venv\Scripts\python.exe -m pytest
  --basetemp=.pytest_tmp_phase34
  tests/test_ledger_authority_invariants.py tests/persistence_test.py
  tests/test_persistence_service_schema_gate.py
  tests/test_replay_mode_mixin.py tests/unit/test_durable_submission_authority.py
  -q` (**96 passed, no skips**). Slow/stress memory selection
  `.venv\Scripts\python.exe -m pytest --basetemp=.pytest_tmp_phase34
  --run-slow --run-stress tests/test_memory_manager.py -q` completed
  successfully (**84 passed**).
  The first attempt without `--basetemp` could not create pytest's existing
  `%TEMP%\pytest-of-josep` path due to Windows access denial; the short
  repository-local temp base resolved the environment issue.
- At this audit stage, Redis/PostgreSQL checkpoint adapters and live checkpoint
  integration tests were absent. The subsequent Phase 3.4 checkpoint gate
  update below records the trusted-local adapters and live results.
- Cross-process cache invalidation and the stale-authority JSON-mirror
  regression were separate open gates at this audit stage. The later checkpoint
  gate update records the mirror rerun; the earlier cache-invalidation evidence
  remains unchanged.

### Authority, task recovery and event-identity repair

- Authenticated WebSocket and admin automation paths now retain and verify
  tenant authority through live effects. Spatial requests use task delegation
  and signed worker publication; admin operations verify runtime-tenant binding
  and revalidate active membership. Direct authenticated action-tool calls
  without authority are rejected; trusted-local mode is unchanged.
- Postgres task recovery now uses a tenant catalog updated in the same
  transaction as task writes. `CompositeStateStore` exposes the same
  tenant-qualified interface for Redis + Postgres, and recovery treats the
  durable backend as authoritative when a fast-store task is stale.
- Event and replay reads derive omitted tenant IDs from the authenticated
  principal and reject an explicitly different tenant. Request canonicalization
  also preserves gateway metadata across persisted task recovery.
- Live integration services were isolated disposable Docker containers using
  the repository Compose images `pgvector/pgvector:pg16` and `redis:7-alpine`.
  The test database used a `NOSUPERUSER NOBYPASSRLS` role; the running
  application containers and their data were not used or modified.
- Live integration command, with the two `DADBOT_TEST_*` environment
  variables set to those isolated services:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests\integration\test_postgres_execution_ledger.py tests\integration\test_owned_postgres_sessions.py tests\integration\test_owned_redis_sessions.py --basetemp=.venv\pytest-temp-live-all -q
  ```

  Result: **30 passed, 0 failed, 0 skipped**. This includes empty and
  multi-tenant persisted Postgres API lifespan recovery; tenant catalog
  discovery and forced RLS; completed-task publication and restart recovery in
  a separate Python process; rejection of a revoked task; and Redis + Postgres
  stale-state conflicts in both directions. Durable queued status wins over
  stale Redis completion, and durable completion suppresses stale Redis
  queued recovery. Identical task IDs remained isolated by tenant.
- Focused regression command:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests\unit\test_durable_submission_authority.py tests\test_service_api_integration.py tests\unit\test_owned_sqlite_sessions.py --basetemp=.venv\pytest-temp-focused-final -q
  ```

  Result: **62 passed, 0 failed, 0 skipped**.
- The first live Postgres run exposed a missing local `psycopg` import in
  `PostgresOwnedTasks.list_task_tenants`; tenant enumeration and API recovery
  failed before use. The import was added, and all affected live and focused
  tests passed on rerun.

### Phase 3.4 checkpoint gate status update

- Completed the selected trusted-local backend work without changing the
  authenticated production ledger policy. `PostgresCheckpointer` uses
  versioned PostgreSQL tables and commits each checkpoint row and success
  write-log row in one transaction. `RedisCheckpointer` uses tenant/session
  namespaced keys and a Lua transaction for checkpoint/index/registry/log
  updates. `RedisPostgresCheckpointer` always reads PostgreSQL as the durable
  authority; Redis is refreshed from PostgreSQL and is never used as a fallback
  if PostgreSQL is unavailable. Each remote adapter requires explicit
  `trusted_local=True`; none is automatically wired into the authenticated
  application.
- Live integration services were isolated disposable Docker containers using
  `pgvector/pgvector:pg16` and `redis:7-alpine`, bound to localhost ports
  56263 and 56264. PostgreSQL tests used a `NOSUPERUSER NOBYPASSRLS` role.
  The existing application containers and their data were not used or
  modified; the disposable test containers were stopped after verification.
- Live command, with `DADBOT_TEST_POSTGRES_DSN` and `DADBOT_TEST_REDIS_URL`
  pointed at those isolated services:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests\integration\test_checkpoint_backends.py --basetemp=.venv\pytest-temp-checkpoint-backends-concurrent -q
  ```

  Result: **9 passed, 0 failed, 0 skipped**. This covered PostgreSQL and Redis
  tenant-qualified recovery across fresh Python processes; same-session and
  same-trace tenant isolation; hash-chain and manifest checks; replacement,
  prune, delete, and concurrent same-trace updates; injected transactional
  PostgreSQL write failure and pre-mutation Redis script rejection; metadata
  tampering; and PostgreSQL precedence over a stale conflicting Redis record.
  Repeated construction verified idempotent PostgreSQL migration and Redis
  schema initialization.
- The final combined checkpoint, mirror, and authority command
  passed:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests\integration\test_checkpoint_backends.py tests\persistence_test.py tests\test_memory_manager.py::test_unavailable_tenant_memory_never_rehydrates_stale_json_mirror tests\test_memory_manager.py::test_composite_tenant_memory_does_not_read_stale_cache_when_durable_unavailable tests\unit\test_durable_submission_authority.py::test_authority_initializer_upgrades_partial_schema_without_resetting_state tests\unit\test_durable_submission_authority.py::test_interrupted_session_termination_rolls_back_revocation_and_keeps_session --basetemp=.venv\pytest-temp-phase34-final-combined -q
  ```

  **39 tests, 0 failed, 0 skipped**. This includes SQLite transactional
  interrupted-write handling, corrupted-payload rejection, tenant-isolated
  fresh-process recovery, and orchestrator restart restoration. The authority
  upgrade test removes the revocation table, verifies unavailable-state
  rejection, reruns initialization, and confirms identity/revision preservation
  and idempotence. The stale JSON-mirror regression confirms an unavailable
  authoritative tenant store raises without restoring or rewriting stale JSON.
- The required focused regression command was rerun:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests\unit\test_durable_submission_authority.py tests\test_service_api_integration.py tests\unit\test_owned_sqlite_sessions.py --basetemp=.venv\pytest-temp-focused-phase34-final2 -q
  ```

  Result: **62 passed, 0 failed, 0 skipped**.
- The final production-path audit confirms that graph checkpoints in the
  application remain `PersistenceService.save_graph_checkpoint` ->
  `ConversationPersistenceManager.persist_graph_checkpoint` ->
  `ExecutionLedger`. No production constructor for `AbstractCheckpointer` was
  found; it remains optional injection. Authenticated startup still requires
  the durable file WAL and rejects PostgreSQL execution-ledger mode. The new
  adapters therefore close the requested trusted-local backend parity gate,
  not authenticated PostgreSQL support or automatic application wiring.
- An initial remote test run found a test-only SQL-template formatting error
  in the PostgreSQL metadata-tampering case; the SQL path was parameterized
  and the final live rerun passed. A prior preliminary combined run also had a
  transient orchestrator restart `StateDivergenceError`; it passed in
  isolation, the full SQLite suite, and the final combined run without a code
  change to suppress it.
- **Phase 3.4 acceptance gates: PASS for the selected trusted-local scope.**
  Authenticated production continues to use the durable file WAL. This is not
  a production-readiness claim for PostgreSQL or Redis in authenticated mode;
  no Phase 3.5/3.6 work was started.
