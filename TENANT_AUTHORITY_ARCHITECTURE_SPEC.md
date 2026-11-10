# Tenant authority architecture and remediation specification

Status: **proposed; incomplete discovery; not implemented or approved**.
This specification covers the current working tree and does not certify tenant
isolation. It does not modify runtime behavior, migrate data, relax existing
safeguards, authorize PostgreSQL application integration, or initiate checkpoint
retention. The companion [execution-path audit](TENANT_EXECUTION_PATH_AUDIT.md)
records prior evidence and outstanding inventory. All requirements below are
proposals until architecture and deployment decisions are approved.

## 1. Evidence and limits

**Reproduced:** a fresh-process local test demonstrated that a second tenant
using the same paths imports the first tenant's synthetic memory. The test in
[`tests/unit/test_tenant_isolation_acceptance.py`](tests/unit/test_tenant_isolation_acceptance.py)
is strict-xfail and fails with `--runxfail`; it is not acceptance evidence.
Prior exploration also demonstrated that `tenant/a` and `tenant-a` normalize to
the same value.

**Observed in code, not independently reproduced across all deployments:**

- [`DadBotConfig`](dadbot/config.py#L344) normalizes tenant IDs and resolves
  paths globally; [`MemoryStorageBackend`](dadbot/memory/storage.py#L690)
  falls back to the shared local mirror during tenant establishment.
- [`NamespacedStateStore`](dadbot_system/state.py#L268) namespaces session and
  event keys but forwards task and response methods unchanged.
- [`SessionRegistry`](dadbot/core/control_plane.py#L323) indexes by session ID;
  pulse reads it by session ID
  ([`api.py`](dadbot_system/api.py#L678)).
- SQLite and PostgreSQL semantic and graph stores use unscoped primary keys;
  PostgreSQL graph replacement truncates its tables
  ([`semantic_index.py`](dadbot_system/semantic_index.py#L129),
  [`semantic_index.py`](dadbot_system/semantic_index.py#L471),
  [`graph_store.py`](dadbot_system/graph_store.py#L78),
  [`graph_store.py`](dadbot_system/graph_store.py#L353)).
- [`SQLiteCheckpointer`](dadbot/core/persistence/checkpointer.py#L109) keys its
  optional checkpoint store by `(session_id, trace_id)`, not tenant.
- [`DadBotTaskProcessor`](dadbot_system/worker.py#L109) has a custom-factory
  `TypeError` fallback that omits tenant; worker retries catch general
  exceptions ([`worker.py`](dadbot_system/worker.py#L210)).
- API checks scopes at handlers, while some automation and spatial handlers
  use a singleton runtime bot
  ([`api.py`](dadbot_system/api.py#L1035),
  [`api.py`](dadbot_system/api.py#L1248)). Actual cross-tenant exposure depends
  on deployment and is not proven by code inspection alone.
- Source-grounding approval rows are keyed by tenant and source
  ([`source_grounding.py`](dadbot/services/source_grounding.py#L58));
  this does not prove that the underlying source data is owned correctly.

**Coverage limit:** the repository contains hundreds of production modules and
dynamic imports. The family inventory and matrix below are a scoped navigation
and test plan, not an exhaustive proof of production reachability. Optional
plugins, dynamic entry points, direct backend users, every core call path,
PostgreSQL and Redis deployments, and OS permissions remain unverified. No
new acceptance tests were run for this specification.

## 2. System invariants

- **I1 — Identity:** a single immutable canonical tenant identity is resolved
  once at a trusted boundary; no lossy transformation aliases two distinct
  owners. Tenant, session, bot, task, trace, resource and operation identities
  are bound and validated throughout the lifetime of work.
- **I2 — Authority:** an authenticated actor or explicitly trusted local
  operator has an auditable, least-privilege grant for an operation on a
  resource owned by the same tenant. User-supplied payloads and serialized
  metadata are not authority.
- **I3 — Ownership:** every tenant-bearing read, write, list, delete, clear,
  export, restore, cache lookup and repair addresses an owned resource.
  Cross-tenant operations require an explicit, separate platform-admin grant,
  never merely a tenant-admin grant.
- **I4 — Persistence:** committed ledger state remains authoritative;
  failure, retry, repair or migration cannot silently assign unowned data,
  revert a committed event or publish a stale projection. Existing safeguards
  stay enabled until replacement gates pass.
- **I5 — Isolation under failure:** races, crash/restart, retries, expired
  grants and partial migration must fail closed for ambiguous ownership while
  preserving previously valid data and authorized single-user access.

## 3. Proposed canonical identity and authorization

### Identity

1. Issue an opaque, stable tenant identifier at a trusted provisioning
   boundary. Use an exact, case-sensitive, versioned canonical encoding;
   reject invalid or ambiguous external forms instead of rewriting characters
   or defaulting malformed input. Keep display names/legacy aliases separate.
   Existing IDs require a collision report and explicit administrator
   resolution before migration; never merge colliding identities.
2. Resolve a verified **subject** (issuer + subject), tenant membership and
   grants at ingress. Client-supplied `tenant_id` is a requested resource,
   never proof of membership. Derive a non-forgeable internal execution
   context carrying tenant, subject/service identity, session, task, trace,
   action, delegation and policy version. Do not equate trace/request IDs
   with ownership.
3. Use typed compound keys, not ambiguous string concatenation:
   `(tenant_id, session_id)`, `(tenant_id, task_id)`,
   `(tenant_id, resource_kind, resource_id)`; scope idempotency, events,
   checkpoints and caches accordingly. Bind bot instances to one tenant and
   assert that principal, task, session, request and bot all agree before
   state hydration and before effect execution.
4. A **tenant administrator** is limited to that tenant. A **platform
   administrator**, if deployment supports one, has a distinct explicit
   cross-tenant grant, audited resource selection and narrowly scoped
   operations; it must not be inferred from `admin` alone. Default access is
   deny. Global operational health may expose only non-sensitive information.

### Shared authorization contract

All interfaces call a common policy decision mechanism with a trusted actor,
canonical tenant, action, resource owner, execution mode and contextual
constraints. Enforcement belongs at ingress **and** at stateful/effectful
service and repository boundaries: internal calls must not rely on an API
check that can be bypassed.

- API and WebSocket ingress authenticate; bind session and ticket to tenant,
  action, path and expiry; prevent replay and reject tenant mismatch.
- Local UI/CLI operate under an explicit trusted-local single-user capability,
  not an implicit cross-tenant admin grant. Remote exposure requires a
  separately authenticated boundary.
- Worker and broker consumers validate task/request/bot/session tenant
  equality and a verifiable delegation envelope. Queue payload fields alone
  are untrusted. Retry and recovery revalidate authority and revocation/expiry
  before side effects; user tokens and secrets are not persisted in plaintext.
- Background jobs have service identities and tenant-scoped least-privilege
  grants; schedule ownership and cancellation survive restart. Missing,
  expired, revoked or mismatched authority fails explicitly. Clarify whether
  previously accepted work can complete under a bounded delegated grant.
- Tool, MCP and integration calls require action-specific permission, tenant,
  destination and user approval where applicable. Authorization must be
  applied to direct calls as well as planner-driven calls; optional middleware
  must not silently imply allow. Distinguish local MCP transport access,
  upstream network identity, approval and workspace filesystem permissions.
- Reads, exports, deletion and recovery require the same owner check as
  mutations. Denial must be explicit and avoid existence leaks as required by
  policy; avoid logging secrets or private payloads.

This specifies *what* the shared mechanism must enforce, not a choice of IAM
product or an assertion that current credentials already implement it.

## 4. Resource ownership model

Shared physical storage is permitted only with enforced logical ownership and
appropriate OS/database permissions. A separate path/table is not by itself
an authorization boundary. Legacy unowned resources are not assigned an owner
by their name, path, first reader or a guessed `default` tenant.

| Resource / current location | Proposed authority and owner | Required enforcement / negative test |
|---|---|---|
| Local memory mirror, backup and migration evidence [`storage.py`](dadbot/memory/storage.py#L690), [`local_tenant_establishment.py`](dadbot/core/local_tenant_establishment.py#L39) | Tenant ledger owns memory; mirror is tenant-tagged derived state, migration evidence is installation + source + claim owner. | Bind paths, manifests and locks to owner; B cannot import, replace, back up or claim A's mirror. |
| File WAL / PostgreSQL ledger [`ledger_backend.py`](dadbot/core/ledger_backend.py), [`postgres_ledger_backend.py`](dadbot/core/postgres_ledger_backend.py) | Per-tenant authoritative event streams or shared append log with tenant-bound verified events; installation-level chain metadata separately owned. | Tenant-scoped append/replay/reconciliation; prevent A querying B and reject mismatched or unowned events. Do not remove existing head/fence checks. |
| SQLite/PG semantic and graph [`semantic_index.py`](dadbot_system/semantic_index.py), [`graph_store.py`](dadbot_system/graph_store.py) | Tenant-owned derived projections; independent schema, revision, generation and dirty metadata tied to tenant and ledger. | Composite tenant keys across all SQL, including reads, clear, upsert, deletion and rebuild; stale candidate or mismatched generation cannot publish; B survives A's clear. |
| Profile and document state [`runtime_storage.py`](dadbot/managers/runtime_storage.py#L88), [`state.py`](dadbot_system/state.py#L268) | Tenant-owned profile; tenant-scoped session and memory documents. | Explicit owner on fallback/import; B cannot load A's profile, session or backup even with identical IDs. |
| Sessions, tasks, results, events and Redis queues [`orchestration.py`](dadbot_system/orchestration.py), [`state.py`](dadbot_system/state.py), [`worker.py`](dadbot_system/worker.py) | Compound tenant/session/task identifiers and trusted task delegation. Shared infrastructure has service ownership; payload has tenant owner. | Backend and caller enforce owner on every operation; test matching IDs, forged results, retries and recovery. |
| SQLite checkpointer and event durability [`checkpointer.py`](dadbot/core/persistence/checkpointer.py#L109), [`event_durability.py`](dadbot_system/event_durability.py#L71) | Checkpoints, runs, events and handoffs owned by tenant plus session/run; chain/lease binds owner. | Validate owner on save/load/delete/replay/import; test identical sessions/traces and forged handoffs. No checkpoint retention changes here. |
| Embedding, semantic, retrieval, model and answer caches [`semantic_manager.py`](dadbot/memory/semantic_manager.py#L126), [`source_grounding.py`](dadbot/services/source_grounding.py#L113) | Tenant-owned unless explicitly classified as safe shared non-sensitive metadata. | Partition or prove non-sensitive sharing, invalidation and deletion; no cached A result for B with same query/key. |
| Profile/session logs, telemetry, audit, exports and backups [`conversation_persistence.py`](dadbot/managers/conversation_persistence.py#L643), [`storage.py`](dadbot/memory/storage.py#L866) | Tenant-specific records/artifacts; installation-wide diagnostics have minimized contents and restricted operator access. | Authorize destination, restore and file access; owner metadata, restrictive permissions, safe naming, encryption/retention policies; B cannot retrieve A export. |
| MCP workspace, calendar, email drafts, browser, network and external vector collections [`local_mcp_server.py`](dadbot_system/local_mcp_server.py#L310), [`vector_memory.py`](dadbot/services/vector_memory.py#L174) | Tenant or explicitly global service ownership per resource; external credentials bound to authorized integration. | Per-operation authorization, path/destination containment and owner-aware collection operations; no implicit global action or shared collection reset. |
| Citation approvals [`source_grounding.py`](dadbot/services/source_grounding.py#L58) | Tenant and source digest. | Preserve composite-key protections; validate source owner and prohibit B's approval/revocation of A's source. |

Unknown resource families discovered later must be added before the inventory
gate is met. Direct database/filesystem access by an operator outside DadBot
is governed by a documented deployment trust model; application checks cannot
replace database roles or filesystem ACLs.

## 5. Legacy single-user compatibility and explicit migration

**Compatibility requirement:** the existing single-user installation must
remain usable in its established trusted local mode. Availability is not a
license to grant a new tenant access to its ambiguous files. Multi-tenant
activation, remote access and ownership reassignment must not be inferred from
first startup or client input.

Proposed protocol, subject to a product decision about operator consent:

1. Discover files, sidecars, ledger streams, database tables and backups
   read-only; identify existing evidence, hashes, active processes and aliases
   without copying private data into the report. Classify each resource as
   owned, legacy-unowned, conflicted or unreadable.
2. Permit the existing local installation to operate under an explicit
   installation-bound legacy mode while blocking new tenant claims and shared
   multi-tenant access to ambiguous data. Record no new assignment merely
   because the local mode is used. Do not change its data or configuration
   without consent.
3. An authenticated local operator explicitly chooses a destination tenant
   and scope after reviewing discovered ownership and collisions. A dry run
   lists affected resource classes, incompatible backends, disk requirements
   and rollback constraints. Reject missing/conflicting proof.
4. Acquire an installation-wide cross-process migration fence before the
   source can be read or changed; block *all* old and new writers, including
   background tasks and other configured paths. If every writer cannot
   cooperate, require offline migration. The existing per-WAL lock alone does
   not prove coverage of profile, indexes, backups and exports.
5. Durably record a versioned intent with operation ID, source fingerprint,
   destination, affected resources and phase; retain source untouched. Stage
   copies/candidates in destination-owned storage, validate contents and owner,
   reconstruct projections from authority as applicable, and verify unchanged
   source plus ledger revisions. Never import an index as authority.
6. Publish through an atomic manifest/generation switch with durable ordering
   appropriate to the filesystem/database; source remains recoverable. Only
   the switch authorizes the new owner. Resume from intent after interruption,
   check destination and source fingerprints, and reject conflicting claims.
   Recovery must be idempotent and distinguish incomplete publish from
   completed ownership. Document Windows sharing violations and power-loss
   assumptions separately; `os.replace` alone is not proof of durability.
7. Release old-mode access only after verification and explicit cutover.
   Preserve backups under the same owner and access policy; rollback must not
   discard commits made after cutover. If correctness is uncertain, block
   migration/multi-tenant operation, not silently rewrite or delete files.

## 6. Dependency map

```text
verified ingress / trusted local operator
   -> canonical identity + principal/grant
   -> tenant-bound session, bot, task and resource keys
   -> shared authorization decision + enforcement at effect boundaries
   -> tenant-owned stores / ledger and explicit legacy ownership
   -> revision-fenced semantic + graph publication and recovery
   -> worker retries, lifecycle, replay, tools, exports, observability
   -> end-to-end gates across both tenants and fresh processes
```

Identity and authenticated ownership are prerequisites for correctly scoped
storage and safe migration. Ledger authority and fenced projection recovery
depend on scoped stores and ownership; enabling indexed mutations depends on
their recovery gates. Worker, tool and replay enforcement depends on a
verifiable delegated identity. Observability must be designed with every
phase, but must not become a second source of authority. The model preserves
the existing local WAL fencing and mutation guard throughout.

## 7. Requirement -> code -> test traceability

**Existing** means a named test exists, not that it proves the full
requirement. **Planned** tests below do not exist merely because they are named.
All planned cases are blocking until actually implemented and run.

| ID / enforcement requirement | Current path / observed status | Existing evidence | Required negative acceptance test |
|---|---|---|---|
| T01 canonical identity: reject aliases and invalid IDs | [`contracts.py`](dadbot_system/contracts.py#L41), [`config.py`](dadbot/config.py#L344): lossy normalization | prior runtime collision probe | **P01:** colliding external IDs never share tenant, keys or grants; invalid/missing ID denied. |
| T02 bind principal -> request -> bot -> task | [`api.py`](dadbot_system/api.py#L250), [`worker.py`](dadbot_system/worker.py#L109): multiple fields/fallback | [`test_api_rejects_tenant_mismatch`](tests/test_service_api_integration.py#L794) | **P02:** forged/missing request, task or factory tenant denied before hydration/effects, including internal `TypeError`. |
| T03 sessions/events/checkpoints compound ownership | [`control_plane.py`](dadbot/core/control_plane.py#L323), [`checkpointer.py`](dadbot/core/persistence/checkpointer.py#L109) | API event stream and session namespace tests | **P03:** A/B same session and trace: no registry, pulse, replay, checkpoint, delete or termination crossover. |
| T04 task/result ownership and recovery | [`state.py`](dadbot_system/state.py#L295), [`orchestration.py`](dadbot_system/orchestration.py#L164) | service API tests | **P04:** B cannot read A task; forged result/recovered task mismatch rejected without publishing. |
| T05 local legacy/profile ownership | [`storage.py`](dadbot/memory/storage.py#L690), [`runtime_storage.py`](dadbot/managers/runtime_storage.py#L88) | strict-xfail local memory test; profile document-store test | **P05:** B cannot import/read/overwrite A legacy memory/profile; single-user mode stays operational. |
| T06 authoritative ledger/replay | [`tenant_memory_ledger.py`](dadbot/core/tenant_memory_ledger.py#L74) | [`test_tenant_memory_ledger.py`](tests/unit/test_tenant_memory_ledger.py) | **P06:** B cannot append/replay/restore A; stale or uncertain append fails, no silent reimport. |
| T07 semantic/graph shared-store isolation | [`semantic_index.py`](dadbot_system/semantic_index.py), [`graph_store.py`](dadbot_system/graph_store.py) | one-tenant recovery and fence tests | **P07:** identical content/IDs in SQLite and live PostgreSQL; A rebuild/clear/delete leaves B unchanged, including direct administrative paths. |
| T08 revision/generation recovery | [`storage.py`](dadbot/memory/storage.py#L191), [`semantic_manager.py`](dadbot/memory/semantic_manager.py#L641), [`graph_views.py`](dadbot/memory/graph_views.py#L169) | interrupted semantic/graph tests | **P08:** independent metadata, interrupted publication, stale source and corrupt metadata never expose stale data or alter tenant revision. |
| T09 auth on every ingress, admin explicit | [`api.py`](dadbot_system/api.py#L171), [`local_mcp_server.py`](dadbot_system/local_mcp_server.py#L310) | auth/mismatch and WebSocket integration tests | **P09:** unauthenticated, expired, wrong scope/tenant, tenant-admin cross-tenant, ticket replay and singleton-bot mismatch denied. |
| T10 worker delegation/retry | [`worker.py`](dadbot_system/worker.py#L164) | worker/service tests | **P10:** expiry/revocation or altered queue metadata on retry/restart denies side effects; same-tenant idempotent recovery succeeds. |
| T11 tools, MCP and background services | [`action_mixin.py`](dadbot/core/action_mixin.py#L27), [`external_tool_runtime.py`](dadbot/core/external_tool_runtime.py#L636), [`boot_mixin.py`](dadbot/core/boot_mixin.py#L432) | external tool and memory policy unit tests | **P11:** direct/planned/MCP/background unauthorized tool, workspace or remote destination is denied even without optional middleware. |
| T12 cache, exports, backup and logs | [`semantic_manager.py`](dadbot/memory/semantic_manager.py#L126), [`conversation_persistence.py`](dadbot/managers/conversation_persistence.py#L643), [`storage.py`](dadbot/memory/storage.py#L866) | source approval cache unit tests | **P12:** A's identical-query cache, exports, backups and logs inaccessible to B; delete/restore remains scoped. |
| T13 ownership migration and lifecycle | [`local_tenant_establishment.py`](dadbot/core/local_tenant_establishment.py#L39), [`boot_mixin.py`](dadbot/core/boot_mixin.py#L575) | one-tenant startup interruption and concurrency tests | **P13:** competing claims, offline/online fence, crash each phase and restart; no source loss, no implicit claim, local access preserved. |
| T14 optional vector/integration and run stores | [`vector_memory.py`](dadbot/services/vector_memory.py#L174), [`event_durability.py`](dadbot_system/event_durability.py#L173), [`tool_client.py`](dadbot/runtime/mcp/tool_client.py#L178) | partial tests; reachability unknown | **P14:** reset/run-id/integration credential or replay cannot cross tenants; prove actual deployed wiring. |
| T15 unknown/dynamic paths | dynamic imports, plugins, `dadbot/core` and alternate adapters | no complete coverage | **P15:** inventory each reachable effect and add an owner check plus negative test before marking discovery complete. |

## 8. System-wide acceptance gates

1. **Discovery gate:** inventory every production entry point and enabled
   adapter, direct store operation, background effect and replay path. Map
   caller, trusted identity, owner, enforcement, failure mode and both positive
   and negative tests; review static and dynamic call graphs. Unknown paths
   keep this gate open.
2. **Identity/authorization gate:** P01-P04 and P09-P11 pass at ingress and
   direct internal boundaries. Tests use two authenticated principals, tenant
   admins and explicit platform operator, identical session IDs, expired/
   revoked grants, tampered tickets and forged task metadata.
3. **Ownership and migration gate:** P05, P07, P12-P14 run with two real
   tenant processes sharing the same physical storage, identical logical
   identifiers, concurrent startup, simulated termination at each migration
   phase, competing claimants and fresh-process verification. Include
   SQLite and live PostgreSQL/Redis where supported; unavailable services are
   reported as unverified, not passed. Preserve legacy single-user access.
4. **Durability/recovery gate:** P06 and P08 exercise both projection failures,
   metadata corruption/interruption, failed publication, concurrent newer
   commit, repeated repair and crash/restart. Verify last valid data,
   independent revisions, no duplicates, deletion and unchanged authoritative
   revision during repair. Keep current guard until supported mutations have
   their own post-commit recovery proofs.
5. **Lifecycle/replay and operational gate:** P10-P15 plus retry, canceled
   work, shutdown, post-restart authorization, replay without external side
   effects and degraded-status visibility. Run focused, integration, broader
   suites and configured architecture gates; publish exact commands, counts
   and known baseline failures. Tests must pass unmarked; xfail/skip is not
   proof. No claims of Windows power-loss durability without dedicated tests.

## 9. Smallest coherent remediation sequence

0. Complete discovery and agree on trust boundaries and legacy-owner consent;
   preserve existing data and current guards. Do not claim a complete audit
   while dynamic callers or resource families are unaccounted for.
1. Introduce canonical tenant/resource identity and a verified execution
   context; make factory, session, task and broker boundaries reject mismatch.
   Establish authorization decision and enforcement interfaces shared by all
   ingress and effectful calls. Gate with P01-P04 and P09-P11.
2. Implement owner-aware repositories and resource bindings for all active
   stores, covering every operation and direct backend entry point. Add
   migration/ownership discovery and explicit single-user-compatible cutover
   under a global fence. Gate with P05-P07 and P12-P14.
3. Integrate ledger revisions and independent projection generations under
   the ownership contract; revalidate before publication, recover after
   interruption, expose degradation, then prove supported mutations before
   relaxing their guard. Gate with P06 and P08.
4. Integrate delegated context with retries, recovery, tools, background
   jobs, exports and shutdown; certify lifecycle and deterministic replay.
   Complete P10-P15 and the broader suites.

These are architectural work packages, not isolated bug fixes. Exact file
layout and deployment policy require approval after inventory.

## 10. Unresolved decisions and discovery

- Who is authorized to bind an unowned legacy installation to a tenant, how
  is consent recorded, and how is the current single local user identified
  without silently declaring ownership?
- Which deployments require multi-tenant shared physical databases versus
  isolated installations? What isolation do database roles and OS accounts
  actually provide?
- How are canonical IDs provisioned, how are existing collisions resolved,
  and which aliases must remain backward compatible?
- Is platform-wide administration required? Who issues that grant, how is
  it audited, and which actions must remain tenant-only?
- When queued work outlives a user session, should it be canceled on expiry,
  reauthorized, or run under a bounded durable service delegation?
- Which MCP transports, integrations, background jobs and optional vector
  stores are enabled in supported production configurations?
- What level of crash/power-loss guarantee is required for local Windows
  installations, backups, SQLite and remote databases?
- How are privacy-preserving telemetry, export destinations, encryption keys,
  retention and deletion obligations defined for each resource?
- Who reviews the completed path inventory and approves final acceptance
  criteria and migration/cutover policy before implementation?

## 11. Next discovery work

Perform static plus runtime-assisted reachability tracing for CLI, Streamlit,
HTTP, WebSocket, worker, MCP and scheduled paths; inspect all configured
store implementations and direct calls, including SQL, Chroma, filesystem,
Redis, caches and handoffs. Link each concrete entry/effect to a matrix row,
its test and configuration. Verify remaining test coverage rather than
assuming absence or applicability. This work remains open; this document is
not the complete production-reachable inventory requested.

## 12. Foundation-boundary discovery

The following is additional code inspection for selecting the initial shared
contracts. It does not establish which options are enabled on a particular
installation, prove runtime reachability of dynamically loaded modules, or
close the discovery gate.

| Surface and dependency path | Current evidence and boundary implication |
|---|---|
| Launcher -> local CLI or UI | [`launch.py`](launch.py) delegates to [`app_runtime.main`](dadbot/app_runtime.py#L1057). That function constructs a local bot before dispatching CLI or Streamlit; clear and export can also act directly on that bot. Streamlit separately constructs a bot in [`_build_bot`](dad_streamlit.py#L852). These are separate initialization surfaces, not necessarily one shared instance. |
| Streamlit persistence | [`_build_bot`](dad_streamlit.py#L874) attaches [`SQLiteStreamlitThreadStore`](dadbot/ui/streamlit_context.py#L141), whose table is keyed only by `thread_id` and whose load reads all rows. It also reads the configured profile file directly after boot. Shared-path isolation, browser access control and actual deployment exposure are unverified. Localhost binding alone is not proof of an authorized user. |
| API -> control plane -> broker -> worker | [`launch_api_service`](dadbot/app_runtime.py#L932) constructs an in-memory event bus, `LocalMultiprocessBroker`, state store, orchestrator, worker manager and singleton runtime bot. [`_kernel_execute`](dadbot_system/api.py#L290) reconstructs a request from scheduler payload and submits it. The ingress boundary must not be the only enforcement point: persisted state, scheduled payloads and process messages cross additional trust boundaries. |
| Worker and restart | [`submit_chat`](dadbot_system/orchestration.py#L87) persists task/request metadata before enqueueing. [`recover_pending_tasks`](dadbot_system/orchestration.py#L164) reconstructs work from stored payload and replaces the request tenant with the task tenant. [`process`](dadbot_system/worker.py#L210) selects a bot using the task tenant and creates a turn context from the request tenant. Equality, delegation integrity and current authorization are not established by these operations alone. |
| Direct local MCP | [`local_mcp_server.py`](dadbot_system/local_mcp_server.py#L60) resolves an injected or global bot and its [server](dadbot_system/local_mcp_server.py#L310) exposes tools. Whether only a trusted local process can connect depends on transport and deployment; no tenant principal is visible in the examined handlers. Do not treat all MCP clients as trusted solely because the module is named local. |
| Boot and storage selection | [`DadBot` boot](dadbot/core/boot_mixin.py#L514) configures paths and stores, initializes the execution ledger, hydrates memory, repairs projections, then starts orchestration and background tasks. [`build_customer_document_store`](dadbot/app_runtime.py#L875) fails if configured stores cannot initialize and production requires PostgreSQL; [`build_service_state_store`](dadbot/app_runtime.py#L842) may fall back from Redis or PostgreSQL but requires a durable store in production. Neither behavior establishes resource ownership. |
| Deployment and optionality | [`pyproject.toml`](pyproject.toml) declares Streamlit, Chroma and Ollama as base dependencies; service, voice and MCP dependencies are extras. [`docker-compose.yml`](docker-compose.yml) describes a local UI plus Ollama and a production-profile API plus Redis/PostgreSQL. This is declared configuration, not proof those services are currently running or safe for multiple tenants. The queue created by the inspected API launcher is local multiprocessing even though a queue backend setting exists. |
| Existing authorization mechanisms | [`ServicePrincipal`](dadbot_system/security.py#L57) provides signed-token identity and scope checks at service ingress. [`SessionAuthorizationPolicy`](dadbot/core/authorization.py#L154) defaults to read/write for unknown sessions unless strict and is documented as in-process protection, not production authentication; [`CapabilityRegistry`](dadbot/core/capability_registry.py#L1) provides stage-level enforcement. They are useful integration points, not proof of a shared owner-aware policy on every path. Pylance references found registry and test uses of the session policy; this does not prove which runtime configurations enable it. |
| Existing execution context | [`SovereignContext`](dadbot/core/execution_contract.py#L30) carries tenant, session, trace, request, mode and policy scope, but no verified principal or resource-owner grant. Its serialized data alone must not become a trusted credential. Preserve replay semantics while separately validating live authority. |
| Alternate launcher | [`dadbot/runtime/launcher.py`](dadbot/runtime/launcher.py#L310) contains another API composition path without the same singleton bot wiring. Its production reachability remains unverified; it cannot be classified as legacy or removed based on filename alone. |

### Trust-boundary proposal for approval

- Treat a local operator as trusted **only through an explicitly selected
  installation-local mode**, subject to OS and transport access controls.
  Loopback, a session ID, a tenant string, or the absence of an API token
  alone must not establish identity. The precise browser and MCP access
  controls need a deployment decision and verification.
- Treat API token verification as ingress authentication, not as permission
  to trust later mutable request metadata. Bind principal and tenant to
  scheduler work, worker tasks and effects using an authenticated delegation
  appropriate to the actual process boundary.
- Treat persisted tasks, queues, session state, snapshots and replay data as
  inputs to be validated; they are not proof that the original authorization
  remains valid. Distinguish replay with no effects from resumed live work.
- Treat database connections and filesystem paths as storage mechanisms, not
  ownership or authorization evidence. The scope of protection against a
  user with direct access to the operating system account or database
  credentials must be explicit.
- Before applying these contracts to any subsystem, approve a canonical ID
  and alias transition, local-mode trust and legacy-owner consent, delegated
  retry/expiry behavior, and tenant versus platform administrator scope.
  Implementing interfaces alone does not establish isolation, and no existing
  guard should be removed until its replacement has passing acceptance tests.

## 13. User-selected trust requirements and remaining decisions

The user selected opaque stable tenant IDs with explicit legacy alias and
collision resolution, live-effect reauthorization on retries and recovery
with fail-closed behavior, and a distinct explicitly granted and audited
platform administrator. The user requested **continued discovery and
specification revision before implementation**; these selections do not
authorize code changes or data migration.

For the existing installation, the user specified an explicit single-user
local trust mode with all of these requirements:

- Bind local UI and services to loopback by default. Local execution or
  loopback alone does not authenticate a caller.
- Enforce OS-account and filesystem permissions for data, databases,
  configuration and credentials; restrict CLI operations to the authorized
  local OS account.
- Authenticate and authorize non-loopback, remote API and externally
  reachable interfaces; browser-accessible interfaces must not bypass
  service-layer authorization.
- Give background workers explicit service identities and tenant scope.
  Migration, recovery, administrative mutations and other sensitive
  operations require explicit authorization. Never infer ownership from a
  local path or access alone.
- Treat same-OS-account processes as one local trust domain; protection
  against a malicious process already running as that account is explicitly
  outside this mode's security claim. Preserve the existing single-user
  experience and support eventual authenticated multi-user operation without
  redesigning the identity/ownership foundations.

**Additional observations:** [`ApiSettings`](dadbot_system/contracts.py#L265)
defaults the API host to `127.0.0.1`, but it is configurable. The Streamlit
launcher [checks whether a loopback port can be bound](dadbot/app_runtime.py#L583)
but does not pass `--server.address` in its
[launch command](dadbot/app_runtime.py#L627); the checked-in
[Streamlit configuration](.streamlit/config.toml) does not set an address.
The actual listener depends on Streamlit configuration/environment and has
not been inspected in a running process. A loopback port preflight is not
proof that the resulting server binds only loopback. Compose host port
publishing uses loopback, but that is not proof of guest-process binding or
browser authorization.

[`BackgroundTaskManager`](dadbot/background.py#L17) accepts arbitrary
callables and metadata, and the boot path
[`_start_boot_background_tasks`](dadbot/core/boot_mixin.py#L436) submits graph
and heartbeat jobs without an explicit service identity in the inspected
calls. This is a propagation gap to test, not proof that all background work
is unauthorized. The container
[entrypoint](docker/entrypoint.sh) changes ownership and runs as an account;
that does not establish equivalent Windows ACLs, authorized-user checks, or
tenant ownership.

### Contract and test consequences

- The local-mode bootstrap must establish the authorized installation and
  user through a documented, OS-specific mechanism, verify permissions
  including parent directories and sensitive files, and fail explicitly when
  its trust assumptions cannot be met. Avoid claiming a portable guarantee
  before Windows ACLs, POSIX permissions and supported container mount
  behavior have each been verified.
- A loopback HTTP connection cannot by itself prove the browser's OS user.
  Design an authenticated bootstrap and session mechanism with origin,
  cross-site request, DNS-rebinding, WebSocket, token leakage and expiry
  protections. It must reach the same service authorization checks as other
  callers; the local-mode UX must still be usable without silently granting
  cross-tenant access. The exact bootstrap mechanism remains undecided.
- Platform-wide administration needs an issuer, grant lifecycle, action and
  tenant selection, auditing, and tests for absence/revocation. Tenant admin
  remains tenant-scoped; no global rights are implied by a generic `admin`
  scope.
- Define how a worker obtains a verifiable tenant-scoped service identity
  and how it reauthorizes each effect after expiry, revocation, restart or
  an unavailable policy service. Recovery that would mutate persistent
  projections is an authorized service action; read-only replay cannot
  acquire live-effect privileges merely by loading a record.
- Add blocking acceptance cases: inspect actual listening addresses; reject
  unauthorized local browser and remote requests including alternate hosts,
  forged origins and WebSockets; deny CLI invocation by an unauthorized OS
  account; verify Windows ACL and supported POSIX/container permissions;
  assert worker identity/tenant propagation, reauthorization and fail-closed
  recovery; prove authorized single-user use and authorized local recovery
  still work. Tests requiring distinct OS accounts, non-loopback networking,
  or supported platforms must run in those real environments; a mock or
  skipped case is not evidence of enforcement.
- Resolve provisioning, bootstrap/session credentials, allowed deployments,
  ACL repair versus refusal, operator consent and availability when the
  authority service is offline before approving implementation. Exhaustive
  direct-resource and optional-path discovery remains open.

### Further execution-path evidence

| Path | Observed behavior | Foundation requirement / missing proof |
|---|---|---|
| Streamlit session -> live turn | [`dad_streamlit.py`](dad_streamlit.py#L928) stores a bot in Streamlit session state and [`_process_user_message`](dad_streamlit.py#L941) calls that bot directly. An API-only authorization guard would not cover this path. | Bind an authenticated browser session to the same trusted execution/authorization contract before direct turns and effects. Negative test: unauthenticated or wrong-tenant browser cannot cause a turn or mutation, even when the API is bypassed. |
| Boot -> local asynchronous effects | [`boot_mixin.py`](dadbot/core/boot_mixin.py#L436) submits graph refresh and proactive heartbeat; [`BackgroundTaskManager.submit`](dadbot/background.py#L43) executes passed callables in a thread pool with optional arbitrary metadata. [`runtime_orchestration.py`](dadbot/managers/runtime_orchestration.py#L258) also submits tracked work. | Establish service principal, tenant binding and policy revalidation at submission and effect boundaries; test both bootstrap and dynamically submitted work, including restart and revocation. |
| CLI -> local mutation/export | [`app_runtime.main`](dadbot/app_runtime.py#L1127) constructs a bot and can clear or export memory directly. Operator commands can run before normal startup validation ([`app_runtime.py`](dadbot/app_runtime.py#L1077)). | Check authorized OS operator and specific operation permission at all CLI and operator entry points, not only the API. Negative tests must use an actually different account on each supported OS. |
| Configuration -> network authentication | [`validate_startup_environment`](dadbot/runtime/env_loader.py#L97) validates a service token secret only when API auth is configured as required; [`_resolve_principal`](dadbot_system/api.py#L177) provides an anonymous default-tenant admin principal when it is disabled. `--api-host` can override the default bind host. | Prove non-loopback/external interfaces cannot start or serve requests without authentication and authorization, including configuration overrides. Test the effective listener and actual requests, not only a config object. No deployed exposure is inferred. |

These observations add concrete foundation integration targets; they do not
establish a complete call graph or determine which OS-level mechanism should
be used. No new runtime or acceptance tests have been executed.
