# Engineering invariants and remediation gates

Status: **in progress; not a complete whole-repository audit**.

The tenant identity and execution-path investigation, including its scope and
remaining unexamined paths, is in
[`TENANT_EXECUTION_PATH_AUDIT.md`](TENANT_EXECUTION_PATH_AUDIT.md).

The worktree contains substantial pre-existing changes. Do not commit or push
without a separate review. Results below describe this worktree and are not a
production-readiness certification.

## Required invariants and order

The user's priority order is:

1. **Tenant isolation and authority.** A verified tenant identity must select
   the same isolated authority and projections throughout ingress, worker
   execution, storage, retrieval, migration, recovery and shutdown. Another
   tenant must not read, overwrite, clear, import or infer its data. Invalid
   or ambiguous ownership must fail closed. Existing single-user operation
   must remain available without silently assigning ambiguous legacy data to
   a new tenant.
2. **Authentication and authorization.** Every ingress and privileged action
   must bind authenticated identity, tenant and scope to execution; failure
   must deny access, including asynchronous and resumed work.
3. **Durable state and ledger authority.** Committed state must have one
   authoritative history per tenant. Repair must not fabricate or undo a
   committed event, and uncertainty must be surfaced.
4. **Atomicity, recovery and projections.** Projection publication must be
   fenced by authoritative revision and generation, atomically verified,
   tenant-scoped and recoverable after interruption.
5. **Lifecycle and concurrency.** Initialization, shutdown and concurrent
   access must preserve authority, resource ownership and operation ordering.
6. **Deterministic execution and replay.** Replaying a recorded execution
   must preserve its declared deterministic outcomes without re-executing
   unrecorded side effects.
7. **Memory retrieval and semantic correctness.** Results must be current,
   tenant-scoped, explain degradation, and never resurrect removed records.
8. **API, tool and integration contracts.** All public and internal entry
   points must preserve their contracts, authorization and failure semantics.
9. **Observability and operational resilience.** Health and telemetry must
   accurately reflect degraded capabilities without exposing sensitive data.

An invariant is complete only when its application-level acceptance tests
pass without xfail, including restart, concurrency and failure cases where
applicable. Tests must verify persisted behavior, not only helper functions.

## First gate: tenant isolation — not met

### Confirmed application-level failure

[`tests/unit/test_tenant_isolation_acceptance.py`](tests/unit/test_tenant_isolation_acceptance.py)
starts separate real DadBot processes for two different tenants, using one
local persistence root and synthetic data. Tenant B imports tenant A's memory.
Running the test with `--runxfail` produces **1 failed**; its strict xfail
records the unresolved gate, not a passing acceptance result.

The demonstrated path is:

1. [`DadBotConfig`](dadbot/config.py) derives memory, profile, semantic, graph
   and session paths from global configuration, not tenant identity.
2. [`DadBotTaskProcessor`](dadbot_system/worker.py) passes tenant identity to
   each bot but does not partition those local paths.
3. [`load_memory_store`](dadbot/memory/storage.py) can establish a new tenant
   by importing the unowned shared JSON mirror; snapshot repair also writes
   that shared mirror.
4. [`SQLiteSemanticIndex`](dadbot_system/semantic_index.py) uses a global
   `summary_key` primary key and unscoped reads/deletes/clear. The graph store
   similarly uses global keys and replacement deletes all graph rows
   ([`graph_store.py`](dadbot_system/graph_store.py)).

Ledger events are tenant-labelled, but that does not make the legacy import,
mirrors, or indexes tenant-isolated. Normalized tenant identifiers can also
collapse distinct input strings; their identity and migration contract needs
explicit review. Namespaced session/document stores do not by themselves
establish isolation across the other paths. This map is incomplete: backups,
profile and session logs, PostgreSQL, caches, exports, tools, web and
background paths must still be traced before claiming the gate is covered.

### Shared mechanism and acceptance work required

- Define one validated tenant identity and resource-ownership boundary shared
  by ingress, workers, persistence and recovery. Reject ambiguous identities
  and ownership rather than guessing; preserve compatibility through an
  explicit, recoverable legacy-data migration/ownership protocol.
- Isolate all tenant-bearing stores, including legacy imports, mirror and
  backup writes, profile, semantic and graph indexes, caches, session logs,
  restore/clear/delete and administrative access. Support shared physical
  indexes only with tenant-scoped keys and every operation scoped; otherwise
  fail closed rather than pretending a tenant filter suffices.
- Coordinate ownership establishment and migration across processes. Test
  crashes and simultaneous claimants without overwriting existing data.
  Preserve existing single-user availability without implicitly handing
  unowned data to an arbitrary new tenant.
- Acceptance tests must use two real tenants and fresh processes: initial
  hydration, identical keys, retrieval, add/update/delete/clear/restore,
  rebuild, concurrent repair, restart, legacy migration and unauthorized
  access. Verify both logical records and physical persisted state. The
  indexed-mutation safeguard remains until its separate recovery gates pass.

Until these gates pass, do not deploy the current shared local data root as a
multi-tenant isolation boundary. The failing test does not establish that
separate directories, PostgreSQL, or any other path is safe.

## Baseline and additional findings

- Test collection: **4,104 tests** collected. No full-suite result is claimed.
- Selected tool registry, fault handling, API integration, egress and import
  firewall run: **175 passed**. Before the user redirected the work, regression
  tests reproduced and fixes were made for four builtin executor name errors,
  probabilistic fault injection and permission-error classification. These
  do not satisfy any system-wide invariant.
- Additional security/startup/durability selection: **123 passed, 3 failed**.
  Failures: the legacy turn-entrypoint freeze test and two manager-smoke tests
  reporting a missing `parse_tool_command`.
- CI production-confluence flags check passed; the kernel boundary validator
  reported **7 violations** involving telemetry and trace-export imports.
  Investigate intended boundaries before modifying either code or gate.
- Ruff across `dadbot` and `dadbot_system` reported **1,797 diagnostics**
  before the fixes; F821/E9 and other F-family diagnostics require a separate
  prioritized baseline, not a mechanical mass cleanup.
- Authentication defaults to enabled, but configuration can disable it and
  give the anonymous principal admin scope
  ([`api.py`](dadbot_system/api.py),
  [`env_loader.py`](dadbot/runtime/env_loader.py)). Determine intended
  deployment policy and assess all entry points before changing it.
- The previous D2.6 state and validation are recorded in
  [`DURABLE_MEMORY_IMPLEMENTATION_STATUS.md`](DURABLE_MEMORY_IMPLEMENTATION_STATUS.md).
  Durable projection metadata and cross-tenant projection isolation remain
  unimplemented.

## Subsequent gates

After tenant isolation is proven, trace authorization on HTTP, WebSocket,
workers and tools; then establish durable append/recovery and projection
metadata gates. Continue in the invariant order above. Each gate needs an
explicit path inventory, negative and crash tests, shared implementation,
focused and broader regression runs, and evidence from fresh processes.
Checkpoint retention and PostgreSQL application integration are outside the
previously authorized milestone scope and are not implemented here.
