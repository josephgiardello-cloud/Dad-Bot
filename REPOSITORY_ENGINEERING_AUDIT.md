# Dad-Bot Repository Engineering Audit

**Inspection date:** 2026-11-09; follow-up verification: 2026-11-10
**Mode:** Repository audit plus bounded follow-up implementation
**Scope:** Current checkout, including uncommitted changes
**Verdict:** The latest worktree and clean-clone `tests/unit` runs are green; an immediately preceding worktree run reproduced the characterized intermittent checkpoint/cache failure. Multiple product surfaces and release gates remain incomplete. Overall status is **PARTIAL**; this is not a production-readiness certification.

## A. Executive engineering assessment

Dad-Bot is a Python 3.13 conversational assistant with a Streamlit user interface, a FastAPI service, an Ollama-backed model path, tenant-aware task orchestration, memory/knowledge projections, and a durable execution path. Its core has real components rather than being only a scaffold: API requests reach delegated tasks and worker execution; memory has graph and semantic backends; file-WAL and SQLite/PostgreSQL/Redis persistence paths exist; replay, recovery, and authority have meaningful tests.

The engineering risk is the gap between this architectural breadth and demonstrated, user-facing completeness:

- The working tree is materially modified. Findings describe this checkout and cannot be attributed to a clean commit baseline.
- The original inspection collected 4,336 tests. The latest post-push worktree run and the clean-clone run each completed with **1,961 passed, 10 skipped, 0 failed**. An immediately preceding worktree run had **1 failed, 1,960 passed, and 10 skipped** in the known source-grounding checkpoint/cache test; this intermittent failure is characterized but its underlying cause remains open. The phase-isolated staged Phase 0 run was also red (**50 failed, 1,856 passed, 10 skipped**), showing the increments are not independently green.
- Runtime composition is split across launchers, control planes, kernel services, and several persistence authorities. Some integrated features coexist with simpler stubs or alternate paths.
- The model path is effectively Ollama-only; several advertised tool and voice capabilities are not end-to-end product workflows; retrieval can degrade to empty results.
- Operational checks are weaker than the system complexity warrants: `/health` is liveness-only; `/ready` covers state, authority, workers, and opt-in model/vector probes, but coverage thresholds are absent and the CI Ruff gate is narrow.

No P0 issue was established in the original audit. The source-grounding checkpoint/cache failure remains intermittent: ten fixed-order trials did not reproduce it; one full worktree run reproduced it; the next full worktree run and the clean-clone run passed. The underlying divergence cause remains unlocated and is tracked as open.

## Roadmap execution update

The original audit was read-only. The user subsequently authorized bounded work
following its roadmap. The increments are combined in one commit because the
staged Phase 0-only snapshot failed while the complete worktree contains
cross-increment dependencies. This is not a claim that the increment commits
are independently reproducible.

| Phase | Status |
|---|---|
| 0 — Baseline | The stale-contract updates retain their intended assertions and the tenant-qualified snapshot fix remains covered. However, the staged Phase 0-only full-unit run was red (50 failed, 1,856 passed, 10 skipped) because it omitted prerequisites and production/test-support changes from later increments. The phase-isolated commit is therefore not independently green; the combined state is the verification unit. The source-grounding/checkpoint failure remains open. |
| 1 — Runtime composition | Three bounded increments verified. `launch.py`, `Dad.py`, and `api_entrypoint.py` converge on `dadbot.app_runtime.main`; an import-linter contract forbids `app_runtime` from importing `dadbot.runtime.launcher`. `/health` is liveness; `/ready` checks state/authority dependencies, worker status, and optionally model/vector probes. Authenticated workers quarantine during authority outage, hold already-dequeued work, and resume after recovery with capped backoff. Model/vector readiness remains opt-in and does not change current Compose behavior. Broader database-failure recovery and task/WAL/effect durability mapping remain open. |
| 2 — Tool capability truthfulness | Two bounded increments verified. The authorized durable `set_reminder` workflow remains available; fake reminder/calendar/web-search defaults were removed from the legacy registries. Registry/planner, kernel replay, and MCP execution remain disjoint. |
| 3 — AI and memory | Implemented and focused-verified. Configured context is now sent exactly; test doubles match the runtime contract; retrieval-only APIs fail explicitly on partial/degraded results; turn status, optional memory-backend readiness, and bounded `/doctor` counts expose retrieval health. SQLite and PostgreSQL semantic-index connections can be probed; graph persistence and non-semantic memory projections remain unprobed. |
| 4 — Browser interaction | In progress. A real local-model chat turn passed in the browser. The canonical `launch.py --ui` path returned HTTP 200 and binds only to loopback, but an earlier 30-second `Invoke-WebRequest` probe timed out despite the server reporting startup; the client discrepancy, session continuity, failure handling, and accessibility remain unverified. |
| 5 — Maintenance overhead | In progress. Added grouped API readiness checks and an admin-protected bounded `/doctor` report; broader maintenance deliverables remain open. |

### Phase 0 disposition

The latest complete command was
`.\.venv\Scripts\pytest.exe --basetemp=$base -o "addopts=" -q --tb=short tests\unit`,
with `$base` set to the isolated directory
`C:\Users\josep\AppData\Local\Temp\dadbot_stale_contract_full_20261110` and
the log captured outside it at
`C:\Users\josep\AppData\Local\Temp\dadbot_stale_contract_full_20261110.log`.
Pytest collected 1,961 tests and finished with **0 failed, 1,951 passed, and
10 skipped** in 212.72 seconds pytest time (215.33 seconds wall-clock). Relative
to the prior complete run (**33 failed, 1,918 passed, 10 skipped**), the
failure count fell by 33 and the pass count rose by 33; the skipped count was
unchanged. This is a dirty-working-tree result, not a clean-commit baseline.
At this point in Phase 0 the historical source-grounding/checkpoint order
failure did not recur; later Phase 1 runs and its separate disposition are
recorded below.
Failure delta: **33 previously failing nodes now pass; 0 failures remain**.
The temporary basetemp and captured output log were removed after recording
the result; rerunning the command recreates them.

- **Reproduced regressions fixed:** background tasks were losing delegated
  `ContextVar` authority in thread-pool workers; authenticated WALs could be
  reopened as unbound local state after binding sidecars were removed. The
  targeted suites passed **10** and **19** tests, respectively.
- **Likely stale test fixtures, not confirmed:** a cluster of test doubles
  omits the runtime configuration now required by production code. Update only
  after confirming the contract rather than changing production behavior to
  satisfy incomplete fixtures.
- **Environment/policy-related candidates:** subprocess tests using the
  default tenant fail when the configured nonproduction tenant gate is absent.
  `test_sqlite_service_state.py` now passes an explicit test-only
  `DADBOT_ALLOW_LEGACY_DEFAULT_TENANT=true` to spawned children; no production
  gate was relaxed.
- **Phase 0 failure resolution:** all 33 stale-contract nodes from the
  33-failure baseline were updated in assertion-preserving batches documented
  in Section F. The latest full unit run is green (1,951 passed, 10 skipped).
  The separate tenant-qualified snapshot serialization defect remains fixed
  and covered by its round-trip test. No authorization, ownership,
  SaveNode, or fail-closed requirement was weakened.
- **Confirmed startup race repaired:** two local processes starting against
  the same WAL could race while recording projection-recovery status and one
  would fail with `LedgerHeadConflict`. Recovery-status writes now reload and
  retry only that non-session status event on a stale head. The
  three concurrent-startup cases pass, and the 41-test WAL-integrity and
  projection-status selection passes; direct stale-writer rejection remains
  intact.
- **Confirmed source-grounding retry defect repaired:** retry trace detection
  derived the candidate job ID without the tenant, while `ExecutionJob` hashes
  tenant, session, and trace. The terminal failed job was therefore not
  recognized, so a retry reused its terminal identity and the scheduler
  rejected it. Candidate identity is now tenant-qualified; the full
  `tests/unit/test_source_grounding.py` selection passes **26 tests**; the
  tenant-session/execution-order regression selection passes **11 tests**.
- **Phase 0 category refresh:** the combined selection
  `tests\unit\test_phase1_chat.py`,
  `test_local_authority_mutation_application.py`,
  `test_sqlite_service_state.py`,
  `test_model_context_budget.py`,
  `test_local_tenant_establishment_application.py`, and
  `test_response_engine_control_plane_integration.py` finished with **14
  failed, 127 passed, 0 skipped** in 111.12 seconds. The phase-one chat,
  SQLite service-state, model-context-budget, and local-tenant-establishment
  files had no failures in this run. Failures were confined to local authority
  mutation (9 in this run) and response/control-plane integration (5).
- **Failure dispositions from observed traces:**
  - **Stale/unsafe test setup, authorization intentionally preserved:** memory
    startup, snapshot-write, migration, and semantic-repair tests supply
    unowned legacy JSON snapshots. Startup rejects these with “no tenant
    ownership evidence”; tests need tenant-bound fixtures or an explicit
    migration contract, not relaxed ownership checks. Worker tests call the
    live worker processor without delegated authority and are likewise
    incompatible with the intentional fail-closed contract.
  - **Stale test boundary/contracts:** response-engine tests call the private
    submit finalizer directly, outside `KernelGateway` scope, and are rejected
    by `KernelBoundary`; graph trace tests omit the required `SaveNode` stage;
    snapshot tests use unqualified session IDs rather than tenant-qualified
    keys; facade tests use incomplete `SimpleNamespace` substitutes and
    assert object identity where the facade returns a projection.
  - **Environment-sensitive:** the explicit-runtime-services test omits the
    required non-default tenant. In the category-only run, a subprocess test
    also failed during `os.replace` with Windows `PermissionError` under the
    repository-local OneDrive temp path; it passed in the full run. This is
    evidence of a temp-file/filesystem interaction, not a product regression.
  - **Unresolved behavior/contract mismatch:** two memory tests expect a
    shared legacy JSON mirror to be updated alongside the ledger-backed
    tenant projection; one observed a stale value and another found no mirror
    file. The intended mirror guarantee needs a decision grounded in the
    durability contract before changing code.
  - **Order-sensitive failure still open:** the source-grounding approval and
    revocation cache test passed in the isolated 26-test file run but failed
    in the full run during strict SaveNode finalization with memory-authority
    divergence. Repeating
    `.\.venv\Scripts\pytest.exe --basetemp=.pytest_tmp_source_cache_triage_N -o addopts='' -q tests\unit\test_source_grounding.py::test_canonical_approval_and_revocation_invalidate_repeated_prompt_cache`
    twice, each with a fresh basetemp, passed both times (**1 passed** each).
    This confirms the failure does not reproduce in isolation, but does not
    identify the preceding test/state that triggers it. Do not classify as a
    stable product regression or stale test until that order-dependent state is
    isolated.
- **Test fixture corrected:** `test_build_creates_runtime_with_session_lock`
  previously relied on the production-forbidden implicit `default` tenant and
  failed in isolation. It now supplies an explicit test tenant and temporary
  state paths; the production tenant gate was not relaxed.
- **Targeted retest:** the affected context-budget, chat, tenant-establishment,
  SQLite service-state, and projection-status files passed **114 tests, with
  1 skip**. The focused authority/API/ownership selection passed **67 tests**.
- **Still unresolved and specifically concerning:** startup correctly hydrates
  the committed memory projection in one test, but its shared legacy JSON
  mirror remains stale. This needs tenant-ownership analysis before deciding
  whether it is a production defect or an obsolete test expectation. Snapshot
  replay behavior and run-to-run variance in the source-grounding cache test
  also need independent diagnosis.

This was the status at the time of the prior triage note. Subsequent
verification completed a full post-retry run and individually dispositioned
the then-current failures; the latest post-snapshot-fix counts and remaining
order-dependent failure are recorded in Section F below. Stale tests will be
corrected only when their intended contracts are confirmed.

### Phase 1 implementation and verification

- The canonical API composition is enforced as `Dad.py` and
  `api_entrypoint.py` → `launch.main` → `dadbot.app_runtime.main`, with an
  import contract forbidding `dadbot.app_runtime` from importing the alternate
  `dadbot.runtime.launcher`. Configured Redis or Postgres initialization
  errors now abort startup instead of silently selecting in-memory state;
  the no-backend local development fallback remains supported.
- Authenticated workers no longer dequeue tasks when revocation authority
  cannot be polled. Per-worker `starting` / `available` / `quarantined`
  status contributes to `/ready`, and manager shutdown signals workers without
  requiring an authority-dependent queue sentinel. `/health` remains a
  lightweight liveness/diagnostic route.
- State stores expose availability checks; `/ready` returns 503 when the state
  or authority store is unavailable or workers are not alive/authority-ready.
  Its response separates `dependencies` from `runtime` checks. Docker Compose
  uses `/ready`; deployment documentation states the durability boundary and
  canonical launch path.
- Added admin-scoped `/doctor`, with the existing admin rate limit. It reports
  readiness dependencies, tool-registry count, trace-provider state, and
  selected cached runtime-health indicators. Model/vector probe behavior
  follows the same opt-in, timeout, and cache policy as `/ready`; disabled
  probes are explicitly reported as `unprobed`, while
  pending-approval status is marked not exposed. The health-level runtime
  snapshot is read with `persist=False`.
- **Increment 3 — opt-in model/vector probes:** `/ready` always includes
  `model` and `vector` groups (`ok`, `degraded`, `unavailable`, or
  `unprobed`), with an ISO timestamp and error type only. Probes are disabled
  by default; `DADBOT_READINESS_PROBES=model,vector` enables checks and
  `DADBOT_READINESS_REQUIRED=model,vector` makes selected checks gate HTTP 200.
  `DADBOT_READINESS_PROBE_TIMEOUT_MS` is bounded to 5 seconds maximum, and
  `DADBOT_READINESS_PROBE_CACHE_SECONDS` defaults to 10 seconds. Concurrent
  requests share one in-flight probe per dependency. The model check calls
  Ollama `/api/tags` and verifies the configured active/default model;
  vector checks use `DADBOT_VECTOR_STORE_URL` as a PostgreSQL DSN (`SELECT 1`)
  or an HTTP(S) endpoint (`HEAD`). Disabled groups remain `unprobed`, so
  existing readiness behavior is preserved. Probe failures do not affect
  authority or worker quarantine decisions. Compose remains unchanged and
  does not opt in: this is deliberate until each production deployment has
  confirmed its model/vector endpoint configuration. For production, opt into
  the model probe when Ollama availability is a turn prerequisite; require the
  vector probe only where a vector endpoint is configured and required for
  that deployment's turn path.
- `tests/test_service_api_integration.py`: **29 passed, 0 failed, 0 skipped**
  after the readiness response change. Coverage includes state-store
  unavailability, readiness groups, admin-scope enforcement, registry and
  runtime diagnostics, and omission of unapproved runtime-snapshot fields.
- The increment 3 focused readiness selection passed **9 tests**; it covers
  default `unprobed` groups, model-name verification, vector HTTP and
  PostgreSQL connectivity, bounded timeout/error shape, concurrent cache
  sharing, environment configuration, required model up/down/up transitions,
  and admin diagnostics.
- The combined API/authority selection passed **82 tests**. The latest full
  `tests/unit` rerun passed **1,960 tests, 10 skipped, 0 failed** in 278.15
  seconds (281.67 seconds wall clock). Scoped Ruff import/undefined-name
  checks and `git diff --check` passed. The canonical app-runtime import
  contract remains **KEPT**; the two existing API-to-telemetry contracts
  remain **BROKEN** and were not changed.
  `tests/unit/test_service_verified_principal.py` and
  `tests/unit/test_delegated_authority_acceptance.py`, excluding the unrelated
  async-only worker test: **35 passed, 1 deselected**. Running that full
  selection first failed because the environment lacks `pytest-asyncio` for
  the test's `pytest.mark.asyncio`; the test is not counted as passed.
  Focused Ruff undefined-name checks, `compileall`, and touched-file
  `git diff --check` passed. Model/vector connectivity and approval status
  remain unverified/unavailable through this API.
- `tests/test_app_runtime_startup.py` and
  `tests/test_service_api_integration.py`: **55 passed**. Targeted Ruff
  undefined-name checks and `compileall` passed.
- This does not certify external-service readiness against live Redis/Postgres,
  a spawned production worker outage/restart, or production deployment. The
  full Ruff run and `git diff --check` still expose pre-existing findings,
  including the explicitly preserved trailing whitespace in
  `tests/unit/test_durable_submission_authority.py`.

### Phase 2 implementation and verification

- The deterministic reminder path previously reached its executor without
  checking the request's `allowed_tools` policy and without applying the
  reminder argument validation used by model-selected tools. Both sync and
  async routes now reject unauthorized or invalid reminder requests before
  recording an execution decision; the executor dispatch also rechecks tool
  authorization.
- The end-to-end regression runs a reminder under live runtime authority,
  proves an empty tool allowlist causes no reminder, permits `set_reminder`
  when explicitly allowlisted, then constructs a new bot over the same local
  memory file and replays the same trace. The persisted reminder ID is stable
  and only one reminder exists.
- `test_e2e_orchestrator_turn.py::test_e2e_orchestrator_full_runtime_lane_certifying_expected_tool_execution`
  and the authorization/restart/invalid-title reminder tests: **3 passed** in
  the last end-to-end selection; the narrower reminder regression selection
  also passed **4 tests**.
- This verifies the local file-backed trusted-local workflow and delegated
  authority in-process. It does not certify a live Postgres-backed reminder,
  production model quality, notification delivery to an external channel, or
  exactly-once behavior across arbitrary power loss after a notification.

### Phase 2, increment 2 — truthful tool surface

The legacy placeholder executors in `dadbot/tools/executors.py` and
`dadbot/core/services.py` returned success-shaped strings or dictionaries
without performing reminder, calendar, or web-search effects. Those
implementations were outside the authorized runtime tool pipeline. Both
default builders now register no placeholder executors; the legacy
`ReActPlanner` receives no tools. The structured definitions for `set_reminder`
and `web_search` remain because the separate runtime pipeline implements those
handlers; calendar is not advertised. Direct dispatch through the legacy
executor function now fails closed with “tool not found.”

| Legacy tool name | Registered before → after | Advertised before → after | Real executor / supported handler? | Approval / authority / audit? | Disposition |
|---|---|---|---|---|---|
| `reminder` (`set_reminder` definition) | Yes, both defaults → no placeholder registration | `set_reminder` definition and legacy planner name → definition remains for runtime pipeline; no legacy planner entry | Placeholder returned a string; runtime `set_reminder` has a real handler | Placeholder path: no; runtime path: allowlist, authority, recording/audit, local durable persistence tested | Remove placeholder; retain the independently dispatched authorized capability |
| `calendar` | Yes, both defaults → no | Planner-name prompt only; no structured definition → none | No; placeholder returned an action dictionary | No | Remove registration and planner advertisement |
| `web_search` | Yes, both defaults → no placeholder registration | Structured definition and legacy planner name → structured definition remains for runtime pipeline; no legacy planner entry | Placeholder returned a string; runtime web-search handler is separate | Placeholder path: no; separate runtime route is not broadened by this increment | Remove placeholder; retain only the separately implemented runtime capability |

Focused tests prove no placeholder is installed, advertised runtime definitions
map to handlers in `ToolPipelineMixin`, selecting removed names through the
legacy planner cannot execute an effect, and direct legacy dispatch fails.
The already-proven authorized reminder workflow remains covered. No new
executor or authorization abstraction was added.

The service `ToolRegistry`/`ReActPlanner`, core kernel tool-recording/replay
path, and MCP validated plan loop remain separate execution paths. Unification
is deferred to a separately scoped increment because this change only removes
false capability claims; merging dispatch, authority, replay, and MCP
semantics together would obscure the security and behavior delta.

Verification for this increment: full `tests/unit` **1,960 passed, 10
skipped, 0 failed** in 218.04 seconds (220.71 seconds wall clock); API and
authority selection **75 passed**; focused surface/reminder selection **17
passed**. Scoped Ruff import/undefined-name checks and `git diff --check`
passed. The canonical app-runtime import contract remains **KEPT**; two
pre-existing API-to-telemetry contracts remain **BROKEN** at
`dadbot_system.api -> dadbot.telemetry.otel_bridge`. The focused set includes
the positive authorized reminder workflow, unsupported direct dispatch and
planner selections, and the handler/catalog contract.

### Phase 3 implementation and verification

- Runtime evidence showed `CONTEXT_TOKEN_BUDGET=6000` produced Ollama
  `num_ctx=8192` because `_prepare_ollama_request` rounded the context
  allocation up to a power of two. It now sends the effective configured
  budget (still clamped to model capacity and explicit caller overrides).
  Pylance-selected-interpreter snippets observed `6000 → 8192` before and
  `6000 → 6000` after the fix.
- Eight context-test failures were caused by `_client()` test doubles omitting
  the required `runtime_config.ollama_keep_alive_seconds`; the fixture now
  provides it. This corrects the test contract without adding production
  fallbacks.
- `MemoryService.retrieve_for_query_with_status()` distinguishes an empty
  successful result (`ok`) from graph-only degradation (`partial`) and
  catalog/backend failure (`degraded`). Semantic-index and post-tool
  degradation is logged without exposing exception messages and is recorded
  in per-turn `memory_retrieval_status` with source and error type.
- The list-only `retrieve_for_query()` now raises `MemoryRetrievalError` for
  partial/degraded outcomes rather than returning a success-shaped empty list;
  empty successful results remain `[]`. Turn status retains contributing
  sources and error types. The opt-in `memory` readiness probe performs a
  bounded, cached `SELECT 1` through the configured SQLite/PostgreSQL semantic
  index connection and does not invoke embedding generation. `/doctor` reports
  counts for the last 100 in-process status observations (`ok`/`partial`/
  `degraded`) without retrieval payloads or exception strings. Graph storage
  and non-semantic memory projections remain configured-but-unprobed. Probe
  connectivity was verified with an injected backend; no live PostgreSQL or
  production memory backend outage was exercised.
- Increment verification: `tests/unit` passed **1,960**, skipped **10**;
  focused memory/readiness selection passed **24**; API/authority regression
  selection passed **146**, skipped **2**, with one unrelated
  `test_contract_single_callable_output_authority` test deselected after it
  failed on a missing `LIGHT_MODE` attribute in its test double. Scoped Ruff
  reported **31 existing findings** in the touched files, none on the newly
  added lines; no mass autofix was applied. Task-scoped `git diff --check`
  passed. The repository-wide check still reports unrelated trailing
  whitespace in `tests/unit/test_durable_submission_authority.py:670`.
- `tests/unit/test_model_context_budget.py` and
  `tests/unit/test_memory_retrieval_efficiency.py`: **33 passed**. This includes
  the formerly failing context budget suite and the empty-versus-failure
  retrieval tests.
- No live vector backend outage was exercised; retrieval errors were injected
  in focused tests. A live Ollama call was subsequently exercised through the
  browser; see Phase 4.
- Durable conversational memory now uses strict Pydantic fact records with an
  exact user-message quote as evidence. Unsupported quotes and malformed
  legacy-shaped extraction output are rejected; code assigns source, extractor,
  bounded trust level, and the turn trace ID. Provenance survives memory
  normalization, and PII scrubbing covers both the fact and its evidence.
  Focused extraction, provenance, normalization, and PII tests passed **6
  tests**. The broader related memory/model/retrieval selection passed **118
  tests; 13 were skipped**. Trust level describes source attribution, not
  objective truth. Extraction was tested with controlled model responses; a
  live model-backed memory extraction and retrieval cycle remains unverified.
- Tool-step working memory now uses a dedicated SQLite database under the
  configured session-log directory (overridable with
  `DADBOT_WORKING_MEMORY_DB_PATH`). Successful, non-replayed tool results are
  stored with tenant/session/trace/step keys, PII scrubbing, a 32 KiB payload
  limit, and a one-hour maximum TTL. Only opaque random references created for
  validated user facts saved by the memory coordinator can mark those rows
  promoted, and that marking occurs from `commit_transaction()` after SaveNode
  finalization. Raw tool
  results remain short-lived and are not copied into episodic memory.
- `tests/test_working_memory.py`, `tests/test_memory_coordinator_pii.py`,
  `tests/test_execute_tool_recording.py`, `tests/test_post_commit_events.py`,
  and the certifying reminder test in `tests/test_e2e_orchestrator_turn.py`:
  **31 passed**. Coverage includes TTL expiry across store reconstruction,
  tenant/session/trace separation, PII scrubbing, failed/denied/cached/replayed
  exclusion, post-commit-only promotion, all four sync/async reminder and
  web-search call paths, and an end-to-end reminder turn. The four-path test
  uses controlled tool results; the reminder E2E uses the local runtime lane.
- Ruff passed for the new store and its tests; Ruff undefined-name checks,
  `compileall`, and `git diff --check` passed for all changed implementation
  and test files. Live model-backed fact extraction and multi-process
  concurrency against the new SQLite database remain unverified.
- The canonical `execute_tool()` spine now emits a JSON-formatted audit log
  for live, failed, cached, replayed, and authority-denied calls. Records carry
  UTC timestamp, tool name, tenant/session/turn trace, authority mode and
  decision, sanitized input parameters, and input/output hashes; authenticated
  principal subjects are represented by a stable hash. Sensitive parameter
  names, common PII, oversized values, and oversized records are redacted or
  bounded. The generic execution spine emits an opt-in `dadbot.tool.execute`
  span, while the actual Home Assistant MCP transport emits `mcp.call_tool`
  spans with `mcp.tool.name`, `mcp.server.name`, `mcp.method.name`, and result
  status. Parameter and result payloads are excluded by the exporter allowlist.
- `tests/test_execute_tool_recording.py` and
  `tests/test_observability_bridges.py` verify structured success/denial logs,
  redaction, pseudonymous authenticated identity, and exported OTel semantic
  attributes without private tool arguments/results. Combined with the working
  memory, extraction, post-commit, and end-to-end reminder selections, the
  focused command executed **51 tests: 51 passed, 0 failed, 0 skipped**.
  Optional OTel and MCP test dependencies were installed in the local Python
  environment; no project dependency manifests were changed. Export remains
  opt-in, and live production log collection/export configuration is not
  verified here.

### Phase 4 implementation and verification

- Ollama exposed local models, including `qwen3.5:9b`. With temporary profile,
  memory, semantic/graph, session-log, and Streamlit-thread paths, the actual
  `dad_streamlit.py` UI started and its `/_stcore/health` endpoint returned
  **200**.
- Browser automation submitted `What is 2 + 2? Reply with just the number.`;
  the UI displayed the user turn, loading state (`Dad is thinking...`), and
  the local model's response (`4`). This verifies one real input-to-response
  path, not session restoration or reliability under dependency failure.
- The attempted `launch.py --ui --model qwen3.5:9b` startup did not expose a
  listening UI during the observed readiness window and was stopped; its
  cause remains undiagnosed. The successful browser run used the direct
  Streamlit entrypoint, so the wrapper launch path is not certified.
- The browser console also reported `Recording error: Container not found`
  from the voice waveform component. Chat still completed; voice capture,
  disconnected-service behavior, session continuity, keyboard/accessibility,
  PWA/offline behavior, and automated browser regression coverage remain
  unverified. Phase 4 remains in progress.

## B. Repository inventory and architecture map

### Composition

| Area | Contents and observed role |
|---|---|
| Application | `dadbot/` contains the assistant, runtime, managers, memory, UI, tools, services, telemetry, and persistence code (509 tracked files). |
| Service/control plane | `dadbot_system/` contains FastAPI, orchestration, workers, authority, task/state stores, events, and local MCP server (24 tracked files). |
| Verification | `tests/` contains 415 tracked Python test files across unit, integration, adversarial, chaos, replay, stress, UI, evaluation, and distribution-readiness categories. |
| Operational/configuration | Root includes `Dockerfile`, Compose files, `pyproject.toml`, `uv.lock`, `pytest.ini`, `.github/workflows/`, and Docker deployment docs. |
| Supporting/legacy | `archive/` has archived docs/tools/audits; `tools/`, `ci/`, and `evaluation/` contain development and validation support. The setuptools include list names `caaf*`, but no tracked `caaf/` package was found. |
| Entrypoints | `Dad.py`, `launch.py`, `api_entrypoint.py`, `launch_streamlit.py`, and the `dadbot` console script converge on or expose the application runtime. `dadbot/runtime/launcher.py` also contains an alternate API composition path. |

### Primary execution/dependency map

```text
Dad.py / launch.py / api_entrypoint.py / console script
    -> dadbot.app_runtime.main
       -> Streamlit UI OR FastAPI + local broker + event bus/state + orchestrator
          -> API lifespan recovery and scheduler
          -> API request -> tenant/session/task store -> broker
             -> worker -> authority verification -> DadBot.execute_turn
                -> prompt/model/runtime -> memory and tool/effect paths
                -> task result -> validation/publication/events

DadBot/core turn and persistence path
    -> conversation persistence -> ExecutionLedger / file WAL
    -> memory manager -> graph and semantic projections

Optional/configured integrations
    -> Ollama (active model path), PostgreSQL/Redis (service or selected stores),
       Chroma, optional LiteLLM, voice libraries, MCP, telemetry, reranker/Torch
```

The service task/state database, event-durability store, core execution ledger/WAL, and memory graph/semantic projections are distinct persistence concerns. They should not be treated as one atomic store. `LocalMultiprocessBroker` and the in-memory event bus are local-process infrastructure, not a durable cross-host queue.

### Supported versus optional evidence

- The normal application path is Python/Streamlit or FastAPI; the live service composition requires its configured dependencies, including PostgreSQL for service state in the production startup path.
- Ollama is the only provider shown on the active `LLMCallAdapter` path. `litellm` is an optional dependency, but no second active provider implementation was identified.
- Redis, PostgreSQL, Chroma/PGVector, MCP, voice, and reranker dependencies are optional or configuration-dependent in at least some runtime modes.
- The repository has Docker/Linux operational support. Windows-specific operational deployment was not demonstrated by the inspected CI/configuration.

## C. Runtime wiring report

| Path | Observed wiring | Status and consequence |
|---|---|---|
| Application launch | `Dad.py`, `launch.py`, `api_entrypoint.py`, and `dadbot/cli.py` reach `dadbot.app_runtime.main`; default launch is Streamlit, while API mode builds broker, state/event components, orchestrator, workers, and FastAPI (`dadbot/app_runtime.py:941-1074`). | **Reachable, with competing composition.** `dadbot/runtime/launcher.py:310-359` is a second API launcher; it is not the path selected by `launch.py`. Configuration and lifecycle behavior can diverge. |
| API/task path | API lifespan recovers incomplete tasks, starts scheduler execution, and shuts services down (`dadbot_system/api.py:448-518`). Requests call `orchestrator.submit_chat()` and enter tenant-scoped task persistence and broker execution (`dadbot_system/orchestration.py:144-223`). | **Implemented and tested in focused paths.** The broker itself is local; cross-host queuing is not supplied by that component. |
| Worker/live execution | Worker verifies delegated authority, loads the tenant/session/model runtime, invokes `bot.execute_turn`, signs the result, and returns it (`dadbot_system/worker.py:360-444`). Orchestration validates result/task/request identity before applying it (`dadbot_system/orchestration.py:318-414`). | **Connected, but broad unit verification is red.** Task recovery does not itself prove exactly-once external effects across every effect boundary. |
| Authenticated authority | Startup requires configured authentication, token secret, durable authority storage, and an enforcing orchestrator (`dadbot_system/api.py:154-191`). Requests bind principal/tenant context; workers verify delegation before execution (`dadbot_system/api.py:220-257`; `dadbot_system/worker.py:367-383`). | **Fail-closed behavior is real and focused-tested.** Authority loss denies work and evicts cache. The worker loop does not act on the boolean returned by revocation polling (`worker.py:467-480`), so availability/quarantine behavior merits explicit design. |
| Core turn/checkpoint | Graph checkpoint calls flow through `PersistenceService.save_graph_checkpoint` → `ConversationPersistenceManager.persist_graph_checkpoint` → `ExecutionLedger`. `AbstractCheckpointer` remains optional injection; no normal production constructor was found. | **Core ledger path connected; remote checkpoint parity is opt-in.** Trusted-local PostgreSQL/Redis adapters are not automatically wired into authenticated application execution. |
| Memory and retrieval | `MemoryManager.initialize()` creates graph, semantic, normalizer, storage, and lifecycle components (`dadbot/memory/manager.py:103-121`); `MemoryService` combines graph and memory retrieval (`dadbot/services/memory_service.py:20-53`). | **Substantially connected, with failure opacity.** Retrieval errors can be swallowed and represented as empty results, making backend failure indistinguishable from no knowledge. |
| Tool choice and execution | Service registry installs a built-in `ToolRegistry`; kernel tool recording/replay uses a separate execution spine. MCP has a separate validated plan loop. | **Partial/disjoint.** The simple registry is not clearly the same path as kernel tool execution. Built-in calendar/reminder executors are placeholders and the advertised definitions omit a registered calendar tool (`dadbot/tools/executors.py:12-29,34-69`). |
| MCP | Local MCP server is optional and separately launched; MCP HTTP client and Home Assistant configuration are separate surfaces (`dadbot_system/local_mcp_server.py:17-57`; `dadbot/runtime/mcp/tool_client.py:118-160`). | **Implemented but not default conversational wiring.** Local MCP exposes broad local capabilities; the complete authenticated approval/runtime path for each capability was not established. |
| Voice/multimodal | Voice UI delegates to `MediaService` (`dadbot/components/voice.py:7-18`). The captured-transcript path now calls the canonical conversational runtime and can synthesize local Piper sentence chunks (`dadbot/ui/media/service.py`; `dadbot/speech/voice_loop.py`). `MultimodalManager` accepts existing transcripts/notes for audio; images use model-name heuristics (`dadbot/managers/multimodal.py:17-49,70-90`). | **Partial.** The one-turn browser capture → local transcription → runtime → Piper response path is wired and unit-tested. Playback is assembled before rendering; VAD-based turn boundaries, incremental browser playback, full-duplex barge-in, and live microphone validation remain unverified. Image capability is still heuristic. |
| Operational health | `/health` is process liveness; `/ready` checks state, authority, worker status and optional model/vector probes (`dadbot_system/api.py`); Compose uses `/ready` (`docker-compose.yml:91-93`). | **Readiness is opt-in for model/vector.** Disabled external probes report `unprobed`, not healthy; explicitly required probes gate readiness while preserving worker authority checks. |

## D. Feature implementation and wiring matrix

Status terms follow the requested definitions. “Verified functional” applies only to the behavior named in its evidence, not to the subsystem as a whole.

| Subsystem / functional area | Key files and runtime entry | Dependencies; actual effects | Wiring status | Implementation status | Test evidence | Defects / confidence | Next action |
|---|---|---|---|---|---|---|---|
| App composition and API | `Dad.py`, `launch.py`, `api_entrypoint.py`, `dadbot/app_runtime.py`, `dadbot/runtime/launcher.py`, `dadbot_system/api.py` | Streamlit/FastAPI, config, PostgreSQL/Redis in service modes; starts workers and routes requests | **Canonical path enforced:** all public entrypoints reach `launch.main` → `dadbot.app_runtime.main`; import contract prevents `app_runtime` from importing the alternate launcher | Implemented for the tested launch paths | Entrypoint and import-contract checks pass; API readiness tests pass | Alternate launcher module remains present and can be invoked separately; medium | Keep alternate path out of canonical runtime; test API/UI startup and shutdown against the supported path |
| Control plane, tasks, workers | `dadbot_system/orchestration.py`, `worker.py`, `state.py`, `owned_tasks.py` | Broker, task/state store, authority, model runtime; enqueue, execute, publish/recover | **Complete for tested local/service scenarios**, not cross-host queueing | Functional with limits | Prior live task integration: 30 passed; current API/authority selection: 71 passed; latest full unit rerun green | Local broker is not durable/distributed; high | Separate queue guarantees from persistence guarantees and prove crash/retry/effect semantics |
| Identity and authorization | `dadbot_system/api.py`, `delegation.py`, `execution_identity.py`, `tenant_context.py`, `worker.py` | Auth principal, durable authority store, signed delegation; gates live effects | **Connected and fail-closed** on authenticated worker path | Verified for focused boundary behaviors | API/authority selection: 71 passed, including delegation, revocation, tenant cases, and queued-work hold/recovery | Worker quarantine is process-local readiness, not a distributed queue guarantee; medium-high | Preserve explicit quarantine/hold/recovery behavior and monitor worker readiness |
| Core turns, ledger, replay | `dadbot/core/`, `dadbot/services/persistence.py`, `dadbot/core/persistence/` | Model/runtime, execution ledger/file WAL, event and projection stores | **Connected but multi-path** | Partial-to-functional depending on path | Replay/durability suites exist; latest full unit rerun is green, but the source-grounding checkpoint/cache case has intermittent malformed-envelope failure evidence below | Separate authority/projection lifecycles create consistency burden; checkpoint failure cause remains open; high | Map transaction boundaries and diagnose the intermittent divergence-checkpoint envelope failure |
| Model and prompt path | `dadbot/services/llm_call_adapter.py`, `dadbot/core/llm_mixin.py`, `dadbot/managers/runtime_client.py` | Ollama client and configured model candidates | **Connected; provider breadth partial** | Functional Ollama path; no active second provider found | Model/context and phase-one chat tests pass in the latest unit run | Duplicate/legacy-looking model paths and no active second provider; high | Establish one provider boundary; add a real second provider only if required |
| Memory and knowledge | `dadbot/memory/manager.py`, `graph_manager.py`, `semantic_manager.py`, `dadbot/services/memory_service.py` | SQLite; optional Postgres/PGVector; graph/vector projections and CoreState | **Substantially wired**, fallback-dependent | Functional with partial failure signaling | Vector, graph, PGVector, restart, and current authority tests exist | Empty-result fallback can conceal retrieval failures; multiple graph/memory abstractions; high | Surface degraded retrieval, add end-to-end source/citation checks, consolidate authority versus projections |
| Tools and external integrations | `dadbot/tools/`, `dadbot/core/tool_executor.py`, `dadbot/tool_registry_builder.py` | Tool registry, kernel executor, local APIs | **Partial/disjoint** | Some real tools; built-in calendar/reminder placeholders | Registry, filesystem, HTTP, replay tests | Tool registry/planner/kernel are competing paths; placeholder executor may appear usable; high | Choose one execution spine and wire one concrete integration through authorization, approval, effect, result, and audit |
| MCP | `dadbot_system/local_mcp_server.py`, `dadbot/runtime/mcp/tool_client.py` | Optional MCP package, remote MCP, Home Assistant configuration | **Optional, separate surface** | Implemented in selected paths | MCP client/local tool tests | Not shown as default conversational runtime; access-policy parity not established; medium-high | Define supported launch/auth/approval model and add end-to-end tests for every exposed capability |
| Voice and multimodal | `dadbot/components/voice.py`, `dadbot/ui/media/service.py`, `dadbot/speech/voice_loop.py`, `dadbot/managers/tts.py`, `dadbot/managers/multimodal.py` | `faster-whisper`, `streamlit-webrtc`, Piper executable/model; transcript/image payloads | **Connected for captured one-turn interaction** | Partial; local TTS requires a configured model | `tests/test_voice_loop.py`, `tests/test_voice_loop_integration.py`, `tests/test_voice_control_plane.py`, `tests/test_oral_history.py`: **26 passed**. Piper readiness and synthesis now share executable discovery and readiness checks the configured model file exists. Streamlit smoke tests timed out while copying/starting the app. | No VAD segmentation or incremental browser playback; controller cancellation is not connected to active audio playback; live mic path not verified; high | Add VAD-driven capture and incremental audio playback with acoustic barge-in, then validate in a browser with a real microphone and Piper model |
| Streamlit/PWA UX | `dad_streamlit.py`, `dadbot/ui/`, `static/` | Streamlit, browser storage/assets, API/WebSocket/Redis for selected flows | **Streamlit reachable; PWA behavior partial/unverified** | Partial | Python smoke/contract tests; no browser automation found | 1,333+ line UI monolith; no service-worker lifecycle found; high | Add browser-level tests and split UI responsibilities around stable user workflows |
| Deployment and operations | `Dockerfile`, `docker-compose.yml`, `docker/`, `dadbot_system/api.py` | Docker, Ollama, PostgreSQL, Redis; JSON logging/OTel options | **Linux/Docker path exists; API liveness/readiness split wired** | Partial | `/health` remains process liveness; `/ready` checks state/authority/worker and has opt-in bounded cached model/vector probes; dependency-down and worker-quarantine transitions return 503; probe transition/cache/timeout tests pass | Model/vector default to unprobed; Compose does not opt into their readiness; image installs voice/MCP/reranker/Torch extras; startup begins as root; high | Keep Compose conservative; enable/require model and configured vector probes per production deployment after validating endpoints; reduce image/privilege overhead and test deployment shutdown/failure |
| Verification and engineering quality | `tests/`, `pyproject.toml`, `pytest.ini`, `.github/workflows/` | pytest, Ruff, import-linter, optional external services | **Large test surface; uneven gates** | Partially verified | 4,336 collected; 1,860 unit passed, 80 failed, 10 skipped; import contracts pass | No coverage threshold; CI Ruff gate covers three files; high | Triage red tests, measure coverage, broaden CI on supported high-value packages |

## E. Code health

### Strengths

- There are explicit architectural import contracts. `lint-imports` analyzed 525 files and 950 dependencies; all four configured contracts were kept.
- The service path has explicit lifespan cleanup, authority checks, tenant-qualified stores, and persisted task recovery.
- The dependency manifest and lock file define a modern Python 3.13 environment with optional extras separated by service, voice, MCP, notifications, and reranker use.
- Docker Compose defines service healthchecks and dependencies; tests include restart, fresh-process, replay, and real-service categories.

### Risks and debt

- `ruff check dadbot dadbot_system caaf evaluation --statistics` found **1,809** findings in the current worktree. Largest groups include 285 line-too-long, 248 blind-except, 149 too-many-arguments, 146 whitespace-only blank lines, and 113 complex-structure. These are counts, not a clean-baseline comparison.
- Type-annotation and docstring lint families are explicitly suppressed in `pyproject.toml`; no configured mypy/pyright gate was identified.
- No coverage tooling, threshold, or coverage report artifact was found in the inspected configuration/CI.
- A 1,333+ line Streamlit entry module, multiple model/tool/runtime paths, and distinct state/ledger/projection stores raise change and incident-triage cost.
- Docker installs voice, service, MCP, and reranker extras in the builder path, including Torch/Transformers, although deployments may not use them. The entrypoint starts as root to initialize directories before dropping privileges.
- Telemetry configuration exists, but central log shipping, metric scraping, alerting, and rollback were not demonstrated.
- `pip check` reported no broken installed requirements. This only checks the current environment’s dependency metadata.

## F. Test and verification report

### Finalization reconciliation (2026-11-10)

The staged Phase 0-only attempt initially failed collection because two tests
imported the unstaged `dadbot/core/tenant_session.py`. After including that
required helper, the isolated staged full-unit run collected 1,916 tests and
reported **50 failed, 1,856 passed, and 10 skipped**. The failures were not
limited to one missing helper: Phase 0 tests also depend on changes outside
that commit, including tenant ownership, runtime configuration, memory
retrieval, and later-phase test support. The staged snapshot therefore did
not represent a dependency-complete checkpoint. No commit was made from that
red staged snapshot.

The selected commit shape is one combined remediation commit (Option C),
staged with explicit paths. Phase-isolated commits are abandoned because the
observed Phase 0 staged result is not green and the worktree increments are
entangled. The original pre-program path manifest is unavailable. The 149
current paths were classified by path-based inference (A=98 remediation,
B=43 pre-existing dirty paths, C=8 audit/notes, D=0 generated paths); this
classification is not hunk-level provenance. The combined commit includes
the current tracked and untracked source, tests, and audit deliverables needed
to reproduce the combined state; generated test databases and the scratch
inventory are excluded.

The fixed-order characterization ran the same 41-test sequence ten times;
all ten trials passed (**410 passing test executions; no failure reproduced
within those trials**). A later full `tests/unit` run did reproduce the
intermittent failure in
`tests/unit/test_source_grounding.py::test_canonical_approval_and_revocation_invalidate_repeated_prompt_cache`:
**1 failed, 1,960 passed, 10 skipped** (1,971 collected). Its observed
exception chain starts with `ValueError: occurred_at missing or invalid in
divergence checkpoint envelope`, then
`PersistenceFailure: PersistenceService.save_graph_checkpoint rejected
malformed checkpoint`, then strict `PersistenceFailure:
PersistenceService.finalize_turn strict-mode failure`. The reported
`occurred_at` validation is the surfaced failure, not established proof of
the underlying divergence cause. No diagnostic envelope was captured during
the ten trials, and this pass made no functional change for the flake.

The one required final import-linter run was
`.\.venv\Scripts\lint-imports.exe --no-cache`: **5 contracts kept, 0 broken**,
with **528 files and 963 dependencies analyzed**. This supersedes the
historical Phase 1 notes below that recorded two API-to-telemetry violations;
the API now uses the runtime-signals adapter, and the complete current
import-linter result is clean.

The latest complete worktree unit result for this finalization pass is the
green rerun: **1,961 passed, 10 skipped, 0 failed** in 285.62 seconds.
The preceding worktree run was red only in
`tests/unit/test_source_grounding.py::test_canonical_approval_and_revocation_invalidate_repeated_prompt_cache`,
with **1 failed, 1,960 passed, and 10 skipped**. This is the already
characterized order-dependent checkpoint/cache flake, not a new
worktree-only failure. The post-push clean clone at commit
`cb8720501cabc07c247d6cd832e9f90cc0fb581a` also completed with **1,961
passed, 10 skipped, 0 failed** in 258.70 seconds, and import-linter reported
**5 kept, 0 broken** (528 files, 963 dependencies). Thus the latest worktree
and clean-clone outcomes agree; the earlier difference is reconciled as the
known intermittent flake resurfacing. The clean-clone result is the
authoritative artifact for the pushed commit.

The combined commit leaves the following Phase 4/5 work explicitly open:
voice/VAD/barge-in completion; PWA/service-worker behavior; Docker profile
split and root-entrypoint review; coverage and type-check gates; a documented
multi-authority persistence contract; and live PostgreSQL/Redis plus live
model/vector outage verification.

### Commands run in this audit

| Command | Result |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest --collect-only -q` | 347 collected test modules; summing per-file collection counts yielded **4,336 test cases**. |
| `.\.venv\Scripts\python.exe -m pytest tests\unit --tb=short --basetemp=$base` (with `$base` an isolated `%TEMP%` directory) | **80 failed, 1,860 passed, 10 skipped** in 231.07s; exit 1. This is the valid rerun with pytest output stored outside its temporary root. |
| `.\.venv\Scripts\python.exe -m pytest tests\unit\test_durable_submission_authority.py tests\test_service_api_integration.py tests\unit\test_owned_sqlite_sessions.py --basetemp=$base --tb=short` | **62 passed** in 27.32s. |
| `.\.venv\Scripts\lint-imports.exe --no-cache` | **4 contracts kept, 0 broken**; 525 files, 950 dependencies analyzed. |
| `.\.venv\Scripts\python.exe -m ruff check dadbot dadbot_system caaf evaluation --statistics` | **1,809 findings**, exit 1. |
| `.\.venv\Scripts\python.exe -m pip check` | No broken requirements. |

Earlier attempts to run the unit suite were invalid due to pytest temporary-directory permissions and then placing the captured output inside pytest’s own cleanup root. They are excluded from the result above. The valid run used an isolated temp root and a sibling log.

### Broad result and Phase 0 attribution

The original audit run reported 80 failed, 1,860 passed, and 10 skipped. The
last pre-remediation full unit run reported 34 failed, 1,916 passed, and 10
skipped; after the snapshot serialization fix it reported 33 failed, 1,918
passed, and 10 skipped. The current result, after the test-contract batches
below, is **0 failed, 1,951 passed, and 10 skipped** across 1,961 tests. The
current result is from a materially modified working tree, not a
clean-checkout comparison. Historical run counts below are retained as
attribution evidence and are not the current status.

The six-file classification selection was:

```text
tests/unit/test_phase1_chat.py
tests/unit/test_local_authority_mutation_application.py
tests/unit/test_sqlite_service_state.py
tests/unit/test_model_context_budget.py
tests/unit/test_local_tenant_establishment_application.py
tests/unit/test_response_engine_control_plane_integration.py
```

Combined result: **14 failed, 127 passed, 0 skipped**. Split by file, phase-one
chat, SQLite service-state, model-context-budget, and local-tenant-establishment
had no failures; local-authority-mutation had 9 failures; response/control-plane
integration had 5. The nine local-authority failures were rerun as individually
selected test cases with a fresh `%TEMP%` basetemp: **8 failed, 1 passed**.
The six-file run used `.\.venv\Scripts\pytest.exe --basetemp=.pytest_tmp_phase0_classification -o addopts='' -q` followed by the six paths listed above.

| Test | Individual disposition |
|---|---|
| `test_mutation_survives_restart` | **Contract-mismatched mirror assertion.** The test's ledger revision and reloaded tenant memory assertions succeed; it fails only because no `memory.json` is written in ledger-authoritative mode. |
| `test_startup_recovers_semantic_and_graph_independently` | **Fixture violates ownership gate.** Startup fails before graph/semantic recovery because the fixture places a legacy memory snapshot with no tenant ownership evidence. |
| `test_semantic_startup_failure_keeps_graph_available_and_reports_degraded` | **Fixture violates ownership gate.** Same precondition and same fail-closed startup rejection; the intended embedding-failure path is never reached. |
| `test_process_exit_during_startup_graph_rebuild_recovers_on_restart` | **Environment-sensitive, passes off OneDrive.** It failed in the repository-local OneDrive temp root with `PermissionError` at `os.replace` while writing tenant evidence; rerunning the case with `%TEMP%` basetemp passed. |
| `test_clear_with_indexed_memories_does_not_append_without_fenced_repairs` | **Fixture violates ownership gate.** Unowned legacy snapshot prevents construction before the fenced-clear behavior runs. |
| `test_semantic_index_rejects_stale_catalog_and_clear` | **Fixture violates ownership gate.** Same unowned snapshot; stale-catalog assertions are not reached. |
| `test_stale_semantic_content_is_not_used_for_local_retrieval` | **Fixture violates ownership gate.** Same unowned snapshot; stale-content retrieval assertions are not reached. |
| `test_semantic_repair_rejects_catalog_after_external_commit` | **Fixture violates ownership gate.** Same unowned snapshot; external-commit repair assertions are not reached. |
| `test_semantic_repair_rejects_candidate_after_concurrent_commit` | **Fixture violates ownership gate.** Same unowned snapshot prevents construction of the first bot, before concurrent repair begins. |

These observations establish no production authorization regression among
these nine tests. Seven fail at the same intended tenant-ownership guard; the
restart test's durable ledger behavior passes and only its JSON expectation
conflicts with the current persistence contract; the interrupted-rebuild test
passes when the OneDrive temp-path interference is removed. No ownership,
authorization, or fail-closed rule was weakened.

### Phase 1 increment: canonical runtime, readiness, and worker quarantine

`launch.py` imports `dadbot.app_runtime.main`; the `Dad.py` and
`api_entrypoint.py` wrappers delegate through `launch.main`. A new
import-linter forbidden contract prevents `dadbot.app_runtime` from importing
`dadbot.runtime.launcher`. The entrypoint tests verify both wrapper routing and
the terminal `launch.main` → `app_runtime.main` delegation.

`/health` remains a process-liveness response. `/ready` returns 503 with
structured check groups if the state store or authenticated authority store
is unavailable, or if workers are not ready; Compose probes `/ready`. Worker
authority status is now explicit (`starting`, `available`, or `quarantined`),
and process failure is `unavailable`. While quarantined, an authenticated
worker continues polling authority but does not dequeue queued tasks. A focused
regression exercises authority loss, confirms no task effect occurs while
quarantined, then restores authority and verifies the queued task executes.

Verification on this materially modified Windows checkout:

- The API/authority selection
  `tests\unit\test_durable_submission_authority.py`,
  `tests\test_service_api_integration.py`, and
  `tests\unit\test_owned_sqlite_sessions.py` passed **68 tests** in 28.32
  seconds (the historical selector had 62 before additional tests were
  present).
- Entrypoint, readiness, and worker tests passed **10 tests** in 2.90 seconds.
- Full `tests\unit` passed **1,951 tests, 10 skipped, 0 failed** in 261.79
  seconds, using isolated `%TEMP%\dadbot_phase1_unit_20261110`.
- The new canonical-runtime import contract was **KEPT** by `lint-imports`.
  The full import-linter run also reports two API-to-telemetry
  violations in the `/doctor` trace-provider path (`dadbot_system/api.py`),
  which are outside this increment.
- Ruff's check of `tests\test_core_import_firewall.py` reports its pre-existing
  complexity finding in `_imported_modules`; the import ordering issue in
  that touched file was corrected. Existing Ruff findings also remain in
  `dadbot_system/api.py`, `dadbot_system/worker.py`, and
  `tests\test_service_api_integration.py`.

No tool-spine consolidation was included. This is a dirty-checkout result,
not clean-commit attribution.

### Phase 1 increment 2: bounded authority quarantine and recovery

Worker authority polling now drives an explicit four-state readiness
projection: `starting` before the first successful poll, `available` after a
successful poll, `quarantined` after one or two consecutive failures, and
`unavailable` after three or more consecutive failures or worker-process
failure. The worker continues recovery polling while unavailable. Retry delay
is exponential from 0.1 seconds and capped at 1.6 seconds; successful polling
resets the failure count and delay.

Queued tasks stay in the broker because the worker does not dequeue without a
successful revocation poll. If the second authority check fails after dequeue,
the worker keeps that single task in process-local pending state rather than
publishing or discarding it. A transient authority-store error during
delegation verification or execution is marked retryable; invalid/revoked
delegations are not retried as transient outages. After recovery, the same
task/request identity is processed again. The before-effect regression proves
the effect happens once after recovery. The after-commit regression uses the
real `ExecutionControlPlane` and `InMemoryExecutionLedger` to prove replay
returns the committed effect without invoking the effect executor twice.
If the worker process exits while holding a task, the durable task remains
incomplete and follows the existing startup recovery path.

`/ready` exposes each worker aggregate state in its structured `checks.runtime`
body and returns 200 only for `available`; `starting`, `quarantined`, and
`unavailable` return 503 with their specific worker state. For multiple
workers, any `unavailable` worker makes the aggregate unavailable; otherwise
any quarantined worker makes it quarantined; all workers must be available to
make the service ready.

Verification:

- Worker quarantine, dequeued-task hold, before/after-effect retry,
  capped-backoff/authority-flap, multi-worker aggregation, readiness states,
  and existing authority tests: **48 passed** in 8.86 seconds.
- API/authority selection (`test_durable_submission_authority.py`,
  `test_service_api_integration.py`, and `test_owned_sqlite_sessions.py`):
  **71 passed** in 25.95 seconds on the final rerun.
- Full `tests/unit`: **1,951 passed, 10 skipped, 0 failed** in 251.25
  seconds, using isolated `%TEMP%\dadbot_phase1_inc2_unit_20261110`. A
  subsequent acceptance rerun had one unrelated failure in
  `test_source_grounding.py::test_canonical_approval_and_revocation_invalidate_repeated_prompt_cache`
  (**1 failed, 1,950 passed, 10 skipped**); its traceback reported
  `PersistenceFailure.save_graph_checkpoint rejected malformed checkpoint`
  due to a missing `occurred_at` in a divergence-checkpoint envelope. The
  same test failed once in isolation and passed in a fresh isolated rerun.
  The final full-unit rerun passed **1,951 passed, 10 skipped, 0 failed** in
  265.33 seconds. This pre-existing checkpoint/cache flake remains open and
  was not changed in this worker increment.
- The canonical-runtime import-linter contract remains **KEPT**. The complete
  import-linter command still exits nonzero for two API-to-telemetry
  violations in the existing `/doctor` trace-provider path.
- Ruff `--select I,F` passed on the touched worker/contracts/test paths.
  `--select E501` still reports two existing long lines in
  `dadbot_system/worker.py:231` and `tests/test_service_api_integration.py:1304`;
  neither is part of this increment. Scoped `git diff --check` reports the
  previously recorded trailing-whitespace line at
  `tests/unit/test_durable_submission_authority.py:670`; this increment did
  not touch that line. Other changed paths pass the whitespace check.

No tool-spine consolidation was included. Full Phase 1 remains in progress;
this is a dirty-checkout result, not clean-commit attribution.

### Memory JSON persistence contract

The checked-in implementation has an explicit mode-dependent contract:

- In file-WAL ledger mode, the **execution ledger is authoritative** after a
  tenant has established memory. Snapshot-only writes are rejected, and the
  shared JSON file is **not a guaranteed derived mirror**. A pre-existing
  shared legacy JSON snapshot without ownership evidence cannot establish or
  grant a tenant's memory.
- In the tenant-document-store path, the document store is authoritative and
  JSON is a **derived compatibility mirror**. A mirror write is best-effort
  for durable document stores, but required for the local ephemeral document
  store.
- Where a legacy snapshot exists, migration/import requires separately
  established tenant ownership; there is no implicit shared-mirror entitlement.

This resolves the failing restart assertion as an obsolete expectation for
ledger mode, not a request to add a shared cross-tenant mirror. Tests that
exercise ledger authority should assert ledger replay/recovery; tests that
exercise document-store mirroring should configure that backend explicitly.
This contract is consistent with `dadbot/memory/storage.py` write guards,
`tests/test_memory_manager.py` mirror failure semantics, and
`docs/decisions/tenant-identity-collisions.md`.

### Cache order-dependence experiment

The cache case
`test_canonical_approval_and_revocation_invalidate_repeated_prompt_cache`
passed twice in fresh isolated runs. It was then run last, in one pytest
process, after the complete local-authority-mutation module and the complete
response/control-plane integration module, in that fixed order:

```text
.\.venv\Scripts\pytest.exe --basetemp=$base -o addopts='' -q tests\unit\test_local_authority_mutation_application.py tests\unit\test_response_engine_control_plane_integration.py tests\unit\test_source_grounding.py::test_canonical_approval_and_revocation_invalidate_repeated_prompt_cache
```

That ordered run finished **13 failed, 28 passed**: the same 8 authority
failures and 5 control-plane boundary failures, with no new failure from the
cache case. Thus neither failure cluster, by itself or in this sequence, was
enough to reproduce the cache failure. The original full-suite failure during
strict SaveNode finalization and malformed divergence-checkpoint envelope is
classified below as an order-dependent flake. The predecessor that leaves the
malformed envelope remains unidentified; this experiment does not rule out an
earlier test or broader suite-order interaction.

### Historical disposition of the 34-failure baseline

The full unit suite at this stage was rerun with an isolated `%TEMP%` basetemp and its
captured output stored outside that temp root:

```text
.\.venv\Scripts\pytest.exe --basetemp=$base -o addopts='' -q --tb=short tests\unit
```

`$base` resolved to
`C:\Users\josep\AppData\Local\Temp\dadbot_phase0_full_triage`. The captured
log was `C:\Users\josep\AppData\Local\Temp\dadbot_phase0_full_triage_utf8.log`.
The command took **246.35 seconds wall-clock**; pytest reported **243.72
seconds**. Result: **34 failed, 1,916 passed, 10 skipped**. This was the
full-suite red result at that point in the work. The temporary log was used to
inspect all failure traces and removed after this disposition was recorded.

Disposition totals for the 34 failing nodes are now **33 stale-contract
(including 3 reclassified snapshot tests), 1 order-dependent-flake, and 0
genuine-unknown**. Diagnosis also found **1 confirmed production defect** in
tenant-qualified snapshot creation; it is a separate valid-input failure, not
the cause of these three tests' assertion tracebacks. No production or test
source was changed.
At this historical point, Phase 0 remained PARTIAL: attribution was complete
for this 34-test set, but the full suite was red and the snapshot defect was
unresolved. The subsequent snapshot fix and stale-contract test updates are
recorded below.

#### Contracts applied while updating stale tests

Tenant-owned startup recovery fixtures must seed memory through a
tenant-qualified document store (tenant ID plus the `"memory"` document key);
a shared JSON file is not ownership evidence and must remain ignored for an
already-established tenant. Recovery status objects include stable status and
source-revision fields plus additive generation/timestamp metadata, so tests
must assert the required status/revision contract without rejecting those
additional fields. In durable file-WAL mode, tenant ledger replay is
authoritative and a shared JSON mirror may be absent or stale; tests must
verify that stale mirror data cannot override the committed tenant projection.

| # | Node ID | Disposition | Observed evidence and attribution |
|---:|---|---|---|
| 1 | `test_execution_ledger_boot_order.py::test_established_tenant_authority_cannot_be_overwritten_by_stale_snapshot` | **stale-contract** | The tenant projection and replay assertions pass; only the assertion that the shared JSON file was rewritten to `positive` fails (`tests/unit/test_execution_ledger_boot_order.py:94`). In file-WAL ledger mode that file is not a guaranteed mirror (`dadbot/memory/storage.py:409-420`). |
| 2 | `test_facade_runtime_delegation.py::test_missing_facade_method_resolves_from_service_registry` | **stale-contract** | The test calls `DadBot.__getattr__` with a `SimpleNamespace` that lacks `_RUNTIME_STATE_ATTR_MAP`, failing before service-registry fallback (`tests/unit/test_facade_runtime_delegation.py:15-22`; `dadbot/core/dadbot.py:148-149`). The fixture does not model a DadBot instance. |
| 3 | `test_facade_runtime_delegation.py::test_memory_store_facade_reads_and_replaces_canonical_projection` | **stale-contract** | The test requires object identity with the backing dict (`tests/unit/test_facade_runtime_delegation.py:25-36`); the facade intentionally returns a projection wrapper (`dadbot/core/dadbot.py:58-70,562-570`), so equal contents do not imply identity. |
| 4 | `test_local_authority_mutation_application.py::test_mutation_survives_restart` | **stale-contract** | Ledger revision and reloaded memory checks pass; the final assertion fails because ledger-authoritative mode did not create `memory.json` (`tests/unit/test_local_authority_mutation_application.py:15-26`). Snapshot-only writes are disabled in this mode (`dadbot/memory/storage.py:409-420`). |
| 5 | `test_local_authority_mutation_application.py::test_startup_recovers_semantic_and_graph_independently` | **stale-contract** | The fixture seeds an unowned shared legacy snapshot (`tests/unit/test_local_authority_mutation_application.py:46-55`); startup stops at the ownership gate before graph/semantic recovery (`dadbot/memory/storage.py:780-798`). |
| 6 | `test_local_authority_mutation_application.py::test_semantic_startup_failure_keeps_graph_available_and_reports_degraded` | **stale-contract** | Same unowned legacy-snapshot setup; the ownership guard runs before the injected embedding failure or degraded-status assertions (`tests/unit/test_local_authority_mutation_application.py:70-87`; `dadbot/memory/storage.py:780-798`). |
| 7 | `test_local_authority_mutation_application.py::test_clear_with_indexed_memories_does_not_append_without_fenced_repairs` | **stale-contract** | Construction fails on the test's unowned legacy snapshot before the fenced-clear operation (`tests/unit/test_local_authority_mutation_application.py:162-174`; `dadbot/memory/storage.py:780-798`). |
| 8 | `test_local_authority_mutation_application.py::test_semantic_index_rejects_stale_catalog_and_clear` | **stale-contract** | The fixture's shared legacy snapshot has no tenant ownership evidence; stale-catalog/clear assertions are never reached (`tests/unit/test_local_authority_mutation_application.py:236-253`; `dadbot/memory/storage.py:780-798`). |
| 9 | `test_local_authority_mutation_application.py::test_stale_semantic_content_is_not_used_for_local_retrieval` | **stale-contract** | Startup rejects the same unowned snapshot before retrieval (`tests/unit/test_local_authority_mutation_application.py:255-272`; `dadbot/memory/storage.py:780-798`). |
| 10 | `test_local_authority_mutation_application.py::test_semantic_repair_rejects_catalog_after_external_commit` | **stale-contract** | Startup rejects the legacy snapshot before the external-commit repair scenario (`tests/unit/test_local_authority_mutation_application.py:274-302`; `dadbot/memory/storage.py:780-798`). |
| 11 | `test_local_authority_mutation_application.py::test_semantic_repair_rejects_candidate_after_concurrent_commit` | **stale-contract** | The first bot cannot be constructed from the unowned shared snapshot, so the concurrent candidate check is not exercised (`tests/unit/test_local_authority_mutation_application.py:304-330`; `dadbot/memory/storage.py:780-798`). |
| 12 | `test_memory_authority_application.py::test_durable_bot_migrates_and_records_memory_mutations` | **stale-contract** | The test imports `legacy-memory.json` without tenant ownership evidence and startup rejects it before migration (`tests/unit/test_memory_authority_application.py:5-10`; `dadbot/memory/storage.py:780-798`). |
| 13 | `test_memory_snapshot_write_guard.py::test_direct_snapshot_write_rejects_established_tenant[_write_memory_store_unlocked]` | **stale-contract** | Setup fails importing its unowned legacy JSON before reaching the direct-write guard (`tests/unit/test_memory_snapshot_write_guard.py:9-27`; ownership gate `dadbot/memory/storage.py:780-798`; intended write guard `:409-420`). |
| 14 | `test_memory_snapshot_write_guard.py::test_direct_snapshot_write_rejects_established_tenant[_write_json_memory_store_unlocked]` | **stale-contract** | Same setup failure as #13; the parameterized writer is not reached. The intended ledger-authoritative contract prohibits snapshot-only writes (`tests/unit/test_memory_snapshot_write_guard.py:9-27`; `dadbot/memory/storage.py:409-420,780-798`). |
| 15 | `test_priority_graph_trace_enrichment.py::test_graph_records_node_and_edge_trace_events` | **stale-contract** | The test executes a bare two-node `TurnGraph` with no `SaveNode` stage and is blocked by the required post-execution invariant (`tests/unit/test_priority_graph_trace_enrichment.py:35-39`; `dadbot/core/invariant_registry.py:70-76`). |
| 16 | `test_priority_graph_trace_enrichment.py::test_graph_turn_trace_ignores_preseeded_snapshot_authority` | **stale-contract** | Same bare-graph fixture lacks the required `SaveNode` stage, so the trace assertion is not reached (`tests/unit/test_priority_graph_trace_enrichment.py:102-110`; `dadbot/core/invariant_registry.py:70-76`). |
| 17 | `test_priority_graph_trace_enrichment.py::test_graph_fails_when_projected_trace_event_count_diverges` | **stale-contract** | The test expects the injected projection-mismatch error, but post-execution rejects the graph first because `SaveNode` is missing (`tests/unit/test_priority_graph_trace_enrichment.py:118-137`; `dadbot/core/invariant_registry.py:70-76`). |
| 18 | `test_production_reranker.py::test_api_worker_preserves_explicit_source_mode[True-expected0]` | **stale-contract** | Direct worker execution has no delegated authority context and is denied before source-mode assertions (`tests/unit/test_production_reranker.py:21-33`; `dadbot_system/worker.py:231-235`; `dadbot_system/delegation.py:126-128`). |
| 19 | `test_production_reranker.py::test_api_worker_preserves_explicit_source_mode[False-expected1]` | **stale-contract** | Same missing-authority failure as #18; source-mode parameter `False` does not bypass the live-execution authority requirement (`tests/unit/test_production_reranker.py:21-33`; `dadbot_system/delegation.py:126-128`). |
| 20 | `test_production_reranker.py::test_api_worker_preserves_explicit_source_mode[true-expected2]` | **stale-contract** | Same missing-authority failure as #18; the string parameter is not evaluated by the worker without delegated authority (`tests/unit/test_production_reranker.py:21-33`; `dadbot_system/delegation.py:126-128`). |
| 21 | `test_replay_alias_contract.py::test_relationship_apply_feedback_records_boundary_trace` | **stale-contract** | `DadBot()` uses the default tenant, which the execution-identity gate rejects unless gated non-production configuration is enabled (`tests/unit/test_replay_alias_contract.py:18-20`; `dadbot_system/execution_identity.py:12-19`). |
| 22 | `TestResponseEngineControlPlaneIntegration::test_finalize_submit_success_persists_response_engine_telemetry` | **stale-contract** | Direct private finalizer invocation at test line 279 reaches `control_plane.ledger_writer` without KernelGateway scope and fails before telemetry assertions (`tests/unit/test_response_engine_control_plane_integration.py:234-279`; `dadbot/core/control_plane.py:2062`). |
| 23 | `TestResponseEngineControlPlaneIntegration::test_finalize_submit_success_persists_structured_decision_reasoning` | **stale-contract** | Direct private finalizer invocation at line 373 fails the same mandatory kernel-scope check before reasoning assertions (`tests/unit/test_response_engine_control_plane_integration.py:293-373`; `dadbot/core/control_plane.py:2062`). |
| 24 | `TestResponseEngineControlPlaneIntegration::test_finalize_submit_success_persists_pending_selection_for_next_turn_learning` | **stale-contract** | Direct private finalizer invocation at line 567 fails the same scope check before persistence/learning assertions (`tests/unit/test_response_engine_control_plane_integration.py:510-567`; `dadbot/core/control_plane.py:2062`). |
| 25 | `TestResponseEngineControlPlaneIntegration::test_finalize_submit_success_updates_identity_state_and_continuity` | **stale-contract** | Direct private finalizer invocation at line 630 fails the same scope check before identity/continuity assertions (`tests/unit/test_response_engine_control_plane_integration.py:582-630`; `dadbot/core/control_plane.py:2062`). |
| 26 | `TestResponseEngineControlPlaneIntegration::test_phase3_behavioral_stability_gate_30_turn_conflict_sequence` | **stale-contract** | The 30-turn test reaches finalization through a direct control-plane path and fails the same `control_plane.ledger_writer` scope guard at its line-791 call; the behavioral sequence does not complete (`tests/unit/test_response_engine_control_plane_integration.py:708-791`; `dadbot/core/control_plane.py:2062`). |
| 27 | `test_runtime_service_provider_transition.py::test_dadbot_prefers_explicit_runtime_services_constructor_injection` | **stale-contract** | The constructor fails on the ungated default tenant before the dependency-injection identity assertions (`tests/unit/test_runtime_service_provider_transition.py:75-92`; `dadbot_system/execution_identity.py:12-19`). |
| 28 | `test_semantic_index_repair_failures.py::test_local_semantic_verification_failure_rolls_back_publication` | **stale-contract** | Bot construction fails importing the test's unowned legacy snapshot before semantic verification/publication (`tests/unit/test_semantic_index_repair_failures.py:69-97`; `dadbot/memory/storage.py:780-798`). |
| 29 | `test_semantic_index_repair_failures.py::test_local_semantic_candidate_rejects_changed_index` | **stale-contract** | Same ownership failure prevents the changed-index candidate scenario from running (`tests/unit/test_semantic_index_repair_failures.py:100-125`; `dadbot/memory/storage.py:780-798`). |
| 30 | `test_semantic_index_repair_failures.py::test_process_exit_during_sqlite_semantic_publication_preserves_previous_index` | **stale-contract** | The child process exits with code 1 at the unowned-snapshot startup guard rather than reaching the injected publication exit code 73 (`tests/unit/test_semantic_index_repair_failures.py:128-159`; `dadbot/memory/storage.py:780-798`). |
| 31 | `test_snapshot_tail_recovery.py::test_snapshot_tail_replay_preserves_checkpoint_state_and_applies_tail` | **reclassified-stale** | Exact input: snapshot `sessions={"existing":{"old_field":"preserved","changed":"old"},"deleted":{"value":1}}`; three tail records for `existing`, `deleted`, and `added`; none has `tenant_id`. Full pytest traceback ends at `tests/unit/test_snapshot_tail_recovery.py:47` with `AssertionError: None == {"old_field":"preserved","changed":"new"}`. The call path is test:45 → `dadbot/core/snapshot_engine.py:180` → `SessionStore.rebuild_from_ledger()` → `dadbot/core/session_store.py:78-83`, where missing tenant causes each record to be discarded. The test expects unqualified replay, contrary to tenant-scoped projection behavior; it does not demonstrate a defect for valid tenant-qualified events. |
| 32 | `test_snapshot_tail_recovery.py::test_snapshot_tail_replay_without_tail_restores_checkpoint` | **reclassified-stale** | Exact input: snapshot `{"head_sequence":10,"session_snapshot":{"sessions":{"existing":{"value":1}}}}`; empty tail. Full pytest traceback ends at `tests/unit/test_snapshot_tail_recovery.py:59` with `AssertionError: None == {"value":1}`. `replay_tail()` passes snapshot seed events to `SessionStore.rebuild_from_ledger()` (`dadbot/core/snapshot_engine.py:180`); the seed event has no `tenant_id` (`:125-129`) and `SessionStore._apply_event()` discards it (`dadbot/core/session_store.py:78-83`). The test asks an unqualified snapshot to create an unqualified session; the current tenant-scoped projection requires tenant identity. |
| 33 | `test_snapshot_tail_recovery.py::test_snapshot_verification_detects_projection_drift_even_when_snapshot_is_rehashed` | **reclassified-stale** | Exact input ledger event: `{"type":"SESSION_STATE_UPDATED","session_id":"s1","trace_id":"test","timestamp":"2026-01-01","kernel_step_id":"test","payload":{"state":{"value":1}}}`; `tenant_id` is absent. Full pytest traceback ends at `tests/unit/test_snapshot_tail_recovery.py:97` with `KeyError: 's1'`: `SessionStore._apply_event()` discards the event at `dadbot/core/session_store.py:78-83`, so the test indexes a projection it never populated. The fixture asserts against a tenantless event and unqualified key, not a valid tenant-bound projection. |
| 34 | `test_source_grounding.py::test_source_store_failure_releases_scheduler_and_allows_retry[OperationalError]` (historical run-order failure; not failed in the final run) | **order-dependent-flake** | In the intermediate post-fix full run it failed with `ValueError: occurred_at missing or invalid in divergence checkpoint envelope`, cascading to `PersistenceFailure: save_graph_checkpoint rejected malformed checkpoint` and strict `finalize_turn` failure. The parameterized test passed in isolation (**2 passed**) and did not fail in the final full run. An earlier full run had the same error in `test_canonical_approval_and_revocation_invalidate_repeated_prompt_cache`; the trigger or predecessor remains unidentified. |

### Stale-contract repair batches and final verification (2026-11-10)

The 33 baseline failures were corrected as test setup/expectation changes only.
No production code, authority gate, ownership rule, or fail-closed behavior
was changed. Assertions for the original intended behavior were retained and
now execute past the previously blocking boundary.

| Batch | Tests and changed locations | Previous test assumption → updated setup/contract |
|---|---|---|
| Response/control-plane (5) | `tests/unit/test_response_engine_control_plane_integration.py:284,344,562,635,810` | Direct finalizer calls had no kernel scope or verified task. Each now uses a tenant-scoped session and signed delegated task inside `KernelGateway.open_scope()`; telemetry, decision reasoning, pending learning, identity/continuity, and the 30-turn assertions remain. |
| Tenant-owned recovery and memory authority (13) | `tests/unit/test_local_authority_mutation_application.py:54,89,186,264,287,310,348`; `tests/unit/test_memory_authority_application.py:12`; `tests/unit/test_memory_snapshot_write_guard.py:17`; `tests/unit/test_semantic_index_repair_failures.py:78,121,161` | Tests seeded recoverable data in an unowned shared JSON file. They now seed tenant-owned `"memory"` documents or establish a tenant WAL before semantic repair. Existing unowned-snapshot denial remains covered by `tests/unit/test_local_tenant_establishment_application.py::test_local_startup_refuses_unowned_shared_snapshot`. The exact recovery-status assertions now check stable status/revision plus additive generation/timestamp metadata. |
| Tenant-qualified identity (5) | `tests/unit/test_snapshot_tail_recovery.py:34,83,146`; `tests/unit/test_replay_alias_contract.py:19`; `tests/unit/test_runtime_service_provider_transition.py:75` | Tenantless session events/snapshot keys and ungated default-tenant construction are replaced with explicit tenant/session identities. Positive projection/replay paths assert tenant-scoped state; the snapshot tests also assert that unqualified lookups do not expose it. |
| Required graph save stage (3) | `tests/unit/test_priority_graph_trace_enrichment.py:35,102,118` | Partial graphs omitted the required `SaveNode` stage. `SaveNode()` is included in each expected graph sequence; trace and projection-divergence assertions remain. |
| Remaining worker, facade, and mirror contracts (7) | `tests/unit/test_production_reranker.py:23`; `tests/unit/test_facade_runtime_delegation.py:15,28`; `tests/unit/test_execution_ledger_boot_order.py:72`; `tests/unit/test_local_authority_mutation_application.py:22` | Worker source-mode cases now carry verified delegated authority and a tenant-qualified request. Facade tests use a `DadBot` instance and assert the projection wrapper's value semantics. Durable JSON mirror tests now verify that stale JSON cannot override the committed ledger state; tenant-owned document persistence is asserted where applicable. |

Batch commands/results:

```text
.\.venv\Scripts\pytest.exe --basetemp=$base -o "addopts=" -q --tb=short tests\unit\test_response_engine_control_plane_integration.py
18 passed

.\.venv\Scripts\pytest.exe --basetemp=$base -o "addopts=" -q --tb=short tests\unit\test_local_authority_mutation_application.py tests\unit\test_memory_authority_application.py tests\unit\test_memory_snapshot_write_guard.py tests\unit\test_semantic_index_repair_failures.py
31 passed

.\.venv\Scripts\pytest.exe --basetemp=$base -o "addopts=" -q --tb=short tests\unit\test_snapshot_tail_recovery.py tests\unit\test_priority_graph_trace_enrichment.py tests\unit\test_production_reranker.py tests\unit\test_replay_alias_contract.py tests\unit\test_runtime_service_provider_transition.py
29 passed, 1 skipped

.\.venv\Scripts\pytest.exe --basetemp=$base -o "addopts=" -q --tb=short tests\unit\test_execution_ledger_boot_order.py tests\unit\test_facade_runtime_delegation.py
16 passed
```

Final changed-file selection (all 12 modified test files): **94 passed, 1
skipped** in 52.95 seconds. Ruff command:

```text
.\.venv\Scripts\ruff.exe check tests\unit\test_response_engine_control_plane_integration.py tests\unit\test_local_authority_mutation_application.py tests\unit\test_memory_authority_application.py tests\unit\test_memory_snapshot_write_guard.py tests\unit\test_semantic_index_repair_failures.py tests\unit\test_snapshot_tail_recovery.py tests\unit\test_priority_graph_trace_enrichment.py tests\unit\test_production_reranker.py tests\unit\test_replay_alias_contract.py tests\unit\test_runtime_service_provider_transition.py tests\unit\test_execution_ledger_boot_order.py tests\unit\test_facade_runtime_delegation.py
```

Result: **All checks passed**. Full unit command:

```text
.\.venv\Scripts\pytest.exe --basetemp=$base -o "addopts=" -q --tb=short tests\unit
```

The isolated basetemp was
`C:\Users\josep\AppData\Local\Temp\dadbot_stale_contract_full_20261110`;
captured output was written outside it to
`C:\Users\josep\AppData\Local\Temp\dadbot_stale_contract_full_20261110.log`.
Pytest collected **1,961** tests and completed with **0 failed, 1,951 passed,
10 skipped** in 212.72 seconds; wall-clock was 215.33 seconds. Compared with
the preceding **33 failed, 1,918 passed, 10 skipped**, this is **33 fewer
failures, 33 additional passes, and no skip-count change**. There are no
remaining failure nodes to re-disposition. The parked source-grounding checkpoint flake was not modified and did not
recur in that run; it later recurred intermittently during Phase 1, as
recorded in the Phase 1 increment 2 results.

Change attribution: this checkout was already materially dirty and includes
changes predating this increment, so no clean commit range can be attributed.
This increment edited the 12 test files listed above and this untracked audit
file; no files were staged or committed. Phase 0's full-unit acceptance is
closed for this checkout, while the overall audit remains **PARTIAL** because
the other roadmap/product and operational gates are not all complete.

#### Snapshot diagnosis: test tracebacks and separate production defect

The three tests above were rerun with:

```text
.\.venv\Scripts\pytest.exe --basetemp=$base -o "addopts=" -q --tb=long tests\unit\test_snapshot_tail_recovery.py::test_snapshot_tail_replay_preserves_checkpoint_state_and_applies_tail tests\unit\test_snapshot_tail_recovery.py::test_snapshot_tail_replay_without_tail_restores_checkpoint tests\unit\test_snapshot_tail_recovery.py::test_snapshot_verification_detects_projection_drift_even_when_snapshot_is_rehashed
```

Result: **3 failed in 0.16s**. All three are assertion failures; the complete
pytest tracebacks contain no thrown production exception. Their final frames
and exact failed values are recorded in rows 31-33. In all three cases, the
input event/snapshot lacks `tenant_id`; `SessionStore._apply_event()` returns
without applying such a record (`dadbot/core/session_store.py:78-83`). The
fixtures therefore test an older unqualified-session behavior. The intended
tenant isolation invariant is not to infer or invent tenant ownership for a
session or ledger event that does not supply it.

Exact failing inputs from the tests:

```python
# test_snapshot_tail_replay_preserves_checkpoint_state_and_applies_tail
snapshot = {
    "head_sequence": 10,
    "session_snapshot": {
        "sessions": {
            "existing": {"old_field": "preserved", "changed": "old"},
            "deleted": {"value": 1},
        },
    },
}
tail = [
    {"type": "SESSION_STATE_UPDATED", "session_id": "existing", "sequence": 11,
     "payload": {"state": {"changed": "new"}}},
    {"type": "SESSION_STATE_DELETED", "session_id": "deleted", "sequence": 12,
     "payload": {}},
    {"type": "SESSION_STATE_UPDATED", "session_id": "added", "sequence": 13,
     "payload": {"state": {"x": 1}}},
]

# test_snapshot_tail_replay_without_tail_restores_checkpoint
snapshot = {
    "head_sequence": 10,
    "session_snapshot": {"sessions": {"existing": {"value": 1}}},
}
tail = []

# test_snapshot_verification_detects_projection_drift_even_when_snapshot_is_rehashed
event = {
    "type": "SESSION_STATE_UPDATED",
    "session_id": "s1",
    "trace_id": "test",
    "timestamp": "2026-01-01",
    "kernel_step_id": "test",
    "payload": {"state": {"value": 1}},
}
```

In the first two snapshots, the session map keys are bare session IDs, not
tenant/session identities. In the first and third cases, every ledger record
also omits `tenant_id`.

The complete pytest failure tracebacks (the assertions raise directly in the
test frame) were:

```text
____ test_snapshot_tail_replay_preserves_checkpoint_state_and_applies_tail ____
tests/unit/test_snapshot_tail_recovery.py:47:
    assert store.get("existing") == {"old_field": "preserved", "changed": "new"}
E AssertionError: assert None == {"changed": "new", "old_field": "preserved"}

_________ test_snapshot_tail_replay_without_tail_restores_checkpoint __________
tests/unit/test_snapshot_tail_recovery.py:59:
    assert store.get("existing") == {"value": 1}
E AssertionError: assert None == {"value": 1}

_ test_snapshot_verification_detects_projection_drift_even_when_snapshot_is_rehashed _
tests/unit/test_snapshot_tail_recovery.py:97:
    snapshot["session_snapshot"]["sessions"]["s1"]["value"] = 2
E KeyError: 's1'
```

Separately, a valid-input runtime probe **confirmed a production defect** in
the same snapshot path. Input event:

```python
{
    "type": "SESSION_STATE_UPDATED",
    "tenant_id": "tenant-a",
    "session_id": "s1",
    "trace_id": "trace-1",
    "timestamp": "2026-01-01",
    "kernel_step_id": "test",
    "payload": {"state": {"value": 1}},
}
```

The resulting `SessionStore.snapshot()` value was
`{"version": 1, "sessions": {("tenant-a", "s1"): {"value": 1}}}`. Calling
`SnapshotEngine.take_snapshot(ledger=ledger, session_store=store)` raised the
full traceback:

```text
Traceback (most recent call last):
  File "<stdin>", line 21, in <module>
  File "dadbot/core/snapshot_engine.py", line 83, in take_snapshot
  snapshot["snapshot_hash"] = self._hash_snapshot(snapshot)
  File "dadbot/core/snapshot_engine.py", line 231, in _hash_snapshot
  serialized = json.dumps(...)
  File "Python313/Lib/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
  File "Python313/Lib/json/encoder.py", line 200, in encode
    chunks = self.iterencode(o, _one_shot=True)
  File "Python313/Lib/json/encoder.py", line 261, in iterencode
    return _iterencode(o, 0)
TypeError: keys must be str, int, float, bool or None, not tuple
```

This violated the **tenant-qualified snapshot round-trip invariant**: a
session projection keyed by `(tenant_id, session_id)` must be serializable,
persistable, and restorable without dropping tenant identity. The affected
path was `SessionStore.snapshot()` (`dadbot/core/session_store.py:23,60-61`)
→ `SnapshotEngine.take_snapshot()` / `_hash_snapshot()`
(`dadbot/core/snapshot_engine.py:72-83,260-267`). The defect and its fix are
recorded in the post-fix verification below. It was not the cause of the
three reclassified-stale test failures, whose supplied inputs omit tenant
identity.

#### Snapshot serialization fix and post-fix verification

The fix is scoped to the snapshot wire/JSON boundary in
`dadbot/core/snapshot_engine.py`: tuple identity keys are encoded as prefixed
JSON arrays in the snapshot's `sessions` mapping before hashing and ledger
publication; prefixed keys are decoded into seed events carrying the original
`tenant_id` and `session_id` during restore. The in-memory `SessionStore`
mapping remains keyed by `(tenant_id, session_id)`. Existing bare legacy
string keys remain readable, while strings colliding with the reserved
prefixes are escaped. No tenant identity is inferred for tenantless records.

New regression test:
`tests/unit/test_snapshot_tail_recovery.py::test_tenant_qualified_session_snapshot_serializes_and_restores`.
It writes the same session ID under two tenants, takes a snapshot, performs an
actual `json.dumps`/`json.loads` persistence round-trip, restores it, verifies
both tenant-specific states and the unqualified lookup miss, and checks
snapshot projection verification. Result: **1 passed**.

Commands and results:

```text
.\.venv\Scripts\ruff.exe check dadbot\core\snapshot_engine.py tests\unit\test_snapshot_tail_recovery.py
All checks passed!

.\.venv\Scripts\pytest.exe --basetemp=$base -o "addopts=" -q tests\unit\test_snapshot_tail_recovery.py::test_tenant_qualified_session_snapshot_serializes_and_restores tests\test_production_hardening_wave3.py::TestSnapshotEngine
7 passed, 1 failed in 10.55s
```

The single focused failure in the existing snapshot-engine class was
`TestSnapshotEngine.test_restore_from_snapshot_and_replay_tail`; its fixture
uses tenantless job records, and `SessionStore` ignores those by design. The
new tenant-qualified regression passed. The full unit command was:

```text
.\.venv\Scripts\pytest.exe --basetemp=$base -o "addopts=" -q --tb=short tests\unit
```

At the time of this snapshot-fix work, before the stale-contract batches, the
first post-fix run (before finalizing the wire-format marker) was **34 failed,
1,917 passed, 10 skipped in 253.69s pytest time** (257.13s wall-clock). The
final snapshot-fix run was **33 failed, 1,918 passed, 10 skipped in 245.79s
pytest time** (249.27s wall-clock), versus the pre-fix baseline **34 failed,
1,916 passed, 10 skipped**. It met the expected 34-to-33 failure delta; the
33 remaining failures were the stale-contract cases later corrected above.
The new regression test accounted for one additional pass; the other pass was
the previously intermittent source-grounding case, which did not fail in
that run. The intermediate
`test_source_store_failure_releases_scheduler_and_allows_retry[OperationalError]`
failure passed in isolation (**2 parameterizations passed**) and did not
recur in the subsequent green full run. Its trigger/predecessor remains
unlocated.

Change-range attribution: implementation is limited to
`dadbot/core/snapshot_engine.py`; the tenant-qualified serialization,
two-tenant isolation, JSON round-trip, and restore regression is in
`tests/unit/test_snapshot_tail_recovery.py`; `REPOSITORY_ENGINEERING_AUDIT.md`
records the evidence. No stale tests were modified in that snapshot-fix
increment. The subsequent stale-contract remediation and final unit result
are recorded in the earlier “Stale-contract repair batches and final
verification” section.

#### Response/control-plane cluster: per-test hypothesis check

The earlier “likely test-side drift” hypothesis is **confirmed for each of the
five cases**, not inferred from the cluster count. Cases #22–#26 individually
reach the private finalizer without the required `KernelGateway` scope; the
shared failure is the enforced `control_plane.ledger_writer` boundary at
`dadbot/core/control_plane.py:2062`. None reaches the behavior assertion it
was intended to test. These are stale test invocation paths, not evidence that
the kernel scope should be relaxed.

The snapshot classifications have been individually diagnosed: those three
fixtures use retired unqualified identity assumptions, while the separate
valid-input production serialization defect has been fixed and verified as
documented above. This does not justify weakening tenant identity or accepting
tenantless replay events.

### Additional verification evidence and gaps

- Earlier live integration work recorded in the existing audit exercised real PostgreSQL and Redis: task-store integration **30 passed**, trusted-local checkpoint adapters **9 passed**, and a combined checkpoint/mirror/authority selection **39 passed**. Those live services were stopped and these tests were **not rerun during this repository-wide audit**.
- The focused authority/API/ownership selection was rerun now and passed. It is not representative of the 4,336-test full suite.
- `tests/` includes unit, integration, adversarial, chaos, replay, stress, property, UI, and certification-oriented tests. Pytest defaults exclude marked stress/slow work, and external-service tests remain configuration-dependent.
- There is no browser-driven UI test evidence in the inspected setup; Python Streamlit smoke/contract tests do not establish browser usability, accessibility, offline behavior, or WebSocket UX.
- The complete `tests/unit` suite was verified on this Windows checkout: 0
  failed, 1,951 passed, 10 skipped. External-service integration,
  multi-host queueing, production deployment, load/soak behavior, and live
  third-party integrations were not verified in this pass.

## G. Prioritized defect and risk register

| Priority | Finding and evidence | Why it matters / affected paths | Confidence | Recommended correction |
|---|---|---|---|---|
| **P1 — Phase 0 stale-contract work closed; separate intermittent test remains open** | Original unit baseline was 80 failed / 1,860 passed / 10 skipped; the stale-contract baseline was 33 failed / 1,918 passed / 10 skipped. Phase 1 full runs include a final green 1,951 passed / 10 skipped run and one run with a failure in `test_canonical_approval_and_revocation_invalidate_repeated_prompt_cache`; the isolated case failed once and passed once. The failure reports a malformed divergence checkpoint missing `occurred_at`. | Dirty-checkout attribution; issue is not caused or altered by the Phase 1 worker changes, but current stability is not deterministic. | High that intermittent failure occurred; cause unknown | Keep Phase 0 disposition; separately diagnose the checkpoint envelope/order behavior if it recurs. |
| **P1 — worker quarantine path implemented; durability limits remain** | Authenticated workers transition `available → quarantined → unavailable` on repeated authority-poll failures, apply capped exponential backoff, hold a dequeued task across transient outages, and resume the same identity after recovery. Tests cover before-effect recovery, post-commit execution-ledger replay, authority flapping, worker aggregation, and `/ready` status codes. | Revoked/invalid delegations are not retried as transient outages. A process crash while holding a task relies on durable startup recovery; ambiguous in-flight effects remain subject to the existing execution-ledger replay/ambiguity contract. | High for covered local worker and readiness behaviors; not a distributed queue guarantee | Preserve these tests; separately validate durable recovery after worker process termination and external side-effect ambiguity before claiming exactly-once behavior. |
| **P2 — Phase 1 readiness expanded; external probes opt-in** | `/health` remains liveness. `/ready` checks state store, authenticated authority store, and worker status; it now has cached, timeout-bounded opt-in Ollama model-name and vector endpoint probes. Disabled probes report `unprobed`; probe details contain timestamp and error type only. Tests cover success, timeout, cache serialization, environment configuration, and required model up/down/up transitions. | Default behavior stays compatible. Deployments can require configured model/vector checks before HTTP 200 without changing authority/quarantine semantics. Compose still does not opt in. | High for local/injected probe semantics; external Ollama, PostgreSQL vector, and HTTP vector service reachability were not tested live | Recommend `DADBOT_READINESS_PROBES=model` and `DADBOT_READINESS_REQUIRED=model` where model availability is required; add `vector` only when `DADBOT_VECTOR_STORE_URL` is part of the serving path. Do not change Compose defaults until deployment-specific endpoint and startup policy are agreed. |
| **P2 — truthful defaults resolved; unification remains open** | Placeholder `reminder`, `calendar`, and `web_search` entries were removed from both default legacy registries. Surface/negative tests verify they are not registered or advertised and cannot execute; the separate authorized `set_reminder` regression remains covered. The service planner, kernel record/replay, and MCP plan loop are still disjoint. | Before removal, model/caller selection could produce success-shaped results without an effect and without approval, authority, or audit. Remaining disjoint paths can still have different semantics. | High for the removed defaults and covered reminder flow; medium-high for cross-path differences | Keep the placeholder surfaces absent. Unify the distinct paths only in a dedicated increment with explicit authorization, audit, replay, and compatibility contracts. |
| **P2 — memory failure signaling implemented; wider backend coverage remains open** | `MemoryService.retrieve_for_query()` now raises `MemoryRetrievalError` for partial/degraded retrieval; status-aware retrieval preserves `ok` for an empty success. Turn state records status, sources, and error types; optional `/ready` probes the semantic-index connection, and `/doctor` reports bounded recent status counts. | Previously, catalog/backend failure could look exactly like “no memory found.” The implementation distinguishes those outcomes without exposing exception messages or retrieval content. | High for injected retrieval failures, turn status, probe behavior, and doctor counts | SQLite/PostgreSQL semantic-index reachability is probed only when opted in; graph storage and non-semantic memory projection remain unprobed. Embedding quality is intentionally not a readiness condition. |
| **P2** | Multiple persistence authorities/projections and local-only broker/event components are used in different parts of execution. | Recovery, cross-process consistency, and external-effect idempotency are harder to reason about than a single “durable task” label suggests. | High | Document transaction/effect boundaries and state which guarantees are exactly-once, at-least-once, or best-effort; test crash boundaries. |
| **P2** | Voice audio processing expects prior transcript/note; multimodal vision selection uses model-name heuristics (`dadbot/managers/multimodal.py:17-49,70-90`). | Voice/multimodal package presence overstates a complete user workflow. | High | Build a tested capture/transcription/turn/respond path or narrow the documented capability. |
| **P2** | Broad Ruff debt (1,809 findings), no coverage threshold, no type-check gate identified, and CI Ruff selection limited to three files. | Static and behavioral quality drift can accumulate outside the narrow enforced slice. | High | First establish a baseline, then add focused package-level lint/type/coverage gates with ratcheting rather than mass autofix. |
| **P2** | Docker builder includes large optional voice/MCP/reranker dependencies and entrypoint starts as root (`Dockerfile:27-28,43,79`; `docker/entrypoint.sh:8-17`). | Larger image, extra dependency surface, slower build/deploy, and greater impact if initialization is compromised. | High | Split deploy profiles/images where warranted and minimize privileged initialization. |
| **P3** | PWA assets exist but no service-worker lifecycle was found; no browser automation was identified. | Install/offline claims and browser-level UX regressions are not demonstrated. | Medium-high | Add browser tests and service-worker behavior only if offline/install support is a product requirement. |
| **P3** | `pyproject.toml` includes `caaf*`, while no tracked `caaf/` package was found; multiple tracked root diagnostics/output/archive assets are present. | Packaging and repository ownership signals may be stale/noisy. | Medium | Confirm intended package targets; remove or relocate only after ownership and usage are established. |

At the time of the original bounded inspection, no exploitability finding was
established. The local MCP surface exposed broad capabilities, but that audit
did not perform a full endpoint-by-endpoint authorization review. Section K
records the subsequent targeted local-MCP authorization verification and its
remaining limits.

## H. Engineering maturity and imbalance

Development effort and architecture are concentrated in control-plane authority, durability, tenant ownership, replay, and memory consistency. This work has tangible implementation and test depth; the recent live task/checkpoint tests provide real service evidence in a selected scope. The concern is not that this work is unnecessary, but that it is substantially more mature than user-visible integration and broad operational proof.

Relative maturity, without a numerical score:

- **Strongest evidence:** authority boundary for focused paths; task ownership/recovery; file-WAL and restart/replay mechanisms; import-boundary contracts.
- **Mixed:** API startup and task/worker lifecycle; memory graph/semantic backends; Docker/Linux deployment.
- **Underdeveloped or unverified:** unified tool execution, real calendar/reminder workflows, model-provider breadth, end-to-end speech input, browser/PWA behavior, truthful readiness, cross-host queueing, coverage and type-checking.
- **Potentially over-complex:** several persistence/event/ledger/projection mechanisms and competing runtime/tool/model compositions require clearer contracts before adding more abstractions.

The immediate imbalance correction should be product-facing and verification-focused: stabilize the full test baseline, select the canonical execution/tool path, then complete a small number of real user workflows end to end. More persistence abstractions should wait until recovery/effect guarantees and current failures are clear.

## I. Balanced remediation roadmap

The following was the original recommendation; execution status is summarized
above and detailed in this report.

### Phase 0 — Establish a trustworthy baseline

- **Deliverables:** Attribute the 80 unit failures to the exact working-tree change set; record which failures are regressions, stale tests, or environment issues; preserve the focused 62-test result and real-service results as separate gates.
- **Dependencies:** None.
- **Acceptance:** All failures are individually dispositioned; required critical suites pass without disabling or weakening assertions; clean-checkout or change-range attribution is recorded.
- **Positive/negative tests:** Repeat deterministic passing paths; verify invalid authority, unavailable stores, stale snapshots, and malformed recovery remain denied.
- **Definition of done:** No unexplained failure remains and a reproducible baseline command set is documented.

### Phase 1 — Clarify runtime composition and reliability contracts

- **Status:** First increment complete: canonical entrypoint contract, liveness/readiness split, Compose readiness probe, worker quarantine/readiness state, and authority hold/recovery behavior are implemented and verified. Phase 1 remains in progress for the remaining reliability and durability work.
- **Deliverables:** Document local broker limits; define behavior for database outages; map task persistence versus event/WAL/effect durability.
- **Dependencies:** Phase 0.
- **Acceptance:** The canonical API path is executable as an import contract; readiness becomes false for the checked dependencies and worker quarantine; authority hold/recovery is tested. Database-outage and task/effect durability boundaries remain open.
- **Positive/negative tests:** Healthy startup and recovery; PostgreSQL/Redis outage; authority revoked during work; task interrupted before/after effect and publication.
- **Definition of done:** Operators can distinguish healthy, degraded, quarantined, and unavailable states without weakening fail-closed rules.

### Phase 2 — Deliver one real tool workflow

- **Deliverables:** Choose a canonical tool planner/registry/executor path; implement one concrete calendar or reminder integration (or remove unsupported capability claims); route effects through approval, authority, audit, and durable result handling.
- **Dependencies:** Phase 0 and the execution contract in Phase 1.
- **Acceptance:** A real user request reaches the selected external or local effect and returns an auditable result; unsupported/unauthorized requests do not cause effects.
- **Positive/negative tests:** Successful integration; missing credentials; invalid arguments; denied approval; revoked authority; downstream timeout and retry.
- **Definition of done:** One end-to-end integration is demonstrably useful, secure, and recoverable before expanding the tool catalog.

### Phase 3 — Make AI and memory failure behavior explicit

- **Deliverables:** Resolve model-context failures; establish provider abstraction boundaries; expose memory/vector/graph degradation; define source grounding and retrieval-result semantics.
- **Dependencies:** Phase 0; Phase 1 for dependency health.
- **Acceptance:** Active model and memory paths are observable; provider/runtime failure is distinct from empty answer or empty memory; grounded outputs identify evidence.
- **Positive/negative tests:** Context budget boundaries; provider unavailable; empty index versus retrieval failure; stale projection; invalid citation/source.
- **Definition of done:** A representative conversational workflow has traceable prompt/model/retrieval/effect/result evidence.

### Phase 4 — Complete one user-facing interaction path

- **Deliverables:** Browser-level coverage for the primary Streamlit/API interaction; choose whether PWA/offline and voice are supported; implement speech capture/transcription only if retained in scope.
- **Dependencies:** Phases 1 and 3.
- **Acceptance:** A browser test can create/resume a session, submit a turn, display errors/loading, and observe the response; voice/PWA claims are backed by actual workflows.
- **Positive/negative tests:** Session continuity, disconnected API/WebSocket, failed transcription, unsupported media, keyboard/accessibility smoke, offline behavior if claimed.
- **Definition of done:** One core workflow is verified from user input through UI and back, not only by Python component tests.

#### Canonical UI launcher follow-up (2026-11-10)

The original `launch.py --ui` path constructed a `DadBot` in the parent before
starting Streamlit, creating duplicate runtime initialization and delaying
readiness. The launcher now avoids that construction for UI mode, passes an
explicit `--model` selection to the Streamlit process, and bounds failed-start
cleanup instead of waiting indefinitely after its readiness timeout. Streamlit
is explicitly bound to `127.0.0.1`; this was added after the live probe showed
that the prior default exposed a `0.0.0.0` listener.

Evidence: the focused launcher/runtime/operator selection passed **49 tests**.
A real `launch.py --ui --model qwen3.5:9b --tenant-id ui-launch-probe` run with
isolated temporary profile, memory, graph, semantic, and session paths returned
HTTP 200 from `http://127.0.0.1:8501/_stcore/health`; the only listener was
`127.0.0.1`, and the port was closed after stopping the exact process tree.
An earlier 30-second probe using `Invoke-WebRequest` did not receive a
response, although the process log reported Uvicorn started on
`127.0.0.1:8501` and the listener was bound there; its request exception was
swallowed by the probe harness. Follow-up confirmed this shell is Windows
PowerShell 5.1, whose `Invoke-WebRequest` has no `-NoProxy` parameter. With an
explicit `127.0.0.1` URL and `[System.Net.WebRequest]::DefaultWebProxy = $null`,
the readiness probe returned HTTP 200 on attempt 3; an initial connection
refusal occurred while the app was starting. This is consistent with startup
timing and/or PowerShell proxy handling, rather than a persistent bind failure;
the original swallowed exception cannot be recovered. The listener remained
loopback-only and was stopped, with port 8501 confirmed closed. A repeated
probe using `curl.exe` also received HTTP 200.
This verifies launcher readiness and network binding, not browser interaction,
session continuity, or a full UI turn. The probe used a local Ollama endpoint
and does not certify external model/service behavior.

### Phase 5 — Reduce operations and maintenance cost

- **Deliverables:** Improve image/profile dependency split and least privilege; add coverage and type-check evidence; ratchet Ruff findings; document migrations, backup, restore, monitoring, and supported OS matrix.
- **Dependencies:** Phases 0–4 can proceed; security/deployment changes should follow explicit operational acceptance.
- **Acceptance:** CI checks all changed production modules and critical contracts; coverage is measured; production image includes only required features; restore drills are repeatable.
- **Positive/negative tests:** Build each supported deployment profile; verify least privilege and health failure; test backup restore and schema upgrade.
- **Definition of done:** Operational claims match verified deployment behavior and static/test gates enforce a measured baseline.

Phases 2 and 3 can proceed in parallel after the execution contract is stable; Phase 4 depends on the selected AI/service workflow. Phase 5 is ongoing but must not substitute for Phase 0 correctness triage.

## J. Evidence, exclusions, and limitations

- This report describes the checkout at HEAD `6616ed01` (`2026-11-09`) plus the current working tree. It is not an assessment of the commit alone.
- The worktree had 71 modified tracked paths, 2 deleted tracked files, and 10 untracked files at inspection. Tracked diff was approximately 4,588 additions and 1,229 deletions. The audit did not clean, stage, revert, or otherwise alter those changes.
- No `.env` or credential contents were read. Secret/config values and private test data were excluded.
- The unit suite’s final valid run used an isolated temporary directory. Earlier invalid harness attempts are not included in pass/fail totals.
- Live PostgreSQL/Redis results are from the immediately preceding integration work and existing audit record, not a fresh service run in this audit. The disposable services were stopped.
- The audit used source/test inspection and targeted tooling; it did not inspect every line of all 1,048 tracked Python files. Major execution paths and major subsystem evidence were sampled with bounded read-only exploration. External integrations, every MCP permission branch, deployment against production-like infrastructure, performance/load, and full Windows operations remain unverified.
- Findings about test failures are observed outcomes, not proven production root causes. Root-cause attribution requires Phase 0.
- At the time of the original audit, no source, test, configuration, migration,
  or existing audit/specification file was changed to produce its findings.
  Section K is a later follow-up that updates this report and the MCP
  implementation.

## K. Improvement 7 follow-up — authenticated local MCP server

The Dad-Bot-owned MCP service now serves Streamable HTTP on loopback by default
and refuses startup without the shared API signing secret and initialized,
durable authority database. Its token-exchange endpoint issues audience-bound
HMAC service tokens capped at 300 seconds. Each grant names the approved tools
and their exact MCP scopes; invocation revalidates active tenant membership and
the configured server tenant. Both tool and resource listings are filtered by
the grant, and resource reads execute through the same authority check as tools.
The checked-in manifest pins the names, scopes, schemas, and descriptions of all
26 registered tools; startup fails closed on schema drift.

Evidence collected for this follow-up:

- The focused MCP/authority/API/identity/WebSocket selection passed **183
  tests**; **1 test was skipped** because it requires an isolated Redis test
  instance. The run covered
  `test_local_mcp_tool_rush.py`, `test_mcp_token_exchange.py`,
  `test_mcp_tool_client.py`, `test_delegated_authority_acceptance.py`,
  `test_service_identity_binding.py`, `test_service_verified_principal.py`,
  `test_sqlite_service_state.py`, `test_service_api_integration.py`, and
  `test_service_websocket_tickets.py`.
- A real local MCP server process was started on loopback with a disposable,
  initialized SQLite authority database. An unauthenticated initialization
  returned HTTP 401; an audience-bound bearer token initialized successfully
  and `tools/list` returned only the one granted tool. The server process was
  stopped after the probe. API token exchange and tool/resource authorization
  were also exercised by the focused tests.
- Python compilation and Ruff import-order checks passed for the changed
  modules. The broader Ruff run still reports pre-existing style/complexity
  findings in the large API and MCP modules; those unrelated findings were not
  mass-fixed.

This closes the targeted authenticated-HTTP and per-tool-scope verification,
not the entire MCP production-security roadmap. The bearer exchange is Dad-Bot's
existing HMAC service-token scheme, **not a standard OAuth 2.1 authorization
server**. The manifest is a reviewed hash-based drift lock, **not a
cryptographically signed manifest**; no external drift-alerting pipeline or
public-network production deployment was verified. The live socket smoke test
used SQLite and did not combine a separately running API process with the MCP
process in one end-to-end deployment.
