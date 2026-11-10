# Durable memory implementation status

Status: **incomplete; not ready to commit or push**.

## D2.6: local semantic candidate publication (incomplete)

- For file-ledger tenants using SQLite semantic indexes, synchronization now
  captures the ledger revision and existing index hashes under the WAL fence,
  builds new embeddings without holding that fence, and rechecks the exact
  revision and committed catalog before publication. SQLite checks the
  captured index hashes and the resulting hashes in the same transaction as
  replacement and deletion. A changed index, changed revision, interrupted
  transaction, or failed verification rejects or rolls back the candidate.
- The real local-manager concurrency test now proves that a tenant commit
  completes while embedding generation is paused and that the older candidate
  is rejected without modifying the previously persisted index. Additional
  SQLite tests exercise a competing index change and rollback on verification
  failure. Existing subprocess interruption/restart coverage remains in the
  broader selection.
- Broader command: `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_semantic_index_repair_failures.py
  tests\unit\test_execution_ledger_boot_order.py
  tests\unit\test_tenant_memory_restore_authority.py
  tests\test_memory_manager.py tests\test_semantic_index_determinism.py
  -q --runxfail -o addopts='' --basetemp=.pytest_tmp_d26_broad`:
  **132 passed, 13 skipped**. Ruff F/E9 and Pylance syntax checks passed.
  `git diff --check` reported the existing LF/CRLF warning.
- D2.6 is **not complete**. There is no durable per-tenant projection metadata
  or generation protocol. The shared SQLite semantic table has a global
  summary-key primary key, and its reads, deletes, and clears are not
  tenant-scoped: sharing the physical index between tenants is unsafe.
  The transactional index comparison is not tenant isolation. The graph
  candidate still uses the bot I/O lock during construction, and shared graph
  storage isolation has not been established. The new non-blocking path is
  specific to local authority with SQLite; other backend paths retain their
  previous behavior. Hash verification does not validate embedding integrity.
  Indexed-mutation safeguards remain in place. The required metadata,
  cross-tenant, and fresh-process acceptance matrix remains unfulfilled.

## Completed work and current limitations

- The execution ledger is initialized before memory hydration and shared with
  the control plane. File and PostgreSQL ledger backends exist, with writer
  locking or transactional head checks, chain validation, tenant revision
  validation, and uncertain-append reconciliation.
- Session restore and rollback have partial guards against overwriting a
  committed tenant projection. Snapshot-only writes are blocked when an
  established tenant stream is observed, but the observation is not a writer
  fence.
- Unsafe ledger trimming is disabled. Existing session snapshots do not provide
  the tenant and control-plane checkpoint needed for safe WAL retention.
- Local WAL writers now reject unsealed tenant events before appending. The
  corruption-replay tests still inject invalid records directly into the WAL
  so that load-time rejection remains tested.

None of the six end-to-end implementation and acceptance phases is complete.
In particular, ledger primitives are not yet the application's memory authority.

## Current changed files

Tracked changes: `README.md`, `dadbot/base/_memory_lifecycle_mixin.py`,
`dadbot/core/boot_mixin.py`, `dadbot/core/compaction.py`,
`dadbot/core/durability.py`, `dadbot/core/execution_ledger.py`,
`dadbot/core/ledger_backend.py`, `dadbot/core/orchestrator.py`,
`dadbot/core/snapshot_engine.py`, `dadbot/core/turn_mixin.py`,
`dadbot/memory/normalizers.py`,
`dadbot/memory/semantic_manager.py`, `dadbot/memory/storage.py`,
`dadbot/services/persistence.py`, `dadbot/state.py`,
`tests/test_memory_manager.py`, `tests/test_production_hardening_wave3.py`,
`tests/test_production_hardening_wave4.py`, and
`tests/unit/test_control_plane_ledger_bounds.py`.

Untracked implementation and tests: `dadbot/core/postgres_ledger_backend.py`,
`dadbot/core/tenant_memory_ledger.py`,
`tests/integration/test_postgres_execution_ledger.py`,
`tests/unit/test_execution_ledger_boot_order.py`,
`tests/unit/test_memory_authority_application.py`,
`tests/unit/test_semantic_index_repair_failures.py`,
`tests/unit/test_snapshot_tail_recovery.py`,
`tests/unit/test_tenant_memory_ledger.py`,
`tests/unit/test_tenant_memory_restore_authority.py`, and
`tests/unit/test_wal_recovery_integrity.py`,
`tests/unit/test_lifecycle_checkpoint_failures.py`. This status file is also
untracked. Preserve these changes; their presence does not mean they have all
been verified in the latest run.

## Actual verification

- Latest D2.5 work: **PARTIAL / NOT ACCEPTED**. Local boot now calls
  `MemoryStorageBackend.recover_local_projections` after hydration and runtime
  cache initialization. It attempts graph and semantic synchronization
  independently, logs expected projection failures with tracebacks, records
  in-process `degraded` or `sync_succeeded` status, and allows startup to
  continue after those projection failures. Ledger corruption and revision
  conflicts still fail startup. Semantic sync uses the existing WAL fence and
  SQLite transactional publication; graph sync uses its revision check.
  These statuses are **not durable projection metadata** and `sync_succeeded`
  does not assert verified content parity.
- Local semantic retrieval now compares stored content hashes to the
  authoritative memory catalog under the WAL fence before reading indexed
  rows; both the public memory-manager retrieval path and the semantic
  manager path return no semantic matches and report `stale_index` if the
  hashes disagree. A test first reproduced a stale hash being returned as a
  normal match. This protects the tested path but is not a substitute for
  per-tenant revision metadata, embedding-version validation, or a full audit
  of all direct backend consumers.
- The real subprocess semantic transaction test now inspects SQLite before
  starting the fresh DadBot process to confirm the previous index survived the
  terminated transaction; after boot, it checks that startup automatically
  repaired the index and retained tenant revision 2. A separate application
  test verifies graph recovery can fail while semantic sync succeeds.
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_semantic_index_repair_failures.py
  tests\unit\test_execution_ledger_boot_order.py -q --runxfail -o addopts=''
  --basetemp=.pytest_tmp_projection_coordinator2`: **37 passed** before adding
  the semantic-failure/graph-success test and renaming the in-memory status.
- Broader run:
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_semantic_index_repair_failures.py
  tests\unit\test_execution_ledger_boot_order.py
  tests\unit\test_tenant_memory_restore_authority.py
  tests\test_memory_manager.py tests\test_semantic_index_determinism.py
  -q --runxfail -o addopts=''
  --basetemp=.pytest_tmp_projection_coordinator_broad`: **130 passed,
  13 skipped**. Ruff F/E9 passed and `git diff --check` emitted the existing
  LF/CRLF warning.
- **Still missing:** independent durable projection revisions, dirty state,
  generations, content verification for graph, tenant isolation for shared
  indexes, safe indexed mutations, and several required crash/failure tests.
  The index-changing mutation safeguard remains. Semantic embedding is still
  performed while holding the WAL writer fence; startup can wait for embedding
  availability. Windows evidence replacement remains intermittently failing.

- Latest startup-recovery work: **PARTIAL / NOT ACCEPTED**. For the local file
  authority, DadBot now rebuilds the graph after ledger replay and runtime
  cache initialization but before orchestration and READY. The existing
  graph sync constructs from hydrated memory and rechecks the tenant revision
  under the WAL lock before SQLite replacement. A startup rebuild failure
  propagates and prevents READY; it is not treated as a successful repair.
  Graphs are rebuilt on every such startup, not tracked with durable projection
  revisions or dirty metadata. This is not a complete recovery coordinator.
- `test_startup_rebuilds_stale_graph_from_ledger` first failed because the
  stale persisted graph survived restart. After the change, the local
  startup/authority selection
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_local_tenant_establishment_application.py
  tests\unit\test_execution_ledger_boot_order.py -q -o addopts=''
  --basetemp=.pytest_tmp_startup_graph_first` produced **43 passed**.
  `test_process_exit_during_startup_graph_rebuild_recovers_on_restart`
  uses a real subprocess exiting inside a SQLite graph replacement transaction;
  another fresh process automatically removes the stale graph and confirms
  tenant revision 1. Its focused command with
  `--basetemp=.pytest_tmp_startup_graph_kill` produced **1 passed**.
- Broader local selection:
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_execution_ledger_boot_order.py tests\test_memory_manager.py
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_memory_authority_application.py
  tests\unit\test_local_tenant_establishment_application.py
  tests\unit\test_tenant_memory_ledger.py
  tests\unit\test_wal_recovery_integrity.py
  tests\unit\test_lifecycle_checkpoint_failures.py
  tests\unit\test_memory_snapshot_write_guard.py
  tests\unit\test_tenant_memory_restore_authority.py
  tests\unit\test_semantic_index_repair_failures.py -q --runxfail
  -o addopts='' --basetemp=.pytest_tmp_startup_graph_full` produced
  **199 passed, 13 skipped**. Ruff F/E9 and `git diff --check` passed
  with the existing LF/CRLF warning.
- Semantic startup recovery, independent projection metadata, verified
  publication generations, degraded semantic retrieval, and indexed
  mutations are **not implemented**. Semantic reconstruction needs embedding
  services that may be unavailable at boot and currently holds the WAL lock
  while computing embeddings; simply invoking it at boot would make
  availability and concurrency worse. The safeguard on indexed mutations
  remains. Graph recovery also depends on the existing graph inputs and has
  not been shown to isolate tenants sharing an index. Windows evidence
  replacement failures remain unresolved.

- Latest D2 recovery work: **PARTIAL / NOT ACCEPTED**. The SQLite semantic
  backend now publishes replacement rows and stale-key deletions in a single
  SQLite transaction through `SQLiteSemanticIndex.apply_changes`. Previously
  these were separate transactions, so a failed deletion left newly inserted
  rows alongside stale rows. The existing semantic manager uses the atomic
  path for SQLite; non-SQLite backend behavior is unchanged.
- A real SQLite trigger reproduces the former partial-publication failure:
  deleting the old row raises `sqlite3.IntegrityError` after the replacement
  row was inserted. Before this change the replacement remained visible;
  after it, the old index is unchanged. An additional real subprocess test
  exits during the SQLite transaction after the replacement insert and
  before stale-row deletion. A fresh DadBot process observes the previous
  index and authoritative tenant revision 2, then explicitly synchronizes
  the committed catalog and verifies exactly one new row and unchanged
  revision. This demonstrates transaction rollback and an explicit repair,
  **not** automatic startup recovery.
- `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_semantic_index_repair_failures.py -q -o addopts=''
  --basetemp=.pytest_tmp_semantic_atomic_process`: **4 passed**.
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_semantic_index_repair_failures.py
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_tenant_memory_restore_authority.py tests\test_memory_manager.py
  -q -o addopts='' --basetemp=.pytest_tmp_semantic_atomic_broad`:
  **99 passed, 13 skipped**.
  `.\.venv\Scripts\python.exe -m pytest tests\test_semantic_index_determinism.py
  -q -o addopts='' --basetemp=.pytest_tmp_semantic_atomic_determinism`:
  **14 passed**. Ruff F/E9 and `git diff --check` passed, with the existing
  LF/CRLF warning.
- Independent semantic/graph revision and dirty metadata, generation
  publication, startup recovery, safe semantic retrieval, and automatic
  repair after process termination are **not implemented**. Indexed memory
  mutations remain guarded. The process test uses an explicit ledger-level
  external commit because the normal indexed mutation path is not yet
  available. Windows evidence replacement remains intermittently failing;
  no fix or new durability claim is made.

- Latest D2 work: **PARTIAL / NOT ACCEPTED**. For local ledger-backed
  tenants, `MemoryGraphManager.clear_graph_store` now explicitly rejects an
  independent graph clear, and `MemoryCoordinator` rejects direct graph-edge
  replacement rather than logging and hiding the failure. Local
  `graph_snapshot` no longer treats an unversioned persisted graph as current:
  it constructs an in-memory projection and checks the captured tenant
  revision and memory projection under the WAL writer fence before returning.
  Stale local instances fail explicitly rather than returning that graph.
  This is a conservative fallback, not revision-aware projection recovery.
- The direct-clear regression failed before the change because no exception
  was raised. A separate retrieval regression reproduced a persisted graph
  containing an unverified deleted node being returned; it passes with the
  in-memory fallback. The focused run
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_semantic_index_repair_failures.py
  tests\unit\test_tenant_memory_restore_authority.py
  tests\test_memory_manager.py -q -o addopts=''
  --basetemp=.pytest_tmp_graph_direct_broad` produced **96 passed,
  13 skipped**. Ruff F/E9 and `git diff --check` passed, with the existing
  LF/CRLF warning. After adding a test that verifies a stale instance's graph
  read fails closed, the focused three-file selection with
  `--basetemp=.pytest_tmp_graph_direct_final` produced **27 passed**.
- A separate run including local establishment produced **37 passed,
  2 failed**: both failures were `PermissionError` from Windows evidence
  `os.replace`, including one during repeated replacement after the explicit
  sharing-violation handle had been closed. The two failures passed when
  rerun individually, but the intermittent issue is not resolved and no
  retry was introduced. A closed intentional test handle alone therefore
  does not guarantee replacement success in this environment.
- Independent projection metadata, index recovery, process-termination
  recovery, direct backend write enforcement, and stale semantic retrieval
  protections remain unimplemented. Direct graph clear and direct edge
  invalidation are currently unavailable for ledger-backed tenants; these
  guards do not satisfy the requested complete functionality. The safeguard
  blocking index-changing memory mutations remains unchanged.

- Milestone D2 follow-up: **PARTIAL / NOT ACCEPTED**. Normal local graph
  `sync_graph_store` now builds outside the WAL writer lock, then takes the
  writer lock and validates the sealed ledger, the captured tenant revision,
  and the normalized source projection before replacing the graph. The
  runtime I/O lock protects the source while the graph is built. A stale
  instance or a newer concurrent tenant commit raises
  `TenantMemoryRevisionConflict` before graph replacement. This does not
  fence direct graph backend writes, graph clearing, or graph reads, and does
  not implement projection metadata, startup rebuild, or crash recovery.
  The index-changing mutation safeguard remains in place.
- The new stale-instance regression first failed because graph sync did not
  raise after a second instance committed revision 2. After the change,
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_tenant_memory_restore_authority.py
  tests\unit\test_semantic_index_repair_failures.py -q -o addopts=''
  --basetemp=.pytest_tmp_graph_fence_v2` produced **23 passed**. It includes
  a coordinated concurrent commit during graph construction and checks that
  the stale replacement is not invoked. It does not test process termination
  during graph publication.
- Windows evidence investigation: a real `CreateFileW` handle opened without
  `FILE_SHARE_DELETE` causes `os.replace` to raise `PermissionError`; the
  prior evidence remains readable and unchanged. After closing that handle,
  20 consecutive evidence replacements succeeded. This demonstrates one
  reproducible sharing-violation mechanism, not the cause of the earlier
  intermittent failure. No retry or exception suppression was introduced.
  The test also repeats two-process initial startup three times and verifies
  each evidence marker's initial-event hash. Command:
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_tenant_establishment_application.py::test_windows_evidence_replacement_sharing_violation_preserves_old_evidence
  tests\unit\test_local_tenant_establishment_application.py::test_concurrent_first_startup_establishes_one_event
  -q -o addopts='' --basetemp=.pytest_tmp_windows_evidence_repeated`:
  **4 passed**. These are not power-loss durability tests.
- Broader local selection using `--basetemp=.pytest_tmp_graph_fence_broad`
  produced **189 passed, 13 skipped**, before parametrizing concurrent startup
  for three iterations. Ruff F/E9 and `git diff --check` passed, with the
  existing LF/CRLF warning. PostgreSQL application integration and retention
  were not started.

- Latest semantic/graph recovery assignment: **PARTIAL / NOT ACCEPTED**.
  `MemoryStorageBackend.fenced_semantic_index_write` now acquires the local
  I/O lock and WAL writer lock, reloads and validates the execution ledger,
  and rejects a semantic input catalog that does not match the committed
  tenant catalog. Existing `SemanticIndexManager.sync_semantic_memory_index`
  and `clear_semantic_memory_index` use this fence for local file authority;
  non-local behavior is unchanged. The fence is held during the existing
  index update, not a second index implementation. This prevents a stale
  queued catalog or direct clear from overwriting the current local semantic
  index, and serializes that update with cooperating WAL writers.
- `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_semantic_index_repair_failures.py -q -o addopts=''
  --basetemp=.pytest_tmp_semantic_fencing_v3`: **13 passed**. Tests use the
  real local memory manager and semantic backend to check stale-catalog
  rejection after another writer's commit, removal using the current
  committed catalog, and writer-fence serialization. They do not prove
  index revision tracking or process-termination recovery. After strengthening
  the serialization assertion to wait for the blocked writer, the same
  selection with `--basetemp=.pytest_tmp_semantic_final` again produced
  **13 passed**. Ruff F/E9 and Pylance syntax checks passed.
- Broader local baseline with `--basetemp=.pytest_tmp_semantic_full`:
  **183 passed, 13 skipped, 1 failed**. The failing subprocess test
  `test_real_startup_reconciles_process_termination[before_repair]` exited
  before its intended termination because Windows `os.replace` raised
  `PermissionError` while writing establishment evidence. A separate earlier
  selection reported **82 passed, 13 skipped, 1 teardown error** from
  `PermissionError` replacing a JSON snapshot. Rerunning the two individual
  failing cases with `--basetemp=.pytest_tmp_semantic_rerun_failures`
  produced **2 passed**; the intermittent Windows replacement failure is
  still unproven and not suppressed.
- Graph writes and reads are not fenced or revision-tracked. Direct graph
  replacement in memory coordination and persistence rollback bypasses
  `sync_graph_store`. Graph snapshot can return a persisted stale graph.
  The semantic background drain logs failures but does not yet expose durable
  dirty/rebuild status; no startup rebuild or per-index revision markers
  exist. The existing safeguard rejecting mutations requiring semantic or
  graph repair remains in place. Consequently additions, indexed clears, the
  requested crash/restart acceptance, and the complete local recovery
  protocol remain open. No PostgreSQL or checkpoint retention work was begun.

- Latest local integration assignment: **PARTIAL / NOT ACCEPTED**. Reproduced
  the preceding combined baseline with `-rs`: **179 passed, 13 skipped**,
  including both formerly failing boot assertions and all seven formerly
  failing fixture teardowns. The 13 skipped tests were then run separately:
  `.\.venv\Scripts\python.exe -m pytest tests\test_memory_manager.py -m slow
  --run-stress -q -rs -o addopts='' --basetemp=.pytest_tmp_slow_durable`:
  **12 passed, 71 deselected**; and
  `.\.venv\Scripts\python.exe -m pytest
  tests\test_memory_manager.py::test_memory_dedup_key_collapses_finance_and_work_stress_variants
  --run-stress -q -o addopts='' --basetemp=.pytest_tmp_stress_dedup`:
  **1 passed**. The unmarked deduplication test was skipped because its name
  contains `stress`; the stress-profile forgetting test is marked slow but
  also skipped by the name-based stress gate. The others have slow markers.
  The 13 individual tests are:
  - `test_memory_state_hash_is_identical_across_10_same_input_runs`
  - `test_memory_dedup_key_collapses_finance_and_work_stress_variants`
  - `test_controlled_forgetting_archives_old_low_signal_noise`
  - `test_controlled_forgetting_soft_prunes_under_memory_saturation`
  - `test_controlled_forgetting_prioritizes_low_signal_conflicting_memory`
  - `test_controlled_forgetting_stress_profile_reduces_retention_under_density`
  - `test_build_memory_context_places_recent_archive_notes_before_older_memories`
  - `test_sync_graph_store_extracts_consolidated_archive_traits_and_patterns`
  - `test_graph_retrieval_for_input_ranks_consolidated_over_weaker_archive`
  - `test_graph_retrieval_surfaces_contradictions_in_supporting_evidence`
  - `test_build_memory_context_uses_graph_first_and_semantic_second`
  - `test_build_memory_context_includes_active_consolidated_context`
  - `test_build_memory_context_includes_deep_pattern_context`
  All concern memory correctness, including graph projection or retrieval;
  none establishes local WAL crash safety or ledger-fenced projection repair.
- Local clear now enters the memory manager's revision-checked local commit
  path rather than snapshot-only persistence. The application test first
  reproduced the previous snapshot-authority rejection, then verified a
  mood-only clear commits revision 3 and survives restart. Stale clear now
  raises a revision conflict. **Clear with existing indexed memories remains
  explicitly blocked** by the existing semantic/graph safeguard; a test
  verifies that such rejection does not change the committed revision.
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_tenant_memory_restore_authority.py -q -o addopts=''
  --basetemp=.pytest_tmp_local_clear_fix2`: **15 passed** before adding the
  explicit indexed-clear guard test. The complete baseline selection rerun
  with that test included and `--basetemp=.pytest_tmp_local_clear_full`
  produced **181 passed, 13 skipped**.
- Blocking implementation detail: `sync_semantic_memory_index` can be called
  by an asynchronous queue with a previously captured memory list, and graph
  synchronization is also called outside the WAL writer boundary. Boot loads
  memory before the runtime/index locks exist. Merely removing the safeguard
  would allow stale unfenced index writes and does not provide revision
  markers or restart repair. Those call sites, index revision tracking, and
  startup repair still require coordinated implementation and tests. Local
  additions, updates, indexed clears, delete, durable restore, crash tests
  for index repair, and concurrent restore remain incomplete.

- Combined Phase A/local establishment/lifecycle run:
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_execution_ledger_boot_order.py tests\test_memory_manager.py
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_memory_authority_application.py
  tests\unit\test_local_tenant_establishment_application.py
  tests\unit\test_tenant_memory_ledger.py
  tests\unit\test_wal_recovery_integrity.py
  tests\unit\test_lifecycle_checkpoint_failures.py
  tests\unit\test_memory_snapshot_write_guard.py
  tests\unit\test_tenant_memory_restore_authority.py -q --runxfail
  -o addopts='' --basetemp=.pytest_tmp_phase_a_combined`: **179 passed,
  13 skipped**. Ruff F/E9 passed for the five edited test files and
  `git diff --check` passed with the existing LF/CRLF warning.
- Phase A baseline follow-up: **160 passed, 13 skipped, no failures or
  teardown errors** using `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_execution_ledger_boot_order.py tests\test_memory_manager.py
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_memory_authority_application.py
  tests\unit\test_local_tenant_establishment_application.py
  tests\unit\test_tenant_memory_ledger.py
  tests\unit\test_wal_recovery_integrity.py -q --runxfail -o addopts=''
  --basetemp=.pytest_tmp_phase_a_verified`.
  The two boot tests now verify replay of the existing tenant and revision 1
  for a newly established tenant rather than asserting the former refusal and
  revision 0. Seven teardown errors were caused by failure-injection patches
  remaining active during fixture shutdown; the scoped
  `persistence_failure_patch` fixture restores those patches before shutdown
  without suppressing actual shutdown failures. The tests still assert the
  original injected failures.
- Additional Phase A lifecycle baseline: the first run of
  `tests\unit\test_lifecycle_checkpoint_failures.py`,
  `tests\unit\test_memory_snapshot_write_guard.py`, and
  `tests\unit\test_tenant_memory_restore_authority.py` produced **13 passed,
  6 failed** because the tests attempted first commits at revision 0 after
  local startup already established revision 1. The updated tests commit at
  revision 1, check revision conflicts for stale mutation/save and the
  snapshot-only guard for clear, and restore the committed projection before
  fixture shutdown. The direct snapshot guard test no longer replaces
  shutdown with a no-op. The rerun command
  `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_lifecycle_checkpoint_failures.py
  tests\unit\test_memory_snapshot_write_guard.py
  tests\unit\test_tenant_memory_restore_authority.py -q -o addopts=''
  --basetemp=.pytest_tmp_phase_a_lifecycle_fix` produced **19 passed**.
  These assertions do not prove concurrent restore or graph repair fencing.
- Phase A has **not** established a complete persistence call graph or clean
  baseline for all application and PostgreSQL tests. The following confirmed
  paths require integration work:
  - Local startup: `DadBotBootMixin` initializes the execution ledger before
    `MemoryStorageBackend.load_memory_store`, which invokes
    `establish_local_tenant` under the WAL lock, repairs the snapshot, then
    emits `MemoryInitialized`.
  - Local mutation: `mutate_memory_store` builds a private candidate and
    `_commit_local_projection` invokes `commit_local_tenant` to check the
    revision, append, publish, and repair the snapshot. Index-changing
    candidates are rejected pending fenced index repair.
  - Direct save and shutdown: `save_memory_store` compares against refreshed
    tenant state and either does nothing or calls the local commit path;
    `shutdown` calls this save and is registered with `atexit`.
  - Non-local mutation: `_coordinated_commit` uses snapshot, semantic and
    graph participants with rollback; it is not the tenant ledger authority.
    `clear_memory_projection` separately writes a snapshot and indexes and
    refuses established authority. Projection replacement is also used for
    staging and rollback, not a durable restore event.
  - Semantic queue/drain and `sync_semantic_memory_index`, graph
    `sync_graph_store`, and persistence transaction rollback can write indexes
    without a tenant-ledger writer fence. They must be included in the repair,
    not just the normal memory-manager mutation.
- An existing ledger event containing an invalid mood can be replayed as one
  value but normalized to another by the memory manager; full-state
  validation/legacy normalization at the authority boundary remains open.
  PostgreSQL application establishment, index recovery, and Windows
  power-loss guarantees remain unimplemented or unproven.

- Latest projection-order change: **PARTIAL / NOT ACCEPTED**. Local mutation
  appends and fsyncs before publishing the committed in-memory projection,
  then repairs the snapshot while holding the same WAL writer lock. If
  snapshot repair raises, the committed in-memory value remains visible, the
  error propagates, and further changes are blocked until restart. A no-op
  save/shutdown does not append another event. Direct staged saves already
  have their candidate in memory. This does not yet implement fenced semantic
  or graph index recovery; changes requiring those indexes remain rejected.
- `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_memory_authority_application.py
  tests\unit\test_local_tenant_establishment_application.py
  tests\unit\test_tenant_memory_ledger.py
  tests\unit\test_wal_recovery_integrity.py -q --runxfail -o addopts=''
  --basetemp=.pytest_tmp_projection_order`: **78 passed**. The unchanged
  strict-xfail regression was exercised with `--runxfail`.
- `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_execution_ledger_boot_order.py tests\test_memory_manager.py
  -q -o addopts='' --basetemp=.pytest_tmp_projection_related`:
  **80 passed, 13 skipped, 2 failed, 7 teardown errors**. The two boot tests
  assert old establishment behavior; teardown errors occur when injected
  persistence failures remain active during shutdown. These failures have not
  been fixed or hidden.
- PostgreSQL application authority and crash tests were **not run or
  implemented** in this change. Windows directory-entry power-loss durability
  remains **UNSUPPORTED / UNPROVEN**. Restore, rollback, mutation coverage,
  index recovery, and application operation-ID reconciliation remain open.

- Latest local mutation work: **PARTIAL / NOT ACCEPTED**. Added
  `tests/unit/test_local_authority_mutation_application.py` and changed
  `dadbot/core/local_tenant_establishment.py`, `dadbot/memory/storage.py`,
  `dadbot/base/_memory_lifecycle_mixin.py`, and `README.md`. The local
  file-backed path constructs a private candidate, checks the expected tenant
  revision and establishment evidence under the WAL lock, appends and fsyncs
  a tenant event, repairs the snapshot under the same lock, reloads the
  ledger, and then published memory in that earlier version. A failed or uncertain append/repair blocks
  further writes until restart; stale staged saves are rejected.
- `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_authority_mutation_application.py
  tests\unit\test_memory_authority_application.py
  tests\unit\test_local_tenant_establishment_application.py
  tests\unit\test_tenant_memory_ledger.py
  tests\unit\test_wal_recovery_integrity.py -q --runxfail -o addopts=''
  --basetemp=.pytest_tmp_local_mutation_final2`: **78 passed**. This includes
  the unchanged strict-xfail regression run with xfail handling disabled,
  staged save and stale competing writer tests, and an actual process exit
  after a mood mutation's append but before snapshot repair. The xfail marker
  remains unchanged.
- PostgreSQL backend integration suite against disposable PostgreSQL 16:
  **21 passed**; this is not an application-level PostgreSQL establishment or
  recovery test. Ruff F/E9 and `git diff --check` passed, with the existing
  LF/CRLF warning. Pylance syntax checks passed for the three changed
  application modules; Pylance did not recognize the new test as a user file.
- **BLOCKED / NOT ACCEPTED**: Memory and other changes requiring semantic or
  graph repair are explicitly rejected to avoid publishing unfenced indexes.
  Clear, delete, durable restore, rollback, and PostgreSQL tenant authority
  remain incomplete. Operation IDs are generated internally rather than
  exposed for application-level retry reconciliation. An uncertain outcome
  requires restart. Windows directory-entry power-loss durability remains
  unproven. All acceptance gates A-M have not been independently proven;
  passing the local tests does not establish full concurrent, crash, index,
  or cross-backend correctness.
- Latest milestone A work: **PARTIAL / NOT ACCEPTED**. Added
  `dadbot/core/local_tenant_establishment.py` and
  `tests/unit/test_local_tenant_establishment_application.py`; modified
  `dadbot/core/ledger_backend.py`, `dadbot/core/boot_mixin.py`,
  `dadbot/memory/storage.py`, and `dadbot/base/_memory_lifecycle_mixin.py`.
  This status file and `README.md` were also updated.
- The local application startup now holds the WAL writer lock while reading
  a tenant-specific pending intent, writing the initial event, marking
  establishment, and repairing the snapshot. It uses a stable tenant-derived
  operation ID and replays before publishing initialization. It checks
  established evidence against the initial event and refuses a missing WAL.
  Existing tenant streams without evidence are used to rebuild evidence.
  Shutdown is idempotent after success and avoids an unnecessary save when
  the projection matches the tenant ledger.
- `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_local_tenant_establishment_application.py -q -o addopts=''
  --basetemp=.pytest_tmp_authority_final`: **10 passed**. Includes real
  application startup and restart, concurrent initial startup, invalid legacy
  snapshot, missing WAL, missing evidence, disabled-ledger rejection, and
  subprocess termination after intent, after append, after evidence, and before
  snapshot repair. These are process-termination tests, not power-loss tests.
- `.\.venv\Scripts\python.exe -m pytest
  tests\unit\test_tenant_memory_ledger.py
  tests\unit\test_wal_recovery_integrity.py -q -o addopts=''
  --basetemp=.pytest_tmp_authority_ledgers`: **61 passed**.
- Unchanged application regression with `--runxfail`: **1 failed**. Startup
  revision 1 and snapshot equality assertions passed, but normal mutation
  raises the existing snapshot-authority guard; the unified durable commit
  path remains absent. This means enabling the file ledger currently blocks
  normal durable memory changes after startup.
- Existing boot-order tests: **10 passed, 2 failed**. Their existing assertions
  expect established streams to be rejected rather than replayed and a fresh
  tenant to remain revision 0; neither test has been changed. The intended
  behavioral shift and full migration compatibility still need review.
- **BLOCKED / NOT PROVEN**: PostgreSQL establishment and transaction tests
  are not implemented. Local Windows directory-entry power-loss durability
  is not proven; `_write_evidence` fsyncs file contents but only fsyncs the
  parent directory on non-Windows systems. Interrupted partial WAL writes
  fail closed rather than automatically repairing the partial tail. Loss or
  rollback of both the WAL and its independent evidence cannot be detected.
  Concurrent writer fencing for semantic/graph repair, all ordinary durable
  mutation paths, and established legacy event normalization remain unresolved.
  Do not call milestone A complete or use this as a production authority.
- Ruff F/E9 and `git diff --check` passed; the latter emitted the existing
  LF/CRLF warning for `tests/test_memory_manager.py`. Pylance syntax checks
  passed for the new establishment module, memory storage, and test module.
- Latest assignment: **NOT IMPLEMENTED**. Inspected the current branch,
  worktree, startup/storage implementation, backend transaction boundaries,
  and existing untracked test files. No tests were run and no authority,
  hydration, or recovery implementation was changed in this attempt.
  Milestone A remains **FAIL**: the previously reproduced application
  regression still needs a serialized establishment protocol and application
  integration. Milestones B-F remain **NOT IMPLEMENTED**.
- Latest snapshot guard work: direct calls to the document-store/JSON snapshot
  writer and to the JSON writer now reject writes when a refreshed ledger has
  a tenant revision. An application-level test first reproduced both the
  former direct-write bypass and the corrected behavior. This is still only a
  check-then-write safeguard; it does not hold a backend writer boundary and
  cannot prevent a concurrent tenant append between the check and the write.
  Future replay repair needs a distinct fenced projection-write path.
- Latest focused run: direct snapshot guard plus tenant-ledger, WAL integrity,
  and boot-order tests: **75 passed**. A broader run including tenant restore
  had **21 passed and 8 teardown errors** from the previously documented
  shutdown snapshot guard. Pylance syntax checks for the two changed Python
  files and Ruff F/E9 passed.
- The application migration regression was run with `--runxfail` again:
  **failed**, revision 0 rather than 1. No crash-safe establishment, evidence,
  replay-based hydration, or application process-termination tests were
  implemented in this run.
- Independent reproduction: constructed an actual `DadBot` with a fresh local
  durable execution WAL and a legacy JSON snapshot containing
  `{"last_mood": "neutral"}`. Startup returned tenant revision **0** and the
  hydrated mood was `neutral`; the standalone process exited with status 1
  after checking for revision 1. This did not rely on the xfailed pytest test.
- Startup trace: `_initialize_execution_ledger` selects and loads the backend,
  then `_hydrate_profile_and_memory` calls `load_memory_store`. At
  `MemoryStorageBackend.load_memory_store`, a nonzero tenant revision is
  rejected; revision 0 falls through to snapshot loading and `MemoryInitialized`
  projection without calling `commit_tenant_memory`. Thus the missing first
  append is between validation/normalization of the legacy snapshot and
  publishing that initial projection. This is not safely fixed by adding a
  commit there: the snapshot read, initial event, independent establishment
  evidence, replay, and snapshot repair require the same backend writer fence.
- In the latest investigation, no establishment implementation or new
  application recovery tests were added. The existing PostgreSQL subprocess
  test checks an already-established ledger primitive, not first migration or
  application hydration. Shutdown still calls the snapshot-only save, whose
  authority guard rejects established tenant streams.
- Latest milestone attempt: the application regression was run with
  `--runxfail` and **failed** at startup: revision was 0, not 1. No authority
  establishment or hydration integration was implemented in this attempt.
- Latest combined local ledger, boot, and memory-manager run:
  **143 passed, 13 skipped, 7 teardown errors**. The errors occur because the
  tests leave injected persistence failures active during fixture shutdown,
  and explicit shutdown now propagates those failures.
- Latest PostgreSQL ledger run on a disposable PostgreSQL 16 container:
  **21 passed, 1 teardown error**. The teardown attempts a snapshot-only
  shutdown save after the test establishes a tenant ledger revision; the
  existing snapshot-authority guard rejects that save. At process exit,
  repeated shutdown also accesses tables already dropped by the fixture.
  Consequently this is not a clean 21-test pass.
- The prior shutdown change exposes these failures and repeated atexit calls;
  handling shutdown lifecycle and fixture cleanup without hiding real flush
  errors remains necessary. The previous narrow 28-test run did not cover
  these cases.
- Lifecycle follow-up: **28 passed** for focused boot, checkpoint, shutdown,
  graph delegation, and runtime-provider tests. A broader selection had
  **86 passed, 3 failed**: one source-layout assertion expects
  `bot.execute_turn(` in `dadbot/app_runtime.py`, and two manager-smoke tests
  fail because `MemoryCommandManager.build_memory_transcript` calls the
  missing `DadBot.parse_tool_command`. The latter call and missing app-runtime
  text also occur in HEAD; neither issue was fixed in this follow-up.
- The facade currently delegates graph refresh, hydrates before READY, invokes
  manager validation by default, and raises when the memory manager is
  missing. Its constructor does not currently declare `validate_managers`;
  boot instead exposes `skip_manager_validation`.
- The turn-summary `MemoryLedger` is independent of the execution ledger and
  is not tenant-memory authority. Missing event taps now produce a warning,
  invalid or failed snapshots and configured checkpoint errors propagate,
  and explicit shutdown propagates final flush errors. This does not implement
  durable tenant-memory replay or durable checkpoints.
- Current run: `tests/unit/test_tenant_memory_ledger.py`,
  `tests/unit/test_wal_recovery_integrity.py`, and
  `tests/unit/test_execution_ledger_boot_order.py`: **73 passed**.
- Current run against disposable PostgreSQL 16:
  `tests/integration/test_postgres_execution_ledger.py`: **21 passed**.
  These are ledger/backend integration tests, not end-to-end memory-manager
  crash or recovery tests.
- Current run: Ruff `--select F,E9` for the four files changed in the latest
  integrity fix passed; `git diff --check` passed with a pre-existing
  LF/CRLF warning for `tests/test_memory_manager.py`. Pylance syntax checks
  for those four files found no syntax errors.
- Current run: the existing application regression, executed with
  `--runxfail`, **failed**: startup returned tenant revision 0 rather than 1.
  Its strict xfail marker predates this work and must not be removed until the
  unchanged test passes.
- Earlier runs reported 129 passed and 14 skipped for a restore/transaction
  selection, plus three failures in a broader service-architecture run.
  Those results were not independently repeated in this run.

## Remaining work and root causes

1. Establish persistent tenant authority before memory hydration, with a
   crash-safe establishment record that prevents silent reimport if a WAL is
   lost. Serialize migration and competing migrations at the backend writer
   boundary. Reject unsupported and unsafe backend transitions.
2. Replace snapshot-only writes with one operation-ID, expected-revision
   tenant commit path. Include normal mutations, staged writes, direct saves,
   clears, durable restores, migrations, and shutdown. The current mutation
   path publishes state before `_coordinated_commit`, and its snapshot
   compensation can restore older state, so it is not ledger-atomic.
3. Replay the tenant stream on startup, repair snapshot and derived semantic
   and graph projections, and roll forward after any post-commit failure.
   Fence repair and writes across processes. Uncertain appends must block
   writes until the operation ID is reconciled.
4. Make session restore and transaction rollback preserve the current committed
   revision under concurrent writers; an intentional durable restore must be a
   new ledger mutation.
5. Add and pass real memory-manager crash and competing-writer tests on local
   and PostgreSQL backends, including startup migration/restarts, stale
   snapshots, tenant isolation, clear, staged/direct saves, restore, rollback,
   and repair failures. The passing backend tests do not establish this.
6. Only after recovery is proven, implement and test a durable checkpoint
   covering all projections and chain/sequence and operation-ID state needed
   for checkpoint-plus-tail replay. Test crashes around checkpoint creation
   and history removal before enabling compaction.

The immediate next action is to design and implement the crash-safe authority
establishment protocol across file and PostgreSQL writer boundaries, then
integrate it into memory hydration. Continue through all remaining steps and
the full acceptance tests; do not replace them with ledger-only tests or
snapshot guards. Update documentation to reflect only verified behavior.
Commit and push only after all acceptance gates pass and after checking that
unrelated user changes are not included.

The local protocol must distinguish an incomplete establishment intent from
completed independent evidence, while retaining enough durable information to
recover an interrupted first append. A marker written only after the WAL
append leaves a crash window; a marker written only before it cannot by
itself distinguish an interrupted append from later WAL loss. The existing
file backend also creates a missing WAL on construction, so missing-file
detection must happen before that creation. Do not repair snapshots outside
the writer fence: another writer could commit between replay and repair.
