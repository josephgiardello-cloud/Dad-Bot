# Branch and commit plan

> **EXECUTED — LOCAL ONLY.**
> Branch `tenant-identity-remediation` was created at `fe314752`.
> No push or merge was performed.

## Risk and recommendation

Before classification, 109 status entries (47 untracked) were present at
the reported base `fe314752`. Several paths mixed tenant changes with
pre-existing durable-execution, ledger, snapshot, and Phase 4A work. Those
ambiguous paths were assigned to the carried-forward WIP set instead of
being forced into a remediation commit.

Recommended: first preserve and review the current worktree, then make a
dedicated branch and split commits by ownership. Do not use blanket `git add
-A`; inspect every staged diff and preserve unrelated edits.

## Proposed ship-set split

1. **Phase 0 implementation:** shared-mirror fail-closed behavior, tenant-
   scoped pulse lookup, production auth-off refusal.
2. **Phase 1 implementation:** canonical validator, opaque minting,
   provisioning and legacy acceptance logging.
3. **Decision documents:** tenant semantics, collision/default decisions,
   compatibility window, and this sign-off packet. Keep all marked proposed
   until signed.
4. **Tests:** split acceptance and regression tests by the implementation
   they validate; retain strict tenant isolation and pre-existing unrelated
   test fixes only after owner review.
5. **Tooling:** Import Linter gate, F821 repairs, contract-anchor/compiler
   fixes, and TenantContext interface.
6. **Audit updates:** audit traceability, D1-D3, Phase 3 entry criteria, and
   conditional assumptions.

Do not combine a sign-off decision with an automatic collision/data
migration. Do not include Phase 2 middleware or Phase 3 store scoping until
their gates are approved and separately reviewed.

## Execution record

All seven local commits were created on `tenant-identity-remediation`. No
push or merge was performed.

| Set | Commit | Files |
|---|---|---|
| Phase 0 implementation | `b06a8fd4b087d414351ffd1332d72eb2986fedeb` | `dadbot/core/local_tenant_establishment.py` |
| Phase 1 implementation | `de1dc3f239186cbae798f596f30a65e6d32f3c1d` | `dadbot/config.py`, `dadbot_system/contracts.py`, `dadbot_system/delegation.py`, `dadbot_system/execution_identity.py`, `dadbot_system/security.py`, `dadbot_system/tenant_identity.py`, `tools/tenant_identity_migration.py` |
| Decision documents | `d48cbc6c2934a8ff724dd81b1b2de8d70e05a0c3` | `docs/decisions/SIGN-OFF-PACKET.md`, `docs/decisions/branch-commit-plan.md`, `docs/decisions/phase3-entry-criteria.md`, `docs/decisions/tenant-identity-collisions.md`, `docs/decisions/tenant-identity-compatibility.md`, `docs/decisions/tenant-identity-migration-report.md`, `docs/decisions/tenant-identity.md` |
| Remediation tests | `8f0491beb775db30ad7d7bac35bb9fa136959d47` | `tests/unit/test_delegated_authority_acceptance.py`, `tests/unit/test_service_identity_binding.py`, `tests/unit/test_service_verified_principal.py`, `tests/unit/test_tenant_context.py`, `tests/unit/test_tenant_identity.py`, `tests/unit/test_tenant_isolation_acceptance.py` |
| Tooling and kernel boundary | `870ad555f2dc226b504dae682c3cde3ab9462ef0` | `dadbot/managers/runtime_client.py`, `dadbot/runtime/launcher.py`, `dadbot/runtime/mcp/tool_client.py`, `dadbot/telemetry/__init__.py`, `dadbot/telemetry/otel_bridge.py`, `dadbot/telemetry/trace_export.py`, `dadbot_system/runtime_signals.py`, `dadbot_system/telemetry.py`, `dadbot_system/tenant_context.py`, `pyproject.toml`, `tests/test_observability_bridges.py`, `tests/unit/test_contract_test_compiler.py`, `tools/contract_test_compiler.py` |
| Audit updates | `b46a4040676585db1fb6868bd9f5ac6d18fcfcb6` | `TENANT_EXECUTION_PATH_AUDIT.md` |
| Carried-forward pre-existing WIP (not tenant remediation) | `ef28cc66c6ca109cc138f03d50ba58585182afc5` | `DURABLE_MEMORY_IMPLEMENTATION_STATUS.md`, `ENGINEERING_INVARIANTS_AUDIT.md`, `README.md`, `TENANT_AUTHORITY_ARCHITECTURE_SPEC.md`, `dadbot/app_runtime.py`, `dadbot/base/_memory_lifecycle_mixin.py`, `dadbot/base/_memory_search_mixin.py`, `dadbot/core/boot_mixin.py`, `dadbot/core/compaction.py`, `dadbot/core/durability.py`, `dadbot/core/execution_ledger.py`, `dadbot/core/fault_injection.py`, `dadbot/core/ledger_backend.py`, `dadbot/core/orchestrator.py`, `dadbot/core/postgres_ledger_backend.py`, `dadbot/core/snapshot_engine.py`, `dadbot/core/tenant_memory_ledger.py`, `dadbot/core/tool_executor.py`, `dadbot/core/tool_registry.py`, `dadbot/core/turn_mixin.py`, `dadbot/managers/memory_coordination.py`, `dadbot/memory/graph_manager.py`, `dadbot/memory/graph_views.py`, `dadbot/memory/manager.py`, `dadbot/memory/normalizers.py`, `dadbot/memory/semantic_manager.py`, `dadbot/memory/storage.py`, `dadbot/services/persistence.py`, `dadbot/state.py`, `dadbot_system/__init__.py`, `dadbot_system/api.py`, `dadbot_system/client.py`, `dadbot_system/orchestration.py`, `dadbot_system/owned_sqlite_tasks.py`, `dadbot_system/semantic_index.py`, `dadbot_system/state.py`, `dadbot_system/worker.py`, `docs/system_contract_map.md`, `tests/conftest.py`, `tests/integration/test_postgres_execution_ledger.py`, `tests/persistence_test.py`, `tests/test_config_drift.py`, `tests/test_determinism_capability_receipts.py`, `tests/test_durable_execution.py`, `tests/test_memory_manager.py`, `tests/test_phase4a_adversarial_collapse.py`, `tests/test_phase4a_fixed.py`, `tests/test_production_hardening_wave3.py`, `tests/test_production_hardening_wave4.py`, `tests/test_service_api_integration.py`, `tests/test_service_architecture.py`, `tests/test_service_websocket_tickets.py`, `tests/unit/test_authenticated_wal_boundary.py`, `tests/unit/test_control_plane_ledger_bounds.py`, `tests/unit/test_delegated_memory_boundary.py`, `tests/unit/test_durable_submission_authority.py`, `tests/unit/test_execution_ledger_boot_order.py`, `tests/unit/test_lifecycle_checkpoint_failures.py`, `tests/unit/test_local_authority_mutation_application.py`, `tests/unit/test_local_tenant_establishment_application.py`, `tests/unit/test_memory_authority_application.py`, `tests/unit/test_memory_snapshot_write_guard.py`, `tests/unit/test_owned_sqlite_tasks.py`, `tests/unit/test_semantic_index_repair_failures.py`, `tests/unit/test_snapshot_tail_recovery.py`, `tests/unit/test_sqlite_service_state.py`, `tests/unit/test_tenant_memory_ledger.py`, `tests/unit/test_tenant_memory_restore_authority.py`, `tests/unit/test_tool_registry.py`, `tests/unit/test_wal_recovery_integrity.py`, `uv.lock` |

`.gitignore` was deliberately not included in the original seven commits.
B1–B4 and the initial Phase 3 inventory were recorded in the later
`6616ed01` follow-up commit. Subsequent Stage 1 changes are separate from
both sets. Do not amend the original seven commits or push/merge.

### Subsequent local follow-up commits

The original seven-set split was followed by two local commits. They were not
replayed or amended:

| Commit | Set | Files |
|---|---|---|
| `5273599567e97cd99ec63748d9978f0522c6f6fd` | Resource-bounded suite and B2 trimming deferral | `TENANT_EXECUTION_PATH_AUDIT.md`, `docs/decisions/checkpoint-trimming-deferral.md`, `tests/test_scheduler_lifecycle_hardening.py` |
| `6616ed0179d4f64e1e77b8634894d7eb56b3f0a8` | Carried-forward durability/config work, explicitly separate from tenant wiring | `.env.production.example`, `.github/workflows/phase4-capability-validation.yml`, `.gitignore`, `ci/production_env_check.py`, `dadbot/core/boot_mixin.py`, `dadbot/core/canonical_event.py`, `dadbot/core/otel_bridge.py`, `dadbot/core/postgres_ledger_backend.py`, `dadbot/core/trace_export.py`, `dadbot/memory/storage.py`, `dadbot_system/state.py`, `docker-compose.yml`, `docs/decisions/branch-commit-plan.md`, `docs/decisions/phase3-scope-inventory.md`, `docs/decisions/postgres-authenticated-boot.md`, `tests/unit/test_authenticated_postgres_boot.py`, `tests/unit/test_projection_status_durability.py` |

Commit `6616ed01` has an explicit provenance body identifying its changes as
carried-forward WIP and excluding them from Phase 2 tenant-context wiring and
Phase 3 tenant scoping. No additional carried-forward source set is present
in the current Stage 1 diff; generated `.phase32-test-data/` is ignored, not
committed. The current Stage 1 implementation and documentation changes are
reviewed and committed separately.

## Reproducible command record

The branch-creation command was:

```powershell
git switch -c tenant-identity-remediation fe314752
```

For each split set, the path list is the exact **Files** entry in the table
above. The reviewed staging/commit sequence was:

```powershell
git add -- <only paths in the selected set>
git diff --cached --check
git diff --cached --stat
git commit -m "<the subject recorded for that set>"
```

Recorded subjects, in order, were:
`fix: fail closed on unowned local tenant establishment`;
`feat: add canonical tenant identity provisioning`;
`docs: add tenant identity sign-off packet`;
`test: cover tenant identity and isolation boundaries`;
`chore: enforce service kernel boundaries`;
`docs: update tenant execution audit`; and
`chore: carry forward existing repository WIP`.
These commands document the completed local operation; do not replay them
against the current worktree.

## Ruff all-rule baseline proposal — decision requested

Do not run sweeping style fixes and do not change Ruff configuration in this
change. Choose one strategy for the remaining style/complexity debt:

| Option | Advantages | Risks / required control |
|---|---|---|
| Per-rule quarantine with owner and expiry | Makes debt and accountability explicit while new regressions remain gated. | Requires a named role, narrow rule/path scope, and an expiry/review date; avoid broad permanent ignores. |
| Incremental baseline | Lets the repository reduce existing debt without a disruptive all-at-once cleanup. | Baseline tooling must distinguish old findings from newly introduced ones and be deterministic in CI. |
| Fix now for a named subset | Delivers immediate value for a bounded, selected rule family or directory. | Scope must be approved and tested; do not opportunistically rewrite unrelated files. |

**Requested selection:** `[ ] quarantine  [ ] incremental baseline  [ ] named fix-now subset`
**Owner role / expiry (if applicable):** `____________________`

No Ruff baseline, rule quarantine, or sweeping style fix was created by this
proposal. Branch/commit execution remains local; push and merge are not
authorized.
