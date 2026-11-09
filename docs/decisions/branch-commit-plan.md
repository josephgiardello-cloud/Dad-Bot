# Branch and commit plan

> **AUTHORIZED AND IN PROGRESS — LOCAL ONLY.**
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

The branch was created and the following two reviewed sets were committed:

| Set | Commit | Files |
|---|---|---|
| Phase 0 implementation | `b06a8fd4` | `dadbot/core/local_tenant_establishment.py` |
| Phase 1 implementation | `de1dc3f2` | `dadbot/config.py`, `dadbot_system/contracts.py`, `dadbot_system/security.py`, `dadbot_system/delegation.py`, `dadbot_system/execution_identity.py`, `dadbot_system/tenant_identity.py`, `tools/tenant_identity_migration.py` |

Remaining sets are decision documents, remediation tests, tooling,
audit updates, and the separately labeled carried-forward WIP. Stage only
the paths assigned to each set, inspect the staged diff, and confirm status
after every commit. Do not push, merge, or include `.gitignore` unless a
specific unignored artifact requires it.

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
