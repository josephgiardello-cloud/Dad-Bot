# Tenant identity and collision decisions

> **SIGNED POLICY — 2026-11-09.**
>
> The three collision groups are to be split into distinct issued tenant IDs.
> This is a policy decision only: no group has been migrated, mapped to an
> owner, assigned membership, or granted shared-mirror access. Ownership must
> be verified before any per-tenant data or membership migration.

**Status: SIGNED POLICY; DATA MIGRATION NOT AUTHORIZED WITHOUT OWNER EVIDENCE**

**Decision owner:** Platform Lead.
**Compatibility expiry:** 2027-02-07.

Source inventory: [tenant-identity-migration-report.md](./tenant-identity-migration-report.md).
The scan recorded 220 observations in 22 lossy-normalizer groups. It found
three collision groups. No live authority database, membership rows, tenant
owner registry, or tenant-owned mirror entitlement was found in the scanned
repository paths. Therefore actual current owners and grants are **unknown**.
The source paths below are primarily test fixtures and code constants, not
proof that the values exist as deployed tenants.

## A. All 22 legacy-normalized groups

The signed choices for `default` / empty, `family-a` / `family/a`, and
`tenant-a` / `tenant/a` are **split** (Items **1–3**); literal `default` is
reserved compatibility-only, prohibited in production, and gated in
dev/local (Item **4**). See [SIGN-OFF-PACKET.md](./SIGN-OFF-PACKET.md) for
the signed consequences and exact gate name.

Each observed input below maps to the normalized form shown only under the
old lossy normalizer. These are not proposed canonical IDs. The source report
contains exact line references for each observation.

| Old normalized form | Exact observed member inputs | Observed source paths | Current ownership / mirror entitlement |
|---|---|---|---|
| `a` | `a` | `tests/unit/test_tenant_memory_ledger.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `alpha` | `alpha` | `tests/integration/test_postgres_execution_ledger.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `another-tenant` | `another-tenant` | `tests/unit/test_execution_ledger_boot_order.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `audit-tenant-a` | `audit-tenant-a` | `tests/unit/test_tenant_isolation_acceptance.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `audit-tenant-b` | `audit-tenant-b` | `tests/unit/test_tenant_isolation_acceptance.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `b` | `b` | `tests/unit/test_tenant_memory_ledger.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `config-tenant` | `config-tenant` | `tests/test_dadbot_regression.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `default` **(collision)** | missing/empty input `""`; literal `default` | `dadbot_system/contracts.py`; `tests/test_kernel_freeze_phase1.py`; `tests/test_models.py`; `tests/test_service_architecture.py`; `tests/test_service_client.py` | Not proven provisioned; auth-off creates an anonymous default principal only in the permitted local/dev mode. No mirror entitlement. |
| `family-a` **(collision)** | `family-a`; `family/a` | `tests/test_api_models_contract.py`; `tests/test_app_runtime_startup.py`; `tests/test_event_durability_security.py`; `tests/test_service_api_integration.py`; `tests/test_service_architecture.py`; `tests/test_service_client.py`; `tests/unit/test_source_grounding.py` | Unknown; tests use each value; no registered sharing entitlement. |
| `family-b` | `family-b` | `tests/test_service_api_integration.py`; `tests/test_service_architecture.py`; `tests/test_service_websocket_tickets.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `my-tenant` | `My Tenant!` | `tests/test_config_drift.py` | Not accepted as an ID by the new validator; observed only as invalid-input test data. |
| `other` | `other` | `tests/unit/test_source_grounding.py`; `tests/unit/test_tenant_memory_ledger.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `other-tenant` | `other-tenant` | `tests/unit/test_execution_ledger_boot_order.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `real-source-test` | `real-source-test` | `tests/unit/test_production_reranker.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `tenant` | `tenant` | `tests/test_service_split.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `tenant-a` **(collision)** | `tenant-a`; `tenant/a` | `tests/integration/test_postgres_execution_ledger.py`; `tests/test_config_drift.py`; `tests/test_dadbot_regression.py`; `tests/test_recovery_and_idempotency.py`; `tests/unit/test_authenticated_wal_boundary.py`; `tests/unit/test_delegated_authority_acceptance.py`; `tests/unit/test_delegated_memory_boundary.py`; `tests/unit/test_durable_submission_authority.py`; `tests/unit/test_owned_sqlite_tasks.py`; `tests/unit/test_service_identity_binding.py`; `tests/unit/test_service_verified_principal.py`; `tests/unit/test_sqlite_service_state.py`; `tests/unit/test_tenant_identity.py` | Unknown; tests exercise both canonical-looking and rejected alias forms; no registered sharing entitlement. |
| `tenant-alpha` | `tenant-alpha` | `tests/integration/test_postgres_execution_ledger.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `tenant-b` | `tenant-b` | `tests/test_dadbot_regression.py`; `tests/unit/test_durable_submission_authority.py`; `tests/unit/test_owned_sqlite_tasks.py`; `tests/unit/test_sqlite_service_state.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `tenant-beta` | `tenant-beta` | `tests/integration/test_postgres_execution_ledger.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `tenant-test` | `tenant-test` | `tests/test_contract_runtime_telemetry.py`; `tests/test_ledger_authority_invariants.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `test` | `test` | `tests/unit/test_production_reranker.py`; `tests/unit/test_source_grounding.py` | Unknown; fixture-only; no implicit mirror entitlement. |
| `test-tenant` | `test-tenant` | `tests/integration/test_postgres_execution_ledger.py`; `tests/test_service_split.py`; `tests/unit/test_execution_ledger_boot_order.py` | Unknown; fixture-only; no implicit mirror entitlement. |

### Policy options (for historical rationale)

The options below apply independently to **each of the 22 rows**, including
non-collision groups. For the three collision rows, the signed split choice
above supersedes this historical options table. No identity mapping or
membership migration follows solely from the policy choice.

| Option | Shared-mirror effect | Membership-row effect | Current test effect |
|---|---|---|---|
| **Merge inputs into one issued tenant** | Sharing is permitted only after an owner proves the records have one owner and grants the migration; never infer mirror entitlement from the old normalized string. | Create one issued tenant and explicitly migrate/merge only verified subjects and scopes; preserve audit provenance. | Tests using different inputs as distinct identities would become invalid. For the three collision groups, tests that assert mismatch/rejection or cross-tenant separation must not be weakened; fixture owners must choose distinct identities or explicitly test same-tenant aliases. |
| **Split inputs into distinct issued tenants** | No cross-tenant mirror access; migrate each owned mirror/state partition only after ownership is verified. | Create a separate issued tenant and membership mapping per verified owner. | Preserve isolation tests; update only fixtures that intentionally assert lossy equality. The alias-rejection tests should continue to reject ambiguous external forms. |
| **Keep legacy aliases** | An alias grants no mirror access by itself. Each exact alias must resolve to one issued tenant and state remains assigned only after owner verification. | Store issuer-scoped exact alias-to-issued-ID records; retain distinct membership rows for distinct issued owners. Empty input cannot be an alias and `default` needs an explicit decision. | Tests may assert exact alias resolution, but must not assert normalization/collapse. Existing fixture inputs can remain as aliases only where the test provisions that mapping. |
| **Defer with named owner** | Keep the Phase 0 fail-closed rule: no implicit shared mirror. | Do not merge or synthesize memberships; assign an accountable owner to resolve the mapping before compatibility expires. | Preserve tests, but do not claim they establish production ownership. Cases that currently depend on the old lossy normalizer remain migration blockers. |

## B. Collision group details

These are the three groups with more than one exact input.

### `default`: `""` and `default`

Observed at the source paths in the group table; exact references are in the
linked scan report. A naive normalizer collapses missing/empty input and the
literal `default`. The validator now rejects empty input but still accepts
literal `default` for compatibility. No member has a verified durable
membership or mirror entitlement in this inventory. The API auth-off path
assigns an anonymous principal to `default` in allowed local/dev mode; this is
not proof that existing data belongs to that principal.

All four options in the previous matrix apply. Test references include
`tests/test_models.py:208`, `tests/test_service_architecture.py:29,364`,
`tests/test_service_client.py:95,101`, and `tests/test_kernel_freeze_phase1.py:23`.
The empty-input behavior is covered in `tests/unit/test_tenant_identity.py`.

### `family-a`: `family-a` and `family/a`

Observed in the seven paths in the group table. A naive normalizer collapses
the slash form to `family-a`. The slash form is currently rejected by strict
request/tenant validation; the hyphen form remains accepted as a legacy ID.
No membership or shared-mirror entitlement is established by the scan.

All four options in the previous matrix apply. Relevant tests include
`tests/test_api_models_contract.py:8,27`,
`tests/test_service_api_integration.py:935-936`, and
`tests/unit/test_source_grounding.py:317`; exact observation line numbers are
in the linked scan report. The API test expecting the slash alias to be
rejected is a safety invariant and must not be changed into silent rewriting.

### `tenant-a`: `tenant-a` and `tenant/a`

Observed in the thirteen paths in the group table. A naive normalizer
collapses the slash form to `tenant-a`. The slash form is currently rejected
by strict request/tenant validation; `tenant-a` remains accepted as a legacy
ID. The inventory proves no current membership or mirror-entitlement mapping.

All four options in the previous matrix apply. Relevant tests include
`tests/test_config_drift.py:116`, `tests/unit/test_service_identity_binding.py:210-244`,
and `tests/unit/test_tenant_identity.py:129-136`; exact observation line
numbers are in the linked scan report. Tests that require rejection of
`tenant/a` or isolation between distinct owners must remain strict.

## C. `default` treatment options

Item **4** in [SIGN-OFF-PACKET.md](./SIGN-OFF-PACKET.md) is signed:
`default` is reserved for compatibility-only use, prohibited in production,
and gated in dev/local. The historical alternatives below are retained for
rationale only; none grants shared-mirror entitlement.

| Option | Token issuance and verification | Membership rows | Shared-mirror fallback | Auth-off anonymous principal | Tests relying on `default` |
|---|---|---|---|---|---|
| **1. Reserved compatibility-only tenant** — prohibit in production; permit only in dev/local behind an explicit environment flag. | Refuse new `default` claims unless the explicit local/dev flag is enabled; retain verification of already-signed tokens only through the approved compatibility window. | Keep only explicitly approved local/dev memberships; prohibit new production membership grants for `default`. | No entitlement; keep the Phase 0 fail-closed behavior even in dev/local unless a separate owner-backed grant is recorded. | May remain `default` only when auth-off is allowed and the explicit environment flag is set; production auth-off remains prohibited. | Tests must set the explicit flag when exercising this path; preserve the expected production rejection. |
| **2. Migrate to an issued ID during migration.** | Issue tokens with the provisioned opaque ID; preserve old signed claims until expiry, then reject unmapped legacy claims. | Migrate each subject/scope row only after owner verification; preserve audit provenance and create explicit memberships for the issued ID. | Migrate state only after owner-backed assignment; never infer mirror access from the old value. | Bind the dev/local anonymous principal to an explicitly provisioned local ID rather than `default`. | Replace intentional default fixtures with the issued ID; retain negative tests for empty/missing input and expired/unmapped legacy claims. |
| **3. Prohibit entirely after the compatibility window.** | Reject new issuance immediately at cutover and reject remaining signed `default` claims after their approved expiry/grace period. | Migrate verified memberships to issued IDs or revoke them; do not retain a `default` production row. | No entitlement; any existing mirror data requires separately approved owner mapping. | Disable auth-off `default` principal; local/dev must use explicit authenticated or provisioned identity. | Convert tests that use `default` as a working principal to explicit issued IDs; keep tests asserting that `default` is rejected. |

Observed test references include `tests/test_kernel_freeze_phase1.py:23`,
`tests/test_models.py:208`, `tests/test_service_architecture.py:29,364`,
`tests/test_service_client.py:95,101,190`,
`tests/test_service_websocket_tickets.py:137`,
`tests/unit/test_service_verified_principal.py:91`, and the empty-input
compatibility contract in `tests/unit/test_tenant_identity.py`. These are
test fixtures and contracts, not evidence that a deployed `default` tenant
has a verified owner.

## Current enforcement and non-decisions

- The local shared-memory mirror provides no implicit entitlement. If a
  shared legacy snapshot or backup exists without tenant ownership evidence,
  establishment fails closed; if neither exists, a fresh tenant starts from
  empty state. This rule applies equally to every row above.
- The authority schema can hold `(issuer, subject, tenant)` membership rows,
  but legacy membership grants do not prove tenant issuance or owner identity.
  No live per-member row set was available for this report.
- Do not map the test literals to customer identities. Test fixtures are not
  production provisioning records.
- No migration script in this change writes state, membership, aliases, or
  tokens. All options require an owner-backed mapping and a separate approved
  migration plan.
