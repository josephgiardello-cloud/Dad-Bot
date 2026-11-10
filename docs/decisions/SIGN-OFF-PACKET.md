# Tenant identity and migration sign-off

> **SIGNED — 2026-11-09.** The decisions below authorize design and
> implementation work within the stated scope; they do not themselves migrate
> tenant data, memberships, tokens, aliases, or mirrors.

**Decision record:** approved choices and parameters were supplied by the
repository owner in the 2026-11-09 task instruction.

Evidence: the dry-run inventory found 220 observations in 22 legacy-normalized
groups, with three collisions and no proof of deployed ownership. See
[tenant-identity-migration-report.md](./tenant-identity-migration-report.md)
and [tenant-identity-collisions.md](./tenant-identity-collisions.md).
In every item below, a legacy spelling or alias alone grants **no shared
mirror entitlement**. The Phase 0 mirror fix remains conditional on Item 6.

## 1. Collision: empty input / literal `default`

**Question:** How should the historical collapse of missing/empty input and
literal `default` be resolved?

| Option | Mirror / membership / token consequence | Tests and Phase 0 impact |
|---|---|---|
| Merge into one issued tenant | Requires proof that all affected state has one owner; explicitly migrate memberships and issue the new ID. No inferred mirror access. | Migrate tests to the issued ID; auth-off dev tests may use it. If the owner model authorizes shared mirrors, revisit Phase 0.1 only under an explicit grant. |
| Split into distinct issued tenants | Map empty/missing to rejection and literal `default` data to an issued tenant only after ownership verification; create explicit membership rows and reissue tokens. | Keep empty-input rejection; migrate only `default` fixtures with an approved owner mapping. Auth-off must use a separately provisioned local identity. Phase 0 remains unchanged. |
| Keep with owner | Retain literal `default` only for a named, provisioned owner; empty input remains invalid. Record issuer/subject memberships; allow tokens only under the signed compatibility policy. | Update auth-off dev tests to require the approved owner/flag. No mirror entitlement follows from the name; Phase 0 remains unchanged absent a sharing grant. |
| Defer with owner | Freeze migrations; assign a role to reconcile state, membership, and tokens. No new mapping or token policy is inferred. | Keep existing default-related tests as compatibility tests, and preserve production auth-off rejection. Phase 0 remains conditional and fail-closed. |

**Pre-sign-off recommendation (superseded by the signed choice):** defer with an accountable tenant-authority
owner until ownership can be established; this avoids an irreversible merge
or an unsupported assignment to the anonymous principal.

**Signed choice:** `[ ] merge  [x] split  [ ] keep-with-owner  [ ] defer-with-owner`
**Owner role:** `Platform Lead`. Empty input remains rejected; any literal
`default` state is assigned only after its owner is verified and receives a
distinct issued ID. No membership or mirror entitlement is inferred.

## 2. Collision: `family-a` / `family/a`

**Question:** Should the old normalized equality become a merge, a split, an
exact alias, or a deferred owner decision?

| Option | Mirror / membership / token consequence | Tests and Phase 0 impact |
|---|---|---|
| Merge into one issued tenant | Only after proving one owner; explicitly consolidate membership rows and reissue claims. The slash spelling is not itself a mirror grant. | Tests intentionally treating the values as distinct must change; alias-rejection tests remain strict unless an exact issuer-scoped alias is provisioned. Phase 0 changes only if an explicit shared-mirror grant is approved. |
| Split into distinct issued tenants | Issue one ID per verified owner, keep separate membership rows, and reissue tokens. No shared mirror. | Preserve cross-tenant mismatch/isolation assertions; migrate fixtures that used the slash form as an external ID. Phase 0 remains unchanged. |
| Keep with owner | Keep `family-a` only as an explicitly owned legacy identity; permit `family/a` solely as an exact registered issuer-scoped alias to one issued ID. Membership and token claims use the resolved ID. | Tests must provision the exact alias and must not expect normalization. No mirror entitlement; Phase 0 remains unchanged. |
| Defer with owner | No merge, alias, or membership migration until an owner resolves the collision. | Preserve rejection and separation tests. No mirror entitlement; Phase 0 remains conditional and fail-closed. |

**Pre-sign-off recommendation (superseded by the signed choice):** defer with owner because the scan found
fixtures, not deployment ownership evidence.

**Signed choice:** `[ ] merge  [x] split  [ ] keep-with-owner  [ ] defer-with-owner`
**Owner role:** `Platform Lead`. The two spellings remain distinct; ownership
verification and separate issued IDs/memberships precede any migration.

## 3. Collision: `tenant-a` / `tenant/a`

**Question:** Should the old normalized equality become a merge, a split, an
exact alias, or a deferred owner decision?

| Option | Mirror / membership / token consequence | Tests and Phase 0 impact |
|---|---|---|
| Merge into one issued tenant | Only after proving one owner; explicitly consolidate memberships and reissue claims. No automatic mirror access. | Distinct-owner and mismatch tests cannot be weakened; update only tests whose fixtures are approved as one owner. Revisit Phase 0 only for a separate explicit mirror grant. |
| Split into distinct issued tenants | Issue separate IDs, memberships, and tokens for verified owners; preserve state partitions. No shared mirror. | Keep tenant-isolation assertions; replace rejected slash-form request inputs with provisioning/alias tests. Phase 0 remains unchanged. |
| Keep with owner | Retain `tenant-a` for an explicitly named owner; `tenant/a` can be an exact issuer-scoped alias only if provisioned. Tokens/membership use the resolved ID. | Alias tests must provision the mapping and isolation tests must retain distinct IDs. No implicit mirror entitlement; Phase 0 remains unchanged. |
| Defer with owner | Freeze mapping pending owner verification; do not merge records or memberships. | Keep strict alias rejection and isolation tests. Phase 0 remains conditional and fail-closed. |

**Pre-sign-off recommendation (superseded by the signed choice):** defer with owner until state and membership
ownership can be verified.

**Signed choice:** `[ ] merge  [x] split  [ ] keep-with-owner  [ ] defer-with-owner`
**Owner role:** `Platform Lead`. The two spellings remain distinct; ownership
verification and separate issued IDs/memberships precede any migration.

## 4. Treatment of literal `default`

**Question:** What policy applies to literal `default` after the compatibility
period?

| Option | Mirror / membership / token consequence | Auth-off principal, tests, and Phase 0 impact |
|---|---|---|
| Reserved compatibility-only; prohibited in production and explicitly env-gated in dev/local | Permit only approved local memberships; block production membership grants and new production `default` tokens. Existing signed tokens survive only to approved expiry. No mirror entitlement. | Anonymous auth-off may use it only in allowed dev/local with the explicit flag; update dev tests to set the flag and keep production rejection. Phase 0 remains fail-closed. |
| Migrate to an issued ID | Migrate each verified membership/state owner; issue new opaque-ID tokens and accept old signed tokens only through expiry. No implicit mirror entitlement. | Rebind the dev/local anonymous principal to an explicitly provisioned ID; update tests at `test_models.py`, `test_service_architecture.py`, `test_service_client.py`, `test_service_websocket_tickets.py`, and `test_service_verified_principal.py`. Phase 0 remains unchanged. |
| Prohibit after the window | Reject new claims and, after expiry/grace, remaining signed `default` claims; migrate or revoke memberships. No mirror entitlement. | Disable auth-off `default`; dev/local must use an explicit identity. Convert tests to issued IDs while retaining rejection tests. Phase 0 remains unchanged. |

**Pre-sign-off recommendation (confirmed by the signed choice):** reserved compatibility-only, explicitly
gated outside production, until ownership and local-mode migration are
approved. This minimizes accidental claim changes while keeping production
closed.

**Signed choice:** `[x] reserved  [ ] migrate  [ ] prohibit`
**Production prohibition confirmed:** `[x] yes`
**Dev/local flag name:** `DADBOT_ALLOW_LEGACY_DEFAULT_TENANT`
`default` is not accepted by the opaque issued-ID TenantContext. Any permitted
dev/local compatibility principal uses a separate, explicitly gated path.

## 5. Compatibility-window owner and expiry

**Question:** Which accountable role owns the compatibility inventory and
what is the cutover/expiry date?

| Option | Mirror / membership / token consequence | Tests and Phase 0 impact |
|---|---|---|
| Name a Tenant Authority Owner role and set a date | Owner maintains evidence and approves membership/token cutover; no mirror entitlement is implied. | Add expiry/cutover tests for each retained identity class. Phase 0 is unchanged absent an explicit sharing grant. |
| Set owner or expiry to `TBD-REQUIRES-SIGN-OFF` | No identity class retires and no membership/token migration is authorized; mirror remains unshared. | Keep compatibility tests and explicitly block Phase 2 identity wiring until resolved; Phase 0 remains conditional and fail-closed. |

**Pre-sign-off recommendation (confirmed by the signed choice):** Tenant Authority Owner role, with a
specific expiry date entered before any provenance enforcement or ID
retirement. A role is accountable without embedding a personal identity.

**Owner role:** `Platform Lead`
**Expiry date (YYYY-MM-DD):** `2027-02-07`
**Signed choice:** `[x] approved  [ ] defer`

## 6. Conditional Phase 0 assumption

**Question:** Acknowledge or reject this explicit assumption:

> Phase 0 mirror fix assumes the shared mirror is a legacy single-owner
> artifact; if the tenant semantics decision selects organization-shared
> mirrors, Phase 0.1 must be revisited.

Acknowledgement does not grant mirror access or restore first-run fallback.
Any valid sharing model still requires explicit ownership evidence and a
separately reviewed grant.

| Choice | Membership / token / tests consequence | Mirror and Phase 0 consequence |
|---|---|---|
| Acknowledge | No membership or token behavior changes; keep the strict isolation acceptance test. | Keep the current Phase 0 behavior; revisit only if a later signed tenant model requires explicitly authorized organization-level sharing. |
| Reject | Record the replacement ownership model and update affected membership/token decisions and tests before implementation. | Phase 0.1 must be reviewed before any sharing behavior is designed; no fallback is restored by rejection alone. |
| Defer | No changes; Phase 2 remains blocked. | Keep the current fail-closed fix while the assumption remains unconfirmed. |

**Pre-sign-off recommendation (confirmed by the signed choice):** acknowledge provisionally; the statement
does not grant access and preserves the security boundary while ownership is
unknown.

**Signed choice:** `[x] acknowledge  [ ] reject  [ ] defer`

## Implementation boundaries

Items 1–6 are signed. Collision splitting is a policy choice, not a data
migration: no owner mapping, membership, token, alias, or mirror is created
without independent ownership evidence. `default` remains reserved for
compatibility-only use outside production and only behind
`DADBOT_ALLOW_LEGACY_DEFAULT_TENANT` in dev/local. Authenticated PostgreSQL
boot remains excluded by
[postgres-authenticated-boot.md](./postgres-authenticated-boot.md); RLS work
may be developed and directly tested with a local `NOBYPASSRLS` role without
reopening that production decision.
