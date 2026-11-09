# Tenant identity decision

**Status: PROPOSED — REQUIRES SIGN-OFF**
**Decision owner:** TBD — a named tenant-authority owner must be assigned.
**Decision expiry:** TBD — requires sign-off.

This record separates the unresolved meaning of “tenant” from the Phase 1
identifier contract. The code must not infer that a tenant is a person,
organization, deployment, or runtime bot from an ID or alias.

## Repository evidence

- The authority database represents membership as `(issuer, subject, tenant)`,
  with scopes and active/version fields; this permits a principal to have a
  membership in a tenant but does not define what the tenant represents
  ([`delegation.py`](../../dadbot_system/delegation.py), `memberships` schema
  and `grant_membership`).
- Service tokens carry a tenant claim; token issuance validates the value but
  does not itself establish membership
  ([`security.py`](../../dadbot_system/security.py), `ServiceTokenManager.issue`).
- The service client can select a tenant from configuration and can issue its
  own service token, while local runtime configuration has historically used
  `default` ([`client.py`](../../dadbot_system/client.py),
  [`contracts.py`](../../dadbot_system/contracts.py)).
- The identity architecture specification calls for an opaque, stable,
  versioned, case-sensitive ID resolved at a trusted provisioning boundary;
  it does not define tenant ownership semantics
  ([`TENANT_AUTHORITY_ARCHITECTURE_SPEC.md`](../../TENANT_AUTHORITY_ARCHITECTURE_SPEC.md),
  “Proposed canonical identity and authorization”).
- The execution-path audit records a shared local memory mirror and singleton
  runtime-bot paths, but does not establish whether either is organization- or
  deployment-owned
  ([`TENANT_EXECUTION_PATH_AUDIT.md`](../../TENANT_EXECUTION_PATH_AUDIT.md)).
- The README describes tenant-scoped service state and memory boundaries, but
  does not define a tenant as a user, organization, or deployment
  ([`README.md`](../../README.md)).
- The durable-memory status explicitly treats cross-tenant recovery and
  projections as incomplete; it cannot establish a semantic ownership model
  ([`DURABLE_MEMORY_IMPLEMENTATION_STATUS.md`](../../DURABLE_MEMORY_IMPLEMENTATION_STATUS.md)).

The identity invariant in the specification is: “Issue an opaque, stable
tenant identifier at a trusted provisioning boundary. Use an exact,
case-sensitive, versioned canonical encoding; reject invalid or ambiguous
external forms instead of rewriting characters or defaulting malformed input”
([`TENANT_AUTHORITY_ARCHITECTURE_SPEC.md:83-86`](../../TENANT_AUTHORITY_ARCHITECTURE_SPEC.md#L83)).

## Candidate semantics

| Candidate | Provisioning and membership implications | Shared-mirror implication | Evidence for / against |
|---|---|---|---|
| **User** | Provision one stable tenant per issuer/subject; membership is effectively one principal per tenant, with explicit account-recovery and identity-linking rules. | A mirror may be owned by one user, but must never be treated as shared with another user without an explicit sharing grant. | The schema has issuer and subject fields, but does not enforce one tenant per subject or say that tenant equals subject. |
| **Organization** | Provision one tenant per customer/workspace; map multiple issuer/subject pairs to the organization and define organization-admin/member lifecycle. | A local mirror is not shared across organization tenants. Sharing inside one organization still needs an explicit ownership and authorization rule. | The membership table supports the shape of multiple principals per tenant. No organization registry, organization claim, or membership lifecycle policy was found. |
| **Deployment** | Provision one tenant per installation/environment; bind issuers and subjects to that installation and define whether isolated namespaces inside it are tenants or resources. | A mirror may belong to one installation only; separate tenants inside that deployment cannot inherit it absent explicit entitlement. | Local file-backed storage and the historical `default` value fit a single-installation history. The service API, tenant membership and tenant-scoped state suggest a broader model may be intended. |
| **Other (for example workspace, household, or runtime bot)** | Define the owning entity and its lifecycle first; then provision its stable ID and grant subjects access to it. | Ownership follows the defined entity; no implicit mirror sharing. | The code contains family/workspace-like examples and runtime-bot integrations, but examples and service objects do not establish the authoritative model. |

## Recommendation

**Recommend organization/workspace as the tenant for hosted multi-user service
deployments, with an explicitly provisioned single-owner tenant for local
single-user installations.** Keep deployment/environment as a separate
isolation boundary rather than equating it with a tenant. This best fits the
existing issuer/subject/tenant membership relation while retaining a defined
local-use path. This is a recommendation only; it is not adopted by this
change.

Regardless of sign-off, no tenant may inherit a shared memory mirror merely
because it is first, has revision zero, or lacks an establishment marker. The
Phase 0 fix assumes the mirror is a legacy single-owner artifact and distinct
tenants have no shared-mirror entitlement. If sign-off defines shared
ownership, that must be modeled as an explicit grant and re-reviewed; it must
not restore implicit fallback.

## Impacts requiring sign-off

- **Shared-mirror fallback:** retain the Phase 0 fail-closed behavior. Define
  any allowed import as an explicit owner/entitlement migration, not
  tenant-equivalence inferred from a path.
- **Token issuance:** issue a versioned opaque tenant ID only after trusted
  provisioning; authorize the issuer/subject/tenant membership independently.
  Keep verification of already-signed v1 tokens until each token's signed
  expiration during the compatibility window. No fixed cutoff or forced
  remapping is authorized here; signature validity alone does not prove that a
  legacy tenant claim has been mapped.
- **Membership table:** decide whether multiple subjects may join a tenant,
  who may grant/revoke memberships, how issuer changes and organization
  mergers work, and whether a deployment can contain multiple tenants.
- **Automation and spatial runtime-bot resolution:** bind each runtime bot to
  one explicit tenant or an explicitly authorized platform-level operation.
  A singleton bot is not an implicit tenant resolver.
- **`default`:** do not use it as a missing/invalid-input fallback. Inventory
  and explicitly map the existing literal as a legacy identity; it may be
  retained only with an owner and provisioning record. No decision in this
  record silently declares it to be a user, organization, or deployment.

## Open questions for human sign-off

1. Is the hosted tenant an organization/workspace, a user, or another named
   owner entity? Is the local single-user installation an allowed separate
   mode?
2. May one issuer/subject belong to multiple tenants, and who can grant,
   revoke, or migrate those memberships?
3. Which trusted system provisions IDs, persists aliases, and authorizes
   token issuance? How are issuer rotation, deleted owners, and collisions
   handled?
4. Does any existing `default` data have an identifiable owner? Which operator
   will approve its mapping, and is it to be retained or migrated?
5. Is sharing ever valid across tenants? If so, which resource types and what
   explicit grant authorizes it?
6. Which runtime-bot and automation operations are tenant-local versus
   platform-wide, and how is the latter authority represented?
