# Tenant identity compatibility window

> **PROPOSED — REQUIRES SIGN-OFF**

**Decision owner:** TBD — a named tenant-authority owner must be assigned.
**Expiry:** TBD — requires sign-off.

This is a proposal for a bounded compatibility period. It does not extend the
window automatically or authorize provenance enforcement changes.

## Classes included

The inventory at
[tenant-identity-migration-report.md](./tenant-identity-migration-report.md)
observed 220 tenant-bearing values in 22 legacy-normalized groups, including
three collision groups. See the options and member IDs in
[tenant-identity-collisions.md](./tenant-identity-collisions.md).

| Class | Current compatibility behavior | Exit criteria for retiring this class |
|---|---|---|
| Versioned opaque IDs (`t1_` plus exact canonical payload) | Accepted by the strict parser and can be minted at the provisioning seam. | Keep as the canonical production class; require provisioning/issuer membership checks at the future tenant-context boundary and prove every active ID has a provisioning record. |
| Exact legacy canonical IDs (lowercase ASCII with legacy separators) | Accepted by `parse_tenant_id`/`require_tenant_id` without rewriting; provenance is not universally checked. | Map every configured and persisted value to an approved issued ID or explicitly retained legacy identity; prove no unexpired signed token or supported client still depends on the old claim; update fixtures and run the cross-tenant acceptance matrix. |
| Literal `default` | Accepted as a legacy value; empty/missing input is rejected by the parser, but several APIs/config models still supply a default value explicitly. | Human sign-off selects a treatment; migrate all explicitly owned records; remove implicit/default construction or constrain it according to the signed decision; test auth-off and production behavior for the selected policy. |
| Previously signed v1 token claims | Signature verification preserves the signed tenant string, including legacy/noncanonical values, until token expiry. | Wait through the last outstanding token's signed expiration after reissue; demonstrate no required client relies on the old claim; reject any remaining unmapped legacy claim at authenticated execution ingress. |
| Exact registered alias references | `resolve_tenant_reference` can resolve an exact issuer-scoped alias to a canonical ID. | Enumerate aliases, ensure every target is provisioned and owner-approved, prove collision-free issuer-scoped resolution, then retire only aliases with no active clients/state references. |

## Interim acceptance logging

Until cutover, every accepted non-versioned legacy tenant identifier is to be
logged with the exact tenant string, issuer, subject, UTC/epoch timestamp, and
acceptance path. The identity validator now emits a warning for accepted
legacy IDs and records those fields; token issuance/verification and
membership provisioning pass the known issuer and subject. Call sites with
no authenticated actor explicitly report `unbound` rather than inventing
provenance. Do not treat an `unbound` record as a resolved owner.

This logging is implemented and tested, but not every old call site has a
bound actor; `unbound` events must be reconciled before cutover. Logs are
operational evidence, not authorization or proof of ownership. They must be
retained under the normal privacy/retention policy and must not include
bearer tokens, secrets, or request payloads.

## Risks while open

- Lossy IDs could already have caused aliases in state, tokens, or fixtures;
  no automatic merge or mirror entitlement is safe.
- Signed legacy tokens can still carry noncanonical tenant strings until
  expiry; signature verification preserves compatibility but does not confirm
  current membership.
- `default` remains accepted in several explicit defaulting paths and is used
  by the auth-off anonymous principal in allowed local/dev execution.
- Legacy membership grants do not prove an ID was provisioned or that all
  subjects have the same owner.
- The audit's references to deficits **9, 18, and 19** are currently
  unnumbered/unmapped in the checked-in audit text. Their exact labels and
  scope are **unanswerable from the repository**; a human should map them
  before using them as compatibility exit gates. The risks above are directly
  evidenced and must not be relabeled as those deficits without that mapping.

## Approval record

| Field | Value |
|---|---|
| Owner | TBD — requires named owner |
| Expiry/cutover date | TBD — requires sign-off |
| Approver(s) | TBD |
| Decision status | PROPOSED — REQUIRES SIGN-OFF |
