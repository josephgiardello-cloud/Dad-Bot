# Archive Policy

## Purpose

archive/ contains historical snapshots, old audits, and reference material.
It is not a runtime dependency boundary.

## Rules

1. Runtime code under dadbot/ must not import from archive/.
2. Product documentation should prefer docs/ over archive/.
3. New design decisions must be recorded in docs/, not archive/.
4. archive/ content can be deleted or moved without runtime behavior changes.

## Operational Guidance

- Keep archive additions rare and intentional.
- If archive size or noise grows, move older content to external storage and keep only minimal in-repo references.
- Treat archive as read-only reference, never as executable authority.
