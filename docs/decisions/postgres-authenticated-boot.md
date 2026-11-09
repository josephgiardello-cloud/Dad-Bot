# PostgreSQL execution ledger and authenticated production

**Decision: PostgreSQL execution-ledger boot is out of scope for authenticated
production.**

Authenticated execution requires the durable file WAL. Selecting
`DADBOT_EXECUTION_LEDGER_BACKEND=postgres` in authenticated mode fails during
boot with an explicit error. The PostgreSQL state-store and execution-ledger
primitives remain experimental for trusted-local use; they are not a supported
authenticated multi-tenant persistence configuration.

The current PostgreSQL state tables use global session/task keys, and the
PostgreSQL ledger stream defaults to a shared `default` stream. Those primitives
do not establish tenant ownership or isolation for authenticated execution.
The authenticated authority database is SQLite and does not make the separate
PostgreSQL state store tenant-scoped.

Reconsider this scope only after a separately reviewed design defines
tenant-qualified keys/constraints and recovery semantics, implements the
corresponding store-scoped access path, and passes same-key cross-tenant
isolation and restart-recovery acceptance tests. This decision does not
authorize tenant middleware, store filters, RLS, or a `default` policy.
