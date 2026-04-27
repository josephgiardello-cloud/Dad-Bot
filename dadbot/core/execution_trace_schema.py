"""Execution trace contract schema versioning and backward-compatible migration.

Design contract
---------------
Every ``execution_trace_contract`` dict written to turn metadata receives a
``schema_version`` stamp.  When loading persisted contracts from older turns,
``ExecutionTraceContractSchemaMigrator`` upgrades the contract to the current
schema before it is used for verification or replay.

Architectural role
------------------
::

    TurnGraph._finalize_execution_trace_contract()
        └── stamp_trace_contract_version(contract)   ← stamps current version

    Replay / persistence load path:
        └── get_trace_migrator().migrate(loaded_contract)   ← upgrades legacy

Contract schema evolution
-------------------------
Schema ``"1.0"`` (current):

.. code-block:: json

    {
        "schema_version": "1.0",
        "version": "1.0",
        "event_count": 12,
        "trace_hash": "<sha256-hex>"
    }

Legacy ``"0.0"`` (contracts written before schema versioning was introduced):

.. code-block:: json

    {
        "version": "1.0",
        "event_count": 12,
        "trace_hash": "<sha256-hex>"
    }

The ``"0.0"`` → ``"1.0"`` migration just adds ``schema_version`` with no
structural change (the ``version`` field was already written in earlier
contracts and is preserved).

Registering custom migrations
-----------------------------
::

    from dadbot.core.execution_trace_schema import get_trace_migrator

    # Upgrade from "1.0" to "2.0" when the contract format changes.
    def my_migration(contract):
        contract["new_field"] = contract.get("old_field", "default")
        return contract

    get_trace_migrator().register("1.0", "2.0", my_migration)
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable


# Bump this string each time the execution trace contract envelope changes.
EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION: str = "1.0"

# Legacy marker used for contracts that pre-date schema versioning.
_LEGACY_VERSION: str = "0.0"


# ---------------------------------------------------------------------------
# Stamping helper
# ---------------------------------------------------------------------------

def stamp_trace_contract_version(contract: dict[str, Any]) -> dict[str, Any]:
    """Add ``schema_version`` to *contract* if absent.

    Mutates the dict in-place and returns it (convenience pattern).

    This function is called by ``TurnGraph._finalize_execution_trace_contract``
    so that every newly written contract carries an explicit schema stamp.
    """
    contract.setdefault("schema_version", EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION)
    return contract


# ---------------------------------------------------------------------------
# Migration engine
# ---------------------------------------------------------------------------

class ExecutionTraceContractMigration:
    """A single schema-version upgrade step for execution trace contracts."""

    def __init__(
        self,
        from_version: str,
        to_version: str,
        migrate_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.from_version = str(from_version)
        self.to_version = str(to_version)
        self._fn = migrate_fn

    def apply(self, contract: dict[str, Any]) -> dict[str, Any]:
        result = self._fn(deepcopy(contract))
        result["schema_version"] = self.to_version
        return result


class ExecutionTraceContractSchemaMigrator:
    """Registry and engine for execution trace contract schema migrations.

    Maintains a chain of ``(from_version, to_version)`` upgrade steps.
    ``migrate(contract)`` walks the chain until the contract reaches
    ``EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION``.

    Backward-compatible legacy handling
    ------------------------------------
    Contracts with no ``schema_version`` field are treated as ``"0.0"``
    (pre-versioning era) and upgraded to ``"1.0"`` by the built-in rule which
    simply adds the ``schema_version`` stamp — no other structural change is
    needed because the ``version`` field was already written correctly.
    """

    def __init__(self) -> None:
        self._migrations: list[ExecutionTraceContractMigration] = []
        # Register the built-in legacy → 1.0 migration.
        self.register(
            _LEGACY_VERSION,
            "1.0",
            lambda c: c,  # no structural change; stamp applied in apply()
        )

    def register(
        self,
        from_version: str,
        to_version: str,
        migrate_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> "ExecutionTraceContractSchemaMigrator":
        """Register a migration step.  Returns self for chaining."""
        self._migrations.append(
            ExecutionTraceContractMigration(from_version, to_version, migrate_fn)
        )
        return self

    def migrate(self, contract: dict[str, Any]) -> dict[str, Any]:
        """Return *contract* upgraded to ``EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION``.

        - Contracts already at the current version are returned as-is.
        - Contracts with no ``schema_version`` are treated as legacy ``"0.0"``.
        - Contracts at unknown intermediate versions are upgraded along any
          registered path; if no path exists they are returned as-is with a
          ``schema_version`` stamp of the latest reachable version.

        The input dict is never mutated; a deep copy is made.
        """
        version = str(contract.get("schema_version") or _LEGACY_VERSION)
        if version == EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION:
            return contract

        contract = deepcopy(contract)

        # Walk the migration chain (max 50 hops to prevent infinite loops).
        for _ in range(50):
            if version == EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION:
                break
            step = next(
                (m for m in self._migrations if m.from_version == version),
                None,
            )
            if step is None:
                # No migration path forward — stamp whatever version we reached.
                contract["schema_version"] = version
                break
            contract = step.apply(contract)
            version = str(contract.get("schema_version") or version)

        return contract

    def migrate_all(
        self, contracts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Migrate a list of contracts in-order."""
        return [self.migrate(c) for c in contracts]

    def needs_migration(self, contract: dict[str, Any]) -> bool:
        """Return True if *contract* is not at the current schema version."""
        return (
            str(contract.get("schema_version") or _LEGACY_VERSION)
            != EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION
        )

    @property
    def migration_count(self) -> int:
        return len(self._migrations)

    def registered_paths(self) -> list[tuple[str, str]]:
        """Return all registered migration paths as ``(from, to)`` tuples."""
        return [(m.from_version, m.to_version) for m in self._migrations]


# ---------------------------------------------------------------------------
# Global default migrator (module-level singleton)
# ---------------------------------------------------------------------------

_DEFAULT_TRACE_MIGRATOR = ExecutionTraceContractSchemaMigrator()


def get_trace_migrator() -> ExecutionTraceContractSchemaMigrator:
    """Return the global execution trace contract schema migrator."""
    return _DEFAULT_TRACE_MIGRATOR


def migrate_trace_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper — migrate *contract* using the global migrator."""
    return _DEFAULT_TRACE_MIGRATOR.migrate(contract)
