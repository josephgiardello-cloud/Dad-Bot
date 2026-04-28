"""Memory store schema migration system.

Provides a versioned migration pipeline that runs on every load_memory_store()
call, BEFORE normalization.  Migrations are one-way, additive, and idempotent.

Usage pattern in MemoryStorageBackend.load_memory_store():

    loaded = json_load(...)
    migrated = MemoryMigrationRegistry.migrate(loaded)
    normalized = manager.normalize_memory_store(migrated)

Adding a new migration:

    @MemoryMigrationRegistry.register(from_version=3)
    def migrate_v3_to_v4(store: dict) -> dict:
        store["new_field"] = store.pop("old_field", default_value)
        return store

The registry auto-increments to_version = from_version + 1.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# The current schema version. Bump this when adding a new migration.
CURRENT_SCHEMA_VERSION: int = 1

MigrationFn = Callable[[dict], dict]

# version -> migration function that upgrades from that version to version+1
_MIGRATIONS: dict[int, MigrationFn] = {}


def _register(from_version: int, fn: MigrationFn) -> MigrationFn:
    if from_version in _MIGRATIONS:
        raise ValueError(
            f"Migration from v{from_version} is already registered as "
            f"{_MIGRATIONS[from_version].__name__!r}; cannot register {fn.__name__!r}"
        )
    _MIGRATIONS[from_version] = fn
    return fn


class MemoryMigrationRegistry:
    """Versioned migration pipeline for the DadBot memory store.

    Call ``MemoryMigrationRegistry.migrate(store)`` before normalizing to
    apply all pending schema upgrades in order.  The method is idempotent:
    a store already at CURRENT_SCHEMA_VERSION is returned unchanged.
    """

    @staticmethod
    def register(from_version: int) -> Callable[[MigrationFn], MigrationFn]:
        """Decorator that registers a migration function.

        Example::

            @MemoryMigrationRegistry.register(from_version=1)
            def migrate_v1_to_v2(store: dict) -> dict:
                store["new_key"] = []
                return store
        """
        def decorator(fn: MigrationFn) -> MigrationFn:
            return _register(from_version, fn)
        return decorator

    @staticmethod
    def migrate(store: dict) -> dict:
        """Apply all pending migrations to *store* and stamp schema_version.

        Always returns the (possibly mutated) store dict.  Never raises — any
        migration error is logged and the store is returned as-is at the
        pre-error version so normalization can still proceed.
        """
        if not isinstance(store, dict):
            return store

        current = int(store.get("schema_version") or 0)

        if current == CURRENT_SCHEMA_VERSION:
            return store

        if current > CURRENT_SCHEMA_VERSION:
            # Store is from a newer code version — leave it alone.
            logger.warning(
                "Memory store schema_version=%d is ahead of CURRENT_SCHEMA_VERSION=%d; "
                "skipping migration to avoid downgrade corruption.",
                current,
                CURRENT_SCHEMA_VERSION,
            )
            return store

        while current < CURRENT_SCHEMA_VERSION:
            migration_fn = _MIGRATIONS.get(current)
            if migration_fn is None:
                # No explicit migration for this gap — treat as implicit no-op and advance.
                logger.debug(
                    "No migration registered for v%d→v%d; advancing schema_version silently.",
                    current,
                    current + 1,
                )
                current += 1
                store["schema_version"] = current
                continue
            try:
                store = migration_fn(store)
                current += 1
                store["schema_version"] = current
                logger.debug("Memory store migrated v%d→v%d via %s.", current - 1, current, migration_fn.__name__)
            except Exception as exc:
                logger.error(
                    "Memory store migration v%d→v%d failed (%s); "
                    "leaving store at v%d and continuing with normalization.",
                    current,
                    current + 1,
                    exc,
                    current,
                )
                break

        return store

    @staticmethod
    def registered_versions() -> list[int]:
        """Return sorted list of registered migration source versions."""
        return sorted(_MIGRATIONS.keys())


# ---------------------------------------------------------------------------
# v0 → v1 : initial version stamp migration
# ---------------------------------------------------------------------------
# Every store without a schema_version is treated as v0 (pre-versioning).
# This migration stamps it to v1 and seeds the mcp_local_store key if absent,
# which was the last key added during the unversioned era.

@MemoryMigrationRegistry.register(from_version=0)
def _migrate_v0_to_v1(store: dict) -> dict:
    """Stamp existing stores to v1; backfill keys added in the unversioned era."""
    if "mcp_local_store" not in store:
        store["mcp_local_store"] = {}
    if "narrative_memories" not in store:
        store["narrative_memories"] = []
    if "heritage_cross_links" not in store:
        store["heritage_cross_links"] = []
    if "advice_audits" not in store:
        store["advice_audits"] = []
    if "environmental_cues_history" not in store:
        store["environmental_cues_history"] = []
    if "longitudinal_insights" not in store:
        store["longitudinal_insights"] = []
    return store


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MemoryMigrationRegistry",
]
