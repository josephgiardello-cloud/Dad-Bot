"""Event schema versioning and backward-compatible migration.

Every event written to the ledger receives a ``_schema_version`` stamp.
When loading events from a WAL, ``EventSchemaMigrator`` upgrades older events
to the current schema before they are applied to in-memory state.

Usage::

    # Stamp new events (LedgerWriter calls this automatically):
    from dadbot.core.event_schema import stamp_schema_version
    event = stamp_schema_version(raw_event)

    # Upgrade old events on load:
    from dadbot.core.event_schema import get_migrator
    events = get_migrator().migrate_all(loaded_events)

    # Register a migration (do this once at startup):
    from dadbot.core.event_schema import get_migrator
    get_migrator().register("0.9", "1.0", lambda e: {**e, "kernel_step_id": e.get("step_id", "")})
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

# Bump this string when the event envelope structure changes.
CURRENT_SCHEMA_VERSION: str = "1.0"


def stamp_schema_version(event: dict[str, Any]) -> dict[str, Any]:
    """Add ``_schema_version`` to event if missing.  Returns event (mutates in-place)."""
    event.setdefault("_schema_version", CURRENT_SCHEMA_VERSION)
    return event


# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------

class EventSchemaMigration:
    """A single schema-version upgrade step."""

    def __init__(
        self,
        from_version: str,
        to_version: str,
        migrate_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.from_version = str(from_version)
        self.to_version   = str(to_version)
        self._fn          = migrate_fn

    def apply(self, event: dict[str, Any]) -> dict[str, Any]:
        result = self._fn(deepcopy(event))
        result["_schema_version"] = self.to_version
        return result


class EventSchemaMigrator:
    """Registry and engine for schema migrations.

    Maintains a list of ``(from_version, to_version)`` migration steps.
    ``migrate(event)`` walks the chain until the event reaches
    ``CURRENT_SCHEMA_VERSION``.
    """

    def __init__(self) -> None:
        self._migrations: list[EventSchemaMigration] = []

    def register(
        self,
        from_version: str,
        to_version: str,
        migrate_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> "EventSchemaMigrator":
        """Register a migration step.  Returns self for chaining."""
        self._migrations.append(
            EventSchemaMigration(from_version, to_version, migrate_fn)
        )
        return self

    def migrate(self, event: dict[str, Any]) -> dict[str, Any]:
        """Return event upgraded to CURRENT_SCHEMA_VERSION.

        - Events already at the current version are returned as-is.
        - Events with no ``_schema_version`` are treated as legacy (0.0)
          and immediately stamped with the current version (no structural
          migration needed for the initial upgrade).
        """
        version = str(event.get("_schema_version") or "0.0")
        if version == CURRENT_SCHEMA_VERSION:
            return event

        event = deepcopy(event)

        # Legacy events: just stamp them.
        if version == "0.0":
            event["_schema_version"] = CURRENT_SCHEMA_VERSION
            return event

        # Walk the migration chain (max 50 hops to prevent infinite loops).
        for _ in range(50):
            if version == CURRENT_SCHEMA_VERSION:
                break
            step = next(
                (m for m in self._migrations if m.from_version == version),
                None,
            )
            if step is None:
                # No path forward â€” stop here.
                break
            event = step.apply(event)
            version = str(event.get("_schema_version") or version)

        return event

    def migrate_all(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.migrate(e) for e in events]

    def needs_migration(self, event: dict[str, Any]) -> bool:
        return str(event.get("_schema_version") or "0.0") != CURRENT_SCHEMA_VERSION

    @property
    def migration_count(self) -> int:
        return len(self._migrations)


# ---------------------------------------------------------------------------
# Global default migrator
# ---------------------------------------------------------------------------

_DEFAULT_MIGRATOR = EventSchemaMigrator()


def get_migrator() -> EventSchemaMigrator:
    return _DEFAULT_MIGRATOR


def migrate_event(event: dict[str, Any]) -> dict[str, Any]:
    return _DEFAULT_MIGRATOR.migrate(event)
