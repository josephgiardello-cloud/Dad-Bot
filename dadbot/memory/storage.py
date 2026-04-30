"""Memory storage I/O sub-component.

Extracted from MemoryManager so that all JSON / document-store persistence
lives in one focused class.  Receives a back-reference to the MemoryManager
instance (same pattern as MemoryGraphManager) to read/write ``memory_store``.
MemoryManager keeps delegation shims so all existing call-sites continue to
work unchanged.
"""

from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy
from pathlib import Path

from dadbot.core.execution_boundary import (
    MemoryWriteOwnerScope,
    enforce_memory_write_owner,
)
from dadbot.core.execution_trace_context import (
    ensure_execution_trace_root,
    record_execution_step,
)
from dadbot.memory.migration import MemoryMigrationRegistry

logger = logging.getLogger(__name__)


_SEMANTIC_TRACKED_KEYS = frozenset({"memories"})
_GRAPH_TRACKED_KEYS = frozenset(
    {
        "memories",
        "consolidated_memories",
        "session_archive",
        "persona_evolution",
        "life_patterns",
        "relationship_history",
        "relationship_state",
    },
)


def _section_payload(store: dict, keys: set[str] | frozenset[str]) -> dict:
    return {key: store.get(key) for key in keys}


def _section_changed(
    before: dict,
    after: dict,
    keys: set[str] | frozenset[str],
) -> bool:
    return _section_payload(before, keys) != _section_payload(after, keys)


class _MemoryStoreParticipant:
    def __init__(self, backend: MemoryStorageBackend, previous_store: dict) -> None:
        self._backend = backend
        self._previous_store = deepcopy(previous_store)

    def prepare(self) -> None:
        return None

    def commit(self) -> None:
        self._backend._write_memory_store_unlocked()

    def rollback(self) -> None:
        self._backend._manager.memory_store = deepcopy(self._previous_store)
        self._backend._write_memory_store_unlocked()


class _SemanticIndexParticipant:
    def __init__(
        self,
        backend: MemoryStorageBackend,
        before_store: dict,
        after_store: dict,
    ) -> None:
        self._backend = backend
        self._before_memories = list(before_store.get("memories") or [])
        self._after_memories = list(after_store.get("memories") or [])

    def prepare(self) -> None:
        return None

    def commit(self) -> None:
        self._backend._manager.sync_semantic_memory_index(self._after_memories)

    def rollback(self) -> None:
        self._backend._manager.sync_semantic_memory_index(self._before_memories)


class _GraphStoreParticipant:
    def __init__(self, backend: MemoryStorageBackend) -> None:
        self._backend = backend
        self._graph_manager = getattr(self._backend._manager, "graph_manager", None)
        self._before_snapshot: dict = {"nodes": [], "edges": [], "updated_at": None}

    def prepare(self) -> None:
        if self._graph_manager is None:
            return
        snapshot = getattr(self._graph_manager, "graph_snapshot", None)
        if callable(snapshot):
            self._before_snapshot = dict(snapshot() or self._before_snapshot)

    def commit(self) -> None:
        if self._graph_manager is None:
            return
        sync_graph_store = getattr(self._graph_manager, "sync_graph_store", None)
        if callable(sync_graph_store):
            sync_graph_store(turn_context=None)

    def rollback(self) -> None:
        if self._graph_manager is None:
            return
        ensure_graph_store = getattr(self._graph_manager, "ensure_graph_store", None)
        backend = getattr(self._graph_manager, "_graph_store_backend", None)
        replace_graph = getattr(backend, "replace_graph", None)
        if callable(ensure_graph_store):
            ensure_graph_store()
        if callable(replace_graph):
            replace_graph(
                deepcopy(list(self._before_snapshot.get("nodes", []) or [])),
                deepcopy(list(self._before_snapshot.get("edges", []) or [])),
            )


class MemoryStorageBackend:
    """Handles all persistence I/O for the DadBot memory store.

    Depends on:
    - ``bot``     â€” for path helpers, locks, document-store access, atomic write
    - ``manager`` â€” back-reference to MemoryManager for ``memory_store`` access
                    and for ``normalize_memory_store`` (lives on normalizer via
                    manager delegation shim)
    """

    def __init__(self, bot, manager) -> None:
        self.bot = bot
        self._manager = manager

    # ------------------------------------------------------------------ private write helpers

    def _write_json_memory_store_unlocked(self):
        try:
            self.bot.write_json_atomically(
                self.bot.MEMORY_PATH,
                self._manager.memory_store,
                backup=True,
            )
        except OSError as exc:
            self.bot.record_runtime_issue(
                "memory store save",
                "keeping in-memory state because the JSON mirror write failed",
                exc,
                level=logging.INFO,
            )

    def _write_memory_store_unlocked(self):
        if getattr(self.bot, "_tenant_document_store", None) is not None:
            try:
                self.bot._tenant_document_store.save_session_state(
                    "memory",
                    self._manager.memory_store,
                )
            except OSError as exc:
                self.bot.record_runtime_issue(
                    "memory store save",
                    "document-store write failed, so Dad Bot fell back to the local JSON mirror",
                    exc,
                    level=logging.INFO,
                )
                self._write_json_memory_store_unlocked()
                return
            self._write_json_memory_store_unlocked()
            return
        self._write_json_memory_store_unlocked()

    # ------------------------------------------------------------------ public mutation

    def _apply_mutation_in_memory(self, *, mutator=None, changes=None, normalize=True):
        if mutator is not None:
            mutator(self._manager.memory_store)
        if changes:
            self._manager.memory_store.update(changes)
        if normalize:
            self._manager.memory_store = self._manager.normalize_memory_store(
                self._manager.memory_store,
            )

    def _coordinated_commit(self, *, before_store: dict, changes: dict) -> None:
        tx_module = __import__(
            "dadbot.core.transaction_coordinator",
            fromlist=["TransactionContext", "TransactionCoordinator"],
        )
        transaction_context_cls = getattr(tx_module, "TransactionContext")
        transaction_coordinator_cls = getattr(tx_module, "TransactionCoordinator")

        after_store = deepcopy(dict(self._manager.memory_store or {}))
        changed_keys = set((changes or {}).keys())

        semantic_changed = bool(
            _SEMANTIC_TRACKED_KEYS.intersection(changed_keys),
        ) or _section_changed(
            before_store,
            after_store,
            _SEMANTIC_TRACKED_KEYS,
        )
        graph_changed = bool(
            _GRAPH_TRACKED_KEYS.intersection(changed_keys),
        ) or _section_changed(
            before_store,
            after_store,
            _GRAPH_TRACKED_KEYS,
        )

        coordinator = transaction_coordinator_cls(
            transaction_context_cls(
                transaction_id=uuid.uuid4().hex,
                metadata={
                    "source": "MemoryStorageBackend.mutate_memory_store",
                    "semantic_changed": semantic_changed,
                    "graph_changed": graph_changed,
                },
            ),
        )
        coordinator.register(_MemoryStoreParticipant(self, before_store))
        if semantic_changed:
            coordinator.register(
                _SemanticIndexParticipant(self, before_store, after_store),
            )
        if graph_changed:
            coordinator.register(_GraphStoreParticipant(self))

        report = coordinator.execute()
        if not report.ok:
            raise RuntimeError(
                f"Cross-store transaction failed and was rolled back: {report.error or 'unknown error'}",
            )

    def mutate_memory_store(
        self,
        mutator=None,
        normalize=True,
        save=True,
        owner=None,
        **changes,
    ):
        with ensure_execution_trace_root(
            operation="memory_write",
            prompt="[memory-storage-mutate]",
            metadata={"source": "MemoryStorageBackend.mutate_memory_store"},
            required=True,
        ):
            enforce_memory_write_owner(owner=str(owner or ""))
            record_execution_step(
                "memory_write",
                payload={
                    "source": "MemoryStorageBackend.mutate_memory_store",
                    "changed_keys": sorted(str(key) for key in (changes or {}).keys()),
                    "normalize": bool(normalize),
                    "save": bool(save),
                },
                required=True,
            )
            io_lock = getattr(self.bot, "_io_lock", None)
            if io_lock is None:
                before_store = deepcopy(dict(self._manager.memory_store or {}))
                self._apply_mutation_in_memory(
                    mutator=mutator,
                    changes=changes,
                    normalize=normalize,
                )
                if save:
                    self._coordinated_commit(before_store=before_store, changes=changes)
                return self._manager.memory_store

            with io_lock:
                before_store = deepcopy(dict(self._manager.memory_store or {}))
                self._apply_mutation_in_memory(
                    mutator=mutator,
                    changes=changes,
                    normalize=normalize,
                )
                if save:
                    self._coordinated_commit(before_store=before_store, changes=changes)
                return self._manager.memory_store

    # ------------------------------------------------------------------ load

    def load_memory_store(self):
        """Load from disk / document store, normalize, and return the dict."""
        tenant_document_store = getattr(self.bot, "_tenant_document_store", None)
        if tenant_document_store is not None:
            try:
                persisted = tenant_document_store.load_session_state("memory")
            except OSError as exc:
                persisted = None
                self.bot.record_runtime_issue(
                    "memory store load",
                    "document-store read failed, so Dad Bot fell back to the local JSON mirror",
                    exc,
                    level=logging.INFO,
                )
            if isinstance(persisted, dict):
                migrated = MemoryMigrationRegistry.migrate(persisted)
                normalized = self._manager.normalize_memory_store(migrated)
                if normalized != persisted:
                    tenant_document_store.save_session_state("memory", normalized)
                    self.bot.write_json_atomically(
                        self.bot.MEMORY_PATH,
                        normalized,
                        backup=False,
                    )
                return normalized

        primary_path = self.bot.MEMORY_PATH
        backup_path = self.bot.json_backup_path(primary_path)
        candidates = [("primary", primary_path), ("backup", backup_path)]
        last_error = None

        for label, path in candidates:
            if not path.exists():
                continue
            try:
                from dadbot.utils import json_load

                with path.open("r", encoding="utf-8") as memory_file:
                    loaded = json_load(memory_file)
            except (OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if label == "primary":
                    self.bot.capture_corrupt_json_snapshot(path)
                continue

            migrated = MemoryMigrationRegistry.migrate(loaded)
            normalized = self._manager.normalize_memory_store(migrated)
            should_restore_primary = label == "backup" or normalized != loaded
            if should_restore_primary:
                self.bot.write_json_atomically(primary_path, normalized, backup=False)
                if label == "backup":
                    self.bot.record_runtime_issue(
                        "memory store load",
                        "restored from the last good backup snapshot",
                        last_error,
                        level=logging.INFO,
                    )
            if tenant_document_store is not None:
                tenant_document_store.save_session_state("memory", normalized)
            return normalized

        if last_error is not None:
            self.bot.record_runtime_issue(
                "memory store load",
                "using a fresh default memory store because no readable snapshot remained",
                last_error,
            )
        store = self.bot.default_memory_store()
        if tenant_document_store is not None:
            tenant_document_store.save_session_state("memory", store)
        return store

    # ------------------------------------------------------------------ save / export / clear

    def prepare_memory_store_for_save(self):
        with MemoryWriteOwnerScope.bind("MemoryManager"):
            return self.mutate_memory_store(save=False, owner="MemoryManager")

    def save_memory_store(self):
        io_lock = getattr(self.bot, "_io_lock", None)
        if io_lock is None:
            self._manager.memory_store = self._manager.normalize_memory_store(
                self._manager.memory_store,
            )
            self._write_memory_store_unlocked()
            return
        with io_lock:
            self._manager.memory_store = self._manager.normalize_memory_store(
                self._manager.memory_store,
            )
            self._write_memory_store_unlocked()

    def export_memory_store(self, export_path):
        destination = Path(export_path)
        with self.bot._io_lock:
            normalized = self._manager.normalize_memory_store(
                self._manager.memory_store,
            )
            self.bot.write_json_atomically(destination, normalized, backup=False)

    def clear_memory_store(self):
        """Reset memory store to defaults and clear semantic + graph indexes."""
        with self.bot._io_lock:
            self._manager.memory_store = self.bot.default_memory_store()
            self._manager.memory_store = self._manager.normalize_memory_store(
                self._manager.memory_store,
            )
            self._write_memory_store_unlocked()
        self._manager.clear_semantic_memory_index()
        self._manager.clear_graph_store()


__all__ = ["MemoryStorageBackend"]
