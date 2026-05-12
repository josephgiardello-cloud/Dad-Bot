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
from dadbot.core.execution_context import (
    ensure_execution_trace_root,
    record_execution_step,
)
from dadbot.core.execution_context import (
    get_active_core_state,
    push_core_state_event,
    require_bound_core_state_for_mutation,
)
from dadbot.core.kernel_locks import KernelEventTotalityLock
from dadbot.core.mutation_entry_invariants import enforce_mutation_entry_invariants
from dadbot.core.core_state import CoreState, InputEvent, memory_projection, transition
from dadbot.core.state_lineage import canonical_state_hash
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
        self._backend._write_memory_store_unlocked(self._backend._projected_store)

    def rollback(self) -> None:
        restored = self._backend.replace_projection_via_canonical_event(
            self._previous_store,
            save=False,
        )
        self._backend._write_memory_store_unlocked(restored)


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
        invalidate_projection_cache = getattr(self._graph_manager, "invalidate_projection_cache", None)
        if callable(invalidate_projection_cache):
            invalidate_projection_cache()
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
        self._projected_store: dict[str, object] = {}

    # ------------------------------------------------------------------ private write helpers

    def _write_json_memory_store_unlocked(self, store_projection: dict):
        try:
            self.bot.write_json_atomically(
                self.bot.MEMORY_PATH,
                store_projection,
                backup=True,
            )
        except OSError as exc:
            self.bot.record_runtime_issue(
                "memory store save",
                "keeping in-memory state because the JSON mirror write failed",
                exc,
                level=logging.INFO,
            )

    def _write_memory_store_unlocked(self, store_projection: dict):
        if getattr(self.bot, "_tenant_document_store", None) is not None:
            try:
                self.bot._tenant_document_store.save_session_state(
                    "memory",
                    store_projection,
                )
            except OSError as exc:
                self.bot.record_runtime_issue(
                    "memory store save",
                    "document-store write failed, so Dad Bot fell back to the local JSON mirror",
                    exc,
                    level=logging.INFO,
                )
                self._write_json_memory_store_unlocked(store_projection)
                return
            self._write_json_memory_store_unlocked(store_projection)
            return
        self._write_json_memory_store_unlocked(store_projection)

    # ------------------------------------------------------------------ public mutation

    def _coordinated_commit(
        self,
        *,
        before_store: dict,
        after_store: dict,
        changes: dict,
    ) -> None:
        tx_module = __import__(
            "dadbot.core.transaction_coordinator",
            fromlist=["TransactionContext", "TransactionCoordinator"],
        )
        transaction_context_cls = tx_module.TransactionContext
        transaction_coordinator_cls = tx_module.TransactionCoordinator

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

    def _resolve_transition_base_state(self) -> CoreState:
        active = get_active_core_state()
        if active is not None:
            return active
        cached = getattr(self._manager, "memory_core_state", None)
        if callable(cached):
            state = cached()
            if isinstance(state, CoreState):
                return state
        session_state = {}
        snapshotter = getattr(self.bot, "snapshot_session_state", None)
        if callable(snapshotter):
            try:
                session_state = dict(snapshotter() or {})
            except Exception:
                session_state = {}
        return CoreState.from_dict(dict(session_state.get("core_state") or {}))

    def _emit_memory_event(self, event_type: str, payload: dict) -> CoreState:
        active = get_active_core_state()
        if active is not None:
            next_state = push_core_state_event(event_type, payload)
            resolved = next_state if next_state is not None else active
            self._manager._set_memory_core_state_cache(resolved)
            return resolved
        base = self._resolve_transition_base_state()
        resolved = transition(base, InputEvent(event_type=event_type, payload=dict(payload or {})))
        self._manager._set_memory_core_state_cache(resolved)
        return resolved

    def _emit_memory_events(self, *, before_store: dict, changes: dict) -> CoreState:
        state = self._resolve_transition_base_state()
        if not changes:
            return state
        for key in sorted(str(item) for item in dict(changes).keys()):
            value = changes.get(key)
            str_key = str(key)
            if str_key == "memories":
                before_memories = list(before_store.get("memories") or [])
                after_memories = list(value or []) if isinstance(value, list) else []
                if not after_memories and before_memories:
                    state = self._emit_memory_event("MemoryCleared", {})
                    continue
                if len(after_memories) == len(before_memories) + 1:
                    prefix_match = before_memories == after_memories[: len(before_memories)]
                    if prefix_match:
                        state = self._emit_memory_event(
                            "MemoryAppended",
                            {"entry": dict(after_memories[-1] or {})},
                        )
                        continue
                state = self._emit_memory_event(
                    "MemoryUpdated",
                    {"key": "memories", "value": after_memories},
                )
                continue
            if value is None:
                state = self._emit_memory_event("MemoryDeleted", {"key": str_key})
            else:
                state = self._emit_memory_event(
                    "MemoryUpdated",
                    {"key": str_key, "value": value},
                )
        return state

    def _project_memory_store(self, state: CoreState) -> dict:
        self._manager._set_memory_core_state_cache(state)
        projected = memory_projection(state, defaults=self.bot.default_memory_store())
        normalized = self._manager.normalize_memory_store(projected)
        self._manager._set_memory_projection_cache(normalized)
        self._projected_store = dict(normalized)
        return dict(normalized)

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
            active_run_id = str(getattr(self.bot, "_active_turn_run_id", "") or "").strip()
            if active_run_id:
                KernelEventTotalityLock.require_event_witness(
                    run_id=active_run_id,
                    source="MemoryStorageBackend.mutate_memory_store",
                )
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
            enforce_mutation_entry_invariants(
                mutation_kind="memory_store",
                source="MemoryStorageBackend.mutate_memory_store",
                changed_keys=sorted(str(key) for key in (changes or {}).keys()),
            )
            require_bound_core_state_for_mutation(
                source="MemoryStorageBackend.mutate_memory_store",
                changed_keys=sorted(str(key) for key in (changes or {}).keys()),
            )
            io_lock = getattr(self.bot, "_io_lock", None)
            if io_lock is None:
                return self._run_mutation_commit_path(
                    mutator=mutator, changes=changes, normalize=normalize, save=save,
                )
            with io_lock:
                return self._run_mutation_commit_path(
                    mutator=mutator, changes=changes, normalize=normalize, save=save,
                )

    def _run_mutation_commit_path(
        self,
        *,
        mutator,
        changes: dict,
        normalize: bool,
        save: bool,
    ) -> dict:
        """Shared mutation + commit + projection routing logic (lock-agnostic)."""
        before_state = self._resolve_transition_base_state()
        before_store = self._manager.normalize_memory_store(
            deepcopy(
                memory_projection(
                    before_state,
                    defaults=self.bot.default_memory_store(),
                )
            )
        )
        before_hash = canonical_state_hash(before_store)
        candidate_store = deepcopy(before_store)
        if mutator is not None:
            mutator(candidate_store)
        if changes:
            candidate_store.update(dict(changes or {}))
        event_changes = {
            key: candidate_store.get(key)
            for key in sorted(set(before_store.keys()) | set(candidate_store.keys()))
            if before_store.get(key) != candidate_store.get(key)
        }
        if normalize:
            normalized_candidate = self._manager.normalize_memory_store(
                {**before_store, **event_changes},
            )
            event_changes = {
                key: normalized_candidate.get(key)
                for key in sorted(event_changes.keys())
            }
        next_state = self._emit_memory_events(before_store=before_store, changes=event_changes)
        after_store = self._project_memory_store(next_state)
        if save:
            self._coordinated_commit(
                before_store=before_store,
                after_store=after_store,
                changes=event_changes,
            )
        after_hash = canonical_state_hash(after_store)
        self.bot._last_memory_state_hash = after_hash
        record_execution_step(
            "memory_projection_updated",
            payload={
                "changed_keys": sorted(str(key) for key in (event_changes or {}).keys()),
                "projection_size": len(after_store),
            },
            required=True,
        )
        record_execution_step(
            "memory_state_canonicalized",
            payload={
                "before_hash": before_hash,
                "after_hash": after_hash,
                "changed": bool(before_hash != after_hash),
                "changed_keys": sorted(str(key) for key in (event_changes or {}).keys()),
            },
            required=True,
        )
        return after_store

    def replace_projection_via_canonical_event(
        self,
        store: dict | None,
        *,
        save: bool = False,
    ) -> dict:
        """Replace projection through canonical MemoryInitialized routing.

        This closes legacy projection-only reset paths so equivalent resets
        always produce equivalent event/state transitions.
        """
        normalized = self._manager.normalize_memory_store(dict(store or {}))
        next_state = self._emit_memory_event(
            "MemoryInitialized",
            {"store": normalized},
        )
        projected = self._project_memory_store(next_state)
        # Replacement is intentionally projection-only to keep writes confined
        # to commit/rollback/clear commit-boundary paths.
        _ = save
        return projected

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
                initialized = self._emit_memory_event(
                    "MemoryInitialized",
                    {"store": normalized},
                )
                return self._project_memory_store(initialized)

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
            initialized = self._emit_memory_event(
                "MemoryInitialized",
                {"store": normalized},
            )
            return self._project_memory_store(initialized)

        if last_error is not None:
            self.bot.record_runtime_issue(
                "memory store load",
                "using a fresh default memory store because no readable snapshot remained",
                last_error,
            )
        store = self.bot.default_memory_store()
        if tenant_document_store is not None:
            tenant_document_store.save_session_state("memory", store)
        initialized = self._emit_memory_event(
            "MemoryInitialized",
            {"store": store},
        )
        return self._project_memory_store(initialized)

    # ------------------------------------------------------------------ save / export / clear

    def prepare_memory_store_for_save(self):
        with MemoryWriteOwnerScope.bind("MemoryManager"):
            return self.mutate_memory_store(save=False, owner="MemoryManager")

    def export_memory_store(self, export_path):
        destination = Path(export_path)
        with self.bot._io_lock:
            normalized = self._manager.normalize_memory_store(self._manager.memory_projection())
            self.bot.write_json_atomically(destination, normalized, backup=False)

    def clear_memory_projection(self):
        """Reset memory by canonical event and persist resulting projection."""
        enforce_mutation_entry_invariants(
            mutation_kind="memory_store",
            source="MemoryStorageBackend.clear_memory_projection",
            changed_keys=["*"],
        )
        require_bound_core_state_for_mutation(
            source="MemoryStorageBackend.clear_memory_projection",
            changed_keys=["*"],
        )
        with self.bot._io_lock:
            cleared_state = self._emit_memory_event("MemoryCleared", {})
            after_store = self._project_memory_store(cleared_state)
            self._write_memory_store_unlocked(after_store)
        self._manager.clear_semantic_memory_index()
        self._manager.clear_graph_store()
        return after_store


__all__ = ["MemoryStorageBackend"]
