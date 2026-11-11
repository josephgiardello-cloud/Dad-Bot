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
import sqlite3
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from dadbot.core.core_state import CoreState, InputEvent, memory_projection, transition
from dadbot.core.execution_boundary import (
    MemoryWriteOwnerScope,
    enforce_memory_write_owner,
)
from dadbot.core.execution_context import (
    ensure_execution_trace_root,
    get_active_core_state,
    push_core_state_event,
    record_execution_step,
    require_bound_core_state_for_mutation,
)
from dadbot.core.kernel_locks import KernelEventTotalityLock
from dadbot.core.mutation_entry_invariants import enforce_mutation_entry_invariants
from dadbot.core.state_lineage import canonical_state_hash
from dadbot.memory.migration import MemoryMigrationRegistry

if TYPE_CHECKING:
    from dadbot.core.ledger_backend import FileWALLedgerBackend

logger = logging.getLogger(__name__)


_SEMANTIC_TRACKED_KEYS = frozenset({"memories"})
_MEMORY_ACCESS_FIELDS = frozenset(
    {"access_count", "confidence_history", "high_confidence_hits", "last_accessed_at"}
)
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


def _only_memory_access_changed(before: dict, after: dict, changed_keys: set[str]) -> bool:
    if changed_keys != {"memories"}:
        return False
    before_memories = before.get("memories")
    after_memories = after.get("memories")
    if not isinstance(before_memories, list) or not isinstance(after_memories, list):
        return False
    if len(before_memories) != len(after_memories):
        return False
    return all(
        isinstance(old, dict)
        and isinstance(new, dict)
        and {k: v for k, v in old.items() if k not in _MEMORY_ACCESS_FIELDS}
        == {k: v for k, v in new.items() if k not in _MEMORY_ACCESS_FIELDS}
        for old, new in zip(before_memories, after_memories)
    )


def _required_projection_revisions(events: list[dict], tenant_id: str) -> dict[str, int]:
    required: dict[str, int] = {}
    previous: dict = {}
    for event in events:
        if event.get("type") != "TENANT_MEMORY_COMMITTED":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("tenant_id") != tenant_id:
            continue
        state = payload["state"]
        changed_keys = {
            key for key in set(previous) | set(state)
            if previous.get(key) != state.get(key)
        }
        if not _only_memory_access_changed(previous, state, changed_keys):
            for name, keys in (("graph", _GRAPH_TRACKED_KEYS), ("semantic", _SEMANTIC_TRACKED_KEYS)):
                if _section_changed(previous, state, keys):
                    required[name] = payload["revision"]
        previous = state
    return required


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
        from dadbot_system.delegation import _execution_authority

        self.bot = bot
        self._manager = manager
        self._authenticated_authority_required = _execution_authority.get() is not None
        self._projected_store: dict[str, object] = {}
        self._tenant_base_revision: int | None = None
        self._tenant_recovery_required = False
        self._fenced_tenant_snapshot: tuple[FileWALLedgerBackend, str, bytes, int, dict] | None = None

    def require_authenticated_authority(self) -> None:
        self._authenticated_authority_required = True

    def _authorize_tenant_memory(self) -> None:
        from dadbot_system.delegation import _execution_authority, require_tenant_authority

        if self._authenticated_authority_required or _execution_authority.get() is not None:
            require_tenant_authority(self.bot.config.tenant_id)

    def _local_authority_backend(self):
        from dadbot.core.ledger_backend import FileWALLedgerBackend

        ledger = getattr(self.bot, "_execution_ledger", None)
        backend = getattr(ledger, "_backend", None)
        return backend if isinstance(backend, FileWALLedgerBackend) else None

    def recover_local_projections(self):
        """Attempt each rebuild independently; the ledger remains authoritative."""
        self._authorize_tenant_memory()
        from dadbot.core.ledger_backend import WALCorruptionError
        from dadbot.core.tenant_memory_ledger import TenantMemoryLedgerCorruption, TenantMemoryRevisionConflict

        if self._local_authority_backend() is None:
            return
        graph = self._manager.graph_manager
        self._projection_recovery_status = self._load_projection_recovery_status()
        repairs = (
            ("graph", lambda: (graph.invalidate_projection_cache(), graph.sync_graph_store(turn_context=None))),
            ("semantic", lambda: self._manager.sync_semantic_memory_index(
                list(self._manager.memory_projection().get("memories") or []),
            )),
        )
        for name, repair in repairs:
            previous = self._projection_recovery_status.get(name, {})
            generation = int(previous.get("generation") or 0) + 1
            try:
                repair()
            except (WALCorruptionError, TenantMemoryLedgerCorruption, TenantMemoryRevisionConflict):
                raise
            except (OSError, sqlite3.Error, RuntimeError) as exc:
                logger.warning("%s projection recovery failed: %s", name, exc, exc_info=True)
                status = {
                    "status": "degraded",
                    "generation": generation,
                    "source_revision": self._tenant_base_revision,
                    "updated_at": time.time(),
                    "error": str(exc),
                }
            else:
                status = {
                    "status": "sync_succeeded",
                    "generation": generation,
                    "source_revision": self._tenant_base_revision,
                    "updated_at": time.time(),
                }
            self._persist_projection_recovery_status(name, status)
            self._projection_recovery_status[name] = status

    def _load_projection_recovery_status(self) -> dict[str, dict]:
        tenant_id = self.bot.config.tenant_id
        latest: dict[str, dict] = {}
        ledger = getattr(self.bot, "_execution_ledger", None)
        if ledger is None:
            return latest
        events = ledger.read()
        for event in events:
            if event.get("type") != "PROJECTION_STATUS_UPDATED":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict) or payload.get("tenant_id") != tenant_id:
                continue
            projection = payload.get("projection")
            status = payload.get("status")
            if projection not in {"graph", "semantic"} or not isinstance(status, dict):
                continue
            previous = latest.get(projection)
            if previous is None or int(status.get("generation") or 0) >= int(previous.get("generation") or 0):
                latest[projection] = deepcopy(status)
        for name, revision in _required_projection_revisions(events, tenant_id).items():
            status = latest.get(name, {"generation": 0, "source_revision": None})
            if int(status.get("source_revision") or 0) < revision:
                latest[name] = {**status, "status": "dirty", "required_revision": revision}
        return latest

    def _persist_projection_recovery_status(self, projection: str, status: dict) -> None:
        ledger = getattr(self.bot, "_execution_ledger", None)
        if ledger is None:
            raise RuntimeError("Projection recovery status requires a durable execution ledger")
        event = {
            "type": "PROJECTION_STATUS_UPDATED",
            "session_id": "",
            "trace_id": f"projection-recovery:{projection}",
            "timestamp": str(status["updated_at"]),
            "kernel_step_id": "memory.projection.recovery",
            "committed": True,
            "payload": {
                "tenant_id": self.bot.config.tenant_id,
                "projection": projection,
                "status": deepcopy(status),
            },
        }
        from dadbot.core.ledger_backend import LedgerHeadConflict

        for attempt in range(3):
            try:
                ledger.write(event)
                return
            except LedgerHeadConflict:
                if attempt == 2:
                    raise
                ledger.load_from_backend()

    @contextmanager
    def fenced_semantic_index_write(self, memories, expected_revision=None):
        self._authorize_tenant_memory()
        """Only index the current committed catalog while holding the WAL fence."""
        backend = self._local_authority_backend()
        if backend is None:
            yield
            return
        from dadbot.core.durability import FileLockMutex
        from dadbot.core.ledger_backend import WALCorruptionError
        from dadbot.core.tenant_memory_ledger import TenantMemoryRevisionConflict

        with self.bot._io_lock, backend._lock:
            lock = FileLockMutex(backend._path.with_name(backend._path.name + ".lock"))
            with lock.locked():
                if not backend._path.exists():
                    raise WALCorruptionError("Execution WAL is missing during semantic index repair")
                revision, state = self._fenced_tenant_projection(backend)
                if list(state.get("memories") or []) != list(memories):
                    raise RuntimeError("Semantic index input does not match committed tenant memory")
                if expected_revision is not None and revision != expected_revision:
                    raise TenantMemoryRevisionConflict(
                        "Semantic candidate is not based on the committed tenant revision"
                    )
                yield revision

    def _fenced_tenant_projection(self, backend: FileWALLedgerBackend) -> tuple[int, dict]:
        """Read committed authority while the caller holds the raw WAL writer fence."""
        from dadbot.core.execution_ledger import ExecutionLedger
        from dadbot.core.ledger_backend import FileWALLedgerBackend, InMemoryLedgerBackend

        tenant_id = self.bot.config.tenant_id
        snapshot = self._fenced_tenant_snapshot
        self._fenced_tenant_snapshot = None
        digest = snapshot[2] if (
            snapshot is not None and snapshot[0] is backend and snapshot[1] == tenant_id
            and type(backend) is FileWALLedgerBackend
        ) else None
        append_snapshot = backend._append_verifier._snapshot if type(backend) is FileWALLedgerBackend else None
        if append_snapshot is not None:
            digest = append_snapshot[0]
        current_digest, events = backend._load_if_changed_unlocked(digest)
        if (
            events is None and snapshot is not None and snapshot[0] is backend
            and snapshot[1] == tenant_id and snapshot[2] == current_digest
        ):
            revision, state = snapshot[3], snapshot[4]
        elif events is None and append_snapshot is not None:
            from dadbot.core.tenant_memory_ledger import replay_tenant_memory, validate_tenant_event_seal

            for event in append_snapshot[1]:
                validate_tenant_event_seal(event)
            current = replay_tenant_memory(append_snapshot[1], tenant_id)
            if not isinstance(current.state, dict) or current.revision == 0:
                raise RuntimeError("Tenant memory authority is unavailable during index repair")
            revision, state = current.revision, current.state
        elif events is None:
            raise RuntimeError("Unchanged WAL requires a previously verified tenant projection")
        else:
            verifier_backend = InMemoryLedgerBackend()
            verifier_backend._events = events
            ledger = ExecutionLedger(verifier_backend)
            ledger.load_from_backend()
            current = ledger.replay_tenant_memory(tenant_id)
            if not isinstance(current.state, dict) or current.revision == 0:
                raise RuntimeError("Tenant memory authority is unavailable during index repair")
            revision, state = current.revision, current.state
        self._fenced_tenant_snapshot = (backend, tenant_id, current_digest, revision, state)
        return revision, deepcopy(state)

    @contextmanager
    def fenced_graph_index_write(self, expected_revision, expected_projection):
        self._authorize_tenant_memory()
        backend = self._local_authority_backend()
        if backend is None:
            yield
            return
        from dadbot.core.durability import FileLockMutex
        from dadbot.core.ledger_backend import WALCorruptionError
        from dadbot.core.tenant_memory_ledger import TenantMemoryRevisionConflict

        with backend._lock:
            lock = FileLockMutex(backend._path.with_name(backend._path.name + ".lock"))
            with lock.locked():
                if not backend._path.exists():
                    raise WALCorruptionError("Execution WAL is missing during graph index repair")
                revision, state = self._fenced_tenant_projection(backend)
                if (
                    revision != expected_revision
                    or self._manager.normalize_memory_store(state) != expected_projection
                ):
                    raise TenantMemoryRevisionConflict("Graph projection is not based on the committed tenant revision")
                yield

    def _commit_local_projection(self, projection: dict, publish=None) -> None:
        from dadbot_system.delegation import tenant_mutation_fence

        with tenant_mutation_fence(
            self.bot.config.tenant_id, required=self._authenticated_authority_required
        ):
            self._commit_local_projection_fenced(projection, publish=publish)

    def _commit_local_projection_fenced(self, projection: dict, publish=None) -> None:
        self._authorize_tenant_memory()
        from dadbot.core.local_tenant_establishment import commit_local_tenant
        from dadbot.core.tenant_memory_ledger import TenantMemoryRevisionConflict

        if self._tenant_recovery_required or self._tenant_base_revision is None:
            raise RuntimeError("Tenant memory requires restart and ledger recovery before writing")
        committed = self.bot._execution_ledger.replay_tenant_memory(self.bot.config.tenant_id)
        if not isinstance(committed.state, dict):
            raise RuntimeError("Committed tenant memory is unavailable")
        semantic_changed = _section_changed(committed.state, projection, _SEMANTIC_TRACKED_KEYS)
        graph_changed = _section_changed(committed.state, projection, _GRAPH_TRACKED_KEYS)
        backend = self._local_authority_backend()
        if backend is None:
            raise RuntimeError("Local tenant memory commits require file-WAL authority")
        if semantic_changed or graph_changed:
            self._require_tenant_isolated_indexes()
        try:
            ledger = self.bot._execution_ledger
            # Keep ordinary event writers behind the raw WAL commit and its head refresh.
            with ledger._lock:
                revision = commit_local_tenant(
                    backend,
                    self.bot.config.tenant_id,
                    self._tenant_base_revision,
                    uuid.uuid4().hex,
                    projection,
                    publish or (lambda: None),
                    self._repair_established_snapshot,
                    execution_ledger=ledger,
                )
                ledger.load_from_backend(only_if_changed=not ledger._backend_uncertain)
        except TenantMemoryRevisionConflict:
            raise
        except Exception:
            self._tenant_recovery_required = True
            raise
        self._tenant_base_revision = revision
        if semantic_changed or graph_changed:
            self._repair_committed_indexes(revision)

    def _require_tenant_isolated_indexes(self) -> None:
        from dadbot.core.local_tenant_establishment import local_projection_path
        from dadbot_system.graph_store import SQLiteGraphStore
        from dadbot_system.semantic_index import SQLiteSemanticIndex

        semantic = self._manager._semantic_manager._semantic_index_backend
        graph = self._manager.graph_manager._graph_store_backend
        if not isinstance(semantic, SQLiteSemanticIndex) or not isinstance(graph, SQLiteGraphStore):
            raise RuntimeError("Indexed local tenant memory writes require tenant-isolated SQLite projections")
        expected_semantic = local_projection_path(self.bot, self.bot.semantic_memory_db_path)
        expected_graph = local_projection_path(self.bot, self.bot.graph_store_db_path)
        if (
            semantic.db_path.resolve() != Path(expected_semantic).resolve()
            or graph.db_path.resolve() != Path(expected_graph).resolve()
        ):
            raise RuntimeError("Indexed local tenant memory writes require tenant-qualified projection paths")

    def _repair_committed_indexes(self, revision: int) -> None:
        repaired = False
        try:
            self.recover_local_projections()
            failed = [
                name for name in ("graph", "semantic")
                if self._projection_recovery_status.get(name, {}).get("status") != "sync_succeeded"
                or self._projection_recovery_status.get(name, {}).get("source_revision") != revision
            ]
            if failed:
                raise RuntimeError(
                    f"Tenant memory committed at revision {revision}; "
                    f"projection repair requires restart ({', '.join(failed)})",
                )
            repaired = True
        finally:
            if not repaired:
                self._tenant_recovery_required = True

    def committed_tenant_projection(self) -> dict | None:
        self._authorize_tenant_memory()
        ledger = getattr(self.bot, "_execution_ledger", None)
        if ledger is None:
            return None
        tenant = ledger.refresh_tenant_memory(self.bot.config.tenant_id)
        if tenant.revision == 0:
            return None
        if not isinstance(tenant.state, dict):
            raise RuntimeError("Committed tenant memory has no valid state")
        return self._manager.normalize_memory_store(deepcopy(tenant.state))

    def _require_snapshot_authority(self) -> None:
        if self.committed_tenant_projection() is not None:
            raise RuntimeError(
                "Tenant memory is ledger-authoritative; snapshot-only writes are disabled "
                "until the ledger-backed memory commit path is available",
            )

    # ------------------------------------------------------------------ private write helpers

    def _write_json_memory_store_unlocked(self, store_projection: dict, *, required: bool = True):
        self._authorize_tenant_memory()
        self._require_snapshot_authority()
        try:
            self.bot.write_json_atomically(
                self.bot.MEMORY_PATH,
                store_projection,
                backup=True,
            )
        except OSError as exc:
            self.bot.record_runtime_issue(
                "memory store save",
                "the JSON memory store write failed",
                exc,
                level=logging.INFO,
            )
            if required:
                raise

    def _write_memory_store_unlocked(self, store_projection: dict):
        self._authorize_tenant_memory()
        self._require_snapshot_authority()
        self._manager.cache_invalidation_revision += 1
        tenant_store = getattr(self.bot, "_tenant_document_store", None)
        if tenant_store is not None:
            try:
                save = getattr(tenant_store, "save_memory_state", tenant_store.save_session_state)
                save(
                    "memory",
                    store_projection,
                )
            except OSError as exc:
                self.bot.record_runtime_issue(
                    "memory store save",
                    "document-store write failed; memory changes require recovery if rollback fails",
                    exc,
                )
                raise
            self._write_json_memory_store_unlocked(
                store_projection,
                required=bool(getattr(self.bot, "_local_ephemeral_document_store", False)),
            )
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
        if _only_memory_access_changed(before_store, after_store, changed_keys):
            semantic_changed = False
            graph_changed = False

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
                f"Cross-store transaction failed; rollback was attempted: {report.error or 'unknown error'}",
            )

    def _resolve_transition_base_state(self) -> CoreState:
        current_state = getattr(self._manager, "_current_memory_core_state", None)
        if callable(current_state):
            state = current_state()
            if isinstance(state, CoreState):
                return state
        active = get_active_core_state()
        if isinstance(active, CoreState):
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
        self._manager.cache_invalidation_revision += 1
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
        self._authorize_tenant_memory()
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
        local_authority = self._local_authority_backend() is not None
        if save and not local_authority:
            self._require_snapshot_authority()
        if local_authority and self._tenant_recovery_required:
            raise RuntimeError("Tenant memory requires restart and ledger recovery before writing")
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
        if save and local_authority:
            candidate_state = before_state
            for key, value in event_changes.items():
                candidate_state = transition(
                    candidate_state,
                    InputEvent(
                        event_type="MemoryDeleted" if value is None else "MemoryUpdated",
                        payload={"key": str(key)} if value is None else {"key": str(key), "value": value},
                    ),
                )
            durable_candidate = self._manager.normalize_memory_store(
                memory_projection(candidate_state, defaults=self.bot.default_memory_store()),
            )
            published = []

            def publish():
                next_state = self._emit_memory_events(before_store=before_store, changes=event_changes)
                result = self._project_memory_store(next_state)
                if result != durable_candidate:
                    raise RuntimeError("Published memory projection differs from committed tenant state")
                published.append(result)

            self._commit_local_projection(durable_candidate, publish=publish)
        try:
            if save and local_authority:
                after_store = published[0]
            else:
                next_state = self._emit_memory_events(before_store=before_store, changes=event_changes)
                after_store = self._project_memory_store(next_state)
            if save and local_authority and after_store != durable_candidate:
                raise RuntimeError("Published memory projection differs from committed tenant state")
        except Exception:
            if save and local_authority:
                self._tenant_recovery_required = True
            raise
        if save and not local_authority:
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
        self._authorize_tenant_memory()
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

    def _load_legacy_for_establishment(self, *, allow_tenant_mirror_fallback: bool) -> dict:
        self._authorize_tenant_memory()
        tenant_store = getattr(self.bot, "_tenant_document_store", None)
        if tenant_store is not None:
            load = getattr(tenant_store, "load_memory_state", tenant_store.load_session_state)
            persisted = load("memory")
            if persisted is not None:
                if not isinstance(persisted, dict):
                    raise ValueError("Invalid tenant memory snapshot: expected an object")
                return self._manager.normalize_memory_store(MemoryMigrationRegistry.migrate(deepcopy(persisted)))
            if not allow_tenant_mirror_fallback:
                return self._manager.normalize_memory_store(self.bot.default_memory_store())
        if not allow_tenant_mirror_fallback:
            return self._manager.normalize_memory_store(self.bot.default_memory_store())
        legacy_paths = (self.bot.MEMORY_PATH, self.bot.json_backup_path(self.bot.MEMORY_PATH))
        if any(path.exists() for path in legacy_paths):
            raise RuntimeError(
                "A shared legacy memory snapshot has no tenant ownership evidence; "
                "establish tenant ownership before importing it"
            )
        return self._manager.normalize_memory_store(self.bot.default_memory_store())

    def _repair_established_snapshot(self, state: dict) -> None:
        self._authorize_tenant_memory()
        tenant_store = getattr(self.bot, "_tenant_document_store", None)
        if tenant_store is not None:
            save = getattr(tenant_store, "save_memory_state", tenant_store.save_session_state)
            save("memory", state)

    def load_memory_store(self, *, allow_tenant_mirror_fallback: bool | None = None):
        """Load from disk / document store, normalize, and return the dict."""
        self._authorize_tenant_memory()
        if allow_tenant_mirror_fallback is None:
            allow_tenant_mirror_fallback = bool(
                getattr(self.bot, "_local_ephemeral_document_store", False)
            )
        ledger = getattr(self.bot, "_execution_ledger", None)
        from dadbot.core.ledger_backend import FileWALLedgerBackend
        from dadbot.core.local_tenant_establishment import establish_local_tenant

        backend = getattr(ledger, "_backend", None)
        if isinstance(backend, FileWALLedgerBackend):
            state = establish_local_tenant(
                backend,
                self.bot.config.tenant_id,
                lambda: self._load_legacy_for_establishment(
                    allow_tenant_mirror_fallback=allow_tenant_mirror_fallback,
                ),
                self._repair_established_snapshot,
                empty_state_factory=self.bot.default_memory_store,
            )
            ledger.load_from_backend()
            tenant = ledger.replay_tenant_memory(self.bot.config.tenant_id)
            if tenant.state != state:
                raise RuntimeError("Tenant memory changed during startup; retry hydration")
            self._tenant_base_revision = tenant.revision
            initialized = self._emit_memory_event("MemoryInitialized", {"store": state})
            return self._project_memory_store(initialized)
        if ledger is not None:
            tenant = ledger.replay_tenant_memory(self.bot.config.tenant_id)
            if tenant.revision:
                raise RuntimeError(
                    "Tenant memory authority exists in the execution ledger, but memory "
                    "hydration and writes are not yet ledger-authoritative; refusing to "
                    "load a snapshot that could overwrite committed memory",
                )
        tenant_document_store = getattr(self.bot, "_tenant_document_store", None)
        if tenant_document_store is not None:
            load = getattr(tenant_document_store, "load_memory_state", tenant_document_store.load_session_state)
            save = getattr(tenant_document_store, "save_memory_state", tenant_document_store.save_session_state)
            try:
                persisted = load("memory")
            except OSError as exc:
                if not allow_tenant_mirror_fallback:
                    raise
                persisted = None
                self.bot.record_runtime_issue(
                    "memory store load",
                    "document-store read failed, so Dad Bot fell back to the local JSON mirror",
                    exc,
                    level=logging.INFO,
                )
            if isinstance(persisted, dict):
                migrated = MemoryMigrationRegistry.migrate(deepcopy(persisted))
                normalized = self._manager.normalize_memory_store(migrated)
                if normalized != persisted:
                    save("memory", normalized)
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
            if persisted is not None:
                raise ValueError("Invalid tenant memory snapshot: expected an object")
            if not allow_tenant_mirror_fallback:
                initialized = self._emit_memory_event(
                    "MemoryInitialized",
                    {"store": self.bot.default_memory_store()},
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
                if not isinstance(loaded, dict):
                    raise ValueError("Memory snapshot must be an object")
                migrated = MemoryMigrationRegistry.migrate(deepcopy(loaded))
                normalized = self._manager.normalize_memory_store(migrated)
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = exc
                if label == "primary":
                    self.bot.capture_corrupt_json_snapshot(path)
                continue

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
                save("memory", normalized)
            initialized = self._emit_memory_event(
                "MemoryInitialized",
                {"store": normalized},
            )
            return self._project_memory_store(initialized)

        if last_error is not None:
            self.bot.record_runtime_issue(
                "memory store load",
                "no valid memory snapshot remained; refusing to initialize an empty store",
                last_error,
            )
            raise RuntimeError("Cannot load memory store: no valid snapshot remains") from last_error
        store = self.bot.default_memory_store()
        if tenant_document_store is not None:
            save("memory", store)
        initialized = self._emit_memory_event(
            "MemoryInitialized",
            {"store": store},
        )
        return self._project_memory_store(initialized)

    # ------------------------------------------------------------------ save / export / clear

    def prepare_memory_store_for_save(self):
        self._authorize_tenant_memory()
        with MemoryWriteOwnerScope.bind("MemoryManager"):
            return self.mutate_memory_store(save=False, owner="MemoryManager")

    def export_memory_store(self, export_path):
        self._authorize_tenant_memory()
        destination = Path(export_path)
        with self.bot._io_lock:
            normalized = self._manager.normalize_memory_store(self._manager.memory_projection())
            self.bot.write_json_atomically(destination, normalized, backup=False)

    def clear_memory_projection(self):
        self._authorize_tenant_memory()
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
            if self._local_authority_backend() is not None:
                def clear(store):
                    store.clear()
                    store.update(self.bot.default_memory_store())

                return self._manager.mutate_memory_store(mutator=clear)
            self._require_snapshot_authority()
            before_store = deepcopy(self._manager.memory_projection())
            cleared_state = self._emit_memory_event("MemoryCleared", {})
            after_store = self._project_memory_store(cleared_state)
            try:
                self._write_memory_store_unlocked(after_store)
            except Exception:
                self.replace_projection_via_canonical_event(before_store)
                raise
        self._manager.clear_semantic_memory_index()
        self._manager.clear_graph_store()
        return after_store


__all__ = ["MemoryStorageBackend"]
