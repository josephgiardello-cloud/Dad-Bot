"""Serialized first-time tenant import for a local execution WAL."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dadbot.core.durability import FileLockMutex
from dadbot.core.execution_ledger import ExecutionLedger
from dadbot.core.ledger_backend import FileWALLedgerBackend, WALCorruptionError, _validate_append_head
from dadbot.core.tenant_memory_ledger import (
    TenantMemoryOperationConflict,
    TenantMemoryRevision,
    TenantMemoryRevisionConflict,
    replay_tenant_memory,
    validate_tenant_memory_state,
)


def local_projection_path(bot: Any, path: str | Path) -> str | Path:
    ledger = getattr(bot, "_execution_ledger", None)
    if not isinstance(getattr(ledger, "_backend", None), FileWALLedgerBackend) or str(path) == ":memory:":
        return path
    tenant_hash = hashlib.sha256(bot.config.tenant_id.encode("utf-8")).hexdigest()
    configured = Path(path)
    return configured.with_name(f"{configured.stem}.tenant-{tenant_hash}{configured.suffix}")


def _write_evidence(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def establish_local_tenant(
    backend: FileWALLedgerBackend,
    tenant_id: str,
    load_legacy: Callable[[], dict[str, Any]],
    repair_snapshot: Callable[[dict[str, Any]], None],
    *,
    empty_state_factory: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Establish or replay one tenant while holding the execution WAL lock."""
    from dadbot_system.delegation import tenant_mutation_fence

    with tenant_mutation_fence(tenant_id, required=False):
        return _establish_local_tenant_fenced(
            backend,
            tenant_id,
            load_legacy,
            repair_snapshot,
            empty_state_factory=empty_state_factory,
        )


def _establish_local_tenant_fenced(
    backend: FileWALLedgerBackend,
    tenant_id: str,
    load_legacy: Callable[[], dict[str, Any]],
    repair_snapshot: Callable[[dict[str, Any]], None],
    *,
    empty_state_factory: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not backend._fsync_enabled:
        raise RuntimeError("Tenant establishment requires fsync")
    tenant_hash = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
    evidence = backend._path.with_name(backend._path.name + ".tenant-" + tenant_hash)
    operation_id = "migration-" + tenant_hash
    lock = FileLockMutex(backend._path.with_name(backend._path.name + ".lock"))
    with backend._lock, lock.locked():
        if not backend._path.exists():
            raise WALCorruptionError("Execution WAL is missing during tenant establishment")
        events = backend._load_unlocked()
        ledger = ExecutionLedger()
        ledger._backend._events = deepcopy(events)
        ledger.load_from_backend()
        current = ledger.replay_tenant_memory(tenant_id)
        # A shared JSON mirror cannot establish ownership for another tenant.
        other_tenant_established = any(
            event.get("type") == "TENANT_MEMORY_COMMITTED"
            and str(dict(event.get("payload") or {}).get("tenant_id") or "") != tenant_id
            for event in events
        )
        if other_tenant_established:
            if empty_state_factory is None:
                raise WALCorruptionError(
                    "A fresh tenant cannot be established from a shared legacy memory snapshot"
                )
            empty_state = validate_tenant_memory_state(empty_state_factory())
        else:
            empty_state = None
        marker = None
        if evidence.exists():
            try:
                marker = json.loads(evidence.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise WALCorruptionError("Invalid tenant establishment evidence") from exc
            if (
                not isinstance(marker, dict)
                or marker.get("tenant_id") != tenant_id
                or marker.get("operation_id") != operation_id
                or marker.get("status") not in {"pending", "established"}
            ):
                raise WALCorruptionError("Invalid tenant establishment evidence")
        if marker and marker["status"] == "established" and current.revision == 0:
            raise WALCorruptionError("Established tenant history is missing")
        if marker and marker["status"] == "established":
            first = next(iter(current.operations.values()))["event"]
            if first.get("event_sha256") != marker.get("initial_event_sha256"):
                raise WALCorruptionError("Established tenant initial event does not match evidence")
        if current.revision == 0:
            if marker is None:
                candidate = empty_state if other_tenant_established else validate_tenant_memory_state(load_legacy())
                marker = {
                    "tenant_id": tenant_id,
                    "operation_id": operation_id,
                    "status": "pending",
                    "state": candidate,
                }
                _write_evidence(evidence, marker)
            candidate = validate_tenant_memory_state(marker.get("state"))
            event = {
                "type": "TENANT_MEMORY_COMMITTED",
                "session_id": "",
                "trace_id": "",
                "kernel_step_id": "tenant-memory",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "committed": True,
                "payload": {
                    "tenant_id": tenant_id,
                    "operation_id": operation_id,
                    "expected_revision": 0,
                    "revision": 1,
                    "state": candidate,
                },
            }
            ledger._seal_payload_hash_chain(event, ledger.get_next_sequence())
            _validate_append_head(events, event)
            with backend._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            events.append(event)
            current = replay_tenant_memory(events, tenant_id)
        if marker and marker["status"] == "pending":
            initial = current.operations.get(operation_id)
            if initial is None or initial["state"] != marker.get("state"):
                raise WALCorruptionError("Pending tenant establishment does not match the WAL")
        if marker is None or marker["status"] != "established":
            first = next(iter(current.operations.values()))["event"]
            _write_evidence(evidence, {
                "tenant_id": tenant_id,
                "operation_id": operation_id,
                "status": "established",
                "initial_event_sha256": first["event_sha256"],
            })
        if current.state is None:
            raise WALCorruptionError("Tenant memory projection is missing")
        repair_snapshot(deepcopy(current.state))
        return deepcopy(current.state)


def commit_local_tenant(
    backend: FileWALLedgerBackend,
    tenant_id: str,
    expected_revision: int,
    operation_id: str,
    state: dict[str, Any],
    publish: Callable[[], None],
    repair_snapshot: Callable[[dict[str, Any]], None],
    *,
    execution_ledger: ExecutionLedger | None = None,
) -> int:
    """Commit a candidate and repair its snapshot while holding the WAL writer lock."""
    from dadbot_system.delegation import tenant_mutation_fence

    if execution_ledger is not None and execution_ledger._backend is not backend:
        raise ValueError("The shared execution ledger must own the tenant WAL backend")
    with tenant_mutation_fence(tenant_id, required=False), (
        execution_ledger._lock if execution_ledger is not None else nullcontext()
    ):
        return _commit_local_tenant_fenced(
            backend, tenant_id, expected_revision, operation_id, state, publish, repair_snapshot,
            execution_ledger=execution_ledger,
        )


def _verified_local_commit_head(
    backend: FileWALLedgerBackend, execution_ledger: ExecutionLedger | None,
) -> tuple[list[dict[str, Any]], bytes, ExecutionLedger, bool, bool]:
    from dadbot_system.delegation import _execution_authority

    backend._authorize_event({})
    backend._read_binding()
    reuse = type(backend) is FileWALLedgerBackend and _execution_authority.get() is None
    events, prefix_bytes, unchanged = backend._append_verifier.load_prefix(
        backend._path, backend._load_unlocked, reuse=reuse,
    )
    shared_matches = (
        reuse and execution_ledger is not None and execution_ledger._backend is backend
        and not execution_ledger._replay_filters and not execution_ledger._backend_uncertain
        and events == execution_ledger._events
    )
    if shared_matches and unchanged and execution_ledger is not None:
        ledger = execution_ledger
        ledger.validate_cache()
    else:
        ledger = ExecutionLedger()
        ledger._backend._events = events
        ledger.load_from_backend()
    return events, prefix_bytes, ledger, shared_matches, reuse


def _require_established_tenant(
    backend: FileWALLedgerBackend, tenant_id: str, current: TenantMemoryRevision,
) -> None:
    tenant_hash = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
    evidence = backend._path.with_name(backend._path.name + ".tenant-" + tenant_hash)
    try:
        marker = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WALCorruptionError("Tenant establishment evidence is missing or invalid") from exc
    first = next(iter(current.operations.values()), {}).get("event")
    if (
        not isinstance(marker, dict)
        or marker.get("tenant_id") != tenant_id
        or marker.get("status") != "established"
        or marker.get("operation_id") != "migration-" + tenant_hash
        or not isinstance(first, dict)
        or marker.get("initial_event_sha256") != first.get("event_sha256")
    ):
        raise WALCorruptionError("Tenant establishment evidence does not match the WAL")


def _commit_local_tenant_fenced(
    backend: FileWALLedgerBackend,
    tenant_id: str,
    expected_revision: int,
    operation_id: str,
    state: dict[str, Any],
    publish: Callable[[], None],
    repair_snapshot: Callable[[dict[str, Any]], None],
    *,
    execution_ledger: ExecutionLedger | None = None,
) -> int:
    candidate = validate_tenant_memory_state(state)
    if not backend._fsync_enabled:
        raise RuntimeError("Tenant commit requires fsync")
    lock = FileLockMutex(backend._path.with_name(backend._path.name + ".lock"))
    with backend._lock, lock.locked():
        if not backend._path.exists():
            raise WALCorruptionError("Execution WAL is missing during tenant commit")
        events, prefix_bytes, ledger, shared_matches, reuse = _verified_local_commit_head(backend, execution_ledger)
        current = ledger.replay_tenant_memory(tenant_id)
        _require_established_tenant(backend, tenant_id, current)
        prior = current.operations.get(operation_id)
        if prior is not None:
            if prior["expected_revision"] != expected_revision or prior["state"] != candidate:
                raise TenantMemoryOperationConflict("Tenant operation ID belongs to another mutation")
            if current.revision != prior["event"]["payload"]["revision"]:
                raise TenantMemoryRevisionConflict("Operation committed before a newer tenant revision")
            if current.state is None:
                raise WALCorruptionError("Committed tenant memory projection is missing")
            publish()
            repair_snapshot(deepcopy(current.state))
            return int(prior["event"]["payload"]["revision"])
        if current.revision != expected_revision or current.revision == 0:
            raise TenantMemoryRevisionConflict(
                f"Tenant memory revision changed: expected {expected_revision}, current {current.revision}",
            )
        event = {
            "type": "TENANT_MEMORY_COMMITTED",
            "session_id": "",
            "trace_id": "",
            "kernel_step_id": "tenant-memory",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "committed": True,
            "payload": {
                "tenant_id": tenant_id,
                "operation_id": operation_id,
                "expected_revision": expected_revision,
                "revision": expected_revision + 1,
                "state": candidate,
            },
        }
        ledger._seal_payload_hash_chain(event, ledger.get_next_sequence())
        _validate_append_head(events, event, verified_prefix=events == ledger._events)
        if execution_ledger is not None:
            execution_ledger._backend_content_digest = None
            execution_ledger._backend_uncertain = True
        line = json.dumps(event, default=str) + "\n"
        with backend._path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        publish()
        repair_snapshot(deepcopy(candidate))
        digest = backend._append_verifier.record(prefix_bytes, line, events, event, reuse=reuse)
        if shared_matches and execution_ledger is not None:
            execution_ledger._record_committed_event(deepcopy(event), ("", ""))
            execution_ledger._backend_content_digest = digest
        return expected_revision + 1
