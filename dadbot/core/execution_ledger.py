from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from typing import Any

from dadbot.core.canonical_event import (
    CANONICAL_EVENT_FIELDS,
    NON_REPLAY_EVENT_TYPES,
    _canonical_trace_payload,
    _chain_hash,
    _deterministic_event_id,
    _event_sha256,
    _verify_persisted_chain,
)
from dadbot.core.canonical_event import (
    _canonical_event_for_hash as _canonical_event_for_hash,  # noqa: PLC0414
)
from dadbot.core.event_schema import get_migrator, stamp_schema_version
from dadbot.core.ledger_backend import (
    CRCFileWALLedgerBackend,
    FileWALLedgerBackend,
    InMemoryLedgerBackend,
    SequenceValidator,
    _copy_ledger_value,
)

_TAIL_LIMIT = 256


def _event_digest_record(event: dict[str, Any]) -> dict[str, str | int]:
    return {
        "sequence": int(event.get("sequence") or 0),
        "type": str(event.get("type") or ""),
        "trace_id": str(event.get("trace_id") or ""),
        "session_id": str(event.get("session_id") or ""),
        "kernel_step_id": str(event.get("kernel_step_id") or ""),
        "payload_hash": hashlib.sha256(
            json.dumps(dict(event.get("payload") or {}), sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest(),
    }


class IntegrityBreachError(RuntimeError):
    """Raised when integrity checks require an immediate hard abort."""


class WriteBoundaryViolationError(RuntimeError):
    """Raised when strict ledger mode rejects a write outside the boundary guard."""


class WriteBoundaryGuard(AbstractContextManager["WriteBoundaryGuard"]):
    """Temporarily allow writes to a strict-mode ledger."""

    def __init__(self, ledger: ExecutionLedger) -> None:
        self._ledger = ledger

    def __enter__(self) -> WriteBoundaryGuard:
        self._ledger._write_guard_depth += 1
        return self

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> None:
        self._ledger._write_guard_depth = max(0, self._ledger._write_guard_depth - 1)


class ExecutionLedger:
    """In-memory append-only execution ledger with replay-hash support."""

    def __init__(
        self,
        backend: Any | None = None,
        *,
        strict_writes: bool = False,
    ) -> None:
        from dadbot_system.delegation import _execution_authority

        self._authenticated_authority_required = _execution_authority.get() is not None
        self._backend = backend or InMemoryLedgerBackend()
        self._backend_content_digest: bytes | None = None
        self._events: list[dict[str, Any]] = []
        self._cache: dict[str, Any] = {
            "sequence_counter": 0,
            "event_count": 0,
            "last_event_hash": None,
            "recent_tail": [],
            "trace_sequence_counters": {},
            "cache_rebuild_count": 0,
            "version": 0,
        }
        self._lock = threading.RLock()
        self._session_heads: dict[tuple[str, str], str] = {}
        self._session_indices: dict[tuple[str, str], int] = {}
        self._session_state_projections: dict[tuple[str, str], dict[str, Any]] = {}
        self._checkpoint_projections: dict[tuple[str, str], dict[str, Any]] = {}
        self._write_guard_depth = 0
        self._tenant_memory_commit_depth = 0
        self._strict_writes = bool(strict_writes)
        self._backend_uncertain = False
        self._replay_filters: list[Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = []

    @property
    def sealed_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.read())

    def add_replay_filter(
        self,
        filter_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> None:
        with self._lock:
            self._replay_filters.append(filter_fn)
            self._backend_content_digest = None

    def _ensure_write_allowed(self) -> None:
        if self._strict_writes and self._write_guard_depth <= 0:
            raise WriteBoundaryViolationError(
                "ExecutionLedger strict mode requires WriteBoundaryGuard",
            )

    def _ensure_backend_reconciled(self) -> None:
        if self._backend_uncertain:
            raise RuntimeError(
                "Execution ledger backend state is uncertain; reload and verify the backend before continuing",
            )

    def get_next_sequence(self) -> int:
        with self._lock:
            self._ensure_backend_reconciled()
            return int(self._cache.get("sequence_counter") or 0) + 1

    def get_next_trace_sequence(self, trace_token: str = "", **legacy_kwargs: Any) -> int:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        with self._lock:
            self._ensure_backend_reconciled()
            counters = dict(self._cache.get("trace_sequence_counters") or {})
            key = str(trace_token or legacy_trace or "").strip()
            return int(counters.get(key, 0)) + 1

    def event_count(self) -> int:
        with self._lock:
            self._ensure_backend_reconciled()
            return int(self._cache.get("event_count") or 0)

    def _rebuild_cache(self) -> None:
        trace_sequence_counters: dict[str, int] = {}
        for event in self._events:
            if str(event.get("type") or "") != "TURN_EVENT":
                continue
            payload = dict(event.get("payload") or {})
            trace_id = str(payload.get("trace_id") or "").strip()
            if not trace_id:
                continue
            seq = int(payload.get("sequence") or 0)
            if seq > int(trace_sequence_counters.get(trace_id, 0)):
                trace_sequence_counters[trace_id] = seq

        self._cache["sequence_counter"] = max(
            len(self._events),
            max((int(event.get("sequence") or 0) for event in self._events), default=0),
        )
        self._cache["event_count"] = len(self._events)
        self._cache["last_event_hash"] = (
            str(self._events[-1].get("chain_hash") or "") if self._events else None
        )
        self._cache["recent_tail"] = list(self._events[-_TAIL_LIMIT:])
        self._cache["trace_sequence_counters"] = trace_sequence_counters
        self._cache["cache_rebuild_count"] = int(self._cache.get("cache_rebuild_count") or 0) + 1
        self._cache["version"] = int(self._cache.get("version") or 0) + 1
        projections: dict[tuple[str, str], dict[str, Any]] = {}
        checkpoints: dict[tuple[str, str], dict[str, Any]] = {}
        for event in self._events:
            self._project_session_event(event, projections)
            self._project_checkpoint_event(event, checkpoints)
        self._session_state_projections = projections
        self._checkpoint_projections = checkpoints

    def telemetry_snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_backend_reconciled()
            self.validate_cache()
            return {
                "sequence_counter": int(self._cache.get("sequence_counter") or 0),
                "event_count": int(self._cache.get("event_count") or 0),
                "cache_rebuild_count": int(self._cache.get("cache_rebuild_count") or 0),
                "tail_size": len(self._cache.get("recent_tail") or []),
                "version": int(self._cache.get("version") or 0),
            }

    def validate_cache(self) -> None:
        if int(self._cache.get("event_count") or 0) != len(self._events):
            self._rebuild_cache()

    def _apply_session_lineage(self, payload: dict[str, Any]) -> tuple[tuple[str, str], str]:
        session_id = str(payload.get("session_id") or "")
        tenant_id = str(payload.get("tenant_id") or "")
        identity = (tenant_id, session_id)
        parent_event_id = str(payload.get("parent_event_id") or "")
        current_head = str(self._session_heads.get(identity) or "")
        if session_id:
            if parent_event_id:
                if parent_event_id != current_head:
                    raise RuntimeError(
                        f"causal chain violation: session_id={session_id!r} parent={parent_event_id!r} head={current_head!r}",
                    )
            else:
                payload["parent_event_id"] = current_head
            payload["session_index"] = int(self._session_indices.get(identity, 0)) + 1
        else:
            payload.setdefault("parent_event_id", "")
            payload.setdefault("session_index", 0)
        return identity, current_head

    def _seal_payload_hash_chain(self, payload: dict[str, Any], next_sequence: int) -> None:
        payload.setdefault("payload", {})
        payload["_seq"] = len(self._events)
        payload.setdefault("sequence", next_sequence)
        payload.setdefault("event_id", _deterministic_event_id(payload))
        stamp_schema_version(payload)

        event_sha256 = _event_sha256(payload)
        prev_chain_hash = str(self._events[-1].get("chain_hash") or "") if self._events else ""
        payload["event_sha256"] = event_sha256
        payload["prev_chain_hash"] = prev_chain_hash
        payload["chain_hash"] = _chain_hash(prev_chain_hash, event_sha256)

    def _update_session_heads(self, payload: dict[str, Any], identity: tuple[str, str]) -> None:
        session_id = identity[1]
        if not session_id:
            return
        event_id = str(payload.get("event_id") or "")
        self._session_heads[identity] = event_id
        self._session_indices[identity] = int(payload.get("session_index") or 0)

    def _update_trace_sequence_cache(self, payload: dict[str, Any], cache: dict[str, Any]) -> None:
        if str(payload.get("type") or "") != "TURN_EVENT":
            return
        turn_payload = dict(payload.get("payload") or {})
        trace_id = str(turn_payload.get("trace_id") or "").strip()
        if not trace_id:
            return
        counters = dict(cache.get("trace_sequence_counters") or {})
        counters[trace_id] = int(turn_payload.get("sequence") or 0)
        cache["trace_sequence_counters"] = counters

    def _update_recent_tail_cache(self, payload: dict[str, Any], cache: dict[str, Any]) -> None:
        tail = list(cache.get("recent_tail") or [])
        tail.append(payload)
        if len(tail) > _TAIL_LIMIT:
            tail = tail[-_TAIL_LIMIT:]
        cache["recent_tail"] = tail

    def _record_committed_event(
        self, payload: dict[str, Any], identity: tuple[str, str], *,
        session_projections: dict[tuple[str, str], dict[str, Any]] | None = None,
        checkpoint_projections: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        if session_projections is None:
            self._project_session_event(payload, self._session_state_projections)
        else:
            self._session_state_projections = session_projections
        if checkpoint_projections is None:
            self._project_checkpoint_event(payload, self._checkpoint_projections)
        else:
            self._checkpoint_projections = checkpoint_projections
        cache = self._cache
        self._update_session_heads(payload, identity)
        self._events.append(payload)
        cache["sequence_counter"] = int(payload.get("sequence") or int(cache.get("sequence_counter") or 0) + 1)
        cache["event_count"] = int(cache.get("event_count") or 0) + 1
        cache["last_event_hash"] = str(payload.get("chain_hash") or "")
        self._update_trace_sequence_cache(payload, cache)
        self._update_recent_tail_cache(payload, cache)
        cache["version"] = int(cache.get("version") or 0) + 1
        self.validate_cache()
        self._backend_uncertain = False

    @staticmethod
    def _project_session_event(
        event: dict[str, Any], projections: dict[tuple[str, str], dict[str, Any]],
    ) -> None:
        from dadbot.core.session_state_codec import apply_session_state_update

        event_type = event.get("type")
        if event_type not in {"SESSION_STATE_UPDATED", "SESSION_STATE_DELETED", "JOB_COMPLETED"}:
            return
        tenant = str(event.get("tenant_id") or "").strip()
        session = str(event.get("session_id") or "default").strip()
        payload = dict(event.get("payload") or {})
        owner = str(payload.get("tenant_id") or "").strip()
        if owner and owner != tenant:
            raise ValueError("ledger event tenant does not match its payload")
        identity = (tenant, session)
        if event_type == "SESSION_STATE_DELETED":
            projections.pop(identity, None)
        elif event_type == "SESSION_STATE_UPDATED":
            projections[identity] = apply_session_state_update(projections.get(identity) or {}, payload)
        else:
            state = dict(projections.get(identity) or {})
            state["last_result"] = _copy_ledger_value(payload.get("result"))
            projections[identity] = state

    @staticmethod
    def _project_checkpoint_event(
        event: dict[str, Any], projections: dict[tuple[str, str], dict[str, Any]],
    ) -> None:
        from dadbot.core.session_state_codec import apply_checkpoint_update

        if event.get("type") != "GRAPH_CHECKPOINT":
            return
        identity = (str(event.get("tenant_id") or "").strip(), str(event.get("session_id") or "default").strip())
        payload = dict(event.get("payload") or {})
        previous = dict(projections.get(identity, {}).get("payload") or {})
        projections[identity] = {
            "event_id": event.get("event_id"),
            "wire_payload_hash": _event_digest_record(event)["payload_hash"],
            "payload": apply_checkpoint_update(previous, payload),
        }

    def _verified_append_digest(self, payload: dict[str, Any], previous_digest: bytes | None) -> bytes | None:
        from dadbot_system.delegation import _execution_authority

        backend = self._backend
        if (
            type(self) is not ExecutionLedger
            or self._replay_filters
            or self._authenticated_authority_required
            or _execution_authority.get() is not None
        ):
            return None
        if type(backend) is InMemoryLedgerBackend:
            with backend._lock:
                snapshot = backend._append_snapshot
                if snapshot is not None:
                    digest, prefix_digest, count, record = snapshot
                    if (
                        count == len(self._events) + 1
                        and record == payload
                        and (
                            prefix_digest == previous_digest
                            or (not self._events and count == 1)
                        )
                    ):
                        return digest
            return None
        if type(backend) not in (FileWALLedgerBackend, CRCFileWALLedgerBackend):
            return None
        with backend._lock:
            snapshot = backend._append_verifier._snapshot
            if (
                snapshot is not None and len(snapshot[1]) == len(self._events) + 1
                and json.dumps(snapshot[1][-1], sort_keys=True, default=str)
                == json.dumps(payload, sort_keys=True, default=str)
            ):
                # Reuse an already bound runtime prefix, or establish exact record equality.
                if (
                    (previous_digest is not None and snapshot[2] == previous_digest)
                    or snapshot[1][:-1] == self._events
                ):
                    return snapshot[0]
        return None

    def write(self, event: dict[str, Any]) -> dict[str, Any]:
        from dadbot.core.ledger.enforcement import LedgerEnforcer

        self._ensure_write_allowed()
        LedgerEnforcer().validate(dict(event or {}))
        backend_lock = self._backend._lock if type(self._backend) is InMemoryLedgerBackend else nullcontext()
        with self._lock, backend_lock:
            self._ensure_backend_reconciled()
            payload = _copy_ledger_value(dict(event or {}))
            if payload.get("type") == "TENANT_MEMORY_COMMITTED" and not self._tenant_memory_commit_depth:
                raise WriteBoundaryViolationError("Tenant memory events must use commit_tenant_memory")
            cache = self._cache
            next_sequence = int(cache.get("sequence_counter") or 0) + 1
            identity, _ = self._apply_session_lineage(payload)
            if payload.get("type") == "GRAPH_CHECKPOINT" and self._history_references_allowed():
                from dadbot.core.session_state_codec import encode_session_state_update

                checkpoint = dict(payload.get("payload") or {})
                if not any(key in checkpoint for key in ("checkpoint_history_delta", "session_state_history_delta")):
                    previous = dict(self._checkpoint_projections.get(identity, {}).get("payload") or {})
                    roots = {"checkpoint", "session_state"}
                    encoded = encode_session_state_update(previous, {
                        key: value for key, value in checkpoint.items() if key in roots
                    })
                    checkpoint = {
                        key: value for key, value in checkpoint.items() if key not in roots
                    } | encoded["state"]
                    if "state_history_delta" in encoded:
                        for root, envelope in (
                            ("checkpoint", "checkpoint_history_delta"),
                            ("session_state", "session_state_history_delta"),
                        ):
                            fields = [
                                field for field in encoded["state_history_delta"]["fields"] if field["path"][0] == root
                            ]
                            if fields:
                                checkpoint[envelope] = {
                                    "version": encoded["state_history_delta"]["version"], "fields": fields,
                                }
                    payload["payload"] = checkpoint
            self._seal_payload_hash_chain(payload, next_sequence)
            session_projections = dict(self._session_state_projections)
            self._project_session_event(payload, session_projections)
            checkpoint_projections = dict(self._checkpoint_projections)
            self._project_checkpoint_event(payload, checkpoint_projections)
            previous_digest = self._backend_content_digest
            self._backend_content_digest = None
            self._backend_uncertain = True
            backend = self._backend
            owns_input = (
                type(backend) is InMemoryLedgerBackend
                and getattr(backend.append, "__func__", None) is InMemoryLedgerBackend.append
            )
            backend.append(
                payload if owns_input else _copy_ledger_value(payload),
                committed=bool(payload.get("committed", False)),
            )
            digest = self._verified_append_digest(payload, previous_digest)
            self._record_committed_event(
                payload, identity, session_projections=session_projections,
                checkpoint_projections=checkpoint_projections,
            )
            self._backend_content_digest = digest
            return _copy_ledger_value(payload)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        return self.write(event)

    def require_authenticated_authority(self) -> None:
        self._authenticated_authority_required = True

    def _authorize_tenant_memory(self, tenant_id: str) -> None:
        from dadbot_system.delegation import _execution_authority, require_tenant_authority

        if self._authenticated_authority_required or _execution_authority.get() is not None:
            require_tenant_authority(tenant_id)

    def replay_tenant_memory(self, tenant_id: str):
        from dadbot.core.tenant_memory_ledger import replay_tenant_memory

        self._authorize_tenant_memory(tenant_id)
        with self._lock:
            self._ensure_backend_reconciled()
            return replay_tenant_memory(self._events, tenant_id)

    def replay_tenant_memory_at_revision(self, tenant_id: str, revision: int):
        from dadbot.core.tenant_memory_ledger import replay_tenant_memory_at_revision

        self._authorize_tenant_memory(tenant_id)
        with self._lock:
            self._ensure_backend_reconciled()
            return replay_tenant_memory_at_revision(self._events, tenant_id, revision)

    def refresh_tenant_memory(self, tenant_id: str):
        """Reload a reconciled ledger before consulting a tenant projection."""
        self._authorize_tenant_memory(tenant_id)
        with self._lock:
            self._ensure_backend_reconciled()
            self.load_from_backend(only_if_changed=True)
            return self.replay_tenant_memory(tenant_id)

    def reconcile_tenant_memory_operation(
        self,
        *,
        tenant_id: str,
        expected_revision: int,
        operation_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Reload an uncertain ledger and find a previously submitted operation."""
        from dadbot.core.tenant_memory_ledger import (
            TenantMemoryOperationConflict,
            same_tenant_memory,
            validate_tenant_memory_state,
        )

        self._authorize_tenant_memory(tenant_id)
        if not isinstance(operation_id, str) or not operation_id or operation_id != operation_id.strip():
            raise ValueError("Operation ID must be a nonempty normalized string")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Expected revision must be a nonnegative integer")
        normalized_state = validate_tenant_memory_state(state)
        with self._lock:
            self.load_from_backend()
            current = self.replay_tenant_memory(tenant_id)
            previous = current.operations.get(operation_id)
            if previous is None:
                return None
            if previous["expected_revision"] != expected_revision or not same_tenant_memory(
                previous["state"], normalized_state
            ):
                raise TenantMemoryOperationConflict("Operation ID already belongs to a different mutation")
            return deepcopy(previous["event"])

    def commit_tenant_memory(
        self,
        *,
        tenant_id: str,
        expected_revision: int,
        operation_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        from dadbot_system.delegation import tenant_mutation_fence

        with tenant_mutation_fence(tenant_id, required=self._authenticated_authority_required):
            return self._commit_tenant_memory(
                tenant_id=tenant_id,
                expected_revision=expected_revision,
                operation_id=operation_id,
                state=state,
            )

    def _commit_tenant_memory(
        self,
        *,
        tenant_id: str,
        expected_revision: int,
        operation_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        from datetime import datetime, timezone

        from dadbot.core.ledger_backend import CRCFileWALLedgerBackend, FileWALLedgerBackend
        from dadbot.core.postgres_ledger_backend import PostgresLedgerBackend
        from dadbot.core.tenant_memory_ledger import (
            TENANT_MEMORY_EVENT,
            TenantMemoryOperationConflict,
            TenantMemoryRevisionConflict,
            same_tenant_memory,
            validate_tenant_memory_state,
        )

        self._authorize_tenant_memory(tenant_id)
        if not isinstance(operation_id, str) or not operation_id or operation_id != operation_id.strip():
            raise ValueError("Operation ID must be a nonempty normalized string")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Expected revision must be a nonnegative integer")
        if not isinstance(self._backend, (FileWALLedgerBackend, CRCFileWALLedgerBackend, PostgresLedgerBackend)):
            raise RuntimeError("Tenant memory requires a supported durable ledger backend")
        if isinstance(self._backend, (FileWALLedgerBackend, CRCFileWALLedgerBackend)):
            if not self._backend._fsync_enabled:
                raise RuntimeError("Tenant memory requires a ledger with fsync enabled")
        normalized_state = validate_tenant_memory_state(state)
        with self._lock:
            self._ensure_backend_reconciled()
            current = self.replay_tenant_memory(tenant_id)
            previous = current.operations.get(operation_id)
            if previous is not None:
                if previous["expected_revision"] != expected_revision or not same_tenant_memory(
                    previous["state"], normalized_state
                ):
                    raise TenantMemoryOperationConflict("Operation ID already belongs to a different mutation")
                return deepcopy(previous["event"])
            if current.revision != expected_revision:
                raise TenantMemoryRevisionConflict(
                    f"Tenant memory revision changed: expected {expected_revision}, current {current.revision}",
                )
            self._authorize_tenant_memory(tenant_id)
            self._tenant_memory_commit_depth += 1
            try:
                return self.write({
                    "type": TENANT_MEMORY_EVENT,
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
                        "state": normalized_state,
                    },
                })
            finally:
                self._tenant_memory_commit_depth -= 1

    def read_digest_records(self) -> list[dict[str, str | int]]:
        """Hash every payload under the ledger lock, returning only isolated scalar records."""
        with self._lock:
            self._ensure_backend_reconciled()
            self.validate_cache()
            return [_event_digest_record(event) for event in self._events]

    def session_state_delta(
        self, *, tenant_id: str, session_id: str, state_patch: dict[str, Any],
    ) -> dict[str, Any]:
        """Omit JSON-identical fields already projected from the verified local ledger."""
        from dadbot.core.tenant_session import resolve_tenant_id
        from dadbot_system.delegation import _execution_authority

        tenant = resolve_tenant_id(tenant_id)
        session = str(session_id or "").strip()
        if not session:
            raise ValueError("session_id is required")
        with self._lock:
            self._ensure_backend_reconciled()
            self.validate_cache()
            eligible = (
                type(self) is ExecutionLedger
                and isinstance(self._backend, (FileWALLedgerBackend, CRCFileWALLedgerBackend))
                and type(self._backend) in (FileWALLedgerBackend, CRCFileWALLedgerBackend)
                and self._backend_content_digest is not None
                and not any((
                    self._replay_filters, self._authenticated_authority_required,
                    _execution_authority.get() is not None,
                    (
                        isinstance(self._backend, FileWALLedgerBackend)
                        and self._backend._security_mode == "AUTHENTICATED"
                    ),
                    not self._backend._fsync_enabled,
                ))
            )
            if not eligible:
                return _copy_ledger_value(state_patch)
            current = self._session_state_fields(tenant, session, frozenset(state_patch))
            return {
                key: _copy_ledger_value(value) for key, value in state_patch.items()
                if key not in current or (
                    json.dumps(value, sort_keys=True, default=str)
                    != json.dumps(current[key], sort_keys=True, default=str)
                )
            }

    def _session_state_fields(
        self, tenant_id: str, session_id: str, fields: frozenset[str],
    ) -> dict[str, Any]:
        current = self._session_state_projections.get((tenant_id, session_id)) or {}
        return {key: value for key, value in current.items() if key in fields}

    def session_state_payload(
        self, *, tenant_id: str, session_id: str, state_patch: dict[str, Any],
    ) -> dict[str, Any]:
        from dadbot.core.session_state_codec import encode_session_state_update
        with self._lock:
            self._ensure_backend_reconciled()
            if not self._history_references_allowed():
                return {"state": state_patch}
            current = self._session_state_projections.get((tenant_id, session_id)) or {}
            return encode_session_state_update(current, state_patch)

    def _history_references_allowed(self) -> bool:
        from dadbot_system.delegation import _execution_authority

        backend = self._backend
        if (
                type(self) is not ExecutionLedger
                or type(backend) not in (InMemoryLedgerBackend, FileWALLedgerBackend, CRCFileWALLedgerBackend)
                or self._replay_filters or self._authenticated_authority_required
                or _execution_authority.get() is not None
                or self._backend_content_digest is None
                or (
                    type(backend) is not InMemoryLedgerBackend
                    and (
                        not backend._fsync_enabled
                        or (type(backend) is FileWALLedgerBackend and backend._security_mode == "AUTHENTICATED")
                    )
                )
        ):
            return False
        return type(backend) is not InMemoryLedgerBackend or (
            self._backend_content_digest == backend._snapshot_token_unlocked()
        )

    def read_checkpoints(self, *, payload_trace_id: str | None = None) -> list[dict[str, Any]]:
        """Return materialized checkpoint views, not replayable sealed event records."""
        from dadbot.core.session_state_codec import apply_session_state_update

        if getattr(self.read, "__func__", None) is not ExecutionLedger.read:
            return self.read()
        with self._lock:
            events = self.read(event_types=frozenset({"GRAPH_CHECKPOINT"}), payload_trace_id=payload_trace_id)
            cold: dict[str, dict[str, Any]] = {}
            for event in events:
                identity = (
                    str(event.get("tenant_id") or "").strip(), str(event.get("session_id") or "default").strip(),
                )
                latest = self._checkpoint_projections.get(identity) or {}
                if (
                    latest.get("event_id") == event.get("event_id")
                    and latest.get("wire_payload_hash") == _event_digest_record(event)["payload_hash"]
                ):
                    checkpoint_payload = latest["payload"]
                else:
                    if not cold:
                        projections: dict[tuple[str, str], dict[str, Any]] = {}
                        wanted = {str(item.get("event_id")) for item in events}
                        for item in self._events:
                            if item.get("type") == "GRAPH_CHECKPOINT":
                                self._project_checkpoint_event(item, projections)
                                key = (
                                    str(item.get("tenant_id") or "").strip(),
                                    str(item.get("session_id") or "default").strip(),
                                )
                                event_id = str(item.get("event_id"))
                                if event_id in wanted:
                                    cold[event_id] = projections[key]["payload"]
                                if len(cold) == len(wanted):
                                    break
                    checkpoint_payload = cold[str(event.get("event_id"))]
                event["payload"] = apply_session_state_update({}, {"state": checkpoint_payload})
                for key in ("chain_hash", "event_sha256", "prev_chain_hash"):
                    event.pop(key, None)
            return events

    def read(  # noqa: PLR0913
        self, full: bool = True, *,
        event_types: frozenset[str] | None = None, payload_types: frozenset[str] | None = None,
        trace_id: str | None = None, job_id: str | None = None, payload_trace_id: str | None = None,
        payload_fields: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return isolated events; supplied trace/job identities select their union."""
        with self._lock:
            self._ensure_backend_reconciled()
            self.validate_cache()
            events = self._events if full else list(self._cache.get("recent_tail") or [])
            if event_types is not None:
                events = [event for event in events if event.get("type") in event_types]
            if payload_trace_id is not None:
                events = [
                    event for event in events
                    if str(dict(event.get("payload") or {}).get("trace_id") or "").strip() == payload_trace_id
                ]
            if trace_id is not None or job_id is not None:
                events = [
                    event for event in events
                    if (
                        trace_id is not None and str(event.get("trace_id") or "").strip() == trace_id
                    ) or (
                        job_id is not None
                        and str(dict(event.get("payload") or {}).get("job_id") or "").strip() == job_id
                    )
                ]
            if payload_types is not None:
                events = [
                    event if event.get("type") in payload_types else {
                        key: value for key, value in event.items() if key != "payload"
                    } for event in events
                ]
            if payload_fields is not None:
                events = [
                    {
                        **event,
                        "payload": {
                            key: value for key, value in dict(event.get("payload") or {}).items()
                            if key in payload_fields
                        },
                    } for event in events
                ]
            return _copy_ledger_value(events)

    def replay_hash(self) -> str:
        events = [event for event in self.read() if str(event.get("type") or "") not in NON_REPLAY_EVENT_TYPES]
        canonical = [
            {field: event.get(field) for field in CANONICAL_EVENT_FIELDS}
            | {
                "payload": _canonical_trace_payload(
                    str(event.get("type") or ""),
                    event.get("payload"),
                ),
            }
            for event in sorted(events, key=lambda item: int(item.get("sequence") or 0))
        ]
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()

    def chain_hash(self) -> str:
        events = self.read()
        if not events:
            return ""
        return str(events[-1].get("chain_hash") or "")

    def verify_replay(self, *, mode: str = "basic") -> dict[str, Any]:
        events = list(self.read())
        violations: list[str] = []

        if str(mode or "basic").strip().lower() == "chain":
            prev = ""
            for idx, event in enumerate(events):
                expected_event_sha = _event_sha256(event)
                expected_prev = prev
                expected_chain = _chain_hash(expected_prev, expected_event_sha)

                actual_event_sha = str(event.get("event_sha256") or "")
                actual_prev = str(event.get("prev_chain_hash") or "")
                actual_chain = str(event.get("chain_hash") or "")

                if actual_event_sha != expected_event_sha:
                    violations.append(f"event[{idx}].event_sha256_mismatch")
                if actual_prev != expected_prev:
                    violations.append(f"event[{idx}].prev_chain_hash_mismatch")
                if actual_chain != expected_chain:
                    violations.append(f"event[{idx}].chain_hash_mismatch")
                prev = expected_chain

        return {
            "ok": len(violations) == 0,
            "mode": str(mode or "basic"),
            "event_count": len(events),
            "replay_hash": self.replay_hash(),
            "chain_hash": self.chain_hash(),
            "violations": violations,
        }

    @staticmethod
    def _has_persisted_chain(events: list[dict[str, Any]]) -> bool:
        return any(
            str(event.get("chain_hash") or "")
            or str(event.get("event_sha256") or "")
            or str(event.get("prev_chain_hash") or "")
            for event in events
        )

    def _apply_replay_filters(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered = list(events)
        for replay_filter in self._replay_filters:
            filtered = list(replay_filter(list(filtered)))
        return filtered

    def _rehydrate_loaded_events(self, events: list[dict[str, Any]], *, preserve_seals: bool = False) -> None:
        self._events = deepcopy(events)
        self._session_heads.clear()
        self._session_indices.clear()
        prev_chain = ""
        for event in self._events:
            if not str(event.get("event_id") or ""):
                event["event_id"] = _deterministic_event_id(event)
            if not preserve_seals:
                event_sha256 = _event_sha256(event)
                event["event_sha256"] = event_sha256
                event["prev_chain_hash"] = prev_chain
                prev_chain = _chain_hash(prev_chain, event_sha256)
                event["chain_hash"] = prev_chain
            session_id = str(event.get("session_id") or "")
            tenant_id = str(event.get("tenant_id") or "")
            identity = (tenant_id, session_id)
            event_id = str(event.get("event_id") or "")
            if session_id and event_id:
                self._session_heads[identity] = event_id
                self._session_indices[identity] = int(
                    event.get("session_index") or 0,
                )

    def load_from_backend(self, *, only_if_changed: bool = False) -> int:
        from dadbot_system.delegation import _execution_authority

        with self._lock:
            if only_if_changed:
                self._ensure_backend_reconciled()
            self._backend_uncertain = True
            digest = None
            if (
                only_if_changed and not self._replay_filters
                and (
                    type(self._backend) in (FileWALLedgerBackend, CRCFileWALLedgerBackend)
                    or (
                        type(self) is ExecutionLedger and type(self._backend) is InMemoryLedgerBackend
                        and not self._authenticated_authority_required
                        and _execution_authority.get() is None
                    )
                )
            ):
                digest, loaded = self._backend.load_if_changed(self._backend_content_digest)
                if loaded is None:
                    self._backend_uncertain = False
                    return len(self._events)
            else:
                loaded = self._backend.load()
            self._backend_content_digest = None
            events = list(get_migrator().migrate_all(list(loaded)))
            from dadbot.core.tenant_memory_ledger import (
                replay_all_tenant_memory,
                validate_tenant_event_seal,
            )

            for event in events:
                validate_tenant_event_seal(event)
            if self._has_persisted_chain(events):
                violations = _verify_persisted_chain(events)
                if violations:
                    raise RuntimeError(
                        "Sovereign checksum chain verification failed while loading ledger: "
                        + ", ".join(violations),
                    )
            report = SequenceValidator.validate(events)
            if not bool(report.get("ok")):
                raise RuntimeError(
                    f"Execution ledger sequence anomaly: {report.get('violations') or []}",
                )
            tenant_events = deepcopy([event for event in events if event.get("type") == "TENANT_MEMORY_COMMITTED"])
            events = self._apply_replay_filters(events)
            if tenant_events != [event for event in events if event.get("type") == "TENANT_MEMORY_COMMITTED"]:
                raise RuntimeError("Replay filters cannot modify tenant memory events")
            replay_all_tenant_memory(events)
            self._rehydrate_loaded_events(
                events,
                preserve_seals=not self._replay_filters and all(
                    event.get("event_id") and event.get("chain_hash")
                    and event.get("event_sha256") and "prev_chain_hash" in event
                    for event in events
                ),
            )
            self._rebuild_cache()
            self.validate_cache()
            self._backend_content_digest = digest
            self._backend_uncertain = False
            return len(self._events)

InMemoryExecutionLedger = ExecutionLedger
