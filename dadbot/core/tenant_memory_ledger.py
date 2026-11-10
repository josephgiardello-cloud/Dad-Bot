"""Validated tenant-memory events for the shared execution ledger."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


TENANT_MEMORY_EVENT = "TENANT_MEMORY_COMMITTED"


class TenantMemoryRevisionConflict(RuntimeError):
    """The candidate was based on a different committed tenant revision."""


class TenantMemoryOperationConflict(RuntimeError):
    """An operation ID was reused for a different mutation."""


class TenantMemoryLedgerCorruption(RuntimeError):
    """Tenant events cannot be safely replayed."""


def validate_tenant_event_seal(event: dict[str, Any]) -> None:
    if event.get("type") != TENANT_MEMORY_EVENT:
        return
    if (
        type(event.get("sequence")) is not int
        or event["sequence"] <= 0
        or not isinstance(event.get("event_id"), str)
        or not event["event_id"]
        or not isinstance(event.get("event_sha256"), str)
        or not event["event_sha256"]
        or not isinstance(event.get("prev_chain_hash"), str)
        or not isinstance(event.get("chain_hash"), str)
        or not event["chain_hash"]
    ):
        raise TenantMemoryLedgerCorruption("Tenant memory event must be sealed and sequenced")


@dataclass(frozen=True)
class TenantMemoryRevision:
    revision: int
    state: dict[str, Any] | None
    operations: dict[str, dict[str, Any]]


def _validate_json(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("Tenant memory must contain only finite JSON values and string object keys")


def validate_tenant_memory_state(state: Any) -> dict[str, Any]:
    if type(state) is not dict:
        raise ValueError("Tenant memory state must be an object")
    _validate_json(state)
    return deepcopy(state)


def replay_all_tenant_memory(events: list[dict[str, Any]]) -> dict[str, TenantMemoryRevision]:
    tenants: dict[str, TenantMemoryRevision] = {}
    for event in events:
        if event.get("type") != TENANT_MEMORY_EVENT:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise TenantMemoryLedgerCorruption("Invalid tenant memory event payload")
        event_tenant = payload.get("tenant_id")
        if not isinstance(event_tenant, str) or not event_tenant or event_tenant != event_tenant.strip():
            raise TenantMemoryLedgerCorruption("Invalid tenant ID in memory event")
        current = tenants.get(event_tenant, TenantMemoryRevision(0, None, {}))
        revision = current.revision
        operations = current.operations
        operation_id = payload.get("operation_id")
        expected = payload.get("expected_revision")
        result_revision = payload.get("revision")
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or operation_id != operation_id.strip()
            or type(expected) is not int
            or type(result_revision) is not int
            or expected != revision
            or result_revision != revision + 1
            or operation_id in operations
            or event.get("committed") is not True
        ):
            raise TenantMemoryLedgerCorruption("Invalid tenant memory revision or operation ID")
        try:
            state = validate_tenant_memory_state(payload.get("state"))
        except ValueError as exc:
            raise TenantMemoryLedgerCorruption("Invalid tenant memory event state") from exc
        operations[operation_id] = {
            "expected_revision": expected,
            "state": state,
            "event": deepcopy(event),
        }
        tenants[event_tenant] = TenantMemoryRevision(result_revision, state, operations)
    return tenants


def replay_tenant_memory(events: list[dict[str, Any]], tenant_id: str) -> TenantMemoryRevision:
    if not isinstance(tenant_id, str) or not tenant_id.strip() or tenant_id != tenant_id.strip():
        raise ValueError("Tenant ID must be a nonempty normalized string")
    return replay_all_tenant_memory(events).get(tenant_id, TenantMemoryRevision(0, None, {}))


def same_tenant_memory(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, allow_nan=False, separators=(",", ":")
    )
