from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


LIFECYCLE_EVENT_SCHEMA_VERSION = "execution-lifecycle.v1"


class LifecycleEventType(StrEnum):
    SUBMITTED = "Submitted"
    CLAIMED = "Claimed"
    LEASE_RENEWED = "LeaseRenewed"
    LEASE_EXPIRED = "LeaseExpired"
    RELEASED = "Released"
    COMPLETED = "Completed"
    FAILED = "Failed"
    REDELIVERED = "Redelivered"


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError("occurred_at must be non-empty")
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _require_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    execution_id: str
    occurred_at: datetime
    event_type: LifecycleEventType

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": LIFECYCLE_EVENT_SCHEMA_VERSION,
            "event_type": self.event_type.value,
            "execution_id": self.execution_id,
            "occurred_at": self.occurred_at.isoformat(),
        }
        payload.update(self._event_fields())
        return payload

    def _event_fields(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True, slots=True)
class Submitted(LifecycleEvent):
    def __init__(self, *, execution_id: str, occurred_at: datetime):
        object.__setattr__(self, "execution_id", _require_text(execution_id, field_name="execution_id"))
        object.__setattr__(self, "occurred_at", _coerce_datetime(occurred_at))
        object.__setattr__(self, "event_type", LifecycleEventType.SUBMITTED)


@dataclass(frozen=True, slots=True)
class Claimed(LifecycleEvent):
    worker_id: str
    lease_expiry: datetime

    def __init__(self, *, execution_id: str, occurred_at: datetime, worker_id: str, lease_expiry: datetime):
        object.__setattr__(self, "execution_id", _require_text(execution_id, field_name="execution_id"))
        object.__setattr__(self, "occurred_at", _coerce_datetime(occurred_at))
        object.__setattr__(self, "event_type", LifecycleEventType.CLAIMED)
        object.__setattr__(self, "worker_id", _require_text(worker_id, field_name="worker_id"))
        object.__setattr__(self, "lease_expiry", _coerce_datetime(lease_expiry))

    def _event_fields(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "lease_expiry": self.lease_expiry.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class LeaseRenewed(LifecycleEvent):
    worker_id: str
    lease_expiry: datetime

    def __init__(self, *, execution_id: str, occurred_at: datetime, worker_id: str, lease_expiry: datetime):
        object.__setattr__(self, "execution_id", _require_text(execution_id, field_name="execution_id"))
        object.__setattr__(self, "occurred_at", _coerce_datetime(occurred_at))
        object.__setattr__(self, "event_type", LifecycleEventType.LEASE_RENEWED)
        object.__setattr__(self, "worker_id", _require_text(worker_id, field_name="worker_id"))
        object.__setattr__(self, "lease_expiry", _coerce_datetime(lease_expiry))

    def _event_fields(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "lease_expiry": self.lease_expiry.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class LeaseExpired(LifecycleEvent):
    worker_id: str

    def __init__(self, *, execution_id: str, occurred_at: datetime, worker_id: str):
        object.__setattr__(self, "execution_id", _require_text(execution_id, field_name="execution_id"))
        object.__setattr__(self, "occurred_at", _coerce_datetime(occurred_at))
        object.__setattr__(self, "event_type", LifecycleEventType.LEASE_EXPIRED)
        object.__setattr__(self, "worker_id", _require_text(worker_id, field_name="worker_id"))

    def _event_fields(self) -> dict[str, Any]:
        return {"worker_id": self.worker_id}


@dataclass(frozen=True, slots=True)
class Released(LifecycleEvent):
    worker_id: str

    def __init__(self, *, execution_id: str, occurred_at: datetime, worker_id: str):
        object.__setattr__(self, "execution_id", _require_text(execution_id, field_name="execution_id"))
        object.__setattr__(self, "occurred_at", _coerce_datetime(occurred_at))
        object.__setattr__(self, "event_type", LifecycleEventType.RELEASED)
        object.__setattr__(self, "worker_id", _require_text(worker_id, field_name="worker_id"))

    def _event_fields(self) -> dict[str, Any]:
        return {"worker_id": self.worker_id}


@dataclass(frozen=True, slots=True)
class Completed(LifecycleEvent):
    result_ref: str

    def __init__(self, *, execution_id: str, occurred_at: datetime, result_ref: str):
        object.__setattr__(self, "execution_id", _require_text(execution_id, field_name="execution_id"))
        object.__setattr__(self, "occurred_at", _coerce_datetime(occurred_at))
        object.__setattr__(self, "event_type", LifecycleEventType.COMPLETED)
        object.__setattr__(self, "result_ref", _require_text(result_ref, field_name="result_ref"))

    def _event_fields(self) -> dict[str, Any]:
        return {"result_ref": self.result_ref}


@dataclass(frozen=True, slots=True)
class Failed(LifecycleEvent):
    error_ref: str

    def __init__(self, *, execution_id: str, occurred_at: datetime, error_ref: str):
        object.__setattr__(self, "execution_id", _require_text(execution_id, field_name="execution_id"))
        object.__setattr__(self, "occurred_at", _coerce_datetime(occurred_at))
        object.__setattr__(self, "event_type", LifecycleEventType.FAILED)
        object.__setattr__(self, "error_ref", _require_text(error_ref, field_name="error_ref"))

    def _event_fields(self) -> dict[str, Any]:
        return {"error_ref": self.error_ref}


@dataclass(frozen=True, slots=True)
class Redelivered(LifecycleEvent):
    previous_worker_id: str
    new_worker_id: str

    def __init__(self, *, execution_id: str, occurred_at: datetime, previous_worker_id: str, new_worker_id: str):
        object.__setattr__(self, "execution_id", _require_text(execution_id, field_name="execution_id"))
        object.__setattr__(self, "occurred_at", _coerce_datetime(occurred_at))
        object.__setattr__(self, "event_type", LifecycleEventType.REDELIVERED)
        object.__setattr__(self, "previous_worker_id", _require_text(previous_worker_id, field_name="previous_worker_id"))
        object.__setattr__(self, "new_worker_id", _require_text(new_worker_id, field_name="new_worker_id"))

    def _event_fields(self) -> dict[str, Any]:
        return {
            "previous_worker_id": self.previous_worker_id,
            "new_worker_id": self.new_worker_id,
        }


def lifecycle_event_from_payload(payload: dict[str, Any]) -> LifecycleEvent:
    body = dict(payload or {})
    if str(body.get("schema_version") or "").strip() != LIFECYCLE_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported lifecycle schema version")
    event_type = LifecycleEventType(_require_text(body.get("event_type"), field_name="event_type"))
    kwargs = {
        "execution_id": body.get("execution_id"),
        "occurred_at": body.get("occurred_at"),
    }
    if event_type is LifecycleEventType.SUBMITTED:
        return Submitted(**kwargs)
    if event_type is LifecycleEventType.CLAIMED:
        return Claimed(**kwargs, worker_id=body.get("worker_id"), lease_expiry=body.get("lease_expiry"))
    if event_type is LifecycleEventType.LEASE_RENEWED:
        return LeaseRenewed(**kwargs, worker_id=body.get("worker_id"), lease_expiry=body.get("lease_expiry"))
    if event_type is LifecycleEventType.LEASE_EXPIRED:
        return LeaseExpired(**kwargs, worker_id=body.get("worker_id"))
    if event_type is LifecycleEventType.RELEASED:
        return Released(**kwargs, worker_id=body.get("worker_id"))
    if event_type is LifecycleEventType.COMPLETED:
        return Completed(**kwargs, result_ref=body.get("result_ref"))
    if event_type is LifecycleEventType.FAILED:
        return Failed(**kwargs, error_ref=body.get("error_ref"))
    if event_type is LifecycleEventType.REDELIVERED:
        return Redelivered(
            **kwargs,
            previous_worker_id=body.get("previous_worker_id"),
            new_worker_id=body.get("new_worker_id"),
        )
    raise ValueError(f"unsupported lifecycle event type: {event_type!r}")


def lifecycle_event_from_ledger_event(event: dict[str, Any]) -> LifecycleEvent | None:
    if str((event or {}).get("type") or "") != "EXECUTION_LIFECYCLE":
        return None
    return lifecycle_event_from_payload(dict((event or {}).get("payload") or {}))
