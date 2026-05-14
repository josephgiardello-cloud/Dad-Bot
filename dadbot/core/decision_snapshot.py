from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(k): _freeze(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(v) for v in value), key=lambda item: repr(item)))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionSnapshot:
    """Frozen decision context with hash lock validation for scoring stages."""

    trace_id: str
    memory: Any
    context: Any
    metadata: Any
    snapshot_hash: str

    @classmethod
    def capture(
        cls,
        *,
        trace_id: str,
        memory: Any,
        context: Any,
        metadata: dict[str, Any] | None = None,
    ) -> "DecisionSnapshot":
        frozen_memory = _freeze(memory)
        frozen_context = _freeze(context)
        frozen_metadata = _freeze(metadata or {})
        payload = {
            "trace_id": str(trace_id or ""),
            "memory": _thaw(frozen_memory),
            "context": _thaw(frozen_context),
            "metadata": _thaw(frozen_metadata),
        }
        snapshot_hash = _stable_hash(payload)
        return cls(
            trace_id=str(trace_id or ""),
            memory=frozen_memory,
            context=frozen_context,
            metadata=frozen_metadata,
            snapshot_hash=snapshot_hash,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "memory": _thaw(self.memory),
            "context": _thaw(self.context),
            "metadata": _thaw(self.metadata),
        }

    def verify_hash_lock(self) -> bool:
        return _stable_hash(self.to_payload()) == self.snapshot_hash
