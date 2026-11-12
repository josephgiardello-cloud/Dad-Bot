"""Tenant-owned loop diagnostics, not canonical recovery or action replay authority."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dadbot.core.durability import FileLockMutex
from dadbot.runtime.agent_driver_loop import DriverLoopPolicy, DriverLoopResult, LoopTurnRecord, ReflectionHook
from dadbot_system.delegation import (
    live_authority_audit_context,
    require_tenant_authority_if_active,
    tenant_mutation_fence,
)
from dadbot_system.execution_identity import require_scoped_tenant_id
from dadbot_system.security import AuthorizationError

_DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent.parent.parent / "session_logs"
_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


def _snapshot_path(session_id: str, base_dir: Path | None = None, *, tenant_id: str) -> Path:
    tenant = require_scoped_tenant_id(tenant_id)
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A nonempty snapshot session ID is required")
    identity = json.dumps([tenant, session_id], ensure_ascii=True, separators=(",", ":"))
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return (Path(base_dir) if base_dir is not None else _DEFAULT_SNAPSHOT_DIR) / f"{key}.loop.json"


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate loop snapshot JSON field: {key}")
        result[key] = value
    return result


@dataclass
class LoopSessionSnapshot:
    """Local driver-reported progress; reconcile canonical receipts before resuming."""

    session_id: str
    initial_observation: str
    tenant_id: str = field(kw_only=True)
    last_committed_turn: int = 0
    completed_turns: int = 0
    failures: int = 0
    stop_reason: str | None = None
    started_at: str = ""
    finished_at: str = ""
    policy: dict[str, Any] = field(default_factory=lambda: asdict(DriverLoopPolicy()))

    def __post_init__(self) -> None:
        self._identity = (self.tenant_id, self.session_id)
        self._authority_required = live_authority_audit_context()["authority_mode"] != "trusted_local"
        self._validate()
        self._check_authority()

    def _check_authority(self) -> None:
        if (self.tenant_id, self.session_id) != self._identity:
            raise AuthorizationError("Loop snapshot identity changed")
        require_tenant_authority_if_active(self.tenant_id, required=self._authority_required)

    def _validate(self) -> None:
        _snapshot_path(self.session_id, tenant_id=self.tenant_id)
        for name in ("initial_observation", "started_at", "finished_at"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"Invalid loop snapshot {name}")
        if self.stop_reason is not None and (not isinstance(self.stop_reason, str) or not self.stop_reason.strip()):
            raise ValueError("Invalid loop snapshot stop_reason")
        for name in ("last_committed_turn", "completed_turns", "failures"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid loop snapshot {name}")
        if self.completed_turns > self.last_committed_turn or (
            (self.completed_turns == 0) != (self.last_committed_turn == 0)
        ):
            raise ValueError("Inconsistent loop snapshot progress")
        self._validate_policy()

    def _validate_policy(self) -> None:
        if not isinstance(self.policy, dict) or set(self.policy) != set(asdict(DriverLoopPolicy())):
            raise ValueError("Invalid loop snapshot policy")
        for name, minimum in (("max_turns", 1), ("max_failures", 0), ("max_consecutive_noop", 1)):
            value = self.policy[name]
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid loop snapshot policy {name}")
        tools = self.policy["allowed_tools"]
        if tools is not None and (
            not isinstance(tools, list) or any(not isinstance(tool, str) or not tool.strip() for tool in tools)
        ):
            raise ValueError("Invalid loop snapshot allowed_tools")
        trigger = self.policy["trigger_type"]
        if not isinstance(trigger, str) or not trigger.strip():
            raise ValueError("Invalid loop snapshot trigger_type")

    @property
    def is_incomplete(self) -> bool:
        return self.stop_reason is None

    def save(self, base_dir: Path | None = None, *, create_only: bool = False) -> Path:
        self._check_authority()
        self._validate()
        payload = {"schema_version": _SCHEMA_VERSION, **asdict(self)}
        serialized = json.dumps(payload, indent=2, allow_nan=False)
        path = _snapshot_path(self.session_id, base_dir, tenant_id=self.tenant_id)
        with tenant_mutation_fence(self.tenant_id, required=self._authority_required):
            with FileLockMutex(path.with_suffix(".json.lock")).locked():
                # Never replace corrupt or foreign source data with a fresh snapshot.
                existing = self.load(self.session_id, base_dir, tenant_id=self.tenant_id)
                if create_only and existing is not None:
                    raise FileExistsError("Reconcile and explicitly delete the existing loop snapshot before a new run")
                temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                try:
                    with temporary.open("w", encoding="utf-8") as handle:
                        handle.write(serialized)
                        handle.flush()
                        os.fsync(handle.fileno())
                    self._check_authority()
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, session_id: str, base_dir: Path | None = None, *, tenant_id: str) -> LoopSessionSnapshot | None:
        tenant = require_scoped_tenant_id(tenant_id)
        require_tenant_authority_if_active(tenant)
        path = _snapshot_path(session_id, base_dir, tenant_id=tenant)
        try:
            serialized = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            require_tenant_authority_if_active(tenant)
            legacy_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
            if len(legacy_name.encode("utf-8")) <= 240 and path.with_name(f"{legacy_name}.loop.json").exists():
                logger.warning("Unowned legacy loop snapshot excluded; canonical recovery reconciliation is required")
            return None
        data = json.loads(serialized, object_pairs_hook=_unique_json_object)
        expected = {item.name for item in fields(cls)} | {"schema_version"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Invalid loop snapshot schema")
        if type(data["schema_version"]) is not int or data.pop("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported loop snapshot schema version")
        if data["tenant_id"] != tenant or data["session_id"] != session_id:
            raise AuthorizationError("Loop snapshot tenant/session mismatch")
        snapshot = cls(**data)
        snapshot._check_authority()
        return snapshot

    @classmethod
    def delete(cls, session_id: str, base_dir: Path | None = None, *, tenant_id: str) -> bool:
        tenant = require_scoped_tenant_id(tenant_id)
        if cls.load(session_id, base_dir, tenant_id=tenant) is None:
            return False
        required = live_authority_audit_context()["authority_mode"] != "trusted_local"
        path = _snapshot_path(session_id, base_dir, tenant_id=tenant)
        with tenant_mutation_fence(tenant, required=required):
            with FileLockMutex(path.with_suffix(".json.lock")).locked():
                if cls.load(session_id, base_dir, tenant_id=tenant) is None:
                    return False
                require_tenant_authority_if_active(tenant, required=required)
                path.unlink()
                return True


class LoopSnapshotManager:
    """Persist prior dispatch records at Reflection and the final returned result.

    Reflection cannot acknowledge its pending action. An interrupted snapshot may
    therefore lag the last dispatch; it must never be used to skip or resend IO.
    """

    def __init__(
        self,
        session_id: str,
        initial_observation: str,
        policy: DriverLoopPolicy | None = None,
        *,
        tenant_id: str,
        base_dir: Path | None = None,
    ) -> None:
        self.session_id = session_id
        self.base_dir = base_dir
        self._snap = LoopSessionSnapshot(
            session_id=session_id,
            tenant_id=tenant_id,
            initial_observation=initial_observation,
            policy=asdict(policy if policy is not None else DriverLoopPolicy()),
            started_at=datetime.now(UTC).isoformat(),
        )
        self._snap.save(base_dir, create_only=True)

    def _record_progress(self, records: list[LoopTurnRecord]) -> None:
        last_index = 0
        committed = []
        failures = 0
        for record in records:
            if not isinstance(record, LoopTurnRecord) or record.turn_index <= last_index:
                raise ValueError("Invalid loop snapshot turn records")
            last_index = record.turn_index
            if record.commit_status == "committed":
                committed.append(record)
            elif record.commit_status in {"failed", "rejected"}:
                failures += 1
            elif record.commit_status != "skipped":
                raise ValueError("Invalid loop snapshot turn status")
        self._snap.last_committed_turn = committed[-1].turn_index if committed else 0
        self._snap.completed_turns = len(committed)
        self._snap.failures = failures

    def wrap_reflection(self, base_hook: ReflectionHook | None = None) -> ReflectionHook:
        def hook(ctx: dict[str, Any]) -> dict[str, Any]:
            if (ctx.get("tenant_id"), ctx.get("session_id")) != self._snap._identity:
                raise AuthorizationError("Loop snapshot dispatch identity mismatch")
            records = ctx.get("records")
            if not isinstance(records, list):
                raise ValueError("Loop snapshot requires prior dispatch records")
            self._record_progress(records)
            self._snap.save(self.base_dir)
            return dict(base_hook(ctx) or {}) if base_hook is not None else {}

        return hook

    def finalize(self, result: DriverLoopResult) -> None:
        if (result.tenant_id, result.session_id) != self._snap._identity:
            raise AuthorizationError("Loop snapshot result identity mismatch")
        self._record_progress(result.records)
        if result.completed_turns != self._snap.completed_turns or result.failures != self._snap.failures:
            raise ValueError("Loop result counters disagree with dispatch records")
        self._snap.stop_reason = result.stop_reason
        self._snap.finished_at = datetime.now(UTC).isoformat()
        self._snap.save(self.base_dir)


__all__ = ["LoopSessionSnapshot", "LoopSnapshotManager"]
