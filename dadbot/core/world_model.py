from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


class AsyncWorldModelPersistence(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str) -> None: ...


class InMemoryAsyncWorldModelPersistence:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._kv.get(str(key))

    async def set(self, key: str, value: str) -> None:
        self._kv[str(key)] = str(value)


class SQLiteAsyncWorldModelPersistence:
    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dadbot_world_model_kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """,
            )
            conn.commit()
        finally:
            conn.close()

    def _get_sync(self, key: str) -> str | None:
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT value FROM dadbot_world_model_kv WHERE key = ?",
                (str(key),),
            ).fetchone()
            return None if row is None else str(row[0])
        finally:
            conn.close()

    def _set_sync(self, key: str, value: str) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                INSERT INTO dadbot_world_model_kv(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(key), str(value)),
            )
            conn.commit()
        finally:
            conn.close()

    async def get(self, key: str) -> str | None:
        return await asyncio.to_thread(self._get_sync, str(key))

    async def set(self, key: str, value: str) -> None:
        await asyncio.to_thread(self._set_sync, str(key), str(value))


class UserWorldModel(BaseModel):
    session_id: str
    policy_version: str = ""
    prompt_context: str = ""
    trust_level: int = 0
    openness_level: int = 0
    emotional_momentum: str = ""
    key_facts: list[str] = Field(default_factory=list)
    active_goals: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    family_map: dict[str, str] = Field(default_factory=dict)
    emotional_timeline: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: str


class WorldModelStore:
    KEY_PREFIX = "dadbot:world_model"

    def __init__(self, persistence: AsyncWorldModelPersistence) -> None:
        self.persistence = persistence

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @classmethod
    def _session_key(cls, session_id: str) -> str:
        normalized = str(session_id or "default").strip() or "default"
        return f"{cls.KEY_PREFIX}:{normalized}"

    @staticmethod
    def _merge_unique_strings(existing: list[str], incoming: list[Any]) -> list[str]:
        out = [str(item).strip() for item in list(existing or []) if str(item).strip()]
        seen = {item.lower() for item in out}
        for raw in list(incoming or []):
            candidate = str(raw).strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        return out

    async def get_model(self, session_id: str) -> UserWorldModel | None:
        raw = await self.persistence.get(self._session_key(session_id))
        if not raw:
            return None
        try:
            return UserWorldModel.model_validate_json(raw)
        except (TypeError, ValueError):
            return None

    async def evolve_from_turn(self, snapshot: dict[str, Any]) -> UserWorldModel:
        base_snapshot = dict(snapshot or {})
        session_id = str(base_snapshot.get("session_id") or "default").strip() or "default"

        existing = await self.get_model(session_id)
        if existing is None:
            model = UserWorldModel(session_id=session_id, updated_at=self._now_iso())
        else:
            model = existing.model_copy(deep=True)

        policy_version = str(base_snapshot.get("policy_version") or "").strip()
        if policy_version:
            model.policy_version = policy_version

        prompt_context = str(base_snapshot.get("prompt_context") or "").strip()
        if prompt_context:
            model.prompt_context = prompt_context[:240]

        model.trust_level = int(base_snapshot.get("trust_level", model.trust_level) or 0)
        model.openness_level = int(base_snapshot.get("openness_level", model.openness_level) or 0)
        momentum = str(base_snapshot.get("emotional_momentum") or "").strip()
        if momentum:
            model.emotional_momentum = momentum

        model.key_facts = self._merge_unique_strings(model.key_facts, list(base_snapshot.get("key_facts") or []))
        model.active_goals = self._merge_unique_strings(model.active_goals, list(base_snapshot.get("active_goals") or []))
        model.contradictions = self._merge_unique_strings(
            model.contradictions,
            list(base_snapshot.get("contradictions") or []),
        )

        incoming_family_map = dict(base_snapshot.get("family_map") or {})
        if incoming_family_map:
            merged = dict(model.family_map or {})
            for key, value in incoming_family_map.items():
                k = str(key).strip()
                v = str(value).strip()
                if k and v:
                    merged[k] = v
            model.family_map = merged

        timeline_point = {
            "timestamp": self._now_iso(),
            "trust_level": model.trust_level,
            "openness_level": model.openness_level,
            "emotional_momentum": model.emotional_momentum,
            "policy_version": model.policy_version,
        }
        timeline = list(model.emotional_timeline or [])
        timeline.append(timeline_point)
        model.emotional_timeline = timeline[-200:]
        model.updated_at = timeline_point["timestamp"]

        await self.persistence.set(self._session_key(session_id), model.model_dump_json())
        return model
