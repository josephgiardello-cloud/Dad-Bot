from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, field_validator


class AsyncPolicyPersistence(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str) -> None: ...

    async def rpush(self, key: str, value: str) -> None: ...

    async def lrange(self, key: str, start: int, end: int) -> list[str]: ...


class InMemoryAsyncPolicyPersistence:
    """Small async persistence adapter for tests/local wiring."""

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._lists: dict[str, list[str]] = {}

    async def get(self, key: str) -> str | None:
        return self._kv.get(str(key))

    async def set(self, key: str, value: str) -> None:
        self._kv[str(key)] = str(value)

    async def rpush(self, key: str, value: str) -> None:
        bucket = self._lists.setdefault(str(key), [])
        bucket.append(str(value))

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = list(self._lists.get(str(key), []))
        if not values:
            return []

        normalized_start = max(int(start), 0)
        normalized_end = len(values) - 1 if int(end) < 0 else int(end)
        if normalized_start > normalized_end:
            return []
        return values[normalized_start : normalized_end + 1]


class DadPolicy(BaseModel):
    version: str
    persona_style: dict[str, Any]
    relationship_rules: dict[str, Any]
    safety_boundaries: dict[str, Any]
    memory_preferences: dict[str, Any]
    created_at: str
    created_by: str = "user"
    comment: str | None = None

    @field_validator("safety_boundaries")
    @classmethod
    def _validate_safety_boundaries(cls, value: dict[str, Any]) -> dict[str, Any]:
        boundaries = dict(value or {})
        harmful_keywords = boundaries.get("harmful_keywords")
        if harmful_keywords is None:
            return boundaries
        if not isinstance(harmful_keywords, (list, tuple, set)):
            raise ValueError("safety_boundaries.harmful_keywords must be a list of non-empty strings")

        normalized_keywords: list[str] = []
        for item in harmful_keywords:
            keyword = str(item).strip().lower()
            if not keyword:
                raise ValueError("safety_boundaries.harmful_keywords cannot contain empty terms")
            normalized_keywords.append(keyword)

        boundaries["harmful_keywords"] = normalized_keywords
        return boundaries


class DadPolicyStore:
    """Versioned policy store for persona/relationship/safety templates."""

    CURRENT_KEY = "dadbot:policy:current"
    HISTORY_KEY = "dadbot:policy:history"

    def __init__(self, persistence: AsyncPolicyPersistence) -> None:
        self.persistence = persistence

    async def get_current_policy(self) -> DadPolicy:
        raw = await self.persistence.get(self.CURRENT_KEY)
        if raw:
            return DadPolicy.model_validate_json(raw)
        return await self.seed_default_policy()

    async def seed_default_policy(self) -> DadPolicy:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        default = DadPolicy(
            version="dad_v1.0",
            persona_style={
                "warmth": 0.9,
                "humor": 0.7,
                "directness": 0.6,
                "encouragement": 0.85,
                "folksy_language": True,
            },
            relationship_rules={
                "build_trust": True,
                "emotional_support_first": True,
                "gentle_accountability": True,
                "avoid_topics": ["politics", "religion", "explicit_content"],
            },
            safety_boundaries={
                "block_harmful_advice": True,
                "protect_privacy": True,
                "age_appropriate": True,
            },
            memory_preferences={
                "prioritize_emotional_events": True,
                "keep_family_memories": True,
                "prune_old_smalltalk": True,
            },
            created_at=now,
            comment="Default warm Dad policy",
        )
        await self.save_policy(default, comment=default.comment)
        return default

    async def save_policy(self, policy: DadPolicy, *, comment: str | None = None) -> None:
        to_save = policy.model_copy(deep=True)
        if comment is not None:
            to_save.comment = str(comment)

        # Keep an append-only history of every saved policy revision.
        await self.persistence.rpush(self.HISTORY_KEY, to_save.model_dump_json())
        await self.persistence.set(self.CURRENT_KEY, to_save.model_dump_json())

    async def list_history(self) -> list[DadPolicy]:
        raw_items = await self.persistence.lrange(self.HISTORY_KEY, 0, -1)
        history: list[DadPolicy] = []
        for raw in raw_items:
            try:
                history.append(DadPolicy.model_validate_json(raw))
            except (TypeError, ValueError):
                continue
        return history

    async def rollback_to_version(self, version: str, *, comment: str | None = None) -> DadPolicy:
        target_version = str(version or "").strip()
        if not target_version:
            raise ValueError("version is required")

        history = await self.list_history()
        for item in reversed(history):
            if str(item.version) == target_version:
                candidate = item.model_copy(deep=True)
                if comment:
                    candidate.comment = str(comment)
                await self.save_policy(candidate, comment=candidate.comment)
                return candidate

        current = await self.get_current_policy()
        if str(current.version) == target_version:
            if comment:
                current.comment = str(comment)
                await self.save_policy(current, comment=current.comment)
            return current

        raise KeyError(f"policy version not found: {target_version}")

    async def policy_template(self) -> dict[str, Any]:
        policy = await self.get_current_policy()
        return {
            "version": policy.version,
            "persona_style": dict(policy.persona_style),
            "relationship_rules": dict(policy.relationship_rules),
            "safety_boundaries": dict(policy.safety_boundaries),
            "memory_preferences": dict(policy.memory_preferences),
        }
