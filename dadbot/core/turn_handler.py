from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dadbot.contracts import AttachmentList, FinalizedTurnResult


def thin_turn_handler_enabled() -> bool:
    # Thin spine is canonical and always enabled.
    # Keep this helper for compatibility with legacy call sites/tests.
    return True


@dataclass(frozen=True)
class TurnContext:
    user_input: str
    session_id: str
    attachments: AttachmentList | None = None
    confluence_key: str | None = None
    metadata: Mapping[str, Any] | None = None
    timeout_seconds: float | None = None


class TurnHandler:
    """Thin, deterministic turn spine adapter.

    This wrapper does not decide policy or mutate global state; it only normalizes
    submit payload shape and delegates into the authoritative execution entrypoint.
    """

    def __init__(
        self,
        *,
        submit_turn: Callable[..., Awaitable[FinalizedTurnResult]],
        policy_store: Any | None = None,
        prompt_builder: Callable[[], str] | None = None,
        memory_ledger: Any | None = None,
        relationship_snapshotter: Callable[[], Mapping[str, Any] | None] | None = None,
        world_model_store: Any | None = None,
    ) -> None:
        self._submit_turn = submit_turn
        self._policy_store = policy_store
        self._prompt_builder = prompt_builder
        self._memory_ledger = memory_ledger
        self._relationship_snapshotter = relationship_snapshotter
        self._world_model_store = world_model_store

    @staticmethod
    def _normalize_goal(goal_text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(goal_text or "")).strip(" .,!?:;\t\n\r")
        if not normalized:
            return ""
        if len(normalized) > 96:
            normalized = normalized[:96].rstrip(" ")
        return normalized

    @classmethod
    def _extract_goal_candidates(cls, user_input: str) -> list[str]:
        text = str(user_input or "").strip()
        if not text:
            return []
        lowered = text.lower()
        patterns = [
            r"\bi\s+want\s+to\s+([^\.!?;]+)",
            r"\bi\s+need\s+to\s+([^\.!?;]+)",
            r"\bi\s+am\s+trying\s+to\s+([^\.!?;]+)",
            r"\bi\s+plan\s+to\s+([^\.!?;]+)",
            r"\bmy\s+goal\s+is\s+to\s+([^\.!?;]+)",
            r"\bhelp\s+me\s+([^\.!?;]+)",
            r"\bremind\s+me\s+to\s+([^\.!?;]+)",
        ]
        candidates: list[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, lowered, flags=re.IGNORECASE):
                for segment in re.split(r"\s+and\s+", str(match)):
                    goal = cls._normalize_goal(segment)
                    if goal:
                        candidates.append(goal)
        # Light fallback: treat short imperative requests as goals.
        if not candidates and lowered.startswith(("please ", "can you ", "could you ")):
            fallback = re.sub(r"^(please|can you|could you)\s+", "", lowered, flags=re.IGNORECASE)
            goal = cls._normalize_goal(fallback)
            if goal:
                candidates.append(goal)
        dedup: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(candidate)
        return dedup[:6]

    @classmethod
    def _extract_contradiction_signals(cls, user_input: str) -> list[str]:
        text = str(user_input or "").strip()
        if not text:
            return []
        lowered = text.lower()
        contradiction_markers = [
            "actually",
            "i changed my mind",
            "on second thought",
            "not anymore",
            "earlier i said",
            "i was wrong",
        ]
        if not any(marker in lowered for marker in contradiction_markers):
            return []
        sentence = re.split(r"[\.!?]", text, maxsplit=1)[0]
        contradiction = cls._normalize_goal(sentence)
        return [contradiction] if contradiction else []

    @classmethod
    def _extract_family_map(cls, user_input: str) -> dict[str, str]:
        text = str(user_input or "").strip()
        if not text:
            return {}
        lowered = text.lower()
        relation_words = [
            "mom",
            "mother",
            "dad",
            "father",
            "wife",
            "husband",
            "daughter",
            "son",
            "sister",
            "brother",
            "grandma",
            "grandpa",
            "partner",
        ]
        family_map: dict[str, str] = {}
        for relation in relation_words:
            pattern = rf"\bmy\s+{relation}\s+(?:is|was|seems|feels)\s+([^\.!?;]+)"
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if not match:
                continue
            descriptor = cls._normalize_goal(str(match.group(1) or ""))
            if descriptor:
                family_map[relation] = descriptor
        return family_map

    @classmethod
    def _extract_world_model_signals(cls, user_input: str) -> dict[str, Any]:
        return {
            "active_goals": cls._extract_goal_candidates(user_input),
            "contradictions": cls._extract_contradiction_signals(user_input),
            "family_map": cls._extract_family_map(user_input),
        }

    async def _inject_policy_metadata(self, metadata: dict[str, Any]) -> None:
        store = self._policy_store
        if store is None:
            return
        getter = getattr(store, "get_current_policy", None)
        if not callable(getter):
            return
        try:
            policy = await getter()
        except (RuntimeError, ValueError, TypeError):
            return

        if metadata.get("dad_policy_version"):
            return

        version = str(getattr(policy, "version", "") or "").strip()
        if version:
            metadata["dad_policy_version"] = version
            metadata.setdefault("policy_version", version)

        template = {
            "persona_style": dict(getattr(policy, "persona_style", {}) or {}),
            "relationship_rules": dict(getattr(policy, "relationship_rules", {}) or {}),
            "safety_boundaries": dict(getattr(policy, "safety_boundaries", {}) or {}),
            "memory_preferences": dict(getattr(policy, "memory_preferences", {}) or {}),
        }
        metadata.setdefault("dad_policy_template", template)

    async def _inject_prompt_context(self, metadata: dict[str, Any]) -> None:
        builder = self._prompt_builder
        if builder is None or metadata.get("prompt_context"):
            return
        try:
            prompt_context = builder()
            if inspect.isawaitable(prompt_context):
                prompt_context = await prompt_context
        except (RuntimeError, ValueError, TypeError):
            return

        normalized_context = str(prompt_context or "").strip()
        if not normalized_context:
            return

        metadata.setdefault("prompt_context", normalized_context)
        rich_context = dict(metadata.get("rich_context") or {})
        rich_context.setdefault("prompt_context", normalized_context)
        metadata["rich_context"] = rich_context

    async def _append_memory_ledger_event(
        self,
        *,
        ctx: TurnContext,
        metadata: Mapping[str, Any],
        result: FinalizedTurnResult,
    ) -> None:
        ledger = self._memory_ledger
        append_event = getattr(ledger, "append_memory_event", None)
        if not callable(append_event):
            return

        reply, should_end = result
        relationship_projection = dict(metadata.get("relationship_projection") or {})
        user_world_model = dict(metadata.get("user_world_model") or {})
        payload = {
            "event_type": "turn.finalized",
            "session_id": str(ctx.session_id or "default"),
            "confluence_key": str(ctx.confluence_key or ""),
            "user_input": str(ctx.user_input or ""),
            "reply_preview": str(reply or "")[:240],
            "should_end": bool(should_end),
            "policy_version": str(metadata.get("policy_version") or metadata.get("dad_policy_version") or ""),
            "prompt_context": str(metadata.get("prompt_context") or "")[:240],
            "relationship_trust_level": int(relationship_projection.get("trust_level", 0) or 0),
            "relationship_openness_level": int(relationship_projection.get("openness_level", 0) or 0),
            "relationship_emotional_momentum": str(
                relationship_projection.get("emotional_momentum") or "",
            ),
            "world_model_trust_level": int(user_world_model.get("trust_level", 0) or 0),
            "world_model_openness_level": int(user_world_model.get("openness_level", 0) or 0),
            "world_model_policy_version": str(user_world_model.get("policy_version") or ""),
        }
        try:
            write_result = append_event(payload)
            if inspect.isawaitable(write_result):
                await write_result
        except (RuntimeError, ValueError, TypeError):
            return

    async def _inject_relationship_projection(self, metadata: dict[str, Any]) -> None:
        if metadata.get("relationship_projection"):
            return
        snapshotter = self._relationship_snapshotter
        if snapshotter is None:
            return
        try:
            projection = snapshotter()
            if inspect.isawaitable(projection):
                projection = await projection
        except (RuntimeError, ValueError, TypeError):
            return
        projection_dict = dict(projection or {})
        if not projection_dict:
            return
        metadata["relationship_projection"] = projection_dict

    async def _inject_world_model_snapshot(self, metadata: dict[str, Any], ctx: TurnContext) -> None:
        if metadata.get("user_world_model"):
            return

        relationship_projection = dict(metadata.get("relationship_projection") or {})
        extracted = self._extract_world_model_signals(ctx.user_input)
        world_model = {
            "session_id": str(ctx.session_id or "default"),
            "policy_version": str(metadata.get("policy_version") or metadata.get("dad_policy_version") or ""),
            "prompt_context": str(metadata.get("prompt_context") or "")[:240],
            "trust_level": int(relationship_projection.get("trust_level", 0) or 0),
            "openness_level": int(relationship_projection.get("openness_level", 0) or 0),
            "emotional_momentum": str(relationship_projection.get("emotional_momentum") or ""),
            "active_goals": list(extracted.get("active_goals") or []),
            "contradictions": list(extracted.get("contradictions") or []),
            "family_map": dict(extracted.get("family_map") or {}),
        }
        metadata["user_world_model"] = world_model

    async def _evolve_persistent_world_model(self, metadata: dict[str, Any]) -> None:
        store = self._world_model_store
        evolve = getattr(store, "evolve_from_turn", None)
        if not callable(evolve):
            return
        snapshot = dict(metadata.get("user_world_model") or {})
        if not snapshot:
            return
        try:
            evolved = evolve(snapshot)
            if inspect.isawaitable(evolved):
                evolved = await evolved
        except (RuntimeError, ValueError, TypeError):
            return

        if isinstance(evolved, dict):
            metadata["user_world_model"] = dict(evolved)
            return

        dump = getattr(evolved, "model_dump", None)
        if callable(dump):
            try:
                metadata["user_world_model"] = dict(dump())
            except (RuntimeError, ValueError, TypeError):
                return

    async def process_turn(self, ctx: TurnContext) -> FinalizedTurnResult:
        outbound_metadata: dict[str, Any] = {
            "confluence_mode": "enforce",
            "confluence_key": str(ctx.confluence_key or "").strip(),
        }
        if ctx.metadata:
            outbound_metadata.update(dict(ctx.metadata))
        await self._inject_policy_metadata(outbound_metadata)
        await self._inject_prompt_context(outbound_metadata)

        result = await self._submit_turn(
            str(ctx.user_input or ""),
            attachments=ctx.attachments,
            session_id=str(ctx.session_id or "default"),
            confluence_key=str(ctx.confluence_key or ""),
            metadata=outbound_metadata,
            timeout_seconds=ctx.timeout_seconds,
        )
        await self._inject_relationship_projection(outbound_metadata)
        await self._inject_world_model_snapshot(outbound_metadata, ctx)
        await self._evolve_persistent_world_model(outbound_metadata)
        await self._append_memory_ledger_event(
            ctx=ctx,
            metadata=outbound_metadata,
            result=result,
        )
        return result
