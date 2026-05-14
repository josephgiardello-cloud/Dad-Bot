from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any


_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def _now() -> float:
    return float(time.time())


def _tokenize(text: str) -> set[str]:
    return {
        token.strip().lower()
        for token in str(text or "").replace("\n", " ").split(" ")
        if len(token.strip()) >= 3
    }


@dataclass(frozen=True)
class RetrievalItem:
    kind: str
    score: float
    payload: dict[str, Any]


class SemanticMemoryGraph:
    """Entity-relation semantic memory with episodic compression and decay."""

    def _extract_entities(self, text: str) -> list[str]:
        values = [match.group(1).strip() for match in _ENTITY_PATTERN.finditer(str(text or ""))]
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped[:16]

    def _relation_pairs(self, entities: list[str]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for idx, source in enumerate(entities):
            for target in entities[idx + 1 :]:
                pairs.append((source, target))
        return pairs[:32]

    def update_from_turn(
        self,
        *,
        state: dict[str, Any],
        session_id: str,
        trace_id: str,
        user_input: str,
        response_text: str,
    ) -> dict[str, Any]:
        memory = dict(state.get("semantic_memory_graph") or {})
        entities = dict(memory.get("entities") or {})
        relations = dict(memory.get("relations") or {})
        episodic = list(memory.get("episodic") or [])
        now = _now()

        combined = f"{str(user_input or '').strip()} {str(response_text or '').strip()}".strip()
        extracted = self._extract_entities(combined)

        for name in extracted:
            key = name.lower()
            current = dict(entities.get(key) or {})
            current["name"] = name
            current["mentions"] = int(current.get("mentions") or 0) + 1
            current["salience"] = float(min(1.0, float(current.get("salience") or 0.0) + 0.1))
            current["last_seen"] = now
            current["session_id"] = str(session_id or "default")
            entities[key] = current

        for source, target in self._relation_pairs(extracted):
            relation_key = f"{source.lower()}->{target.lower()}"
            current = dict(relations.get(relation_key) or {})
            current["source"] = source
            current["target"] = target
            current["weight"] = float(min(1.0, float(current.get("weight") or 0.0) + 0.08))
            current["last_seen"] = now
            current["type"] = str(current.get("type") or "cooccur")
            relations[relation_key] = current

        summary = str(user_input or "").strip()
        if len(summary) > 220:
            summary = summary[:220]
        if summary:
            episodic.append(
                {
                    "trace_id": str(trace_id or ""),
                    "session_id": str(session_id or "default"),
                    "summary": summary,
                    "response": str(response_text or "")[:220],
                    "importance": self._estimate_importance(user_input=user_input, response_text=response_text),
                    "created_at": now,
                },
            )

        memory["entities"] = entities
        memory["relations"] = relations
        memory["episodic"] = episodic[-512:]
        memory["updated_at"] = now
        state["semantic_memory_graph"] = memory
        return memory

    def apply_decay(self, *, state: dict[str, Any], half_life_seconds: float = 86_400.0) -> dict[str, Any]:
        memory = dict(state.get("semantic_memory_graph") or {})
        entities = dict(memory.get("entities") or {})
        relations = dict(memory.get("relations") or {})
        now = _now()
        hl = max(60.0, float(half_life_seconds or 86_400.0))

        for key, payload in entities.items():
            item = dict(payload)
            age = max(0.0, now - float(item.get("last_seen") or now))
            decay = math.exp(-math.log(2.0) * age / hl)
            item["salience"] = float(max(0.0, min(1.0, float(item.get("salience") or 0.0) * decay)))
            entities[key] = item

        for key, payload in relations.items():
            item = dict(payload)
            age = max(0.0, now - float(item.get("last_seen") or now))
            decay = math.exp(-math.log(2.0) * age / hl)
            item["weight"] = float(max(0.0, min(1.0, float(item.get("weight") or 0.0) * decay)))
            relations[key] = item

        memory["entities"] = entities
        memory["relations"] = relations
        memory["updated_at"] = now
        state["semantic_memory_graph"] = memory
        return memory

    def retrieval_plan(self, *, state: dict[str, Any], query: str, limit: int = 5) -> dict[str, Any]:
        memory = dict(state.get("semantic_memory_graph") or {})
        entities = dict(memory.get("entities") or {})
        relations = dict(memory.get("relations") or {})
        episodic = list(memory.get("episodic") or [])
        tokens = _tokenize(query)

        ranked: list[RetrievalItem] = []

        for payload in entities.values():
            item = dict(payload)
            name_tokens = _tokenize(str(item.get("name") or ""))
            overlap = len(tokens.intersection(name_tokens)) if tokens else 0
            score = (0.65 * float(item.get("salience") or 0.0)) + (0.35 * float(overlap > 0))
            if score > 0.0:
                ranked.append(RetrievalItem(kind="entity", score=float(score), payload=item))

        for payload in relations.values():
            item = dict(payload)
            rel_tokens = _tokenize(f"{item.get('source', '')} {item.get('target', '')}")
            overlap = len(tokens.intersection(rel_tokens)) if tokens else 0
            score = (0.6 * float(item.get("weight") or 0.0)) + (0.4 * float(overlap > 0))
            if score > 0.0:
                ranked.append(RetrievalItem(kind="relation", score=float(score), payload=item))

        for item in episodic[-256:]:
            episode = dict(item)
            overlap = len(tokens.intersection(_tokenize(str(episode.get("summary") or "")))) if tokens else 0
            score = (0.55 * float(episode.get("importance") or 0.0)) + (0.45 * float(overlap > 0))
            if score > 0.0:
                ranked.append(RetrievalItem(kind="episodic", score=float(score), payload=episode))

        ranked.sort(key=lambda entry: entry.score, reverse=True)
        selected = ranked[: max(1, int(limit))]
        return {
            "query": str(query or ""),
            "items": [
                {
                    "kind": item.kind,
                    "score": round(float(item.score), 6),
                    "payload": dict(item.payload),
                }
                for item in selected
            ],
            "strategy": "semantic_graph_weighted",
            "identity_continuity": self._identity_continuity(state=state, query=query),
        }

    def _estimate_importance(self, *, user_input: str, response_text: str) -> float:
        text = str(user_input or "").strip().lower()
        importance = 0.35
        if "my " in text or "i " in text:
            importance += 0.25
        if "?" not in text:
            importance += 0.15
        if len(str(response_text or "").strip()) > 80:
            importance += 0.1
        return float(max(0.0, min(1.0, importance)))

    def _identity_continuity(self, *, state: dict[str, Any], query: str) -> dict[str, Any]:
        profile = dict(state.get("identity_profile") or {})
        user_alias = str(profile.get("user_alias") or "default_user")
        continuity_score = 0.8 if user_alias != "default_user" else 0.5
        continuity_score += 0.1 if "my" in str(query or "").lower() else 0.0
        return {
            "user_alias": user_alias,
            "continuity_score": float(max(0.0, min(1.0, continuity_score))),
        }
