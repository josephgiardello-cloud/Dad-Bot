"""MemoryGraphManager â€” owns memory graph construction, projection, store sync, and retrieval.
Extracted from MemoryManager to thin the god class.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import TYPE_CHECKING

from dadbot.memory.graph_entity_resolver import GraphEntityResolver
from dadbot.utils import env_truthy
from dadbot_system.graph_compression import GraphPromptCompressor
from dadbot_system.graph_store import PostgresGraphStore, SQLiteGraphStore

if TYPE_CHECKING:
    from dadbot.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class MemoryGraphManager:
    """Owns graph projection, graph store persistence, and graph-based memory retrieval."""

    GRAPH_SOURCE_NODE_TYPES = {
        "archive_session",
        "consolidated_memory",
        "life_pattern",
        "persona_trait",
    }
    # Entity knowledge is owned by GraphEntityResolver; constants kept as
    # class-level aliases so existing code referencing them still works.
    GRAPH_ENTITY_ALIASES = GraphEntityResolver.ENTITY_ALIASES
    GRAPH_ENTITY_TYPES = GraphEntityResolver.ENTITY_TYPES
    GRAPH_RELATION_VOCAB = GraphEntityResolver.RELATION_VOCAB

    def __init__(self, bot, memory_manager: MemoryManager) -> None:
        self._bot = bot
        self._mm = memory_manager
        self._entity_resolver = GraphEntityResolver(bot)
        self._graph_store_backend = self._build_graph_store_backend()
        self._graph_prompt_compressor = GraphPromptCompressor(
            bot,
            max_tokens=bot.runtime_config.graph_context_token_budget,
            max_hops=bot.runtime_config.graph_walk_hops,
            max_edges=bot.runtime_config.graph_walk_edge_limit,
            max_nodes=bot.runtime_config.graph_walk_node_limit,
        )

    # ------------------------------------------------------------------
    # Backend construction
    # ------------------------------------------------------------------

    def _build_graph_store_backend(self):
        postgres_dsn = str(os.environ.get("DADBOT_POSTGRES_DSN") or "").strip()
        table_prefix = str(os.environ.get("DADBOT_GRAPH_TABLE_PREFIX") or "dadbot_graph").strip() or "dadbot_graph"
        if postgres_dsn:
            try:
                backend = PostgresGraphStore(postgres_dsn, table_prefix=table_prefix)
                backend.ensure_storage()
                return backend
            except Exception as exc:
                logger.warning(
                    "Postgres graph store unavailable, falling back to SQLite: %s",
                    exc,
                )
        return SQLiteGraphStore(self._bot, self._bot.GRAPH_STORE_DB_PATH)

    def ensure_graph_store(self):
        self._graph_store_backend.ensure_storage()

    def clear_graph_store(self):
        self._graph_store_backend.clear()

    # ------------------------------------------------------------------
    # Key / ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def graph_node_key(node_type, identifier):
        return f"{node_type}:{identifier}"

    @staticmethod
    def graph_edge_key(source_key, target_key, relation_type):
        return hashlib.sha1(
            f"{source_key}|{relation_type}|{target_key}".encode(),
        ).hexdigest()[:20]

    @staticmethod
    def graph_slug(text):
        return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")

    def graph_source_id(self, source_type, entry):
        if source_type == "archive_session":
            return str(
                entry.get("id")
                or hashlib.sha1(
                    f"{entry.get('summary', '')}|{entry.get('created_at', '')}".encode(),
                ).hexdigest()[:12],
            )
        if source_type == "persona_trait":
            return hashlib.sha1(
                f"{entry.get('trait', '')}|{entry.get('applied_at', '')}".encode(),
            ).hexdigest()[:12]
        if source_type == "life_pattern":
            return hashlib.sha1(
                f"{entry.get('summary', '')}|{entry.get('topic', '')}|{entry.get('day_hint', '')}|{entry.get('mood', '')}".encode(),
            ).hexdigest()[:12]
        return hashlib.sha1(
            f"{entry.get('summary', '')}|{entry.get('updated_at', '')}".encode(),
        ).hexdigest()[:12]

    def graph_keyword_tokens(self, text, limit=4):
        ignored = {
            "carry",
            "carries",
            "dad",
            "lately",
            "often",
            "really",
            "shared",
            "talk",
            "talking",
            "tony",
            "want",
            "wants",
            "with",
            "your",
            "been",
            "just",
            "more",
            "about",
            "again",
            "this",
            "that",
        }
        ordered = []
        seen = set()
        for token in re.findall(r"[a-z0-9']+", str(text or "").lower()):
            if token in seen or token in ignored:
                continue
            if token not in self._bot.significant_tokens(token):
                continue
            ordered.append(token)
            seen.add(token)
            if len(ordered) >= max(1, int(limit or 4)):
                break
        return ordered

    # ------------------------------------------------------------------
    # Entity / relation helpers
    # ------------------------------------------------------------------

    # --- Entity canonicalization (delegates to GraphEntityResolver) ---

    def canonical_graph_entity(self, label, semantic_type=None):
        return self._entity_resolver.canonical_entity(label, semantic_type)

    def normalize_relation_type(self, relation_type):
        return self._entity_resolver.normalize_relation_type(relation_type)

    def relation_type_for_entity(
        self,
        source_type,
        entity_type,
        text,
        fallback="mentions",
    ):
        return self._entity_resolver.relation_type_for_entity(
            source_type,
            entity_type,
            text,
            fallback,
        )

    # ------------------------------------------------------------------
    # Fact extraction
    # ------------------------------------------------------------------

    def extract_typed_graph_facts(
        self,
        text,
        source_type,
        *,
        default_category="general",
        default_mood="neutral",
        default_day="",
    ):
        summary = str(text or "").strip()
        lowered = summary.lower()
        facts = []
        seen = set()

        def add_fact(
            label,
            semantic_type,
            relation_type=None,
            weight=1.0,
            confidence=0.7,
        ):
            entity_type, canonical = self.canonical_graph_entity(
                label,
                semantic_type=semantic_type,
            )
            if not canonical:
                return
            resolved_relation = self.relation_type_for_entity(
                source_type,
                entity_type,
                lowered,
                fallback=relation_type or "mentions",
            )
            key = (entity_type, canonical, resolved_relation)
            if key in seen:
                return
            seen.add(key)
            facts.append(
                {
                    "entity_type": entity_type,
                    "label": canonical,
                    "relation_type": resolved_relation,
                    "weight": weight,
                    "confidence": confidence,
                },
            )

        if default_category and default_category != "general":
            add_fact(
                default_category,
                "category",
                relation_type="concerns",
                weight=1.1,
                confidence=0.8,
            )
        if default_mood and default_mood != "neutral":
            add_fact(
                default_mood,
                "emotion",
                relation_type="feels",
                weight=1.0,
                confidence=0.82,
            )
        if default_day:
            add_fact(
                default_day,
                "day",
                relation_type="mentioned_on",
                weight=0.9,
                confidence=0.78,
            )

        for canonical, aliases in self.GRAPH_ENTITY_ALIASES.items():
            if lowered == canonical or canonical in lowered or any(alias in lowered for alias in aliases):
                add_fact(
                    canonical,
                    self.GRAPH_ENTITY_TYPES.get(canonical, "topic"),
                    weight=1.1,
                    confidence=0.8,
                )

        if any(token in lowered for token in ["proud", "relieved", "grateful"]):
            add_fact(
                "positive",
                "emotion",
                relation_type="feels",
                weight=0.95,
                confidence=0.78,
            )
        if any(
            token in lowered
            for token in [
                "stress",
                "stressed",
                "overwhelmed",
                "anxious",
                "worried",
                "heavy",
                "pressure",
            ]
        ):
            if default_category and default_category != "general":
                add_fact(
                    f"{default_category}_pressure",
                    "stressor",
                    relation_type="struggles_with",
                    weight=1.15,
                    confidence=0.82,
                )
            else:
                add_fact(
                    "stress",
                    "stressor",
                    relation_type="struggles_with",
                    weight=1.0,
                    confidence=0.78,
                )

        llm_facts = self.llm_typed_graph_facts(summary, source_type)
        for fact in llm_facts:
            entity_type, canonical = self.canonical_graph_entity(
                fact.get("object") or fact.get("label"),
                semantic_type=fact.get("object_type") or fact.get("entity_type") or "topic",
            )
            if not canonical:
                continue
            relation_type = self.normalize_relation_type(
                fact.get("predicate") or fact.get("relation_type") or "mentions",
            )
            key = (entity_type, canonical, relation_type)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                {
                    "entity_type": entity_type,
                    "label": canonical,
                    "relation_type": relation_type,
                    "weight": float(fact.get("weight", 1.0) or 1.0),
                    "confidence": max(
                        0.3,
                        min(0.95, float(fact.get("confidence", 0.7) or 0.7)),
                    ),
                },
            )

        for token in self.graph_keyword_tokens(summary, limit=3):
            add_fact(
                token,
                "topic",
                relation_type="mentions",
                weight=0.7,
                confidence=0.65,
            )
        return facts

    def llm_typed_graph_facts(self, text, source_type):
        if not env_truthy("DADBOT_GRAPH_ENABLE_LLM_EXTRACTION", default=False):
            return []
        summary = str(text or "").strip()
        if not summary:
            return []
        prompt = f"""
From the text below, extract entities and typed relations as JSON.
Return a JSON array of objects with keys:
- object
- object_type
- predicate
- confidence

Use short canonical object names when possible.
Prefer predicates from this vocabulary:
feels, concerns, mentions, plans_for, struggles_with, proud_of, mentioned_on, responds_to.
Do not include duplicates.

Source type: {source_type}
Text: {summary}
""".strip()
        try:
            response = self._bot.call_ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                response_format="json",
                purpose="graph_fact_extraction",
            )
            parsed = self._bot.parse_model_json_content(response["message"]["content"])
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    # ------------------------------------------------------------------
    # Source confidence / weight
    # ------------------------------------------------------------------

    def graph_source_confidence(self, source_type, entry):
        if source_type == "consolidated_memory":
            return round(
                self._bot.normalize_confidence(
                    entry.get("confidence"),
                    source_count=entry.get("source_count", 1),
                    contradiction_count=len(entry.get("contradictions", [])),
                    updated_at=entry.get("updated_at"),
                ),
                2,
            )
        if source_type == "archive_session":
            try:
                turn_count = max(1, int(entry.get("turn_count", 1)))
            except (TypeError, ValueError):
                turn_count = 1
            numeric = min(
                0.92,
                0.5 + min(turn_count, 8) * 0.04 + 0.04 * len(entry.get("topics", [])),
            )
            return round(max(0.45, numeric), 2)
        if source_type == "persona_trait":
            try:
                critique = max(0, min(10, int(entry.get("critique_score", 0) or 0)))
            except (TypeError, ValueError):
                critique = 0
            try:
                impact = max(0.0, float(entry.get("impact_score", 0.0) or 0.0))
            except (TypeError, ValueError):
                impact = 0.0
            numeric = min(0.98, 0.52 + critique * 0.035 + min(impact, 4.0) * 0.04)
            return round(max(0.45, numeric), 2)
        if source_type == "life_pattern":
            try:
                pattern_confidence = max(
                    0,
                    min(100, int(entry.get("confidence", 0) or 0)),
                )
            except (TypeError, ValueError):
                pattern_confidence = 0
            numeric = min(0.96, 0.4 + pattern_confidence / 175.0)
            return round(max(0.45, numeric), 2)
        return 0.6

    def graph_source_weight(self, source_type, entry):
        if source_type == "consolidated_memory":
            try:
                return max(1.0, float(entry.get("source_count", 1)))
            except (TypeError, ValueError):
                return 1.0
        if source_type == "archive_session":
            try:
                return max(1.0, min(4.0, int(entry.get("turn_count", 1)) / 2.0))
            except (TypeError, ValueError):
                return 1.0
        if source_type == "persona_trait":
            strength = self._bot.long_term_signals.decayed_trait_strength(entry)
            return max(1.0, min(3.0, float(strength)))
        if source_type == "life_pattern":
            try:
                return max(1.0, min(3.0, int(entry.get("confidence", 0)) / 35.0))
            except (TypeError, ValueError):
                return 1.0
        return 1.0

    # ------------------------------------------------------------------
    # Node / edge upsert
    # ------------------------------------------------------------------

    @staticmethod
    def upsert_graph_node(node_map, **node):
        node_key = node["node_key"]
        existing = node_map.get(node_key)
        attributes = dict(node.get("attributes") or {})
        candidate = {
            "node_key": node_key,
            "node_type": node["node_type"],
            "label": str(node.get("label") or "").strip().lower(),
            "source_type": node.get("source_type"),
            "source_id": node.get("source_id"),
            "content": str(node.get("content") or "").strip(),
            "category": node.get("category"),
            "mood": node.get("mood"),
            "confidence": round(float(node.get("confidence", 0.0) or 0.0), 4),
            "updated_at": node.get("updated_at"),
            "attributes": attributes,
        }
        if existing is None:
            node_map[node_key] = candidate
            return candidate
        existing["confidence"] = max(
            existing.get("confidence", 0.0),
            candidate["confidence"],
        )
        existing["updated_at"] = (
            max(
                str(existing.get("updated_at") or ""),
                str(candidate.get("updated_at") or ""),
            )
            or None
        )
        if candidate.get("content") and len(candidate["content"]) > len(
            existing.get("content", ""),
        ):
            existing["content"] = candidate["content"]
        for key, value in candidate["attributes"].items():
            if key not in existing["attributes"] or value:
                existing["attributes"][key] = value
        return existing

    @staticmethod
    def upsert_graph_edge(edge_map, **edge):
        edge_key = edge["edge_key"]
        existing = edge_map.get(edge_key)
        evidence = list(edge.get("evidence") or [])
        # Bi-temporal fields: valid_from / valid_until represent the knowledge
        # validity window; event_time is when the fact occurred; ingestion_time
        # is when we first observed it.  All default to updated_at so legacy
        # projections without explicit turn-time still get a sensible value.
        updated_at_str = edge.get("updated_at") or ""
        candidate = {
            "edge_key": edge_key,
            "source_key": edge["source_key"],
            "target_key": edge["target_key"],
            "relation_type": edge["relation_type"],
            "weight": round(float(edge.get("weight", 0.0) or 0.0), 4),
            "confidence": round(float(edge.get("confidence", 0.0) or 0.0), 4),
            "updated_at": edge.get("updated_at"),
            "valid_from": edge.get("valid_from") or updated_at_str or None,
            "valid_until": edge.get("valid_until"),
            "event_time": edge.get("event_time") or updated_at_str or None,
            "ingestion_time": edge.get("ingestion_time") or updated_at_str or None,
            "evidence": evidence,
            "attributes": dict(edge.get("attributes") or {}),
        }
        if existing is None:
            edge_map[edge_key] = candidate
            return candidate
        existing["weight"] = round(existing.get("weight", 0.0) + candidate["weight"], 4)
        existing["confidence"] = max(
            existing.get("confidence", 0.0),
            candidate["confidence"],
        )
        existing["updated_at"] = (
            max(
                str(existing.get("updated_at") or ""),
                str(candidate.get("updated_at") or ""),
            )
            or None
        )
        # Preserve the original valid_from / event_time / ingestion_time on
        # subsequent upserts â€” only update valid_until if explicitly supplied.
        if existing.get("valid_from") is None and candidate["valid_from"]:
            existing["valid_from"] = candidate["valid_from"]
        if existing.get("event_time") is None and candidate["event_time"]:
            existing["event_time"] = candidate["event_time"]
        if existing.get("ingestion_time") is None and candidate["ingestion_time"]:
            existing["ingestion_time"] = candidate["ingestion_time"]
        if candidate.get("valid_until") is not None:
            existing["valid_until"] = candidate["valid_until"]
        for item in candidate["evidence"]:
            if item not in existing["evidence"]:
                existing["evidence"].append(item)
        for key, value in candidate["attributes"].items():
            if key not in existing["attributes"] or value:
                existing["attributes"][key] = value
        return existing

    @staticmethod
    def is_edge_valid(edge: dict, current_time_str: str) -> bool:
        """Return True if the edge is valid at ``current_time_str`` (ISO string).

        Edges without a ``valid_from`` field are treated as always-valid for
        backward compatibility with projections produced before Step 3.
        """
        valid_from = edge.get("valid_from")
        valid_until = edge.get("valid_until")
        if not valid_from:
            return True
        if valid_from > current_time_str:
            return False
        if valid_until is not None and valid_until <= current_time_str:
            return False
        return True

    @staticmethod
    def invalidate_edge(
        edge: dict,
        turn_time_str: str,
        *,
        reason: str = "",
    ) -> dict:
        """Explicitly invalidate an edge at ``turn_time_str``.

        Sets ``valid_until`` (bi-temporal close), ``invalidated_at`` (audit
        timestamp), and ``invalidation_reason`` (explainability/auditability).
        Returns the mutated edge (in-place) for convenience.

        Rules:
        - No silent overwrite — caller must pass ``reason`` for contradiction cases.
        - Edge is never deleted; it becomes non-visible but remains auditable.
        """
        edge["valid_until"] = turn_time_str
        edge["invalidated_at"] = turn_time_str
        edge["invalidation_reason"] = str(reason or "explicit_invalidation").strip() or "explicit_invalidation"
        return edge

    def has_conflicting_edges(
        self,
        edge_map: dict,
        source_key: str,
        relation_type: str,
        target_key: str,
        current_time_str: str,
    ) -> bool:
        """Return True if a valid (non-invalidated) edge with the same
        source→relation→target triple already exists in ``edge_map``.

        Used as a pre-commit invariant check before adding a new edge:
        if True, the caller must first call ``invalidate_edge`` on the
        conflicting edge before creating the replacement.
        """
        for edge in edge_map.values():
            if (
                str(edge.get("source_key") or "") == source_key
                and str(edge.get("relation_type") or "") == relation_type
                and str(edge.get("target_key") or "") == target_key
                and self.is_edge_valid(edge, current_time_str)
                and not str(edge.get("invalidated_at") or "")
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Graph projection
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Graph projection helpers (hoisted from build_graph_projection)
    # ------------------------------------------------------------------

    def _graph_add_source_node(
        self,
        node_map,
        source_type,
        entry,
        *,
        label,
        content,
        category=None,
        mood=None,
        updated_at=None,
        attributes=None,
    ):
        source_id = self.graph_source_id(source_type, entry)
        source_key = self.graph_node_key(source_type, source_id)
        self.upsert_graph_node(
            node_map,
            node_key=source_key,
            node_type=source_type,
            label=label,
            source_type=source_type,
            source_id=source_id,
            content=content,
            category=category,
            mood=mood,
            confidence=self.graph_source_confidence(source_type, entry),
            updated_at=updated_at,
            attributes=attributes,
        )
        return source_key, source_id

    def _graph_add_semantic_link(
        self,
        node_map,
        edge_map,
        source_key,
        source_type,
        source_id,
        semantic_type,
        label,
        relation_type,
        *,
        updated_at,
        weight=1.0,
        confidence=0.8,
        excerpt="",
        attributes=None,
    ):
        entity_type, normalized_label = self.canonical_graph_entity(
            label,
            semantic_type=semantic_type,
        )
        if not normalized_label:
            return
        normalized_relation = self.normalize_relation_type(relation_type)
        semantic_key = self.graph_node_key(entity_type, normalized_label)
        self.upsert_graph_node(
            node_map,
            node_key=semantic_key,
            node_type=entity_type,
            label=normalized_label,
            confidence=confidence,
            updated_at=updated_at,
            attributes=attributes,
        )
        self.upsert_graph_edge(
            edge_map,
            edge_key=self.graph_edge_key(source_key, semantic_key, normalized_relation),
            source_key=source_key,
            target_key=semantic_key,
            relation_type=normalized_relation,
            weight=weight,
            confidence=confidence,
            updated_at=updated_at,
            evidence=[
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "excerpt": str(excerpt or "").strip(),
                },
            ],
            attributes=attributes,
        )

    def _graph_add_typed_links_from_text(
        self,
        node_map,
        edge_map,
        source_key,
        source_type,
        source_id,
        text,
        *,
        default_category="general",
        default_mood="neutral",
        default_day="",
        updated_at=None,
        confidence=0.75,
    ):
        for fact in self.extract_typed_graph_facts(
            text,
            source_type,
            default_category=default_category,
            default_mood=default_mood,
            default_day=default_day,
        ):
            self._graph_add_semantic_link(
                node_map,
                edge_map,
                source_key,
                source_type,
                source_id,
                fact["entity_type"],
                fact["label"],
                fact["relation_type"],
                updated_at=updated_at,
                weight=fact.get("weight", 1.0),
                confidence=max(confidence, fact.get("confidence", confidence)),
                excerpt=text,
            )

    def _require_turn_temporal(self, turn_context=None):
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            if turn_context is not None:
                # TurnGraph is active but temporal node missing — hard fail.
                raise RuntimeError("TemporalNode required — execution invalid")
            runtime_temporal = getattr(self._bot, "_current_turn_time_base", None)
            runtime_wall_time = str(
                getattr(runtime_temporal, "wall_time", "") or "",
            ).strip()
            runtime_wall_date = str(
                getattr(runtime_temporal, "wall_date", "") or "",
            ).strip()
            if runtime_wall_time and runtime_wall_date:
                return runtime_temporal
            # Maintenance / direct-API path (no TurnGraph): synthesize temporal.
            from datetime import date as _date
            from datetime import datetime
            from types import SimpleNamespace

            return SimpleNamespace(
                wall_time=datetime.now().isoformat(timespec="seconds"),
                wall_date=_date.today().isoformat(),
            )
        wall_time = str(getattr(temporal, "wall_time", "")).strip()
        wall_date = str(getattr(temporal, "wall_date", "")).strip()
        if not wall_time or not wall_date:
            raise RuntimeError("TemporalNode required — execution invalid")
        return temporal

    def _project_consolidated_memories(self, node_map, edge_map, *, turn_context=None):
        temporal = self._require_turn_temporal(turn_context)
        for entry in self._mm.consolidated_memories():
            if bool(entry.get("superseded", False)):
                continue
            summary = str(entry.get("summary", "")).strip()
            updated_at = str(
                entry.get("updated_at") or getattr(temporal, "wall_time", ""),
            )
            category = str(entry.get("category", "general")).strip().lower() or "general"
            source_key, source_id = self._graph_add_source_node(
                node_map,
                "consolidated_memory",
                entry,
                label=summary,
                content=summary,
                category=category,
                mood="neutral",
                updated_at=updated_at,
                attributes={
                    "summary": summary,
                    "supporting_summaries": list(entry.get("supporting_summaries", []))[:4],
                    "contradictions": list(entry.get("contradictions", []))[:4],
                    "source_count": int(entry.get("source_count", 1) or 1),
                },
            )
            self._graph_add_semantic_link(
                node_map,
                edge_map,
                source_key,
                "consolidated_memory",
                source_id,
                "category",
                category,
                "categorized_as",
                updated_at=updated_at,
                weight=self.graph_source_weight("consolidated_memory", entry),
                confidence=self.graph_source_confidence("consolidated_memory", entry),
                excerpt=summary,
            )
            self._graph_add_typed_links_from_text(
                node_map,
                edge_map,
                source_key,
                "consolidated_memory",
                source_id,
                summary,
                default_category=category,
                default_mood="neutral",
                updated_at=updated_at,
                confidence=self.graph_source_confidence("consolidated_memory", entry),
            )
            for token in self.graph_keyword_tokens(summary, limit=4):
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "consolidated_memory",
                    source_id,
                    "topic",
                    token,
                    "mentions",
                    updated_at=updated_at,
                    weight=1.0,
                    confidence=self.graph_source_confidence(
                        "consolidated_memory",
                        entry,
                    ),
                    excerpt=summary,
                )
            for contradiction in list(entry.get("contradictions", []))[:2]:
                # Explicitly invalidate any existing edge that the contradiction supersedes,
                # then create a new contradicted_by edge. No silent overwrite; no dual-truth.
                contradiction_relation = "contradicted_by"
                contradiction_entity_type = "contradiction"
                _, norm_contradiction = self.canonical_graph_entity(
                    contradiction,
                    semantic_type=contradiction_entity_type,
                )
                if norm_contradiction:
                    contradiction_key = self.graph_node_key(
                        contradiction_entity_type,
                        norm_contradiction,
                    )
                    old_edge_key = self.graph_edge_key(
                        source_key,
                        contradiction_key,
                        contradiction_relation,
                    )
                    if old_edge_key in edge_map:
                        self.invalidate_edge(
                            edge_map[old_edge_key],
                            updated_at,
                            reason=f"superseded_by_contradiction:{contradiction[:80]}",
                        )
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "consolidated_memory",
                    source_id,
                    contradiction_entity_type,
                    contradiction,
                    contradiction_relation,
                    updated_at=updated_at,
                    weight=1.0,
                    confidence=0.5,
                    excerpt=contradiction,
                )

    def _project_archive_sessions(self, node_map, edge_map, *, turn_context=None):
        temporal = self._require_turn_temporal(turn_context)
        for entry in self._mm.session_archive():
            summary = str(entry.get("summary", "")).strip()
            updated_at = str(
                entry.get("created_at") or getattr(temporal, "wall_time", ""),
            )
            mood = self._bot.normalize_mood(entry.get("dominant_mood"))
            source_key, source_id = self._graph_add_source_node(
                node_map,
                "archive_session",
                entry,
                label=summary,
                content=summary,
                category="archive",
                mood=mood,
                updated_at=updated_at,
                attributes={
                    "summary": summary,
                    "topics": list(entry.get("topics", []))[:5],
                    "turn_count": int(entry.get("turn_count", 0) or 0),
                },
            )
            for topic in entry.get("topics", [])[:5]:
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "archive_session",
                    source_id,
                    "topic",
                    topic,
                    "covers_topic",
                    updated_at=updated_at,
                    weight=1.2,
                    confidence=self.graph_source_confidence("archive_session", entry),
                    excerpt=summary,
                )
            if mood != "neutral":
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "archive_session",
                    source_id,
                    "mood",
                    mood,
                    "reflects_mood",
                    updated_at=updated_at,
                    weight=1.0,
                    confidence=self.graph_source_confidence("archive_session", entry),
                    excerpt=summary,
                )
            archive_topics = [str(topic).strip().lower() for topic in entry.get("topics", []) if str(topic).strip()]
            primary_archive_topic = archive_topics[0] if archive_topics else "general"
            self._graph_add_typed_links_from_text(
                node_map,
                edge_map,
                source_key,
                "archive_session",
                source_id,
                summary,
                default_category=primary_archive_topic,
                default_mood=mood,
                updated_at=updated_at,
                confidence=self.graph_source_confidence("archive_session", entry),
            )
            for token in self.graph_keyword_tokens(summary, limit=3):
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "archive_session",
                    source_id,
                    "topic",
                    token,
                    "mentions",
                    updated_at=updated_at,
                    weight=0.9,
                    confidence=self.graph_source_confidence("archive_session", entry),
                    excerpt=summary,
                )

    def _project_persona_traits(self, node_map, edge_map, *, turn_context=None):
        temporal = self._require_turn_temporal(turn_context)
        for entry in self._mm.persona_evolution_history():
            trait = str(entry.get("trait", "")).strip()
            if not trait:
                continue
            updated_at = str(
                entry.get("last_reinforced_at") or entry.get("applied_at") or getattr(temporal, "wall_time", ""),
            )
            reason = str(entry.get("reason") or "").strip()
            content = " ".join(part for part in [reason, str(entry.get("announcement") or "").strip()] if part)
            source_key, source_id = self._graph_add_source_node(
                node_map,
                "persona_trait",
                entry,
                label=trait,
                content=content,
                category="persona",
                mood="neutral",
                updated_at=updated_at,
                attributes={
                    "trait": trait,
                    "reason": reason,
                    "impact_score": float(entry.get("impact_score", 0.0) or 0.0),
                    "strength": self._bot.long_term_signals.decayed_trait_strength(
                        entry,
                    ),
                    "critique_score": int(entry.get("critique_score", 0) or 0),
                },
            )
            self._graph_add_semantic_link(
                node_map,
                edge_map,
                source_key,
                "persona_trait",
                source_id,
                "trait",
                trait,
                "expresses_trait",
                updated_at=updated_at,
                weight=self.graph_source_weight("persona_trait", entry),
                confidence=self.graph_source_confidence("persona_trait", entry),
                excerpt=content or trait,
            )
            self._graph_add_typed_links_from_text(
                node_map,
                edge_map,
                source_key,
                "persona_trait",
                source_id,
                content or trait,
                default_category="persona",
                updated_at=updated_at,
                confidence=self.graph_source_confidence("persona_trait", entry),
            )
            for token in self.graph_keyword_tokens(reason, limit=3):
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "persona_trait",
                    source_id,
                    "topic",
                    token,
                    "reinforced_by",
                    updated_at=updated_at,
                    weight=0.8,
                    confidence=self.graph_source_confidence("persona_trait", entry),
                    excerpt=reason,
                )

    def _project_life_patterns(self, node_map, edge_map, *, turn_context=None):
        temporal = self._require_turn_temporal(turn_context)
        for entry in self._mm.life_patterns():
            summary = str(entry.get("summary", "")).strip()
            updated_at = str(
                entry.get("last_seen_at") or getattr(temporal, "wall_time", ""),
            )
            topic = str(entry.get("topic") or "general").strip().lower() or "general"
            mood = self._bot.normalize_mood(entry.get("mood"))
            day_hint = str(entry.get("day_hint") or "").strip()
            source_key, source_id = self._graph_add_source_node(
                node_map,
                "life_pattern",
                entry,
                label=summary,
                content=str(entry.get("proactive_message") or summary).strip(),
                category=topic,
                mood=mood,
                updated_at=updated_at,
                attributes={
                    "summary": summary,
                    "topic": topic,
                    "day_hint": day_hint,
                    "confidence": int(entry.get("confidence", 0) or 0),
                },
            )
            if topic != "general":
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "life_pattern",
                    source_id,
                    "topic",
                    topic,
                    "pattern_topic",
                    updated_at=updated_at,
                    weight=1.2,
                    confidence=self.graph_source_confidence("life_pattern", entry),
                    excerpt=summary,
                )
            if mood != "neutral":
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "life_pattern",
                    source_id,
                    "mood",
                    mood,
                    "pattern_mood",
                    updated_at=updated_at,
                    weight=1.0,
                    confidence=self.graph_source_confidence("life_pattern", entry),
                    excerpt=summary,
                )
            if day_hint:
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "life_pattern",
                    source_id,
                    "day",
                    day_hint,
                    "recurs_on",
                    updated_at=updated_at,
                    weight=1.0,
                    confidence=self.graph_source_confidence("life_pattern", entry),
                    excerpt=summary,
                )
            self._graph_add_typed_links_from_text(
                node_map,
                edge_map,
                source_key,
                "life_pattern",
                source_id,
                summary,
                default_category=topic,
                default_mood=mood,
                default_day=day_hint,
                updated_at=updated_at,
                confidence=self.graph_source_confidence("life_pattern", entry),
            )
            for token in self.graph_keyword_tokens(summary, limit=3):
                self._graph_add_semantic_link(
                    node_map,
                    edge_map,
                    source_key,
                    "life_pattern",
                    source_id,
                    "topic",
                    token,
                    "mentions",
                    updated_at=updated_at,
                    weight=0.85,
                    confidence=self.graph_source_confidence("life_pattern", entry),
                    excerpt=summary,
                )

    def build_graph_projection(self, turn_context=None):
        temporal = self._require_turn_temporal(turn_context)
        enforce_temporal_window = turn_context is not None
        node_map = {}
        edge_map = {}
        self._project_consolidated_memories(
            node_map,
            edge_map,
            turn_context=turn_context,
        )
        self._project_archive_sessions(node_map, edge_map, turn_context=turn_context)
        self._project_persona_traits(node_map, edge_map, turn_context=turn_context)
        self._project_life_patterns(node_map, edge_map, turn_context=turn_context)
        # Apply bi-temporal validity filter: only surface edges that are valid
        # at the canonical turn time so stale/invalidated edges are excluded.
        current_time_str = str(getattr(temporal, "wall_time", "")) if enforce_temporal_window else ""
        nodes = sorted(
            node_map.values(),
            key=lambda item: (item["node_type"], item["label"], item["node_key"]),
        )
        node_keys = {item.get("node_key") for item in nodes}
        edges = sorted(
            (
                e
                for e in edge_map.values()
                if (
                    self.is_edge_valid(e, current_time_str)
                    if current_time_str
                    else self.is_edge_valid(e, str(e.get("updated_at") or ""))
                )
                and e.get("source_key") in node_keys
                and e.get("target_key") in node_keys
            ),
            key=lambda item: (
                item["source_key"],
                item["relation_type"],
                item["target_key"],
            ),
        )
        for edge in edges:
            if not edge.get("valid_from"):
                edge["valid_from"] = edge.get("updated_at") or current_time_str
        updated_at = None
        for item in [*nodes, *edges]:
            value = item.get("updated_at")
            if value and (updated_at is None or str(value) > str(updated_at)):
                updated_at = value
        return {"nodes": nodes, "edges": edges, "updated_at": updated_at}

    def preview_memory_graph(self, snapshot=None):
        graph = snapshot or self.graph_snapshot()
        nodes_by_key = {node["node_key"]: node for node in graph.get("nodes", [])}
        semantic_weights = {}
        semantic_types = {}
        source_neighbors = {}

        for edge in graph.get("edges", []):
            source = nodes_by_key.get(edge.get("source_key"))
            target = nodes_by_key.get(edge.get("target_key"))
            if source is None or target is None:
                continue
            if source.get("node_type") not in self.GRAPH_SOURCE_NODE_TYPES:
                continue
            if target.get("node_type") == "contradiction":
                continue
            label = str(target.get("label") or "").strip().lower()
            if not label:
                continue
            semantic_weights[label] = semantic_weights.get(label, 0.0) + float(
                edge.get("weight", 0.0) or 0.0,
            )
            semantic_types[label] = target.get("node_type")
            source_neighbors.setdefault(source["node_key"], []).append(label)

        edge_weights = {}
        for labels in source_neighbors.values():
            unique = []
            for label in labels:
                if label not in unique:
                    unique.append(label)
            for index, left in enumerate(unique):
                for right in unique[index + 1 :]:
                    edge_key = tuple(sorted((left, right)))
                    edge_weights[edge_key] = edge_weights.get(edge_key, 0.0) + 1.0

        preview_nodes = []
        for label, weight in sorted(
            semantic_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )[:18]:
            node_type = semantic_types.get(label, "topic")
            preview_type = node_type if node_type in {"topic", "category", "mood"} else "topic"
            preview_nodes.append(
                {
                    "id": f"{preview_type}:{label}",
                    "label": label,
                    "type": preview_type,
                    "weight": max(1, int(round(weight))),
                },
            )

        preview_edges = [
            {"source": left, "target": right, "weight": max(1, int(round(weight)))}
            for (left, right), weight in sorted(
                edge_weights.items(),
                key=lambda item: (-item[1], item[0]),
            )[:18]
        ]
        return {
            "nodes": preview_nodes,
            "edges": preview_edges,
            "updated_at": graph.get("updated_at"),
        }

    # ------------------------------------------------------------------
    # Store sync / snapshot
    # ------------------------------------------------------------------

    def sync_graph_store(self, turn_context=None):
        runtime = getattr(self, "_bot", None)
        commit_active = bool(getattr(runtime, "_graph_commit_active", False))
        if turn_context is not None:
            # TurnGraph path: strict temporal + commit boundary enforcement.
            self._require_turn_temporal(turn_context)
            active_stage = (
                str(
                    (getattr(turn_context, "state", None) or {}).get(
                        "_active_graph_stage",
                    )
                    or "",
                )
                .strip()
                .lower()
            )
            if not commit_active or active_stage not in {"save", ""}:
                raise RuntimeError(
                    "Graph sync violation: graph writes are only allowed at SaveNode commit boundary "
                    f"(active_stage={active_stage!r}, commit_active={commit_active!r}).",
                )
        # Maintenance / direct-API path (turn_context=None): no boundary check.
        snapshot = self.build_graph_projection(turn_context=turn_context)
        try:
            self.ensure_graph_store()
            self._graph_store_backend.replace_graph(
                snapshot.get("nodes", []),
                snapshot.get("edges", []),
            )
        except Exception as exc:
            logger.warning("Graph store sync failed: %s", exc)
        return snapshot

    def graph_snapshot(self):
        try:
            self.ensure_graph_store()
            snapshot = self._graph_store_backend.fetch_graph()
            if snapshot.get("nodes") or snapshot.get("edges"):
                if not snapshot.get("updated_at"):
                    updated_at = None
                    for item in [
                        *list(snapshot.get("nodes") or []),
                        *list(snapshot.get("edges") or []),
                    ]:
                        if not isinstance(item, dict):
                            continue
                        value = item.get("updated_at")
                        if value and (updated_at is None or str(value) > str(updated_at)):
                            updated_at = value
                    snapshot = dict(snapshot)
                    snapshot["updated_at"] = updated_at
                return snapshot
        except Exception as exc:
            logger.warning(
                "Graph store fetch failed, using in-memory projection: %s",
                exc,
            )
        # Graph store is empty or unavailable: fall back to in-memory projection.
        try:
            return self.build_graph_projection(turn_context=None)
        except Exception as exc:
            logger.warning("In-memory graph projection failed: %s", exc)
            return {"nodes": [], "edges": [], "updated_at": None}

    # ------------------------------------------------------------------
    # Source display helpers
    # ------------------------------------------------------------------

    def graph_source_summary(self, node):
        attributes = dict(node.get("attributes") or {})
        return str(
            attributes.get("summary") or attributes.get("reason") or node.get("content") or node.get("label") or "",
        ).strip()

    def graph_source_lines(self, selected_items):
        lines = []
        for score, node, matched_labels in selected_items:
            attributes = dict(node.get("attributes") or {})
            source_type = node.get("node_type")
            summary = self.graph_source_summary(node)
            if source_type == "consolidated_memory":
                confidence = self._bot.confidence_label(
                    float(node.get("confidence", 0.5) or 0.5),
                )
                line = f"- [consolidated insight | confidence={confidence}] {summary}"
                contradictions = list(attributes.get("contradictions", []))[:2]
                if contradictions:
                    line += f" Tension: {'; '.join(contradictions)}."
            elif source_type == "archive_session":
                mood = self._bot.normalize_mood(node.get("mood"))
                line = f"- [archived chat | mood={mood}] {summary}"
            elif source_type == "persona_trait":
                line = (
                    f"- [dad trait | impact={float(attributes.get('impact_score', 0.0) or 0.0):.2f}, "
                    f"strength={float(attributes.get('strength', 1.0) or 1.0):.2f}] Dad has become more {node.get('label', '')} with Tony."
                )
                if attributes.get("reason"):
                    line += f" {attributes['reason']}"
            else:
                confidence = int(attributes.get("confidence", 0) or 0)
                line = f"- [life pattern | confidence={confidence}%] {summary}"
            if matched_labels:
                line += f" Connected nodes: {', '.join(matched_labels[:3])}."
            line += f" (score={score:.2f})"
            lines.append(line)
        return lines

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    _RETRIEVAL_SOURCE_TYPE_WEIGHTS = {
        "consolidated_memory": 1.18,
        "archive_session": 1.0,
        "persona_trait": 1.08,
        "life_pattern": 1.05,
    }

    def _score_graph_node(
        self,
        node,
        adjacency,
        nodes,
        query_tokens,
        query_category,
        query_mood,
        mood_trend,
        recent_topics,
    ):
        """Score a single source node for query relevance.
        Returns (score, matched_labels) or None if node should be excluded.
        """
        attributes = dict(node.get("attributes") or {})
        summary = self.graph_source_summary(node)
        text = " ".join(
            part
            for part in [
                node.get("label", ""),
                node.get("content", ""),
                summary,
                " ".join(attributes.get("supporting_summaries", [])),
                " ".join(attributes.get("contradictions", [])),
            ]
            if part
        )
        source_tokens = self._bot.significant_tokens(text)
        overlap = len(query_tokens & source_tokens)
        if query_category != "general" and str(node.get("category") or "general").strip().lower() == query_category:
            overlap += 2
        if query_mood != "neutral" and self._bot.normalize_mood(node.get("mood")) == query_mood:
            overlap += 1
        if mood_trend != "neutral" and self._bot.normalize_mood(node.get("mood")) == mood_trend:
            overlap += 0.5
        matched_labels = []
        for edge in adjacency.get(node["node_key"], []):
            neighbor_key = (
                edge.get("target_key") if edge.get("source_key") == node["node_key"] else edge.get("source_key")
            )
            neighbor = nodes.get(neighbor_key)
            if (
                neighbor is None
                or neighbor.get("node_type") in self.GRAPH_SOURCE_NODE_TYPES
                or neighbor.get("node_type") == "contradiction"
            ):
                continue
            label = str(neighbor.get("label") or "").strip().lower()
            if not label:
                continue
            if (
                label in query_tokens
                or (query_category != "general" and label == query_category)
                or label in recent_topics
            ):
                matched_labels.append(label)
                overlap += 1.1
            elif query_mood != "neutral" and label == query_mood:
                matched_labels.append(label)
                overlap += 0.8
        if overlap <= 0 and query_tokens:
            if not any(token in text.lower() for token in query_tokens):
                return None
            overlap = 0.75
        freshness = self._bot.memory_freshness_weight(node.get("updated_at"))
        confidence = max(
            0.4,
            min(1.15, float(node.get("confidence", 0.6) or 0.6) + 0.15),
        )
        contradictions = len(attributes.get("contradictions", []))
        contradiction_penalty = max(0.5, 1.0 - contradictions * 0.12)
        score = (
            overlap
            * freshness
            * confidence
            * contradiction_penalty
            * self._RETRIEVAL_SOURCE_TYPE_WEIGHTS.get(node.get("node_type"), 1.0)
        )
        if score <= 0.15:
            return None
        return round(score, 4), sorted(set(matched_labels))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def graph_retrieval_for_input(self, query, limit=3):
        snapshot = self.graph_snapshot()
        nodes = {node["node_key"]: node for node in snapshot.get("nodes", [])}
        if not nodes:
            return None

        adjacency = {}
        for edge in snapshot.get("edges", []):
            adjacency.setdefault(edge.get("source_key"), []).append(edge)
            adjacency.setdefault(edge.get("target_key"), []).append(edge)

        query_tokens = self._bot.significant_tokens(query)
        query_category = self._bot.infer_memory_category(query)
        query_mood = self._bot.normalize_mood(query)
        recent_topics = set(self._bot.recent_memory_topics(limit=4))
        mood_trend = self._bot.current_memory_mood_trend()

        ranked = []
        for node in nodes.values():
            if node.get("node_type") not in self.GRAPH_SOURCE_NODE_TYPES:
                continue
            result = self._score_graph_node(
                node,
                adjacency,
                nodes,
                query_tokens,
                query_category,
                query_mood,
                mood_trend,
                recent_topics,
            )
            if result is None:
                continue
            score, matched_labels = result
            ranked.append((score, node, matched_labels))

        if not ranked:
            return None

        selected = sorted(
            ranked,
            key=lambda item: (
                item[0],
                item[1].get("updated_at", ""),
                item[1].get("label", ""),
            ),
            reverse=True,
        )[: max(1, int(limit or 3))]
        return {
            "compressed_summary": self._graph_prompt_compressor.compress_neighborhood(
                query,
                snapshot.get("nodes", []),
                snapshot.get("edges", []),
                max_tokens=self._bot.runtime_config.graph_context_token_budget,
            ),
            "summary_lines": self.graph_source_lines(selected),
            "supporting_evidence": [
                {
                    "source_type": node.get("node_type"),
                    "label": node.get("label", ""),
                    "summary": self.graph_source_summary(node),
                    "category": str(node.get("category") or "general").strip().lower() or "general",
                    "mood": self._bot.normalize_mood(node.get("mood")),
                    "confidence": round(float(node.get("confidence", 0.0) or 0.0), 2),
                    "updated_at": node.get("updated_at"),
                    "score": score,
                    "matched_nodes": matched_labels,
                    "contradictions": list(
                        (node.get("attributes") or {}).get("contradictions", []),
                    )[:2],
                }
                for score, node, matched_labels in selected
            ],
            "updated_at": snapshot.get("updated_at"),
        }

    def _rank_graph_summary_nodes(self, nodes, adjacency):
        """Rank semantic nodes by total weighted source-node connections.
        Returns list of (total_weight, node, source_types, companion_labels).
        """
        ranked = []
        for node in nodes.values():
            if node.get("node_type") in self.GRAPH_SOURCE_NODE_TYPES or node.get("node_type") == "contradiction":
                continue
            source_neighbors = []
            companion_labels = []
            total_weight = 0.0
            for edge in adjacency.get(node["node_key"], []):
                neighbor_key = (
                    edge.get("target_key") if edge.get("source_key") == node["node_key"] else edge.get("source_key")
                )
                neighbor = nodes.get(neighbor_key)
                if neighbor is None:
                    continue
                if neighbor.get("node_type") in self.GRAPH_SOURCE_NODE_TYPES:
                    source_neighbors.append(neighbor)
                    total_weight += float(edge.get("weight", 0.0) or 0.0) * max(
                        0.4,
                        float(neighbor.get("confidence", 0.0) or 0.0),
                    )
                    for peer_edge in adjacency.get(neighbor["node_key"], []):
                        peer_key = (
                            peer_edge.get("target_key")
                            if peer_edge.get("source_key") == neighbor["node_key"]
                            else peer_edge.get("source_key")
                        )
                        peer = nodes.get(peer_key)
                        if (
                            peer is None
                            or peer.get("node_key") == node["node_key"]
                            or peer.get("node_type") in self.GRAPH_SOURCE_NODE_TYPES
                            or peer.get("node_type") == "contradiction"
                        ):
                            continue
                        companion_labels.append(
                            str(peer.get("label") or "").strip().lower(),
                        )
            if not source_neighbors:
                continue
            source_types = sorted(
                {neighbor.get("node_type") for neighbor in source_neighbors if neighbor.get("node_type")},
            )
            ranked.append(
                (round(total_weight, 4), node, source_types, companion_labels),
            )
        return ranked

    def _format_graph_summary_lines(self, ranked, limit):
        """Format top-ranked semantic nodes into human-readable summary lines."""
        source_labels = {
            "archive_session": "archived chats",
            "consolidated_memory": "consolidated insights",
            "persona_trait": "dad traits",
            "life_pattern": "life patterns",
        }
        lines = []
        for _score, node, source_types, companion_labels in sorted(
            ranked,
            key=lambda item: (item[0], item[1].get("label", "")),
            reverse=True,
        )[: max(1, int(limit or 3))]:
            sources_text = self._bot.natural_list(
                [source_labels.get(name, name.replace("_", " ")) for name in source_types],
            )
            companion_unique = []
            for label in companion_labels:
                if label and label not in companion_unique:
                    companion_unique.append(label)
            line = f"- {str(node.get('label', '')).title()} is reinforced across {sources_text}"
            if companion_unique:
                line += f", often alongside {self._bot.natural_list([label.title() for label in companion_unique[:2]])}"
            line += "."
            lines.append(line)
        return lines

    def build_graph_summary_context(self, limit=3):
        snapshot = self.graph_snapshot()
        nodes = {node["node_key"]: node for node in snapshot.get("nodes", [])}
        if not nodes:
            return None

        compressed = self._graph_prompt_compressor.compress_neighborhood(
            "long-term relationship state with Tony",
            snapshot.get("nodes", []),
            snapshot.get("edges", []),
            max_tokens=min(260, self._bot.runtime_config.graph_context_token_budget),
        )
        if compressed:
            return "Relationship graph summary:\n" + compressed

        adjacency = {}
        for edge in snapshot.get("edges", []):
            adjacency.setdefault(edge.get("source_key"), []).append(edge)
            adjacency.setdefault(edge.get("target_key"), []).append(edge)

        ranked = self._rank_graph_summary_nodes(nodes, adjacency)
        if not ranked:
            return None

        lines = self._format_graph_summary_lines(ranked, limit)
        if not lines:
            return None
        return "Relationship graph summary:\n" + "\n".join(lines)
