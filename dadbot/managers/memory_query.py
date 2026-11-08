from __future__ import annotations

import heapq
from datetime import date
from time import perf_counter

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.memory.lexical_index import LexicalMemoryIndex
from dadbot.memory.reranker import get_local_reranker, reranked_candidates, select_candidates
from dadbot.utils import SIGNIFICANT_TOKEN_STOPWORDS, tokenize_text

_RETRIEVAL_QUERY_WORDS = frozenset(
    {"about", "can", "earlier", "mentioned", "please", "recall", "remember", "remind", "tell", "what", "you"}
)
MEMORY_DENSE_WEIGHT = 0.75


class MemoryQueryManager:
    """Owns memory retrieval scoring, memory replies, and saved-memory CRUD flows."""

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self._lexical_index = LexicalMemoryIndex()

    @staticmethod
    def lexical_query_tokens(user_input: str) -> set[str]:
        return {
            token for token in tokenize_text(user_input)
            if len(token) >= 3 and token not in SIGNIFICANT_TOKEN_STOPWORDS and token not in _RETRIEVAL_QUERY_WORDS
        }

    def close(self) -> None:
        self._lexical_index.close()

    def rerank_memories(self, query, ranked, limit):
        if not self.bot.runtime_config.memory_reranker_enabled:
            return self.select_diverse_ranked_memories(ranked, limit), {}
        fired, gap, candidates = select_candidates(
            ranked, self.memory_diversity_key,
            top_score_threshold=self.bot.runtime_config.memory_reranker_top_score_threshold,
        )
        elapsed = 0.0
        if fired:
            start = perf_counter()
            model = get_local_reranker(self.bot.runtime_config.memory_reranker_path)
            scores = model.score(query, [memory["summary"] for _, memory in candidates])
            candidates = reranked_candidates(candidates, scores)
            elapsed = perf_counter() - start
            selected = self.select_diverse_ranked_memories([*candidates, *ranked], limit)
        else:
            selected = self.select_diverse_ranked_memories(ranked, limit)
        return selected, {
            "reranker_fired": fired, "reranker_seconds": round(elapsed, 6),
            "fused_margin": gap if len(candidates) > 1 else None,
            "fused_top_score": ranked[0][0] if ranked else 0.0,
        }

    def memory_freshness_weight(self, value):
        elapsed_days = self.bot.days_since_iso_date(value)
        if elapsed_days is None:
            return 1.0
        half_life = max(1, int(self.bot.runtime_config.memory_freshness_half_life_days))
        min_weight = float(self.bot.runtime_config.memory_min_freshness_weight)
        raw = 0.5 ** (elapsed_days / float(half_life))
        return round(max(min_weight, min(1.0, raw)), 3)

    @staticmethod
    def memory_impact_score(memory):
        try:
            return float(memory.get("impact_score", 0.0))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def memory_importance_score(memory):
        try:
            return max(0.0, min(1.0, float(memory.get("importance_score", 0.0) or 0.0)))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def semantic_memory_freshness_weight(self, memory):
        elapsed_days = self.bot.days_since_iso_date(
            memory.get("updated_at") or memory.get("created_at"),
        )
        if elapsed_days is None or elapsed_days <= 0:
            return 1.0

        impact = self.memory_impact_score(memory)
        if elapsed_days <= 7:
            raw = 0.5 ** (elapsed_days / 7.0)
        else:
            raw = 0.5 * (0.5 ** ((elapsed_days - 7) / 45.0))

        if elapsed_days > 60 and impact < 2.5:
            raw = min(raw, 0.25)
        elif elapsed_days > 30 and impact < 1.5:
            raw = min(raw, 0.35)

        if impact >= 2.5:
            raw = max(raw, 0.45)
        elif impact >= 1.5:
            raw = max(raw, 0.30)

        return round(max(0.08, min(1.0, raw)), 3)

    def recent_memory_topics(self, limit=5):
        topics = []
        desired = max(1, int(limit or 5))
        for entry in reversed(self.bot.session_archive()[-desired * 2 :]):
            for topic in entry.get("topics", []):
                normalized = str(topic or "").strip().lower()
                if normalized and normalized != "general" and normalized not in topics:
                    topics.append(normalized)
                if len(topics) >= desired:
                    return topics
        return topics

    def current_memory_mood_trend(self):
        trend = self.bot.long_term_signals.recent_mood_trend() if self.bot.session_moods else self.bot.last_saved_mood()
        return self.bot.normalize_mood(trend)

    def memory_alignment_weight(
        self,
        memory,
        query_tokens=None,
        query_category="general",
        query_mood="neutral",
        recent_topics=None,
        mood_trend=None,
    ):
        summary = str(memory.get("summary", ""))
        tokens = self.bot.significant_tokens(summary)
        query_tokens = set(query_tokens or [])
        category = str(memory.get("category", "general")).strip().lower() or "general"
        recent_topics = set(recent_topics or [])
        memory_mood = self.bot.normalize_mood(memory.get("mood"))
        impact = self.memory_impact_score(memory)
        topical_overlap = bool(query_tokens & tokens)
        topic_match = (
            (query_category != "general" and category == query_category) or category in recent_topics or topical_overlap
        )

        weight = 1.0
        if recent_topics and category not in recent_topics and query_category == "general" and not topical_overlap:
            weight *= 0.75 if impact < 2.0 else 0.9

        if memory_mood != "neutral":
            if query_mood != "neutral" and memory_mood != query_mood:
                weight *= 0.6 if impact < 2.0 else 0.8
            elif mood_trend not in {None, "", "neutral"} and memory_mood != mood_trend:
                weight *= 0.75 if impact < 2.0 else 0.9

        if (
            not topic_match
            and memory_mood != "neutral"
            and query_mood != "neutral"
            and memory_mood != query_mood
            and impact < 1.5
        ):
            return 0.0
        return round(max(0.0, min(1.2, weight)), 3)

    def memory_diversity_key(self, memory):
        return self.bot.memory_dedup_key(memory)

    def select_diverse_ranked_memories(self, ranked_items, limit, memory_index=1):
        desired = max(1, int(limit or 1))
        selected = []
        seen_keys = set()
        for item in ranked_items:
            key = self.memory_diversity_key(item[memory_index])
            if key in seen_keys:
                continue
            selected.append(item)
            seen_keys.add(key)
            if len(selected) >= desired:
                break

        # Diversify display order only after the strongest distinct set is fixed.
        first_in_category = []
        repeated_categories = []
        seen_categories = set()
        for item in selected:
            category = str(item[memory_index].get("category", "general")).strip().lower() or "general"
            if category in seen_categories:
                repeated_categories.append(item)
            else:
                first_in_category.append(item)
                seen_categories.add(category)
        return first_in_category + repeated_categories

    def memory_context_limit_for_input(self, user_input):
        current_mood = self.bot.last_saved_mood()
        baseline_sections = self.bot.base_request_sections(current_mood)
        baseline_sections.extend(
            [
                self.bot.build_daily_checkin_context(current_mood),
                self.bot.build_active_tool_observation_context(),
                self.bot.build_cross_session_context(),
                self.bot.build_session_summary_context(),
                self.bot.build_relevant_context(user_input),
                self.bot.build_wisdom_context(user_input),
                self.bot.build_escalation_context(current_mood, self.bot.session_moods),
            ],
        )
        baseline_prompt = "\n\n".join(section for section in baseline_sections if section)
        history_budget = self.bot.prompt_history_token_budget(
            baseline_prompt,
            user_input,
        )
        if history_budget < 120:
            return 1
        if history_budget < 220:
            return 2
        if history_budget < 360:
            return 3
        return 4

    def memory_relevance_score(self, user_input, memory_summary):
        query_tokens = self.bot.tokenize(user_input)
        memory_tokens = self.bot.tokenize(memory_summary)
        overlap = query_tokens & memory_tokens

        score = len(overlap)
        if any(term in user_input.lower() for term in ["remember", "recall", "last time", "before", "previous"]):
            score += 2
        return score

    def _recent_turn_context(self, *, max_turns=3):
        history = list(self.bot.conversation_history() or [])
        if not history:
            return ""
        recent = [item for item in history[-max(2, int(max_turns or 3) * 2) :] if isinstance(item, dict)]
        parts = []
        for item in recent:
            role = str(item.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            speaker = "Tony" if role == "user" else "Dad"
            parts.append(f"{speaker}: {content}")
        return "\n".join(parts)

    def _importance_signal(self, memory):
        base = self.memory_importance_score(memory)
        category = str(memory.get("category", "general") or "general").strip().lower()
        summary = str(memory.get("summary", "") or "").strip().lower()

        boost = 0.0
        if category in {"identity", "relationship", "relationships", "preferences", "preference"}:
            boost += 0.25
        if any(token in summary for token in ["my name", "i am", "wife", "husband", "daughter", "son"]):
            boost += 0.15
        if any(token in summary for token in ["prefers", "likes", "favorite", "always", "usually"]):
            boost += 0.15

        return max(0.0, min(1.0, base + boost))

    def score(self, memory, query_context):
        """Score one memory for current turn relevance.

        Blend the existing semantic, importance, and recency score with
        normalized FTS5 BM25 when the user supplied meaningful terms.
        """
        memory_key = self.bot.semantic_memory_key(memory)
        semantic_similarity = float((query_context.get("semantic_scores") or {}).get(memory_key, 0.0) or 0.0)
        graph_boost_tokens = set(query_context.get("graph_boost_tokens") or [])
        if graph_boost_tokens and graph_boost_tokens & self.bot.significant_tokens(str(memory.get("summary") or "")):
            semantic_similarity = min(1.0, semantic_similarity + 0.15)
        semantic_similarity = max(0.0, min(1.0, semantic_similarity))

        importance = self._importance_signal(memory)
        recency = self.semantic_memory_freshness_weight(memory)

        base_score = (0.6 * semantic_similarity) + (0.25 * importance) + (0.15 * recency)
        query_words = query_context.get("lexical_query_tokens")
        if query_words is None:
            query_words = self.lexical_query_tokens(str(query_context.get("user_message") or ""))
        lexical_overlap = float((query_context.get("lexical_scores") or {}).get(memory_key, 0.0))
        if query_words:
            base_score = MEMORY_DENSE_WEIGHT * base_score + (1.0 - MEMORY_DENSE_WEIGHT) * lexical_overlap
        return round(base_score, 6)

    @staticmethod
    def _confidence_bucket(score):
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def _record_memory_access_signals(self, selected, memories):
        if not selected:
            return

        today = date.today().isoformat()
        changed = False
        for score, memory in selected:
            try:
                memory["access_count"] = max(0, int(memory.get("access_count", 0) or 0)) + 1
            except (TypeError, ValueError):
                memory["access_count"] = 1

            history = memory.get("confidence_history")
            if not isinstance(history, dict):
                history = {}
            memory["confidence_history"] = {
                "high": max(0, int(history.get("high", 0) or 0)),
                "medium": max(0, int(history.get("medium", 0) or 0)),
                "low": max(0, int(history.get("low", 0) or 0)),
            }

            bucket = self._confidence_bucket(float(score or 0.0))
            memory["confidence_history"][bucket] += 1
            memory["high_confidence_hits"] = int(memory["confidence_history"].get("high", 0))
            memory["last_accessed_at"] = today
            changed = True

        if changed:
            self.bot.mutate_memory_store(memories=memories)

    def relevant_archive_entries_for_input(self, user_input, limit=2):
        query_tokens = self.bot.significant_tokens(user_input)
        query_category = self.bot.infer_memory_category(user_input)
        mood_trend = self.current_memory_mood_trend()
        recent_topics = set(self.recent_memory_topics(limit=4))
        scored = []

        for entry in reversed(self.bot.session_archive()):
            topics = [str(topic or "").strip().lower() for topic in entry.get("topics", []) if str(topic or "").strip()]
            searchable = f"{' '.join(topics)} {entry.get('summary', '')}"
            overlap = len(query_tokens & self.bot.significant_tokens(searchable))
            if query_category != "general" and query_category in topics:
                overlap += 2
            elif recent_topics and recent_topics.intersection(topics):
                overlap += 1
            if mood_trend != "neutral" and self.bot.normalize_mood(entry.get("dominant_mood")) == mood_trend:
                overlap += 1

            freshness = self.memory_freshness_weight(entry.get("created_at"))
            score = overlap * max(0.45, freshness)
            if score > 0:
                scored.append((round(score, 4), entry))

        ranked = sorted(
            scored,
            key=lambda item: (
                item[0],
                item[1].get("created_at", ""),
                item[1].get("summary", ""),
            ),
            reverse=True,
        )
        return [entry for _, entry in ranked[: max(1, int(limit or 2))]]

    def relevant_memories_for_input(self, user_input, limit=5, graph_result=None):
        started = perf_counter()
        stage_seconds = {}
        requested_limit = int(limit or 5)
        effective_limit = max(1, min(7, requested_limit))
        if requested_limit >= 3:
            effective_limit = max(3, effective_limit)
        stage_start = perf_counter()
        memories = self.bot.memory_catalog()
        stage_seconds["catalog"] = perf_counter() - stage_start
        if not memories:
            self._lexical_index.scores([], set())
            self.bot._last_memory_retrieval_diagnostics = {
                "retrieved_count": 0,
                "top_score_gap": 0.0,
                "has_high_confidence": False,
                "catalog_count": 0,
                "total_seconds": round(perf_counter() - started, 6),
            }
            return []

        stage_start = perf_counter()
        graph_result = (
            graph_result
            if graph_result is not None
            else self.bot.graph_retrieval_for_input(
                user_input,
                limit=min(3, max(1, effective_limit)),
            )
        )
        stage_seconds["graph"] = perf_counter() - stage_start
        graph_boost_tokens = set()
        if graph_result:
            for evidence in graph_result.get("supporting_evidence", []):
                graph_boost_tokens.update(
                    self.bot.significant_tokens(evidence.get("summary", "")),
                )
                graph_boost_tokens.update(
                    self.bot.significant_tokens(
                        " ".join(evidence.get("matched_nodes", [])),
                    ),
                )

        recent_turns = self._recent_turn_context(max_turns=3)
        personality_state = str(self.bot.last_saved_mood() or "neutral").strip().lower() or "neutral"
        # Metadata labels must not displace user terms in the candidate filter.
        # Recent turns remain available to contextual scoring below.
        semantic_query = str(user_input or "").strip()

        stage_start = perf_counter()
        semantic_manager = self.bot.memory_manager._semantic_manager
        semantic_manager._last_retrieval_status = {}
        semantic_matches = list(
            self.bot.semantic_memory_matches(
                semantic_query,
                memories,
                # Do not discard dense evidence before BM25 fusion can rank it.
                limit=len(memories),
            )
            or []
        )
        stage_seconds["semantic"] = perf_counter() - stage_start
        semantic_status = dict(semantic_manager._last_retrieval_status)
        max_semantic = max((float(score or 0.0) for score, _memory in semantic_matches), default=0.0)
        semantic_scores = {}
        for semantic_score, memory in semantic_matches:
            key = self.bot.semantic_memory_key(memory)
            if max_semantic > 0:
                semantic_scores[key] = max(0.0, min(1.0, float(semantic_score) / max_semantic))
            else:
                semantic_scores[key] = 0.0

        stage_start = perf_counter()
        bounded = bool(semantic_status.get("bounded"))
        candidate_limit = self.bot.runtime_config.memory_candidate_limit
        lexical_tokens = self.lexical_query_tokens(str(user_input or ""))
        lexical_scores = self._lexical_index.scores(
            [(self.bot.semantic_memory_key(memory), str(memory.get("summary") or "")) for memory in memories],
            lexical_tokens,
            limit=candidate_limit if bounded else None,
        )
        stage_seconds["lexical"] = perf_counter() - stage_start
        query_context = {
            "user_message": str(user_input or ""),
            "recent_turn_context": recent_turns,
            "personality_state": personality_state,
            "semantic_scores": semantic_scores,
            "lexical_query_tokens": lexical_tokens,
            "query_tokens": self.bot.significant_tokens(f"{user_input or ''!s} {recent_turns}"),
            "graph_boost_tokens": graph_boost_tokens,
            "lexical_scores": lexical_scores,
        }

        stage_start = perf_counter()
        candidate_keys = set(semantic_scores) | set(lexical_scores)
        graph_candidate_keys = set()
        if bounded and graph_boost_tokens:
            graph_ranked = heapq.nsmallest(
                candidate_limit,
                (
                    (
                        len(graph_boost_tokens & self.bot.significant_tokens(str(memory.get("summary") or ""))),
                        self.bot.semantic_memory_key(memory),
                    )
                    for memory in memories
                ),
                key=lambda item: (-item[0], item[1]),
            )
            graph_candidate_keys = {key for overlap, key in graph_ranked if overlap}
            candidate_keys.update(graph_candidate_keys)
        candidates = [
            memory for memory in memories
            if not bounded or self.bot.semantic_memory_key(memory) in candidate_keys
        ]
        ranked = sorted(
            ((self.score(memory, query_context), memory) for memory in candidates if str(memory.get("summary") or "").strip()),
            key=lambda item: (
                float(item[0]),
                item[1].get("updated_at", ""),
                item[1].get("summary", ""),
            ),
            reverse=True,
        )
        min_relevance = 0.18
        filtered_ranked = [item for item in ranked if float(item[0]) >= min_relevance]
        if not filtered_ranked and ranked:
            filtered_ranked = ranked[:1]

        stage_seconds["ranking"] = perf_counter() - stage_start
        stage_start = perf_counter()
        selected, reranker_diagnostics = self.rerank_memories(
            str(user_input or ""), filtered_ranked, effective_limit,
        )
        stage_seconds["reranking"] = perf_counter() - stage_start
        selected_scores = sorted((float(score) for score, _ in selected), reverse=True)
        top_score = selected_scores[0] if selected_scores else 0.0
        second_score = selected_scores[1] if len(selected_scores) > 1 else 0.0
        stage_start = perf_counter()
        self._record_memory_access_signals(selected, memories)
        stage_seconds["access_update"] = perf_counter() - stage_start
        self.bot._last_memory_retrieval_diagnostics = {
            "retrieved_count": len(selected),
            "top_score_gap": round(max(0.0, top_score - second_score), 4),
            "has_high_confidence": bool(top_score >= 0.7),
            "top_score": round(top_score, 4),
            "catalog_count": len(memories),
            "candidate_count": len(candidates),
            "lexical_candidate_count": len(lexical_scores),
            "graph_candidate_count": len(graph_candidate_keys),
            "semantic_status": semantic_status,
            "stage_seconds": {key: round(value, 6) for key, value in stage_seconds.items()},
            "total_seconds": round(perf_counter() - started, 6),
            **reranker_diagnostics,
        }
        return [memory for _score, memory in selected]

    def get_retrieval_diagnostics(self) -> dict:
        """Return a copy of the diagnostics from the most recent retrieval call.

        This is the single authorised accessor for retrieval diagnostics.
        Callers (PromptAssemblyManager, context.py) MUST use this method instead
        of reading ``bot._last_memory_retrieval_diagnostics`` directly, which is
        a private implementation detail of this class.
        """
        return dict(getattr(self.bot, "_last_memory_retrieval_diagnostics", {}) or {})

    @staticmethod
    def retrieval_strategy_for_input(user_input):
        text = str(user_input or "").strip().lower()
        if not text:
            return "hybrid"
        if any(token in text for token in ["pattern", "trend", "chapter", "timeline", "arc"]):
            return "graph_heavy"
        if any(
            token in text
            for token in [
                "remember",
                "recall",
                "what do you remember",
                "last time",
                "before",
            ]
        ):
            return "consolidated_heavy"
        if any(token in text for token in ["exactly", "specific", "detail", "quote", "verbatim"]):
            return "semantic_heavy"
        return "hybrid"

    def retrieve_context(self, user_input, strategy="hybrid", limit=4):
        requested = str(strategy or "hybrid").strip().lower() or "hybrid"
        effective_strategy = self.retrieval_strategy_for_input(user_input) if requested == "auto" else requested
        try:
            effective_limit = max(1, int(limit or 4))
        except (TypeError, ValueError):
            effective_limit = 4

        graph_limit = min(4, effective_limit)
        semantic_limit = min(5, max(1, effective_limit))
        archive_limit = min(3, effective_limit)
        consolidated_limit = min(4, effective_limit)

        if effective_strategy == "graph_heavy":
            graph_limit = min(5, max(2, effective_limit))
            semantic_limit = max(1, effective_limit - 1)
        elif effective_strategy == "semantic_heavy":
            semantic_limit = min(6, max(2, effective_limit + 1))
            graph_limit = max(1, effective_limit - 2)
        elif effective_strategy == "consolidated_heavy":
            consolidated_limit = min(5, max(2, effective_limit))
            graph_limit = max(1, effective_limit - 2)

        graph_result = self.bot.graph_retrieval_for_input(user_input, limit=graph_limit)
        archive_entries = (
            []
            if graph_result
            else self.relevant_archive_entries_for_input(
                user_input,
                limit=archive_limit,
            )
        )
        semantic_memories = self.relevant_memories_for_input(
            user_input,
            limit=semantic_limit,
            graph_result=graph_result or {},
        )
        consolidated_memories = self.bot.select_active_consolidated_memories(
            user_input,
            max_items=consolidated_limit,
        )

        bundle = []
        for memory in semantic_memories:
            freshness = self.memory_freshness_weight(
                memory.get("updated_at") or memory.get("created_at"),
            )
            impact_importance = min(
                1.0,
                max(0.0, self.memory_impact_score(memory) / 3.0),
            )
            importance = self.memory_importance_score(memory)
            score = 0.45 * freshness + 0.25 * impact_importance + 0.3 * importance
            bundle.append(
                {"type": "semantic", "score": round(score, 4), "payload": memory},
            )

        for entry in consolidated_memories:
            confidence = max(0.05, min(1.0, float(entry.get("confidence", 0.5) or 0.5)))
            importance = max(
                0.0,
                min(1.0, float(entry.get("importance_score", 0.0) or 0.0)),
            )
            recency = self.memory_freshness_weight(entry.get("updated_at"))
            score = 0.35 * confidence + 0.4 * importance + 0.25 * recency
            bundle.append(
                {"type": "consolidated", "score": round(score, 4), "payload": entry},
            )

        if graph_result:
            bundle.append(
                {
                    "type": "graph",
                    "score": 0.72,
                    "payload": {
                        "summary_lines": list(graph_result.get("summary_lines", []))[:4],
                        "compressed_summary": graph_result.get(
                            "compressed_summary",
                            "",
                        ),
                    },
                },
            )

        for entry in archive_entries:
            freshness = self.memory_freshness_weight(entry.get("created_at"))
            bundle.append(
                {
                    "type": "archive",
                    "score": round(0.55 * freshness, 4),
                    "payload": entry,
                },
            )

        ranked_bundle = sorted(
            bundle,
            key=lambda item: (item.get("score", 0.0), str(item.get("type", ""))),
            reverse=True,
        )
        return {
            "strategy": effective_strategy,
            "semantic_memories": semantic_memories,
            "consolidated_memories": consolidated_memories,
            "graph_result": graph_result,
            "archive_entries": archive_entries,
            "bundle": ranked_bundle[: max(3, effective_limit + 1)],
        }

    @staticmethod
    def format_memories_for_reply(memories):
        if not memories:
            return ""

        grouped = {}
        for memory in memories:
            category = memory.get("category", "general").title()
            grouped.setdefault(category, []).append(memory["summary"].rstrip("."))

        parts = []
        for category, summaries in grouped.items():
            parts.append(f"{category}: {'; '.join(summaries)}")

        return " | ".join(parts)

    def get_memory_reply(self, user_input):
        message = user_input.lower()

        if not any(
            term in message
            for term in [
                "remember",
                "recall",
                "what do you remember",
                "last time",
                "before",
            ]
        ):
            return None

        memories = self.relevant_memories_for_input(user_input)
        if not memories:
            return None

        remembered_points = self.format_memories_for_reply(memories)
        return f"Yeah, buddy, I remember a few things you've shared with me before. {remembered_points}"

    def find_memory_matches(self, query):
        query_tokens = self.bot.tokenize(query)
        matches = []

        for memory in self.bot.memory_catalog():
            searchable = f"{memory.get('category', '')} {memory.get('summary', '')}"
            overlap = query_tokens & self.bot.tokenize(searchable)
            if overlap:
                matches.append((len(overlap), memory))

        matches.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in matches]

    def add_memory(self, summary, category=None, *, oral_history_evidence=None):
        today_stamp = date.today().isoformat()
        normalized = self.bot.normalize_memory_entry(
            {
                "summary": summary,
                "category": category,
                "oral_history_evidence": oral_history_evidence,
                "mood": self.bot.last_saved_mood(),
                "created_at": today_stamp,
                "updated_at": today_stamp,
            },
        )
        if normalized is None:
            return None

        existing = self.bot.memory_catalog()
        normalized_lookup = {self.bot.normalize_memory_text(memory["summary"]): memory for memory in existing}
        key = self.bot.normalize_memory_text(normalized["summary"])

        if key in normalized_lookup:
            memory = normalized_lookup[key]
            memory["updated_at"] = today_stamp
            memory["category"] = normalized["category"]
            if normalized["category"] == "oral_history":
                memory["summary"] = normalized["summary"]
            if "oral_history_evidence" in normalized:
                memory["oral_history_evidence"] = normalized["oral_history_evidence"]
            else:
                memory.pop("oral_history_evidence", None)
            if normalized.get("mood"):
                memory["mood"] = normalized["mood"]
            self.bot.mutate_memory_store(memories=existing)
            self.bot.queue_semantic_memory_index(existing)
            return memory

        existing.append(normalized)
        self.bot.save_memory_catalog(existing)
        return normalized

    def update_memory_entry(
        self,
        original_summary,
        new_summary,
        category=None,
        mood=None,
    ):
        current_catalog = self.bot.memory_catalog()
        target_key = self.bot.normalize_memory_text(original_summary)
        updated_memories = []
        target_found = False

        for memory in current_catalog:
            if self.bot.normalize_memory_text(memory.get("summary", "")) != target_key:
                updated_memories.append(memory)
                continue

            replacement = self.bot.normalize_memory_entry(
                {
                    "summary": new_summary,
                    "category": category if category not in {None, ""} else memory.get("category"),
                    "mood": mood if mood not in {None, ""} else memory.get("mood"),
                    "created_at": memory.get("created_at") or date.today().isoformat(),
                    "updated_at": date.today().isoformat(),
                },
            )
            if replacement is None:
                return None

            updated_memories.append(replacement)
            target_found = True

        if not target_found:
            return None

        self.bot.save_memory_catalog(updated_memories)
        return next(
            (
                memory
                for memory in self.bot.memory_catalog()
                if self.bot.normalize_memory_text(memory.get("summary", ""))
                == self.bot.normalize_memory_text(new_summary)
            ),
            None,
        )

    def delete_memory_entry(self, summary):
        target_key = self.bot.normalize_memory_text(summary)
        current_catalog = self.bot.memory_catalog()
        kept = []
        removed = []

        for memory in current_catalog:
            if self.bot.normalize_memory_text(memory.get("summary", "")) == target_key:
                removed.append(memory)
            else:
                kept.append(memory)

        if not removed:
            return []

        self.bot.save_memory_catalog(kept)
        return removed

    def forget_memories(self, query):
        matches = self.find_memory_matches(query)
        if not matches:
            return []

        match_keys = {self.bot.normalize_memory_text(memory["summary"]) for memory in matches}
        current_catalog = self.bot.memory_catalog()
        kept = [
            memory for memory in current_catalog if self.bot.normalize_memory_text(memory["summary"]) not in match_keys
        ]
        removed = [
            memory for memory in current_catalog if self.bot.normalize_memory_text(memory["summary"]) in match_keys
        ]
        self.bot.save_memory_catalog(kept)
        return removed


__all__ = ["MemoryQueryManager"]
