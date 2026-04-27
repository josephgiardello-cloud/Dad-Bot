from __future__ import annotations

import json

from dadbot.contracts import DadBotContext, SupportsContextRuntime


class ContextBuilder:
    """Owns prompt-facing context sections that compose profile, relationship, memory, and long-term signals."""

    def __init__(self, bot: DadBotContext | SupportsContextRuntime):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    def build_core_persona_prompt(self) -> str:
        behavior_rules = "\n".join(f"- {rule}" for rule in self.bot.profile_runtime.effective_behavior_rules())
        return f"""
You are '{self.bot.STYLE['name']}'.
You are warm, encouraging, informal, supportive, and honest.
Keep most replies brief and easy to read in a terminal.
Default to 2-4 sentences unless Tony asks for more detail.
Prefer one clear thought at a time over long multi-part answers.

Behavior rules:
{behavior_rules}

Never contradict established profile facts.
If asked about a personal fact that is not in the profile, say you do not want to guess rather than inventing a detail.
""".strip()

    def build_dynamic_profile_context(self) -> str:
        previous_mood = self.bot.last_saved_mood()
        prior_mood_context = ""

        if previous_mood != "neutral":
            prior_mood_context = (
                f"\nPrevious conversation continuity:\n"
                f"- Tony's last remembered mood was {previous_mood}.\n"
                f"- If his current tone is similar, respond with gentle continuity, but do not assume he feels exactly the same unless he signals it.\n"
            )

        return f"""
Full profile facts:
{self.bot.profile_context.build_profile_block()}
{prior_mood_context}

If prior conversation memories are provided, treat them as real things Tony has shared with you before.
Keep family relationships, ages, timelines, and education history consistent with the profile.
""".strip()

    def build_relationship_context(self) -> str:
        return self.bot.relationship_manager.build_prompt_context()

    def build_session_summary_context(self) -> str | None:
        if not self.bot.session_summary:
            return None

        return (
            "Rolling summary of earlier turns in this chat:\n"
            f"{self.bot.session_summary}\n"
            "Treat this as conversation context from the same ongoing session."
        )

    def build_relevant_context(self, user_input: str) -> str | None:
        fact_ids = self.bot.profile_context.relevant_fact_ids_for_input(user_input)

        if not fact_ids:
            return None

        return (
            "Relevant profile facts for this specific user message:\n"
            f"{self.bot.profile_context.build_profile_block(fact_ids)}\n"
            "If any part of your answer touches these facts, keep them exactly consistent."
        )

    def build_wisdom_context(self, user_input: str) -> str | None:
        return self.bot.long_term_signals.build_wisdom_context(user_input)

    def _fit_sections_to_token_budget(self, sections: list[str], budget: int) -> list[str]:
        """Drop lower-priority trailing sections until the joined result fits within *budget* tokens.

        Sections are ordered highest-priority first. The footer sentence is
        always appended after trimming, so it is not counted against the budget.
        """
        if not sections:
            return sections
        kept = []
        running_tokens = 0
        for section in sections:
            section_tokens = self.bot.estimate_token_count(section)
            if running_tokens + section_tokens > budget and kept:
                # Budget exceeded â€” drop this and all remaining lower-priority sections.
                break
            kept.append(section)
            running_tokens += section_tokens
        return kept

    def _record_memory_context_stats(
        self,
        *,
        user_input: str,
        tokens: int,
        selected_sections: int,
        total_sections: int,
        pruned: bool,
    ) -> None:
        self.bot.record_memory_context_stats(
            tokens=tokens,
            budget_tokens=max(1, int(self.bot.effective_context_token_budget(self.bot.ACTIVE_MODEL) or self.bot.CONTEXT_TOKEN_BUDGET or 1)),
            selected_sections=selected_sections,
            total_sections=total_sections,
            pruned=pruned,
            user_input=user_input,
        )

    def _build_memory_sections(
        self,
        *,
        deep_pattern_context: str | None,
        consolidated_context: str | None,
        graph_result: dict | None,
        archive_entries: list[dict],
        memories: list[dict],
    ) -> list[str]:
        sections: list[str] = []
        if deep_pattern_context:
            sections.append(deep_pattern_context)
        if consolidated_context:
            sections.append(consolidated_context)
        if graph_result:
            graph_text = graph_result.get("compressed_summary") or "\n".join(graph_result.get("summary_lines", []))
            sections.append("Graph-connected long-term context most relevant right now:\n" + graph_text)
        if archive_entries:
            archive_lines = "\n".join(
                f"- [{entry.get('created_at', '')[:10]}, mood={entry.get('dominant_mood', 'neutral')}] {entry.get('summary', '')}"
                for entry in archive_entries
            )
            sections.append("Recent prior session notes most relevant right now:\n" + archive_lines)
        if memories:
            memory_lines = "\n".join(
                f"- [{memory.get('category', 'general')}, mood={memory.get('mood', 'unknown')}] {memory['summary']}"
                for memory in memories
            )
            sections.append("Semantic fallback from older remembered details with Tony:\n" + memory_lines)
        return sections

    def _trim_memory_sections_to_budget(self, sections: list[str]) -> tuple[list[str], int, int, int]:
        total_sections = len(sections)
        pre_trim_tokens = sum(self.bot.estimate_token_count(section) for section in sections)
        baseline_memory_budget = min(800, max(120, self.bot.CONTEXT_TOKEN_BUDGET // 4))
        memory_token_budget = self.bot.adaptive_memory_context_budget(baseline_memory_budget)
        trimmed_sections = self._fit_sections_to_token_budget(sections, memory_token_budget)
        post_trim_tokens = sum(self.bot.estimate_token_count(section) for section in trimmed_sections)
        return trimmed_sections, total_sections, pre_trim_tokens, post_trim_tokens

    def _recent_turn_messages(self, *, max_turns: int = 12) -> list[dict]:
        history = list(self.bot.conversation_history())
        if not history:
            return []
        # Keep the most recent user+assistant turns at full fidelity.
        max_messages = max(2, int(max_turns or 12) * 2)
        return [dict(message) for message in history[-max_messages:] if isinstance(message, dict)]

    @staticmethod
    def _clip_message_text(text: str, *, limit: int = 220) -> str:
        normalized = str(text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(1, limit - 3)].rstrip() + "..."

    def _build_recent_buffer_context(self, *, max_turns: int = 12) -> str | None:
        recent_messages = self._recent_turn_messages(max_turns=max_turns)
        if not recent_messages:
            return None

        lines: list[str] = []
        for message in recent_messages:
            role = str(message.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = self._clip_message_text(str(message.get("content") or ""))
            if not content:
                continue
            speaker = "Tony" if role == "user" else "Dad"
            lines.append(f"- {speaker}: {content}")

        if not lines:
            return None
        return "Recent conversation buffer (highest priority):\n" + "\n".join(lines)

    @staticmethod
    def _structured_claim_type(category: str, summary: str) -> str:
        text = f"{category} {summary}".lower()
        if any(token in text for token in ["prefer", "likes", "favorite", "wants", "enjoys"]):
            return "preferences"
        if any(token in text for token in ["decided", "decision", "will", "chose", "plan"]):
            return "decisions"
        if any(token in text for token in ["goal", "working on", "trying to", "aim", "target"]):
            return "goals"
        if any(token in text for token in ["cannot", "can't", "constraint", "limited", "budget", "deadline", "schedule"]):
            return "constraints"
        return "entities"

    def _build_structured_claims_context(self, user_input: str, *, max_items: int = 8) -> tuple[str | None, list[dict[str, object]]]:
        claims: list[dict[str, object]] = []
        for entry in self.bot.select_active_consolidated_memories(user_input, max_items=max_items):
            summary = str(entry.get("summary") or "").strip()
            if not summary:
                continue
            category = str(entry.get("category") or "general").strip().lower() or "general"
            claim = {
                "type": self._structured_claim_type(category, summary),
                "summary": summary,
                "category": category,
                "mood": str(entry.get("mood") or "neutral"),
                "confidence": round(float(entry.get("confidence", 0.0) or 0.0), 3),
                "importance": round(float(entry.get("importance_score", 0.0) or 0.0), 3),
                "updated_at": str(entry.get("updated_at") or ""),
            }
            claims.append(claim)

        if not claims:
            return None, []

        payload = json.dumps(claims[: max(1, int(max_items or 8))], ensure_ascii=True)
        return "Structured memory claims (JSON):\n" + payload, claims

    def build_hierarchical_memory_context(self, user_input: str) -> tuple[str | None, dict[str, object]]:
        recent_section = self._build_recent_buffer_context(max_turns=12)
        summary_section = self.build_session_summary_context()
        structured_section, structured_claims = self._build_structured_claims_context(user_input, max_items=8)

        prioritized_sections = [
            ("recent", recent_section),
            ("summary", summary_section),
            ("structured", structured_section),
        ]
        section_lookup = {name: section for name, section in prioritized_sections if section}
        sections = [section for _name, section in prioritized_sections if section]
        if not sections:
            return None, {
                "recent_tokens": 0,
                "summary_tokens": 0,
                "structured_tokens": 0,
                "structured_claims": [],
                "selected_layers": [],
            }

        memory_token_budget = min(1400, max(220, int(self.bot.CONTEXT_TOKEN_BUDGET or 6000) // 3))
        trimmed_sections = self._fit_sections_to_token_budget(sections, memory_token_budget)

        selected_layers: list[str] = []
        for layer_name in ("recent", "summary", "structured"):
            layer_text = section_lookup.get(layer_name)
            if layer_text and layer_text in trimmed_sections:
                selected_layers.append(layer_name)

        layer_stats = {
            "recent_tokens": self.bot.estimate_token_count(section_lookup.get("recent") or "") if "recent" in selected_layers else 0,
            "summary_tokens": self.bot.estimate_token_count(section_lookup.get("summary") or "") if "summary" in selected_layers else 0,
            "structured_tokens": self.bot.estimate_token_count(section_lookup.get("structured") or "") if "structured" in selected_layers else 0,
            "structured_claims": list(structured_claims),
            "selected_layers": selected_layers,
        }

        if not trimmed_sections:
            return None, layer_stats

        context_text = "\n\n".join(trimmed_sections + ["Use these only as prior context Tony has already shared."])
        return context_text, layer_stats

    def build_memory_context(self, user_input: str) -> str | None:
        """Build query-aware memory context with graph, archive, consolidated, and semantic sections."""
        user_input_str = str(user_input or "").strip()
        
        sections = []
        
        # 1. Try to get graph and archive results
        graph_result = None
        archive_entries = []
        semantic_memories = []
        
        try:
            graph_result = self.bot.graph_retrieval_for_input(user_input_str, limit=3) if user_input_str else None
        except Exception:
            pass
        
        try:
            if not graph_result and user_input_str:
                archive_entries = list(self.bot.relevant_archive_entries_for_input(user_input_str, limit=2) or [])
        except Exception:
            pass
        
        try:
            if user_input_str:
                semantic_memories = list(self.bot.relevant_memories_for_input(user_input_str, limit=3, graph_result=graph_result) or [])
        except Exception:
            pass
        
        # 2. Get consolidated and deep pattern context
        deep_pattern_context = None
        consolidated_context = None
        
        if user_input_str:
            try:
                deep_pattern_context = self.bot.long_term_signals.build_deep_pattern_context(user_input_str)
            except Exception:
                pass
            try:
                consolidated_context = self.bot.build_active_consolidated_context(user_input_str)
            except Exception:
                pass
        
        # 3. Build all sections using _build_memory_sections
        if deep_pattern_context or consolidated_context or graph_result or archive_entries or semantic_memories:
            sections = self._build_memory_sections(
                deep_pattern_context=deep_pattern_context,
                consolidated_context=consolidated_context,
                graph_result=graph_result,
                archive_entries=archive_entries,
                memories=semantic_memories,
            )
        
        # 4. If no sections from query-aware retrieval, fall back to hierarchical memory
        if not sections and user_input_str:
            context_text, layer_stats = self.build_hierarchical_memory_context(user_input_str)
            if context_text:
                self.bot._last_hierarchical_memory_stats = dict(layer_stats)
                self._record_memory_context_stats(
                    user_input=user_input_str,
                    tokens=self.bot.estimate_token_count(context_text),
                    selected_sections=1,
                    total_sections=1,
                    pruned=False,
                )
                return context_text
            return None
        
        if not sections:
            return None
        
        # 5. Trim sections to budget
        try:
            trimmed_sections, total_sections, pre_trim_tokens, post_trim_tokens = self._trim_memory_sections_to_token_budget(sections)
        except Exception:
            # If trimming fails, just use all sections
            trimmed_sections = sections
            total_sections = len(sections)
            pre_trim_tokens = sum(self.bot.estimate_token_count(s) for s in sections)
            post_trim_tokens = pre_trim_tokens
        
        if not trimmed_sections:
            return None
        
        combined_context = "\n\n".join(trimmed_sections)
        
        self._record_memory_context_stats(
            user_input=user_input_str,
            tokens=post_trim_tokens,
            selected_sections=len(trimmed_sections),
            total_sections=total_sections,
            pruned=(len(trimmed_sections) < total_sections) or (post_trim_tokens < pre_trim_tokens),
        )
        
        # Store stats for context service
        layer_stats = {
            "recent_tokens": 0,
            "summary_tokens": 0,
            "structured_tokens": 0,
            "structured_claims": [],
            "selected_layers": [],
        }
        self.bot._last_hierarchical_memory_stats = dict(layer_stats)
        
        return combined_context

    def _cross_session_traits_section(self) -> str | None:
        evolution_history = self.bot.persona_evolution_history()
        if not evolution_history:
            return None
        prioritized_traits = sorted(
            self.bot.active_persona_trait_entries(limit=4),
            key=lambda entry: (
                self.bot.long_term_signals.trait_impact(entry) > 0,
                self.bot.long_term_signals.trait_impact(entry),
                self.bot.long_term_signals.decayed_trait_strength(entry),
                str(entry.get("applied_at", "")),
            ),
            reverse=True,
        )
        trait_lines = [
            f"- {entry.get('trait', '')} (impact={self.bot.long_term_signals.trait_impact(entry):.2f}, strength={self.bot.long_term_signals.decayed_trait_strength(entry):.2f})"
            for entry in prioritized_traits
            if entry.get("trait")
        ]
        if not trait_lines:
            return None
        return "Dad's strongest evolved traits with Tony right now:\n" + "\n".join(trait_lines)

    def _cross_session_patterns_section(self) -> str | None:
        patterns = self.bot.life_patterns()
        if not patterns:
            return None
        pattern_lines = "\n".join(
            f"- [{pattern.get('confidence', 0)}%] {pattern.get('summary', '')}"
            for pattern in patterns[-3:]
        )
        return "Long-term life patterns Dad has noticed:\n" + pattern_lines

    def _cross_session_timeline_section(self) -> str | None:
        timeline = self.bot.relationship_timeline()
        if not timeline:
            return None
        return f"Long-term relationship digest from prior chats:\n{timeline}"

    def _cross_session_archive_section(self) -> str | None:
        archive = self.bot.session_archive()
        if not archive:
            return None
        recent_notes = []
        for entry in archive[-3:]:
            topics = ", ".join(entry.get("topics", [])) or "general"
            recent_notes.append(
                f"- {entry.get('created_at', '')[:10]} | mood={entry.get('dominant_mood', 'neutral')} | topics={topics}: {entry.get('summary', '')}"
            )
        return "Recent prior session notes:\n" + "\n".join(recent_notes)

    def _cross_session_deep_and_consolidated_sections(self, user_input: str) -> list[str]:
        if str(user_input or "").strip():
            deep_pattern_context = self.bot.long_term_signals.build_deep_pattern_context(str(user_input))
            consolidated_context = self.bot.build_active_consolidated_context(str(user_input))
        else:
            deep_pattern_context = None
            consolidated_context = self.bot.build_consolidated_memory_context()
        sections = []
        if deep_pattern_context:
            sections.append(deep_pattern_context)
        if consolidated_context:
            sections.append(consolidated_context)
        return sections

    def _cross_session_graph_summary_section(self) -> str | None:
        return self.bot.build_graph_summary_context(limit=3)

    def _cross_session_heritage_section(self, user_input: str) -> str | None:
        if not str(user_input or "").strip():
            return None
        try:
            from dadbot.managers.heritage_graph import HeritageGraphManager

            return HeritageGraphManager(self.bot).heritage_context_block(str(user_input))
        except Exception:
            return None

    @staticmethod
    def _append_if_present(sections: list[str], section: str | None) -> None:
        if section:
            sections.append(section)

    def build_cross_session_context(self, user_input: str = "") -> str | None:
        sections: list[str] = []
        self._append_if_present(sections, self._cross_session_traits_section())
        self._append_if_present(sections, self._cross_session_patterns_section())
        self._append_if_present(sections, self._cross_session_timeline_section())
        self._append_if_present(sections, self._cross_session_archive_section())

        sections.extend(self._cross_session_deep_and_consolidated_sections(user_input))
        self._append_if_present(sections, self._cross_session_graph_summary_section())
        self._append_if_present(sections, self._cross_session_heritage_section(user_input))

        if not sections:
            return None

        return "\n\n".join(sections)


__all__ = ["ContextBuilder"]
