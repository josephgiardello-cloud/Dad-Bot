from __future__ import annotations

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
                # Budget exceeded — drop this and all remaining lower-priority sections.
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

    def build_memory_context(self, user_input: str) -> str | None:
        memory_limit = self.bot.memory_context_limit_for_input(user_input)
        retrieval = self.bot.retrieve_context(user_input, strategy="hybrid", limit=memory_limit)
        graph_result = retrieval.get("graph_result")
        archive_entries = retrieval.get("archive_entries") or []
        memories = retrieval.get("semantic_memories") or []
        deep_pattern_context = self.bot.long_term_signals.build_deep_pattern_context(user_input)
        consolidated_context = self.bot.build_active_consolidated_context(user_input)

        if not deep_pattern_context and not consolidated_context and not graph_result and not archive_entries and not memories:
            self._record_memory_context_stats(user_input=user_input, tokens=0, selected_sections=0, total_sections=0, pruned=False)
            return None

        sections = self._build_memory_sections(
            deep_pattern_context=deep_pattern_context,
            consolidated_context=consolidated_context,
            graph_result=graph_result,
            archive_entries=archive_entries,
            memories=memories,
        )

        # Trim sections to a token budget so that a large memory store never
        # displaces conversation history.  Budget = 25 % of the context window,
        # capped at 800 tokens.  Sections are already in priority order so the
        # lowest-priority ones (semantic fallback) are dropped first.
        sections, total_sections, pre_trim_tokens, post_trim_tokens = self._trim_memory_sections_to_budget(sections)
        self._record_memory_context_stats(
            user_input=user_input,
            tokens=post_trim_tokens,
            selected_sections=len(sections),
            total_sections=total_sections,
            pruned=(len(sections) < total_sections) or (post_trim_tokens < pre_trim_tokens),
        )

        if not sections:
            return None

        return "\n\n".join(sections + ["Use these only as prior context Tony has already shared."])

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