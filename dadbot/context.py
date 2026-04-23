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

    def build_memory_context(self, user_input: str) -> str | None:
        memory_limit = self.bot.memory_context_limit_for_input(user_input)
        retrieval = self.bot.retrieve_context(user_input, strategy="hybrid", limit=memory_limit)
        graph_result = retrieval.get("graph_result")
        archive_entries = retrieval.get("archive_entries") or []
        memories = retrieval.get("semantic_memories") or []
        deep_pattern_context = self.bot.long_term_signals.build_deep_pattern_context(user_input)
        consolidated_context = self.bot.build_active_consolidated_context(user_input)

        if not deep_pattern_context and not consolidated_context and not graph_result and not archive_entries and not memories:
            self.bot.record_memory_context_stats(
                tokens=0,
                budget_tokens=max(1, int(self.bot.effective_context_token_budget(self.bot.ACTIVE_MODEL) or self.bot.CONTEXT_TOKEN_BUDGET or 1)),
                selected_sections=0,
                total_sections=0,
                pruned=False,
                user_input=user_input,
            )
            return None

        sections = []
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

        # Trim sections to a token budget so that a large memory store never
        # displaces conversation history.  Budget = 25 % of the context window,
        # capped at 800 tokens.  Sections are already in priority order so the
        # lowest-priority ones (semantic fallback) are dropped first.
        total_sections = len(sections)
        pre_trim_tokens = sum(self.bot.estimate_token_count(section) for section in sections)
        baseline_memory_budget = min(800, max(120, self.bot.CONTEXT_TOKEN_BUDGET // 4))
        memory_token_budget = self.bot.adaptive_memory_context_budget(baseline_memory_budget)
        sections = self._fit_sections_to_token_budget(sections, memory_token_budget)
        post_trim_tokens = sum(self.bot.estimate_token_count(section) for section in sections)
        self.bot.record_memory_context_stats(
            tokens=post_trim_tokens,
            budget_tokens=max(1, int(self.bot.effective_context_token_budget(self.bot.ACTIVE_MODEL) or self.bot.CONTEXT_TOKEN_BUDGET or 1)),
            selected_sections=len(sections),
            total_sections=total_sections,
            pruned=(len(sections) < total_sections) or (post_trim_tokens < pre_trim_tokens),
            user_input=user_input,
        )

        if not sections:
            return None

        return "\n\n".join(sections + ["Use these only as prior context Tony has already shared."])

    def build_cross_session_context(self, user_input: str = "") -> str | None:
        sections = []

        evolution_history = self.bot.persona_evolution_history()
        if evolution_history:
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
            if trait_lines:
                sections.append("Dad's strongest evolved traits with Tony right now:\n" + "\n".join(trait_lines))

        patterns = self.bot.life_patterns()
        if patterns:
            pattern_lines = "\n".join(
                f"- [{pattern.get('confidence', 0)}%] {pattern.get('summary', '')}"
                for pattern in patterns[-3:]
            )
            sections.append("Long-term life patterns Dad has noticed:\n" + pattern_lines)

        timeline = self.bot.relationship_timeline()
        if timeline:
            sections.append(f"Long-term relationship digest from prior chats:\n{timeline}")

        archive = self.bot.session_archive()
        if archive:
            recent_notes = []
            for entry in archive[-3:]:
                topics = ", ".join(entry.get("topics", [])) or "general"
                recent_notes.append(
                    f"- {entry.get('created_at', '')[:10]} | mood={entry.get('dominant_mood', 'neutral')} | topics={topics}: {entry.get('summary', '')}"
                )
            sections.append("Recent prior session notes:\n" + "\n".join(recent_notes))

        if str(user_input or "").strip():
            deep_pattern_context = self.bot.long_term_signals.build_deep_pattern_context(str(user_input))
            consolidated_context = self.bot.build_active_consolidated_context(str(user_input))
        else:
            deep_pattern_context = None
            consolidated_context = self.bot.build_consolidated_memory_context()
        if deep_pattern_context:
            sections.append(deep_pattern_context)
        if consolidated_context:
            sections.append(consolidated_context)

        graph_summary = self.bot.build_graph_summary_context(limit=3)
        if graph_summary:
            sections.append(graph_summary)

        # Heritage cross-links: connect present context to resonant past arcs
        if str(user_input or "").strip():
            try:
                from dadbot.managers.heritage_graph import HeritageGraphManager
                heritage_block = HeritageGraphManager(self.bot).heritage_context_block(str(user_input))
                if heritage_block:
                    sections.append(heritage_block)
            except Exception:
                pass

        if not sections:
            return None

        return "\n\n".join(sections)


__all__ = ["ContextBuilder"]