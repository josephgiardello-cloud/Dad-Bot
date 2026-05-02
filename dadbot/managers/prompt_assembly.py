from __future__ import annotations

from dadbot.contracts import AttachmentList, DadBotContext, SupportsDadBotAccess
from dadbot.core.turn_coherence import mark_turn_coherence


class PromptAssemblyManager:
    """Builds the layered system prompt and final chat request message list."""

    VISUAL_TASK_PROFILES = {
        "debug_screenshot": {
            "label": "Debug Screenshot",
            "context": "Treat the image like a screenshot or technical artifact. Prioritize visible error text, tracebacks, code fragments, UI state, and the next grounded troubleshooting step.",
            "analysis": "Focus on visible error text, stack traces, terminal output, app state, code snippets, and likely failing components. Avoid generic scenic description.",
        },
        "homework_help": {
            "label": "Homework Help",
            "context": "Treat the image like schoolwork. Prioritize visible instructions, equations, diagrams, labels, and what Tony is being asked to solve before offering help.",
            "analysis": "Read visible problem text, equations, diagrams, tables, labels, and instructions. Identify the subject and what is explicitly shown.",
        },
        "creative_feedback": {
            "label": "Creative Feedback",
            "context": "Treat the image like something Tony made or wants feedback on. Notice effort, subject, color, composition, and what deserves encouragement before critique.",
            "analysis": "Describe the main subject, colors, style, and effort in a supportive way. Notice what seems intentional or expressive.",
        },
        "repair_coach": {
            "label": "Repair Coach",
            "context": "Treat the image like a repair or DIY problem. Prioritize parts, materials, labels, damage points, tools, and any obvious safety-relevant detail.",
            "analysis": "Identify visible parts, tools, labels, damage points, wiring, leaks, fasteners, connectors, and safety-relevant details.",
        },
        "general_photo": {
            "label": "General Photo",
            "context": "Treat the image like a personal photo or general visual reference. Notice the scene, visible text, project context, and emotional tone without guessing hidden details.",
            "analysis": "Describe the scene, visible text, project context, and emotional tone in a grounded way.",
        },
    }

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self._turn_memory_context: str | None = None

    def _memory_confidence_label(self) -> tuple[str, str]:
        diagnostics = dict(getattr(self.bot, "_last_memory_retrieval_diagnostics", {}) or {})
        retrieved_count = int(diagnostics.get("retrieved_count", 0) or 0)
        top_score = float(diagnostics.get("top_score", 0.0) or 0.0)
        has_high_confidence = bool(diagnostics.get("has_high_confidence", False))

        if retrieved_count <= 0:
            return (
                "LOW",
                "Memory may be weak for this turn. Do not force memory into the answer unless it clearly fits.",
            )
        if has_high_confidence or top_score >= 0.7:
            return (
                "HIGH",
                "At least one memory is highly relevant. Let it influence factual content when answering.",
            )
        if top_score >= 0.4:
            return (
                "MEDIUM",
                "Memory is moderately relevant. Use it when it supports the current user request.",
            )
        return (
            "LOW",
            "Memory confidence is low. Prioritize current user input over older memory.",
        )

    def begin_turn_memory_context(
        self,
        user_input: str,
        *,
        user_id: str,
        session_id: str,
    ) -> str:
        """Retrieve memory context once at turn start and cache for prompt assembly."""
        raw_context = self.bot.context_builder.build_memory_context(str(user_input or ""))
        confidence_label, confidence_guidance = self._memory_confidence_label()
        if raw_context:
            section = (
                f"Memory context (user={user_id}, session={session_id}):\n"
                f"Memory confidence: {confidence_label}\n"
                f"{confidence_guidance}\n\n"
                f"{raw_context}"
            )
        else:
            section = (
                f"Memory context (user={user_id}, session={session_id}):\n"
                f"Memory confidence: LOW\n"
                "No prior memory context for this turn."
            )
        self._turn_memory_context = section
        mark_turn_coherence(self.bot, "memory_included")
        return section

    def _resolve_turn_memory_context(self, user_input: str) -> str:
        if self._turn_memory_context is None:
            self.begin_turn_memory_context(
                str(user_input or ""),
                user_id=str(getattr(self.bot, "TENANT_ID", "default") or "default"),
                session_id=str(getattr(self.bot, "active_thread_id", "default") or "default"),
            )
        return str(self._turn_memory_context or "Memory context:\nNo prior memory context for this turn.")

    def base_request_sections(self, current_mood: str) -> list[str | None]:
        return [
            self.bot.context_builder.build_core_persona_prompt(),
            self.bot.context_builder.build_dynamic_profile_context(),
            self.bot.context_builder.build_relationship_context(),
            self.bot.build_style_examples(),
            self.bot.personality_service.build_personality_context(current_mood),
        ]

    def infer_visual_task(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> dict[str, str] | None:
        image_attachments = [
            attachment
            for attachment in attachments or []
            if isinstance(attachment, dict) and attachment.get("type") == "image"
        ]
        if not image_attachments:
            return None

        combined = " ".join(
            str(part or "").strip()
            for attachment in image_attachments
            for part in [
                user_input,
                attachment.get("note"),
                attachment.get("analysis"),
                attachment.get("name"),
                attachment.get("mime_type"),
            ]
        ).lower()

        if any(
            token in combined
            for token in [
                "traceback",
                "stack trace",
                "terminal",
                "console",
                "error",
                "exception",
                "crash",
                "bug",
                "screenshot",
            ]
        ):
            return {
                "mode": "debug_screenshot",
                **self.VISUAL_TASK_PROFILES["debug_screenshot"],
            }
        if any(
            token in combined
            for token in [
                "homework",
                "worksheet",
                "equation",
                "solve",
                "assignment",
                "quiz",
                "math",
                "diagram",
                "study",
            ]
        ):
            return {
                "mode": "homework_help",
                **self.VISUAL_TASK_PROFILES["homework_help"],
            }
        if any(
            token in combined
            for token in [
                "draw",
                "drawing",
                "sketch",
                "painting",
                "art",
                "doodle",
                "made this",
                "coloring",
            ]
        ):
            return {
                "mode": "creative_feedback",
                **self.VISUAL_TASK_PROFILES["creative_feedback"],
            }
        if any(
            token in combined
            for token in [
                "fix",
                "broken",
                "repair",
                "leak",
                "wire",
                "wiring",
                "tool",
                "engine",
                "appliance",
                "install",
            ]
        ):
            return {"mode": "repair_coach", **self.VISUAL_TASK_PROFILES["repair_coach"]}
        return {"mode": "general_photo", **self.VISUAL_TASK_PROFILES["general_photo"]}

    def build_visual_task_context(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> str | None:
        profile = self.infer_visual_task(user_input, attachments)
        if profile is None:
            return None
        return (
            f"Visual mode for this turn: {profile['label']}.\n"
            f"{profile['context']}\n"
            "Use only what is actually visible or already described by Tony. If the image is ambiguous, say what you can and cannot see."
        )

    def contextual_request_sections(
        self,
        user_input: str,
        current_mood: str,
        attachments: AttachmentList | None = None,
    ) -> list[str | None]:
        return [
            self.bot.tone_context.build_daily_checkin_context(current_mood),
            self.bot.build_active_tool_observation_context(),
            self.build_visual_task_context(user_input, attachments),
            self.bot.context_builder.build_cross_session_context(user_input),
            self.bot.context_builder.build_session_summary_context(),
            self.bot.context_builder.build_relevant_context(user_input),
            self.bot.context_builder.build_wisdom_context(user_input),
            self._resolve_turn_memory_context(user_input),
            self.bot.tone_context.build_escalation_context(
                current_mood,
                self.bot.session_moods,
            ),
        ]

    def build_request_system_prompt(
        self,
        user_input: str,
        current_mood: str,
        attachments: AttachmentList | None = None,
    ) -> str:
        sections = self.base_request_sections(current_mood)
        sections.extend(
            self.contextual_request_sections(user_input, current_mood, attachments),
        )
        return "\n\n".join(section for section in sections if section)

    def build_image_analysis_prompt(
        self,
        note: str = "",
        user_input: str = "",
        attachment: dict | None = None,
    ) -> str:
        profile = self.infer_visual_task(
            user_input or note,
            [attachment] if attachment else None,
        )
        if profile is None:
            profile = {
                "mode": "general_photo",
                **self.VISUAL_TASK_PROFILES["general_photo"],
            }
        note_section = f"Tony's note about the image: {note}\n" if note else ""
        request_section = f"Tony's request: {user_input}\n" if user_input else ""
        return (
            "You are helping a warm dad chatbot understand an image.\n"
            f"Visual mode: {profile['label']}.\n"
            f"Task guidance: {profile['analysis']}\n"
            f"{note_section}"
            f"{request_section}"
            "Return only 2 short sentences. Mention visible text, labels, or emotional context if plainly visible, and do not guess identity-sensitive or hidden details."
        )

    def build_chat_request_messages(
        self,
        user_input: str,
        current_mood: str,
        attachments: AttachmentList | None = None,
    ) -> list[dict[str, object]]:
        system_prompt = self.build_request_system_prompt(
            user_input,
            current_mood,
            attachments,
        )
        request_messages = [{"role": "system", "content": system_prompt}]
        request_messages.extend(
            self.bot.token_budgeted_prompt_history(system_prompt, user_input),
        )
        request_messages.append(
            self.bot.build_user_request_message(user_input, attachments),
        )
        return request_messages

    # ------------------------------------------------------------------
    # guard_chat_request_messages pipeline helpers
    # ------------------------------------------------------------------

    def _compute_prompt_budget(self) -> int:
        """Compute the effective token budget for the prompt, applying pressure factor."""
        context_budget = max(
            256,
            int(
                self.bot.effective_context_token_budget(self.bot.ACTIVE_MODEL) or self.bot.CONTEXT_TOKEN_BUDGET or 0,
            ),
        )
        reserved_tokens = max(64, int(self.bot.RESERVED_RESPONSE_TOKENS or 0))
        prompt_budget = max(128, context_budget - reserved_tokens)
        pressure_factor = self.bot.adaptive_prompt_pressure_factor()
        if pressure_factor < 1.0:
            prompt_budget = max(96, int(prompt_budget * pressure_factor))
        return prompt_budget

    def _total_message_tokens(self, messages: list) -> int:
        return sum(self.bot.message_token_cost(m) for m in messages)

    def _drop_oldest_messages(self, messages: list, prompt_budget: int) -> list:
        """Drop oldest non-system, non-last messages until within budget."""

        def removable_indices(items):
            indexes = []
            for idx, msg in enumerate(items):
                role = str(msg.get("role") or "").strip().lower()
                if idx == len(items) - 1:
                    continue
                if idx == 0 and role == "system":
                    continue
                indexes.append(idx)
            return indexes

        while len(messages) > 1 and self._total_message_tokens(messages) > prompt_budget:
            candidates = removable_indices(messages)
            if not candidates:
                break
            messages.pop(candidates[0])
        return messages

    def _trim_last_message(self, messages: list, prompt_budget: int) -> list:
        """Trim the last message to fit within the remaining budget after leading messages."""
        if self._total_message_tokens(messages) > prompt_budget and messages:
            leading_cost = sum(self.bot.message_token_cost(m) for m in messages[:-1])
            remaining_budget = max(32, prompt_budget - leading_cost)
            messages[-1] = self.bot.trim_message_to_token_budget(
                messages[-1],
                remaining_budget,
            )
        return messages

    def _trim_all_messages(self, messages: list, prompt_budget: int) -> list:
        """Per-message trim pass â€” trim each message to its proportional share."""
        if self._total_message_tokens(messages) > prompt_budget:
            for idx, msg in enumerate(messages):
                running_total = self._total_message_tokens(messages)
                if running_total <= prompt_budget:
                    break
                other_cost = running_total - self.bot.message_token_cost(msg)
                allowed_budget = max(16, prompt_budget - other_cost)
                messages[idx] = self.bot.trim_message_to_token_budget(
                    msg,
                    allowed_budget,
                )
        return messages

    def _apply_system_prompt_fallback(
        self,
        messages: list,
        prompt_budget: int,
        purpose: str,
    ) -> list:
        """Replace an oversized system prompt with a minimal safety fallback."""
        import logging

        _logger = logging.getLogger(__name__)
        if not messages:
            return messages
        first_role = str(messages[0].get("role") or "").strip().lower()
        if first_role == "system" and self._total_message_tokens(messages) > prompt_budget:
            minimal_system_prompt = (
                "You are a warm, grounded dad speaking to Tony. Be supportive, concise, honest, and safe."
            )
            messages[0] = {**messages[0], "content": minimal_system_prompt}
            if self._total_message_tokens(messages) > prompt_budget:
                other_cost = self._total_message_tokens(
                    messages,
                ) - self.bot.message_token_cost(messages[0])
                allowed_budget = max(16, prompt_budget - other_cost)
                messages[0] = self.bot.trim_message_to_token_budget(
                    messages[0],
                    allowed_budget,
                )
            _logger.warning(
                "Prompt guard replaced oversized system prompt with minimal fallback for %s",
                purpose,
            )
        return messages

    def _trim_to_hard_limit(self, messages: list, prompt_budget: int) -> list:
        """Final trim/drop loop â€” last resort to fit within budget."""
        while messages and self._total_message_tokens(messages) > prompt_budget:
            last_idx = len(messages) - 1
            other_cost = sum(self.bot.message_token_cost(m) for m in messages[:-1])
            allowed_budget = max(8, prompt_budget - other_cost)
            trimmed = self.bot.trim_message_to_token_budget(
                messages[last_idx],
                allowed_budget,
            )
            if trimmed.get("content", "") == messages[last_idx].get("content", ""):
                if len(messages) > 1:
                    messages.pop(last_idx)
                    continue
                messages[last_idx] = {**trimmed, "content": "..."}
                break
            messages[last_idx] = trimmed
        return messages

    def _record_guard_stats(
        self,
        original_tokens: int,
        final_tokens: int,
        purpose: str,
    ) -> None:
        """Update prompt_guard_stats and emit a log line if trimming occurred."""
        import logging
        from datetime import datetime

        _logger = logging.getLogger(__name__)
        prompt_guard_stats = self.bot.prompt_guard_stats()
        trimmed_flag = final_tokens < original_tokens
        prompt_guard_stats.update(
            {
                "last_purpose": str(purpose or "chat"),
                "last_original_tokens": int(original_tokens),
                "last_final_tokens": int(final_tokens),
                "last_trimmed": bool(trimmed_flag),
                "last_updated": datetime.now().isoformat(timespec="seconds"),
            },
        )
        if trimmed_flag:
            prompt_guard_stats["trim_count"] = int(prompt_guard_stats.get("trim_count", 0) or 0) + 1
            prompt_guard_stats["trimmed_tokens_total"] = int(
                prompt_guard_stats.get("trimmed_tokens_total", 0) or 0,
            ) + max(0, int(original_tokens) - int(final_tokens))
        self.bot._prompt_guard_stats = prompt_guard_stats
        if trimmed_flag:
            _logger.info(
                "Prompt guard trimmed %s request from %s to %s tokens for %s",
                self.bot.ACTIVE_MODEL,
                original_tokens,
                final_tokens,
                purpose,
            )

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def guard_chat_request_messages(self, messages, purpose="chat"):
        """Apply a final prompt-size guard before calling Ollama.

        Strategy:
        - Keep the newest user turn whenever possible.
        - Drop older non-system context first.
        - Trim remaining messages to fit the prompt budget.
        - If an oversized system prompt still dominates, replace it with a
          minimal safety/persona fallback and continue.
        """
        normalized_messages = [dict(m) for m in list(messages or []) if isinstance(m, dict)]
        if not normalized_messages:
            return []

        prompt_budget = self._compute_prompt_budget()
        original_tokens = self._total_message_tokens(normalized_messages)
        if original_tokens <= prompt_budget:
            return normalized_messages

        normalized_messages = self._drop_oldest_messages(
            normalized_messages,
            prompt_budget,
        )
        normalized_messages = self._trim_last_message(
            normalized_messages,
            prompt_budget,
        )
        normalized_messages = self._trim_all_messages(
            normalized_messages,
            prompt_budget,
        )
        normalized_messages = self._apply_system_prompt_fallback(
            normalized_messages,
            prompt_budget,
            purpose,
        )
        normalized_messages = self._trim_to_hard_limit(
            normalized_messages,
            prompt_budget,
        )

        self._record_guard_stats(
            original_tokens,
            self._total_message_tokens(normalized_messages),
            purpose,
        )
        return normalized_messages


__all__ = ["PromptAssemblyManager"]
