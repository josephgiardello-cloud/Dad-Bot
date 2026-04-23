from __future__ import annotations

from dadbot.contracts import AttachmentList, DadBotContext, SupportsDadBotAccess


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

	def base_request_sections(self, current_mood: str) -> list[str | None]:
		return [
			self.bot.context_builder.build_core_persona_prompt(),
			self.bot.context_builder.build_dynamic_profile_context(),
			self.bot.context_builder.build_relationship_context(),
			self.bot.build_style_examples(),
			self.bot.tone_context.build_mood_context(current_mood),
		]

	def infer_visual_task(self, user_input: str, attachments: AttachmentList | None = None) -> dict[str, str] | None:
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

		if any(token in combined for token in ["traceback", "stack trace", "terminal", "console", "error", "exception", "crash", "bug", "screenshot"]):
			return {"mode": "debug_screenshot", **self.VISUAL_TASK_PROFILES["debug_screenshot"]}
		if any(token in combined for token in ["homework", "worksheet", "equation", "solve", "assignment", "quiz", "math", "diagram", "study"]):
			return {"mode": "homework_help", **self.VISUAL_TASK_PROFILES["homework_help"]}
		if any(token in combined for token in ["draw", "drawing", "sketch", "painting", "art", "doodle", "made this", "coloring"]):
			return {"mode": "creative_feedback", **self.VISUAL_TASK_PROFILES["creative_feedback"]}
		if any(token in combined for token in ["fix", "broken", "repair", "leak", "wire", "wiring", "tool", "engine", "appliance", "install"]):
			return {"mode": "repair_coach", **self.VISUAL_TASK_PROFILES["repair_coach"]}
		return {"mode": "general_photo", **self.VISUAL_TASK_PROFILES["general_photo"]}

	def build_visual_task_context(self, user_input: str, attachments: AttachmentList | None = None) -> str | None:
		profile = self.infer_visual_task(user_input, attachments)
		if profile is None:
			return None
		return (
			f"Visual mode for this turn: {profile['label']}.\n"
			f"{profile['context']}\n"
			"Use only what is actually visible or already described by Tony. If the image is ambiguous, say what you can and cannot see."
		)

	def contextual_request_sections(self, user_input: str, current_mood: str, attachments: AttachmentList | None = None) -> list[str | None]:
		return [
			self.bot.tone_context.build_daily_checkin_context(current_mood),
			self.bot.build_active_tool_observation_context(),
			self.build_visual_task_context(user_input, attachments),
			self.bot.context_builder.build_cross_session_context(user_input),
			self.bot.context_builder.build_session_summary_context(),
			self.bot.context_builder.build_relevant_context(user_input),
			self.bot.context_builder.build_wisdom_context(user_input),
			self.bot.context_builder.build_memory_context(user_input),
			self.bot.tone_context.build_escalation_context(current_mood, self.bot.session_moods),
		]

	def build_request_system_prompt(self, user_input: str, current_mood: str, attachments: AttachmentList | None = None) -> str:
		sections = self.base_request_sections(current_mood)
		sections.extend(self.contextual_request_sections(user_input, current_mood, attachments))
		return "\n\n".join(section for section in sections if section)

	def build_image_analysis_prompt(self, note: str = "", user_input: str = "", attachment: dict | None = None) -> str:
		profile = self.infer_visual_task(user_input or note, [attachment] if attachment else None)
		if profile is None:
			profile = {"mode": "general_photo", **self.VISUAL_TASK_PROFILES["general_photo"]}
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

	def build_chat_request_messages(self, user_input: str, current_mood: str, attachments: AttachmentList | None = None) -> list[dict[str, object]]:
		system_prompt = self.build_request_system_prompt(user_input, current_mood, attachments)
		request_messages = [{"role": "system", "content": system_prompt}]
		request_messages.extend(self.bot.token_budgeted_prompt_history(system_prompt, user_input))
		request_messages.append(self.bot.build_user_request_message(user_input, attachments))
		return request_messages

	def guard_chat_request_messages(self, messages, purpose="chat"):
		"""Apply a final prompt-size guard before calling Ollama.

		Strategy:
		- Keep the newest user turn whenever possible.
		- Drop older non-system context first.
		- Trim remaining messages to fit the prompt budget.
		- If an oversized system prompt still dominates, replace it with a
		  minimal safety/persona fallback and continue.
		"""
		import logging
		from datetime import datetime
		_logger = logging.getLogger(__name__)

		normalized_messages = [dict(m) for m in list(messages or []) if isinstance(m, dict)]
		if not normalized_messages:
			return []

		context_budget = max(256, int(self.bot.effective_context_token_budget(self.bot.ACTIVE_MODEL) or self.bot.CONTEXT_TOKEN_BUDGET or 0))
		reserved_tokens = max(64, int(self.bot.RESERVED_RESPONSE_TOKENS or 0))
		prompt_budget = max(128, context_budget - reserved_tokens)
		pressure_factor = self.bot.adaptive_prompt_pressure_factor()
		if pressure_factor < 1.0:
			prompt_budget = max(96, int(prompt_budget * pressure_factor))

		def total_tokens(items):
			return sum(self.bot.message_token_cost(m) for m in items)

		original_tokens = total_tokens(normalized_messages)
		if original_tokens <= prompt_budget:
			return normalized_messages

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

		while len(normalized_messages) > 1 and total_tokens(normalized_messages) > prompt_budget:
			candidates = removable_indices(normalized_messages)
			if not candidates:
				break
			normalized_messages.pop(candidates[0])

		if total_tokens(normalized_messages) > prompt_budget and normalized_messages:
			leading_cost = sum(self.bot.message_token_cost(m) for m in normalized_messages[:-1])
			remaining_budget = max(32, prompt_budget - leading_cost)
			normalized_messages[-1] = self.bot.trim_message_to_token_budget(normalized_messages[-1], remaining_budget)

		if total_tokens(normalized_messages) > prompt_budget:
			for idx, msg in enumerate(normalized_messages):
				running_total = total_tokens(normalized_messages)
				if running_total <= prompt_budget:
					break
				other_cost = running_total - self.bot.message_token_cost(msg)
				allowed_budget = max(16, prompt_budget - other_cost)
				normalized_messages[idx] = self.bot.trim_message_to_token_budget(msg, allowed_budget)

		if normalized_messages:
			first_role = str(normalized_messages[0].get("role") or "").strip().lower()
			if first_role == "system" and total_tokens(normalized_messages) > prompt_budget:
				minimal_system_prompt = "You are a warm, grounded dad speaking to Tony. Be supportive, concise, honest, and safe."
				normalized_messages[0] = {**normalized_messages[0], "content": minimal_system_prompt}
				if total_tokens(normalized_messages) > prompt_budget:
					other_cost = total_tokens(normalized_messages) - self.bot.message_token_cost(normalized_messages[0])
					allowed_budget = max(16, prompt_budget - other_cost)
					normalized_messages[0] = self.bot.trim_message_to_token_budget(normalized_messages[0], allowed_budget)
				_logger.warning("Prompt guard replaced oversized system prompt with minimal fallback for %s", purpose)

		while normalized_messages and total_tokens(normalized_messages) > prompt_budget:
			last_idx = len(normalized_messages) - 1
			other_cost = sum(self.bot.message_token_cost(m) for m in normalized_messages[:-1])
			allowed_budget = max(8, prompt_budget - other_cost)
			trimmed = self.bot.trim_message_to_token_budget(normalized_messages[last_idx], allowed_budget)
			if trimmed.get("content", "") == normalized_messages[last_idx].get("content", ""):
				if len(normalized_messages) > 1:
					normalized_messages.pop(last_idx)
					continue
				normalized_messages[last_idx] = {**trimmed, "content": "..."}
				break
			normalized_messages[last_idx] = trimmed

		final_tokens = total_tokens(normalized_messages)
		prompt_guard_stats = self.bot.prompt_guard_stats()
		trimmed_flag = final_tokens < original_tokens
		prompt_guard_stats.update({
			"last_purpose": str(purpose or "chat"),
			"last_original_tokens": int(original_tokens),
			"last_final_tokens": int(final_tokens),
			"last_trimmed": bool(trimmed_flag),
			"last_updated": datetime.now().isoformat(timespec="seconds"),
		})
		if trimmed_flag:
			prompt_guard_stats["trim_count"] = int(prompt_guard_stats.get("trim_count", 0) or 0) + 1
			prompt_guard_stats["trimmed_tokens_total"] = int(prompt_guard_stats.get("trimmed_tokens_total", 0) or 0) + max(0, int(original_tokens) - int(final_tokens))
		self.bot._prompt_guard_stats = prompt_guard_stats

		if final_tokens < original_tokens:
			_logger.info("Prompt guard trimmed %s request from %s to %s tokens for %s", self.bot.ACTIVE_MODEL, original_tokens, final_tokens, purpose)

		return normalized_messages


__all__ = ["PromptAssemblyManager"]