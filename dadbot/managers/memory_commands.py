from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from dadbot.contracts import DadBotContext, SupportsDadBotAccess


class MemoryCommandManager:
	"""Owns conversational memory-command parsing, execution, and transcript filtering."""

	def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot

	@staticmethod
	def parse_memory_command(user_input):
		stripped = user_input.strip()

		update_match = re.match(r"^(?:update memory|change memory)(?: that)? (.+?) to (.+)$", stripped, flags=re.IGNORECASE)
		if update_match:
			return {
				"action": "update",
				"old": update_match.group(1).strip(),
				"new": update_match.group(2).strip(),
			}

		remember_match = re.match(r"^(?:remember this|remember that|please remember)(?::)?\s+(.+)$", stripped, flags=re.IGNORECASE)
		if remember_match:
			return {
				"action": "remember",
				"summary": remember_match.group(1).strip(),
			}

		forget_match = re.match(r"^(?:forget|delete memory about|remove memory about|forget that)(?::)?\s+(.+)$", stripped, flags=re.IGNORECASE)
		if forget_match:
			return {
				"action": "forget",
				"query": forget_match.group(1).strip(),
			}

		# ── GDPR / CCPA data-rights commands ─────────────────────────────────
		stripped_lower = stripped.lower()

		if re.match(r"^/export(?:\s+my\s+data)?$", stripped, flags=re.IGNORECASE):
			return {"action": "gdpr_export"}

		if re.match(r"^/(?:delete\s+my\s+data|forget\s+me|erase\s+me)$", stripped, flags=re.IGNORECASE):
			return {"action": "gdpr_delete"}

		if re.match(r"^/what\s+do\s+you\s+know(?:\s+about\s+me)?$", stripped, flags=re.IGNORECASE):
			return {"action": "gdpr_summary"}

		return None

	def handle_memory_command(self, user_input):
		command = self.parse_memory_command(user_input)
		if command is None:
			return None

		if command["action"] == "remember":
			memory = self.bot.add_memory(command["summary"])
			if memory is None:
				return "I tried to hang onto that for you, buddy, but I couldn't turn it into a usable memory."
			return f"I've got it, Tony. I'll remember that under {memory['category']} as: {memory['summary'].rstrip('.')}"

		if command["action"] == "forget":
			removed = self.bot.forget_memories(command["query"])
			if not removed:
				return "I don't think I had that saved in my memory, buddy."
			removed_text = self.bot.format_memories_for_reply(removed)
			return f"Okay, Tony. I forgot those saved details: {removed_text}"

		if command["action"] == "update":
			removed = self.bot.forget_memories(command["old"])
			memory = self.bot.add_memory(command["new"])
			if memory is None:
				return "I couldn't update that memory cleanly, buddy."
			if removed:
				return f"Okay, Tony. I updated that memory. I'll remember: {memory['summary'].rstrip('.')}"
			return f"I didn't find the old memory exactly, but I've saved the new one: {memory['summary'].rstrip('.')}"

		if command["action"] == "gdpr_export":
			return self._handle_data_export()

		if command["action"] == "gdpr_delete":
			return self._handle_data_delete()

		if command["action"] == "gdpr_summary":
			return self._handle_data_summary()

		return None

	# ── GDPR / CCPA helpers ───────────────────────────────────────────────────

	def _handle_data_export(self) -> str:
		"""Export all personal data to a timestamped JSON file in the user's home directory."""
		try:
			memories = list(self.bot.memory_catalog() or [])
			consolidated = list(getattr(self.bot, "consolidated_memories", lambda: [])() or [])
			relationship = {}
			try:
				relationship = self.bot.relationship_state()
			except Exception:
				pass

			export_data = {
				"export_version": "1.0",
				"exported_at": datetime.now(timezone.utc).isoformat(),
				"memories": memories,
				"consolidated_memories": consolidated,
				"relationship_state": relationship,
			}

			timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
			export_path = Path.home() / f"dadbot-export-{timestamp}.json"
			export_path.write_text(json.dumps(export_data, indent=2, default=str), encoding="utf-8")

			counts = (
				f"{len(memories)} memories, "
				f"{len(consolidated)} consolidated entries"
			)
			return (
				f"Done, buddy. I've saved everything I know about you to:\n\n"
				f"  {export_path}\n\n"
				f"That file contains {counts}, plus your relationship state. "
				"You can open it in any text editor — it's plain JSON."
			)
		except Exception as exc:  # pragma: no cover
			return f"I tried to export your data but ran into a problem: {exc}"

	def _handle_data_delete(self) -> str:
		"""Clear all stored personal data (memories, relationship state)."""
		try:
			self.bot.memory.clear_memory_store()
			try:
				# Reset relationship state if the method exists
				reset = getattr(self.bot, "reset_relationship_state", None)
				if callable(reset):
					reset()
			except Exception:
				pass
			return (
				"Done. I've erased everything I had saved about you — memories, relationship history, all of it. "
				"We're starting completely fresh. I'll still be your dad, I just won't remember the specifics from before."
			)
		except Exception as exc:  # pragma: no cover
			return f"I tried to delete your data but hit a snag: {exc}"

	def _handle_data_summary(self) -> str:
		"""Return a human-readable summary of all stored data categories."""
		try:
			memories = list(self.bot.memory_catalog() or [])
			consolidated = list(getattr(self.bot, "consolidated_memories", lambda: [])() or [])
			relationship = {}
			try:
				relationship = self.bot.relationship_state()
			except Exception:
				pass

			categories: dict[str, int] = {}
			for m in memories:
				cat = str(m.get("category") or "general").strip() or "general"
				categories[cat] = categories.get(cat, 0) + 1

			category_lines = "\n".join(
				f"  • {cat}: {count} entr{'y' if count == 1 else 'ies'}"
				for cat, count in sorted(categories.items())
			) or "  (none yet)"

			trust = relationship.get("trust_level") if relationship else None
			trust_line = f"  • Trust level: {trust:.2f}" if isinstance(trust, (int, float)) else ""
			rel_tenure = relationship.get("relationship_tenure_days") if relationship else None
			tenure_line = f"  • Relationship tenure: {rel_tenure} day(s)" if rel_tenure is not None else ""
			rel_lines = "\n".join(filter(None, [trust_line, tenure_line])) or "  (no relationship data)"

			return (
				f"Here's everything I have stored about you, Tony:\n\n"
				f"**Individual memories ({len(memories)} total):**\n{category_lines}\n\n"
				f"**Consolidated long-term insights:** {len(consolidated)} entr{'y' if len(consolidated) == 1 else 'ies'}\n\n"
				f"**Relationship state:**\n{rel_lines}\n\n"
				f"You can use `/export my data` to get the full details as a file, "
				f"or `/delete my data` to erase everything."
			)
		except Exception as exc:  # pragma: no cover
			return f"I ran into a problem while summarising your data: {exc}"

	def build_memory_transcript(self, history):
		transcript_lines = []

		for message in history:
			if message["role"] == "user":
				if self.parse_memory_command(message["content"]) is not None:
					continue
				if self.bot.parse_tool_command(message["content"]) is not None:
					continue
				if self.bot.get_memory_reply(message["content"]) is not None:
					continue

				mood = self.bot.normalize_mood(message.get("mood"))
				transcript_lines.append(f"Tony (mood={mood}): {message['content']}")

		return "\n".join(transcript_lines)


__all__ = ["MemoryCommandManager"]
