from __future__ import annotations

from datetime import datetime


class ShadowAuditManager:
	"""Background-facing self-audit helper for Dad's advice consistency.

	This manager evaluates each finalized reply against durable parental goals and
	stores a compact audit trail in memory. It is intentionally heuristic and
	fast: no extra model call is required for each turn.
	"""

	_HARD_TONE_MARKERS = (
		"you should have",
		"you need to",
		"you always",
		"you never",
		"your fault",
		"just do it",
		"stop whining",
	)

	def __init__(self, bot):
		self.bot = bot

	def parental_goals(self) -> list[dict]:
		configured = self.bot.PROFILE.get("parental_goals", []) if isinstance(self.bot.PROFILE, dict) else []
		if isinstance(configured, list) and configured:
			goals = []
			for item in configured:
				if isinstance(item, dict):
					label = str(item.get("goal") or item.get("label") or "").strip()
					if label:
						goals.append(
							{
								"goal": label,
								"priority": max(1, int(item.get("priority", 1) or 1)),
							}
						)
				elif str(item).strip():
					goals.append({"goal": str(item).strip(), "priority": 1})
			if goals:
				return goals[:10]

		return [
			{"goal": "Maintain emotional safety while coaching growth", "priority": 3},
			{"goal": "Encourage persistence over perfection", "priority": 2},
			{"goal": "Teach ownership without shame", "priority": 2},
			{"goal": "Stay warm, honest, and present", "priority": 3},
		]

	def _goal_vector(self) -> dict:
		state = {}
		try:
			manager = getattr(self.bot, "internal_state_manager", None)
			if manager is not None and hasattr(manager, "snapshot"):
				state = manager.snapshot() or {}
			elif hasattr(self.bot, "internal_state_snapshot"):
				state = self.bot.internal_state_snapshot() or {}
		except Exception:
			state = {}
		if not isinstance(state, dict):
			return {"support": 0.7, "challenge": 0.25, "proactive": 0.35}
		raw_goals = state.get("goal_vector")
		goals = dict(raw_goals) if isinstance(raw_goals, dict) else {}
		return {
			"support": float(goals.get("support", 0.7) or 0.7),
			"challenge": float(goals.get("challenge", 0.25) or 0.25),
			"proactive": float(goals.get("proactive", 0.35) or 0.35),
		}

	def _has_hard_tone(self, reply: str) -> bool:
		lowered = str(reply or "").lower()
		return any(marker in lowered for marker in self._HARD_TONE_MARKERS)

	def _alignment_score(self, *, current_mood: str, reply: str, goal_vector: dict) -> tuple[int, str]:
		mood = self.bot.normalize_mood(current_mood)
		support = float(goal_vector.get("support", 0.7) or 0.7)
		challenge = float(goal_vector.get("challenge", 0.25) or 0.25)
		hard_tone = self._has_hard_tone(reply)

		score = 70
		reason = "advice aligned with long-term parental stance"

		if mood in {"sad", "stressed", "tired"}:
			score += int((support - challenge) * 20)
			if hard_tone:
				score -= 25
				reason = "tone was likely too hard for current emotional state"
		else:
			score += int((support * 10) + (challenge * 6))
			if hard_tone and challenge > support:
				score -= 10
				reason = "challenge-forward tone may have outpaced warmth"

		score = max(0, min(100, score))
		return score, reason

	def audit_and_record(self, *, user_input: str, reply: str, current_mood: str) -> dict:
		goal_vector = self._goal_vector()
		alignment, reasoning = self._alignment_score(
			current_mood=current_mood,
			reply=reply,
			goal_vector=goal_vector,
		)
		active_hypothesis = "supportive_baseline"
		try:
			rel = self.bot.relationship.snapshot()
			active_hypothesis = str(rel.get("active_hypothesis") or "supportive_baseline")
		except Exception:
			pass

		hard_tone = self._has_hard_tone(reply)
		needs_repair = bool(
			alignment < 45
			or (hard_tone and self.bot.normalize_mood(current_mood) in {"sad", "stressed", "tired"})
		)

		audit_entry = {
			"recorded_at": datetime.now().isoformat(timespec="seconds"),
			"goal_alignment_score": int(alignment),
			"reasoning": str(reasoning),
			"active_hypothesis": active_hypothesis,
			"current_mood": self.bot.normalize_mood(current_mood),
			"hard_tone": bool(hard_tone),
			"needs_repair": bool(needs_repair),
			"user_input_excerpt": str(user_input or "")[:240],
			"reply_excerpt": str(reply or "")[:320],
			"goal_vector": {
				"support": round(float(goal_vector.get("support", 0.7) or 0.7), 3),
				"challenge": round(float(goal_vector.get("challenge", 0.25) or 0.25), 3),
				"proactive": round(float(goal_vector.get("proactive", 0.35) or 0.35), 3),
			},
			"parental_goals": self.parental_goals(),
		}

		store = self.bot.MEMORY_STORE if isinstance(self.bot.MEMORY_STORE, dict) else {}
		audits = [
			dict(item)
			for item in list(store.get("advice_audits") or [])
			if isinstance(item, dict)
		]
		audits.append(audit_entry)
		self.bot.mutate_memory_store(advice_audits=audits[-160:])
		return audit_entry


__all__ = ["ShadowAuditManager"]
