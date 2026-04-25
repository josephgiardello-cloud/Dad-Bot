from __future__ import annotations

import json
import logging
import time

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.models import ReplySupervisorSnapshot, SupervisorDecisionState, SupervisorJudgment
from dadbot.utils import json_dumps
from pydantic import ValidationError


class ReplySupervisorManager:
	"""Owns reply-supervision prompts, grading, and revision decisions."""

	def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot

	@staticmethod
	def _supervisor_state(
		stage: str,
		source: str,
		*,
		approved: bool = True,
		score: int = 0,
		dad_likeness: int = 0,
		groundedness: int = 0,
		emotional_fit: int = 0,
		issues: list[str] | None = None,
		revised: bool = False,
		duration_ms: int = 0,
	) -> dict[str, object]:
		validated = SupervisorDecisionState.model_validate(
			{
				"stage": str(stage or "idle").strip() or "idle",
				"approved": bool(approved),
				"score": max(0, min(10, int(score or 0))),
				"dad_likeness": max(0, min(10, int(dad_likeness or 0))),
				"groundedness": max(0, min(10, int(groundedness or 0))),
				"emotional_fit": max(0, min(10, int(emotional_fit or 0))),
				"issues": list(issues or []),
				"revised": bool(revised),
				"duration_ms": max(0, int(duration_ms or 0)),
				"source": str(source or "none").strip() or "none",
			}
		)
		return validated.model_dump(mode="python")

	@staticmethod
	def _normalize_supervisor_issues(issues: object) -> list[str]:
		if not isinstance(issues, list):
			return []
		return [
			str(item).strip()
			for item in issues
			if str(item).strip()
		][:4]

	@staticmethod
	def _coerce_supervisor_rating(value: object, *, default: int = 1) -> int:
		try:
			rating = int(value)
		except (TypeError, ValueError):
			return default
		return max(1, min(10, rating))

	def _last_supervisor_duration_ms(self) -> int:
		return max(0, int((self.bot._last_reply_supervisor or {}).get("duration_ms", 0) or 0))

	def _coerce_supervisor_judgment(self, judgment, *, stage: str) -> SupervisorJudgment | None:
		if isinstance(judgment, SupervisorJudgment):
			return judgment.model_copy(update={"stage": stage})
		if not isinstance(judgment, dict):
			return None

		revised_reply = str(judgment.get("revised_reply") or "").strip() or None
		payload = {
			"approved": bool(judgment.get("approved", not revised_reply)),
			"score": self._coerce_supervisor_rating(judgment.get("score")),
			"dad_likeness": self._coerce_supervisor_rating(judgment.get("dad_likeness")),
			"groundedness": self._coerce_supervisor_rating(judgment.get("groundedness")),
			"emotional_fit": self._coerce_supervisor_rating(judgment.get("emotional_fit")),
			"issues": self._normalize_supervisor_issues(judgment.get("issues")),
			"revised_reply": revised_reply,
			"stage": stage,
		}
		try:
			return SupervisorJudgment.model_validate(payload)
		except ValidationError:
			return None

	def _supervisor_messages(self, user_input: str, candidate_reply: str, current_mood: str) -> list[dict[str, str]]:
		return [{"role": "user", "content": self.build_reply_supervisor_prompt(user_input, candidate_reply, current_mood)}]

	@staticmethod
	def _elapsed_ms(started_at: float) -> int:
		return max(0, int((time.perf_counter() - started_at) * 1000))

	def _record_supervisor_failure(self, stage: str, source: str, started_at: float) -> str:
		self.bot._last_reply_supervisor = self._supervisor_state(
			stage,
			source,
			duration_ms=self._elapsed_ms(started_at),
		)
		return ""

	def _finalize_supervisor_response(self, content: str, candidate_reply: str, *, stage: str, started_at: float) -> str:
		try:
			judgment = self.bot.parse_model_json_content(content)
		except (json.JSONDecodeError, TypeError):
			self.bot._last_reply_supervisor = self._supervisor_state(
				stage,
				"invalid_json",
				duration_ms=self._elapsed_ms(started_at),
			)
			return candidate_reply

		self.bot._last_reply_supervisor = {
			**dict(self.bot._last_reply_supervisor or {}),
			"duration_ms": self._elapsed_ms(started_at),
		}
		return self.apply_reply_supervisor_decision(judgment, candidate_reply, stage=stage)

	def build_reply_critique_prompt(self, user_input: str, draft_reply: str, current_mood: str) -> str:
		supervisor_context = self.build_reply_supervisor_context(current_mood)
		relevant_facts = self.bot.build_profile_block(self.bot.relevant_fact_ids_for_input(user_input))
		return f"""
You are reviewing a draft reply from a warm, supportive dad.

Known profile facts:
{self.bot.build_profile_block()}

Relevant facts for this message:
{relevant_facts}

Supervisor context:
{supervisor_context}

Rules:
- Never invent facts not in the profile.
- If the draft guesses at unknown personal details, revise it to say "I don't want to guess".
- Keep the warm, casual dad tone and the signoff when natural.
- Only make minimal changes needed for accuracy.
- Preserve emotional fit for Tony's current mood: {self.bot.normalize_mood(current_mood)}.
- Respect the current relationship state. If openness is guarded or momentum is heavy, be gentler and less forceful.
- Stay consistent with the active evolved persona traits and behavior rules.

Return only JSON:
{{"approved": true/false, "revised_reply": "string or null"}}

Tony: {json_dumps(user_input)}
Draft: {json_dumps(draft_reply)}
""".strip()

	def build_reply_alignment_judge_prompt(self, user_input: str, candidate_reply: str, current_mood: str) -> str:
		return self.build_reply_supervisor_prompt(user_input, candidate_reply, current_mood)

	def build_reply_supervisor_prompt(self, user_input: str, draft_reply: str, current_mood: str) -> str:
		supervisor_context = self.build_reply_supervisor_context(current_mood)
		relevant_facts = self.bot.build_profile_block(self.bot.relevant_fact_ids_for_input(user_input))
		return f"""
You are reviewing and grading a draft reply from a warm, supportive dad.

Known profile facts:
{self.bot.build_profile_block()}

Relevant facts for this message:
{relevant_facts}

Supervisor context:
{supervisor_context}

Current mood: {self.bot.normalize_mood(current_mood)}

Rules:
- Never invent facts not in the profile.
- If the draft guesses at unknown personal details, revise it to say "I don't want to guess".
- Keep the reply natural, concise, emotionally fitting, and dad-like.
- Respect the current relationship state. If openness is guarded or momentum is heavy, be gentler and less forceful.
- Stay consistent with the active evolved persona traits and behavior rules.
- If the draft is already strong, approve it and leave revised_reply null.
- If it is generic, cold, too formal, too intense, verbose, or inaccurate, fix it with the smallest useful rewrite.

Scoring guide:
- dad_likeness: warm, steady, grounded, naturally fatherly, not corporate or clinical.
- groundedness: avoids invented facts, avoids overconfident guesses, stays specific and believable.
- emotional_fit: matches Tony's mood and the current relationship baseline without over- or under-reacting.

Anchor examples:
- Good reply: "That sounds heavy, buddy. Let us slow it down and take one steady next step together."
- Bad reply: "As an AI assistant, I recommend optimizing your workflow immediately and reframing your mindset."

Think silently through groundedness, dad_likeness, and emotional_fit before deciding. Do not reveal your reasoning. Return only the final JSON.

Return only JSON:
{{
  "approved": true/false,
  "score": 1-10,
  "dad_likeness": 1-10,
  "groundedness": 1-10,
  "emotional_fit": 1-10,
  "issues": ["short strings"],
  "revised_reply": "string or null"
}}

Tony: {json_dumps(user_input)}
Draft reply: {json_dumps(draft_reply)}
""".strip()

	def apply_reply_supervisor_decision(self, judgment, candidate_reply, stage="reply_supervisor"):
		judgment_model = self._coerce_supervisor_judgment(judgment, stage=stage)
		if judgment_model is None:
			self.bot._last_reply_supervisor = self._supervisor_state(
				stage,
				"invalid_payload",
				duration_ms=self._last_supervisor_duration_ms(),
			)
			return candidate_reply

		revised_reply = str(judgment_model.revised_reply or "").strip()
		self.bot._last_reply_supervisor = self._supervisor_state(
			judgment_model.stage,
			"llm",
			approved=judgment_model.approved,
			score=judgment_model.score,
			dad_likeness=judgment_model.dad_likeness,
			groundedness=judgment_model.groundedness,
			emotional_fit=judgment_model.emotional_fit,
			issues=judgment_model.issues,
			revised=bool(revised_reply),
			duration_ms=self._last_supervisor_duration_ms(),
		)
		return revised_reply or candidate_reply

	def run_reply_supervisor(
		self,
		user_input: str,
		candidate_reply: str,
		current_mood: str,
		*,
		stage: str = "reply_supervisor",
	) -> str:
		started_at = time.perf_counter()
		if self.bot.LIGHT_MODE:
			self.bot._last_reply_supervisor = self._supervisor_state(stage="disabled", source="light_mode")
			return candidate_reply

		try:
			response = self.bot.call_ollama_chat(
				messages=self._supervisor_messages(user_input, candidate_reply, current_mood),
				options={"temperature": 0.0},
				response_format="json",
				purpose="reply supervisor",
			)
			content = self.bot.extract_ollama_message_content(response)
		except (RuntimeError, KeyError, TypeError) as exc:
			self.bot.record_runtime_issue("reply supervisor", "keeping the draft reply without supervisor refinement", exc, level=logging.INFO)
			self._record_supervisor_failure(stage, "error", started_at)
			return candidate_reply

		return self._finalize_supervisor_response(content, candidate_reply, stage=stage, started_at=started_at)

	async def run_reply_supervisor_async(
		self,
		user_input: str,
		candidate_reply: str,
		current_mood: str,
		*,
		stage: str = "reply_supervisor",
	) -> str:
		started_at = time.perf_counter()
		if self.bot.LIGHT_MODE:
			self.bot._last_reply_supervisor = self._supervisor_state(stage="disabled", source="light_mode")
			return candidate_reply

		try:
			response = await self.bot.call_ollama_chat_async(
				messages=self._supervisor_messages(user_input, candidate_reply, current_mood),
				options={"temperature": 0.0},
				response_format="json",
				purpose="reply supervisor",
			)
			content = self.bot.extract_ollama_message_content(response)
		except (RuntimeError, KeyError, TypeError) as exc:
			self.bot.record_runtime_issue("reply supervisor", "keeping the draft reply without supervisor refinement", exc, level=logging.INFO)
			self._record_supervisor_failure(stage, "error", started_at)
			return candidate_reply

		return self._finalize_supervisor_response(content, candidate_reply, stage=stage, started_at=started_at)

	def build_reply_supervisor_context(self, current_mood):
		relationship = self.bot.relationship.snapshot()
		top_topics = ", ".join(relationship.get("top_topics", [])) or "none"
		active_traits = ", ".join(self.bot.profile_runtime.active_persona_traits(limit=3)) or "none"
		hypothesis_lines = "\n".join(
			f"- {entry.get('label', entry.get('name', 'theory'))}: {float(entry.get('probability', 0.0) or 0.0):.2f}"
			for entry in relationship.get("hypotheses", [])[:3]
		) or "- none"
		behavior_rules = "\n".join(f"- {rule}" for rule in self.bot.profile_runtime.effective_behavior_rules()[:6]) or "- none"
		last_reflection = str(relationship.get("last_reflection") or "none").strip() or "none"
		return f"""
- trust_level: {relationship.get('trust_level', 0)} ({relationship.get('trust_label', 'steady')})
- openness_level: {relationship.get('openness_level', 0)} ({relationship.get('openness_label', 'steady')})
- emotional_momentum: {relationship.get('emotional_momentum', 'steady')}
- active_hypothesis: {relationship.get('active_hypothesis', 'supportive_baseline')} ({float(relationship.get('active_hypothesis_probability', 0.0) or 0.0):.2f})
- current_mood: {self.bot.normalize_mood(current_mood)}
- top_topics: {top_topics}
- last_reflection: {last_reflection}
- active_persona_traits: {active_traits}
Relationship hypotheses:
{hypothesis_lines}
Behavior rules:
{behavior_rules}
""".strip()

	def reply_supervisor_snapshot(self):
		relationship = self.bot.relationship.snapshot()
		decision = self._supervisor_state(
			str((self.bot._last_reply_supervisor or {}).get("stage") or "idle"),
			str((self.bot._last_reply_supervisor or {}).get("source") or "none"),
			approved=bool((self.bot._last_reply_supervisor or {}).get("approved", True)),
			score=max(0, min(10, int((self.bot._last_reply_supervisor or {}).get("score", 0) or 0))),
			dad_likeness=max(0, min(10, int((self.bot._last_reply_supervisor or {}).get("dad_likeness", 0) or 0))),
			groundedness=max(0, min(10, int((self.bot._last_reply_supervisor or {}).get("groundedness", 0) or 0))),
			emotional_fit=max(0, min(10, int((self.bot._last_reply_supervisor or {}).get("emotional_fit", 0) or 0))),
			issues=[str(item) for item in (self.bot._last_reply_supervisor or {}).get("issues", []) if str(item).strip()][:4],
			revised=bool((self.bot._last_reply_supervisor or {}).get("revised")),
			duration_ms=max(0, int((self.bot._last_reply_supervisor or {}).get("duration_ms", 0) or 0)),
		)
		validated = ReplySupervisorSnapshot.model_validate(
			{
				"enabled": not self.bot.LIGHT_MODE,
				"active_hypothesis": str(relationship.get("active_hypothesis") or "supportive_baseline").strip() or "supportive_baseline",
				"active_hypothesis_label": str(relationship.get("active_hypothesis_label") or "Supportive Baseline").strip() or "Supportive Baseline",
				"active_hypothesis_probability": float(relationship.get("active_hypothesis_probability", 0.0) or 0.0),
				"last_decision": decision,
			}
		)
		return validated.model_dump(mode="python")

	def judge_reply_alignment(self, user_input, candidate_reply, current_mood):
		return self.run_reply_supervisor(user_input, candidate_reply, current_mood, stage="alignment_judge")

	async def judge_reply_alignment_async(self, user_input, candidate_reply, current_mood):
		return await self.run_reply_supervisor_async(user_input, candidate_reply, current_mood, stage="alignment_judge")

	def critique_reply(self, user_input, draft_reply, current_mood):
		return self.run_reply_supervisor(user_input, draft_reply, current_mood, stage="reply_supervisor")

	async def critique_reply_async(self, user_input, draft_reply, current_mood):
		return await self.run_reply_supervisor_async(user_input, draft_reply, current_mood, stage="reply_supervisor")


__all__ = ["ReplySupervisorManager"]
