from __future__ import annotations

import hashlib
from datetime import date

from dadbot.contracts import DadBotContext, SupportsDadBotAccess


class ProfileRuntimeManager:
	"""Owns profile-derived runtime settings and profile update normalization."""

	def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot
		self._profile = {}

	@property
	def profile(self):
		return self._profile

	@profile.setter
	def profile(self, value):
		self._profile = value if isinstance(value, dict) else {}

	@property
	def style(self):
		return self.get_style()

	@style.setter
	def style(self, value):
		self.set_style(value)

	def get_style(self):
		cached_style = getattr(self.bot, "_STYLE", None)
		if isinstance(cached_style, dict):
			return cached_style
		profile_style = self._profile.get("style") if isinstance(self._profile, dict) else None
		if isinstance(profile_style, dict):
			return profile_style
		return {}

	def set_style(self, value):
		self.bot._STYLE = value if isinstance(value, dict) else {}
		return self.bot._STYLE

	def load_profile(self):
		self.profile = self.bot.runtime_storage.load_profile()
		self.initialize_profile_defaults()
		return self.profile

	def initialize_profile_defaults(self):
		self._profile.setdefault("voice", {}).setdefault("tts_backend", "pyttsx3")
		self._profile.setdefault("voice", {}).setdefault("piper_model_path", "")
		self._profile.setdefault("avatar", {})
		self._profile.setdefault("ical_feed_url", "")
		return self._profile

	def refresh_profile_runtime(self):
		self.bot.STYLE = self.bot.PROFILE["style"]
		self.bot.FAMILY = self.bot.PROFILE["family"]
		self.bot.EDUCATION = self.bot.PROFILE["education"]
		self.bot.SECURITY = self.bot.PROFILE.get("security", {})
		self.bot.SUPPORT_ESCALATION = self.bot.PROFILE.get("support_escalation", {})
		self.bot.CRISIS_SUPPORT = self.bot.PROFILE.get("crisis_support", {})
		self.bot.AGENTIC_TOOLS = self.bot.PROFILE.get("agentic_tools", {})
		self.bot.RELATIONSHIP_CALIBRATION = self.bot.PROFILE.get("relationship_calibration", {})
		self.bot.CADENCE = self.bot.PROFILE.get("cadence", {})
		self.bot.RUNTIME = self.bot.PROFILE.get("runtime", {})
		self.bot.OPENING_MESSAGES = self.bot.PROFILE.get("opening_messages", [])
		self.bot.CHAT_ROUTING = self.bot.PROFILE["chat_routing"]
		self.bot.FACT_DEFINITIONS = self.bot.PROFILE["facts"]

		runtime = self.runtime_settings()
		self.bot.PREFERRED_EMBEDDING_MODELS = tuple(runtime["preferred_embedding_models"])
		self.bot.STREAM_TIMEOUT_SECONDS = runtime["max_thinking_time_seconds"]
		self.bot.STREAM_MAX_CHARS = runtime["stream_max_chars"]
		self.bot.GRAPH_REFRESH_DEBOUNCE_SECONDS = runtime["graph_refresh_debounce_seconds"]

		self.bot.DAD_BIRTHDATE = date.fromisoformat(self.bot.FAMILY["dad"]["birthdate"])
		self.bot.CARRIE_BIRTHDATE = date.fromisoformat(self.bot.FAMILY["carrie"]["birthdate"])
		self.bot.TONY_BIRTHDATE = date.fromisoformat(self.bot.FAMILY["tony"]["birthdate"])
		self.bot.MARRIAGE_DATE = date.fromisoformat(self.bot.FAMILY["marriage"]["date"])

		self.bot.TOPIC_RULES = self.bot.CHAT_ROUTING["topic_rules"]
		self.bot.CORE_FACT_IDS = self.bot.CHAT_ROUTING["core_fact_ids"]

	def cadence_settings(self):
		defaults = self.bot.runtime_config.cadence_defaults
		configured = self.bot.CADENCE if isinstance(self.bot.CADENCE, dict) else {}
		normalized = dict(defaults)
		for key, default in defaults.items():
			try:
				value = int(configured.get(key, default))
			except (TypeError, ValueError):
				value = default
			normalized[key] = max(1, value)

		return normalized

	def current_persona_preset(self):
		preset = str(self.bot.STYLE.get("persona_preset") or "").strip().lower()
		if preset in self.bot.persona_preset_catalog():
			return preset
		return "classic"

	def evolved_persona_traits(self):
		return [entry["trait"] for entry in self.bot.persona_evolution_history() if entry.get("trait")]

	def active_persona_traits(self, limit=3):
		return [
			entry.get("trait", "")
			for entry in self.bot.active_persona_trait_entries(limit=limit)
			if entry.get("trait")
		]

	def effective_behavior_rules(self):
		style = self.style if isinstance(self.style, dict) else {}
		rules = list(style.get("behavior_rules", []))
		for trait in self.active_persona_traits(limit=3):
			evolved_rule = f"Over time with Tony, you've grown into this trait too: {trait}."
			if evolved_rule not in rules:
				rules.append(evolved_rule)
		return rules

	def living_dad_snapshot(self, limit=3):
		limit = max(1, int(limit or 3))
		persona_shifts = list(reversed(self.bot.persona_evolution_history()[-limit:]))
		wisdom = list(reversed(self.bot.wisdom_catalog()[-limit:]))
		patterns = list(reversed(self.bot.life_patterns()[-limit:]))
		proactive_queue = list(reversed(self.bot.pending_proactive_messages()[-limit:]))
		return {
			"persona_shifts": persona_shifts,
			"wisdom": wisdom,
			"patterns": patterns,
			"proactive_queue": proactive_queue,
			"counts": {
				"persona_shifts": len(self.bot.persona_evolution_history()),
				"wisdom": len(self.bot.wisdom_catalog()),
				"patterns": len(self.bot.life_patterns()),
				"proactive_queue": len(self.bot.pending_proactive_messages()),
			},
		}

	def opening_message_candidates(self):
		configured = self.bot.OPENING_MESSAGES if isinstance(self.bot.OPENING_MESSAGES, list) else []
		candidates = []
		for message in configured:
			normalized = str(message or "").strip()
			if normalized and normalized not in candidates:
				candidates.append(normalized)
		return candidates

	def opening_message(self, default_message="That's my boy. I love hearing that, Tony."):
		self.bot.run_scheduled_proactive_jobs()
		proactive = self.bot.consume_proactive_message()
		if proactive is not None:
			message = proactive.get("message", "")
			if self.bot.memory.should_do_daily_checkin() and "how's your day" not in message.lower():
				return f"{message} How's your day going so far?"
			return message

		if self.bot.memory.should_do_daily_checkin():
			return self.bot.daily_checkin_greeting()

		candidates = self.opening_message_candidates()
		if candidates:
			return candidates[len(self.bot.session_archive()) % len(candidates)]
		return default_message

	def apply_persona_preset(self, preset_key, save=True):
		preset = self.bot.persona_preset_catalog().get(str(preset_key or "").strip().lower())
		if preset is None:
			return False

		style = dict(self.bot.PROFILE.get("style", {}))
		style["name"] = preset["name"]
		style["signoff"] = preset["signoff"]
		style["behavior_rules"] = list(preset["behavior_rules"])
		style["persona_preset"] = str(preset_key).strip().lower()
		style.setdefault("listener_name", "Tony")
		self.bot.PROFILE["style"] = style
		self.refresh_profile_runtime()
		if save:
			self.bot.save_profile()
		return True

	def update_style_profile(self, name=None, listener_name=None, signoff=None, behavior_rules=None, persona_preset=None, save=True):
		style = dict(self.bot.PROFILE.get("style", {}))
		if name is not None:
			style["name"] = str(name).strip() or style.get("name") or "Dad"
		if listener_name is not None:
			style["listener_name"] = str(listener_name).strip() or style.get("listener_name") or "Tony"
		if signoff is not None:
			style["signoff"] = str(signoff).strip() or style.get("signoff") or "Love you, buddy."
		if behavior_rules is not None:
			cleaned_rules = [str(rule).strip() for rule in behavior_rules if str(rule).strip()]
			if cleaned_rules:
				style["behavior_rules"] = cleaned_rules
		if persona_preset is not None:
			style["persona_preset"] = str(persona_preset).strip().lower() or self.current_persona_preset()
		self.bot.PROFILE["style"] = style
		self.refresh_profile_runtime()
		if save:
			self.bot.save_profile()
		return dict(style)

	def update_cadence_profile(self, cadence=None, save=True, **overrides):
		defaults = self.cadence_settings()
		current = self.bot.PROFILE.get("cadence", {})
		merged = dict(current) if isinstance(current, dict) else {}

		if isinstance(cadence, dict):
			merged.update(cadence)
		merged.update(overrides)

		normalized = {}
		for key, default in defaults.items():
			try:
				value = int(merged.get(key, default))
			except (TypeError, ValueError):
				value = default
			normalized[key] = max(1, value)

		self.bot.PROFILE["cadence"] = normalized
		self.refresh_profile_runtime()
		if save:
			self.bot.save_profile()
		return dict(normalized)

	def runtime_settings(self):
		configured = self.bot.RUNTIME if isinstance(self.bot.RUNTIME, dict) else {}
		defaults = {
			"preferred_embedding_models": list(self.bot.runtime_config.preferred_embedding_models),
			"max_thinking_time_seconds": self.bot.runtime_config.stream_timeout_seconds,
			"stream_max_chars": self.bot.runtime_config.stream_max_chars,
			"graph_refresh_debounce_seconds": self.bot.runtime_config.graph_refresh_debounce_seconds,
		}

		raw_models = configured.get("preferred_embedding_models", defaults["preferred_embedding_models"])
		if isinstance(raw_models, str):
			raw_models = [part.strip() for part in raw_models.split(",")]

		preferred_embedding_models = []
		for model_name in raw_models or []:
			normalized = str(model_name or "").strip().lower()
			if normalized and normalized not in preferred_embedding_models:
				preferred_embedding_models.append(normalized)
		if not preferred_embedding_models:
			preferred_embedding_models = list(defaults["preferred_embedding_models"])

		try:
			max_thinking_time_seconds = int(
				configured.get(
					"max_thinking_time_seconds",
					configured.get("stream_timeout_seconds", defaults["max_thinking_time_seconds"]),
				)
			)
		except (TypeError, ValueError):
			max_thinking_time_seconds = defaults["max_thinking_time_seconds"]

		try:
			stream_max_chars = int(configured.get("stream_max_chars", defaults["stream_max_chars"]))
		except (TypeError, ValueError):
			stream_max_chars = defaults["stream_max_chars"]

		try:
			graph_refresh_debounce_seconds = int(
				configured.get("graph_refresh_debounce_seconds", defaults["graph_refresh_debounce_seconds"])
			)
		except (TypeError, ValueError):
			graph_refresh_debounce_seconds = defaults["graph_refresh_debounce_seconds"]

		return {
			"preferred_embedding_models": preferred_embedding_models,
			"max_thinking_time_seconds": max(1, max_thinking_time_seconds),
			"stream_max_chars": max(256, stream_max_chars),
			"graph_refresh_debounce_seconds": max(0, graph_refresh_debounce_seconds),
		}

	def update_runtime_profile(self, settings=None, save=True, **overrides):
		current = self.bot.PROFILE.get("runtime", {})
		merged = dict(current) if isinstance(current, dict) else {}
		if isinstance(settings, dict):
			merged.update(settings)
		merged.update(overrides)

		self.bot.RUNTIME = merged
		normalized = self.runtime_settings()
		self.bot.PROFILE["runtime"] = dict(normalized)
		self.refresh_profile_runtime()
		if save:
			self.bot.save_profile()
		return dict(normalized)

	def update_opening_messages_profile(self, opening_messages=None, save=True):
		values = opening_messages if isinstance(opening_messages, list) else []
		normalized = []
		for message in values:
			cleaned = str(message or "").strip()
			if cleaned and cleaned not in normalized:
				normalized.append(cleaned)

		self.bot.PROFILE["opening_messages"] = normalized
		self.refresh_profile_runtime()
		if save:
			self.bot.save_profile()
		return list(normalized)

	def agentic_tool_settings(self):
		configured = self.bot.AGENTIC_TOOLS if isinstance(self.bot.AGENTIC_TOOLS, dict) else {}
		return {
			"enabled": bool(configured.get("enabled", True)),
			"auto_reminders": bool(configured.get("auto_reminders", True)),
			"auto_web_lookup": bool(configured.get("auto_web_lookup", True)),
		}

	def update_agentic_tool_profile(self, settings=None, save=True, **overrides):
		current = self.bot.PROFILE.get("agentic_tools", {})
		merged = dict(current) if isinstance(current, dict) else {}
		if isinstance(settings, dict):
			merged.update(settings)
		merged.update(overrides)

		normalized = {
			"enabled": bool(merged.get("enabled", True)),
			"auto_reminders": bool(merged.get("auto_reminders", True)),
			"auto_web_lookup": bool(merged.get("auto_web_lookup", True)),
		}
		self.bot.PROFILE["agentic_tools"] = normalized
		self.refresh_profile_runtime()
		if save:
			self.bot.save_profile()
		return dict(normalized)

	def relationship_calibration_settings(self):
		defaults = {
			"enabled": True,
			"protected_moods": ["sad", "stressed", "tired"],
			"trigger_patterns": [
				r"\b(i keep procrastinating|i keep avoiding|i keep sabotaging)\b",
				r"\b(it's (all|always) someone else's fault)\b",
				r"\b(i know it's wrong but i don't care)\b",
				r"\b(should i just give up|i should give up)\b",
			],
			"opening_line": "I love you enough to be honest with you, buddy: part of this pattern sounds like it's hurting you.",
		}
		configured = self.bot.RELATIONSHIP_CALIBRATION if isinstance(self.bot.RELATIONSHIP_CALIBRATION, dict) else {}
		protected_moods = [
			self.bot.normalize_mood(item)
			for item in configured.get("protected_moods", defaults["protected_moods"])
			if self.bot.normalize_mood(item) != "neutral"
		]
		trigger_patterns = [
			str(pattern).strip()
			for pattern in configured.get("trigger_patterns", defaults["trigger_patterns"])
			if str(pattern).strip()
		]
		return {
			"enabled": bool(configured.get("enabled", defaults["enabled"])),
			"protected_moods": protected_moods or list(defaults["protected_moods"]),
			"trigger_patterns": trigger_patterns or list(defaults["trigger_patterns"]),
			"opening_line": str(configured.get("opening_line") or defaults["opening_line"]).strip(),
		}

	def update_relationship_calibration_profile(self, settings=None, save=True, **overrides):
		current = self.bot.PROFILE.get("relationship_calibration", {})
		merged = dict(current) if isinstance(current, dict) else {}
		if isinstance(settings, dict):
			merged.update(settings)
		merged.update(overrides)

		existing_defaults = self.relationship_calibration_settings()
		protected_moods = [
			self.bot.normalize_mood(item)
			for item in merged.get("protected_moods", existing_defaults["protected_moods"])
			if self.bot.normalize_mood(item) != "neutral"
		]
		trigger_patterns = [
			str(pattern).strip()
			for pattern in merged.get("trigger_patterns", existing_defaults["trigger_patterns"])
			if str(pattern).strip()
		]
		normalized = {
			"enabled": bool(merged.get("enabled", existing_defaults["enabled"])),
			"protected_moods": protected_moods or list(existing_defaults["protected_moods"]),
			"trigger_patterns": trigger_patterns or list(existing_defaults["trigger_patterns"]),
			"opening_line": str(merged.get("opening_line") or existing_defaults["opening_line"]).strip(),
		}
		self.bot.PROFILE["relationship_calibration"] = normalized
		self.refresh_profile_runtime()
		if save:
			self.bot.save_profile()
		return dict(normalized)

	def streamlit_security_settings(self):
		if not isinstance(self.bot.SECURITY, dict):
			return {
				"require_pin": False,
				"pin_hash": "",
				"pin_hint": "",
			}

		raw_pin = str(self.bot.SECURITY.get("streamlit_pin") or "").strip()
		pin_hash = str(self.bot.SECURITY.get("streamlit_pin_hash") or "").strip().lower()
		if not pin_hash and raw_pin:
			pin_hash = hashlib.sha256(raw_pin.encode("utf-8")).hexdigest()

		require_pin = bool(self.bot.SECURITY.get("require_pin")) or bool(pin_hash)
		return {
			"require_pin": require_pin,
			"pin_hash": pin_hash,
			"pin_hint": str(self.bot.SECURITY.get("pin_hint") or "").strip(),
		}

	def verify_streamlit_pin(self, pin_attempt):
		security = self.streamlit_security_settings()
		if not security["require_pin"]:
			return True

		attempted_pin = str(pin_attempt or "")
		attempted_hash = hashlib.sha256(attempted_pin.encode("utf-8")).hexdigest()
		return attempted_hash == security["pin_hash"]


__all__ = ["ProfileRuntimeManager"]
