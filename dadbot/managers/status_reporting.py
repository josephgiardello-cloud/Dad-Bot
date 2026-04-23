from __future__ import annotations

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.models import ActiveThreadSnapshot, BackgroundTaskOverview, CircuitBreakerStatusSnapshot, DashboardStatusSnapshot, GraphFallbackStatusSnapshot, MemoryContextStatusSnapshot, ModerationSnapshot, PersistenceStatusSnapshot, PromptGuardStatusSnapshot, RelationshipStatusSnapshot, RuntimeHealthSnapshot, RuntimeHealthTrendPoint, RuntimeIssueSnapshot, RuntimeStatusSnapshot, SecurityStatusSnapshot, ServiceStatusSnapshot, SessionStatusSnapshot, StatusTraitMetric, ThreadsStatusSnapshot, VisionStatusSnapshot
from dadbot_system import DadServiceClient


class StatusReportingManager:
	"""Owns service health snapshots, dashboard payloads, and human-readable status formatting."""

	def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot

	@staticmethod
	def _graph_fallback_message(event_count: int, degraded_mode: str) -> str:
		if event_count <= 0:
			return "Graph turn orchestration is healthy."
		mode_label = str(degraded_mode or "legacy").replace("_", " ")
		if event_count == 1:
			return f"Graph orchestration degraded once and switched to {mode_label}. Replies continue through legacy turn processing."
		return f"Graph orchestration degraded {event_count} times and switched to {mode_label}. Replies continue through legacy turn processing."

	def graph_fallback_snapshot(self) -> dict:
		issues = list(self.bot.recent_runtime_issues(limit=12) or [])
		graph_issues = [
			item
			for item in issues
			if str(item.get("purpose") or "").strip().lower() == "turn_graph"
		]
		latest = graph_issues[0] if graph_issues else {}
		metadata = dict(latest.get("metadata") or {})
		degraded_mode = str(metadata.get("degraded_mode") or "none").strip() or "none"
		payload = {
			"active": bool(graph_issues),
			"degraded_mode": degraded_mode,
			"event_count": len(graph_issues),
			"last_timestamp": str(latest.get("timestamp") or "").strip(),
			"last_purpose": str(latest.get("purpose") or "").strip(),
			"last_fallback": str(latest.get("fallback") or "").strip(),
			"last_error": str(latest.get("error") or "").strip(),
			"message": self._graph_fallback_message(len(graph_issues), degraded_mode),
		}
		return GraphFallbackStatusSnapshot.model_validate(payload).model_dump(mode="python")

	@staticmethod
	def _reputation_score(relationship: dict) -> int:
		trust = int(relationship.get("trust_level", 0) or 0)
		openness = int(relationship.get("openness_level", 0) or 0)
		return max(0, min(100, int(round((trust + openness) / 2))))

	def circuit_breaker_snapshot(self, *, health: dict, relationship: dict, graph_fallback: dict | None = None) -> dict:
		fallback = dict(graph_fallback or {})
		last_mood = str(self.bot.last_saved_mood() or "neutral")
		reasons = []
		if health.get("clarification_recommended"):
			reasons.append(str(health.get("clarification_message") or "Dad asked for a short clarification.").strip())
		if fallback.get("active"):
			mode = str(fallback.get("degraded_mode") or "legacy").replace("_", " ")
			reasons.append(f"Turn graph degraded and switched to {mode}.")
		if str(health.get("level") or "green").strip().lower() == "red":
			severity = "high"
		elif reasons:
			severity = "medium"
		else:
			severity = "low"
		message = reasons[0] if reasons else "Dad is tracking well right now."
		active = bool(reasons)
		return CircuitBreakerStatusSnapshot.model_validate(
			{
				"active": active,
				"severity": severity,
				"title": "Dad needs a quick clarification" if active else "Dad is steady",
				"message": message,
				"reasons": reasons,
				"suggested_prompts": [
					"Can you answer using just the most important point?",
					"Want me to restate the goal in one sentence?",
					"Should I focus on the practical next step or the emotional side?",
				] if active else [],
				"reputation_score": self._reputation_score(relationship),
				"current_mood": last_mood,
				"reasoning_confidence": float(health.get("reasoning_confidence", 1.0) or 1.0),
			}
		).model_dump(mode="python")

	def status_snapshot(self) -> dict:
		relationship = self.bot.relationship.snapshot()
		semantic_status = self.bot.semantic_memory_status()
		health = RuntimeHealthSnapshot.model_validate(
			self.bot.current_runtime_health_snapshot(log_warnings=False, persist=True)
		).model_dump(mode="python")
		top_trait_entries = self.bot.active_persona_trait_entries(limit=2)
		top_trait_metrics = [
			StatusTraitMetric.model_validate(
				{
					"trait": entry.get("trait", ""),
					"strength": round(float(entry.get("strength", 1.0)), 2),
					"impact_score": round(float(entry.get("impact_score", 0.0)), 2),
				}
			).model_dump(mode="python")
			for entry in top_trait_entries
			if entry.get("trait")
		]
		validated = RuntimeStatusSnapshot.model_validate(
			{
				"active_model": self.bot.ACTIVE_MODEL,
				"embedding_model": semantic_status.get("embedding_model") or "inactive",
				"saved_memories": len(self.bot.memory_catalog()),
				"archived_chats": len(self.bot.session_archive()),
				"pending_proactive": len(self.bot.pending_proactive_messages()),
				"trust_level": relationship.get("trust_level", 0),
				"openness_level": relationship.get("openness_level", 0),
				"emotional_momentum": relationship.get("emotional_momentum", "steady"),
				"last_mood": self.bot.last_saved_mood(),
				"tenant_id": self.bot.TENANT_ID,
				"top_trait_metrics": top_trait_metrics,
				"health": health,
			}
		)
		return validated.model_dump(mode="python")

	def service_status_snapshot(self) -> dict:
		client = DadServiceClient()
		snapshot = {
			"base_url": client.base_url,
			"event_stream_url": client.session_event_stream_url(self.bot.runtime_state_container.session_id, tenant_id=self.bot.TENANT_ID),
			"reachable": False,
			"port_open": False,
			"status": "offline",
			"workers": 0,
			"queue_backend": "",
			"state_backend": "",
			"service_name": "",
			"error": "",
		}

		try:
			snapshot["port_open"] = bool(client._port_is_open())
		except Exception:
			snapshot["port_open"] = False

		try:
			payload = client.health()
		except Exception as exc:
			snapshot["status"] = "unhealthy" if snapshot["port_open"] else "offline"
			snapshot["error"] = str(exc).strip() or exc.__class__.__name__
			return snapshot

		try:
			workers = int(payload.get("workers") or 0)
		except (TypeError, ValueError):
			workers = 0

		status = str(payload.get("status") or "unknown").strip().lower() or "unknown"
		snapshot.update(
			{
				"reachable": status == "ok",
				"status": status,
				"workers": max(0, workers),
				"queue_backend": str(payload.get("queue_backend") or "").strip(),
				"state_backend": str(payload.get("state_backend") or "").strip(),
				"service_name": str(payload.get("service_name") or "").strip(),
				"error": "",
			}
		)
		return ServiceStatusSnapshot.model_validate(snapshot).model_dump(mode="python")

	def dashboard_status_snapshot(self) -> dict:
		self.bot.ensure_chat_thread_state(preserve_active_runtime=True)
		status = self.status_snapshot()
		semantic_status = self.bot.semantic_memory_status()
		planner_debug = self.bot.planner_debug_snapshot()
		service_status = self.service_status_snapshot()
		persistence = self.bot.customer_persistence_status()
		moderation = self.bot.moderation_snapshot()
		vision_ready, vision_message = self.bot.vision_fallback_status()
		runtime = self.bot.runtime_settings()
		tool_settings = self.bot.agentic_tool_settings()
		security = self.bot.streamlit_security_settings()
		relationship = self.bot.relationship.snapshot()
		maintenance = self.bot.maintenance_snapshot()
		supervisor = self.bot.reply_supervisor_snapshot()
		living = self.bot.profile_runtime.living_dad_snapshot(limit=2)
		active_thread = dict(self.bot.active_chat_thread() or {})
		threads = self.bot.list_chat_threads()
		open_threads = len([thread for thread in threads if not thread.get("closed")])
		history_messages = len([message for message in self.bot.history if message.get("role") in {"user", "assistant"}])
		top_topics = []
		for topic in relationship.get("top_topics", [])[:4]:
			cleaned = str(topic or "").strip().lower()
			if cleaned and cleaned not in top_topics:
				top_topics.append(cleaned)
		recent_runtime_issues = [
			RuntimeIssueSnapshot.model_validate(item).model_dump(mode="python")
			for item in self.bot.recent_runtime_issues(limit=3)
		]
		graph_fallback = self.graph_fallback_snapshot()
		memory_context = MemoryContextStatusSnapshot.model_validate(self.bot.memory_context_stats()).model_dump(mode="python")
		prompt_guard = PromptGuardStatusSnapshot.model_validate(self.bot.prompt_guard_stats()).model_dump(mode="python")
		health = RuntimeHealthSnapshot.model_validate(
			self.bot.current_runtime_health_snapshot(log_warnings=True, persist=True)
		).model_dump(mode="python")
		circuit_breaker = self.circuit_breaker_snapshot(
			health=health,
			relationship=relationship,
			graph_fallback=graph_fallback,
		)
		health_history = [
			RuntimeHealthTrendPoint.model_validate(item).model_dump(mode="python")
			for item in self.bot.health_history(limit=72)
		]
		relationship_history = [dict(item) for item in self.bot.relationship_history(limit=90)]
		memory_contradictions = [dict(item) for item in self.bot.consolidated_contradictions(limit=8)]

		validated = DashboardStatusSnapshot.model_validate(
			{
				"status": status,
				"service": service_status,
				"persistence": PersistenceStatusSnapshot.model_validate(persistence).model_dump(mode="python"),
				"moderation": ModerationSnapshot.model_validate(moderation).model_dump(mode="python"),
				"background_tasks": BackgroundTaskOverview.model_validate(self.bot.background_task_snapshot()).model_dump(mode="python"),
				"semantic_memory": dict(semantic_status or {}),
				"vision": VisionStatusSnapshot.model_validate({"ready": vision_ready, "message": vision_message}).model_dump(mode="python"),
				"planner_debug": planner_debug,
				"runtime": dict(runtime or {}),
				"agentic_tools": dict(tool_settings or {}),
				"security": SecurityStatusSnapshot.model_validate(
					{
						"require_pin": bool(security.get("require_pin")),
						"has_pin_hint": bool(str(security.get("pin_hint") or "").strip()),
					}
				).model_dump(mode="python"),
				"session": SessionStatusSnapshot.model_validate(
					{
						"turn_count": self.bot.session_turn_count(),
						"history_messages": history_messages,
						"summary_updated_at": self.bot.session_summary_updated_at,
						"summary_covered_messages": self.bot.session_summary_covered_messages,
					}
				).model_dump(mode="python"),
				"threads": ThreadsStatusSnapshot.model_validate(
					{
						"total": len(threads),
						"open": open_threads,
						"closed": max(0, len(threads) - open_threads),
					}
				).model_dump(mode="python"),
				"active_thread": ActiveThreadSnapshot.model_validate(
					{
						"thread_id": active_thread.get("thread_id", ""),
						"title": active_thread.get("title", ""),
						"turn_count": int(active_thread.get("turn_count", 0) or 0),
						"closed": bool(active_thread.get("closed")),
						"last_message": str(active_thread.get("last_message") or "").strip(),
					}
				).model_dump(mode="python"),
				"relationship": RelationshipStatusSnapshot.model_validate(
					{
						"trust_level": int(relationship.get("trust_level", 0) or 0),
						"openness_level": int(relationship.get("openness_level", 0) or 0),
						"emotional_momentum": str(relationship.get("emotional_momentum") or "steady").strip() or "steady",
						"active_hypothesis": str(relationship.get("active_hypothesis") or "supportive_baseline").strip() or "supportive_baseline",
						"active_hypothesis_label": str(relationship.get("active_hypothesis_label") or "Supportive Baseline").strip() or "Supportive Baseline",
						"active_hypothesis_probability": float(relationship.get("active_hypothesis_probability", 0.0) or 0.0),
						"hypotheses": [dict(item) for item in relationship.get("hypotheses", [])[:3]],
						"top_topics": top_topics,
					}
				).model_dump(mode="python"),
				"relationship_history": relationship_history,
				"memory_contradictions": memory_contradictions,
				"memory_context": memory_context,
				"prompt_guard": prompt_guard,
				"health": health,
				"circuit_breaker": circuit_breaker,
				"health_history": health_history,
				"recent_runtime_issues": recent_runtime_issues,
				"graph_fallback": graph_fallback,
				"maintenance": dict(maintenance or {}),
				"supervisor": dict(supervisor or {}),
				"living": dict(living or {}),
				"turn_pipeline": self.bot.turn_pipeline_snapshot(),
			}
		)
		return validated.model_dump(mode="python")

	def ui_shell_snapshot(self) -> dict:
		try:
			models = list(self.bot.available_model_names() or [])
		except Exception as exc:
			models = []
			connection_note = str(exc).strip() or "offline"
		else:
			connection_note = "online" if models else "no models"

		health = RuntimeHealthSnapshot.model_validate(
			self.bot.current_runtime_health_snapshot(force=True, log_warnings=False, persist=True)
		).model_dump(mode="python")
		relationship = dict(self.bot.relationship.snapshot() or {})
		internal_state = dict(self.bot.internal_state_manager.snapshot() or {})
		recent_moods = [dict(item) for item in self.bot.recent_mood_history()[-8:]]
		circuit_breaker = self.circuit_breaker_snapshot(
			health=health,
			relationship=relationship,
			graph_fallback=self.graph_fallback_snapshot(),
		)
		local_mcp = dict(self.bot.local_mcp_status() or {})
		persona_traits = [
			entry.get("trait", "")
			for entry in self.bot.active_persona_trait_entries(limit=4)
			if entry.get("trait")
		]
		return {
			"ollama": {
				"connected": bool(models),
				"model_count": len(models),
				"connection_note": connection_note,
			},
			"health": health,
			"reputation_score": self._reputation_score(relationship),
			"local_mcp": local_mcp,
			"narrative_memory_count": len(self.bot.narrative_memories()),
			"circuit_breaker": circuit_breaker,
			"persona_preset": str(self.bot.current_persona_preset() or "classic"),
			"last_mood": str(self.bot.last_saved_mood() or "neutral"),
			"internal_debug": {
				"relationship_hypotheses": list(relationship.get("hypotheses", [])[:4]),
				"active_persona_traits": persona_traits,
				"health_snapshot": health,
				"recent_moods": recent_moods,
				"inner_state": internal_state,
			},
		}

	def format_status_snapshot(self) -> str:
		snapshot = self.status_snapshot()
		dashboard = self.dashboard_status_snapshot()
		graph_fallback = dashboard.get("graph_fallback", {})
		memory_context = dashboard.get("memory_context", {})
		prompt_guard = dashboard.get("prompt_guard", {})
		health = dashboard.get("health", {})
		health_warnings = [str(item).strip() for item in health.get("warnings", []) if str(item).strip()]
		warning_text = " | ".join(health_warnings) if health_warnings else "none"
		memory_pruned = "pruned" if memory_context.get("pruned") else "full"
		if snapshot["top_trait_metrics"]:
			trait_text = "; ".join(
				f"{item['trait']} (strength={item['strength']:.2f}, impact={item['impact_score']:.2f})"
				for item in snapshot["top_trait_metrics"]
			)
		else:
			trait_text = "none yet"
		return (
			f"Status check, buddy: model={snapshot['active_model']}, embeddings={snapshot['embedding_model']}, "
			f"saved memories={snapshot['saved_memories']}, archived chats={snapshot['archived_chats']}, "
			f"pending proactive={snapshot['pending_proactive']}, trust={snapshot['trust_level']}/100, "
			f"openness={snapshot['openness_level']}/100, momentum={snapshot['emotional_momentum']}, "
			f"last mood={snapshot['last_mood']}, top traits={trait_text}, "
			f"memory context={memory_context.get('tokens', 0)}/{memory_context.get('budget_tokens', 0)} ({memory_pruned}), "
			f"prompt guard trims={prompt_guard.get('trim_count', 0)}, "
			f"recent degradations={len(self.bot.recent_runtime_issues(limit=3))}, "
			f"graph fallback={graph_fallback.get('degraded_mode', 'none')} ({graph_fallback.get('event_count', 0)}), "
			f"health={health.get('level', 'green')} (worker limit={health.get('background_worker_limit', 0)}, "
			f"prompt factor={health.get('prompt_budget_factor', 1.0):.2f}, warnings={warning_text})."
		)

	def format_dad_snapshot(self) -> str:
		living = self.bot.profile_runtime.living_dad_snapshot(limit=2)
		relationship = self.bot.relationship.snapshot()
		memory_context = self.bot.memory_context_stats()
		prompt_guard = self.bot.prompt_guard_stats()
		persona_trait_entries = self.bot.active_persona_trait_entries(limit=2)
		top_topics = relationship.get("top_topics", []) or []
		if persona_trait_entries:
			trait_text = "; ".join(
				f"{entry.get('trait', '')} (strength={self.bot.long_term_signals.decayed_trait_strength(entry):.2f}, impact={self.bot.long_term_signals.trait_impact(entry):.2f})"
				for entry in persona_trait_entries
				if entry.get("trait")
			)
		else:
			trait_text = "none yet"
		positive_trait_text = ", ".join(self.bot.most_positive_persona_traits(limit=2)) or "none yet"
		topic_text = ", ".join(top_topics) if top_topics else "nothing recurring yet"
		return (
			f"Dad snapshot: preset={self.bot.current_persona_preset()}, evolved traits={trait_text}, "
			f"best trait impact={positive_trait_text}, "
			f"momentum={relationship.get('emotional_momentum', 'steady')}, top topics={topic_text}, "
			f"memory context={memory_context.get('tokens', 0)}/{memory_context.get('budget_tokens', 0)}, "
			f"guard trims={prompt_guard.get('trim_count', 0)}, "
			f"health quiet mode={self.bot.health_quiet_mode_enabled()}, "
			f"wisdom notes={living['counts']['wisdom']}, patterns={living['counts']['patterns']}, "
			f"queued proactive={living['counts']['proactive_queue']}."
		)

	def format_proactive_snapshot(self) -> str:
		queued = self.bot.pending_proactive_messages()
		if not queued:
			return "No proactive openings are queued right now, buddy."

		items = []
		for entry in queued[:5]:
			source = str(entry.get("source", "general")).replace("-", " ")
			message = str(entry.get("message", "")).strip()
			if message:
				items.append(f"[{source}] {message}")
		return "Queued proactive openings: " + " | ".join(items) + "."


__all__ = ["StatusReportingManager"]