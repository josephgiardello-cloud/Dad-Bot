from __future__ import annotations

from datetime import date

from dadbot.contracts import DadBotContext, SupportsDadBotAccess


class MemoryQueryManager:
	"""Owns memory retrieval scoring, memory replies, and saved-memory CRUD flows."""

	def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot

	def memory_freshness_weight(self, value):
		elapsed_days = self.bot.days_since_iso_date(value)
		if elapsed_days is None:
			return 1.0
		half_life = max(1, int(self.bot.runtime_config.memory_freshness_half_life_days))
		min_weight = float(self.bot.runtime_config.memory_min_freshness_weight)
		raw = 0.5 ** (elapsed_days / float(half_life))
		return round(max(min_weight, min(1.0, raw)), 3)

	@staticmethod
	def memory_impact_score(memory):
		try:
			return float(memory.get("impact_score", 0.0))
		except (AttributeError, TypeError, ValueError):
			return 0.0

	@staticmethod
	def memory_importance_score(memory):
		try:
			return max(0.0, min(1.0, float(memory.get("importance_score", 0.0) or 0.0)))
		except (AttributeError, TypeError, ValueError):
			return 0.0

	def semantic_memory_freshness_weight(self, memory):
		elapsed_days = self.bot.days_since_iso_date(memory.get("updated_at") or memory.get("created_at"))
		if elapsed_days is None or elapsed_days <= 0:
			return 1.0

		impact = self.memory_impact_score(memory)
		if elapsed_days <= 7:
			raw = 0.5 ** (elapsed_days / 7.0)
		else:
			raw = 0.5 * (0.5 ** ((elapsed_days - 7) / 45.0))

		if elapsed_days > 60 and impact < 2.5:
			raw = min(raw, 0.25)
		elif elapsed_days > 30 and impact < 1.5:
			raw = min(raw, 0.35)

		if impact >= 2.5:
			raw = max(raw, 0.45)
		elif impact >= 1.5:
			raw = max(raw, 0.30)

		return round(max(0.08, min(1.0, raw)), 3)

	def recent_memory_topics(self, limit=5):
		topics = []
		desired = max(1, int(limit or 5))
		for entry in reversed(self.bot.session_archive()[-desired * 2:]):
			for topic in entry.get("topics", []):
				normalized = str(topic or "").strip().lower()
				if normalized and normalized != "general" and normalized not in topics:
					topics.append(normalized)
				if len(topics) >= desired:
					return topics
		return topics

	def current_memory_mood_trend(self):
		trend = self.bot.long_term_signals.recent_mood_trend() if self.bot.session_moods else self.bot.last_saved_mood()
		return self.bot.normalize_mood(trend)

	def memory_alignment_weight(self, memory, query_tokens=None, query_category="general", query_mood="neutral", recent_topics=None, mood_trend=None):
		summary = str(memory.get("summary", ""))
		tokens = self.bot.significant_tokens(summary)
		query_tokens = set(query_tokens or [])
		category = str(memory.get("category", "general")).strip().lower() or "general"
		recent_topics = set(recent_topics or [])
		memory_mood = self.bot.normalize_mood(memory.get("mood"))
		impact = self.memory_impact_score(memory)
		topical_overlap = bool(query_tokens & tokens)
		topic_match = (query_category != "general" and category == query_category) or category in recent_topics or topical_overlap

		weight = 1.0
		if query_category != "general" and category != query_category and not topical_overlap:
			weight *= 0.55 if impact < 2.0 else 0.75
		elif recent_topics and category not in recent_topics and query_category == "general" and not topical_overlap:
			weight *= 0.75 if impact < 2.0 else 0.9

		if memory_mood != "neutral":
			if query_mood != "neutral" and memory_mood != query_mood:
				weight *= 0.6 if impact < 2.0 else 0.8
			elif mood_trend not in {None, "", "neutral"} and memory_mood != mood_trend:
				weight *= 0.75 if impact < 2.0 else 0.9

		if not topic_match and memory_mood != "neutral" and query_mood != "neutral" and memory_mood != query_mood and impact < 1.5:
			return 0.0
		return round(max(0.0, min(1.2, weight)), 3)

	def memory_diversity_key(self, memory):
		return self.bot.memory_dedup_key(memory)

	def select_diverse_ranked_memories(self, ranked_items, limit, memory_index=1):
		desired = max(1, int(limit or 1))
		selected = []
		seen_keys = set()
		category_counts = {}

		def category_name(memory):
			return str(memory.get("category", "general")).strip().lower() or "general"

		def try_add(item, max_per_category):
			memory = item[memory_index]
			diversity_key = self.memory_diversity_key(memory)
			category = category_name(memory)
			if diversity_key in seen_keys or category_counts.get(category, 0) >= max_per_category:
				return False
			selected.append(item)
			seen_keys.add(diversity_key)
			category_counts[category] = category_counts.get(category, 0) + 1
			return True

		for item in ranked_items:
			if try_add(item, 1) and len(selected) >= desired:
				return selected

		for item in ranked_items:
			if try_add(item, 2) and len(selected) >= desired:
				return selected

		for item in ranked_items:
			memory = item[memory_index]
			diversity_key = self.memory_diversity_key(memory)
			if diversity_key in seen_keys:
				continue
			selected.append(item)
			seen_keys.add(diversity_key)
			if len(selected) >= desired:
				break

		return selected

	def memory_context_limit_for_input(self, user_input):
		current_mood = self.bot.last_saved_mood()
		baseline_sections = self.bot.base_request_sections(current_mood)
		baseline_sections.extend([
			self.bot.build_daily_checkin_context(current_mood),
			self.bot.build_active_tool_observation_context(),
			self.bot.build_cross_session_context(),
			self.bot.build_session_summary_context(),
			self.bot.build_relevant_context(user_input),
			self.bot.build_wisdom_context(user_input),
			self.bot.build_escalation_context(current_mood, self.bot.session_moods),
		])
		baseline_prompt = "\n\n".join(section for section in baseline_sections if section)
		history_budget = self.bot.prompt_history_token_budget(baseline_prompt, user_input)
		if history_budget < 120:
			return 1
		if history_budget < 220:
			return 2
		if history_budget < 360:
			return 3
		return 4

	def memory_relevance_score(self, user_input, memory_summary):
		query_tokens = self.bot.tokenize(user_input)
		memory_tokens = self.bot.tokenize(memory_summary)
		overlap = query_tokens & memory_tokens

		score = len(overlap)
		if any(term in user_input.lower() for term in ["remember", "recall", "last time", "before", "previous"]):
			score += 2
		return score

	def relevant_archive_entries_for_input(self, user_input, limit=2):
		query_tokens = self.bot.significant_tokens(user_input)
		query_category = self.bot.infer_memory_category(user_input)
		mood_trend = self.current_memory_mood_trend()
		recent_topics = set(self.recent_memory_topics(limit=4))
		scored = []

		for entry in reversed(self.bot.session_archive()):
			topics = [str(topic or "").strip().lower() for topic in entry.get("topics", []) if str(topic or "").strip()]
			searchable = f"{' '.join(topics)} {entry.get('summary', '')}"
			overlap = len(query_tokens & self.bot.significant_tokens(searchable))
			if query_category != "general" and query_category in topics:
				overlap += 2
			elif recent_topics and recent_topics.intersection(topics):
				overlap += 1
			if mood_trend != "neutral" and self.bot.normalize_mood(entry.get("dominant_mood")) == mood_trend:
				overlap += 1

			freshness = self.memory_freshness_weight(entry.get("created_at"))
			score = overlap * max(0.45, freshness)
			if score > 0:
				scored.append((round(score, 4), entry))

		ranked = sorted(
			scored,
			key=lambda item: (item[0], item[1].get("created_at", ""), item[1].get("summary", "")),
			reverse=True,
		)
		return [entry for _, entry in ranked[:max(1, int(limit or 2))]]

	def relevant_memories_for_input(self, user_input, limit=3, graph_result=None):
		effective_limit = max(1, min(int(limit or 3), self.memory_context_limit_for_input(user_input)))
		memories = self.bot.memory_catalog()
		scored_memories = []
		combined_scores = {}
		graph_result = graph_result if graph_result is not None else self.bot.graph_retrieval_for_input(user_input, limit=min(3, effective_limit))
		graph_boost_tokens = set()
		query_tokens = self.bot.significant_tokens(user_input)
		query_category = self.bot.infer_memory_category(user_input)
		query_mood = self.bot.normalize_mood(user_input)
		recent_topics = self.recent_memory_topics(limit=4)
		mood_trend = self.current_memory_mood_trend()

		if graph_result:
			for evidence in graph_result.get("supporting_evidence", []):
				graph_boost_tokens.update(self.bot.significant_tokens(evidence.get("summary", "")))
				graph_boost_tokens.update(self.bot.significant_tokens(" ".join(evidence.get("matched_nodes", []))))

		for memory in memories:
			summary = memory.get("summary", "")
			if not summary:
				continue

			base_score = self.memory_relevance_score(user_input, f"{memory.get('category', '')} {summary}")
			freshness = self.memory_freshness_weight(memory.get("updated_at") or memory.get("created_at"))
			alignment = self.memory_alignment_weight(
				memory,
				query_tokens=query_tokens,
				query_category=query_category,
				query_mood=query_mood,
				recent_topics=recent_topics,
				mood_trend=mood_trend,
			)
			score = base_score * freshness * alignment
			if graph_boost_tokens and graph_boost_tokens & self.bot.significant_tokens(summary):
				score += 1.4
			if score > 0:
				score += min(1.25, max(0.0, self.memory_impact_score(memory)) * 0.25)
				score += self.memory_importance_score(memory) * 0.8
				scored_memories.append((float(score), memory))

		for score, memory in scored_memories:
			combined_scores[self.bot.semantic_memory_key(memory)] = (score, memory)

		semantic_matches = self.bot.semantic_memory_matches(user_input, memories, limit=max(effective_limit * 3, 4))
		for semantic_score, memory in semantic_matches:
			memory_key = self.bot.semantic_memory_key(memory)
			existing = combined_scores.get(memory_key)
			if existing is None:
				combined_scores[memory_key] = (semantic_score, memory)
				continue
			combined_scores[memory_key] = (existing[0] + semantic_score, memory)

		ranked = sorted(
			combined_scores.values(),
			key=lambda item: (item[0], item[1].get("updated_at", ""), item[1].get("summary", "")),
			reverse=True,
		)
		selected = self.select_diverse_ranked_memories(ranked, effective_limit)
		return [memory for _, memory in selected]

	@staticmethod
	def retrieval_strategy_for_input(user_input):
		text = str(user_input or "").strip().lower()
		if not text:
			return "hybrid"
		if any(token in text for token in ["pattern", "trend", "chapter", "timeline", "arc"]):
			return "graph_heavy"
		if any(token in text for token in ["remember", "recall", "what do you remember", "last time", "before"]):
			return "consolidated_heavy"
		if any(token in text for token in ["exactly", "specific", "detail", "quote", "verbatim"]):
			return "semantic_heavy"
		return "hybrid"

	def retrieve_context(self, user_input, strategy="hybrid", limit=4):
		requested = str(strategy or "hybrid").strip().lower() or "hybrid"
		effective_strategy = self.retrieval_strategy_for_input(user_input) if requested == "auto" else requested
		try:
			effective_limit = max(1, int(limit or 4))
		except (TypeError, ValueError):
			effective_limit = 4

		graph_limit = min(4, effective_limit)
		semantic_limit = min(5, max(1, effective_limit))
		archive_limit = min(3, effective_limit)
		consolidated_limit = min(4, effective_limit)

		if effective_strategy == "graph_heavy":
			graph_limit = min(5, max(2, effective_limit))
			semantic_limit = max(1, effective_limit - 1)
		elif effective_strategy == "semantic_heavy":
			semantic_limit = min(6, max(2, effective_limit + 1))
			graph_limit = max(1, effective_limit - 2)
		elif effective_strategy == "consolidated_heavy":
			consolidated_limit = min(5, max(2, effective_limit))
			graph_limit = max(1, effective_limit - 2)

		graph_result = self.bot.graph_retrieval_for_input(user_input, limit=graph_limit)
		archive_entries = [] if graph_result else self.relevant_archive_entries_for_input(user_input, limit=archive_limit)
		semantic_memories = self.relevant_memories_for_input(user_input, limit=semantic_limit, graph_result=graph_result)
		consolidated_memories = self.bot.select_active_consolidated_memories(user_input, max_items=consolidated_limit)

		bundle = []
		for memory in semantic_memories:
			freshness = self.memory_freshness_weight(memory.get("updated_at") or memory.get("created_at"))
			impact_importance = min(1.0, max(0.0, self.memory_impact_score(memory) / 3.0))
			importance = self.memory_importance_score(memory)
			score = 0.45 * freshness + 0.25 * impact_importance + 0.3 * importance
			bundle.append({"type": "semantic", "score": round(score, 4), "payload": memory})

		for entry in consolidated_memories:
			confidence = max(0.05, min(1.0, float(entry.get("confidence", 0.5) or 0.5)))
			importance = max(0.0, min(1.0, float(entry.get("importance_score", 0.0) or 0.0)))
			recency = self.memory_freshness_weight(entry.get("updated_at"))
			score = 0.35 * confidence + 0.4 * importance + 0.25 * recency
			bundle.append({"type": "consolidated", "score": round(score, 4), "payload": entry})

		if graph_result:
			bundle.append(
				{
					"type": "graph",
					"score": 0.72,
					"payload": {
						"summary_lines": list(graph_result.get("summary_lines", []))[:4],
						"compressed_summary": graph_result.get("compressed_summary", ""),
					},
				}
			)

		for entry in archive_entries:
			freshness = self.memory_freshness_weight(entry.get("created_at"))
			bundle.append({"type": "archive", "score": round(0.55 * freshness, 4), "payload": entry})

		ranked_bundle = sorted(bundle, key=lambda item: (item.get("score", 0.0), str(item.get("type", ""))), reverse=True)
		return {
			"strategy": effective_strategy,
			"semantic_memories": semantic_memories,
			"consolidated_memories": consolidated_memories,
			"graph_result": graph_result,
			"archive_entries": archive_entries,
			"bundle": ranked_bundle[: max(3, effective_limit + 1)],
		}

	@staticmethod
	def format_memories_for_reply(memories):
		if not memories:
			return ""

		grouped = {}
		for memory in memories:
			category = memory.get("category", "general").title()
			grouped.setdefault(category, []).append(memory["summary"].rstrip("."))

		parts = []
		for category, summaries in grouped.items():
			parts.append(f"{category}: {'; '.join(summaries)}")

		return " | ".join(parts)

	def get_memory_reply(self, user_input):
		message = user_input.lower()

		if not any(term in message for term in ["remember", "recall", "what do you remember", "last time", "before"]):
			return None

		memories = self.relevant_memories_for_input(user_input)
		if not memories:
			return None

		remembered_points = self.format_memories_for_reply(memories)
		return f"Yeah, buddy, I remember a few things you've shared with me before. {remembered_points}"

	def find_memory_matches(self, query):
		query_tokens = self.bot.tokenize(query)
		matches = []

		for memory in self.bot.memory_catalog():
			searchable = f"{memory.get('category', '')} {memory.get('summary', '')}"
			overlap = query_tokens & self.bot.tokenize(searchable)
			if overlap:
				matches.append((len(overlap), memory))

		matches.sort(key=lambda item: item[0], reverse=True)
		return [memory for _, memory in matches]

	def add_memory(self, summary, category=None):
		today_stamp = date.today().isoformat()
		normalized = self.bot.normalize_memory_entry({
			"summary": summary,
			"category": category,
			"mood": self.bot.last_saved_mood(),
			"created_at": today_stamp,
			"updated_at": today_stamp,
		})
		if normalized is None:
			return None

		existing = self.bot.memory_catalog()
		normalized_lookup = {self.bot.normalize_memory_text(memory["summary"]): memory for memory in existing}
		key = self.bot.normalize_memory_text(normalized["summary"])

		if key in normalized_lookup:
			memory = normalized_lookup[key]
			memory["updated_at"] = today_stamp
			memory["category"] = normalized["category"]
			if normalized.get("mood"):
				memory["mood"] = normalized["mood"]
			self.bot.mutate_memory_store(memories=existing)
			self.bot.queue_semantic_memory_index(existing)
			return memory

		existing.append(normalized)
		self.bot.save_memory_catalog(existing)
		return normalized

	def update_memory_entry(self, original_summary, new_summary, category=None, mood=None):
		current_catalog = self.bot.memory_catalog()
		target_key = self.bot.normalize_memory_text(original_summary)
		updated_memories = []
		target_found = False

		for memory in current_catalog:
			if self.bot.normalize_memory_text(memory.get("summary", "")) != target_key:
				updated_memories.append(memory)
				continue

			replacement = self.bot.normalize_memory_entry({
				"summary": new_summary,
				"category": category if category not in {None, ""} else memory.get("category"),
				"mood": mood if mood not in {None, ""} else memory.get("mood"),
				"created_at": memory.get("created_at") or date.today().isoformat(),
				"updated_at": date.today().isoformat(),
			})
			if replacement is None:
				return None

			updated_memories.append(replacement)
			target_found = True

		if not target_found:
			return None

		self.bot.save_memory_catalog(updated_memories)
		return next(
			(
				memory for memory in self.bot.memory_catalog()
				if self.bot.normalize_memory_text(memory.get("summary", "")) == self.bot.normalize_memory_text(new_summary)
			),
			None,
		)

	def delete_memory_entry(self, summary):
		target_key = self.bot.normalize_memory_text(summary)
		current_catalog = self.bot.memory_catalog()
		kept = []
		removed = []

		for memory in current_catalog:
			if self.bot.normalize_memory_text(memory.get("summary", "")) == target_key:
				removed.append(memory)
			else:
				kept.append(memory)

		if not removed:
			return []

		self.bot.save_memory_catalog(kept)
		return removed

	def forget_memories(self, query):
		matches = self.find_memory_matches(query)
		if not matches:
			return []

		match_keys = {self.bot.normalize_memory_text(memory["summary"]) for memory in matches}
		current_catalog = self.bot.memory_catalog()
		kept = [memory for memory in current_catalog if self.bot.normalize_memory_text(memory["summary"]) not in match_keys]
		removed = [memory for memory in current_catalog if self.bot.normalize_memory_text(memory["summary"]) in match_keys]
		self.bot.save_memory_catalog(kept)
		return removed


__all__ = ["MemoryQueryManager"]
