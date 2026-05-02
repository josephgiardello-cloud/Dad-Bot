from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from datetime import date

from dadbot.memory.conflict_detector import ConflictDetector
from dadbot.memory.scoring import MemoryScorer
from dadbot.pii_scrubber import scrub_memory_entry

logger = logging.getLogger(__name__)


class MemoryCoordinator:
    """Router of memory lifecycle stages.

    Delegates scoring math to ``MemoryScorer`` and contradiction logic to
    ``ConflictDetector``.  This class owns only the orchestration of each
    stage: extraction â†’ reinforcement â†’ selection â†’ consolidation â†’ archive.
    """

    def __init__(self, bot) -> None:
        self.bot = bot
        self._active_consolidated_selection_cache: dict = {}
        self.scorer = MemoryScorer(bot)
        self.conflicts = ConflictDetector(bot, self.scorer)

    def _require_turn_temporal(self, turn_context=None):
        temporal = getattr(turn_context, "temporal", None)
        if temporal is None:
            raise RuntimeError(
                "TemporalNode missing — deterministic execution violated",
            )
        wall_time = str(getattr(temporal, "wall_time", "")).strip()
        wall_date = str(getattr(temporal, "wall_date", "")).strip()
        if not wall_time or not wall_date:
            raise RuntimeError(
                "TemporalNode missing — deterministic execution violated",
            )
        return temporal

    def _assert_save_commit_boundary(self, turn_context=None) -> bool:
        commit_active = bool(getattr(self.bot, "_graph_commit_active", False))
        active_stage = str(getattr(turn_context, "state", {}).get("_active_graph_stage") or "").strip().lower()
        if not commit_active or active_stage not in {"save", ""}:
            return False
        return True

    def _turn_wall_time(self, turn_context=None) -> str:
        temporal = self._require_turn_temporal(turn_context)
        wall_time = str(getattr(temporal, "wall_time", "")).strip()
        if not wall_time:
            raise RuntimeError(
                "TemporalNode missing — deterministic execution violated",
            )
        return wall_time

    def _turn_wall_date(self, turn_context=None) -> str:
        temporal = self._require_turn_temporal(turn_context)
        wall_date = str(getattr(temporal, "wall_date", "")).strip()
        if not wall_date:
            raise RuntimeError(
                "TemporalNode missing — deterministic execution violated",
            )
        return wall_date

    @staticmethod
    def memory_extraction_prompt():
        return """
You extract durable memory summaries from Tony's prior messages.
Return only JSON.
Rules:
- Include only lasting personal context or ongoing concerns Tony shared.
    - Prefer short, specific statements Dad can actually use later.
    - Keep concrete details over generic labels. Good: "Tony is stressed about a work deadline because his boss moved it up." Bad: "Tony feels stressed."
    - Capture actionable context when present, including what Tony is dealing with, why it matters, or what follow-up would help.
- Return an array of objects with keys 'summary', 'category', and 'mood'.
- Return ONLY a JSON array of objects. Example:
    [{"summary": "Tony is stressed about a work deadline his boss moved up.", "category": "work", "mood": "stressed"}]
- Do not include one-off greetings, filler, or facts already covered by Dad's built-in profile.
    - Avoid vague summaries like "personal struggles", "mental health", or "emotional state" unless the user gave a specific enduring detail.
- Keep each memory under 25 words.
- If there is nothing durable to remember, return [].
""".strip()

    def normalize_parsed_memories(self, parsed):
        if isinstance(parsed, dict):
            if "memories" in parsed and isinstance(parsed["memories"], list):
                parsed = parsed["memories"]
            else:
                parsed = [
                    {
                        "summary": value,
                        "category": key,
                        "mood": self.bot.last_saved_mood(),
                    }
                    for key, value in parsed.items()
                ]

        if not isinstance(parsed, list):
            return []

        cleaned = []
        for item in self.bot.flatten_memory_payload(parsed):
            if isinstance(item, str):
                summary = self.bot.coerce_memory_summary(item)
                if summary:
                    cleaned.append(
                        {
                            "summary": self.bot.naturalize_memory_summary(summary),
                            "category": self.bot.infer_memory_category(summary),
                            "mood": self.bot.last_saved_mood(),
                        },
                    )
            elif isinstance(item, dict):
                summary = self.bot.coerce_memory_summary(item.get("summary", ""))
                if summary:
                    cleaned.append(
                        {
                            "summary": self.bot.naturalize_memory_summary(summary),
                            "category": item.get("category") or self.bot.infer_memory_category(summary),
                            "mood": self.bot.normalize_mood(
                                item.get("mood") or self.bot.last_saved_mood(),
                            ),
                        },
                    )

        return cleaned[:5]

    def extract_session_memories(self, history):
        transcript = self.bot.build_memory_transcript(history)

        if not transcript.strip():
            return []

        try:
            response = self.bot.call_ollama_chat(
                messages=[
                    {"role": "system", "content": self.memory_extraction_prompt()},
                    {"role": "user", "content": transcript},
                ],
                response_format="json",
                purpose="memory extraction",
            )
            content = self.bot.extract_ollama_message_content(response)
        except (RuntimeError, KeyError, TypeError) as exc:
            self.bot.record_runtime_issue(
                "memory extraction",
                "skipping new durable memory capture for this pass",
                exc,
            )
            return []

        try:
            parsed = self.bot.parse_model_json_content(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Memory extraction returned invalid JSON: %s", exc)
            return []

        return self.normalize_parsed_memories(parsed)

    def update_memory_store(self, history, turn_context=None):
        if not self._assert_save_commit_boundary(turn_context):
            return
        extracted_memories = self.extract_session_memories(history)

        if not extracted_memories:
            self.apply_memory_reinforcement_signals(history, turn_context=turn_context)
            return

        existing = self.bot.memory_catalog()
        normalized_existing = {self.bot.normalize_memory_text(memory.get("summary", "")): memory for memory in existing}
        today_stamp = self._turn_wall_date(turn_context)

        for memory_entry in extracted_memories:
            memory = self.bot.normalize_memory_entry(
                {
                    "summary": memory_entry.get("summary", ""),
                    "category": memory_entry.get("category"),
                    "mood": memory_entry.get("mood"),
                    "created_at": today_stamp,
                    "updated_at": today_stamp,
                },
            )
            if memory is None:
                continue

            normalized = self.bot.normalize_memory_text(memory["summary"])
            if not normalized:
                continue

            if normalized in normalized_existing:
                normalized_existing[normalized]["updated_at"] = today_stamp
                normalized_existing[normalized]["category"] = memory["category"]
                if memory.get("mood"):
                    normalized_existing[normalized]["mood"] = memory["mood"]
                rescored = self.score_memory_entry(normalized_existing[normalized])
                normalized_existing[normalized].update(rescored)
                continue

            memory = scrub_memory_entry(memory)
            memory.update(self.score_memory_entry(memory))
            existing.append(memory)
            normalized_existing[normalized] = memory

        self.apply_memory_reinforcement_signals(
            history,
            existing_catalog=existing,
            turn_context=turn_context,
        )

        self.bot.save_memory_catalog(existing)

    # ------------------------------------------------------------------ #
    # Stage: Scoring  (delegates to MemoryScorer)                         #
    # ------------------------------------------------------------------ #

    def score_memory_entry(self, memory: dict) -> dict:
        return self.scorer.score_memory_entry(memory)

    def consolidated_importance_score(self, entry: dict) -> float:
        return self.scorer.consolidated_importance_score(entry)

    def contradiction_weight(self, left: dict, right: dict) -> float:
        return self.scorer.contradiction_weight(left, right)

    def consolidated_resolution_rank(self, entry: dict) -> float:
        return self.scorer.consolidated_resolution_rank(entry)

    # ------------------------------------------------------------------ #
    # Stage: Contradiction  (delegates to ConflictDetector)               #
    # ------------------------------------------------------------------ #

    def contradiction_signal_reason(
        self,
        left_summary: str,
        right_summary: str,
    ) -> str | None:
        return self.conflicts.contradiction_signal_reason(left_summary, right_summary)

    def detect_memory_contradictions(
        self,
        memories: list,
        existing_insights: list | None = None,
    ) -> list:
        return self.conflicts.detect_memory_contradictions(memories, existing_insights)

    def consolidated_contradictions(self, limit: int = 8) -> list:
        return self.conflicts.consolidated_contradictions(limit)

    def resolve_consolidated_contradiction(
        self,
        left_summary: str,
        right_summary: str,
        keep: str = "auto",
        reason: str = "user_review",
    ) -> dict | None:
        return self.conflicts.resolve_consolidated_contradiction(
            left_summary,
            right_summary,
            keep,
            reason,
        )

    # ------------------------------------------------------------------ #
    # Stage: Reinforcement                                                 #
    # ------------------------------------------------------------------ #

    def apply_memory_reinforcement_signals(
        self,
        history,
        existing_catalog=None,
        turn_context=None,
    ):
        catalog = existing_catalog if isinstance(existing_catalog, list) else self.bot.memory_catalog()
        if not catalog:
            return 0

        user_messages = [
            str(message.get("content", "")).strip()
            for message in list(history or [])[-8:]
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if not user_messages:
            return 0

        correction_pattern = re.compile(
            r"\b(?:actually|correction|i meant|i now|used to|not anymore|no,? i)\b",
            flags=re.IGNORECASE,
        )
        praise_pattern = re.compile(
            r"\b(?:good memory|thanks for remembering|you remembered|that's right|exactly right)\b",
            flags=re.IGNORECASE,
        )
        reinforced = 0
        today_stamp = self._turn_wall_date(turn_context)

        for message in user_messages:
            tokens = self.bot.significant_tokens(message)
            if not tokens:
                continue
            is_correction = bool(correction_pattern.search(message))
            is_praise = bool(praise_pattern.search(message))
            if not is_correction and not is_praise:
                continue

            for memory in catalog:
                overlap = tokens & self.bot.significant_tokens(
                    memory.get("summary", ""),
                )
                if len(overlap) < 2:
                    continue
                base_confidence = float(memory.get("confidence", 0.5) or 0.5)
                base_impact = float(memory.get("impact_score", 1.0) or 1.0)
                if is_correction:
                    memory["confidence"] = round(min(0.98, base_confidence + 0.08), 3)
                    memory["impact_score"] = round(base_impact + 0.2, 3)
                else:
                    memory["confidence"] = round(min(0.98, base_confidence + 0.04), 3)
                    memory["impact_score"] = round(base_impact + 0.1, 3)
                memory["updated_at"] = today_stamp
                memory.update(self.scorer.score_memory_entry(memory))
                reinforced += 1

        return reinforced

    def apply_consolidated_feedback(self, summary, vote, turn_context=None):
        if not self._assert_save_commit_boundary(turn_context):
            return None
        normalized_target = self.bot.normalize_memory_text(summary)
        if not normalized_target:
            return None

        try:
            vote_value = int(vote)
        except (TypeError, ValueError):
            vote_value = 0
        if vote_value == 0:
            return None

        direction = 1 if vote_value > 0 else -1
        now_stamp = self._turn_wall_time(turn_context)
        updated_entry = None
        consolidated = []
        for entry in self.bot.consolidated_memories():
            candidate = dict(entry)
            if self.bot.normalize_memory_text(
                candidate.get("summary", ""),
            ) == normalized_target and not bool(candidate.get("superseded", False)):
                importance = max(
                    0.0,
                    min(1.0, float(candidate.get("importance_score", 0.0) or 0.0)),
                )
                confidence = max(
                    0.05,
                    min(0.98, float(candidate.get("confidence", 0.5) or 0.5)),
                )
                candidate["importance_score"] = round(
                    max(0.0, min(1.0, importance + 0.08 * direction)),
                    3,
                )
                candidate["confidence"] = round(
                    max(0.05, min(0.98, confidence + 0.04 * direction)),
                    3,
                )
                candidate["last_reinforced_at"] = now_stamp
                candidate["updated_at"] = now_stamp
                updated_entry = candidate
            consolidated.append(candidate)

        if updated_entry is None:
            return None

        self.bot.mutate_memory_store(consolidated_memories=consolidated)
        self.bot.mark_memory_graph_dirty()
        return updated_entry

    def apply_consolidated_feedback_from_recent_turns(self, turn_context=None) -> int:
        """Scan recent conversation turns for inline feedback signals (thumbs-up/down
        stored in MEMORY_STORE["relationship_history"]) and apply them to consolidated
        memories whose summary overlaps with the feedback text.
        Returns the number of memories updated.
        """
        updated = 0
        feedback_entries = list(
            (self.bot.MEMORY_STORE.get("relationship_history") or [])[-20:],
        )
        if not feedback_entries:
            return 0

        for entry in feedback_entries:
            feedback_text = str(
                entry.get("message") or entry.get("content") or "",
            ).strip()
            vote_raw = entry.get("vote") or entry.get("feedback_score")
            if not feedback_text or vote_raw is None:
                continue
            try:
                vote = int(vote_raw)
            except (TypeError, ValueError):
                continue
            if vote == 0:
                continue

            # Find the consolidated memory whose summary best matches the feedback text
            best_match = None
            norm_feedback = self.bot.normalize_memory_text(feedback_text)
            for mem in self.bot.consolidated_memories():
                norm_summary = self.bot.normalize_memory_text(mem.get("summary", ""))
                # Simple token overlap check
                fb_tokens = set(norm_feedback.split())
                sm_tokens = set(norm_summary.split())
                if fb_tokens and sm_tokens and len(fb_tokens & sm_tokens) / max(len(fb_tokens), 1) > 0.35:
                    best_match = mem
                    break

            if best_match:
                result = self.apply_consolidated_feedback(
                    best_match.get("summary", ""),
                    vote,
                    turn_context=turn_context,
                )
                if result:
                    updated += 1

        return updated

    def apply_controlled_forgetting(self, force=False, turn_context=None):
        # Previously gated by _assert_save_commit_boundary. The gate is now
        # structural: this method is only callable from PostCommitWorker
        # after PersistenceService publishes a post-commit-ready event.
        memories = list(self.bot.memory_catalog())
        if not memories:
            return {"removed": 0, "backup_path": "", "ran": False}

        wall_time = self._turn_wall_time(turn_context)
        turn_wall_date = self._turn_wall_date(turn_context)
        try:
            turn_date = date.fromisoformat(str(turn_wall_date)[:10])
        except ValueError as exc:
            raise RuntimeError(
                "TemporalNode wall_date invalid — deterministic execution violated",
            ) from exc
        backup_stamp = wall_time.replace("-", "").replace(":", "").replace("T", "_")[:15]
        decay_threshold = 0.15
        identity_categories = {"identity", "relationship", "relationships"}

        def _confidence_history(entry):
            raw = entry.get("confidence_history") or {}
            if not isinstance(raw, dict):
                raw = {}
            return {
                "high": max(0, int(raw.get("high", 0) or 0)),
                "medium": max(0, int(raw.get("medium", 0) or 0)),
                "low": max(0, int(raw.get("low", 0) or 0)),
            }

        def _days_old_from_turn(entry):
            anchor = str(entry.get("updated_at") or entry.get("created_at") or "").strip()
            if not anchor:
                return 0
            try:
                entry_date = date.fromisoformat(anchor[:10])
            except ValueError:
                return 0
            return max(0, (turn_date - entry_date).days)

        def _decay_score(entry):
            days_old = _days_old_from_turn(entry)
            age_factor = min(1.0, max(0.0, float(days_old or 0.0) / 365.0))

            importance = max(
                0.0,
                min(1.0, float(entry.get("importance_score", 0.0) or 0.0)),
            )

            access_count = max(0, int(entry.get("access_count", 0) or 0))
            access_frequency = min(1.0, float(access_count) / 10.0)

            history = _confidence_history(entry)
            high_hits = max(0, int(entry.get("high_confidence_hits", history["high"]) or history["high"]))
            high_confidence_factor = min(1.0, float(high_hits) / 10.0)

            score = (
                (0.35 * age_factor)
                - (0.30 * importance)
                - (0.20 * access_frequency)
                - (0.15 * high_confidence_factor)
            )
            components = {
                "age_factor": round(age_factor, 4),
                "importance_score": round(importance, 4),
                "access_frequency": round(access_frequency, 4),
                "high_confidence_factor": round(high_confidence_factor, 4),
            }
            return round(score, 4), components

        kept = []
        archived = []
        for memory in memories:
            if bool(memory.get("pinned", False)):
                kept.append(memory)
                continue

            # Keep importance current if not already meaningful.
            importance = max(
                0.0,
                min(1.0, float(memory.get("importance_score", 0.0) or 0.0)),
            )
            if importance <= 0.0:
                memory.update(self.scorer.score_memory_entry(memory))

            decay_score, components = _decay_score(memory)
            memory = dict(memory)
            memory["decay_score"] = decay_score
            memory["decay_components"] = components

            category = str(memory.get("category", "general") or "general").strip().lower()
            if category in identity_categories:
                kept.append(memory)
                continue

            should_archive = bool(decay_score > decay_threshold)
            if force and decay_score > 0.0:
                should_archive = True

            if should_archive:
                archived.append(memory)
            else:
                kept.append(memory)

        if not archived:
            return {"removed": 0, "backup_path": "", "ran": True}

        backup_name = f"memory_forgetting_backup_{backup_stamp}.json"
        backup_path = str(self.bot.script_path.with_name(backup_name))
        self.bot.memory.export_memory_store(backup_path)

        archive_entries = list(self.bot.session_archive())
        for memory in archived:
            archive_entry = self.bot.normalize_session_archive_entry(
                {
                    "summary": str(memory.get("summary") or "").strip(),
                    "topics": [str(memory.get("category") or "general").strip().lower() or "general"],
                    "dominant_mood": str(memory.get("mood") or "neutral"),
                    "turn_count": 1,
                    "created_at": wall_time,
                },
            )
            if archive_entry is not None:
                archive_entries.append(archive_entry)

        if archive_entries:
            self.bot.mutate_memory_store(session_archive=archive_entries[-24:])

        self.bot.save_memory_catalog(kept)
        return {
            "removed": len(archived),
            "archived": len(archived),
            "retained": len(kept),
            "threshold": decay_threshold,
            "backup_path": backup_path,
            "ran": True,
        }

    def select_active_consolidated_memories(
        self,
        user_input: str,
        max_items: int = 3,
    ) -> list:
        try:
            effective_max = max(1, int(max_items or 3))
        except (TypeError, ValueError):
            effective_max = 3

        consolidated = [entry for entry in self.bot.consolidated_memories() if not bool(entry.get("superseded", False))]
        if not consolidated:
            return []

        recent_consolidated = consolidated[-20:]
        if len(recent_consolidated) <= effective_max:
            return recent_consolidated[:effective_max]

        normalized_input = str(user_input or "").strip()
        if not normalized_input:
            return recent_consolidated[-effective_max:]

        cache_payload = "\n".join(
            f"{entry.get('category', 'general')}|{entry.get('confidence', 0.5)}|{entry.get('updated_at', '')}|{entry.get('summary', '')}"
            for entry in recent_consolidated
        )
        cache_key = (
            self.bot.normalize_memory_text(normalized_input),
            effective_max,
            hashlib.sha1(cache_payload.encode("utf-8")).hexdigest(),
        )
        cached_indices = self._active_consolidated_selection_cache.get(cache_key)
        if cached_indices is not None:
            return [recent_consolidated[index] for index in cached_indices if 0 <= index < len(recent_consolidated)][
                :effective_max
            ]

        memory_list = []
        for index, entry in enumerate(recent_consolidated, start=1):
            category = entry.get("category", "general")
            confidence = self.bot.confidence_label(entry.get("confidence", 0.5))
            memory_list.append(
                f"{index}. [{category} | conf={confidence}] {entry.get('summary', '')}",
            )

        selector_prompt = f"""
You are helping Dad decide which long-term insights about Tony are most relevant for this new message.

Tony's current message: \"{normalized_input}\"

Recent consolidated memories (numbered):
{chr(10).join(memory_list)}

Return ONLY a JSON array of the 1-{effective_max} most relevant memory numbers (e.g. [3, 7, 12]).
Choose the ones that best help understand Tony's current situation, emotions, or patterns.
If none are strongly relevant, return an empty array [].
""".strip()

        try:
            response = self.bot.call_ollama_chat(
                messages=[{"role": "user", "content": selector_prompt}],
                options={"temperature": 0.0},
                response_format="json",
                purpose="active memory selection",
            )
            content = self.bot.extract_ollama_message_content(response)
            selected_indices = self.bot.parse_model_json_content(content)
            if not isinstance(selected_indices, list):
                selected_indices = []

            active = []
            selected_positions = []
            seen = set()
            for raw_index in selected_indices:
                try:
                    selected_index = int(raw_index) - 1
                except (TypeError, ValueError):
                    continue
                if not 0 <= selected_index < len(recent_consolidated):
                    continue

                entry = recent_consolidated[selected_index]
                entry_key = self.bot.normalize_memory_text(entry.get("summary", "")) or f"index:{selected_index}"
                if entry_key in seen:
                    continue
                seen.add(entry_key)
                active.append(entry)
                selected_positions.append(selected_index)
                if len(active) >= effective_max:
                    break
            if len(self._active_consolidated_selection_cache) >= 128:
                self._active_consolidated_selection_cache.pop(
                    next(iter(self._active_consolidated_selection_cache)),
                )
            self._active_consolidated_selection_cache[cache_key] = tuple(
                selected_positions,
            )
            return active
        except Exception as exc:
            self.bot.record_runtime_issue(
                "active memory selection",
                "falling back to top recent consolidated",
                exc,
            )
            return recent_consolidated[-effective_max:]

    def build_active_consolidated_context(self, user_input: str) -> str | None:
        active_memories = self.select_active_consolidated_memories(
            user_input,
            max_items=3,
        )
        if not active_memories:
            return None

        lines = []
        for entry in active_memories:
            category = entry.get("category", "general")
            confidence = self.bot.confidence_label(entry.get("confidence", 0.5))
            importance = float(entry.get("importance_score", 0.0) or 0.0)
            summary = entry.get("summary", "")
            tension = (
                f" Tension: {'; '.join(entry.get('contradictions', [])[:2])}." if entry.get("contradictions") else ""
            )
            lines.append(
                f"- [{category} | conf={confidence} | importance={importance:.2f}]{tension} {summary}",
            )

        return (
            "Most relevant long-term insights about Tony right now:\n"
            + "\n".join(lines)
            + "\nUse these only as supporting context - keep replies natural."
        )

    def build_consolidated_memory_context(self) -> str | None:
        """Format all consolidated memories as a static context block (non-query-aware)."""
        consolidated = self.bot.memory_manager.consolidated_memories()
        if not consolidated:
            return None

        lines = "\n".join(
            (
                f"- [{entry.get('category', 'general')} | confidence={self.bot.confidence_label(entry.get('confidence', 0.5))}] "
                f"{entry.get('summary', '')}"
                + (
                    f" Tension to watch: {'; '.join(entry.get('contradictions', [])[:2])}."
                    if entry.get("contradictions")
                    else ""
                )
            )
            for entry in consolidated[-5:]
        )
        return "Consolidated long-term insights about Tony from prior chats:\n" + lines

    def should_run_memory_consolidation(self, force=False, turn_context=None):
        if force:
            return True
        if str(
            self.bot.MEMORY_STORE.get("last_consolidated_at") or "",
        ).strip() == self._turn_wall_date(turn_context):
            return False
        return len(self.bot.memory_catalog()) >= 3

    def build_memory_consolidation_prompt(self, memories, existing_insights):
        memory_lines = [
            f"- [{memory.get('category', 'general')}, mood={self.bot.normalize_mood(memory.get('mood'))}] {memory.get('summary', '')}"
            for memory in memories[-18:]
            if memory.get("summary")
        ]
        prior_lines = [
            (
                f"- [{entry.get('category', 'general')}, confidence={entry.get('confidence', 0.5):.2f}] "
                f"{entry.get('summary', '')}"
            )
            for entry in existing_insights[-5:]
            if entry.get("summary")
        ] or ["- None yet."]
        contradiction_lines = [
            f"- {item['left']} <-> {item['right']} ({item['reason']})"
            for item in self.detect_memory_contradictions(memories, existing_insights)[:6]
        ] or ["- None detected."]

        return f"""
You are consolidating long-term memory about Tony from repeated prior conversations.

Existing consolidated insights:
{chr(10).join(prior_lines)}

Possible contradictions or tension points:
{chr(10).join(contradiction_lines)}

Recent durable memories:
{chr(10).join(memory_lines)}

Return only JSON as an array of up to 6 objects with keys:
- summary
- category
- source_count
- confidence
- supporting_summaries
- contradictions

Rules:
- Write higher-level durable insights, not one-off diary lines.
- Merge overlapping memories into one clearer long-term fact when possible.
- If sources conflict, lower confidence and mention the contradiction briefly instead of pretending the conflict does not exist.
- Confidence should be a number from 0.05 to 0.98, based on source agreement, repetition, and recency.
- Preserve uncertainty by staying specific and modest; do not invent anything.
- Prefer stable patterns, ongoing concerns, recurring goals, or notable positive progress.
- Keep each summary under 22 words.
""".strip()

    def _with_consolidated_defaults(self, entry):
        normalized = self.bot.normalize_consolidated_memory_entry(entry)
        if normalized is None:
            return None
        try:
            version = max(
                1,
                int(entry.get("version", normalized.get("version", 1)) or 1),
            )
        except (TypeError, ValueError, AttributeError):
            version = max(1, int(normalized.get("version", 1) or 1))
        try:
            importance_score = float(
                entry.get("importance_score", normalized.get("importance_score", 0.0)) or 0.0,
            )
        except (TypeError, ValueError, AttributeError):
            importance_score = float(normalized.get("importance_score", 0.0) or 0.0)
        return {
            **normalized,
            "importance_score": max(0.0, min(1.0, importance_score)),
            "version": version,
            "superseded": bool(
                entry.get("superseded", normalized.get("superseded", False)),
            ),
            "superseded_by": str(
                entry.get("superseded_by") or normalized.get("superseded_by") or "",
            ).strip(),
            "superseded_reason": str(
                entry.get("superseded_reason") or normalized.get("superseded_reason") or "",
            ).strip(),
            "superseded_at": entry.get("superseded_at") or normalized.get("superseded_at"),
            "last_reinforced_at": entry.get("last_reinforced_at") or normalized.get("last_reinforced_at"),
        }

    @staticmethod
    def consolidated_entry_rank(entry):
        return (
            float(entry.get("confidence", 0.0)),
            int(entry.get("source_count", 1)),
            entry.get("updated_at", ""),
            entry.get("summary", ""),
        )

    def _invalidate_superseded_graph_edges(
        self,
        loser: dict,
        turn_time: str,
        turn_context=None,
    ) -> int:
        if not self._assert_save_commit_boundary(turn_context):
            return 0
        memory_manager = getattr(self.bot, "memory_manager", None)
        graph_manager = getattr(memory_manager, "graph_manager", None)
        if graph_manager is None:
            return 0

        loser_label = str(loser.get("summary") or "").strip().lower()
        if not loser_label:
            return 0

        try:
            snapshot = graph_manager.graph_snapshot()
            nodes = list(snapshot.get("nodes", []))
            edges = list(snapshot.get("edges", []))
            source_keys = {
                str(node.get("node_key") or "")
                for node in nodes
                if str(node.get("node_type") or "") == "consolidated_memory"
                and str(node.get("label") or "").strip().lower() == loser_label
            }
            if not source_keys:
                return 0

            invalidated = 0
            for edge in edges:
                if str(edge.get("source_key") or "") in source_keys:
                    graph_manager.invalidate_edge(edge, turn_time)
                    invalidated += 1

            if invalidated:
                graph_manager.ensure_graph_store()
                graph_manager._graph_store_backend.replace_graph(nodes, edges)
            return invalidated
        except Exception as exc:
            logger.warning("Failed to invalidate superseded graph edges: %s", exc)
            return 0

    def merge_consolidated_memories(self, entries, turn_context=None):
        if not self._assert_save_commit_boundary(turn_context):
            return self.bot.consolidated_memories()
        merged = {}
        for entry in [*self.bot.consolidated_memories(), *entries]:
            normalized = self._with_consolidated_defaults(entry)
            if normalized is None:
                continue
            normalized["importance_score"] = self.scorer.consolidated_importance_score(
                normalized,
            )
            key = self.bot.normalize_memory_text(normalized.get("summary", ""))
            existing = merged.get(key)
            if existing is None:
                merged[key] = normalized
                continue

            combined_support = []
            for summary in [
                *existing.get("supporting_summaries", []),
                *normalized.get("supporting_summaries", []),
            ]:
                if summary not in combined_support:
                    combined_support.append(summary)

            combined_contradictions = []
            for summary in [
                *existing.get("contradictions", []),
                *normalized.get("contradictions", []),
            ]:
                if summary not in combined_contradictions:
                    combined_contradictions.append(summary)

            preferred = (
                normalized
                if self.consolidated_entry_rank(normalized) >= self.consolidated_entry_rank(existing)
                else existing
            )
            merged[key] = {
                **preferred,
                "source_count": max(
                    existing.get("source_count", 1),
                    normalized.get("source_count", 1),
                ),
                "confidence": max(
                    existing.get("confidence", 0.0),
                    normalized.get("confidence", 0.0),
                ),
                "importance_score": max(
                    existing.get("importance_score", 0.0),
                    normalized.get("importance_score", 0.0),
                ),
                "version": max(
                    existing.get("version", 1),
                    normalized.get("version", 1),
                ),
                "supporting_summaries": combined_support[:4],
                "contradictions": combined_contradictions[:4],
            }

        merged_values = sorted(
            merged.values(),
            key=lambda item: (item.get("updated_at", ""), item.get("summary", "")),
        )
        now_stamp = self._turn_wall_time(turn_context)
        active_entries = [entry for entry in merged_values if not bool(entry.get("superseded", False))]

        for index, left in enumerate(active_entries):
            if left.get("superseded"):
                continue
            for right in active_entries[index + 1 :]:
                if right.get("superseded"):
                    continue
                if str(left.get("category") or "general") != str(
                    right.get("category") or "general",
                ):
                    continue
                reason = self.contradiction_signal_reason(
                    left.get("summary", ""),
                    right.get("summary", ""),
                )
                if not reason:
                    continue

                left_rank = self.consolidated_resolution_rank(left)
                right_rank = self.consolidated_resolution_rank(right)
                left_recency = self.scorer._recency_score(left.get("updated_at"))
                right_recency = self.scorer._recency_score(right.get("updated_at"))
                left_conf = max(
                    0.05,
                    min(1.0, float(left.get("confidence", 0.5) or 0.5)),
                )
                right_conf = max(
                    0.05,
                    min(1.0, float(right.get("confidence", 0.5) or 0.5)),
                )

                # Prefer the latest user statement when confidence is not weak.
                if abs(left_recency - right_recency) >= 0.2:
                    if left_recency > right_recency and left_conf >= 0.35:
                        winner, loser = left, right
                    elif right_recency > left_recency and right_conf >= 0.35:
                        winner, loser = right, left
                    else:
                        winner, loser = (left, right) if left_rank >= right_rank else (right, left)
                else:
                    winner, loser = (left, right) if left_rank >= right_rank else (right, left)
                winner["version"] = max(
                    int(winner.get("version", 1) or 1),
                    int(loser.get("version", 1) or 1) + 1,
                )
                loser.update(
                    {
                        "superseded": True,
                        "superseded_by": ConflictDetector._entry_identity(winner),
                        "superseded_reason": reason,
                        "superseded_at": now_stamp,
                    },
                )
                self._invalidate_superseded_graph_edges(
                    loser,
                    now_stamp,
                    turn_context=turn_context,
                )

        return merged_values[-16:]

    def consolidate_memories(self, force=False, turn_context=None):
        # Previously gated by _assert_save_commit_boundary. The gate is now
        # structural: this method is only callable from PostCommitWorker
        # after PersistenceService publishes a post-commit-ready event.
        if not self.should_run_memory_consolidation(
            force=force,
            turn_context=turn_context,
        ):
            return self.bot.consolidated_memories()

        memories = self.bot.memory_catalog()
        if not memories:
            return self.bot.consolidated_memories()

        try:
            response = self.bot.call_ollama_chat(
                messages=[
                    {
                        "role": "user",
                        "content": self.build_memory_consolidation_prompt(
                            memories,
                            self.bot.consolidated_memories(),
                        ),
                    },
                ],
                options={"temperature": 0.1},
                response_format="json",
                purpose="memory consolidation",
            )
            content = self.bot.extract_ollama_message_content(response)
        except (RuntimeError, KeyError, TypeError) as exc:
            self.bot.record_runtime_issue(
                "memory consolidation",
                "keeping the existing consolidated memory view",
                exc,
            )
            return self.bot.consolidated_memories()

        try:
            parsed = self.bot.parse_model_json_content(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Memory consolidation returned invalid JSON: %s", exc)
            return self.bot.consolidated_memories()

        if not isinstance(parsed, list):
            logger.warning("Memory consolidation returned non-list payload: %r", parsed)
            return self.bot.consolidated_memories()

        merged = self.merge_consolidated_memories(parsed, turn_context=turn_context)
        self.bot.mutate_memory_store(
            consolidated_memories=merged,
            last_consolidated_at=self._turn_wall_date(turn_context),
        )
        self.bot.mark_memory_graph_dirty()
        return merged

    def session_topics_from_history(self, history):
        topic_counts = {}
        for message in history:
            if message.get("role") != "user":
                continue
            topic = self.bot.infer_memory_category(str(message.get("content", "")))
            if topic == "general":
                topic_matches = self.bot.matching_topics(
                    str(message.get("content", "")),
                )
                if topic_matches:
                    topic = topic_matches[0]["name"].replace("_", " ")
            if topic and topic != "general":
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        ranked = sorted(topic_counts.items(), key=lambda item: (-item[1], item[0]))
        return [topic for topic, _ in ranked[:3]]

    @staticmethod
    def build_session_archive_fallback(history):
        user_messages = [
            str(message.get("content", "")).strip() for message in history if message.get("role") == "user"
        ]
        snippets = [snippet for snippet in user_messages[-3:] if snippet]
        if not snippets:
            return ""
        return " | ".join(snippets)

    def archive_session_context(self, history, turn_context=None):
        if not self._assert_save_commit_boundary(turn_context):
            return None
        summary = (self.bot.session_summary or "").strip() or self.build_session_archive_fallback(history)
        if not summary:
            return None

        created_at = self._turn_wall_time(turn_context)
        entry = self.bot.normalize_session_archive_entry(
            {
                "created_at": created_at,
                "summary": summary,
                "topics": self.session_topics_from_history(history),
                "dominant_mood": Counter(self.bot.session_moods).most_common(1)[0][0]
                if self.bot.session_moods
                else self.bot.last_saved_mood(),
                "turn_count": len(
                    [message for message in history if message.get("role") == "user"],
                ),
            },
        )
        if entry is None:
            return None

        archive = self.bot.session_archive()
        if archive:
            latest = archive[-1]
            if latest.get("summary") == entry["summary"] and latest.get("turn_count") == entry["turn_count"]:
                return latest
            if latest.get("turn_count") == entry["turn_count"]:
                archive[-1] = entry
                self.bot.mutate_memory_store(session_archive=archive[-24:])
                self.bot.mark_memory_graph_dirty()
                return entry

        archive.append(entry)
        self.bot.mutate_memory_store(session_archive=archive[-24:])
        self.bot.mark_memory_graph_dirty()
        return entry

    @staticmethod
    def build_relationship_timeline_prompt(previous_timeline, archive_entries):
        if not archive_entries:
            return None

        prior_timeline = previous_timeline or "None yet."
        entry_lines = []
        for entry in archive_entries[-6:]:
            topics = ", ".join(entry.get("topics", [])) or "general"
            entry_lines.append(
                f"- {entry.get('created_at', '')[:10]} | mood={entry.get('dominant_mood', 'neutral')} | topics={topics} | summary={entry.get('summary', '')}",
            )

        return f"""
You are maintaining a long-term relationship digest for ongoing chats between Tony and his dad.

Previous digest:
{prior_timeline}

Recent session summaries:
{chr(10).join(entry_lines)}

Write an updated digest in 5 short bullet lines or fewer.
Focus on recurring struggles or wins, relationship movement, notable follow-ups, and what should carry across future chats.
Do not invent anything.
""".strip()

    def refresh_relationship_timeline(self, force=False, turn_context=None):
        if not self._assert_save_commit_boundary(turn_context):
            return self.bot.relationship_timeline()
        self._turn_wall_time(turn_context)
        archive = self.bot.session_archive()
        if not archive:
            return ""
        if not force and len(archive) < 2:
            return self.bot.relationship_timeline()

        prompt = self.build_relationship_timeline_prompt(
            self.bot.relationship_timeline(),
            archive,
        )
        if prompt is None:
            return self.bot.relationship_timeline()

        try:
            response = self.bot.call_ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
                purpose="relationship timeline",
            )
            timeline = self.bot.extract_ollama_message_content(response).strip()
        except (RuntimeError, KeyError, TypeError) as exc:
            self.bot.record_runtime_issue(
                "relationship timeline",
                "keeping the previous relationship digest",
                exc,
            )
            return self.bot.relationship_timeline()

        if timeline:
            self.bot.mutate_memory_store(relationship_timeline=timeline)

        return self.bot.relationship_timeline()
