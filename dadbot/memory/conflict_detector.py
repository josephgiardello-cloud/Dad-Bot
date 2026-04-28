from __future__ import annotations

import hashlib
import re
from datetime import date, datetime


class ConflictDetector:
    """Contradiction detection and resolution for consolidated memories.

    No LLM calls, no prompts.  Uses text-overlap heuristics and scoring weights
    (delegated to ``MemoryScorer``) to find and resolve conflicts.
    """

    def __init__(self, bot, scorer) -> None:
        self.bot = bot
        self.scorer = scorer

    # --- Detection ---

    def contradiction_signal_reason(self, left_summary: str, right_summary: str) -> str | None:
        left = self.bot.normalize_memory_text(left_summary)
        right = self.bot.normalize_memory_text(right_summary)
        if not left or not right or left == right:
            return None

        stripped_left = re.sub(
            r"\b(?:not|never|no|doesn't|dont|don't|isn't|wasn't|can't|cannot|won't|stopped|quit)\b", "", left
        )
        stripped_right = re.sub(
            r"\b(?:not|never|no|doesn't|dont|don't|isn't|wasn't|can't|cannot|won't|stopped|quit)\b", "", right
        )
        shared_tokens = self.bot.significant_tokens(stripped_left) & self.bot.significant_tokens(stripped_right)
        negation_words = {
            "not", "never", "no", "doesn't", "don't", "isn't", "wasn't",
            "can't", "cannot", "won't", "stopped", "quit",
        }
        left_has_negation = bool(negation_words & self.bot.tokenize(left))
        right_has_negation = bool(negation_words & self.bot.tokenize(right))
        if len(shared_tokens) >= 2 and left_has_negation != right_has_negation:
            return "same topic appears with opposite polarity"

        opposite_pairs = [
            ("single", "married"),
            ("saving", "debt"),
            ("employed", "unemployed"),
            ("working", "jobless"),
            ("together", "separated"),
        ]
        for left_word, right_word in opposite_pairs:
            if left_word in left and right_word in right or left_word in right and right_word in left:
                return f"contains opposing states ({left_word} vs {right_word})"

        return None

    def detect_memory_contradictions(self, memories: list, existing_insights: list | None = None) -> list:
        candidate_entries: list[dict] = []
        for item in memories[-18:]:
            summary = self.bot.naturalize_memory_summary(item.get("summary", ""))
            if summary:
                candidate_entries.append(
                    {
                        "summary": summary,
                        "category": str(item.get("category") or self.bot.infer_memory_category(summary)).strip().lower() or "general",
                        "confidence": float(item.get("confidence", 0.5) or 0.5),
                        "updated_at": str(item.get("updated_at") or item.get("created_at") or date.today().isoformat()),
                    }
                )
        for item in existing_insights or []:
            summary = self.bot.naturalize_memory_summary(item.get("summary", ""))
            if summary:
                candidate_entries.append(
                    {
                        "summary": summary,
                        "category": str(item.get("category") or self.bot.infer_memory_category(summary)).strip().lower() or "general",
                        "confidence": float(item.get("confidence", 0.5) or 0.5),
                        "updated_at": str(item.get("updated_at") or date.today().isoformat()),
                    }
                )

        contradiction_candidates = []
        for index, left in enumerate(candidate_entries):
            for right in candidate_entries[index + 1:]:
                if left["category"] != right["category"]:
                    continue
                reason = self.contradiction_signal_reason(left["summary"], right["summary"])
                if not reason:
                    continue
                contradiction_candidates.append(
                    {
                        "left": left["summary"],
                        "right": right["summary"],
                        "reason": reason,
                        "weight": self.scorer.contradiction_weight(left, right),
                    }
                )

        unique: list[dict] = []
        seen: set = set()
        for item in contradiction_candidates:
            key = tuple(sorted((item["left"], item["right"])))
            if key in seen:
                continue
            if float(item.get("weight", 0.0) or 0.0) < 0.15:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def consolidated_contradictions(self, limit: int = 8) -> list:
        entries = [entry for entry in self.bot.consolidated_memories() if not bool(entry.get("superseded", False))]
        if len(entries) < 2:
            return []

        findings: list[dict] = []
        for index, left in enumerate(entries):
            for right in entries[index + 1:]:
                if str(left.get("category") or "general") != str(right.get("category") or "general"):
                    continue
                reason = self.contradiction_signal_reason(left.get("summary", ""), right.get("summary", ""))
                if not reason:
                    continue
                findings.append(
                    {
                        "left_summary": str(left.get("summary") or "").strip(),
                        "right_summary": str(right.get("summary") or "").strip(),
                        "category": str(left.get("category") or "general").strip().lower() or "general",
                        "reason": reason,
                        "left_confidence": float(left.get("confidence", 0.5) or 0.5),
                        "right_confidence": float(right.get("confidence", 0.5) or 0.5),
                        "left_importance": float(left.get("importance_score", 0.0) or 0.0),
                        "right_importance": float(right.get("importance_score", 0.0) or 0.0),
                        "left_updated_at": str(left.get("updated_at") or ""),
                        "right_updated_at": str(right.get("updated_at") or ""),
                    }
                )

        findings.sort(
            key=lambda item: (
                max(float(item.get("left_importance", 0.0) or 0.0), float(item.get("right_importance", 0.0) or 0.0)),
                max(float(item.get("left_confidence", 0.0) or 0.0), float(item.get("right_confidence", 0.0) or 0.0)),
                str(item.get("left_updated_at") or ""),
                str(item.get("right_updated_at") or ""),
            ),
            reverse=True,
        )
        return findings[: max(1, int(limit or 8))]

    # --- Resolution ---

    def resolve_consolidated_contradiction(
        self,
        left_summary: str,
        right_summary: str,
        keep: str = "auto",
        reason: str = "user_review",
    ) -> dict | None:
        left_norm = self.bot.normalize_memory_text(left_summary)
        right_norm = self.bot.normalize_memory_text(right_summary)
        if not left_norm or not right_norm or left_norm == right_norm:
            return None

        now_stamp = datetime.now().isoformat(timespec="seconds")
        entries = [dict(item) for item in self.bot.consolidated_memories()]
        left_entry = right_entry = None
        for entry in entries:
            if bool(entry.get("superseded", False)):
                continue
            normalized = self.bot.normalize_memory_text(entry.get("summary", ""))
            if normalized == left_norm:
                left_entry = entry
            elif normalized == right_norm:
                right_entry = entry

        if left_entry is None or right_entry is None:
            return None

        keep_value = str(keep or "auto").strip().lower()
        if keep_value not in {"auto", "left", "right"}:
            keep_value = "auto"

        if keep_value == "left":
            winner, loser = left_entry, right_entry
        elif keep_value == "right":
            winner, loser = right_entry, left_entry
        else:
            left_rank = self.scorer.consolidated_resolution_rank(left_entry)
            right_rank = self.scorer.consolidated_resolution_rank(right_entry)
            winner, loser = (left_entry, right_entry) if left_rank >= right_rank else (right_entry, left_entry)

        winner["version"] = max(int(winner.get("version", 1) or 1), int(loser.get("version", 1) or 1) + 1)
        winner["updated_at"] = now_stamp
        winner["last_reinforced_at"] = now_stamp
        winner["confidence"] = round(min(0.98, float(winner.get("confidence", 0.5) or 0.5) + 0.04), 3)
        winner["importance_score"] = round(min(1.0, float(winner.get("importance_score", 0.0) or 0.0) + 0.06), 3)

        loser.update(
            {
                "superseded": True,
                "superseded_by": self._entry_identity(winner),
                "superseded_reason": str(reason or "user_review").strip() or "user_review",
                "superseded_at": now_stamp,
            }
        )

        self.bot.mutate_memory_store(consolidated_memories=entries)
        self.bot.mark_memory_graph_dirty()
        return {"winner": winner, "loser": loser, "resolved_at": now_stamp}

    @staticmethod
    def _entry_identity(entry: dict) -> str:
        summary = str(entry.get("summary") or "").strip()
        category = str(entry.get("category") or "general").strip().lower() or "general"
        return hashlib.sha1(f"{category}|{summary}".encode("utf-8")).hexdigest()[:12]


__all__ = ["ConflictDetector"]
