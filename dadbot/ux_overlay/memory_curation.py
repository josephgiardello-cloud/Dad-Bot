from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from dadbot.ux_overlay.models import CuratedMemory


@dataclass
class MemoryIngestionItem:
    text: str
    created_at: datetime
    emotional_intensity: float


class MemoryCurator:
    """Converts raw interaction snippets into narrative-memory units."""

    def __init__(
        self,
        minimum_length: int = 20,
        meaningful_terms: tuple[str, ...] = ("feel", "remember", "important", "family"),
    ) -> None:
        self.minimum_length = minimum_length
        self.meaningful_terms = tuple(t.lower() for t in meaningful_terms)

    def ingestion_filter(
        self,
        raw_events: list[dict[str, Any]],
    ) -> list[MemoryIngestionItem]:
        items: list[MemoryIngestionItem] = []
        for event in raw_events:
            text = str(event.get("text") or "").strip()
            if len(text) < self.minimum_length:
                continue
            ltext = text.lower()
            if not any(term in ltext for term in self.meaningful_terms):
                continue
            created = event.get("created_at")
            if not isinstance(created, datetime):
                created = datetime.now(UTC)
            intensity = float(event.get("emotional_intensity") or 0.0)
            items.append(
                MemoryIngestionItem(
                    text=text,
                    created_at=created,
                    emotional_intensity=max(0.0, min(1.0, intensity)),
                ),
            )
        return items

    def importance_score(
        self,
        item: MemoryIngestionItem,
        *,
        recency_hours: float,
        repetition_count: int,
    ) -> float:
        emotional = item.emotional_intensity * 0.5
        repetition = min(1.0, repetition_count / 5.0) * 0.3
        recency = max(0.0, 1.0 - min(1.0, recency_hours / (24.0 * 14.0))) * 0.2
        return max(0.0, min(1.0, emotional + repetition + recency))

    def compress(self, items: list[MemoryIngestionItem]) -> list[CuratedMemory]:
        if not items:
            return []

        buckets: dict[str, list[MemoryIngestionItem]] = defaultdict(list)
        for item in items:
            topic_key = self._topic_key(item.text)
            buckets[topic_key].append(item)

        curated: list[CuratedMemory] = []
        now = datetime.now(UTC)
        for key, bucket in buckets.items():
            latest = max(bucket, key=lambda x: x.created_at)
            avg_emotion = sum(x.emotional_intensity for x in bucket) / len(bucket)
            age_hours = max(0.0, (now - latest.created_at).total_seconds() / 3600.0)
            score = self.importance_score(
                latest,
                recency_hours=age_hours,
                repetition_count=len(bucket),
            )
            curated.append(
                CuratedMemory(
                    summary=f"{key}: {self._summary_from_bucket(bucket)}",
                    emotional_weight=max(
                        0.0,
                        min(1.0, (score * 0.7) + (avg_emotion * 0.3)),
                    ),
                    last_reinforced=latest.created_at,
                ),
            )

        curated.sort(
            key=lambda x: (x.emotional_weight, x.last_reinforced),
            reverse=True,
        )
        return curated

    @staticmethod
    def _topic_key(text: str) -> str:
        tokens = [t.strip(".,!?;:").lower() for t in text.split()]
        tokens = [t for t in tokens if len(t) > 3]
        if not tokens:
            return "general"
        return " ".join(tokens[:3])

    @staticmethod
    def _summary_from_bucket(items: list[MemoryIngestionItem]) -> str:
        lead = items[0].text.strip()
        if len(lead) > 96:
            lead = lead[:93] + "..."
        if len(items) == 1:
            return lead
        return f"{lead} (+{len(items) - 1} related)"
