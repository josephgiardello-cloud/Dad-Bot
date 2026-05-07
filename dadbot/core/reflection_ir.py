"""
Drift Reflection IR: Longitudinal behavioral modeling and pattern detection.

This module provides high-quality behavioral interpretation without simplistic heuristics.
It accumulates evidence across turns, detects recurring patterns, and surfaces actionable
insights about drift triggers, recovery effectiveness, and burnout signals.

Key Objects:
- DriftEpisode: A contiguous divergence period with context
- BehavioralPattern: Recurring drift structures (late-night, rabbit-holes, avoidance, etc.)
- ReflectionSummary: High-level risk assessment and recommendation

Principles:
- Centralized evidence accumulation (not scattered if/else heuristics)
- Temporal and topic clustering with statistical confidence
- Focus on recovery effectiveness (what helps, not just what fails)
- Support executive coherence, not enforce obedience
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Iterator
from datetime import datetime, timedelta
from collections import defaultdict
import json
import math
import re


@dataclass
class DriftEpisode:
    """
    Represents a contiguous divergence period.
    
    Fields:
        start_turn: Turn index when drift began
        end_turn: Turn index when drift ended (None if ongoing)
        duration: Number of consecutive divergent turns
        trigger_topic: Primary topic/goal deviation
        trigger_time_bucket: Hour-of-day or day-of-week when triggered
        emotional_signature: Observed mood/state (frustrated, tired, bored, distracted)
        recovery_method: What intervention helped (realignment phrase, break, context shift)
        halt_count: Number of mandatory halts during this episode
        resumed_successfully: Whether user recovered to goal alignment
        recovery_time_minutes: Minutes from halt to successful realignment
        burnout_probability: Estimated likelihood this is fatigue-driven (0.0-1.0)
    """
    start_turn: int
    end_turn: Optional[int]
    duration: int
    trigger_topic: Optional[str]
    trigger_time_bucket: Optional[str]  # "11_PM" or "Wednesday" format
    emotional_signature: Optional[str]  # e.g., "fatigue", "frustration", "avoidance"
    recovery_method: Optional[str]  # e.g., "break", "realignment_phrase", "context_shift"
    halt_count: int
    resumed_successfully: bool
    recovery_time_minutes: Optional[int] = None
    burnout_probability: float = 0.0


@dataclass
class BehavioralPattern:
    """
    Represents a recurring drift structure.
    
    Examples:
    - "late-night burnout" (drifts after 11 PM, recovery slow)
    - "tool rabbit-hole" (drifts into same distraction topic repeatedly)
    - "doomscroll diversion" (sequential topic-switching avoidance)
    - "post-difficult-task avoidance" (drifts after completing hard objectives)
    
    Fields:
        pattern_id: Unique identifier (e.g., "late_night_fatigue")
        pattern_name: Human-readable name
        frequency: Number of observed episodes
        confidence: Statistical confidence (0.0-1.0)
        correlated_times: List of time buckets where pattern occurs (["11_PM", "12_AM"])
        correlated_topics: List of distraction domains (["fishing", "gaming"])
        average_episode_duration: Mean turns per drift episode
        average_recovery_time: Mean minutes to recovery
        success_after_intervention: Percentage of episodes recovered successfully
        last_observed_turn: Most recent turn this pattern occurred
        evidence_weight: Combined statistical strength for pattern confidence
    """
    pattern_id: str
    pattern_name: str
    frequency: int = 0
    confidence: float = 0.0
    correlated_times: List[str] = field(default_factory=list)
    correlated_topics: List[str] = field(default_factory=list)
    average_episode_duration: float = 0.0
    average_recovery_time: Optional[float] = None
    success_after_intervention: float = 0.0  # Percentage
    last_observed_turn: int = 0
    evidence_weight: float = 0.0


@dataclass
class ReflectionSummary:
    """
    High-level behavioral synthesis and risk assessment.
    
    Fields:
        current_risk_level: "low", "moderate", "high"
        predicted_drift_probability: Likelihood of drift in next N turns (0.0-1.0)
        likely_trigger_category: Most probable drift cause ("fatigue", "avoidance", "distraction", "unknown")
        primary_pattern: The dominant recurring pattern (or None)
        secondary_patterns: Other observed patterns
        recommended_intervention: Specific, actionable suggestion
        intervention_justification: Why this intervention is recommended
        confidence_score: Overall confidence in this assessment (0.0-1.0)
        observable_signals: List of current behavioral signals to monitor
        recent_episode_count: Number of drift episodes in last N turns
    """
    current_risk_level: str
    predicted_drift_probability: float
    likely_trigger_category: str
    primary_pattern: Optional[BehavioralPattern]
    secondary_patterns: List[BehavioralPattern] = field(default_factory=list)
    recommended_intervention: str = ""
    intervention_justification: str = ""
    confidence_score: float = 0.0
    observable_signals: List[str] = field(default_factory=list)
    recent_episode_count: int = 0


@dataclass
class EvidenceEdge:
    """Weighted edge between evidence graph nodes."""

    source: str
    target: str
    weight: float = 0.0
    observations: int = 0


@dataclass
class EvidenceGraph:
    """Event -> Episode -> Pattern -> Outcome evidence scaffold."""

    edges: Dict[Tuple[str, str], EvidenceEdge] = field(default_factory=dict)

    def add_observation(self, source: str, target: str, weight: float = 1.0) -> None:
        key = (source, target)
        edge = self.edges.get(key)
        if edge is None:
            self.edges[key] = EvidenceEdge(
                source=source,
                target=target,
                weight=max(0.0, float(weight)),
                observations=1,
            )
            return
        edge.observations += 1
        edge.weight += max(0.0, float(weight))


class DriftReflectionEngine:
    """
    Analyzes relational_ledger.jsonl to detect drift patterns, track recovery effectiveness,
    and build longitudinal behavioral models.
    
    Design Principles:
    - NO simplistic if/else heuristics scattered throughout
    - Centralized evidence accumulation and weighting
    - Temporal and topic clustering with confidence thresholds
    - Focus on recovery effectiveness (what interventions help?)
    - Support executive coherence, not enforce obedience
    """

    def __init__(self, ledger_path: Optional[str] = None):
        """
        Initialize the reflection engine.
        
        Args:
            ledger_path: Path to relational_ledger.jsonl. If None, engine is ready but inactive.
        """
        self.ledger_path = ledger_path
        self.drift_episodes: List[DriftEpisode] = []
        self.behavioral_patterns: Dict[str, BehavioralPattern] = {}
        self.turn_alignment_history: List[Tuple[int, bool, Optional[str], Optional[datetime]]] = []  # (turn, is_aligned, goal, occurred_at)
        self.turn_halt_history: Dict[int, bool] = {}
        self.turn_topic_labels: Dict[int, str] = {}
        self._reflection_entries: List[Dict[str, Any]] = []
        self.evidence_graph = EvidenceGraph()
        self.session_start_time = datetime.now()
        self.min_pattern_frequency = 2  # Confidence requires at least 2 observations
        self.recent_window_turns = 20  # Consider last N turns for "recent" signals
        self.burnout_trust_drop_threshold = 0.20
        self.burnout_window = timedelta(hours=2)

    def analyze_ledger(self) -> ReflectionSummary:
        """
        Analyze the relational ledger to produce a reflection summary.
        
        Returns:
            ReflectionSummary with current risk assessment and recommendations.
        """
        if not self.ledger_path:
            # Return neutral summary if no ledger available
            return ReflectionSummary(
                current_risk_level="low",
                predicted_drift_probability=0.0,
                likely_trigger_category="unknown",
                primary_pattern=None,
                confidence_score=0.0,
                observable_signals=[]
            )

        try:
            self._load_and_parse_ledger()
        except Exception as e:
            # Graceful degradation if ledger read fails
            return ReflectionSummary(
                current_risk_level="unknown",
                predicted_drift_probability=0.0,
                likely_trigger_category="unknown",
                primary_pattern=None,
                confidence_score=0.0,
                observable_signals=[f"ledger_read_error: {str(e)}"]
            )

        # Detect drift episodes and patterns
        self._extract_drift_episodes()
        self._detect_temporal_patterns()
        self._detect_topic_patterns()
        self._compute_recovery_effectiveness()
        self._build_evidence_graph()

        # Build summary from accumulated evidence
        return self._synthesize_reflection_summary()

    def _load_and_parse_ledger(self) -> None:
        """Load and parse relational_ledger.jsonl into turn history."""
        if not self.ledger_path:
            return

        self.turn_alignment_history.clear()
        self.turn_halt_history.clear()
        self.turn_topic_labels.clear()
        self._reflection_entries.clear()

        try:
            for entry in self._stream_ledger_entries():
                    self._reflection_entries.append(dict(entry))

                    # Extract turn-level alignment data
                    turn_num_raw = entry.get("turn_index", len(self.turn_alignment_history))
                    try:
                        turn_num = int(turn_num_raw)
                    except (TypeError, ValueError):
                        turn_num = len(self.turn_alignment_history)

                    session_goal = self._extract_goal_label(entry)
                    occurred_at = self._parse_ledger_timestamp(entry)

                    # Determine alignment from relational_state or goal_alignment_diversion_streak
                    is_aligned = entry.get("relational_state", {}).get("is_aligned", True)
                    goal_drift_streak = entry.get("goal_alignment_diversion_streak", 0)
                    decision = str(entry.get("decision") or "").strip().lower()

                    # Override with streak if available
                    if goal_drift_streak > 0:
                        is_aligned = False
                    if decision == "diverted_from_intent":
                        is_aligned = False
                    if decision == "followed_intent":
                        is_aligned = True

                    self.turn_halt_history[turn_num] = bool(entry.get("goal_alignment_mandatory_halt", False))
                    self.turn_topic_labels[turn_num] = self._extract_topic_label(entry, fallback=session_goal)

                    self.turn_alignment_history.append((turn_num, is_aligned, session_goal, occurred_at))
        except FileNotFoundError:
            # Ledger doesn't exist yet; engine remains initialized but empty
            pass

    def _stream_ledger_entries(self, *, max_entries: int = 4096) -> Iterator[Dict[str, Any]]:
        """Yield ledger entries incrementally to avoid full-buffer blocking reads."""
        if not self.ledger_path:
            return
        emitted = 0
        with open(self.ledger_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                if emitted >= max(1, int(max_entries)):
                    break
                line = str(raw_line or "").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                emitted += 1
                yield payload

    def _extract_drift_episodes(self) -> None:
        """
        Identify contiguous divergence periods from turn_alignment_history.
        
        A DriftEpisode is a sequence of consecutive misaligned turns.
        """
        self.drift_episodes.clear()

        if not self.turn_alignment_history:
            return

        episode_start = None
        episode_topic = None

        for turn_num, is_aligned, goal, _occurred_at in self.turn_alignment_history:
            if not is_aligned:
                if episode_start is None:
                    # Start new episode
                    episode_start = turn_num
                    episode_topic = self.turn_topic_labels.get(turn_num) or goal
            else:
                if episode_start is not None:
                    # End episode
                    duration = turn_num - episode_start
                    halt_count = sum(
                        1
                        for index in range(episode_start, turn_num)
                        if self.turn_halt_history.get(index, False)
                    )
                    time_bucket = self._infer_time_bucket(episode_start)
                    episode = DriftEpisode(
                        start_turn=episode_start,
                        end_turn=turn_num - 1,
                        duration=duration,
                        trigger_topic=episode_topic,
                        trigger_time_bucket=time_bucket,
                        emotional_signature=None,
                        recovery_method=None,
                        halt_count=halt_count,
                        resumed_successfully=True,
                        burnout_probability=self._estimate_burnout_probability(
                            duration=duration,
                            halt_count=halt_count,
                            time_bucket=time_bucket,
                            recovered=True,
                        ),
                    )
                    self.drift_episodes.append(episode)
                    episode_start = None
                    episode_topic = None

        # Handle ongoing episode
        if episode_start is not None:
            latest_turn = max(t for t, _, _, _ in self.turn_alignment_history)
            duration = max(1, (latest_turn - episode_start) + 1)
            halt_count = sum(
                1
                for index in range(episode_start, latest_turn + 1)
                if self.turn_halt_history.get(index, False)
            )
            time_bucket = self._infer_time_bucket(episode_start)
            episode = DriftEpisode(
                start_turn=episode_start,
                end_turn=None,
                duration=duration,
                trigger_topic=episode_topic,
                trigger_time_bucket=time_bucket,
                emotional_signature=None,
                recovery_method=None,
                halt_count=halt_count,
                resumed_successfully=False,
                burnout_probability=self._estimate_burnout_probability(
                    duration=duration,
                    halt_count=halt_count,
                    time_bucket=time_bucket,
                    recovered=False,
                ),
            )
            self.drift_episodes.append(episode)

    def _detect_temporal_patterns(self) -> None:
        """
        Detect recurring temporal patterns (hour-of-day, day-of-week clustering).
        
        Creates or updates BehavioralPattern for time-based drifts.
        """
        time_buckets: Dict[str, List[DriftEpisode]] = defaultdict(list)

        # Group episodes by observed time buckets from parsed ledger timestamps.
        for episode in self.drift_episodes:
            time_bucket = episode.trigger_time_bucket or self._infer_time_bucket(episode.start_turn)
            if time_bucket:
                time_buckets[time_bucket].append(episode)

        # Create pattern for frequent time-based clusters
        for time_bucket, episodes in time_buckets.items():
            if len(episodes) >= self.min_pattern_frequency:
                pattern_id = f"temporal_{time_bucket}"
                pattern = BehavioralPattern(
                    pattern_id=pattern_id,
                    pattern_name=f"Drift during {time_bucket}",
                    frequency=len(episodes),
                    confidence=self._compute_confidence(len(episodes), len(self.drift_episodes)),
                    correlated_times=[time_bucket],
                    average_episode_duration=sum(e.duration for e in episodes) / len(episodes),
                    last_observed_turn=max(e.end_turn or e.start_turn for e in episodes),
                    evidence_weight=len(episodes) / max(1, len(self.drift_episodes))
                )
                self.behavioral_patterns[pattern_id] = pattern

    def _detect_topic_patterns(self) -> None:
        """
        Detect recurring topic-based drift patterns (rabbit holes, avoidance loops).
        
        Creates patterns for repeated distractions or avoidance of specific goals.
        """
        topic_clusters: Dict[str, List[DriftEpisode]] = defaultdict(list)

        for episode in self.drift_episodes:
            topic = episode.trigger_topic or "unknown"
            topic_clusters[topic].append(episode)

        # Create pattern for recurring topics
        for topic, episodes in topic_clusters.items():
            if len(episodes) >= self.min_pattern_frequency and topic != "unknown":
                pattern_id = f"topic_{topic.lower().replace(' ', '_')}"
                pattern = BehavioralPattern(
                    pattern_id=pattern_id,
                    pattern_name=f"Recurrent diversion: {topic}",
                    frequency=len(episodes),
                    confidence=self._compute_confidence(len(episodes), len(self.drift_episodes)),
                    correlated_topics=[topic],
                    average_episode_duration=sum(e.duration for e in episodes) / len(episodes),
                    last_observed_turn=max(e.end_turn or e.start_turn for e in episodes),
                    evidence_weight=len(episodes) / max(1, len(self.drift_episodes))
                )
                self.behavioral_patterns[pattern_id] = pattern

    def _compute_recovery_effectiveness(self) -> None:
        """
        Analyze what interventions and recovery methods correlate with successful alignment restoration.
        
        Updates each pattern with success metrics.
        """
        for pattern in self.behavioral_patterns.values():
            # Calculate success rate for this pattern
            matching_episodes = [e for e in self.drift_episodes if self._episode_matches_pattern(e, pattern)]
            if matching_episodes:
                recovered = sum(1 for e in matching_episodes if e.resumed_successfully)
                pattern.success_after_intervention = (recovered / len(matching_episodes)) * 100.0

                recovery_times = [e.recovery_time_minutes for e in matching_episodes if e.recovery_time_minutes]
                if recovery_times:
                    pattern.average_recovery_time = sum(recovery_times) / len(recovery_times)

    def _build_evidence_graph(self) -> None:
        """Build weighted Event -> Episode -> Pattern -> Outcome evidence edges."""
        self.evidence_graph = EvidenceGraph()

        for episode in self.drift_episodes:
            episode_label = f"episode:{episode.start_turn}-{episode.end_turn if episode.end_turn is not None else 'ongoing'}"
            if episode.trigger_time_bucket:
                self.evidence_graph.add_observation(
                    source=f"event:time_bucket:{episode.trigger_time_bucket}",
                    target=episode_label,
                )
            if episode.trigger_topic:
                self.evidence_graph.add_observation(
                    source=f"event:topic:{episode.trigger_topic}",
                    target=episode_label,
                )
            outcome = "recovered" if episode.resumed_successfully else "unrecovered"
            self.evidence_graph.add_observation(
                source=episode_label,
                target=f"outcome:{outcome}",
                weight=max(0.1, 1.0 - episode.burnout_probability),
            )

        for pattern in self.behavioral_patterns.values():
            pattern_label = f"pattern:{pattern.pattern_id}"
            success_rate = max(0.0, min(1.0, pattern.success_after_intervention / 100.0))
            self.evidence_graph.add_observation(
                source=pattern_label,
                target="outcome:recovered",
                weight=success_rate,
            )
            self.evidence_graph.add_observation(
                source=pattern_label,
                target="outcome:unrecovered",
                weight=max(0.0, 1.0 - success_rate),
            )

    def _synthesize_reflection_summary(self) -> ReflectionSummary:
        """
        Build high-level reflection summary from accumulated evidence.
        
        Returns:
            ReflectionSummary with risk assessment and recommendations.
        """
        if not self.drift_episodes:
            return ReflectionSummary(
                current_risk_level="low",
                predicted_drift_probability=0.0,
                likely_trigger_category="unknown",
                primary_pattern=None,
                confidence_score=0.1,
                observable_signals=["no_drift_episodes_observed"]
            )

        # Rank patterns by confidence and recency
        sorted_patterns = sorted(
            self.behavioral_patterns.values(),
            key=lambda p: (p.confidence, p.last_observed_turn),
            reverse=True
        )

        primary_pattern = sorted_patterns[0] if sorted_patterns else None
        secondary_patterns = sorted_patterns[1:3] if len(sorted_patterns) > 1 else []

        # Assess current risk level
        recent_episodes = [e for e in self.drift_episodes if self._is_recent(e)]
        drift_prob = self._estimate_drift_probability(recent_episodes)

        if drift_prob >= 0.65:
            risk_level = "high"
        elif drift_prob >= 0.35:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # Determine trigger category
        trigger_category = self._infer_trigger_category(primary_pattern, recent_episodes)

        # Build observable signals
        observable_signals = self._extract_observable_signals(recent_episodes, primary_pattern)
        burnout_signal, burnout_context = self._detect_burnout_signal()
        if burnout_signal:
            observable_signals.append(
                "BURNOUT_PROBABLE"
                f": trust_drop={burnout_context.get('trust_drop', 0.0):.3f}"
                f", window_minutes={burnout_context.get('window_minutes', 0)}"
            )

        # Generate recommendation
        recommendation = self._generate_recommendation(primary_pattern, risk_level, drift_prob)

        recent_turn_count = self._recent_turn_count()
        data_confidence = min(1.0, recent_turn_count / 30.0)
        pattern_confidence = primary_pattern.confidence if primary_pattern else 0.0
        summary_confidence = max(0.0, min(1.0, (0.6 * pattern_confidence) + (0.4 * data_confidence)))

        return ReflectionSummary(
            current_risk_level=risk_level,
            predicted_drift_probability=drift_prob,
            likely_trigger_category=trigger_category,
            primary_pattern=primary_pattern,
            secondary_patterns=secondary_patterns,
            recommended_intervention=recommendation,
            intervention_justification=self._generate_justification(primary_pattern, trigger_category),
            confidence_score=summary_confidence,
            observable_signals=observable_signals,
            recent_episode_count=len(recent_episodes)
        )

    def _detect_burnout_signal(self) -> Tuple[bool, Dict[str, Any]]:
        """Flag burnout when trust credit drops >20% within 2h of high-complexity activity."""
        timed_entries: List[Tuple[datetime, Dict[str, Any]]] = []
        for entry in self._reflection_entries:
            recorded_at = self._parse_ledger_timestamp(entry)
            if recorded_at is None:
                continue
            timed_entries.append((recorded_at, entry))

        if not timed_entries:
            return False, {}

        timed_entries.sort(key=lambda item: item[0])
        for index, (start_at, start_entry) in enumerate(timed_entries):
            if not self._is_high_complexity_activity(start_entry):
                continue

            start_credit = self._extract_trust_credit(start_entry)
            if start_credit <= 0.0:
                continue

            window_end = start_at + self.burnout_window
            min_credit = start_credit
            for probe_at, probe_entry in timed_entries[index:]:
                if probe_at > window_end:
                    break
                probe_credit = self._extract_trust_credit(probe_entry)
                if probe_credit <= 0.0:
                    continue
                min_credit = min(min_credit, probe_credit)

            drop_ratio = (start_credit - min_credit) / max(start_credit, 1e-9)
            if drop_ratio > self.burnout_trust_drop_threshold:
                return True, {
                    "trust_drop": round(drop_ratio, 3),
                    "window_minutes": int(self.burnout_window.total_seconds() // 60),
                    "start_credit": round(start_credit, 3),
                    "min_credit": round(min_credit, 3),
                }

        return False, {}

    @staticmethod
    def _extract_trust_credit(entry: Dict[str, Any]) -> float:
        behavioral = entry.get("behavioral_ledger")
        if isinstance(behavioral, dict):
            value = behavioral.get("trust_credit")
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass

        for key in ("trust_credit_after", "trust_credit_before"):
            value = entry.get(key)
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _is_high_complexity_activity(entry: Dict[str, Any]) -> bool:
        explicit_complexity = str(
            entry.get("complexity")
            or entry.get("planner_complexity")
            or entry.get("turn_complexity")
            or ""
        ).strip().lower()
        if explicit_complexity in {"complex", "high", "very_high"}:
            return True

        active_goals = entry.get("active_goals")
        if isinstance(active_goals, list) and len(active_goals) >= 3:
            return True

        excerpt = str(entry.get("user_input_excerpt") or "")
        return len(excerpt) >= 180

    # ============ Helper Methods ============

    def _infer_time_bucket(self, turn_num: int) -> Optional[str]:
        """Infer weekday/hour bucket from parsed ledger timestamp."""
        for history_turn, _aligned, _goal, occurred_at in self.turn_alignment_history:
            if history_turn != turn_num:
                continue
            if occurred_at is None:
                return None
            return f"{occurred_at.strftime('%a')}_h{occurred_at.hour:02d}"
        return None

    def _compute_confidence(self, frequency: int, total: int) -> float:
        """
        Compute statistical confidence (0.0-1.0) for a pattern.
        
        Higher frequency and lower total variance = higher confidence.
        """
        if total <= 0 or frequency <= 0:
            return 0.0
        n = float(max(1, total))
        p = max(0.0, min(1.0, float(frequency) / n))
        z = 1.96
        z2 = z * z
        denominator = 1.0 + (z2 / n)
        center = p + (z2 / (2.0 * n))
        margin = z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
        wilson_lower = (center - margin) / denominator
        sample_coverage = min(1.0, n / 12.0)
        return max(0.0, min(1.0, wilson_lower * sample_coverage))

    def _episode_matches_pattern(self, episode: DriftEpisode, pattern: BehavioralPattern) -> bool:
        """Check if a drift episode matches a behavioral pattern."""
        if pattern.correlated_topics and episode.trigger_topic not in pattern.correlated_topics:
            return False
        if pattern.correlated_times and episode.trigger_time_bucket not in pattern.correlated_times:
            return False
        return True

    def _is_recent(self, episode: DriftEpisode) -> bool:
        """Check if episode is in the recent window."""
        end_turn = episode.end_turn if episode.end_turn is not None else episode.start_turn
        latest_turn = max(t for t, _, _, _ in self.turn_alignment_history) if self.turn_alignment_history else 0
        return (latest_turn - end_turn) < self.recent_window_turns

    def _recent_turn_count(self) -> int:
        if not self.turn_alignment_history:
            return 0
        latest_turn = max(t for t, _, _, _ in self.turn_alignment_history)
        lower_bound = latest_turn - self.recent_window_turns
        return sum(1 for t, _, _, _ in self.turn_alignment_history if t > lower_bound)

    def _estimate_drift_probability(self, recent_episodes: List[DriftEpisode]) -> float:
        """Estimate drift probability using posterior and recovery pressure signals."""
        if not self.turn_alignment_history:
            return 0.0

        latest_turn = max(t for t, _, _, _ in self.turn_alignment_history)
        lower_bound = latest_turn - self.recent_window_turns
        recent_turns = [(t, aligned) for t, aligned, _goal, _at in self.turn_alignment_history if t > lower_bound]
        if not recent_turns:
            return 0.0

        drift_turns = sum(1 for _t, aligned in recent_turns if not aligned)
        alpha = 1.0 + drift_turns
        beta = 1.0 + (len(recent_turns) - drift_turns)
        posterior_mean = alpha / (alpha + beta)

        episode_density = min(1.0, len(recent_episodes) / max(1.0, len(recent_turns) / 4.0))
        recovery_success = 0.0
        if recent_episodes:
            recovery_success = sum(1.0 for e in recent_episodes if e.resumed_successfully) / float(len(recent_episodes))
        recovery_penalty = 1.0 - recovery_success

        burnout_pressure = 0.0
        if recent_episodes:
            burnout_pressure = sum(e.burnout_probability for e in recent_episodes) / float(len(recent_episodes))

        probability = (
            (0.55 * posterior_mean)
            + (0.20 * episode_density)
            + (0.15 * recovery_penalty)
            + (0.10 * burnout_pressure)
        )
        return max(0.0, min(1.0, probability))

    def _estimate_burnout_probability(
        self,
        *,
        duration: int,
        halt_count: int,
        time_bucket: Optional[str],
        recovered: bool,
    ) -> float:
        """Estimate burnout pressure from episode-level evidence."""
        duration_factor = min(1.0, float(max(0, duration)) / 6.0)
        halt_factor = min(1.0, float(max(0, halt_count)) / 3.0)
        night_factor = 0.0
        if time_bucket:
            match = re.search(r"_h(\d{2})$", time_bucket)
            if match:
                hour = int(match.group(1))
                if hour >= 22 or hour <= 5:
                    night_factor = 1.0
        unresolved_factor = 0.0 if recovered else 1.0

        burnout = (0.35 * duration_factor) + (0.25 * halt_factor) + (0.20 * night_factor) + (0.20 * unresolved_factor)
        return max(0.0, min(1.0, burnout))

    @staticmethod
    def _extract_goal_label(entry: Dict[str, Any]) -> str:
        goals = entry.get("session_goals")
        if isinstance(goals, list) and goals:
            return str(goals[0])
        active_goals = entry.get("active_goals")
        if isinstance(active_goals, list) and active_goals:
            first_goal = active_goals[0]
            if isinstance(first_goal, dict):
                description = str(first_goal.get("description") or "").strip()
                if description:
                    return description
        return "unknown"

    @staticmethod
    def _extract_topic_label(entry: Dict[str, Any], fallback: str) -> str:
        excerpt = str(entry.get("user_input_excerpt") or "").strip().lower()
        if not excerpt:
            return str(fallback or "unknown")
        tokens = [token for token in re.findall(r"[a-z0-9]+", excerpt) if len(token) >= 4]
        if not tokens:
            return str(fallback or "unknown")
        return " ".join(tokens[:3])

    @staticmethod
    def _parse_ledger_timestamp(entry: Dict[str, Any]) -> Optional[datetime]:
        for key in ("recorded_at", "occurred_at", "wall_time", "created_at", "timestamp"):
            value = entry.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                try:
                    return datetime.fromtimestamp(float(value))
                except Exception:
                    continue
            if isinstance(value, str):
                raw = value.strip()
                if not raw:
                    continue
                normalized = raw.replace("Z", "+00:00")
                try:
                    return datetime.fromisoformat(normalized)
                except ValueError:
                    continue
        return None

    def _infer_trigger_category(self, pattern: Optional[BehavioralPattern], episodes: List[DriftEpisode]) -> str:
        """
        Infer primary trigger category: fatigue, avoidance, distraction, or unknown.
        """
        if not pattern:
            return "unknown"

        pattern_name_lower = pattern.pattern_name.lower()
        if "fatigue" in pattern_name_lower or "late" in pattern_name_lower or "night" in pattern_name_lower:
            return "fatigue"
        elif "avoid" in pattern_name_lower:
            return "avoidance"
        elif "distract" in pattern_name_lower or "rabbit" in pattern_name_lower:
            return "distraction"

        return "unknown"

    def _extract_observable_signals(self, episodes: List[DriftEpisode], pattern: Optional[BehavioralPattern]) -> List[str]:
        """Extract observable behavioral signals to monitor."""
        signals = []
        if episodes:
            avg_duration = sum(e.duration for e in episodes) / len(episodes)
            signals.append(f"recent_avg_episode_duration: {avg_duration:.1f} turns")
            avg_burnout = sum(e.burnout_probability for e in episodes) / len(episodes)
            signals.append(f"recent_burnout_pressure: {avg_burnout:.2f}")

        if pattern:
            signals.append(f"primary_pattern_observed: {pattern.pattern_name}")
            if pattern.average_recovery_time:
                signals.append(f"avg_recovery_time: {pattern.average_recovery_time:.0f} minutes")

        return signals

    def _generate_recommendation(self, pattern: Optional[BehavioralPattern], risk_level: str, drift_prob: float) -> str:
        """Generate specific, actionable recommendation."""
        if risk_level == "high":
            if pattern:
                return f"Strong signal: {pattern.pattern_name}. Consider proactive intervention before drift occurs."
            return "High drift risk detected. Return to active task or initiate goal recalibration."
        elif risk_level == "moderate":
            if pattern and pattern.success_after_intervention > 60:
                return "Moderate drift risk. Past episodes suggest structured realignment prompts are effective."
            return "Monitor for drift signals. Realignment may be needed soon."
        else:
            return "No immediate intervention needed. Stay aligned with declared goals."

    def _generate_justification(self, pattern: Optional[BehavioralPattern], category: str) -> str:
        """Generate justification for recommendation."""
        if not pattern:
            return "Pattern confidence too low for specific recommendation."

        return f"Pattern '{pattern.pattern_name}' observed {pattern.frequency} times with {pattern.confidence * 100:.0f}% confidence. " \
               f"Trigger category: {category}. Success rate: {pattern.success_after_intervention:.0f}%."

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Return a summary of detected patterns for debugging/analysis."""
        return {
            "total_episodes": len(self.drift_episodes),
            "patterns_detected": len(self.behavioral_patterns),
            "evidence_edges": len(self.evidence_graph.edges),
            "patterns": [
                {
                    "id": p.pattern_id,
                    "name": p.pattern_name,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                    "success_rate": p.success_after_intervention
                }
                for p in self.behavioral_patterns.values()
            ]
        }

    def get_evidence_graph_snapshot(self, max_edges: int = 20) -> Dict[str, Any]:
        """Return a bounded, JSON-safe evidence graph snapshot for session state."""
        try:
            limit = max(0, int(max_edges))
        except (TypeError, ValueError):
            limit = 20

        ranked_edges = sorted(
            self.evidence_graph.edges.values(),
            key=lambda edge: (edge.weight, edge.observations),
            reverse=True,
        )
        selected_edges = ranked_edges[:limit]
        return {
            "node_count": len({node for edge in self.evidence_graph.edges.values() for node in (edge.source, edge.target)}),
            "edge_count": len(self.evidence_graph.edges),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": round(float(edge.weight), 4),
                    "observations": int(edge.observations),
                }
                for edge in selected_edges
            ],
        }
