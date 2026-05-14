"""Response generation and ranking mid-layer.

Pipeline:
    generate -> hard filter -> normalized scoring -> select best

Phase 1:
    - hard coherence/relevance filter before scoring
    - diversity injection via intent reinterpretation and response-goal shifts
    - normalized dimensions for stable weight behavior

Phase 2:
    - stateful emotion biasing field over selection (adaptive, not static)
    - response risk gating tied to relationship attachment
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any

from dadbot.core.emotion_state import EmotionState

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Scoring dimension weights (tuned per phase)."""

    coherence: float = 0.25
    relevance: float = 0.25
    persona_consistency: float = 0.20
    novelty: float = 0.20
    redundancy_penalty: float = 0.10


@dataclass
class ResponseCandidate:
    """Single response candidate with metadata."""

    text: str
    source: str
    confidence: float = 1.0
    tone: str = "neutral"
    intensity: float = 0.5
    stance: str = "balanced"
    response_goal: str = "inform"
    risk_level: float = 0.3
    depth: str = "medium"


class ResponseEngine:
    """Multi-candidate generation + ranking decision layer.

    Public interface:
        run(context) -> str

    Internal methods:
        generate_candidates(context, n=5) -> list[ResponseCandidate]
        score_candidate(candidate, context) -> float
        select_best(candidates_with_scores) -> ResponseCandidate
    """

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()
        self._recent_responses: list[str] = []
        self._emotion_state = EmotionState.neutral()
        self._last_turn_telemetry: dict[str, Any] = {}
        # Lightweight learned preference model (online-updated linear head).
        self._reward_weights: dict[str, float] = {
            "base_score": 0.25,
            "emotion_bias": 0.10,
            "memory_relevance": 0.20,
            "user_alignment": 0.20,
            "trajectory_alignment": 0.15,
            "predicted_user_reaction": 0.20,
            "risk_level": -0.10,
        }
        self._reward_bias: float = 0.0
        self._reward_learning_rate: float = 0.08
        self._reward_anchor_weights: dict[str, float] = dict(self._reward_weights)
        self._reward_weight_bound: float = 1.25
        self._reward_decay: float = 0.015
        self._feedback_samples: int = 0
        self._feedback_ema_abs: float = 0.0
        self._selection_style_counts: dict[str, int] = {}
        self._selection_count: int = 0
        self._recent_selected_sources: list[str] = []
        self._source_history_window: int = 24
        self._embedding_salt: str = "dadbot-memory-v1"
        self._source_rebalance_enabled: bool = False
        self._history_adaptation_enabled: bool = False

    def generate_candidates(
        self,
        context: Any,
        n: int = 5,
    ) -> list[ResponseCandidate]:
        """Generate 3–5 distinct response candidates from execution context.

        Strategy: create candidates by varying approach, depth, and tone.

        Args:
            context: ExecutionContext with user input, execution results, persona state
            n: target number of candidates (default 5, typically 3–5)

        Returns:
            list of ResponseCandidate with distinct generation strategies
        """
        candidates: list[ResponseCandidate] = []
        user_input = str(getattr(context, "user_input", "") or "")
        intent = self._infer_intent(user_input)

        # Strategy 1: Direct response (base approach)
        direct = self._generate_direct_response(context)
        if direct:
            candidates.append(
                ResponseCandidate(
                    text=direct,
                    source="direct",
                    confidence=0.95,
                    tone="neutral",
                    intensity=0.35,
                    stance="balanced",
                    response_goal="inform",
                    risk_level=0.25,
                )
            )

        # Strategy 2: Persona-inflected (add persona flavor)
        persona_flavored = self._generate_persona_inflected(context)
        if persona_flavored:
            candidates.append(
                ResponseCandidate(
                    text=persona_flavored,
                    source="persona_inflection",
                    confidence=0.90,
                    tone="warm",
                    intensity=0.45,
                    stance="supportive",
                    response_goal="engage",
                    risk_level=0.45,
                )
            )

        # Strategy 3: Question echo (reflect back question in response)
        question_echo = self._generate_question_echo(context)
        if question_echo:
            candidates.append(
                ResponseCandidate(
                    text=question_echo,
                    source="question_echo",
                    confidence=0.85,
                    tone="reflective",
                    intensity=0.40,
                    stance="curious",
                    response_goal="clarify",
                    risk_level=0.20,
                )
            )

        # Strategy 4: Concise variant (shorter, punchier)
        concise = self._generate_concise_variant(context)
        if concise and concise != direct:  # avoid duplication
            candidates.append(
                ResponseCandidate(
                    text=concise,
                    source="concise_variant",
                    confidence=0.80,
                    tone="calm",
                    intensity=0.30,
                    stance="pragmatic",
                    response_goal="clarify" if intent == "question" else "inform",
                    risk_level=0.15,
                )
            )

        # Strategy 5: Elaborated variant (longer, more detailed)
        if len(candidates) < n:
            elaborated = self._generate_elaborated_variant(context)
            if elaborated and elaborated != direct:
                candidates.append(
                    ResponseCandidate(
                        text=elaborated,
                        source="elaborated_variant",
                        confidence=0.80,
                        tone="warm",
                        intensity=0.55,
                        stance="supportive",
                        response_goal="engage" if intent == "emotional" else "inform",
                        risk_level=0.40,
                    )
                )

        # Diversity injection: reinterpret intent and change response goal.
        reinterpret = self._generate_reinterpreted_intent(context, intent=intent)
        if reinterpret:
            candidates.append(
                ResponseCandidate(
                    text=reinterpret,
                    source="reinterpret_intent",
                    confidence=0.78,
                    tone="analytical",
                    intensity=0.50,
                    stance="reframing",
                    response_goal="clarify",
                    risk_level=0.35,
                )
            )

        goal_shift = self._generate_goal_shift_variant(context, intent=intent)
        if goal_shift:
            candidates.append(
                ResponseCandidate(
                    text=goal_shift,
                    source="goal_shift",
                    confidence=0.76,
                    tone="engaging",
                    intensity=0.60,
                    stance="forward",
                    response_goal="engage",
                    risk_level=0.55,
                    depth="short",
                )
            )

        # Structured diversity axes: tone/depth/risk/intensity combinations.
        if direct:
            candidates.extend(self._generate_structured_variants(base_text=direct, intent=intent))

        # Stochastic prompt-style variants (deterministic seed by context).
        candidates.extend(self._generate_stochastic_variants(context=context, intent=intent))

        # Optional multi-model candidate ingestion when provided by runtime metadata.
        candidates.extend(self._ingest_external_model_candidates(context=context))

        return candidates[:n]  # cap at requested count

    def _seed_for_context(self, context: Any) -> str:
        raw_trace_id = getattr(context, "trace_id", "")
        if isinstance(raw_trace_id, (str, int, float)):
            trace_id = str(raw_trace_id or "")
        else:
            trace_id = ""
        user_input = str(getattr(context, "user_input", "") or "")
        seed_material = f"{trace_id}|{user_input}".encode("utf-8", errors="ignore")
        return hashlib.sha256(seed_material).hexdigest()

    @staticmethod
    def _deterministic_unit(seed: str, slot: int) -> float:
        digest = hashlib.sha256(f"{seed}|{slot}".encode("utf-8", errors="ignore")).hexdigest()
        return int(digest[:12], 16) / float(16**12 - 1)

    def _generate_stochastic_variants(self, *, context: Any, intent: str) -> list[ResponseCandidate]:
        user_input = str(getattr(context, "user_input", "") or "").strip()
        if not user_input:
            return []
        seed = self._seed_for_context(context)
        templates = [
            "Let's solve this as a next-step plan: {text}",
            "If we optimize for stability first, here's a framing: {text}",
            "Quick triage approach for this: {text}",
            "I can give you the 30-second and 5-minute version for: {text}",
        ]
        tones = ["calm", "warm", "engaging", "analytical"]
        depths = ["short", "medium", "deep"]

        variants: list[ResponseCandidate] = []
        for idx in range(2):
            u0 = self._deterministic_unit(seed, (idx * 5) + 0)
            u1 = self._deterministic_unit(seed, (idx * 5) + 1)
            u2 = self._deterministic_unit(seed, (idx * 5) + 2)
            u3 = self._deterministic_unit(seed, (idx * 5) + 3)
            template = templates[int(u0 * len(templates)) % len(templates)]
            tone = tones[int(u1 * len(tones)) % len(tones)]
            depth = depths[int(u2 * len(depths)) % len(depths)]
            risk = 0.15 + (0.7 * u2)
            intensity = 0.2 + (0.7 * u3)
            text = template.format(text=user_input)
            text = self._render_structured_variant(base_text=text, tone=tone, depth=depth, intent=intent)
            variants.append(
                ResponseCandidate(
                    text=text,
                    source=f"stochastic_prompt_{idx+1}",
                    confidence=0.70,
                    tone=tone,
                    intensity=max(0.0, min(1.0, intensity)),
                    stance="balanced",
                    response_goal="clarify" if intent == "question" else "inform",
                    risk_level=max(0.0, min(1.0, risk)),
                    depth=depth,
                )
            )
        return variants

    def _ingest_external_model_candidates(self, *, context: Any) -> list[ResponseCandidate]:
        metadata = getattr(context, "job_metadata", None)
        if not isinstance(metadata, dict):
            return []

        external = list(metadata.get("model_candidate_texts") or [])
        if not external:
            return []

        accepted: list[ResponseCandidate] = []
        for idx, value in enumerate(external[:3]):
            text = str(value or "").strip()
            if not text:
                continue
            accepted.append(
                ResponseCandidate(
                    text=text,
                    source=f"multi_model_{idx+1}",
                    confidence=0.72,
                    tone="neutral",
                    intensity=0.45,
                    stance="balanced",
                    response_goal="inform",
                    risk_level=0.35,
                    depth="medium",
                )
            )
        return accepted

    def _generate_structured_variants(self, *, base_text: str, intent: str) -> list[ResponseCandidate]:
        tones = ["calm", "warm", "engaging"]
        depths = ["short", "medium", "deep"]
        risk_levels = [0.2, 0.45, 0.7]
        intensities = [0.25, 0.5, 0.8]

        variants: list[ResponseCandidate] = []
        # Keep this deterministic and small: three controlled variants.
        for idx in range(3):
            tone = tones[idx]
            depth = depths[idx]
            risk = risk_levels[idx]
            intensity = intensities[idx]
            text = self._render_structured_variant(base_text=base_text, tone=tone, depth=depth, intent=intent)
            variants.append(
                ResponseCandidate(
                    text=text,
                    source=f"structured_axis_{idx+1}",
                    confidence=0.74,
                    tone=tone,
                    intensity=intensity,
                    stance="balanced" if idx != 2 else "forward",
                    response_goal="clarify" if depth == "short" else ("engage" if tone == "engaging" else "inform"),
                    risk_level=risk,
                    depth=depth,
                )
            )
        return variants

    def _render_structured_variant(self, *, base_text: str, tone: str, depth: str, intent: str) -> str:
        text = base_text
        if tone == "calm":
            text = f"Let's keep this steady. {base_text}"
        elif tone == "warm":
            text = f"I hear you and I'm with you. {base_text}"
        elif tone == "engaging":
            text = f"Let's tackle this together. {base_text}"

        if depth == "short":
            return text.split(".")[0] + "."
        if depth == "deep":
            if intent == "question":
                return f"{text} Step 1: define the outcome. Step 2: compare options. Step 3: choose the smallest next action."
            return f"{text} Here's a deeper read: what matters most, what tradeoff is hardest, and what's the safest next move."
        return text

    def filter_candidates(self, candidates: list[ResponseCandidate], context: Any) -> list[ResponseCandidate]:
        """Hard pre-score filter: only coherent and relevant candidates pass."""
        return [
            candidate
            for candidate in candidates
            if self.is_coherent(candidate)
            and self.is_relevant(candidate, context)
            and self._satisfies_persona_constraints(candidate, context)
        ]

    def is_coherent(self, candidate: ResponseCandidate) -> bool:
        return self._score_coherence(candidate.text) >= 0.25

    def is_relevant(self, candidate: ResponseCandidate, context: Any) -> bool:
        return self._score_relevance(candidate.text, context) >= 0.15

    def score_candidate(self, candidate: ResponseCandidate, context: Any) -> float:
        """Score a single candidate across Phase 1 dimensions.

        Dimensions:
            - coherence: internal logical consistency (0–1)
            - relevance: matches context/question appropriateness (0–1)
            - persona_consistency: aligns with known persona traits (0–1)
            - novelty: encourages variation from recent responses (0–1)
            - redundancy_penalty: penalizes similarity to recent responses (negative)

        Args:
            candidate: ResponseCandidate to score
            context: ExecutionContext for reference

        Returns:
            float score in approximate range [0–1] after weighting
        """
        persona_calibration = self._persona_calibration_factor(context)
        raw_persona_score = self._score_persona_consistency(candidate.text, context)
        persona_signal = max(0.0, min(1.0, raw_persona_score * persona_calibration))
        emotion_state = self._resolve_emotion_state(context)
        base_score = self._base_score(
            coherence=self._score_coherence(candidate.text),
            relevance=self._score_relevance(candidate.text, context),
            persona=persona_signal,
            novelty=self._score_novelty(candidate.text),
            redundancy=self._score_redundancy(candidate.text),
        )
        emotion_score = self.emotion_alignment(candidate.text, emotion_state)
        emotion_weight = self.compute_emotion_weight(emotion_state)
        emotion_bias = emotion_weight * (emotion_score - 0.5)
        memory_relevance = self._score_memory_relevance(candidate, context)
        user_alignment = self._score_user_alignment(candidate, context)
        trajectory_alignment = self._score_trajectory_alignment(candidate, context)
        user_reaction = self._simulate_user_reaction(candidate, context, emotion_state)

        final_score = (
            base_score
            + emotion_bias
            + 0.18 * memory_relevance
            + 0.18 * user_alignment
            + 0.12 * trajectory_alignment
            + 0.12 * user_reaction
        )
        diversity_penalty = self._distribution_collapse_penalty(candidate)
        interaction_bonus = self._interaction_bonus(
            emotion_bias=emotion_bias,
            memory_relevance=memory_relevance,
            user_alignment=user_alignment,
            trajectory_alignment=trajectory_alignment,
            predicted_user_reaction=user_reaction,
        )
        learned_preference = self._learned_preference_score(
            {
                "base_score": base_score,
                "emotion_bias": emotion_bias,
                "memory_relevance": memory_relevance,
                "user_alignment": user_alignment,
                "trajectory_alignment": trajectory_alignment,
                "predicted_user_reaction": user_reaction,
                "risk_level": candidate.risk_level,
            }
        )
        final_score = final_score + (0.12 * interaction_bonus) + (0.15 * learned_preference) - (0.08 * diversity_penalty)
        return max(0.0, min(1.0, final_score))

    def score_candidates(
        self,
        candidates: list[ResponseCandidate],
        context: Any,
    ) -> list[tuple[ResponseCandidate, float]]:
        """Score all candidates with normalized dimensions and adaptive emotion bias."""
        if not candidates:
            return []

        emotion_state = self._resolve_emotion_state(context)
        emotion_weight = self.compute_emotion_weight(emotion_state)
        persona_calibration = self._persona_calibration_factor(context)

        raw: dict[str, list[float]] = {
            "coherence": [self._score_coherence(c.text) for c in candidates],
            "relevance": [self._score_relevance(c.text, context) for c in candidates],
            "persona": [self._score_persona_consistency(c.text, context) for c in candidates],
            "novelty": [self._score_novelty(c.text) for c in candidates],
            "redundancy": [self._score_redundancy(c.text) for c in candidates]
            if self._history_adaptation_enabled
            else [0.0 for _ in candidates],
            "emotion": [self.emotion_alignment(c.text, emotion_state) for c in candidates],
            "memory": [self._score_memory_relevance(c, context) for c in candidates],
            "user_alignment": [self._score_user_alignment(c, context) for c in candidates],
            "trajectory": [self._score_trajectory_alignment(c, context) for c in candidates],
            "reaction": [self._simulate_user_reaction(c, context, emotion_state) for c in candidates],
        }
        normalized = {name: self._normalize(values) for name, values in raw.items()}
        shadow_events = self._collect_shadow_events(context)
        entropy_before = self._source_entropy()

        scored: list[tuple[ResponseCandidate, float]] = []
        telemetry_candidates: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            base_score = self._base_score(
                coherence=normalized["coherence"][idx],
                relevance=normalized["relevance"][idx],
                persona=max(0.0, min(1.0, normalized["persona"][idx] * persona_calibration)),
                novelty=normalized["novelty"][idx],
                redundancy=normalized["redundancy"][idx],
            )
            emotion_score = normalized["emotion"][idx]
            emotion_bias = emotion_weight * (emotion_score - 0.5)
            base_decision_score = (
                base_score
                + emotion_bias
                + 0.18 * normalized["user_alignment"][idx]
                + 0.12 * normalized["trajectory"][idx]
                + 0.12 * normalized["reaction"][idx]
            )
            shadow_influences = self._compute_shadow_influences(
                candidate=candidate,
                events=shadow_events,
            )
            coherence_weight = 0.08 * normalized["coherence"][idx] + float(shadow_influences["coherence_delta"])
            memory_weight = 0.18 * normalized["memory"][idx] + float(shadow_influences["memory_delta"])
            tool_weight = float(shadow_influences["tool_delta"])
            safety_weight = float(shadow_influences["safety_delta"])
            final_score = (
                base_decision_score
                + coherence_weight
                + memory_weight
                + tool_weight
                + safety_weight
            )
            diversity_penalty = self._distribution_collapse_penalty(candidate) if self._source_rebalance_enabled else 0.0
            interaction_bonus = self._interaction_bonus(
                emotion_bias=emotion_bias,
                memory_relevance=normalized["memory"][idx],
                user_alignment=normalized["user_alignment"][idx],
                trajectory_alignment=normalized["trajectory"][idx],
                predicted_user_reaction=normalized["reaction"][idx],
            )
            learned_preference = self._learned_preference_score(
                {
                    "base_score": base_score,
                    "emotion_bias": emotion_bias,
                    "memory_relevance": normalized["memory"][idx],
                    "user_alignment": normalized["user_alignment"][idx],
                    "trajectory_alignment": normalized["trajectory"][idx],
                    "predicted_user_reaction": normalized["reaction"][idx],
                    "risk_level": candidate.risk_level,
                }
            )
            final_score = final_score + (0.12 * interaction_bonus) + (0.15 * learned_preference) - (0.08 * diversity_penalty)
            source_dominance_penalty = self._source_dominance_penalty(candidate) if self._source_rebalance_enabled else 0.0
            final_score = final_score - (0.12 * source_dominance_penalty)
            bounded_final_score = max(0.0, min(1.0, final_score))
            scored.append((candidate, bounded_final_score))
            telemetry_candidates.append(
                {
                    "source": candidate.source,
                    "tone": candidate.tone,
                    "depth": candidate.depth,
                    "response_goal": candidate.response_goal,
                    "risk_level": float(candidate.risk_level),
                    "components": {
                        "base_decision_score": float(base_decision_score),
                        "base_score": float(base_score),
                        "emotion_score": float(emotion_score),
                        "emotion_weight": float(emotion_weight),
                        "emotion_bias": float(emotion_bias),
                        "persona_signal": float(max(0.0, min(1.0, normalized["persona"][idx] * persona_calibration))),
                        "coherence_weight": float(coherence_weight),
                        "memory_weight": float(memory_weight),
                        "tool_weight": float(tool_weight),
                        "safety_weight": float(safety_weight),
                        "memory_relevance": float(normalized["memory"][idx]),
                        "user_alignment": float(normalized["user_alignment"][idx]),
                        "trajectory_alignment": float(normalized["trajectory"][idx]),
                        "predicted_user_reaction": float(normalized["reaction"][idx]),
                        "distribution_collapse_penalty": float(diversity_penalty),
                        "source_dominance_penalty": float(source_dominance_penalty),
                        "interaction_bonus": float(interaction_bonus),
                        "learned_preference": float(learned_preference),
                    },
                    "reasoning": {
                        "safety": str(shadow_influences["reasoning"]["safety"]),
                        "tools": str(shadow_influences["reasoning"]["tools"]),
                        "memory": str(shadow_influences["reasoning"]["memory"]),
                        "coherence": str(shadow_influences["reasoning"]["coherence"]),
                    },
                    "final_score": float(bounded_final_score),
                }
            )
            telemetry_candidates[-1]["influence_share"] = self._influence_share(
                dict(telemetry_candidates[-1].get("components") or {}),
            )

        self._last_turn_telemetry = {
            "emotion_weight": float(emotion_weight),
            "candidate_count": len(candidates),
            "candidates": telemetry_candidates,
            "reward_model": {
                "weights": dict(self._reward_weights),
                "bias": float(self._reward_bias),
                "learning_rate": float(self._reward_learning_rate),
            },
            "behavioral_closure": {
                "source_entropy_before": float(entropy_before),
                "dominant_source_ratio_before": float(self._dominant_source_ratio()),
            },
            "persona_calibration": {
                "factor": float(persona_calibration),
                "traits_count": int(len(list(getattr(context, "persona_traits", []) or []))),
                "has_constraints": bool(isinstance(getattr(context, "persona_constraints", None), dict) and bool(getattr(context, "persona_constraints", None))),
            },
            "shadow_event_count": len(shadow_events),
        }
        return scored

    def select_best(
        self,
        candidates_with_scores: list[tuple[ResponseCandidate, float]],
    ) -> ResponseCandidate:
        """Select highest-scored candidate.

        Args:
            candidates_with_scores: list of (candidate, score) tuples

        Returns:
            ResponseCandidate with highest score
        """
        if not candidates_with_scores:
            raise ValueError("No candidates to select from")

        ranked = sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)
        best_candidate, best_score = ranked[0]
        dominant_ratio = self._dominant_source_ratio()
        if self._source_rebalance_enabled and dominant_ratio >= 0.60:
            score_margin = 0.08
            for candidate, score in ranked[1:]:
                if candidate.source != best_candidate.source and (best_score - score) <= score_margin:
                    best_candidate, best_score = candidate, score
                    break
        logger.debug(
            f"Selected response from {best_candidate.source} "
            f"(score={best_score:.3f})"
        )
        return best_candidate

    def _collect_shadow_events(self, context: Any) -> list[dict[str, Any]]:
        metadata = getattr(context, "job_metadata", None)
        if not isinstance(metadata, dict):
            return []

        direct = list(metadata.get("shadow_decision_bus") or [])
        telemetry_container = dict(metadata.get("response_engine_telemetry") or {})
        nested = list(telemetry_container.get("shadow_decision_bus") or [])
        events = direct or nested
        return [dict(event) for event in events if isinstance(event, dict)]

    @staticmethod
    def _shadow_source_group(source: str) -> str:
        src = str(source or "").strip().lower()
        if not src:
            return "generic"
        if "safety" in src or "guard" in src:
            return "safety"
        if any(token in src for token in ("tool", "mcp", "calendar", "email", "reminder", "agentic")):
            return "tool"
        if any(token in src for token in ("memory", "profile", "relationship", "context")):
            return "memory"
        return "generic"

    def _compute_shadow_influences(
        self,
        *,
        candidate: ResponseCandidate,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        safety_delta = 0.0
        tool_delta = 0.0
        memory_delta = 0.0
        coherence_delta = 0.0

        safety_notes: list[str] = []
        tool_notes: list[str] = []
        memory_notes: list[str] = []
        coherence_notes: list[str] = []

        candidate_source = str(candidate.source or "").strip().lower()
        for event in events:
            event_type = str(event.get("type") or "").strip().lower()
            event_source = str(event.get("source") or "unknown").strip().lower()
            source_group = self._shadow_source_group(event_source)

            try:
                priority = float(event.get("priority", 0.0) or 0.0)
            except Exception:
                priority = 0.0
            priority = max(0.0, min(1.0, priority))

            would_replace = bool(event.get("would_replace", False))
            reason = str(event.get("reason") or "").strip() or "no_reason"
            source_match_scale = 1.0 if event_source == candidate_source else 0.45
            replace_scale = 1.0 if would_replace else 0.6
            strength = (0.02 + (0.10 * priority)) * source_match_scale * replace_scale

            if event_type == "veto" or source_group == "safety":
                penalty = min(strength * 1.3, 0.18)
                safety_delta -= penalty
                if len(safety_notes) < 2:
                    safety_notes.append(f"{event_source}: {reason}")
                continue

            if source_group == "tool":
                boost = min(strength * 1.1, 0.14)
                tool_delta += boost
                if len(tool_notes) < 2:
                    tool_notes.append(f"{event_source}: {reason}")
                continue

            if source_group == "memory":
                boost = min(strength, 0.12)
                memory_delta += boost
                if len(memory_notes) < 2:
                    memory_notes.append(f"{event_source}: {reason}")
                continue

            coherence_delta += min(strength * 0.8, 0.08)
            if len(coherence_notes) < 2:
                coherence_notes.append(f"{event_source}: {reason}")

        safety_delta = max(-0.35, min(0.0, safety_delta))
        tool_delta = max(0.0, min(0.20, tool_delta))
        memory_delta = max(0.0, min(0.16, memory_delta))
        coherence_delta = max(0.0, min(0.10, coherence_delta))

        return {
            "safety_delta": float(safety_delta),
            "tool_delta": float(tool_delta),
            "memory_delta": float(memory_delta),
            "coherence_delta": float(coherence_delta),
            "reasoning": {
                "safety": "; ".join(safety_notes) if safety_notes else "No active safety veto pressure",
                "tools": "; ".join(tool_notes) if tool_notes else "No tool-driven boost applied",
                "memory": "; ".join(memory_notes) if memory_notes else "No memory-side shadow boost applied",
                "coherence": "; ".join(coherence_notes) if coherence_notes else "No generic shadow coherence adjustment",
            },
        }

    @staticmethod
    def _influence_share(components: dict[str, Any]) -> dict[str, float]:
        magnitudes = {
            "safety": abs(float(components.get("safety_weight", 0.0) or 0.0)),
            "tools": abs(float(components.get("tool_weight", 0.0) or 0.0)),
            "memory": abs(float(components.get("memory_weight", 0.0) or 0.0)),
            "coherence": abs(float(components.get("coherence_weight", 0.0) or 0.0)),
        }
        total = float(sum(magnitudes.values()))
        if total <= 1e-9:
            return {"safety": 0.0, "tools": 0.0, "memory": 0.0, "coherence": 0.0}
        return {name: float(value / total) for name, value in magnitudes.items()}

    def run(self, context: Any) -> str:
        """Orchestrator: generate → score → select → track.

        Single atomic call from control-plane execution flow.

        Args:
            context: ExecutionContext

        Returns:
            str: selected response text
        """
        self._apply_feedback_update(context)
        source_rebalance_flag = getattr(context, "enable_source_rebalancing", False)
        history_adaptation_flag = getattr(context, "enable_history_adaptation", False)
        self._source_rebalance_enabled = source_rebalance_flag if isinstance(source_rebalance_flag, bool) else False
        self._history_adaptation_enabled = (
            history_adaptation_flag if isinstance(history_adaptation_flag, bool) else False
        )

        generated = self.generate_candidates(context, n=8)
        if not generated:
            return self._terminal_no_candidate_result(context, reason="none_generated")

        filtered = self.filter_candidates(generated, context)
        if not filtered:
            return self._terminal_no_candidate_result(context, reason="all_filtered")

        scored = self.score_candidates(filtered, context)

        best = self.select_best(scored)
        ranked_scores = sorted([float(score) for _candidate, score in scored], reverse=True)
        selected_score = float(ranked_scores[0]) if ranked_scores else 0.0
        second_best_score = float(ranked_scores[1]) if len(ranked_scores) > 1 else selected_score
        decision_confidence = max(0.0, selected_score - second_best_score)

        selected_telemetry = None
        for item in list(self._last_turn_telemetry.get("candidates") or []):
            if str(item.get("source") or "") == best.source:
                selected_telemetry = item
                break
        if selected_telemetry is not None:
            self._last_turn_telemetry["selected"] = dict(selected_telemetry)
            self._last_turn_telemetry["selected_text_preview"] = best.text[:180]
            selected_components = dict(self._last_turn_telemetry["selected"].get("components") or {})
            self._last_turn_telemetry["selected"]["influence_share"] = self._influence_share(selected_components)
            self._last_turn_telemetry["selected"]["selected_score"] = float(selected_score)
            self._last_turn_telemetry["selected"]["second_best_score"] = float(second_best_score)
            self._last_turn_telemetry["selected"]["decision_confidence"] = float(decision_confidence)
        self._last_turn_telemetry["selected_score"] = float(selected_score)
        self._last_turn_telemetry["second_best_score"] = float(second_best_score)
        self._last_turn_telemetry["decision_confidence"] = float(decision_confidence)

        self._record_selection(best)

        try:
            context.response_engine_telemetry = dict(self._last_turn_telemetry)
        except Exception:
            logger.debug("Unable to attach response_engine_telemetry to context", exc_info=True)

        self._recent_responses.append(best.text)
        if len(self._recent_responses) > 10:
            self._recent_responses.pop(0)

        return best.text

    def _terminal_no_candidate_result(self, context: Any, *, reason: str) -> str:
        """Return a deterministic terminal fallback when no candidates survive."""
        terminal_text = str(getattr(context, "initial_response", "") or "").strip() or "I hear you."
        terminal = ResponseCandidate(
            text=terminal_text,
            source="canonical_empty",
            confidence=1.0,
            tone="neutral",
            intensity=0.0,
            stance="balanced",
            response_goal="fallback_terminal",
            risk_level=0.0,
            depth="minimal",
        )
        self._last_turn_telemetry = {
            "candidate_count": 0,
            "status": "no_valid_candidates",
            "reason": str(reason or "all_filtered"),
            "selected": {
                "source": terminal.source,
                "tone": terminal.tone,
                "depth": terminal.depth,
                "response_goal": terminal.response_goal,
                "risk_level": float(terminal.risk_level),
                "selected_score": 0.0,
                "second_best_score": 0.0,
                "decision_confidence": 0.0,
                "influence_share": {
                    "safety": 0.0,
                    "tools": 0.0,
                    "memory": 0.0,
                    "coherence": 0.0,
                },
            },
            "selected_score": 0.0,
            "second_best_score": 0.0,
            "decision_confidence": 0.0,
        }
        self._record_selection(terminal)
        try:
            context.response_engine_telemetry = dict(self._last_turn_telemetry)
        except Exception:
            logger.debug("Unable to attach response_engine_telemetry to context", exc_info=True)

        self._recent_responses.append(terminal.text)
        if len(self._recent_responses) > 10:
            self._recent_responses.pop(0)
        return terminal.text

    def get_last_turn_telemetry(self) -> dict[str, Any]:
        return dict(self._last_turn_telemetry)

    # ────────────────────────────────────────────────────────────────────────
    # Generation strategies
    # ────────────────────────────────────────────────────────────────────────

    def _generate_direct_response(self, context: Any) -> str | None:
        """Base strategy: straightforward, factual response."""
        # Placeholder: would extract from execution context
        user_input = getattr(context, "user_input", "")
        if not user_input:
            return None
        return f"I hear you: {user_input.strip()[:50]}..."

    def _generate_persona_inflected(self, context: Any) -> str | None:
        """Add persona flavor: inject personality traits."""
        base = self._generate_direct_response(context)
        if not base:
            return None
        # In real implementation, would apply persona transforms
        return f"[Dad voice] {base}"

    def _generate_question_echo(self, context: Any) -> str | None:
        """Reflect back the question as part of response."""
        user_input = getattr(context, "user_input", "")
        if not user_input or "?" not in user_input:
            return None
        return f"So you're asking... {user_input.strip()}? Here's the thing..."

    def _generate_concise_variant(self, context: Any) -> str | None:
        """Shorter, punchier variant."""
        base = self._generate_direct_response(context)
        if not base:
            return None
        # Simulate brevity
        return base.split(".")[0] + "."

    def _generate_elaborated_variant(self, context: Any) -> str | None:
        """Longer, more detailed variant."""
        base = self._generate_direct_response(context)
        if not base:
            return None
        return f"{base} And here's why that matters..."

    def _generate_reinterpreted_intent(self, context: Any, *, intent: str) -> str | None:
        user_input = str(getattr(context, "user_input", "") or "").strip()
        if not user_input:
            return None
        if intent == "question":
            return f"Instead of just answering quickly, let's frame this as a decision: {user_input}"
        if intent == "emotional":
            return f"Underneath this, it sounds like you're asking for stability and a next step: {user_input}"
        return f"One useful reframe here is to identify the core tradeoff in: {user_input}"

    def _generate_goal_shift_variant(self, context: Any, *, intent: str) -> str | None:
        user_input = str(getattr(context, "user_input", "") or "").strip()
        if not user_input:
            return None
        if intent == "question":
            return f"Quick clarifier before we go deep: what outcome do you want most from this?"
        if intent == "emotional":
            return "You're not alone in this. Want a steady 2-step plan for the next 10 minutes?"
        return f"If you want, I can give you a short plan and then a deeper version for: {user_input}"

    # ────────────────────────────────────────────────────────────────────────
    # Scoring helpers
    # ────────────────────────────────────────────────────────────────────────

    def _score_coherence(self, text: str) -> float:
        """Score internal logical consistency (0–1)."""
        # Placeholder: real implementation would check sentence structure,
        # contradiction detection, etc.
        if not text or len(text) < 3:
            return 0.0
        # Heuristic: longer, structured text is more coherent
        return min(len(text) / 180.0, 1.0)

    def _score_relevance(self, text: str, context: Any) -> float:
        """Score appropriateness to context/question (0–1)."""
        user_input = getattr(context, "user_input", "")
        if not user_input:
            return 0.5  # unknown relevance
        text_lower = str(text or "").lower()
        tokens = [word.strip(".,!?;:").lower() for word in str(user_input).split() if word.strip()]
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "can",
            "could",
            "do",
            "for",
            "i",
            "is",
            "it",
            "me",
            "my",
            "of",
            "please",
            "should",
            "tell",
            "the",
            "this",
            "to",
            "what",
            "you",
        }
        meaningful_tokens = [token for token in tokens if token and token not in stopwords]
        overlap_pool = meaningful_tokens or tokens
        overlap = sum(1 for token in overlap_pool if token and token in text_lower)
        lexical_score = min(overlap / max(len(overlap_pool), 1), 1.0)

        input_markers = set(tokens)
        intent_score = 0.0
        if input_markers.intersection({"help", "stuck", "decide", "decision", "next", "should"}):
            if any(marker in text_lower for marker in ["step", "plan", "steady", "next action", "solve", "decide", "clarif"]):
                intent_score = max(intent_score, 0.45)
        if "?" in str(user_input) and any(marker in text_lower for marker in ["here's", "let's", "quick", "step", "answer"]):
            intent_score = max(intent_score, 0.35)
        if input_markers.intersection({"worried", "anxious", "stress", "stressed", "overwhelmed"}):
            if any(marker in text_lower for marker in ["steady", "with you", "together", "calm", "breathe"]):
                intent_score = max(intent_score, 0.40)

        return max(0.0, min(1.0, max(lexical_score, intent_score)))

    def _score_persona_consistency(self, text: str, context: Any) -> float:
        """Score alignment with known persona traits (0–1)."""
        dad_markers = ["dad", "advice", "practical", "steady", "support"]
        matches = sum(1 for marker in dad_markers if marker.lower() in text.lower())
        trait_bonus = 0.0
        traits = getattr(context, "persona_traits", None)
        if isinstance(traits, list):
            trait_bonus = min(
                sum(1 for trait in traits if str(trait).lower() in text.lower()) / max(len(traits), 1),
                1.0,
            )
        return min((matches / 5.0) * 0.7 + trait_bonus * 0.3, 1.0)

    def _persona_calibration_factor(self, context: Any) -> float:
        traits = list(getattr(context, "persona_traits", []) or [])
        constraints = getattr(context, "persona_constraints", None)
        has_constraints = isinstance(constraints, dict) and bool(constraints)

        # Personality is influence-only: damp when no persona context exists,
        # modestly boost when explicit constraints are present.
        factor = 0.80 if not traits else 1.0
        if has_constraints:
            factor += 0.10
        return max(0.70, min(1.10, float(factor)))

    def _satisfies_persona_constraints(self, candidate: ResponseCandidate, context: Any) -> bool:
        """Persona as constraints, not only a soft score."""
        text = candidate.text.lower()
        constraints = getattr(context, "persona_constraints", None)
        if not isinstance(constraints, dict):
            constraints = {}

        banned_words = [str(word).lower() for word in list(constraints.get("disallow_words") or [])]
        if any(word and word in text for word in banned_words):
            return False

        required_tone = str(constraints.get("required_tone") or "").strip().lower()
        if required_tone and candidate.tone.lower() != required_tone:
            return False

        max_risk = constraints.get("max_risk")
        if max_risk is not None and candidate.risk_level > float(max_risk):
            return False

        traits = getattr(context, "persona_traits", None)
        if isinstance(traits, list):
            lowered = [str(item).lower() for item in traits]
            if "nonjudgmental" in lowered and "you should have" in text:
                return False
            if "supportive" in lowered and not any(token in text for token in ["i hear", "with you", "together", "steady"]):
                return False
        return True

    @staticmethod
    def extract_emotional_features(text: str) -> dict[str, float]:
        """Fast heuristic emotional classifier with no model dependency."""
        text_lower = str(text or "").lower()
        valence = 0.0
        if any(word in text_lower for word in ["great", "glad", "nice", "love"]):
            valence = 0.6
        elif any(word in text_lower for word in ["sorry", "bad", "unfortunately"]):
            valence = -0.6

        arousal = 0.8 if any(word in text_lower for word in ["!", "really", "very", "definitely"]) else 0.2
        attachment = 0.8 if any(word in text_lower for word in ["we", "us", "together", "i get you"]) else 0.3

        if any(word in text_lower for word in ["definitely", "clearly", "for sure"]):
            confidence = 0.8
        elif any(word in text_lower for word in ["might", "maybe", "could"]):
            confidence = 0.4
        else:
            confidence = 0.6

        risk = 0.8 if any(word in text_lower for word in ["personally", "i feel", "honestly"]) else 0.3

        return {
            "valence": valence,
            "arousal": arousal,
            "attachment": attachment,
            "confidence": confidence,
            "risk": risk,
        }

    def _score_novelty(self, text: str) -> float:
        """Score variation from typical responses (0–1)."""
        # Placeholder: heuristic based on text length and word diversity
        words = text.lower().split()
        unique_words = len(set(words))
        diversity = unique_words / max(len(words), 1)
        return diversity

    def _score_redundancy(self, text: str) -> float:
        """Score similarity to recent responses (0–1, higher = more redundant)."""
        # Placeholder: simple substring matching
        if not self._recent_responses:
            return 0.0
        # Check overlap with recent responses
        max_overlap = 0.0
        for recent in self._recent_responses:
            recent_words = set(recent.lower().split())
            current_words = set(text.lower().split())
            if recent_words and current_words:
                overlap = len(recent_words & current_words) / len(recent_words | current_words)
                max_overlap = max(max_overlap, overlap)
        return max_overlap

    def _score_memory_relevance(self, candidate: ResponseCandidate, context: Any) -> float:
        memory_blobs = []
        direct_memory = getattr(context, "memory_context", None)
        if isinstance(direct_memory, list):
            for item in direct_memory:
                if isinstance(item, dict):
                    memory_blobs.append(str(item.get("text") or item.get("content") or ""))
                else:
                    memory_blobs.append(str(item))

        session_state = getattr(context, "session_state", None)
        if isinstance(session_state, dict):
            semantic_items = list(dict(session_state.get("semantic_memory") or {}).get("items") or [])
            for item in semantic_items[:8]:
                if isinstance(item, dict):
                    memory_blobs.append(str(item.get("text") or item.get("content") or ""))

        memory_blobs = [blob for blob in memory_blobs if str(blob).strip()]
        if not memory_blobs:
            return 0.5

        # Scale guard: cap memory set by lexical prefilter before semantic scoring.
        if len(memory_blobs) > 24:
            tokens = [token.strip(".,!?;:").lower() for token in candidate.text.split() if token.strip()]
            ranked = sorted(
                memory_blobs,
                key=lambda blob: sum(1 for token in tokens if token and token in blob.lower()),
                reverse=True,
            )
            memory_blobs = ranked[:24]

        candidate_embedding = self._embed_text(candidate.text)
        similarities = [self._cosine_similarity(candidate_embedding, self._embed_text(blob)) for blob in memory_blobs]
        semantic_similarity = max(similarities) if similarities else 0.0

        memory_text = " ".join(memory_blobs).lower()
        tokens = [token.strip(".,!?;:").lower() for token in candidate.text.split() if token.strip()]
        lexical_overlap = 0.0
        if tokens:
            overlap = sum(1 for token in tokens if token in memory_text)
            lexical_overlap = min(overlap / max(len(tokens), 1), 1.0)

        return max(0.0, min(1.0, (0.7 * semantic_similarity) + (0.3 * lexical_overlap)))

    @staticmethod
    def _embed_text(text: str, dim: int = 24) -> list[float]:
        vector = [0.0 for _ in range(dim)]
        tokens = [token.strip(".,!?;:\\n\\t").lower() for token in str(text or "").split() if token.strip()]
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(f"dadbot-memory-v1::{token}".encode("utf-8", errors="ignore")).hexdigest()
            bucket = int(digest[:8], 16) % dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    @staticmethod
    def _interaction_bonus(
        *,
        emotion_bias: float,
        memory_relevance: float,
        user_alignment: float,
        trajectory_alignment: float,
        predicted_user_reaction: float,
    ) -> float:
        # Nonlinear cross terms capture interaction effects.
        t1 = max(min(emotion_bias * user_alignment, 0.35), -0.35)
        t2 = max(min(user_alignment * trajectory_alignment, 0.35), -0.35)
        t3 = max(min(memory_relevance * predicted_user_reaction, 0.35), -0.35)
        bonus = (0.35 * t1) + (0.30 * t2) + (0.35 * t3)
        return max(-1.0, min(1.0, math.tanh(bonus * 1.8)))

    def _learned_preference_score(self, features: dict[str, float]) -> float:
        score = self._reward_bias
        for key, value in features.items():
            score += self._reward_weights.get(key, 0.0) * float(value)
        # Keep model output bounded and stable.
        return max(0.0, min(1.0, 0.5 + (0.5 * math.tanh(score))))

    def _apply_feedback_update(self, context: Any) -> None:
        metadata = getattr(context, "job_metadata", None)
        if not isinstance(metadata, dict):
            return
        feedback = metadata.get("reward_feedback")
        if not isinstance(feedback, dict):
            return

        reward = float(feedback.get("reward", 0.0))
        # Skip no-op updates to avoid drift from placeholder payloads.
        if abs(reward) < 1e-9:
            return
        reward = max(-1.0, min(1.0, reward))

        confidence = float(feedback.get("confidence", 0.6))
        confidence = max(0.0, min(1.0, confidence))
        if confidence < 0.2:
            return

        features_raw = feedback.get("features")
        features: dict[str, float]
        if isinstance(features_raw, dict) and features_raw:
            features = {str(k): float(v) for k, v in features_raw.items() if isinstance(v, (int, float))}
        else:
            selected = dict(self._last_turn_telemetry.get("selected") or {})
            components = dict(selected.get("components") or {})
            features = {
                "base_score": float(components.get("base_score", 0.0)),
                "emotion_bias": float(components.get("emotion_bias", 0.0)),
                "memory_relevance": float(components.get("memory_relevance", 0.0)),
                "user_alignment": float(components.get("user_alignment", 0.0)),
                "trajectory_alignment": float(components.get("trajectory_alignment", 0.0)),
                "predicted_user_reaction": float(components.get("predicted_user_reaction", 0.0)),
                "risk_level": float(selected.get("risk_level", 0.0)),
            }

        if not features:
            return

        # Credit assignment: optional per-feature attribution map.
        attribution_raw = feedback.get("attribution")
        attribution: dict[str, float] = {}
        if isinstance(attribution_raw, dict):
            attribution = {
                str(k): max(0.0, min(1.0, float(v)))
                for k, v in attribution_raw.items()
                if isinstance(v, (int, float))
            }

        learning_rate = float(feedback.get("learning_rate", self._reward_learning_rate))
        effective_lr = max(0.0, min(0.2, learning_rate * (0.4 + (0.6 * confidence))))
        for key, value in features.items():
            if key not in self._reward_weights:
                continue
            credit = attribution.get(key, 1.0 if not attribution else 0.0)
            if credit <= 0.0:
                continue
            delta = effective_lr * reward * value * credit
            updated = self._reward_weights[key] + delta
            # Stabilize drift by decaying toward anchor and clipping.
            anchor = self._reward_anchor_weights.get(key, 0.0)
            updated = ((1.0 - self._reward_decay) * updated) + (self._reward_decay * anchor)
            self._reward_weights[key] = float(max(-self._reward_weight_bound, min(self._reward_weight_bound, updated)))

        bias_updated = self._reward_bias + (effective_lr * reward * 0.1)
        self._reward_bias = float(max(-0.5, min(0.5, (1.0 - self._reward_decay) * bias_updated)))

        self._feedback_samples += 1
        self._feedback_ema_abs = (0.9 * self._feedback_ema_abs) + (0.1 * abs(reward))

    def _distribution_collapse_penalty(self, candidate: ResponseCandidate) -> float:
        if self._selection_count < 4:
            return 0.0
        signature = f"{candidate.source}|{candidate.tone}|{candidate.depth}|{candidate.response_goal}"
        count = int(self._selection_style_counts.get(signature, 0))
        return min(count / max(self._selection_count, 1), 1.0)

    def _record_selection(self, candidate: ResponseCandidate) -> None:
        signature = f"{candidate.source}|{candidate.tone}|{candidate.depth}|{candidate.response_goal}"
        self._selection_style_counts[signature] = int(self._selection_style_counts.get(signature, 0)) + 1
        self._selection_count += 1
        self._recent_selected_sources.append(str(candidate.source or ""))
        if len(self._recent_selected_sources) > self._source_history_window:
            self._recent_selected_sources = self._recent_selected_sources[-self._source_history_window :]

    def _source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for source in self._recent_selected_sources:
            key = str(source or "")
            counts[key] = int(counts.get(key, 0)) + 1
        return counts

    def _dominant_source_ratio(self) -> float:
        if not self._recent_selected_sources:
            return 0.0
        counts = self._source_counts()
        if not counts:
            return 0.0
        dominant = max(counts.values())
        return float(dominant) / float(len(self._recent_selected_sources))

    def _source_entropy(self) -> float:
        if not self._recent_selected_sources:
            return 0.0
        counts = self._source_counts()
        total = float(sum(counts.values()))
        if total <= 0:
            return 0.0
        entropy = 0.0
        for value in counts.values():
            p = float(value) / total
            if p > 0:
                entropy -= p * math.log(p, 2)
        return float(entropy)

    def _source_dominance_penalty(self, candidate: ResponseCandidate) -> float:
        counts = self._source_counts()
        total = len(self._recent_selected_sources)
        if total < 6:
            return 0.0
        source = str(candidate.source or "")
        ratio = float(counts.get(source, 0)) / float(max(total, 1))
        if ratio <= 0.55:
            return 0.0
        return min((ratio - 0.55) / 0.45, 1.0)

    def _score_user_alignment(self, candidate: ResponseCandidate, context: Any) -> float:
        prefs = getattr(context, "user_preferences", None)
        if not isinstance(prefs, dict):
            return 0.5

        score = 0.5
        preferred_tone = str(prefs.get("preferred_tone") or "").strip().lower()
        if preferred_tone:
            score += 0.2 if candidate.tone.lower() == preferred_tone else -0.1

        preferred_depth = str(prefs.get("preferred_depth") or "").strip().lower()
        if preferred_depth:
            score += 0.2 if candidate.depth.lower() == preferred_depth else -0.1

        style = str(prefs.get("style") or "").strip().lower()
        if style == "concise":
            score += 0.15 if candidate.depth == "short" else -0.05
        elif style == "detailed":
            score += 0.15 if candidate.depth == "deep" else -0.05

        return max(0.0, min(1.0, score))

    def _score_trajectory_alignment(self, candidate: ResponseCandidate, context: Any) -> float:
        trajectory = getattr(context, "conversation_trajectory", None)
        desired_goal = ""
        preferred_response_goal = ""
        preferred_tone = ""
        continuity_pressure = 0.0
        emotional_target: dict[str, Any] = {}
        continuity_markers: list[str] = []
        felt_state: dict[str, Any] = {}
        if isinstance(trajectory, dict):
            desired_goal = str(trajectory.get("desired_goal") or "").strip().lower()
            preferred_response_goal = str(trajectory.get("preferred_response_goal") or "").strip().lower()
            preferred_tone = str(trajectory.get("preferred_tone") or "").strip().lower()
            continuity_pressure = float(trajectory.get("continuity_pressure", 0.0) or 0.0)
            emotional_target = dict(trajectory.get("emotional_target") or {})
            continuity_markers = [str(item).lower() for item in list(trajectory.get("continuity_markers") or [])]
            felt_state = dict(trajectory.get("felt_state") or {})
        if not felt_state:
            felt_state = dict(getattr(context, "felt_persona_state", None) or {})

        if not desired_goal:
            intent = self._infer_intent(str(getattr(context, "user_input", "") or ""))
            if intent == "question":
                desired_goal = "clarify"
            elif intent == "emotional":
                desired_goal = "engage"
            else:
                desired_goal = "inform"

        if candidate.response_goal == desired_goal:
            score = 1.0
        elif desired_goal == "clarify" and candidate.response_goal == "inform":
            score = 0.7
        elif desired_goal == "engage" and candidate.response_goal in {"inform", "clarify"}:
            score = 0.6
        else:
            score = 0.45

        pressure = max(0.0, min(1.0, continuity_pressure))
        if preferred_tone:
            if str(candidate.tone or "").strip().lower() == preferred_tone:
                score += 0.14 * pressure
            else:
                score -= 0.08 * pressure

        if continuity_markers:
            text = str(candidate.text or "").lower()
            marker_match = any(marker and marker in text for marker in continuity_markers)
            if marker_match:
                score += 0.10 * pressure
            else:
                score -= 0.05 * pressure

        if preferred_response_goal:
            if candidate.response_goal == preferred_response_goal:
                score += 0.08 * pressure
            else:
                score -= 0.03 * pressure

        target_stance = str(felt_state.get("target_stance") or "").strip().lower()
        if target_stance:
            if str(candidate.stance or "").strip().lower() == target_stance:
                score += 0.07 * pressure
            else:
                score -= 0.03 * pressure

        narrative_phase = str(felt_state.get("narrative_phase") or "").strip().lower()
        if narrative_phase == "stabilizing" and float(candidate.risk_level) > 0.45:
            score -= 0.10 * pressure
        if narrative_phase == "building" and str(candidate.stance or "").strip().lower() == "forward":
            score += 0.05 * pressure

        emotional_momentum = float(felt_state.get("emotional_momentum", 0.0) or 0.0)
        target_arousal = float(emotional_target.get("arousal", 0.35) or 0.35)
        target_arousal += max(-0.15, min(0.15, emotional_momentum * 0.20))
        target_arousal = max(0.0, min(1.0, target_arousal))

        target_intensity = max(0.0, min(1.0, 0.15 + (0.8 * target_arousal)))
        intensity_gap = abs(float(candidate.intensity) - target_intensity)
        score -= 0.10 * pressure * intensity_gap

        return max(0.0, min(1.0, score))

    def _simulate_user_reaction(self, candidate: ResponseCandidate, context: Any, emotion_state: EmotionState) -> float:
        """Lightweight internal simulation: likely user affect to this response."""
        relevance = self._score_relevance(candidate.text, context)
        emotional_fit = self.emotion_alignment(candidate.text, emotion_state)
        trajectory = self._score_trajectory_alignment(candidate, context)

        # Risk comfort depends on relationship attachment and confidence.
        risk_tolerance = 0.2 + (0.6 * emotion_state.attachment) + (0.2 * emotion_state.confidence)
        risk_penalty = max(candidate.risk_level - risk_tolerance, 0.0)

        score = (0.40 * relevance) + (0.30 * emotional_fit) + (0.30 * trajectory) - (0.6 * risk_penalty)
        return max(0.0, min(1.0, score))

    def _resolve_emotion_state(self, context: Any) -> EmotionState:
        # Stateful field: context can override, otherwise reuse engine state.
        raw = getattr(context, "emotion_state", None)
        if raw is None:
            session_state = getattr(context, "session_state", None)
            if isinstance(session_state, dict):
                raw = session_state.get("emotion_state")
        if isinstance(raw, EmotionState):
            self._emotion_state = raw
        elif isinstance(raw, dict):
            self._emotion_state = EmotionState.from_mapping(raw)
        return self._emotion_state

    def emotion_alignment(self, candidate: str, emotion: EmotionState) -> float:
        """Distance-based emotional alignment with attachment-aware risk gating."""
        features = self.extract_emotional_features(candidate)

        valence_score = 1.0 - abs(features["valence"] - emotion.valence)
        arousal_score = 1.0 - abs(features["arousal"] - emotion.arousal)
        attachment_score = 1.0 - abs(features["attachment"] - emotion.attachment)
        confidence_score = 1.0 - abs(features["confidence"] - emotion.confidence)

        risk = features["risk"]
        max_allowed_risk = 0.3 + (emotion.attachment * 0.7)
        if risk > max_allowed_risk:
            risk_penalty = (risk - max_allowed_risk) * 1.5
        else:
            risk_penalty = 0.0

        score = (
            0.30 * valence_score
            + 0.20 * arousal_score
            + 0.25 * attachment_score
            + 0.25 * confidence_score
            - risk_penalty
        )
        return max(0.0, min(1.0, score))

    def compute_emotion_weight(self, emotion: EmotionState) -> float:
        """Dynamically adjust emotional influence on selection."""
        attachment_factor = 0.5 + (emotion.attachment * 0.5)
        arousal_factor = 0.7 + (emotion.arousal * 0.6)
        confidence_factor = 0.8 + (emotion.confidence * 0.4)
        weight = 0.15 * attachment_factor * arousal_factor * confidence_factor
        return min(0.35, weight)

    def _adaptive_emotion_weight(self, emotion_state: EmotionState) -> float:
        """Back-compat shim for existing tests/callers."""
        return self.compute_emotion_weight(emotion_state)

    def _emotion_alignment(self, candidate: ResponseCandidate, emotion_state: EmotionState) -> float:
        """Back-compat shim for existing tests/callers."""
        return self.emotion_alignment(candidate.text, emotion_state)

    def _base_score(self, *, coherence: float, relevance: float, persona: float, novelty: float, redundancy: float) -> float:
        score = (
            self.weights.coherence * coherence
            + self.weights.relevance * relevance
            + self.weights.persona_consistency * persona
            + self.weights.novelty * novelty
            - self.weights.redundancy_penalty * redundancy
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi - lo < 1e-9:
            return [0.5 for _ in values]
        return [max(min((value - lo) / (hi - lo), 1.0), 0.0) for value in values]

    @staticmethod
    def _infer_intent(user_input: str) -> str:
        text = str(user_input or "").strip().lower()
        if not text:
            return "statement"
        if "?" in text:
            return "question"
        if any(token in text for token in ("worried", "anxious", "overwhelmed", "sad", "stress")):
            return "emotional"
        return "statement"


if __name__ == "__main__":
    # Quick test
    engine = ResponseEngine()

    class MockContext:
        user_input = "What's the best way to handle conflict?"

    context = MockContext()
    result = engine.run(context)
    print(f"Selected response: {result}")
