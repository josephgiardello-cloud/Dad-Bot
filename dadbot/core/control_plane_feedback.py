from __future__ import annotations

import time
from typing import Any


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _normalize_sentiment_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _sentiment_reward(value: str) -> tuple[float, float]:
    sentiment = _normalize_sentiment_label(value)
    mapping = {
        "happy": (0.45, 0.70),
        "excited": (0.55, 0.72),
        "grateful": (0.50, 0.75),
        "relieved": (0.35, 0.65),
        "neutral": (0.0, 0.0),
        "sad": (-0.40, 0.65),
        "anxious": (-0.45, 0.68),
        "stressed": (-0.50, 0.70),
        "frustrated": (-0.65, 0.75),
        "angry": (-0.75, 0.80),
    }
    return mapping.get(sentiment, (0.0, 0.0))


def _textual_feedback_signal(text: str) -> tuple[float, float, str]:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return 0.0, 0.0, "neutral"

    negative_markers = (
        "that didn't help",
        "that did not help",
        "not helpful",
        "that's wrong",
        "you are wrong",
        "not what i asked",
        "too much",
        "too long",
        "irrelevant",
        "stop",
        "no,",
    )
    positive_markers = (
        "thank you",
        "thanks",
        "that helps",
        "helpful",
        "good point",
        "exactly",
        "makes sense",
        "perfect",
    )

    if any(marker in lowered for marker in negative_markers):
        return -0.70, 0.74, "frustrated"
    if any(marker in lowered for marker in positive_markers):
        return 0.55, 0.68, "grateful"
    return 0.0, 0.0, "neutral"


def _selected_feedback_features(telemetry: dict[str, Any]) -> dict[str, float]:
    selected = dict(telemetry.get("selected") or {})
    components = dict(selected.get("components") or {})
    return {
        "base_score": float(components.get("base_score", 0.0)),
        "emotion_bias": float(components.get("emotion_bias", 0.0)),
        "memory_relevance": float(components.get("memory_relevance", 0.0)),
        "user_alignment": float(components.get("user_alignment", 0.0)),
        "trajectory_alignment": float(components.get("trajectory_alignment", 0.0)),
        "predicted_user_reaction": float(components.get("predicted_user_reaction", 0.0)),
        "risk_level": float(selected.get("risk_level", 0.0)),
    }


def _synthesize_reward_feedback(
    *,
    pending_selection: dict[str, Any],
    current_user_input: str,
    metadata: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any] | None:
    features = _selected_feedback_features(pending_selection)
    if not any(abs(value) > 1e-9 for value in features.values()):
        return None

    explicit = {}
    for key in ("response_feedback", "user_feedback", "response_outcome", "reaction"):
        candidate = metadata.get(key)
        if isinstance(candidate, dict):
            explicit = dict(candidate)
            break

    reward_terms: list[tuple[float, float, str]] = []
    attribution: dict[str, float] = {}
    evidence: list[str] = []

    accepted = explicit.get("accepted")
    rejected = explicit.get("rejected")
    if isinstance(accepted, bool):
        reward_terms.append((1.0 if accepted else -1.0, 0.92, "explicit_acceptance"))
        evidence.append(f"accepted={accepted}")
        attribution.update({
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.40),
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.35),
            "trajectory_alignment": max(attribution.get("trajectory_alignment", 0.0), 0.25),
        })
    elif isinstance(rejected, bool):
        reward_terms.append((-1.0 if rejected else 0.6, 0.88, "explicit_rejection"))
        evidence.append(f"rejected={rejected}")
        attribution.update({
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.35),
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.30),
            "risk_level": max(attribution.get("risk_level", 0.0), 0.20),
        })

    rating = _coerce_float(explicit.get("rating"))
    if rating is not None:
        normalized = rating
        if rating > 1.0:
            normalized = ((rating - 3.0) / 2.0) if rating <= 5.0 else max(-1.0, min(1.0, rating / 5.0))
        reward_terms.append((_clamp(normalized, -1.0, 1.0), 0.85, "explicit_rating"))
        evidence.append(f"rating={rating}")
        attribution.update({
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.45),
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.35),
        })

    sentiment_sources = [
        explicit.get("sentiment"),
        metadata.get("user_sentiment"),
        dict(session_state.get("last_turn_ux_feedback") or {}).get("mood_hint"),
    ]
    for sentiment_source in sentiment_sources:
        reward_value, confidence = _sentiment_reward(str(sentiment_source or ""))
        if confidence > 0.0:
            reward_terms.append((reward_value, confidence, "sentiment"))
            evidence.append(f"sentiment={sentiment_source}")
            attribution.update({
                "emotion_bias": max(attribution.get("emotion_bias", 0.0), 0.35),
                "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.40),
                "user_alignment": max(attribution.get("user_alignment", 0.0), 0.25),
            })
            break

    engagement_level = _coerce_float(explicit.get("engagement_level"))
    if engagement_level is None:
        engagement_level = _coerce_float(metadata.get("engagement_level"))
    if engagement_level is None:
        interaction_state = dict(session_state.get("interaction_state") or {})
        engagement_level = _coerce_float(interaction_state.get("engagement_level"))
    if engagement_level is not None:
        normalized_engagement = _clamp((engagement_level - 0.5) * 1.2, -1.0, 1.0)
        reward_terms.append((normalized_engagement, 0.58, "engagement_level"))
        evidence.append(f"engagement_level={engagement_level}")
        attribution.update({
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.45),
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.35),
            "trajectory_alignment": max(attribution.get("trajectory_alignment", 0.0), 0.20),
        })

    engagement_delta = _coerce_float(explicit.get("engagement_delta"))
    if engagement_delta is None:
        engagement_delta = _coerce_float(metadata.get("engagement_delta"))
    if engagement_delta is not None:
        reward_terms.append((_clamp(engagement_delta, -1.0, 1.0), 0.68, "engagement_delta"))
        evidence.append(f"engagement_delta={engagement_delta}")
        attribution.update({
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.50),
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.30),
        })

    textual_reward, textual_confidence, inferred_sentiment = _textual_feedback_signal(current_user_input)
    if textual_confidence > 0.0:
        reward_terms.append((textual_reward, textual_confidence, "follow_up_text"))
        evidence.append(f"follow_up_text={inferred_sentiment}")
        attribution.update({
            "user_alignment": max(attribution.get("user_alignment", 0.0), 0.40),
            "predicted_user_reaction": max(attribution.get("predicted_user_reaction", 0.0), 0.30),
            "emotion_bias": max(attribution.get("emotion_bias", 0.0), 0.20),
        })
        if textual_reward < 0.0:
            attribution["risk_level"] = max(attribution.get("risk_level", 0.0), 0.25)

    if not reward_terms:
        return None

    confidence_total = sum(confidence for _reward, confidence, _source in reward_terms)
    if confidence_total <= 1e-9:
        return None

    blended_reward = sum(reward * confidence for reward, confidence, _source in reward_terms) / confidence_total
    blended_confidence = _clamp(confidence_total / max(len(reward_terms), 1), 0.0, 1.0)
    if blended_confidence < 0.35:
        return None

    return {
        "reward": _clamp(blended_reward, -1.0, 1.0),
        "confidence": blended_confidence,
        "features": features,
        "attribution": {key: _clamp(value, 0.0, 1.0) for key, value in attribution.items() if value > 0.0},
        "source": "control_plane.synthesized_outcome",
        "evidence": evidence[-6:],
        "pending_trace_id": str(pending_selection.get("trace_id") or ""),
        "created_at": float(time.time()),
    }


def _response_influence_share(selected: dict[str, Any]) -> dict[str, float]:
    existing = dict(selected.get("influence_share") or {})
    if existing:
        return {
            "safety": float(existing.get("safety", 0.0) or 0.0),
            "tools": float(existing.get("tools", 0.0) or 0.0),
            "memory": float(existing.get("memory", 0.0) or 0.0),
            "coherence": float(existing.get("coherence", 0.0) or 0.0),
        }

    components = dict(selected.get("components") or {})
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


def _update_response_engine_drift_monitor(
    *,
    monitor: dict[str, Any],
    selected: dict[str, Any],
    selected_reasoning: dict[str, Any],
    shadow_event_count: int,
    trace_id: str,
) -> dict[str, Any]:
    history = list(monitor.get("history") or [])
    influence_share = _response_influence_share(selected)
    decision_confidence = max(0.0, float(selected.get("decision_confidence", 0.0) or 0.0))
    selected_source = str(selected.get("source") or "unknown")
    required_reasoning = ("safety", "tools", "memory", "coherence")
    missing_reasoning = any(not str(selected_reasoning.get(name) or "").strip() for name in required_reasoning)

    history.append(
        {
            "trace_id": str(trace_id or ""),
            "timestamp": float(time.time()),
            "selected_source": selected_source,
            "influence_share": influence_share,
            "decision_confidence": float(decision_confidence),
            "shadow_event_count": int(max(0, shadow_event_count)),
            "missing_reasoning": bool(missing_reasoning),
        },
    )
    history = history[-200:]

    window_size = min(50, len(history))
    window = history[-window_size:] if window_size > 0 else []

    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / float(len(values)))

    rolling = {
        "safety": _mean([float(dict(item.get("influence_share") or {}).get("safety", 0.0) or 0.0) for item in window]),
        "tools": _mean([float(dict(item.get("influence_share") or {}).get("tools", 0.0) or 0.0) for item in window]),
        "memory": _mean([float(dict(item.get("influence_share") or {}).get("memory", 0.0) or 0.0) for item in window]),
        "coherence": _mean([float(dict(item.get("influence_share") or {}).get("coherence", 0.0) or 0.0) for item in window]),
    }
    rates = {
        "veto_presence_rate": _mean([1.0 if int(item.get("shadow_event_count", 0) or 0) > 0 else 0.0 for item in window]),
        "missing_reasoning_rate": _mean([1.0 if bool(item.get("missing_reasoning")) else 0.0 for item in window]),
        "avg_decision_confidence": _mean([float(item.get("decision_confidence", 0.0) or 0.0) for item in window]),
    }

    trend = {"safety": 0.0, "tools": 0.0, "memory": 0.0, "coherence": 0.0}
    if window_size >= 10:
        half = max(1, window_size // 2)
        recent = window[-half:]
        prior = window[:-half]
        if prior:
            for key in trend:
                recent_mean = _mean(
                    [float(dict(item.get("influence_share") or {}).get(key, 0.0) or 0.0) for item in recent],
                )
                prior_mean = _mean(
                    [float(dict(item.get("influence_share") or {}).get(key, 0.0) or 0.0) for item in prior],
                )
                trend[key] = float(recent_mean - prior_mean)

    anomalies: list[str] = []
    if rolling["safety"] >= 0.55 and rates["veto_presence_rate"] >= 0.35:
        anomalies.append("safety_overriding_too_often")
    if rolling["tools"] >= 0.60:
        anomalies.append("tools_over_contributing")
    if rolling["memory"] <= 0.08:
        anomalies.append("memory_underutilized")
    if rates["avg_decision_confidence"] <= 0.08:
        anomalies.append("decision_confidence_low")
    if rates["missing_reasoning_rate"] >= 0.10:
        anomalies.append("reasoning_attribution_missing")

    return {
        "history": history,
        "window_size": int(window_size),
        "rolling_averages": rolling,
        "rates": rates,
        "trend": trend,
        "anomalies": anomalies,
        "last_entry": history[-1] if history else {},
        "updated_at": float(time.time()),
    }


__all__ = [
    "_clamp",
    "_coerce_float",
    "_normalize_sentiment_label",
    "_sentiment_reward",
    "_textual_feedback_signal",
    "_selected_feedback_features",
    "_synthesize_reward_feedback",
    "_response_influence_share",
    "_update_response_engine_drift_monitor",
]