from __future__ import annotations

from datetime import datetime


class InternalStateManager:
    """Maintains a lightweight persistent internal belief/goal state across turns."""

    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    def default_state():
        return {
            "belief_vector": {
                "stability": 0.5,
                "stress_load": 0.3,
                "connection": 0.5,
            },
            "goal_vector": {
                "support": 0.7,
                "challenge": 0.25,
                "proactive": 0.35,
            },
            "tone_targets": {
                "warmth": 0.7,
                "directness": 0.45,
                "playfulness": 0.3,
            },
            "proactive_recommended": False,
            "last_reflection": "",
            "last_reflection_at": None,
            "recent_reflections": [],
            "target_history": [],
            "turn_count": 0,
            "updated_at": None,
        }

    @staticmethod
    def _clamp_unit(value, fallback=0.5):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float(fallback)
        return max(0.0, min(1.0, numeric))

    def snapshot(self):
        store = self.bot.MEMORY_STORE if isinstance(self.bot.MEMORY_STORE, dict) else {}
        raw = store.get("internal_state") if isinstance(store, dict) else None
        if not isinstance(raw, dict):
            normalized = self.default_state()
            self.bot.mutate_memory_store(internal_state=normalized, save=False)
            return dict(normalized)

        defaults = self.default_state()
        normalized = dict(defaults)
        normalized["belief_vector"] = {
            **defaults["belief_vector"],
            **dict(raw.get("belief_vector") or {}),
        }
        normalized["goal_vector"] = {
            **defaults["goal_vector"],
            **dict(raw.get("goal_vector") or {}),
        }
        normalized["tone_targets"] = {
            **defaults["tone_targets"],
            **dict(raw.get("tone_targets") or {}),
        }
        normalized["proactive_recommended"] = bool(raw.get("proactive_recommended", False))
        normalized["last_reflection"] = str(raw.get("last_reflection") or "").strip()
        normalized["last_reflection_at"] = str(raw.get("last_reflection_at") or "").strip() or None
        normalized["recent_reflections"] = [
            str(item).strip()
            for item in list(raw.get("recent_reflections") or [])[-8:]
            if str(item).strip()
        ]
        history = []
        for item in list(raw.get("target_history") or [])[-64:]:
            if not isinstance(item, dict):
                continue
            try:
                history.append(
                    {
                        "recorded_at": str(item.get("recorded_at") or "").strip(),
                        "support": self._clamp_unit(item.get("support", 0.0), fallback=0.0),
                        "challenge": self._clamp_unit(item.get("challenge", 0.0), fallback=0.0),
                        "proactive": self._clamp_unit(item.get("proactive", 0.0), fallback=0.0),
                    }
                )
            except Exception:
                continue
        normalized["target_history"] = history
        try:
            normalized["turn_count"] = max(0, int(raw.get("turn_count", 0) or 0))
        except (TypeError, ValueError):
            normalized["turn_count"] = 0
        normalized["updated_at"] = str(raw.get("updated_at") or "").strip() or None

        for bucket_name in ("belief_vector", "goal_vector", "tone_targets"):
            normalized[bucket_name] = {
                key: self._clamp_unit(value, fallback=defaults[bucket_name].get(key, 0.5))
                for key, value in normalized[bucket_name].items()
            }
        return normalized

    def internal_state_snapshot(self):
        return self.snapshot()

    @staticmethod
    def _blend(previous, target, alpha=0.2):
        return (1.0 - alpha) * float(previous) + alpha * float(target)

    def reflect_after_turn(self, user_input, current_mood, dad_reply=""):
        state = self.snapshot()
        mood = self.bot.normalize_mood(current_mood)
        mood_targets = {
            "positive": {"stability": 0.78, "stress_load": 0.18, "connection": 0.75, "warmth": 0.72, "directness": 0.48, "playfulness": 0.5},
            "neutral": {"stability": 0.62, "stress_load": 0.3, "connection": 0.58, "warmth": 0.68, "directness": 0.5, "playfulness": 0.35},
            "stressed": {"stability": 0.34, "stress_load": 0.82, "connection": 0.66, "warmth": 0.82, "directness": 0.34, "playfulness": 0.1},
            "sad": {"stability": 0.28, "stress_load": 0.74, "connection": 0.8, "warmth": 0.88, "directness": 0.3, "playfulness": 0.08},
            "frustrated": {"stability": 0.36, "stress_load": 0.7, "connection": 0.54, "warmth": 0.72, "directness": 0.62, "playfulness": 0.12},
            "tired": {"stability": 0.33, "stress_load": 0.6, "connection": 0.6, "warmth": 0.8, "directness": 0.3, "playfulness": 0.12},
        }
        target = mood_targets.get(mood, mood_targets["neutral"])

        beliefs = dict(state.get("belief_vector") or {})
        beliefs["stability"] = self._clamp_unit(self._blend(beliefs.get("stability", 0.5), target["stability"], alpha=0.2))
        beliefs["stress_load"] = self._clamp_unit(self._blend(beliefs.get("stress_load", 0.3), target["stress_load"], alpha=0.2))
        beliefs["connection"] = self._clamp_unit(self._blend(beliefs.get("connection", 0.5), target["connection"], alpha=0.16))

        goals = dict(state.get("goal_vector") or {})
        goals["support"] = self._clamp_unit(self._blend(goals.get("support", 0.7), 0.82 if mood in {"sad", "stressed", "tired"} else 0.68, alpha=0.2))
        goals["challenge"] = self._clamp_unit(self._blend(goals.get("challenge", 0.25), 0.2 if mood in {"sad", "stressed", "tired"} else 0.42, alpha=0.18))
        goals["proactive"] = self._clamp_unit(self._blend(goals.get("proactive", 0.35), 0.56 if mood in {"sad", "stressed", "tired"} else 0.32, alpha=0.16))

        tone = dict(state.get("tone_targets") or {})
        tone["warmth"] = self._clamp_unit(self._blend(tone.get("warmth", 0.7), target["warmth"], alpha=0.22))
        tone["directness"] = self._clamp_unit(self._blend(tone.get("directness", 0.45), target["directness"], alpha=0.22))
        tone["playfulness"] = self._clamp_unit(self._blend(tone.get("playfulness", 0.3), target["playfulness"], alpha=0.22))

        proactive_recommended = bool(goals.get("proactive", 0.0) >= 0.5 or beliefs.get("stress_load", 0.0) >= 0.7)
        target_history = list(state.get("target_history") or [])
        target_history.append(
            {
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "support": round(float(goals.get("support", 0.0)), 3),
                "challenge": round(float(goals.get("challenge", 0.0)), 3),
                "proactive": round(float(goals.get("proactive", 0.0)), 3),
            }
        )
        summary = (
            f"Mood={mood}, support={goals['support']:.2f}, challenge={goals['challenge']:.2f}, "
            f"stress={beliefs['stress_load']:.2f}, proactive={str(proactive_recommended).lower()}"
        )

        recent = list(state.get("recent_reflections") or [])
        recent.append(summary)
        payload = {
            "belief_vector": beliefs,
            "goal_vector": goals,
            "tone_targets": tone,
            "proactive_recommended": proactive_recommended,
            "last_reflection": summary,
            "last_reflection_at": datetime.now().isoformat(timespec="seconds"),
            "recent_reflections": recent[-8:],
            "target_history": target_history[-64:],
            "turn_count": max(0, int(state.get("turn_count", 0) or 0)) + 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "last_user_input": str(user_input or "")[:180],
            "last_reply_excerpt": str(dad_reply or "")[:180],
        }
        self.bot.mutate_memory_store(internal_state=payload, save=True)
        return payload

    def reflect_internal_state(self, user_input, current_mood, dad_reply=""):
        return self.reflect_after_turn(user_input, current_mood, dad_reply)


__all__ = ["InternalStateManager"]
