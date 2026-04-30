"""Runtime health monitoring, adaptive pressure management, and hardware optimization."""

from __future__ import annotations

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class RuntimeHealthManager:
    """Owns all runtime-health pressure signals, adaptive budgets, and optimization controls."""

    def __init__(self, bot):
        self.bot = bot

    # â”€â”€ Issue tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def record_runtime_issue(
        self,
        purpose,
        fallback,
        exc=None,
        *,
        level=logging.WARNING,
        metadata=None,
    ):
        summary = self.bot.ollama_error_summary(exc) if exc is not None else ""
        level_name = logging.getLevelName(level)
        if not isinstance(level_name, str):
            level_name = str(level_name)
        issue = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "purpose": str(purpose or "runtime"),
            "fallback": str(fallback or ""),
            "error": summary,
            "level": level_name.lower(),
            "metadata": dict(metadata or {}),
        }
        recent = getattr(self.bot, "_recent_runtime_issues", None)
        if recent is not None:
            recent.append(issue)
        if summary:
            logger.log(
                level,
                "%s degraded: %s. Fallback: %s",
                issue["purpose"],
                summary,
                issue["fallback"],
            )
        else:
            logger.log(
                level,
                "%s degraded. Fallback: %s",
                issue["purpose"],
                issue["fallback"],
            )

    def recent_runtime_issues(self, limit=3):
        try:
            max_items = max(1, int(limit or 1))
        except (TypeError, ValueError):
            max_items = 3
        recent = getattr(self.bot, "_recent_runtime_issues", None)
        if recent is None:
            return []
        return list(reversed(list(recent)[-max_items:]))

    # â”€â”€ Stats accessors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def prompt_guard_stats(self):
        stats = getattr(self.bot, "_prompt_guard_stats", None)
        if not isinstance(stats, dict):
            return {
                "trim_count": 0,
                "trimmed_tokens_total": 0,
                "last_purpose": "",
                "last_original_tokens": 0,
                "last_final_tokens": 0,
                "last_trimmed": False,
                "last_updated": None,
            }
        return dict(stats)

    def memory_context_stats(self):
        stats = getattr(self.bot, "_last_memory_context_stats", None)
        if not isinstance(stats, dict):
            return {
                "tokens": 0,
                "budget_tokens": max(1, int(self.bot.CONTEXT_TOKEN_BUDGET or 0)),
                "selected_sections": 0,
                "total_sections": 0,
                "pruned": False,
                "last_user_input": "",
                "last_updated": None,
            }
        return dict(stats)

    def record_memory_context_stats(
        self,
        *,
        tokens,
        budget_tokens,
        selected_sections,
        total_sections,
        pruned,
        user_input="",
    ):
        self.bot._last_memory_context_stats = {
            "tokens": max(0, int(tokens or 0)),
            "budget_tokens": max(
                1,
                int(budget_tokens or self.bot.CONTEXT_TOKEN_BUDGET or 1),
            ),
            "selected_sections": max(0, int(selected_sections or 0)),
            "total_sections": max(0, int(total_sections or 0)),
            "pruned": bool(pruned),
            "last_user_input": str(user_input or "")[:180],
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        }

    # â”€â”€ Health history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def health_history(self, limit=72):
        try:
            max_items = max(1, int(limit or 1))
        except (TypeError, ValueError):
            max_items = 72
        history = self.bot.MEMORY_STORE.get("health_history", []) if isinstance(self.bot.MEMORY_STORE, dict) else []
        if not isinstance(history, list):
            return []
        normalized = []
        for item in history[-max_items:]:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "recorded_at": str(
                        item.get("recorded_at") or item.get("updated_at") or self.bot.runtime_timestamp(),
                    ).strip(),
                    "level": str(item.get("level") or "green").strip().lower() or "green",
                    "memory_context_ratio": max(
                        0.0,
                        float(item.get("memory_context_ratio", 0.0) or 0.0),
                    ),
                    "prompt_guard_trim_count": max(
                        0,
                        int(item.get("prompt_guard_trim_count", 0) or 0),
                    ),
                    "recent_runtime_issue_count": max(
                        0,
                        int(item.get("recent_runtime_issue_count", 0) or 0),
                    ),
                },
            )
        return normalized

    def record_runtime_health_snapshot(self, snapshot, *, max_points=240):
        if not isinstance(snapshot, dict):
            return
        point = {
            "recorded_at": self.bot.runtime_timestamp(),
            "level": str(snapshot.get("level") or "green").strip().lower() or "green",
            "memory_context_ratio": round(
                max(0.0, float(snapshot.get("memory_context_ratio", 0.0) or 0.0)),
                3,
            ),
            "prompt_guard_trim_count": max(
                0,
                int(snapshot.get("prompt_guard_trim_count", 0) or 0),
            ),
            "recent_runtime_issue_count": max(
                0,
                int(snapshot.get("recent_runtime_issue_count", 0) or 0),
            ),
        }
        history = list(self.health_history(limit=max_points))
        history.append(point)
        self.bot.mutate_memory_store(
            health_history=history[-max(8, int(max_points or 240)) :],
            save=False,
        )

    # â”€â”€ Adaptive budgets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def adaptive_prompt_pressure_factor(self):
        prompt_guard = self.prompt_guard_stats()
        memory_context = self.memory_context_stats()
        issue_count = len(self.recent_runtime_issues(limit=3))
        trim_count = max(0, int(prompt_guard.get("trim_count", 0) or 0))
        try:
            budget_tokens = max(1, int(memory_context.get("budget_tokens", 1) or 1))
            used_tokens = max(0, int(memory_context.get("tokens", 0) or 0))
            memory_ratio = used_tokens / max(1, budget_tokens)
        except (TypeError, ValueError):
            memory_ratio = 0.0
        factor = 1.0
        if trim_count >= 20 or issue_count >= 3 or memory_ratio >= 0.95:
            factor = 0.72
        elif trim_count >= 8 or issue_count >= 1 or memory_ratio >= 0.85:
            factor = 0.86
        if self.bot.LIGHT_MODE:
            factor = min(factor, 0.82)
        return round(max(0.6, min(1.0, factor)), 2)

    def adaptive_background_worker_limit(self, health_snapshot=None):
        if health_snapshot is None:
            health_snapshot = self.current_runtime_health_snapshot(
                log_warnings=False,
                persist=False,
                max_age_seconds=60,
            )
        base_limit = max(
            2,
            int(getattr(self.bot, "_base_background_worker_limit", 12) or 12),
        )
        if not self.bot.LIGHT_MODE:
            return base_limit
        level = "green"
        if isinstance(health_snapshot, dict):
            level = str(health_snapshot.get("level") or "green").strip().lower() or "green"
        if level == "red":
            return 1
        if level == "yellow":
            return min(2, base_limit)
        return min(4, base_limit)

    def should_delay_noncritical_maintenance(self, health_snapshot=None):
        if health_snapshot is None:
            health_snapshot = self.current_runtime_health_snapshot(
                log_warnings=False,
                persist=False,
                max_age_seconds=60,
            )
        level = "green"
        if isinstance(health_snapshot, dict):
            level = str(health_snapshot.get("level") or "green").strip().lower() or "green"
        return level == "red" or (level == "yellow" and self.bot.LIGHT_MODE)

    def adaptive_memory_context_budget(self, baseline_budget):
        return max(
            96,
            int(
                max(1, int(baseline_budget or 1)) * self.adaptive_prompt_pressure_factor(),
            ),
        )

    # â”€â”€ Forecasting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def forecast_minutes_to_red(self, history=None):
        samples = list(history or self.health_history(limit=24))
        if len(samples) < 4:
            return -1
        ratios = []
        for item in samples[-8:]:
            try:
                ratios.append(float(item.get("memory_context_ratio", 0.0) or 0.0))
            except (TypeError, ValueError, AttributeError):
                continue
        if len(ratios) < 4:
            return -1
        delta = ratios[-1] - ratios[0]
        step_count = max(1, len(ratios) - 1)
        growth_per_sample = delta / float(step_count)
        if growth_per_sample <= 0.001:
            return -1
        remaining = max(0.0, 0.95 - ratios[-1])
        if remaining <= 0:
            return 0
        estimated_samples = remaining / growth_per_sample
        return max(0, min(24 * 60, int(round(estimated_samples * 5.0))))

    @staticmethod
    def runtime_health_score(level, memory_ratio, trim_count, issue_count):
        base_map = {"green": 92, "yellow": 72, "red": 42}
        base = int(base_map.get(str(level or "green"), 72))
        memory_penalty = int(max(0.0, min(1.0, float(memory_ratio or 0.0))) * 22)
        trim_penalty = min(24, int(max(0, int(trim_count or 0)) * 0.8))
        issue_penalty = min(28, int(max(0, int(issue_count or 0)) * 8))
        return max(
            0,
            min(100, base - memory_penalty - trim_penalty - issue_penalty + 8),
        )

    @staticmethod
    def runtime_reasoning_confidence(level, memory_ratio, trim_count, issue_count):
        confidence = 1.0
        confidence -= max(0.0, min(1.0, float(memory_ratio or 0.0))) * 0.4
        confidence -= min(20, max(0, int(trim_count or 0))) / 20.0 * 0.25
        confidence -= min(3, max(0, int(issue_count or 0))) / 3.0 * 0.25
        if str(level or "green").strip().lower() == "yellow":
            confidence -= 0.1
        elif str(level or "green").strip().lower() == "red":
            confidence -= 0.2
        return round(max(0.0, min(1.0, confidence)), 3)

    @staticmethod
    def clarification_guidance(level, reasoning_confidence, trim_count, issue_count):
        normalized_level = str(level or "green").strip().lower() or "green"
        confidence = float(reasoning_confidence or 0.0)
        clarification_needed = (
            normalized_level == "red" or confidence < 0.55 or int(trim_count or 0) >= 8 or int(issue_count or 0) >= 2
        )
        if not clarification_needed:
            return False, ""
        if normalized_level == "red":
            return (
                True,
                "Dad is under heavier runtime pressure right now. A short, specific clarification will keep the reply grounded.",
            )
        if int(issue_count or 0) >= 2:
            return (
                True,
                "Dad hit a couple runtime bumps. A quick restatement of what matters most will reduce drift.",
            )
        if int(trim_count or 0) >= 8:
            return (
                True,
                "Dad has been trimming context more than usual. A concise clarification will help preserve the important part.",
            )
        return (
            True,
            "Dad's reasoning confidence dipped a bit. A short clarification will help lock onto the right thread.",
        )

    # â”€â”€ Hardware optimization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def hardware_optimization_status(self):
        payload = (
            self.bot.MEMORY_STORE.get("runtime_optimization", {}) if isinstance(self.bot.MEMORY_STORE, dict) else {}
        )
        if not isinstance(payload, dict):
            return {
                "applied": False,
                "last_applied_at": None,
                "worker_limit": self.bot._base_background_worker_limit,
            }
        return {
            "applied": bool(payload.get("applied", False)),
            "last_applied_at": payload.get("last_applied_at"),
            "worker_limit": int(
                payload.get("worker_limit", self.bot._base_background_worker_limit)
                or self.bot._base_background_worker_limit,
            ),
        }

    def suggest_hardware_optimization(self, health_snapshot=None):
        snapshot = (
            health_snapshot
            if isinstance(health_snapshot, dict)
            else self.current_runtime_health_snapshot(log_warnings=False, persist=False)
        )
        level = str(snapshot.get("level") or "green").strip().lower() or "green"
        recommended = level in {"yellow", "red"}
        worker_limit = self.adaptive_background_worker_limit(snapshot)
        if level == "red":
            worker_limit = min(worker_limit, 1)
        prompt_factor = self.adaptive_prompt_pressure_factor()
        return {
            "recommended": recommended,
            "worker_limit": max(1, int(worker_limit or 1)),
            "prompt_budget_factor": float(prompt_factor),
        }

    def apply_hardware_optimization(self, *, confirm=False):
        suggestion = self.suggest_hardware_optimization()
        if not confirm or not suggestion.get("recommended"):
            return {
                "applied": False,
                "worker_limit": self.bot._base_background_worker_limit,
                "prompt_budget_factor": suggestion.get("prompt_budget_factor", 1.0),
            }
        self.bot._base_background_worker_limit = max(
            1,
            int(
                suggestion.get("worker_limit", self.bot._base_background_worker_limit)
                or self.bot._base_background_worker_limit,
            ),
        )
        self.bot.set_health_quiet_mode(True, save=False)
        self.bot.mutate_memory_store(
            runtime_optimization={
                "applied": True,
                "worker_limit": self.bot._base_background_worker_limit,
                "last_applied_at": self.bot.runtime_timestamp(),
            },
        )
        return {
            "applied": True,
            "worker_limit": self.bot._base_background_worker_limit,
            "prompt_budget_factor": suggestion.get("prompt_budget_factor", 1.0),
        }

    # â”€â”€ Main snapshot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def current_runtime_health_snapshot(
        self,
        *,
        force=False,
        log_warnings=False,
        persist=False,
        max_age_seconds=None,
    ):
        now = time.monotonic()
        if max_age_seconds is None:
            max_age_seconds = max(
                30,
                int(getattr(self.bot, "_health_snapshot_interval_seconds", 300) or 300),
            )
        else:
            max_age_seconds = max(0, int(max_age_seconds or 0))
        cached = getattr(self.bot, "_cached_runtime_health_snapshot", None)
        last = float(
            getattr(self.bot, "_last_runtime_health_snapshot_monotonic", 0.0) or 0.0,
        )
        if not force and isinstance(cached, dict) and (now - last) <= max_age_seconds:
            return dict(cached)
        snapshot = self.runtime_health_snapshot(
            log_warnings=log_warnings,
            persist=persist,
        )
        self.bot._cached_runtime_health_snapshot = dict(snapshot)
        self.bot._last_runtime_health_snapshot_monotonic = now
        return dict(snapshot)

    def health_quiet_mode_enabled(self):
        return bool(self.bot.MEMORY_STORE.get("health_quiet_mode"))

    def set_health_quiet_mode(self, enabled, *, save=True):
        self.bot.mutate_memory_store(health_quiet_mode=bool(enabled), save=save)
        return self.health_quiet_mode_enabled()

    def runtime_health_snapshot(self, *, log_warnings=True, persist=True):
        """Compute runtime-health pressure signals and return an operational snapshot dict."""
        memory_context = self.memory_context_stats()
        prompt_guard = self.prompt_guard_stats()
        recent_runtime_issues = self.recent_runtime_issues(limit=3)
        try:
            budget_tokens = max(1, int(memory_context.get("budget_tokens", 1) or 1))
            used_tokens = max(0, int(memory_context.get("tokens", 0) or 0))
        except (TypeError, ValueError):
            budget_tokens = max(1, int(self.bot.CONTEXT_TOKEN_BUDGET or 1))
            used_tokens = 0
        memory_ratio = min(5.0, used_tokens / max(1, budget_tokens))
        trim_count = max(0, int(prompt_guard.get("trim_count", 0) or 0))
        issue_count = len(recent_runtime_issues)
        level = "green"
        warnings = []
        if memory_ratio >= 0.95:
            level = "red"
            warnings.append(
                "Dad is working a bit hard today - memory context is near the model limit.",
            )
        elif memory_ratio >= 0.85 and level == "green":
            level = "yellow"
            warnings.append(
                "Dad is getting close to the memory limit and may prune more often.",
            )
        if trim_count >= 20:
            level = "red"
            warnings.append(
                "Dad trimmed prompts a lot today - consider clearing older memories.",
            )
        elif trim_count >= 5 and level == "green":
            level = "yellow"
            warnings.append("Dad has started trimming prompts more than usual.")
        if issue_count >= 3:
            level = "red"
            warnings.append(
                "Dad hit multiple runtime bumps, but fallback safety is active.",
            )
        elif issue_count >= 1 and level == "green":
            level = "yellow"
            warnings.append(
                "Dad saw a small runtime hiccup and switched to fallback behavior.",
            )
        adaptive_worker_limit = self.adaptive_background_worker_limit({"level": level})
        prompt_budget_factor = self.adaptive_prompt_pressure_factor()
        delayed_noncritical = self.should_delay_noncritical_maintenance(
            {"level": level},
        )
        quiet_mode_active = self.health_quiet_mode_enabled() and level in {
            "yellow",
            "red",
        }
        projected_minutes_to_red = self.forecast_minutes_to_red()
        health_score = self.runtime_health_score(
            level,
            memory_ratio,
            trim_count,
            issue_count,
        )
        reasoning_confidence = self.runtime_reasoning_confidence(
            level,
            memory_ratio,
            trim_count,
            issue_count,
        )
        clarification_recommended, clarification_message = self.clarification_guidance(
            level,
            reasoning_confidence,
            trim_count,
            issue_count,
        )
        optimization = self.suggest_hardware_optimization({"level": level})
        optimization_state = self.hardware_optimization_status()
        if quiet_mode_active:
            warnings.append(
                "Quiet mode is active while Dad recovers, so proactive nudges are paused.",
            )
        snapshot = {
            "level": level,
            "warnings": warnings,
            "memory_context_ratio": round(memory_ratio, 3),
            "prompt_guard_trim_count": trim_count,
            "recent_runtime_issue_count": issue_count,
            "health_score": int(health_score),
            "reasoning_confidence": float(reasoning_confidence),
            "projected_minutes_to_red": int(projected_minutes_to_red),
            "background_worker_limit": adaptive_worker_limit,
            "prompt_budget_factor": prompt_budget_factor,
            "delayed_noncritical_maintenance": delayed_noncritical,
            "quiet_mode_active": quiet_mode_active,
            "clarification_recommended": bool(clarification_recommended),
            "clarification_message": clarification_message,
            "optimization_recommended": bool(optimization.get("recommended", False)),
            "optimization_applied": bool(optimization_state.get("applied", False)),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if persist:
            self.record_runtime_health_snapshot(snapshot)
        if log_warnings and warnings:
            now = time.monotonic()
            last_logged = float(
                getattr(self.bot, "_last_runtime_health_log_monotonic", 0.0) or 0.0,
            )
            if now - last_logged >= 60.0:
                logger.warning(
                    "Runtime self-health is %s: %s",
                    level,
                    " | ".join(warnings),
                )
                self.bot._last_runtime_health_log_monotonic = now
        return snapshot
