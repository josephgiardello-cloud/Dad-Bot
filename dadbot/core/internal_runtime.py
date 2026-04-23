from __future__ import annotations

from collections import deque


class DadBotInternalRuntime:
    """Holds mutable runtime-only diagnostics/state to keep DadBot.__init__ lean."""

    def __init__(self, context_token_budget: int):
        self.prompt_guard_stats = {
            "trim_count": 0,
            "trimmed_tokens_total": 0,
            "last_purpose": "",
            "last_original_tokens": 0,
            "last_final_tokens": 0,
            "last_trimmed": False,
            "last_updated": None,
        }
        self.last_memory_context_stats = {
            "tokens": 0,
            "budget_tokens": max(1, int(context_token_budget or 0)),
            "selected_sections": 0,
            "total_sections": 0,
            "pruned": False,
            "last_user_input": "",
            "last_updated": None,
        }
        self.last_output_moderation = {
            "approved": True,
            "action": "allow",
            "category": "none",
            "source": "none",
            "reason": "",
        }
        self.last_reply_supervisor = {
            "stage": "idle",
            "approved": True,
            "score": 0,
            "dad_likeness": 0,
            "groundedness": 0,
            "emotional_fit": 0,
            "issues": [],
            "revised": False,
            "duration_ms": 0,
            "source": "none",
        }
        self.last_turn_pipeline = None
        self.background_task_ids = deque(maxlen=32)
