import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from Dad import DadBot


def _make_temp_bot(temp_path: Path) -> DadBot:
    bot = DadBot(light_mode=True)
    bot.CONTEXT_TOKEN_BUDGET = 384
    bot.RESERVED_RESPONSE_TOKENS = 128
    bot.effective_context_token_budget = lambda _model_name=None: 384
    bot.MEMORY_PATH = temp_path / "dad_memory.json"
    bot.SEMANTIC_MEMORY_DB_PATH = temp_path / "dad_memory_semantic.sqlite3"
    bot.GRAPH_STORE_DB_PATH = temp_path / "dad_memory_graph.sqlite3"
    bot.SESSION_LOG_DIR = temp_path / "session_logs"
    bot.MEMORY_STORE = bot.default_memory_store()
    bot.save_memory_store()
    bot.embed_texts = lambda texts, purpose="semantic retrieval": (
        [[0.0] * 12] * len(texts) if isinstance(texts, list) else [[0.0] * 12]
    )
    return bot


@pytest.mark.soak
def test_long_session_stress_harness_tracks_pressure_and_stability():
    with TemporaryDirectory() as temp_dir:
        bot = _make_temp_bot(Path(temp_dir))
        try:
            mood_cycle = ["neutral", "stressed", "positive", "tired", "frustrated", "sad"]
            turn_count = max(100, int(os.environ.get("DADBOT_STRESS_TURNS", "180") or 180))
            observed_levels = []
            memory_sizes = []
            fallback_events = 0

            for turn in range(turn_count):
                current_mood = mood_cycle[turn % len(mood_cycle)]
                user_text = f"Turn {turn}: checking sustained load behavior for mood {current_mood}."
                context_budget = max(128, int(bot.CONTEXT_TOKEN_BUDGET or 0))
                synthetic_tokens = int(context_budget * (0.78 + (turn % 12) * 0.02))
                bot.record_memory_context_stats(
                    tokens=synthetic_tokens,
                    budget_tokens=context_budget,
                    selected_sections=3,
                    total_sections=5,
                    pruned=synthetic_tokens >= int(context_budget * 0.9),
                    user_input=user_text,
                )

                oversized_messages = [
                    {"role": "system", "content": "Core dad persona " + ("context " * 700)},
                    {"role": "user", "content": user_text + " " + ("details " * (300 + (turn % 40)))},
                ]
                guarded_messages = bot.guard_chat_request_messages(oversized_messages, purpose="stress-harness")
                assert guarded_messages

                if turn % 23 == 0:
                    fallback_events += 1
                    bot.record_runtime_issue(
                        "stress-harness",
                        "fallback model",
                        RuntimeError("synthetic fallback event"),
                    )

                bot.submit_background_task(
                    lambda: {"ok": True}, task_kind="post-turn-maintenance", metadata={"turn": turn}
                )
                if turn % 9 == 0:
                    bot.submit_background_task(
                        lambda: {"ok": True}, task_kind="conversation-persist", metadata={"turn": turn}
                    )

                snapshot = bot.runtime_health_snapshot(log_warnings=False, persist=True)
                observed_levels.append(str(snapshot.get("level") or "green"))

                if turn % 15 == 0:
                    memory_sizes.append(len(json.dumps(bot.MEMORY_STORE)))

            for _ in range(10):
                if bot.background_task_snapshot().get("running", 0) == 0:
                    break
                time.sleep(0.05)

            history = bot.health_history(limit=1000)
            assert 12 <= len(history) <= 240

            assert memory_sizes
            min_size = min(memory_sizes)
            max_size = max(memory_sizes)
            assert min_size > 0
            assert (max_size / min_size) < 22.0
            assert max_size < 90000

            guard_stats = bot.prompt_guard_stats()
            assert int(guard_stats.get("trim_count", 0) or 0) >= 40

            fallback_rate = fallback_events / float(turn_count)
            assert 0.02 <= fallback_rate <= 0.15

            background = bot.background_task_snapshot(limit=12)
            assert int(background.get("failed", 0) or 0) == 0
            assert int(background.get("tracked", 0) or 0) >= 10

            assert "yellow" in observed_levels or "red" in observed_levels
        finally:
            bot.shutdown()
