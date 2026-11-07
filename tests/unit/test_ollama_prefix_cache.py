import os
from statistics import median

import ollama
import pytest

from dadbot.core.execution_boundary import ModelGatewayScope
from dadbot.managers.runtime_client import RuntimeClientManager

pytestmark = pytest.mark.unit
_LIVE_TRANSPORT = RuntimeClientManager._call_ollama_sync


@pytest.mark.skipif(
    os.environ.get("DADBOT_RUN_REAL_OLLAMA_PREFIX_CACHE") != "1",
    reason="requires local Ollama and a downloaded chat model",
)
def test_num_keep_prefix_cache_probe(bot, monkeypatch):
    def checked_transport(**kwargs):
        assert kwargs["messages"] == messages, "Prompt trimming would confound the cache experiment"
        return _LIVE_TRANSPORT(**kwargs)

    monkeypatch.setattr(bot.runtime_client, "_call_ollama_sync", checked_transport)
    model = os.environ.get("DADBOT_OPTIONS_TEST_MODEL", "llama3.2")
    system = (
        "You are a concise assistant for a synthetic cache experiment. "
        "Follow the user's instruction exactly. Do not add explanation. "
    ) * 30
    samples = {0: [], 128: []}
    with ModelGatewayScope.bind("ModelPort"):
        for run, num_keep in enumerate((0, 128, 128, 0)):
            messages = [{
                "role": "system",
                "content": f"Synthetic experiment sequence {run}.\n{system}",
            }]
            for turn in range(3):
                messages.append({"role": "user", "content": f"Turn {turn}: reply only READY."})
                response = bot.runtime_client.call_ollama_chat_with_model(
                    model, messages, caller="ModelPort",
                    options={
                        "num_ctx": 4096, "num_predict": 8,
                        "num_keep": num_keep, "temperature": 0.0,
                    },
                    purpose="prefix-cache probe",
                )
                assert response.message.content.strip()
                assert response.prompt_eval_count > 0
                assert response.prompt_eval_duration > 0
                messages.append({"role": "assistant", "content": response.message.content})
                if turn:
                    samples[num_keep].append(response.prompt_eval_duration / 1e6)
                print(
                    f"num_keep={num_keep} sequence={run} turn={turn} "
                    f"prompt_tokens={response.prompt_eval_count} "
                    f"prompt_ms={response.prompt_eval_duration / 1e6:.1f} "
                    f"load_ms={response.load_duration / 1e6:.1f}"
                )
    for num_keep, durations in samples.items():
        assert len(durations) == 4
        print(f"num_keep={num_keep} warm_prefill_median_ms={median(durations):.1f}")
    loaded = next(item for item in ollama.ps().models if item.name in {model, f"{model}:latest"})
    assert loaded.context_length == 4096
