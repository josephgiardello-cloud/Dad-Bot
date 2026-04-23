import os

import pytest

pytest.importorskip("litellm")


def _cloud_e2e_provider_settings():
    provider = str(os.environ.get("DADBOT_E2E_PROVIDER") or "openai").strip().lower()
    model = str(os.environ.get("DADBOT_E2E_MODEL") or "gpt-4o-mini").strip()
    api_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
    }
    return provider, model, api_key_map.get(provider, "")


def test_process_user_message_cloud_provider_e2e_voice_disabled(bot):
    if not bool(os.environ.get("DADBOT_RUN_CLOUD_E2E")):
        pytest.skip("Set DADBOT_RUN_CLOUD_E2E=1 to run the live cloud-provider E2E test.")

    provider, model, required_key = _cloud_e2e_provider_settings()
    if provider == "ollama":
        pytest.skip("Set DADBOT_E2E_PROVIDER to a non-Ollama provider for this cloud E2E test.")
    if not required_key:
        pytest.skip(f"Unsupported DADBOT_E2E_PROVIDER value: {provider}")
    if not str(os.environ.get(required_key) or "").strip():
        pytest.skip(f"Set {required_key} to run the live {provider} E2E test.")

    bot.LLM_PROVIDER = provider
    bot.LLM_MODEL = model
    bot.APPEND_SIGNOFF = False
    bot.PROFILE["llm"] = {
        "provider": provider,
        "model": model,
    }
    bot.PROFILE["voice"] = {
        "enabled": False,
        "stt_enabled": False,
        "tts_enabled": False,
    }
    bot.update_agentic_tool_profile(enabled=False, auto_reminders=False, auto_web_lookup=False, save=False)

    reply, should_end = bot.process_user_message("Reply with exactly CLOUD_E2E_OK and nothing else.")

    assert should_end is False
    assert isinstance(reply, str)
    assert "CLOUD_E2E_OK" in reply