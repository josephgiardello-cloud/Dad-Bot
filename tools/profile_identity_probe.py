from __future__ import annotations

import json
from pathlib import Path

from dadbot.core.boot_mixin import DadBotBootMixin


def canonical_profile(payload: dict) -> dict:
    style = payload.get("style") if isinstance(payload.get("style"), dict) else {}
    llm = payload.get("llm") if isinstance(payload.get("llm"), dict) else {}
    conversation_style = payload.get("conversation_style") if isinstance(payload.get("conversation_style"), dict) else {}
    preferences = payload.get("preferences") if isinstance(payload.get("preferences"), dict) else {}

    tone_value = style.get("tone", conversation_style.get("tone", "supportive"))
    if isinstance(tone_value, str):
        tone_tokens = [tone_value.strip().lower()] if tone_value.strip() else []
    elif isinstance(tone_value, list):
        tone_tokens = [str(item).strip().lower() for item in tone_value if str(item).strip()]
    else:
        tone_tokens = []
    tone_tokens = sorted(set(tone_tokens))
    if "supportive" in tone_tokens:
        tone_family = "supportive"
    elif "warm" in tone_tokens:
        tone_family = "warm"
    else:
        tone_family = "supportive"

    return {
        "name": str(payload.get("name") or style.get("name") or "Dad"),
        "listener_name": str(style.get("listener_name") or "Tony"),
        "relationship": str(payload.get("relationship") or "father"),
        "signoff": str(style.get("signoff") or "Love you, buddy."),
        "llm_provider": str(llm.get("provider") or "ollama").strip().lower(),
        "llm_model": str(llm.get("model") or "llama3.2").strip(),
        "append_signoff": bool(preferences.get("append_signoff", True)),
        "tone_family": tone_family,
    }


def canonical_hash(payload: dict) -> str:
    return json.dumps(canonical_profile(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def scenario_with_template() -> dict:
    payload = DadBotBootMixin.default_profile()
    return {
        "canonical": canonical_profile(payload),
        "hash": canonical_hash(payload),
    }


def scenario_without_template() -> dict:
    template_path = Path(__file__).resolve().parents[1] / "dad_profile.template.json"
    real_open = Path.open

    def raise_for_template(self: Path, *args, **kwargs):
        if self.resolve() == template_path.resolve():
            raise FileNotFoundError("simulated missing template")
        return real_open(self, *args, **kwargs)

    Path.open = raise_for_template
    try:
        payload = DadBotBootMixin.default_profile()
    finally:
        Path.open = real_open

    return {
        "canonical": canonical_profile(payload),
        "hash": canonical_hash(payload),
    }


def main() -> None:
    out = {
        "workspace": str(Path.cwd()),
        "with_template": scenario_with_template(),
        "without_template": scenario_without_template(),
    }
    out["boot_equivalent"] = out["with_template"]["hash"] == out["without_template"]["hash"]
    out["identity_key"] = out["with_template"]["hash"] + "|" + out["without_template"]["hash"]
    print(json.dumps(out, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
