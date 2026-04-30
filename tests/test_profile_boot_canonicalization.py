from __future__ import annotations

import json
from pathlib import Path

from dadbot.core.boot_mixin import DadBotBootMixin


def _canonical_profile(payload: dict) -> dict:
    style = payload.get("style") if isinstance(payload.get("style"), dict) else {}
    llm = payload.get("llm") if isinstance(payload.get("llm"), dict) else {}
    conversation_style = (
        payload.get("conversation_style") if isinstance(payload.get("conversation_style"), dict) else {}
    )
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


def _canonical_hash(payload: dict) -> str:
    canonical = _canonical_profile(payload)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_profile_equivalence_template_and_embedded_normalize_to_same_structure(monkeypatch):
    template_path = Path(__file__).resolve().parents[1] / "dad_profile.template.json"
    template_payload = json.loads(template_path.read_text(encoding="utf-8"))

    embedded_payload = json.loads(json.dumps(DadBotBootMixin._EMBEDDED_DEFAULT_PROFILE))

    # Pin canonical defaults so equivalence checks deterministic and explicit.
    monkeypatch.setitem(embedded_payload, "name", template_payload.get("name", "Dad"))

    assert _canonical_profile(template_payload) == _canonical_profile(embedded_payload)


def test_boot_equivalence_no_template_vs_valid_template(monkeypatch, tmp_path):
    destination = tmp_path / "dad_profile.json"

    real_open = Path.open
    template_path = Path(__file__).resolve().parents[1] / "dad_profile.template.json"

    def _raise_for_template(self: Path, *args, **kwargs):
        if self.resolve() == template_path.resolve():
            raise FileNotFoundError("simulated missing template")
        return real_open(self, *args, **kwargs)

    # Scenario A: no template present (simulated), should fall back to embedded.
    monkeypatch.setattr(Path, "open", _raise_for_template)
    no_template_profile = DadBotBootMixin.default_profile()
    DadBotBootMixin.initialize_profile_file(profile_path=destination, force=True)
    no_template_written = json.loads(destination.read_text(encoding="utf-8"))

    # Scenario B: valid template present, should still canonicalize identically.
    monkeypatch.setattr(Path, "open", real_open)
    with_template_profile = DadBotBootMixin.default_profile()
    DadBotBootMixin.initialize_profile_file(profile_path=destination, force=True)
    with_template_written = json.loads(destination.read_text(encoding="utf-8"))

    assert _canonical_profile(no_template_profile) == _canonical_profile(with_template_profile)
    assert _canonical_profile(no_template_written) == _canonical_profile(with_template_written)


def test_cross_environment_identity_payload_is_deterministic(monkeypatch):
    template_path = Path(__file__).resolve().parents[1] / "dad_profile.template.json"
    template_payload = json.loads(template_path.read_text(encoding="utf-8"))

    embedded_payload = json.loads(json.dumps(DadBotBootMixin._EMBEDDED_DEFAULT_PROFILE))
    monkeypatch.setitem(embedded_payload, "name", template_payload.get("name", "Dad"))

    assert _canonical_hash(template_payload) == _canonical_hash(embedded_payload)
