"""Avatar generation and management via Ollama image models."""
from __future__ import annotations

import base64
from pathlib import Path

import ollama


class AvatarManager:
    """Owns Dad's avatar image: generation, persistence, and removal."""

    def __init__(self, bot):
        self.bot = bot

    def avatar_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "static" / "dad_avatar.png"

    def current_avatar_exists(self) -> bool:
        return self.avatar_path().exists()

    def generate_avatar(self, custom_prompt: str | None = None, model: str | None = None) -> bool:
        """Generate and save a new avatar image via Ollama. Returns True on success."""
        prompt = (custom_prompt or "").strip() or (
            "Photorealistic warm portrait of a friendly 56-year-old father with kind eyes, "
            "short neatly trimmed graying hair, gentle reassuring smile, wearing a soft flannel shirt, "
            "standing in a cozy home kitchen with wooden cabinets and soft natural window light, "
            "heartwarming atmosphere, high detail, cinematic lighting"
        )
        if not model:
            for candidate in ["flux", "flux-dev", "flux-schnell", "sdxl", "stable-diffusion"]:
                try:
                    ollama.show(candidate)
                    model = candidate
                    break
                except Exception:
                    continue
        if not model:
            return False
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={"num_predict": 1},
            )
            images = (response if isinstance(response, dict) else {}).get("images", [])
            if images:
                self.avatar_path().parent.mkdir(parents=True, exist_ok=True)
                self.avatar_path().write_bytes(base64.b64decode(images[0]))
                self.bot.PROFILE.setdefault("avatar", {})["last_prompt"] = prompt
                self.bot.PROFILE["avatar"]["last_generated_at"] = self.bot.runtime_timestamp()
                self.bot.save_profile()
                return True
        except Exception:
            pass
        return False

    def remove_avatar(self) -> bool:
        """Delete custom avatar and clear profile metadata. Returns True if a file was removed."""
        path = self.avatar_path()
        if path.exists():
            path.unlink(missing_ok=True)
            self.bot.PROFILE.setdefault("avatar", {}).pop("last_prompt", None)
            self.bot.PROFILE["avatar"]["last_generated_at"] = None
            self.bot.save_profile()
            return True
        return False
