"""Avatar generation and management via Ollama image models."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import ollama

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - fallback handles missing Pillow
    Image = None
    ImageDraw = None
    ImageFont = None


class AvatarManager:
    """Owns Dad's avatar image: generation, persistence, and removal."""

    MOOD_PROMPT_MODIFIERS = {
        "positive": "bright, joyful, smiling warmly, energetic",
        "neutral": "calm, focused, thoughtful expression",
        "sad": "tender, compassionate, understanding gaze",
        "frustrated": "determined, focused, slightly concerned",
        "tired": "relaxed, gentle, warm despite fatigue",
    }

    def __init__(self, bot):
        self.bot = bot

    def avatar_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "static" / "dad_avatar.png"

    def current_avatar_exists(self) -> bool:
        return self.avatar_path().exists()

    def _create_fallback_avatar_image(self, mood: str | None = None) -> bytes:
        """Generate a local placeholder avatar when image models are unavailable."""
        if Image is None or ImageDraw is None or ImageFont is None:
            # Minimal valid 1x1 PNG fallback if Pillow is unavailable.
            return base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO0Z6l8AAAAASUVORK5CYII="
            )

        mood_key = str(mood or "neutral").lower().strip()
        mood_colors = {
            "positive": (52, 152, 219),
            "neutral": (149, 165, 166),
            "sad": (155, 89, 182),
            "frustrated": (231, 76, 60),
            "tired": (243, 156, 18),
        }
        accent = mood_colors.get(mood_key, (120, 132, 140))

        image = Image.new("RGB", (480, 480), color=(248, 249, 251))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        # Framed card
        draw.rounded_rectangle((24, 24, 456, 456), radius=24, fill=(255, 255, 255), outline=accent, width=6)

        # Simple avatar face
        draw.ellipse((150, 120, 330, 300), fill=(240, 205, 170), outline=(95, 60, 45), width=3)
        draw.ellipse((190, 180, 210, 200), fill=(50, 40, 35))
        draw.ellipse((270, 180, 290, 200), fill=(50, 40, 35))
        draw.arc((205, 215, 275, 265), 15, 165, fill=(80, 45, 40), width=4)
        draw.arc((145, 95, 335, 250), 205, -25, fill=(120, 120, 120), width=10)

        label = f"Dad ({mood_key or 'neutral'})"
        draw.text((170, 338), label, fill=accent, font=font)
        draw.text((84, 370), "Local fallback avatar (no image model)", fill=(105, 114, 124), font=font)

        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def _save_avatar_image(self, image_bytes: bytes, prompt: str, mood: str | None) -> bool:
        self.avatar_path().parent.mkdir(parents=True, exist_ok=True)
        self.avatar_path().write_bytes(image_bytes)
        self.bot.PROFILE.setdefault("avatar", {})["last_prompt"] = prompt
        self.bot.PROFILE["avatar"]["last_generated_at"] = self.bot.runtime_timestamp()
        self.bot.PROFILE["avatar"]["last_mood"] = mood or "neutral"
        self.bot.save_profile()
        return True

    def _build_prompt_for_mood(self, base_prompt: str | None, mood: str | None) -> str:
        """Build a mood-aware prompt by incorporating mood modifiers."""
        if base_prompt:
            return base_prompt
        
        mood_str = str(mood or "neutral").lower().strip()
        mood_mod = self.MOOD_PROMPT_MODIFIERS.get(mood_str, "warm and approachable")
        
        return (
            f"Photorealistic portrait of a friendly 56-year-old father with kind eyes, "
            f"short neatly trimmed graying hair, {mood_mod}, wearing a soft flannel shirt, "
            f"standing in a cozy home kitchen with wooden cabinets and soft natural window light, "
            f"heartwarming atmosphere, high detail, cinematic lighting"
        )

    def generate_avatar(
        self,
        custom_prompt: str | None = None,
        model: str | None = None,
        mood: str | None = None,
    ) -> bool:
        """Generate and save a new avatar image via Ollama. Returns True on success.
        
        Args:
            custom_prompt: Custom prompt for generation (overrides mood-based generation)
            model: Specific model to use (if None, auto-detect from available)
            mood: Current mood to incorporate into prompt (positive, neutral, sad, frustrated, tired)
        """
        prompt = (custom_prompt or "").strip() or self._build_prompt_for_mood(custom_prompt, mood)
        
        if not model:
            for candidate in [
                "flux",
                "flux-dev",
                "flux-schnell",
                "sdxl",
                "stable-diffusion",
            ]:
                try:
                    ollama.show(candidate)
                    model = candidate
                    break
                except Exception:
                    continue
        if not model:
            fallback = self._create_fallback_avatar_image(mood=mood)
            return self._save_avatar_image(fallback, prompt=f"fallback:{prompt}", mood=mood)
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={"num_predict": 1},
            )
            images = (response if isinstance(response, dict) else {}).get("images", [])
            if images:
                return self._save_avatar_image(base64.b64decode(images[0]), prompt=prompt, mood=mood)
        except Exception:
            fallback = self._create_fallback_avatar_image(mood=mood)
            return self._save_avatar_image(fallback, prompt=f"fallback:{prompt}", mood=mood)
        fallback = self._create_fallback_avatar_image(mood=mood)
        return self._save_avatar_image(fallback, prompt=f"fallback:{prompt}", mood=mood)

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

